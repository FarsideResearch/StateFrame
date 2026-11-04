"""Checkpoint write failures and restore refusals across complete scripted runs."""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
import zlib
from pathlib import Path

import pytest

from _journal import measurement_journal_paths
from ._support import Endpoint, FIXTURES, PYTHON, ROOT, copy_repo, environment, run_case
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from engine_run import EngineProcess, run_monitored, start_monitored


pytestmark = pytest.mark.host_sandbox
SCRIPT = FIXTURES / "scripts" / "checkpoint_resume.script.json"
TASK = "Set app.py VALUE to after and verify it"


# Edit receipt IDs are per-run identifiers: two independent runs with checkpoints off differ there
# too. Separate runs are compared with only those IDs blanked; every other byte must match.
_EDIT_RECEIPT = re.compile(rb'edit_receipt\\*":\\*"([0-9a-f]{20})')


def _blank_receipts(data: bytes) -> bytes:
    # A receipt ID is also quoted elsewhere in the frame ("changed by edit <id>"), so every
    # occurrence of each ID the request declares is blanked, in order of first declaration.
    for index, receipt in enumerate(dict.fromkeys(_EDIT_RECEIPT.findall(data))):
        data = data.replace(receipt, b"<receipt-%d>" % index)
    return data


def _wire(run_dir: Path) -> tuple[bytes, ...]:
    return tuple(_blank_receipts(path.read_bytes())
                 for path in sorted((run_dir / "wire").glob("*.request.json")))


def _events(run_dir: Path) -> list[dict]:
    path = run_dir / "checkpoints" / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith("\n")]


def _completed(case) -> None:
    assert case.process.returncode == 0, case.process.stdout + case.process.stderr
    assert case.summary and case.summary["succeeded"] is True, case.summary
    assert (case.run_dir / "workspace" / "app.py").read_text().count('VALUE = "after"') == 1


def _case(tmp_path: Path, name: str, repo: Path, *, checkpoint: str,
          run_env: bool = False, script: Path = SCRIPT):
    folder = tmp_path / name
    folder.mkdir()
    case = run_case(folder, str(script), repo=repo, task=TASK,
                    flags=("--shell", "read-only" if run_env else "off", "--keep-checkpoints",
                           *(("--run-env", "--shell-read", str(ROOT / "src"))
                             if run_env else ())),
                    env_overrides={"STATEFRAME_CHECKPOINT": checkpoint})
    _completed(case)
    if run_env:
        assert (case.run_dir / "env-seed.json").is_file(), case.process.stdout + case.process.stderr
    return case


def _fault_files(tmp_path: Path) -> tuple[Path, Path]:
    injection = tmp_path / "injection"
    injection.mkdir()
    marker = tmp_path / "writer-entered"
    site_source = (
        "import errno, json, os, shutil, signal, sys\n"
        "from pathlib import Path\n"
        "import kernel_checkpoint\n"
        "_save = kernel_checkpoint.save\n"
        "_snapshot_appended_file = kernel_checkpoint._snapshot_appended_file\n"
        "_copy_tree = kernel_checkpoint._copy_tree\n"
        "_clonefile = kernel_checkpoint._clonefile\n"
        "_restore_files = kernel_checkpoint._restore_files\n"
        "_compressobj = kernel_checkpoint.zlib.compressobj\n"
        "if os.environ['CKPT_FAULT_MODE'] in ('journal', 'env'):\n"
        "    kernel_checkpoint._async_enabled = lambda: False\n"
        "_save_count = 0\n"
        "def _pause(**details):\n"
        "    Path(os.environ['CKPT_FAULT_MARKER']).write_text(json.dumps(\n"
        "        {'pid': os.getpid(), 'save': _save_count, **details}))\n"
        "    os.kill(os.getpid(), signal.SIGSTOP)\n"
        "def _save_hook(*args, **kwargs):\n"
        "    global _save_count\n"
        "    _save_count += 1\n"
        "    run = args[2]\n"
        "    journals = list((run / 'worker' / 'measurement-runs').glob('*/events.jsonl'))\n"
        "    events = run / 'checkpoints' / 'events.jsonl'\n"
        "    saved = sum(json.loads(line).get('event') == 'checkpoint_saved'\n"
        "                for line in events.read_text().splitlines()) if events.exists() else 0\n"
        "    with open(os.environ['CKPT_FAULT_TRACE'], 'a') as trace:\n"
        "        trace.write(json.dumps({'pid': os.getpid(), 'save': _save_count,\n"
        "                                'completed_saves': saved,\n"
        "                                'journals': [(str(path), path.stat().st_size)\n"
        "                                             for path in journals]}) + '\\n')\n"
        "    if os.environ['CKPT_FAULT_MODE'] == 'env' and _save_count == 2:\n"
        "        seed = kernel_checkpoint._seed_for_run(run)\n"
        "        assert seed is not None\n"
        "        source = seed / 'seed.json'\n"
        "        target = run / 'env' / 'seed.json'\n"
        "        shutil.copy2(source, target)\n"
        "        target.chmod(0o600)\n"
        "        def directory_clone(path, destination):\n"
        "            return False if path == run / 'env' else _clonefile(path, destination)\n"
        "        kernel_checkpoint._clonefile = directory_clone\n"
        "    return _save(*args, **kwargs)\n"
        "def _journal(*args, **kwargs):\n"
        "    source, _target, relative, root, previous = args[:5]\n"
        "    if os.environ['CKPT_FAULT_MODE'] != 'journal' or _save_count != 2 \\\n"
        "            or not relative.startswith('worker/measurement-runs/') \\\n"
        "            or not relative.endswith('/events.jsonl'):\n"
        "        return _snapshot_appended_file(*args, **kwargs)\n"
        "    saved = sum(json.loads(line).get('event') == 'checkpoint_saved'\n"
        "                for line in (root / 'events.jsonl').read_text().splitlines())\n"
        "    assert saved, 'journal fault requires a completed first checkpoint'\n"
        "    assert previous, 'first checkpoint did not include the audit journal'\n"
        "    covered = previous[-1][2]\n"
        "    size = source.stat().st_size\n"
        "    assert size > covered, f'audit journal did not grow: {size} <= {covered}'\n"
        "    class Compressor:\n"
        "        def __init__(self, level):\n"
        "            self.inner = _compressobj(level)\n"
        "        def compress(self, data):\n"
        "            return self.inner.compress(data)\n"
        "        def flush(self):\n"
        "            caller = sys._getframe(1).f_locals\n"
        "            output = caller['output']\n"
        "            temporary = caller['temporary']\n"
        "            output.flush()\n"
        "            os.fsync(output.fileno())\n"
        "            assert temporary.stat().st_size > 0\n"
        "            _pause(point='journal', temporary=str(temporary), journal=str(source),\n"
        "                   covered=covered, journal_size=size, completed_saves=saved)\n"
        "            return self.inner.flush()\n"
        "    kernel_checkpoint.zlib.compressobj = Compressor\n"
        "    try:\n"
        "        return _snapshot_appended_file(*args, **kwargs)\n"
        "    finally:\n"
        "        kernel_checkpoint.zlib.compressobj = _compressobj\n"
        "def _env(source, target, previous, durable_inodes, *, seed=None):\n"
        "    if os.environ['CKPT_FAULT_MODE'] != 'env' or _save_count != 2:\n"
        "        return _copy_tree(source, target, previous, durable_inodes, seed=seed)\n"
        "    assert source.name == 'env' and seed is not None\n"
        "    def clone(path, destination):\n"
        "        cloned = _clonefile(path, destination)\n"
        "        if cloned and Path(path).name == 'seed.json':\n"
        "            written = sum(1 for file in target.rglob('*') if file.is_file())\n"
        "            assert written >= 2 and destination.is_file()\n"
        "            _pause(point='env', cloned=str(destination), files=written)\n"
        "        return cloned\n"
        "    kernel_checkpoint._clonefile = clone\n"
        "    try:\n"
        "        return _copy_tree(source, target, previous, durable_inodes, seed=seed)\n"
        "    finally:\n"
        "        kernel_checkpoint._clonefile = _clonefile\n"
        "def _restore(checkpoint, run_dir, meta):\n"
        "    result = _restore_files(checkpoint, run_dir, meta)\n"
        "    if os.environ['CKPT_FAULT_MODE'] == 'observe_restore':\n"
        "        journals = list((run_dir / 'worker' / 'measurement-runs').glob('*/events.jsonl'))\n"
        "        assert len(journals) == 1\n"
        "        Path(os.environ['CKPT_FAULT_RESTORED']).write_bytes(journals[0].read_bytes())\n"
        "    return result\n"
        "def _snapshot(*args, **kwargs):\n"
        "    Path(os.environ['CKPT_FAULT_MARKER']).write_text(str(os.getpid()))\n"
        "    if os.environ['CKPT_FAULT_MODE'] == 'pause':\n"
        "        os.kill(os.getpid(), signal.SIGSTOP)\n"
        "    raise OSError(errno.ENOSPC, 'checkpoint storage full')\n"
        "kernel_checkpoint.save = _save_hook\n"
        "kernel_checkpoint._snapshot_appended_file = _journal\n"
        "kernel_checkpoint._copy_tree = _env\n"
        "kernel_checkpoint._restore_files = _restore\n"
        "if os.environ['CKPT_FAULT_MODE'] in ('pause', 'enospc'):\n"
        "    kernel_checkpoint._fsync_snapshot = _snapshot\n")
    (injection / "sitecustomize.py").write_text(
        "try:\n" + textwrap.indent(site_source, "    ")
        + "except BaseException:\n"
        + "    import os as _os, traceback as _traceback\n"
        + "    with open(_os.environ['CKPT_FAULT_IMPORT_ERROR'], 'w') as _error:\n"
        + "        _traceback.print_exc(file=_error)\n"
        + "    raise\n")
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "from stateframe import cli, runner\n"
        "runner.PASS_ENV += ('PYTHONPATH', 'CKPT_FAULT_MARKER', 'CKPT_FAULT_MODE', "
        "'CKPT_FAULT_RESTORED', 'CKPT_FAULT_IMPORT_ERROR', 'CKPT_FAULT_TRACE')\n"
        "raise SystemExit(cli.main())\n")
    return launcher, marker


def _fault_environment(tmp_path: Path, marker: Path, mode: str,
                       restored: Path | None = None, *, checkpoint_every: int = 1) -> dict[str, str]:
    env = environment()
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path / "injection"), env["PYTHONPATH"]))
    env["CKPT_FAULT_MARKER"] = str(marker)
    env["CKPT_FAULT_MODE"] = mode
    env["CKPT_FAULT_IMPORT_ERROR"] = str(tmp_path / "sitecustomize-error")
    env["CKPT_FAULT_TRACE"] = str(tmp_path / "checkpoint-save-trace.jsonl")
    env["STATEFRAME_CHECKPOINT_EVERY"] = str(checkpoint_every)
    if restored is not None:
        env["CKPT_FAULT_RESTORED"] = str(restored)
    return env


def _assert_injection_ok(tmp_path: Path, minimum_saves: int = 0) -> None:
    error = tmp_path / "sitecustomize-error"
    assert not error.exists(), error.read_text() if error.exists() else ""
    trace = tmp_path / "checkpoint-save-trace.jsonl"
    saves = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    assert len(saves) >= minimum_saves, saves


def _run_command(launcher: Path, repo: Path, run_dir: Path, endpoint: str, *,
                 run_env: bool = False) -> list[str]:
    return [PYTHON, str(launcher), "run", "--repo", str(repo), "--task", TASK,
            "--endpoint", endpoint, "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
            "--max-turns", "32", "--shell", "read-only" if run_env else "off",
            "--keep-checkpoints", "--quiet",
            *(("--run-env", "--shell-read", str(ROOT / "src")) if run_env else ())]


def _engine_child(parent_pid: int) -> int:
    rows = subprocess.run(("ps", "-eo", "pid=,ppid=,args="), capture_output=True,
                          text=True, check=True).stdout.splitlines()
    for row in rows:
        parts = row.strip().split(maxsplit=2)
        if len(parts) == 3 and int(parts[1]) == parent_pid and "_child.py" in parts[2]:
            return int(parts[0])
    raise AssertionError("engine child was not found")


def _wait_pid_gone(pid: int, timeout: float = 40) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = subprocess.run(("ps", "-p", str(pid), "-o", "pid="),
                               capture_output=True, text=True, check=False)
        if probe.returncode != 0 or not probe.stdout.strip():
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} was still present after {timeout} seconds")


def _stop_paused_run(run_dir: Path, runner: EngineProcess, engine_pid: int,
                     writer_pid: int, launched_after: float) -> None:
    # The synchronous fault modes use the engine as their writer; keep both PIDs explicit.
    children = tuple(dict.fromkeys((engine_pid, writer_pid)))
    for pid in children:
        try:
            identity = record_identity(pid, launched_after=launched_after,
                                       command_fragment="_child.py")
            # The writer is deliberately paused inside its second checkpoint save.
            signal_run_verified(run_dir, pid, identity, signal.SIGKILL,
                                deadline=time.monotonic(), what="mid-save child cleanup",
                                require_checkpoint_saved=False)
        except ProcessLookupError:
            pass
    try:
        for pid in children:
            _wait_pid_gone(pid)
    finally:
        if runner.poll() is None:
            try:
                # The writer may still be paused inside its second checkpoint save.
                signal_run_owned(run_dir, runner, signal.SIGKILL, deadline=time.monotonic(),
                                 what="mid-save run cleanup", require_checkpoint_saved=False)
            except ProcessLookupError:
                pass
        runner.communicate(timeout=40)
    _wait_pid_gone(runner.pid)


def _wait_for(path: Path, process: EngineProcess, timeout: float = 600,
              run_dir: Path | None = None) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            out, err = process.communicate(timeout=5)
            error = path.parent / "sitecustomize-error"
            trace = path.parent / "checkpoint-save-trace.jsonl"
            journals = ([(str(journal), journal.stat().st_size)
                         for journal in measurement_journal_paths(run_dir)]
                        if run_dir is not None else [])
            saves = _events(run_dir) if run_dir is not None else []
            raise AssertionError(f"run exited before {path.name}: {out} {err}; "
                                 f"import_error={error.read_text() if error.exists() else None}; "
                                 f"save_trace={trace.read_text() if trace.exists() else None}; "
                                 f"checkpoint_events={saves}; journals={journals}")
        time.sleep(0.05)
    assert path.exists(), f"timed out waiting for {path.name}"


def _checkpoint_journal(checkpoint: Path) -> bytes:
    meta = json.loads((checkpoint / "checkpoint.json").read_text())
    journals = {name: rows for name, rows in meta["file_chunks"].items()
                if name.startswith("worker/measurement-runs/") and name.endswith("/events.jsonl")}
    assert len(journals) == 1
    rows = next(iter(journals.values()))
    assert rows
    result = bytearray()
    for name, start, end, raw_sha, stored_sha in rows:
        stored = (checkpoint.parent / "journal-cache" / name).read_bytes()
        assert hashlib.sha256(stored).hexdigest() == stored_sha
        raw = zlib.decompress(stored)
        assert hashlib.sha256(raw).hexdigest() == raw_sha
        assert start == len(result) and end == start + len(raw)
        result.extend(raw)
    return bytes(result)


def _assert_latest(pointer: Path, previous: str) -> None:
    assert pointer.read_text().strip() == previous
    checkpoint = pointer.parent / previous
    assert checkpoint.is_dir() and (checkpoint / "checkpoint.json").is_file()


@pytest.mark.parametrize("mode", (
    "journal",
    pytest.param("env", marks=pytest.mark.skipif(sys.platform != "darwin",
                                                  reason="APFS clonefile path is Darwin only")),
))
def test_killed_second_save_restores_previous_complete_journal(
        tmp_path: Path, mode: str) -> None:
    repo = copy_repo(tmp_path)
    script = SCRIPT
    if mode == "journal":
        document = json.loads(SCRIPT.read_text())
        document["steps"][2:2] = [dict(document["steps"][1]) for _ in range(2)]
        script = tmp_path / "journal.script.json"
        script.write_text(json.dumps(document))
    control = _case(tmp_path, "control", repo, checkpoint="off", run_env=mode == "env",
                    script=script)
    launcher, marker = _fault_files(tmp_path)
    run_dir = tmp_path / mode
    restored_journal = tmp_path / "restored-journal"
    runner = None
    engine_pid = None
    writer_pid = None
    stopped = False
    try:
        with Endpoint(script) as endpoint:
            command = _run_command(launcher, repo, run_dir, endpoint.url,
                                   run_env=mode == "env")
            launched_after = time.time()
            runner = start_monitored(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=_fault_environment(tmp_path, marker, mode,
                                        checkpoint_every=2 if mode == "journal" else 1),
                run_dir=run_dir, expected_seconds=600,
                step="paused second checkpoint save", watch_paths=(run_dir / "checkpoints",))
            if mode == "env":
                _wait_for(run_dir / "env-seed.json", runner)
            _wait_for(marker, runner, run_dir=run_dir)
            _assert_injection_ok(tmp_path, minimum_saves=2)
            assert sum(row.get("event") == "checkpoint_saved" for row in _events(run_dir)) >= 1
            fault = json.loads(marker.read_text())
            writer_pid = int(fault["pid"])
            engine_pid = _engine_child(runner.pid)
            assert fault["save"] == 2 and fault["point"] == mode, fault
            if mode == "journal":
                assert fault["completed_saves"] >= 1, fault
                assert fault["journal_size"] > fault["covered"], fault
                assert Path(fault["journal"]).is_file(), fault
            assert writer_pid == engine_pid
            root = run_dir / "checkpoints"
            pointer = root / "LATEST"
            previous = pointer.read_text().strip()
            expected_journal = _checkpoint_journal(root / previous)
            assert expected_journal
            _assert_latest(pointer, previous)
            pending = list(root.glob(".*.tmp"))
            assert len(pending) == 1 and not (pending[0] / "checkpoint.json").exists()
            assert not (root / pending[0].name[1:-4]).exists()
            if mode == "journal":
                chunk = Path(fault["temporary"])
                assert chunk.is_file() and chunk.stat().st_size > 0
                assert chunk.parent == root / "journal-cache"
                assert chunk.name.startswith(".file-") and chunk.name.endswith(".tmp")
            else:
                clone = Path(fault["cloned"])
                assert fault["files"] >= 2 and clone.is_file()
                assert clone.is_relative_to(pending[0] / "run" / "env")
            interrupted_wire = _wire(run_dir)
            assert len(interrupted_wire) >= 2
            assert interrupted_wire == _wire(control.run_dir)[:len(interrupted_wire)]

            # A planted pointer to the killed temporary must fail the same publication check.
            planted = tmp_path / "planted" / "checkpoints" / "LATEST"
            planted.parent.mkdir(parents=True)
            planted.write_text(pending[0].name + "\n")
            with pytest.raises(AssertionError):
                _assert_latest(planted, previous)

            _stop_paused_run(run_dir, runner, engine_pid, writer_pid, launched_after)
            stopped = True
            _assert_latest(pointer, previous)
            assert pending[0].is_dir() and not (root / pending[0].name[1:-4]).exists()

            resumed = run_monitored(
                [PYTHON, str(launcher), "resume", str(run_dir), "--endpoint", endpoint.url,
                 "--model", "scripted", "--quiet", "--keep-checkpoints"],
                capture_output=True, text=True,
                env=_fault_environment(tmp_path, marker, "observe_restore", restored_journal,
                                       checkpoint_every=2 if mode == "journal" else 1),
                expected_seconds=240, check=False)
            assert resumed.returncode == 0, resumed.stdout + resumed.stderr
            _assert_injection_ok(tmp_path, minimum_saves=2)
            assert json.loads((run_dir / "summary.json").read_text())["succeeded"] is True
            assert restored_journal.read_bytes() == expected_journal
            assert _wire(run_dir) == _wire(control.run_dir)
            assert not list(root.glob(".*.tmp"))
            events = [json.loads(line) for path in run_dir.rglob("checkpoints/events.jsonl")
                      for line in path.read_text().splitlines()]
            assert any(row.get("event") == "checkpoint_restored"
                       and Path(row["source"]).name == previous for row in events)
            assert not any(row.get("event") == "checkpoint_restore_refused" for row in events)
    finally:
        if runner is not None and not stopped:
            if engine_pid is None and runner.poll() is None:
                try:
                    engine_pid = _engine_child(runner.pid)
                except AssertionError:
                    pass
            if writer_pid is None and marker.is_file():
                try:
                    writer_pid = int(json.loads(marker.read_text())["pid"])
                except (ValueError, KeyError, TypeError):
                    pass
            if engine_pid is not None and writer_pid is not None:
                _stop_paused_run(run_dir, runner, engine_pid, writer_pid, launched_after)
            else:
                if runner.poll() is None:
                    try:
                        # Cleanup can interrupt the writer inside its second save.
                        signal_run_owned(run_dir, runner, signal.SIGKILL,
                                         deadline=time.monotonic(), what="mid-save run cleanup",
                                         require_checkpoint_saved=False)
                    except ProcessLookupError:
                        pass
                runner.communicate(timeout=40)
                _wait_pid_gone(runner.pid)


def test_killed_writer_leaves_no_published_partial_checkpoint_and_replays(tmp_path: Path) -> None:
    repo = copy_repo(tmp_path)
    control = _case(tmp_path, "control", repo, checkpoint="off")
    launcher, writer_marker = _fault_files(tmp_path)
    paused_request = tmp_path / "paused-request"
    run_dir = tmp_path / "interrupted"
    endpoint = subprocess.Popen(
        [PYTHON, str(FIXTURES / "pause_endpoint.py"), "--script", str(SCRIPT),
         "--marker", str(paused_request), "--pause-at", "2", "--pause-seconds", "600"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment())
    runner = None
    writer_pid = None
    try:
        assert endpoint.stdout is not None
        url = endpoint.stdout.readline().strip()
        assert url.startswith("http://127.0.0.1:")
        runner = start_monitored(
            _run_command(launcher, repo, run_dir, url),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=_fault_environment(tmp_path, writer_marker, "pause"), run_dir=run_dir,
            expected_seconds=600, step="interrupted checkpoint writer run",
            watch_paths=(run_dir / "checkpoints",),
        )
        _wait_for(writer_marker, runner)
        _assert_injection_ok(tmp_path, minimum_saves=1)
        writer_pid = int(writer_marker.read_text())
        _wait_for(paused_request, runner)
        assert _wire(run_dir)[1] == _wire(control.run_dir)[1]
        pointer = run_dir / "checkpoints" / "LATEST"
        assert not pointer.exists()
        child = _engine_child(runner.pid)
        child_identity = record_identity(child, launched_after=0, command_fragment="_child.py")
        # This fault injection must kill while the checkpoint writer is stopped mid-save.
        signal_run_verified(run_dir, child, child_identity, signal.SIGKILL,
                            deadline=time.monotonic(), what="mid-save engine fault",
                            require_checkpoint_saved=False)
        try:
            writer_identity = record_identity(writer_pid, launched_after=0,
                                              command_fragment="_child.py")
            signal_run_verified(run_dir, writer_pid, writer_identity, signal.SIGKILL,
                                deadline=time.monotonic(), what="mid-save writer fault",
                                require_checkpoint_saved=False)
        except ProcessLookupError:
            pass
        writer_pid = None
        runner.communicate(timeout=40)
        assert not pointer.exists()
        assert list((run_dir / "checkpoints").glob(".*.tmp"))
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), expected_seconds=240, check=False)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert json.loads((run_dir / "summary.json").read_text())["succeeded"] is True
        assert _wire(run_dir)[1] == _wire(control.run_dir)[1]
        assert not list((run_dir / "checkpoints").glob(".*.tmp"))
    finally:
        if runner is not None and runner.poll() is None:
            # Failure cleanup must also work when the writer has not saved.
            signal_run_owned(run_dir, runner, signal.SIGKILL, deadline=time.monotonic(),
                             what="mid-save run cleanup", require_checkpoint_saved=False)
            runner.wait(timeout=5)
        if writer_pid is not None:
            try:
                writer_identity = record_identity(writer_pid, launched_after=0,
                                                  command_fragment="_child.py")
                # The writer is deliberately stopped before any save event.
                signal_run_verified(run_dir, writer_pid, writer_identity, signal.SIGKILL,
                                    deadline=time.monotonic(), what="mid-save writer cleanup",
                                    require_checkpoint_saved=False)
            except ProcessLookupError:
                pass
        if endpoint.poll() is None:
            endpoint.terminate()
            endpoint.wait(timeout=5)


@pytest.mark.parametrize("damage", ("truncate", "flip"))
def test_corrupt_state_refuses_restore_and_replays(tmp_path: Path, damage: str) -> None:
    from .test_resume_completion import _interrupted_run

    repo = copy_repo(tmp_path)
    control = _case(tmp_path, "control", repo, checkpoint="off")
    _, run_dir, _, endpoint, url = _interrupted_run(
        tmp_path, wait_for_checkpoint=True, pause_at=2, repo=repo, task=TASK)
    pointer = run_dir / "checkpoints" / "LATEST"
    state = pointer.parent / pointer.read_text().strip() / "state.pkl"
    original = state.read_bytes()
    assert len(original) > 8
    if damage == "truncate":
        state.write_bytes(original[: len(original) // 2])
    else:
        changed = bytearray(original)
        changed[len(changed) // 2] ^= 0x80
        state.write_bytes(changed)
    try:
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), expected_seconds=240, check=False)
    finally:
        endpoint.terminate()
        endpoint.wait(timeout=5)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert json.loads((run_dir / "summary.json").read_text())["succeeded"] is True
    assert _wire(run_dir)[1] == _wire(control.run_dir)[1]
    spec = json.loads((run_dir / "spec.json").read_text())
    assert "checkpoint" not in spec["resume"]
    archived = run_dir / "attempts" / "1" / "checkpoints" / "events.jsonl"
    assert any(row.get("event") == "checkpoint_restore_refused"
               and "checksum mismatch" in row.get("detail", "")
               for row in map(json.loads, archived.read_text().splitlines()))


def test_writer_enospc_does_not_block_next_request(tmp_path: Path) -> None:
    repo = copy_repo(tmp_path)
    control = _case(tmp_path, "control", repo, checkpoint="off")
    launcher, marker = _fault_files(tmp_path)
    run_dir = tmp_path / "failed-write"
    with Endpoint(SCRIPT) as endpoint:
        process = run_monitored(
            _run_command(launcher, repo, run_dir, endpoint.url),
            capture_output=True, text=True,
            env=_fault_environment(tmp_path, marker, "enospc"),
            expected_seconds=240, run_dir=run_dir, check=False,
        )
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads((run_dir / "summary.json").read_text())["succeeded"] is True
    _assert_injection_ok(tmp_path, minimum_saves=1)
    assert marker.is_file()
    deadline = time.monotonic() + 5
    while not any(row.get("event") == "checkpoint_failed" for row in _events(run_dir)) \
            and time.monotonic() < deadline:
        time.sleep(0.05)
    assert any(row.get("event") == "checkpoint_failed" and "Errno 28" in row.get("error", "")
               for row in _events(run_dir))
    assert _wire(run_dir) == _wire(control.run_dir)


def test_checkpoint_on_and_off_have_identical_wire_requests(tmp_path: Path) -> None:
    repo = copy_repo(tmp_path)
    enabled = _case(tmp_path, "enabled", repo, checkpoint="save")
    disabled = _case(tmp_path, "disabled", repo, checkpoint="off")
    assert any(row.get("event") == "checkpoint_saved" for row in _events(enabled.run_dir))
    assert not (disabled.run_dir / "checkpoints").exists()
    assert len(_wire(enabled.run_dir)) >= 3
    # Raw bytes match because both runs have no deadline: the frame's countdown reads "unbounded" in
    # every request instead of a live clock that would differ between runs.
    for case in (enabled, disabled):
        for path in sorted((case.run_dir / "wire").glob("*.request.json")):
            assert b"sandbox_seconds_remaining=unbounded" in path.read_bytes(), path.name
    assert _wire(enabled.run_dir) == _wire(disabled.run_dir)

    # Planted controls: the comparison still sees one changed byte outside the receipt IDs, and a
    # change in how many distinct receipts a request declares.
    first = (enabled.run_dir / "wire").glob("*.request.json")
    raw = sorted(first)[0].read_bytes()
    position = raw.index(TASK.encode()) + len("Set ")
    mutated = raw[:position] + b"b" + raw[position + 1:]
    assert _blank_receipts(mutated) != _blank_receipts(raw)
    same = b'edit_receipt":"' + b"a" * 20 + b'" and edit_receipt":"' + b"a" * 20 + b'"'
    distinct = b'edit_receipt":"' + b"a" * 20 + b'" and edit_receipt":"' + b"b" * 20 + b'"'
    assert _blank_receipts(same) != _blank_receipts(distinct)


def test_low_disk_skips_checkpoints_without_changing_wire_requests(tmp_path: Path) -> None:
    repo = copy_repo(tmp_path)
    control = _case(tmp_path, "control", repo, checkpoint="off")
    folder = tmp_path / "low-disk"
    folder.mkdir()
    constrained = run_case(
        folder, str(SCRIPT), repo=repo, task=TASK,
        flags=("--shell", "off", "--keep-checkpoints"),
        env_overrides={"STATEFRAME_CHECKPOINT": "on",
                       "STATEFRAME_CHECKPOINT_MIN_FREE_BYTES": str(1 << 63)},
    )
    _completed(constrained)
    assert not list((constrained.run_dir / "checkpoints").glob("*/state.pkl"))
    assert not (constrained.run_dir / "checkpoints" / "LATEST").exists()
    skipped = [row for row in _events(constrained.run_dir)
               if row.get("event") == "checkpoint_skipped_low_disk"]
    assert len(skipped) == 1
    assert skipped[0]["floor_bytes"] == 1 << 63
    assert _wire(constrained.run_dir) == _wire(control.run_dir)
