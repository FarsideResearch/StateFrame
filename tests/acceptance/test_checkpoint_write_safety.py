"""Checkpoint write failures and restore refusals across complete scripted runs."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

from ._support import Endpoint, FIXTURES, PYTHON, copy_repo, environment, run_case


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


def _case(tmp_path: Path, name: str, repo: Path, *, checkpoint: str):
    folder = tmp_path / name
    folder.mkdir()
    case = run_case(folder, str(SCRIPT), repo=repo, task=TASK,
                    flags=("--shell", "off", "--keep-checkpoints"),
                    env_overrides={"STATEFRAME_CHECKPOINT": checkpoint})
    _completed(case)
    return case


def _fault_files(tmp_path: Path) -> tuple[Path, Path]:
    injection = tmp_path / "injection"
    injection.mkdir()
    marker = tmp_path / "writer-entered"
    (injection / "sitecustomize.py").write_text(
        "import errno, os, signal\n"
        "from pathlib import Path\n"
        "import kernel_checkpoint\n"
        "_original = kernel_checkpoint._fsync_snapshot\n"
        "def _snapshot(*args, **kwargs):\n"
        "    Path(os.environ['CKPT_FAULT_MARKER']).write_text(str(os.getpid()))\n"
        "    if os.environ['CKPT_FAULT_MODE'] == 'pause':\n"
        "        os.kill(os.getpid(), signal.SIGSTOP)\n"
        "    raise OSError(errno.ENOSPC, 'checkpoint storage full')\n"
        "kernel_checkpoint._fsync_snapshot = _snapshot\n")
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "from stateframe import cli, runner\n"
        "runner.PASS_ENV += ('PYTHONPATH', 'CKPT_FAULT_MARKER', 'CKPT_FAULT_MODE')\n"
        "raise SystemExit(cli.main())\n")
    return launcher, marker


def _fault_environment(tmp_path: Path, marker: Path, mode: str) -> dict[str, str]:
    env = environment()
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path / "injection"), env["PYTHONPATH"]))
    env["CKPT_FAULT_MARKER"] = str(marker)
    env["CKPT_FAULT_MODE"] = mode
    return env


def _run_command(launcher: Path, repo: Path, run_dir: Path, endpoint: str) -> list[str]:
    return [PYTHON, str(launcher), "run", "--repo", str(repo), "--task", TASK,
            "--endpoint", endpoint, "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
            "--max-turns", "32", "--shell", "off", "--keep-checkpoints", "--quiet"]


def _engine_child(parent_pid: int) -> int:
    rows = subprocess.run(("ps", "-eo", "pid=,ppid=,args="), capture_output=True,
                          text=True, check=True).stdout.splitlines()
    for row in rows:
        parts = row.strip().split(maxsplit=2)
        if len(parts) == 3 and int(parts[1]) == parent_pid and "_child.py" in parts[2]:
            return int(parts[0])
    raise AssertionError("engine child was not found")


def _wait_for(path: Path, process: subprocess.Popen[str], timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            out, err = process.communicate(timeout=5)
            raise AssertionError(f"run exited before {path.name}: {out} {err}")
        time.sleep(0.05)
    assert path.exists(), f"timed out waiting for {path.name}"


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
        runner = subprocess.Popen(_run_command(launcher, repo, run_dir, url),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  env=_fault_environment(tmp_path, writer_marker, "pause"))
        _wait_for(writer_marker, runner)
        writer_pid = int(writer_marker.read_text())
        _wait_for(paused_request, runner)
        assert _wire(run_dir)[1] == _wire(control.run_dir)[1]
        pointer = run_dir / "checkpoints" / "LATEST"
        assert not pointer.exists()
        os.kill(_engine_child(runner.pid), signal.SIGKILL)
        try:
            os.kill(writer_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        writer_pid = None
        runner.communicate(timeout=40)
        assert not pointer.exists()
        assert list((run_dir / "checkpoints").glob(".*.tmp"))
        resumed = subprocess.run(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), timeout=240, check=False)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert json.loads((run_dir / "summary.json").read_text())["succeeded"] is True
        assert _wire(run_dir)[1] == _wire(control.run_dir)[1]
        assert not list((run_dir / "checkpoints").glob(".*.tmp"))
    finally:
        if runner is not None and runner.poll() is None:
            runner.kill()
            runner.wait(timeout=5)
        if writer_pid is not None:
            try:
                os.kill(writer_pid, signal.SIGKILL)
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
        resumed = subprocess.run(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), timeout=240, check=False)
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
        process = subprocess.run(_run_command(launcher, repo, run_dir, endpoint.url),
                                 capture_output=True, text=True,
                                 env=_fault_environment(tmp_path, marker, "enospc"),
                                 timeout=240, check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads((run_dir / "summary.json").read_text())["succeeded"] is True
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
    assert _wire(enabled.run_dir) == _wire(disabled.run_dir)


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
