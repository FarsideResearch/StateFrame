"""Cross-version checkpoint resume must fail with a named, non-destructive refusal."""
from __future__ import annotations

import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
import kernel_checkpoint
from stateframe import resume


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "acceptance" / "fixtures"
SCRIPT = FIXTURES / "scripts" / "checkpoint_resume.script.json"
TASK = "Set app.py VALUE to after and verify it"
PYTHON = sys.executable

_PINNED_TREES = {
    "base": ("7ea0b8c5699e6acfec1b9d575180098018156859", None),
    "a": ("9686beb110c08187dd5cd984b96a63423a394d43", "zlib1"),
}
_RELEASE_REFUSAL = (
    "resume replays the run on the same engine, so it needs the release that started it"
)


def _archive_tree(parent: Path, revision: str) -> Path:
    """Materialize a pinned producer tree without changing the current checkout."""
    parent.mkdir(parents=True)
    archive = parent.with_suffix(".tar")
    subprocess.run(
        ["git", "-C", str(ROOT), "archive", "--format=tar", "-o", str(archive), revision],
        check=True, capture_output=True, text=True,
    )
    shutil.unpack_archive(str(archive), str(parent), "tar")
    return parent


def _environment(tree: Path, home: Path, *, checkpoint_every: int = 1) -> dict[str, str]:
    """Give each producer/resumer the same isolated install key and the selected source tree."""
    env = {key: os.environ[key] for key in ("PATH", "LANG", "TMPDIR") if key in os.environ}
    env.update({
        "HOME": str(home),
        "STATEFRAME_HOME": str(home / ".stateframe"),
        "PYTHONPATH": os.pathsep.join((str(tree / "src"),
                                        str(tree / "src" / "stateframe" / "engine"))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATEFRAME_CHECKPOINT": "save",
        "STATEFRAME_CHECKPOINT_EVERY": str(checkpoint_every),
    })
    return env


def _start_pause_server(home: Path, script: Path, marker: Path, pidfile: Path):
    process = subprocess.Popen(
        [PYTHON, str(FIXTURES / "pause_endpoint.py"), "--script", str(script),
         "--marker", str(marker), "--pause-at", "3", "--pause-seconds", "600"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=_environment(ROOT, home), start_new_session=True,
    )
    pidfile.write_text(f"{process.pid}\n", encoding="ascii")
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], 15)
    if not ready:
        _stop_pause_server(process, pidfile)
        stdout, stderr = process.communicate(timeout=5)
        raise AssertionError(f"scripted endpoint did not start: {stdout} {stderr}")
    url = process.stdout.readline().strip()
    assert url.startswith("http://127.0.0.1:") and url.endswith("/v1"), url
    return process, url


def _stop_pause_server(server: subprocess.Popen[str], pidfile: Path) -> None:
    if server.poll() is not None:
        return
    pid = int(pidfile.read_text(encoding="ascii").strip())
    assert pid == server.pid, "server pidfile must contain the actual launched PID"
    try:
        server.terminate()
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=5)
    except ProcessLookupError:
        server.wait(timeout=5)


def _engine_child(parent_pid: int) -> int:
    rows = subprocess.run(
        ("ps", "-ww", "-eo", "pid=,ppid=,args="),
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    for row in rows:
        fields = row.strip().split(maxsplit=2)
        if len(fields) == 3 and int(fields[1]) == parent_pid and "_child.py" in fields[2]:
            return int(fields[0])
    raise AssertionError("engine child was not found under the launched runner")


def _wait_for_checkpoint(marker: Path, run_dir: Path, process: subprocess.Popen[str]) -> Path:
    deadline = time.monotonic() + 180
    pointer = run_dir / "checkpoints" / "LATEST"
    while time.monotonic() < deadline and (not marker.is_file() or not pointer.is_file()):
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=5)
            raise AssertionError(f"producer exited before checkpoint/pause: {stdout} {stderr}")
        time.sleep(0.05)
    assert marker.is_file(), "scripted run did not reach its paused request"
    assert pointer.is_file(), "scripted run did not publish a complete checkpoint"
    folder = run_dir / "checkpoints" / pointer.read_text(encoding="utf-8").strip()
    assert (folder / "checkpoint.json").is_file()
    assert (folder / "state.pkl").is_file()
    assert (run_dir / "wire" / "0003.request.json").is_file()
    return folder


def _interrupted_checkpoint(tmp_path: Path, producer: str):
    revision, expected_codec = _PINNED_TREES[producer]
    tree = _archive_tree(tmp_path / "producer-tree", revision)

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    home.chmod(0o700)
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copytree(FIXTURES / "simple_repo", repo, dirs_exist_ok=True)
    run_dir = tmp_path / "run"
    marker = tmp_path / "paused"
    server_pidfile = tmp_path / "pause-endpoint.pid"
    server, url = _start_pause_server(home, SCRIPT, marker, server_pidfile)
    runner: subprocess.Popen[str] | None = None
    keep_server = False
    try:
        runner = subprocess.Popen(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", TASK,
             "--endpoint", url, "--model", "scripted", "--run-dir", str(run_dir),
             "--test-python", PYTHON, "--max-turns", "32", "--shell", "off", "--quiet",
             "--keep-checkpoints"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=_environment(tree, home, checkpoint_every=1), start_new_session=True,
        )
        folder = _wait_for_checkpoint(marker, run_dir, runner)
        metadata = json.loads((folder / "checkpoint.json").read_text(encoding="utf-8"))
        assert metadata["schema"] == kernel_checkpoint.SCHEMA
        assert metadata.get("state_codec") == expected_codec
        pending_request = (run_dir / "wire" / "0003.request.json").read_bytes()

        # Kill only the verified engine child. The old runner and local endpoint remain ours.
        child_pid = _engine_child(runner.pid)
        identity = record_identity(child_pid, launched_after=0, command_fragment="_child.py")
        signal_run_verified(
            run_dir, child_pid, identity, signal.SIGKILL,
            deadline=time.monotonic() + 180, what="checkpoint B load-version interruption",
        )
        runner.communicate(timeout=40)
        runner = None
        keep_server = True
        summary_path = run_dir / "summary.json"
        summary_before = summary_path.read_bytes() if summary_path.is_file() else None
        return (run_dir, home, pending_request, folder, metadata, summary_before,
                server, url, server_pidfile)
    finally:
        if runner is not None and runner.poll() is None:
            signal_run_owned(
                run_dir, runner, signal.SIGKILL, deadline=time.monotonic(),
                what="checkpoint B load-version cleanup", require_checkpoint_saved=False,
            )
            runner.wait(timeout=5)
        if not keep_server:
            _stop_pause_server(server, server_pidfile)
            if server.poll() is None:
                server.wait(timeout=5)


@pytest.mark.parametrize("producer", ("a", "base"), ids=("from-a", "from-0.3.30"))
def test_old_checkpoint_refuses_by_engine_release_without_restore(
    tmp_path: Path, producer: str,
) -> None:
    (run_dir, home, pending_request, checkpoint, metadata, summary_before, server, url,
     server_pidfile) = _interrupted_checkpoint(tmp_path, producer)
    try:
        assert metadata["engine_files_sha256"]
        checkpoint_state = (checkpoint / "state.pkl").read_bytes()
        checkpoint_meta = (checkpoint / "checkpoint.json").read_bytes()
        run_spec = (run_dir / "spec.json").read_bytes()

        # The public resume guard must refuse before restoring or changing the stopped run.
        resumed = subprocess.run(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", url,
             "--model", "scripted", "--quiet", "--keep-checkpoints"],
            capture_output=True, text=True, env=_environment(ROOT, home, checkpoint_every=999),
            timeout=240, check=False,
        )
        output = resumed.stdout + resumed.stderr
        assert resumed.returncode != 0, output
        assert "stateframe: cannot resume:" in output, output
        assert _RELEASE_REFUSAL in output, output
        assert "Traceback" not in output, output
        assert (run_dir / "wire" / "0003.request.json").read_bytes() == pending_request
        assert (checkpoint / "state.pkl").read_bytes() == checkpoint_state
        assert (checkpoint / "checkpoint.json").read_bytes() == checkpoint_meta
        assert (run_dir / "spec.json").read_bytes() == run_spec
        summary_path = run_dir / "summary.json"
        assert (summary_path.read_bytes() if summary_path.is_file() else None) == summary_before
        assert not (run_dir / "attempts/2").exists()
        events_path = run_dir / "checkpoints" / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        assert not any(event.get("event") == "checkpoint_restored" for event in events), events
    finally:
        _stop_pause_server(server, server_pidfile)


def test_unknown_schema_is_refused_with_named_reason(tmp_path: Path) -> None:
    run = tmp_path / "run"
    root = run / "checkpoints"
    folder = root / "0001-unknown-schema"
    run.mkdir(mode=0o700)
    root.mkdir(mode=0o700)
    folder.mkdir(mode=0o700)
    root.chmod(0o700)
    folder.chmod(0o700)
    (folder / kernel_checkpoint.META_FILE).write_text(
        json.dumps({"schema": "stateframe-engine-checkpoint/999", "sends": 0, "seq": 1}),
        encoding="utf-8")
    (folder / kernel_checkpoint.STATE_FILE).write_bytes(b"not reached: schema must be checked first")
    (folder / kernel_checkpoint.META_FILE).chmod(0o600)
    (folder / kernel_checkpoint.STATE_FILE).chmod(0o600)

    reason = ("checkpoint schema 'stateframe-engine-checkpoint/999' is not "
              f"{kernel_checkpoint.SCHEMA}")
    with pytest.raises(kernel_checkpoint.CheckpointError, match=re.escape(reason)):
        kernel_checkpoint.validate(folder)

    # The normal selector must safely skip it; direct validation retains the named refusal.
    assert resume.select_checkpoint(run, [{"send": 1}], run / "wire") is None


def test_cross_release_without_checkpoint_still_refuses(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    parent = run / "parent"
    run.mkdir()
    parent.mkdir()
    spec = {
        "schema": "stateframe-run/1",
        "environment": {},
        "public_row": {},
        "parent_snapshot": str(parent.resolve()),
        "chat_url": "http://127.0.0.1:1/v1",
        "model": "scripted",
        "max_turns": 32,
        "repo": str(tmp_path / "repo"),
        "engine_source": {"shipped_hash": "older-release"},
    }
    (run / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(resume.outcomes, "terminal_receipt", lambda _run: {})
    monkeypatch.setattr(resume, "_terminal_resume_refusal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(resume, "_tool_safety", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(resume.profiles, "engine_source_summary",
                        lambda: {"shipped_hash": "current-release"})
    monkeypatch.setattr(resume, "_lock_holder", lambda _path: None)
    monkeypatch.setattr(resume, "_journal_lineage_ok", lambda _run: None)
    monkeypatch.setattr(resume.replay_tape, "build", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(resume, "carry_forward", lambda _spec, tape, *_args, **_kwargs: tape)

    with pytest.raises(resume.ResumeRefused, match="same engine, so it needs the release that started it"):
        resume.assess(run)
