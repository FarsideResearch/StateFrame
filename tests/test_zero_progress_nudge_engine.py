"""The real engine loop delivers the no-progress nudge once, never after an edit or check, and not again after resume."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from _timeouts import scaled
from engine_run import run_monitored, start_monitored
from fixtures.process_signals import record_identity, signal_run_verified, wait_for_checkpoint_saved
from stateframe.scripted_endpoint import Script, serve
from test_hook_checkpoint_resume import InterruptAfterActions, _launcher, _repo

ROOT = Path(__file__).resolve().parents[1]
NUDGE = "[kernel] No edit or check yet after"
STEP = "3"


def _step(tool: str, **arguments: object) -> dict:
    return {"tool": tool, "required": True, "arguments": arguments}


def _reads(count: int) -> list[dict]:
    return [_step("read", path="pkg/value.py") for _ in range(count)]


EDIT = _step("str_replace", path="pkg/value.py", old_str="VALUE = 1", new_str="VALUE = 3")
CHECK = _step("run_tests", target="tests/test_value.py")
DONE = _step("done", summary="Checks passed after the edit.")


def _environment() -> dict[str, str]:
    return {**os.environ, "STATEFRAME_ZERO_PROGRESS_NUDGE_STEP": STEP,
            "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "src/stateframe/engine")))}


def _command(tmp_path: Path, server, run: Path, *extra: str) -> list[str]:
    return [sys.executable, str(_launcher(tmp_path)), "run", "--repo", str(_repo(tmp_path)),
            "--task", "Inspect and update VALUE.", "--endpoint",
            f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
            "--run-dir", str(run), "--test-python", sys.executable,
            "--max-turns", "12", "--shell", "off", "--quiet", *extra]


def _nudges(run: Path) -> list[str]:
    """Request bodies where the nudge first appears (StateFrame then carries it forward in later requests)."""
    bodies = [p.read_text() for p in sorted((run / "wire").glob("*.request.json"))]
    return [body for i, body in enumerate(bodies)
            if NUDGE in body and (i == 0 or NUDGE not in bodies[i - 1])]


def _run(tmp_path: Path, steps: list[dict]) -> Path:
    server = serve(Script({"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}}))
    run = tmp_path / "run"
    try:
        run_monitored(_command(tmp_path, server, run), env=_environment(), run_dir=run,
                      step="nudge engine run", capture_output=True, text=True,
                      expected_seconds=scaled(240))
    finally:
        server.shutdown()
        server.server_close()
    return run


@pytest.mark.host_sandbox
def test_reading_run_gets_exactly_one_nudge_at_the_threshold(tmp_path):
    run = _run(tmp_path, [*_reads(6), CHECK, DONE])
    nudges = _nudges(run)
    assert len(nudges) == 1
    text = nudges[0]
    assert "after 3 turns" in text and "pkg/value.py x" in text
    # Once delivered it stays visible in every later request (no gaps), and is never re-delivered.
    present = [NUDGE in p.read_text() for p in sorted((run / "wire").glob("*.request.json"))]
    first = present.index(True)
    assert all(present[first:])


@pytest.mark.host_sandbox
def test_run_that_edits_before_the_threshold_gets_no_nudge(tmp_path):
    run = _run(tmp_path, [EDIT, *_reads(5), CHECK, DONE])
    assert _nudges(run) == []


@pytest.mark.host_sandbox
def test_run_that_checks_before_the_threshold_gets_no_nudge(tmp_path):
    run = _run(tmp_path, [CHECK, *_reads(5), DONE])
    assert _nudges(run) == []


@pytest.mark.host_sandbox
def test_resume_after_the_nudge_does_not_repeat_it(tmp_path):
    steps = [*_reads(7), CHECK, DONE]
    script = InterruptAfterActions({"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}}, 5)
    server = serve(script)
    run = tmp_path / "interrupted"
    env = _environment()
    try:
        import json as _json
        profile = _json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
        profile["switches"]["STATEFRAME_CHECKPOINT_EVERY"] = "1"
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(_json.dumps(profile))
        process = start_monitored(
            _command(tmp_path, server, run, "--keep-checkpoints", "--profile", str(profile_path)),
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, run_dir=run,
            expected_seconds=scaled(600), step="nudge resume interrupted run",
            watch_paths=(run / "checkpoints",))
        while not script.waiting.wait(0.5):
            assert process.poll() is None, "run stopped before the interrupted request"
        deadline = time.monotonic() + scaled(120)
        wait_for_checkpoint_saved(run, deadline=deadline, what="nudge resume")
        assert len(_nudges(run)) == 1, "the nudge must be delivered before the interruption"
        child = int(subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()[0])
        identity = record_identity(child, launched_after=0, command_fragment="_child.py")
        signal_run_verified(run, child, identity, signal.SIGKILL, deadline=deadline, what="nudge resume")
        script.release.set()
        assert script.interrupted.wait(scaled(10))
        process.communicate(timeout=scaled(90))
        resumed = run_monitored(
            [sys.executable, "-m", "stateframe", "resume", str(run),
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--keep-checkpoints"],
            env=env, capture_output=True, text=True, expected_seconds=scaled(240))
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    finally:
        server.shutdown()
        server.server_close()
    assert len(_nudges(run)) == 1
