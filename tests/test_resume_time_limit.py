"""A run with --timeout resumes faithfully and keeps its time limit.

The [CONTINUATION LIMITS] line shows the live countdown, so a rebuilt request can never
byte-match the request it replays. Replay compares with the recorded countdown put back
(only that field), and the resumed run gets the time it had left once replay ends.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from _journal import measurement_events
from stateframe.replay_tape import (
    Replayer, recorded_countdown_seconds, with_recorded_countdown)
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve
from engine_run import run_monitored, start_monitored
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"


def _limits(seconds: str, decisions: int = 7) -> bytes:
    return json.dumps({"messages": [{"role": "user", "content": (
        "[CONTINUATION LIMITS | stateframe-continuation-limits/1] "
        f"model_decisions_remaining={decisions}; sandbox_seconds_remaining={seconds}; "
        "next_action_admission=allowed")}]}).encode()


RECORDED = _limits("10737.6 (about 179 minutes left, ~12 turns at your pace)")


def test_countdown_swap_moves_only_the_countdown():
    rebuilt = _limits("unbounded")
    assert rebuilt != RECORDED
    assert with_recorded_countdown(rebuilt, RECORDED) == RECORDED
    # Planted: any other difference stays visible after the swap.
    other = _limits("unbounded", decisions=6)
    assert with_recorded_countdown(other, RECORDED) != RECORDED
    # A different number of countdowns is not a countdown-only difference.
    assert with_recorded_countdown(rebuilt + rebuilt, RECORDED) is None
    assert with_recorded_countdown(b"no limits line", RECORDED) is None


def test_recorded_countdown_seconds_reads_the_last_finite_value():
    assert recorded_countdown_seconds(RECORDED) == 10737.6
    assert recorded_countdown_seconds(_limits("unbounded")) is None
    assert recorded_countdown_seconds(b"{}") is None


def _tape(tmp_path: Path, request: bytes) -> list[dict]:
    request_path, response_path = tmp_path / "0001.request.json", tmp_path / "0001.response.json"
    request_path.write_bytes(request)
    response = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
    response_path.write_bytes(response)
    import hashlib
    return [{"send": 1, "request": str(request_path), "response": str(response_path),
             "response_sha256": hashlib.sha256(response).hexdigest()}]


@pytest.mark.parametrize("strict", [False, True])
def test_replay_answers_a_rebuilt_request_whose_only_difference_is_the_countdown(tmp_path, strict):
    replayer = Replayer(_tape(tmp_path, RECORDED), strict=strict)
    assert replayer.answer(_limits("unbounded")) is not None
    assert replayer.diverged is None and replayer.cursor == 1


@pytest.mark.parametrize("strict", [False, True])
def test_replay_still_diverges_on_any_other_difference(tmp_path, strict):
    replayer = Replayer(_tape(tmp_path, RECORDED), strict=strict)
    assert replayer.answer(_limits("unbounded", decisions=6)) is None
    assert replayer.diverged is not None


def test_pending_request_check_takes_the_recorded_countdown(tmp_path):
    pending = tmp_path / "0002.request.json"
    pending.write_bytes(RECORDED)
    replayer = Replayer([], pending={"send": 2, "request": str(pending)})
    assert replayer.answer(_limits("4000.0 (about 67 minutes left)")) is None
    assert replayer.diverged is None and replayer.pending_verified is not None
    planted = Replayer([], pending={"send": 2, "request": str(pending)})
    planted.answer(_limits("unbounded", decisions=6))
    assert planted.pending_mismatch


def test_resumed_time_prefers_the_in_flight_request(tmp_path):
    from stateframe.resume import _resumed_time

    first, last, in_flight = (tmp_path / name for name in ("1.json", "2.json", "3.json"))
    first.write_bytes(_limits("600.0 (about 10 minutes left)"))
    last.write_bytes(_limits("420.0 (about 7 minutes left)"))
    in_flight.write_bytes(_limits("400.0 (about 7 minutes left)"))
    tape = [{"request": str(first)}, {"request": str(last)}]
    assert _resumed_time(tape, {"request": str(in_flight)}, tmp_path) == {
        "seconds_left": 400.0, "time_allowance_seconds": 600.0}
    assert _resumed_time(tape, None, tmp_path)["seconds_left"] == 420.0
    unbounded = tmp_path / "u.json"
    unbounded.write_bytes(_limits("unbounded"))
    assert _resumed_time([{"request": str(unbounded)}], None, tmp_path) == {}


def test_resumed_deadline_overrides_every_budget(monkeypatch):
    sys.path.insert(0, str(ENGINE))
    import kernel_read_fixes

    monkeypatch.setattr(kernel_read_fixes, "_RESUMED_DEADLINE_MONOTONIC", None)
    budget = kernel_read_fixes.ContinuationLimits(10, sandbox_deadline_monotonic=None)
    assert budget._sandbox_seconds_remaining(now=100.0) is None
    kernel_read_fixes.resume_sandbox_deadline(250.0, now=100.0)
    assert budget._sandbox_seconds_remaining(now=150.0) == 200.0
    with pytest.raises(ValueError):
        kernel_read_fixes.resume_sandbox_deadline(float("nan"))


def test_resume_keeps_the_run_time_limit_unless_one_is_given():
    from stateframe.runner import RunOptions, _apply_resumed_time_limit

    def options():
        return RunOptions(repo=Path("."), task="t", endpoint=None)  # type: ignore[arg-type]

    lines: list[str] = []
    kept = options()
    resume = {"seconds_left": 400.0, "time_allowance_seconds": 600.0}
    _apply_resumed_time_limit(kept, resume, 1000.0, None, lines.append)
    assert kept.timeout_s == 600.0 and kept._deadline == 1600.0
    assert "400s left" in lines[0] and resume["seconds_left"] == 400.0
    given = options()
    resume = {"seconds_left": 400.0, "time_allowance_seconds": 600.0}
    _apply_resumed_time_limit(given, resume, 1000.0, 90.0, lines.append)
    assert "seconds_left" not in resume and given._deadline is None


class _StopAtThirdRequest(Script):
    """Answers two requests, then holds the third until the test kills the engine."""

    def __init__(self):
        super().__init__(json.loads((ROOT / "examples/discount_bug.script.json").read_text()))
        self.waiting = threading.Event()
        self.release = threading.Event()

    def next_call(self, offered):
        if self.position == 2 and not self.release.is_set():
            self.waiting.set()
            if not self.release.wait(120):
                raise TimeoutError("held request was not released")
            raise ScriptStepUnavailable("request interrupted before a response")
        return super().next_call(offered)


def _events(run: Path) -> list[dict]:
    return measurement_events(run)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("checkpoint", ["on", "off"])
def test_timeout_run_killed_mid_run_resumes_faithfully(tmp_path, checkpoint):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    run = tmp_path / "run"
    script = _StopAtThirdRequest()
    server = serve(script)
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "STATEFRAME_CHECKPOINT": checkpoint}
    process = None
    try:
        launched_after = time.time()
        process = start_monitored(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Fix the pricing code so the tests pass.", "--endpoint", endpoint,
             "--model", "scripted", "--run-dir", str(run), "--test-python", sys.executable,
             "--max-turns", "16", "--timeout", "3600", "--quiet"],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            run_dir=run, expected_seconds=600, step="time-limit interrupted run",
            watch_paths=(run / "checkpoints",))
        while not script.waiting.wait(0.5):
            assert process.poll() is None, "run ended before its third request"
        original = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
        assert len(original) == 3  # two answered, the third in flight
        assert all(recorded_countdown_seconds(body) is not None for body in original)
        children = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
        assert len(children) == 1, children
        child = int(children[0])
        identity = record_identity(child, launched_after=launched_after,
                                   command_fragment="_child.py")
        # Checkpoint-off runs have no save event to wait for.
        signal_run_verified(run, child, identity, signal.SIGKILL,
                            deadline=time.monotonic() + 120,
                            what="time-limit resume interruption",
                            require_checkpoint_saved=checkpoint == "on")
        script.release.set()
        process.communicate(timeout=120)
        resumed = run_monitored([sys.executable, "-m", "stateframe", "resume", str(run),
                                 "--endpoint", endpoint], env=environment,
                                capture_output=True, text=True, expected_seconds=600)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        script.assert_consumed()
    finally:
        script.release.set()
        if process is not None and process.poll() is None:
            # Failure cleanup may precede any completed checkpoint.
            signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="time-limit run cleanup", require_checkpoint_saved=False,
                             group=True)
            process.communicate(timeout=10)
        server.shutdown()
        server.server_close()

    assert "time limit: this run's --timeout continues" in resumed.stdout
    assert "time limit: none" not in resumed.stdout
    assert not (run / "checkpoint-refused.json").exists()
    assert not (run / "resume-divergence.json").exists()
    rebuilt = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
    # Every recorded request, the in-flight one included, was rebuilt byte for byte apart from
    # the live countdown, which replay compared at its recorded value.
    # Both resume paths number the new attempt's sends from 1, so send N pairs with send N.
    replayed = rebuilt[:len(original)]
    assert len(replayed) == len(original)
    for before, after in zip(original, replayed):
        assert (after == before or with_recorded_countdown(after, before) == before)
    spec = json.loads((run / "spec.json").read_text())
    seconds_left = spec["resume"]["seconds_left"]
    assert 0 < seconds_left <= 3600
    kinds = [event.get("kind") for event in _events(run)]
    assert "checkpoint_unfaithful" not in kinds and "replay_diverged" not in kinds
    # Requests built after replay show the resumed time, never more than what was left.
    live = [recorded_countdown_seconds(body) for body in rebuilt[len(replayed):]]
    assert live and all(value is not None and value <= seconds_left for value in live)
