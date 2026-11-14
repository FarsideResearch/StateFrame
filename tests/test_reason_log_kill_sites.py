"""Kill sites report one stderr line per cause per run, and a repeat of the same cause stays silent."""
import io
import re
import subprocess
import sys
import time

import pytest

from _timeouts import scaled
import kernel_exec_tracker as tracker
import kernel_execution_policy as policy
import kernel_job_watch as jobs
import kernel_reason_log as rl

LINE = re.compile(r"^\[stateframe\] (WARN|ERROR) (?P<component>[a-z_]+): .+", re.M)


@pytest.fixture(autouse=True)
def stream(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    out = io.StringIO()
    rl.configure(level="WARN", stream=out)
    yield out
    rl.reset_for_tests()


def _sleeper():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    assert tracker.capture_launch_identity(process, session_leader=True) is not None
    return process


def _reap(process):
    if process.poll() is None:
        process.kill()
    process.wait(timeout=scaled(5))
    process.stdout.close()
    process.stderr.close()


def _components(out):
    return [m.group("component") for m in LINE.finditer(out.getvalue())]


def test_kill_launch_unit_logs_once_per_cause(stream):
    first, second = _sleeper(), _sleeper()
    try:
        for process in (first, second):
            tracker.kill_launch_unit(process, launch_identity=tracker.recorded_launch_identity(process),
                                     deadline_seconds=2, reason="unit_test_reason")
            if process is first:
                assert _components(stream) == ["exec_tracker"]
        assert _components(stream) == ["exec_tracker"]
        assert "unit_test_reason" in stream.getvalue()
        assert rl.run_summary()["by_cause"]["kill:kill_unit:unit_test_reason"]["count"] == 2
    finally:
        _reap(first), _reap(second)


def test_kill_launch_unit_default_reason_is_safe(stream):
    process = _sleeper()
    try:
        assert tracker.kill_launch_unit(process, launch_identity=tracker.recorded_launch_identity(process)) == ()
    finally:
        _reap(process)
    assert _components(stream) == ["exec_tracker"]


def test_exec_policy_deadline_kill_logs_once_per_cause(stream):
    first, second = _sleeper(), _sleeper()
    try:
        assert policy._kill_group(first, cause="wall_deadline", why="deadline reached") is True
        assert _components(stream) == ["exec_policy"]
        assert policy._kill_group(second, cause="wall_deadline", why="deadline reached") is True
        assert _components(stream) == ["exec_policy"]
        # A different cause is a different line; the tracker's own line is muted for this caller.
        third = _sleeper()
        try:
            policy._kill_group(third, cause="collect_aborted", why="aborted")
        finally:
            _reap(third)
        assert _components(stream) == ["exec_policy", "exec_policy"]
    finally:
        _reap(first), _reap(second)


def test_exec_policy_kill_group_default_args_still_work(stream):
    process = _sleeper()
    try:
        assert policy._kill_group(process) is True
    finally:
        _reap(process)
    assert _components(stream) == ["exec_policy"]


def test_job_watch_job_kill_logs_once_per_cause(tmp_path, stream):
    owner = jobs.JobWatch(tmp_path)
    procs = []
    try:
        for name in ("J1", "J2"):
            process = _sleeper()
            procs.append(process)
            record = jobs.JobRecord(name, time.time(), expected_s=30)
            owner._jobs[name] = jobs._ManagedJob(record, process, time.monotonic(), b"", time.monotonic)
            owner.job_kill(name)
            assert _components(stream).count("job_watch") == 1
    finally:
        for process in procs:
            _reap(process)
    assert rl.run_summary()["by_cause"]["kill:job_watch:model_requested"]["count"] == 2


def test_job_wait_unavailable_logs_reason_and_still_reconciles(tmp_path, stream, monkeypatch):
    owner = jobs.JobWatch(tmp_path)
    process = _sleeper()
    wait = process.wait

    def unavailable(*args, **kwargs):
        raise AttributeError("wait unavailable")

    try:
        record = jobs.JobRecord("J1", time.time(), expected_s=30)
        managed = jobs._ManagedJob(record, process, time.monotonic(), b"", time.monotonic)
        managed.reader_done = True  # no output reader is attached to this primitive test process
        owner._jobs["J1"] = managed
        monkeypatch.setattr(process, "wait", unavailable)
        owner.job_kill("J1")
        cause = rl.run_summary()["by_cause"]["refusal:job_watch:wait_unavailable"]
        assert cause["level"] == "INFO"
        assert "reconciling status" in cause["why"]
        assert "exit wait unavailable" not in stream.getvalue()  # INFO is quiet at WARN.
        assert not managed.cleanup_in_progress
        assert managed.record.state != "running"
    finally:
        monkeypatch.setattr(process, "wait", wait)
        _reap(process)
