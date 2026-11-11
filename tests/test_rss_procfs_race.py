"""A procfs race while sampling a live check must never kill it."""
from __future__ import annotations

import errno
import threading
import time

import pytest

import kernel_execution_policy as policy
from kernel_exec_tracker import LaunchIdentity, ProcessIdentity

LEADER = ProcessIdentity(123, 1, 123, 123, (10, 0), False)
RACES = [
    ProcessLookupError(errno.ESRCH, "No such process"),
    FileNotFoundError(errno.ENOENT, "No such file or directory"),
    ValueError("invalid literal for int() with base 10: ''"),
    IndexError("list index out of range"),
    RuntimeError("Linux process status omits VmPTE"),
]


class _Process:
    pid = LEADER.pid

    def __init__(self):
        self.returncode = None
        self.memory_limit_killed = False
        self.memory_limit_reason = None
        self.memory_limit_samples_skipped = 0
        self.memory_limit_sample_skip_reason = None


class _Table:
    """Raise one injected procfs race per sample, then measure normally."""

    def __init__(self, process, races, *, breach=""):
        self.process = process
        self.races = list(races)
        self.breach = breach
        self.samples = 0
        self.done = threading.Event()

    def identity(self, pid):
        return LEADER if pid == LEADER.pid else None

    @staticmethod
    def pid_namespace_inode(_pid):
        return None

    def refresh_members(self, _members, **_kwargs):
        self.samples += 1
        if self.races:
            raise self.races.pop(0)
        if not self.breach and self.samples > 20:
            # The healthy check kept running through every injected race.
            self.done.set()
        return {LEADER.pid: LEADER}

    @staticmethod
    def snapshot():
        return {LEADER.pid: LEADER}

    def resident_memory_limit_breach(self, _live, _memory_bytes):
        return self.breach


def _supervise(monkeypatch, races, *, breach="", failure_limit=None):
    process = _Process()
    table = _Table(process, races, breach=breach)
    stops = []
    monkeypatch.setattr(policy, "ProcessTable", lambda: table)
    monkeypatch.setattr(policy, "recorded_launch_identity",
                        lambda _process: LaunchIdentity(LEADER.pid, LEADER.start, "test-boot"))
    if failure_limit is not None:
        monkeypatch.setattr(policy, "PROCFS_SAMPLE_MAX_CONSECUTIVE_FAILURES", failure_limit)

    class Supervisor(policy._ResidentMemorySupervisor):
        def _stop_launch_unit(self):
            stops.append(self.error)
            table.done.set()

    supervisor = Supervisor(process, 128 << 20)
    supervisor.start()
    assert table.done.wait(10)
    supervisor._stopped.set()
    supervisor._thread.join(5)
    return process, supervisor, stops


@pytest.mark.parametrize("race", RACES, ids=lambda error: type(error).__name__)
def test_injected_procfs_race_does_not_kill_live_check(monkeypatch, race):
    process, supervisor, stops = _supervise(monkeypatch, [race, race, race])

    assert stops == []
    assert supervisor.error is None
    assert process.memory_limit_killed is False
    assert process.memory_limit_samples_skipped == 3
    assert type(race).__name__ in process.memory_limit_sample_skip_reason


def test_positive_over_budget_measurement_still_kills(monkeypatch):
    process, _supervisor, stops = _supervise(
        monkeypatch, [RACES[0]], breach="aggregate resident-memory limit exceeded")

    assert stops == [None]
    assert process.memory_limit_stop_cause == "memory_limit"
    assert process.memory_limit_killed is True
    assert process.memory_limit_reason == "aggregate resident-memory limit exceeded"
    assert process.memory_limit_samples_skipped == 1


def test_non_race_procfs_error_still_fails_closed(monkeypatch):
    _process, supervisor, stops = _supervise(
        monkeypatch, [PermissionError(errno.EACCES, "Permission denied")])

    assert len(stops) == 1
    assert "PermissionError" in supervisor.error


def test_persistent_unreadable_samples_stop_failing_open(monkeypatch):
    _process, supervisor, stops = _supervise(
        monkeypatch, [RACES[0]] * 10_000, failure_limit=5)

    assert len(stops) == 1
    assert "ProcessLookupError" in supervisor.error
    assert _process.memory_limit_stop_cause.startswith("supervisor_error: ")
    assert _process.memory_limit_killed is False


def test_long_run_of_races_is_bounded_by_count_not_time(monkeypatch):
    process, _supervisor, stops = _supervise(monkeypatch, [RACES[2]] * 40, failure_limit=40)

    assert stops == []
    assert process.memory_limit_samples_skipped == 40


def test_receipt_records_skipped_samples(monkeypatch):
    class Process:
        memory_limit_samples_skipped = 2
        memory_limit_sample_skip_reason = "procfs sample raced process state (ValueError)"

    assert int(getattr(Process(), "memory_limit_samples_skipped", 0)) == 2
    assert {"memory_limit_samples_skipped", "memory_limit_sample_skip_reason"} <= {
        field.name for field in policy.PolicyResult.__dataclass_fields__.values()}
    from kernel_wire_identity import WIRE_AUDIT_ONLY_FIELDS
    assert {"memory_limit_samples_skipped", "memory_limit_sample_skip_reason"} <= WIRE_AUDIT_ONLY_FIELDS


class _SlowExitTable:
    """Post-exit procfs that is slow (well past the old 15 s cap, scaled) and racy."""

    def __init__(self, delay, races):
        self.delay = delay
        self.races = list(races)

    def identity(self, _pid):
        return None

    @staticmethod
    def pid_namespace_inode(_pid):
        return None

    def refresh_members(self, _members, **_kwargs):
        time.sleep(self.delay)
        if self.races:
            raise self.races.pop(0)
        return {}

    def snapshot(self):
        time.sleep(self.delay)
        return {}

    @staticmethod
    def resident_memory_limit_breach(_live, _memory_bytes):
        return ""


def _exited_supervisor(monkeypatch, table):
    class Process:
        pid = LEADER.pid
        returncode = 0

    process = Process()
    monkeypatch.setattr(policy, "ProcessTable", lambda: table)
    monkeypatch.setattr(policy, "recorded_launch_identity",
                        lambda _process: LaunchIdentity(LEADER.pid, None, "test-boot"))
    return process, policy._ResidentMemorySupervisor(process, 128 << 20)


def test_slow_and_erroring_post_exit_procfs_verifies_without_failing(monkeypatch):
    # Each sample takes longer than the old 2 s idle-stall cutoff would allow
    # between progress marks once the race retries are added up.
    process, supervisor = _exited_supervisor(
        monkeypatch, _SlowExitTable(0.4, [RACES[0], RACES[2], RACES[4]] * 3))
    supervisor.start()

    assert supervisor.finish() is True
    assert supervisor.finish_error is None
    assert getattr(process, "memory_limit_stop_cause", None) is None
    assert process.memory_limit_samples_skipped == 9


def test_stalled_exit_scan_is_waited_out_not_failed(monkeypatch):
    release = threading.Event()

    class StalledTable(_SlowExitTable):
        def refresh_members(self, _members, **_kwargs):
            release.wait()
            return {}

    # Any former wall-clock cap (15 s) is shrunk below the stall so this test proves
    # completion no longer depends on elapsed time.
    monkeypatch.setattr(policy, "RSS_FINISH_MAX_WAIT_SECONDS", 1.0, raising=False)
    _process, supervisor = _exited_supervisor(monkeypatch, StalledTable(0, []))
    supervisor.start()
    timer = threading.Timer(3.0, release.set)
    timer.start()
    try:
        started = time.monotonic()
        assert supervisor.finish() is True
        assert time.monotonic() - started >= 2.5
        assert supervisor.finish_error is None
    finally:
        release.set()
        timer.cancel()
