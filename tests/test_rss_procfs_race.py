"""A procfs race while sampling a live check must never kill it."""
from __future__ import annotations

import errno
import threading

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


def _supervise(monkeypatch, races, *, breach="", race_window=None):
    process = _Process()
    table = _Table(process, races, breach=breach)
    stops = []
    monkeypatch.setattr(policy, "ProcessTable", lambda: table)
    monkeypatch.setattr(policy, "recorded_launch_identity",
                        lambda _process: LaunchIdentity(LEADER.pid, LEADER.start, "test-boot"))
    if race_window is not None:
        monkeypatch.setattr(policy, "PROCFS_SAMPLE_RACE_MAX_SECONDS", race_window)

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
        monkeypatch, [RACES[0]] * 10_000, race_window=0.05)

    assert len(stops) == 1
    assert "ProcessLookupError" in supervisor.error


def test_receipt_records_skipped_samples(monkeypatch):
    class Process:
        memory_limit_samples_skipped = 2
        memory_limit_sample_skip_reason = "procfs sample raced process state (ValueError)"

    assert int(getattr(Process(), "memory_limit_samples_skipped", 0)) == 2
    assert {"memory_limit_samples_skipped", "memory_limit_sample_skip_reason"} <= {
        field.name for field in policy.PolicyResult.__dataclass_fields__.values()}
    from kernel_wire_identity import WIRE_AUDIT_ONLY_FIELDS
    assert {"memory_limit_samples_skipped", "memory_limit_sample_skip_reason"} <= WIRE_AUDIT_ONLY_FIELDS
