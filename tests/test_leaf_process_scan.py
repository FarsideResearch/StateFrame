"""Periodic procfs scans retain private-namespace escapees after reparenting."""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest

import kernel_execution_policy as policy
import kernel_exec_supervisor
from kernel_exec_tracker import ProcessIdentity


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision")
LIMIT = 32 << 20


def _rows():
    leader = ProcessIdentity(100, 1, 100, 100, (10, 0), False)
    namespace_init = ProcessIdentity(101, 100, 100, 100, (11, 0), False)
    escaped = ProcessIdentity(102, 101, 102, 102, (12, 0), False)
    return leader, namespace_init, escaped


def test_memory_supervisor_scans_private_namespace_immediately(
        monkeypatch):
    leader, namespace_init, escaped = _rows()

    class Table:
        snapshots = 0
        identities = {row.pid: row for row in (leader, namespace_init, escaped)}
        namespaces = {100: 10, 101: 20, 102: 20}
        resident = {100: 1 << 20, 101: 1 << 20, 102: 2 * LIMIT}

        def identity(self, pid):
            return self.identities.get(pid)

        def pid_namespace_inode(self, pid):
            return self.namespaces.get(pid)

        def refresh_members(self, _members, *, namespace_inode=None):
            return {leader.pid: leader, namespace_init.pid: namespace_init}

        def snapshot(self):
            self.snapshots += 1
            return self.identities.copy()

        def resident_memory_limit_breach(self, live, _memory_bytes):
            return ("aggregate resident-memory limit exceeded"
                    if sum(self.resident.get(row.pid, 0) for row in live) > LIMIT else "")

    table = Table()

    class StopAfterOneSample:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _timeout):
            self.stopped = True

        def set(self):
            self.stopped = True

    process = SimpleNamespace(pid=leader.pid, returncode=None)
    monkeypatch.setattr(policy, "ProcessTable", lambda: table)
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda _process: SimpleNamespace(
        pid=leader.pid, start=leader.start))
    supervisor = policy._ResidentMemorySupervisor(process, LIMIT)
    # The first sample is immediate even when the regular cadence is not due.
    supervisor._stopped = StopAfterOneSample()
    supervisor._stop_launch_unit = lambda: None

    supervisor._run()

    assert table.snapshots == 1
    assert supervisor.family.private_pid_namespace_inode == 20
    assert escaped.pid in supervisor.family.members
    assert process.returncode is None
    assert process.memory_limit_killed is True
    assert process.memory_limit_reason == "aggregate resident-memory limit exceeded"


def test_memory_supervisor_scans_and_cleans_after_leader_exit(monkeypatch):
    escaped = ProcessIdentity(102, 1, 102, 102, (12, 0), False)

    class Table:
        snapshots = 0

        def pid_namespace_inode(self, pid):
            return 20 if pid == escaped.pid else None

        def refresh_members(self, _members, *, namespace_inode=None):
            return {}

        def snapshot(self):
            self.snapshots += 1
            return {escaped.pid: escaped}

        def resident_memory_limit_breach(self, live, _memory_bytes):
            return ("aggregate resident-memory limit exceeded"
                    if escaped in live else "")

    class Family:
        members = {}
        private_pid_namespace_inode = 20

        def __init__(self, *_args):
            self.reaped = False

        def leader_reaped(self):
            self.reaped = True

        def update(self, rows):
            return [escaped] if escaped.pid in rows else []

    table = Table()
    process = SimpleNamespace(pid=100, returncode=0)
    stopped = []
    monkeypatch.setattr(policy, "ProcessTable", lambda: table)
    monkeypatch.setattr(policy, "Descendants", Family)
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda _process: object())
    supervisor = policy._ResidentMemorySupervisor(process, LIMIT)
    supervisor._next_full_process_scan = time.monotonic() + 5
    supervisor._stop_launch_unit = lambda: stopped.append(True)

    supervisor._run()

    assert table.snapshots == 1
    assert supervisor.family.reaped
    assert process.memory_limit_killed is True
    assert stopped == [True]


def test_memory_supervisor_keeps_private_scan_on_one_second_cadence(monkeypatch):
    row = ProcessIdentity(100, 1, 100, 100, (10, 0), False)

    class Table:
        snapshots = 0
        refreshes = 0

        def refresh_members(self, _members, *, namespace_inode=None):
            self.refreshes += 1
            return {row.pid: row}

        def snapshot(self):
            self.snapshots += 1
            return {row.pid: row}

        def resident_memory_limit_breach(self, _live, _memory_bytes):
            return ""

    class Family:
        members = {row.pid: row}
        private_pid_namespace_inode = 20

        def __init__(self, *_args):
            pass

        def update(self, _rows):
            return [row]

        def leader_reaped(self):
            pass

    class StopAfterTwoSamples:
        def __init__(self):
            self.waits = 0
            self.stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _timeout):
            self.waits += 1
            self.stopped = self.waits >= 2

        def set(self):
            self.stopped = True

    table = Table()
    process = SimpleNamespace(pid=row.pid, returncode=None)
    monkeypatch.setattr(policy, "ProcessTable", lambda: table)
    monkeypatch.setattr(policy, "Descendants", Family)
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda _process: object())
    supervisor = policy._ResidentMemorySupervisor(process, LIMIT)
    supervisor._next_full_process_scan = time.monotonic() - 1
    supervisor._stopped = StopAfterTwoSamples()

    supervisor._run()

    assert table.refreshes == 2
    assert table.snapshots == 1


def test_memory_supervisor_schedules_from_scan_completion(monkeypatch):
    row = ProcessIdentity(100, 1, 100, 100, (10, 0), False)
    clock = [0.0]
    fake_time = SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _delay: None)

    class Table:
        def __init__(self):
            self.scan_spans = []
            self.refresh_times = []

        def refresh_members(self, _members, *, namespace_inode=None):
            self.refresh_times.append(clock[0])
            return {row.pid: row}

        def snapshot(self):
            started = clock[0]
            clock[0] += 1.2
            self.scan_spans.append((started, clock[0]))
            return {row.pid: row}

        def resident_memory_limit_breach(self, _live, _memory_bytes):
            return ""

    class Family:
        members = {row.pid: row}
        private_pid_namespace_inode = 20

        def __init__(self, *_args):
            pass

        def update(self, _rows):
            return [row]

        def leader_reaped(self):
            pass

    class StopAfterSamples:
        def __init__(self):
            self.waits = 0
            self.stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _timeout):
            clock[0] += 0.25
            self.waits += 1
            self.stopped = self.waits >= 9

        def set(self):
            self.stopped = True

    table = Table()
    process = SimpleNamespace(pid=row.pid, returncode=None)
    monkeypatch.setattr(policy, "time", fake_time)
    monkeypatch.setattr(policy, "ProcessTable", lambda: table)
    monkeypatch.setattr(policy, "Descendants", Family)
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda _process: object())
    supervisor = policy._ResidentMemorySupervisor(process, LIMIT)
    supervisor._stopped = StopAfterSamples()

    supervisor._run()

    assert len(table.scan_spans) == 3
    for previous, following in zip(table.scan_spans, table.scan_spans[1:]):
        assert following[0] - previous[1] >= 1.0 - 1e-9
        assert any(previous[1] < checked < following[0]
                   for checked in table.refresh_times)


def test_exec_supervisor_scans_private_namespace_at_one_second_interval(monkeypatch):
    leader, namespace_init, escaped = _rows()

    class Table:
        def __init__(self):
            self.snapshots = 0
            self.refreshes = []

        def refresh_members(self, members, *, namespace_inode=None):
            self.refreshes.append((members, namespace_inode))
            return {leader.pid: leader, namespace_init.pid: namespace_init}

        def snapshot(self):
            self.snapshots += 1
            return {row.pid: row for row in (leader, namespace_init, escaped)}

    table = Table()
    supervisor = object.__new__(kernel_exec_supervisor.Supervisor)
    supervisor.table = table
    supervisor._next_full_process_scan = time.monotonic() - 1
    job = SimpleNamespace(
        cleanup=None,
        family=SimpleNamespace(
            members={leader.pid: leader, namespace_init.pid: namespace_init},
            private_pid_namespace_inode=20,
        ),
    )
    supervisor.jobs = {"private": job}

    rows = supervisor._sample_rows()

    assert table.refreshes == [(job.family.members, 20)]
    assert table.snapshots == 1
    assert escaped.pid in rows
    assert supervisor._sample_rows()
    assert table.snapshots == 1


def test_exec_supervisor_schedules_from_scan_completion(monkeypatch):
    row = ProcessIdentity(100, 1, 100, 100, (10, 0), False)
    clock = [0.0]

    class Table:
        def __init__(self):
            self.scan_spans = []
            self.refresh_times = []

        def refresh_members(self, members, *, namespace_inode=None):
            self.refresh_times.append(clock[0])
            return {row.pid: row}

        def snapshot(self):
            started = clock[0]
            clock[0] += 1.2
            self.scan_spans.append((started, clock[0]))
            return {row.pid: row}

    table = Table()
    supervisor = object.__new__(kernel_exec_supervisor.Supervisor)
    supervisor.table = table
    supervisor._next_full_process_scan = 0.0
    job = SimpleNamespace(
        cleanup=None,
        initial_scan_pending=False,
        post_exit_scan_pending=False,
        family=SimpleNamespace(
            members={row.pid: row}, private_pid_namespace_inode=20),
    )
    supervisor.jobs = {"slow-scan": job}
    monkeypatch.setattr(
        kernel_exec_supervisor, "time",
        SimpleNamespace(monotonic=lambda: clock[0]))

    for _ in range(20):
        supervisor._sample_rows()
        if len(table.scan_spans) == 3:
            break
        clock[0] += 0.25

    assert len(table.scan_spans) == 3
    for previous, following in zip(table.scan_spans, table.scan_spans[1:]):
        assert following[0] - previous[1] >= 1.0 - 1e-9
        assert any(previous[1] < checked < following[0]
                   for checked in table.refresh_times)


def test_exec_supervisor_forces_initial_and_post_exit_scans(monkeypatch):
    row = ProcessIdentity(100, 1, 100, 100, (10, 0), False)

    class Table:
        snapshots = 0

        def refresh_members(self, _members, *, namespace_inode=None):
            return {}

        def snapshot(self):
            self.snapshots += 1
            return {row.pid: row}

    table = Table()
    supervisor = object.__new__(kernel_exec_supervisor.Supervisor)
    supervisor.table = table
    supervisor._next_full_process_scan = time.monotonic() + 5
    job = SimpleNamespace(
        cleanup=None,
        initial_scan_pending=True,
        post_exit_scan_pending=False,
        post_exit_scan_complete=False,
        family=SimpleNamespace(members={}, private_pid_namespace_inode=None),
    )
    supervisor.jobs = {"fast": job}

    supervisor._sample_rows()
    assert table.snapshots == 1
    assert not job.initial_scan_pending

    job.post_exit_scan_pending = True
    supervisor._sample_rows()
    assert table.snapshots == 2
    assert not job.post_exit_scan_pending
    assert job.post_exit_scan_complete
