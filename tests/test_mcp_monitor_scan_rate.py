"""MCP resource monitoring avoids repeated full host process-table scans."""
from __future__ import annotations

import sys
import time as real_time
from types import SimpleNamespace

import pytest

import kernel_mcp_driver as driver
from kernel_exec_tracker import LaunchIdentity, ProcessIdentity


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux MCP supervision")

SERVER, PRIVATE_CHILD, DETACHED_PRIVATE = 91001, 91002, 91003


def _monitor(monkeypatch, *, snapshot_delay=0.0, clock=None):
    server = ProcessIdentity(SERVER, 1, SERVER, SERVER, (10, 0), False)
    child = ProcessIdentity(PRIVATE_CHILD, SERVER, PRIVATE_CHILD, PRIVATE_CHILD,
                            (11, 0), False)

    class Table:
        def __init__(self):
            self.rows = {SERVER: server, PRIVATE_CHILD: child}
            self.namespaces = {SERVER: 10, PRIVATE_CHILD: 20}
            self.resident = {SERVER: 1, PRIVATE_CHILD: 1}
            self.snapshots = []
            self.scan_spans = []
            self.refresh_times = []
            self.refreshes = 0
            self.last_live = set()

        def identity(self, pid):
            return self.rows.get(pid)

        def pid_namespace(self, pid):
            return "private" if self.namespaces.get(pid) == 20 else "host"

        def pid_namespace_inode(self, pid):
            return self.namespaces.get(pid, 10)

        def refresh_members(self, members, *, namespace_inode=None):
            self.refreshes += 1
            self.refresh_times.append(
                clock[0] if clock is not None else real_time.monotonic())
            return {pid: self.rows[pid] for pid in members if pid in self.rows}

        def snapshot(self):
            started = clock[0] if clock is not None else real_time.monotonic()
            self.snapshots.append(started)
            if snapshot_delay:
                if clock is not None:
                    clock[0] += snapshot_delay
                else:
                    real_time.sleep(snapshot_delay)
            ended = clock[0] if clock is not None else real_time.monotonic()
            self.scan_spans.append((started, ended))
            return self.rows.copy()

        def limit_breach(self, live, limits):
            self.last_live = {row.pid for row in live}
            if sum(self.resident.get(row.pid, 0) for row in live) > limits.memory_bytes:
                return "aggregate resident-memory limit exceeded"
            if len(live) > limits.nproc:
                return "process limit exceeded"
            return ""

    table = Table()
    process = SimpleNamespace(
        pid=SERVER,
        returncode=None,
        exited=False,
        _stateframe_launch_identity=LaunchIdentity(SERVER, server.start, "boot"),
    )
    monkeypatch.setattr(driver, "ProcessTable", lambda: table)
    if clock is not None:
        monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monitor = driver._ContainmentMonitor(
        process,
        SimpleNamespace(limits=SimpleNamespace(wall_seconds=60, memory_bytes=100, nproc=20)),
    )
    return monitor, table, process


def test_full_scan_cadence_starts_one_second_after_completion_and_leader_exit(monkeypatch):
    clock = [100.0]
    monitor, table, process = _monitor(monkeypatch, snapshot_delay=0.02, clock=clock)

    assert monitor._sample() == ""
    assert monitor.namespace == "private"
    assert PRIVATE_CHILD in monitor.family.members
    assert table.snapshots == [100.0]

    for offset in (0.1, 0.4, 0.99):
        clock[0] = 100.0 + offset
        assert monitor._sample() == ""
    assert table.snapshots == [100.0]

    clock[0] = 101.02
    assert monitor._sample() == ""
    assert table.snapshots == [100.0, 101.02]

    # This process is already inside the private namespace but is detached from known parents.
    table.rows[DETACHED_PRIVATE] = ProcessIdentity(
        DETACHED_PRIVATE, 999, 900, 900, (12, 0), False)
    table.namespaces[DETACHED_PRIVATE] = 20
    table.resident[DETACHED_PRIVATE] = 1

    class StopAfterExit:
        def is_set(self):
            return False

        def wait(self, _timeout):
            clock[0] += 0.01
            process.exited = True

        def set(self):
            process.exited = True

    monitor.stopped = StopAfterExit()
    monkeypatch.setattr(driver, "child_exited_unreaped", lambda _process: process.exited)
    monitor._run()

    assert table.snapshots == [100.0, 101.02, 101.05]
    assert DETACHED_PRIVATE in monitor.family.members
    assert DETACHED_PRIVATE in table.last_live


def test_slow_full_scans_are_spaced_from_completion_and_refresh_known_members(monkeypatch):
    clock = [50.0]
    monitor, table, _process = _monitor(
        monkeypatch, snapshot_delay=1.2, clock=clock)

    monitor._sample()
    for _ in range(8):
        clock[0] += 0.25
        monitor._sample()

    assert len(table.scan_spans) == 3
    for previous, following in zip(table.scan_spans, table.scan_spans[1:]):
        assert following[0] - previous[1] >= 1.0 - 1e-9
        assert any(previous[1] < checked < following[0]
                   for checked in table.refresh_times)


def test_known_member_memory_breach_is_checked_without_waiting_for_slow_scan(monkeypatch):
    monitor, table, _process = _monitor(monkeypatch, snapshot_delay=0.25)

    assert monitor._sample() == ""
    assert table.snapshots and PRIVATE_CHILD in monitor.family.members
    table.resident[PRIVATE_CHILD] = 101

    started = real_time.monotonic()
    breach = monitor._sample()
    elapsed = real_time.monotonic() - started

    assert breach == "aggregate resident-memory limit exceeded"
    assert elapsed < 0.1
    assert len(table.snapshots) == 1


def test_verified_monitor_kill_is_not_swept_twice(monkeypatch):
    process = SimpleNamespace(
        pid=SERVER,
        _stateframe_launch_identity=LaunchIdentity(SERVER, (10, 0), "boot"),
    )
    calls = []
    monkeypatch.setattr(driver, "kill_launch_unit",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or ())
    monitor = driver._ContainmentMonitor.__new__(driver._ContainmentMonitor)
    monitor.process = process
    monitor.family = SimpleNamespace()
    monitor.failure = None
    monitor.launch_unit_terminated = False
    monitor.stopped = SimpleNamespace(is_set=lambda: False)
    monitor._sample = lambda: "aggregate resident-memory limit exceeded"

    monitor._run()

    assert monitor.launch_unit_terminated is True
    assert len(calls) == 1
    assert driver._cleanup_launch_unit(process, monitor) == ()
    assert len(calls) == 1
