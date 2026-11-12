"""Regressions for cleanup authority held across child exit and PID reuse."""

import errno
import io
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from _timeouts import scaled
import kernel_exec_tracker as tracker
import kernel_mcp_driver as mcp
from kernel_exec_session import ExecSessionError
from stateframe import runner
from stateframe import outer_checks


def test_pidfd_open_failure_keeps_launch_start_identity(monkeypatch):
    row = tracker.ProcessIdentity(42100, 1, 42100, 42100, (100, 0), False)

    class Table:
        def identity(self, pid):
            assert pid == row.pid
            return row

    class Process:
        pid = row.pid

    def denied(_pid):
        raise OSError(errno.ENOSYS, "pidfd disabled")

    monkeypatch.setattr(tracker.sys, "platform", "linux")
    monkeypatch.setattr(tracker, "_boot_id", lambda: "boot-a")
    monkeypatch.setattr(tracker.os, "pidfd_open", denied, raising=False)
    process = Process()
    identity = tracker.capture_launch_identity(process, Table())
    assert identity == tracker.LaunchIdentity(row.pid, row.start, "boot-a", None)
    assert tracker.recorded_launch_identity(process) == identity


def test_fast_exited_waitable_leader_signals_group_without_pidfd(monkeypatch):
    row = tracker.ProcessIdentity(42100, 1, 42100, 42100, (100, 0), True)

    class Table:
        def identity(self, pid):
            assert pid == row.pid
            return row

    class Process:
        pid = row.pid
        returncode = None
        _waitpid_lock = threading.Lock()

    calls = []
    monkeypatch.setattr(tracker.sys, "platform", "linux")
    monkeypatch.setattr(tracker, "_boot_id", lambda: "boot-a")
    monkeypatch.setattr(tracker.os, "pidfd_open", lambda _pid: (_ for _ in ()).throw(
        OSError(errno.ENOSYS, "pidfd disabled")), raising=False)
    monkeypatch.setattr(tracker.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    process = Process()
    identity = tracker.capture_launch_identity(process, Table())
    assert tracker.signal_launch_group(process, identity, signal.SIGKILL, table=Table())
    assert calls == [(row.pid, signal.SIGKILL)]


def test_mcp_monitor_rejects_recycled_leader_late_adoption(monkeypatch):
    old = tracker.ProcessIdentity(42100, 1, 42100, 42100, (100, 0), False)
    reused = tracker.ProcessIdentity(42100, 1, 42100, 42100, (101, 0), False)

    class Table:
        def identity(self, _pid):
            return None

        def snapshot(self):
            return {reused.pid: reused}

        def limit_breach(self, rows, _limits):
            assert rows == []
            return ""

    process = SimpleNamespace(pid=old.pid,
                              _stateframe_launch_identity=tracker.LaunchIdentity(
                                  old.pid, old.start, tracker._boot_id()))
    monitor = mcp._ContainmentMonitor.__new__(mcp._ContainmentMonitor)
    monitor.process = process
    monitor.launch_identity = tracker.recorded_launch_identity(process)
    monitor.table = Table()
    monitor.family = tracker.Descendants(old.pid, monitor.table, monitor.launch_identity)
    monitor.cap = SimpleNamespace(limits=SimpleNamespace(wall_seconds=60))
    monitor.started = time.monotonic()
    monitor.namespace = None
    monkeypatch.setattr(mcp.sys, "platform", "darwin")
    assert monitor._sample() == ""
    assert monitor.family.members == {}


def test_mcp_monitor_refuses_missing_launch_identity():
    with pytest.raises(ExecSessionError, match="cannot capture MCP launch identity"):
        mcp._ContainmentMonitor(SimpleNamespace(pid=42100),
                                SimpleNamespace(limits=SimpleNamespace(wall_seconds=60)))


def test_family_construction_rechecks_identity_under_wait_lock(monkeypatch):
    old = tracker.ProcessIdentity(42100, 1, 42100, 42100, (100, 0), False)
    reused = tracker.ProcessIdentity(42100, 1, 42100, 42100, (101, 0), False)

    class Table:
        def __init__(self):
            self.lookups = 0

        def identity(self, _pid):
            self.lookups += 1
            return old if self.lookups == 1 else reused

        def snapshot(self):
            return {old.pid: reused}

    class Process:
        pid = old.pid
        returncode = None
        _waitpid_lock = threading.Lock()

        def kill(self):
            self.returncode = -9

        def wait(self, timeout):
            return self.returncode

        def poll(self):
            return self.returncode

    table = Table()
    monkeypatch.setattr(tracker, "ProcessTable", lambda: table)
    signals = []
    monkeypatch.setattr(tracker.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    identity = tracker.LaunchIdentity(old.pid, old.start, tracker._boot_id())
    with pytest.raises(tracker.ContainmentUnverified, match="survivors unknown"):
        tracker.kill_launch_unit(Process(), launch_identity=identity)
    assert signals == []
    assert table.lookups >= 2


def _alive(pid):
    row = tracker.ProcessTable().identity(pid)
    return row is not None and not row.zombie


@pytest.mark.parametrize("cleanup", ["runner", "policy"])
def test_exited_leader_descendant_is_stopped_before_reap(cleanup):
    from stateframe.engine import kernel_exec_tracker as package_tracker

    launch_tracker = package_tracker if cleanup == "runner" else tracker
    code = ("import os,sys,time\n"
            "child=os.fork()\n"
            "if child == 0:\n"
            "    time.sleep(30)\n"
            "    os._exit(0)\n"
            "print(child, flush=True)\n")
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                               start_new_session=True)
    assert launch_tracker.capture_launch_identity(process) is not None
    child = int(process.stdout.readline())
    try:
        until = time.monotonic() + scaled(2)
        while not launch_tracker.child_exited_unreaped(process) and time.monotonic() < until:
            time.sleep(.01)
        assert launch_tracker.child_exited_unreaped(process)
        assert _alive(child)
        if cleanup == "runner":
            runner._stop_group(process)
        else:
            from kernel_execution_policy import _kill_group
            try:
                assert _kill_group(process)
            except tracker.ContainmentUnverified as error:
                assert "descendant family unavailable" in str(error)
        until = time.monotonic() + scaled(2)
        while _alive(child) and time.monotonic() < until:
            time.sleep(.01)
        assert not _alive(child)
    finally:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdout.close()


@pytest.mark.parametrize("survives", [False, True])
def test_runner_drive_cleans_group_before_reaping_exited_leader(tmp_path, monkeypatch, survives):
    from stateframe.engine import kernel_exec_tracker as package_tracker
    from stateframe import shell_host

    order = []

    class Engine:
        returncode = None

        def __init__(self):
            self.stdout = io.StringIO("")

        def poll(self):
            return 0

        def wait(self, timeout):
            order.append("wait")
            assert "stop" in order
            self.returncode = 0
            return 0

    engine = Engine()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(package_tracker, "capture_launch_identity", lambda _process, **_kwargs: None)
    monkeypatch.setattr(package_tracker, "child_exited_unreaped", lambda _process: True)
    monkeypatch.setattr(runner, "_remaining", lambda _options: 10)
    monkeypatch.setattr(runner, "_child_env", lambda _options: {})
    monkeypatch.setattr(runner, "_kill_deadline", lambda *_args: float("inf"))
    monkeypatch.setattr(runner, "engine_reads_test_python", lambda: False)
    monkeypatch.setattr(shell_host, "answer_pending", lambda *_args: None)

    def stop(_process):
        order.append("stop")
        if not survives:
            engine.returncode = 0

    monkeypatch.setattr(runner, "_stop_group", stop)
    options = SimpleNamespace(timeout_s=10, _deadline=None, python=sys.executable,
                              shell_confirm=None, quiet=True)
    if survives:
        with pytest.raises(RuntimeError, match="engine survived process-group cleanup"):
            runner.drive(tmp_path, options)
    else:
        assert runner.drive(tmp_path, options) == 0
    assert order == ["stop", "stop"]


def test_outer_check_parent_has_no_transmitted_group_signal():
    source = (Path(__file__).resolve().parents[1] / "src/stateframe/outer_checks.py").read_text()
    assert '"check_pid"' not in source
    assert '"cleanup_ok"' in source
    assert "signal_launch_group(process, recorded_launch_identity(process)" in source


@pytest.mark.parametrize("ack,expected", [
    (False, "containment unverified"),
    (True, "outer check refused: failed check"),
])
def test_outer_check_requires_cleanup_ack_even_on_helper_error(tmp_path, monkeypatch, ack, expected):
    from stateframe.engine import kernel_exec_tracker as package_tracker

    def reader(payload):
        read_fd, write_fd = os.pipe()
        os.write(write_fd, payload)
        os.close(write_fd)
        return os.fdopen(read_fd, "rb")

    class Helper:
        pid = 42100
        returncode = 2

        def __init__(self):
            self.stdout = reader((b'{"error":"failed check"}\n'
                                  + b'{"cleanup_ok":' + (b'true' if ack else b'false') + b'}\n'))
            self.stderr = reader(b"")

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr(outer_checks.subprocess, "Popen", lambda *_args, **_kwargs: Helper())
    monkeypatch.setattr(package_tracker, "capture_launch_identity", lambda _process, **_kwargs: None)
    with pytest.raises(outer_checks.OuterCheckError, match=expected):
        outer_checks.run("python -m pytest", tmp_path, python=sys.executable)
