"""Stress partial launch identity through the RLIMIT_DATA RSS fallback."""
from __future__ import annotations

import errno
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

import kernel_execution_policy as policy
import kernel_exec_tracker as tracker
from kernel_exec_tracker import (
    ContainmentUnverified,
    Descendants,
    LaunchIdentity,
    ProcessIdentity,
    kill_launch_unit,
)


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux RSS supervision")


class _ErrnoOSError(OSError):
    """Generic OSError subclass avoids Python's errno-specific exception factory."""


def _identity_isolation(command, _root, environment, **_options):
    return list(command), dict(environment)


def test_fast_completion_race_stress(tmp_path, monkeypatch):
    """Clean fast exits survive partial identity capture and delayed RSS scans."""
    real_table_type = policy.ProcessTable
    real_supervisor_type = policy._ResidentMemorySupervisor
    report = policy.CapabilityReport("Linux", "test", True, True, True, True, True, True)
    monkeypatch.setattr(policy, "require_capabilities", lambda: report)
    monkeypatch.setattr(policy, "_command_cgroup", lambda *_args, **_kwargs: None)
    original_popen = policy.sandboxed_popen

    def without_namespace(command, cwd, **options):
        return original_popen(command, cwd, isolation_builder=_identity_isolation, **options)

    monkeypatch.setattr(policy, "sandboxed_popen", without_namespace)

    full_stress = os.environ.get("STATEFRAME_RSS_RACE_STRESS") == "1"
    iterations = 200 if full_stress else 8
    slow_scan_limit = 8 if full_stress else 2
    state: dict[str, object] = {
        "hide_identity": True,
        "leader_pid": None,
        "slow_scans": 0,
    }
    state_lock = threading.Lock()
    supervisors = []

    class ProcessTableView:
        def __init__(self):
            self.table = real_table_type()

        def identity(self, pid):
            with state_lock:
                if state["leader_pid"] is None:
                    state["leader_pid"] = pid
                if state["hide_identity"]:
                    state["hide_identity"] = False
                    return None
            return self.table.identity(pid)

        def snapshot(self):
            rows = self.table.snapshot()
            if threading.current_thread().name == "sandbox-resident-memory":
                delay = False
                with state_lock:
                    if state["slow_scans"] < slow_scan_limit:
                        state["slow_scans"] += 1
                        delay = True
                if delay:
                    # Let the check exit while its supervisor holds the scan result.
                    time.sleep(0.5)
            return rows

        def resident_memory_limit_breach(self, live, memory_bytes):
            return self.table.resident_memory_limit_breach(live, memory_bytes)

    class ObservedSupervisor(real_supervisor_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            supervisors.append(self)

    monkeypatch.setattr(policy, "ProcessTable", ProcessTableView)
    monkeypatch.setattr(tracker, "ProcessTable", ProcessTableView)
    monkeypatch.setattr(policy, "_ResidentMemorySupervisor", ObservedSupervisor)

    stop_load = threading.Event()

    def load_worker():
        value = 0
        while not stop_load.is_set():
            for number in range(100):
                value = (value * 33 + number) & 0xFFFFFF
            stop_load.wait(0.01)

    workers = [threading.Thread(target=load_worker, daemon=True) for _ in range(2)]
    for worker in workers:
        worker.start()

    failures = []
    try:
        for _index in range(iterations):
            with state_lock:
                state.update({"hide_identity": True, "leader_pid": None})
            try:
                result = policy.run_sandboxed(
                    [sys.executable, "-c", "pass"], tmp_path,
                    timeout=5, cpu_seconds=5, memory_bytes=128 << 20,
                    file_bytes=64 << 20, nproc=32,
                )
            except ContainmentUnverified as error:
                failures.append(str(error))
                if supervisors:
                    supervisors.pop()
                continue

            assert result.completed.returncode == 0, result.completed.stderr
            assert result.memory_limit_mechanism == "rlimit_data+rss_supervisor"
            supervisor = supervisors.pop()
            assert not supervisor.error, supervisor.error

            # Exercise the supervisor's post-exit cleanup boundary after the runner
            # has reaped the leader. A clean, empty launch unit is verifiable even
            # when the spawn-time procfs identity read was transiently unavailable.
            supervisor._stop_launch_unit()
            if supervisor.error:
                failures.append(supervisor.error)
    finally:
        stop_load.set()
        for worker in workers:
            worker.join(timeout=1)

    rate = f"{len(failures)}/{iterations} clean checks"
    assert not failures, f"{rate}; first error: {failures[:1]}"


@pytest.mark.parametrize("_attempt", range(10), ids=lambda attempt: f"attempt-{attempt + 1}")
def test_run_tests_clean_exit_while_rss_scan_blocked_preserves_family(
        tmp_path, monkeypatch, _attempt):
    """A slow post-exit scan must not turn a clean check into unknown containment."""
    real_supervisor_type = policy._ResidentMemorySupervisor
    real_collect = policy._collect
    popen = policy.sandboxed_popen
    report = policy.CapabilityReport("Linux", "test", True, True, True, True, True, True)
    scan_started = threading.Event()
    release_scan = threading.Event()
    collected = threading.Event()
    finish_started = threading.Event()
    completed = threading.Event()
    outcome: dict[str, object] = {}
    supervisors = []

    monkeypatch.setattr(policy, "require_capabilities", lambda: report)
    monkeypatch.setattr(policy, "_command_cgroup", lambda *_args, **_kwargs: None)

    if policy.sandbox_executable("bwrap") is None:
        def without_namespace(command, cwd, **options):
            return popen(command, cwd, isolation_builder=_identity_isolation, **options)

        monkeypatch.setattr(policy, "sandboxed_popen", without_namespace)

    def observed_collect(*args, **kwargs):
        result = real_collect(*args, **kwargs)
        collected.set()
        return result

    monkeypatch.setattr(policy, "_collect", observed_collect)

    class GatedSupervisor(real_supervisor_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            supervisors.append(self)
            real_snapshot = self.table.snapshot
            blocked = False

            def gated_snapshot():
                nonlocal blocked
                if (not blocked
                        and threading.current_thread().name == "sandbox-resident-memory"):
                    blocked = True
                    scan_started.set()
                    if not release_scan.wait(10):
                        raise RuntimeError("test did not release blocked RSS scan")
                return real_snapshot()

            self.table.snapshot = gated_snapshot

        def finish(self, *args, **kwargs):
            finish_started.set()
            return super().finish(*args, **kwargs)

    monkeypatch.setattr(policy, "_ResidentMemorySupervisor", GatedSupervisor)

    def run_check():
        try:
            outcome["result"] = policy.run_sandboxed(
                [sys.executable, "-c", "pass"], tmp_path,
                timeout=8, cpu_seconds=8, memory_bytes=128 << 20,
                file_bytes=64 << 20, nproc=32,
            )
        except BaseException as error:
            outcome["error"] = error
        finally:
            completed.set()

    runner = threading.Thread(target=run_check, name="rss-clean-exit-regression")
    runner.start()
    try:
        assert scan_started.wait(5), "supervisor never entered its real procfs scan"
        assert collected.wait(5), "_collect did not reap the clean command"
        assert finish_started.wait(2), "run_sandboxed did not enter supervisor finish()"
        assert not release_scan.wait(3.2), "scan gate released before the old 3 s deadline"
        release_scan.set()
        assert completed.wait(5), "run_sandboxed did not finish after releasing the scan"
        assert "error" not in outcome, outcome.get("error")
        result = outcome["result"]
        assert isinstance(result, policy.PolicyResult)
        assert result.completed.returncode == 0
        assert result.memory_limit_mechanism == "rlimit_data+rss_supervisor"
        assert len(supervisors) == 1 and supervisors[0].containment_verified, (
            "successful finish must record verified containment; "
            f"family_empty={supervisors[0].family_verified_empty}, "
            f"cleanup_empty={supervisors[0].cleanup_verified_no_survivors}")
    finally:
        release_scan.set()
        runner.join(timeout=5)
    assert not runner.is_alive(), "run_sandboxed worker remained blocked"


def test_run_sandboxed_cleanup_uses_supervisor_family_and_chains_original(
        tmp_path, monkeypatch):
    report = policy.CapabilityReport("Linux", "test", True, True, True, True, True, True)
    family = object()
    launch_identity = object()

    class Process:
        pid = 123
        returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    process = Process()

    class Supervisor:
        error = None
        finish_error = "resident-memory scan timed out in this test"
        family_verified_empty = False
        containment_verified = False

        def __init__(self):
            self.family = family
            self.launch_identity = launch_identity
            self.reap_notifications = 0
            self._leader_reaped = False

        def notify_leader_reaped(self):
            if not self._leader_reaped:
                self.reap_notifications += 1
                self._leader_reaped = True

        def finish(self, *, timeout, wait_for_post_exit=True):
            return False

    process._stateframe_memory_supervisor = Supervisor()
    cleanup_calls = []

    def record_cleanup(child, captured_family=None, *, launch_identity=None):
        cleanup_calls.append((child, captured_family, launch_identity))
        return False

    monkeypatch.setattr(policy, "require_capabilities", lambda: report)
    monkeypatch.setattr(policy, "_command_cgroup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(policy, "sandboxed_popen", lambda *_args, **_kwargs: (process, report))
    monkeypatch.setattr(policy, "_collect", lambda *_args, **_kwargs: ("", "", False, False, False))
    monkeypatch.setattr(policy, "_kill_group", record_cleanup)

    with pytest.raises(ContainmentUnverified, match="survivors remain after cleanup") as raised:
        policy.run_sandboxed(["true"], tmp_path)

    assert cleanup_calls == [(process, family, launch_identity)]
    assert process._stateframe_memory_supervisor.reap_notifications == 1
    assert isinstance(raised.value.__cause__, ContainmentUnverified)
    assert "scan timed out in this test" in str(raised.value.__cause__)


@pytest.mark.parametrize(
    "gone_error",
    [ProcessLookupError(errno.ESRCH, "gone"), _ErrnoOSError(errno.ESRCH, "gone")],
    ids=["process-lookup", "esrch-oserror"],
)
def test_procfs_resident_bytes_treats_esrch_as_gone(monkeypatch, gone_error):
    row = ProcessIdentity(123, 1, 123, 123, (10, 0), False)
    table = tracker.ProcessTable()
    monkeypatch.setattr(table, "identity", lambda _pid: row)

    def vanished_statm(self, **_kwargs):
        if self.name == "statm":
            raise gone_error
        raise AssertionError(f"unexpected procfs read: {self}")

    monkeypatch.setattr(Path, "read_text", vanished_statm)
    assert table.resident_bytes(row) == 0


@pytest.mark.parametrize(
    "gone_error",
    [ProcessLookupError(errno.ESRCH, "gone"), _ErrnoOSError(errno.ESRCH, "gone")],
    ids=["process-lookup", "esrch-oserror"],
)
@pytest.mark.parametrize("location", ["task-list", "children-file"])
def test_procfs_children_treat_esrch_as_gone(monkeypatch, gone_error, location):
    table = tracker.ProcessTable()
    task_root = Path("/proc/123/task")
    task_children = task_root / "123" / "children"

    def list_tasks(self):
        if self == task_root:
            if location == "task-list":
                raise gone_error
            return iter([task_root / "123"])
        return iter(())

    monkeypatch.setattr(Path, "iterdir", list_tasks)

    def vanished_children(self, **_kwargs):
        if self == task_children and location == "children-file":
            raise gone_error
        raise AssertionError(f"unexpected procfs read: {self}")

    monkeypatch.setattr(Path, "read_text", vanished_children)
    assert table.children(123) == ()


@pytest.mark.parametrize("helper", ["resident_bytes", "children"])
def test_procfs_helpers_propagate_non_esrch_errors(monkeypatch, helper):
    table = tracker.ProcessTable()
    error = OSError(errno.EIO, "procfs I/O failure")
    if helper == "resident_bytes":
        row = ProcessIdentity(123, 1, 123, 123, (10, 0), False)
        monkeypatch.setattr(table, "identity", lambda _pid: row)

        def fail_statm(self, **_kwargs):
            if self.name == "statm":
                raise error
            raise AssertionError(f"unexpected procfs read: {self}")

        monkeypatch.setattr(Path, "read_text", fail_statm)
        with pytest.raises(OSError) as raised:
            table.resident_bytes(row)
    else:
        task_root = Path("/proc/123/task")
        task_children = task_root / "123" / "children"
        monkeypatch.setattr(Path, "iterdir", lambda self: iter([task_root / "123"])
                            if self == task_root else iter(()))

        def fail_children(self, **_kwargs):
            if self == task_children:
                raise error
            raise AssertionError(f"unexpected procfs read: {self}")

        monkeypatch.setattr(Path, "read_text", fail_children)
        with pytest.raises(OSError) as raised:
            table.children(123)
    assert raised.value.errno == errno.EIO


def test_unverifiable_live_process_still_fails_closed():
    class UnverifiableLiveProcess:
        pid = 1
        returncode = None

        def kill(self):
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("unverifiable", timeout)

        def poll(self):
            return None

    with pytest.raises(ContainmentUnverified, match="survivors unknown"):
        kill_launch_unit(
            UnverifiableLiveProcess(), launch_identity=None, deadline_seconds=0)


def test_unstamped_leader_requires_explicit_spawn_authority():
    leader = ProcessIdentity(123, 1, 123, 123, (10, 0), False)

    class Table:
        pid_namespace_inode = None

        def identity(self, pid):
            return leader if pid == leader.pid else None

    identity = LaunchIdentity(leader.pid, None, "boot-id")
    family = Descendants(leader.pid, Table(), identity)
    assert leader.pid not in family.members

    spawn_family = Descendants(
        leader.pid, Table(), identity, allow_unstamped_leader=True)
    assert spawn_family.members[leader.pid] is leader
    assert spawn_family.launch_leader_start == leader.start


def _empty_family_supervisor(monkeypatch, table):
    class Process:
        pid = 123
        returncode = 0

    process = Process()
    identity = LaunchIdentity(process.pid, None, "test-boot")
    monkeypatch.setattr(policy, "ProcessTable", lambda: table)
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda _process: identity)
    return policy._ResidentMemorySupervisor(process, 128 << 20)


def _run_sandboxed_with_test_supervisor(tmp_path, monkeypatch, *, scan_error=None,
                                        verify_empty=True):
    report = policy.CapabilityReport("Linux", "test", True, True, True, True, True, True)

    class Process:
        pid = 123
        returncode = 0
        memory_limit_mechanism = "rlimit_data+rss_supervisor"
        memory_limit_killed = False
        memory_limit_reason = None

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

    process = Process()

    class Table:
        @staticmethod
        def identity(_pid):
            return None

        @staticmethod
        def pid_namespace_inode(_pid):
            return None

        def refresh_members(self, _members, **_kwargs):
            if scan_error is not None:
                raise scan_error
            return {}

        @staticmethod
        def snapshot():
            return {}

        @staticmethod
        def resident_memory_limit_breach(_live, _memory_bytes):
            return ""

    real_supervisor_type = policy._ResidentMemorySupervisor

    class TestSupervisor(real_supervisor_type):
        # The runner's cleanup path is separately exercised with a verified result.
        def _stop_launch_unit(self):
            pass

    monkeypatch.setattr(policy, "require_capabilities", lambda: report)
    monkeypatch.setattr(policy, "_command_cgroup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(policy, "ProcessTable", Table)
    monkeypatch.setattr(
        policy, "recorded_launch_identity",
        lambda _process: LaunchIdentity(process.pid, None, "test-boot"),
    )
    monkeypatch.setattr(policy, "_ResidentMemorySupervisor", TestSupervisor)

    def spawn(_command, _root, **_options):
        supervisor = TestSupervisor(process, 128 << 20)
        process._stateframe_memory_supervisor = supervisor
        supervisor.start()
        return process, report

    def collect(_process, *_args, **kwargs):
        callback = kwargs.get("on_leader_reaped")
        if callable(callback):
            callback()
        return "", "", False, False, False

    monkeypatch.setattr(policy, "sandboxed_popen", spawn)
    monkeypatch.setattr(policy, "_collect", collect)
    monkeypatch.setattr(policy, "_kill_group", lambda *_args, **_kwargs: True)
    if not verify_empty:
        # Exercise a completed leader whose supervisor skipped its required full scan.
        monkeypatch.setattr(policy.platform, "system", lambda: "Darwin")

    result = policy.run_sandboxed(["true"], tmp_path, memory_bytes=128 << 20)
    return result, process._stateframe_memory_supervisor


@pytest.mark.parametrize("error_type", [IndexError, RuntimeError])
def test_run_sandboxed_fails_closed_and_chains_post_exit_supervisor_error(
        tmp_path, monkeypatch, error_type):
    cause = error_type("post-exit scan failed")

    with pytest.raises(ContainmentUnverified, match="containment unverified") as raised:
        _run_sandboxed_with_test_supervisor(
            tmp_path, monkeypatch, scan_error=cause)

    assert raised.value.__cause__ is cause
    assert f"{error_type.__name__}: post-exit scan failed" in str(raised.value)


def test_run_sandboxed_fails_closed_without_empty_family_verification(tmp_path, monkeypatch):
    with pytest.raises(ContainmentUnverified,
                       match="without verifying containment"):
        _run_sandboxed_with_test_supervisor(tmp_path, monkeypatch, verify_empty=False)


def test_run_sandboxed_clean_supervisor_still_succeeds(tmp_path, monkeypatch):
    result, supervisor = _run_sandboxed_with_test_supervisor(tmp_path, monkeypatch)

    assert result.completed.returncode == 0
    assert supervisor.containment_verified


def test_finish_accepts_verified_cleanup_without_empty_family(monkeypatch):
    leader = ProcessIdentity(123, 1, 123, 123, (10, 0), False)

    class Process:
        pid = leader.pid
        returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

    class Table:
        @staticmethod
        def identity(pid):
            return leader if pid == leader.pid else None

        @staticmethod
        def pid_namespace_inode(_pid):
            return None

        @staticmethod
        def snapshot():
            return {}

    process = Process()
    table = Table()
    identity = LaunchIdentity(process.pid, None, "test-boot")
    monkeypatch.setattr(policy, "ProcessTable", lambda: table)
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda _process: identity)
    supervisor = policy._ResidentMemorySupervisor(process, 128 << 20)
    supervisor._stop_launch_unit()
    supervisor._leader_exit_processed.set()

    assert not supervisor.family_verified_empty
    assert supervisor.cleanup_verified_no_survivors
    assert supervisor.containment_verified
    assert supervisor.finish()


def test_finish_waits_for_blocked_refresh_after_leader_exit(monkeypatch):
    refresh_started = threading.Event()
    release_refresh = threading.Event()

    class Table:
        @staticmethod
        def identity(_pid):
            return None

        @staticmethod
        def pid_namespace_inode(_pid):
            return None

        def refresh_members(self, _members, **_kwargs):
            refresh_started.set()
            if not release_refresh.wait(10):
                raise RuntimeError("test did not release blocked member refresh")
            return {}

        @staticmethod
        def snapshot():
            return {}

        @staticmethod
        def resident_memory_limit_breach(_live, _memory_bytes):
            return ""

    supervisor = _empty_family_supervisor(monkeypatch, Table())
    supervisor.start()
    assert refresh_started.wait(2), "supervisor never entered member refresh"
    release_timer = threading.Timer(5.0, release_refresh.set)
    release_timer.start()
    started = time.monotonic()
    try:
        assert supervisor.finish(timeout=0.1), supervisor.finish_error
        assert time.monotonic() - started >= 4.5
        assert supervisor.containment_verified
    finally:
        release_refresh.set()
        release_timer.cancel()
        release_timer.join(timeout=1)
        supervisor._thread.join(timeout=1)
    assert not supervisor._thread.is_alive()


def test_finish_fails_at_hard_cap_while_supervisor_work_is_stuck(monkeypatch):
    refresh_started = threading.Event()
    release_refresh = threading.Event()

    class Table:
        @staticmethod
        def identity(_pid):
            return None

        @staticmethod
        def pid_namespace_inode(_pid):
            return None

        def refresh_members(self, _members, **_kwargs):
            refresh_started.set()
            release_refresh.wait()
            return {}

        @staticmethod
        def snapshot():
            return {}

        @staticmethod
        def resident_memory_limit_breach(_live, _memory_bytes):
            return ""

    monkeypatch.setattr(policy, "RSS_FINISH_MAX_WAIT_SECONDS", 0.25)
    supervisor = _empty_family_supervisor(monkeypatch, Table())
    supervisor.start()
    assert refresh_started.wait(2), "supervisor never entered member refresh"
    started = time.monotonic()
    try:
        assert not supervisor.finish(timeout=0.1)
        assert time.monotonic() - started < 1.0
        assert supervisor.finish_error is not None
        assert supervisor.finish_error.startswith(
            "resident-memory supervisor post-exit verification timed out after ")
        assert "(work active " in supervisor.finish_error
        assert "; last progress " in supervisor.finish_error
        assert "scan active 0.00s; last completed scan 0.00s; maximum wait 0.25s)" in (
            supervisor.finish_error)
    finally:
        release_refresh.set()
        supervisor._thread.join(timeout=2)
    assert not supervisor._thread.is_alive()


def test_finish_fails_quickly_when_supervisor_thread_is_dead(monkeypatch):
    class Table:
        @staticmethod
        def identity(_pid):
            return None

        @staticmethod
        def pid_namespace_inode(_pid):
            return None

    supervisor = _empty_family_supervisor(monkeypatch, Table())
    dead_thread = threading.Thread(target=lambda: None, name="dead-rss-supervisor")
    dead_thread.start()
    dead_thread.join()
    supervisor._thread = dead_thread

    started = time.monotonic()
    assert not supervisor.finish()
    assert time.monotonic() - started < 0.5
    assert supervisor.finish_error == (
        "resident-memory supervisor thread exited before post-exit verification "
        "completed (scan active 0.00s; last completed scan 0.00s; maximum wait 15s)"
    )
