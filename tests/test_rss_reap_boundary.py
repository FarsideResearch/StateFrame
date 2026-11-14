"""RSS supervision must honor the collector's exact leader-reap boundary."""
from __future__ import annotations

import sys
import threading

import pytest

import kernel_execution_policy as policy
import kernel_exec_tracker as tracker
from kernel_exec_tracker import ProcessIdentity


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux RSS supervision")


def _identity_isolation(command, _root, environment, **_options):
    return list(command), dict(environment)


def test_collector_reap_blocks_reused_group_pid_from_rss_admission(
        tmp_path, monkeypatch):
    real_supervisor_type = policy._ResidentMemorySupervisor
    real_collect = policy._collect
    popen = policy.sandboxed_popen
    report = policy.CapabilityReport("Linux", "test", True, True, True, True, True, True)
    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()
    collector_returned = threading.Event()
    state: dict[str, ProcessIdentity] = {}
    supervisors = []
    reap_callbacks: list[str] = []

    monkeypatch.setattr(policy, "require_capabilities", lambda: report)
    monkeypatch.setattr(policy, "_command_cgroup", lambda *_args, **_kwargs: None)

    def without_namespace(command, cwd, **options):
        return popen(command, cwd, isolation_builder=_identity_isolation, **options)

    monkeypatch.setattr(policy, "sandboxed_popen", without_namespace)

    def observed_collect(*args, **kwargs):
        callback = kwargs.get("on_leader_reaped")
        if callable(callback):
            def record_reap_callback():
                callback()
                reap_callbacks.append(threading.current_thread().name)

            kwargs["on_leader_reaped"] = record_reap_callback
        result = real_collect(*args, **kwargs)
        collector_returned.set()
        return result

    monkeypatch.setattr(policy, "_collect", observed_collect)

    class GatedSupervisor(real_supervisor_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            supervisors.append(self)
            self.stop_calls = []
            real_snapshot = self.table.snapshot

            def gated_snapshot():
                snapshot_entered.set()
                if not release_snapshot.wait(8):
                    raise RuntimeError("test did not release RSS snapshot")
                rows = real_snapshot()
                row = state.get("reused_group_member")
                if row is not None:
                    rows[row.pid] = row
                return rows

            self.table.snapshot = gated_snapshot
            self._next_full_process_scan = 0

        def _stop_launch_unit(self):
            self.stop_calls.append(True)

    monkeypatch.setattr(policy, "_ResidentMemorySupervisor", GatedSupervisor)
    outcome: dict[str, object] = {}

    def run_check():
        try:
            outcome["result"] = policy.run_sandboxed(
                [sys.executable, "-c", "pass"], tmp_path,
                timeout=8, cpu_seconds=8, memory_bytes=128 << 20,
                file_bytes=64 << 20, nproc=32,
            )
        except BaseException as error:
            outcome["error"] = error

    runner = threading.Thread(target=run_check, name="rss-reap-boundary-regression")
    runner.start()
    try:
        assert snapshot_entered.wait(5), "RSS supervisor never entered its gated snapshot"
        assert collector_returned.wait(5), "collector did not reap the clean leader"
        supervisor = supervisors[0]
        leader_pid = supervisor.process.pid
        reused = ProcessIdentity(
            pid=leader_pid + 1_000_000,
            ppid=1,
            pgid=leader_pid,
            sid=leader_pid,
            start=tracker.process_start_now(),
            zombie=False,
        )
        state["reused_group_member"] = reused
        release_snapshot.set()
        runner.join(timeout=8)
        assert not runner.is_alive(), "run_sandboxed remained blocked"
        assert "result" in outcome, outcome.get("error")
        assert reused.pid not in supervisor.family.members
        assert supervisor.stop_calls == []
        assert reap_callbacks == [runner.name]
        assert supervisor.family.reaped_bound is not None
        assert supervisor.family.reaped_bound <= reused.start
    finally:
        release_snapshot.set()
        runner.join(timeout=2)


def test_under_budget_check_completes_with_rss_supervision(tmp_path, monkeypatch):
    report = policy.CapabilityReport("Linux", "test", True, True, True, True, True, True)
    popen = policy.sandboxed_popen
    monkeypatch.setattr(policy, "require_capabilities", lambda: report)
    monkeypatch.setattr(policy, "_command_cgroup", lambda *_args, **_kwargs: None)

    def without_namespace(command, cwd, **options):
        return popen(command, cwd, isolation_builder=_identity_isolation, **options)

    monkeypatch.setattr(policy, "sandboxed_popen", without_namespace)
    result = policy.run_sandboxed(
        [sys.executable, "-c", "pass"], tmp_path,
        timeout=8, cpu_seconds=8, memory_bytes=128 << 20,
        file_bytes=64 << 20, nproc=32,
    )

    assert result.completed.returncode == 0
    assert result.memory_limit_mechanism == "rlimit_data+rss_supervisor"
    assert not result.memory_limit_killed


def test_pre_reap_group_member_remains_tracked_after_collector_notification(monkeypatch):
    leader = ProcessIdentity(41001, 1, 41001, 41001, (10, 0), False)
    member = ProcessIdentity(41002, 1, 41001, 41001, (11, 0), False)

    class Process:
        pid = leader.pid

        def __init__(self):
            self.returncode = None

    process = Process()

    class Table:
        def identity(self, pid):
            if pid == leader.pid and process.returncode is None:
                return leader
            return None

        @staticmethod
        def pid_namespace_inode(_pid):
            return None

    identity = tracker.LaunchIdentity(leader.pid, leader.start, "test-boot")
    monkeypatch.setattr(policy, "ProcessTable", Table)
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda _process: identity)
    supervisor = policy._ResidentMemorySupervisor(process, 128 << 20)

    before_reap = supervisor._update_family({leader.pid: leader, member.pid: member})
    assert member in before_reap
    assert supervisor.family.members[member.pid] == member

    process.returncode = 0
    supervisor.notify_leader_reaped()
    after_reap = supervisor._update_family({member.pid: member})

    assert member in after_reap
    assert supervisor.family.members[member.pid] == member


def test_cleanup_update_cannot_admit_or_signal_a_new_group_row(monkeypatch):
    leader = ProcessIdentity(42001, 1, 42001, 42001, (10, 0), False)
    member = ProcessIdentity(42002, 1, 42001, 42001, (11, 0), False)

    class Process:
        pid = leader.pid

        def __init__(self):
            self.returncode = None

    process = Process()

    class Table:
        def identity(self, pid):
            if pid == leader.pid and process.returncode is None:
                return leader
            if pid == member.pid:
                return member
            return None

        @staticmethod
        def pid_namespace_inode(_pid):
            return None

    identity = tracker.LaunchIdentity(leader.pid, leader.start, "test-boot")
    monkeypatch.setattr(policy, "ProcessTable", Table)
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda _process: identity)
    supervisor = policy._ResidentMemorySupervisor(process, 128 << 20)
    supervisor._update_family({leader.pid: leader, member.pid: member})
    process.returncode = 0
    supervisor.notify_leader_reaped()
    assert supervisor.family.reaped_bound is not None

    reused = ProcessIdentity(
        pid=42003, ppid=1, pgid=leader.pid, sid=leader.pid,
        start=supervisor.family.reaped_bound, zombie=False,
    )
    signalled: list[int] = []

    def inspect_cleanup(_process, family, *, launch_identity, **_log_fields):
        live = family.update({reused.pid: reused})
        signalled.extend(row.pid for row in live)
        return ()

    monkeypatch.setattr(policy, "kill_launch_unit", inspect_cleanup)
    supervisor._stop_launch_unit()

    assert reused.pid not in supervisor.family.members
    assert reused.pid not in signalled
