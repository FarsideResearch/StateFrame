"""Deterministic process-table regressions for automatic probe cleanup."""
from dataclasses import replace
import os
import signal
from types import SimpleNamespace

import pytest

import kernel_exec_supervisor as supervisor_module
from kernel_exec_session import ExecResult, ExecSession, JobHandle, _cleanup
from kernel_exec_supervisor import ManagedProcess, Supervisor
from kernel_exec_tracker import ProcessIdentity, ProcessTable
from kernel_safe_probe import ProbeCleanupError, run_probe_session


def _row(pid, ppid, group):
    return ProcessIdentity(pid, ppid, group, group, (pid, 0), False)


class _Probe:
    def __init__(self, monkeypatch, leader, descendant):
        self.clock = 0.0
        self.rows = {row.pid: row for row in (leader, descendant)}
        self.leader_exited = False
        self.signals = []
        monkeypatch.setattr(supervisor_module, "time", SimpleNamespace(
            monotonic=lambda: self.clock, sleep=lambda _seconds: None))
        monkeypatch.setattr(supervisor_module, "emit_kill", lambda *args, **kwargs: None)
        self.table = SimpleNamespace(
            identity=self.rows.get, snapshot=self.rows.copy,
            limit_breach=lambda *_args: "", pid_namespace=lambda _pid: "pid:[42]",
        )
        process = SimpleNamespace(
            pid=leader.pid, poll=lambda: 0 if self.leader_exited else None,
            stdout=SimpleNamespace(fileno=lambda: 0, close=lambda: None),
        )
        collector = SimpleNamespace(
            total_bytes=0, drain=lambda _fd: True, seal=lambda: None, close=lambda: None,
        )
        cap = SimpleNamespace(purpose="probe", limits=SimpleNamespace(
            wall_seconds=10, kill_on_output=False))
        self.job = ManagedProcess("j1", process, collector, cap, self.table)

        def record_signals(rows, sig):
            self.signals.extend((row.pid, sig) for row in rows)
            return {row.pid for row in rows}

        monkeypatch.setattr(self.job.family, "signal", record_signals)

    def finish(self):
        assert self.job.cleanup is not None
        return run_probe_session(SimpleNamespace(close=lambda: None), lambda: ExecResult(
            self.job.exit_code, False, b"", False, 0, self.clock, _cleanup(self.job.cleanup)))

    def stop(self, leader):
        self.leader_exited = True
        del self.rows[leader.pid]
        self.job.stop(0)

    def tick(self, *, check_orphans=None):
        self.clock += 3  # exceed the former wall deadline between every observation
        self.job.tick(self.rows.copy(), check_orphans=check_orphans)

    def supervisor(self, on_select):
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.jobs = {"j1": self.job}
        supervisor.table = self.table
        supervisor.namespace_network = "none"
        supervisor.closing = True
        supervisor.outgoing = bytearray()
        supervisor.owner_fd = -1
        supervisor.control = SimpleNamespace(close=lambda: None)
        supervisor._sample_rows = self.rows.copy
        supervisor.selector = SimpleNamespace(
            select=on_select, modify=lambda *_args: None, close=lambda: None)
        return supervisor


def test_probe_gives_sigkill_time_to_take_effect_after_delayed_tick(monkeypatch):
    leader = _row(3, 1, 3)
    child = _row(5, 3, 4)
    probe = _Probe(monkeypatch, leader, child)
    probe.job.family.update(probe.rows)
    probe.leader_exited = True
    del probe.rows[leader.pid]
    probe.rows[child.pid] = replace(child, ppid=1)
    probe.job.stop(0)
    # The supervisor was descheduled before it could send the first SIGKILL.
    probe.clock += 7
    probe.job.tick(probe.rows)
    assert (child.pid, signal.SIGKILL) in probe.signals
    probe.rows[child.pid] = replace(probe.rows[child.pid], zombie=True)
    probe.clock += 3
    probe.job.tick(probe.rows)
    del probe.rows[child.pid]
    probe.job.tick(probe.rows)
    assert probe.finish().cleanup.ok


@pytest.mark.parametrize("leader_exited", [True, False], ids=["leader-reaped", "leader-still-live"])
def test_probe_rechecks_reparented_child_while_sandbox_exits(monkeypatch, leader_exited):
    supervisor_pid = os.getpid()
    leader = _row(supervisor_pid + 10, supervisor_pid, supervisor_pid + 10)
    orphan = _row(supervisor_pid + 12, supervisor_pid, supervisor_pid + 11)
    probe = _Probe(monkeypatch, leader, orphan)
    if leader_exited:
        probe.leader_exited = True
        del probe.rows[leader.pid]
    signals = []
    monkeypatch.setattr(supervisor_module.os, "kill", lambda *args: signals.append(args))

    def reaper_runs(_timeout):
        # The sandbox's intermediate parent exited between scans, so the
        # supervisor first sees its child as an orphan still in kernel exit.
        if orphan.pid in probe.rows:
            probe.rows[orphan.pid] = replace(orphan, zombie=True)
        probe.leader_exited = True
        probe.rows.pop(leader.pid, None)
        probe.clock += 3
        return []

    def reap(pid, _flags):
        assert probe.rows[pid].zombie
        del probe.rows[pid]
        return pid, 0

    monkeypatch.setattr(supervisor_module.os, "waitpid", reap)
    probe.supervisor(reaper_runs).serve()
    assert probe.finish().cleanup.ok
    assert signals == []


def test_probe_waits_through_slow_exit_reparenting_and_reap(monkeypatch):
    monkeypatch.setattr(supervisor_module, "PROBE_REAP_STALLED_SAMPLES", 2)
    leader = _row(3, os.getpid(), 3)
    child = _row(5, 3, 4)
    probe = _Probe(monkeypatch, leader, child)
    probe.job.family.update(probe.rows)
    probe.stop(leader)
    supervisor = probe.supervisor(lambda _timeout: [])
    attempts = []
    reap_after = probe.clock + 18

    def reap(pid, _flags):
        attempts.append(pid)
        if probe.clock < reap_after:
            return 0, 0
        del probe.rows[pid]
        return pid, 0

    monkeypatch.setattr(supervisor_module.os, "waitpid", reap)
    monkeypatch.setattr(supervisor_module.os, "kill", lambda *_args: pytest.fail("already owned"))
    probe.tick(check_orphans=supervisor._check_orphans)
    probe.tick(check_orphans=supervisor._check_orphans)
    probe.rows[child.pid] = replace(child, ppid=os.getpid(), exiting=True)
    probe.tick(check_orphans=supervisor._check_orphans)
    probe.tick(check_orphans=supervisor._check_orphans)
    probe.rows[child.pid] = replace(probe.rows[child.pid], zombie=True)
    probe.tick(check_orphans=supervisor._check_orphans)
    assert probe.job.cleanup is None  # waitpid has not reaped the known zombie yet
    probe.tick(check_orphans=supervisor._check_orphans)
    assert probe.finish().cleanup.ok
    assert attempts == [child.pid] * 3


@pytest.mark.parametrize("exiting", [False, True], ids=["live", "stuck-in-exit"])
def test_probe_fails_when_sigkill_and_reaping_make_no_progress(monkeypatch, exiting):
    monkeypatch.setattr(supervisor_module, "PROBE_REAP_STALLED_SAMPLES", 2)
    leader = _row(3, 1, 3)
    child = replace(_row(5, 3, 4), exiting=exiting)
    probe = _Probe(monkeypatch, leader, child)
    probe.job.family.update(probe.rows)
    probe.stop(leader)
    probe.rows[child.pid] = replace(child, ppid=1)

    probe.tick()
    probe.tick()
    assert probe.job.cleanup is None
    probe.tick()
    with pytest.raises(ProbeCleanupError, match=r"descendants survived SIGKILL: 5\(ppid=1"):
        probe.finish()
    assert probe.job.cleanup["survivors"] == (child.pid,)


@pytest.mark.parametrize("reuse", [False, True], ids=["unknown", "reused-pid"])
def test_probe_still_fails_for_persistent_unattributed_process(monkeypatch, reuse):
    monkeypatch.setattr(supervisor_module, "PROBE_REAP_STALLED_SAMPLES", 2)
    leader = _row(3, os.getpid(), 3)
    child = _row(5, 3, 4)
    probe = _Probe(monkeypatch, leader, child)
    if reuse:
        probe.job.family.update(probe.rows)
    probe.rows[child.pid] = replace(child, ppid=os.getpid(), start=(6, 0))
    probe.stop(leader)
    killed = []
    monkeypatch.setattr(supervisor_module.os, "kill", lambda *args: killed.append(args))
    supervisor = probe.supervisor(lambda _timeout: [])

    probe.tick(check_orphans=supervisor._check_orphans)
    assert probe.job.cleanup is None
    probe.tick(check_orphans=supervisor._check_orphans)
    probe.tick(check_orphans=supervisor._check_orphans)
    with pytest.raises(ProbeCleanupError, match="unattributed namespace descendant; session stopped"):
        probe.finish()
    assert killed == [(child.pid, signal.SIGKILL)]
    assert probe.signals == []  # no authority to signal it as a member of the probe family


def test_probe_orphan_churn_cannot_hide_a_stalled_survivor(monkeypatch):
    monkeypatch.setattr(supervisor_module, "PROBE_REAP_STALLED_SAMPLES", 2)
    leader = _row(3, os.getpid(), 3)
    orphan = _row(5, os.getpid(), 4)
    probe = _Probe(monkeypatch, leader, orphan)
    probe.stop(leader)
    monkeypatch.setattr(supervisor_module.os, "kill", lambda *_args: None)
    supervisor = probe.supervisor(lambda _timeout: [])

    probe.tick(check_orphans=supervisor._check_orphans)
    probe.rows[6] = _row(6, os.getpid(), 4)
    probe.tick(check_orphans=supervisor._check_orphans)
    del probe.rows[6]
    probe.tick(check_orphans=supervisor._check_orphans)

    with pytest.raises(ProbeCleanupError, match="unattributed namespace descendant; session stopped"):
        probe.finish()
    assert probe.job.cleanup["survivors"] == (orphan.pid,)


def test_probe_rejects_persistent_orphan_while_leader_is_running(monkeypatch):
    monkeypatch.setattr(supervisor_module, "PROBE_REAP_STALLED_SAMPLES", 2)
    supervisor_pid = os.getpid()
    leader = _row(supervisor_pid + 10, supervisor_pid, supervisor_pid + 10)
    orphan = _row(supervisor_pid + 12, supervisor_pid, supervisor_pid + 11)
    probe = _Probe(monkeypatch, leader, orphan)
    killed = []
    monkeypatch.setattr(supervisor_module.os, "kill", lambda *args: killed.append(args))
    supervisor = probe.supervisor(lambda _timeout: [])

    probe.tick(check_orphans=supervisor._check_orphans)
    assert probe.job.stop_at is None
    for _ in range(2):
        probe.clock += 3
        probe.job.tick({leader.pid: leader}, check_orphans=supervisor._check_orphans)
    assert probe.job.stop_at == 9  # the probe's own wall limit is still in the future
    assert killed == [(orphan.pid, signal.SIGKILL)]

    probe.leader_exited = True
    probe.rows.clear()
    probe.tick(check_orphans=supervisor._check_orphans)
    with pytest.raises(ProbeCleanupError, match="unattributed namespace descendant; session stopped"):
        probe.finish()


def test_probe_retains_verified_cgroup_member_after_reparenting(monkeypatch, tmp_path):
    leader = _row(3, os.getpid(), 3)
    child = _row(5, os.getpid(), 4)
    probe = _Probe(monkeypatch, leader, child)
    group = tmp_path / "command-cgroup"
    group.mkdir()
    pids = group / "cgroup.procs"
    pids.write_text(f"{child.pid}\n", encoding="ascii")
    probe.job.cgroup = group
    supervisor = probe.supervisor(lambda _timeout: [])
    probe.stop(leader)
    probe.tick(check_orphans=supervisor._check_orphans)
    assert probe.job.family.members[child.pid] == child
    assert (child.pid, signal.SIGKILL) in probe.signals
    pids.write_text("", encoding="ascii")
    probe.tick(check_orphans=supervisor._check_orphans)
    assert probe.signals.count((child.pid, signal.SIGKILL)) == 2


@pytest.mark.skipif(supervisor_module.sys.platform != "linux", reason="Linux procfs flags")
def test_linux_stat_distinguishes_exiting_task_from_zombie(monkeypatch, tmp_path):
    import kernel_exec_tracker as tracker

    stat = tmp_path / "5" / "stat"
    stat.parent.mkdir()
    monkeypatch.setattr(tracker, "Path", lambda _root: tmp_path)
    fields = ["D", "1", "4", "4", "0", "0", "4", *(["0"] * 12), "30"]
    stat.write_text("5 (worker) " + " ".join(fields), encoding="ascii")
    row = ProcessTable().identity(5)
    assert row.exiting and not row.zombie


@pytest.mark.parametrize("stop_all", [False, True], ids=["job-stop", "session-stop"])
def test_probe_close_waits_for_supervisor_result_past_old_deadline(monkeypatch, stop_all):
    import kernel_exec_session as session_module

    clock = 0.0
    running = {"finished": False}
    finished = {"finished": True,
                "cleanup": {"ok": True, "killed": 0, "survivors": [], "detail": ""}}
    observations = iter((running, running, finished))
    session = ExecSession.__new__(ExecSession)
    session._process = object()
    session._probe_family = object()

    def rpc(request):
        if request["op"] == "all":
            return {"jobs": {"j1": next(observations)}}
        if request["op"] == "status":
            return next(observations)
        return {}

    def sleep(_seconds):
        nonlocal clock
        clock += 10

    session._rpc = rpc
    monkeypatch.setattr(session_module, "time", SimpleNamespace(monotonic=lambda: clock, sleep=sleep))
    report = session.stop_all(0) if stop_all else JobHandle("j1", session).stop(0)
    assert report.ok
