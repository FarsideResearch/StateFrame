"""Recorded subframe fencing must authenticate the leader before signalling."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import stateframe.parallel_host as host
import stateframe.parallel_worker as worker_module
from stateframe.engine import kernel_unit_policy
from stateframe.engine import kernel_exec_tracker
from stateframe.engine import kernel_execution_policy
from stateframe.engine.kernel_exec_tracker import (ContainmentUnverified, LaunchIdentity, ProcessIdentity,
                                                   ProcessTable, signal_launch_group,
                                                   signal_mac_verified_process)


def _run_dir(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "parallel" / "call" / "units" / "a"
    run_dir.mkdir(parents=True)
    child = tmp_path / "engine" / "_child.py"
    (run_dir / "spec.json").write_text(json.dumps({"engine_dir": str(child.parent / "engine")}))
    return run_dir, child


def _record(run_dir: Path, *, birth: str = "darwin:100:1") -> Path:
    path = run_dir / "proc" / "1" / "process.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"incarnation": 1, "pid": 4444, "pgid": 4444,
                                "started_at": "display only", "started_epoch": 100.0,
                                "kernel_identity": birth, "verified_members": {}}))
    return path


def _identity(monkeypatch: pytest.MonkeyPatch, identity):
    monkeypatch.setattr(host, "_process_identity", identity)
    monkeypatch.setattr(host, "_live_group_members", lambda group: [])
    calls = []
    monkeypatch.setattr(host, "_signal_one", lambda pid, pgid, birth, sig: calls.append(
        (pid, pgid, birth, sig)))
    return calls


def test_unparseable_command_still_counts_as_live_group_member(monkeypatch):
    monkeypatch.setattr(host, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(host.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=f"4444 4444 S python -c 'unterminated\n{os.getpid()} {os.getpgrp()} S observer\n"))
    assert host._process_table() == [(4444, 4444, "S", []),
                                    (os.getpid(), os.getpgrp(), "S", ["observer"])]
    assert host._group_has_live_processes(4444) is True


def test_kernel_process_probe_keeps_exact_birth_and_argv(monkeypatch):
    monkeypatch.setattr(host, "ProcessTable", lambda: SimpleNamespace(identity=lambda pid:
                        SimpleNamespace(pgid=pid, zombie=False, start=(123, 456))))
    monkeypatch.setattr(host, "kernel_start_identity", lambda pid, row: "darwin:123:456")
    monkeypatch.setattr(host, "mac_process_argv", lambda pid: ["/tmp/_child.py", "/tmp/run with spaces"])
    monkeypatch.setattr(host, "sys", SimpleNamespace(platform="darwin"))
    assert host._process_identity(4444) == (4444, "R", "darwin:123:456",
                                           ["/tmp/_child.py", "/tmp/run with spaces"])


def test_matching_command_with_wrong_kernel_birth_is_not_signalled(tmp_path, monkeypatch):
    run_dir, child = _run_dir(tmp_path)
    record = _record(run_dir)
    calls = _identity(monkeypatch, lambda pid: (pid, "R", "darwin:100:2",
                                              [str(child), str(run_dir)]))
    host._signal_recorded_group(run_dir, 4444, 4444, "darwin:100:1", signal.SIGTERM, record)
    assert calls == []


def test_missing_leader_refuses_live_group_and_accepts_empty_group(tmp_path, monkeypatch):
    run_dir, _ = _run_dir(tmp_path)
    record = _record(run_dir)
    calls = _identity(monkeypatch, lambda pid: None)
    monkeypatch.setattr(host, "_live_group_members", lambda group: [5555])

    with pytest.raises(host.ProcessInspectionDeferred,
                       match="^process disappeared during family inspection; retry pending$"):
        host._signal_recorded_group(run_dir, 4444, 4444, "darwin:100:1", signal.SIGTERM, record)
    assert calls == []

    monkeypatch.setattr(host, "_live_group_members", lambda group: [])
    host._signal_recorded_group(run_dir, 4444, 4444, "darwin:100:1", signal.SIGTERM, record)
    assert calls == []


def test_reused_pid_in_another_group_is_not_signalled(tmp_path, monkeypatch):
    run_dir, child = _run_dir(tmp_path)
    record = _record(run_dir)
    calls = _identity(monkeypatch, lambda pid: (9999, "R", "darwin:100:1",
                                              [str(child), str(run_dir)]))
    host._signal_recorded_group(run_dir, 4444, 4444, "darwin:100:1", signal.SIGTERM, record)
    assert calls == []


def test_zombie_leader_and_own_group_are_refused(tmp_path, monkeypatch):
    run_dir, child = _run_dir(tmp_path)
    record = _record(run_dir)
    calls = _identity(monkeypatch, lambda pid: (pid, "Z", "darwin:100:1",
                                              [str(child), str(run_dir)]))
    host._signal_recorded_group(run_dir, 4444, 4444, "darwin:100:1", signal.SIGTERM, record)
    with pytest.raises(host.ParallelHostError, match="kernel identity is invalid"):
        host._signal_recorded_group(run_dir, os.getpgrp(), os.getpgrp(),
                                    "darwin:100:1", signal.SIGTERM, record)
    assert calls == []


def test_leader_is_rechecked_before_member_identity_is_persisted(tmp_path, monkeypatch):
    run_dir, child = _run_dir(tmp_path)
    record = _record(run_dir)
    identities = iter([(4444, "R", "darwin:100:1", [str(child), str(run_dir)]),
                       (4444, "R", "darwin:100:2", [str(child), str(run_dir)])])
    calls = _identity(monkeypatch, lambda pid: next(identities))

    with pytest.raises(host.ParallelHostError, match="leader exited during inspection"):
        host._signal_recorded_group(run_dir, 4444, 4444, "darwin:100:1", signal.SIGTERM, record)
    assert calls == []
    assert json.loads(record.read_text())["verified_members"] == {}

    calls = _identity(monkeypatch, lambda pid: (4444, "R", "darwin:100:1",
                                              [str(child), str(run_dir)]))
    # Signals go only to verified live members, so the live leader must be in the group listing.
    monkeypatch.setattr(host, "_live_group_members", lambda group: [4444])
    host._signal_recorded_group(run_dir, 4444, 4444, "darwin:100:1", signal.SIGTERM, record)
    assert calls == [(4444, 4444, "darwin:100:1", signal.SIGTERM)]


@pytest.mark.skipif(not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"),
                    reason="Linux pidfd APIs are unavailable")
def test_linux_signals_only_pinned_verified_members_then_leader(tmp_path, monkeypatch):
    run_dir, child = _run_dir(tmp_path)
    monkeypatch.setattr(host, "sys", SimpleNamespace(platform="linux"))
    read_fd, write_fd = os.pipe()
    opened = []
    signalled = []
    current_pid = []

    def open_pidfd(pid):
        opened.append(pid)
        current_pid.append(pid)
        return os.dup(read_fd)

    monkeypatch.setattr(host.os, "pidfd_open", open_pidfd)
    monkeypatch.setattr(host.signal, "pidfd_send_signal",
                        lambda _fd, sig: signalled.append((current_pid[-1], sig)))
    monkeypatch.setattr(host, "_live_group_members", lambda group: [4444, 5555])
    monkeypatch.setattr(host, "_process_identity",
                        lambda pid: (4444, "R", "linux:00000000-0000-0000-0000-000000000000:100",
                                     [str(child), str(run_dir)])
                        if pid == 4444 else (4444, "R", "linux:00000000-0000-0000-0000-000000000000:101",
                                             ["descendant"]))
    record = _record(run_dir, birth="linux:00000000-0000-0000-0000-000000000000:100")
    try:
        host._signal_recorded_group(run_dir, 4444, 4444,
                                    "linux:00000000-0000-0000-0000-000000000000:100",
                                    signal.SIGTERM, record)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert opened == [5555, 4444]
    assert signalled == [(5555, signal.SIGTERM), (4444, signal.SIGTERM)]


@pytest.mark.skipif(not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"),
                    reason="Linux pidfd APIs are unavailable")
def test_linux_rechecks_birth_after_pidfd_open_before_signal(monkeypatch):
    monkeypatch.setattr(host, "sys", SimpleNamespace(platform="linux"))
    read_fd, write_fd = os.pipe()
    opened = []
    delivered = []
    monkeypatch.setattr(host.os, "pidfd_open", lambda pid: opened.append(pid) or os.dup(read_fd))
    monkeypatch.setattr(host.signal, "pidfd_send_signal",
                        lambda fd, sig: delivered.append((fd, sig)))
    monkeypatch.setattr(host, "_process_identity", lambda pid: (pid, "R", "linux:"
                         "00000000-0000-0000-0000-000000000000:99", ["child"]))
    try:
        assert not host._signal_one(4444, 4444,
                                    "linux:00000000-0000-0000-0000-000000000000:100",
                                    signal.SIGTERM)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert opened == [4444]
    assert delivered == []


def test_recovered_macos_member_refuses_numeric_pid_signal(monkeypatch):
    monkeypatch.setattr(host.sys, "platform", "darwin")
    monkeypatch.setattr(host, "_process_identity", lambda pid:
                        (4444, "R", "darwin:100:1", ["subframe-child"]))
    delivered = []
    monkeypatch.setattr(os, "kill", lambda *args: delivered.append(args))

    with pytest.raises(host.ParallelHostError, match="containment unverified"):
        host._signal_one(4444, 4444, "darwin:100:1", signal.SIGTERM)
    assert delivered == []


def test_reaped_worker_leader_with_recycled_pid_is_never_signalled(tmp_path, monkeypatch):
    script = tmp_path / "exit.py"
    script.write_text("import time\ntime.sleep(0.2)\n", encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(script)], start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    handle = worker_module.WorkerHandle(process, process.pid, process.pid, 1, tmp_path,
                                        tmp_path / "worker.log", time.time(), time.monotonic(),
                                        threading.Thread(target=lambda: None))
    assert handle.launch_identity is not None
    process.wait(timeout=5)
    reused = ProcessIdentity(process.pid, 1, process.pid, process.pid, (999999, 0), False)
    monkeypatch.setattr(worker_module, "ProcessTable", lambda: SimpleNamespace(
        snapshot=lambda: {process.pid: reused}))
    delivered = []
    monkeypatch.setattr(os, "killpg", lambda *args: delivered.append(args))

    with pytest.raises(ContainmentUnverified, match="launch identity"):
        handle.reap()
    assert delivered == []


def test_forced_worker_death_reaps_recorded_detached_check(tmp_path, monkeypatch):
    ready = tmp_path / "check-ready.json"
    check = (
        "import json, os, pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"ready = pathlib.Path({str(ready)!r})\n"
        "ready.with_suffix('.tmp').write_text(json.dumps(os.getpid()))\n"
        "ready.with_suffix('.tmp').rename(ready)\n"
        "time.sleep(120)\n"
    )
    worker = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {check!r}], start_new_session=True)\n"
        "time.sleep(120)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", worker], start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    reader = threading.Thread(target=lambda: None)
    reader.start()
    handle = worker_module.WorkerHandle(process, process.pid, process.pid, 1, tmp_path,
                                        tmp_path / "worker.log", time.time(), time.monotonic(), reader)
    table = ProcessTable()
    cleanup = kernel_exec_tracker.Descendants(process.pid, table, handle.launch_identity)
    unrelated = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
                                  stdin=subprocess.PIPE, start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            assert handle.poll() is None
            time.sleep(0.01)
        assert ready.is_file(), "worker did not start its detached check"
        check_pid = json.loads(ready.read_text())
        child = table.identity(check_pid)
        assert child is not None and child.ppid == process.pid
        assert child.pgid != process.pid and child.sid != process.pid
        cleanup.update(table.snapshot())
        # The supervisor observes the family before the leader is killed and children reparent.
        assert handle.poll() is None
        assert signal_launch_group(process, handle.launch_identity, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while not kernel_exec_tracker.child_exited_unreaped(process) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert kernel_exec_tracker.child_exited_unreaped(process)
        monkeypatch.setattr(worker_module, "cleanup_seconds", lambda: 1.0)

        assert handle.reap(timeout=1) == -signal.SIGKILL

        remaining = table.identity(check_pid)
        assert remaining is None or remaining.zombie, "reap left the detached check alive"
        assert unrelated.poll() is None, "cleanup signalled a process outside the recorded family"
    finally:
        # Keep the red test safe too: signal only birth-verified members observed before death.
        cleanup.signal(cleanup.update(table.snapshot()), signal.SIGKILL)
        process.wait(timeout=5)
        assert unrelated.stdin is not None
        unrelated.stdin.close()
        unrelated.wait(timeout=5)
        reader.join(timeout=2)


@pytest.mark.parametrize("child_state", ["exited", "recycled", "survivor"])
def test_worker_reap_requires_recorded_detached_family_to_exit(tmp_path, monkeypatch, child_state):
    leader = ProcessIdentity(4444, 1, 4444, 4444, (100, 0), False)
    child = ProcessIdentity(5555, 4444, 5555, 5555, (101, 0), False)
    rows = {leader.pid: leader, child.pid: child}
    table = SimpleNamespace(identity=rows.get, snapshot=lambda: dict(rows))
    monkeypatch.setattr(worker_module, "ProcessTable", lambda: table)
    monkeypatch.setattr(worker_module, "child_exited_unreaped", lambda process: rows[4444].zombie)
    monkeypatch.setattr(worker_module, "signal_launch_group", lambda *args: False)
    monkeypatch.setattr(worker_module, "cleanup_seconds", lambda: 0.0)
    monkeypatch.setattr(kernel_exec_tracker, "sys", SimpleNamespace(platform="darwin"))
    delivered = []
    monkeypatch.setattr(kernel_exec_tracker, "signal_mac_verified_process",
                        lambda row, sig: delivered.append((row.pid, sig)) or False)
    waited = []
    process = SimpleNamespace(pid=leader.pid, returncode=None,
                              wait=lambda timeout: waited.append(timeout) or -signal.SIGKILL)
    reader = threading.Thread(target=lambda: None)
    reader.start()
    handle = worker_module.WorkerHandle(process, leader.pid, leader.pgid, 1, tmp_path,
                                        tmp_path / "worker.log", time.time(), time.monotonic(), reader,
                                        launch_identity=LaunchIdentity(leader.pid, leader.start, None))
    assert handle.poll() is None
    rows[4444] = ProcessIdentity(4444, 1, 4444, 4444, leader.start, True)
    rows[5555] = ProcessIdentity(5555, 1, 5555, 5555,
                                 (202, 0) if child_state == "recycled" else child.start, False)
    if child_state == "exited":
        del rows[5555]

    if child_state == "survivor":
        with pytest.raises(ContainmentUnverified, match="remains live"):
            handle.reap()
        assert waited == [], "an uncontained family must keep the leader unreaped"
        assert handle.exit_code is None
        assert delivered == [(5555, signal.SIGTERM), (5555, signal.SIGKILL)]
    else:
        assert handle.reap() == -signal.SIGKILL
        assert len(waited) == 1
        assert delivered == [], "exited or recycled members must never receive signals"
    reader.join(timeout=2)


def test_unreaped_worker_leader_with_wrong_birth_is_never_signalled(tmp_path, monkeypatch):
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(script)], start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        identity = kernel_exec_tracker.capture_launch_identity(process, session_leader=True)
        assert identity is not None and identity.start is not None
        assert process.returncode is None
        wrong_birth = (identity.start[0] + 1, identity.start[1])
        recycled = ProcessIdentity(process.pid, 1, process.pid, process.pid, wrong_birth, False)
        table = SimpleNamespace(identity=lambda pid: recycled)
        delivered = []
        monkeypatch.setattr(kernel_exec_tracker.os, "killpg",
                            lambda *args: delivered.append(args))

        assert signal_launch_group(process, identity, signal.SIGTERM, table=table) is False
        assert process.returncode is None
        assert delivered == []
    finally:
        process.kill()
        process.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS audit-token signal API")
def test_macos_audit_token_signal_checks_birth_and_stops_owned_process(tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(script)], start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        row = ProcessTable().identity(process.pid)
        assert row is not None and not row.zombie
        wrong = ProcessIdentity(row.pid, row.ppid, row.pgid, row.sid,
                                (row.start[0] + 1, row.start[1]), False)
        assert signal_mac_verified_process(wrong, signal.SIGKILL) is False
        assert process.poll() is None
        assert signal_mac_verified_process(row, signal.SIGKILL) == kernel_exec_tracker.MAC_KERNEL_SIGNAL_PATH
        assert process.wait(timeout=5) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_completed_check_refuses_a_live_group_without_numeric_signal(monkeypatch):
    live = ProcessIdentity(2222, 1, 2222, 2222, (100, 1), False)
    monkeypatch.setattr(kernel_exec_tracker, "ProcessTable", lambda: SimpleNamespace(
        snapshot=lambda: {2222: live}))
    tick = iter((0.0, 1.0, 3.0))
    monkeypatch.setattr(kernel_unit_policy.time, "monotonic", lambda: next(tick))
    delivered = []
    monkeypatch.setattr(os, "killpg", lambda *args: delivered.append(args))

    with pytest.raises(kernel_unit_policy.UnitCheckCleanupError, match="containment unverified"):
        kernel_unit_policy.reap_check_group(2222)
    assert delivered == []


def test_detached_check_member_is_refused_without_numeric_signal(monkeypatch):
    monkeypatch.setattr(kernel_unit_policy.sys, "platform", "darwin")
    monkeypatch.setattr(kernel_unit_policy, "_matching_check_processes", lambda token: (3333,))
    ticks = iter((0.0, 0.0, 11.0))
    monkeypatch.setattr(kernel_unit_policy.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(kernel_unit_policy.time, "sleep", lambda seconds: None)
    delivered = []
    monkeypatch.setattr(os, "kill", lambda *args: delivered.append(args))

    with pytest.raises(kernel_unit_policy.UnitCheckCleanupError, match="detached descendants"):
        kernel_unit_policy._reap_check_token("a" * 32)
    assert delivered == []


def test_detached_check_keeps_clone_until_short_lived_member_is_gone(monkeypatch):
    monkeypatch.setattr(kernel_unit_policy.sys, "platform", "darwin")
    observations = iter(((3333,), (), ()))
    inspected = []

    def scan(token):
        inspected.append(token)
        return next(observations)

    monkeypatch.setattr(kernel_unit_policy, "_matching_check_processes", scan)
    monkeypatch.setattr(kernel_unit_policy.time, "sleep", lambda seconds: None)
    with pytest.raises(kernel_unit_policy.UnitCheckCleanupError,
                       match="detached descendants \\(3333,\\) exited after the check"):
        kernel_unit_policy._reap_check_token("b" * 32)
    assert inspected == ["b" * 32] * 3


def test_detached_check_cleanup_signals_only_stable_kernel_identity(monkeypatch):
    monkeypatch.setattr(kernel_unit_policy.sys, "platform", "darwin")
    row = ProcessIdentity(3333, 1, 3333, 3333, (100, 1), False)
    scans = iter(((3333,), (3333,), (), ()))
    monkeypatch.setattr(kernel_unit_policy, "_matching_check_processes", lambda token: next(scans))
    monkeypatch.setattr(kernel_exec_tracker, "ProcessTable", lambda: SimpleNamespace(
        identity=lambda pid: row))
    delivered = []
    monkeypatch.setattr(kernel_exec_tracker, "signal_mac_verified_process",
                        lambda identity, sig: delivered.append((identity, sig)) or True)
    monkeypatch.setattr(kernel_unit_policy.time, "sleep", lambda seconds: None)

    finding = kernel_unit_policy._terminate_check_token("c" * 32)
    assert finding == "check result invalid: detached descendants (3333,) terminated after the check"
    assert delivered == [(row, signal.SIGKILL)]


def test_detached_check_cleanup_refuses_recycled_identity(monkeypatch):
    monkeypatch.setattr(kernel_unit_policy.sys, "platform", "darwin")
    before = ProcessIdentity(3333, 1, 3333, 3333, (100, 1), False)
    after = ProcessIdentity(3333, 1, 3333, 3333, (101, 1), False)
    identities = iter((before, after))
    monkeypatch.setattr(kernel_unit_policy, "_matching_check_processes", lambda token: (3333,))
    monkeypatch.setattr(kernel_exec_tracker, "ProcessTable", lambda: SimpleNamespace(
        identity=lambda pid: next(identities)))
    delivered = []
    monkeypatch.setattr(kernel_exec_tracker, "signal_mac_verified_process",
                        lambda identity, sig: delivered.append((identity, sig)) or True)

    with pytest.raises(kernel_unit_policy.UnitCheckCleanupError, match="changed identity"):
        kernel_unit_policy._terminate_check_token("d" * 32)
    assert delivered == []


def test_check_cleanup_runs_before_sandbox_leader_is_reaped(monkeypatch):
    calls = []

    class Process:
        pid = 3333
        stdin = None
        stdout = None
        stderr = None
        returncode = None

        def wait(self, timeout):
            calls.append("wait")
            self.returncode = 0
            return 0

    monkeypatch.setattr(kernel_execution_policy, "child_exited_unreaped", lambda process: True)
    monkeypatch.setattr(kernel_execution_policy, "recorded_launch_identity",
                        lambda process: LaunchIdentity(3333, (100, 1), None))
    monkeypatch.setattr(kernel_execution_policy, "signal_launch_group",
                        lambda *args, **kwargs: calls.append("group-signal") or True)
    monkeypatch.setattr(kernel_execution_policy, "ProcessTable", lambda: SimpleNamespace(
        snapshot=lambda: {}))
    result = kernel_execution_policy._collect(
        Process(), None, time.monotonic() + 1, 1024,
        pre_reap_cleanup=lambda: calls.append("token-cleanup"))

    assert calls == ["group-signal", "token-cleanup", "wait"]
    assert result[-1] is True
