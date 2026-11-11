"""Execution cleanup kills only on positive evidence that a process is stuck or escaped.

A fixed 5-second wait in ExecSession.close() SIGKILLed supervisors that were merely slow
to exit on a loaded host, and the kernel then killed their whole sandbox (exit 137). The
supervisor also killed a session when a reparented grandchild had not been recorded yet.
"""
import os
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

import kernel_exec_session as session_module
from kernel_exec_session import CleanupReport, ExecSession, ExecSessionError
from kernel_exec_supervisor import Supervisor
from kernel_exec_tracker import ProcessIdentity, capture_launch_identity


def _session(process, control):
    session = ExecSession.__new__(ExecSession)
    session._closed = False
    session._process = process
    session._control = control
    session._owner_write_fd = None
    session._log = None
    session._lock = threading.RLock()
    session.stop_all = lambda grace_seconds=2.0: CleanupReport(True, 0, (), "")
    session._rpc = lambda request: {}
    return session


def _sleeper(seconds):
    # Launched like the real supervisor: its own session, launch identity recorded.
    process = subprocess.Popen([sys.executable, "-I", "-c", f"import time; time.sleep({seconds})"],
                               start_new_session=True)
    capture_launch_identity(process, session_leader=True)
    return process


def test_slow_supervisor_exit_after_eof_is_waited_for_not_killed(monkeypatch):
    monkeypatch.setenv("STATEFRAME_EXEC_RPC_SECONDS", "10")
    monkeypatch.setattr(session_module, "_host_load", lambda: (0.0, os.cpu_count() or 1))
    ours, theirs = socket.socketpair()
    theirs.close()  # the supervisor released the session promptly
    process = _sleeper(6)  # ... and exits later than the old fixed 5 s wait
    _session(process, ours).close()
    assert process.returncode == 0


def test_unresponsive_live_supervisor_is_killed_and_reported(monkeypatch):
    monkeypatch.setenv("STATEFRAME_EXEC_RPC_SECONDS", "1")
    monkeypatch.setattr(session_module, "_host_load", lambda: (0.0, os.cpu_count() or 1))
    ours, theirs = socket.socketpair()
    process = _sleeper(60)
    try:
        with pytest.raises(ExecSessionError, match="stopped answering and did not exit"):
            _session(process, ours).close()
        assert process.returncode == -9
    finally:
        theirs.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def _row(pid, *, ppid, pgid, sid, start=(100, 0), zombie=False):
    return ProcessIdentity(pid, ppid, pgid, sid, start, zombie)


def _supervisor(rows, members, leader):
    supervisor = Supervisor.__new__(Supervisor)
    family = SimpleNamespace(members=dict(members), leader=leader, update=lambda _rows: [])
    supervisor.jobs = {"j1": SimpleNamespace(cleanup=None, family=family)}
    supervisor.table = SimpleNamespace(snapshot=lambda: dict(rows))
    return supervisor, family


def test_reparented_grandchild_sharing_a_job_session_is_attributed():
    own = os.getpid()
    leader = _row(500, ppid=own, pgid=os.getpgrp(), sid=os.getsid(0), start=(90, 0))
    inner = _row(501, ppid=500, pgid=501, sid=501)
    orphan = _row(777, ppid=own, pgid=501, sid=501, start=(120, 0))  # its parent exited between sweeps
    supervisor, family = _supervisor({500: leader, 501: inner, 777: orphan}, {500: leader, 501: inner}, 500)
    assert supervisor._unattributed([orphan]) == []
    assert 777 in family.members


def test_child_sharing_only_the_supervisor_session_is_still_an_escape():
    own = os.getpid()
    leader = _row(500, ppid=own, pgid=os.getpgrp(), sid=os.getsid(0), start=(90, 0))
    escaped = _row(778, ppid=own, pgid=os.getpgrp(), sid=os.getsid(0), start=(120, 0))
    supervisor, family = _supervisor({500: leader, 778: escaped}, {500: leader}, 500)
    assert supervisor._unattributed([escaped]) == [escaped]
    assert 778 not in family.members


def test_orphan_gone_by_the_second_look_is_not_killed():
    own = os.getpid()
    leader = _row(500, ppid=own, pgid=os.getpgrp(), sid=os.getsid(0), start=(90, 0))
    orphan = _row(779, ppid=own, pgid=9, sid=9, start=(120, 0))
    supervisor, _family = _supervisor({500: leader}, {500: leader}, 500)
    assert supervisor._unattributed([orphan]) == []


def test_orphan_older_than_the_job_is_not_attributed_to_it():
    own = os.getpid()
    leader = _row(500, ppid=own, pgid=os.getpgrp(), sid=os.getsid(0), start=(90, 0))
    inner = _row(501, ppid=500, pgid=501, sid=501)
    older = _row(780, ppid=own, pgid=501, sid=501, start=(10, 0))
    supervisor, _family = _supervisor({500: leader, 501: inner, 780: older}, {500: leader, 501: inner}, 500)
    assert supervisor._unattributed([older]) == [older]
