"""close() kills a supervisor only on evidence of no progress, never on elapsed time.

A loaded host can keep a supervisor tearing down its jobs for longer than any fixed bound.
Killing it then takes its whole sandbox with it (exit 137). close() samples the launch unit
(CPU ticks, members, zombies) and escalates only after a load-scaled run of samples in which
nothing moved; a vanished or unreadable process is unknown, never a stall.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

import kernel_exec_session as sessions
from kernel_exec_session import CleanupReport, ExecSession, ExecSessionError
from kernel_exec_tracker import ProcessIdentity, ProcessTable
from fixtures.process_signals import signal_run_owned


class _Process:
    """A supervisor double that exits after `exit_after` waits."""

    def __init__(self, exit_after: int | None):
        self.pid = 4242
        self.waits = 0
        self.exit_after = exit_after

    def poll(self):
        return None

    def wait(self, timeout):
        self.waits += 1
        if self.exit_after is not None and self.waits > self.exit_after:
            return 0
        raise subprocess.TimeoutExpired("supervisor", timeout)


class _Control:
    def settimeout(self, timeout):
        pass

    def recv(self, size):
        return b""

    def close(self):
        pass


def _session(tmp_path, process) -> ExecSession:
    session = object.__new__(ExecSession)
    session._closed = False
    session._process = process
    session._control = _Control()
    session._log = (tmp_path / "supervisor.log").open("wb")
    session._initial_capability = SimpleNamespace(purpose="mutation")
    session._close_owner_write_end = lambda: None
    session.stop_all = lambda *_args: CleanupReport(True, 0, (), "")
    session._rpc = lambda _request: None
    return session


@pytest.fixture
def harness(monkeypatch):
    kills = []
    monkeypatch.setattr(sessions, "CLOSE_SAMPLE_SECONDS", 0.25)
    monkeypatch.setattr(sessions, "rpc_timeout_seconds", lambda: 1.0)  # window: 4 samples
    monkeypatch.setattr(sessions, "recorded_launch_identity", lambda process: None)
    monkeypatch.setattr(sessions, "kill_launch_unit",
                        lambda *args, **kwargs: kills.append(args) or [])
    return kills


def _members(*pids):
    return frozenset((pid, (pid, 0)) for pid in pids)


def test_slow_but_progressing_stop_is_never_killed(tmp_path, monkeypatch, harness, caplog):
    # CPU ticks keep advancing for far longer than the stall window, then the supervisor exits.
    process = _Process(exit_after=200)
    ticks = iter(range(10_000))
    monkeypatch.setattr(sessions, "_launch_unit_progress",
                        lambda proc, table: (next(ticks), _members(4242, 4243), 0))
    session = _session(tmp_path, process)
    with caplog.at_level(logging.WARNING):
        session.close()
    assert session._closed and process.waits == 201
    assert harness == []
    assert "made no progress" not in caplog.text


def test_members_exiting_counts_as_progress(tmp_path, monkeypatch, harness, caplog):
    # Constant CPU, but the member set keeps shrinking and refilling as jobs are reaped.
    process = _Process(exit_after=60)
    sets = iter([_members(4242, 4243), _members(4242)] * 40)
    monkeypatch.setattr(sessions, "_launch_unit_progress",
                        lambda proc, table: (5, next(sets), 0))
    session = _session(tmp_path, process)
    with caplog.at_level(logging.WARNING):
        session.close()
    assert session._closed and harness == []
    assert "made no progress" not in caplog.text


def test_stalled_supervisor_is_killed_with_the_reason_logged(tmp_path, monkeypatch, harness, caplog):
    process = _Process(exit_after=None)
    monkeypatch.setattr(sessions, "_launch_unit_progress",
                        lambda proc, table: (9, _members(4242, 4243), 0))
    session = _session(tmp_path, process)
    with caplog.at_level(logging.WARNING), \
            pytest.raises(ExecSessionError, match="made no progress for 1 s"):
        session.close()
    assert len(harness) == 1
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["[stateframe] execution supervisor 4242 made no progress for 4 samples "
                        "(cpu ticks 9, members 2); killing launch unit"]
    # First sample is the baseline; four unchanged samples after it escalate.
    assert process.waits == 5
    # The kill reaped it without survivors: a retry only re-verifies and then finishes.
    session.close()
    assert session._closed and len(harness) == 1 and process.waits == 5


def test_stalled_supervisor_window_scales_with_load(tmp_path, monkeypatch, harness):
    monkeypatch.setattr(sessions, "rpc_timeout_seconds", lambda: 60.0)
    monkeypatch.setattr(sessions, "CLOSE_SAMPLE_SECONDS", 0.5)
    process = _Process(exit_after=None)
    process.wait = lambda timeout: (setattr(process, "waits", process.waits + 1),
                                    (_ for _ in ()).throw(
                                        subprocess.TimeoutExpired("supervisor", timeout)))
    monkeypatch.setattr(sessions, "_launch_unit_progress",
                        lambda proc, table: (1, _members(4242), 0))
    with pytest.raises(ExecSessionError, match="made no progress for 60 s"):
        _session(tmp_path, process).close()
    assert process.waits == 121 and len(harness) == 1


@pytest.mark.parametrize("unknown", ["vanished", "procfs-error"])
def test_unreadable_fingerprint_is_never_a_stall(tmp_path, monkeypatch, harness, caplog, unknown):
    process = _Process(exit_after=50)
    table = SimpleNamespace(cpu_ticks=lambda row: 3, refresh_members=lambda m: dict(m))
    if unknown == "vanished":
        table.identity = lambda pid: None
    else:
        def identity(pid):
            raise PermissionError("procfs unreadable")
        table.identity = identity
    monkeypatch.setattr(sessions, "ProcessTable", lambda: table)
    session = _session(tmp_path, process)
    with caplog.at_level(logging.WARNING):
        session.close()
    assert session._closed and harness == []
    assert "made no progress" not in caplog.text


def test_reused_pid_is_unknown_not_progress_or_stall(monkeypatch):
    from kernel_exec_tracker import LaunchIdentity

    process = SimpleNamespace(pid=77)
    row = ProcessIdentity(77, 1, 77, 77, (900, 0), False)
    table = SimpleNamespace(identity=lambda pid: row, refresh_members=lambda m: dict(m),
                            cpu_ticks=lambda r: 4)
    monkeypatch.setattr(sessions, "recorded_launch_identity",
                        lambda proc: LaunchIdentity(77, (100, 0), "boot"))
    assert sessions._launch_unit_progress(process, table) is None
    monkeypatch.setattr(sessions, "recorded_launch_identity",
                        lambda proc: LaunchIdentity(77, (900, 0), "boot"))
    assert sessions._launch_unit_progress(process, table) == (4, frozenset({(77, (900, 0))}), 0)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads Linux /proc")
def test_cpu_ticks_reads_live_process_and_rejects_reuse():
    table = ProcessTable()
    row = table.identity(os.getpid())
    assert isinstance(table.cpu_ticks(row), int)
    assert table.cpu_ticks(ProcessIdentity(row.pid, row.ppid, row.pgid, row.sid,
                                           (row.start[0] + 1, 0), False)) is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads Linux /proc")
def test_real_slow_cleanup_under_load_is_reaped_without_sigkill(tmp_path, monkeypatch):
    # A supervisor double that burns CPU in a slow cleanup well past the stall window.
    monkeypatch.setattr(sessions, "rpc_timeout_seconds", lambda: 1.0)
    monkeypatch.setattr(sessions, "CLOSE_SAMPLE_SECONDS", 0.1)
    kills = []
    monkeypatch.setattr(sessions, "kill_launch_unit", lambda *a, **k: kills.append(a) or [])
    script = textwrap.dedent("""
        import time
        end = time.monotonic() + 3.0
        while time.monotonic() < end:
            sum(range(20000))
    """)
    process = subprocess.Popen(["nice", "-n", "19", sys.executable, "-c", script])
    try:
        assert sessions._await_supervisor_progress(process) is None
    finally:
        if process.poll() is None:
            signal_run_owned(tmp_path, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="slow cleanup double", require_checkpoint_saved=False)
            process.wait()
    assert kills == []
    assert process.returncode == 0 and process.returncode != -signal.SIGKILL


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads Linux /proc")
def test_real_idle_supervisor_is_reported_stalled(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "rpc_timeout_seconds", lambda: 0.5)
    monkeypatch.setattr(sessions, "CLOSE_SAMPLE_SECONDS", 0.1)
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        stall = sessions._await_supervisor_progress(process)
    finally:
        signal_run_owned(tmp_path, process, signal.SIGKILL, deadline=time.monotonic(),
                         what="idle supervisor double", require_checkpoint_saved=False)
        process.wait()
    assert stall is not None
    samples, (_ticks, members, _zombies) = stall
    assert samples == 5
    assert any(pid == process.pid for pid, _start in members)
