"""Security regressions for mutation shutdown cleanup."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_exec_session import CleanupReport, ExecSession, ExecSessionError
from kernel_mutate_exec import MutationContainmentError, MutationExecutor


def test_supervisor_stop_timeout_retains_artifacts_until_reaped(tmp_path, monkeypatch):
    import kernel_exec_session as sessions

    events = []
    finalizer = lambda: events.append("pidfd")

    class Process:
        _stateframe_pidfd_finalizer = staticmethod(finalizer)

        def poll(self):
            return None

        def wait(self, timeout):
            events.append("wait")
            if events.count("wait") == 1:
                raise subprocess.TimeoutExpired("supervisor", timeout)

    class Control:
        def settimeout(self, timeout):
            pass

        def recv(self, size):
            return b""

        def close(self):
            events.append("control")

    session = object.__new__(ExecSession)
    session._closed = False
    session._process = Process()
    session._control = Control()
    session._log = (tmp_path / "supervisor.log").open("wb")
    session._initial_capability = SimpleNamespace(purpose="mutation")
    session._close_owner_write_end = lambda: events.append("owner")
    session.stop_all = lambda *args: CleanupReport(True, 0, (), "")
    session._rpc = lambda request: events.append("rpc")
    survivors = iter(([123], []))
    monkeypatch.setattr(sessions, "kill_launch_unit", lambda *args, **kwargs: next(survivors))
    monkeypatch.setattr(sessions, "recorded_launch_identity", lambda process: None)

    root = tmp_path / "mutation"
    for name in ("session", "boot", Path("stage") / ("a" * 32)):
        (root / name).mkdir(parents=True)
    executor = object.__new__(MutationExecutor)
    executor._closed = False
    executor._session = session
    executor._session_dir = root / "session"
    executor._boot = root / "boot"
    executor._stage = root / "stage"
    executor._ledger = (root / "launches.jsonl").open("wb")

    with pytest.raises(MutationContainmentError, match="did not terminate"):
        executor.close()
    assert session._log.closed
    assert events.count("pidfd") == 1
    assert events.count("owner") == 1
    assert events.count("control") == 1
    assert executor._session_dir.exists()
    assert executor._boot.exists()
    assert executor._stage.exists()
    assert executor._ledger.closed
    assert not session._closed and not executor._closed

    assert executor.close().ok
    assert session._closed and executor._closed
    assert not executor._session_dir.exists()
    assert not executor._boot.exists()
    assert not executor._stage.exists()
    assert executor._ledger.closed
    assert events.count("wait") == 1
    assert events.count("pidfd") == 1


def test_failed_stop_rpc_releases_descriptors_for_retry(tmp_path, monkeypatch):
    import kernel_exec_session as sessions
    from kernel_exec_tracker import LaunchIdentity

    events = []

    class Process:
        _stateframe_pidfd_finalizer = staticmethod(lambda: events.append("pidfd closed"))

        def wait(self, timeout):
            events.append("wait")

    class Control:
        def close(self):
            events.append("control closed")

    session = object.__new__(ExecSession)
    session._closed = False
    session._process = Process()
    session._process._stateframe_launch_identity = LaunchIdentity(123, (1, 2), "boot", 77)
    session._control = Control()
    session._log = (tmp_path / "supervisor.log").open("wb")
    session._initial_capability = None
    session._close_owner_write_end = lambda: events.append("owner closed")
    attempts = 0

    def stop_all(*_args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ExecSessionError("transient stop RPC failure")
        events.append("stopped")
        return CleanupReport(True, 0, (), "")

    session.stop_all = stop_all
    session._rpc = lambda request: events.append(request["op"])
    survivors = iter(([123], []))
    monkeypatch.setattr(sessions, "kill_launch_unit", lambda *args, **kwargs: next(survivors))

    with pytest.raises(ExecSessionError, match="survivors"):
        session.close()
    # The supervisor gets its load-scaled exit wait before the launch unit is verified.
    assert events == ["owner closed", "control closed", "wait", "pidfd closed"]
    assert session._process._stateframe_launch_identity.pidfd is None
    assert not session._closed and session._log.closed

    session.close()
    assert events == ["owner closed", "control closed", "wait", "pidfd closed", "owner closed"]
    assert session._closed and session._log.closed


def test_failed_close_rpc_does_not_poison_verified_retry(tmp_path):
    events = []

    class Process:
        def poll(self):
            return None

        def wait(self, timeout):
            events.append("wait")

    class Control:
        def settimeout(self, timeout):
            pass

        def recv(self, size):
            return b""

        def close(self):
            events.append("control closed")

    session = object.__new__(ExecSession)
    session._closed = False
    session._process = Process()
    session._control = Control()
    session._log = (tmp_path / "supervisor.log").open("wb")
    session._initial_capability = SimpleNamespace(purpose="mutation")
    session._close_owner_write_end = lambda: None
    session.stop_all = lambda *_args: CleanupReport(True, 0, (), "")

    def close_rpc(_request):
        events.append("close RPC")
        raise ExecSessionError("close RPC unavailable")

    session._rpc = close_rpc
    with pytest.raises(ExecSessionError, match="close RPC unavailable"):
        session.close()
    assert not session._closed
    assert session._log.closed
    assert events == ["close RPC", "control closed", "wait"]

    session.close()
    assert session._closed
    assert events == ["close RPC", "control closed", "wait"]


def test_failed_stop_does_not_sweep_unverified_mutation_artifacts(tmp_path):
    root = tmp_path / "mutation"
    for name in ("session", "boot", Path("stage") / ("a" * 32)):
        (root / name).mkdir(parents=True)
    executor = object.__new__(MutationExecutor)
    executor._closed = False
    executor._session_dir = root / "session"
    executor._boot = root / "boot"
    executor._stage = root / "stage"
    executor._ledger = (root / "launches.jsonl").open("wb")
    attempts = 0

    class Session:
        def stop_all(self, *_args):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ExecSessionError("stop RPC unavailable")
            return CleanupReport(True, 0, (), "")

        def close(self):
            if attempts == 1:
                raise ExecSessionError("exit unverified")

    executor._session = Session()
    try:
        with pytest.raises(MutationContainmentError, match="exit unverified"):
            executor.close()
        assert (root / "stage" / ("a" * 32)).is_dir()
        assert executor._session_dir.is_dir() and executor._boot.is_dir()
        assert executor._ledger.closed
        assert not executor._closed

        assert executor.close().ok
        assert not executor._stage.exists()
        assert not executor._session_dir.exists() and not executor._boot.exists()
        assert executor._ledger.closed and executor._closed
    finally:
        if not executor._ledger.closed:
            executor._ledger.close()


def test_survivor_report_retains_artifacts_but_closes_ledger(tmp_path):
    root = tmp_path / "mutation"
    for name in ("session", "boot", Path("stage") / ("a" * 32)):
        (root / name).mkdir(parents=True)
    executor = object.__new__(MutationExecutor)
    executor._closed = False
    executor._session_dir = root / "session"
    executor._boot = root / "boot"
    executor._stage = root / "stage"
    executor._ledger = (root / "launches.jsonl").open("wb")
    attempts = 0

    class Session:
        def stop_all(self, *_args):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return CleanupReport(False, 0, (123,), "survivor")
            return CleanupReport(True, 0, (), "")

        def close(self):
            pass

    executor._session = Session()
    with pytest.raises(MutationContainmentError, match="survivor"):
        executor.close()
    assert executor._ledger.closed
    assert executor._stage.is_dir()
    assert executor._session_dir.is_dir() and executor._boot.is_dir()
    assert not executor._closed

    assert executor.close().ok
    assert executor._closed and not executor._stage.exists()


def test_slow_stop_waits_for_supervisor_exit_before_verifying(tmp_path, monkeypatch):
    """A stop that misses its deadline on a loaded host must not SIGKILL a supervisor
    that is still exiting on its own; the launch unit is verified only after the wait."""
    import kernel_exec_session as sessions

    events = []

    class Process:
        def poll(self):
            return None

        def wait(self, timeout):
            events.append(("wait", timeout))

    class Control:
        def close(self):
            events.append("control closed")

    session = object.__new__(ExecSession)
    session._closed = False
    session._process = Process()
    session._control = Control()
    session._log = (tmp_path / "supervisor.log").open("wb")
    session._initial_capability = None
    session._close_owner_write_end = lambda: None
    session.stop_all = lambda *_args: CleanupReport(False, 0, (), "session cleanup deadline exceeded")
    session._rpc = lambda request: events.append(request["op"])
    monkeypatch.setattr(sessions, "rpc_timeout_seconds", lambda: 42.0)
    monkeypatch.setattr(sessions, "recorded_launch_identity", lambda process: None)
    monkeypatch.setattr(sessions, "kill_launch_unit",
                        lambda *args, **kwargs: events.append("verify launch unit") or [])

    session.close()

    assert events == ["control closed", ("wait", 42.0), "verify launch unit"]
    assert session._closed and session._log.closed
