"""Startup probe failures retain bounded diagnostics before their temp tree is removed."""
from __future__ import annotations

import errno
import os
import re
import signal
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from fixtures.process_signals import (
    close_signal_binding,
    record_identity,
    retain_signal_binding,
    signal_bound,
)
from stateframe import runner
from stateframe.engine import kernel_exec_session, kernel_safe_probe
from stateframe.engine.kernel_exec_session import ExecSession, ExecSessionError
from stateframe.engine.kernel_exec_tracker import ProcessTable
from stateframe.engine.kernel_safe_probe import ProbeCleanupError
from kernel_exec_session import ExecSession as SmokeExecSession
from kernel_ship_tree_smoke import ImportSmokeCleanupError, _execute_smoke


def _retained_process_status_pattern():
    if sys.platform.startswith("linux"):
        return r"launcher \(bwrap\) exit status: -?\d+"
    return r"supervisor exit status: -?\d+"


def _supervisor_command_fragment() -> str:
    return "kernel_exec_supervisor.py"


def _linux_supervisor_pid(launcher_pid: int, *, timeout: float = 5.0) -> int:
    """Find this bwrap launcher's supervisor, rather than signalling bwrap itself."""
    table = ProcessTable()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = table.snapshot()
        descendants = {launcher_pid}
        while True:
            children = {row.pid for row in rows.values() if row.ppid in descendants}
            new_children = children - descendants
            if not new_children:
                break
            descendants.update(new_children)

        matches = []
        for pid in sorted(descendants - {launcher_pid}):
            try:
                command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
            except FileNotFoundError:
                continue
            if b"kernel_exec_supervisor.py" in command:
                matches.append(pid)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError("bwrap launch has multiple execution supervisors")
        time.sleep(0.01)
    raise RuntimeError("could not find execution supervisor below the bwrap launcher")


def _signal_target_pid(session) -> int:
    process = session._process
    if process is None or process.poll() is not None:
        raise RuntimeError("execution launcher exited before the death plant")
    if sys.platform.startswith("linux") and session._process_role == "launcher (bwrap)":
        return _linux_supervisor_pid(process.pid)
    return process.pid


def _wait_for_signal_target_exit(identity, *, timeout: float = 5.0) -> None:
    if not sys.platform.startswith("linux"):
        return
    expected_start = (int(identity.kernel_identity.rsplit(":", 1)[1]), 0)
    table = ProcessTable()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = table.identity(identity.pid)
        if current is None or current.start != expected_start or current.zombie:
            return
        time.sleep(0.01)
    raise RuntimeError(f"retained supervisor {identity.pid} remained live after SIGKILL")


def _plant_supervisor_death(session, *, launched_after: float) -> None:
    """Kill and reap the supervisor, failing the test as setup if that cannot be proved."""
    binding = None
    try:
        target_pid = _signal_target_pid(session)
        identity = record_identity(
            target_pid, launched_after=launched_after,
            command_fragment=_supervisor_command_fragment())
        binding = retain_signal_binding(identity)
        signal_bound(binding, signal.SIGKILL)
        _wait_for_signal_target_exit(identity)
        session._process.wait(timeout=5)
    except Exception as error:
        cleanup_error = None
        try:
            session.close()
        except Exception as close_error:
            cleanup_error = close_error
        detail = f"setup: could not plant supervisor death: {type(error).__name__}: {error}"
        if cleanup_error is not None:
            detail += f"; session cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
        pytest.fail(detail, pytrace=False)
    finally:
        if binding is not None:
            try:
                close_signal_binding(binding)
            except Exception as error:
                pytest.fail(
                    f"setup: could not close retained supervisor binding: "
                    f"{type(error).__name__}: {error}",
                    pytrace=False)


@pytest.mark.host_sandbox
def test_supervisor_death_before_start_keeps_exit_status_log_tail_and_base_error(
        monkeypatch, tmp_path):
    marker = "planted supervisor log tail"

    class DiedBeforeStartSession(ExecSession):
        def _rpc(self, request):
            if request.get("op") == "start" and not getattr(self, "_death_planted", False):
                self._death_planted = True
                _plant_supervisor_death(self, launched_after=launched_after)
                self._log.write((marker + "\n").encode())
            return super()._rpc(request)

    monkeypatch.setattr(kernel_safe_probe, "ExecSession", DiedBeforeStartSession)

    launched_after = time.time()
    with pytest.raises(ProbeCleanupError) as caught:
        runner._startup_probe(sys.executable, "pass", label="engine interpreter test")

    message = str(caught.value)
    assert type(caught.value) is ProbeCleanupError
    assert message.startswith(
        "automatic probe cleanup failed: execution supervisor communication failed:")
    assert "primary error:" not in message
    assert re.search(_retained_process_status_pattern(), message)
    assert "supervisor.log tail (up to 8192 bytes):" in message
    assert marker in message


def test_missing_supervisor_log_is_reported_as_absent(tmp_path):
    session = ExecSession(tmp_path)

    diagnostic = session.startup_diagnostics()

    assert "execution process status unknown" in diagnostic
    assert "supervisor.log absent" in diagnostic


def test_poll_returning_none_reports_supervisor_status_unknown(tmp_path):
    session = ExecSession(tmp_path)

    class RunningSupervisor:
        def poll(self):
            return None

    session._process = RunningSupervisor()
    session._process_role = "supervisor"

    diagnostic = session.startup_diagnostics()

    assert diagnostic.startswith("supervisor status unknown\n")


def test_poll_error_reports_status_unknown(tmp_path):
    session = ExecSession(tmp_path)

    class UnreadableStatus:
        def poll(self):
            raise OSError("status unavailable")

    session._process = UnreadableStatus()
    session._process_role = "supervisor"

    diagnostic = session.startup_diagnostics()

    assert diagnostic.startswith("supervisor status unknown\n")


def test_symlinked_supervisor_log_is_not_followed(tmp_path):
    session = ExecSession(tmp_path)
    outside = tmp_path / "outside.log"
    outside.write_text("must not be read", encoding="utf-8")
    (session._directory / "supervisor.log").symlink_to(outside)

    diagnostic = session.startup_diagnostics()

    assert "supervisor.log absent" in diagnostic
    assert "must not be read" not in diagnostic


def test_symlinked_supervisor_log_parent_is_not_followed(tmp_path):
    session = ExecSession(tmp_path)
    real_directory = session._directory
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "supervisor.log").write_text("must not be read", encoding="utf-8")
    real_directory.rmdir()
    real_directory.symlink_to(outside, target_is_directory=True)

    diagnostic = session.startup_diagnostics()

    assert "supervisor.log absent" in diagnostic
    assert "must not be read" not in diagnostic


def test_non_regular_supervisor_log_is_reported_as_absent(tmp_path):
    session = ExecSession(tmp_path)
    os.mkfifo(session._directory / "supervisor.log")

    diagnostic = session.startup_diagnostics()

    assert "supervisor.log absent" in diagnostic
    assert "supervisor.log tail" not in diagnostic


def test_supervisor_log_read_error_is_reported_as_absent(tmp_path, monkeypatch):
    session = ExecSession(tmp_path)
    (session._directory / "supervisor.log").write_text("log", encoding="utf-8")

    def unreadable(_descriptor, _size):
        raise OSError(errno.EIO, "simulated read failure")

    monkeypatch.setattr(kernel_exec_session.os, "read", unreadable)

    diagnostic = session.startup_diagnostics()

    assert "supervisor.log absent" in diagnostic
    assert "simulated read failure" not in diagnostic


def test_log_tail_is_bounded_and_decodes_invalid_utf8_with_replacement(tmp_path):
    session = ExecSession(tmp_path)
    log = session._directory / "supervisor.log"
    log.write_bytes(b"old prefix " + b"A" * 9000 + b"\xfftail")

    diagnostic = session.startup_diagnostics()
    tail = diagnostic.split("supervisor.log tail (up to 8192 bytes):\n", 1)[1]

    assert tail.count("A") == 8187
    assert "old prefix" not in tail
    assert "\ufffd" in tail
    assert tail.endswith("tail")


@pytest.mark.host_sandbox
@pytest.mark.parametrize(("sandboxed", "expected_status", "wrapped"), [
    (True, "launcher (bwrap) exit status: 19", True),
    (False, "supervisor exit status: 19", False),
])
def test_status_label_names_the_retained_bwrap_launcher_or_supervisor(
        monkeypatch, tmp_path, sandboxed, expected_status, wrapped):
    session = ExecSession(tmp_path, platform_backend="bwrap")
    commands = []

    class Process:
        def poll(self):
            return 19

        def wait(self, timeout):
            return 19

    monkeypatch.setattr(kernel_exec_session, "sandbox_executable", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(kernel_exec_session, "capture_launch_identity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kernel_exec_session, "_current_cgroup", lambda: None)
    if sys.platform.startswith("linux"):
        monkeypatch.setattr(kernel_exec_session.os, "getuid", lambda: 1000)

    def fake_popen(command, **_kwargs):
        commands.append(command)
        return Process()

    monkeypatch.setattr(kernel_exec_session.subprocess, "Popen", fake_popen)
    capability = SimpleNamespace(
        sandboxed=sandboxed,
        network="none",
        write_roots=(),
        workspace=tmp_path,
        workspace_writable=False,
    )

    session._start(capability)
    diagnostic = session.startup_diagnostics()
    assert expected_status in diagnostic
    if wrapped:
        assert commands[0][0] == "/usr/bin/bwrap"
        assert "supervisor exit status" not in diagnostic
    else:
        assert commands[0][0] == sys.executable

    monkeypatch.setattr(session, "_rpc", lambda _request: {"jobs": {}})
    session.close()


@pytest.mark.parametrize("failure", ["timeout", "eof", "socket"])
def test_rpc_failure_keeps_exact_leading_message_and_appends_diagnostics(
        monkeypatch, tmp_path, failure):
    session = ExecSession(tmp_path)

    class RunningSupervisor:
        def poll(self):
            return None

    class FailingControl:
        def sendall(self, _payload):
            if failure == "socket":
                raise OSError("broken socket")

        def recv(self, _size):
            if failure == "timeout":
                raise socket.timeout("timed out")
            return b""

        def gettimeout(self):
            return 0.25

    session._process = RunningSupervisor()
    session._process_role = "supervisor"
    session._control = FailingControl()
    monkeypatch.setattr(kernel_exec_session, "_host_load", lambda: (0.5, 2))

    expected_leading = {
        "timeout": (
            "execution supervisor did not answer within 0.25 s (host load 0.5 on 2 CPUs); "
            "retry when the machine is less busy or raise STATEFRAME_EXEC_RPC_SECONDS"
        ),
        "eof": f"execution supervisor exited; inspect {session._directory / 'supervisor.log'}",
        "socket": "execution supervisor communication failed: broken socket",
    }[failure]
    expected = (f"{expected_leading}\nstartup supervisor diagnostics:\n"
                "supervisor status unknown\nsupervisor.log absent")

    with pytest.raises(ExecSessionError) as caught:
        session._rpc({"op": "start"})

    assert type(caught.value) is ExecSessionError
    assert str(caught.value) == expected


@pytest.mark.host_sandbox
def test_import_smoke_supervisor_death_includes_diagnostics(monkeypatch, tmp_path):
    marker = "planted import smoke supervisor log tail"

    class DiedBeforeStartSession(SmokeExecSession):
        def _rpc(self, request):
            if request.get("op") == "start" and not getattr(self, "_death_planted", False):
                self._death_planted = True
                _plant_supervisor_death(self, launched_after=launched_after)
                self._log.write((marker + "\n").encode())
            return super()._rpc(request)

    monkeypatch.setattr("kernel_exec_session.ExecSession", DiedBeforeStartSession)
    monkeypatch.setenv("STATEFRAME_TEST_PYTHON", sys.executable)
    (tmp_path / "smoke.py").write_text("VALUE = 1\n", encoding="utf-8")

    launched_after = time.time()
    with pytest.raises(ImportSmokeCleanupError) as caught:
        _execute_smoke(tmp_path, ("smoke.py",), (str(tmp_path),), 3)

    message = str(caught.value)
    assert "execution supervisor communication failed:" in message
    assert "startup supervisor diagnostics:" in message
    assert re.search(_retained_process_status_pattern(), message)
    assert marker in message
