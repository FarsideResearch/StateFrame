"""Focused unit coverage for orphan-guard lifetime and handle contracts."""
from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import _orphan_guard as orphan_guard
from test_orphan_guard_acceptance import _STRIP_TOKEN_SCRIPT


def test_stripped_token_helper_target_compiles(monkeypatch):
    executed_argv: list[list[str]] = []

    def capture_execve(_executable, argv, _environment):
        executed_argv.append(argv)

    monkeypatch.setattr(os, "execve", capture_execve)
    monkeypatch.setattr(sys, "argv", ["stripped-token-helper", "SF_TEST_TOKEN"])
    exec(_STRIP_TOKEN_SCRIPT, {})

    assert len(executed_argv) == 1
    compile(executed_argv[0][-1], "<stripped-token-target>", "exec")


class _LiveHandle:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.identity = ("test-handle", pid)
        self.alive = True
        self.signals: list[int] = []
        self.closed = False

    def is_current(self) -> bool:
        return not self.closed

    def is_alive(self) -> bool:
        return self.alive

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)
        self.alive = False

    def wait_exit(self, timeout: float) -> bool:
        return not self.alive

    def close(self) -> None:
        self.closed = True


def _launch_helper(tmp_path: Path):
    script = tmp_path / "registered_helper.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    handles: list[_LiveHandle] = []

    def handle_factory(process: subprocess.Popen[bytes], *, platform: str) -> _LiveHandle:
        handle = _LiveHandle(process.pid)
        handles.append(handle)
        return handle

    guard = orphan_guard.OrphanGuard(tmp_path, handle_factory=handle_factory)
    scope = guard.scope("unit-helper")
    launch = scope.launch(
        [sys.executable, str(script)], cwd=tmp_path, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return guard, scope, launch, handles[0]


def _stop_helper(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5.0)


def test_registered_launch_waits_until_scope_ends(tmp_path):
    guard, scope, launch, handle = _launch_helper(tmp_path)
    try:
        report = guard.check()

        assert report.outcomes == ()
        assert not report.failed
        assert launch.process.poll() is None
        assert handle.signals == []
    finally:
        scope.end()
        _stop_helper(launch.process)
        guard.close()


def test_ended_registered_live_launch_is_a_leak_stopped_by_bound_handle(tmp_path):
    guard, scope, launch, handle = _launch_helper(tmp_path)
    scope.end()
    try:
        report = guard.check()

        assert [outcome.kind for outcome in report.outcomes] == ["leak", "confirmed exit"]
        assert all(outcome.pid == launch.pid for outcome in report.outcomes)
        assert report.failed
        assert handle.signals == [signal.SIGTERM]
    finally:
        _stop_helper(launch.process)
        guard.close()


def test_live_registered_launch_remains_a_leak_when_bound_signal_fails(tmp_path):
    script = tmp_path / "registered_helper.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    handles = []
    signal_observations = []

    class _SignalFailureHandle(_LiveHandle):
        def send_signal(self, signum: int) -> None:
            signal_observations.append(tuple(outcome.kind for outcome in guard._outcomes))
            self.signals.append(signum)
            raise PermissionError(errno.EPERM, "signal denied")

    def handle_factory(process: subprocess.Popen[bytes], *, platform: str):
        handle = _SignalFailureHandle(process.pid)
        handles.append(handle)
        return handle

    guard = orphan_guard.OrphanGuard(tmp_path, handle_factory=handle_factory)
    scope = guard.scope("signal-failure")
    launch = scope.launch(
        [sys.executable, str(script)], cwd=tmp_path, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    scope.end()
    try:
        report = guard.check()

        assert [outcome.kind for outcome in report.outcomes] == [
            "leak", "cleanup unverified"]
        assert all(outcome.pid == launch.pid for outcome in report.outcomes)
        assert "remained alive" in report.outcomes[0].detail
        assert "signal denied" in report.outcomes[1].detail
        assert report.unit_outcomes[0].kind == "cleanup unverified"
        assert handles[0].signals == [signal.SIGTERM]
        assert signal_observations == [("leak",)]
    finally:
        _stop_helper(launch.process)
        guard.close()


def test_live_registered_launch_remains_a_leak_when_check_deadline_expires(tmp_path, monkeypatch):
    guard, scope, launch, handle = _launch_helper(tmp_path)
    scope.end()
    monkeypatch.setattr(orphan_guard, "_CHECK_DEADLINE_SECONDS", 0.0)
    try:
        report = guard.check()

        assert [outcome.kind for outcome in report.outcomes] == [
            "leak", "cleanup unverified"]
        assert all(outcome.pid == launch.pid for outcome in report.outcomes)
        assert "remained alive" in report.outcomes[0].detail
        assert "deadline expired before SIGTERM" in report.outcomes[1].detail
        assert report.unit_outcomes[0].kind == "cleanup unverified"
        assert handle.signals == []
    finally:
        _stop_helper(launch.process)
        guard.close()


@pytest.mark.parametrize(
    ("event", "expected_kind"),
    [
        (orphan_guard.select.POLLIN, "confirmed exit"),
        (orphan_guard.select.POLLHUP, "confirmed exit"),
        (orphan_guard.select.POLLERR, "cleanup unverified"),
        (orphan_guard.select.POLLNVAL, "cleanup unverified"),
    ],
)
def test_pidfd_poll_only_confirms_exit_for_pollin_or_pollhup(
        tmp_path, monkeypatch, event, expected_kind):
    pid = 7341

    class _ReadyPoller:
        def poll(self, _timeout_ms):
            return [(23, event)]

    handle = object.__new__(orphan_guard._PidfdHandle)
    handle.pid = pid
    handle._fd = 23
    handle.identity = ("linux-pidfd", pid, 23)
    handle._poller = _ReadyPoller()
    monkeypatch.setattr(orphan_guard.os, "close", lambda _fd: None)

    guard = orphan_guard.OrphanGuard(tmp_path)
    scope = guard.scope("pidfd-poll")
    scope.register_adopted(
        pid, bound_handle=handle, attributed=True,
        parent_transition=(123, 456), parent_exit_confirmed=True,
        tmp_path_use=True)
    scope.end()
    try:
        report = guard.check()

        assert [outcome.kind for outcome in report.outcomes] == [expected_kind]
        assert report.unit_outcomes[0].kind == expected_kind
        if expected_kind == "cleanup unverified":
            assert "poll" in report.outcomes[0].detail.lower()
        assert report.failed is (expected_kind != "confirmed exit")
    finally:
        guard.close()


def test_open_scope_adopted_candidate_remains_pending_until_scope_ends(tmp_path):
    guard = orphan_guard.OrphanGuard(tmp_path)
    scope = guard.scope("pending-adopted")
    handle = _LiveHandle(7342)
    adopted = scope.register_adopted(
        7342, bound_handle=handle, attributed=True,
        parent_transition=(123, 456), parent_exit_confirmed=True,
        tmp_path_use=True)
    try:
        before_scope_end = guard.check()

        pending = [outcome for outcome in before_scope_end.outcomes if outcome.pid == adopted.pid]
        assert [outcome.kind for outcome in pending] == ["cleanup unverified"]
        assert "scope" in pending[0].detail.lower()
        assert "end" in pending[0].detail.lower()
        assert before_scope_end.failed
        assert before_scope_end.unit_outcomes == ()
        assert not adopted.checked
        assert handle.signals == []

        repeated_check = guard.check()
        assert [outcome.kind for outcome in repeated_check.outcomes if outcome.pid == adopted.pid] == [
            "cleanup unverified"]
        assert not adopted.checked
        assert handle.signals == []

        scope.end()
        after_scope_end = guard.check()

        assert [outcome.kind for outcome in after_scope_end.outcomes if outcome.pid == adopted.pid] == [
            "cleanup unverified", "leak", "confirmed exit"]
        assert adopted.checked
        assert handle.signals == [signal.SIGTERM]
    finally:
        guard.close()


def test_teardown_reports_missing_scope_end_for_open_adopted_candidate(tmp_path):
    guard = orphan_guard.OrphanGuard(tmp_path)
    scope = guard.scope("teardown-missing-scope-end")
    handle = _LiveHandle(7349)
    scope.register_adopted(
        7349, bound_handle=handle, attributed=True,
        parent_transition=(123, 456), parent_exit_confirmed=True,
        tmp_path_use=True)
    item = SimpleNamespace(stash={orphan_guard._GUARD_KEY: guard})

    failures = orphan_guard._finish_guard(item)

    assert len(failures) == 1
    assert "cleanup unverified" in failures[0]
    assert "owning scope end is not observed" in failures[0]
    assert handle.signals == []
    assert item.stash == {}


def test_exited_registered_adoption_owner_does_not_end_scope(tmp_path):
    handles: list[_LiveHandle] = []

    def handle_factory(process: subprocess.Popen[bytes], *, platform: str) -> _LiveHandle:
        handle = _LiveHandle(process.pid)
        handles.append(handle)
        return handle

    guard = orphan_guard.OrphanGuard(tmp_path, handle_factory=handle_factory)
    scope = guard.scope("exited-adoption-owner")
    owner = scope.launch(
        [sys.executable, "-c", "pass"], cwd=tmp_path,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    owner.process.wait(timeout=5.0)
    handle = _LiveHandle(7350)
    adopted = scope.register_adopted(
        7350, bound_handle=handle, attributed=True,
        parent_transition=(123, owner.pid), parent_exit_confirmed=True,
        tmp_path_use=True)
    try:
        before_scope_end = guard.check()

        adopted_outcomes = [
            outcome for outcome in before_scope_end.outcomes if outcome.pid == adopted.pid]
        assert [outcome.kind for outcome in adopted_outcomes] == ["cleanup unverified"]
        assert "scope" in adopted_outcomes[0].detail.lower()
        assert "end" in adopted_outcomes[0].detail.lower()
        assert before_scope_end.unit_outcomes == ()
        assert not adopted.checked
        assert handle.alive
        assert handle.signals == []

        scope.end()
        after_scope_end = guard.check()

        adopted_outcomes = [
            outcome for outcome in after_scope_end.outcomes if outcome.pid == adopted.pid]
        assert [outcome.kind for outcome in adopted_outcomes] == [
            "cleanup unverified", "leak", "confirmed exit"]
        assert adopted.checked
        assert handle.signals == [signal.SIGTERM]
    finally:
        guard.close()


def test_close_reaps_registered_child_and_closes_owned_pipes(tmp_path):
    handles = []

    def handle_factory(process: subprocess.Popen[bytes], *, platform: str):
        handle = _LiveHandle(process.pid)
        handles.append(handle)
        return handle

    guard = orphan_guard.OrphanGuard(tmp_path, handle_factory=handle_factory)
    scope = guard.scope("close-reaps-child")
    launch = scope.launch(
        [sys.executable, "-c", "pass"], cwd=tmp_path,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    streams = (launch.process.stdin, launch.process.stdout, launch.process.stderr)
    try:
        guard.close()

        assert launch.process.returncode is not None
        assert all(stream is not None and stream.closed for stream in streams)
        assert handles[0].closed
    finally:
        if launch.process.poll() is None:
            _stop_helper(launch.process)


def test_close_bounds_wait_for_live_registered_child_and_still_closes_pipes(
        tmp_path, monkeypatch):
    handles = []

    def handle_factory(process: subprocess.Popen[bytes], *, platform: str):
        handle = _LiveHandle(process.pid)
        handles.append(handle)
        return handle

    monkeypatch.setattr(orphan_guard, "_CLOSE_REAP_TIMEOUT_SECONDS", 0.05)
    guard = orphan_guard.OrphanGuard(tmp_path, handle_factory=handle_factory)
    scope = guard.scope("close-bounds-live-child")
    launch = scope.launch(
        [sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    streams = (launch.process.stdin, launch.process.stdout, launch.process.stderr)
    try:
        with pytest.raises(orphan_guard.OrphanGuardError, match="reap"):
            guard.close()

        assert launch.process.poll() is None
        assert all(stream is not None and stream.closed for stream in streams)
        assert handles[0].signals == []
    finally:
        _stop_helper(launch.process)


def test_close_discards_buffered_stdin_without_exceeding_teardown_deadline(
        tmp_path, monkeypatch):
    handles = []

    def handle_factory(process: subprocess.Popen[bytes], *, platform: str):
        handle = _LiveHandle(process.pid)
        handles.append(handle)
        return handle

    monkeypatch.setattr(orphan_guard, "_CLOSE_REAP_TIMEOUT_SECONDS", 0.05)
    guard = orphan_guard.OrphanGuard(tmp_path, handle_factory=handle_factory)
    scope = guard.scope("close-bounds-buffered-stdin")
    launch = scope.launch(
        [sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdin = launch.process.stdin
    assert stdin is not None
    stdin.write(b"buffered input" * 256)
    stdin_fd = stdin.fileno()
    os.set_blocking(stdin_fd, False)
    try:
        while True:
            try:
                os.write(stdin_fd, b"pipe fill" * 512)
            except BlockingIOError:
                break
    finally:
        os.set_blocking(stdin_fd, True)

    def kill_non_reader() -> None:
        try:
            if launch.process.poll() is None:
                launch.process.kill()
        except ProcessLookupError:
            pass

    safety_kill = threading.Timer(0.35, kill_non_reader)
    safety_kill.daemon = True
    safety_kill.start()
    started = time.monotonic()
    try:
        with pytest.raises(orphan_guard.OrphanGuardError, match="reap"):
            guard.close()

        elapsed = time.monotonic() - started
        assert elapsed < 0.2
        assert stdin.closed
        assert handles[0].closed
    finally:
        safety_kill.cancel()
        _stop_helper(launch.process)


def test_pidfd_handle_accepts_signal_enum_and_passes_integer_to_bound_syscall(monkeypatch):
    calls: list[tuple[int, int]] = []
    handle = object.__new__(orphan_guard._PidfdHandle)
    handle._fd = 23
    monkeypatch.setattr(
        orphan_guard.signal, "pidfd_send_signal",
        lambda pidfd, signum: calls.append((pidfd, signum)), raising=False)

    handle.send_signal(signal.SIGTERM)

    assert calls == [(23, int(signal.SIGTERM))]


@pytest.mark.parametrize("signum", [0, -1, True])
def test_pidfd_handle_rejects_invalid_signal_numbers(monkeypatch, signum):
    calls: list[tuple[int, int]] = []
    handle = object.__new__(orphan_guard._PidfdHandle)
    handle._fd = 23
    monkeypatch.setattr(
        orphan_guard.signal, "pidfd_send_signal",
        lambda pidfd, sent: calls.append((pidfd, sent)), raising=False)

    with pytest.raises(ValueError, match="signals must be positive"):
        handle.send_signal(signum)

    assert calls == []


def test_macos_bound_handle_probes_the_captured_process_generation(monkeypatch):
    pid = 7341
    captured = orphan_guard._MacProcessIdentity(
        pid=pid, uid=501, start_sec=100, start_usec=200,
        pid_version=9, zombie=False)
    observed = {pid: captured}
    monkeypatch.setattr(orphan_guard, "_read_macos_identity", lambda requested: observed[requested])
    handle = orphan_guard._MacAuditTokenHandle(pid)

    assert handle.is_current()
    assert handle.is_alive()

    observed[pid] = replace(captured, zombie=True)
    assert not handle.is_alive()
    observed[pid] = replace(captured, start_usec=201, zombie=False)
    assert not handle.is_current()
    assert not handle.is_alive()


def test_macos_identity_capture_uses_complete_combined_proc_pidinfo_record(monkeypatch):
    class _ProcPidInfo:
        argtypes: object = None
        restype: object = None

        def __call__(self, pid, flavor, arg, buffer, buffer_size):
            assert flavor == orphan_guard._PROC_PIDT_BSDINFOWITHUNIQID
            assert arg == 0
            assert buffer_size == orphan_guard.ctypes.sizeof(
                orphan_guard._BSDInfoWithUnique)
            info = orphan_guard.ctypes.cast(
                buffer, orphan_guard.ctypes.POINTER(orphan_guard._BSDInfoWithUnique)).contents
            info.bsd.pid = pid
            info.bsd.uid = 501
            info.bsd.start_sec = 100
            info.bsd.start_usec = 200
            info.bsd.status = 2
            info.unique.idversion = 9
            return buffer_size

    class _Libproc:
        def __init__(self):
            self.proc_pidinfo = _ProcPidInfo()

    monkeypatch.setattr(orphan_guard.ctypes, "CDLL", lambda *args, **kwargs: _Libproc())

    identity = orphan_guard._read_macos_identity(7346)

    assert identity == orphan_guard._MacProcessIdentity(
        pid=7346, uid=501, start_sec=100, start_usec=200,
        pid_version=9, zombie=False)


def test_macos_bound_handle_confirms_exit_through_registered_child_wait(monkeypatch):
    pid = 7347
    identity = orphan_guard._MacProcessIdentity(
        pid=pid, uid=501, start_sec=100, start_usec=200,
        pid_version=10, zombie=False)
    monkeypatch.setattr(orphan_guard, "_read_macos_identity", lambda requested: identity)

    class _WaitableProcess:
        def __init__(self):
            self.pid = pid
            self.waits: list[float] = []

        def wait(self, timeout: float) -> int:
            self.waits.append(timeout)
            return -int(signal.SIGTERM)

    process = _WaitableProcess()
    handle = orphan_guard._MacAuditTokenHandle(pid, process=process)

    assert handle.wait_exit(0.25)
    assert process.waits == [0.25]


def _macos_launch_with_signal_result(tmp_path, monkeypatch, signal_result):
    identities: dict[int, orphan_guard._MacProcessIdentity] = {}
    signal_calls: list[int] = []

    def read_identity(pid: int):
        return identities.get(pid)

    class _SignalApi:
        argtypes: object = None
        restype: object = None

        def __call__(self, _audit_token, signum: int) -> int:
            signal_calls.append(signum)
            return signal_result

    class _Libproc:
        def __init__(self):
            if signal_result is not None:
                self.proc_signal_with_audittoken = _SignalApi()

    libproc = _Libproc()
    monkeypatch.setattr(orphan_guard, "_read_macos_identity", read_identity)
    monkeypatch.setattr(orphan_guard.ctypes, "CDLL", lambda *args, **kwargs: libproc)

    def handle_factory(process: subprocess.Popen[bytes], *, platform: str):
        identity = orphan_guard._MacProcessIdentity(
            pid=process.pid, uid=501, start_sec=100, start_usec=200,
            pid_version=9, zombie=False)
        identities[process.pid] = identity
        return orphan_guard._MacAuditTokenHandle(process.pid, process=process)

    guard = orphan_guard.OrphanGuard(
        tmp_path, platform="darwin", handle_factory=handle_factory)
    scope = guard.scope("macos-signal-capability")
    launch = scope.launch(
        [sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    scope.end()
    return guard, launch, signal_calls


@pytest.mark.parametrize(
    ("signal_result", "expected_calls"),
    [(None, []), (errno.ENOSYS, [signal.SIGTERM])],
    ids=["symbol-missing", "enosys"],
)
def test_macos_missing_strict_signal_api_merges_live_leak_evidence(
        tmp_path, monkeypatch, signal_result, expected_calls):
    guard, launch, signal_calls = _macos_launch_with_signal_result(
        tmp_path, monkeypatch, signal_result)
    try:
        report = guard.check()

        outcomes = [outcome for outcome in report.outcomes if outcome.pid == launch.pid]
        assert [outcome.kind for outcome in outcomes] == ["cleanup unverified"]
        detail = outcomes[0].detail.lower()
        assert "registered process alive after scope end" in detail
        assert "strict" in detail
        assert "audit-token" in detail
        assert "api unavailable" in detail
        assert report.failed
        assert launch.process.poll() is None
        assert signal_calls == expected_calls
    finally:
        _stop_helper(launch.process)
        guard.close()


def test_macos_available_strict_signal_failure_preserves_leak_and_unverified(
        tmp_path, monkeypatch):
    guard, launch, signal_calls = _macos_launch_with_signal_result(
        tmp_path, monkeypatch, errno.EPERM)
    try:
        report = guard.check()

        outcomes = [outcome for outcome in report.outcomes if outcome.pid == launch.pid]
        assert [outcome.kind for outcome in outcomes] == ["leak", "cleanup unverified"]
        assert "remained alive" in outcomes[0].detail
        assert "eperm" in outcomes[1].detail.lower()
        assert report.failed
        assert launch.process.poll() is None
        assert signal_calls == [signal.SIGTERM]
    finally:
        _stop_helper(launch.process)
        guard.close()


def test_macos_zombie_status_needs_waitable_child_exit_confirmation(monkeypatch):
    pid = 7349
    identity = orphan_guard._MacProcessIdentity(
        pid=pid, uid=501, start_sec=100, start_usec=200,
        pid_version=12, zombie=True)
    monkeypatch.setattr(orphan_guard, "_read_macos_identity", lambda requested: identity)

    class _LiveChild:
        def __init__(self):
            self.pid = pid

        def poll(self) -> None:
            return None

    handle = orphan_guard._MacAuditTokenHandle(pid, process=_LiveChild())

    with pytest.raises(orphan_guard.OrphanGuardError, match="zombie.*wait-confirmed"):
        handle.is_alive()


def test_macos_esrch_for_live_registered_child_is_cleanup_unverified(tmp_path, monkeypatch):
    class _ProcPidInfo:
        argtypes: object = None
        restype: object = None
        calls = 0

        def __call__(self, pid, _flavor, _arg, buffer, buffer_size):
            self.calls += 1
            if self.calls > 1:
                orphan_guard.ctypes.set_errno(errno.ESRCH)
                return 0
            info = orphan_guard.ctypes.cast(
                buffer, orphan_guard.ctypes.POINTER(orphan_guard._BSDInfoWithUnique)).contents
            info.bsd.pid = pid
            info.bsd.uid = 501
            info.bsd.start_sec = 100
            info.bsd.start_usec = 200
            info.bsd.status = 2
            info.unique.idversion = 9
            return buffer_size

    class _Libproc:
        def __init__(self):
            self.proc_pidinfo = _ProcPidInfo()

    libproc = _Libproc()
    monkeypatch.setattr(orphan_guard.ctypes, "CDLL", lambda *args, **kwargs: libproc)
    guard = orphan_guard.OrphanGuard(tmp_path, platform="darwin")
    scope = guard.scope("macos-esrch-live-child")
    launch = scope.launch(
        [sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    scope.end()
    try:
        report = guard.check()

        assert [outcome.kind for outcome in report.outcomes] == ["cleanup unverified"]
        assert len(report.unit_outcomes) == 1
        assert report.unit_outcomes[0].kind == "cleanup unverified"
        assert report.unit_outcomes[0].pid == launch.pid
        assert launch.process.poll() is None
        assert libproc.proc_pidinfo.calls == 2
    finally:
        _stop_helper(launch.process)
        guard.close()


def test_macos_missing_identity_without_waitable_child_is_unverified(tmp_path, monkeypatch):
    pid = 7348
    captured = orphan_guard._MacProcessIdentity(
        pid=pid, uid=501, start_sec=100, start_usec=200,
        pid_version=11, zombie=False)
    observed = iter((captured, None))
    monkeypatch.setattr(
        orphan_guard, "_read_macos_identity", lambda requested: next(observed))
    guard = orphan_guard.OrphanGuard(tmp_path, platform="darwin")
    scope = guard.scope("macos-adopted-missing-identity")
    handle = orphan_guard._MacAuditTokenHandle(pid)
    scope.register_adopted(
        pid, bound_handle=handle, attributed=True,
        parent_transition=(123, 456), parent_exit_confirmed=True, tmp_path_use=True)
    scope.end()
    try:
        report = guard.check()

        assert [outcome.kind for outcome in report.outcomes] == ["cleanup unverified"]
        assert len(report.unit_outcomes) == 1
        assert report.unit_outcomes[0].kind == "cleanup unverified"
        assert report.unit_outcomes[0].pid == pid
    finally:
        guard.close()


def test_ended_registered_units_have_one_canonical_outcome_each(tmp_path):
    script = tmp_path / "registered_helper.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    handles: list[_LiveHandle] = []

    class _UninspectableHandle(_LiveHandle):
        def is_current(self) -> bool:
            raise OSError("identity probe failed")

    def handle_factory(process: subprocess.Popen[bytes], *, platform: str):
        handle = (_UninspectableHandle(process.pid) if not handles else
                  _LiveHandle(process.pid))
        handles.append(handle)
        return handle

    guard = orphan_guard.OrphanGuard(tmp_path, handle_factory=handle_factory)
    launches = []
    adopted_pid = 98765
    adopted_handle = _LiveHandle(adopted_pid)
    try:
        for index in range(2):
            scope = guard.scope(f"registered-unit-{index}")
            launches.append(scope.launch(
                [sys.executable, str(script)], cwd=tmp_path,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL))
            if index == 1:
                scope.register_adopted(
                    adopted_pid, bound_handle=adopted_handle, attributed=True,
                    parent_transition=(123, 456), parent_exit_confirmed=True,
                    tmp_path_use=True)
            scope.end()

        report = guard.check()
        assert len(report.unit_outcomes) == len(launches) + 1
        assert {outcome.pid for outcome in report.unit_outcomes} == (
            {launch.pid for launch in launches} | {adopted_pid})
        assert {outcome.kind for outcome in report.unit_outcomes} == {
            "cleanup unverified", "leak"}
        assert guard.check().unit_outcomes == report.unit_outcomes
    finally:
        for launch in launches:
            _stop_helper(launch.process)
        guard.close()


@pytest.mark.parametrize(
    ("stale_errno", "probe_errno", "expected_errno"),
    [(errno.ESRCH, 0, 0), (0, errno.EPERM, errno.EPERM)],
)
def test_macos_identity_probe_does_not_treat_stale_or_permission_errno_as_exit(
        monkeypatch, stale_errno, probe_errno, expected_errno):
    class _ProcPidInfo:
        argtypes: object = None

        def __call__(self, *args: object) -> int:
            if probe_errno:
                orphan_guard.ctypes.set_errno(probe_errno)
            return 0

    class _Libproc:
        def __init__(self) -> None:
            self.proc_pidinfo = _ProcPidInfo()

    monkeypatch.setattr(orphan_guard.ctypes, "CDLL", lambda *args, **kwargs: _Libproc())
    orphan_guard.ctypes.set_errno(stale_errno)

    with pytest.raises(OSError) as error:
        orphan_guard._read_macos_identity(7345)

    assert error.value.errno == expected_errno


def test_macos_bound_handle_accepts_signal_enum(monkeypatch):
    pid = 7342
    identity = orphan_guard._MacProcessIdentity(
        pid=pid, uid=501, start_sec=100, start_usec=200,
        pid_version=10, zombie=False)
    monkeypatch.setattr(orphan_guard, "_read_macos_identity", lambda requested: identity)

    class _ProcSignal:
        argtypes: object = None

        def __init__(self) -> None:
            self.signals: list[int] = []

        def __call__(self, audit_token: object, signum: int) -> int:
            self.signals.append(signum)
            return 0

    class _Libproc:
        def __init__(self) -> None:
            self.proc_signal_with_audittoken = _ProcSignal()

    libproc = _Libproc()
    monkeypatch.setattr(orphan_guard.ctypes, "CDLL", lambda *args, **kwargs: libproc)
    handle = orphan_guard._MacAuditTokenHandle(pid)

    handle.send_signal(signal.SIGTERM)

    assert libproc.proc_signal_with_audittoken.signals == [int(signal.SIGTERM)]


def test_teardown_failure_marks_passed_report_as_failed():
    report = SimpleNamespace(outcome="passed", longrepr=None, sections=[])

    orphan_guard._apply_teardown_failures(report, ["leak: pid 7343: scope ended"])

    assert report.outcome == "failed"
    assert report.longrepr == "orphan guard: leak: pid 7343: scope ended"


def test_makereport_hook_applies_live_registered_leak_to_teardown_report(tmp_path):
    guard, scope, launch, _handle = _launch_helper(tmp_path)
    scope.end()
    item = SimpleNamespace(stash={orphan_guard._GUARD_KEY: guard})
    call = SimpleNamespace(when="teardown")
    report = SimpleNamespace(outcome="passed", longrepr=None, sections=[])
    outcome = SimpleNamespace(get_result=lambda: report)
    hook = orphan_guard.pytest_runtest_makereport(item, call)

    try:
        assert next(hook) is None
        with pytest.raises(StopIteration):
            hook.send(outcome)

        assert report.outcome == "failed"
        assert "orphan guard: leak:" in report.longrepr
        assert item.stash == {}
    finally:
        _stop_helper(launch.process)
        guard.close()


def test_teardown_failure_preserves_existing_failure_and_adds_guard_section():
    report = SimpleNamespace(outcome="failed", longrepr="existing teardown error", sections=[])

    orphan_guard._apply_teardown_failures(report, ["cleanup unverified: pid 7344"])

    assert report.outcome == "failed"
    assert report.longrepr == "existing teardown error"
    assert report.sections == [("orphan guard", "orphan guard: cleanup unverified: pid 7344")]
