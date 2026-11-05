"""Focused unit coverage for orphan-guard lifetime and handle contracts."""
from __future__ import annotations

import errno
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import _orphan_guard as orphan_guard


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
