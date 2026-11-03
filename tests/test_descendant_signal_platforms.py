"""Descendants.signal: macOS uses the generation-checked kernel signal; Linux keeps pidfd delivery."""
from __future__ import annotations

import ctypes
import errno
import signal
import sys
from types import SimpleNamespace

import pytest

from stateframe.engine import kernel_exec_tracker as tracker
from stateframe.engine.kernel_exec_tracker import ContainmentUnverified, Descendants, ProcessIdentity
from kernel_exec_supervisor import ManagedProcess

ROW = ProcessIdentity(4242, 1, 4242, 4242, (100, 7), False)


def _family(monkeypatch, platform: str, current: ProcessIdentity | None = ROW) -> Descendants:
    monkeypatch.setattr(tracker.sys, "platform", platform)
    family = Descendants(1, SimpleNamespace(identity=lambda pid: current))
    return family


def _complete_with_signal_errors(signal_errors: dict[int, str], survivors: tuple[int, ...], detail=""):
    process = SimpleNamespace(poll=lambda: -signal.SIGKILL,
                              stdout=SimpleNamespace(close=lambda: None))
    job = ManagedProcess.__new__(ManagedProcess)
    job.process = process
    job.collector = SimpleNamespace(seal=lambda: None)
    job.family = SimpleNamespace(signal_errors=signal_errors, signal_paths={})
    job.killed, job.term_sent = set(), set()
    job.containment_failed, job.limit_detail, job.cgroup = False, "", None
    job.complete(False, survivors, detail)
    return job.cleanup


def _fake_libproc(*, starts: tuple[tuple[int, int], ...] = (ROW.start,),
                  signal_result: int | None = None):
    reads = iter(starts)

    def proc_pidinfo(pid, flavor, arg, buffer, size):
        assert (pid, flavor, arg, size) == (ROW.pid, 18, 0, ctypes.sizeof(tracker._BSDInfoWithUnique))
        info = ctypes.cast(buffer, ctypes.POINTER(tracker._BSDInfoWithUnique)).contents
        info.bsd.pid, info.bsd.uid, info.bsd.status = ROW.pid, tracker.os.getuid(), 2
        info.bsd.start_sec, info.bsd.start_usec = next(reads)
        info.unique.idversion = 7
        return size

    lib = SimpleNamespace(proc_pidinfo=proc_pidinfo)
    if signal_result is not None:
        def proc_signal_with_audittoken(_token, _sig):
            return signal_result

        lib.proc_signal_with_audittoken = proc_signal_with_audittoken
    return lib


def test_macos_detached_descendant_is_signalled_through_generation_check(monkeypatch):
    family = _family(monkeypatch, "darwin")
    sent = []
    monkeypatch.setattr(tracker, "signal_mac_verified_process",
                        lambda row, sig: sent.append((row, sig)) or tracker.MAC_KERNEL_SIGNAL_PATH)
    assert family.signal([ROW], signal.SIGKILL) == {4242}
    assert sent == [(ROW, signal.SIGKILL)]
    assert family.signal_errors == {}
    assert family.signal_paths == {ROW.pid: tracker.MAC_KERNEL_SIGNAL_PATH}


def test_macos_kernel_signal_path_is_labelled(monkeypatch):
    family = _family(monkeypatch, "darwin")
    monkeypatch.setattr(tracker, "_libproc", lambda: _fake_libproc(signal_result=0))
    monkeypatch.setattr(tracker.os, "kill", lambda *_args: pytest.fail("kernel signal must not use os.kill"))
    assert family.signal([ROW], signal.SIGKILL) == {ROW.pid}
    assert family.signal_paths == {ROW.pid: tracker.MAC_KERNEL_SIGNAL_PATH}


@pytest.mark.parametrize("unsupported", [None, errno.ENOTSUP, errno.ENOSYS])
def test_macos_missing_or_unsupported_signal_rechecks_then_uses_os_kill(monkeypatch, unsupported):
    family = _family(monkeypatch, "darwin")
    monkeypatch.setattr(tracker, "_libproc", lambda: _fake_libproc(
        starts=(ROW.start, ROW.start), signal_result=unsupported))
    sent = []
    monkeypatch.setattr(tracker.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert family.signal([ROW], signal.SIGTERM) == {ROW.pid}
    assert sent == [(ROW.pid, signal.SIGTERM)]
    assert family.signal_paths == {ROW.pid: tracker.MAC_FALLBACK_SIGNAL_PATH}
    assert family.signal_errors == {}


def test_macos_fallback_refuses_changed_birth_time(monkeypatch):
    family = _family(monkeypatch, "darwin")
    monkeypatch.setattr(tracker, "_libproc", lambda: _fake_libproc(starts=(ROW.start, (101, 0))))
    monkeypatch.setattr(tracker.os, "kill", lambda *_args: pytest.fail("changed identity must not be signalled"))
    assert family.signal([ROW], signal.SIGKILL) == set()
    assert family.signal_paths == {}


def test_macos_missing_proc_pidinfo_records_containment_error(monkeypatch):
    family = _family(monkeypatch, "darwin")
    monkeypatch.setattr(tracker, "_libproc", lambda: SimpleNamespace())
    monkeypatch.setattr(tracker.os, "kill", lambda *_args: pytest.fail("unverified PID must not be signalled"))
    with pytest.raises(ContainmentUnverified, match="macOS libproc unavailable: cannot read process identity"):
        tracker.signal_mac_verified_process(ROW, signal.SIGKILL)
    assert family.signal([ROW], signal.SIGKILL) == set()
    assert family.signal_errors == {ROW.pid: "macOS libproc unavailable: cannot read process identity"}


def test_macos_libproc_load_failure_records_containment_error(monkeypatch):
    family = _family(monkeypatch, "darwin")

    def unavailable(*_args, **_kwargs):
        raise OSError("libproc failed to load")

    monkeypatch.setattr(tracker.ctypes, "CDLL", unavailable)
    monkeypatch.setattr(tracker.os, "kill", lambda *_args: pytest.fail("unverified PID must not be signalled"))
    assert family.signal([ROW], signal.SIGKILL) == set()
    assert family.signal_errors == {ROW.pid: "macOS libproc unavailable: cannot read process identity"}


@pytest.mark.parametrize("error", [ProcessLookupError, PermissionError])
def test_macos_fallback_preserves_os_kill_errors(monkeypatch, error):
    monkeypatch.setattr(tracker.sys, "platform", "darwin")
    monkeypatch.setattr(tracker, "_libproc", lambda: _fake_libproc(starts=(ROW.start, ROW.start)))

    def fail(_pid, _sig):
        raise error("process unavailable")

    monkeypatch.setattr(tracker.os, "kill", fail)
    if error is ProcessLookupError:
        assert tracker.signal_mac_verified_process(ROW, signal.SIGKILL) is False
    else:
        with pytest.raises(PermissionError, match="process unavailable"):
            tracker.signal_mac_verified_process(ROW, signal.SIGKILL)


def test_macos_recycled_pid_is_skipped_not_reported_as_permission(monkeypatch):
    family = _family(monkeypatch, "darwin")
    monkeypatch.setattr(tracker, "signal_mac_verified_process", lambda row, sig: False)
    assert family.signal([ROW], signal.SIGTERM) == set()
    assert family.signal_errors == {}


def test_macos_unreadable_generation_is_a_signal_error_not_a_crash(monkeypatch):
    family = _family(monkeypatch, "darwin")

    def unverified(row, sig):
        raise ContainmentUnverified("cannot read macOS process birth and PID version together")
    monkeypatch.setattr(tracker, "signal_mac_verified_process", unverified)
    assert family.signal([ROW], signal.SIGTERM) == set()
    assert "PID version" in family.signal_errors[4242]


def test_changed_start_time_is_never_signalled_on_macos(monkeypatch):
    other = ProcessIdentity(4242, 1, 4242, 4242, (101, 0), False)
    family = _family(monkeypatch, "darwin", current=other)
    monkeypatch.setattr(tracker, "signal_mac_verified_process",
                        lambda row, sig: pytest.fail("recycled PID must not reach the kernel signal"))
    assert family.signal([ROW], signal.SIGKILL) == set()


def test_linux_keeps_pidfd_delivery(monkeypatch):
    family = _family(monkeypatch, "linux")
    sent, closed = [], []
    monkeypatch.setattr(tracker.os, "pidfd_open", lambda pid: 99, raising=False)
    monkeypatch.setattr(tracker.os, "close", closed.append)
    monkeypatch.setattr(tracker.signal, "pidfd_send_signal",
                        lambda fd, sig: sent.append((fd, sig)), raising=False)
    monkeypatch.setattr(tracker, "signal_mac_verified_process",
                        lambda row, sig: pytest.fail("Linux must not use the macOS path"))
    assert family.signal([ROW], signal.SIGKILL) == {4242}
    assert sent == [(99, signal.SIGKILL)] and closed == [99]
    assert family.signal_paths == {}


def test_complete_exposes_signal_paths_without_changing_cleanup_result(monkeypatch):
    family = _family(monkeypatch, "darwin")
    monkeypatch.setattr(tracker, "_libproc", lambda: _fake_libproc(starts=(ROW.start, ROW.start)))
    monkeypatch.setattr(tracker.os, "kill", lambda *_args: None)
    assert family.signal([ROW], signal.SIGKILL) == {ROW.pid}
    cleanup = _complete_with_signal_errors({}, ())
    assert "signal_paths" not in cleanup

    process = SimpleNamespace(poll=lambda: -signal.SIGKILL,
                              stdout=SimpleNamespace(close=lambda: None))
    job = ManagedProcess.__new__(ManagedProcess)
    job.process, job.collector, job.family = process, SimpleNamespace(seal=lambda: None), family
    job.killed, job.term_sent = set(), set()
    job.containment_failed, job.limit_detail, job.cgroup = False, "", None
    job.complete(True, ())
    assert job.cleanup["ok"] is True
    assert job.cleanup["signal_paths"] == {ROW.pid: tracker.MAC_FALLBACK_SIGNAL_PATH}


@pytest.mark.parametrize(("platform", "reason"), [
    ("freebsd", "detached descendant signal unavailable without pidfd"),
    ("darwin", "cannot read macOS process birth and PID version together"),
])
def test_complete_reports_real_descendant_signal_reason_without_false_permission_error(monkeypatch, platform, reason):
    family = _family(monkeypatch, platform)
    if platform == "darwin":
        def unverified(_row, _sig):
            raise ContainmentUnverified(reason)

        monkeypatch.setattr(tracker, "signal_mac_verified_process", unverified)

    assert family.signal([ROW], signal.SIGKILL) == set()
    assert family.signal_errors == {ROW.pid: reason}
    cleanup = _complete_with_signal_errors(family.signal_errors, (ROW.pid,), "descendants survived SIGKILL")
    assert reason in cleanup["detail"]
    assert "signal permission denied" not in cleanup["detail"]


def test_complete_reports_permission_error_recorded_by_descendants_signal(monkeypatch):
    family = _family(monkeypatch, "linux")

    def deny_pidfd(_pid):
        raise PermissionError("pidfd access denied")

    monkeypatch.setattr(tracker.os, "pidfd_open", deny_pidfd, raising=False)
    assert family.signal([ROW], signal.SIGKILL) == set()
    assert family.signal_errors == {ROW.pid: "signal permission denied"}

    cleanup = _complete_with_signal_errors(family.signal_errors, (ROW.pid,))
    assert "signal permission denied" in cleanup["detail"]


def test_complete_uses_only_sorted_distinct_signal_errors_for_survivors():
    errors = {
        4242: "zebra survivor reason",
        4243: "alpha survivor reason",
        4244: "zebra survivor reason",
        4245: "unrelated descendant reason",
    }
    cleanup = _complete_with_signal_errors(errors, (4242, 4243, 4244), "descendants survived SIGKILL")
    assert cleanup["detail"] == (
        "descendants survived SIGKILL; alpha survivor reason; zebra survivor reason"
    )


def test_complete_bounds_appended_signal_error_detail():
    cleanup = _complete_with_signal_errors({ROW.pid: "x" * 500}, (ROW.pid,))
    assert cleanup["detail"] == "; " + "x" * 300


@pytest.mark.skipif(sys.platform != "darwin", reason="real macOS kernel signal")
def test_real_macos_setsid_child_is_killed(tmp_path):
    """On 03469b8, Descendants.signal recorded "unavailable without pidfd" and never signalled this child."""
    import subprocess
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        table = tracker.ProcessTable()
        row = table.identity(child.pid)
        family = Descendants(child.pid, table)
        assert family.signal([row], signal.SIGKILL) == {child.pid}
        assert child.wait(timeout=5) == -signal.SIGKILL
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.skipif(sys.platform != "darwin", reason="real macOS process identity")
def test_real_macos_setsid_child_falls_back_when_kernel_symbol_is_missing(monkeypatch):
    import subprocess

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        table = tracker.ProcessTable()
        row = table.identity(child.pid)
        assert row is not None
        lib = tracker._libproc()
        monkeypatch.setattr(tracker, "_libproc", lambda: SimpleNamespace(proc_pidinfo=lib.proc_pidinfo))
        family = Descendants(child.pid, table)
        assert family.signal([row], signal.SIGKILL) == {child.pid}
        assert family.signal_paths == {child.pid: tracker.MAC_FALLBACK_SIGNAL_PATH}
        assert child.wait(timeout=5) == -signal.SIGKILL
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
