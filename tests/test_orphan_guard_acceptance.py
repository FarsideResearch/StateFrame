"""Acceptance contract for the test-harness orphan guard in tests/_orphan_guard.py.

The public API under test is ``OrphanGuard(tmp_path, *, token=None, platform=None)``;
``guard.scope(name)`` returns a scope with ``scope.launch(argv, *, cwd, **popen_options)``
and ``scope.end()``; ``guard.check()`` returns a report with ``outcomes`` (each has
``kind``, ``pid`` and ``detail``) and ``failed``; ``guard.close()`` closes owned handles
and cleans up test-owned subprocesses. A launch adds the guard token to its explicit
child environment and is registered with a Linux pidfd or strict macOS audit-token
handle at launch. The pytest adapter calls ``check()`` from a teardown hookwrapper after
fixture finalizers through ``pytest_runtest_teardown(item, nextitem)``; that hook gets
the guard from ``item._orphan_guard`` and fails the test when ``report.failed`` is true.
Outcome kinds are exactly ``leak``, ``cleanup unverified`` and ``confirmed exit``;
successful signaling does not itself produce ``confirmed exit``.

The module-level boundaries patched below are ``inspect_tmp_path(root, *, token)``,
``bind_process(process, *, platform)``, ``signal_bound(handle, signum)``,
``proc_signal_with_audittoken(pid, audit_token, signum)`` and ``subprocess.Popen``.
The guard exports ``TOKEN_ENV``; ``signal_bound`` dispatches through the registered
handle's ``send_signal`` method and never uses numeric ``os.kill`` for nonzero signals.
Each returned launch exposes ``pid``, ``process`` and ``bound_handle`` for teardown
assertions.
``ProcessObservation`` accepts ``pid``, ``ppid``, ``cwd``, ``token``,
``tmp_path_use=True`` and ``parent_transition_observed=False``. A bound handle exposes
``pid``, ``is_alive()``, ``send_signal(signum)``, ``wait_exit(timeout)`` and ``close()``.
For the planted controls, candidate classification belongs in
``OrphanGuard._handle_candidate(candidate)``: it must return without signaling when
``candidate.scope_ended`` is false, and use these exact branches::

    if not candidate.scope_ended:
        return
    if candidate.bound_handle is None:
        self._record_outcome("cleanup unverified", candidate.pid, "missing bound handle")
        return

Proven candidates are signaled only through ``_signal_bound(candidate.bound_handle,
signum)``. These two classification seams are also the targets of the unapplied
planted-control patches at the worktree root.
"""
from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import _orphan_guard as orphan_guard


_IS_LINUX = sys.platform == "linux"


def _kinds(report, pid: int) -> list[str]:
    return [outcome.kind for outcome in report.outcomes if outcome.pid == pid]


def _sleep_command() -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(60)"]


def _permission_denied(*_args, **_kwargs):
    raise PermissionError(errno.EPERM, "permission denied while inspecting process")


def _watch_numeric_signals(monkeypatch):
    calls = []

    def record_kill(pid, signum):
        if signum:
            calls.append(("kill", pid, signum))

    def record_killpg(pgid, signum):
        if signum:
            calls.append(("killpg", pgid, signum))

    monkeypatch.setattr(orphan_guard.os, "kill", record_kill)
    monkeypatch.setattr(orphan_guard.os, "killpg", record_killpg)
    return calls


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class _FakeBoundHandle:
    def __init__(self, pid: int, *, replace_before: int | None = None) -> None:
        self.pid = pid
        self.replace_before = replace_before
        self.signals: list[int] = []
        self.replacement_signals: list[int] = []
        self.replaced = False

    def is_alive(self) -> bool:
        return True

    def send_signal(self, signum: int) -> None:
        if signum == self.replace_before:
            self.replaced = True
            raise ProcessLookupError("bound identity changed before signal")
        if self.replaced:
            self.replacement_signals.append(signum)
        else:
            self.signals.append(signum)

    def wait_exit(self, timeout: float) -> bool:
        return False

    def close(self) -> None:
        return None


class _FakeMacBoundHandle(_FakeBoundHandle):
    audit_token = object()

    def send_signal(self, signum: int) -> None:
        api = orphan_guard.proc_signal_with_audittoken
        if api is None:
            raise OSError(errno.ENOSYS, "proc_signal_with_audittoken is unavailable")
        api(self.pid, self.audit_token, signum)


def _fake_launch(monkeypatch, guard, tmp_path, bound_handle, *, pid: int = 71031):
    process = _FakeProcess(pid)
    monkeypatch.setattr(orphan_guard.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        orphan_guard,
        "bind_process",
        lambda _process, *, platform: bound_handle,
    )
    monkeypatch.setattr(
        orphan_guard,
        "inspect_tmp_path",
        lambda _root, *, token: (
            orphan_guard.ProcessObservation(
                pid=pid,
                ppid=4321,
                cwd=tmp_path,
                token=token,
                tmp_path_use=True,
                parent_transition_observed=True,
            ),
        ),
    )
    scope = guard.scope("test-owned")
    launch = scope.launch(_sleep_command(), cwd=tmp_path)
    return scope, launch, process


@pytest.fixture
def live_registered_fixture_helper(tmp_path):
    guard = orphan_guard.OrphanGuard(tmp_path, token="fixture-live")
    scope = guard.scope("fixture")
    helper = scope.launch(_sleep_command(), cwd=tmp_path, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield guard, scope, helper
    finally:
        scope.end()
        guard.close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="real pidfd-backed subprocess control")
def test_live_registered_fixture_helper_is_not_a_leak(live_registered_fixture_helper, monkeypatch):
    """A registered helper inside its still-running fixture scope is not a leak."""
    guard, _scope, helper = live_registered_fixture_helper
    signaled = []
    numeric_signals = _watch_numeric_signals(monkeypatch)
    original_signal = orphan_guard.signal_bound

    def record_signal(handle, signum):
        signaled.append((handle, signum))
        return original_signal(handle, signum)

    monkeypatch.setattr(orphan_guard, "signal_bound", record_signal)
    report = guard.check()
    assert "leak" not in _kinds(report, helper.pid)
    assert not report.failed
    assert helper.process.poll() is None
    assert signaled == []
    assert numeric_signals == []


def test_pytest_hook_checks_after_fixture_finalizers(monkeypatch, tmp_path):
    """The teardown hook runs the guard after fixture finalizers complete."""
    guard = orphan_guard.OrphanGuard(tmp_path, token="finalizer-order")
    order = []

    class Item:
        _orphan_guard = guard
        finalizers_complete = False

    class HookOutcome:
        def get_result(self):
            return None

    def check_after_finalizers():
        order.append(Item.finalizers_complete)
        return guard.check()

    monkeypatch.setattr(guard, "check", check_after_finalizers)
    hook = orphan_guard.pytest_runtest_teardown(Item(), None)
    next(hook)
    Item.finalizers_complete = True
    with pytest.raises(StopIteration):
        hook.send(HookOutcome())

    assert order == [True]
    guard.close()


def test_pid_one_direct_child_is_not_attributed(monkeypatch, tmp_path):
    """A PID-1 direct child without a registered launch is not attributed to this test."""
    guard = orphan_guard.OrphanGuard(tmp_path, token="ours")
    observed = orphan_guard.ProcessObservation(
        pid=81001, ppid=1, cwd=tmp_path, token=None, tmp_path_use=True,
        parent_transition_observed=False,
    )
    signaled = []
    numeric_signals = _watch_numeric_signals(monkeypatch)
    monkeypatch.setattr(orphan_guard, "inspect_tmp_path",
                        lambda _root, *, token: (observed,))
    monkeypatch.setattr(orphan_guard, "signal_bound",
                        lambda handle, signum: signaled.append((handle, signum)))

    report = guard.check()

    assert "leak" not in _kinds(report, observed.pid)
    assert signaled == []
    assert numeric_signals == []


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="real pidfd-backed subprocess control")
def test_ended_registered_launch_is_leak_and_stops_through_bound_handle(
        tmp_path, monkeypatch):
    """A real leftover in an ended registered scope is stopped through its bound pidfd."""
    guard = orphan_guard.OrphanGuard(tmp_path, token="ended-scope")
    scope = guard.scope("fixture")
    helper = scope.launch(_sleep_command(), cwd=tmp_path, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bound_handles = []
    original_signal = orphan_guard.signal_bound
    numeric_signals = []

    def record_signal(handle, signum):
        bound_handles.append(handle)
        return original_signal(handle, signum)

    scope.end()
    try:
        with monkeypatch.context() as patch:
            numeric_signals = _watch_numeric_signals(patch)
            patch.setattr(orphan_guard, "signal_bound", record_signal)
            report = guard.check()
        assert _kinds(report, helper.pid) == ["leak", "confirmed exit"]
        assert report.failed
        assert bound_handles
        assert all(handle is helper.bound_handle for handle in bound_handles)
        assert numeric_signals == []
        assert helper.process.wait(timeout=5) is not None
    finally:
        guard.close()


def test_unobserved_double_fork_is_unverified_and_untouched(monkeypatch, tmp_path):
    """An unobserved double-fork with no registered handle is unverified and untouched."""
    guard = orphan_guard.OrphanGuard(tmp_path, token="ours")
    observed = orphan_guard.ProcessObservation(
        pid=81002, ppid=1, cwd=tmp_path, token=guard.token, tmp_path_use=True,
        parent_transition_observed=False,
    )
    signals = []
    numeric_signals = _watch_numeric_signals(monkeypatch)

    monkeypatch.setattr(orphan_guard, "inspect_tmp_path",
                        lambda _root, *, token: (observed,))
    monkeypatch.setattr(orphan_guard, "signal_bound",
                        lambda handle, signum: signals.append((handle, signum)))

    report = guard.check()

    outcomes = [outcome for outcome in report.outcomes if outcome.pid == observed.pid]
    assert [outcome.kind for outcome in outcomes] == ["cleanup unverified"]
    assert "bound handle" in outcomes[0].detail.lower()
    assert report.failed
    assert signals == []
    assert numeric_signals == []


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="Linux /proc environment-token control")
def test_stripped_token_does_not_attribute_unregistered_process(tmp_path, monkeypatch):
    """A helper that execs after stripping its token is not attributed as a leak."""
    token = "stripped-token-control"
    guard = orphan_guard.OrphanGuard(tmp_path, token=token)
    token_name = orphan_guard.TOKEN_ENV
    scrub_and_exec = (
        "import os, sys; "
        "clean = {key: value for key, value in os.environ.items() if key != sys.argv[1]}; "
        "os.execve(sys.executable, [sys.executable, '-c', 'import time; time.sleep(60)'], clean)"
    )
    env = os.environ.copy()
    env[token_name] = token
    child = subprocess.Popen([sys.executable, "-c", scrub_and_exec, token_name], cwd=tmp_path,
                             env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    signals = []
    try:
        deadline = time.monotonic() + 5
        environ_path = Path(f"/proc/{child.pid}/environ")
        while time.monotonic() < deadline:
            if environ_path.exists() and token_name.encode() not in environ_path.read_bytes():
                break
            time.sleep(0.01)
        assert token_name.encode() not in environ_path.read_bytes()

        with monkeypatch.context() as patch:
            numeric_signals = _watch_numeric_signals(patch)
            patch.setattr(orphan_guard, "signal_bound",
                          lambda handle, signum: signals.append((handle, signum)))
            report = guard.check()

        assert "leak" not in _kinds(report, child.pid)
        assert signals == []
        assert numeric_signals == []
        assert child.poll() is None
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
        guard.close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="real pidfd-backed subprocess control")
def test_permission_denied_inspection_is_unverified_not_gone(tmp_path, monkeypatch):
    """EPERM during inspection is unverified and leaves the registered process alive."""
    guard = orphan_guard.OrphanGuard(tmp_path, token="denied-inspection")
    scope = guard.scope("fixture")
    helper = scope.launch(_sleep_command(), cwd=tmp_path, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    signaled = []
    numeric_signals = _watch_numeric_signals(monkeypatch)
    monkeypatch.setattr(orphan_guard, "inspect_tmp_path", _permission_denied)
    monkeypatch.setattr(orphan_guard, "signal_bound",
                        lambda handle, signum: signaled.append((handle, signum)))
    scope.end()
    try:
        report = guard.check()
        outcomes = [outcome for outcome in report.outcomes if outcome.pid == helper.pid]
        assert [outcome.kind for outcome in outcomes] == ["cleanup unverified"]
        assert "permission" in outcomes[0].detail.lower() or "eperm" in outcomes[0].detail.lower()
        assert "confirmed exit" not in _kinds(report, helper.pid)
        assert helper.process.poll() is None
        assert signaled == []
        assert numeric_signals == []
    finally:
        monkeypatch.undo()
        guard.close()


@pytest.mark.parametrize("replace_before", [signal.SIGTERM, signal.SIGKILL])
def test_identity_replacement_never_signals_reused_pid(monkeypatch, tmp_path, replace_before):
    """A bound-handle identity change before TERM or KILL leaves the replacement untouched."""
    guard = orphan_guard.OrphanGuard(tmp_path, token="race", platform="linux")
    binding = _FakeBoundHandle(71032, replace_before=replace_before)
    scope, launch, _process = _fake_launch(monkeypatch, guard, tmp_path, binding, pid=binding.pid)

    def record_numeric_kill(_pid, signum):
        if signum:
            binding.replacement_signals.append(signum)
            return None
        return None

    def record_numeric_killpg(_pgid, signum):
        if signum:
            binding.replacement_signals.append(signum)
            return None
        return None

    monkeypatch.setattr(orphan_guard.os, "kill", record_numeric_kill)
    monkeypatch.setattr(orphan_guard.os, "killpg", record_numeric_killpg)
    scope.end()

    report = guard.check()

    assert launch.pid == binding.pid
    assert binding.replaced
    assert "cleanup unverified" in _kinds(report, binding.pid)
    assert "confirmed exit" not in _kinds(report, binding.pid)
    assert binding.replacement_signals == []
    if replace_before == signal.SIGKILL:
        assert binding.signals == [signal.SIGTERM]
    else:
        assert binding.signals == []
    guard.close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="real pidfd signaling control")
def test_nonzero_signals_never_use_numeric_kill_or_killpg(tmp_path, monkeypatch):
    """The guard sends nonzero signals only through the registered bound handle."""
    guard = orphan_guard.OrphanGuard(tmp_path, token="numeric-signal-control")
    scope = guard.scope("fixture")
    helper = scope.launch(_sleep_command(), cwd=tmp_path, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    real_kill = os.kill
    real_killpg = os.killpg
    numeric_signals = []

    def guarded_kill(pid, signum):
        if signum != 0:
            numeric_signals.append(("kill", pid, signum))
            pytest.fail("guard used numeric os.kill for a nonzero signal")
        return real_kill(pid, signum)

    def guarded_killpg(pgid, signum):
        if signum != 0:
            numeric_signals.append(("killpg", pgid, signum))
            pytest.fail("guard used numeric os.killpg for a nonzero signal")
        return real_killpg(pgid, signum)

    scope.end()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(orphan_guard.os, "kill", guarded_kill)
            patch.setattr(orphan_guard.os, "killpg", guarded_killpg)
            report = guard.check()
        assert "leak" in _kinds(report, helper.pid)
        assert "confirmed exit" in _kinds(report, helper.pid)
        assert numeric_signals == []
    finally:
        guard.close()


def test_macos_signal_zero_and_unavailable_strict_path_fail_closed(monkeypatch, tmp_path):
    """macOS probes avoid audit-token signal zero and missing strict signaling has no fallback."""
    guard = orphan_guard.OrphanGuard(tmp_path, token="mac-control", platform="darwin")
    binding = _FakeMacBoundHandle(71033)
    scope, launch, _process = _fake_launch(monkeypatch, guard, tmp_path, binding, pid=binding.pid)
    api_calls = []
    numeric_signals = []

    def audit_signal(_pid, _audit_token, signum):
        api_calls.append(signum)

    monkeypatch.setattr(orphan_guard, "proc_signal_with_audittoken", audit_signal)
    report_while_owned = guard.check()
    assert "leak" not in _kinds(report_while_owned, launch.pid)
    assert 0 not in api_calls

    def unavailable_numeric(_pid, signum):
        if signum:
            numeric_signals.append(signum)

    monkeypatch.setattr(orphan_guard, "proc_signal_with_audittoken", None)
    monkeypatch.setattr(orphan_guard.os, "kill", unavailable_numeric)
    monkeypatch.setattr(orphan_guard.os, "killpg", unavailable_numeric)
    scope.end()
    report = guard.check()

    outcomes = [outcome for outcome in report.outcomes if outcome.pid == launch.pid]
    assert [outcome.kind for outcome in outcomes] == ["cleanup unverified"]
    assert "audittoken" in outcomes[0].detail.lower() or "strict" in outcomes[0].detail.lower()
    assert 0 not in api_calls
    assert numeric_signals == []
    assert "confirmed exit" not in _kinds(report, launch.pid)
    guard.close()
