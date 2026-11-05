"""Behavior contract for the pytest orphan guard.

"Leftover" means surviving an explicitly ended test-owned lifetime. A process
launched and registered by a scope is a leak when its scope ends while it is
alive; its retained bound handle proves ownership, so no adoption proof is
required. For a process not launched by the harness, a leak requires all four
facts: attribution to a registered scope, that scope has ended, an identity-
bound record of a parent transition and confirmed death of the former parent,
and observed cwd/open-file use beneath the test's resolved tmp_path. If a
candidate is known but any required fact is missing, report "cleanup unverified"
and leave it untouched.

The public surface exercised here is OrphanGuard(root, *, token=None,
platform=None, handle_factory=None), guard.scope(name), scope.launch(...),
scope.end(), scope.register_adopted(...), guard.check(), guard.close(), and the
pytest fixture `orphan_guard`. `scope.launch()` returns a registration exposing
`pid`, `process`, and its retained `bound_handle`. Reports expose `outcomes`
and `failed`; outcomes expose `kind`, `pid`, and `detail`. Kinds are `leak`,
`cleanup unverified`, and `confirmed exit`; a successful signal alone never
confirms exit.

An injected bound handle exposes `pid`, stable `identity`, `is_current()`,
`is_alive()`, `send_signal(signum)`, `wait_exit(timeout)`, and `close()`.
`handle_factory(process, *, platform)` and `scope.register_adopted(pid, *,
bound_handle, attributed, parent_transition, parent_exit_confirmed,
tmp_path_use)` are the test seams. `parent_transition` is `(old_parent_pid,
new_parent_pid)` when observed. Tests assert behavior through this surface and
do not require private method names, hook storage, branch text, or report order.
"""
from __future__ import annotations

import errno
import json
import os
import select
import secrets
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

import _orphan_guard as orphan_guard
from orphan_guard_mutants import MUTANTS

pytest_plugins = ("pytester",)


_IS_LINUX = sys.platform == "linux"
_SETUP_TIMEOUT = 8.0
_CHECK_TIMEOUT = 5.0
_TOKEN_ENV = "SF_TEST_TOKEN"
_READY_SCRIPT = "import os,signal; os.write(1,b'ready\\n'); signal.pause()"
_STRIP_TOKEN_SCRIPT = """import os,sys
name=sys.argv[1]
clean={key:value for key,value in os.environ.items() if key != name}
target="import os,signal; os.write(1,b'ready\\n'); signal.pause()"
os.execve(sys.executable,[sys.executable,"-c",target],clean)
"""


_SUBREAPER_SCRIPT = r'''import ctypes, json, os, select, signal, sys
event_fd = int(sys.argv[1])
control_fd = int(sys.argv[2])
use_path = sys.argv[3]
libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                       ctypes.c_ulong, ctypes.c_ulong)
libc.prctl.restype = ctypes.c_int

def emit(value):
    os.write(event_fd, (json.dumps(value, separators=(",", ":")) + "\n").encode())

if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
    emit({"stage": "setup_error", "detail": "PR_SET_CHILD_SUBREAPER failed",
          "errno": ctypes.get_errno()})
    raise SystemExit(2)

ready_r, ready_w = os.pipe()
go_r, go_w = os.pipe()
adopted_r, adopted_w = os.pipe()
pid_r, pid_w = os.pipe()
intermediate = os.fork()
if intermediate == 0:
    os.close(pid_r)
    os.close(go_w)
    os.close(adopted_r)
    child = os.fork()
    if child > 0:
        os.write(pid_w, (str(child) + "\n").encode())
        os.close(pid_w)
        os.close(adopted_w)
        os.read(ready_r, 1)
        os.read(go_r, 1)
        raise SystemExit(0)

    os.close(pid_w)
    os.close(ready_r)
    os.close(go_r)
    old_parent = os.getppid()
    path_fd = os.open(use_path, os.O_CREAT | os.O_RDWR, 0o600)

    def report_adoption(_signum, _frame):
        emit({"stage": "adopted", "pid": os.getpid(),
              "old_parent_pid": old_parent, "new_parent_pid": os.getppid(),
              "open_path": use_path})
        os.write(adopted_w, b"1")

    signal.signal(signal.SIGUSR1, report_adoption)
    emit({"stage": "ready", "pid": os.getpid(),
          "old_parent_pid": old_parent, "open_path": use_path})
    os.write(ready_w, b"1")
    os.close(ready_w)
    while True:
        signal.pause()

os.close(pid_w)
os.close(ready_r)
os.close(ready_w)
os.close(go_r)
os.close(adopted_w)
pid_bytes = os.read(pid_r, 64)
os.close(pid_r)
child_pid = int(pid_bytes.strip())
readable, _, _ = select.select([control_fd], [], [], 10.0)
command = os.read(control_fd, 1) if readable else b""
if command != b"g":
    for pid in (intermediate, child_pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    emit({"stage": "setup_error", "detail": "test did not release adoption setup"})
    raise SystemExit(2)

try:
    parent_pidfd = os.pidfd_open(intermediate, 0)
except OSError as exc:
    for pid in (intermediate, child_pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    emit({"stage": "setup_error", "detail": "could not retain intermediate parent identity",
          "errno": exc.errno})
    raise SystemExit(2)
os.write(go_w, b"1")
os.close(go_w)
parent_exit = select.poll()
parent_exit.register(parent_pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
if not parent_exit.poll(5_000):
    os.close(parent_pidfd)
    for pid in (intermediate, child_pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    emit({"stage": "setup_error", "detail": "intermediate parent exit timed out"})
    raise SystemExit(2)
_, status = os.waitpid(intermediate, 0)
os.close(parent_pidfd)
emit({"stage": "parent_exit", "parent_pid": intermediate,
      "confirmed": True, "wait_status": status})
os.kill(child_pid, signal.SIGUSR1)
readable, _, _ = select.select([adopted_r], [], [], 5.0)
if not readable or not os.read(adopted_r, 1):
    emit({"stage": "setup_error", "detail": "adopted child did not acknowledge transition"})
    raise SystemExit(2)
os.close(adopted_r)
readable, _, _ = select.select([control_fd], [], [], 10.0)
if not readable or os.read(control_fd, 1) != b"x":
    emit({"stage": "setup_error", "detail": "test did not release subreaper helper"})
    raise SystemExit(2)
os.close(control_fd)
os.close(event_fd)
'''


_PID1_CHILD_SCRIPT = r'''import ctypes, json, os, select, signal, sys
use_path = sys.argv[1]
control_fd = int(sys.argv[2])
libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                       ctypes.c_ulong, ctypes.c_ulong)
libc.prctl.restype = ctypes.c_int

def emit(value):
    os.write(1, (json.dumps(value, separators=(",", ":")) + "\n").encode())

child_ready_r, child_ready_w = os.pipe()
parent_go_r, parent_go_w = os.pipe()
pid_r, pid_w = os.pipe()
intermediate = os.fork()
if intermediate == 0:
    os.close(parent_go_w)
    os.close(pid_r)
    child = os.fork()
    if child > 0:
        os.write(pid_w, (str(child) + "\n").encode())
        os.close(pid_w)
        os.close(child_ready_w)
        os.read(child_ready_r, 1)
        os.read(parent_go_r, 1)
        os._exit(0)

    os.close(pid_w)
    os.close(child_ready_r)
    os.close(parent_go_r)
    old_parent = os.getppid()
    path_fd = os.open(use_path, os.O_CREAT | os.O_RDWR, 0o600)
    def report_parent(_signum, _frame):
        emit({"stage": "adopted", "pid": os.getpid(), "ppid": os.getppid(),
              "old_parent_pid": old_parent, "open_path": use_path})
    signal.signal(signal.SIGUSR1, report_parent)
    if libc.prctl(1, signal.SIGUSR1, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        emit({"stage": "setup_error", "detail": "PR_SET_PDEATHSIG failed",
              "errno": ctypes.get_errno()})
        os._exit(3)
    emit({"stage": "created", "pid": os.getpid(), "old_parent_pid": old_parent})
    os.write(child_ready_w, b"1")
    os.close(child_ready_w)
    while True:
        signal.pause()

os.close(child_ready_r)
os.close(child_ready_w)
os.close(parent_go_r)
os.close(pid_w)
pid_bytes = os.read(pid_r, 64)
os.close(pid_r)
child_pid = int(pid_bytes.strip())
emit({"stage": "created", "pid": child_pid, "old_parent_pid": intermediate})
readable, _, _ = select.select([control_fd], [], [], 10.0)
command = os.read(control_fd, 1) if readable else b""
if command != b"g":
    for pid in (intermediate, child_pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    emit({"stage": "setup_error", "detail": "test did not release PID-1 setup"})
    raise SystemExit(2)
try:
    parent_pidfd = os.pidfd_open(intermediate, 0)
except OSError as exc:
    for pid in (intermediate, child_pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    emit({"stage": "setup_error", "detail": "could not retain intermediate parent identity",
          "errno": exc.errno})
    raise SystemExit(2)
os.write(parent_go_w, b"1")
os.close(parent_go_w)
parent_exit = select.poll()
parent_exit.register(parent_pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
if not parent_exit.poll(5_000):
    os.close(parent_pidfd)
    for pid in (intermediate, child_pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    emit({"stage": "setup_error", "detail": "intermediate parent exit timed out"})
    raise SystemExit(2)
_, status = os.waitpid(intermediate, 0)
os.close(parent_pidfd)
emit({"stage": "parent_exit", "parent_pid": intermediate,
      "confirmed": True, "wait_status": status})
os.close(control_fd)
'''


def _token() -> str:
    return secrets.token_hex(16)


def _readline_fd(fd: int, deadline: float, label: str) -> bytes:
    data = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail(f"setup: timed out waiting for {label}", pytrace=False)
        readable, _, _ = select.select([fd], [], [], remaining)
        if not readable:
            pytest.fail(f"setup: timed out waiting for {label}", pytrace=False)
        chunk = os.read(fd, 1)
        if not chunk:
            pytest.fail(f"setup: stream closed before {label}", pytrace=False)
        data.extend(chunk)
        if chunk == b"\n":
            return bytes(data[:-1])
        if len(data) > 4096:
            pytest.fail(f"setup: oversized readiness record for {label}", pytrace=False)


def _read_json_fd(fd: int, deadline: float, label: str) -> dict[str, object]:
    try:
        value = json.loads(_readline_fd(fd, deadline, label))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        pytest.fail(f"setup: invalid {label} record: {exc}", pytrace=False)
    if not isinstance(value, dict):
        pytest.fail(f"setup: {label} record was not an object", pytrace=False)
    return value


def _wait_ready(process: subprocess.Popen[bytes], label: str = "helper readiness") -> None:
    if process.stdout is None:
        pytest.fail(f"setup: {label} has no readiness pipe", pytrace=False)
    line = _readline_fd(process.stdout.fileno(), time.monotonic() + _SETUP_TIMEOUT, label)
    if line != b"ready":
        pytest.fail(f"setup: unexpected {label} payload {line!r}", pytrace=False)


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        try:
            process.wait(timeout=_SETUP_TIMEOUT)
        except subprocess.TimeoutExpired:
            pytest.fail("setup: exited helper could not be reaped", pytrace=False)
        return
    process.terminate()
    try:
        process.wait(timeout=_SETUP_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=_SETUP_TIMEOUT)
        except subprocess.TimeoutExpired:
            pytest.fail("setup: helper survived TERM and KILL", pytrace=False)


def _outcome_kinds(report, pid: int) -> Counter[str]:
    return Counter(outcome.kind for outcome in report.outcomes if outcome.pid == pid)


def _assert_empty_report(report) -> None:
    assert list(report.outcomes) == []
    assert not report.failed


def _assert_unverified(report, pid: int, terms: tuple[str, ...]) -> None:
    kinds = _outcome_kinds(report, pid)
    assert kinds == Counter({"cleanup unverified": 1})
    outcome = next(outcome for outcome in report.outcomes if outcome.pid == pid)
    detail = outcome.detail.casefold()
    assert any(term in detail for term in terms), detail
    assert report.failed


def _new_guard(root: Path, *, token: str, platform: str | None = None,
               handle_factory=None, variant=None, token_pids=()):
    base = orphan_guard.OrphanGuard(
        root, token=token, platform=platform, handle_factory=handle_factory)
    if variant is None:
        return base
    return variant(base, token_pids=token_pids)


def _start_registered_helper(scope, tmp_path: Path, *, script: str = _READY_SCRIPT,
                             script_args: tuple[str, ...] = ()):
    try:
        launch = scope.launch(
            [sys.executable, "-c", script, *script_args], cwd=tmp_path,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError as exc:
        pytest.fail(f"setup: helper launch failed: {exc}", pytrace=False)
    _wait_ready(launch.process)
    return launch


class _PidfdHandle:
    """Real Linux bound handle used to retain an adopted child's identity."""

    def __init__(self, pid: int) -> None:
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            pytest.fail("setup: Linux pidfd support is required for this control", pytrace=False)
        self.pid = pid
        try:
            self._fd = os.pidfd_open(pid, 0)
        except OSError as exc:
            pytest.fail(f"setup: could not retain pidfd for {pid}: {exc}", pytrace=False)
        self.identity = (pid, self._fd)
        self._poller = select.poll()
        self._poller.register(self._fd, select.POLLIN | select.POLLHUP | select.POLLERR)

    def is_current(self) -> bool:
        return self.is_alive()

    def is_alive(self) -> bool:
        return not self._poller.poll(0)

    def send_signal(self, signum: int) -> None:
        signal.pidfd_send_signal(self._fd, signum)

    def wait_exit(self, timeout: float) -> bool:
        return bool(self._poller.poll(max(0, int(timeout * 1000))))

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


class _SubreaperRun:
    def __init__(self, scope, tmp_path: Path) -> None:
        self.scope = scope
        self.tmp_path = tmp_path
        self.event_read, event_write = os.pipe()
        control_read, self.control_write = os.pipe()
        self.use_path = tmp_path / "subreaper-open-file"
        self.launch = None
        self.child_handle = None
        self.observer_handle = None
        self.closed = False
        try:
            try:
                self.launch = scope.launch(
                    [sys.executable, "-c", _SUBREAPER_SCRIPT,
                     str(event_write), str(control_read), str(self.use_path)],
                    cwd=tmp_path, pass_fds=(event_write, control_read),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
            except OSError as exc:
                pytest.fail(f"setup: subreaper launch failed: {exc}", pytrace=False)
            os.close(event_write)
            event_write = -1
            os.close(control_read)
            control_read = -1
            ready = _read_json_fd(
                self.event_read, time.monotonic() + _SETUP_TIMEOUT, "subreaper readiness")
            if ready.get("stage") != "ready":
                pytest.fail(f"setup: subreaper did not become ready: {ready}", pytrace=False)
            self.pid = int(ready["pid"])
            self.ready = ready
            self.child_handle = _PidfdHandle(self.pid)
            self.observer_handle = _PidfdHandle(self.pid)
            self._send_control(b"g")
            events: dict[str, dict[str, object]] = {}
            deadline = time.monotonic() + _SETUP_TIMEOUT
            while not {"parent_exit", "adopted"}.issubset(events):
                event = _read_json_fd(self.event_read, deadline, "subreaper adoption record")
                stage = str(event.get("stage"))
                if stage == "setup_error":
                    pytest.fail(f"setup: subreaper failed: {event}", pytrace=False)
                events[stage] = event
            self.parent_exit = events["parent_exit"]
            self.adopted = events["adopted"]
            if self.adopted.get("pid") != self.pid:
                pytest.fail("setup: subreaper reported a different adopted pid", pytrace=False)
            if self.ready.get("old_parent_pid") != self.adopted.get("old_parent_pid"):
                pytest.fail("setup: parent identity changed across adoption record", pytrace=False)
            if self.adopted.get("new_parent_pid") != self.launch.process.pid:
                pytest.fail("setup: grandchild was not adopted by the registered subreaper", pytrace=False)
            if self.parent_exit.get("parent_pid") != self.ready.get("old_parent_pid"):
                pytest.fail("setup: confirmed parent death did not match the old parent", pytrace=False)
            if self.parent_exit.get("confirmed") is not True:
                pytest.fail("setup: subreaper did not confirm old-parent death", pytrace=False)
            if Path(str(self.adopted.get("open_path"))).resolve().parent != tmp_path.resolve():
                pytest.fail("setup: adopted child did not use a file beneath tmp_path", pytrace=False)
        except BaseException:
            self.close()
            raise
        finally:
            if event_write >= 0:
                os.close(event_write)
            if control_read >= 0:
                os.close(control_read)

    def _send_control(self, value: bytes) -> None:
        try:
            os.write(self.control_write, value)
        except BrokenPipeError:
            pass

    def release_owner(self) -> None:
        if self.launch is None:
            return
        if self.launch.process.poll() is None:
            self._send_control(b"x")
            try:
                self.launch.process.wait(timeout=_SETUP_TIMEOUT)
            except subprocess.TimeoutExpired:
                pytest.fail("setup: subreaper helper did not exit after release", pytrace=False)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.release_owner()
        if self.observer_handle is not None:
            if self.observer_handle.is_alive():
                try:
                    self.observer_handle.send_signal(signal.SIGKILL)
                    if not self.observer_handle.wait_exit(_SETUP_TIMEOUT):
                        pytest.fail("setup: adopted helper survived test cleanup", pytrace=False)
                except ProcessLookupError:
                    pass
        for handle in (self.child_handle, self.observer_handle):
            if handle is not None:
                handle.close()
        if self.launch is not None and self.launch.process.poll() is None:
            _stop_process(self.launch.process)
        for fd in (self.control_write, self.event_read):
            try:
                os.close(fd)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise


class _InjectedHandle:
    def __init__(self, pid: int, *, signal_api=None, confirm_after_signal: bool = False) -> None:
        self.pid = pid
        self.identity = object()
        self.current_identity = self.identity
        self.signal_api = signal_api
        self.confirm_after_signal = confirm_after_signal
        self.liveness_probes = 0
        self.signals: list[int] = []
        self.wait_timeouts: list[float] = []
        self.exited = False
        self.closed = False

    def is_current(self) -> bool:
        return self.current_identity is self.identity

    def is_alive(self) -> bool:
        self.liveness_probes += 1
        return not self.exited

    def send_signal(self, signum: int) -> None:
        if self.current_identity is not self.identity:
            raise ProcessLookupError("bound identity no longer names the original process")
        if self.signal_api is not None:
            self.signal_api(self.pid, self.identity, signum)
        self.signals.append(signum)
        if self.confirm_after_signal:
            self.exited = True

    def wait_exit(self, timeout: float) -> bool:
        self.wait_timeouts.append(timeout)
        return self.exited

    def close(self) -> None:
        self.closed = True


class _EvidenceRaceHandle(_InjectedHandle):
    def __init__(self, pid: int, *, replace_at: str) -> None:
        super().__init__(pid)
        self.replace_at = replace_at
        self.identity_checks = 0
        self.armed = False

    def is_current(self) -> bool:
        if not self.armed:
            return self.current_identity is self.identity
        self.identity_checks += 1
        if self.replace_at == "term" and self.identity_checks == 1:
            evidence_identity = self.current_identity
            self.current_identity = object()
            return evidence_identity is self.identity
        return self.current_identity is self.identity

    def send_signal(self, signum: int) -> None:
        super().send_signal(signum)
        if self.replace_at == "kill" and signum == signal.SIGTERM:
            self.current_identity = object()

    def wait_exit(self, timeout: float) -> bool:
        self.wait_timeouts.append(timeout)
        return self.exited


class _PermissionDeniedHandle(_InjectedHandle):
    def is_current(self) -> bool:
        raise PermissionError(errno.EPERM, "permission denied while checking bound identity")

    def is_alive(self) -> bool:
        raise PermissionError(errno.EPERM, "permission denied while probing liveness")


def _forbid_numeric_signals(monkeypatch):
    calls = []
    original_kill = os.kill
    original_killpg = os.killpg

    def guarded_kill(pid, signum):
        if signum != 0:
            calls.append(("kill", pid, signum))
            return None
        return original_kill(pid, signum)

    def guarded_killpg(pgid, signum):
        if signum != 0:
            calls.append(("killpg", pgid, signum))
            return None
        return original_killpg(pgid, signum)

    monkeypatch.setattr(os, "kill", guarded_kill)
    monkeypatch.setattr(os, "killpg", guarded_killpg)
    return calls


class _CheckDeadline(Exception):
    pass


def _check_before_deadline(guard):
    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_deadline(_signum, _frame):
        raise _CheckDeadline

    signal.signal(signal.SIGALRM, raise_deadline)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, _CHECK_TIMEOUT)
    started = time.monotonic()
    try:
        result = guard.check()
    except _CheckDeadline:
        pytest.fail("guard.check() exceeded its deadline", pytrace=False)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
    assert time.monotonic() - started < _CHECK_TIMEOUT
    return result


def _registered_leak_scenario(tmp_path: Path, monkeypatch, variant=None, *,
                               script: str = _READY_SCRIPT, script_args: tuple[str, ...] = ()) -> None:
    token = _token()
    guard = _new_guard(tmp_path, token=token, variant=variant)
    scope = guard.scope("registered-leftover")
    launch = _start_registered_helper(scope, tmp_path, script=script, script_args=script_args)
    scope.end()
    try:
        numeric = _forbid_numeric_signals(monkeypatch)
        report = guard.check()
        kinds = _outcome_kinds(report, launch.pid)
        assert kinds["leak"] == 1
        assert kinds["confirmed exit"] == 1
        assert report.failed
        assert numeric == []
        assert launch.process.wait(timeout=_SETUP_TIMEOUT) is not None
    finally:
        monkeypatch.undo()
        _stop_process(launch.process)
        guard.close()


def _unverified_adoption_scenario(tmp_path: Path, monkeypatch, variant=None) -> None:
    token = _token()
    guard = _new_guard(tmp_path, token=token, variant=variant)
    scope = guard.scope("partial-adoption-proof")
    adopted = _SubreaperRun(scope, tmp_path)
    scope.register_adopted(
        adopted.pid, bound_handle=adopted.child_handle, attributed=True,
        parent_transition=None, parent_exit_confirmed=True, tmp_path_use=True)
    adopted.release_owner()
    scope.end()
    try:
        numeric = _forbid_numeric_signals(monkeypatch)
        report = guard.check()
        _assert_unverified(report, adopted.pid, ("parent", "transition"))
        assert adopted.observer_handle.is_alive()
        assert numeric == []
    finally:
        monkeypatch.undo()
        adopted.close()
        guard.close()


def _pid1_token_scenario(tmp_path: Path, monkeypatch, variant=None) -> None:
    token = _token()
    use_path = tmp_path / "pid1-open-file"
    control_read, control_write = os.pipe()
    env = os.environ.copy()
    env[_TOKEN_ENV] = token
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", _PID1_CHILD_SCRIPT, str(use_path), str(control_read)],
            cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            pass_fds=(control_read,))
    except OSError as exc:
        os.close(control_read)
        os.close(control_write)
        pytest.fail(f"setup: PID-1 helper launch failed: {exc}", pytrace=False)
    os.close(control_read)
    pidfd = None
    guard = None
    try:
        if child.stdout is None:
            pytest.fail("setup: PID-1 helper has no observation pipe", pytrace=False)
        created = _read_json_fd(
            child.stdout.fileno(), time.monotonic() + _SETUP_TIMEOUT, "PID-1 child creation")
        if created.get("stage") != "created":
            pytest.fail(f"setup: PID-1 child was not created: {created}", pytrace=False)
        pid = int(created["pid"])
        pidfd = _PidfdHandle(pid)
        os.write(control_write, b"g")
        try:
            child.wait(timeout=_SETUP_TIMEOUT)
        except subprocess.TimeoutExpired:
            pytest.fail("setup: PID-1 parent did not confirm child-parent death", pytrace=False)
        events: dict[str, dict[str, object]] = {"created": created}
        deadline = time.monotonic() + _SETUP_TIMEOUT
        while not {"parent_exit", "adopted"}.issubset(events):
            event = _read_json_fd(child.stdout.fileno(), deadline, "PID-1 adoption")
            stage = str(event.get("stage"))
            if stage == "setup_error":
                pytest.fail(f"setup: PID-1 helper failed: {event}", pytrace=False)
            events[stage] = event
        value = events["adopted"]
        if int(value["ppid"]) != 1:
            pytest.fail(
                f"setup: expected PID-1 adoption, observed parent {value['ppid']}",
                pytrace=False)
        if Path(str(value["open_path"])).resolve().parent != tmp_path.resolve():
            pytest.fail("setup: PID-1 child did not use a path beneath tmp_path", pytrace=False)
        guard = _new_guard(
            tmp_path, token=token, variant=variant, token_pids=(pid,))
        numeric = _forbid_numeric_signals(monkeypatch)
        report = guard.check()
        _assert_empty_report(report)
        assert numeric == []
    finally:
        if child.poll() is None:
            try:
                os.write(control_write, b"x")
            except BrokenPipeError:
                pass
            try:
                child.wait(timeout=_SETUP_TIMEOUT)
            except subprocess.TimeoutExpired:
                _stop_process(child)
        if guard is not None:
            guard.close()
        if pidfd is not None:
            if pidfd.is_alive():
                pidfd.send_signal(signal.SIGKILL)
                if not pidfd.wait_exit(_SETUP_TIMEOUT):
                    pytest.fail("setup: PID-1 child survived test cleanup", pytrace=False)
            pidfd.close()
        if child.poll() is None:
            _stop_process(child)
        try:
            os.close(control_write)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise


@pytest.mark.host_sandbox
def test_active_registered_helper_is_not_a_leak(tmp_path, monkeypatch):
    """A live registered helper remains unreported while its scope is active."""
    guard = _new_guard(tmp_path, token=_token())
    scope = guard.scope("active-fixture")
    launch = _start_registered_helper(scope, tmp_path)
    try:
        numeric = _forbid_numeric_signals(monkeypatch)
        report = _check_before_deadline(guard)
        _assert_empty_report(report)
        assert launch.process.poll() is None
        assert numeric == []
    finally:
        monkeypatch.undo()
        _stop_process(launch.process)
        scope.end()
        guard.close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="real registered helper with exec-stripped token")
def test_registered_stripped_token_in_active_scope_is_not_a_leak(tmp_path):
    """A registered helper stays unreported while active even after exec strips its token."""
    guard = _new_guard(tmp_path, token=_token())
    scope = guard.scope("active-stripped-token")
    launch = _start_registered_helper(
        scope, tmp_path, script=_STRIP_TOKEN_SCRIPT,
        script_args=(_TOKEN_ENV,))
    try:
        report = guard.check()
        _assert_empty_report(report)
        assert launch.process.poll() is None
    finally:
        _stop_process(launch.process)
        scope.end()
        guard.close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="real pidfd-backed child control")
def test_registered_launch_is_a_leak_without_token_or_adoption_proof(tmp_path, monkeypatch):
    """A registered launch that survives its ended scope is a leak with a retained handle."""
    _registered_leak_scenario(tmp_path, monkeypatch)


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="real registered helper with exec-stripped token")
def test_ended_registered_stripped_token_is_a_leak(tmp_path, monkeypatch):
    """A retained launch handle proves a leftover after its exec removes the token."""
    _registered_leak_scenario(
        tmp_path, monkeypatch, script=_STRIP_TOKEN_SCRIPT,
        script_args=(_TOKEN_ENV,))


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="real PID-1 adoption control")
def test_pid_one_token_bearing_child_is_not_attributed(tmp_path, monkeypatch):
    """A token-bearing PID-1 child without a registered scope or transition proof is ignored."""
    _pid1_token_scenario(tmp_path, monkeypatch)


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="Linux subreaper adoption control")
def test_unobserved_subreaper_double_fork_is_unverified_and_untouched(tmp_path, monkeypatch):
    """A real adopted grandchild without guard-recorded transition evidence stays untouched."""
    guard = _new_guard(tmp_path, token=_token())
    scope = guard.scope("unobserved-double-fork")
    adopted = _SubreaperRun(scope, tmp_path)
    adopted.release_owner()
    scope.end()
    try:
        numeric = _forbid_numeric_signals(monkeypatch)
        report = guard.check()
        _assert_unverified(report, adopted.pid, ("parent", "transition", "evidence"))
        assert adopted.observer_handle.is_alive()
        assert numeric == []
    finally:
        monkeypatch.undo()
        adopted.close()
        guard.close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="Linux subreaper adoption control")
def test_real_subreaper_adoption_with_all_four_facts_is_a_leak(tmp_path, monkeypatch):
    """A real double-fork adopted through a registered subreaper is stopped after proof."""
    guard = _new_guard(tmp_path, token=_token())
    scope = guard.scope("observed-double-fork")
    adopted = _SubreaperRun(scope, tmp_path)
    transition = (
        int(adopted.ready["old_parent_pid"]), int(adopted.adopted["new_parent_pid"]))
    assert adopted.parent_exit["confirmed"] is True
    assert Path(str(adopted.adopted["open_path"])).resolve().is_relative_to(tmp_path.resolve())
    scope.register_adopted(
        adopted.pid, bound_handle=adopted.child_handle, attributed=True,
        parent_transition=transition, parent_exit_confirmed=True, tmp_path_use=True)
    adopted.release_owner()
    scope.end()
    try:
        numeric = _forbid_numeric_signals(monkeypatch)
        report = guard.check()
        kinds = _outcome_kinds(report, adopted.pid)
        assert kinds["leak"] == 1
        assert kinds["confirmed exit"] == 1
        assert report.failed
        assert numeric == []
        assert not adopted.observer_handle.is_alive()
    finally:
        monkeypatch.undo()
        adopted.close()
        guard.close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="Linux subreaper adoption control")
@pytest.mark.parametrize(
    ("missing", "detail_terms"),
    [
        ("attribution", ("attribution", "scope")),
        ("scope end", ("scope", "end")),
        ("parent transition", ("parent", "transition")),
        ("parent death", ("parent", "death", "exit")),
        ("tmp_path use", ("path", "cwd", "file")),
    ],
)
def test_each_missing_adoption_fact_is_unverified_and_untouched(
        tmp_path, monkeypatch, missing, detail_terms):
    """Each missing strict adoption fact is named and leaves the grandchild alive."""
    guard = _new_guard(tmp_path, token=_token())
    scope = guard.scope("incomplete-adoption-proof")
    adopted = _SubreaperRun(scope, tmp_path)
    transition = (
        int(adopted.ready["old_parent_pid"]), int(adopted.adopted["new_parent_pid"]))
    scope.register_adopted(
        adopted.pid, bound_handle=adopted.child_handle,
        attributed=missing != "attribution",
        parent_transition=None if missing == "parent transition" else transition,
        parent_exit_confirmed=missing != "parent death",
        tmp_path_use=missing != "tmp_path use")
    adopted.release_owner()
    if missing != "scope end":
        scope.end()
    try:
        numeric = _forbid_numeric_signals(monkeypatch)
        report = guard.check()
        _assert_unverified(report, adopted.pid, detail_terms)
        assert adopted.observer_handle.is_alive()
        assert adopted.child_handle.signals == []
        assert numeric == []
    finally:
        monkeypatch.undo()
        if missing == "scope end":
            scope.end()
        adopted.close()
        guard.close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="real pidfd-backed permission control")
def test_permission_denied_identity_check_is_unverified_not_gone(tmp_path, monkeypatch):
    """EPERM while checking a registered process is unverified and never treated as exit."""
    handles = []

    def handle_factory(process, *, platform):
        handle = _PermissionDeniedHandle(process.pid)
        handles.append(handle)
        return handle

    guard = _new_guard(tmp_path, token=_token(), handle_factory=handle_factory)
    scope = guard.scope("denied-inspection")
    launch = _start_registered_helper(scope, tmp_path)
    scope.end()
    try:
        numeric = _forbid_numeric_signals(monkeypatch)
        report = guard.check()
        _assert_unverified(report, launch.pid, ("permission", "eperm", "identity"))
        assert "confirmed exit" not in _outcome_kinds(report, launch.pid)
        assert launch.process.poll() is None
        assert handles[0].signals == []
        assert numeric == []
    finally:
        monkeypatch.undo()
        _stop_process(launch.process)
        guard.close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="real pidfd-backed identity race")
def test_identity_replacement_before_term_never_signals_replacement(tmp_path, monkeypatch):
    """A replacement observed after evidence collection is never signalled with TERM."""
    handles = []

    def handle_factory(process, *, platform):
        handle = _EvidenceRaceHandle(process.pid, replace_at="term")
        handles.append(handle)
        return handle

    guard = _new_guard(tmp_path, token=_token(), handle_factory=handle_factory)
    scope = guard.scope("term-identity-race")
    launch = _start_registered_helper(scope, tmp_path)
    scope.end()
    handle = handles[0]
    handle.armed = True
    try:
        numeric = _forbid_numeric_signals(monkeypatch)
        report = guard.check()
        assert "cleanup unverified" in _outcome_kinds(report, launch.pid)
        assert handle.current_identity is not handle.identity
        assert handle.signals == []
        assert numeric == []
    finally:
        monkeypatch.undo()
        _stop_process(launch.process)
        guard.close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="real pidfd-backed identity race")
def test_identity_replacement_before_kill_never_signals_replacement(tmp_path, monkeypatch):
    """A replacement observed after TERM is never signalled with KILL."""
    handles = []

    def handle_factory(process, *, platform):
        handle = _EvidenceRaceHandle(process.pid, replace_at="kill")
        handles.append(handle)
        return handle

    guard = _new_guard(tmp_path, token=_token(), handle_factory=handle_factory)
    scope = guard.scope("kill-identity-race")
    launch = _start_registered_helper(scope, tmp_path)
    scope.end()
    handle = handles[0]
    handle.armed = True
    try:
        numeric = _forbid_numeric_signals(monkeypatch)
        report = guard.check()
        assert "cleanup unverified" in _outcome_kinds(report, launch.pid)
        assert handle.current_identity is not handle.identity
        assert handle.signals == [signal.SIGTERM]
        assert numeric == []
    finally:
        monkeypatch.undo()
        _stop_process(launch.process)
        guard.close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="bound-handle confirmation control")
def test_successful_term_and_kill_without_confirmed_exit_stay_unverified(tmp_path, monkeypatch):
    """Successful TERM and KILL calls do not count as exit without handle confirmation."""
    handles = []

    def handle_factory(process, *, platform):
        handle = _InjectedHandle(process.pid)
        handles.append(handle)
        return handle

    guard = _new_guard(tmp_path, token=_token(), handle_factory=handle_factory)
    scope = guard.scope("signal-without-exit")
    launch = _start_registered_helper(scope, tmp_path)
    scope.end()
    try:
        numeric = _forbid_numeric_signals(monkeypatch)
        report = guard.check()
        kinds = _outcome_kinds(report, launch.pid)
        assert kinds["leak"] == 1
        assert kinds["cleanup unverified"] == 1
        assert "confirmed exit" not in kinds
        assert report.failed
        assert handles[0].signals == [signal.SIGTERM, signal.SIGKILL]
        assert all(timeout > 0 for timeout in handles[0].wait_timeouts)
        assert numeric == []
    finally:
        monkeypatch.undo()
        _stop_process(launch.process)
        guard.close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="real pidfd-backed launch control")
def test_nonzero_signals_use_only_the_registered_handle(tmp_path, monkeypatch):
    """Nonzero cleanup signals never use numeric PID or process-group APIs."""
    _registered_leak_scenario(tmp_path, monkeypatch)


def test_check_returns_before_its_deadline(tmp_path):
    """A live active-scope check returns within a bounded deadline."""
    guard = _new_guard(tmp_path, token=_token())
    scope = guard.scope("bounded-check")
    launch = _start_registered_helper(scope, tmp_path)
    try:
        report = _check_before_deadline(guard)
        _assert_empty_report(report)
    finally:
        _stop_process(launch.process)
        scope.end()
        guard.close()


def test_macos_active_scope_is_empty_and_never_uses_signal_zero(tmp_path):
    """An active macOS scope has no outcomes and its audit-token API receives no signal zero."""
    api_calls = []
    handles = []

    def audit_api(pid, audit_token, signum):
        api_calls.append((pid, audit_token, signum))
        return 0

    def handle_factory(process, *, platform):
        handle = _InjectedHandle(process.pid, signal_api=audit_api)
        handles.append(handle)
        return handle

    guard = _new_guard(
        tmp_path, token=_token(), platform="darwin", handle_factory=handle_factory)
    scope = guard.scope("active-macos")
    launch = _start_registered_helper(scope, tmp_path)
    try:
        report = guard.check()
        _assert_empty_report(report)
        assert 0 not in [signum for _pid, _token, signum in api_calls]
        assert handles[0].signals == []
        assert launch.process.poll() is None
    finally:
        _stop_process(launch.process)
        scope.end()
        guard.close()


def test_macos_ended_candidate_probes_liveness_without_audit_signal_zero_or_fallback(
        tmp_path, monkeypatch):
    """An ended macOS candidate is probed and missing strict signaling stays unverified."""
    api_calls = []
    handles = []

    def unavailable_api(pid, audit_token, signum):
        api_calls.append((pid, audit_token, signum))
        raise OSError(errno.ENOSYS, "strict audit-token signal API unavailable")

    def handle_factory(process, *, platform):
        handle = _InjectedHandle(process.pid, signal_api=unavailable_api)
        handles.append(handle)
        return handle

    guard = _new_guard(
        tmp_path, token=_token(), platform="darwin", handle_factory=handle_factory)
    scope = guard.scope("ended-macos")
    launch = _start_registered_helper(scope, tmp_path)
    scope.end()
    try:
        numeric = _forbid_numeric_signals(monkeypatch)
        report = guard.check()
        _assert_unverified(report, launch.pid, ("audittoken", "strict", "api"))
        assert handles[0].liveness_probes > 0
        assert 0 not in [signum for _pid, _token, signum in api_calls]
        assert numeric == []
        assert "confirmed exit" not in _outcome_kinds(report, launch.pid)
    finally:
        monkeypatch.undo()
        _stop_process(launch.process)
        guard.close()


def test_macos_successful_strict_signal_confirms_exit_without_signal_zero(tmp_path, monkeypatch):
    """A successful strict macOS handle signal plus confirmed exit reports leak and exit."""
    api_calls = []
    handles = []

    def audit_api(pid, audit_token, signum):
        api_calls.append((pid, audit_token, signum))
        return 0

    def handle_factory(process, *, platform):
        handle = _InjectedHandle(
            process.pid, signal_api=audit_api, confirm_after_signal=True)
        handles.append(handle)
        return handle

    guard = _new_guard(
        tmp_path, token=_token(), platform="darwin", handle_factory=handle_factory)
    scope = guard.scope("strict-macos-success")
    launch = _start_registered_helper(scope, tmp_path)
    scope.end()
    try:
        numeric = _forbid_numeric_signals(monkeypatch)
        report = guard.check()
        kinds = _outcome_kinds(report, launch.pid)
        assert kinds["leak"] == 1
        assert kinds["confirmed exit"] == 1
        assert report.failed
        assert [signum for _pid, _token, signum in api_calls] == [signal.SIGTERM]
        assert numeric == []
    finally:
        monkeypatch.undo()
        _stop_process(launch.process)
        guard.close()


def _configure_pytester(pytester, monkeypatch) -> None:
    tests_dir = str(Path(__file__).resolve().parent)
    inherited = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(value for value in (tests_dir, inherited) if value))


def _run_inner(pytester, monkeypatch, source: str):
    pytester.makepyfile(test_inner=source)
    _configure_pytester(pytester, monkeypatch)
    return pytester.runpytest_subprocess(
        "-p", "_orphan_guard", "-q", "-rA", "--tb=short")


@pytest.mark.host_sandbox
def test_pytest_hook_runs_after_fixture_finalizer_stops_helper(pytester, monkeypatch):
    """The real teardown hook sees the helper only after its fixture finalizer stops it."""
    result = _run_inner(pytester, monkeypatch, r'''import os
import select
import signal
import subprocess
import sys
import pytest

HELPER = "import os,signal; os.write(1,b'ready\\n'); signal.pause()"

def wait_ready(process):
    if process.stdout is None:
        pytest.fail("setup: helper has no readiness stream", pytrace=False)
    if not select.select([process.stdout], [], [], 5.0)[0]:
        pytest.fail("setup: helper readiness timed out", pytrace=False)
    if process.stdout.readline() != b"ready\n":
        pytest.fail("setup: helper sent invalid readiness", pytrace=False)

@pytest.fixture
def ended_scope_with_helper(orphan_guard, tmp_path):
    scope = orphan_guard.scope("fixture-finalizer")
    launch = scope.launch([sys.executable, "-c", HELPER], cwd=tmp_path,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    wait_ready(launch.process)
    scope.end()
    yield launch
    launch.process.terminate()
    try:
        launch.process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pytest.fail("setup: fixture finalizer could not stop helper", pytrace=False)

def test_fixture_finalizer_stops_helper(ended_scope_with_helper):
    """The fixture finalizer stops its helper before orphan teardown inspection."""
    assert ended_scope_with_helper.process.poll() is None
''')
    result.assert_outcomes(passed=1)
    assert "ERROR test_inner.py::test_fixture_finalizer_stops_helper" not in result.stdout.str()


@pytest.mark.host_sandbox
def test_pytest_hook_fails_inner_run_for_registered_leak(pytester, monkeypatch):
    """The real teardown hook fails an inner test and names its registered leak."""
    result = _run_inner(pytester, monkeypatch, r'''import atexit
import os
import select
import signal
import subprocess
import sys
import pytest

HELPER = "import os,signal; os.write(1,b'ready\\n'); signal.pause()"

def wait_ready(process):
    if process.stdout is None:
        pytest.fail("setup: helper has no readiness stream", pytrace=False)
    if not select.select([process.stdout], [], [], 5.0)[0]:
        pytest.fail("setup: helper readiness timed out", pytrace=False)
    if process.stdout.readline() != b"ready\n":
        pytest.fail("setup: helper sent invalid readiness", pytrace=False)

def cleanup(process):
    if process.poll() is None:
        process.kill()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pytest.fail("setup: atexit cleanup could not stop helper", pytrace=False)

def test_leaked_registered_helper_fails_teardown(orphan_guard, tmp_path):
    """A registered helper left alive fails the inner teardown as a leak."""
    scope = orphan_guard.scope("inner-leak")
    try:
        launch = scope.launch([sys.executable, "-c", HELPER], cwd=tmp_path,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
    except OSError as exc:
        pytest.fail(f"setup: helper launch failed: {exc}", pytrace=False)
    atexit.register(cleanup, launch.process)
    wait_ready(launch.process)
    scope.end()
    assert launch.process.poll() is None
''')
    result.assert_outcomes(passed=1, errors=1)
    output = result.stdout.str()
    assert "leak" in output.casefold()
    assert "test_leaked_registered_helper_fails_teardown" in output


@pytest.mark.host_sandbox
def test_pytest_hook_fails_inner_run_for_unverified_process(pytester, monkeypatch):
    """The real teardown hook reports missing adoption evidence and fails the inner test."""
    result = _run_inner(pytester, monkeypatch, r'''import atexit
import os
import select
import signal
import subprocess
import sys
import pytest

HELPER = "import os,signal,sys; fd=os.open(sys.argv[1],os.O_CREAT|os.O_RDWR,0o600); os.write(1,b'ready\\n'); signal.pause()"

class Handle:
    def __init__(self, pid):
        self.pid = pid
        self.identity = object()
        self.signals = []
    def is_current(self): return True
    def is_alive(self): return True
    def send_signal(self, signum): self.signals.append(signum)
    def wait_exit(self, timeout): return False
    def close(self): pass

def wait_ready(process):
    if process.stdout is None:
        pytest.fail("setup: helper has no readiness stream", pytrace=False)
    if not select.select([process.stdout], [], [], 5.0)[0]:
        pytest.fail("setup: helper readiness timed out", pytrace=False)
    if process.stdout.readline() != b"ready\n":
        pytest.fail("setup: helper sent invalid readiness", pytrace=False)

def cleanup(process):
    if process.poll() is None:
        process.kill()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pytest.fail("setup: atexit cleanup could not stop helper", pytrace=False)

def test_unverified_candidate_fails_teardown(orphan_guard, tmp_path):
    """An unverifiable process fails the inner teardown with its missing evidence."""
    use_path = tmp_path / "unverified-open-file"
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", HELPER, str(use_path)], cwd=tmp_path,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
    except OSError as exc:
        pytest.fail(f"setup: helper launch failed: {exc}", pytrace=False)
    atexit.register(cleanup, process)
    wait_ready(process)
    scope = orphan_guard.scope("inner-unverified")
    scope.register_adopted(process.pid, bound_handle=Handle(process.pid), attributed=True,
                           parent_transition=None, parent_exit_confirmed=True,
                           tmp_path_use=True)
    scope.end()
''')
    result.assert_outcomes(passed=1, errors=1)
    output = result.stdout.str().casefold()
    assert "cleanup unverified" in output
    assert "parent transition" in output


def _signal_means_exit_scenario(tmp_path: Path, monkeypatch, variant=None) -> None:
    handles = []

    def handle_factory(process, *, platform):
        handle = _InjectedHandle(process.pid)
        handles.append(handle)
        return handle

    guard = _new_guard(
        tmp_path, token=_token(), handle_factory=handle_factory, variant=variant)
    scope = guard.scope("signal-does-not-confirm")
    launch = _start_registered_helper(scope, tmp_path)
    scope.end()
    try:
        report = guard.check()
        kinds = _outcome_kinds(report, launch.pid)
        assert "confirmed exit" not in kinds
        assert kinds["cleanup unverified"] == 1
        assert handles[0].signals == [signal.SIGTERM, signal.SIGKILL]
    finally:
        _stop_process(launch.process)
        guard.close()


_MUTANT_SCENARIOS = {
    "numeric_pid_signal": _registered_leak_scenario,
    "signal_unverified_process": _unverified_adoption_scenario,
    "signal_means_exit": _signal_means_exit_scenario,
    "token_only_attribution": _pid1_token_scenario,
    "never_reports": _registered_leak_scenario,
}


@pytest.mark.host_sandbox
@pytest.mark.skipif(not _IS_LINUX, reason="Linux behavior mutation controls")
@pytest.mark.parametrize("mutant_name", sorted(MUTANTS))
def test_mutants_are_rejected(mutant_name, tmp_path, monkeypatch):
    """Each deliberately wrong public-API variant is rejected by a behavior scenario."""
    variant = MUTANTS[mutant_name]
    scenario = _MUTANT_SCENARIOS[mutant_name]
    failures = []
    try:
        scenario(tmp_path, monkeypatch, variant)
    except AssertionError as exc:
        failures.append(str(exc))
    assert failures, f"mutant {mutant_name} survived its acceptance scenario"
