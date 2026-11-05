"""Pytest support for reporting registered child processes left alive by a test."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import ctypes
import errno
import math
import os
from pathlib import Path
import select
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

import pytest


_TOKEN_ENV = "SF_TEST_TOKEN"
_TERM_GRACE_SECONDS = 0.35
_KILL_GRACE_SECONDS = 0.65
_CHECK_DEADLINE_SECONDS = 8.0
_PIDFD_EXIT_EVENTS = select.POLLIN | select.POLLHUP
_PIDFD_ERROR_EVENTS = select.POLLERR | select.POLLNVAL
_PIDFD_EVENTS = _PIDFD_EXIT_EVENTS | _PIDFD_ERROR_EVENTS
_CLOSE_REAP_TIMEOUT_SECONDS = 1.0
_PROC_PIDT_BSDINFOWITHUNIQID = 18

OutcomeKind = Literal["leak", "cleanup unverified", "confirmed exit"]


class OrphanGuardError(RuntimeError):
    """The guard cannot safely establish process ownership or cleanup state."""


class BoundHandle(Protocol):
    pid: int
    identity: object

    def is_current(self) -> bool: ...
    def is_alive(self) -> bool: ...
    def send_signal(self, signum: int) -> None: ...
    def wait_exit(self, timeout: float) -> bool: ...
    def close(self) -> None: ...


class HandleFactory(Protocol):
    def __call__(self, process: subprocess.Popen[bytes], *, platform: str) -> BoundHandle: ...


class _TestReport(Protocol):
    outcome: str
    longrepr: object | None
    sections: list[tuple[str, str]]


@dataclass(frozen=True)
class Outcome:
    kind: OutcomeKind
    pid: int
    detail: str


@dataclass(frozen=True)
class Report:
    outcomes: tuple[Outcome, ...]
    failed: bool
    # One final disposition per registered unit in an ended scope. `outcomes`
    # remains the event log, where a detected leak and its confirmed cleanup
    # are intentionally separate events.
    unit_outcomes: tuple[Outcome, ...] = ()


@dataclass
class _Launch:
    pid: int
    process: subprocess.Popen[bytes]
    bound_handle: BoundHandle
    identity: object
    checked: bool = False
    outcome: Outcome | None = None


@dataclass
class _Adopted:
    pid: int
    bound_handle: BoundHandle
    identity: object
    attributed: bool
    parent_transition: tuple[int, int] | None
    parent_exit_confirmed: bool
    tmp_path_use: bool
    checked: bool = False
    outcome: Outcome | None = None


@dataclass
class _Scope:
    guard: OrphanGuard
    name: str
    ended: bool = False
    launches: list[_Launch] = field(default_factory=list)
    adopted: list[_Adopted] = field(default_factory=list)

    def launch(self, args: Sequence[str], **kwargs: object) -> _Launch:
        if self.guard._closed:
            raise OrphanGuardError("orphan guard is closed")
        if self.ended:
            raise OrphanGuardError(f"scope {self.name!r} has ended; it cannot launch another process")
        launch_kwargs = dict(kwargs)
        supplied_env = launch_kwargs.get("env")
        if supplied_env is None:
            child_env = os.environ.copy()
        elif isinstance(supplied_env, Mapping):
            child_env = dict(supplied_env)
        else:
            raise TypeError("scope.launch env must be a mapping or None")
        if self.guard.token is not None:
            child_env[_TOKEN_ENV] = self.guard.token
        launch_kwargs["env"] = child_env
        process = subprocess.Popen(args, **launch_kwargs)  # type: ignore[arg-type]
        handle, identity = self.guard._new_handle(process)
        record = _Launch(process.pid, process, handle, identity)
        self.launches.append(record)
        return record

    def end(self) -> None:
        self.ended = True

    def register_adopted(
        self,
        pid: int,
        *,
        bound_handle: BoundHandle,
        attributed: bool,
        parent_transition: tuple[int, int] | None,
        parent_exit_confirmed: bool,
        tmp_path_use: bool,
    ) -> _Adopted:
        if self.guard._closed:
            raise OrphanGuardError("orphan guard is closed")
        if self.ended:
            raise OrphanGuardError(f"scope {self.name!r} has ended; it cannot register a process")
        _validate_pid(pid)
        bound_handle, identity = _validate_handle(bound_handle, pid)
        if type(attributed) is not bool:
            raise TypeError("attributed must be bool")
        if type(parent_exit_confirmed) is not bool:
            raise TypeError("parent_exit_confirmed must be bool")
        if type(tmp_path_use) is not bool:
            raise TypeError("tmp_path_use must be bool")
        if any(record.pid == pid and record.identity == identity
               for record in (*self.launches, *self.adopted)):
            raise ValueError(f"process {pid} is already registered in scope {self.name!r}")
        record = _Adopted(pid, bound_handle, identity, attributed, parent_transition,
                          parent_exit_confirmed, tmp_path_use)
        self.adopted.append(record)
        self.guard._handles.append(bound_handle)
        return record


@dataclass(frozen=True)
class _MacProcessIdentity:
    pid: int
    uid: int
    start_sec: int
    start_usec: int
    pid_version: int
    zombie: bool

    @property
    def audit_token(self) -> tuple[int, ...]:
        # XNU validates the PID version while resolving this audit token.
        return (0, 0, 0, 0, 0, self.pid, 0, self.pid_version & 0xFFFFFFFF)


class _BSDInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "flags", "status", "xstatus", "pid", "ppid", "uid", "gid", "ruid", "rgid",
        "svuid", "svgid", "rfu")]
    _fields_ += [("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32)]
    _fields_ += [(name, ctypes.c_uint32) for name in (
        "nfiles", "pgid", "pjobc", "tdev", "tpgid", "nice")]
    _fields_ += [(name, ctypes.c_uint64) for name in ("start_sec", "start_usec")]


class _UniqueInfo(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_ubyte * 16), ("uniqueid", ctypes.c_uint64),
                ("parent_uniqueid", ctypes.c_uint64), ("idversion", ctypes.c_int32),
                ("reserved2", ctypes.c_uint32), ("reserved3", ctypes.c_uint64),
                ("reserved4", ctypes.c_uint64)]


class _BSDInfoWithUnique(ctypes.Structure):
    _fields_ = [("bsd", _BSDInfo), ("unique", _UniqueInfo)]


def _read_macos_identity(pid: int) -> _MacProcessIdentity | None:
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    except OSError as error:
        raise OrphanGuardError("macOS libproc is unavailable for audit-token identity") from error
    proc_pidinfo = getattr(libproc, "proc_pidinfo", None)
    if proc_pidinfo is None:
        raise OrphanGuardError("macOS proc_pidinfo is unavailable for audit-token identity")
    proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                             ctypes.c_void_p, ctypes.c_int]
    proc_pidinfo.restype = ctypes.c_int
    info = _BSDInfoWithUnique()
    ctypes.set_errno(0)
    expected_size = ctypes.sizeof(info)
    count = proc_pidinfo(pid, _PROC_PIDT_BSDINFOWITHUNIQID, 0,
                         ctypes.byref(info), expected_size)
    if count == 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.ESRCH, errno.ENOENT):
            return None
        raise OSError(error_number, "cannot read macOS audit-token process identity", pid)
    if count != expected_size:
        raise OrphanGuardError(
            f"macOS proc_pidinfo returned {count} bytes; expected {expected_size} "
            "for combined process identity")
    if info.bsd.pid != pid or info.unique.idversion <= 0:
        raise OrphanGuardError("macOS process identity has no valid PID version")
    return _MacProcessIdentity(
        pid, info.bsd.uid, info.bsd.start_sec, info.bsd.start_usec,
        info.unique.idversion, info.bsd.status == 5)


def _positive_signal(signum: int, handle_name: str) -> int:
    if not isinstance(signum, int) or isinstance(signum, bool) or signum <= 0:
        raise ValueError(f"{handle_name} process signals must be positive")
    return int(signum)


class _PidfdHandle:
    """A Linux process reference retained from registration through cleanup."""

    def __init__(self, pid: int) -> None:
        pidfd_open = getattr(os, "pidfd_open", None)
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        if pidfd_open is None or pidfd_send_signal is None:
            raise OrphanGuardError("Linux pidfd open and pidfd_send_signal are required")
        self.pid = pid
        self._fd = pidfd_open(pid, 0)
        self.identity = ("linux-pidfd", pid, self._fd)
        try:
            self._poller = select.poll()
            self._poller.register(self._fd, _PIDFD_EVENTS)
        except Exception:
            os.close(self._fd)
            self._fd = -1
            raise

    def is_current(self) -> bool:
        if self._fd < 0:
            raise OrphanGuardError("retained Linux pidfd is closed")
        return True

    def is_alive(self) -> bool:
        return not self.wait_exit(0.0)

    def send_signal(self, signum: int) -> None:
        signum = _positive_signal(signum, "bound")
        if self._fd < 0:
            raise OrphanGuardError("retained Linux pidfd is closed")
        signal.pidfd_send_signal(self._fd, signum)

    def wait_exit(self, timeout: float) -> bool:
        timeout_ms = _timeout_milliseconds(timeout)
        if self._fd < 0:
            raise OrphanGuardError("retained Linux pidfd is closed")
        events = self._poller.poll(timeout_ms)
        if not events:
            return False
        event_mask = 0
        for ready_fd, mask in events:
            if ready_fd != self._fd:
                raise OrphanGuardError(
                    f"Linux pidfd poll returned unexpected descriptor {ready_fd}")
            event_mask |= mask
        if event_mask & _PIDFD_ERROR_EVENTS:
            raise OrphanGuardError(
                f"Linux pidfd poll reported cleanup-unverified event mask 0x{event_mask:x}")
        if event_mask & ~_PIDFD_EXIT_EVENTS:
            raise OrphanGuardError(
                f"Linux pidfd poll reported unrecognized event mask 0x{event_mask:x}")
        return bool(event_mask & _PIDFD_EXIT_EVENTS)

    def close(self) -> None:
        if self._fd >= 0:
            fd = self._fd
            self._fd = -1
            os.close(fd)


class _MacAuditTokenHandle:
    """A Darwin process generation with strict audit-token-only signal authority."""

    def __init__(self, pid: int, *, process: subprocess.Popen[bytes] | None = None) -> None:
        if process is not None and process.pid != pid:
            raise ValueError("macOS bound process pid does not match its captured identity")
        identity = _read_macos_identity(pid)
        if identity is None:
            raise ProcessLookupError(f"process {pid} exited before audit-token registration")
        self.pid = pid
        self._process = process
        self._identity = identity
        self.identity = ("darwin-audit-token", identity)
        self._closed = False

    def _read_current_identity(self) -> _MacProcessIdentity | None:
        current = _read_macos_identity(self.pid)
        if current is not None and current == self._identity:
            return current
        if self._process is None:
            if current is None:
                raise OrphanGuardError(
                    "macOS process identity is unavailable without a waitable child handle")
            return current
        if self._process.poll() is None:
            reason = "not found" if current is None else "does not match the captured generation"
            raise OrphanGuardError(
                f"macOS process identity {reason} while registered child remains alive")
        return None

    def is_current(self) -> bool:
        if self._closed:
            raise OrphanGuardError("retained macOS audit-token identity is closed")
        current = self._read_current_identity()
        return current is not None and current == self._identity

    def is_alive(self) -> bool:
        if self._closed:
            raise OrphanGuardError("retained macOS audit-token identity is closed")
        current = self._read_current_identity()
        if current is None or current != self._identity:
            return False
        if current.zombie and self._process is not None and self._process.poll() is None:
            raise OrphanGuardError(
                "macOS process is marked zombie but registered child exit is not wait-confirmed")
        return not current.zombie

    def send_signal(self, signum: int) -> None:
        signum = _positive_signal(signum, "audit-token")
        if self._closed:
            raise OrphanGuardError("retained macOS audit-token identity is closed")
        if not self.is_current():
            raise ProcessLookupError(f"process {self.pid} no longer matches its audit-token identity")
        try:
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        except OSError as error:
            raise OrphanGuardError("strict macOS audit-token API unavailable: libproc load failed") from error
        proc_signal = getattr(libproc, "proc_signal_with_audittoken", None)
        if proc_signal is None:
            raise OrphanGuardError("strict macOS audit-token API unavailable: signal symbol missing")
        audit_token = (ctypes.c_uint32 * 8)(*self._identity.audit_token)
        proc_signal.argtypes = [ctypes.c_void_p, ctypes.c_int]
        result = proc_signal(ctypes.byref(audit_token), signum)
        if result in (errno.ESRCH, errno.ENOENT):
            raise ProcessLookupError(f"process {self.pid} exited before audit-token signal")
        if result != 0:
            if result in (errno.ENOSYS, getattr(errno, "ENOTSUP", errno.ENOSYS)):
                raise OrphanGuardError(f"strict macOS audit-token API unavailable: errno {result}")
            raise OSError(result, "strict audit-token signal failed", self.pid)

    def wait_exit(self, timeout: float) -> bool:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("exit wait timeout must be finite and nonnegative")
        if self._process is not None:
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return False
            return True
        deadline = time.monotonic() + timeout
        while True:
            if not self.is_alive():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.02, remaining))

    def close(self) -> None:
        self._closed = True


class _UnavailableHandle:
    """A launch is retained for reporting when a platform cannot bind signals safely."""

    def __init__(self, process: subprocess.Popen[bytes], error: Exception) -> None:
        self.pid = process.pid
        self.identity = ("unavailable", process.pid, id(error))
        self._process = process
        self._error = error
        self._closed = False

    def is_current(self) -> bool:
        return not self._closed

    def is_alive(self) -> bool:
        if self._closed:
            raise OrphanGuardError("unavailable process handle is closed")
        return self._process.poll() is None

    def send_signal(self, signum: int) -> None:
        _positive_signal(signum, "bound")
        raise OrphanGuardError(f"bound signaling unavailable: {_error_text(self._error)}")

    def wait_exit(self, timeout: float) -> bool:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("exit wait timeout must be finite and nonnegative")
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def close(self) -> None:
        self._closed = True


def _system_handle_factory(process: subprocess.Popen[bytes], *, platform: str) -> BoundHandle:
    if platform.startswith("linux"):
        return _PidfdHandle(process.pid)
    if platform == "darwin":
        return _MacAuditTokenHandle(process.pid, process=process)
    raise OrphanGuardError(f"bound process signaling is unsupported on {platform}")


class OrphanGuard:
    """Track only explicitly registered test-owned processes and adopted candidates."""

    def __init__(self, root: Path, *, token: str | None = None,
                 platform: str | None = None, handle_factory: HandleFactory | None = None) -> None:
        try:
            self.root = Path(root).resolve(strict=True)
        except OSError as error:
            raise ValueError(f"orphan guard root is not accessible: {root}") from error
        if not self.root.is_dir():
            raise ValueError("orphan guard root must be a directory")
        if token is not None and (not isinstance(token, str) or not token or "\x00" in token):
            raise ValueError("orphan guard token must be a nonempty string without NUL")
        self.token = token if token is not None else secrets.token_hex(24)
        self.platform = platform or sys.platform
        self.handle_factory = handle_factory
        self._scopes: list[_Scope] = []
        self._handles: list[BoundHandle] = []
        self._outcomes: list[Outcome] = []
        self._closed = False

    def scope(self, name: str) -> _Scope:
        if self._closed:
            raise OrphanGuardError("orphan guard is closed")
        if not isinstance(name, str) or not name.strip() or "\x00" in name:
            raise ValueError("scope name must be a nonempty string without NUL")
        scope = _Scope(self, name)
        self._scopes.append(scope)
        return scope

    def _new_handle(self, process: subprocess.Popen[bytes]) -> tuple[BoundHandle, object]:
        factory = self.handle_factory or _system_handle_factory
        candidate: object | None = None
        try:
            candidate = factory(process, platform=self.platform)
            bound_handle, identity = _validate_handle(candidate, process.pid)
        except Exception as error:
            if candidate is not None:
                close = getattr(candidate, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as close_error:
                        error = OrphanGuardError(
                            f"{_error_text(error)}; failed to close invalid handle: "
                            f"{_error_text(close_error)}")
            bound_handle = _UnavailableHandle(process, error)
            identity = bound_handle.identity
        self._handles.append(bound_handle)
        return bound_handle, identity

    def _end_open_scopes(self) -> None:
        """Close any scopes still active once pytest has finished fixture finalizers."""
        for scope in self._scopes:
            scope.end()

    def check(self) -> Report:
        if self._closed:
            raise OrphanGuardError("orphan guard is closed")
        deadline = time.monotonic() + _CHECK_DEADLINE_SECONDS
        for scope in self._scopes:
            for launch in scope.launches:
                if scope.ended and not launch.checked:
                    self._check_unit(
                        scope, launch,
                        lambda scope=scope, launch=launch: self._check_launch(
                            scope, launch, deadline))
            for adopted in scope.adopted:
                if scope.ended and not adopted.checked:
                    self._check_unit(
                        scope, adopted,
                        lambda scope=scope, adopted=adopted: self._check_adopted(
                            scope, adopted, deadline))
        failed = any(outcome.kind != "confirmed exit" for outcome in self._outcomes)
        unit_outcomes = tuple(
            unit.outcome
            for scope in self._scopes if scope.ended
            for unit in (*scope.launches, *scope.adopted)
            if unit.checked and unit.outcome is not None)
        return Report(tuple(self._outcomes), failed, unit_outcomes)

    def _check_unit(self, scope: _Scope, unit: _Launch | _Adopted,
                    check: Callable[[], None]) -> None:
        """Record one canonical disposition even when an adapter check fails."""
        if unit.checked:
            return
        unit.checked = True
        first_outcome = len(self._outcomes)
        try:
            check()
        except Exception as error:
            self._unverified(
                unit.pid,
                f"registered process in scope {scope.name!r} could not be checked: "
                f"{_error_text(error)}")
        events = self._outcomes[first_outcome:]
        if not events:
            self._unverified(
                unit.pid,
                f"registered process in scope {scope.name!r} produced no check outcome")
            events = self._outcomes[first_outcome:]
        unverified = next((event for event in reversed(events)
                           if event.kind == "cleanup unverified"), None)
        leak = next((event for event in events if event.kind == "leak"), None)
        unit.outcome = unverified or leak or events[-1]

    def _check_launch(self, scope: _Scope, launch: _Launch, deadline: float) -> None:
        current, alive, detail = _probe(launch.bound_handle, launch.identity)
        if detail:
            self._outcomes.append(Outcome(
                "cleanup unverified", launch.pid,
                f"registered launch in ended scope {scope.name!r}: {detail}"))
            return
        if not alive:
            self._outcomes.append(Outcome(
                "confirmed exit", launch.pid,
                "registered process exit confirmed through its retained bound handle"))
            return
        if not current:
            self._outcomes.append(Outcome(
                "cleanup unverified", launch.pid,
                "registered launch identity is no longer current; cleanup left untouched"))
            return
        self._cleanup_live(
            launch.pid, launch.bound_handle, launch.identity, deadline,
            f"registered process remained alive after scope {scope.name!r} ended")

    def _check_adopted(self, scope: _Scope, adopted: _Adopted, deadline: float) -> None:
        missing = _missing_adoption_facts(scope, adopted)
        if missing:
            self._unverified(adopted.pid, "adopted candidate cleanup unverified: " + "; ".join(missing))
            return
        current, alive, detail = _probe(adopted.bound_handle, adopted.identity)
        if detail:
            self._unverified(adopted.pid, f"adopted candidate identity/liveness: {detail}")
            return
        if not alive:
            self._confirmed_exit(adopted.pid, "adopted process exit confirmed through its retained bound handle")
            return
        if not current:
            self._unverified(adopted.pid, "adopted candidate bound identity is no longer current")
            return
        self._cleanup_live(
            adopted.pid, adopted.bound_handle, adopted.identity, deadline,
            f"adopted process remained alive after scope {scope.name!r} ended with all required evidence")

    def _cleanup_live(self, pid: int, handle: BoundHandle, identity: object,
                      deadline: float, leak_detail: str) -> None:
        self._outcomes.append(Outcome("leak", pid, leak_detail))
        if not self._has_time(deadline):
            self._unverified(pid, "check deadline expired before SIGTERM")
            return
        try:
            if not _same_bound_identity(handle, identity):
                self._unverified(pid, "bound identity changed before SIGTERM; process left untouched")
                return
            handle.send_signal(signal.SIGTERM)
        except Exception as error:
            self._signal_error(pid, handle, identity, "SIGTERM", error)
            return
        if self._wait_confirmed_exit(pid, handle, identity, _TERM_GRACE_SECONDS, deadline):
            return
        if not self._has_time(deadline):
            self._unverified(pid, "check deadline expired before SIGKILL escalation")
            return
        try:
            if not _same_bound_identity(handle, identity):
                self._unverified(pid, "bound identity changed before SIGKILL; escalation withheld")
                return
            if not handle.is_alive():
                self._confirmed_exit(pid, "process exit observed through retained bound handle")
                return
            if not _same_bound_identity(handle, identity):
                self._unverified(pid, "bound identity changed before SIGKILL; escalation withheld")
                return
            handle.send_signal(signal.SIGKILL)
        except Exception as error:
            self._signal_error(pid, handle, identity, "SIGKILL", error)
            return
        if not self._wait_confirmed_exit(pid, handle, identity, _KILL_GRACE_SECONDS, deadline):
            self._unverified(pid, "SIGKILL returned but bound-handle exit was not confirmed")

    def _wait_confirmed_exit(self, pid: int, handle: BoundHandle, identity: object,
                             grace: float, deadline: float) -> bool:
        timeout = min(grace, max(0.0, deadline - time.monotonic()))
        if timeout <= 0:
            return False
        try:
            if handle.wait_exit(timeout):
                self._confirmed_exit(pid, "process exit confirmed by bounded retained-handle wait")
                return True
            current, alive, detail = _probe(handle, identity)
        except Exception as error:
            self._unverified(pid, f"bounded exit wait failed: {_error_text(error)}")
            return True
        if detail:
            self._unverified(pid, f"after bounded exit wait: {detail}")
            return True
        if not alive:
            self._confirmed_exit(pid, "process exit observed through retained bound handle")
            return True
        if not current:
            self._unverified(pid, "bound identity changed during bounded exit wait")
            return True
        return False

    def _signal_error(self, pid: int, handle: BoundHandle, identity: object,
                      phase: str, error: Exception) -> None:
        current, alive, detail = _probe(handle, identity)
        if detail:
            self._unverified(pid, f"{phase} failed: {_error_text(error)}; liveness unknown: {detail}")
            return
        if not alive:
            self._confirmed_exit(pid, f"process exit observed after {phase} failure through retained handle")
            return
        if not current:
            self._unverified(pid, f"{phase} failed after bound identity changed")
            return
        self._unverified(pid, f"{phase} failed through bound handle: {_error_text(error)}")

    def _confirmed_exit(self, pid: int, detail: str) -> None:
        self._outcomes.append(Outcome("confirmed exit", pid, detail))

    def _unverified(self, pid: int, detail: str) -> None:
        self._outcomes.append(Outcome("cleanup unverified", pid, detail))

    @staticmethod
    def _has_time(deadline: float) -> bool:
        return time.monotonic() < deadline

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        deadline = time.monotonic() + _CLOSE_REAP_TIMEOUT_SECONDS
        launches = [launch for scope in self._scopes for launch in scope.launches]
        for launch in launches:
            process = launch.process
            try:
                if process.returncode is None:
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        process.wait(timeout=remaining)
                    elif process.poll() is None:
                        errors.append(
                            f"could not reap registered child {launch.pid} before close deadline")
            except subprocess.TimeoutExpired:
                errors.append(
                    f"could not reap registered child {launch.pid} before close deadline")
            except Exception as error:
                errors.append(
                    f"could not reap registered child {launch.pid}: {_error_text(error)}")
        for launch in launches:
            for stream_name in ("stdin", "stdout", "stderr"):
                stream = getattr(launch.process, stream_name)
                if stream is None:
                    continue
                try:
                    stream.close()
                except Exception as error:
                    errors.append(
                        f"could not close registered child {launch.pid} {stream_name} pipe: "
                        f"{_error_text(error)}")
        closed: set[int] = set()
        for handle in self._handles:
            if id(handle) in closed:
                continue
            closed.add(id(handle))
            try:
                handle.close()
            except Exception as error:
                errors.append(_error_text(error))
        if errors:
            raise OrphanGuardError("could not close all retained process resources: " + "; ".join(errors))


def _validate_pid(pid: int) -> None:
    if type(pid) is not int or pid <= 0:
        raise ValueError("a positive process id is required")


def _validate_handle(handle: object, pid: int) -> tuple[BoundHandle, object]:
    _validate_pid(pid)
    if getattr(handle, "pid", None) != pid:
        raise ValueError(f"bound handle pid does not match registered pid {pid}")
    identity = getattr(handle, "identity", None)
    if identity is None:
        raise ValueError("bound handle must expose a stable identity")
    for name in ("is_current", "is_alive", "send_signal", "wait_exit", "close"):
        if not callable(getattr(handle, name, None)):
            raise TypeError(f"bound handle must provide {name}()")
    return cast(BoundHandle, handle), identity


def _same_bound_identity(handle: BoundHandle, identity: object) -> bool:
    if handle.identity != identity:
        return False
    return handle.is_current() and handle.identity == identity


def _probe(handle: BoundHandle, identity: object) -> tuple[bool, bool, str]:
    try:
        if handle.identity != identity:
            return False, False, "retained bound handle identity changed"
        current = handle.is_current()
        alive = handle.is_alive()
        if handle.identity != identity:
            return False, False, "retained bound handle identity changed during liveness check"
    except Exception as error:
        return False, False, "bound identity/liveness check failed: " + _error_text(error)
    if not alive:
        return bool(current), False, ""
    if not current:
        return False, True, "bound identity is no longer current while candidate remains alive"
    return True, True, ""


def _missing_adoption_facts(scope: _Scope, adopted: _Adopted) -> list[str]:
    missing = []
    if not adopted.attributed:
        missing.append("attribution to a registered scope is missing")
    if not scope.ended:
        missing.append("owning scope end is not observed")
    transition = adopted.parent_transition
    if (not isinstance(transition, tuple) or len(transition) != 2
            or any(type(pid) is not int or pid <= 0 for pid in transition)
            or transition[0] == transition[1]):
        missing.append("parent transition evidence is missing or invalid")
    if not adopted.parent_exit_confirmed:
        missing.append("former parent death/exit is not confirmed")
    if not adopted.tmp_path_use:
        missing.append("cwd/open-file use beneath tmp_path is not observed")
    return missing


def _timeout_milliseconds(timeout: float) -> int:
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("exit wait timeout must be finite and nonnegative")
    if timeout >= 2_147_483_647 / 1000:
        return 2_147_483_647
    return min(2_147_483_647, math.ceil(timeout * 1000))


def _error_text(error: Exception) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


_GUARD_KEY = pytest.StashKey[OrphanGuard]()


def _finish_guard(item: pytest.Item) -> list[str]:
    guard = item.stash.get(_GUARD_KEY, None)
    if guard is None:
        return []
    failures: list[str] = []
    try:
        guard._end_open_scopes()
        report = guard.check()
        if report.failed:
            failures.extend(
                f"{outcome.kind}: pid {outcome.pid}: {outcome.detail}"
                for outcome in report.outcomes if outcome.kind != "confirmed exit")
    except Exception as error:
        failures.append("guard check failed: " + _error_text(error))
    finally:
        try:
            guard.close()
        except Exception as error:
            failures.append("guard close failed: " + _error_text(error))
        del item.stash[_GUARD_KEY]
    return failures


def _apply_teardown_failures(report: _TestReport, failures: Sequence[str]) -> None:
    if not failures:
        return
    message = "orphan guard: " + "\n".join(failures)
    if report.outcome == "failed":
        report.sections.append(("orphan guard", message))
        return
    report.outcome = "failed"
    report.longrepr = message


@pytest.fixture
def orphan_guard(tmp_path: Path, request: pytest.FixtureRequest) -> OrphanGuard:
    guard = OrphanGuard(tmp_path)
    request.node.stash[_GUARD_KEY] = guard
    return guard


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[object]):
    """Fail the generated teardown report after fixture finalizers and guard cleanup."""
    failures = _finish_guard(item) if call.when == "teardown" else []
    outcome = yield
    if failures:
        report = outcome.get_result()
        _apply_teardown_failures(report, failures)
