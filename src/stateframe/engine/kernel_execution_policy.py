"""One fail-closed execution policy for model-influenced child processes.

This module is the only v46 entry point for new model-controlled subprocesses.
It deliberately separates host capability discovery from child execution so a
missing sandbox, unsupported platform, invalid limit, or output flood cannot
fall back to a raw host process.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import errno
import logging
import math
import os
from pathlib import Path
import platform
import re
import resource
import selectors
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Callable, Iterator, TypeVar

from kernel_exec_tracker import (ContainmentUnverified, Descendants, LaunchIdentity,
                                 ProcessIdentity, ProcessTable, capture_launch_identity,
                                 child_exited_unreaped, kill_launch_unit,
                                 recorded_launch_identity, signal_launch_group)
from kernel_test_isolation import SandboxReadGrants, isolated_command, sandbox_executable
from sf_launch import (
    _cgroup_leaf_pattern_status,
    _cgroup_memory_limit_configured,
    _command_cgroup,
)


MAX_WALL_SECONDS = 900.0
MAX_INPUT_BYTES = 4 * 1024 * 1024
DEFAULT_CPU_SECONDS = 30.0
# User checks get RLIMIT_CPU only as a runaway backstop, never as a budget: real suites can
# legitimately burn many CPU minutes. STATEFRAME_PUBLIC_CHECK_CPU_SECONDS overrides it.
CHECK_CPU_BACKSTOP_SECONDS = 3600.0
MAX_CPU_SECONDS = 86400.0
DEFAULT_MEMORY_BYTES = 512 * 1024 * 1024
DEFAULT_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_NOFILE = 128
DEFAULT_NPROC = 64
MAX_OUTPUT_BYTES = 256 * 1024
MAX_PROFILE_OUTPUT_BYTES = 16 * 1024
CGROUP_CLEANUP_ATTEMPTS = 40
CGROUP_CLEANUP_INTERVAL_SECONDS = 0.025
FULL_PROCESS_SCAN_INTERVAL_SECONDS = 1.0
# A procfs sample can race process exit or return a torn read, and under memory
# pressure it can be slow. Such a sample is skipped and taken again. Only this many
# consecutive failed samples (a count, never elapsed time) prove procfs unreadable.
PROCFS_SAMPLE_MAX_CONSECUTIVE_FAILURES = 500
_PROCFS_RACE_ERRNOS = frozenset({errno.ESRCH, errno.ENOENT})
_Result = TypeVar("_Result")
_PROFILE_FD_ENV = "STATEFRAME_TEST_RUN_SPANS_FD"
_LOGGER = logging.getLogger(__name__)

_ALLOWED_ENV = frozenset({
    "PATH", "LANG", "LC_ALL", "PYTHONHASHSEED", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
})


class ExecutionPolicyError(RuntimeError):
    """A child was refused because the required execution boundary is absent."""


@dataclass(frozen=True)
class CapabilityReport:
    platform: str
    backend: str | None
    available: bool
    network_isolated: bool
    path_contained: bool
    cpu_limited: bool
    memory_limited: bool
    output_bounded: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-execution-capabilities/1",
            "platform": self.platform,
            "backend": self.backend,
            "available": self.available,
            "network_isolated": self.network_isolated,
            "path_contained": self.path_contained,
            "cpu_limited": self.cpu_limited,
            "memory_limited": self.memory_limited,
            "output_bounded": self.output_bounded,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PolicyResult:
    completed: subprocess.CompletedProcess[str]
    timed_out: bool
    process_group_killed: bool
    output_truncated: bool
    capabilities: CapabilityReport
    pid: int | None = None
    execution_trace: Mapping[str, int] = field(default_factory=dict)
    profile_payload: bytes | None = None
    read_grants: SandboxReadGrants | None = None
    memory_limit_mechanism: str | None = None
    memory_limit_killed: bool = False
    memory_limit_reason: str | None = None
    cgroup_leaf_pattern: str | None = None
    memory_limit_samples_skipped: int = 0
    memory_limit_sample_skip_reason: str | None = None
    memory_limit_stop_cause: str | None = None
    # RLIMIT_CPU requested for the check, and the CPU its whole launch unit used.
    cpu_limit_seconds: float | None = None
    cpu_seconds_used: float | None = None


def policy_read_grants(result: object) -> SandboxReadGrants | None:
    """Return grants when recorded; older policy results have no such field."""
    return getattr(result, "read_grants", None)


def policy_memory_limit_mechanism(result: object) -> str | None:
    """Return the recorded memory limit mechanism, if this is a newer result."""
    value = getattr(result, "memory_limit_mechanism", None)
    return value if isinstance(value, str) else None


def capability_report() -> CapabilityReport:
    """Return the host capability matrix without starting a child process."""
    system = platform.system()
    backend_name = "sandbox-exec" if system == "Darwin" else "bwrap" if system == "Linux" else None
    executable = sandbox_executable(backend_name) if backend_name else None
    required_limits = (
        "RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_CORE", "RLIMIT_NOFILE",
    )
    limits_available = all(hasattr(resource, name) for name in required_limits)
    memory_available = (
        hasattr(resource, "RLIMIT_DATA") if system == "Linux"
        else hasattr(resource, "RLIMIT_AS") or hasattr(resource, "RLIMIT_DATA")
    )
    network_isolated = system in {"Darwin", "Linux"}
    available = bool(
        executable and network_isolated and limits_available and memory_available
    )
    reasons: list[str] = []
    if system not in {"Darwin", "Linux"}:
        reasons.append("unsupported platform")
    if executable is None:
        reasons.append(f"missing {backend_name or 'sandbox backend'}")
    if not limits_available:
        reasons.append("required POSIX resource limits unavailable")
    if not memory_available:
        reasons.append("memory resource limit unavailable")
    if system == "Darwin" and memory_available:
        reasons.append("Darwin address-space limit is best effort")
    return CapabilityReport(
        platform=system,
        backend=executable,
        available=available,
        network_isolated=network_isolated and executable is not None,
        path_contained=available,
        cpu_limited=limits_available,
        memory_limited=memory_available,
        output_bounded=True,
        reason="; ".join(reasons) if reasons else None,
    )


def available() -> bool:
    return capability_report().available


def require_capabilities() -> CapabilityReport:
    report = capability_report()
    if not report.available:
        raise ExecutionPolicyError(
            "model-influenced execution refused: sandbox capability unavailable "
            f"({report.reason or 'unknown reason'})"
        )
    return report


def _contained_root(root: str | os.PathLike[str]) -> Path:
    candidate = Path(root).resolve()
    if not candidate.is_dir() or candidate == Path(candidate.anchor):
        raise ExecutionPolicyError("execution root must be an existing non-root directory")
    return candidate


def _arguments(command: Sequence[object]) -> list[str]:
    if isinstance(command, (str, bytes, bytearray)):
        raise ExecutionPolicyError("execution command must be an argv sequence, not shell text")
    values = [str(item) for item in command]
    if not values or any("\x00" in value for value in values):
        raise ExecutionPolicyError("execution command is empty or contains NUL")
    if len(values) > 128 or sum(len(value) for value in values) > 64 * 1024:
        raise ExecutionPolicyError("execution command exceeds the policy bound")
    return values


# The HOST (never the model) may declare task-environment keys a project's tests need,
# e.g. a stage6 sample project worker sets STATEFRAME_TASK_ENV_KEYS=DATABASE_URL,TEST_DATABASE_URL,...
# 10-03: without this every sample project check died at settings import (pytest exit 4).
TASK_ENV_KEYS_VAR = "STATEFRAME_TASK_ENV_KEYS"
_TASK_ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def _parse_host_task_env_keys(raw: str) -> frozenset[str]:
    keys = {item.strip() for item in raw.split(",") if item.strip()}
    return frozenset(key for key in keys if _TASK_ENV_KEY_RE.match(key))


# These values are host policy, not model input. Capture them before any
# request/turn code can mutate os.environ and use the same snapshot for every
# child in this process.
_HOST_TASK_ENV_KEYS = _parse_host_task_env_keys(os.environ.get(TASK_ENV_KEYS_VAR, ""))
_HOST_TASK_ENV_VALUES = {
    key: os.environ[key] for key in _HOST_TASK_ENV_KEYS if key in os.environ
}


def _host_task_env_keys() -> frozenset[str]:
    return _HOST_TASK_ENV_KEYS


def _safe_environment(env: Mapping[object, object] | None) -> dict[str, str]:
    source = os.environ if env is None else env
    allowed = _ALLOWED_ENV | _host_task_env_keys()
    result = {str(key): str(value) for key, value in source.items() if str(key) in allowed}
    # Declared task keys come from the host process env and are AUTHORITATIVE: they
    # override caller-supplied values. 10-03: run_tests pre-fills DATABASE_URL with a
    # localhost default via setdefault, which silently beat the container's real value
    # and every sample project check still died at import (exit 4). The model cannot set host env.
    result.update(_HOST_TASK_ENV_VALUES)
    return result


def check_cpu_backstop_seconds() -> float:
    """RLIMIT_CPU for a user check: the policy value, else the generous backstop."""
    raw = os.environ.get("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", "").strip()
    try:
        value = float(raw) if raw else CHECK_CPU_BACKSTOP_SECONDS
    except ValueError:
        value = CHECK_CPU_BACKSTOP_SECONDS
    if not math.isfinite(value) or value <= 0:
        value = CHECK_CPU_BACKSTOP_SECONDS
    return min(value, MAX_CPU_SECONDS)


def _limit_value(name: str, requested: float | int, *, maximum: float | int) -> float | int:
    try:
        value = float(requested)
    except (TypeError, ValueError) as error:
        raise ExecutionPolicyError(f"{name} must be finite and positive") from error
    if not math.isfinite(value) or value <= 0 or value > maximum:
        raise ExecutionPolicyError(f"{name} is outside the execution policy bound")
    return value


def _lowered_limits(current: tuple[int, int], limit: int) -> tuple[int, int]:
    """Return (soft, hard) no higher than `limit` and never soft > hard.

    RLIM_INFINITY is -1 on Linux, so a naive min(limit, soft) picks -1 ("unlimited")
    and setrlimit rejects soft=unlimited over a finite hard limit.  10-03: this killed
    every run_tests call in the Linux worker container (macOS reports finite values).
    """
    soft, hard = current
    infinity = resource.RLIM_INFINITY
    bound = limit if hard == infinity else min(limit, hard)
    new_soft = bound if soft == infinity else min(bound, soft)
    return new_soft, bound


def _resource_preexec(cpu_seconds: float, memory_bytes: int,
                      file_bytes: int, nofile: int, nproc: int,
                      cgroup: Path | None = None,
                      memory_report_fd: int | None = None) -> Callable[[], None]:
    cpu = int(math.ceil(_limit_value("cpu_seconds", cpu_seconds, maximum=MAX_CPU_SECONDS)))
    memory = int(_limit_value("memory_bytes", memory_bytes, maximum=4 * 1024 * 1024 * 1024))
    file_limit = int(_limit_value("file_bytes", file_bytes, maximum=64 * 1024 * 1024))
    nofile_limit = int(_limit_value("nofile", nofile, maximum=1024))
    _limit_value("nproc", nproc, maximum=256)

    def _lower_only(which: int, value: int) -> None:
        # A non-root process may only LOWER a hard limit.  Containers often start
        # with hard limits below our policy values; raising them makes setrlimit
        # fail inside preexec_fn, which killed every L82 run at its first check.
        _soft, hard = resource.getrlimit(which)
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        resource.setrlimit(which, (value, value))

    def apply() -> None:
        cgroup_joined = False
        cgroup_memory_configured = _cgroup_memory_limit_configured(cgroup, memory)
        if cgroup is not None:
            try:
                (cgroup / "cgroup.procs").write_text(str(os.getpid()), encoding="ascii")
                cgroup_joined = True
            except OSError:
                pass  # The in-namespace RLIMIT_NPROC is mandatory; cgroup is optional.
        # Soft one second under hard: a runaway dies by SIGXCPU, a signal no other limit
        # sends, so the receipt can name RLIMIT_CPU exactly instead of guessing at a SIGKILL.
        _cpu_soft, cpu_hard = resource.getrlimit(resource.RLIMIT_CPU)
        cpu_hard = cpu if cpu_hard == resource.RLIM_INFINITY else min(cpu, cpu_hard)
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, cpu_hard - 1), cpu_hard))
        _lower_only(resource.RLIMIT_FSIZE, file_limit)
        _lower_only(resource.RLIMIT_CORE, 0)
        resource.setrlimit(resource.RLIMIT_NOFILE,
                           _lowered_limits(resource.getrlimit(resource.RLIMIT_NOFILE), nofile_limit))
        system = platform.system()
        if system == "Linux":
            if cgroup_memory_configured and cgroup_joined:
                memory_mechanism = "cgroup_memory_max"
            elif hasattr(resource, "RLIMIT_DATA"):
                _lower_only(resource.RLIMIT_DATA, memory)
                memory_mechanism = "rlimit_data"
            else:  # Capability discovery should have rejected this before spawn.
                raise RuntimeError("Linux RLIMIT_DATA memory fallback is unavailable")
        elif system == "Darwin" and hasattr(resource, "RLIMIT_AS"):
            try:
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            except (OSError, ValueError):
                if system != "Darwin":
                    raise
            memory_mechanism = "rlimit_as"
        elif hasattr(resource, "RLIMIT_DATA"):
            _lower_only(resource.RLIMIT_DATA, memory)
            memory_mechanism = "rlimit_data"
        else:  # Capability discovery should have rejected this before spawn.
            raise RuntimeError("memory resource limit unavailable")
        if memory_report_fd is not None:
            marker = {"cgroup_memory_max": b"C", "rlimit_data": b"D", "rlimit_as": b"A"}[
                memory_mechanism]
            os.write(memory_report_fd, marker)
            os.close(memory_report_fd)

    return apply


def _process_cpu_seconds(row: ProcessIdentity) -> float | None:
    """User plus system CPU seconds of one live Linux process; None when gone or reused."""
    try:
        raw = (Path("/proc") / str(row.pid) / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        if (int(fields[19]), 0) != tuple(row.start):
            return None
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _note_peak_process_cpu(process: subprocess.Popen[bytes],
                           rows: Sequence[ProcessIdentity]) -> None:
    """Keep the highest per-process CPU time seen; RLIMIT_CPU is a per-process limit."""
    if not sys.platform.startswith("linux"):
        return
    peak = getattr(process, "check_peak_process_cpu_seconds", None)
    for row in rows:
        seconds = _process_cpu_seconds(row)
        if seconds is not None and (peak is None or seconds > peak):
            peak = seconds
    if peak is not None:
        process.check_peak_process_cpu_seconds = peak


def _exited_unit_cpu_seconds(process: subprocess.Popen[bytes]) -> float | None:
    """CPU seconds of an exited, still-unreaped leader plus every descendant it reaped.

    The leader's zombie keeps its own and its waited-for children's times, which covers the
    sandboxed check inside bwrap. Anything unreadable is unknown, never zero.
    """
    if not sys.platform.startswith("linux") or not child_exited_unreaped(process):
        return None
    try:
        raw = (Path("/proc") / str(process.pid) / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        ticks = sum(int(fields[index]) for index in (11, 12, 13, 14))
        return ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _check_cpu_seconds_used(process: subprocess.Popen[bytes]) -> float | None:
    """Peak per-process CPU from supervision; without it, the exited unit's total."""
    peak = getattr(process, "check_peak_process_cpu_seconds", None)
    if isinstance(peak, float):
        return peak
    total = getattr(process, "check_cpu_seconds_used", None)
    return total if isinstance(total, float) else None


def _kill_group(process: subprocess.Popen[bytes], family: Descendants | None = None, *,
                launch_identity: LaunchIdentity | None = None) -> bool:
    # The whole launch unit, not only the group: a bwrap child still setting up its
    # sandbox, or one that already left the group, must not outlive the timeout.
    identity = (launch_identity if launch_identity is not None
                else recorded_launch_identity(process))
    return not kill_launch_unit(process, family, launch_identity=identity)


class _ProgressTrackingProcessTable:
    """Track table operations made directly or through launch-unit cleanup."""

    def __init__(self, supervisor: _ResidentMemorySupervisor, table: ProcessTable):
        object.__setattr__(self, "_supervisor", supervisor)
        object.__setattr__(self, "_table", table)

    def __getattr__(self, name: str) -> object:
        table = object.__getattribute__(self, "_table")
        value = getattr(table, name)
        if not callable(value):
            return value
        operation = value
        supervisor = object.__getattribute__(self, "_supervisor")

        def tracked(*args: object, **kwargs: object) -> object:
            return supervisor._track_work(lambda: operation(*args, **kwargs))

        return tracked

    def __setattr__(self, name: str, value: object) -> None:
        table = object.__getattribute__(self, "_table")
        setattr(table, name, value)


class _ResidentMemorySupervisor:
    """Reuse process-table RSS and page-table accounting for the RLIMIT_DATA fallback."""

    def __init__(self, process: subprocess.Popen[bytes], memory_bytes: int):
        self.process = process
        self.memory_bytes = memory_bytes
        self.table = _ProgressTrackingProcessTable(self, ProcessTable())
        self.launch_identity = recorded_launch_identity(process)
        if self.launch_identity is None:
            raise ExecutionPolicyError("cannot capture launch identity for RSS supervision")
        allow_unstamped_leader = (
            isinstance(self.launch_identity, LaunchIdentity)
            and self.launch_identity.start is None
        )
        if allow_unstamped_leader:
            # sandboxed_popen still owns this unreaped child during supervisor setup.
            self.family = Descendants(
                process.pid, self.table, self.launch_identity,
                allow_unstamped_leader=True,
            )
        else:
            self.family = Descendants(process.pid, self.table, self.launch_identity)
        self.error: str | None = None
        self.finish_error: str | None = None
        self.family_verified_empty = False
        self.cleanup_verified_no_survivors = False
        self.containment_verified = False
        self._error_cause: BaseException | None = None
        self._leader_reaped = False
        self._collector_reap_expected = False
        self._family_lock = threading.Lock()
        self._leader_reaped_event = threading.Event()
        self._next_full_process_scan = time.monotonic()
        self._scan_state_lock = threading.Lock()
        self._scan_started_at: float | None = None
        self._last_scan_duration = 0.0
        self._work_depth = 0
        self._busy_since: float | None = None
        self._last_progress_at = time.monotonic()
        self._stopped = threading.Event()
        self._failed_samples = 0
        self._leader_exit_processed = threading.Event()
        self._completion_lock = threading.Lock()
        self._completion_verified = False
        self._thread = threading.Thread(
            target=self._run, name="sandbox-resident-memory", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def expect_collector_reap(self) -> None:
        """Prefer the collector's exact reap callback over supervisor observation."""
        with self._family_lock:
            self._collector_reap_expected = True

    def notify_leader_reaped(self) -> None:
        """Record the collector's reap once, at its wait site, under the family lock."""
        self._leader_reaped_event.set()
        with self._family_lock:
            if not self._leader_reaped:
                self.family.leader_reaped()
                self._leader_reaped = True

    def _rows_allowed_after_exit(
            self, rows: dict[int, ProcessIdentity]) -> dict[int, ProcessIdentity]:
        """Keep post-exit admission to known identities, their children, and our PID namespace."""
        candidates = dict(rows)
        members = self.family.members
        allowed: set[int] = set()
        for pid, previous in members.items():
            row = candidates.get(pid)
            if row is None:
                identity = getattr(self.table, "identity", None)
                row = identity(pid) if callable(identity) else None
                if row is not None:
                    candidates[pid] = row
            if row is not None and row.start == previous.start:
                allowed.add(pid)

        private_namespace = getattr(self.family, "private_pid_namespace_inode", None)
        namespace_inode = getattr(self.table, "pid_namespace_inode", None)
        if private_namespace is not None and callable(namespace_inode):
            allowed.update(
                pid for pid in candidates
                if namespace_inode(pid) == private_namespace
            )

        # The exited leader's PID may already belong to a new process. Keep its
        # recorded row for identity continuity, but never use it as a new PPID root.
        parents = allowed - {getattr(self.family, "leader", self.process.pid)}
        changed = True
        while changed:
            changed = False
            for pid, row in candidates.items():
                if pid not in allowed and row.ppid in parents:
                    allowed.add(pid)
                    parents.add(pid)
                    changed = True
        return {pid: row for pid, row in candidates.items() if pid in allowed}

    def _update_family(self, rows: dict[int, ProcessIdentity]) -> list[ProcessIdentity]:
        with self._family_lock:
            leader_exited = (
                self.process.returncode is not None or self._leader_reaped_event.is_set())
            # Unit callers and exceptional reaping paths may not have a collector
            # callback. Normal run_sandboxed collection always marks it expected.
            if (self.process.returncode is not None and not self._collector_reap_expected
                    and not self._leader_reaped):
                self.family.leader_reaped()
                self._leader_reaped = True
                self._leader_reaped_event.set()
            update_rows = self._rows_allowed_after_exit(rows) if leader_exited else rows
            return self.family.update(update_rows)

    def _begin_work(self) -> bool:
        if threading.current_thread() is not getattr(self, "_thread", None):
            return False
        now = time.monotonic()
        with self._scan_state_lock:
            if self._work_depth == 0:
                self._busy_since = now
            self._work_depth += 1
            self._last_progress_at = now
        return True

    def _end_work(self, tracked: bool) -> None:
        if not tracked:
            return
        now = time.monotonic()
        with self._scan_state_lock:
            self._work_depth -= 1
            self._last_progress_at = now
            if self._work_depth == 0:
                self._busy_since = None

    def _track_work(self, operation: Callable[[], _Result]) -> _Result:
        tracked = self._begin_work()
        try:
            return operation()
        finally:
            self._end_work(tracked)

    def _finish_diagnostics(
            self, now: float) -> tuple[float, float, float, float, bool]:
        with self._scan_state_lock:
            busy_since = self._busy_since
            last_progress_at = self._last_progress_at
            scan_started_at = self._scan_started_at
            last_scan_duration = self._last_scan_duration
        busy_for = max(0.0, now - busy_since) if busy_since is not None else 0.0
        progress_age = max(0.0, now - last_progress_at)
        scan_active = max(0.0, now - scan_started_at) if scan_started_at is not None else 0.0
        return busy_for, progress_age, scan_active, last_scan_duration, busy_since is not None

    def _finish_failure(self, reason: str) -> bool:
        self.finish_error = reason
        self._record_stop_cause(reason)
        self._stopped.set()
        if self._thread.ident is not None:
            self._thread.join(0)
        return False

    def _finish_verified_result(self) -> bool:
        if self._thread.is_alive():
            return self._finish_failure(
                "resident-memory supervisor thread remained active after post-exit processing")
        if self.error:
            return self._finish_failure(
                f"resident-memory supervisor recorded an error: {self.error}")
        if not self._leader_exit_processed.is_set():
            return self._finish_failure(
                "resident-memory supervisor stopped before post-exit processing completed")
        if not self.containment_verified:
            return self._finish_failure(
                "resident-memory supervisor post-exit processing completed without "
                "verifying containment")
        self.finish_error = None
        return True

    def _record_error(self, error: BaseException, *,
                      context: str = "resident-memory supervision failed") -> None:
        if self._error_cause is not None:
            return
        try:
            message = " ".join(str(error).split())
        except BaseException:
            message = "error message unavailable"
        if len(message) > 500:
            message = message[:497] + "..."
        self.error = f"{context} ({type(error).__name__}: {message})"
        self._error_cause = error
        self._record_stop_cause(self.error)

    def _record_stop_cause(self, reason: str) -> None:
        """Name a supervisor-side stop in the receipt so it is never an anonymous kill."""
        if getattr(self.process, "memory_limit_stop_cause", None) == "memory_limit":
            return
        try:
            self.process.memory_limit_stop_cause = f"supervisor_error: {reason}"
        except (AttributeError, TypeError):
            pass

    def finish(self, timeout: float = 2.0, *, wait_for_post_exit: bool = True) -> bool:
        if not wait_for_post_exit:
            self._stopped.set()
            self._thread.join(max(0.0, timeout))
            return self._finish_verified_result()
        if self.process.returncode is None:
            self._stopped.set()
            self._thread.join(max(0.0, timeout))
            return self._finish_verified_result()

        # Wait for the supervisor's own post-exit scan. There is no wall-clock cap:
        # under memory pressure a scan can be slow, and a slow scan is not a failure.
        # The supervisor loop always ends, either verified or with a recorded error.
        while not self._leader_exit_processed.is_set():
            if not self._thread.is_alive():
                if self._leader_exit_processed.is_set():
                    break
                if self.error:
                    return self._finish_failure(
                        f"resident-memory supervisor recorded an error: {self.error}")
                _busy_for, _progress_age, scan_active, last_scan_duration, _busy = (
                    self._finish_diagnostics(time.monotonic()))
                return self._finish_failure(
                    "resident-memory supervisor thread exited before post-exit verification "
                    f"completed (scan active {scan_active:.2f}s; "
                    f"last completed scan {last_scan_duration:.2f}s)"
                )
            self._leader_exit_processed.wait(0.05)
        self._stopped.set()
        if self._thread.ident is not None:
            self._thread.join()
        return self._finish_verified_result()

    def _snapshot(self) -> dict[int, ProcessIdentity]:
        started = time.monotonic()
        with self._scan_state_lock:
            self._scan_started_at = started
        try:
            return self.table.snapshot()
        finally:
            duration = max(0.0, time.monotonic() - started)
            with self._scan_state_lock:
                self._last_scan_duration = duration
                self._scan_started_at = None

    def _stop_launch_unit(self) -> None:
        tracked = self._begin_work()
        try:
            try:
                supervisor = self
                family = self.family

                class AdmissionFamilyView:
                    @property
                    def table(self):
                        return family.table

                    @property
                    def members(self):
                        return family.members

                    @property
                    def launch_leader_start(self):
                        return getattr(family, "launch_leader_start", None)

                    def update(self, rows):
                        return supervisor._track_work(
                            lambda: supervisor._update_family(rows))

                    def signal(self, rows, sig):
                        return supervisor._track_work(lambda: family.signal(rows, sig))

                survivors = kill_launch_unit(
                    self.process, AdmissionFamilyView(), launch_identity=self.launch_identity)
                if survivors:
                    self.cleanup_verified_no_survivors = False
                    if self.error is None:
                        self.error = "resident-memory supervisor could not stop the full launch unit"
                else:
                    self.cleanup_verified_no_survivors = True
                    self.containment_verified = True
            except (ContainmentUnverified, OSError, RuntimeError, ValueError) as error:
                self._record_error(
                    error, context="resident-memory supervisor could not stop the launch unit")
        finally:
            self._end_work(tracked)

    def _skip_raced_sample(self, error: BaseException) -> bool:
        """Treat a raced or torn procfs sample as unreadable, never as a breach.

        Only a positive over-budget measurement may kill a check. A non-race
        OS error, or PROCFS_SAMPLE_MAX_CONSECUTIVE_FAILURES failed samples in a
        row, is a supervisor error instead.
        """
        if isinstance(error, OSError) and error.errno not in _PROCFS_RACE_ERRNOS:
            return False
        self._failed_samples += 1
        if self._failed_samples > PROCFS_SAMPLE_MAX_CONSECUTIVE_FAILURES:
            return False
        skipped = int(getattr(self.process, "memory_limit_samples_skipped", 0) or 0) + 1
        self.process.memory_limit_samples_skipped = skipped
        self.process.memory_limit_sample_skip_reason = (
            f"procfs sample raced process state ({type(error).__name__})")
        return True

    def _run(self) -> None:
        try:
            while not self._stopped.is_set():
                post_exit_scan_completed = False
                try:
                    refresh_members = getattr(self.table, "refresh_members", None)
                    if callable(refresh_members):
                        rows = refresh_members(
                            self.family.members,
                            namespace_inode=getattr(self.family, "private_pid_namespace_inode", None),
                        )
                    else:  # Compatibility with older process-table test doubles.
                        rows = self._snapshot()
                    leader_exited = (
                        self.process.returncode is not None or self._leader_reaped_event.is_set())
                    # Check known members before a potentially slow full procfs scan so
                    # an already-visible memory breach can stop the launch immediately.
                    live = self._track_work(lambda: self._update_family(rows))
                    _note_peak_process_cpu(self.process, live)
                    breach = self.table.resident_memory_limit_breach(
                        live, self.memory_bytes)
                    now = time.monotonic()
                    if (not breach and platform.system() == "Linux"
                            and (now >= self._next_full_process_scan or leader_exited)):
                        rows.update(self._snapshot())
                        self._next_full_process_scan = (
                            time.monotonic() + FULL_PROCESS_SCAN_INTERVAL_SECONDS)
                        # Recheck the expanded set after the scan finds new processes.
                        live = self._track_work(lambda: self._update_family(rows))
                        breach = self.table.resident_memory_limit_breach(
                            live, self.memory_bytes)
                        post_exit_scan_completed = leader_exited
                except (OSError, RuntimeError, ValueError, IndexError) as error:
                    if self._skip_raced_sample(error):
                        self._stopped.wait(0.02)
                        continue
                    self._record_error(error)
                    self._stop_launch_unit()
                    return
                self._failed_samples = 0
                if breach:
                    self.process.memory_limit_stop_cause = "memory_limit"
                    self.process.memory_limit_killed = True
                    self.process.memory_limit_reason = breach
                    self._stop_launch_unit()
                    return
                if leader_exited:
                    # The bwrap PID namespace kills its members when its init exits.
                    # In the no-namespace fallback, stop every descendant recorded by
                    # the launch-unit scan before the check can report completion.
                    if live:
                        self._stop_launch_unit()
                    else:
                        self.family_verified_empty = post_exit_scan_completed
                        if self.family_verified_empty:
                            self.containment_verified = True
                    return
                self._stopped.wait(0.02)
        except BaseException as error:
            self._record_error(error)
        finally:
            if self.process.returncode is not None:
                self._leader_exit_processed.set()


def _transient_cgroup_cleanup_error(error: OSError) -> bool:
    return error.errno in {
        errno.EBUSY, errno.EAGAIN, errno.EINTR, errno.ENOTEMPTY,
    }


def _remove_command_cgroup(cgroup: Path) -> bool:
    """Remove an empty per-command cgroup, retrying transient cleanup failures."""
    last_error: OSError | UnicodeError | None = None
    last_reason = "unknown error"
    attempts = 0
    for attempt in range(CGROUP_CLEANUP_ATTEMPTS):
        attempts = attempt + 1
        if not cgroup.exists():
            return True
        members: str | None
        try:
            members = (cgroup / "cgroup.procs").read_text(encoding="ascii")
        except OSError as error:
            if error.errno == errno.ENOENT:
                if not cgroup.exists():
                    return True
                # A delegated group can disappear between the stat and read, or
                # its control file can be unavailable while the directory remains.
                members = ""
            else:
                last_error = error
                if not cgroup.exists():
                    return True
                if not _transient_cgroup_cleanup_error(error):
                    break
                members = None
        except UnicodeError as error:
            last_error = error
            break
        if members is not None:
            if members.split():
                last_error = None
                last_reason = "cgroup.procs is not empty"
            else:
                try:
                    cgroup.rmdir()
                    return True
                except OSError as error:
                    if error.errno == errno.ENOENT:
                        return True
                    last_error = error
                    if not cgroup.exists():
                        return True
                    if not _transient_cgroup_cleanup_error(error):
                        break
        if attempt + 1 < CGROUP_CLEANUP_ATTEMPTS:
            time.sleep(CGROUP_CLEANUP_INTERVAL_SECONDS)
    _LOGGER.error(
        "unable to remove per-command cgroup %s after %d attempts (%s)",
        cgroup.name, attempts,
        type(last_error).__name__ if last_error is not None else last_reason,
    )
    return False


def _prepared(command: Sequence[object], root: Path, env: Mapping[object, object] | None,
              internal_env: Mapping[str, object] | None,
              readonly_workspace: bool, nproc: int,
              generic_check: bool = False,
              isolation_builder: Callable[..., tuple[list[str], dict[str, str]]] = isolated_command,
              read_grants_out: list[SandboxReadGrants] | None = None,
              ) -> tuple[list[str], dict[str, str], CapabilityReport]:
    report = require_capabilities()
    arguments = _arguments(command)
    isolation_options = {"readonly_workspace": readonly_workspace}
    if generic_check:
        isolation_options["generic_check"] = True
    if platform.system() == "Linux":
        isolation_options["nproc_budget"] = int(_limit_value("nproc", nproc, maximum=256))
    if read_grants_out is not None:
        isolation_options["read_grants_out"] = read_grants_out
    isolated, clean = isolation_builder(arguments, root, _safe_environment(env),
                                        **isolation_options)
    for key, value in (internal_env or {}).items():
        if not str(key).startswith("STATEFRAME_"):
            raise ExecutionPolicyError("internal execution variables must be STATEFRAME_*")
        clean[str(key)] = str(value)
    return isolated, clean, report


def sandboxed_popen(command: Sequence[object], root: str | os.PathLike[str], *,
                    env: Mapping[object, object] | None = None,
                    internal_env: Mapping[str, object] | None = None,
                    cpu_seconds: float = DEFAULT_CPU_SECONDS,
                    memory_bytes: int = DEFAULT_MEMORY_BYTES,
                    file_bytes: int = DEFAULT_FILE_BYTES,
                    nofile: int = DEFAULT_NOFILE,
                    nproc: int = DEFAULT_NPROC,
                    stdin: object = subprocess.PIPE,
                    stdout: object = subprocess.PIPE,
                    stderr: object = subprocess.PIPE,
                    readonly_workspace: bool = False,
                    generic_check: bool = False,
                    pass_fds: Sequence[int] = (),
                    cgroup: Path | None = None,
                    isolation_builder: Callable[..., tuple[list[str], dict[str, str]]] = isolated_command,
                    read_grants_out: list[SandboxReadGrants] | None = None,
                    popen_factory: Callable[..., subprocess.Popen] | None = None) -> tuple[subprocess.Popen[bytes], CapabilityReport]:
    """Start an argv child behind the required sandbox and resource policy."""
    contained = _contained_root(root)
    command_line, clean, report = _prepared(
        command, contained, env, internal_env, readonly_workspace, nproc,
        generic_check=generic_check,
        isolation_builder=isolation_builder,
        read_grants_out=read_grants_out)
    factory = popen_factory or subprocess.Popen
    memory_reader = memory_writer = None
    report_memory = os.name != "nt" and factory is subprocess.Popen
    inherited_fds = tuple(pass_fds)
    if report_memory:
        try:
            memory_reader, memory_writer = os.pipe()
            os.set_inheritable(memory_writer, True)
            inherited_fds = tuple(dict.fromkeys((*inherited_fds, memory_writer)))
        except OSError as error:
            for fd in (memory_reader, memory_writer):
                if fd is not None:
                    os.close(fd)
            raise ExecutionPolicyError("cannot establish memory limit reporting channel") from error
    try:
        process = factory(
            command_line,
            cwd=str(contained),
            env=clean,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            start_new_session=True,
            pass_fds=inherited_fds,
            preexec_fn=_resource_preexec(
                cpu_seconds, memory_bytes, file_bytes, nofile, nproc, cgroup,
                memory_report_fd=memory_writer)
            if os.name != "nt" else None,
        )
    except BaseException:
        for fd in (memory_reader, memory_writer):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        raise
    if memory_writer is not None:
        os.close(memory_writer)
    capture_launch_identity(process, session_leader=True)
    if memory_reader is not None:
        try:
            marker = os.read(memory_reader, 1)
        finally:
            os.close(memory_reader)
        mechanisms = {b"C": "cgroup_memory_max", b"D": "rlimit_data", b"A": "rlimit_as"}
        mechanism = mechanisms.get(marker)
        if mechanism is None:
            _kill_group(process)
            try:
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise ExecutionPolicyError("sandbox child did not report its memory limit mechanism")
        process.memory_limit_mechanism = mechanism
    if (platform.system() == "Linux"
            and policy_memory_limit_mechanism(process) == "rlimit_data"):
        process.memory_limit_mechanism = "rlimit_data+rss_supervisor"
        process.memory_limit_killed = False
        process.memory_limit_reason = None
        process.memory_limit_samples_skipped = 0
        process.memory_limit_sample_skip_reason = None
        process.memory_limit_stop_cause = None
        try:
            supervisor = _ResidentMemorySupervisor(process, memory_bytes)
            process._stateframe_memory_supervisor = supervisor
            supervisor.start()
        except (ExecutionPolicyError, OSError, RuntimeError, ValueError) as error:
            _kill_group(process)
            try:
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise ExecutionPolicyError(
                "cannot establish resident-memory supervision for RLIMIT_DATA fallback") from error
    return process, report


def finish_supervision(process: subprocess.Popen[bytes], timeout: float = 3.0) -> None:
    """Verify post-exit RSS supervision for a completed direct sandbox launch."""
    supervisor = getattr(process, "_stateframe_memory_supervisor", None)
    if supervisor is None:
        return

    completion_lock = getattr(supervisor, "_completion_lock", None)
    if completion_lock is None:
        completion_lock = threading.Lock()
        try:
            supervisor._completion_lock = completion_lock
        except (AttributeError, TypeError):
            pass
    with completion_lock:
        if getattr(supervisor, "_completion_verified", False):
            return
        prior_failure = getattr(supervisor, "_completion_failure", None)
        if isinstance(prior_failure, ContainmentUnverified):
            raise prior_failure from prior_failure.__cause__

        if getattr(process, "returncode", None) is None:
            reason = "resident-memory supervision completion requested before process exit"
            failure = ContainmentUnverified(f"containment unverified: {reason}")
            raise failure from RuntimeError(reason)

        reap_error: BaseException | None = None
        if not getattr(supervisor, "_leader_reaped", False):
            notify_reaped = getattr(supervisor, "notify_leader_reaped", None)
            if not callable(notify_reaped):
                reap_error = RuntimeError("resident-memory supervisor cannot record leader reap")
            else:
                try:
                    notify_reaped()
                except BaseException as error:
                    reap_error = error

        finish = getattr(supervisor, "finish", None)
        if not callable(finish):
            reason = "resident-memory supervisor has no completion verifier"
            cause = reap_error if reap_error is not None else RuntimeError(reason)
            _raise_supervision_failure(supervisor, reason, cause)
        try:
            stopped = finish(timeout=timeout)
        except BaseException as error:
            try:
                finish(timeout=0.1, wait_for_post_exit=False)
            except BaseException:
                pass
            reason = f"resident-memory supervisor completion raised {type(error).__name__}"
            _raise_supervision_failure(supervisor, reason, error)

        supervisor_error = getattr(supervisor, "error", None)
        if reap_error is not None or not stopped or supervisor_error:
            reason = (str(reap_error) if reap_error is not None else
                      (getattr(supervisor, "finish_error", None) if not stopped else None)
                      or supervisor_error
                      or "resident-memory supervisor did not verify cleanup")
            cause_candidate = reap_error or getattr(supervisor, "_error_cause", None)
            cause = (cause_candidate if isinstance(cause_candidate, BaseException) else None)
            _raise_supervision_failure(supervisor, str(reason), cause)

        supervisor._completion_verified = True


def _raise_supervision_failure(supervisor: object, reason: str,
                               cause: BaseException | None) -> None:
    if not isinstance(cause, BaseException):
        cause = RuntimeError(reason)
    failure = ContainmentUnverified(f"containment unverified: {reason}")
    try:
        setattr(supervisor, "_completion_failure", failure)
    except (AttributeError, TypeError):
        pass
    raise failure from cause


@contextmanager
def supervised_process(process: subprocess.Popen[bytes], timeout: float = 3.0
                       ) -> Iterator[subprocess.Popen[bytes]]:
    """Bound launch cleanup and verify RSS supervision before closing its pipes."""
    def cleanup() -> BaseException | None:
        cleanup_error: BaseException | None = None
        poll = getattr(process, "poll", None)
        try:
            still_running = (poll() is None if callable(poll)
                            else getattr(process, "returncode", None) is None)
        except BaseException as error:
            still_running = True
            cleanup_error = error

        if still_running:
            try:
                supervisor = getattr(process, "_stateframe_memory_supervisor", None)
                family = getattr(supervisor, "family", None)
                identity = getattr(supervisor, "launch_identity", None)
                cleaned = (_kill_group(process, family, launch_identity=identity)
                           if supervisor is not None else _kill_group(process))
                if not cleaned:
                    raise ContainmentUnverified(
                        "containment unverified: launch unit survived context exit")
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error

        # The launch-unit path reaps a killed leader. For an already-exited
        # child, poll above does the same nonblocking observation before finish.
        if callable(poll):
            try:
                poll()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        try:
            finish_supervision(process, timeout=timeout)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error

        for stream in (getattr(process, "stdin", None), getattr(process, "stdout", None),
                       getattr(process, "stderr", None)):
            if stream is not None:
                try:
                    stream.close()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
        wait = getattr(process, "wait", None)
        if callable(wait):
            try:
                wait(timeout=max(0.0, timeout))
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        elif getattr(process, "returncode", None) is None and cleanup_error is None:
            cleanup_error = TypeError("supervised process has no bounded wait method")
        return cleanup_error

    try:
        yield process
    except BaseException as original_error:
        cleanup_error = cleanup()
        if cleanup_error is not None:
            raise original_error from cleanup_error
        raise
    else:
        cleanup_error = cleanup()
        if cleanup_error is not None:
            raise cleanup_error


def _trace_mark(trace: dict[str, int] | None, name: str) -> None:
    if trace is not None and name not in trace:
        trace[name] = time.monotonic_ns()


def _collect(process: subprocess.Popen[bytes], input_data: bytes | None, deadline: float,
             output_limit: int, *, observer_fd: int | None = None,
             observer_buffer: bytearray | None = None,
             trace: dict[str, int] | None = None,
             on_leader_reaped: Callable[[], None] | None = None
             ) -> tuple[str, str, bool, bool, bool]:
    if input_data is not None and len(input_data) > MAX_INPUT_BYTES:
        raise ExecutionPolicyError("child input exceeds the policy bound")
    selector = selectors.DefaultSelector()
    streams: dict[int, str] = {}
    handles: list[object] = []
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    truncated = False
    input_offset = 0
    group_killed = False
    if process.stdin is not None:
        handles.append(process.stdin)
        os.set_blocking(process.stdin.fileno(), False)
        if input_data:
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        else:
            process.stdin.close()
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        if stream is None:
            continue
        handles.append(stream)
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
        streams[stream.fileno()] = name
    if observer_fd is not None and observer_buffer is not None:
        try:
            os.set_blocking(observer_fd, False)
            selector.register(observer_fd, selectors.EVENT_READ, "observer")
        except (OSError, ValueError, KeyError):  # telemetry channel failure is fail-open
            observer_fd = None
            observer_buffer = None
    _trace_mark(trace, "pipe_drain_start_ns")
    timed_out = False
    while selector.get_map():
        if trace is not None and child_exited_unreaped(process):
            _trace_mark(trace, "child_exit_observed_ns")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = not child_exited_unreaped(process)
            if timed_out:
                group_killed = _kill_group(process)
            break
        for key, mask in selector.select(min(remaining, 0.05)):
            stream = key.fileobj
            kind = key.data
            if kind == "stdin":
                broken_pipe = False
                try:
                    written = os.write(stream.fileno(), input_data[input_offset:input_offset + 8192])
                    input_offset += written
                except BlockingIOError:
                    written = 0
                except OSError:
                    broken_pipe = True
                if broken_pipe or input_data is None or input_offset >= len(input_data):
                    selector.unregister(stream)
                    stream.close()
                continue
            try:
                stream_fd = stream if isinstance(stream, int) else stream.fileno()
                data = os.read(stream_fd, 8192)
            except (BlockingIOError, OSError):
                data = b""
            if not data:
                if kind in {"stdout", "stderr"}:
                    _trace_mark(trace, f"{kind}_eof_ns")
                selector.unregister(stream)
                continue
            if kind == "observer":
                if observer_buffer is not None:
                    available = max(0, MAX_PROFILE_OUTPUT_BYTES - len(observer_buffer))
                    if available:
                        observer_buffer.extend(data[:available])
                    if len(data) > available and trace is not None:
                        trace["profile_report_truncated"] = 1
                continue
            available = max(0, output_limit - total)
            if available:
                buffers[kind].extend(data[:available])
            if len(data) > available:
                truncated = True
            total += len(data)
        if trace is not None and child_exited_unreaped(process):
            _trace_mark(trace, "child_exit_observed_ns")
            # The main runner owns the profile writer. Drain all currently available
            # bytes, then stop observing this private channel so a test-spawned child
            # cannot extend the host wait by inheriting the descriptor.
            if observer_fd is not None and observer_buffer is not None:
                try:
                    observer_key = selector.get_key(observer_fd)
                except KeyError:
                    observer_key = None
                if observer_key is not None:
                    while True:
                        try:
                            data = os.read(observer_fd, 8192)
                        except (BlockingIOError, OSError):
                            data = b""
                        if not data:
                            break
                        available = max(0, MAX_PROFILE_OUTPUT_BYTES - len(observer_buffer))
                        if available:
                            observer_buffer.extend(data[:available])
                        if len(data) > available and trace is not None:
                            trace["profile_report_truncated"] = 1
                    try:
                        selector.unregister(observer_fd)
                    except (KeyError, OSError):
                        pass
    for key in list(selector.get_map().values()):
        try:
            selector.unregister(key.fileobj)
        except (KeyError, OSError):
            pass
    for handle in handles:
        try:
            handle.close()
        except OSError:
            pass
    selector.close()
    _trace_mark(trace, "pipe_drain_finished_ns")
    # Stop group members before wait() releases the leader's PID.
    if not timed_out:
        until = max(time.monotonic() + 1.0, deadline + 1.0)
        while not child_exited_unreaped(process) and time.monotonic() < until:
            time.sleep(.02)
        if not child_exited_unreaped(process):
            timed_out = True
            group_killed = _kill_group(process) or group_killed
        else:
            # The exited leader remains waitable and pins its group identity.
            # A launch-unit sweep cannot discover its now-reparented family.
            identity = recorded_launch_identity(process)
            if identity is None or identity.pid != process.pid:
                raise ContainmentUnverified("containment unverified: completed check launch identity missing")
            try:
                signalled = signal_launch_group(process, identity, signal.SIGKILL)
            except PermissionError:
                signalled = False
            table = ProcessTable()
            sweep_deadline = time.monotonic() + 2.0
            while any(row.pgid == identity.pid and not row.zombie
                      for row in table.snapshot().values()):
                if not signalled or time.monotonic() >= sweep_deadline:
                    raise ContainmentUnverified(
                        "containment unverified: cannot sweep completed check process group")
                time.sleep(.02)
            group_killed = True
    process.check_cpu_seconds_used = _exited_unit_cpu_seconds(process)
    process.wait(timeout=1)
    if on_leader_reaped is not None:
        on_leader_reaped()
    if trace is not None:
        _trace_mark(trace, "child_exit_observed_ns")
    return (
        bytes(buffers["stdout"]).decode("utf-8", errors="replace"),
        bytes(buffers["stderr"]).decode("utf-8", errors="replace"),
        timed_out,
        truncated,
        group_killed,
    )


def run_sandboxed(command: Sequence[object], root: str | os.PathLike[str], *,
                  env: Mapping[object, object] | None = None,
                  internal_env: Mapping[str, object] | None = None,
                  input_data: str | bytes | None = None,
                  timeout: float = MAX_WALL_SECONDS,
                  output_limit: int = MAX_OUTPUT_BYTES,
                  readonly_workspace: bool = False,
                  generic_check: bool = False,
                  cpu_seconds: float = DEFAULT_CPU_SECONDS,
                  memory_bytes: int = DEFAULT_MEMORY_BYTES,
                  file_bytes: int = DEFAULT_FILE_BYTES,
                  nofile: int = DEFAULT_NOFILE,
                  nproc: int = DEFAULT_NPROC,
                  observe_pytest: bool = False) -> PolicyResult:
    """Run one bounded child; every failure is a refusal or typed result."""
    timeout = float(_limit_value("timeout", timeout, maximum=MAX_WALL_SECONDS))
    output_limit = int(_limit_value("output_limit", output_limit, maximum=4 * 1024 * 1024))
    nproc_budget = (int(_limit_value("nproc", nproc, maximum=256))
                    if platform.system() == "Linux" else None)
    payload = input_data.encode("utf-8") if isinstance(input_data, str) else input_data
    trace: dict[str, int] | None = {} if observe_pytest else None
    profile_reader: int | None = None
    profile_writer: int | None = None
    profile_buffer: bytearray | None = None
    child_internal_env = dict(internal_env or {})
    pass_fds: tuple[int, ...] = ()
    if observe_pytest and _PROFILE_FD_ENV not in child_internal_env:
        try:
            profile_reader, profile_writer = os.pipe()
            os.set_inheritable(profile_writer, True)
            child_internal_env[_PROFILE_FD_ENV] = str(profile_writer)
            pass_fds = (profile_writer,)
            profile_buffer = bytearray()
        except OSError:
            for fd in (profile_reader, profile_writer):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            profile_reader = profile_writer = None
            profile_buffer = None
            pass_fds = ()
            child_internal_env.pop(_PROFILE_FD_ENV, None)
    if trace is not None:
        trace["sandbox_spawn_start_ns"] = time.monotonic_ns()
    cgroup = (_command_cgroup(uuid.uuid4().hex, nproc_budget, memory_bytes=memory_bytes)
              if nproc_budget is not None else None)
    child_reaped = False
    read_grants: list[SandboxReadGrants] = []
    try:
        spawn_options = {
            "env": env, "internal_env": child_internal_env,
            "readonly_workspace": readonly_workspace,
            "cpu_seconds": cpu_seconds, "memory_bytes": memory_bytes,
            "file_bytes": file_bytes, "nofile": nofile, "nproc": nproc,
        }
        if generic_check:
            spawn_options["generic_check"] = True
        if pass_fds:
            spawn_options["pass_fds"] = pass_fds
        if cgroup is not None:
            spawn_options["cgroup"] = cgroup
        if platform.system() == "Darwin":
            spawn_options["read_grants_out"] = read_grants
        process, report = sandboxed_popen(command, root, **spawn_options)
    except BaseException:
        for fd in (profile_reader, profile_writer):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if cgroup is not None:
            _remove_command_cgroup(cgroup)
        raise
    if trace is not None:
        trace["sandbox_spawn_complete_ns"] = time.monotonic_ns()
    if profile_writer is not None:
        try:
            os.close(profile_writer)
        except OSError:
            pass
        profile_writer = None
    try:
        deadline = time.monotonic() + timeout
        memory_supervisor = getattr(process, "_stateframe_memory_supervisor", None)
        on_leader_reaped = getattr(memory_supervisor, "notify_leader_reaped", None)
        if callable(on_leader_reaped):
            expect_reap = getattr(memory_supervisor, "expect_collector_reap", None)
            if callable(expect_reap):
                expect_reap()
        collect_options = ({"on_leader_reaped": on_leader_reaped}
                           if callable(on_leader_reaped) else {})
        if trace is None:
            stdout, stderr, timed_out, truncated, group_killed = _collect(
                process, payload, deadline, output_limit, **collect_options)
        else:
            stdout, stderr, timed_out, truncated, group_killed = _collect(
                process, payload, deadline, output_limit,
                observer_fd=profile_reader, observer_buffer=profile_buffer, trace=trace,
                **collect_options)
        finish_supervision(process, timeout=3.0)
        child_reaped = True
    except BaseException as original_error:
        memory_supervisor = getattr(process, "_stateframe_memory_supervisor", None)
        if memory_supervisor is not None:
            memory_supervisor.finish(timeout=0.1, wait_for_post_exit=False)
        cleanup_error: BaseException | None = None
        if not (memory_supervisor is not None and memory_supervisor.containment_verified):
            try:
                if memory_supervisor is None:
                    cleaned = _kill_group(process)
                else:
                    cleaned = _kill_group(
                        process, memory_supervisor.family,
                        launch_identity=memory_supervisor.launch_identity,
                    )
                if not cleaned:
                    cleanup_error = ContainmentUnverified(
                        "containment unverified: launch-unit survivors remain after cleanup")
            except BaseException as error:
                cleanup_error = error
        try:
            process.wait(timeout=2)
            child_reaped = True
        except (OSError, subprocess.TimeoutExpired):
            child_reaped = process.returncode is not None
        else:
            if memory_supervisor is not None:
                on_leader_reaped = getattr(memory_supervisor, "notify_leader_reaped", None)
                if callable(on_leader_reaped):
                    on_leader_reaped()
        if cleanup_error is not None:
            raise cleanup_error from original_error
        raise
    finally:
        if profile_reader is not None:
            try:
                os.close(profile_reader)
            except OSError:
                pass
        if cgroup is not None:
            if child_reaped:
                if not _remove_command_cgroup(cgroup):
                    raise ContainmentUnverified(
                        "containment unverified: per-command cgroup cleanup failed")
            else:
                _LOGGER.error(
                    "retaining per-command cgroup %s because its child was not reaped",
                    cgroup.name,
                )
    profile_payload = bytes(profile_buffer) if profile_buffer else None
    return PolicyResult(
        completed=subprocess.CompletedProcess(
            _arguments(command), process.returncode, stdout, stderr),
        timed_out=timed_out,
        process_group_killed=group_killed,
        output_truncated=truncated,
        capabilities=report,
        pid=process.pid if isinstance(process.pid, int) else None,
        execution_trace=trace or {},
        profile_payload=profile_payload,
        read_grants=read_grants[0] if read_grants else None,
        memory_limit_mechanism=policy_memory_limit_mechanism(process),
        memory_limit_killed=bool(getattr(process, "memory_limit_killed", False)),
        memory_limit_reason=getattr(process, "memory_limit_reason", None),
        cgroup_leaf_pattern=_cgroup_leaf_pattern_status(),
        memory_limit_samples_skipped=int(getattr(process, "memory_limit_samples_skipped", 0) or 0),
        memory_limit_sample_skip_reason=getattr(process, "memory_limit_sample_skip_reason", None),
        memory_limit_stop_cause=getattr(process, "memory_limit_stop_cause", None),
        cpu_limit_seconds=float(math.ceil(cpu_seconds)),
        cpu_seconds_used=_check_cpu_seconds_used(process),
    )


__all__ = [
    "CHECK_CPU_BACKSTOP_SECONDS", "CapabilityReport", "DEFAULT_CPU_SECONDS", "DEFAULT_FILE_BYTES",
    "DEFAULT_MEMORY_BYTES", "ExecutionPolicyError", "MAX_OUTPUT_BYTES",
    "PolicyResult", "available", "capability_report", "policy_memory_limit_mechanism",
    "check_cpu_backstop_seconds", "finish_supervision", "policy_read_grants", "require_capabilities",
    "run_sandboxed", "sandboxed_popen", "supervised_process",
]
