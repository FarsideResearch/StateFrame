"""Host-observed check progress and conservative inactivity evidence.

A clock schedules observations; it never supplies progress. Unknown telemetry or
an observation gap invalidates the entire unchanged-evidence window.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import ctypes
import math
import os
import sys
from types import MappingProxyType

try:
    from .kernel_exec_tracker import (Descendants, ProcessIdentity, ProcessTable, _TaskInfo,
                                      recorded_launch_identity)
except ImportError:
    from kernel_exec_tracker import (Descendants, ProcessIdentity, ProcessTable, _TaskInfo,
                                     recorded_launch_identity)

SAMPLE_SECONDS = 0.5
MAX_SAMPLE_GAP = 1.5
DEFAULT_STALL_SECONDS = 300.0
TELEMETRY_GRACE_SECONDS = 8.0
STALL_ENV = "STATEFRAME_CHECK_STALL_SECONDS"
CPU_ENV = "STATEFRAME_PUBLIC_CHECK_CPU_SECONDS"
Identity = tuple[int, tuple[int, int]]
EventSink = Callable[[str, dict[str, object]], None]


def _diagnostic_event(name: str, fields: dict[str, object]) -> None:
    # Journal positions carry request identity. The executed check and its
    # receipt carry terminal evidence; timing samples are host diagnostics.
    try:
        from .kernel_reason_log import emit_reason
    except ImportError:
        from kernel_reason_log import emit_reason
    stopped = name == "execution_stopped"
    emit_reason("degrade", name, "check_watchdog", "check supervision",
                str(fields["cause" if stopped else "evidence_kind"]),
                level="WARN" if stopped else "DEBUG", journal=False)


@dataclass(frozen=True)
class CheckWatchdogConfig:
    stall_seconds: float = DEFAULT_STALL_SECONDS
    on_event: EventSink = field(default=_diagnostic_event, compare=False, repr=False)
    on_progress: Callable[[int, str], None] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if (isinstance(self.stall_seconds, bool)
                or not isinstance(self.stall_seconds, (int, float))
                or not math.isfinite(self.stall_seconds) or self.stall_seconds <= 0):
            raise ValueError("check inactivity window must be finite and positive")


def host_check_cpu_seconds(environ: Mapping[str, str] | None = None) -> float | None:
    """Parse the optional host cap without importing the execution policy."""
    source = os.environ if environ is None else environ
    try:
        value = float(source.get(CPU_ENV, "0"))
    except (ValueError, TypeError) as error:
        raise ValueError("check CPU cap must be finite and nonnegative") from error
    if not math.isfinite(value) or not 0 <= value <= 86400:
        raise ValueError("check CPU cap is outside the execution policy bound")
    return value if value else None


def host_check_stall_seconds(environ: Mapping[str, str] | None = None) -> float:
    source = os.environ if environ is None else environ
    try:
        value = float(source.get(STALL_ENV, str(DEFAULT_STALL_SECONDS)))
    except (ValueError, TypeError) as error:
        raise ValueError("check inactivity window must be finite and positive") from error
    return CheckWatchdogConfig(value).stall_seconds


def check_watchdog_config(environ: Mapping[str, str] | None = None) -> CheckWatchdogConfig:
    """Scale the selected inactivity window for current host load."""
    value = host_check_stall_seconds(environ)
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError):
        load = 0.0
    cpus = max(1, os.cpu_count() or 1)
    scale = min(8.0, max(1.0, load / cpus)) if math.isfinite(load) else 1.0
    return CheckWatchdogConfig(value * scale)


@dataclass(frozen=True)
class ProgressSnapshot:
    cpu: Mapping[Identity, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cpu", MappingProxyType(dict(self.cpu)))


def process_snapshot(table: ProcessTable, rows: Sequence[ProcessIdentity], *,
                     previous: ProgressSnapshot | None = None) -> ProgressSnapshot | None:
    """Require every identity-checked CPU reading; a partial sample is unknown."""
    cpu: dict[Identity, int] = {}
    try:
        for row in rows:
            if row.start is None:
                return None
            ticks = table.cpu_ticks(row)
            if ticks is None and sys.platform == "darwin":
                # libproc task CPU values are cumulative nanoseconds. Validate
                # identity on both sides of the read, as procfs cpu_ticks does.
                current = table.identity(row.pid)
                lib = getattr(table, "_lib", None)
                if current is None or current.start != row.start or lib is None:
                    return None
                info = _TaskInfo()
                count = lib.proc_pidinfo(row.pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
                after = table.identity(row.pid)
                if count != ctypes.sizeof(info) or after is None or after.start != row.start:
                    return None
                ticks = info.user + info.system
            if type(ticks) is not int or ticks < 0:
                return None
            cpu[(row.pid, row.start)] = ticks
        for pid, start in (previous.cpu.keys() - cpu.keys() if previous is not None else ()):
            current = table.identity(pid)
            if current is not None:
                if current.start != start or not current.zombie:
                    return None
            else:
                # identity(None) can mean a partial read, not only exit. A
                # successful signal-0 probe proves the missing row is unknown.
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    return None
    except (OSError, RuntimeError, ValueError, IndexError):
        return None
    return ProgressSnapshot(cpu) if cpu else None


def progress_sample(table: ProcessTable, live: Sequence[ProcessIdentity],
                    previous: ProgressSnapshot | None) -> tuple[ProgressSnapshot | None, ProgressSnapshot | None]:
    current = process_snapshot(table, live, previous=previous)
    if current is not None:
        return current, current
    if previous is not None:
        try:
            # Only positively identified PID reuse breaks the old baseline.
            # A still-live member with a missing reading remains required.
            live_pids = {row.pid for row in live}
            for pid, start in previous.cpu:
                if pid not in live_pids:
                    identity = table.identity(pid)
                    if identity is not None and identity.start != start:
                        return None, None
        except (OSError, RuntimeError, ValueError, IndexError):
            pass
    return None, previous


class LaunchProgressSampler:
    """Parent-side launch-family sampler; one refresh per bounded observation."""

    def __init__(self, process: object):
        self.process = process
        self.table = ProcessTable()
        self.identity = recorded_launch_identity(process)
        self.family = Descendants(process.pid, self.table, self.identity)
        self.previous: ProgressSnapshot | None = None

    def snapshot(self) -> ProgressSnapshot | None:
        try:
            if self.identity is None:
                return None
            if self.identity.start is None:
                lock = getattr(self.process, "_waitpid_lock", None)
                if lock is None:
                    return None
                with lock:
                    if self.process.returncode is not None:
                        return None
                    root = self.table.identity(self.process.pid)
                    if (root is None or root.pgid != self.identity.pid
                            or root.sid != self.identity.pid):
                        return None
                    # The wait lock and unreaped child pin the launch PID.
                    self.identity = replace(self.identity, start=root.start)
                    self.family.launch_leader_start = root.start
                    self.family.members[root.pid] = root
            root = self.table.identity(self.process.pid)
            if root is None or root.start != self.identity.start:
                return None
            rows = self.table.refresh_members(self.family.members)
            live = self.family.update(rows)
            current, self.previous = progress_sample(self.table, live, self.previous)
            return current
        except (OSError, RuntimeError, ValueError, IndexError):
            return None


class CheckWatchdog:
    def __init__(self, config: CheckWatchdogConfig):
        self.config = config
        self.progress_sequence = 0
        self.last_progress_kind: str | None = None
        self.cpu_ticks = 0
        self._previous: dict[Identity, int] | None = None
        self._output = 0
        self._work = 0
        self._sample_at: float | None = None
        self._unchanged_since: float | None = None
        self.stop_cause: str | None = None
        self._terminal_emitted = False
        self._unsupervised_since: float | None = None

    def _emit(self, name: str, fields: dict[str, object]) -> None:
        try:
            self.config.on_event(name, fields)
        except Exception:
            # A diagnostic sink cannot interrupt containment or alter a verdict.
            pass

    def observe(self, now: float, current: ProgressSnapshot | None, *,
                output_bytes: int = 0, completed_work: int = 0) -> bool:
        if self.stop_cause is not None:
            return True
        previous_time = self._sample_at
        if previous_time is not None and now - previous_time < SAMPLE_SECONDS:
            return False
        self._sample_at = now
        continuous = previous_time is not None and 0 <= now - previous_time <= MAX_SAMPLE_GAP
        kind = None
        if output_bytes > self._output:
            kind = "output"
        if completed_work > self._work:
            kind = "completed_work"
        self._output = max(self._output, output_bytes)
        self._work = max(self._work, completed_work)
        previous = self._previous
        valid = current is not None and bool(current.cpu)
        if valid:
            values = dict(current.cpu)
            # Same PID with a different start identity, or a regressed CPU
            # counter, breaks evidence. Neither manufactures activity.
            old_pids = {pid: start for pid, start in (previous or {})}
            discontinuity = any(
                (pid in old_pids and old_pids[pid] != start)
                or (identity in (previous or {}) and ticks < previous[identity])
                for identity, ticks in values.items() for pid, start in (identity,))
            if discontinuity:
                valid = False
                self._previous = None
            else:
                # Accumulate positive deltas by identity, never subtract CPU of
                # exiting children. Storage is bounded by the active family.
                delta = sum(max(0, ticks - (previous or {}).get(identity, ticks))
                            for identity, ticks in values.items())
                self.cpu_ticks += delta
                if delta:
                    kind = kind or "cpu"
                if previous is not None and values.keys() != previous.keys():
                    kind = kind or "members"
                self._previous = values
        else:
            self._previous = None
        if valid or kind is not None:
            self._unsupervised_since = None
        elif self._unsupervised_since is None:
            self._unsupervised_since = now
        elif now - self._unsupervised_since >= TELEMETRY_GRACE_SECONDS:
            self.stop_cause = "telemetry_unavailable"
            return True
        if kind is not None:
            self.progress_sequence += 1
            self.last_progress_kind = kind
            # Coalesce diagnostics over increasingly long healthy executions.
            if self.progress_sequence & (self.progress_sequence - 1) == 0:
                self._emit("execution_progress", {"progress_sequence": self.progress_sequence,
                           "scope": "check", "evidence_kind": kind})
        if kind is not None and self.config.on_progress is not None:
            try:
                self.config.on_progress(self.progress_sequence, kind)
            except Exception:
                pass
        if previous_time is not None and not continuous:
            self._previous = None
            self._unchanged_since = None
            return False
        if not valid or previous is None or kind is not None:
            self._unchanged_since = now if valid else None
            return False
        if self._unchanged_since is None:
            self._unchanged_since = now
        if now - self._unchanged_since >= self.config.stall_seconds:
            self.stop_cause = "no_progress"
        return self.stop_cause is not None

    def stop_payload(self, *, cleanup_ok: bool, cause: str = "no_progress") -> dict[str, object]:
        return {"schema": "stateframe-execution-stop/1", "cause": cause, "scope": "check",
                "resource": None, "requested": None, "effective": None, "observed": None,
                "mechanism": "host_progress_watchdog", "evidence": "observed",
                "inactivity_seconds": self.config.stall_seconds if cause == "no_progress" else None,
                "progress_sequence": self.progress_sequence,
                "last_progress_kind": self.last_progress_kind, "cleanup_ok": cleanup_ok}

    def stopped(self, *, cleanup_ok: bool, cause: str = "no_progress") -> dict[str, object]:
        payload = self.stop_payload(cleanup_ok=cleanup_ok, cause=cause)
        if not self._terminal_emitted:
            self._terminal_emitted = True
            self._emit("execution_stopped", payload.copy())
        return payload


def execution_stop_outcome(stop: Mapping[str, object]) -> str:
    cause = stop["cause"]
    if cause == "no_progress":
        return "NO_PROGRESS"
    if cause == "resource_limit":
        return "RESOURCE_LIMIT"
    return "INCOMPLETE"


def validate_execution_stop(value: object) -> dict[str, object] | None:
    """Detach and validate additive terminal metadata; legacy absence stays None."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("schema") != "stateframe-execution-stop/1":
        raise ValueError("invalid execution stop schema")
    allowed = {"schema", "cause", "scope", "resource", "requested", "effective", "observed",
               "mechanism", "evidence", "inactivity_seconds", "progress_sequence",
               "last_progress_kind", "cleanup_ok"}
    if set(value) != allowed:
        raise ValueError("invalid execution stop fields")
    cause = value["cause"]
    if not isinstance(cause, str) or cause not in {
            "no_progress", "resource_limit", "user_deadline", "cancelled",
            "external_signal", "supervision_error", "telemetry_unavailable"}:
        raise ValueError("invalid execution stop cause")
    scope = value["scope"]
    if not isinstance(scope, str) or scope not in {"check", "shell", "job"}:
        raise ValueError("invalid execution stop scope")
    resource = value["resource"]
    if ((resource is not None and not isinstance(resource, str))
            or resource not in {None, "memory_bytes", "processes", "file_bytes",
                                "open_files", "cpu_seconds", "output_bytes"}):
        raise ValueError("invalid execution stop resource")
    for key in ("requested", "effective", "observed", "inactivity_seconds"):
        number = value[key]
        if number is not None and (type(number) not in {int, float}
                                   or not math.isfinite(number) or number < 0):
            raise ValueError("invalid execution stop number")
    if type(value["progress_sequence"]) is not int or value["progress_sequence"] < 0:
        raise ValueError("invalid execution stop sequence")
    if (type(value["cleanup_ok"]) is not bool or not isinstance(value["evidence"], str)
            or value["evidence"] not in {"observed", "inherited", "unknown"}):
        raise ValueError("invalid execution stop evidence")
    if not isinstance(value["mechanism"], str) or not 1 <= len(value["mechanism"]) <= 128:
        raise ValueError("invalid execution stop mechanism")
    kind = value["last_progress_kind"]
    if kind is not None and (not isinstance(kind, str) or not 1 <= len(kind) <= 64):
        raise ValueError("invalid execution stop progress kind")
    if value["cause"] == "no_progress" and (value["resource"] is not None
                                             or not value["inactivity_seconds"]):
        raise ValueError("invalid no-progress stop")
    return dict(value)


class IncompleteCheck(str):
    """String-compatible frozen-check observation with private terminal evidence."""

    def __new__(cls, text: str, execution_stop: Mapping[str, object], *, exit_code: int | None = None):
        instance = super().__new__(cls, text)
        instance.execution_stop = validate_execution_stop(execution_stop)
        instance.exit_code = exit_code
        return instance
