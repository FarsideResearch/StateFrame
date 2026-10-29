"""Host-owned background jobs and bounded job-watch projections.

The organ is deliberately inert unless ``STATEFRAME_JOB_WATCH=on``.  A job is
started only by the model's typed ``launch_job`` call; overdue and stalled are
observations, never automatic cancellation decisions.
"""
from __future__ import annotations

import copy
import json
import math
import os
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


MODE_ENV = "STATEFRAME_JOB_WATCH"
MAX_CONCURRENT_JOBS = 2
DEFAULT_MAX_TOTAL_JOB_WALL_S = 900.0
DEFAULT_TAIL_LINES = 40
MAX_TAIL_LINES = 500
MAX_COMMAND_CHARS = 16_384
MAX_DONE_SIGNAL_CHARS = 256
_TAIL_BUFFER_LINES = 4096


class JobWatchError(ValueError):
    """A typed job operation was rejected without affecting another job."""

    def __init__(self, message: str, *, code: str = "JOB_WATCH_REJECTED") -> None:
        super().__init__(message)
        self.code = code


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    if mode() != "on":
        return False
    try:
        from kernel_execution_policy import available
        return available()
    except (ImportError, RuntimeError):
        return False


def _max_total_wall_from_env() -> float:
    raw = os.environ.get("STATEFRAME_JOB_WATCH_MAX_WALL_S", str(DEFAULT_MAX_TOTAL_JOB_WALL_S))
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("STATEFRAME_JOB_WATCH_MAX_WALL_S must be finite and positive") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("STATEFRAME_JOB_WATCH_MAX_WALL_S must be finite and positive")
    return value


def configuration() -> dict[str, object]:
    """Return the public limits for the active organ configuration."""
    from kernel_execution_policy import capability_report
    return {
        "schema": "stateframe-job-watch-config/1",
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "max_total_job_wall_s": _max_total_wall_from_env(),
        "host_kills_automatically": False,
        "execution_policy": capability_report().as_dict(),
    }


def _why_property() -> dict[str, str]:
    return {
        "type": "string",
        "description": "one line: why this job operation is the next useful observation",
    }


def _tool(name: str, description: str, properties: Mapping[str, object],
          required: Sequence[str] = ()) -> dict[str, object]:
    values = {key: copy.deepcopy(value) for key, value in properties.items()}
    values.setdefault("why", _why_property())
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": values,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


def _job_tools() -> list[dict[str, object]]:
    return [
        _tool(
            "launch_job",
            "Run one long workspace command in the background. The host observes it "
            "but never cancels it automatically.",
            {
                "cmd": {"type": "string", "description": "workspace command to run"},
                "expected_s": {
                    "type": "number", "exclusiveMinimum": 0,
                    "description": "expected wall time in seconds",
                },
                "done_signal": {
                    "type": "string", "description": "output text that marks the expected completion signal",
                },
            },
            ("cmd", "expected_s", "done_signal"),
        ),
        _tool(
            "job_status",
            "Return host-computed state, elapsed time, output growth, and overdue/stalled flags.",
            {"job_id": {"type": "string"}},
            ("job_id",),
        ),
        _tool(
            "job_output",
            "Return the bounded tail of one background job's captured output.",
            {
                "job_id": {"type": "string"},
                "tail_lines": {
                    "type": "integer", "minimum": 1, "maximum": MAX_TAIL_LINES,
                    "default": DEFAULT_TAIL_LINES,
                },
            },
            ("job_id",),
        ),
        _tool(
            "job_kill",
            "Stop one background job because you decided it should no longer run.",
            {"job_id": {"type": "string"}},
            ("job_id",),
        ),
    ]


def tool_surface(tools: object) -> object:
    """Append typed job operations only in the explicit ON arm.

    The OFF path returns the exact object it received.  This is important for
    the legacy request/menu byte-parity contract.
    """
    if not enabled():
        return tools
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        raise TypeError("job tool surface requires a sequence of tool definitions")
    existing = {
        str(tool.get("function", {}).get("name"))
        for tool in tools if isinstance(tool, Mapping)
    }
    return [*tools, *[
        tool for tool in _job_tools()
        if tool["function"]["name"] not in existing
    ]]


@dataclass
class JobRecord:
    """The serializable host state for one background process."""

    job_id: str
    started: float
    elapsed: float = 0.0
    last_output_at: float | None = None
    output_bytes: int = 0
    bytes_growth_last_interval: int = 0
    exit_code: int | None = None
    state: str = "running"
    done_signal_seen: bool = False
    expected_s: float = 0.0
    done_signal: str = ""
    overdue: bool = False
    stalled: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-job/1",
            "job_id": self.job_id,
            "started": self.started,
            "elapsed": self.elapsed,
            "expected_s": self.expected_s,
            "done_signal": self.done_signal,
            "last_output_at": self.last_output_at,
            "last_output_age_s": None,
            "output_bytes": self.output_bytes,
            "bytes_growth_last_interval": self.bytes_growth_last_interval,
            "exit_code": self.exit_code,
            "state": self.state,
            "alive": self.state == "running",
            "done_signal_seen": self.done_signal_seen,
            "overdue": self.overdue,
            "stalled": self.stalled,
        }


@dataclass
class _ManagedJob:
    record: JobRecord
    process: subprocess.Popen[bytes]
    started_mono: float
    done_signal_bytes: bytes
    clock: Callable[[], float]
    reader: threading.Thread | None = None
    last_output_mono: float | None = None
    output_sample_bytes: int = 0
    signal_tail: bytes = b""
    tail_lines: deque[str] = field(
        default_factory=lambda: deque(maxlen=_TAIL_BUFFER_LINES))
    partial_line: str = ""
    reader_done: bool = False
    kill_requested: bool = False
    completion_journaled: bool = False


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise JobWatchError(f"{label} must be a finite positive number", code="INVALID_ARGUMENT")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise JobWatchError(f"{label} must be a finite positive number", code="INVALID_ARGUMENT") from error
    if not math.isfinite(converted) or converted <= 0:
        raise JobWatchError(f"{label} must be a finite positive number", code="INVALID_ARGUMENT")
    return converted


def _required_text(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise JobWatchError(f"{label} must be a non-empty string of at most {limit} characters",
                            code="INVALID_ARGUMENT")
    return value


def _duration(seconds: object) -> str:
    try:
        value = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        value = 0
    hours, value = divmod(value, 3600)
    minutes, seconds = divmod(value, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s" if seconds else f"{minutes}m"
    return f"{seconds}s"


def _row_records(records: object) -> list[Mapping[str, object]]:
    if isinstance(records, Mapping):
        records = records.get("jobs", ())
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        return []
    return [row for row in records if isinstance(row, Mapping)]


def render_jobs_block(records: object) -> str:
    """Render the compact model-facing JOBS block from host-computed rows."""
    rows = _row_records(records)
    lines = ["[JOBS]"]
    if not rows:
        lines.append("none")
    for row in rows:
        job_id = str(row.get("job_id", "UNKNOWN"))
        state = str(row.get("state", "unknown"))
        elapsed = _duration(row.get("elapsed", 0))
        expected = _duration(row.get("expected_s", 0))
        age = row.get("last_output_age_s")
        last_output = "never" if age is None else f"{_duration(age)} ago"
        parts = [f"{job_id} {state} {elapsed} / expected {expected}",
                 f"last output {last_output}"]
        if row.get("exit_code") is not None:
            parts.append(f"exit {row['exit_code']}")
        if row.get("done_signal_seen"):
            parts.append("SIGNAL_SEEN")
        if row.get("overdue"):
            parts.append("OVERDUE")
        if row.get("stalled"):
            parts.append("STALLED")
        lines.append(" · ".join(parts))
    lines.append("[/JOBS]")
    return "\n".join(lines)


class JobWatch:
    """Manage background subprocesses without making liveness decisions for the model."""

    def __init__(self, root: str | os.PathLike[str], *, journal: object = None,
                 max_total_wall_s: float | None = None,
                 clock: Callable[[], float] | None = None,
                 wall_clock: Callable[[], float] | None = None,
                 popen_factory: Callable[..., subprocess.Popen] | None = None):
        self.root = Path(root).resolve()
        self.journal = journal
        self.max_total_wall_s = (_max_total_wall_from_env()
                                 if max_total_wall_s is None else
                                 _finite_positive(max_total_wall_s, "max_total_wall_s"))
        self.clock = clock or time.monotonic
        self.wall_clock = wall_clock or time.time
        # The optional factory is a deterministic test seam only; production
        # launch always delegates to kernel_execution_policy.
        self.popen_factory = popen_factory
        self._custom_popen = popen_factory is not None
        self._lock = threading.RLock()
        self._jobs: dict[str, _ManagedJob] = {}
        self._next_id = 1
        self._wall_cap_hit = False

    def _emit(self, kind: str, **fields: object) -> dict[str, object] | None:
        emit = getattr(self.journal, "emit", None)
        if not callable(emit):
            return None
        return emit(kind, producer="kernel_job_watch", **fields)

    def _new_id(self) -> str:
        job_id = f"J{self._next_id}"
        self._next_id += 1
        return job_id

    def _total_wall_locked(self) -> float:
        return sum(max(0.0, float(job.record.elapsed)) for job in self._jobs.values())

    def _active_ids_locked(self) -> list[str]:
        return [job.record.job_id for job in self._jobs.values()
                if job.record.state == "running"]

    def _check_wall_cap_locked(self) -> None:
        total = self._total_wall_locked()
        if total < self.max_total_wall_s or self._wall_cap_hit:
            return
        self._wall_cap_hit = True
        self._emit(
            "job_wall_cap_hit",
            total_job_wall_s=total,
            max_total_job_wall_s=self.max_total_wall_s,
            active_job_ids=self._active_ids_locked(),
            host_killed=False,
            dispatch=False,
        )

    def _status_locked(self, managed: _ManagedJob) -> dict[str, object]:
        row = managed.record.to_dict()
        if managed.record.last_output_at is not None:
            row["last_output_age_s"] = max(
                0.0, float(self.wall_clock()) - managed.record.last_output_at)
        return row

    def _reconcile(self, job_id: str, *, wait_for_exit: bool = False) -> None:
        with self._lock:
            managed = self._jobs.get(job_id)
            if managed is None:
                return
            process = managed.process
            if (wait_for_exit and managed.reader_done and process.poll() is None):
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
            now_mono = float(self.clock())
            exit_code = process.poll()
            record = managed.record
            if exit_code is None:
                record.elapsed = max(0.0, now_mono - managed.started_mono)
            else:
                record.elapsed = max(0.0, now_mono - managed.started_mono)
                # A process can exit before its pipe reader has drained the
                # final bytes.  Delay the terminal transition and calibration
                # event until that host-owned reader has seen EOF, so
                # ``done_signal_seen`` and output accounting are final.
                if managed.reader_done:
                    record.exit_code = int(exit_code)
                    if record.state == "running":
                        record.state = "killed" if managed.kill_requested else (
                            "completed" if exit_code == 0 else "failed")
                    if not managed.completion_journaled:
                        actual = record.elapsed
                        self._emit(
                            "job_completed",
                            job_id=record.job_id,
                            expected_s=record.expected_s,
                            actual_elapsed_s=actual,
                            expected_vs_actual={
                                "expected_s": record.expected_s,
                                "actual_s": actual,
                                "delta_s": actual - record.expected_s,
                                "ratio": actual / record.expected_s if record.expected_s else None,
                            },
                            exit_code=record.exit_code,
                            state=record.state,
                            done_signal=record.done_signal,
                            done_signal_seen=record.done_signal_seen,
                            output_bytes=record.output_bytes,
                            dispatch=False,
                        )
                        managed.completion_journaled = True
            record.bytes_growth_last_interval = max(
                0, record.output_bytes - managed.output_sample_bytes)
            managed.output_sample_bytes = record.output_bytes
            record.overdue = record.elapsed > 2.5 * record.expected_s
            if record.state == "running":
                quiet_limit = max(60.0, record.expected_s)
                quiet_for = (record.elapsed if managed.last_output_mono is None
                             else max(0.0, now_mono - managed.last_output_mono))
                record.stalled = quiet_for >= quiet_limit
            else:
                record.stalled = False
            self._check_wall_cap_locked()

    def _reader_loop(self, job_id: str, process: subprocess.Popen[bytes]) -> None:
        stream = process.stdout
        if stream is not None:
            try:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    self._append_output(job_id, chunk)
            finally:
                stream.close()
        with self._lock:
            managed = self._jobs.get(job_id)
            if managed is not None:
                managed.reader_done = True
        self._reconcile(job_id, wait_for_exit=True)

    def _append_output(self, job_id: str, chunk: bytes) -> None:
        with self._lock:
            managed = self._jobs.get(job_id)
            if managed is None:
                return
            record = managed.record
            record.output_bytes += len(chunk)
            record.last_output_at = float(self.wall_clock())
            managed.last_output_mono = float(self.clock())
            signal = managed.done_signal_bytes
            if signal:
                candidate = managed.signal_tail + chunk
                if signal in candidate:
                    record.done_signal_seen = True
                managed.signal_tail = candidate[
                    -max(0, len(managed.done_signal_bytes) - 1):]
            text = chunk.decode("utf-8", errors="replace")
            combined = managed.partial_line + text
            lines = combined.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                managed.partial_line = lines.pop()
            else:
                managed.partial_line = ""
            managed.tail_lines.extend(lines)

    def refresh(self) -> list[dict[str, object]]:
        """Poll all processes and return a stable, host-computed projection."""
        with self._lock:
            job_ids = tuple(self._jobs)
        for job_id in job_ids:
            self._reconcile(job_id)
        with self._lock:
            return [self._status_locked(self._jobs[job_id]) for job_id in job_ids
                    if job_id in self._jobs]

    def _lookup(self, job_id: object) -> _ManagedJob:
        if not isinstance(job_id, str) or not job_id.strip():
            raise JobWatchError("job_id must be a non-empty string", code="INVALID_ARGUMENT")
        with self._lock:
            managed = self._jobs.get(job_id)
        if managed is None:
            raise JobWatchError(f"unknown job_id: {job_id}", code="UNKNOWN_JOB")
        return managed

    def launch_job(self, cmd: object, expected_s: object, done_signal: object) -> dict[str, object]:
        command = _required_text(cmd, "cmd", MAX_COMMAND_CHARS)
        expected = _finite_positive(expected_s, "expected_s")
        signal = _required_text(done_signal, "done_signal", MAX_DONE_SIGNAL_CHARS)
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as error:
            raise JobWatchError(f"invalid command quoting: {error}", code="INVALID_ARGUMENT") from error
        if not argv or any("\x00" in item for item in argv):
            raise JobWatchError("cmd must contain one non-empty argv", code="INVALID_ARGUMENT")
        if any(any(operator in item for operator in (";", "|", "&", ">", "<"))
               for item in argv):
            raise JobWatchError(
                "shell operators are not accepted; launch_job requires argv-safe command text",
                code="INVALID_ARGUMENT",
            )
        self.refresh()
        with self._lock:
            active = self._active_ids_locked()
            if len(active) >= MAX_CONCURRENT_JOBS:
                self._emit("job_concurrency_cap_hit", active_job_ids=active,
                           max_concurrent_jobs=MAX_CONCURRENT_JOBS,
                           requested_expected_s=expected, host_killed=False,
                           dispatch=False)
                raise JobWatchError(
                    f"concurrent job cap reached ({MAX_CONCURRENT_JOBS}); inspect or kill a job",
                    code="CONCURRENT_CAP")
            self._check_wall_cap_locked()
            if self._wall_cap_hit:
                self._emit("job_wall_cap_rejected", active_job_ids=active,
                           total_job_wall_s=self._total_wall_locked(),
                           max_total_job_wall_s=self.max_total_wall_s,
                           requested_expected_s=expected, host_killed=False,
                           dispatch=False)
                raise JobWatchError(
                    "total job wall cap reached; inspect or kill an existing job",
                    code="WALL_CAP")
            job_id = self._new_id()
            started_mono = float(self.clock())
            started = float(self.wall_clock())
            try:
                from kernel_execution_policy import sandboxed_popen
                process, _capabilities = sandboxed_popen(
                    argv,
                    self.root,
                    stdin=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    cpu_seconds=min(60.0, max(1.0, expected * 2.0)),
                    popen_factory=self.popen_factory if self._custom_popen else None,
                )
            except (OSError, subprocess.SubprocessError) as error:
                self._emit("job_launch_failed", job_id=job_id,
                           expected_s=expected, error_type=type(error).__name__,
                           error=str(error)[:400], dispatch=False)
                raise JobWatchError(f"could not launch job: {error}", code="LAUNCH_FAILED") from error
            except Exception as error:
                from kernel_execution_policy import ExecutionPolicyError
                if isinstance(error, ExecutionPolicyError):
                    self._emit("job_launch_refused", job_id=job_id,
                               reason="sandbox_unavailable", dispatch=False)
                    raise JobWatchError(str(error), code="SANDBOX_UNAVAILABLE") from error
                raise
            record = JobRecord(job_id=job_id, started=started,
                               expected_s=expected, done_signal=signal)
            managed = _ManagedJob(
                record=record, process=process, started_mono=started_mono,
                done_signal_bytes=signal.encode("utf-8"), clock=self.clock,
            )
            self._jobs[job_id] = managed
            self._emit("job_started", job_id=job_id, command=command,
                       expected_s=expected, done_signal=signal,
                       max_concurrent_jobs=MAX_CONCURRENT_JOBS,
                       max_total_job_wall_s=self.max_total_wall_s,
                       dispatch=True)
            reader = threading.Thread(
                target=self._reader_loop, args=(job_id, process),
                name=f"stateframe-job-{job_id}", daemon=True,
            )
            managed.reader = reader
            reader.start()
            return self._status_locked(managed)

    def job_status(self, job_id: object) -> dict[str, object]:
        managed = self._lookup(job_id)
        self._reconcile(managed.record.job_id)
        with self._lock:
            return self._status_locked(managed)

    def job_output(self, job_id: object, tail_lines: object = DEFAULT_TAIL_LINES) -> dict[str, object]:
        managed = self._lookup(job_id)
        if isinstance(tail_lines, bool) or not isinstance(tail_lines, int):
            raise JobWatchError("tail_lines must be an integer", code="INVALID_ARGUMENT")
        if not 1 <= tail_lines <= MAX_TAIL_LINES:
            raise JobWatchError(
                f"tail_lines must be between 1 and {MAX_TAIL_LINES}", code="INVALID_ARGUMENT")
        self._reconcile(managed.record.job_id)
        with self._lock:
            lines = list(managed.tail_lines)
            if managed.partial_line:
                lines.append(managed.partial_line)
            output = "".join(lines[-tail_lines:])
            status = self._status_locked(managed)
        return {
            "schema": "stateframe-job-output/1",
            "job_id": status["job_id"],
            "tail_lines": tail_lines,
            "output": output,
            "status": status,
        }

    def job_kill(self, job_id: object) -> dict[str, object]:
        managed = self._lookup(job_id)
        self._reconcile(managed.record.job_id)
        with self._lock:
            if managed.record.state != "running":
                return self._status_locked(managed)
            managed.kill_requested = True
            self._emit("job_kill_requested", job_id=managed.record.job_id,
                       reason="model_selected_job_kill", initiator="model",
                       host_killed_automatically=False,
                       dispatch=True)
            process = managed.process
            signalled = False
            pid = getattr(process, "pid", None)
            if os.name == "posix" and isinstance(pid, int) and pid > 0:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    signalled = True
                except (OSError, ProcessLookupError):
                    pass
            if not signalled:
                terminate = getattr(process, "terminate", None)
                if callable(terminate):
                    terminate()
                else:
                    kill = getattr(process, "kill", None)
                    if callable(kill):
                        kill()

        # This escalation is still part of the explicit model-selected
        # operation; there is no background timeout or automatic kill path.
        try:
            process.wait(timeout=1.0)
        except (AttributeError, subprocess.TimeoutExpired):
            escalated = False
            pid = getattr(process, "pid", None)
            if os.name == "posix" and isinstance(pid, int) and pid > 0:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                    escalated = True
                except (OSError, ProcessLookupError):
                    pass
            if not escalated:
                kill = getattr(process, "kill", None)
                if callable(kill):
                    kill()
            try:
                process.wait(timeout=1.0)
            except (AttributeError, subprocess.TimeoutExpired):
                pass
        reader = managed.reader
        if reader is not None and reader.is_alive():
            reader.join(timeout=1.0)
        self._reconcile(managed.record.job_id, wait_for_exit=True)
        with self._lock:
            return self._status_locked(managed)

    def frame(self) -> dict[str, object]:
        rows = self.refresh()
        with self._lock:
            total = self._total_wall_locked()
            remaining = max(0.0, self.max_total_wall_s - total)
            cap_hit = self._wall_cap_hit
        return {
            "schema": "stateframe-jobs/1",
            "jobs": rows,
            "limits": {
                "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
                "max_total_job_wall_s": self.max_total_wall_s,
                "host_kills_automatically": False,
            },
            "total_job_wall_s": total,
            "total_job_wall_remaining_s": remaining,
            "total_job_wall_cap_hit": cap_hit,
            "block": render_jobs_block(rows),
        }

    def execute(self, name: object, arguments: Mapping[str, object] | None = None) -> str:
        if not isinstance(name, str):
            raise JobWatchError("job operation name must be a string", code="INVALID_ARGUMENT")
        args = dict(arguments or {})
        if name == "launch_job":
            result = self.launch_job(args.get("cmd"), args.get("expected_s"), args.get("done_signal"))
        elif name == "job_status":
            result = self.job_status(args.get("job_id"))
        elif name == "job_output":
            result = self.job_output(args.get("job_id"), args.get("tail_lines", DEFAULT_TAIL_LINES))
        elif name == "job_kill":
            result = self.job_kill(args.get("job_id"))
        else:
            raise JobWatchError(f"unknown job operation: {name}", code="UNKNOWN_OPERATION")
        return json.dumps(result, sort_keys=True, ensure_ascii=False, default=str)


__all__ = [
    "DEFAULT_MAX_TOTAL_JOB_WALL_S", "DEFAULT_TAIL_LINES", "JobRecord", "JobWatch",
    "JobWatchError", "MAX_CONCURRENT_JOBS", "MAX_TAIL_LINES", "MODE_ENV",
    "configuration", "enabled", "mode", "render_jobs_block", "tool_surface",
]
