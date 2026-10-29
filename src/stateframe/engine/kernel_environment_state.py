"""Scoped, host-owned environment state for the environment-state organ.

This module is deliberately small and boring.  It owns processes launched
through its launcher, while facts already owned by another organ are ingested
as projections (for example ``JobWatch`` rows and test receipts).  It never
turns an observation into readiness by itself and it never exposes raw command
arguments or unbounded process output.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Generic, TypeVar

from kernel_environment_bindings import ApplicabilityResult, EnvironmentLedger
from kernel_execution_policy import ExecutionPolicyError, sandboxed_popen
from kernel_environment_recipes import RecipeRegistry, SetupRecipe
from kernel_environment_types import RequirementBinding, ResourceFact, ResourcePredicate


MODE_ENV = "STATEFRAME_ENVIRONMENT_STATE"
SCHEMA = "stateframe-environment-snapshot/1"
SCOPE_SCHEMA = "stateframe-environment-scope/1"
OBSERVATION_SCHEMA = "stateframe-environment-observation/1"
PROCESS_SCHEMA = "stateframe-environment-process/1"
EVENT_SCHEMA = "stateframe-environment-event/1"
RECEIPT_SCHEMA = "stateframe-environment-receipt-index/1"

DEFAULT_EVENT_LIMIT = 128
DEFAULT_TAIL_BYTES = 4096
MAX_LABEL_LENGTH = 96
MAX_TAIL_BYTES = 16 * 1024
MAX_RECEIPTS = 64

_RECEIPT_ARTIFACT_DIRS = frozenset({
    ".gate", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".venv", ".test-tmp", "htmlcov", "node_modules", "backup", "backups",
})

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|secret)\b\s*[=:]\s*)([^\s,;]+)"
)
_SECRET_KEY = re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|secret)")
_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._:/ -]+")
_OFF_VALUES = frozenset({"", "off", "0", "false", "no"})


def _receipt_file_digest(root: Path, relative: str) -> str:
    """Hash only receipt-referenced confined files, never workspace metadata."""
    path = Path(relative)

    def excluded(candidate: Path) -> bool:
        return (any(part in _RECEIPT_ARTIFACT_DIRS for part in candidate.parts)
                or candidate.suffix in {".pyc", ".pyo"}
                or candidate.name == ".coverage"
                or candidate.name.startswith(".coverage."))

    if path.is_absolute() or ".." in path.parts or excluded(path):
        return "excluded"
    try:
        candidate = (root / path).resolve()
        confined = candidate.relative_to(root)
        if excluded(confined):
            return "excluded"
        if not candidate.is_file():
            return "unavailable"
        content = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                content.update(chunk)
        return "sha256:" + content.hexdigest()
    except (OSError, RuntimeError, ValueError):
        return "unavailable"


T = TypeVar("T")


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict organ switch; typos must not silently enable it."""
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _copy(value: T) -> T:
    if isinstance(value, Mapping):
        return {str(key): _copy(item) for key, item in value.items()}  # type: ignore[return-value]
    if isinstance(value, list):
        return [_copy(item) for item in value]  # type: ignore[return-value]
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)  # type: ignore[return-value]
    return value


def _safe_text(value: object, limit: int) -> str:
    text = str(value or "")
    text = text.replace("\x00", "")
    text = "".join(char if char in "\n\r\t" or ord(char) >= 32 else "?" for char in text)
    text = _SECRET_ASSIGNMENT.sub(r"\1<redacted>", text)
    if len(text) <= limit:
        return text
    marker = "...[tail truncated]"
    return text[:max(0, limit - len(marker))] + marker


def safe_label(value: object, *, fallback: str = "process") -> str:
    text = " ".join(str(value or "").split())
    text = _SAFE_LABEL.sub("?", text).strip(" .")
    return (text or fallback)[:MAX_LABEL_LENGTH]


def _safe_value(value: object, *, depth: int = 0) -> object:
    """Keep observed values JSON-safe, bounded, and free of obvious secrets."""
    if depth > 3:
        return "<omitted:depth>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "<non-finite>"
    if isinstance(value, str):
        return _safe_text(value, 1024)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 32:
                result["<omitted>"] = "<field-limit>"
                break
            public_key = safe_label(key, fallback="field")
            result[public_key] = (
                "<redacted>" if _SECRET_KEY.search(str(key))
                else _safe_value(item, depth=depth + 1)
            )
        return result
    if isinstance(value, (list, tuple)):
        items = [_safe_value(item, depth=depth + 1) for item in value[:32]]
        if len(value) > 32:
            items.append("<omitted:item-limit>")
        return items
    return _safe_text(value, 256)


def _relative_ref(root: Path, candidate: Path) -> str:
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix() or "."
    except (OSError, ValueError):
        return "<outside-workspace>"


def _pid_identity(pid: object, started_mono_ns: int) -> str:
    if isinstance(pid, int) and pid > 0:
        return f"pid:{pid}:launch:{started_mono_ns}"
    return f"unknown:launch:{started_mono_ns}"


def _exit_fields(returncode: object) -> tuple[int | None, int | None, str]:
    if not isinstance(returncode, int):
        return None, None, "unknown"
    if returncode < 0:
        return None, -returncode, "signal"
    return returncode, None, "exit"


@dataclass(frozen=True)
class EnvironmentScope:
    run_id: str
    workspace_id: str
    grant_id: str
    host_epoch: str

    @property
    def schema(self) -> str:
        return SCOPE_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCOPE_SCHEMA,
            "run_id": self.run_id,
            "workspace_id": self.workspace_id,
            "grant_id": self.grant_id,
            "host_epoch": self.host_epoch,
        }


@dataclass(frozen=True)
class EnvironmentObservation(Generic[T]):
    observation_id: str
    entity_id: str
    entity_generation: int
    observer_id: str
    observer_version: str
    availability: str
    freshness: str
    coverage: str
    observed_at: float
    valid_until: float | None
    value_revision: str
    value: T
    evidence_ref: str | None = None

    @property
    def schema(self) -> str:
        return OBSERVATION_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": OBSERVATION_SCHEMA,
            "observation_id": self.observation_id,
            "entity_id": self.entity_id,
            "entity_generation": self.entity_generation,
            "observer_id": self.observer_id,
            "observer_version": self.observer_version,
            "availability": self.availability,
            "freshness": self.freshness,
            "coverage": self.coverage,
            "observed_at": self.observed_at,
            "valid_until": self.valid_until,
            "value_revision": self.value_revision,
            "value": _copy(self.value),
            "evidence_ref": self.evidence_ref,
        }


@dataclass(frozen=True)
class ProcessRecord:
    launch_id: str
    entity_generation: int
    role: str
    pid: int | None
    birth_identity: str
    containment_id: str
    command_label: str
    cwd_ref: str
    state: str
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    exit_signal: int | None = None
    stop_reason: str | None = None
    output_tail: str | None = None
    output_truncated: bool = False
    ownership: str = "owned"

    @property
    def schema(self) -> str:
        return PROCESS_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PROCESS_SCHEMA,
            "launch_id": self.launch_id,
            "entity_generation": self.entity_generation,
            "role": self.role,
            "pid": self.pid,
            "birth_identity": self.birth_identity,
            "containment_id": self.containment_id,
            "command_label": self.command_label,
            "cwd_ref": self.cwd_ref,
            "state": self.state,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exit_code": self.exit_code,
            "exit_signal": self.exit_signal,
            "stop_reason": self.stop_reason,
            "output_tail": self.output_tail,
            "output_truncated": self.output_truncated,
            "ownership": self.ownership,
        }


@dataclass(frozen=True)
class EnvironmentEvent:
    event_id: str
    sequence: int
    entity_id: str
    entity_generation: int
    kind: str
    observed_at: float
    observation_id: str | None = None
    previous_observation_id: str | None = None

    @property
    def schema(self) -> str:
        return EVENT_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": EVENT_SCHEMA,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "entity_id": self.entity_id,
            "entity_generation": self.entity_generation,
            "kind": self.kind,
            "observed_at": self.observed_at,
            "observation_id": self.observation_id,
            "previous_observation_id": self.previous_observation_id,
        }


@dataclass(frozen=True)
class ReceiptIndexRow:
    receipt_id: str
    kind: str
    result_class: str
    revision: str
    owner: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": RECEIPT_SCHEMA,
            "receipt_id": self.receipt_id,
            "kind": self.kind,
            "result_class": self.result_class,
            "revision": self.revision,
            "owner": self.owner,
        }


@dataclass(frozen=True)
class EnvironmentSnapshot:
    scope: EnvironmentScope
    snapshot_seq: int
    baseline_id: str
    assembled_at: float
    processes: tuple[ProcessRecord, ...] = ()
    observations: tuple[EnvironmentObservation[object], ...] = ()
    events: tuple[EnvironmentEvent, ...] = ()
    receipts: tuple[ReceiptIndexRow, ...] = ()
    required_checks: tuple[dict[str, object], ...] = ()
    resource_facts: tuple[dict[str, object], ...] = ()
    predicates: tuple[dict[str, object], ...] = ()
    omissions: tuple[str, ...] = ()
    coverage: tuple[str, ...] = ()

    @property
    def schema(self) -> str:
        return SCHEMA

    def semantic_dict(self) -> dict[str, object]:
        """Stable panel input; sensor clocks are intentionally excluded."""
        return {
            "scope": self.scope.to_dict(),
            "snapshot_seq": self.snapshot_seq,
            "baseline_id": self.baseline_id,
            "processes": [row.to_dict() for row in self.processes],
            "observations": [
                {
                    key: value
                    for key, value in row.to_dict().items()
                    if key not in {"observation_id", "observed_at", "valid_until"}
                }
                for row in self.observations
            ],
            "events": [
                {key: value for key, value in row.to_dict().items() if key != "observed_at"}
                for row in self.events
            ],
            "receipts": [row.to_dict() for row in self.receipts],
            "required_checks": [_copy(row) for row in self.required_checks],
            "resource_facts": [_copy(row) for row in self.resource_facts],
            "predicates": [_copy(row) for row in self.predicates],
            "omissions": list(self.omissions),
            "coverage": list(self.coverage),
        }

    @property
    def semantic_digest(self) -> str:
        return digest(self.semantic_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "scope": self.scope.to_dict(),
            "snapshot_seq": self.snapshot_seq,
            "baseline_id": self.baseline_id,
            "assembled_at": self.assembled_at,
            "semantic_digest": self.semantic_digest,
            "processes": [row.to_dict() for row in self.processes],
            "observations": [row.to_dict() for row in self.observations],
            "events": [row.to_dict() for row in self.events],
            "receipts": [row.to_dict() for row in self.receipts],
            "required_checks": [_copy(row) for row in self.required_checks],
            "resource_facts": [_copy(row) for row in self.resource_facts],
            "predicates": [_copy(row) for row in self.predicates],
            "omissions": list(self.omissions),
            "coverage": list(self.coverage),
        }


@dataclass
class _OwnedProcess:
    process: subprocess.Popen[bytes]
    record: ProcessRecord
    started_mono: float
    tail: deque[bytes]
    tail_size: int = 0
    reader: threading.Thread | None = None
    reader_done: bool = False
    terminal_recorded: bool = False
    cleanup_requested: bool = False


class EnvironmentState:
    """One per-run owner for environment records and owned process handles."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        journal: object = None,
        run_id: str = "run:unknown",
        workspace_id: str | None = None,
        grant_id: str = "grant:environment",
        event_limit: int = DEFAULT_EVENT_LIMIT,
        tail_bytes: int = DEFAULT_TAIL_BYTES,
        wall_clock: Callable[[], float] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        popen_factory: Callable[..., subprocess.Popen[bytes]] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError("environment root must be a directory")
        if event_limit < 1:
            raise ValueError("event_limit must be positive")
        if not 1 <= tail_bytes <= MAX_TAIL_BYTES:
            raise ValueError(f"tail_bytes must be between 1 and {MAX_TAIL_BYTES}")
        self.journal = journal
        self.wall_clock = wall_clock or time.time
        self.monotonic_clock = monotonic_clock or time.monotonic
        # The optional factory is a deterministic test seam only.  Production
        # launches always enter kernel_execution_policy before Popen is reached.
        self.popen_factory = popen_factory
        epoch_seed = f"{os.getpid()}:{self.monotonic_clock():.9f}"
        host_epoch = "host:" + hashlib.sha256(epoch_seed.encode()).hexdigest()[:16]
        workspace = workspace_id or "workspace:" + digest(str(self.root))[-16:]
        self.scope = EnvironmentScope(
            safe_label(run_id, fallback="run:unknown"),
            safe_label(workspace, fallback="workspace"),
            safe_label(grant_id, fallback="grant:environment"),
            host_epoch,
        )
        self.baseline_id = "baseline:" + digest({"root": str(self.root), "scope": self.scope.to_dict()})[-16:]
        self.event_limit = event_limit
        self.tail_bytes = tail_bytes
        self._lock = threading.RLock()
        self._next_launch = 1
        self._sequence = 0
        self._events: deque[EnvironmentEvent] = deque(maxlen=event_limit)
        self._processes: dict[str, _OwnedProcess] = {}
        self._observations: dict[tuple[str, int], EnvironmentObservation[object]] = {}
        self._receipt_rows: tuple[ReceiptIndexRow, ...] = ()
        self._receipt_index_digest = digest([])
        self._required_checks: dict[str, dict[str, object]] = {}
        # The ledger is a projection owned by this state object; it is not a
        # second readiness or receipt authority.
        self.environment_ledger = EnvironmentLedger()
        self.model_ledger = self.environment_ledger
        self.recipe_registry = RecipeRegistry()
        self._omissions: set[str] = set()
        self._coverage: set[str] = {"direct-launches", "versioned-resource-model"}
        self._job_watch: object | None = None
        self._closed = False

    def _emit(self, kind: str, **fields: object) -> None:
        emit = getattr(self.journal, "emit", None)
        if callable(emit):
            emit(kind, producer="kernel_environment_state", **fields, dispatch=False)

    def _event(
        self,
        kind: str,
        entity_id: str,
        generation: int,
        *,
        observation_id: str | None = None,
        previous_observation_id: str | None = None,
    ) -> EnvironmentEvent:
        with self._lock:
            self._sequence += 1
            event = EnvironmentEvent(
                event_id=f"env-event:{self._sequence}",
                sequence=self._sequence,
                entity_id=entity_id,
                entity_generation=generation,
                kind=kind,
                observed_at=float(self.wall_clock()),
                observation_id=observation_id,
                previous_observation_id=previous_observation_id,
            )
            self._events.append(event)
        self._emit(
            f"environment_{kind}",
            environment_event_id=event.event_id,
            event_entity_id=entity_id,
            entity_generation=generation,
            event_observation_id=observation_id,
            event_previous_observation_id=previous_observation_id,
        )
        return event

    def _append_tail(self, launch_id: str, chunk: bytes) -> None:
        if not chunk:
            return
        with self._lock:
            managed = self._processes.get(launch_id)
            if managed is None:
                return
            managed.tail.append(bytes(chunk))
            managed.tail_size += len(chunk)
            while managed.tail_size > self.tail_bytes and managed.tail:
                managed.tail_size -= len(managed.tail.popleft())

    def _reader_loop(self, launch_id: str, process: subprocess.Popen[bytes]) -> None:
        stream = process.stdout
        if stream is not None:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                self._append_tail(launch_id, chunk)
        with self._lock:
            managed = self._processes.get(launch_id)
            if managed is not None:
                managed.reader_done = True

    def _tail_text(self, managed: _OwnedProcess) -> tuple[str | None, bool]:
        raw = b"".join(managed.tail)
        if not raw:
            return None, managed.tail_size >= self.tail_bytes
        return _safe_text(raw.decode("utf-8", errors="replace"), self.tail_bytes), managed.tail_size >= self.tail_bytes

    def _replace_record(self, launch_id: str, record: ProcessRecord) -> None:
        managed = self._processes[launch_id]
        managed.record = record

    def _invalidate_required_checks(self, entity_ids: set[str]) -> None:
        changed = False
        with self._lock:
            for check_id, row in self._required_checks.items():
                if str(row.get("entity_id") or "") not in entity_ids:
                    continue
                if row.get("status") in {"failed", "unknown", "expired"}:
                    continue
                updated = dict(row)
                updated["status"] = "unknown"
                updated["invalidation"] = "owned_process_terminated"
                self._required_checks[check_id] = updated
                changed = True
        if changed:
            self._event("readiness_changed", sorted(entity_ids)[0] if entity_ids else "process", 1)

    def _reconcile_one(self, launch_id: str, *, wait_for_exit: bool = False) -> None:
        with self._lock:
            managed = self._processes.get(launch_id)
            if managed is None:
                return
            process = managed.process
            prior = managed.record
        returncode = process.poll()
        if returncode is None:
            return
        if wait_for_exit:
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        reader = managed.reader
        if reader is not None and reader.is_alive():
            reader.join(timeout=0.5)
        try:
            process.wait(timeout=0.1)
        except (AttributeError, subprocess.TimeoutExpired):
            pass
        code, exit_signal, terminal_kind = _exit_fields(process.returncode)
        state = "killed" if managed.cleanup_requested else ("exited" if code == 0 else "crashed")
        tail, tail_truncated = self._tail_text(managed)
        record = ProcessRecord(
            launch_id=prior.launch_id,
            entity_generation=prior.entity_generation,
            role=prior.role,
            pid=prior.pid,
            birth_identity=prior.birth_identity,
            containment_id=prior.containment_id,
            command_label=prior.command_label,
            cwd_ref=prior.cwd_ref,
            state=state,
            started_at=prior.started_at,
            ended_at=float(self.wall_clock()),
            exit_code=code,
            exit_signal=exit_signal,
            stop_reason=("cleanup" if managed.cleanup_requested else terminal_kind),
            output_tail=tail,
            output_truncated=tail_truncated,
            ownership=prior.ownership,
        )
        with self._lock:
            self._replace_record(launch_id, record)
            should_emit = not managed.terminal_recorded
            managed.terminal_recorded = True
        if should_emit:
            self._event(
                "process_crashed" if state == "crashed" else "process_exited",
                launch_id,
                prior.entity_generation,
            )
            self._invalidate_required_checks({launch_id, prior.containment_id})
            self._emit(
                "environment_process_reaped",
                launch_id=launch_id,
                process_state=state,
                exit_code=code,
                exit_signal=exit_signal,
                log_tail_present=bool(tail),
            )

    def refresh(self) -> tuple[ProcessRecord, ...]:
        with self._lock:
            launch_ids = tuple(self._processes)
        for launch_id in launch_ids:
            self._reconcile_one(launch_id)
        with self._lock:
            return tuple(self._processes[launch_id].record for launch_id in launch_ids
                         if launch_id in self._processes)

    def _expire_observations(self) -> None:
        now = float(self.wall_clock())
        expired: list[EnvironmentObservation[object]] = []
        with self._lock:
            for key, observation in tuple(self._observations.items()):
                if (observation.freshness != "fresh"
                        or observation.valid_until is None
                        or now < observation.valid_until):
                    continue
                stale = replace(observation, freshness="stale")
                self._observations[key] = stale
                expired.append(stale)
        for observation in expired:
            self._event(
                "observation_expired",
                observation.entity_id,
                observation.entity_generation,
                observation_id=observation.observation_id,
            )
            self._invalidate_required_checks({observation.entity_id})

    def launch(
        self,
        command: Sequence[object],
        *,
        role: str = "agent_process",
        command_label: str = "agent process",
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ProcessRecord:
        if isinstance(command, (str, bytes, bytearray)) or not command:
            raise TypeError("command must be a non-empty argument sequence")
        arguments = tuple(str(item) for item in command)
        workdir = self.root if cwd is None else Path(cwd).resolve()
        try:
            workdir.relative_to(self.root)
        except ValueError as error:
            raise ValueError("process cwd must stay inside the environment workspace") from error
        with self._lock:
            if self._closed:
                raise RuntimeError("environment state is closed")
            launch_id = f"P{self._next_launch}"
            self._next_launch += 1
            generation = 1
            started = float(self.wall_clock())
            started_mono_ns = time.monotonic_ns()
        label = safe_label(command_label)
        public_role = safe_label(role, fallback="agent_process")
        self._emit("environment_process_launch_intent", launch_id=launch_id,
                   role=public_role,
                   command_label=label, cwd_ref=_relative_ref(self.root, workdir))
        try:
            process, _capabilities = sandboxed_popen(
                list(arguments),
                workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cpu_seconds=60.0,
                memory_bytes=512 * 1024 * 1024,
                file_bytes=8 * 1024 * 1024,
                popen_factory=self.popen_factory,
            )
        except (ExecutionPolicyError, OSError, subprocess.SubprocessError) as error:
            self._event("process_launch_failed", launch_id, generation)
            self._emit("environment_process_launch_failed", launch_id=launch_id,
                       error_type=type(error).__name__)
            raise
        pid = getattr(process, "pid", None)
        pid_value = pid if isinstance(pid, int) and pid > 0 else None
        record = ProcessRecord(
            launch_id=launch_id,
            entity_generation=generation,
            role=public_role,
            pid=pid_value,
            birth_identity=_pid_identity(pid_value, started_mono_ns),
            containment_id=f"pgid:{pid_value}" if pid_value is not None else f"launch:{launch_id}",
            command_label=label,
            cwd_ref=_relative_ref(self.root, workdir),
            state="running",
            started_at=started,
        )
        managed = _OwnedProcess(
            process=process,
            record=record,
            started_mono=float(self.monotonic_clock()),
            tail=deque(),
        )
        with self._lock:
            self._processes[launch_id] = managed
        reader = threading.Thread(
            target=self._reader_loop,
            args=(launch_id, process),
            name=f"stateframe-environment-{launch_id}",
            daemon=True,
        )
        managed.reader = reader
        reader.start()
        self._event("process_started", launch_id, generation)
        self._emit("environment_process_started", launch_id=launch_id,
                   process_id=pid_value, role=record.role, containment_id=record.containment_id)
        return record

    def _signal_group(self, managed: _OwnedProcess, signum: int) -> bool:
        pid = managed.record.pid
        if os.name == "posix" and isinstance(pid, int) and pid > 0:
            try:
                os.killpg(os.getpgid(pid), signum)
                return True
            except (OSError, ProcessLookupError):
                return False
        return False

    def cleanup(self, *, grace_seconds: float = 1.0) -> tuple[ProcessRecord, ...]:
        if grace_seconds <= 0:
            raise ValueError("grace_seconds must be positive")
        with self._lock:
            managed_rows = tuple(
                managed for managed in self._processes.values()
                if managed.record.ownership == "owned"
            )
        for managed in managed_rows:
            if managed.process.poll() is None:
                managed.cleanup_requested = True
                signalled = self._signal_group(managed, signal.SIGTERM)
                if not signalled:
                    terminate = getattr(managed.process, "terminate", None)
                    if callable(terminate):
                        terminate()
                self._emit("environment_process_cleanup_requested",
                           launch_id=managed.record.launch_id, signal="TERM")
        deadline = float(self.monotonic_clock()) + grace_seconds
        for managed in managed_rows:
            remaining = max(0.0, deadline - float(self.monotonic_clock()))
            try:
                managed.process.wait(timeout=remaining)
            except (AttributeError, subprocess.TimeoutExpired):
                if managed.process.poll() is None:
                    signalled = self._signal_group(managed, signal.SIGKILL)
                    if not signalled:
                        kill = getattr(managed.process, "kill", None)
                        if callable(kill):
                            kill()
                    self._emit("environment_process_cleanup_escalated",
                               launch_id=managed.record.launch_id, signal="KILL")
                    try:
                        managed.process.wait(timeout=grace_seconds)
                    except (AttributeError, subprocess.TimeoutExpired):
                        pass
            self._reconcile_one(managed.record.launch_id, wait_for_exit=True)
        rows = self.refresh()
        self._emit("environment_process_cleanup_finished",
                   process_count=len(managed_rows),
                   active_count=sum(
                       row.state == "running" and row.ownership == "owned"
                       for row in rows
                   ))
        return rows

    def close(self) -> tuple[ProcessRecord, ...]:
        with self._lock:
            if self._closed:
                return tuple(item.record for item in self._processes.values())
            self._closed = True
        return self.cleanup()

    reap_all = cleanup

    def record_observation(
        self,
        entity_id: str,
        value: object,
        *,
        observer_id: str,
        availability: str = "present",
        coverage: str = "complete",
        valid_for: float | None = None,
        entity_generation: int = 1,
        evidence_ref: str | None = None,
        journal_value_change: bool = True,
    ) -> EnvironmentObservation[object]:
        if availability not in {"present", "absent", "unknown", "unsupported", "denied", "error"}:
            raise ValueError("invalid observation availability")
        if coverage not in {"complete", "partial"}:
            raise ValueError("invalid observation coverage")
        if (isinstance(entity_generation, bool)
                or not isinstance(entity_generation, int)
                or entity_generation < 1):
            raise ValueError("entity_generation must be a positive integer")
        if (valid_for is not None
                and (not isinstance(valid_for, (int, float))
                     or isinstance(valid_for, bool)
                     or not math.isfinite(float(valid_for))
                     or valid_for < 0)):
            raise ValueError("valid_for must be a finite non-negative number")
        public_value = _safe_value(value)
        value_revision = digest({"availability": availability, "value": public_value})
        public_entity_id = safe_label(entity_id, fallback="entity")
        key = (public_entity_id, int(entity_generation))
        now = float(self.wall_clock())
        with self._lock:
            prior = self._observations.get(key)
            observation = EnvironmentObservation(
                observation_id=f"env-observation:{self._sequence + 1}:{public_entity_id}",
                entity_id=public_entity_id,
                entity_generation=int(entity_generation),
                observer_id=safe_label(observer_id, fallback="environment"),
                observer_version="1",
                availability=availability,
                freshness="fresh",
                coverage=coverage,
                observed_at=now,
                valid_until=(now + valid_for if valid_for is not None else None),
                value_revision=value_revision,
                value=_copy(public_value),
                evidence_ref=(safe_label(evidence_ref, fallback="evidence")
                              if evidence_ref else None),
            )
            self._observations[key] = observation
        value_changed = prior is None or prior.value_revision != value_revision or prior.availability != availability
        # Volatile sensors may update their exact value without changing the
        # audit-relevant availability. Keep their latest value and check
        # invalidation, but do not consume a global journal sequence number.
        if prior is None or prior.availability != availability or (
            journal_value_change and prior.value_revision != value_revision
        ):
            self._event(
                "observation_changed",
                public_entity_id,
                int(entity_generation),
                observation_id=observation.observation_id,
                previous_observation_id=prior.observation_id if prior else None,
            )
        if value_changed:
            self._invalidate_required_checks({public_entity_id})
        return observation

    def set_required_check(
        self,
        check_id: str,
        *,
        status: str,
        required: bool = True,
        predicate: str = "environment_check",
        evidence_ref: str | None = None,
        entity_id: str | None = None,
    ) -> None:
        normalized = str(status).lower()
        if normalized not in {"passed", "failed", "unknown", "expired"}:
            raise ValueError("environment check status must be passed, failed, unknown, or expired")
        if normalized == "passed" and not evidence_ref:
            raise ValueError("passed environment checks require evidence_ref")
        public_check_id = safe_label(check_id, fallback="check")
        row = {
            "check_id": public_check_id,
            "status": normalized,
            "required": bool(required),
            "predicate": safe_label(predicate, fallback="environment_check"),
            "evidence_ref": (safe_label(evidence_ref, fallback="evidence")
                              if evidence_ref else None),
            "entity_id": safe_label(entity_id, fallback="") if entity_id else None,
        }
        previous = self._required_checks.get(public_check_id)
        self._required_checks[public_check_id] = row
        if previous != row:
            self._event("readiness_changed", public_check_id, 1)

    def register_resource_predicate(self, predicate: ResourcePredicate) -> None:
        """Register a typed requirement without observing or executing anything."""
        self.environment_ledger.register_predicate(predicate)
        self._coverage.add("versioned-resource-predicates")

    def register_discovery(self, report: object) -> None:
        """Adopt host-produced discovery output without executing candidates."""
        for candidate in getattr(report, "recipes", ()):
            recipe = getattr(candidate, "recipe", candidate)
            if isinstance(recipe, SetupRecipe):
                self.register_recipe(recipe)
        for predicate in getattr(report, "predicates", ()):
            if isinstance(predicate, ResourcePredicate):
                self.register_resource_predicate(predicate)
        for binding in getattr(report, "bindings", ()):
            if isinstance(binding, RequirementBinding):
                self.bind_environment_check(binding)
        if tuple(getattr(report, "recipes", ())):
            self._coverage.add("manifest-recipe-provider")

    def register_recipe(self, recipe: SetupRecipe) -> None:
        """Store an evidence-bound candidate; admission remains a host decision."""
        self.recipe_registry.register(recipe)
        self._coverage.add("setup-recipe-registry")

    def effective_capability_receipt(self) -> dict[str, object]:
        """Report capabilities proven active by this owner instance."""
        adapters = {
            "resource_predicates": bool(self.environment_ledger.predicates()),
            "recipe_provider": bool(self.recipe_registry.all()),
            "freshness_bindings": bool(self.environment_ledger.facts()),
            "restore_adapter": False,
            "owned_process_lifecycle": True,
        }
        return {
            "schema": "stateframe-environment-capability/1",
            "owner": "kernel_environment_state",
            "effective": tuple(sorted(key for key, value in adapters.items() if value)),
            "adapters": adapters,
            "coverage": tuple(sorted(self._coverage)),
        }

    def bind_environment_check(self, binding: RequirementBinding) -> None:
        """Bind one check to predicates owned by this environment state."""
        self.environment_ledger.bind_check(binding)
        self._coverage.add("environment-check-bindings")

    def record_resource_fact(self, fact: ResourceFact) -> None:
        """Admit one host-produced fact and leave receipt/readiness ownership intact."""
        self.environment_ledger.put_fact(fact)
        self._coverage.add("versioned-resource-facts")

    def environment_applicability(
        self, check_id: str, *, receipt_vector: object | None = None,
        descriptor: object | None = None, fence_token: str = "fence:unknown",
    ) -> ApplicabilityResult:
        from kernel_environment_types import EnvironmentInputVector
        vector = receipt_vector if isinstance(receipt_vector, EnvironmentInputVector) else None
        return self.environment_ledger.applicability(
            check_id, receipt_vector=vector, descriptor=descriptor, fence_token=fence_token,
        )

    check_applicability = environment_applicability

    def ingest_receipts(self, receipts: Iterable[object]) -> tuple[ReceiptIndexRow, ...]:
        by_id: dict[str, ReceiptIndexRow] = {}
        content_by_id: dict[str, dict[str, object]] = {}
        file_digests: dict[str, str] = {}
        for raw in receipts:
            if not isinstance(raw, Mapping):
                continue
            receipt_id = raw.get("receipt_id") or raw.get("event_id") or raw.get("id")
            if not receipt_id:
                continue
            row = ReceiptIndexRow(
                receipt_id=safe_label(receipt_id, fallback="receipt"),
                kind=safe_label(raw.get("kind") or raw.get("tool") or "unknown", fallback="unknown"),
                result_class=safe_label(raw.get("result_class") or raw.get("result") or raw.get("status") or "unknown",
                                        fallback="unknown"),
                revision=safe_label(raw.get("revision") or raw.get("snapshot_digest") or "unknown",
                                    fallback="unknown"),
                owner="existing_receipt_owner",
            )
            by_id[row.receipt_id] = row
            path = str(raw.get("target_path") or raw.get("path") or "").replace("\\", "/")
            if path and path not in file_digests:
                file_digests[path] = _receipt_file_digest(self.root, path)
            content_by_id[row.receipt_id] = {
                "kind": row.kind, "result_class": row.result_class, "revision": row.revision,
                "workspace_revision": raw.get("workspace_revision"),
                "fingerprint": raw.get("fingerprint"), "path": path,
                "file_digest": file_digests.get(path),
            }
        # Compare semantic content before the bounded projection; random receipt
        # IDs must not control which rows survive overflow or change detection.
        ordered = sorted(by_id, key=lambda key: _canonical(content_by_id[key]))
        index_digest = digest([content_by_id[key] for key in ordered])
        normalized = tuple(by_id[key] for key in ordered[-MAX_RECEIPTS:])
        with self._lock:
            changed = index_digest != self._receipt_index_digest
            self._receipt_index_digest = index_digest
            self._receipt_rows = normalized
            if changed:
                self._event("receipt_index_changed", "receipts", 1)
        return normalized

    def bind_job_watch(self, job_watch: object | None) -> None:
        self._job_watch = job_watch
        if job_watch is not None:
            self._coverage.add("job-watch-projection")

    def _ingest_job_watch(self) -> None:
        watch = self._job_watch
        frame = getattr(watch, "frame", None)
        if not callable(frame):
            return
        try:
            payload = frame()
        except Exception as error:  # observer failure is typed and non-fatal
            self._omissions.add("job-watch-error:" + type(error).__name__)
            return
        rows = payload.get("jobs", ()) if isinstance(payload, Mapping) else ()
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            return
        output_for = getattr(watch, "job_output", None)
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            job_id = str(raw.get("job_id") or "")
            if not job_id:
                continue
            state = str(raw.get("state") or ("running" if raw.get("alive") else "exited"))
            if state == "completed":
                state = "exited"
            if state not in {"running", "exited", "failed", "killed"}:
                state = "unknown"
            output_tail = None
            output_truncated = False
            if callable(output_for) and state in {"exited", "failed", "killed"}:
                try:
                    output_payload = output_for(job_id, 32)
                    if isinstance(output_payload, Mapping):
                        raw_output = output_payload.get("output")
                        if raw_output:
                            output_tail = _safe_text(raw_output, self.tail_bytes)
                            output_truncated = len(str(raw_output)) > self.tail_bytes
                except Exception as error:  # existing owner remains authoritative
                    self._omissions.add("job-output-error:" + type(error).__name__)
            row = ProcessRecord(
                launch_id="job:" + safe_label(job_id, fallback="unknown"),
                entity_generation=1,
                role="background_job",
                pid=None,
                birth_identity="job-watch:" + safe_label(job_id, fallback="unknown"),
                containment_id="job-watch:" + safe_label(job_id, fallback="unknown"),
                command_label="background job",
                cwd_ref=".",
                state=state,
                started_at=float(raw.get("started") or 0.0),
                ended_at=None,
                exit_code=(raw.get("exit_code") if isinstance(raw.get("exit_code"), int) else None),
                output_tail=output_tail,
                output_truncated=output_truncated,
                ownership="job_watch",
            )
            key = row.launch_id
            with self._lock:
                prior = self._processes.get(key)
                changed = prior is None or prior.record.to_dict() != row.to_dict()
                if prior is None:
                    self._processes[key] = _OwnedProcess(
                        process=_UnavailableProcess(),
                        record=row,
                        started_mono=float(self.monotonic_clock()),
                        tail=deque(),
                        terminal_recorded=state != "running",
                    )
                elif changed and prior.record.ownership == "job_watch":
                    prior.record = row
            if changed:
                self._event("job_projection_changed", key, 1)

    def snapshot(self, *, receipts: Iterable[object] | None = None) -> EnvironmentSnapshot:
        self.refresh()
        self._expire_observations()
        self._ingest_job_watch()
        if receipts is not None:
            self.ingest_receipts(receipts)
        now = float(self.wall_clock())
        with self._lock:
            processes = tuple(self._processes[key].record for key in sorted(self._processes))
            observations = tuple(self._observations[key] for key in sorted(self._observations))
            events = tuple(self._events)
            receipt_rows = self._receipt_rows
            checks = tuple(_copy(self._required_checks[key]) for key in sorted(self._required_checks))
            resource_facts = tuple(item.to_dict() for item in self.environment_ledger.facts())
            predicates = tuple(item.to_dict() for item in self.environment_ledger.predicates())
            omissions = tuple(sorted(self._omissions))
            coverage = tuple(sorted(self._coverage))
            sequence = self._sequence
        return EnvironmentSnapshot(
            scope=self.scope,
            snapshot_seq=sequence,
            baseline_id=self.baseline_id,
            assembled_at=now,
            processes=processes,
            observations=observations,
            events=events,
            receipts=receipt_rows,
            required_checks=checks,
            resource_facts=resource_facts,
            predicates=predicates,
            omissions=omissions,
            coverage=coverage,
        )


class _UnavailableProcess:
    """Popen-shaped no-op used for projections owned by another organ."""

    pid = None
    returncode = None

    def poll(self) -> int | None:
        return None


__all__ = [
    "DEFAULT_EVENT_LIMIT", "DEFAULT_TAIL_BYTES", "EnvironmentEvent",
    "EnvironmentObservation", "EnvironmentScope", "EnvironmentSnapshot",
    "EnvironmentState", "MODE_ENV", "ProcessRecord", "ReceiptIndexRow",
    "SCHEMA", "digest", "enabled", "mode", "safe_label",
]
