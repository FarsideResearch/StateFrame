"""Opt-in namespace boundaries for cross-run memory.

The normal StateFrame path does not import or construct this boundary's state
for model requests unless ``STATEFRAME_MEMORY_ISOLATION=on``.  When enabled,
memory is addressed by ``worker/task/run`` and only a target-bound,
host-constructed typed capsule may cross a namespace boundary.  A mapping that
merely calls itself memory is not a carry protocol.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


MODE_ENV = "STATEFRAME_MEMORY_ISOLATION"
SCHEMA = "stateframe-memory-isolation/1"
RECORD_SCHEMA = "stateframe-memory-record/1"
CAPSULE_FIELD = "memory_capsule"
MAX_IDENTIFIER = 160
MAX_RECORDS = 32
MAX_STRING = 4800
MAX_DEPTH = 5
MAX_VALUE_BYTES = 16000
MAX_CAPSULE_CHARS = 6000
CAPSULE_KINDS = frozenset({"explicit", "prior_attempt"})
CARRY_SCHEMA = "stateframe-memory-carry-envelope/1"
UNIDENTIFIED_PREFIX = "unidentified:"
CAPSULE_HEADER = "[Memory isolation capsule"
BLOCKED_NOTICE = "[Memory isolation] blocked"
# Rendered carry text that did not come from THIS boundary is removed in
# isolation mode (the legacy prior-attempt capsule header included).
RENDERED_CARRY_MARKERS = (CAPSULE_HEADER, "PRIOR ATTEMPTS: agent-world only")
RECORD_TYPES = frozenset({"observation", "failed_attempt", "procedure", "receipt", "prior_attempt"})
_IDENTIFIER_RE = re.compile(r"^[^\x00\r\n]{1,160}$")
_RECORD_TYPE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,47}$")
_RECORD_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")
_AUTHORITY_KEYS = frozenset({
    "allow", "allow_tools", "allowed_tools", "authorization", "authorizations",
    "capability", "capabilities", "credential", "credentials", "grant", "grants",
    "permission", "permissions", "tool_grant", "tool_grants", "tool_permissions",
})


class MemoryIsolationError(ValueError):
    """Base error for malformed namespace or capsule data."""


class NamespaceViolation(MemoryIsolationError):
    """Raised when a caller attempts an implicit cross-namespace read or carry."""


class CapsuleValidationError(MemoryIsolationError):
    """Raised when a typed capsule is malformed or fails its content seal."""


def mode(value: object = None) -> str:
    """Return the validated isolation switch; absent is the byte-preserving arm."""

    raw = os.environ.get(MODE_ENV, "off") if value is None else value
    if not isinstance(raw, str):
        raise ValueError(f"{MODE_ENV} must be off or on")
    value = raw.strip().lower()
    if value in {"off", "0"}:
        return "off"
    if value in {"on", "1"}:
        return "on"
    raise ValueError(f"{MODE_ENV} must be off or on")


def enabled(value: object = None) -> bool:
    return mode(value) == "on"


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    value = value.strip()
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{label} must be a nonempty single-line identifier")
    return value


def _canonical(value: object) -> str:
    return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class _FrozenMap(Mapping[str, object]):
    __slots__ = ("_items", "_values")

    def __init__(self, values: Mapping[str, object]):
        self._items = tuple(sorted(values.items(), key=lambda item: item[0]))
        self._values = dict(self._items)

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"_FrozenMap({dict(self._items)!r})"


def _normal_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _bounded(value: object, *, depth: int = 0, label: str = "value") -> object:
    if depth > MAX_DEPTH:
        raise CapsuleValidationError(f"{label} exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > MAX_STRING:
            raise CapsuleValidationError(f"{label} exceeds {MAX_STRING} characters")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CapsuleValidationError(f"{label} must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        if len(value) > 32:
            raise CapsuleValidationError(f"{label} has too many fields")
        normalized: dict[str, object] = {}
        for key, child in value.items():
            key_text = _identifier(str(key), f"{label} key")
            if _normal_key(key_text) in _AUTHORITY_KEYS:
                raise CapsuleValidationError(f"{label} contains an authority field")
            normalized[key_text] = _bounded(child, depth=depth + 1,
                                            label=f"{label}.{key_text}")
        return _FrozenMap(normalized)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 32:
            raise CapsuleValidationError(f"{label} has too many items")
        return tuple(_bounded(child, depth=depth + 1, label=f"{label}[{index}]")
                     for index, child in enumerate(value))
    raise CapsuleValidationError(f"{label} must be JSON-compatible")


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(child) for child in value]
    return value


def _record_type(value: object) -> str:
    value = _identifier(value, "record_type").lower()
    if not _RECORD_TYPE_RE.fullmatch(value):
        raise CapsuleValidationError("record_type has an invalid shape")
    if value not in RECORD_TYPES:
        raise CapsuleValidationError("record_type is not an approved memory type")
    return value


@dataclass(frozen=True)
class TaskNamespace:
    """The complete address of one worker's task/run memory."""

    worker_id: str
    task_id: str
    run_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_id", _identifier(self.worker_id, "worker_id"))
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))

    def to_dict(self) -> dict[str, str]:
        return {"worker_id": self.worker_id, "task_id": self.task_id, "run_id": self.run_id}

    @classmethod
    def from_dict(cls, value: object) -> "TaskNamespace":
        if not isinstance(value, Mapping):
            raise CapsuleValidationError("namespace must be an object")
        return cls(value.get("worker_id"), value.get("task_id"), value.get("run_id"))

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def label(self) -> str:
        return self.digest[:23]


@dataclass(frozen=True)
class MemoryRecord:
    """Immutable, advisory state that belongs to exactly one namespace."""

    record_type: str
    key: str
    value: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", _record_type(self.record_type))
        key = _identifier(self.key, "record key")
        if not _RECORD_KEY_RE.fullmatch(key):
            raise CapsuleValidationError("record key has an invalid shape")
        object.__setattr__(self, "key", key)
        value = _bounded(self.value)
        if len(_canonical(value).encode("utf-8")) > MAX_VALUE_BYTES:
            raise CapsuleValidationError("record value exceeds the bounded byte budget")
        object.__setattr__(self, "value", value)

    def to_dict(self) -> dict[str, object]:
        return {"schema": RECORD_SCHEMA, "record_type": self.record_type,
                "key": self.key, "value": _thaw(self.value)}

    @classmethod
    def from_dict(cls, value: object) -> "MemoryRecord":
        if not isinstance(value, Mapping) or value.get("schema") != RECORD_SCHEMA:
            raise CapsuleValidationError("memory record schema is invalid")
        return cls(value.get("record_type"), value.get("key"), value.get("value"))


def _capsule_payload(capsule: "TypedCarryCapsule") -> dict[str, object]:
    return {
        "schema": capsule.schema,
        "kind": capsule.kind,
        "source": capsule.source.to_dict(),
        "target": capsule.target.to_dict(),
        "records": [record.to_dict() for record in capsule.records],
        "reason": capsule.reason,
    }


@dataclass(frozen=True)
class TypedCarryCapsule:
    """A host-created, target-bound carry envelope."""

    source: TaskNamespace
    target: TaskNamespace
    records: tuple[MemoryRecord, ...]
    kind: str = "explicit"
    reason: str = ""
    schema: str = SCHEMA
    seal: str = ""

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise CapsuleValidationError("unsupported memory capsule schema")
        if not isinstance(self.kind, str) or self.kind not in CAPSULE_KINDS:
            raise CapsuleValidationError("unsupported memory capsule kind")
        if not isinstance(self.reason, str):
            raise CapsuleValidationError("memory capsule reason must be text")
        reason = self.reason.strip()
        if len(reason) > 240:
            raise CapsuleValidationError("memory capsule reason is too long")
        object.__setattr__(self, "reason", reason)
        rows = tuple(self.records)
        if not rows or len(rows) > MAX_RECORDS:
            raise CapsuleValidationError("memory capsule must contain 1-32 records")
        if any(not isinstance(record, MemoryRecord) for record in rows):
            raise CapsuleValidationError("memory capsule records must be typed records")
        if len({record.key for record in rows}) != len(rows):
            raise CapsuleValidationError("memory capsule record keys must be unique")
        object.__setattr__(self, "records", rows)
        expected = _digest(_capsule_payload(self))
        if self.seal and self.seal != expected:
            raise CapsuleValidationError("memory capsule seal does not match its contents")
        object.__setattr__(self, "seal", expected)

    @classmethod
    def build(cls, *, source: TaskNamespace, target: TaskNamespace,
              records: Iterable[MemoryRecord], kind: str = "explicit",
              reason: str = "") -> "TypedCarryCapsule":
        return cls(source=source, target=target, records=tuple(records),
                   kind=kind, reason=reason)

    def to_dict(self) -> dict[str, object]:
        payload = _capsule_payload(self)
        return {**payload, "seal": self.seal}

    @classmethod
    def from_dict(cls, value: object) -> "TypedCarryCapsule":
        if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
            raise CapsuleValidationError("memory capsule schema is invalid")
        raw_records = value.get("records")
        if not isinstance(raw_records, list):
            raise CapsuleValidationError("memory capsule records must be a list")
        seal = value.get("seal")
        if not isinstance(seal, str) or not seal:
            raise CapsuleValidationError("memory capsule seal is required")
        return cls(
            source=TaskNamespace.from_dict(value.get("source")),
            target=TaskNamespace.from_dict(value.get("target")),
            records=tuple(MemoryRecord.from_dict(item) for item in raw_records),
            kind=value.get("kind"), reason=value.get("reason", ""),
            seal=seal,
        )

    def render(self) -> str:
        lines = [f"[Memory isolation capsule | kind={self.kind} | target={self.target.label}]"]
        for record in self.records:
            value = _thaw(record.value)
            rendered = value if isinstance(value, str) else json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            lines.append(f"{record.record_type}:{record.key}={rendered}")
        text = "\n".join(lines)
        if len(text) > MAX_CAPSULE_CHARS:
            raise CapsuleValidationError("rendered memory capsule exceeds its bound")
        return text


class WorkerMemoryRegistry:
    """Worker-local registry whose reads always name the exact namespace."""

    def __init__(self, worker_id: str, *, max_namespaces: int = 128):
        self.worker_id = _identifier(worker_id, "worker_id")
        self.max_namespaces = max(1, int(max_namespaces))
        self._records: dict[TaskNamespace, dict[str, MemoryRecord]] = {}
        self._lock = threading.RLock()

    def _check(self, namespace: TaskNamespace) -> None:
        if not isinstance(namespace, TaskNamespace) or namespace.worker_id != self.worker_id:
            raise NamespaceViolation("namespace is not owned by this worker")

    def put(self, namespace: TaskNamespace, record: MemoryRecord) -> None:
        self._check(namespace)
        if not isinstance(record, MemoryRecord):
            raise TypeError("only typed memory records may enter the registry")
        with self._lock:
            if namespace not in self._records and len(self._records) >= self.max_namespaces:
                raise MemoryIsolationError("worker memory namespace capacity reached")
            self._records.setdefault(namespace, {})[record.key] = record

    def snapshot(self, namespace: TaskNamespace) -> tuple[MemoryRecord, ...]:
        self._check(namespace)
        with self._lock:
            rows = self._records.get(namespace, {})
            return tuple(rows[key] for key in sorted(rows))

    def read(self, requesting: TaskNamespace, owner: TaskNamespace, key: str) -> MemoryRecord:
        self._check(requesting)
        self._check(owner)
        if requesting != owner:
            raise NamespaceViolation("implicit cross-namespace memory read refused")
        with self._lock:
            try:
                return self._records.get(owner, {})[_identifier(key, "record key")]
            except KeyError as error:
                raise KeyError(key) from error

    def issue_capsule(self, source: TaskNamespace, target: TaskNamespace, *,
                      keys: Sequence[str] | None = None,
                      reason: str = "") -> TypedCarryCapsule:
        self._check(source)
        self._check(target)
        if not isinstance(reason, str) or not reason.strip():
            raise NamespaceViolation("a typed carry requires an explicit reason")
        with self._lock:
            available = self._records.get(source, {})
            selected = sorted(available) if keys is None else list(keys)
            if len(selected) > MAX_RECORDS:
                raise MemoryIsolationError("typed carry exceeds record bound")
            if len(set(selected)) != len(selected):
                raise NamespaceViolation("typed carry keys must be unique")
            try:
                records = tuple(available[_identifier(key, "record key")] for key in selected)
            except KeyError as error:
                raise NamespaceViolation("typed carry names a missing source record") from error
        return TypedCarryCapsule.build(source=source, target=target, records=records,
                                       reason=reason)

    def accept_capsule(self, target: TaskNamespace, capsule: TypedCarryCapsule) -> None:
        self._check(target)
        if not isinstance(capsule, TypedCarryCapsule):
            raise NamespaceViolation("only a typed carry capsule may cross a boundary")
        if capsule.target != target:
            raise NamespaceViolation("typed carry target does not match the active namespace")
        if capsule.source.worker_id != self.worker_id:
            raise NamespaceViolation("typed carry came from another worker identity")
        for record in capsule.records:
            self.put(target, record)


_REGISTRY_LOCK = threading.RLock()
_WORKER_REGISTRIES: dict[str, WorkerMemoryRegistry] = {}


def registry_for_worker(worker_id: str) -> WorkerMemoryRegistry:
    """Return the bounded worker registry; every record lookup still names a task namespace."""

    worker_id = _identifier(worker_id, "worker_id")
    with _REGISTRY_LOCK:
        return _WORKER_REGISTRIES.setdefault(worker_id, WorkerMemoryRegistry(worker_id))


def _row_text(row: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
            return str(value).strip()
    return None


def worker_identity_from_row(row: Mapping[str, object] | None = None, owner: object = None) -> str:
    row = row if isinstance(row, Mapping) else {}
    explicit = _row_text(row, "worker_id", "worker", "worker_identity")
    if explicit is None and owner is not None:
        value = getattr(owner, "worker_id", None)
        explicit = str(value).strip() if isinstance(value, str) and value.strip() else None
    if explicit is not None:
        return _identifier(explicit, "worker_id")
    configured = os.environ.get("STATEFRAME_WORKER_ID", "").strip()
    if configured:
        return _identifier(configured, "worker_id")
    # Thread identity is the safe fallback for the existing concurrent runner:
    # two tasks in different worker threads must not accidentally share a holder.
    return f"process:{os.getpid()}:thread:{threading.get_ident()}"


def explicit_task_identity(row: Mapping[str, object] | None) -> str | None:
    """The task row's persisted identity, or None.  Never derived from fn/sha:
    two tasks in different repos can share a function name and source sha."""
    if not isinstance(row, Mapping):
        return None
    value = _row_text(row, "task_id", "id")
    return _identifier(value, "task_id") if value is not None else None


def task_identity_from_row(row: Mapping[str, object] | None = None, owner: object = None) -> str:
    """Namespace task identity.  Unidentified rows get a distinct, non-carryable
    namespace (``unidentified:`` + row digest) instead of an fn@sha guess."""
    del owner  # owner attributes (e.g. a sha-prefix bolt id) are not task identity
    explicit = explicit_task_identity(row)
    if explicit is not None:
        return explicit
    row = row if isinstance(row, Mapping) else {}
    fields = {key: row.get(key) for key in ("repo", "fn", "rel", "path", "sha", "task_sha", "msg")
              if isinstance(row.get(key), (str, int)) and str(row.get(key)).strip()}
    return UNIDENTIFIED_PREFIX + hashlib.sha256(
        json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def namespace_for_owner(owner: object, *, run_id: str) -> TaskNamespace:
    row = getattr(owner, "task_row", {})
    row = row if isinstance(row, Mapping) else {}
    return TaskNamespace(worker_id=worker_identity_from_row(row, owner),
                         task_id=task_identity_from_row(row), run_id=run_id)


def _strip_rendered_carry(text: str, own: Iterable[str] = ()) -> tuple[str, int]:
    """Remove carry blocks that were not admitted by this boundary.

    A rendered header is presentation, never enforcement: in isolation mode the
    only carry the model sees is the one this boundary renders itself.  This
    boundary's own renders/notices are removed silently (re-rendered fresh);
    anything else carrying a carry header counts as blocked.
    """
    if not any(marker in text for marker in (*RENDERED_CARRY_MARKERS, BLOCKED_NOTICE)):
        return text, 0
    own = set(own)
    removed = 0
    kept = []
    for block in text.split("\n\n"):
        stripped = block.strip()
        if stripped in own or stripped.startswith(BLOCKED_NOTICE):
            continue
        if stripped.startswith(RENDERED_CARRY_MARKERS):
            removed += 1
            continue
        kept.append(block)
    return "\n\n".join(kept), removed


class MemoryIsolationBoundary:
    """Admit typed carries against the runtime's own namespace; scrub the rest.

    Admission happens once per capsule seal and is validated completely
    (namespace, typed records, rendered and aggregate size) BEFORE any registry
    mutation.  Rendering happens on every prepared request from the admitted
    set, so a carry is never consumed by an earlier (or non-provider) prepare.
    """

    def __init__(self, namespace: TaskNamespace, *, registry: WorkerMemoryRegistry | None = None):
        self.namespace = namespace
        self.registry = registry or WorkerMemoryRegistry(namespace.worker_id)
        self.blocked_count = 0
        self.accepted_count = 0
        self.rejections: list[str] = []
        self._admitted: dict[str, str] = {}   # seal -> pre-rendered text

    @property
    def carryable(self) -> bool:
        return not self.namespace.task_id.startswith(UNIDENTIFIED_PREFIX)

    def _reject(self, reason: str) -> bool:
        self.blocked_count += 1
        self.rejections.append(reason)
        return False

    def _admit(self, capsule: TypedCarryCapsule) -> bool:
        if capsule.seal in self._admitted:
            return True
        if not self.carryable:
            return self._reject("active task has no explicit identity")
        if capsule.target != self.namespace:
            return self._reject("carry target is not the active namespace")
        if capsule.source.worker_id != self.namespace.worker_id:
            return self._reject("carry came from another worker identity")
        try:
            rendered = capsule.render()
        except CapsuleValidationError:
            return self._reject("rendered carry exceeds its bound")
        total = sum(len(text) + 2 for text in self._admitted.values()) + len(rendered)
        if total > MAX_CAPSULE_CHARS:
            return self._reject("aggregate carry exceeds its bound")
        try:
            self.registry.accept_capsule(self.namespace, capsule)
        except MemoryIsolationError as error:
            return self._reject(str(error)[:120])
        self._admitted[capsule.seal] = rendered
        self.accepted_count += 1
        return True

    def admit_envelope(self, envelope: object) -> bool:
        """Bind a producer's target-less carry envelope to THIS namespace, then admit."""
        try:
            if not isinstance(envelope, Mapping) or envelope.get("schema") != CARRY_SCHEMA:
                raise CapsuleValidationError("carry envelope schema is invalid")
            source = TaskNamespace.from_dict(envelope.get("source"))
            if source.task_id != self.namespace.task_id:
                return self._reject("carry envelope belongs to another task")
            raw = envelope.get("records")
            if not isinstance(raw, list):
                raise CapsuleValidationError("carry envelope records must be a list")
            capsule = TypedCarryCapsule.build(
                source=source, target=self.namespace,
                records=tuple(MemoryRecord.from_dict(item) for item in raw),
                kind=envelope.get("kind"), reason=envelope.get("reason", ""))
        except (MemoryIsolationError, TypeError, ValueError) as error:
            return self._reject(str(error)[:120])
        return self._admit(capsule)

    def _admit_payload_capsule(self, value: object) -> None:
        try:
            capsule = value if isinstance(value, TypedCarryCapsule) else TypedCarryCapsule.from_dict(value)
        except (MemoryIsolationError, TypeError, ValueError) as error:
            self._reject(str(error)[:120])
            return
        self._admit(capsule)

    def _scrub(self, value: object) -> tuple[object, bool]:
        if isinstance(value, str):
            cleaned, removed = _strip_rendered_carry(value, self._admitted.values())
            if removed:
                self.blocked_count += removed
                self.rejections.extend(["unadmitted rendered carry"] * removed)
            return cleaned, cleaned != value
        if isinstance(value, Mapping):
            result: dict[object, object] = {}
            changed = False
            for key, child in value.items():
                normalized_key = _normal_key(str(key))
                if normalized_key in {"memory_capsule", "typed_memory_capsule"}:
                    self._admit_payload_capsule(child)
                    changed = True
                    continue
                if normalized_key == "memory":
                    if isinstance(child, TypedCarryCapsule) or (
                            isinstance(child, Mapping) and child.get("schema") == SCHEMA):
                        self._admit_payload_capsule(child)
                    else:
                        # A reserved memory field is never an implicit carry.
                        self._reject("untyped memory payload")
                    changed = True
                    continue
                cleaned, child_changed = self._scrub(child)
                result[key] = cleaned
                changed = changed or child_changed
            return result, changed
        if isinstance(value, (list, tuple)):
            rows, changed = [], False
            for child in value:
                cleaned, child_changed = self._scrub(child)
                rows.append(cleaned)
                changed = changed or child_changed
            return (tuple(rows) if isinstance(value, tuple) else rows), changed
        return value, False

    def scrub_payload(self, payload: object) -> object:
        cleaned, _ = self._scrub(payload)
        return cleaned

    @staticmethod
    def _append_notice(messages: list, notice: str) -> list:
        messages = [copy.deepcopy(dict(item)) if isinstance(item, Mapping)
                    else {"role": "user", "content": str(item)} for item in messages]
        for message in messages:
            if message.get("role") == "user":
                prior = message.get("content")
                message["content"] = (str(prior) + "\n\n" + notice) if prior else notice
                return messages
        messages.append({"role": "user", "content": notice})
        return messages

    def prepare(self, messages: Sequence[dict], uncapped: Sequence[dict]) -> tuple[list[dict], list[dict]]:
        """Scrub both views and render every admitted carry.  Never raises on
        carry content: rendering uses text validated at admission."""
        before, before_reasons = self.blocked_count, len(self.rejections)
        clean_messages, changed_messages = self._scrub(list(messages))
        blocked_messages = self.blocked_count - before
        clean_uncapped, changed_uncapped = self._scrub(list(uncapped))
        # ``uncapped`` is usually a superset view of the same conversation;
        # count each blocked payload once, not once per view.
        blocked = max(blocked_messages, self.blocked_count - before - blocked_messages)
        self.blocked_count = before + blocked
        del self.rejections[before_reasons + blocked:]
        if not (changed_messages or changed_uncapped or blocked or self._admitted):
            return list(messages), list(uncapped)
        sections: list[str] = []
        if blocked:
            sections.append(
                f"[Memory isolation] blocked {blocked} foreign or untyped memory payload(s); "
                "only task-local state and explicit typed carries are available.")
        sections.extend(self._admitted.values())
        if not sections:
            return clean_messages, clean_uncapped
        notice = "\n\n".join(sections)
        return self._append_notice(clean_messages, notice), self._append_notice(clean_uncapped, notice)


def _record_key(record: object) -> Mapping[str, object]:
    if not isinstance(record, Mapping):
        to_dict = getattr(record, "to_dict", None)
        record = to_dict() if callable(to_dict) else getattr(record, "__dict__", {})
    if not isinstance(record, Mapping):
        return {}
    key = record.get("key")
    return key if isinstance(key, Mapping) else record


def _record_world(record: object) -> Mapping[str, object]:
    world = record.get("world") if isinstance(record, Mapping) else getattr(record, "world", None)
    return world if isinstance(world, Mapping) else {}


def eligible_prior_records(records: Iterable[object], *, task_id: str, task_fn: str = "",
                           task_sha: str = "", repo: str | None = None) -> tuple[object, ...]:
    """Keep only records persisted with the active EXPLICIT task identity.

    attempt_memory labels an artifact without a persisted id as ``fn@sha``;
    that label is ambiguous across repos/tasks, so it is rejected outright.
    """
    expected = _identifier(task_id, "task_id")
    ambiguous = {f"{task_fn}@{task_sha}", f"{task_fn}@unknown", f"{task_fn}@"}
    result = []
    for record in records:
        value = _record_key(record).get("task_id")
        if not isinstance(value, str) or value.strip() != expected or expected in ambiguous:
            continue
        record_repo = _record_world(record).get("repo")
        if repo and record_repo and str(record_repo) != str(repo):
            continue
        result.append(record)
    return tuple(result)


def build_prior_carry_envelope(artifact_root: str, task_fn: str, task_sha: str,
                               current_attempt: int, *, row: Mapping[str, object],
                               worker_id: str, max_tokens: int = 800,
                               diagnostics: dict[str, object] | None = None) -> dict[str, object] | None:
    """Production producer: exact-task prior history as a TARGET-LESS typed envelope.

    The runtime binds the envelope to its own namespace (``admit_envelope``);
    nothing is rendered into the brief here.  No explicit task id -> no carry.
    """
    task_id = explicit_task_identity(row)
    if task_id is None:
        return None
    from detreason.attempt_capsule import render_capsule
    from detreason.attempt_memory import distill_prior_attempts

    distilled = tuple(distill_prior_attempts(artifact_root, task_fn, task_sha, current_attempt))
    records = eligible_prior_records(
        distilled, task_id=task_id, task_fn=task_fn, task_sha=task_sha,
        repo=_row_text(row, "repo"))
    if diagnostics is not None:
        # r44 M6: transcripts are file-named by fn, so a same-fn task can overwrite this
        # task's history.  Counts only (never foreign identities) so the runtime can
        # journal a refused/lost carry instead of dropping it silently.
        diagnostics.update({"distilled": len(distilled), "eligible": len(records),
                            "filtered": len(distilled) - len(records)})
    if not records:
        return None
    rendered = render_capsule(records, max_tokens=max_tokens)
    if not rendered:
        return None
    source = TaskNamespace(worker_id=_identifier(worker_id, "worker_id"), task_id=task_id,
                           run_id="prior-history")
    return {"schema": CARRY_SCHEMA, "kind": "prior_attempt", "source": source.to_dict(),
            "records": [MemoryRecord("prior_attempt", "summary", rendered).to_dict()],
            "reason": "same-task prior agent-world history"}


__all__ = [
    "CAPSULE_FIELD", "CARRY_SCHEMA", "CapsuleValidationError", "MemoryIsolationBoundary",
    "MemoryIsolationError", "MemoryRecord", "NamespaceViolation", "RECORD_SCHEMA",
    "SCHEMA", "TaskNamespace", "TypedCarryCapsule", "WorkerMemoryRegistry",
    "build_prior_carry_envelope", "eligible_prior_records", "enabled", "explicit_task_identity",
    "mode", "namespace_for_owner", "registry_for_worker", "task_identity_from_row",
    "worker_identity_from_row",
]
