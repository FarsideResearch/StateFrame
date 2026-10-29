"""Pure, bounded presentation of host-owned progress receipts.

This module is deliberately a reducer, not a ledger authority.  It accepts
already-recorded mappings, folds amendments at a supplied event cutoff, and
returns immutable views suitable for ordinary or F4 presentation.  It never
executes a check, resolves a requirement, authorizes an action, suppresses a
read, or contacts a provider.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


MODE_ENV = "STATEFRAME_PROGRESS_LEDGER"
SCHEMA = "stateframe-progress-ledger/1"
RENDER_SCHEMA = "stateframe-progress-render/1"
CHECK_STATES = (
    "NO_CHECK_RECORDED",
    "CHECK_PASS",
    "CHECK_FAIL",
    "CHECK_NOT_REACHED",
    "STALE",
    "BINDING_UNKNOWN",
)
DEFAULT_LIMITS = {"edits": 4, "requirements": 4, "checks": 3, "reads": 3}
DEFAULT_RENDER_TOKENS = 768
HARD_RENDER_TOKENS = 1024
_EVENT_KINDS = frozenset({
    "action_receipt",
    "action_receipt_amended",
    "completed_action",
    "observation_committed",
    "check_receipt",
    "snapshot_committed",
    "request_submitted",
})
_EDIT_KINDS = frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace", "patch"})
_READ_KINDS = frozenset({"read", "file_read", "cat", "open", "search", "grep", "find", "ripgrep", "rg"})
_CHECK_KINDS = frozenset({"check", "test", "run_tests", "pytest", "unittest", "verification", "probe"})
_PASS_WORDS = frozenset({"PASS", "PASSED", "GREEN", "SUCCESS", "OK", "CURRENT", "VERIFIED"})
_FAIL_WORDS = frozenset({"FAIL", "FAILED", "RED", "VIOLATED", "ERROR"})
_NOT_REACHED_WORDS = frozenset({
    "NOT_RUN", "NOT_REACHED", "NO_EXECUTION", "NO_EXECUTION_RECEIPT", "UNRUNNABLE",
    "TIMEOUT", "TIMED_OUT", "IMPORT", "COLLECTION", "PROCESS_INFRA_ERROR",
})
_OPEN_WORDS = frozenset({"OPEN", "UNKNOWN", "UNRESOLVED", "PENDING", "REQUIRED", "NEEDS_WORK", "BLOCKED"})
_MODEL_WORDS = frozenset({
    "MODEL", "MODEL_CLAIM", "MODEL_CLAIMED", "MODEL_REQUESTED", "MODEL_REPORTED",
    "ASSISTANT", "SOLVER", "USER",
})


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strictly validated OFF-default presentation switch."""
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    for name in ("mapping", "to_dict", "as_dict"):
        converter = getattr(value, name, None)
        if callable(converter):
            try:
                result = converter()
            except (AttributeError, TypeError, ValueError):
                result = None
            if isinstance(result, Mapping):
                return result
    return {}


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _upper(value: object, default: str = "") -> str:
    return _text(value, default).strip().upper().replace("-", "_")


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _seq(value: object, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    for key in ("sequence", "seq", "event_sequence", "event_seq", "ordinal"):
        if isinstance(value, Mapping):
            candidate = value.get(key)
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return candidate
    return default


def _cutoff_value(cutoff: object) -> int | None:
    if cutoff is None:
        return None
    if isinstance(cutoff, Mapping):
        cutoff = cutoff.get("sequence", cutoff.get("seq", cutoff.get("event_seq")))
    if isinstance(cutoff, int) and not isinstance(cutoff, bool):
        return cutoff
    raise TypeError("cutoff must be an integer, mapping, or None")


def _event_kind(event: Mapping[str, object]) -> str:
    # Deliberately read only the top-level kind.  Nested payload kinds are data,
    # not additional events in the replay stream.
    return _text(event.get("kind", event.get("event_kind", event.get("type", "")))).strip().lower()


def _event_payload(event: Mapping[str, object]) -> dict[str, object]:
    payload = event.get("payload")
    result = dict(payload) if isinstance(payload, Mapping) else {}
    for key, value in event.items():
        if key not in {"payload", "kind", "event_kind", "type"}:
            result[str(key)] = value
    return result


def _run_id(row: Mapping[str, object], default: str = "run:unknown") -> str:
    value = row.get("run_id", row.get("run", row.get("execution_run_id", default)))
    return _text(value, default) or default


def _receipt_id(row: Mapping[str, object], default: str) -> str:
    for key in ("receipt_id", "action_receipt_id", "check_receipt_id", "id", "event_id"):
        value = row.get(key)
        if value not in (None, ""):
            return _text(value)
    return default


def _identity(row: Mapping[str, object], *, default_run: str, sequence: int, prefix: str) -> tuple[str, str]:
    return _run_id(row, default_run), _receipt_id(row, f"{prefix}:{sequence}")


def _path(value: object) -> str:
    return _text(value).replace("\\", "/").lstrip("./")


def _binding_revision(binding: Mapping[str, object], path: str) -> object:
    if not path:
        return None
    for group_name in ("product_files", "harness_files", "environment_files", "files"):
        group = binding.get(group_name)
        if isinstance(group, Mapping) and path in group:
            item = group[path]
            if isinstance(item, Mapping):
                return item.get("revision", item.get("file_revision", item.get("hash")))
            return item
    direct = binding.get(path)
    if isinstance(direct, Mapping):
        return direct.get("revision", direct.get("file_revision", direct.get("hash")))
    return direct


def _row_binding(row: Mapping[str, object]) -> Mapping[str, object]:
    binding = row.get("binding", row.get("revision_binding", row.get("bindings")))
    if isinstance(binding, Mapping):
        return binding
    return {}


def _effective_binding(row: Mapping[str, object]) -> dict[str, object]:
    """Retain explicit product, harness, and environment bindings on receipts."""
    result = dict(_row_binding(row))
    for key in (
        "product_revision", "artifact_revision", "workspace_revision",
        "harness_revision", "test_harness_revision", "environment_revision",
    ):
        if row.get(key) is not None and key not in result:
            result[key] = row[key]
    return result


def _binding_value(binding: Mapping[str, object], kind: str, path: str = "") -> object:
    if path and kind == "product":
        # An explicitly named but absent file is unknown; do not borrow an
        # unrelated sole product revision and mislabel the check as stale.
        if any(key in binding for key in ("product_files", "files")):
            return _binding_revision(binding, path)
        value = _binding_revision(binding, path)
        if value is not None:
            return value
    aliases = {
        "product": ("product_revision", "artifact_revision", "workspace_revision"),
        "harness": ("harness_revision", "test_harness_revision"),
        "environment": ("environment_revision", "environment"),
    }
    for key in aliases[kind]:
        if binding.get(key) is not None:
            return binding[key]
    groups = {
        "product": ("product_files", "files"),
        "harness": ("harness_files",),
        "environment": ("environment_files",),
    }
    values: set[str] = set()
    for key in groups[kind]:
        group = binding.get(key)
        if isinstance(group, Mapping):
            values.update(_text(value) for value in group.values() if value not in (None, ""))
    return next(iter(values)) if len(values) == 1 else None


def _revision(row: Mapping[str, object], *keys: str) -> object:
    binding = _row_binding(row)
    for key in (*keys, "execution_revision", "measured_revision", "revision", "workspace_revision"):
        value = row.get(key)
        if value is not None:
            return value
        value = binding.get(key)
        if value is not None:
            return value
    return None


def _range_info(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row.get("range", row.get("span", row.get("observed_range")))
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    start = row.get("start_line", row.get("line"))
    end = row.get("end_line", start)
    if isinstance(start, int) and not isinstance(start, bool):
        result: dict[str, object] = {"start_line": start}
        if isinstance(end, int) and not isinstance(end, bool):
            result["end_line"] = end
        for key in ("start_byte", "end_byte", "byte_start", "byte_end"):
            if key in row:
                result[key] = row[key]
        return result
    return {}


def _files_for_edit(row: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    edit_view = _mapping(row.get("edit_view"))
    source = {**edit_view, **row} if edit_view else row
    raw = source.get("changes", source.get("files", source.get("changed_files")))
    rows = _sequence(raw)
    if not rows:
        rows = (source,)
    result: list[Mapping[str, object]] = []
    for item in rows:
        value = _mapping(item)
        path = _path(value.get("path", value.get("target_path", value.get("file"))))
        if not path and len(rows) > 1:
            continue
        result.append({
            "path": path,
            "role": _text(value.get("role", "product")),
            "before": value.get("before", value.get("before_revision")),
            "after": value.get("after", value.get("after_revision", value.get("revision"))),
            "before_hash": value.get("before_hash", value.get("before_revision")),
            "after_hash": value.get("after_hash", value.get("after_revision", value.get("revision"))),
            "symbol": value.get("symbol", value.get("observed_symbol", value.get("outline_symbol"))),
            "range": dict(_range_info(value)),
            "diff": value.get("diff", value.get("delta", "")),
            "operation": _text(value.get("operation", value.get("relation", "modified"))),
        })
    return tuple(result)


def _bounded_diff(value: object, limit: int = 1400) -> tuple[str, bool]:
    text = _text(value)
    if len(text) <= limit:
        return text, False
    left = max(1, limit // 2 - 1)
    right = max(1, limit - left - 1)
    return text[:left] + "…" + text[-right:], True


def _diff_identity(identity: tuple[str, str], row: Mapping[str, object], files: Sequence[Mapping[str, object]]) -> tuple[str, str, str]:
    raw_diff = row.get("diff", row.get("delta", ""))
    if not raw_diff:
        raw_diff = "\n".join(_text(item.get("diff")) for item in files if item.get("diff"))
    bounded, excerpt = _bounded_diff(raw_diff)
    diff_hash = "sha256:" + hashlib.sha256(_text(raw_diff).encode("utf-8")).hexdigest()
    return diff_hash, "diff_excerpt" if excerpt else "diff_full", bounded


def _symbol(row: Mapping[str, object], files: Sequence[Mapping[str, object]]) -> str:
    for value in (row.get("symbol"), row.get("observed_symbol"), row.get("outline_symbol")):
        if value not in (None, ""):
            return _text(value)
    for item in files:
        if item.get("symbol") not in (None, ""):
            return _text(item["symbol"])
    range_value = _range_info(row)
    if range_value:
        return "unresolved@" + _range_label(range_value)
    byte_values = [item.get("before_hash") for item in files if item.get("before_hash") is not None]
    return "unresolved@bytes" if byte_values else "<module>"


def _range_label(value: Mapping[str, object]) -> str:
    if value.get("start_line") is not None:
        end = value.get("end_line", value.get("start_line"))
        return f"L{value.get('start_line')}-L{end}"
    if value.get("start_byte") is not None or value.get("byte_start") is not None:
        return f"bytes {value.get('start_byte', value.get('byte_start'))}-{value.get('end_byte', value.get('byte_end', '?'))}"
    return "unknown-range"


def _explicit_overlap(row: Mapping[str, object]) -> str:
    if row.get("overlap_proven") is True or row.get("exact_overlap") is True:
        return "exact"
    return "unknown"


@dataclass(frozen=True)
class ProgressInputs:
    """Immutable, public-only input envelope for the projection."""

    events: tuple[Mapping[str, object], ...] = ()
    edits: tuple[Mapping[str, object], ...] = ()
    checks: tuple[Mapping[str, object], ...] = ()
    requirements: tuple[Mapping[str, object], ...] = ()
    reads: tuple[Mapping[str, object], ...] = ()
    current_binding: Mapping[str, object] = field(default_factory=dict)
    delivery_manifest: Mapping[str, object] = field(default_factory=dict)
    ticket_version: object = None
    dependency_identity: object = None
    presentation_version: str = SCHEMA

    def __post_init__(self) -> None:
        for name in ("events", "edits", "checks", "requirements", "reads"):
            value = tuple(dict(_mapping(item)) for item in getattr(self, name) if _mapping(item))
            object.__setattr__(self, name, value)
        object.__setattr__(self, "current_binding", dict(_mapping(self.current_binding)))
        object.__setattr__(self, "delivery_manifest", dict(_mapping(self.delivery_manifest)))
        if not isinstance(self.presentation_version, str) or not self.presentation_version.strip():
            object.__setattr__(self, "presentation_version", SCHEMA)

    @classmethod
    def from_value(cls, value: object) -> "ProgressInputs":
        if isinstance(value, cls):
            return value
        source = _mapping(value)
        return cls(
            events=tuple(dict(_mapping(item)) for item in _sequence(source.get("events", source.get("journal", ())))),
            edits=tuple(dict(_mapping(item)) for item in _sequence(source.get("edits", source.get("recent_changes", ())))),
            checks=tuple(dict(_mapping(item)) for item in _sequence(source.get("checks", source.get("recent_test_feedback", ())))),
            requirements=tuple(dict(_mapping(item)) for item in _sequence(source.get("requirements", source.get("obligations", ())))),
            reads=tuple(dict(_mapping(item)) for item in _sequence(source.get("reads", source.get("recent_reads", ())))),
            current_binding=_mapping(source.get("current_binding", source.get("binding", source.get("artifact_revisions", {})))),
            delivery_manifest=_mapping(source.get("delivery_manifest", source.get("final_delivery_manifest", {}))),
            ticket_version=source.get("ticket_version", source.get("ticket_revision")),
            dependency_identity=source.get("dependency_identity", source.get("dependencies")),
            presentation_version=_text(source.get("presentation_version", SCHEMA), SCHEMA),
        )


@dataclass(frozen=True)
class AppliedEditView:
    identity: tuple[str, str]
    sequence: int
    step: int | None
    paths: tuple[str, ...]
    symbol: str
    range_info: Mapping[str, object]
    before_hashes: tuple[str, ...]
    after_hashes: tuple[str, ...]
    diff_hash: str
    diff_label: str
    diff_excerpt: str
    relation: str
    overlap: str
    result: str
    currentness: str
    rejected: bool
    no_op: bool
    verification_ids: tuple[str, ...] = ()

    @property
    def row_id(self) -> str:
        return "edit:" + self.identity[0] + ":" + self.identity[1]

    def mapping(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "identity": list(self.identity),
            "sequence": self.sequence,
            "step": self.step,
            "paths": list(self.paths),
            "symbol": self.symbol,
            "range": dict(self.range_info),
            "before_hashes": list(self.before_hashes),
            "after_hashes": list(self.after_hashes),
            "diff_hash": self.diff_hash,
            "diff_label": self.diff_label,
            "diff_excerpt": self.diff_excerpt,
            "relation": self.relation,
            "overlap": self.overlap,
            "result": self.result,
            "currentness": self.currentness,
            "rejected": self.rejected,
            "no_op": self.no_op,
            "verification_ids": list(self.verification_ids),
        }


@dataclass(frozen=True)
class VerificationView:
    identity: tuple[str, str]
    execution_identity: str
    sequence: int
    selector: str
    state: str
    outcome: str
    failure_phase: str
    provenance: str
    executed: bool
    model_claimed: bool
    execution_revision: object
    current_revision: object
    binding: Mapping[str, object]
    edit_identity: tuple[str, str] | None
    historical_state: str
    # Optional provenance used by the final display join.  These remain host
    # facts; omission from older receipts must stay distinguishable from a
    # successful binding.
    path: str = ""
    operation: str = ""
    failed_condition: str = ""

    @property
    def row_id(self) -> str:
        return "check:" + self.identity[0] + ":" + self.identity[1]

    def mapping(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "identity": list(self.identity),
            "execution_identity": self.execution_identity,
            "sequence": self.sequence,
            "selector": self.selector,
            "state": self.state,
            "outcome": self.outcome,
            "failure_phase": self.failure_phase,
            "provenance": self.provenance,
            "executed": self.executed,
            "model_claimed": self.model_claimed,
            "execution_revision": self.execution_revision,
            "current_revision": self.current_revision,
            "binding": dict(self.binding),
            "edit_identity": list(self.edit_identity) if self.edit_identity else None,
            "historical_state": self.historical_state,
            "path": self.path,
            "operation": self.operation,
            "failed_condition": self.failed_condition,
        }


@dataclass(frozen=True)
class RequirementView:
    requirement_id: str
    exact_text: str
    source_id: str
    source_span: Mapping[str, object]
    provenance: str
    status: str
    open_reason: str | None
    edit_ids: tuple[str, ...]
    check_ids: tuple[str, ...]

    @property
    def row_id(self) -> str:
        return "requirement:" + self.requirement_id + ":" + self.source_id

    def mapping(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "requirement_id": self.requirement_id,
            "exact_text": self.exact_text,
            "source_id": self.source_id,
            "source_span": dict(self.source_span),
            "provenance": self.provenance,
            "status": self.status,
            "open_reason": self.open_reason,
            "edit_ids": list(self.edit_ids),
            "check_ids": list(self.check_ids),
        }


@dataclass(frozen=True)
class ReadView:
    identity: tuple[str, str]
    sequence: int
    file_identity: str | None
    path: str
    range_info: Mapping[str, object]
    partial: bool
    revision: object
    current_revision: object
    state: str
    delivered_now: bool
    delivery_known: bool
    reread_allowed: bool
    content_hash: str | None

    @property
    def row_id(self) -> str:
        return "read:" + self.identity[0] + ":" + self.identity[1]

    def mapping(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "identity": list(self.identity),
            "sequence": self.sequence,
            "file_identity": self.file_identity,
            "path": self.path,
            "range": dict(self.range_info),
            "partial": self.partial,
            "revision": self.revision,
            "current_revision": self.current_revision,
            "state": self.state,
            "delivered_now": self.delivered_now,
            "delivery_known": self.delivery_known,
            "reread_allowed": self.reread_allowed,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ProgressView:
    cutoff: int | None
    edits: tuple[AppliedEditView, ...]
    verifications: tuple[VerificationView, ...]
    requirements: tuple[RequirementView, ...]
    reads: tuple[ReadView, ...]
    input_receipt_ids: tuple[tuple[str, str], ...]
    current_binding: Mapping[str, object]
    delivery_manifest: Mapping[str, object]
    cache_key: str
    projection_digest: str

    def mapping(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "cutoff": self.cutoff,
            "edits": [item.mapping() for item in self.edits],
            "verifications": [item.mapping() for item in self.verifications],
            "requirements": [item.mapping() for item in self.requirements],
            "reads": [item.mapping() for item in self.reads],
            "input_receipt_ids": [list(item) for item in self.input_receipt_ids],
            "current_binding": dict(self.current_binding),
            "delivery_manifest": dict(self.delivery_manifest),
            "cache_key": self.cache_key,
            "projection_digest": self.projection_digest,
        }


@dataclass(frozen=True)
class RenderedProgress:
    text: str
    estimated_tokens: int
    projection_digest: str
    rendered_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]
    omission_manifest: tuple[Mapping[str, object], ...]
    token_budget: int
    final_wire_delivery: bool = False

    def mapping(self) -> dict[str, object]:
        return {
            "schema": RENDER_SCHEMA,
            "text": self.text,
            "estimated_tokens": self.estimated_tokens,
            "projection_digest": self.projection_digest,
            "rendered_ids": list(self.rendered_ids),
            "omitted_ids": list(self.omitted_ids),
            "omission_manifest": [dict(item) for item in self.omission_manifest],
            "token_budget": self.token_budget,
            "final_wire_delivery": self.final_wire_delivery,
        }


def _allowed_events(events: Sequence[Mapping[str, object]], cutoff: int | None) -> tuple[Mapping[str, object], ...]:
    rows = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or _event_kind(event) not in _EVENT_KINDS:
            continue
        sequence = _seq(event, index)
        if cutoff is not None and sequence > cutoff:
            continue
        rows.append((sequence, index, dict(event)))
    return tuple(row[2] for row in sorted(rows, key=lambda item: (item[0], item[1])))


def _fold_receipts(events: Sequence[Mapping[str, object]], cutoff: int | None) -> tuple[Mapping[str, object], ...]:
    folded: dict[tuple[str, str], dict[str, object]] = {}
    for index, event in enumerate(_allowed_events(events, cutoff)):
        kind = _event_kind(event)
        sequence = _seq(event, index)
        payload = _event_payload(event)
        nested = payload.get("receipt")
        row = dict(_mapping(nested) or payload)
        row.setdefault("run_id", event.get("run_id", event.get("run")))
        row.setdefault("sequence", sequence)
        row.setdefault("event_kind", kind)
        if kind == "check_receipt":
            row.setdefault("kind", "check")
        elif kind == "observation_committed":
            row.setdefault("kind", "read")
        identity = _identity(row, default_run=_run_id(event), sequence=sequence, prefix=kind)
        if kind == "action_receipt_amended":
            amended_id = row.get(
                "amends_receipt_id",
                row.get("amended_receipt_id", row.get("original_receipt_id", row.get("amendment_of"))),
            )
            if amended_id not in (None, ""):
                amended_identity = (
                    _text(row.get("amends_run_id", row.get("original_run_id", _run_id(event)))),
                    _text(amended_id),
                )
            else:
                amended_identity = identity
            base = dict(folded.get(amended_identity, folded.get(identity, {})))
            base.update(row)
            base["run_id"], base["receipt_id"] = amended_identity
            folded[amended_identity] = base
        elif kind == "completed_action":
            action = _mapping(payload.get("action"))
            base = dict(folded.get(identity, {}))
            base.update(action)
            base.update(row)
            base.setdefault("receipt_id", payload.get("receipt_id"))
            folded[identity] = base
        elif kind in {"action_receipt", "check_receipt", "observation_committed", "snapshot_committed", "request_submitted"}:
            base = dict(folded.get(identity, {}))
            base.update(row)
            folded[identity] = base
    return tuple(folded[key] for key in sorted(folded, key=lambda item: (item[0], item[1])))


def _direct_rows(inputs: ProgressInputs, name: str) -> tuple[Mapping[str, object], ...]:
    return tuple(getattr(inputs, name))


def _event_rows(events: Sequence[Mapping[str, object]], kinds: frozenset[str], cutoff: int | None) -> tuple[Mapping[str, object], ...]:
    result: list[Mapping[str, object]] = []
    for index, event in enumerate(_allowed_events(events, cutoff)):
        if _event_kind(event) not in kinds:
            continue
        payload = _event_payload(event)
        nested = payload.get("receipt", payload.get("observation", payload.get("requirement")))
        row = dict(_mapping(nested) or payload)
        event_kind = _event_kind(event)
        if event_kind == "check_receipt":
            row.setdefault("kind", "check")
        elif event_kind == "observation_committed":
            row.setdefault("kind", "read")
        edit_view = _mapping(row.get("edit_view"))
        if edit_view:
            for target, source in (
                ("diff", "diff"),
                ("path", "path"),
                ("symbol", "symbol"),
                ("step", "step"),
                ("after_revision", "revision"),
            ):
                if row.get(target) in (None, "") and edit_view.get(source) not in (None, ""):
                    row[target] = edit_view[source]
        row.setdefault("run_id", event.get("run_id", event.get("run")))
        row.setdefault("sequence", _seq(event, index))
        result.append(row)
    return tuple(result)


def _kind(row: Mapping[str, object]) -> str:
    action = _mapping(row.get("action"))
    value = row.get("kind", row.get("action_kind"))
    if value in (None, "") and action:
        value = action.get("action_kind", action.get("kind"))
    if value in (None, ""):
        value = row.get("tool", row.get("action", row.get("type", "")))
    return _text(value).strip().lower().replace("-", "_")


def _event_binding(events: Sequence[Mapping[str, object]], cutoff: int | None) -> dict[str, object]:
    """Use the latest explicit host binding available at this replay cutoff."""
    binding: dict[str, object] = {}
    for event in _allowed_events(events, cutoff):
        payload = _event_payload(event)
        for candidate in (event.get("current_binding"), payload.get("current_binding")):
            value = _mapping(candidate)
            if value:
                binding = dict(value)
    return binding


def _result(row: Mapping[str, object]) -> str:
    return _upper(row.get("result_class", row.get("result", row.get("status", row.get("outcome", "UNKNOWN")))), "UNKNOWN")


def _edit_views(rows: Sequence[Mapping[str, object]], binding: Mapping[str, object]) -> tuple[AppliedEditView, ...]:
    result: dict[tuple[str, str], AppliedEditView] = {}
    for index, raw in enumerate(rows):
        row = dict(raw)
        if _kind(row) not in _EDIT_KINDS and not row.get("changes") and not row.get("changed_files"):
            continue
        sequence = _seq(row, index)
        identity = _identity(row, default_run="run:unknown", sequence=sequence, prefix="edit")
        files = _files_for_edit(row)
        paths = tuple(_path(item.get("path")) for item in files if _path(item.get("path")))
        before = tuple(_text(item.get("before_hash")) for item in files if item.get("before_hash") is not None)
        after = tuple(_text(item.get("after_hash")) for item in files if item.get("after_hash") is not None)
        diff_hash, diff_label, diff_excerpt = _diff_identity(identity, row, files)
        range_value = _range_info(row)
        result_value = _result(row)
        rejected = result_value in {"REJECTED", "REFUSED", "DENIED", "ERROR"} or row.get("accepted") is False
        no_op = result_value in {"NO_OP", "NOOP", "UNCHANGED"} or row.get("no_op") is True
        currentness = "UNKNOWN"
        if paths:
            statuses = []
            for file_row in files:
                path = _path(file_row.get("path"))
                current = _binding_revision(binding, path)
                after_revision = file_row.get("after")
                if after_revision is None:
                    after_revision = file_row.get("after_hash")
                if after_revision is None:
                    after_revision = _revision(row, "after_revision", "after_hash")
                if current is None or after_revision is None:
                    statuses.append("UNKNOWN")
                else:
                    statuses.append("CURRENT" if str(current) == str(after_revision) else "STALE")
            currentness = "STALE" if "STALE" in statuses else "UNKNOWN" if "UNKNOWN" in statuses else "CURRENT"
        view = AppliedEditView(
            identity=identity,
            sequence=sequence,
            step=row.get("step", row.get("turn")) if isinstance(row.get("step", row.get("turn")), int) else None,
            paths=paths,
            symbol=_symbol(row, files),
            range_info=range_value,
            before_hashes=before,
            after_hashes=after,
            diff_hash=diff_hash,
            diff_label=diff_label,
            diff_excerpt=diff_excerpt,
            relation=_text(row.get("relation", row.get("edit_relation", "modified"))) or "modified",
            overlap=_explicit_overlap(row),
            result=result_value,
            currentness=currentness,
            rejected=rejected,
            no_op=no_op,
        )
        result[identity] = view
    return tuple(sorted(result.values(), key=lambda item: (item.sequence, item.identity)))


def _model_claim(row: Mapping[str, object]) -> bool:
    if row.get("model_claimed") is True or row.get("executed") is False:
        return True
    provenance = row.get("provenance", row.get("author", row.get("actor", "")))
    if isinstance(provenance, Mapping):
        provenance = provenance.get("kind", provenance.get("class", ""))
    return _upper(provenance) in _MODEL_WORDS


def _check_base_state(row: Mapping[str, object]) -> tuple[str, str, str, bool]:
    status = _result(row)
    reported_outcome = _upper(row.get("outcome", status), status)
    phase = _upper(row.get("failure_phase", row.get("phase", "")), "UNKNOWN")
    if _model_claim(row):
        return "CHECK_NOT_REACHED", "UNEXECUTED", phase or "UNEXECUTED", False
    if phase in {"IMPORT", "COLLECTION", "EXECUTION", "TIMEOUT", "RESOURCE", "ENVIRONMENT"} or status in _NOT_REACHED_WORDS:
        return "CHECK_NOT_REACHED", reported_outcome or status or "NOT_REACHED", phase or status or "UNKNOWN", False
    if status in _PASS_WORDS:
        return "CHECK_PASS", "PASS", phase or "NONE", True
    if status in _FAIL_WORDS or status in {"ASSERTION", "BEHAVIOR_FAIL", "FAILURE"}:
        return "CHECK_FAIL", "FAIL", phase or "ASSERTION", True
    return "CHECK_NOT_REACHED", status or "UNEXECUTED", phase or "UNKNOWN", False


def _check_views(rows: Sequence[Mapping[str, object]], edits: Sequence[AppliedEditView], binding: Mapping[str, object]) -> tuple[VerificationView, ...]:
    unique: dict[tuple[str, str], tuple[int, Mapping[str, object]]] = {}
    for index, raw in enumerate(rows):
        row = dict(raw)
        if _kind(row) not in _CHECK_KINDS and not row.get("selector") and not row.get("check_receipt_id"):
            continue
        sequence = _seq(row, index)
        identity = _identity(row, default_run="run:unknown", sequence=sequence, prefix="check")
        execution_identity = _text(row.get("execution_id", row.get("check_execution_id", row.get("execution_identity", identity[1]))))
        dedup_key = (identity[0], execution_identity)
        prior = unique.get(dedup_key)
        if prior is not None and (prior[0], prior[1].get("sequence", 0)) >= (sequence, _seq(prior[1])):
            continue
        unique[dedup_key] = (sequence, row)
    result: list[VerificationView] = []
    edit_by_identity = {item.identity: item for item in edits}
    for (run_id, execution_identity), (sequence, row) in unique.items():
        identity = _identity(row, default_run="run:unknown", sequence=sequence, prefix="check")
        state, outcome, phase, executed = _check_base_state(row)
        execution_revision = _revision(row, "execution_revision", "measured_revision", "product_revision")
        current_revision = execution_revision
        receipt_binding = _effective_binding(row)
        explicit_binding = bool(receipt_binding) or execution_revision is not None
        edit_identity = None
        edit_ref = row.get("edit_identity", row.get("edit_receipt_id", row.get("post_edit_receipt_id")))
        if isinstance(edit_ref, Sequence) and not isinstance(edit_ref, (str, bytes, bytearray)) and len(edit_ref) == 2:
            edit_identity = (_text(edit_ref[0]), _text(edit_ref[1]))
        elif edit_ref not in (None, ""):
            edit_identity = (run_id or _run_id(row), _text(edit_ref))
        linked_edit = edit_by_identity.get(edit_identity) if edit_identity is not None else None
        if state not in {"CHECK_NOT_REACHED"} and not _model_claim(row):
            if not explicit_binding or not binding:
                state = "BINDING_UNKNOWN"
            else:
                file_path = _path(row.get("path", row.get("target_path", "")))
                if not file_path and linked_edit is not None and linked_edit.paths:
                    file_path = linked_edit.paths[0]
                current_revision = _binding_value(binding, "product", file_path)
                binding_checks = (
                    ("product", _binding_value(receipt_binding, "product", file_path)),
                    ("harness", _binding_value(receipt_binding, "harness")),
                    ("environment", _binding_value(receipt_binding, "environment")),
                )
                unknown = current_revision is None or execution_revision is None
                stale = current_revision is not None and execution_revision is not None and str(current_revision) != str(execution_revision)
                for kind, recorded in binding_checks[1:]:
                    if recorded is None:
                        continue
                    current_value = _binding_value(binding, kind)
                    if current_value is None:
                        unknown = True
                    elif str(current_value) != str(recorded):
                        stale = True
                if unknown:
                    state = "BINDING_UNKNOWN"
                elif stale or row.get("stale") is True:
                    state = "STALE"
        provenance = "MODEL_CLAIM" if _model_claim(row) else _text(
            row.get("provenance", "HOST_AUTO" if row.get("auto_verify") is True or row.get("automatic") is True else "HOST_EXECUTION")
        )
        action = _mapping(row.get("action"))
        path = _path(row.get("path", row.get("target_path", "")))
        operation = _text(row.get(
            "operation", row.get("action_kind", action.get("action_kind", "") if action else "")
        ))
        failed_condition = _text(row.get(
            "failed_condition", row.get("failure_condition", row.get("condition", ""))
        ))
        result.append(VerificationView(
            identity=identity,
            execution_identity=execution_identity,
            sequence=sequence,
            selector=_text(row.get("selector", row.get("target", row.get("check", "")))),
            state=state,
            outcome=outcome,
            failure_phase=phase,
            provenance=provenance,
            executed=executed and not _model_claim(row),
            model_claimed=_model_claim(row),
            execution_revision=execution_revision,
            current_revision=current_revision,
            binding=receipt_binding,
            edit_identity=edit_identity if edit_identity in edit_by_identity else edit_identity,
            historical_state=state if state not in {"STALE", "BINDING_UNKNOWN"} else ("CHECK_PASS" if outcome == "PASS" else "CHECK_FAIL" if outcome == "FAIL" else "CHECK_NOT_REACHED"),
            path=path,
            operation=operation,
            failed_condition=failed_condition,
        ))
    for edit in edits:
        if any(item.edit_identity == edit.identity for item in result):
            continue
        synthetic_identity = (edit.identity[0], "no-check:" + edit.identity[1])
        result.append(VerificationView(
            identity=synthetic_identity,
            execution_identity=synthetic_identity[1],
            sequence=edit.sequence,
            selector="post-edit verification",
            state="NO_CHECK_RECORDED",
            outcome="NO_CHECK_RECORDED",
            failure_phase="NOT_RECORDED",
            provenance="HOST_ABSENCE",
            executed=False,
            model_claimed=False,
            execution_revision=None,
            current_revision=None,
            binding={},
            edit_identity=edit.identity,
            historical_state="NO_CHECK_RECORDED",
        ))
    return tuple(sorted(result, key=lambda item: (item.sequence, item.identity)))


def _requirement_views(rows: Sequence[Mapping[str, object]]) -> tuple[RequirementView, ...]:
    result: list[RequirementView] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(rows):
        row = dict(raw)
        requirement_id = _text(row.get("requirement_id", row.get("obligation_id", row.get("id", f"requirement-{index}"))))
        source_id = _text(row.get("source_id", row.get("ticket_source_id", row.get("source", "unknown"))))
        span = _mapping(row.get("source_span", row.get("span", row.get("locator", {}))))
        key = (requirement_id, source_id, _digest(span))
        if key in seen:
            continue
        seen.add(key)
        assessment = _mapping(row.get("assessment"))
        raw_status = _upper(assessment.get("status", row.get("status", row.get("state", "OPEN"))), "OPEN")
        status = "OPEN" if raw_status in _OPEN_WORDS else raw_status
        result.append(RequirementView(
            requirement_id=requirement_id,
            exact_text=_text(row.get("exact_text", row.get("quote", row.get("text", row.get("exact_clause", ""))))),
            source_id=source_id,
            source_span=dict(span),
            provenance=_text(row.get("provenance", row.get("provenance_class", "HOST_SOURCE"))),
            status=status,
            open_reason="EVIDENCE_UNRESOLVED" if status == "OPEN" else None,
            edit_ids=tuple(_text(item) for item in _sequence(row.get("edit_ids", row.get("edit_links", ())))),
            check_ids=tuple(_text(item) for item in _sequence(row.get("check_ids", row.get("check_links", ())))),
        ))
    return tuple(result)


def _manifest_is_final(manifest: Mapping[str, object]) -> bool:
    status = _upper(manifest.get("delivery_status", manifest.get("status", "")))
    return bool(
        manifest.get("final_wire_delivery") is True
        or manifest.get("submitted") is True
        or manifest.get("delivered") is True
        or status in {"SUBMITTED", "DELIVERED", "SENT", "SUCCESS"}
    )


def _delivery_ids(manifest: Mapping[str, object]) -> tuple[set[str], bool]:
    ids: set[str] = set()
    known = _manifest_is_final(manifest)
    if not known:
        return ids, False
    for key in ("delivered_ids", "delivered_item_ids", "sent_item_ids", "wire_item_ids", "rendered_ids"):
        value = manifest.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            ids.update(_text(item) for item in value)
    items = manifest.get("items")
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        for item in items:
            row = _mapping(item)
            if row and (row.get("delivered") is True or row.get("sent") is True or row.get("in_wire") is True):
                value = row.get("item_id", row.get("id"))
                if value not in (None, ""):
                    ids.add(_text(value))
    return ids, known


def _successful_request_submission(event: Mapping[str, object], payload: Mapping[str, object]) -> bool:
    status = _upper(payload.get("delivery_status", payload.get("status", payload.get("result", ""))))
    if payload.get("delivered") is False or payload.get("submitted") is False:
        return False
    if status in {"ATTEMPTED", "FAILED", "REJECTED", "ERROR", "NOT_SENT"}:
        return False
    if payload.get("delivered") is True or payload.get("submitted") is True:
        return True
    if status in {"SUBMITTED", "DELIVERED", "SENT", "SUCCESS"}:
        return True
    # Native request_submitted records carry the exact payload or its UTF-8
    # serialization.  A render attempt carries neither and is not delivery.
    return isinstance(payload.get("messages"), Sequence) or event.get("payload_utf8") not in (None, "")


def _event_delivery_manifest(events: Sequence[Mapping[str, object]], cutoff: int | None) -> dict[str, object]:
    delivered_ids: set[str] = set()
    final_wire_delivery = False
    for event in _allowed_events(events, cutoff):
        if _event_kind(event) != "request_submitted":
            continue
        payload = _event_payload(event)
        if not _successful_request_submission(event, payload):
            continue
        final_wire_delivery = True
        for key in ("delivered_ids", "rendered_ids", "item_ids", "receipt_ids"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                delivered_ids.update(_text(item) for item in value)
    result: dict[str, object] = {"final_wire_delivery": final_wire_delivery}
    if delivered_ids:
        result["delivered_ids"] = sorted(delivered_ids)
    return result


def projection_cache_key(inputs: object, cutoff: object = None) -> str:
    """Build a disposable cache identity from every evidence/version boundary."""
    envelope = ProgressInputs.from_value(inputs)
    cutoff_value = _cutoff_value(cutoff)
    material = {
        "schema": SCHEMA,
        "presentation_version": envelope.presentation_version,
        "cutoff": cutoff_value,
        "events": [dict(item) for item in _allowed_events(envelope.events, cutoff_value)],
        "edits": [dict(item) for item in envelope.edits],
        "checks": [dict(item) for item in envelope.checks],
        "requirements": [dict(item) for item in envelope.requirements],
        "reads": [dict(item) for item in envelope.reads],
        "binding": dict(envelope.current_binding),
        "delivery": dict(envelope.delivery_manifest),
        "ticket_version": envelope.ticket_version,
        "dependency_identity": envelope.dependency_identity,
    }
    return _digest(material)


def _read_views(rows: Sequence[Mapping[str, object]], binding: Mapping[str, object], manifest: Mapping[str, object]) -> tuple[ReadView, ...]:
    delivered, delivery_known = _delivery_ids(manifest)
    result: list[ReadView] = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        if _kind(row) not in _READ_KINDS and not row.get("read_evidence") and not row.get("file_identity") and not row.get("path"):
            continue
        evidence = _mapping(row.get("read_evidence"))
        merged = {**evidence, **row}
        sequence = _seq(merged, index)
        identity = _identity(merged, default_run="run:unknown", sequence=sequence, prefix="read")
        path = _path(merged.get("path", merged.get("file", merged.get("target_path", ""))))
        file_identity = merged.get("file_identity", merged.get("file_id"))
        file_identity_text = _text(file_identity) if file_identity not in (None, "") else (path or None)
        range_value = _range_info(merged)
        partial = bool(range_value) and (merged.get("total_lines") is not None or merged.get("partial") is True or merged.get("complete") is False)
        revision = merged.get("file_revision", merged.get("source_revision", merged.get("revision")))
        current_revision = _binding_revision(binding, path)
        explicit_state = _upper(merged.get("state", merged.get("freshness", "")))
        if not path or file_identity_text is None:
            state = "UNKNOWN"
        elif explicit_state in {"CHANGED", "UNCHANGED", "UNKNOWN"}:
            state = explicit_state
        elif current_revision is None or revision is None:
            state = "UNKNOWN"
        else:
            state = "UNCHANGED" if str(current_revision) == str(revision) else "CHANGED"
        row_id = "read:" + identity[0] + ":" + identity[1]
        delivered_now = delivery_known and bool({row_id, identity[1], file_identity_text} & delivered)
        result.append(ReadView(
            identity=identity,
            sequence=sequence,
            file_identity=file_identity_text,
            path=path,
            range_info=dict(range_value),
            partial=partial,
            revision=revision,
            current_revision=current_revision,
            state=state,
            delivered_now=delivered_now,
            delivery_known=delivery_known,
            reread_allowed=not delivered_now or state in {"CHANGED", "UNKNOWN"},
            content_hash=_text(merged.get("content_hash", merged.get("fingerprint", merged.get("sha256")))) or None,
        ))
    return tuple(sorted(result, key=lambda item: (item.sequence, item.identity)))


def _sort_checks(items: Sequence[VerificationView]) -> tuple[VerificationView, ...]:
    rank = {"CHECK_FAIL": 0, "STALE": 1, "BINDING_UNKNOWN": 2, "CHECK_NOT_REACHED": 3, "NO_CHECK_RECORDED": 4, "CHECK_PASS": 5}
    return tuple(sorted(items, key=lambda item: (rank.get(item.state, 6), item.sequence, item.identity)))


def project_progress(inputs: object, cutoff: object = None) -> ProgressView:
    """Fold bounded host facts into one deterministic, non-authoritative view."""
    envelope = ProgressInputs.from_value(inputs)
    cutoff_value = _cutoff_value(cutoff)
    cache_key = projection_cache_key(envelope, cutoff_value)
    current_binding = dict(envelope.current_binding) or _event_binding(envelope.events, cutoff_value)
    folded = _fold_receipts(envelope.events, cutoff_value)
    edit_rows = [*envelope.edits, *[row for row in folded if _kind(row) in _EDIT_KINDS or row.get("changes")]]
    check_rows = [*envelope.checks, *[row for row in folded if _kind(row) in _CHECK_KINDS or row.get("check_receipt_id")]]
    read_rows = [*envelope.reads, *[row for row in folded if _kind(row) in _READ_KINDS or row.get("read_evidence")]]
    requirement_rows = [*envelope.requirements]
    requirement_rows.extend(_event_rows(envelope.events, frozenset({"snapshot_committed"}), cutoff_value))
    event_delivery = _event_delivery_manifest(envelope.events, cutoff_value)
    delivery_manifest = dict(envelope.delivery_manifest)
    if event_delivery.get("final_wire_delivery") is True:
        delivery_manifest["final_wire_delivery"] = True
    delivered_ids = set(_sequence(delivery_manifest.get("delivered_ids")))
    delivered_ids.update(_sequence(event_delivery.get("delivered_ids")))
    if delivered_ids:
        delivery_manifest["delivered_ids"] = sorted(_text(item) for item in delivered_ids)
    edits = _edit_views(edit_rows, current_binding)
    checks = _check_views(check_rows, edits, current_binding)
    requirements = _requirement_views(requirement_rows)
    reads = _read_views(read_rows, current_binding, delivery_manifest)
    material = {
        "schema": SCHEMA,
        "cutoff": cutoff_value,
        "edits": [item.mapping() for item in edits],
        "verifications": [item.mapping() for item in checks],
        "requirements": [item.mapping() for item in requirements],
        "reads": [item.mapping() for item in reads],
        "binding": current_binding,
        "delivery": delivery_manifest,
        "ticket_version": envelope.ticket_version,
        "dependency_identity": envelope.dependency_identity,
        "presentation_version": envelope.presentation_version,
    }
    input_ids = sorted({item.identity for item in (*edits, *checks, *reads)}, key=lambda item: (item[0], item[1]))
    return ProgressView(
        cutoff=cutoff_value,
        edits=edits,
        verifications=_sort_checks(checks),
        requirements=requirements,
        reads=reads,
        input_receipt_ids=tuple(input_ids),
        current_binding=current_binding,
        delivery_manifest=delivery_manifest,
        cache_key=cache_key,
        projection_digest=_digest(material),
    )


def _row_id(item: object) -> str:
    return _text(getattr(item, "row_id", ""))


def _priority(item: object) -> tuple[int, int, str]:
    if isinstance(item, VerificationView):
        return ({"CHECK_FAIL": 0, "STALE": 1, "BINDING_UNKNOWN": 2, "CHECK_NOT_REACHED": 3, "NO_CHECK_RECORDED": 4, "CHECK_PASS": 5}.get(item.state, 6), item.sequence, item.row_id)
    if isinstance(item, AppliedEditView):
        return (0 if item.rejected or item.currentness in {"STALE", "UNKNOWN"} else 1, item.sequence, item.row_id)
    if isinstance(item, RequirementView):
        return (0 if item.status == "OPEN" else 1, len(item.exact_text), item.row_id)
    if isinstance(item, ReadView):
        return (0 if item.state in {"CHANGED", "UNKNOWN"} or not item.delivered_now else 1, item.sequence, item.row_id)
    return (9, 0, "")


def _json_data(value: object) -> str:
    """Serialize receipt data without leaving markup-looking ticket text."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _render_item(item: object, *, compact: bool, minimal: bool, diff_display: Mapping[str, str]) -> str:
    if isinstance(item, AppliedEditView):
        value = item.mapping()
        value["diff_display"] = diff_display.get(item.diff_hash, item.diff_hash)
        if compact:
            value.pop("diff_excerpt", None)
        if minimal:
            value = {key: value[key] for key in ("row_id", "paths", "symbol", "relation", "result", "currentness", "verification_ids")}
        return "- edit " + _json_data(value)
    if isinstance(item, VerificationView):
        value = item.mapping()
        if minimal:
            value = {key: value[key] for key in ("row_id", "state", "outcome", "historical_state", "failure_phase", "execution_revision", "current_revision", "edit_identity")}
        return "- check " + _json_data(value)
    if isinstance(item, RequirementView):
        value = item.mapping()
        if compact:
            value.pop("exact_text", None)
        if minimal:
            value = {key: value[key] for key in ("row_id", "requirement_id", "source_id", "status", "open_reason")}
        return "- requirement " + _json_data(value)
    if isinstance(item, ReadView):
        value = item.mapping()
        if minimal:
            value = {key: value[key] for key in ("row_id", "path", "range", "state", "delivered_now", "reread_allowed")}
        return "- read " + _json_data(value)
    return "- unknown " + _json_data(str(item))


def _render_text(view: ProgressView, rows: Mapping[str, Sequence[object]], omissions: Sequence[Mapping[str, object]], *, compact: bool, minimal: bool = False, diff_display: Mapping[str, str]) -> str:
    lines = ["[Progress ledger | presentation data | schema=" + SCHEMA + "]"]
    for name, label in (("requirements", "requirements"), ("edits", "applied_edits"), ("checks", "verification"), ("reads", "active_reads")):
        lines.append(label + ":")
        lines.extend(_render_item(item, compact=compact, minimal=minimal, diff_display=diff_display) for item in rows.get(name, ()))
        if not rows.get(name):
            lines.append("- none")
    if omissions:
        shown_handles = [_text(item.get("handle")) for item in omissions[:6]]
        suffix = ";more=" + str(len(omissions) - len(shown_handles)) if len(omissions) > len(shown_handles) else ""
        lines.append("omissions: count=" + str(len(omissions)) + " handles=" + ",".join(shown_handles) + suffix)
    else:
        lines.append("omissions: count=0 handles=none")
    lines.append("projection_digest: " + view.projection_digest)
    return "\n".join(lines)


def _render_floor_text(view: ProgressView, rows: Mapping[str, Sequence[object]], omissions: Sequence[Mapping[str, object]]) -> str:
    """Render a bounded identity/status floor when even compact rows overflow."""
    lines = ["[Progress ledger | presentation data | schema=" + SCHEMA + "]"]
    for name, label in (("requirements", "requirements"), ("edits", "applied_edits"), ("checks", "verification"), ("reads", "active_reads")):
        lines.append(label + ":")
        for item in rows.get(name, ()):
            if isinstance(item, RequirementView):
                detail = f"status={item.status};open={item.open_reason or 'none'};source={_text(item.source_id)}"
            elif isinstance(item, AppliedEditView):
                detail = f"result={item.result};currentness={item.currentness};relation={item.relation}"
            elif isinstance(item, VerificationView):
                detail = f"state={item.state};outcome={item.outcome};phase={item.failure_phase};selector={_text(item.selector)[:96]}"
            elif isinstance(item, ReadView):
                detail = f"state={item.state};delivered_now={str(item.delivered_now).lower()};reread={str(item.reread_allowed).lower()}"
            else:
                detail = "unknown"
            lines.append(f"- {name[:-1]} {_row_id(item)[:160]} {detail}")
        if not rows.get(name):
            lines.append("- none")
    handles = ",".join(_text(item.get("handle"))[:160] for item in omissions[:6]) or "none"
    more = len(omissions) - min(6, len(omissions))
    lines.append(f"omissions: count={len(omissions)} handles={handles}" + (f";more={more}" if more else ""))
    lines.append("projection_digest: " + view.projection_digest)
    return "\n".join(lines)


def _budget(value: object) -> tuple[int, dict[str, int]]:
    limits = dict(DEFAULT_LIMITS)
    if isinstance(value, Mapping):
        token_value = value.get("max_tokens", value.get("token_budget", DEFAULT_RENDER_TOKENS))
        for key in limits:
            raw = value.get(key)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
                limits[key] = raw
    elif isinstance(value, int) and not isinstance(value, bool):
        token_value = value
    else:
        token_value = DEFAULT_RENDER_TOKENS
    if not isinstance(token_value, int) or isinstance(token_value, bool):
        raise TypeError("render budget must contain an integer max_tokens")
    return max(1, min(HARD_RENDER_TOKENS, token_value)), limits


def render_progress(view: ProgressView, budget: object = None) -> RenderedProgress:
    """Render one deterministic data block with explicit overflow handles."""
    token_budget, limits = _budget(budget)
    all_rows: dict[str, tuple[object, ...]] = {
        "requirements": tuple(view.requirements),
        "edits": tuple(sorted(view.edits, key=_priority)),
        "checks": tuple(sorted(view.verifications, key=_priority)),
        "reads": tuple(sorted(view.reads, key=_priority)),
    }
    diff_hashes = sorted({item.diff_hash for item in view.edits})
    diff_display: dict[str, str] = {}
    for value in diff_hashes:
        digest = value.split(":", 1)[-1]
        width = min(12, len(digest))
        while any(other != value and other.split(":", 1)[-1][:width] == digest[:width] for other in diff_hashes) and width < len(digest):
            width += 4
        diff_display[value] = "sha256:" + digest[:width]
    selected: dict[str, list[object]] = {key: list(value[: limits[key]]) for key, value in all_rows.items()}
    omissions: list[dict[str, object]] = []
    for name, rows in all_rows.items():
        for item in rows[limits[name]:]:
            omissions.append({"handle": _row_id(item), "kind": name, "reason": "section_limit"})
    text = _render_text(view, selected, omissions, compact=False, diff_display=diff_display)
    if len(text) // 4 + 1 > token_budget:
        text = _render_text(view, selected, omissions, compact=True, diff_display=diff_display)
    if len(text) // 4 + 1 > token_budget:
        protected = {
            name: [item for item in rows if isinstance(item, (VerificationView, RequirementView)) and _priority(item)[0] <= 2]
            if name in {"checks", "requirements"} else [item for item in rows if _priority(item)[0] == 0]
            for name, rows in selected.items()
        }
        candidates = [(name, item) for name, rows in selected.items() for item in rows if item not in protected.get(name, [])]
        candidates.sort(key=lambda pair: (_priority(pair[1]), pair[0], _row_id(pair[1])), reverse=True)
        for name, item in candidates:
            if len(text) // 4 + 1 <= token_budget:
                break
            selected[name].remove(item)
            omission = {"handle": _row_id(item), "kind": name, "reason": "token_budget"}
            if omission not in omissions:
                omissions.append(omission)
            text = _render_text(view, selected, omissions, compact=True, diff_display=diff_display)
    rendered_ids = tuple(_row_id(item) for rows in selected.values() for item in rows)
    omitted_ids = tuple(_text(item.get("handle")) for item in omissions)
    estimate = len(text) // 4 + 1
    if estimate > token_budget:
        # Preserve the manifest and all protected identity fields even when the
        # caller deliberately requests an unusually small presentation budget.
        protected = {
            name: [item for item in rows if _priority(item)[0] <= 2]
            for name, rows in selected.items()
        }
        for name, rows in selected.items():
            for item in rows:
                if item not in protected[name]:
                    omission = {"handle": _row_id(item), "kind": name, "reason": "token_budget"}
                    if omission not in omissions:
                        omissions.append(omission)
            selected[name] = protected[name]
        text = _render_text(view, protected, omissions, compact=True, minimal=True, diff_display=diff_display)
        estimate = len(text) // 4 + 1
    if estimate > HARD_RENDER_TOKENS:
        text = _render_floor_text(view, selected, omissions)
        estimate = len(text) // 4 + 1
    rendered_ids = tuple(_row_id(item) for rows in selected.values() for item in rows)
    return RenderedProgress(
        text=text,
        estimated_tokens=estimate,
        projection_digest=view.projection_digest,
        rendered_ids=rendered_ids,
        omitted_ids=omitted_ids,
        omission_manifest=tuple(dict(item) for item in omissions),
        token_budget=token_budget,
    )


def inputs_from_frame(frame: Mapping[str, object]) -> ProgressInputs:
    """Adapt existing frame/StepLedger fields without reparsing source or journal text."""
    source = _mapping(frame.get("progress_inputs")) or frame
    return ProgressInputs.from_value(source)


def render_frame_progress(frame: Mapping[str, object], budget: object = None) -> RenderedProgress | None:
    if not isinstance(frame, Mapping):
        return None
    view = project_progress(inputs_from_frame(frame), frame.get("progress_cutoff"))
    if not (view.edits or view.verifications or view.requirements or view.reads):
        return None
    return render_progress(view, budget)


def replay_progress(events: Sequence[Mapping[str, object]], cutoffs: Sequence[int]) -> dict[int, ProgressView]:
    """Replay only the named top-level event kinds at each bounded cutoff."""
    return {int(cutoff): project_progress(ProgressInputs(events=tuple(events)), cutoff) for cutoff in cutoffs}


def presentation_manifest(view: ProgressView, rendered: RenderedProgress | None, *, final_wire_delivery: bool = False) -> dict[str, object]:
    """Return host telemetry for the presentation seam; no authority fields are added."""
    if rendered is None:
        return {
            "schema": RENDER_SCHEMA,
            "mode": "off",
            "cache_key": view.cache_key,
            "projection_digest": view.projection_digest,
            "input_receipt_ids": [list(item) for item in view.input_receipt_ids],
            "rendered_ids": [],
            "omitted_ids": [],
            "estimated_tokens": 0,
            "actual_tokens": None,
            "token_count_basis": "proxy:chars/4",
            "final_wire_delivery": False,
        }
    return {
        "schema": RENDER_SCHEMA,
        "mode": "on",
        "cache_key": view.cache_key,
        "projection_digest": view.projection_digest,
        "input_receipt_ids": [list(item) for item in view.input_receipt_ids],
        "rendered_ids": list(rendered.rendered_ids),
        "omitted_ids": list(rendered.omitted_ids),
        "estimated_tokens": rendered.estimated_tokens,
        "actual_tokens": rendered.estimated_tokens if final_wire_delivery else None,
        "token_count_basis": "proxy:chars/4",
        "final_wire_delivery": final_wire_delivery,
    }


__all__ = [
    "AppliedEditView", "CHECK_STATES", "DEFAULT_LIMITS", "MODE_ENV", "ProgressInputs",
    "ProgressView", "RENDER_SCHEMA", "RenderedProgress", "RequirementView", "ReadView",
    "SCHEMA", "VerificationView", "enabled", "inputs_from_frame", "mode",
    "presentation_manifest", "projection_cache_key", "project_progress", "render_frame_progress", "render_progress",
    "replay_progress",
]
