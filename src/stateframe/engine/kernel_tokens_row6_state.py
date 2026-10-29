"""Bounded, advisory repetition state for the v29 TOKENS lane."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from kernel_tokens_row6_basis import Basis, KnownBasis, UnknownBasis

SCHEMA = "tokens-row6-state/1"
_EDIT_TOOLS = frozenset({"edit", "str_replace", "file_create", "write", "replace",
                         "replace_module"})
_OBSERVATION_TOOLS = frozenset({"read", "grep", "list", "ls", "probe", "search"})


class ResetReason(str, Enum):
    INITIAL = "initial"
    BASIS_CHANGED = "basis_changed"
    SOURCE_CHANGED = "source_changed"
    REQUIREMENTS_CHANGED = "requirements_changed"
    AUTHORITY_CHANGED = "authority_changed"
    NEW_EVIDENCE = "new_evidence"
    NEW_OBSERVATION_SPAN = "new_observation_span"
    REPAIR_CAPABILITY_CHANGED = "repair_capability_changed"
    EFFECT_UNRESOLVED = "effect_unresolved"
    SCOPE_MISMATCH = "scope_mismatch"
    INVALID_RESTORE = "invalid_restore"


@dataclass(frozen=True)
class RepeatEvent:
    kind: str
    action_id: str
    response_id: str
    signature: str | None
    count: int
    detected: bool
    guidance_enabled: bool
    reset_reason: ResetReason | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "action_id": self.action_id,
            "response_id": self.response_id,
            "signature": self.signature,
            "count": self.count,
            "detected": self.detected,
            "guidance_enabled": self.guidance_enabled,
            "reset_reason": self.reset_reason.value if self.reset_reason else None,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClosedAction:
    action_id: str
    kind: str
    tool: str
    result: str
    receipt_id: str
    before_digest: str
    after_digest: str
    effect: str
    observation_span: str

    def to_dict(self) -> dict[str, str]:
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "tool": self.tool,
            "result": self.result,
            "receipt_id": self.receipt_id,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "effect": self.effect,
            "observation_span": self.observation_span,
        }


def _text(value: object) -> str:
    return str(value).strip() if isinstance(value, (str, int, float, bool)) else ""


def _first(row: Mapping[str, object], names: Sequence[str]) -> object:
    for name in names:
        if row.get(name) is not None:
            return row[name]
    return None


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                         default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _record_mapping(record: object) -> Mapping[str, object]:
    return record if isinstance(record, Mapping) else {}


def _result(record: Mapping[str, object]) -> str:
    return _text(_first(record, ("result", "result_class", "status", "outcome"))).upper()


def _tool(record: Mapping[str, object]) -> str:
    return _text(_first(record, ("tool", "operation", "action"))).casefold()


def _receipt_id(record: Mapping[str, object]) -> str:
    value = _first(record, ("receipt_id", "step_record_id", "action_id"))
    receipts = record.get("receipt_ids")
    if not value and isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes)):
        value = next((item for item in receipts if _text(item)), None)
    return _text(value)


def _binding_digest(record: Mapping[str, object], name: str) -> str:
    value = record.get(name)
    if isinstance(value, Mapping):
        allowed = {
            str(key): value[key] for key in (
                "artifact", "product_revision", "source", "source_digest", "environment",
                "environment_revision", "harness", "test_harness_revision", "content_digest",
                "product_files", "harness_files",
            ) if key in value
        }
        return _digest(allowed) if allowed else ""
    return _text(value)


def _observation_span(record: Mapping[str, object]) -> str:
    return _text(_first(record, ("observation_span", "span", "cursor", "range")))


def _is_closed(record: Mapping[str, object]) -> bool:
    # Closure records completion of the action, not success of its result.
    # Negative and partial observations remain evidence and must not disappear
    # from the advisory ledger merely because they did not pass.
    return bool(record.get("closed", True))


def _is_no_op(record: Mapping[str, object]) -> bool:
    tool = _tool(record)
    before = _binding_digest(record, "before_binding")
    after = _binding_digest(record, "after_binding")
    explicit = _result(record) == "NO_OP"
    return _is_closed(record) and tool in _EDIT_TOOLS and bool(before and after and before == after) and (
        explicit or _text(record.get("effect")).upper() in {"NO_OP", "UNCHANGED"})


def _open_ids(record: Mapping[str, object]) -> tuple[str, ...]:
    values = _first(record, ("open_clause_ids", "open_clauses", "violations", "remaining"))
    if isinstance(values, Mapping):
        values = values.keys()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return tuple(sorted(dict.fromkeys(_text(item) for item in values if _text(item))))
    return ()


def _semantic_done_signature(basis: KnownBasis, record: Mapping[str, object]) -> str:
    """Bind refusal identity to semantic facts, never transport IDs or snapshots."""
    pending_effects = record.get("pending_effects", ())
    if not isinstance(pending_effects, Sequence) or isinstance(pending_effects, (str, bytes)):
        pending_effects = ()
    payload = {
        "basis": basis.semantic_key,
        "accepted": bool(record.get("accepted", False)),
        "disposition": _text(_first(record, ("disposition", "action", "result"))).upper(),
        "open_clause_ids": list(_open_ids(record)),
        "pending_effects": sorted(_text(item) for item in pending_effects),
        "repair_reachable": bool(record.get("repair_reachable", basis.repair_witness.reachable)),
        "witness_refs": list(basis.repair_witness.witness_refs),
    }
    return _digest(payload)


def _semantic_no_op_signature(basis: KnownBasis, record: Mapping[str, object]) -> str:
    """Bind unchanged-edit identity to normalized action and exact host bindings."""
    arguments = record.get("arguments")
    if arguments is None:
        arguments = record.get("action_parameters", {})
    return _digest({
        "basis": basis.semantic_key,
        "tool": _tool(record),
        "arguments": arguments,
        "arguments_hash": _text(record.get("arguments_hash")),
        "before_binding": _binding_digest(record, "before_binding"),
        "after_binding": _binding_digest(record, "after_binding"),
    })


def _basis_reset_reason(previous: KnownBasis, current: KnownBasis) -> ResetReason:
    if previous.request_identity != current.request_identity:
        return ResetReason.SCOPE_MISMATCH
    if previous.contract_revision != current.contract_revision:
        return ResetReason.REQUIREMENTS_CHANGED
    if previous.product != current.product:
        return ResetReason.SOURCE_CHANGED
    if (previous.harness != current.harness
            or previous.environment != current.environment
            or previous.selectors != current.selectors):
        return ResetReason.SOURCE_CHANGED
    if previous.tool_authority != current.tool_authority:
        return ResetReason.AUTHORITY_CHANGED
    if previous.clauses != current.clauses or previous.unexecuted_batch_suffix != current.unexecuted_batch_suffix:
        return ResetReason.REQUIREMENTS_CHANGED
    if previous.evidence != current.evidence:
        return ResetReason.NEW_EVIDENCE
    if previous.coverage != current.coverage:
        return ResetReason.NEW_OBSERVATION_SPAN
    if previous.repair_witness != current.repair_witness:
        return ResetReason.REPAIR_CAPABILITY_CHANGED
    if previous.effects != current.effects:
        return ResetReason.EFFECT_UNRESOLVED
    return ResetReason.BASIS_CHANGED


@dataclass
class RepeatState:
    """Per-run/request state with separate done and no-op counters.

    Counters are telemetry even when guidance is off.  No method changes an
    authority decision or removes an observation/tool opportunity.
    """

    run_id: str
    request_id: str
    max_signatures: int = 32
    guidance_enabled: bool = False
    _basis_key: str | None = field(default=None, init=False, repr=False)
    _basis_snapshot: KnownBasis | None = field(default=None, init=False, repr=False)
    _done_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _noop_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _seen_action_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _seen_response_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _detected_action_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _last_closed: ClosedAction | None = field(default=None, init=False, repr=False)
    _last_event: RepeatEvent | None = field(default=None, init=False, repr=False)
    _reset_reason: ResetReason | None = field(default=ResetReason.INITIAL, init=False, repr=False)

    def __post_init__(self) -> None:
        self.run_id = _text(self.run_id) or "UNKNOWN_RUN"
        self.request_id = _text(self.request_id) or "UNKNOWN_REQUEST"
        self.max_signatures = max(2, min(int(self.max_signatures), 256))

    @property
    def last_closed(self) -> ClosedAction | None:
        return self._last_closed

    @property
    def last_event(self) -> RepeatEvent | None:
        return self._last_event

    @property
    def reset_reason(self) -> ResetReason | None:
        return self._reset_reason

    def _reset(self, reason: ResetReason) -> None:
        self._done_counts.clear()
        self._noop_counts.clear()
        self._detected_action_ids.clear()
        self._basis_key = None
        self._basis_snapshot = None
        self._reset_reason = reason

    def _trim_seen_ids(self) -> None:
        limit = self.max_signatures * 4
        self._seen_action_ids = set(sorted(self._seen_action_ids)[-limit:])
        self._seen_response_ids = set(sorted(self._seen_response_ids)[-limit:])
        self._detected_action_ids = set(sorted(self._detected_action_ids)[-limit:])

    def _bind_basis(self, basis: Basis) -> tuple[str | None, ResetReason | None]:
        if isinstance(basis, UnknownBasis):
            return None, None
        key = basis.semantic_key
        reset: ResetReason | None = None
        if self._basis_key is not None and self._basis_key != key:
            reset = (_basis_reset_reason(self._basis_snapshot, basis)
                     if self._basis_snapshot is not None else ResetReason.BASIS_CHANGED)
            self._reset(reset)
        self._basis_key = key
        self._basis_snapshot = basis
        self._reset_reason = None
        return key, reset

    def observe_closed(self, action_id: str, record: Mapping[str, object],
                       basis: Basis | None = None) -> RepeatEvent:
        """Record one closed action; duplicate action IDs are idempotent."""
        normalized_id = _text(action_id) or _receipt_id(record)
        if normalized_id in self._seen_action_ids:
            event = RepeatEvent("closed_duplicate", normalized_id, "", None, 0, False,
                                self.guidance_enabled, reason="action_id_already_seen")
            self._last_event = event
            return event
        self._seen_action_ids.add(normalized_id)
        if len(self._seen_action_ids) > self.max_signatures * 4:
            self._seen_action_ids = set(sorted(self._seen_action_ids)[-self.max_signatures * 4:])
        row = _record_mapping(record)
        tool = _tool(row)
        result = _result(row)
        closed = _is_closed(row)
        action = ClosedAction(
            action_id=normalized_id,
            kind="no_op" if _is_no_op(row) else "observation" if tool in _OBSERVATION_TOOLS else "effect",
            tool=tool,
            result=result,
            receipt_id=_receipt_id(row),
            before_digest=_binding_digest(row, "before_binding"),
            after_digest=_binding_digest(row, "after_binding"),
            effect=_text(row.get("effect")),
            observation_span=_observation_span(row),
        )
        if closed:
            self._last_closed = action
        reset: ResetReason | None = None
        if basis is not None:
            _, reset = self._bind_basis(basis)
        event = RepeatEvent("closed", normalized_id, "", None, 1, False,
                            self.guidance_enabled, reset_reason=reset,
                            reason="closed" if closed else "not_closed")
        self._last_event = event
        return event

    def observe_done(self, *, action_id: str, response_id: str, basis: Basis,
                     accepted: bool, record: Mapping[str, object] | None = None) -> RepeatEvent:
        """Observe exactly one completed done action and detect semantic repeats."""
        normalized_action = _text(action_id) or "UNKNOWN_ACTION"
        normalized_response = _text(response_id) or "UNKNOWN_RESPONSE"
        if normalized_action in self._seen_action_ids or normalized_response in self._seen_response_ids:
            event = RepeatEvent("done_duplicate", normalized_action, normalized_response, None, 0,
                                False, self.guidance_enabled, reason="action_or_response_already_seen")
            self._last_event = event
            return event
        self._seen_action_ids.add(normalized_action)
        self._seen_response_ids.add(normalized_response)
        self._trim_seen_ids()
        if accepted or isinstance(basis, UnknownBasis):
            reason = "accepted" if accepted else "unknown_basis"
            event = RepeatEvent("done", normalized_action, normalized_response, None, 0, False,
                                self.guidance_enabled, reason=reason)
            self._last_event = event
            return event
        basis_key, reset = self._bind_basis(basis)
        if basis_key is None:
            event = RepeatEvent("done", normalized_action, normalized_response, None, 0, False,
                                self.guidance_enabled, reason="unknown_basis")
            self._last_event = event
            return event
        signature = _semantic_done_signature(basis, _record_mapping(record))
        counts = self._done_counts
        count = counts.get(signature, 0) + 1
        if len(counts) >= self.max_signatures and signature not in counts:
            oldest = next(iter(counts))
            counts.pop(oldest, None)
        counts[signature] = count
        detected = count >= 2
        if detected:
            self._detected_action_ids.add(normalized_action)
            self._trim_seen_ids()
        event = RepeatEvent("done", normalized_action, normalized_response, signature, count,
                            detected, self.guidance_enabled, reset_reason=reset,
                            reason="equivalent_refused_done" if detected else "first_refused_done")
        self._last_event = event
        return event

    def observe_no_op(self, *, action_id: str, response_id: str, basis: Basis,
                      record: Mapping[str, object]) -> RepeatEvent:
        """Track a proven unchanged edit separately from refused done actions."""
        normalized_action = _text(action_id) or "UNKNOWN_ACTION"
        normalized_response = _text(response_id) or "UNKNOWN_RESPONSE"
        if normalized_action in self._seen_action_ids or normalized_response in self._seen_response_ids:
            event = RepeatEvent("no_op_duplicate", normalized_action, normalized_response, None, 0,
                                False, self.guidance_enabled, reason="action_or_response_already_seen")
            self._last_event = event
            return event
        self._seen_action_ids.add(normalized_action)
        self._seen_response_ids.add(normalized_response)
        self._trim_seen_ids()
        if not _is_no_op(record) or isinstance(basis, UnknownBasis):
            event = RepeatEvent("no_op", normalized_action, normalized_response, None, 0, False,
                                self.guidance_enabled, reason="not_proven_no_op")
            self._last_event = event
            return event
        basis_key, reset = self._bind_basis(basis)
        if basis_key is None:
            event = RepeatEvent("no_op", normalized_action, normalized_response, None, 0, False,
                                self.guidance_enabled, reason="unknown_basis")
            self._last_event = event
            return event
        signature = _semantic_no_op_signature(basis, record)
        count = self._noop_counts.get(signature, 0) + 1
        if len(self._noop_counts) >= self.max_signatures and signature not in self._noop_counts:
            oldest = next(iter(self._noop_counts))
            self._noop_counts.pop(oldest, None)
        self._noop_counts[signature] = count
        detected = count >= 2
        event = RepeatEvent("no_op", normalized_action, normalized_response, signature, count,
                            detected, self.guidance_enabled, reset_reason=reset,
                            reason="equivalent_no_op" if detected else "first_no_op")
        self._last_event = event
        return event

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "max_signatures": self.max_signatures,
            "guidance_enabled": self.guidance_enabled,
            "basis_key": self._basis_key,
            "done_counts": dict(list(self._done_counts.items())[-self.max_signatures:]),
            "no_op_counts": dict(list(self._noop_counts.items())[-self.max_signatures:]),
            "seen_action_ids": sorted(self._seen_action_ids)[-self.max_signatures * 4:],
            "seen_response_ids": sorted(self._seen_response_ids)[-self.max_signatures * 4:],
            "last_closed": self._last_closed.to_dict() if self._last_closed else None,
            "last_event": self._last_event.to_dict() if self._last_event else None,
            "reset_reason": self._reset_reason.value if self._reset_reason else None,
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, object], *, run_id: str, request_id: str,
                guidance_enabled: bool = False) -> "RepeatState":
        state = cls(run_id, request_id, guidance_enabled=guidance_enabled)
        if (snapshot.get("schema") != SCHEMA or _text(snapshot.get("run_id")) != state.run_id
                or _text(snapshot.get("request_id")) != state.request_id):
            state._reset(ResetReason.INVALID_RESTORE)
            return state
        state.max_signatures = max(2, min(int(snapshot.get("max_signatures", 32)), 256))
        state._basis_key = _text(snapshot.get("basis_key")) or None
        for name, target in (("done_counts", state._done_counts), ("no_op_counts", state._noop_counts)):
            value = snapshot.get(name)
            if isinstance(value, Mapping):
                for key, count in list(value.items())[-state.max_signatures:]:
                    if _text(key) and type(count) is int and count > 0:
                        target[_text(key)] = count
        for name, target in (("seen_action_ids", state._seen_action_ids),
                             ("seen_response_ids", state._seen_response_ids)):
            value = snapshot.get(name)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                target.update(_text(item) for item in value if _text(item))
        closed = snapshot.get("last_closed")
        if isinstance(closed, Mapping):
            state._last_closed = ClosedAction(
                action_id=_text(closed.get("action_id")),
                kind=_text(closed.get("kind")),
                tool=_text(closed.get("tool")),
                result=_text(closed.get("result")),
                receipt_id=_text(closed.get("receipt_id")),
                before_digest=_text(closed.get("before_digest")),
                after_digest=_text(closed.get("after_digest")),
                effect=_text(closed.get("effect")),
                observation_span=_text(closed.get("observation_span")),
            )
        event = snapshot.get("last_event")
        if isinstance(event, Mapping):
            event_reason = _text(event.get("reset_reason"))
            try:
                reset_reason = ResetReason(event_reason) if event_reason else None
            except ValueError:
                state._reset(ResetReason.INVALID_RESTORE)
                return state
            signature = _text(event.get("signature")) or None
            count = event.get("count", 0)
            state._last_event = RepeatEvent(
                kind=_text(event.get("kind")),
                action_id=_text(event.get("action_id")),
                response_id=_text(event.get("response_id")),
                signature=signature,
                count=count if type(count) is int and count >= 0 else 0,
                detected=bool(event.get("detected", False)),
                guidance_enabled=bool(event.get("guidance_enabled", state.guidance_enabled)),
                reset_reason=reset_reason,
                reason=_text(event.get("reason")),
            )
        reason = _text(snapshot.get("reset_reason"))
        try:
            state._reset_reason = ResetReason(reason) if reason else None
        except ValueError:
            state._reset(ResetReason.INVALID_RESTORE)
        return state


__all__ = ["ClosedAction", "RepeatEvent", "RepeatState", "ResetReason", "SCHEMA"]
