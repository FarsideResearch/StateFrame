"""Append-only receipt and idempotency adapters for the System One facade.

The durable journal remains the record owner.  ``ExecutionLedger`` is only a
small in-process dispatch projection: a key becomes deduplicated after a
completed packet and is released for retry when no packet was dispatched or a
transport attempt failed.
"""

from __future__ import annotations

import hashlib
import json
import math
from threading import RLock
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Protocol

try:  # Flat v46 module loading.
    from kernel_continuation_journal import RESERVED, journal_safe_fields
except ImportError:  # pragma: no cover - package import path.
    from ..kernel_continuation_journal import RESERVED, journal_safe_fields

try:
    from .contracts import Answerability, QuestionType
except ImportError:  # pragma: no cover - direct module loading.
    from sys1.contracts import Answerability, QuestionType


SCHEMA = "sys1-facade-receipt/1"
RECORD_SCHEMA = "sys1.records/1"
DECISION_RECORDED = "decision_recorded"
DECISION_LABELLED = "decision_labelled"
RAW_SCORE_CACHE_SCHEMA = "sys1.raw-score-cache/1"


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class LedgerReservation:
    """A non-authoritative local reservation for one possible dispatch."""

    key: str
    reservation_id: str
    reserved_tokens: int


class ExecutionLedger:
    """Track completion only; skipped and failed attempts remain retryable."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._active: set[str] = set()
        self._completed: dict[str, str] = {}
        self._sequence = 0

    def begin(self, key: str, *, reserved_tokens: int = 1) -> LedgerReservation | None:
        if not isinstance(key, str) or not key:
            raise ValueError("ledger key must be non-empty")
        if isinstance(reserved_tokens, bool) or not isinstance(reserved_tokens, int) or reserved_tokens < 1:
            raise ValueError("reserved_tokens must be a positive integer")
        with self._lock:
            if key in self._completed or key in self._active:
                return None
            self._sequence += 1
            self._active.add(key)
            return LedgerReservation(key, f"sys1-reservation:{self._sequence}", reserved_tokens)

    def complete(self, reservation: LedgerReservation, receipt_id: str) -> None:
        with self._lock:
            if reservation.key not in self._active:
                raise ValueError("ledger reservation is not active")
            if not isinstance(receipt_id, str) or not receipt_id:
                raise ValueError("receipt_id must be non-empty")
            self._active.remove(reservation.key)
            self._completed[reservation.key] = receipt_id

    def release(self, reservation: LedgerReservation) -> None:
        """Release a non-dispatch or retryable failed attempt."""

        with self._lock:
            self._active.discard(reservation.key)

    def completed_receipt(self, key: str) -> str | None:
        with self._lock:
            return self._completed.get(key)


@dataclass(frozen=True, slots=True)
class FacadeReceipt:
    """Bounded accounting for one facade attempt."""

    receipt_id: str
    decision_id: str
    status: str
    dispatch_status: str
    reason: str | None = None
    request_sha256: str | None = None
    wire_sha256: str | None = None
    reserved_tokens: int = 0
    observed_tokens: int | None = None
    usage_status: str = "UNKNOWN"
    transport_status: str | None = None
    proposal_status: str | None = None
    owner_status: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "receipt_id": self.receipt_id,
            "decision_id": self.decision_id,
            "status": self.status,
            "dispatch_status": self.dispatch_status,
            "reason": self.reason,
            "request_sha256": self.request_sha256,
            "wire_sha256": self.wire_sha256,
            "reserved_tokens": self.reserved_tokens,
            "observed_tokens": self.observed_tokens,
            "usage_status": self.usage_status,
            "transport_status": self.transport_status,
            "proposal_status": self.proposal_status,
            "owner_status": self.owner_status,
        }


class ReceiptAdapter:
    """Forward receipts to the existing journal and retain a test projection."""

    def __init__(self, journal: object | None = None) -> None:
        self.journal = journal
        self._events: list[dict[str, object]] = []

    @property
    def events(self) -> tuple[Mapping[str, object], ...]:
        return tuple(dict(event) for event in self._events)

    def append(self, kind: str, **fields: object) -> dict[str, object]:
        if not isinstance(kind, str) or not kind:
            raise ValueError("receipt event kind must be non-empty")
        event = {"kind": kind, **dict(fields)}
        self._events.append(dict(event))
        emit = getattr(self.journal, "emit", None)
        if callable(emit):
            return dict(emit(kind, **dict(fields)))
        return event

    def record(self, receipt: FacadeReceipt) -> dict[str, object]:
        return self.append("system_one_receipt", receipt=receipt.to_dict())


def receipt_id(decision_id: str, attempt: int, status: str) -> str:
    """Create a stable bounded receipt identity without persisting request data."""

    return "sys1-receipt:" + _digest(
        {"decision_id": decision_id, "attempt": attempt, "status": status}
    )[7:31]


__all__ = [
    "ExecutionLedger",
    "FacadeReceipt",
    "LedgerReservation",
    "ReceiptAdapter",
    "SCHEMA",
    "receipt_id",
]


# ---------------------------------------------------------------------------
# Package-G immutable evidence/outcome projection


class RecordError(ValueError):
    """An immutable record or projection boundary is invalid."""


class JournalRecordError(RecordError):
    """A record cannot safely cross the existing journal boundary."""


class CacheIdentityError(RecordError):
    """A raw-score cache operation is not identity-safe."""


class CacheMissError(CacheIdentityError):
    """No exact raw-score cache entry exists."""


class StaleConsumptionError(CacheIdentityError):
    """A historical result lost its host-owned freshness basis."""

    def __init__(self, check: "ConsumptionCheck") -> None:
        super().__init__("stale decision basis: " + ", ".join(check.stale_reasons))
        self.check = check


class ArtifactMismatchError(CacheIdentityError):
    """A calibrated/action artifact differs from the recorded artifact."""


class IdempotencyConflictError(RecordError):
    """One immutable event identity was reused for different content."""


class Disposition(str, Enum):
    ACCEPT = "ACCEPT"
    ABSTAIN = "ABSTAIN"
    FALLBACK = "FALLBACK"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


class ObservationStatus(str, Enum):
    OBSERVED = "OBSERVED"
    UNCHOSEN = "UNCHOSEN"
    CENSORED = "CENSORED"
    UNRESOLVED = "UNRESOLVED"
    CONTESTED = "CONTESTED"
    NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"


def _rtext(value: object, name: str, *, required: bool = True, limit: int = 4096) -> str:
    if not isinstance(value, str):
        if required:
            raise RecordError(f"{name} must be a string")
        return ""
    result = value.strip()
    if required and not result:
        raise RecordError(f"{name} must not be empty")
    if "\x00" in result or "\n" in result or "\r" in result:
        raise RecordError(f"{name} contains a forbidden control character")
    if len(result) > limit:
        raise RecordError(f"{name} exceeds {limit} characters")
    return result


def _rfinite(value: object, name: str, *, minimum: float | None = None,
             maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise RecordError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise RecordError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise RecordError(f"{name} must be <= {maximum}")
    return result


def _rfreeze(value: object, name: str = "value") -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise RecordError(f"{name} keys must be non-empty strings")
            result[key] = _rfreeze(item, f"{name}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(_rfreeze(item, f"{name}[]") for item in value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise RecordError(f"{name} contains a non-finite number")
        return value
    raise RecordError(f"{name} must contain JSON-shaped values")


def _rmap(value: object, name: str) -> dict[str, object]:
    frozen = _rfreeze(value, name)
    if not isinstance(frozen, Mapping):
        raise RecordError(f"{name} must be a mapping")
    return {str(key): item for key, item in frozen.items()}


def _rthaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _rthaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_rthaw(item) for item in value]
    return value


def _rdigest(value: object) -> str:
    encoded = json.dumps(_rthaw(_rfreeze(value)), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _rseq(value: object, name: str, *, identifiers: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RecordError(f"{name} must be a sequence")
    result = tuple(_rtext(item, f"{name}[]") for item in value)
    if len(set(result)) != len(result):
        raise RecordError(f"{name} must not contain duplicates")
    if identifiers and any(any(char.isspace() for char in item) for item in result):
        raise RecordError(f"{name} must contain stable identifiers")
    return result


def _renum(value: object, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    result = _rtext(value, name).upper()
    if result not in {str(item.value) for item in enum_type}:
        raise RecordError(f"{name} has an unsupported value")
    return result


def _rquestion_type(value: object) -> str:
    return _renum(value, QuestionType, "question_type")


def _ranswerability(value: object) -> str:
    return _renum(value, Answerability, "answerability")


def _safe_public_evidence(value: object) -> tuple[Mapping[str, object], ...]:
    allowed = frozenset({
        "id", "item_id", "receipt_id", "source_ref", "content_digest", "digest", "sha256",
        "visibility", "role", "selection_rank", "omission_reason", "presented_at",
    })
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RecordError("presented_evidence must be a sequence")
    rows: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise RecordError(f"presented_evidence[{index}] must be a mapping")
        safe = {key: item[key] for key in allowed if key in item}
        if not safe:
            raise RecordError("presented evidence needs a public reference or digest")
        rows.append(MappingProxyType(_rmap(safe, "presented_evidence[]")))
    return tuple(rows)


def _safe_receipt(value: object, name: str) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        return _rtext(value, name, limit=512)
    if not isinstance(value, Mapping):
        raise RecordError(f"{name} must be a receipt mapping or ID")
    allowed = frozenset({
        "id", "receipt_id", "kind", "status", "result", "digest", "sha256", "timestamp",
        "timestamp_ns", "observed_at", "at", "usage_known", "prompt_tokens",
        "completion_tokens", "reasoning_tokens", "total_tokens", "cached_tokens",
        "queue_ms", "prefill_ms", "response_ms", "wall_ms", "dispatch_id", "request_id",
    })
    safe = {key: value[key] for key in allowed if key in value}
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "cached_tokens"):
        item = safe.get(key)
        if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 0):
            raise RecordError(f"{name}.{key} must be a non-negative integer")
    return _rfreeze(safe, name)


def _safe_receipts(value: Mapping[str, object] | None) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType({
        _rtext(key, "receipts.key", limit=64): _safe_receipt(item, f"receipts.{key}")
        for key, item in value.items()
    })


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Complete immutable observation; it has no permission or dispatch power."""

    decision_id: str
    task_id: str
    site_id: str
    question_id: str
    question_version: str
    state_epoch: str
    prompt_digest: str
    evidence_digest: str
    model_identity: str
    tokenizer_identity: str
    template_identity: str
    score_identity: str
    run_id: str | None = None
    parent_id: str | None = None
    schema_id: str = "sys1.records"
    schema_version: str = "1"
    question_type: str = "CHOICE"
    question_digest: str = ""
    evidence_snapshot_ref: str = ""
    presented_evidence: Sequence[Mapping[str, object]] = ()
    public_source_refs: Sequence[str] = ()
    omissions: Sequence[str] = ()
    options: Sequence[Mapping[str, object]] = ()
    semantic_definitions: Mapping[str, object] = field(default_factory=dict)
    label_token_map: Mapping[str, object] = field(default_factory=dict)
    row_order: Sequence[str] = ()
    permutation_seed: str | int | None = None
    permutation_identity: str = "identity"
    model_build: str = ""
    score_domain: str = "raw_pre_sampler_logits"
    score_position: str = "first_generated"
    scorer_identity: str = ""
    sampling_settings: Mapping[str, object] = field(default_factory=dict)
    raw_scores: Mapping[str, float] | None = None
    complete_ids: Sequence[str] = ()
    missing_ids: Sequence[str] = ()
    legal_mass: float | None = None
    calibrated_values: Mapping[str, float] | None = None
    calibration_id: str | None = None
    policy_id: str | None = None
    predicted_action: str | None = None
    disposition: str = Disposition.UNAVAILABLE.value
    disposition_reason: str = ""
    answerability: str = "ANSWERABLE"
    acceptable_option_ids: Sequence[str] = ()
    loss: Mapping[str, object] = field(default_factory=dict)
    loss_estimate: float | None = None
    applied: bool = False
    receipts: Mapping[str, object] = field(default_factory=dict)
    admission_receipt: object = None
    dispatch_receipt: object = None
    response_receipt: object = None
    consumption_receipt: object = None
    receipt_timestamps: Mapping[str, object] = field(default_factory=dict)
    reserved_tokens: int | None = None
    actual_prompt_tokens: int | None = None
    actual_completion_tokens: int | None = None
    actual_reasoning_tokens: int | None = None
    usage_known: bool = False
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    dependency_digests: Mapping[str, str] = field(default_factory=dict)
    graph_watermark: str = ""
    permissions_digest: str = ""
    claimed_attention_refs: Sequence[str] = ()
    claimed_attention_status: str = "UNVERIFIED"
    provenance: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("decision_id", "task_id", "site_id", "question_id", "question_version",
                     "state_epoch", "prompt_digest", "evidence_digest", "model_identity",
                     "tokenizer_identity", "template_identity", "score_identity",
                     "graph_watermark", "permissions_digest"):
            object.__setattr__(self, name, _rtext(getattr(self, name), name, limit=1024))
        object.__setattr__(self, "schema_id", _rtext(self.schema_id, "schema_id", limit=256))
        object.__setattr__(self, "schema_version", _rtext(self.schema_version, "schema_version", limit=64))
        object.__setattr__(self, "question_type", _rquestion_type(self.question_type))
        object.__setattr__(self, "answerability", _ranswerability(self.answerability))
        object.__setattr__(self, "question_digest", _rtext(self.question_digest, "question_digest", required=False, limit=1024))
        object.__setattr__(self, "evidence_snapshot_ref", _rtext(self.evidence_snapshot_ref, "evidence_snapshot_ref", required=False, limit=1024))
        object.__setattr__(self, "run_id", None if self.run_id is None else _rtext(self.run_id, "run_id", limit=1024))
        object.__setattr__(self, "parent_id", None if self.parent_id is None else _rtext(self.parent_id, "parent_id", limit=1024))
        object.__setattr__(self, "presented_evidence", _safe_public_evidence(self.presented_evidence))
        object.__setattr__(self, "public_source_refs", _rseq(self.public_source_refs, "public_source_refs"))
        object.__setattr__(self, "omissions", _rseq(self.omissions, "omissions"))
        object.__setattr__(self, "options", tuple(_rfreeze(item, "options[]") for item in self.options))
        object.__setattr__(self, "semantic_definitions", _rfreeze(self.semantic_definitions, "semantic_definitions"))
        object.__setattr__(self, "label_token_map", _rfreeze(self.label_token_map, "label_token_map"))
        object.__setattr__(self, "row_order", _rseq(self.row_order, "row_order", identifiers=True))
        if self.permutation_seed is not None and not isinstance(self.permutation_seed, (str, int)):
            raise RecordError("permutation_seed must be text, integer, or null")
        for name in ("permutation_identity", "model_build", "score_domain", "score_position", "scorer_identity"):
            object.__setattr__(self, name, _rtext(getattr(self, name), name))
        object.__setattr__(self, "sampling_settings", _rfreeze(self.sampling_settings, "sampling_settings"))
        if self.raw_scores is not None:
            object.__setattr__(self, "raw_scores", MappingProxyType({
                str(key): _rfinite(value, f"raw_scores.{key}") for key, value in self.raw_scores.items()
            }))
        complete, missing = _rseq(self.complete_ids, "complete_ids", identifiers=True), _rseq(self.missing_ids, "missing_ids", identifiers=True)
        if set(complete) & set(missing):
            raise RecordError("complete_ids and missing_ids must be disjoint")
        if self.raw_scores is not None and not complete:
            complete = tuple(self.raw_scores)
        if self.raw_scores is not None and not set(self.raw_scores).issubset(set(complete) | set(missing)):
            raise RecordError("raw_scores contain an undeclared option ID")
        object.__setattr__(self, "complete_ids", complete)
        object.__setattr__(self, "missing_ids", missing)
        object.__setattr__(self, "legal_mass", None if self.legal_mass is None else _rfinite(self.legal_mass, "legal_mass", minimum=0.0, maximum=1.0))
        if self.calibrated_values is not None:
            object.__setattr__(self, "calibrated_values", MappingProxyType({
                str(key): _rfinite(value, f"calibrated_values.{key}", minimum=0.0, maximum=1.0)
                for key, value in self.calibrated_values.items()
            }))
        for name in ("calibration_id", "policy_id"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else _rtext(value, name, limit=1024))
        object.__setattr__(self, "predicted_action", None if self.predicted_action is None else _rtext(self.predicted_action, "predicted_action", limit=256))
        object.__setattr__(self, "disposition", _renum(self.disposition, Disposition, "disposition"))
        object.__setattr__(self, "disposition_reason", _rtext(self.disposition_reason, "disposition_reason", required=False, limit=1024))
        object.__setattr__(self, "acceptable_option_ids", _rseq(self.acceptable_option_ids, "acceptable_option_ids", identifiers=True))
        object.__setattr__(self, "loss", _rfreeze(self.loss, "loss"))
        object.__setattr__(self, "loss_estimate", None if self.loss_estimate is None else _rfinite(self.loss_estimate, "loss_estimate", minimum=0.0))
        if not isinstance(self.applied, bool) or not isinstance(self.usage_known, bool):
            raise RecordError("applied and usage_known must be boolean")
        for name in ("receipts",):
            object.__setattr__(self, name, _safe_receipts(getattr(self, name)))
        for name in ("admission_receipt", "dispatch_receipt", "response_receipt", "consumption_receipt"):
            object.__setattr__(self, name, _safe_receipt(getattr(self, name), name))
        object.__setattr__(self, "receipt_timestamps", _rfreeze(self.receipt_timestamps, "receipt_timestamps"))
        for name in ("reserved_tokens", "actual_prompt_tokens", "actual_completion_tokens", "actual_reasoning_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise RecordError(f"{name} must be a non-negative integer")
        for name, value in self.timings_ms.items():
            _rfinite(value, f"timings_ms.{name}", minimum=0.0)
        object.__setattr__(self, "timings_ms", _rfreeze(self.timings_ms, "timings_ms"))
        object.__setattr__(self, "dependency_digests", _rfreeze(self.dependency_digests, "dependency_digests"))
        object.__setattr__(self, "claimed_attention_refs", _rseq(self.claimed_attention_refs, "claimed_attention_refs"))
        object.__setattr__(self, "claimed_attention_status", _rtext(self.claimed_attention_status, "claimed_attention_status").upper())
        object.__setattr__(self, "provenance", _rfreeze(self.provenance, "provenance"))
        object.__setattr__(self, "metadata", _rfreeze(self.metadata, "metadata"))
        if self.applied and self.consumption_receipt is None:
            raise RecordError("applied records require an observed consumption receipt")

    @property
    def raw_scores_complete(self) -> bool:
        return self.raw_scores is not None and bool(self.complete_ids) and not self.missing_ids

    @property
    def action_authority(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "record_schema": RECORD_SCHEMA, "decision_id": self.decision_id, "task_id": self.task_id,
            "site_id": self.site_id, "question_id": self.question_id, "question_version": self.question_version,
            "question_type": self.question_type, "state_epoch": self.state_epoch, "run_id": self.run_id,
            "parent_id": self.parent_id, "schema_id": self.schema_id, "schema_version": self.schema_version,
            "question_digest": self.question_digest, "prompt_digest": self.prompt_digest,
            "evidence_digest": self.evidence_digest, "evidence_snapshot_ref": self.evidence_snapshot_ref,
            "presented_evidence": _rthaw(self.presented_evidence), "public_source_refs": list(self.public_source_refs),
            "omissions": list(self.omissions), "options": _rthaw(self.options),
            "semantic_definitions": _rthaw(self.semantic_definitions), "label_token_map": _rthaw(self.label_token_map),
            "row_order": list(self.row_order), "permutation_seed": self.permutation_seed,
            "permutation_identity": self.permutation_identity, "model_identity": self.model_identity,
            "model_build": self.model_build, "tokenizer_identity": self.tokenizer_identity,
            "template_identity": self.template_identity, "score_identity": self.score_identity,
            "score_domain": self.score_domain, "score_position": self.score_position,
            "scorer_identity": self.scorer_identity, "sampling_settings": _rthaw(self.sampling_settings),
            "raw_scores": _rthaw(self.raw_scores), "complete_ids": list(self.complete_ids),
            "missing_ids": list(self.missing_ids), "legal_mass": self.legal_mass,
            "calibrated_values": _rthaw(self.calibrated_values), "calibration_id": self.calibration_id,
            "policy_id": self.policy_id, "predicted_action": self.predicted_action,
            "disposition": self.disposition, "disposition_reason": self.disposition_reason,
            "answerability": self.answerability, "acceptable_option_ids": list(self.acceptable_option_ids),
            "loss": _rthaw(self.loss), "loss_estimate": self.loss_estimate, "applied": self.applied,
            "action_authority": False, "receipts": _rthaw(self.receipts),
            "admission_receipt": _rthaw(self.admission_receipt), "dispatch_receipt": _rthaw(self.dispatch_receipt),
            "response_receipt": _rthaw(self.response_receipt), "consumption_receipt": _rthaw(self.consumption_receipt),
            "receipt_timestamps": _rthaw(self.receipt_timestamps), "reserved_tokens": self.reserved_tokens,
            "actual_prompt_tokens": self.actual_prompt_tokens, "actual_completion_tokens": self.actual_completion_tokens,
            "actual_reasoning_tokens": self.actual_reasoning_tokens, "usage_known": self.usage_known,
            "timings_ms": _rthaw(self.timings_ms), "dependency_digests": _rthaw(self.dependency_digests),
            "graph_watermark": self.graph_watermark, "permissions_digest": self.permissions_digest,
            "claimed_attention_refs": list(self.claimed_attention_refs),
            "claimed_attention_status": self.claimed_attention_status, "provenance": _rthaw(self.provenance),
            "metadata": _rthaw(self.metadata),
        }

    @property
    def cache_identity(self) -> str:
        return raw_score_cache_identity(self)


@dataclass(frozen=True, slots=True)
class OutcomeLabel:
    """Later host/auditor label; corrections append and supersede by ID."""

    decision_id: str
    task_id: str
    observed_at: str | int | float
    answerability: str = "ANSWERABLE"
    label_id: str = ""
    revision: int = 1
    target_id: str | None = None
    acceptable_option_ids: Sequence[str] = ()
    outcome_scope: str = "STATE_TRUTH"
    observation_status: str = ObservationStatus.OBSERVED.value
    chosen_action_id: str | None = None
    intermediate_correct: bool | None = None
    eventual_success: bool | None = None
    actual_chosen_action_loss: float | None = None
    evidence_refs: Sequence[str] = ()
    executed_outcome_refs: Sequence[str] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)
    adjudication: Mapping[str, object] = field(default_factory=dict)
    label_source: str = ""
    rubric_id: str = ""
    rubric_version: str = ""
    adjudication_status: str = "UNADJUDICATED"
    sampling_weight: float = 1.0
    supersedes_label_id: str | None = None
    censored: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _rtext(self.decision_id, "decision_id", limit=1024))
        object.__setattr__(self, "task_id", _rtext(self.task_id, "task_id", limit=1024))
        object.__setattr__(self, "observed_at", _rtext(self.observed_at, "observed_at", limit=128) if isinstance(self.observed_at, str) else _rfinite(self.observed_at, "observed_at", minimum=0.0))
        object.__setattr__(self, "answerability", _ranswerability(self.answerability))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise RecordError("revision must be a positive integer")
        object.__setattr__(self, "target_id", None if self.target_id is None else _rtext(self.target_id, "target_id", limit=256))
        accepted = _rseq(self.acceptable_option_ids, "acceptable_option_ids", identifiers=True)
        object.__setattr__(self, "acceptable_option_ids", accepted)
        object.__setattr__(self, "outcome_scope", _rtext(self.outcome_scope, "outcome_scope").upper())
        object.__setattr__(self, "observation_status", _renum(self.observation_status, ObservationStatus, "observation_status"))
        object.__setattr__(self, "chosen_action_id", None if self.chosen_action_id is None else _rtext(self.chosen_action_id, "chosen_action_id", limit=256))
        for name in ("intermediate_correct", "eventual_success"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), bool):
                raise RecordError(f"{name} must be boolean or null")
        if not isinstance(self.censored, bool):
            raise RecordError("censored must be boolean")
        object.__setattr__(self, "actual_chosen_action_loss", None if self.actual_chosen_action_loss is None else _rfinite(self.actual_chosen_action_loss, "actual_chosen_action_loss", minimum=0.0))
        object.__setattr__(self, "evidence_refs", _rseq(self.evidence_refs, "evidence_refs"))
        object.__setattr__(self, "executed_outcome_refs", _rseq(self.executed_outcome_refs, "executed_outcome_refs"))
        object.__setattr__(self, "provenance", _rfreeze(self.provenance, "provenance"))
        object.__setattr__(self, "adjudication", _rfreeze(self.adjudication, "adjudication"))
        for name in ("label_source", "rubric_id", "rubric_version"):
            object.__setattr__(self, name, _rtext(getattr(self, name), name, required=False, limit=1024))
        object.__setattr__(self, "adjudication_status", _rtext(self.adjudication_status, "adjudication_status", limit=256).upper())
        object.__setattr__(self, "sampling_weight", _rfinite(self.sampling_weight, "sampling_weight", minimum=0.0))
        object.__setattr__(self, "supersedes_label_id", None if self.supersedes_label_id is None else _rtext(self.supersedes_label_id, "supersedes_label_id", limit=1024))
        object.__setattr__(self, "metadata", _rfreeze(self.metadata, "metadata"))
        if self.answerability != Answerability.ANSWERABLE.value and (self.target_id or accepted):
            raise RecordError("non-answerable labels cannot carry truth targets")
        if self.censored or self.observation_status in {ObservationStatus.CENSORED.value, ObservationStatus.UNCHOSEN.value}:
            if any(value is not None for value in (self.actual_chosen_action_loss, self.intermediate_correct, self.eventual_success)):
                raise RecordError("censored/unchosen outcomes cannot claim observed correctness or loss")
        if self.actual_chosen_action_loss is not None and self.chosen_action_id is None:
            raise RecordError("actual chosen-action loss requires a chosen action")
        label_id = self.label_id or _rdigest({"decision_id": self.decision_id, "task_id": self.task_id,
                                              "revision": self.revision, "answerability": self.answerability,
                                              "target_id": self.target_id, "acceptable": list(accepted),
                                              "observed_at": self.observed_at})
        object.__setattr__(self, "label_id", _rtext(label_id, "label_id", limit=1024))

    @property
    def acceptable_set(self) -> tuple[str, ...]:
        return tuple(self.acceptable_option_ids)

    @property
    def action_authority(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "record_schema": RECORD_SCHEMA, "decision_id": self.decision_id, "task_id": self.task_id,
            "label_id": self.label_id, "revision": self.revision, "answerability": self.answerability,
            "target_id": self.target_id, "acceptable_option_ids": list(self.acceptable_option_ids),
            "outcome_scope": self.outcome_scope, "observation_status": self.observation_status,
            "chosen_action_id": self.chosen_action_id, "intermediate_correct": self.intermediate_correct,
            "eventual_success": self.eventual_success, "actual_chosen_action_loss": self.actual_chosen_action_loss,
            "evidence_refs": list(self.evidence_refs), "executed_outcome_refs": list(self.executed_outcome_refs),
            "provenance": _rthaw(self.provenance), "adjudication": _rthaw(self.adjudication),
            "label_source": self.label_source, "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version, "adjudication_status": self.adjudication_status,
            "sampling_weight": self.sampling_weight, "observed_at": self.observed_at,
            "supersedes_label_id": self.supersedes_label_id, "censored": self.censored,
            "action_authority": False, "metadata": _rthaw(self.metadata),
        }


class JournalLike(Protocol):
    def emit(self, kind: str, **fields: object) -> Mapping[str, object]: ...


def _record_payload(value: object) -> dict[str, object]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    if isinstance(value, Mapping):
        return dict(value)
    raise JournalRecordError("event payload must be an immutable record or mapping")


_JOURNAL_PRIVATE_KEYS = frozenset({
    "prompt", "prompt_body", "evidence_body", "content", "body", "response_body",
    "raw_envelope", "secret", "token", "access_token", "authorization",
})


def _journal_safe_projection(value: object) -> object:
    """Remove content-bearing diagnostic fields before a record reaches a journal."""
    if isinstance(value, Mapping):
        return {
            str(key): _journal_safe_projection(item)
            for key, item in value.items()
            if str(key).casefold() not in _JOURNAL_PRIVATE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_journal_safe_projection(item) for item in value]
    return value


def _event_fields(kind: str, value: object) -> dict[str, object]:
    if kind not in {DECISION_RECORDED, DECISION_LABELLED}:
        raise JournalRecordError("unsupported decision journal event kind")
    payload = _record_payload(value)
    for key in payload:
        if key.startswith("receipt_") and key[8:] in RESERVED:
            raise JournalRecordError(f"event field collides with journal-safe rename: {key}")
    required = (("decision_id", "task_id", "site_id", "prompt_digest", "evidence_digest")
                if kind == DECISION_RECORDED else
                ("decision_id", "task_id", "label_id", "answerability", "observed_at"))
    missing = tuple(name for name in required if name not in payload)
    if missing:
        raise JournalRecordError("event missing required fields: " + ", ".join(missing))
    projected = _journal_safe_projection(payload)
    if not isinstance(projected, Mapping):
        raise JournalRecordError("journal projection must remain a mapping")
    payload = dict(projected)
    safe = journal_safe_fields(payload)
    if RESERVED.intersection(safe):
        raise JournalRecordError("journal-safe event contains a reserved field")
    safe["idempotency_key"] = _rdigest({"kind": kind, "payload": payload})
    safe["projection_only"] = True
    safe["action_authority"] = False
    return safe


def _journal_events(journal: object) -> Iterable[Mapping[str, object]]:
    records = getattr(journal, "records", None)
    if callable(records):
        try:
            raw_records = records()
            if not isinstance(raw_records, Iterable):
                return
            for event in raw_records:
                if isinstance(event, Mapping):
                    yield event
            return
        except (OSError, RuntimeError, ValueError):
            return
    for attr in ("events", "calls"):
        values = getattr(journal, attr, None)
        if not isinstance(values, Sequence):
            continue
        for value in values:
            if isinstance(value, Mapping):
                yield value
            elif isinstance(value, Sequence) and len(value) == 2 and isinstance(value[1], Mapping):
                yield {"kind": value[0], **dict(value[1])}


def _append_once(journal: JournalLike, kind: str, value: object) -> dict[str, object]:
    fields = _event_fields(kind, value)
    key = fields["idempotency_key"]
    for existing in _journal_events(journal):
        if existing.get("idempotency_key") == key:
            return dict(existing)
    emitted = journal.emit(kind, **fields)
    if not isinstance(emitted, Mapping):
        raise JournalRecordError("journal emit must return a mapping")
    return dict(emitted)


def append_decision_recorded(journal: JournalLike, record: DecisionRecord | Mapping[str, object]) -> dict[str, object]:
    return _append_once(journal, DECISION_RECORDED, record)


def append_decision_labelled(journal: JournalLike, label: OutcomeLabel | Mapping[str, object]) -> dict[str, object]:
    return _append_once(journal, DECISION_LABELLED, label)


emit_decision_recorded = append_decision_recorded
emit_decision_labelled = append_decision_labelled


class RecordLedger:
    """An in-process journal projection, never a competing truth store."""

    def __init__(self, journal: JournalLike) -> None:
        self.journal = journal
        self._decisions: dict[str, DecisionRecord] = {}
        self._labels: dict[str, list[OutcomeLabel]] = {}
        self._events: list[dict[str, object]] = []

    def append_decision(self, record: DecisionRecord) -> Mapping[str, object]:
        previous = self._decisions.get(record.decision_id)
        if previous is not None and previous != record:
            raise IdempotencyConflictError("decision_id identifies different immutable content")
        event = append_decision_recorded(self.journal, record)
        if not any(item.get("idempotency_key") == event.get("idempotency_key") for item in self._events):
            self._events.append(event)
        self._decisions[record.decision_id] = record
        return event

    def append_label(self, label: OutcomeLabel) -> Mapping[str, object]:
        decision = self._decisions.get(label.decision_id)
        if decision is None:
            raise RecordError("cannot label a decision outside this task projection")
        if decision.task_id != label.task_id:
            raise RecordError("decision and label task IDs do not match")
        prior = {item.label_id: item for item in self._labels.get(label.decision_id, ())}
        if label.label_id in prior and prior[label.label_id] != label:
            raise IdempotencyConflictError("label_id identifies different immutable content")
        if label.supersedes_label_id is not None and label.supersedes_label_id not in prior:
            raise RecordError("label correction must supersede an existing label")
        event = append_decision_labelled(self.journal, label)
        if label.label_id not in prior:
            self._labels.setdefault(label.decision_id, []).append(label)
        if not any(item.get("idempotency_key") == event.get("idempotency_key") for item in self._events):
            self._events.append(event)
        return event

    def labels(self, decision_id: str) -> tuple[OutcomeLabel, ...]:
        return tuple(self._labels.get(decision_id, ()))

    def current_label(self, decision_id: str) -> OutcomeLabel | None:
        values = self._labels.get(decision_id, ())
        return values[-1] if values else None

    @property
    def events(self) -> tuple[Mapping[str, object], ...]:
        return tuple(MappingProxyType(dict(item)) for item in self._events)


def raw_score_cache_identity(record: DecisionRecord | Mapping[str, object]) -> str:
    """Exact question/evidence/model/scoring identity; excludes artifacts/policy."""
    payload = record.to_dict() if isinstance(record, DecisionRecord) else dict(record)
    identity = {key: payload.get(key) for key in (
        "task_id", "site_id", "question_id", "question_version", "question_digest", "prompt_digest",
        "evidence_digest", "evidence_snapshot_ref", "presented_evidence", "options",
        "semantic_definitions", "label_token_map", "row_order", "permutation_seed", "permutation_identity",
        "model_identity", "model_build", "tokenizer_identity", "template_identity", "score_identity",
        "score_domain", "score_position", "scorer_identity", "sampling_settings", "state_epoch",
        "dependency_digests", "graph_watermark", "permissions_digest")}
    identity["schema"] = RAW_SCORE_CACHE_SCHEMA
    return _rdigest(identity)


cache_identity = raw_score_cache_identity


@dataclass(frozen=True, slots=True)
class CachedRawScores:
    cache_identity: str
    decision_id: str
    raw_scores: Mapping[str, float]
    source_record: DecisionRecord


class RawScoreCache:
    """Raw scores only; downstream calibration and policy are always recomputed."""

    def __init__(self) -> None:
        self._entries: dict[str, CachedRawScores] = {}

    def put(self, record: DecisionRecord) -> CachedRawScores:
        if record.raw_scores is None:
            raise CacheIdentityError("cannot cache missing raw scores")
        key = record.cache_identity
        entry = CachedRawScores(key, record.decision_id, MappingProxyType(dict(record.raw_scores)), record)
        prior = self._entries.get(key)
        if prior is not None and dict(prior.raw_scores) != dict(entry.raw_scores):
            raise CacheIdentityError("exact cache identity maps to different raw scores")
        self._entries[key] = entry
        return entry

    def get(self, record: DecisionRecord) -> CachedRawScores:
        entry = self._entries.get(record.cache_identity)
        if entry is None:
            raise CacheMissError("no exact raw-score cache entry")
        return entry

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class ConsumptionCheck:
    stale_reasons: tuple[str, ...] = ()
    artifact_mismatch: bool = False
    policy_mismatch: bool = False
    recalculation_required: bool = False

    @property
    def ok(self) -> bool:
        return not self.stale_reasons


def _same(left: object, right: object) -> bool:
    return _rthaw(_rfreeze(left)) == _rthaw(_rfreeze(right))


def revalidate_before_consumption(record: DecisionRecord, current_state: Mapping[str, object], *,
                                  calibration_id: str | None = None,
                                  policy_id: str | None = None) -> ConsumptionCheck:
    expected = record.to_dict()
    reasons: list[str] = []
    if current_state.get("task_id") != expected["task_id"]:
        reasons.append("task_id")
    for name in ("evidence_digest", "dependency_digests", "graph_watermark", "permissions_digest", "state_epoch"):
        if name not in current_state or not _same(current_state.get(name), expected.get(name)):
            reasons.append(name)
    artifact_mismatch = calibration_id is not None and record.calibration_id != calibration_id
    policy_changed = policy_id is not None and record.policy_id != policy_id
    return ConsumptionCheck(tuple(dict.fromkeys(reasons)), artifact_mismatch, policy_changed,
                            artifact_mismatch or policy_changed)


@dataclass(frozen=True, slots=True)
class CachedDecision:
    source_cache_identity: str
    raw_scores: Mapping[str, float]
    calibrated_values: Mapping[str, float] | None
    predicted_action: str | None
    calibration_id: str | None
    policy_id: str | None
    artifact_recomputed: bool
    action_authority: bool = False


def consume_cached_raw_scores(record: DecisionRecord, cache: RawScoreCache,
                              current_state: Mapping[str, object], *,
                              calibration_id: str | None = None,
                              policy_id: str | None = None,
                              calibrate: Callable[[Mapping[str, float]], Mapping[str, float]] | None = None,
                              select: Callable[[Mapping[str, float]], str | None] | None = None) -> CachedDecision:
    entry = cache.get(record)
    check = revalidate_before_consumption(record, current_state,
                                          calibration_id=calibration_id, policy_id=policy_id)
    if not check.ok:
        raise StaleConsumptionError(check)
    if check.recalculation_required and (calibrate is None or select is None):
        raise ArtifactMismatchError("calibration/policy artifact changed; calibrated and action outputs must be recomputed")
    calibrated = None
    if calibrate is not None:
        calibrated = MappingProxyType({
            str(key): _rfinite(value, f"calibrated_values.{key}", minimum=0.0, maximum=1.0)
            for key, value in calibrate(entry.raw_scores).items()
        })
    elif not check.recalculation_required:
        calibrated = record.calibrated_values
    action = select(calibrated or entry.raw_scores) if select is not None else (
        record.predicted_action if not check.recalculation_required else None)
    return CachedDecision(entry.cache_identity, entry.raw_scores, calibrated, action,
                          calibration_id or record.calibration_id, policy_id or record.policy_id,
                          check.recalculation_required)


__all__ = [
    "ArtifactMismatchError", "CacheIdentityError", "CacheMissError", "CachedDecision", "CachedRawScores",
    "ConsumptionCheck", "DECISION_LABELLED", "DECISION_RECORDED", "DecisionRecord", "Disposition",
    "ExecutionLedger", "FacadeReceipt", "IdempotencyConflictError", "JournalRecordError",
    "LedgerReservation", "ObservationStatus", "OutcomeLabel", "ReceiptAdapter",
    "RAW_SCORE_CACHE_SCHEMA", "RawScoreCache", "RECORD_SCHEMA", "RecordError", "RecordLedger",
    "SCHEMA", "StaleConsumptionError", "append_decision_labelled", "append_decision_recorded",
    "cache_identity", "consume_cached_raw_scores", "emit_decision_labelled", "emit_decision_recorded",
    "raw_score_cache_identity", "receipt_id", "revalidate_before_consumption",
]
