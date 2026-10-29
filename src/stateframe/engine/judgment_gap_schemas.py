"""Typed, deterministic contracts for the judgment-gap composition.

Slice 0 is intentionally a projection-only foundation.  These records do not
execute tools, mutate a world cache, or register a kernel organ.  Producers in
later slices can exchange typed observations and render one bounded
``SituationCard`` without allowing model prose to become evidence.

The module reuses the repository's two important evidence primitives:

* :class:`kernel_semantic_ir.SourceSpan` for source-level provenance;
* :class:`detreason.world_knowledge.WorldFact` for versioned world facts.

``FrameSnapshot`` serializes the existing ``poc.stateframe.shadow/2`` record
shape exactly.  It is a typed view of that projector's record, not a competing
StateFrame implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from typing import ClassVar, TypeAlias

from detreason.world_knowledge import WorldFact
from kernel_nitpicker import ContractAtom
from kernel_semantic_ir import SourceSpan


JUDGMENT_GAP_SCHEMA = "poc.judgment_gap/1"
OBSERVATION_BUNDLE_SCHEMA = "ObservationBundle/1"
SITUATION_CARD_SCHEMA = "SituationCard/1"
DECISION_DECLARATION_SCHEMA = "DecisionDeclaration/1"
CERTAINTY_DEMAND_SCHEMA = "CertaintyDemand/1"
HYPOTHESIS_ASSESSMENT_SCHEMA = "HypothesisAssessment/1"
FRAME_SNAPSHOT_SCHEMA = "poc.stateframe.shadow/2"

MAX_VISIBLE_ADVISORIES = 8
MAX_VISIBLE_UNKNOWN = 1
MAX_VISIBLE_CONTRACT_ATOMS = 3

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | Sequence["JsonValue"]


class SchemaError(ValueError):
    """Raised when a typed judgment record cannot be made safe and explicit."""


class ScopeViolation(SchemaError):
    """Raised when evidence from another workspace/run is mixed into a record."""


class TruthState(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class Freshness(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class FailureClassification(str, Enum):
    GREEN = "GREEN"
    PRODUCT_RED = "PRODUCT_RED"
    ENVIRONMENT_RED = "ENVIRONMENT_RED"
    INFRASTRUCTURE_RED = "INFRASTRUCTURE_RED"
    UNKNOWN = "UNKNOWN"
    NOT_OBSERVED = "NOT_OBSERVED"


class ActionKind(str, Enum):
    READ = "READ"
    TEST = "TEST"
    EDIT = "EDIT"
    FINISH = "FINISH"
    OTHER = "OTHER"


class AdvisoryKind(str, Enum):
    SAFETY = "SAFETY"
    ENVIRONMENT = "ENVIRONMENT"
    FRONTIER = "FRONTIER"
    CONTRADICTION = "CONTRADICTION"
    CONTRACT = "CONTRACT"
    DEPENDENCY = "DEPENDENCY"
    WORLD = "WORLD"
    NEGATIVE_WORK = "NEGATIVE_WORK"
    UNKNOWN = "UNKNOWN"


class CompletionState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    NOT_YET_VERIFIED = "NOT_YET_VERIFIED"
    BLOCKED = "BLOCKED"
    UNVERIFIED = "UNVERIFIED"
    SUCCESS_CERTIFIED = "SUCCESS_CERTIFIED"


class PersistenceState(str, Enum):
    DURABLE = "DURABLE"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


class CertaintyStatus(str, Enum):
    VERIFIED = "VERIFIED"
    NOT_YET = "NOT_YET"
    BLOCKED = "BLOCKED"
    UNVERIFIABLE = "UNVERIFIABLE"


class EditPermission(str, Enum):
    YES = "YES"
    NO = "NO"
    YES_WITH_RECHECK = "YES_WITH_RECHECK"


class DemandPoint(str, Enum):
    ENVIRONMENT = "ENVIRONMENT"
    AFTER_RED = "AFTER_RED"
    PRE_EDIT = "PRE_EDIT"
    CHECKPOINT = "CHECKPOINT"
    POST_GREEN = "POST_GREEN"
    BEFORE_DONE = "BEFORE_DONE"
    PLAN_REVISION = "PLAN_REVISION"


class HypothesisStatus(str, Enum):
    LIVE = "LIVE"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"


def _text(value: object, field_name: str, *, limit: int = 512) -> str:
    result = str(value).strip() if value is not None else ""
    if not result:
        raise SchemaError(f"{field_name} is required")
    if len(result) > limit:
        raise SchemaError(f"{field_name} exceeds {limit} characters")
    return result


def _clip(value: object, *, limit: int = 512) -> str:
    result = " ".join(str(value or "").split())
    return result if len(result) <= limit else result[: limit - 1] + "…"


def _optional_text(value: object, *, limit: int = 512) -> str:
    return _clip(value, limit=limit) if value not in (None, "") else ""


def _enum(enum_type: type[Enum], value: object, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value).strip().upper())
    except ValueError as exc:
        allowed = ", ".join(str(item.value) for item in enum_type)
        raise SchemaError(f"{field_name} must be one of {allowed}") from exc


def _items(value: object, field_name: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(f"{field_name} must be a sequence")
    return tuple(value)


def _strings(value: object, field_name: str, *, sort: bool = False) -> tuple[str, ...]:
    values = [_clip(item) for item in _items(value, field_name)]
    values = [item for item in values if item]
    if sort:
        return tuple(sorted(set(values)))
    return tuple(dict.fromkeys(values))


def _mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field_name} must be a mapping")
    return {str(key): value[key] for key in sorted(value, key=lambda item: str(item))}


def _canonical(value: object) -> object:
    """Convert supported values into deterministic JSON-safe data."""

    if isinstance(value, Enum):
        return value.value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, float) and not isfinite(value):
        raise TypeError("non-finite floats are not valid in judgment schemas")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported value in judgment schema: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Serialize a schema value with stable key and collection handling."""

    return json.dumps(
        _canonical(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _check_schema(value: Mapping[str, object], expected: str) -> None:
    supplied = value.get("schema")
    if supplied not in (None, expected):
        raise SchemaError(f"schema must be {expected}, got {supplied!r}")


def _check_digest(
    value: Mapping[str, object],
    expected: str,
    *,
    fields: tuple[str, ...] = ("digest",),
) -> None:
    for field_name in fields:
        supplied = value.get(field_name)
        if supplied not in (None, "") and str(supplied) != expected:
            raise SchemaError(f"{field_name} does not match the record content")


def _span(value: object, field_name: str = "source_span") -> SourceSpan:
    if isinstance(value, SourceSpan):
        return value
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field_name} must be a SourceSpan or mapping")
    try:
        start_line = int(value.get("start_line", value.get("line", 0)))
        end_value = value.get("end_line", value.get("end", start_line))
        end_line = int(end_value) if end_value is not None else start_line
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{field_name} has invalid line numbers") from exc
    return SourceSpan(
        path=str(value.get("path") or ""),
        start_line=start_line,
        end_line=end_line,
        symbol=str(value.get("symbol") or ""),
        role=str(value.get("role") or ""),
    )


def _spans(value: object, field_name: str = "source_spans") -> tuple[SourceSpan, ...]:
    spans = {_span(item, field_name) for item in _items(value, field_name)}
    return tuple(sorted(spans))


def _provenance(value: object, field_name: str = "provenance") -> "Provenance":
    if not isinstance(value, Provenance):
        raise SchemaError(f"{field_name} must be a Provenance record")
    if not value.has_reference:
        raise SchemaError(f"{field_name} needs a source span, event, receipt, or fact reference")
    return value


def _same_scope(left: "RunScope", right: "RunScope") -> bool:
    return left.to_dict() == right.to_dict()


def _assert_optional_scope_fields(
    values: Mapping[str, object],
    scope: "RunScope",
    field_name: str,
) -> None:
    for key in (
        "mission_id",
        "unit_id",
        "task_id",
        "workspace_id",
        "sandbox_id",
        "source_revision",
        "env_id",
        "oracle_id",
    ):
        actual = values.get(key)
        if actual not in (None, "") and str(actual) != getattr(scope, key):
            raise ScopeViolation(
                f"{field_name}.{key}={actual!r} does not match this run scope"
            )


@dataclass(frozen=True, slots=True)
class RunScope:
    """Identity boundary shared by every live judgment record."""

    mission_id: str = ""
    unit_id: str = ""
    task_id: str = ""
    workspace_id: str = ""
    sandbox_id: str = ""
    source_revision: str = ""
    env_id: str = ""
    oracle_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "mission_id",
            "unit_id",
            "task_id",
            "workspace_id",
            "sandbox_id",
            "source_revision",
            "env_id",
            "oracle_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))

    def to_dict(self) -> dict[str, str]:
        return {
            "mission_id": self.mission_id,
            "unit_id": self.unit_id,
            "task_id": self.task_id,
            "workspace_id": self.workspace_id,
            "sandbox_id": self.sandbox_id,
            "source_revision": self.source_revision,
            "env_id": self.env_id,
            "oracle_id": self.oracle_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RunScope":
        return cls(**{name: str(value.get(name) or "") for name in cls.__dataclass_fields__})


# Short compatibility alias for callers that call this object simply Scope.
Scope = RunScope


@dataclass(frozen=True, slots=True)
class Provenance:
    """Explicit origin references for a material claim or advisory."""

    source_spans: tuple[SourceSpan, ...] = ()
    event_ids: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()
    fact_ids: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_spans", _spans(self.source_spans))
        object.__setattr__(self, "event_ids", _strings(self.event_ids, "event_ids", sort=True))
        object.__setattr__(self, "receipt_ids", _strings(self.receipt_ids, "receipt_ids", sort=True))
        object.__setattr__(self, "fact_ids", _strings(self.fact_ids, "fact_ids", sort=True))
        object.__setattr__(self, "detail", _optional_text(self.detail))

    @property
    def has_reference(self) -> bool:
        return bool(self.source_spans or self.event_ids or self.receipt_ids or self.fact_ids)

    def merged(self, other: "Provenance") -> "Provenance":
        if not isinstance(other, Provenance):
            raise SchemaError("can only merge Provenance records")
        details = sorted({detail for detail in (self.detail, other.detail) if detail})
        return Provenance(
            source_spans=(*self.source_spans, *other.source_spans),
            event_ids=(*self.event_ids, *other.event_ids),
            receipt_ids=(*self.receipt_ids, *other.receipt_ids),
            fact_ids=(*self.fact_ids, *other.fact_ids),
            detail=" | ".join(details),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_spans": [span.to_dict() for span in self.source_spans],
            "event_ids": list(self.event_ids),
            "receipt_ids": list(self.receipt_ids),
            "fact_ids": list(self.fact_ids),
            "detail": self.detail,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Provenance":
        return cls(
            source_spans=_spans(value.get("source_spans", value.get("spans", ()))),
            event_ids=value.get("event_ids", ()),
            receipt_ids=value.get("receipt_ids", ()),
            fact_ids=value.get("fact_ids", ()),
            detail=str(value.get("detail") or ""),
        )


@dataclass(frozen=True, slots=True)
class UnknownReason:
    """One bounded explanation for an epistemically unknown value."""

    unknown_id: str
    subject: str
    reason: str
    provenance: Provenance
    required_action: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "unknown_id", _text(self.unknown_id, "unknown_id"))
        object.__setattr__(self, "subject", _text(self.subject, "unknown subject"))
        object.__setattr__(self, "reason", _text(self.reason, "unknown reason", limit=1000))
        object.__setattr__(self, "required_action", _optional_text(self.required_action))
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, object]:
        return {
            "unknown_id": self.unknown_id,
            "subject": self.subject,
            "reason": self.reason,
            "required_action": self.required_action,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "UnknownReason":
        return cls(
            unknown_id=str(value.get("unknown_id") or ""),
            subject=str(value.get("subject") or ""),
            reason=str(value.get("reason") or ""),
            required_action=str(value.get("required_action") or ""),
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )


@dataclass(frozen=True, slots=True)
class Receipt:
    """A canonical, provenance-backed executor result."""

    receipt_id: str
    action_id: str
    kind: str
    result_class: str
    turn: int
    source_revision: str
    env_id: str
    oracle_id: str
    provenance: Provenance
    selector: str = ""
    target: str = ""
    fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_id", _text(self.receipt_id, "receipt_id"))
        object.__setattr__(self, "action_id", _text(self.action_id, "action_id"))
        object.__setattr__(self, "kind", _text(self.kind, "receipt kind"))
        object.__setattr__(self, "result_class", _text(self.result_class, "result_class").upper())
        if not isinstance(self.turn, int) or self.turn < 0:
            raise SchemaError("receipt turn must be a non-negative integer")
        object.__setattr__(self, "source_revision", _text(self.source_revision, "source_revision"))
        object.__setattr__(self, "env_id", _text(self.env_id, "env_id"))
        object.__setattr__(self, "oracle_id", _text(self.oracle_id, "oracle_id"))
        object.__setattr__(self, "selector", _optional_text(self.selector))
        object.__setattr__(self, "target", _optional_text(self.target))
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        identity = {
            "receipt_id": self.receipt_id,
            "action_id": self.action_id,
            "kind": self.kind,
            "result_class": self.result_class,
            "turn": self.turn,
            "source_revision": self.source_revision,
            "env_id": self.env_id,
            "oracle_id": self.oracle_id,
            "selector": self.selector,
            "target": self.target,
            "provenance": self.provenance.to_dict(),
        }
        expected = _digest(identity)
        supplied = str(self.fingerprint).strip()
        if supplied and supplied != expected:
            raise SchemaError(f"receipt {self.receipt_id} fingerprint does not match its content")
        object.__setattr__(self, "fingerprint", expected)

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "action_id": self.action_id,
            "kind": self.kind,
            "result_class": self.result_class,
            "turn": self.turn,
            "source_revision": self.source_revision,
            "env_id": self.env_id,
            "oracle_id": self.oracle_id,
            "selector": self.selector,
            "target": self.target,
            "fingerprint": self.fingerprint,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Receipt":
        return cls(
            receipt_id=str(value.get("receipt_id") or ""),
            action_id=str(value.get("action_id") or ""),
            kind=str(value.get("kind") or ""),
            result_class=str(value.get("result_class") or ""),
            turn=int(value.get("turn", 0)),
            source_revision=str(value.get("source_revision") or ""),
            env_id=str(value.get("env_id") or ""),
            oracle_id=str(value.get("oracle_id") or ""),
            selector=str(value.get("selector") or ""),
            target=str(value.get("target") or ""),
            fingerprint=str(value.get("fingerprint") or ""),
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )


@dataclass(frozen=True, slots=True)
class SemanticEvidence:
    """One typed semantic observation, separate from model prose."""

    evidence_id: str
    predicate: str
    state: TruthState | str
    source_revision: str
    env_id: str
    oracle_id: str
    provenance: Provenance
    value: JsonValue = None
    unknown_reason: str | None = None
    fact_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "predicate", _text(self.predicate, "predicate"))
        object.__setattr__(self, "state", _enum(TruthState, self.state, "state"))
        object.__setattr__(self, "source_revision", _text(self.source_revision, "source_revision"))
        object.__setattr__(self, "env_id", _text(self.env_id, "env_id"))
        object.__setattr__(self, "oracle_id", _text(self.oracle_id, "oracle_id"))
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        normalized_value = _canonical(self.value)
        object.__setattr__(self, "value", normalized_value)
        reason = _optional_text(self.unknown_reason, limit=1000)
        if self.state is TruthState.UNKNOWN and not reason:
            raise SchemaError(f"UNKNOWN evidence {self.evidence_id} needs unknown_reason")
        if self.state is not TruthState.UNKNOWN and reason:
            raise SchemaError(f"non-UNKNOWN evidence {self.evidence_id} has unknown_reason")
        object.__setattr__(self, "unknown_reason", reason or None)
        object.__setattr__(self, "fact_id", _optional_text(self.fact_id))

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "predicate": self.predicate,
            "state": self.state.value,
            "source_revision": self.source_revision,
            "env_id": self.env_id,
            "oracle_id": self.oracle_id,
            "value": self.value,
            "fact_id": self.fact_id,
            "unknown_reason": self.unknown_reason,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SemanticEvidence":
        return cls(
            evidence_id=str(value.get("evidence_id") or ""),
            predicate=str(value.get("predicate") or ""),
            state=str(value.get("state") or "UNKNOWN"),
            source_revision=str(value.get("source_revision") or ""),
            env_id=str(value.get("env_id") or ""),
            oracle_id=str(value.get("oracle_id") or ""),
            value=value.get("value"),
            fact_id=str(value.get("fact_id") or ""),
            unknown_reason=str(value.get("unknown_reason") or "") or None,
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )


@dataclass(frozen=True, slots=True)
class ModelProposal:
    """Model prose kept as an unverified proposal, never as a fact."""

    proposal_id: str
    text: str
    provenance: Provenance
    status: str = "UNVERIFIED"

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_id", _text(self.proposal_id, "proposal_id"))
        object.__setattr__(self, "text", _text(self.text, "proposal text", limit=1000))
        status = _text(self.status, "proposal status").upper()
        if status != "UNVERIFIED":
            raise SchemaError("model prose proposals must remain UNVERIFIED")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "text": self.text,
            "status": self.status,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ModelProposal":
        return cls(
            proposal_id=str(value.get("proposal_id") or ""),
            text=str(value.get("text") or ""),
            status=str(value.get("status") or "UNVERIFIED"),
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )


@dataclass(frozen=True, slots=True)
class NegativeWork:
    """A bounded no-progress or ruled-out action receipt."""

    kind: str
    action_signature: str
    turn: int
    provenance: Provenance
    detail: str = ""
    ruled_out: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _text(self.kind, "negative-work kind").upper())
        object.__setattr__(self, "action_signature", _text(self.action_signature, "action_signature"))
        if not isinstance(self.turn, int) or self.turn < 0:
            raise SchemaError("negative-work turn must be a non-negative integer")
        object.__setattr__(self, "detail", _optional_text(self.detail, limit=1000))
        object.__setattr__(self, "ruled_out", bool(self.ruled_out))
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    @property
    def is_no_op(self) -> bool:
        return "NOOP" in self.kind or "NO_OP" in self.kind

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "action_signature": self.action_signature,
            "turn": self.turn,
            "detail": self.detail,
            "ruled_out": self.ruled_out,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "NegativeWork":
        return cls(
            kind=str(value.get("kind") or ""),
            action_signature=str(value.get("action_signature") or ""),
            turn=int(value.get("turn", 0)),
            detail=str(value.get("detail") or ""),
            ruled_out=bool(value.get("ruled_out", False)),
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )


@dataclass(frozen=True, slots=True)
class ObservationBundle:
    """Canonical executor observation consumed by future state reducers."""

    SCHEMA: ClassVar[str] = OBSERVATION_BUNDLE_SCHEMA

    scope: RunScope
    action_id: str
    intent_id: str
    event_ids: tuple[str, ...]
    turn: int
    workspace_revision: int
    source_revision: str
    env_id: str
    oracle_id: str
    receipts: tuple[Receipt, ...] = ()
    world_fact_candidates: tuple[WorldFact, ...] = ()
    semantic_evidence: tuple[SemanticEvidence, ...] = ()
    negative_work: tuple[NegativeWork, ...] = ()
    failure_classification: FailureClassification | str = FailureClassification.NOT_OBSERVED
    model_proposals: tuple[ModelProposal, ...] = ()
    unknown_reasons: tuple[UnknownReason, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, RunScope):
            raise SchemaError("observation scope must be RunScope")
        object.__setattr__(self, "action_id", _text(self.action_id, "action_id"))
        object.__setattr__(self, "intent_id", _text(self.intent_id, "intent_id"))
        if not isinstance(self.turn, int) or self.turn < 0:
            raise SchemaError("observation turn must be a non-negative integer")
        if not isinstance(self.workspace_revision, int) or self.workspace_revision < 0:
            raise SchemaError("workspace_revision must be a non-negative integer")
        source_revision = _text(self.source_revision, "source_revision")
        env_id = _text(self.env_id, "env_id")
        oracle_id = _text(self.oracle_id, "oracle_id")
        if source_revision != self.scope.source_revision:
            raise ScopeViolation("observation source_revision does not match scope")
        if env_id != self.scope.env_id or oracle_id != self.scope.oracle_id:
            raise ScopeViolation("observation environment/oracle does not match scope")
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "env_id", env_id)
        object.__setattr__(self, "oracle_id", oracle_id)
        object.__setattr__(self, "event_ids", _strings(self.event_ids, "event_ids", sort=True))
        object.__setattr__(self, "failure_classification", _enum(
            FailureClassification, self.failure_classification, "failure_classification"))

        receipts = []
        for item in _items(self.receipts, "receipts"):
            receipt = item if isinstance(item, Receipt) else Receipt.from_mapping(item)
            if receipt.action_id != self.action_id:
                raise ScopeViolation(f"receipt {receipt.receipt_id} belongs to another action")
            if (receipt.source_revision, receipt.env_id, receipt.oracle_id) != (
                self.source_revision,
                self.env_id,
                self.oracle_id,
            ):
                raise ScopeViolation(f"receipt {receipt.receipt_id} has a mismatched run identity")
            receipts.append(receipt)
        object.__setattr__(self, "receipts", tuple(sorted(receipts, key=lambda item: (item.turn, item.receipt_id))))

        facts = []
        for item in _items(self.world_fact_candidates, "world_fact_candidates"):
            if not isinstance(item, WorldFact):
                if isinstance(item, Mapping):
                    item = WorldFact.from_dict(item)
                else:
                    raise SchemaError("world_fact_candidates must contain WorldFact records")
            _assert_world_fact_scope(item, self.scope)
            facts.append(item)
        object.__setattr__(self, "world_fact_candidates", tuple(sorted(facts, key=lambda item: item.fact_id)))

        semantic = []
        for item in _items(self.semantic_evidence, "semantic_evidence"):
            evidence = item if isinstance(item, SemanticEvidence) else SemanticEvidence.from_mapping(item)
            if (evidence.source_revision, evidence.env_id, evidence.oracle_id) != (
                self.source_revision,
                self.env_id,
                self.oracle_id,
            ):
                raise ScopeViolation(f"semantic evidence {evidence.evidence_id} has a mismatched run identity")
            semantic.append(evidence)
        object.__setattr__(self, "semantic_evidence", tuple(sorted(semantic, key=lambda item: item.evidence_id)))

        negatives = [
            item if isinstance(item, NegativeWork) else NegativeWork.from_mapping(item)
            for item in _items(self.negative_work, "negative_work")
        ]
        object.__setattr__(self, "negative_work", tuple(sorted(
            negatives, key=lambda item: (item.turn, item.kind, item.action_signature))))
        proposals = [
            item if isinstance(item, ModelProposal) else ModelProposal.from_mapping(item)
            for item in _items(self.model_proposals, "model_proposals")
        ]
        object.__setattr__(self, "model_proposals", tuple(sorted(proposals, key=lambda item: item.proposal_id)))
        unknowns = [
            item if isinstance(item, UnknownReason) else UnknownReason.from_mapping(item)
            for item in _items(self.unknown_reasons, "unknown_reasons")
        ]
        by_id = {item.unknown_id: item for item in unknowns}
        object.__setattr__(self, "unknown_reasons", tuple(by_id[key] for key in sorted(by_id)))

    @property
    def usable_world_facts(self) -> tuple[WorldFact, ...]:
        return tuple(item for item in self.world_fact_candidates if _world_fact_status(
            item, self.workspace_revision) == Freshness.FRESH)

    @property
    def stale_world_facts(self) -> tuple[WorldFact, ...]:
        return tuple(item for item in self.world_fact_candidates if _world_fact_status(
            item, self.workspace_revision) == Freshness.STALE)

    @property
    def unknown_world_facts(self) -> tuple[WorldFact, ...]:
        return tuple(item for item in self.world_fact_candidates if _world_fact_status(
            item, self.workspace_revision) == Freshness.UNKNOWN)

    def _payload(self) -> dict[str, object]:
        return {
            "scope": self.scope.to_dict(),
            "action_id": self.action_id,
            "intent_id": self.intent_id,
            "event_ids": list(self.event_ids),
            "turn": self.turn,
            "workspace_revision": self.workspace_revision,
            "source_revision": self.source_revision,
            "env_id": self.env_id,
            "oracle_id": self.oracle_id,
            "receipts": [item.to_dict() for item in self.receipts],
            "world_fact_candidates": [item.to_dict() for item in self.world_fact_candidates],
            "world_fact_status": {
                "fresh": [item.fact_id for item in self.usable_world_facts],
                "stale": [item.fact_id for item in self.stale_world_facts],
                "unknown": [item.fact_id for item in self.unknown_world_facts],
            },
            "semantic_evidence": [item.to_dict() for item in self.semantic_evidence],
            "negative_work": [item.to_dict() for item in self.negative_work],
            "failure_classification": self.failure_classification.value,
            "model_proposals": [item.to_dict() for item in self.model_proposals],
            "unknown_reasons": [item.to_dict() for item in self.unknown_reasons],
        }

    @property
    def digest(self) -> str:
        return _digest({"schema": self.SCHEMA, **self._payload()})

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, **self._payload(), "digest": self.digest}

    def serialize(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ObservationBundle":
        _check_schema(value, cls.SCHEMA)
        scope = RunScope.from_mapping(value.get("scope", {}))
        raw_facts = value.get("world_fact_candidates", ())
        if isinstance(raw_facts, Mapping):
            raw_facts = (*_items(raw_facts.get("fresh", ()), "fresh facts"),
                         *_items(raw_facts.get("stale", ()), "stale facts"),
                         *_items(raw_facts.get("unknown", ()), "unknown facts"))
        record = cls(
            scope=scope,
            action_id=str(value.get("action_id") or ""),
            intent_id=str(value.get("intent_id") or ""),
            event_ids=value.get("event_ids", ()),
            turn=int(value.get("turn", 0)),
            workspace_revision=int(value.get("workspace_revision", 0)),
            source_revision=str(value.get("source_revision") or ""),
            env_id=str(value.get("env_id") or ""),
            oracle_id=str(value.get("oracle_id") or ""),
            receipts=tuple(Receipt.from_mapping(item) for item in _items(value.get("receipts", ()), "receipts")),
            world_fact_candidates=tuple(raw_facts),
            semantic_evidence=tuple(
                SemanticEvidence.from_mapping(item)
                for item in _items(value.get("semantic_evidence", ()), "semantic_evidence")
            ),
            negative_work=tuple(
                NegativeWork.from_mapping(item)
                for item in _items(value.get("negative_work", ()), "negative_work")
            ),
            failure_classification=str(value.get("failure_classification") or "NOT_OBSERVED"),
            model_proposals=tuple(
                ModelProposal.from_mapping(item)
                for item in _items(value.get("model_proposals", ()), "model_proposals")
            ),
            unknown_reasons=tuple(
                UnknownReason.from_mapping(item)
                for item in _items(value.get("unknown_reasons", ()), "unknown_reasons")
            ),
        )
        _check_digest(value, record.digest)
        return record


def _assert_world_fact_scope(fact: WorldFact, scope: RunScope) -> None:
    fact_scope = {str(key): str(value) for key, value in fact.scope.items()}
    for key in (
        "sandbox_id",
        "workspace_id",
        "mission_id",
        "unit_id",
        "task_id",
        "source_revision",
        "env_id",
        "oracle_id",
    ):
        actual = fact_scope.get(key, "")
        expected = getattr(scope, key)
        if actual and actual != expected:
            raise ScopeViolation(f"world fact {fact.fact_id} belongs to {key}={actual!r}, not this run")


def _world_fact_status(fact: WorldFact, workspace_revision: int) -> Freshness:
    if fact.freshness == "UNKNOWN" or fact.verification != "DETERMINISTIC":
        return Freshness.UNKNOWN
    if fact.workspace_revision != workspace_revision or fact.freshness != "FRESH":
        return Freshness.STALE
    return Freshness.FRESH


@dataclass(frozen=True, slots=True)
class Objective:
    acceptance_atoms: tuple[str, ...]
    non_goals: tuple[str, ...]
    completion_state: CompletionState | str
    provenance: Provenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "acceptance_atoms", _strings(self.acceptance_atoms, "acceptance_atoms", sort=True))
        object.__setattr__(self, "non_goals", _strings(self.non_goals, "non_goals", sort=True))
        object.__setattr__(self, "completion_state", _enum(
            CompletionState, self.completion_state, "completion_state"))
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, object]:
        return {
            "acceptance_atoms": list(self.acceptance_atoms),
            "non_goals": list(self.non_goals),
            "completion_state": self.completion_state.value,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Objective":
        return cls(
            acceptance_atoms=value.get("acceptance_atoms", ()),
            non_goals=value.get("non_goals", ()),
            completion_state=str(value.get("completion_state") or "NOT_STARTED"),
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )


@dataclass(frozen=True, slots=True)
class CurrentState:
    world_revision: int
    last_product_result: str
    last_action: str
    progress_state: str
    persistence_state: PersistenceState | str
    provenance: Provenance
    world_facts: tuple[WorldFact, ...] = ()
    unknown_world_fact_count: int = field(init=False, default=0)
    unknown_world_fact_digest: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.world_revision, int) or self.world_revision < 0:
            raise SchemaError("world_revision must be a non-negative integer")
        object.__setattr__(self, "last_product_result", _optional_text(self.last_product_result))
        object.__setattr__(self, "last_action", _optional_text(self.last_action))
        object.__setattr__(self, "progress_state", _text(self.progress_state, "progress_state"))
        object.__setattr__(self, "persistence_state", _enum(
            PersistenceState, self.persistence_state, "persistence_state"))
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        facts = []
        for item in _items(self.world_facts, "world_facts"):
            if not isinstance(item, WorldFact):
                if isinstance(item, Mapping):
                    item = WorldFact.from_dict(item)
                else:
                    raise SchemaError("world_facts must contain WorldFact records")
            facts.append(item)
        normalized_facts = tuple(sorted(facts, key=lambda item: item.fact_id))
        object.__setattr__(self, "world_facts", normalized_facts)
        unknown = [
            item for item in normalized_facts
            if _world_fact_status(item, self.world_revision) is Freshness.UNKNOWN
        ]
        omitted = unknown[MAX_VISIBLE_UNKNOWN:]
        object.__setattr__(self, "unknown_world_fact_count", len(unknown))
        object.__setattr__(
            self,
            "unknown_world_fact_digest",
            _digest([item.to_dict() for item in omitted]) if omitted else None,
        )

    def to_dict(self) -> dict[str, object]:
        fresh = [item for item in self.world_facts if _world_fact_status(item, self.world_revision) is Freshness.FRESH]
        stale = [item for item in self.world_facts if _world_fact_status(item, self.world_revision) is Freshness.STALE]
        unknown = [item for item in self.world_facts if _world_fact_status(item, self.world_revision) is Freshness.UNKNOWN]
        return {
            "world_revision": self.world_revision,
            "last_product_result": self.last_product_result,
            "last_action": self.last_action,
            "progress_state": self.progress_state,
            "persistence_state": self.persistence_state.value,
            "world_facts": {
                "fresh": [item.to_dict() for item in fresh],
                "stale": [item.to_dict() for item in stale],
                "unknown": [item.to_dict() for item in unknown[:MAX_VISIBLE_UNKNOWN]],
                "unknown_count": self.unknown_world_fact_count,
                "unknown_digest": self.unknown_world_fact_digest,
            },
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CurrentState":
        facts_value = value.get("world_facts", ())
        unknown_count: int | None = None
        unknown_digest: str | None = None
        if isinstance(facts_value, Mapping):
            raw_unknown_count = facts_value.get("unknown_count")
            if raw_unknown_count is not None:
                unknown_count = int(raw_unknown_count)
            raw_unknown_digest = facts_value.get("unknown_digest")
            unknown_digest = str(raw_unknown_digest) if raw_unknown_digest else None
            facts_value = (*_items(facts_value.get("fresh", ()), "fresh facts"),
                           *_items(facts_value.get("stale", ()), "stale facts"),
                           *_items(facts_value.get("unknown", ()), "unknown facts"))
        record = cls(
            world_revision=int(value.get("world_revision", 0)),
            last_product_result=str(value.get("last_product_result") or ""),
            last_action=str(value.get("last_action") or ""),
            progress_state=str(value.get("progress_state") or "UNKNOWN"),
            persistence_state=str(value.get("persistence_state") or "UNKNOWN"),
            world_facts=tuple(WorldFact.from_dict(item) for item in _items(facts_value, "world_facts")),
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )
        if unknown_count is not None:
            if unknown_count < 0:
                raise SchemaError("world_facts.unknown_count must be non-negative")
            if unknown_count < record.unknown_world_fact_count:
                raise SchemaError("world_facts.unknown_count is smaller than visible unknown facts")
            if unknown_count > record.unknown_world_fact_count and not unknown_digest:
                raise SchemaError("world_facts.unknown_digest is required for omitted unknown facts")
            if unknown_count == record.unknown_world_fact_count and unknown_digest:
                raise SchemaError("world_facts.unknown_digest must be empty when no facts are omitted")
            object.__setattr__(record, "unknown_world_fact_count", unknown_count)
            object.__setattr__(record, "unknown_world_fact_digest", unknown_digest)
        return record


@dataclass(frozen=True, slots=True)
class Frontier:
    highest_value_question: str
    recommended_action: str
    why_now: str
    unlocks: tuple[str, ...]
    safe_alternatives: tuple[str, ...]
    off_frontier_policy: str
    provenance: Provenance

    def __post_init__(self) -> None:
        for name in ("highest_value_question", "recommended_action", "why_now", "off_frontier_policy"):
            object.__setattr__(self, name, _text(getattr(self, name), name, limit=1000))
        object.__setattr__(self, "unlocks", _strings(self.unlocks, "unlocks", sort=True))
        object.__setattr__(self, "safe_alternatives", _strings(
            self.safe_alternatives, "safe_alternatives", sort=True))
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, object]:
        return {
            "highest_value_question": self.highest_value_question,
            "recommended_action": self.recommended_action,
            "why_now": self.why_now,
            "unlocks": list(self.unlocks),
            "safe_alternatives": list(self.safe_alternatives),
            "off_frontier_policy": self.off_frontier_policy,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Frontier":
        return cls(
            highest_value_question=str(value.get("highest_value_question") or ""),
            recommended_action=str(value.get("recommended_action") or ""),
            why_now=str(value.get("why_now") or ""),
            unlocks=value.get("unlocks", ()),
            safe_alternatives=value.get("safe_alternatives", ()),
            off_frontier_policy=str(value.get("off_frontier_policy") or ""),
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )


@dataclass(frozen=True, slots=True)
class DependencySummary:
    blocking: tuple[str, ...]
    ready: tuple[str, ...]
    reopened: tuple[str, ...]
    unknown_count: int
    cross_dependency_summary: str
    provenance: Provenance

    def __post_init__(self) -> None:
        for name in ("blocking", "ready", "reopened"):
            object.__setattr__(self, name, _strings(getattr(self, name), name, sort=True))
        if not isinstance(self.unknown_count, int) or self.unknown_count < 0:
            raise SchemaError("dependency unknown_count must be non-negative")
        object.__setattr__(self, "cross_dependency_summary", _text(
            self.cross_dependency_summary, "cross_dependency_summary", limit=1000))
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, object]:
        return {
            "blocking": list(self.blocking),
            "ready": list(self.ready),
            "reopened": list(self.reopened),
            "unknown_count": self.unknown_count,
            "cross_dependency_summary": self.cross_dependency_summary,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DependencySummary":
        return cls(
            blocking=value.get("blocking", ()),
            ready=value.get("ready", ()),
            reopened=value.get("reopened", ()),
            unknown_count=int(value.get("unknown_count", 0)),
            cross_dependency_summary=str(value.get("cross_dependency_summary") or ""),
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )


@dataclass(frozen=True, slots=True)
class ResumeState:
    next_safe_action: str
    last_exit: str
    frame_digest: str
    event_cursor: int
    provenance: Provenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "next_safe_action", _text(self.next_safe_action, "next_safe_action"))
        object.__setattr__(self, "last_exit", _optional_text(self.last_exit))
        object.__setattr__(self, "frame_digest", _text(self.frame_digest, "frame_digest"))
        if not isinstance(self.event_cursor, int) or self.event_cursor < 0:
            raise SchemaError("event_cursor must be a non-negative integer")
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, object]:
        return {
            "next_safe_action": self.next_safe_action,
            "last_exit": self.last_exit,
            "frame_digest": self.frame_digest,
            "event_cursor": self.event_cursor,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ResumeState":
        return cls(
            next_safe_action=str(value.get("next_safe_action") or ""),
            last_exit=str(value.get("last_exit") or ""),
            frame_digest=str(value.get("frame_digest") or ""),
            event_cursor=int(value.get("event_cursor", 0)),
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )


def _contract_atom(value: object) -> ContractAtom:
    if isinstance(value, ContractAtom):
        return value
    if not isinstance(value, Mapping):
        raise SchemaError("contract atoms must be kernel_nitpicker.ContractAtom records")
    spans = tuple(_span(item, "contract provenance") for item in _items(
        value.get("provenance", ()), "contract provenance"))
    target_spans = tuple(_span(item, "contract target_spans") for item in _items(
        value.get("target_spans", ()), "contract target_spans"))
    return ContractAtom(
        kind=str(value.get("kind") or "policy"),
        subject=str(value.get("subject") or ""),
        expected=value.get("expected"),
        observed=value.get("observed"),
        state=str(value.get("state") or "UNKNOWN"),
        basis=str(value.get("basis") or "TEST_AST"),
        provenance=spans,
        target_spans=target_spans,
        task_id=str(value.get("task_id") or ""),
        target_symbol=str(value.get("target_symbol") or ""),
        contract_key=str(value.get("contract_key") or ""),
        fingerprint=str(value.get("fingerprint") or ""),
        notes=value.get("notes", ()),
    )


@dataclass(frozen=True, slots=True)
class ContractSummary:
    violated_atoms: tuple[ContractAtom, ...]
    unresolved_atoms: tuple[ContractAtom, ...]
    supported_atoms: tuple[ContractAtom, ...]
    action_unlocking_unknown: "CandidateAdvisory | None" = None

    def __post_init__(self) -> None:
        atom_groups = []
        for name in ("violated_atoms", "unresolved_atoms", "supported_atoms"):
            atoms = [_contract_atom(item) for item in _items(getattr(self, name), name)]
            atoms = _dedupe_atoms(atoms, name)
            atom_groups.append(atoms)
            object.__setattr__(self, name, tuple(atoms))
        if sum(len(group) for group in atom_groups) > MAX_VISIBLE_CONTRACT_ATOMS:
            raise SchemaError(
                f"contract atoms exceed the {MAX_VISIBLE_CONTRACT_ATOMS}-atom attention floor"
            )
        if self.action_unlocking_unknown is not None:
            if not isinstance(self.action_unlocking_unknown, CandidateAdvisory):
                raise SchemaError("action_unlocking_unknown must be CandidateAdvisory")
            if (self.action_unlocking_unknown.kind is not AdvisoryKind.UNKNOWN
                    or not self.action_unlocking_unknown.action_unlocking):
                raise SchemaError("action_unlocking_unknown must be action-unlocking UNKNOWN")

    @property
    def provenance(self) -> Provenance:
        spans = tuple(span for atom in (
            *self.violated_atoms, *self.unresolved_atoms, *self.supported_atoms)
            for span in atom.provenance)
        if self.action_unlocking_unknown is not None:
            return Provenance(
                source_spans=spans + self.action_unlocking_unknown.provenance.source_spans,
                event_ids=self.action_unlocking_unknown.provenance.event_ids,
                receipt_ids=self.action_unlocking_unknown.provenance.receipt_ids,
                fact_ids=self.action_unlocking_unknown.provenance.fact_ids,
            )
        return Provenance(source_spans=spans)

    def to_dict(self) -> dict[str, object]:
        return {
            "violated_atoms": [item.to_dict() for item in self.violated_atoms],
            "unresolved_atoms": [item.to_dict() for item in self.unresolved_atoms],
            "supported_atoms": [item.to_dict() for item in self.supported_atoms],
            "action_unlocking_unknown": (
                self.action_unlocking_unknown.to_dict()
                if self.action_unlocking_unknown is not None else None
            ),
            "provenance": self.provenance.to_dict(),
        }


def _atom_key(atom: ContractAtom) -> str:
    return str(atom.contract_key or canonical_json(atom.to_dict()))


def _dedupe_atoms(atoms: Sequence[ContractAtom], field_name: str) -> list[ContractAtom]:
    by_key: dict[str, ContractAtom] = {}
    for atom in atoms:
        key = _atom_key(atom)
        prior = by_key.get(key)
        if prior is not None and canonical_json(prior.to_dict()) != canonical_json(atom.to_dict()):
            raise SchemaError(f"conflicting duplicate contract atom {key!r} in {field_name}")
        by_key[key] = atom
    return [by_key[key] for key in sorted(by_key)]


@dataclass(frozen=True, slots=True)
class CandidateAdvisory:
    """One ranked model-facing candidate with a mandatory provenance anchor."""

    candidate_id: str
    kind: AdvisoryKind | str
    text: str
    provenance: Provenance
    mandatory: bool = False
    blocks_frontier: bool = False
    contradicts_theory: bool = False
    unlock_value: int = 0
    risk: int = 0
    freshness: int = 0
    priority: int = 0
    action_unlocking: bool = False
    unknown_reason: str | None = None
    scope: RunScope | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "kind", _enum(AdvisoryKind, self.kind, "advisory kind"))
        object.__setattr__(self, "text", _text(self.text, "advisory text", limit=1000))
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        for name in ("unlock_value", "risk", "freshness", "priority"):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise SchemaError(f"advisory {name} must be an integer")
        object.__setattr__(self, "mandatory", bool(self.mandatory))
        object.__setattr__(self, "blocks_frontier", bool(self.blocks_frontier))
        object.__setattr__(self, "contradicts_theory", bool(self.contradicts_theory))
        object.__setattr__(self, "action_unlocking", bool(self.action_unlocking))
        reason = _optional_text(self.unknown_reason, limit=1000)
        if self.kind is AdvisoryKind.UNKNOWN and not reason:
            raise SchemaError(f"UNKNOWN advisory {self.candidate_id} needs unknown_reason")
        if self.kind is not AdvisoryKind.UNKNOWN and reason:
            raise SchemaError(f"non-UNKNOWN advisory {self.candidate_id} has unknown_reason")
        object.__setattr__(self, "unknown_reason", reason or None)
        if self.scope is not None and not isinstance(self.scope, RunScope):
            raise SchemaError("advisory scope must be RunScope")

    @property
    def rank_key(self) -> tuple[object, ...]:
        return (
            -int(self.mandatory),
            -int(self.blocks_frontier),
            -int(self.contradicts_theory),
            -self.unlock_value,
            -self.risk,
            -self.freshness,
            -self.priority,
            self.candidate_id,
        )

    @property
    def content_key(self) -> str:
        return canonical_json({
            "kind": self.kind.value,
            "text": self.text,
            "mandatory": self.mandatory,
            "blocks_frontier": self.blocks_frontier,
            "contradicts_theory": self.contradicts_theory,
            "unlock_value": self.unlock_value,
            "risk": self.risk,
            "freshness": self.freshness,
            "priority": self.priority,
            "action_unlocking": self.action_unlocking,
            "unknown_reason": self.unknown_reason,
        })

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind.value,
            "text": self.text,
            "mandatory": self.mandatory,
            "blocks_frontier": self.blocks_frontier,
            "contradicts_theory": self.contradicts_theory,
            "unlock_value": self.unlock_value,
            "risk": self.risk,
            "freshness": self.freshness,
            "priority": self.priority,
            "action_unlocking": self.action_unlocking,
            "unknown_reason": self.unknown_reason,
            "provenance": self.provenance.to_dict(),
            "scope": self.scope.to_dict() if self.scope is not None else None,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CandidateAdvisory":
        scope_value = value.get("scope")
        return cls(
            candidate_id=str(value.get("candidate_id") or ""),
            kind=str(value.get("kind") or "UNKNOWN"),
            text=str(value.get("text") or ""),
            mandatory=bool(value.get("mandatory", False)),
            blocks_frontier=bool(value.get("blocks_frontier", False)),
            contradicts_theory=bool(value.get("contradicts_theory", False)),
            unlock_value=int(value.get("unlock_value", 0)),
            risk=int(value.get("risk", 0)),
            freshness=int(value.get("freshness", 0)),
            priority=int(value.get("priority", 0)),
            action_unlocking=bool(value.get("action_unlocking", False)),
            unknown_reason=str(value.get("unknown_reason") or "") or None,
            scope=RunScope.from_mapping(scope_value) if isinstance(scope_value, Mapping) else None,
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )


def _dedupe_advisories(
    values: Sequence[CandidateAdvisory],
) -> tuple[list[CandidateAdvisory], int]:
    by_id: dict[str, CandidateAdvisory] = {}
    duplicates = 0
    for candidate in values:
        prior = by_id.get(candidate.candidate_id)
        if prior is None:
            by_id[candidate.candidate_id] = candidate
            continue
        if prior.content_key != candidate.content_key:
            raise SchemaError(f"conflicting duplicate candidate advisory {candidate.candidate_id!r}")
        by_id[candidate.candidate_id] = replace(
            prior, provenance=prior.provenance.merged(candidate.provenance))
        duplicates += 1
    return [by_id[key] for key in sorted(by_id)], duplicates


def _unknown_projection(
    values: Sequence[CandidateAdvisory],
) -> tuple[list[CandidateAdvisory], int, str | None, list[str]]:
    unknown = [item for item in values if item.kind is AdvisoryKind.UNKNOWN]
    if not unknown:
        return [], 0, None, []
    selectable = [item for item in unknown if item.action_unlocking]
    selected = (
        sorted(
            selectable,
            key=lambda item: (
                -item.unlock_value,
                -int(item.mandatory),
                -int(item.blocks_frontier),
                -int(item.contradicts_theory),
                -item.risk,
                -item.freshness,
                item.priority,
                item.candidate_id,
            ),
        )[0]
        if selectable
        else None
    )
    omitted = [item for item in unknown if selected is None or item.candidate_id != selected.candidate_id]
    digest = _digest([item.to_dict() for item in omitted]) if omitted else None
    return ([selected] if selected is not None else [], len(unknown), digest,
            [item.candidate_id for item in omitted])


@dataclass(frozen=True, slots=True)
class CertaintyPreAnswer:
    """Deterministic answer presented at a certainty demand point."""

    certainty_status: CertaintyStatus | str
    what_is_proven: tuple[str, ...]
    what_is_not_proven: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    required_next_evidence: tuple[str, ...]
    safe_next_action: str
    can_edit: EditPermission | str
    stop_allowed: bool
    success_certified: bool
    provenance: Provenance
    unknown_reasons: tuple[UnknownReason, ...] = ()
    unknown_reason_count: int = field(init=False, default=0)
    unknown_reason_digest: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "certainty_status", _enum(
            CertaintyStatus, self.certainty_status, "certainty_status"))
        object.__setattr__(self, "what_is_proven", _strings(
            self.what_is_proven, "what_is_proven", sort=True))
        object.__setattr__(self, "what_is_not_proven", _strings(
            self.what_is_not_proven, "what_is_not_proven", sort=True))
        object.__setattr__(self, "evidence_refs", _strings(
            self.evidence_refs, "evidence_refs", sort=True))
        object.__setattr__(self, "required_next_evidence", _strings(
            self.required_next_evidence, "required_next_evidence", sort=True))
        object.__setattr__(self, "safe_next_action", _text(
            self.safe_next_action, "safe_next_action", limit=1000))
        object.__setattr__(self, "can_edit", _enum(EditPermission, self.can_edit, "can_edit"))
        object.__setattr__(self, "stop_allowed", bool(self.stop_allowed))
        object.__setattr__(self, "success_certified", bool(self.success_certified))
        if self.success_certified and self.certainty_status is not CertaintyStatus.VERIFIED:
            raise SchemaError("success_certified requires certainty_status=VERIFIED")
        if self.success_certified and not self.evidence_refs:
            raise SchemaError("success_certified requires evidence_refs")
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        reasons = []
        for item in _items(self.unknown_reasons, "unknown_reasons"):
            reasons.append(item if isinstance(item, UnknownReason) else UnknownReason.from_mapping(item))
        by_id = {item.unknown_id: item for item in reasons}
        normalized_reasons = tuple(by_id[key] for key in sorted(by_id))
        object.__setattr__(self, "unknown_reasons", normalized_reasons)
        omitted = normalized_reasons[MAX_VISIBLE_UNKNOWN:]
        object.__setattr__(self, "unknown_reason_count", len(normalized_reasons))
        object.__setattr__(
            self,
            "unknown_reason_digest",
            _digest([item.to_dict() for item in omitted]) if omitted else None,
        )

    @property
    def unknown_count(self) -> int:
        return self.unknown_reason_count

    @property
    def visible_unknown_reasons(self) -> tuple[UnknownReason, ...]:
        return self.unknown_reasons[:MAX_VISIBLE_UNKNOWN]

    @property
    def unknown_digest(self) -> str | None:
        return self.unknown_reason_digest

    def to_dict(self) -> dict[str, object]:
        return {
            "certainty_status": self.certainty_status.value,
            "what_is_proven": list(self.what_is_proven),
            "what_is_not_proven": list(self.what_is_not_proven),
            "evidence_refs": list(self.evidence_refs),
            "required_next_evidence": list(self.required_next_evidence),
            "safe_next_action": self.safe_next_action,
            "can_edit": self.can_edit.value,
            "stop_allowed": self.stop_allowed,
            "success_certified": self.success_certified,
            "unknown_reasons": [item.to_dict() for item in self.visible_unknown_reasons],
            "unknown_count": self.unknown_count,
            "unknown_digest": self.unknown_digest,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CertaintyPreAnswer":
        raw_count = value.get("unknown_count")
        raw_digest = value.get("unknown_digest")
        unknown_digest = str(raw_digest) if raw_digest else None
        record = cls(
            certainty_status=str(value.get("certainty_status") or "NOT_YET"),
            what_is_proven=value.get("what_is_proven", ()),
            what_is_not_proven=value.get("what_is_not_proven", ()),
            evidence_refs=value.get("evidence_refs", ()),
            required_next_evidence=value.get("required_next_evidence", ()),
            safe_next_action=str(value.get("safe_next_action") or ""),
            can_edit=str(value.get("can_edit") or "NO"),
            stop_allowed=bool(value.get("stop_allowed", False)),
            success_certified=bool(value.get("success_certified", False)),
            unknown_reasons=tuple(
                UnknownReason.from_mapping(item)
                for item in _items(value.get("unknown_reasons", ()), "unknown_reasons")
            ),
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )
        if raw_count is not None:
            unknown_count = int(raw_count)
            if unknown_count < 0:
                raise SchemaError("unknown_count must be non-negative")
            if unknown_count < len(record.visible_unknown_reasons):
                raise SchemaError("unknown_count is smaller than visible unknown reasons")
            if unknown_count > len(record.visible_unknown_reasons) and not unknown_digest:
                raise SchemaError("unknown_digest is required for omitted unknown reasons")
            if unknown_count == len(record.visible_unknown_reasons) and unknown_digest:
                raise SchemaError("unknown_digest must be empty when no reasons are omitted")
            object.__setattr__(record, "unknown_reason_count", unknown_count)
            object.__setattr__(record, "unknown_reason_digest", unknown_digest)
        elif unknown_digest:
            raise SchemaError("unknown_count is required when unknown_digest is present")
        return record


@dataclass(frozen=True, slots=True)
class CertaintyDemand:
    """One deduplicable get-sure demand with its deterministic pre-answer."""

    SCHEMA: ClassVar[str] = CERTAINTY_DEMAND_SCHEMA

    scope: RunScope
    demand_point: DemandPoint | str
    subject: str
    source_revision: str
    dependency_generation: int
    pre_answer: CertaintyPreAnswer

    def __post_init__(self) -> None:
        if not isinstance(self.scope, RunScope):
            raise SchemaError("certainty demand scope must be RunScope")
        object.__setattr__(self, "demand_point", _enum(
            DemandPoint, self.demand_point, "demand_point"))
        object.__setattr__(self, "subject", _text(self.subject, "demand subject"))
        source_revision = _text(self.source_revision, "source_revision")
        if source_revision != self.scope.source_revision:
            raise ScopeViolation("certainty demand source_revision does not match scope")
        object.__setattr__(self, "source_revision", source_revision)
        if not isinstance(self.dependency_generation, int) or self.dependency_generation < 0:
            raise SchemaError("dependency_generation must be non-negative")
        if not isinstance(self.pre_answer, CertaintyPreAnswer):
            raise SchemaError("pre_answer must be CertaintyPreAnswer")

    @property
    def required_evidence(self) -> tuple[str, ...]:
        return self.pre_answer.required_next_evidence

    @property
    def stop_allowed(self) -> bool:
        return self.pre_answer.stop_allowed

    @property
    def success_certified(self) -> bool:
        return self.pre_answer.success_certified

    @property
    def dedup_key(self) -> tuple[object, ...]:
        return (
            self.demand_point.value,
            self.subject,
            self.source_revision,
            _digest(self.pre_answer.to_dict()),
            self.dependency_generation,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "scope": self.scope.to_dict(),
            "demand_point": self.demand_point.value,
            "subject": self.subject,
            "source_revision": self.source_revision,
            "dependency_generation": self.dependency_generation,
            "dedup_key": list(self.dedup_key),
            "pre_answer": self.pre_answer.to_dict(),
            "required_evidence": list(self.required_evidence),
            "stop_allowed": self.stop_allowed,
            "success_certified": self.success_certified,
        }

    @property
    def digest(self) -> str:
        return _digest({"schema": self.SCHEMA, **self._payload()})

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, **self._payload(), "digest": self.digest}

    def serialize(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CertaintyDemand":
        _check_schema(value, cls.SCHEMA)
        record = cls(
            scope=RunScope.from_mapping(value.get("scope", {})),
            demand_point=str(value.get("demand_point") or ""),
            subject=str(value.get("subject") or ""),
            source_revision=str(value.get("source_revision") or ""),
            dependency_generation=int(value.get("dependency_generation", 0)),
            pre_answer=CertaintyPreAnswer.from_mapping(value.get("pre_answer", {})),
        )
        _check_digest(value, record.digest)
        return record


@dataclass(frozen=True, slots=True)
class HypothesisAssessment:
    """Receipt-backed status and deterministic weights for one theory."""

    SCHEMA: ClassVar[str] = HYPOTHESIS_ASSESSMENT_SCHEMA

    scope: RunScope
    hypothesis_id: str
    claim: str
    status: HypothesisStatus | str
    evidence_weights: Mapping[str, int]
    evidence_refs: tuple[str, ...]
    provenance: Provenance
    kill_signal: str | None = None
    unknown_reason: str | None = None
    retention_reason_code: str = ""
    new_prediction: str = ""
    next_test_or_observation: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.scope, RunScope):
            raise SchemaError("hypothesis scope must be RunScope")
        object.__setattr__(self, "hypothesis_id", _text(self.hypothesis_id, "hypothesis_id"))
        object.__setattr__(self, "claim", _text(self.claim, "hypothesis claim", limit=1000))
        object.__setattr__(self, "status", _enum(HypothesisStatus, self.status, "hypothesis status"))
        weights: dict[str, int] = {}
        if not isinstance(self.evidence_weights, Mapping):
            raise SchemaError("evidence_weights must be a mapping")
        for key, value in self.evidence_weights.items():
            if not isinstance(value, int):
                raise SchemaError("hypothesis evidence weights must be integers")
            weights[str(key)] = value
        object.__setattr__(self, "evidence_weights", {
            key: weights[key] for key in sorted(weights)
        })
        refs = _strings(self.evidence_refs, "evidence_refs", sort=True)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        kill = _optional_text(self.kill_signal, limit=1000)
        reason = _optional_text(self.unknown_reason, limit=1000)
        if self.status is HypothesisStatus.CONTRADICTED:
            if not kill or not refs:
                raise SchemaError("CONTRADICTED hypotheses need kill_signal and evidence_refs")
        if self.status is HypothesisStatus.UNKNOWN and not reason:
            raise SchemaError(f"UNKNOWN hypothesis {self.hypothesis_id} needs unknown_reason")
        if self.status is not HypothesisStatus.UNKNOWN and reason:
            raise SchemaError(f"non-UNKNOWN hypothesis {self.hypothesis_id} has unknown_reason")
        object.__setattr__(self, "kill_signal", kill or None)
        object.__setattr__(self, "unknown_reason", reason or None)
        for name in ("retention_reason_code", "new_prediction", "next_test_or_observation"):
            object.__setattr__(self, name, _optional_text(getattr(self, name), limit=1000))

    def _payload(self) -> dict[str, object]:
        return {
            "scope": self.scope.to_dict(),
            "hypothesis_id": self.hypothesis_id,
            "claim": self.claim,
            "status": self.status.value,
            "evidence_weights": dict(self.evidence_weights),
            "evidence_refs": list(self.evidence_refs),
            "kill_signal": self.kill_signal,
            "unknown_reason": self.unknown_reason,
            "retention_reason_code": self.retention_reason_code,
            "new_prediction": self.new_prediction,
            "next_test_or_observation": self.next_test_or_observation,
            "provenance": self.provenance.to_dict(),
        }

    @property
    def digest(self) -> str:
        return _digest({"schema": self.SCHEMA, **self._payload()})

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, **self._payload(), "digest": self.digest}

    def serialize(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "HypothesisAssessment":
        _check_schema(value, cls.SCHEMA)
        record = cls(
            scope=RunScope.from_mapping(value.get("scope", {})),
            hypothesis_id=str(value.get("hypothesis_id") or ""),
            claim=str(value.get("claim") or ""),
            status=str(value.get("status") or "UNKNOWN"),
            evidence_weights=value.get("evidence_weights", {}),
            evidence_refs=value.get("evidence_refs", ()),
            kill_signal=str(value.get("kill_signal") or "") or None,
            unknown_reason=str(value.get("unknown_reason") or "") or None,
            retention_reason_code=str(value.get("retention_reason_code") or ""),
            new_prediction=str(value.get("new_prediction") or ""),
            next_test_or_observation=str(value.get("next_test_or_observation") or ""),
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )
        _check_digest(value, record.digest)
        return record


@dataclass(frozen=True, slots=True)
class DecisionDeclaration:
    """Model action declaration; it explains intent but never writes truth."""

    SCHEMA: ClassVar[str] = DECISION_DECLARATION_SCHEMA

    scope: RunScope
    selected_frontier_id: str
    action_kind: ActionKind | str
    target_scope: Mapping[str, str]
    intent: str
    expected_observation: str
    hypothesis_id: str
    contract_keys: tuple[str, ...]
    why_code: str
    off_frontier_reason: str
    certainty_acknowledgement: CertaintyStatus | str
    provenance: Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.scope, RunScope):
            raise SchemaError("decision scope must be RunScope")
        object.__setattr__(self, "selected_frontier_id", _text(
            self.selected_frontier_id, "selected_frontier_id"))
        object.__setattr__(self, "action_kind", _enum(ActionKind, self.action_kind, "action_kind"))
        target = _mapping(self.target_scope, "target_scope")
        normalized_target: dict[str, str] = {}
        for key, value in target.items():
            normalized_target[key] = _text(value, f"target_scope.{key}")
        if not normalized_target:
            raise SchemaError("target_scope needs at least one target atom")
        _assert_optional_scope_fields(normalized_target, self.scope, "target_scope")
        object.__setattr__(self, "target_scope", normalized_target)
        for name in ("intent", "expected_observation", "why_code"):
            object.__setattr__(self, name, _text(getattr(self, name), name, limit=1000))
        object.__setattr__(self, "hypothesis_id", _optional_text(self.hypothesis_id))
        object.__setattr__(self, "contract_keys", _strings(self.contract_keys, "contract_keys", sort=True))
        object.__setattr__(self, "off_frontier_reason", _optional_text(self.off_frontier_reason, limit=1000))
        object.__setattr__(self, "certainty_acknowledgement", _enum(
            CertaintyStatus, self.certainty_acknowledgement, "certainty_acknowledgement"))
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def _payload(self) -> dict[str, object]:
        return {
            "scope": self.scope.to_dict(),
            "selected_frontier_id": self.selected_frontier_id,
            "action_kind": self.action_kind.value,
            "target_scope": dict(self.target_scope),
            "intent": self.intent,
            "expected_observation": self.expected_observation,
            "hypothesis_id": self.hypothesis_id,
            "contract_keys": list(self.contract_keys),
            "why_code": self.why_code,
            "off_frontier_reason": self.off_frontier_reason,
            "certainty_acknowledgement": self.certainty_acknowledgement.value,
            "provenance": self.provenance.to_dict(),
        }

    @property
    def digest(self) -> str:
        return _digest({"schema": self.SCHEMA, **self._payload()})

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, **self._payload(), "digest": self.digest}

    def serialize(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DecisionDeclaration":
        _check_schema(value, cls.SCHEMA)
        record = cls(
            scope=RunScope.from_mapping(value.get("scope", {})),
            selected_frontier_id=str(value.get("selected_frontier_id") or ""),
            action_kind=str(value.get("action_kind") or "OTHER"),
            target_scope=value.get("target_scope", {}),
            intent=str(value.get("intent") or ""),
            expected_observation=str(value.get("expected_observation") or ""),
            hypothesis_id=str(value.get("hypothesis_id") or ""),
            contract_keys=value.get("contract_keys", ()),
            why_code=str(value.get("why_code") or ""),
            off_frontier_reason=str(value.get("off_frontier_reason") or ""),
            certainty_acknowledgement=str(value.get("certainty_acknowledgement") or "NOT_YET"),
            provenance=Provenance.from_mapping(value.get("provenance", {})),
        )
        _check_digest(value, record.digest)
        return record


def _merge_hypotheses(values: Sequence[HypothesisAssessment]) -> tuple[list[HypothesisAssessment], int, str | None]:
    by_id: dict[str, HypothesisAssessment] = {}
    duplicates = 0
    for item in values:
        prior = by_id.get(item.hypothesis_id)
        if prior is None:
            by_id[item.hypothesis_id] = item
            continue
        if prior.status != item.status or prior.claim != item.claim:
            raise SchemaError(f"conflicting duplicate hypothesis {item.hypothesis_id!r}")
        if dict(prior.evidence_weights) != dict(item.evidence_weights):
            raise SchemaError(f"conflicting evidence weights for hypothesis {item.hypothesis_id!r}")
        by_id[item.hypothesis_id] = replace(
            prior,
            evidence_refs=(*prior.evidence_refs, *item.evidence_refs),
            provenance=prior.provenance.merged(item.provenance),
        )
        duplicates += 1
    normalized = [by_id[key] for key in sorted(by_id)]
    unknown = [item for item in normalized if item.status is HypothesisStatus.UNKNOWN]
    visible_unknown = unknown[:MAX_VISIBLE_UNKNOWN]
    omitted_digest = _digest([item.to_dict() for item in unknown[MAX_VISIBLE_UNKNOWN:]]) \
        if len(unknown) > MAX_VISIBLE_UNKNOWN else None
    return [item for item in normalized if item.status is not HypothesisStatus.UNKNOWN] + visible_unknown, duplicates, omitted_digest


@dataclass(frozen=True, slots=True)
class SituationCard:
    """The single bounded, model-facing composition projection."""

    SCHEMA: ClassVar[str] = SITUATION_CARD_SCHEMA

    run_scope: RunScope
    objective: Objective
    current_state: CurrentState
    frontier: Frontier
    dependencies: DependencySummary
    hypotheses: tuple[HypothesisAssessment, ...]
    contracts: ContractSummary
    certainty: CertaintyDemand
    negative_work: tuple[NegativeWork, ...]
    resume: ResumeState
    advisories: tuple[CandidateAdvisory, ...] = ()
    duplicate_candidate_count: int = field(init=False, default=0)
    unknown_count: int = field(init=False, default=0)
    unknown_digest: str | None = field(init=False, default=None)
    omitted_candidate_count: int = field(init=False, default=0)
    hypothesis_duplicate_count: int = field(init=False, default=0)
    hypothesis_unknown_digest: str | None = field(init=False, default=None)
    hypothesis_unknown_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not isinstance(self.run_scope, RunScope):
            raise SchemaError("situation card run_scope must be RunScope")
        for name, expected_type in (
            ("objective", Objective),
            ("current_state", CurrentState),
            ("frontier", Frontier),
            ("dependencies", DependencySummary),
            ("contracts", ContractSummary),
            ("certainty", CertaintyDemand),
            ("resume", ResumeState),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise SchemaError(f"{name} has the wrong typed schema")
        if not _same_scope(self.certainty.scope, self.run_scope):
            raise ScopeViolation("certainty demand belongs to another run scope")
        for fact in self.current_state.world_facts:
            _assert_world_fact_scope(fact, self.run_scope)
        hypotheses = []
        for item in _items(self.hypotheses, "hypotheses"):
            assessment = item if isinstance(item, HypothesisAssessment) else HypothesisAssessment.from_mapping(item)
            if not _same_scope(assessment.scope, self.run_scope):
                raise ScopeViolation(f"hypothesis {assessment.hypothesis_id} belongs to another run scope")
            hypotheses.append(assessment)
        normalized_hypotheses, duplicate_hypotheses, omitted_hypothesis_digest = _merge_hypotheses(hypotheses)
        object.__setattr__(self, "hypotheses", tuple(normalized_hypotheses))
        object.__setattr__(self, "hypothesis_duplicate_count", duplicate_hypotheses)
        object.__setattr__(self, "hypothesis_unknown_count", sum(
            1 for item in hypotheses if item.status is HypothesisStatus.UNKNOWN))
        object.__setattr__(self, "hypothesis_unknown_digest", omitted_hypothesis_digest)
        for atom_group in (self.contracts.violated_atoms, self.contracts.unresolved_atoms, self.contracts.supported_atoms):
            for atom in atom_group:
                if atom.task_id and atom.task_id != self.run_scope.task_id:
                    raise ScopeViolation(f"contract atom {atom.contract_key} belongs to another task")
        if self.contracts.action_unlocking_unknown is not None:
            candidate_scope = self.contracts.action_unlocking_unknown.scope
            if candidate_scope is not None and not _same_scope(candidate_scope, self.run_scope):
                raise ScopeViolation("contract unknown belongs to another run scope")
        negative = [
            item if isinstance(item, NegativeWork) else NegativeWork.from_mapping(item)
            for item in _items(self.negative_work, "negative_work")
        ]
        object.__setattr__(self, "negative_work", tuple(sorted(
            negative, key=lambda item: (item.turn, item.kind, item.action_signature))))

        candidates = []
        for item in _items(self.advisories, "advisories"):
            candidate = item if isinstance(item, CandidateAdvisory) else CandidateAdvisory.from_mapping(item)
            if candidate.scope is not None and not _same_scope(candidate.scope, self.run_scope):
                raise ScopeViolation(f"advisory {candidate.candidate_id} belongs to another run scope")
            candidates.append(candidate)
        unique_candidates, duplicate_count = _dedupe_advisories(candidates)
        visible_unknown, unknown_count, unknown_digest, omitted_unknown_ids = _unknown_projection(unique_candidates)
        non_unknown = [item for item in unique_candidates if item.kind is not AdvisoryKind.UNKNOWN]
        visible_non_unknown = sorted(non_unknown, key=lambda item: item.rank_key)
        capacity = MAX_VISIBLE_ADVISORIES - len(visible_unknown)
        visible_non_unknown = visible_non_unknown[:max(capacity, 0)]
        visible = sorted((*visible_non_unknown, *visible_unknown), key=lambda item: item.rank_key)
        omitted_ids = [item.candidate_id for item in unique_candidates
                       if item.candidate_id not in {candidate.candidate_id for candidate in visible}]
        omitted_ids.extend(item_id for item_id in omitted_unknown_ids if item_id not in omitted_ids)
        object.__setattr__(self, "advisories", tuple(visible))
        object.__setattr__(self, "duplicate_candidate_count", duplicate_count)
        object.__setattr__(self, "unknown_count", unknown_count)
        object.__setattr__(self, "unknown_digest", unknown_digest)
        object.__setattr__(self, "omitted_candidate_count", len(set(omitted_ids)))

    @property
    def card_digest(self) -> str:
        return _digest({"schema": self.SCHEMA, **self._payload()})

    @property
    def digest(self) -> str:
        return self.card_digest

    def _hypotheses_payload(self) -> dict[str, object]:
        statuses = {
            "live": HypothesisStatus.LIVE,
            "supported": HypothesisStatus.SUPPORTED,
            "contradicted": HypothesisStatus.CONTRADICTED,
            "unknown": HypothesisStatus.UNKNOWN,
        }
        by_status = {
            name: [item.to_dict() for item in self.hypotheses if item.status is status]
            for name, status in statuses.items()
        }
        by_status["unknown_count"] = self.hypothesis_unknown_count
        by_status["unknown_digest"] = self.hypothesis_unknown_digest
        by_status["kill_signal"] = [
            {
                "hypothesis_id": item.hypothesis_id,
                "signal": item.kill_signal,
                "evidence_refs": list(item.evidence_refs),
                "provenance": item.provenance.to_dict(),
            }
            for item in self.hypotheses if item.status is HypothesisStatus.CONTRADICTED
        ]
        by_status["evidence_weights"] = {
            item.hypothesis_id: dict(item.evidence_weights) for item in self.hypotheses
        }
        by_status["provenance"] = [
            item.provenance.to_dict() for item in self.hypotheses
        ]
        return by_status

    def _negative_payload(self) -> dict[str, object]:
        attempted = sorted({item.action_signature for item in self.negative_work})
        ruled_out = sorted({item.detail for item in self.negative_work if item.ruled_out and item.detail})
        counts: dict[str, int] = {}
        for item in self.negative_work:
            counts[item.action_signature] = counts.get(item.action_signature, 0) + 1
        repeated = sorted(
            ((count, signature) for signature, count in counts.items() if count > 1),
            reverse=True,
        )
        return {
            "attempted_actions": attempted,
            "ruled_out_paths": ruled_out,
            "no_op_count": sum(1 for item in self.negative_work if item.is_no_op),
            "repeated_action_signature": repeated[0][1] if repeated else None,
            "entries": [item.to_dict() for item in self.negative_work],
        }

    def _payload(self) -> dict[str, object]:
        return {
            "run_scope": self.run_scope.to_dict(),
            "objective": self.objective.to_dict(),
            "current_state": self.current_state.to_dict(),
            "frontier": self.frontier.to_dict(),
            "dependencies": self.dependencies.to_dict(),
            "hypotheses": self._hypotheses_payload(),
            "contracts": self.contracts.to_dict(),
            "certainty": self.certainty.to_dict(),
            "negative_work": self._negative_payload(),
            "resume": self.resume.to_dict(),
            "advisories": [item.to_dict() for item in self.advisories],
            "attention": {
                "budget": MAX_VISIBLE_ADVISORIES,
                "visible_count": len(self.advisories),
                "unknown_count": self.unknown_count,
                "unknown_digest": self.unknown_digest,
                "omitted_candidate_count": self.omitted_candidate_count,
                "duplicate_candidate_count": self.duplicate_candidate_count,
                "policy": "one action-unlocking UNKNOWN maximum; omitted UNKNOWNs retain count and digest",
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "card_digest": self.card_digest, **self._payload()}

    def serialize(self) -> str:
        return canonical_json(self.to_dict())

    def render(self) -> str:
        """Return the canonical bounded JSON surface sent to a model."""

        return self.serialize()

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SituationCard":
        _check_schema(value, cls.SCHEMA)
        raw_contracts = value.get("contracts", {})
        if not isinstance(raw_contracts, Mapping):
            raise SchemaError("contracts must be a mapping")
        raw_attention = value.get("attention")
        if raw_attention is not None and not isinstance(raw_attention, Mapping):
            raise SchemaError("attention must be a mapping")
        raw_hypotheses = value.get("hypotheses")
        if raw_hypotheses is not None and not isinstance(raw_hypotheses, Mapping):
            raise SchemaError("hypotheses must be a mapping")
        raw_unknown = raw_contracts.get("action_unlocking_unknown")
        record = cls(
            run_scope=RunScope.from_mapping(value.get("run_scope", {})),
            objective=Objective.from_mapping(value.get("objective", {})),
            current_state=CurrentState.from_mapping(value.get("current_state", {})),
            frontier=Frontier.from_mapping(value.get("frontier", {})),
            dependencies=DependencySummary.from_mapping(value.get("dependencies", {})),
            hypotheses=tuple(
                HypothesisAssessment.from_mapping(item)
                for group_name in ("live", "supported", "contradicted", "unknown")
                for item in _items(
                    value.get("hypotheses", {}).get(group_name, ())
                    if isinstance(value.get("hypotheses"), Mapping) else (),
                    f"hypotheses.{group_name}",
                )
            ),
            contracts=ContractSummary(
                violated_atoms=tuple(_contract_atom(item) for item in _items(
                    raw_contracts.get("violated_atoms", ()), "violated_atoms")),
                unresolved_atoms=tuple(_contract_atom(item) for item in _items(
                    raw_contracts.get("unresolved_atoms", ()), "unresolved_atoms")),
                supported_atoms=tuple(_contract_atom(item) for item in _items(
                    raw_contracts.get("supported_atoms", ()), "supported_atoms")),
                action_unlocking_unknown=(CandidateAdvisory.from_mapping(raw_unknown)
                                          if isinstance(raw_unknown, Mapping) else None),
            ),
            certainty=CertaintyDemand.from_mapping(value.get("certainty", {})),
            negative_work=tuple(
                NegativeWork.from_mapping(item)
                for item in _items(
                    value.get("negative_work", {}).get("entries", ())
                    if isinstance(value.get("negative_work"), Mapping) else (),
                    "negative_work.entries",
                )
            ),
            resume=ResumeState.from_mapping(value.get("resume", {})),
            advisories=tuple(
                CandidateAdvisory.from_mapping(item)
                for item in _items(value.get("advisories", ()), "advisories")
            ),
        )
        if isinstance(raw_attention, Mapping):
            raw_unknown_count = raw_attention.get("unknown_count")
            raw_unknown_digest = raw_attention.get("unknown_digest")
            unknown_digest = str(raw_unknown_digest) if raw_unknown_digest else None
            visible_unknown_count = sum(
                1 for item in record.advisories if item.kind is AdvisoryKind.UNKNOWN
            )
            if raw_unknown_count is not None:
                unknown_count = int(raw_unknown_count)
                if unknown_count < visible_unknown_count:
                    raise SchemaError("attention.unknown_count is smaller than visible UNKNOWNs")
                if unknown_count > visible_unknown_count and not unknown_digest:
                    raise SchemaError("attention.unknown_digest is required for omitted UNKNOWNs")
                if unknown_count == visible_unknown_count and unknown_digest:
                    raise SchemaError("attention.unknown_digest must be empty when no UNKNOWNs are omitted")
                object.__setattr__(record, "unknown_count", unknown_count)
                object.__setattr__(record, "unknown_digest", unknown_digest)
            elif unknown_digest:
                raise SchemaError("attention.unknown_count is required when unknown_digest is present")
            for field_name in ("duplicate_candidate_count", "omitted_candidate_count"):
                raw_value = raw_attention.get(field_name)
                if raw_value is not None:
                    count = int(raw_value)
                    if count < 0:
                        raise SchemaError(f"attention.{field_name} must be non-negative")
                    current = getattr(record, field_name)
                    if count < current:
                        raise SchemaError(f"attention.{field_name} is smaller than visible candidates imply")
                    object.__setattr__(record, field_name, count)
        if isinstance(raw_hypotheses, Mapping):
            raw_unknown_count = raw_hypotheses.get("unknown_count")
            raw_unknown_digest = raw_hypotheses.get("unknown_digest")
            unknown_digest = str(raw_unknown_digest) if raw_unknown_digest else None
            visible_unknown_count = sum(
                1 for item in record.hypotheses if item.status is HypothesisStatus.UNKNOWN
            )
            if raw_unknown_count is not None:
                unknown_count = int(raw_unknown_count)
                if unknown_count < visible_unknown_count:
                    raise SchemaError("hypotheses.unknown_count is smaller than visible UNKNOWNs")
                if unknown_count > visible_unknown_count and not unknown_digest:
                    raise SchemaError("hypotheses.unknown_digest is required for omitted UNKNOWNs")
                if unknown_count == visible_unknown_count and unknown_digest:
                    raise SchemaError("hypotheses.unknown_digest must be empty when no UNKNOWNs are omitted")
                object.__setattr__(record, "hypothesis_unknown_count", unknown_count)
                object.__setattr__(record, "hypothesis_unknown_digest", unknown_digest)
            elif unknown_digest:
                raise SchemaError("hypotheses.unknown_count is required when unknown_digest is present")
        _check_digest(value, record.card_digest, fields=("card_digest", "digest"))
        return record


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    """Typed view of an existing ``poc.stateframe.shadow/2`` snapshot record."""

    SCHEMA: ClassVar[str] = FRAME_SNAPSHOT_SCHEMA
    REQUIRED_STATE_KEYS: ClassVar[frozenset[str]] = frozenset({
        "source",
        "objective",
        "phase",
        "action",
        "facts",
        "cursor",
        "world",
        "verified_facts",
        "hypotheses",
        "negative_work",
        "obligations",
        "tri_logic",
        "kernel",
        "resume",
    })

    source_sha256: str
    turn: int
    state: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_sha256", _text(self.source_sha256, "source_sha256"))
        if not isinstance(self.turn, int) or self.turn < 0:
            raise SchemaError("frame turn must be a non-negative integer")
        if not isinstance(self.state, Mapping):
            raise SchemaError("FrameSnapshot state must be a mapping")
        state = _canonical(self.state)
        if not isinstance(state, dict):
            raise SchemaError("FrameSnapshot state must serialize as a mapping")
        missing = self.REQUIRED_STATE_KEYS.difference(state)
        if missing:
            raise SchemaError(f"FrameSnapshot state is not StateFrame/2 compatible; missing {sorted(missing)}")
        if not isinstance(state.get("negative_work"), list):
            raise SchemaError("StateFrame/2 negative_work must be a list")
        object.__setattr__(self, "state", state)

    @property
    def key(self) -> str:
        return f"{self.source_sha256}:{self.turn}"

    @property
    def state_digest(self) -> str:
        return _digest(self.state)

    @property
    def digest(self) -> str:
        return self.state_digest

    def to_dict(self) -> dict[str, object]:
        # Deliberately matches probes.stateframe_shadow.seal_frame exactly:
        # no rival schema field is inserted into the per-turn record.
        return {
            "kind": "snapshot",
            "key": self.key,
            "turn": self.turn,
            "state_digest": self.state_digest,
            "state": self.state,
        }

    def serialize(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FrameSnapshot":
        key = str(value.get("key") or "")
        source_sha = str(value.get("source_sha256") or "")
        if not source_sha and ":" in key:
            source_sha, _ = key.rsplit(":", 1)
        snapshot = cls(
            source_sha256=source_sha,
            turn=int(value.get("turn", 0)),
            state=value.get("state") if isinstance(value.get("state"), Mapping) else {},
        )
        supplied_digest = str(value.get("state_digest") or "")
        if supplied_digest and supplied_digest != snapshot.state_digest:
            raise SchemaError("FrameSnapshot state_digest does not match state")
        if key and key != snapshot.key:
            raise SchemaError("FrameSnapshot key does not match source_sha256 and turn")
        return snapshot


__all__ = [
    "ActionKind",
    "AdvisoryKind",
    "CandidateAdvisory",
    "CertaintyDemand",
    "CertaintyPreAnswer",
    "CertaintyStatus",
    "CERTAINTY_DEMAND_SCHEMA",
    "CompletionState",
    "ContractSummary",
    "CurrentState",
    "DecisionDeclaration",
    "DECISION_DECLARATION_SCHEMA",
    "DemandPoint",
    "DependencySummary",
    "EditPermission",
    "FailureClassification",
    "Freshness",
    "Frontier",
    "FRAME_SNAPSHOT_SCHEMA",
    "FrameSnapshot",
    "HypothesisAssessment",
    "HypothesisStatus",
    "HYPOTHESIS_ASSESSMENT_SCHEMA",
    "JUDGMENT_GAP_SCHEMA",
    "MAX_VISIBLE_ADVISORIES",
    "MAX_VISIBLE_CONTRACT_ATOMS",
    "MAX_VISIBLE_UNKNOWN",
    "ModelProposal",
    "NegativeWork",
    "OBSERVATION_BUNDLE_SCHEMA",
    "ObservationBundle",
    "Objective",
    "PersistenceState",
    "Provenance",
    "Receipt",
    "ResumeState",
    "RunScope",
    "SITUATION_CARD_SCHEMA",
    "SchemaError",
    "Scope",
    "ScopeViolation",
    "SemanticEvidence",
    "SituationCard",
    "TruthState",
    "UnknownReason",
    "WorldFact",
    "canonical_json",
]
