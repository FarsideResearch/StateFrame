"""Language-neutral, clause-bound observation records and pure relations.

This module is deliberately independent of the existing kernel organs.  It
describes what a later host adapter may observe; it does not execute product
code, grant completion, dispatch task IDs, or infer an owner for missing
source.  A receipt is useful only when its scope, identity, and revision are
explicit and current.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

try:  # Python 3.9 does not expose TypeAlias in typing.
    from typing import TypeAlias
except ImportError:  # pragma: no cover - exercised by the repository's 3.9 runner
    TypeAlias = object  # type: ignore[misc,assignment]


SCHEMA = "kernel-observation-contract/1"

JsonValue: TypeAlias = object


class ObservationStatus(str, Enum):
    """Host outcome; no status is silently coerced into another one."""

    OBSERVED_PASS = "OBSERVED_PASS"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    ENVIRONMENT = "ENVIRONMENT"
    UNRUNNABLE = "UNRUNNABLE"
    UNCOVERED = "UNCOVERED"
    UNKNOWN = "UNKNOWN"


class ScopeState(str, Enum):
    """Completeness of a discovered scope, never a coverage count."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    TRUNCATED = "TRUNCATED"
    UNREADABLE = "UNREADABLE"
    UNRESOLVED = "UNRESOLVED"


class EvidenceScope(str, Enum):
    CASE = "CASE"
    CLAUSE = "CLAUSE"


class AdmissionState(str, Enum):
    ADMITTED = "ADMITTED"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"


class ControlKind(str, Enum):
    OMITTED_SITE = "OMITTED_SITE"
    BYPASS = "BYPASS"
    DELAYED_FIRST_ACTION = "DELAYED_FIRST_ACTION"
    ZERO_WAIT = "ZERO_WAIT"
    REMOVED_RENEWAL = "REMOVED_RENEWAL"
    POST_AWAIT_INVALIDATION = "POST_AWAIT_INVALIDATION"
    UNCONDITIONAL_SKIP = "UNCONDITIONAL_SKIP"
    LATE_GUARD = "LATE_GUARD"
    EQUIVALENT_VALID_VARIATION = "EQUIVALENT_VALID_VARIATION"


def _canonical(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON-serializable: {type(value).__name__}")


def _digest(value: object) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value.strip()


def _tuple_text(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(_nonempty(str(value), label) for value in values)
    return tuple(dict.fromkeys(result))


@dataclass(frozen=True)
class SourceSpan:
    path: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    source_digest: str = ""

    def __post_init__(self) -> None:
        _nonempty(self.path, "span path")
        if min(self.start_line, self.start_column, self.end_line, self.end_column) < 1:
            raise ValueError("source span coordinates are one-based")
        if (self.end_line, self.end_column) < (self.start_line, self.start_column):
            raise ValueError("source span end precedes start")

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "start_column": self.start_column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class CallableIdentity:
    language: str
    runtime: str
    qualified_name: str
    module: str = ""
    entrypoint_kind: str = "function"

    def __post_init__(self) -> None:
        _nonempty(self.language, "callable language")
        _nonempty(self.runtime, "callable runtime")
        _nonempty(self.qualified_name, "callable qualified name")
        _nonempty(self.entrypoint_kind, "entrypoint kind")

    def as_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "runtime": self.runtime,
            "qualified_name": self.qualified_name,
            "module": self.module,
            "entrypoint_kind": self.entrypoint_kind,
        }


@dataclass(frozen=True)
class Revision:
    current: str
    source: str
    candidate: str
    check: str
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for label in ("current", "source", "candidate", "check"):
            _nonempty(getattr(self, label), f"revision {label}")

    def as_dict(self) -> dict[str, object]:
        return {
            "current": self.current,
            "source": self.source,
            "candidate": self.candidate,
            "check": self.check,
            "dependencies": list(self.dependencies),
        }


@dataclass(frozen=True)
class ScopePartition:
    name: str
    state: ScopeState
    members: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unreadable: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        _nonempty(self.name, "scope partition name")
        for label in ("members", "missing", "unreadable", "unresolved"):
            object.__setattr__(self, label, _tuple_text(getattr(self, label), label))

    @property
    def usable_for_universal(self) -> bool:
        return self.state is ScopeState.COMPLETE and not (
            self.missing or self.unreadable or self.unresolved
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state": self.state.value,
            "complete": self.usable_for_universal,
            "truncated": self.state is ScopeState.TRUNCATED,
            "unreadable": list(self.unreadable),
            "unresolved": list(self.unresolved),
            "members": list(self.members),
            "missing": list(self.missing),
            "note": self.note,
        }


@dataclass(frozen=True)
class LexicalSite:
    site_id: str
    span: SourceSpan
    literal: str
    role: str = "construction"

    def __post_init__(self) -> None:
        _nonempty(self.site_id, "lexical site id")
        _nonempty(self.literal, "lexical site literal")

    def as_dict(self) -> dict[str, object]:
        return {"site_id": self.site_id, "span": self.span.as_dict(), "literal": self.literal, "role": self.role}


@dataclass(frozen=True)
class ResolvedConstructionSite:
    site_id: str
    lexical_site_id: str
    span: SourceSpan
    callable: CallableIdentity
    policy_id: str
    policy_value: JsonValue
    consumers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for label in ("site_id", "lexical_site_id", "policy_id"):
            _nonempty(getattr(self, label), label)
        object.__setattr__(self, "consumers", _tuple_text(self.consumers, "consumer"))

    def as_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "lexical_site_id": self.lexical_site_id,
            "span": self.span.as_dict(),
            "callable": self.callable.as_dict(),
            "policy_id": self.policy_id,
            "policy_value": _canonical(self.policy_value),
            "consumers": list(self.consumers),
        }


@dataclass(frozen=True)
class DiscoveryObservation:
    lexical_sites: tuple[LexicalSite, ...]
    resolved_construction_sites: tuple[ResolvedConstructionSite, ...]
    scope: tuple[ScopePartition, ...]
    source_revision: str
    phase: str = "DISCOVERY"

    def __post_init__(self) -> None:
        _nonempty(self.source_revision, "discovery source revision")
        if self.phase != "DISCOVERY":
            raise ValueError("discovery record has an invalid phase")

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "lexical_sites": [site.as_dict() for site in self.lexical_sites],
            "resolved_construction_sites": [site.as_dict() for site in self.resolved_construction_sites],
            "scope": [partition.as_dict() for partition in self.scope],
            "source_revision": self.source_revision,
        }


@dataclass(frozen=True)
class AdmissionObservation:
    state: AdmissionState
    target: CallableIdentity | None
    fixture_id: str
    dependency_ids: tuple[str, ...]
    capability_ids: tuple[str, ...]
    missing: tuple[str, ...] = ()
    reason: str = ""
    phase: str = "ADMISSION"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dependency_ids", _tuple_text(self.dependency_ids, "dependency id"))
        object.__setattr__(self, "capability_ids", _tuple_text(self.capability_ids, "capability id"))
        object.__setattr__(self, "missing", _tuple_text(self.missing, "missing capability"))
        if self.state is AdmissionState.ADMITTED:
            if self.target is None or not self.fixture_id.strip() or not self.dependency_ids:
                raise ValueError("admitted observations require target, fixture, and dependencies")
        if self.phase != "ADMISSION":
            raise ValueError("admission record has an invalid phase")

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "state": self.state.value,
            "target": self.target.as_dict() if self.target else None,
            "fixture_id": self.fixture_id,
            "dependency_ids": list(self.dependency_ids),
            "capability_ids": list(self.capability_ids),
            "missing": list(self.missing),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ConstructionObservation:
    site_id: str
    policy_id: str
    effective_value: JsonValue
    observed: bool = True

    def __post_init__(self) -> None:
        _nonempty(self.site_id, "construction observation site id")
        _nonempty(self.policy_id, "construction observation policy id")

    def as_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "policy_id": self.policy_id,
            "effective_value": _canonical(self.effective_value),
            "observed": self.observed,
        }


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    kind: str
    name: str
    arguments: Mapping[str, JsonValue] = field(default_factory=dict)
    logical_time: int = 0
    interval_seconds: float | None = None

    def __post_init__(self) -> None:
        for label in ("event_id", "kind", "name"):
            _nonempty(getattr(self, label), label)
        if self.logical_time < 0:
            raise ValueError("logical event time cannot be negative")
        if self.interval_seconds is not None and self.interval_seconds < 0:
            raise ValueError("timer interval cannot be negative")
        if "interval_seconds" in self.arguments:
            raise ValueError("timer interval must remain separate from action arguments")

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "name": self.name,
            "arguments": _canonical(self.arguments),
            "logical_time": self.logical_time,
            "interval_seconds": self.interval_seconds,
        }


@dataclass(frozen=True)
class EffectObservation:
    effect_id: str
    kind: str
    scope: str
    count: int
    arguments: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label in ("effect_id", "kind", "scope"):
            _nonempty(getattr(self, label), label)
        if self.count < 0:
            raise ValueError("effect count cannot be negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "effect_id": self.effect_id,
            "kind": self.kind,
            "scope": self.scope,
            "count": self.count,
            "arguments": _canonical(self.arguments),
        }


@dataclass(frozen=True)
class SerializedObservation:
    observation_id: str
    payload: JsonValue
    observer_id: str
    persistence_boundary: str = ""
    independent: bool = False
    digest: str = ""

    def __post_init__(self) -> None:
        for label in ("observation_id", "observer_id"):
            _nonempty(getattr(self, label), label)
        object.__setattr__(self, "digest", self.digest or _digest(self.payload))

    def as_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "payload": _canonical(self.payload),
            "observer_id": self.observer_id,
            "persistence_boundary": self.persistence_boundary,
            "independent": self.independent,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class RevisionBoundReceipt:
    ticket_id: str
    source_id: str
    candidate_id: str
    check_id: str
    adapter_id: str
    receipt_id: str
    revision: Revision

    def __post_init__(self) -> None:
        for label in ("ticket_id", "source_id", "candidate_id", "check_id", "adapter_id", "receipt_id"):
            _nonempty(getattr(self, label), label)

    def as_dict(self) -> dict[str, object]:
        return {
            "ticket_id": self.ticket_id,
            "source_id": self.source_id,
            "candidate_id": self.candidate_id,
            "check_id": self.check_id,
            "adapter_id": self.adapter_id,
            "receipt_id": self.receipt_id,
            "revision": self.revision.as_dict(),
        }


@dataclass(frozen=True)
class ExecutionObservation:
    adapter_id: str
    status: ObservationStatus
    events: tuple[TraceEvent, ...] = ()
    effects: tuple[EffectObservation, ...] = ()
    constructions: tuple[ConstructionObservation, ...] = ()
    serialized: tuple[SerializedObservation, ...] = ()
    phase: str = "EXECUTION"

    def __post_init__(self) -> None:
        _nonempty(self.adapter_id, "execution adapter id")
        if self.phase != "EXECUTION":
            raise ValueError("execution record has an invalid phase")

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "adapter_id": self.adapter_id,
            "status": self.status.value,
            "events": [event.as_dict() for event in self.events],
            "effects": [effect.as_dict() for effect in self.effects],
            "constructions": [construction.as_dict() for construction in self.constructions],
            "serialized": [item.as_dict() for item in self.serialized],
        }


@dataclass(frozen=True)
class EvidenceObservation:
    status: ObservationStatus
    scope_kind: EvidenceScope
    partition: str
    case_id: str
    receipt: RevisionBoundReceipt
    observed: JsonValue
    covered_partitions: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()
    phase: str = "EVIDENCE"

    def __post_init__(self) -> None:
        for label in ("partition", "case_id"):
            _nonempty(getattr(self, label), label)
        object.__setattr__(self, "covered_partitions", _tuple_text(self.covered_partitions, "covered partition"))
        object.__setattr__(self, "uncertainty", _tuple_text(self.uncertainty, "uncertainty"))
        if self.phase != "EVIDENCE":
            raise ValueError("evidence record has an invalid phase")

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "status": self.status.value,
            "scope_kind": self.scope_kind.value,
            "partition": self.partition,
            "case_id": self.case_id,
            "receipt": self.receipt.as_dict(),
            "observed": _canonical(self.observed),
            "covered_partitions": list(self.covered_partitions),
            "uncertainty": list(self.uncertainty),
        }


@dataclass(frozen=True)
class ObservationContract:
    ticket_span: SourceSpan
    callable: CallableIdentity
    revision: Revision
    discovery: DiscoveryObservation
    admission: AdmissionObservation
    execution: ExecutionObservation
    evidence: EvidenceObservation

    def validate_binding(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.discovery.source_revision != self.revision.source:
            errors.append("discovery source revision is stale")
        if self.execution.adapter_id != self.evidence.receipt.adapter_id:
            errors.append("execution and receipt adapter identities differ")
        if self.evidence.receipt.revision != self.revision:
            errors.append("receipt revision is not current")
        if self.admission.target is not None and self.admission.target != self.callable:
            errors.append("admission target differs from callable identity")
        return tuple(errors)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "ticket_span": self.ticket_span.as_dict(),
            "callable": self.callable.as_dict(),
            "revision": self.revision.as_dict(),
            "discovery": self.discovery.as_dict(),
            "admission": self.admission.as_dict(),
            "execution": self.execution.as_dict(),
            "evidence": self.evidence.as_dict(),
            "binding_errors": list(self.validate_binding()),
        }


@dataclass(frozen=True)
class RelationResult:
    relation: str
    status: ObservationStatus
    detail: str
    value: JsonValue = None
    certifies_clause: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "relation": self.relation,
            "status": self.status.value,
            "detail": self.detail,
            "value": _canonical(self.value),
            "certifies_clause": self.certifies_clause,
        }


def _scope_blocker(scope: Sequence[ScopePartition]) -> str | None:
    for partition in scope:
        if not partition.usable_for_universal:
            return f"scope partition {partition.name!r} is {partition.state.value}"
    return None


def effective_construction_policy(
    discovery: DiscoveryObservation,
    observations: Sequence[ConstructionObservation],
    *,
    expected_policy_id: str,
    expected_value: JsonValue,
) -> RelationResult:
    """Relate resolved construction sites to observed effective policy."""

    blocker = _scope_blocker(discovery.scope)
    if blocker:
        return RelationResult("effective_construction_policy", ObservationStatus.UNCOVERED, blocker)
    sites = discovery.resolved_construction_sites
    if not sites:
        return RelationResult("effective_construction_policy", ObservationStatus.UNCOVERED, "no resolved construction sites")
    observed = {item.site_id: item for item in observations if item.observed}
    missing = [site.site_id for site in sites if site.site_id not in observed]
    if missing:
        return RelationResult(
            "effective_construction_policy", ObservationStatus.UNCOVERED,
            "executed observations are missing resolved sites", {"missing_sites": missing},
        )
    for site in sites:
        item = observed[site.site_id]
        if item.policy_id != expected_policy_id or item.effective_value != expected_value:
            return RelationResult(
                "effective_construction_policy", ObservationStatus.COUNTEREXAMPLE,
                f"resolved site {site.site_id} has an effective policy mismatch",
                {"site_id": site.site_id, "policy_id": item.policy_id, "value": item.effective_value},
            )
    return RelationResult(
        "effective_construction_policy", ObservationStatus.OBSERVED_PASS,
        "every resolved construction site observed the expected policy",
        {"site_count": len(sites), "policy_id": expected_policy_id},
    )


def initialization_partial_order(trace: Sequence[TraceEvent], before: str, after: str) -> RelationResult:
    """Check a cold-start event order without relying on source line proximity."""

    names = [event.name for event in trace]
    if before not in names or after not in names:
        return RelationResult("initialization_partial_order", ObservationStatus.UNCOVERED, "required event is absent")
    before_index = names.index(before)
    after_index = names.index(after)
    if before_index >= after_index:
        return RelationResult("initialization_partial_order", ObservationStatus.COUNTEREXAMPLE,
                              f"{before!r} did not precede {after!r}")
    return RelationResult("initialization_partial_order", ObservationStatus.OBSERVED_PASS,
                          f"{before!r} precedes {after!r}", {"before": before_index, "after": after_index})


def logical_timer_event_trace(
    trace: Sequence[TraceEvent],
    *,
    first_action: str,
    wait_event: str,
    interval_seconds: float,
    subsequent_action: str | None = None,
) -> RelationResult:
    """Check logical timer ordering; interval is never an action argument."""

    action_indices = [index for index, event in enumerate(trace) if event.name == first_action]
    waits = [(index, event) for index, event in enumerate(trace) if event.name == wait_event]
    if not action_indices or not waits:
        return RelationResult("logical_timer_event_trace", ObservationStatus.UNCOVERED,
                              "first action or wait event is absent")
    first_action_index = action_indices[0]
    first_wait_index, first_wait = waits[0]
    if first_action_index > first_wait_index:
        return RelationResult("logical_timer_event_trace", ObservationStatus.COUNTEREXAMPLE,
                              "first action was delayed until after the first wait")
    if first_wait.interval_seconds != interval_seconds:
        return RelationResult("logical_timer_event_trace", ObservationStatus.COUNTEREXAMPLE,
                              "timer interval differs from the required interval",
                              {"observed": first_wait.interval_seconds, "expected": interval_seconds})
    if subsequent_action is not None and not any(
        event.name == subsequent_action for event in trace[first_wait_index + 1:]
    ):
        return RelationResult("logical_timer_event_trace", ObservationStatus.UNCOVERED,
                              "continued action after the first wait was not observed")
    return RelationResult("logical_timer_event_trace", ObservationStatus.OBSERVED_PASS,
                          "logical action/wait ordering and interval observed",
                          {"first_action_index": first_action_index, "first_wait_index": first_wait_index})


def independent_persisted_readback(
    expected: SerializedObservation,
    readback: SerializedObservation,
) -> RelationResult:
    """Require an independent observer at a persistence boundary."""

    if not readback.independent or not readback.persistence_boundary:
        return RelationResult("independent_persisted_readback", ObservationStatus.UNCOVERED,
                              "read-back is not independently persisted")
    if readback.observer_id == expected.observer_id:
        return RelationResult("independent_persisted_readback", ObservationStatus.UNCOVERED,
                              "read-back observer is not independent")
    if readback.digest != expected.digest:
        return RelationResult("independent_persisted_readback", ObservationStatus.COUNTEREXAMPLE,
                              "independent persisted state differs", {"expected": expected.digest, "readback": readback.digest})
    return RelationResult("independent_persisted_readback", ObservationStatus.OBSERVED_PASS,
                          "independent persisted read-back matches", {"digest": readback.digest})


def scoped_effect_counts(
    effects: Sequence[EffectObservation],
    expected: Sequence[tuple[str, str, int]],
    *,
    telemetry_complete: bool = True,
) -> RelationResult:
    """Compare effect counts only within the requested effect scope."""

    if not telemetry_complete:
        return RelationResult("scoped_effect_counts", ObservationStatus.UNCOVERED,
                              "effect telemetry is incomplete")
    index = {(effect.kind, effect.scope): effect.count for effect in effects}
    missing = [(kind, scope) for kind, scope, _count in expected if (kind, scope) not in index]
    if missing:
        return RelationResult("scoped_effect_counts", ObservationStatus.UNCOVERED,
                              "requested effect scopes were not observed", {"missing": missing})
    mismatches = [
        {"kind": kind, "scope": scope, "observed": index[(kind, scope)], "expected": count}
        for kind, scope, count in expected if index[(kind, scope)] != count
    ]
    if mismatches:
        return RelationResult("scoped_effect_counts", ObservationStatus.COUNTEREXAMPLE,
                              "one or more scoped effect counts differ", mismatches)
    return RelationResult("scoped_effect_counts", ObservationStatus.OBSERVED_PASS,
                          "requested scoped effect counts match", {"checked": len(expected)})


def _state_at(payload: object, scope: str) -> object:
    if not scope:
        return payload
    if not isinstance(payload, Mapping) or scope not in payload:
        raise KeyError(scope)
    return payload[scope]


def state_equality(before: SerializedObservation, after: SerializedObservation, *, scope: str) -> RelationResult:
    """Compare only the named state partition, not unrelated application activity."""

    try:
        before_state = _state_at(before.payload, scope)
        after_state = _state_at(after.payload, scope)
    except KeyError:
        return RelationResult("state_equality", ObservationStatus.UNCOVERED,
                              f"state scope {scope!r} is absent")
    if before_state != after_state:
        return RelationResult("state_equality", ObservationStatus.COUNTEREXAMPLE,
                              f"state scope {scope!r} changed", {"before": before_state, "after": after_state})
    return RelationResult("state_equality", ObservationStatus.OBSERVED_PASS,
                          f"state scope {scope!r} is equal", {"scope": scope})


def failure_preserves_primary_state(
    before: SerializedObservation,
    after: SerializedObservation,
    *,
    primary_scope: str,
    failure_observed: bool,
) -> RelationResult:
    """Require an observed optional failure before comparing primary state."""

    if not failure_observed:
        return RelationResult("failure_preserves_primary_state", ObservationStatus.UNCOVERED,
                              "the optional failure was not observed")
    result = state_equality(before, after, scope=primary_scope)
    return RelationResult("failure_preserves_primary_state", result.status, result.detail,
                          result.value, result.certifies_clause)


@dataclass(frozen=True)
class Clause:
    obligation_id: str
    ticket_span: SourceSpan
    predicate: str
    quantifier: str
    partitions: tuple[ScopePartition, ...]
    revision: Revision

    def __post_init__(self) -> None:
        _nonempty(self.obligation_id, "obligation id")
        _nonempty(self.predicate, "clause predicate")
        _nonempty(self.quantifier, "clause quantifier")
        if not self.partitions:
            raise ValueError("clause must declare at least one scope partition")


def assess_clause(
    clause: Clause,
    evidence: Sequence[EvidenceObservation],
    *,
    current_revision: Revision,
) -> RelationResult:
    """Assess scoped evidence without promoting one scalar case to a clause."""

    relation = "clause_assessment"
    if clause.revision != current_revision:
        return RelationResult(relation, ObservationStatus.UNCOVERED, "clause revision is stale")
    blocker = _scope_blocker(clause.partitions)
    if blocker:
        return RelationResult(relation, ObservationStatus.UNCOVERED, blocker)
    relevant = [item for item in evidence if item.receipt.revision == current_revision]
    if not relevant:
        return RelationResult(relation, ObservationStatus.UNCOVERED, "no current evidence")
    identity_fields = ("ticket_id", "source_id", "candidate_id", "check_id")
    identity = tuple(getattr(relevant[0].receipt, field) for field in identity_fields)
    if any(
        tuple(getattr(item.receipt, field) for field in identity_fields) != identity
        for item in relevant[1:]
    ):
        return RelationResult(
            relation,
            ObservationStatus.UNCOVERED,
            "current evidence receipts do not share ticket/source/candidate/check identity",
        )
    if any(item.status is ObservationStatus.COUNTEREXAMPLE for item in relevant):
        return RelationResult(relation, ObservationStatus.COUNTEREXAMPLE, "a current evidence receipt is a counterexample")
    if any(item.status is ObservationStatus.ENVIRONMENT for item in relevant):
        return RelationResult(relation, ObservationStatus.ENVIRONMENT, "current evidence contains an environment failure")
    if any(item.status is ObservationStatus.UNRUNNABLE for item in relevant):
        return RelationResult(relation, ObservationStatus.UNRUNNABLE, "current evidence is not runnable")
    if any(item.status is ObservationStatus.UNKNOWN for item in relevant):
        return RelationResult(relation, ObservationStatus.UNKNOWN, "current evidence has an unresolved host outcome")
    if any(item.status is ObservationStatus.UNCOVERED for item in relevant):
        return RelationResult(relation, ObservationStatus.UNCOVERED, "current evidence has an uncovered obligation")
    passes = [item for item in relevant if item.status is ObservationStatus.OBSERVED_PASS]
    if clause.quantifier.upper() not in {"CASE", "FOR_ALL", "UNIVERSAL"}:
        return RelationResult(relation, ObservationStatus.UNCOVERED, "clause quantifier is unresolved")
    if clause.quantifier.upper() == "CASE":
        if not passes:
            return RelationResult(relation, ObservationStatus.UNCOVERED, "no passing case evidence")
        return RelationResult(relation, ObservationStatus.OBSERVED_PASS,
                              "case evidence observed; no universal claim made", {"cases": len(passes)}, False)
    clause_receipts = [
        item for item in passes
        if item.scope_kind is EvidenceScope.CLAUSE and not item.uncertainty
    ]
    required = {partition.name for partition in clause.partitions}
    covered = set().union(*(set(item.covered_partitions) for item in clause_receipts)) if clause_receipts else set()
    if not clause_receipts:
        return RelationResult(relation, ObservationStatus.UNCOVERED,
                              "single-case evidence cannot certify a universal clause")
    missing = sorted(required - covered)
    if missing:
        return RelationResult(relation, ObservationStatus.UNCOVERED,
                              "universal clause evidence has uncovered partitions", {"missing": missing})
    return RelationResult(relation, ObservationStatus.OBSERVED_PASS,
                          "current clause-scoped evidence covers every complete partition",
                          {"partitions": sorted(covered)}, True)


@dataclass(frozen=True)
class ControlObservation:
    control_id: str
    kind: ControlKind
    mutated: bool
    operation_reached: bool
    status: ObservationStatus
    equivalent_valid: bool = False
    setup_failure: bool = False

    def __post_init__(self) -> None:
        _nonempty(self.control_id, "control id")


def assess_adversarial_control(control: ControlObservation) -> RelationResult:
    """Count only an executed semantic violation as a mutant kill."""

    if control.setup_failure or control.status in {ObservationStatus.ENVIRONMENT, ObservationStatus.UNRUNNABLE}:
        return RelationResult("adversarial_control", ObservationStatus.ENVIRONMENT,
                              "setup, fixture, resource, or runner failure is not a mutant kill",
                              {"control_id": control.control_id, "killed": False})
    if control.status is ObservationStatus.UNKNOWN:
        return RelationResult("adversarial_control", ObservationStatus.UNKNOWN,
                              "host outcome is unresolved; no control kill is admitted",
                              {"control_id": control.control_id, "killed": False})
    if control.equivalent_valid:
        if control.status is ObservationStatus.OBSERVED_PASS:
            return RelationResult("adversarial_control", ObservationStatus.OBSERVED_PASS,
                                  "equivalent valid variation remains valid",
                                  {"control_id": control.control_id, "killed": False, "valid_control": True})
        return RelationResult("adversarial_control", control.status,
                              "equivalent valid variation did not preserve behavior",
                              {"control_id": control.control_id, "killed": False, "valid_control": True})
    if control.mutated and control.operation_reached and control.status is ObservationStatus.COUNTEREXAMPLE:
        return RelationResult("adversarial_control", ObservationStatus.COUNTEREXAMPLE,
                              "executed semantic violation killed the control",
                              {"control_id": control.control_id, "killed": True})
    return RelationResult("adversarial_control", ObservationStatus.UNCOVERED,
                          "control did not produce an executed semantic kill",
                          {"control_id": control.control_id, "killed": False})


__all__ = [
    "SCHEMA", "AdmissionObservation", "AdmissionState", "CallableIdentity", "Clause", "ConstructionObservation",
    "ControlKind", "ControlObservation", "DiscoveryObservation", "EffectObservation", "EvidenceObservation",
    "EvidenceScope", "ExecutionObservation", "LexicalSite", "ObservationContract", "ObservationStatus",
    "RelationResult", "ResolvedConstructionSite", "Revision", "RevisionBoundReceipt", "ScopePartition",
    "ScopeState", "SerializedObservation", "SourceSpan", "TraceEvent", "assess_adversarial_control",
    "assess_clause", "effective_construction_policy", "failure_preserves_primary_state",
    "independent_persisted_readback", "initialization_partial_order", "logical_timer_event_trace",
    "scoped_effect_counts", "state_equality",
]
