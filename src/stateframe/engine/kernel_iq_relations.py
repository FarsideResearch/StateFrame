"""Finite, language-neutral relation contracts for procedural work.

This module records typed facts and evaluates a deliberately small relation
set.  It is an evidence adapter, not an executor, frame builder, or second
dependency graph.  Values are useful only when their writer and provenance
reach a host-registered observation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


SCHEMA = "iq-procedural-relations/1"


class RelationContractError(ValueError):
    """Raised when a host cannot validate a relation's evidence boundary."""


class RelationState(str, Enum):
    OBSERVED_PASS = "OBSERVED_PASS"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"
    UNEXAMINED = "UNEXAMINED"
    UNSUPPORTED = "UNSUPPORTED"


class RelationOperator(str, Enum):
    EQUALITY = "equality"
    PRESERVATION = "preservation"
    TRANSITION = "transition"
    PRECEDENCE = "precedence"
    INTERVAL_CONTAINS = "interval_contains"
    BUDGET_AT_MOST = "budget_at_most"


class Unit:
    """A finite unit definition with a conversion to its dimension base."""

    _DEFINITIONS = {
        "": ("dimensionless", 1.0),
        "count": ("count", 1.0),
        "rows": ("count", 1.0),
        "bytes": ("bytes", 1.0),
        "seconds": ("time", 1.0),
        "milliseconds": ("time", 0.001),
        "minutes": ("time", 60.0),
        "hours": ("time", 3600.0),
        "meters": ("length", 1.0),
    }

    def __init__(
        self, symbol: str, *, dimension: str | None = None, scale: float | None = None
    ) -> None:
        symbol = _text(symbol, "unit symbol")
        if dimension is None or scale is None:
            try:
                dimension, scale = self._DEFINITIONS[symbol]
            except KeyError as exc:
                raise RelationContractError(f"unsupported unit: {symbol}") from exc
        if not isinstance(dimension, str) or not dimension.strip():
            raise RelationContractError("unit dimension must be non-empty")
        if (
            not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0
        ):
            raise RelationContractError("unit scale must be finite and positive")
        self.symbol = symbol
        self.dimension = dimension.strip()
        self.scale = float(scale)

    @classmethod
    def parse(cls, value: str | "Unit") -> "Unit":
        return value if isinstance(value, Unit) else cls(value)

    def compatible(self, other: "Unit") -> bool:
        return self.dimension == other.dimension

    def convert(self, value: object, other: "Unit") -> float:
        if not self.compatible(other):
            raise RelationContractError(
                f"incompatible units: {self.symbol} and {other.symbol}"
            )
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RelationContractError("unit-bearing values must be numeric")
        return float(value) * self.scale / other.scale

    def as_dict(self) -> dict[str, object]:
        return {"symbol": self.symbol, "dimension": self.dimension, "scale": self.scale}

    def __repr__(self) -> str:
        return f"Unit({self.symbol!r})"


def units_compatible(left: str | Unit | None, right: str | Unit | None) -> bool:
    if left is None or right is None:
        return left is right
    return Unit.parse(left).compatible(Unit.parse(right))


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    observation_id: str
    source_id: str
    revision: str
    payload: object = None

    def __post_init__(self) -> None:
        for field_name in ("observation_id", "source_id", "revision"):
            _text(getattr(self, field_name), field_name)
        _json_value(self.payload, "observation payload")


@dataclass(frozen=True, slots=True)
class Provenance:
    provenance_id: str
    source_id: str
    revision: str
    observation_id: str
    parent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("provenance_id", "source_id", "revision", "observation_id"):
            _text(getattr(self, field_name), field_name)
        object.__setattr__(
            self, "parent_ids", _text_tuple(self.parent_ids, "parent provenance")
        )


@dataclass(frozen=True, slots=True)
class TypedEntity:
    entity_id: str
    kind: str
    writer_ids: tuple[str, ...]
    language: str = ""

    def __post_init__(self) -> None:
        _text(self.entity_id, "entity_id")
        _text(self.kind, "entity kind")
        object.__setattr__(self, "writer_ids", _text_tuple(self.writer_ids, "writer"))
        if self.language:
            _text(self.language, "entity language")


@dataclass(frozen=True, slots=True)
class TypedValue:
    value_id: str
    entity_id: str
    type_name: str
    value: object
    provenance_id: str
    unit: Unit | None = None

    def __post_init__(self) -> None:
        for field_name in ("value_id", "entity_id", "type_name", "provenance_id"):
            _text(getattr(self, field_name), field_name)
        _json_value(self.value, "typed value")
        if self.unit is not None and not isinstance(self.unit, Unit):
            object.__setattr__(self, "unit", Unit.parse(str(self.unit)))


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    state_id: str
    entity_id: str
    label: str
    writer_ids: tuple[str, ...]
    provenance_id: str

    def __post_init__(self) -> None:
        for field_name in ("state_id", "entity_id", "label", "provenance_id"):
            _text(getattr(self, field_name), field_name)
        object.__setattr__(
            self, "writer_ids", _text_tuple(self.writer_ids, "state writer")
        )


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_id: str
    name: str
    logical_time: int
    entity_id: str
    provenance_id: str

    def __post_init__(self) -> None:
        for field_name in ("event_id", "name", "entity_id", "provenance_id"):
            _text(getattr(self, field_name), field_name)
        if not isinstance(self.logical_time, int) or self.logical_time < 0:
            raise RelationContractError(
                "event logical_time must be a non-negative integer"
            )


@dataclass(frozen=True, slots=True)
class PartialOrderEdge:
    before_id: str
    after_id: str
    edge_id: str = ""

    def __post_init__(self) -> None:
        _text(self.before_id, "before event")
        _text(self.after_id, "after event")
        if self.before_id == self.after_id:
            raise RelationContractError("an event cannot precede itself")
        object.__setattr__(
            self, "edge_id", self.edge_id.strip() or f"{self.before_id}<{self.after_id}"
        )


@dataclass(frozen=True, slots=True)
class RelationClaim:
    relation_id: str
    operator: RelationOperator
    expected: TypedValue | StateSnapshot | None = None
    observed: TypedValue | StateSnapshot | None = None
    before_event_id: str = ""
    after_event_id: str = ""
    provenance_ids: tuple[str, ...] = ()
    initial_state: RelationState | None = None

    def __post_init__(self) -> None:
        _text(self.relation_id, "relation_id")
        object.__setattr__(self, "operator", RelationOperator(self.operator))
        object.__setattr__(
            self, "provenance_ids", _text_tuple(self.provenance_ids, "claim provenance")
        )
        if self.before_event_id:
            _text(self.before_event_id, "before event")
        if self.after_event_id:
            _text(self.after_event_id, "after event")
        if self.initial_state is not None:
            object.__setattr__(self, "initial_state", RelationState(self.initial_state))


@dataclass(frozen=True, slots=True)
class GenericRelationResult:
    relation_id: str
    operator: RelationOperator
    state: RelationState
    detail: str
    value: object = None
    provenance_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "relation_id": self.relation_id,
            "operator": self.operator.value,
            "state": self.state.value,
            "detail": self.detail,
            "value": self.value,
            "provenance_ids": list(self.provenance_ids),
        }


class RelationHost:
    """Host-owned registry and evaluator for the finite relation set."""

    def __init__(self) -> None:
        self.entities: dict[str, TypedEntity] = {}
        self.values: dict[str, TypedValue] = {}
        self.states: dict[str, StateSnapshot] = {}
        self.events: dict[str, EventRecord] = {}
        self.observations: dict[str, ObservationRecord] = {}
        self.provenance: dict[str, Provenance] = {}
        self.edges: dict[tuple[str, str], PartialOrderEdge] = {}

    def add_entity(self, entity: TypedEntity) -> None:
        _unique(self.entities, entity.entity_id, "entity")
        self.entities[entity.entity_id] = entity

    def add_observation(self, observation: ObservationRecord) -> None:
        _unique(self.observations, observation.observation_id, "observation")
        self.observations[observation.observation_id] = observation

    def add_provenance(self, provenance: Provenance) -> None:
        _unique(self.provenance, provenance.provenance_id, "provenance")
        self.provenance[provenance.provenance_id] = provenance

    def add_value(self, value: TypedValue) -> None:
        _unique(self.values, value.value_id, "value")
        self.values[value.value_id] = value

    def add_state(self, state: StateSnapshot) -> None:
        _unique(self.states, state.state_id, "state")
        self.states[state.state_id] = state

    def add_event(self, event: EventRecord) -> None:
        _unique(self.events, event.event_id, "event")
        self.events[event.event_id] = event

    def add_edge(self, edge: PartialOrderEdge) -> None:
        if edge.before_id not in self.events or edge.after_id not in self.events:
            raise RelationContractError("temporal edge references an unknown event")
        before = self.events[edge.before_id]
        after = self.events[edge.after_id]
        if before.logical_time > after.logical_time:
            raise RelationContractError("temporal edge contradicts event logical time")
        key = (edge.before_id, edge.after_id)
        if key in self.edges:
            raise RelationContractError("duplicate temporal edge")
        self.edges[key] = edge
        if self._has_cycle():
            del self.edges[key]
            raise RelationContractError("temporal edges must remain acyclic")

    def validate(self) -> None:
        for entity in self.entities.values():
            if not entity.writer_ids:
                raise RelationContractError(
                    f"entity {entity.entity_id} has no known writer"
                )
            for writer_id in entity.writer_ids:
                if writer_id not in self.entities and writer_id != entity.entity_id:
                    raise RelationContractError(
                        f"entity {entity.entity_id} has unknown writer {writer_id}"
                    )
        for value in self.values.values():
            self._validate_entity(
                value.entity_id, value.provenance_id, f"value {value.value_id}"
            )
        for state in self.states.values():
            self._validate_entity(
                state.entity_id, state.provenance_id, f"state {state.state_id}"
            )
            if not state.writer_ids:
                raise RelationContractError(
                    f"state {state.state_id} has no known writer"
                )
            for writer_id in state.writer_ids:
                if writer_id not in self.entities:
                    raise RelationContractError(
                        f"state {state.state_id} has unknown writer {writer_id}"
                    )
        for event in self.events.values():
            self._validate_entity(
                event.entity_id, event.provenance_id, f"event {event.event_id}"
            )
        for provenance_id in self.provenance:
            self._validate_provenance(provenance_id, set())
        if self._has_cycle():
            raise RelationContractError("temporal edges must remain acyclic")

    def evaluate(self, claim: RelationClaim) -> GenericRelationResult:
        self.validate()
        self._validate_claim(claim)
        evidence_ids = self._claim_provenance(claim)
        if claim.initial_state in {RelationState.UNKNOWN, RelationState.UNEXAMINED}:
            return GenericRelationResult(
                claim.relation_id,
                claim.operator,
                claim.initial_state,
                "host received an explicit unavailable state",
                provenance_ids=evidence_ids,
            )
        if claim.operator in {RelationOperator.EQUALITY, RelationOperator.PRESERVATION}:
            assert isinstance(claim.expected, (TypedValue, StateSnapshot))
            assert isinstance(claim.observed, (TypedValue, StateSnapshot))
            same = _same_observable(claim.expected, claim.observed)
            return self._result(
                claim,
                same,
                "expected and observed values agree",
                "expected and observed values differ",
                evidence_ids,
            )
        if claim.operator is RelationOperator.TRANSITION:
            assert isinstance(claim.expected, StateSnapshot)
            assert isinstance(claim.observed, StateSnapshot)
            same = (
                claim.expected.entity_id == claim.observed.entity_id
                and claim.expected.label == claim.observed.label
            )
            return self._result(
                claim,
                same,
                "expected state transition observed",
                "observed state transition differs",
                evidence_ids,
            )
        if claim.operator is RelationOperator.PRECEDENCE:
            if self._reachable(claim.before_event_id, claim.after_event_id):
                return GenericRelationResult(
                    claim.relation_id,
                    claim.operator,
                    RelationState.OBSERVED_PASS,
                    "partial-order path reaches the requested event",
                    provenance_ids=evidence_ids,
                )
            if self._reachable(claim.after_event_id, claim.before_event_id):
                return GenericRelationResult(
                    claim.relation_id,
                    claim.operator,
                    RelationState.CONTRADICTED,
                    "partial-order path proves the reverse order",
                    provenance_ids=evidence_ids,
                )
            return GenericRelationResult(
                claim.relation_id,
                claim.operator,
                RelationState.UNKNOWN,
                "events are observed but incomparable",
                provenance_ids=evidence_ids,
            )
        assert isinstance(claim.expected, TypedValue)
        assert isinstance(claim.observed, TypedValue)
        expected, observed = _numeric_pair(claim.expected, claim.observed)
        if claim.operator is RelationOperator.INTERVAL_CONTAINS:
            return self._result(
                claim,
                0 <= observed <= expected,
                "observed value is inside the expected interval"
                if 0 <= observed <= expected
                else "observed value is outside the expected interval",
                evidence_ids,
            )
        if claim.operator is RelationOperator.BUDGET_AT_MOST:
            return self._result(
                claim,
                observed <= expected,
                "observed value is within the budget"
                if observed <= expected
                else "observed value exceeds the budget",
                evidence_ids,
            )
        return GenericRelationResult(
            claim.relation_id,
            claim.operator,
            RelationState.UNSUPPORTED,
            "relation operator is not implemented",
            provenance_ids=evidence_ids,
        )

    def _validate_claim(self, claim: RelationClaim) -> None:
        if claim.operator is RelationOperator.PRECEDENCE:
            if not claim.before_event_id or not claim.after_event_id:
                raise RelationContractError(
                    "precedence requires before_event_id and after_event_id"
                )
            if (
                claim.before_event_id not in self.events
                or claim.after_event_id not in self.events
            ):
                raise RelationContractError("precedence references an unknown event")
            return
        if claim.expected is None or claim.observed is None:
            if claim.initial_state in {RelationState.UNKNOWN, RelationState.UNEXAMINED}:
                return
            raise RelationContractError(
                f"{claim.operator.value} requires expected and observed operands"
            )
        for operand in (claim.expected, claim.observed):
            if isinstance(operand, TypedValue):
                if self.values.get(operand.value_id) != operand:
                    raise RelationContractError(
                        f"value {operand.value_id} is not host-registered"
                    )
                self._validate_entity(
                    operand.entity_id,
                    operand.provenance_id,
                    f"value {operand.value_id}",
                )
            elif isinstance(operand, StateSnapshot):
                if self.states.get(operand.state_id) != operand:
                    raise RelationContractError(
                        f"state {operand.state_id} is not host-registered"
                    )
                self._validate_entity(
                    operand.entity_id,
                    operand.provenance_id,
                    f"state {operand.state_id}",
                )
            else:
                raise RelationContractError(
                    "relation operands must be typed values or states"
                )
        if isinstance(claim.expected, TypedValue) and isinstance(
            claim.observed, TypedValue
        ):
            if claim.expected.type_name != claim.observed.type_name:
                raise RelationContractError("relation operands have incompatible types")
            if not units_compatible(claim.expected.unit, claim.observed.unit):
                raise RelationContractError("relation operands have incompatible units")

    def _validate_entity(self, entity_id: str, provenance_id: str, label: str) -> None:
        entity = self.entities.get(entity_id)
        if entity is None:
            raise RelationContractError(
                f"{label} references unknown entity {entity_id}"
            )
        if not entity.writer_ids:
            raise RelationContractError(
                f"{label} references entity {entity_id} with no known writer"
            )
        for writer_id in entity.writer_ids:
            if writer_id not in self.entities and writer_id not in {entity_id}:
                raise RelationContractError(
                    f"{label} references unknown writer {writer_id}"
                )
        self._validate_provenance(provenance_id, set())

    def _validate_provenance(self, provenance_id: str, visiting: set[str]) -> None:
        record = self.provenance.get(provenance_id)
        if record is None:
            raise RelationContractError(f"provenance {provenance_id} is not registered")
        if record.observation_id not in self.observations:
            raise RelationContractError(
                f"provenance {provenance_id} cannot reach an observation"
            )
        observation = self.observations[record.observation_id]
        if (
            record.source_id != observation.source_id
            or record.revision != observation.revision
        ):
            raise RelationContractError(
                f"provenance {provenance_id} does not match its observation revision"
            )
        if provenance_id in visiting:
            raise RelationContractError("provenance chain is cyclic")
        visiting.add(provenance_id)
        for parent_id in record.parent_ids:
            self._validate_provenance(parent_id, visiting)
        visiting.remove(provenance_id)

    def _claim_provenance(self, claim: RelationClaim) -> tuple[str, ...]:
        ids = list(claim.provenance_ids)
        for operand in (claim.expected, claim.observed):
            if hasattr(operand, "provenance_id"):
                ids.append(str(getattr(operand, "provenance_id")))
        result = tuple(dict.fromkeys(ids))
        for provenance_id in result:
            self._validate_provenance(provenance_id, set())
        return result

    def _result(
        self,
        claim: RelationClaim,
        same: bool,
        passed: str,
        failed: str,
        provenance_ids: tuple[str, ...],
    ) -> GenericRelationResult:
        return GenericRelationResult(
            claim.relation_id,
            claim.operator,
            RelationState.OBSERVED_PASS if same else RelationState.CONTRADICTED,
            passed if same else failed,
            provenance_ids=provenance_ids,
        )

    def _reachable(self, source: str, target: str) -> bool:
        pending = [source]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(after for before, after in self.edges if before == current)
        return False

    def _has_cycle(self) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(visit(after) for before, after in self.edges if before == node):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(event_id) for event_id in self.events)


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RelationContractError(f"{field_name} must be non-empty")
    return value.strip()


def _text_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise RelationContractError(f"{field_name} must be a sequence")
    return tuple(dict.fromkeys(_text(value, field_name) for value in values))


def _json_value(value: object, field_name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RelationContractError(f"{field_name} must be JSON-compatible") from exc


def _unique(store: Mapping[str, object], key: str, kind: str) -> None:
    if key in store:
        raise RelationContractError(f"duplicate {kind} id: {key}")


def _same_observable(
    left: TypedValue | StateSnapshot, right: TypedValue | StateSnapshot
) -> bool:
    if isinstance(left, StateSnapshot) and isinstance(right, StateSnapshot):
        return left.entity_id == right.entity_id and left.label == right.label
    if isinstance(left, TypedValue) and isinstance(right, TypedValue):
        if left.unit is not None and right.unit is not None:
            return math.isclose(
                left.unit.convert(left.value, right.unit),
                float(right.value),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        return left.value == right.value
    return False


def _numeric_pair(expected: TypedValue, observed: TypedValue) -> tuple[float, float]:
    if expected.unit is None or observed.unit is None:
        if not isinstance(expected.value, (int, float)) or isinstance(
            expected.value, bool
        ):
            raise RelationContractError(
                "numeric relation requires numeric expected value"
            )
        if not isinstance(observed.value, (int, float)) or isinstance(
            observed.value, bool
        ):
            raise RelationContractError(
                "numeric relation requires numeric observed value"
            )
        return float(expected.value), float(observed.value)
    return expected.unit.convert(
        expected.value, Unit.parse(expected.unit)
    ), expected.unit.convert(observed.value, expected.unit)


__all__ = [
    "EventRecord",
    "GenericRelationResult",
    "ObservationRecord",
    "PartialOrderEdge",
    "Provenance",
    "RelationClaim",
    "RelationContractError",
    "RelationHost",
    "RelationOperator",
    "RelationState",
    "SCHEMA",
    "StateSnapshot",
    "TypedEntity",
    "TypedValue",
    "Unit",
    "units_compatible",
]
