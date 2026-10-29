"""Deterministic task obligations and conservative completion evidence.

This module is the first, deliberately narrow slice of the completeness
governor. It compiles obligations from existing task rows and unit plans,
replays test/artifact receipts, and exposes a bounded coverage receipt for a
done gate. It does not own a runner and it does not mutate any hot probe.

The important safety property is monotonic conservatism: an observed green
selector can satisfy an obligation, an observed red selector can mark it
missing, and the absence of an observation remains UNKNOWN.

The graph also retains state transitions so a later SATISFIED-to-MISSING change
can be surfaced as a bounded regression notice with its preceding edit receipt.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TypeAlias

from kernel_liveness import Evidence, OrganCensusSpec, TriState, run_census as _run_liveness_census
from unit_planner import UnitPlan, derive_unit_plan


logger = logging.getLogger(__name__)

SCHEMA = "poc.kernel_obligations/1"
ORGAN_ID = "obligation_graph"
MAX_COVERAGE_CHARS = 600
MAX_FAILURE_TAIL_CHARS = 360
MAX_REGRESSION_NOTICE_CHARS = 300
MAX_PENDING_REGRESSION_NOTICES = 128  # legacy view page size; the tape is lossless

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | Mapping[str, "JSONValue"] | Sequence["JSONValue"]
Record: TypeAlias = dict[str, object]
ReceiptSink: TypeAlias = Callable[[Mapping[str, object]], object]


# The queue is intentionally process-local: the harness drains it at the same
# delivery boundary that emits organ_out("obligation_regression", ...).
_PENDING_REGRESSION_NOTICES: list[tuple[tuple[str, str], str]] = []
_PENDING_REGRESSION_KEYS: set[tuple[str, str]] = set()
# Append-only notice tape.  Pending delivery may be drained, but history is
# retained for recall and run audits.
_REGRESSION_NOTICE_STORE: list[dict[str, object]] = []
_REGRESSION_NOTICE_STORE_KEYS: set[tuple[str, str]] = set()


class ObligationState(str, Enum):
    """Four-state obligation result; UNKNOWN is never treated as success."""

    SATISFIED = "SATISFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"
    MISSING = "MISSING"


class ObligationKind(str, Enum):
    """Source lane that produced an obligation."""

    ORACLE = "oracle"
    UNIT = "unit"
    CONTRACT = "contract"


# Compatibility aliases make the state vocabulary easy to discover for
# callers that use "status" or "kind" naming.
ObligationStatus = ObligationState
State = ObligationState

_KIND_ORDER = {
    ObligationKind.ORACLE.value: 0,
    ObligationKind.UNIT.value: 1,
    ObligationKind.CONTRACT.value: 2,
}
_GREEN_RESULTS = frozenset(
    {
        "ACCEPTED",
        "CERTIFIED",
        "GREEN",
        "OK",
        "PASS",
        "PASSED",
        "SUCCESS",
        "SUCCEEDED",
    }
)
_MISSING_RESULTS = frozenset(
    {
        "ASSERTION_FAIL",
        "COLLECTION_ERROR",
        "ERROR",
        "FAILED",
        "FAIL",
        "HARNESS_ERROR",
        "INFRA_ERROR",
        "MISSING",
        "NO_TESTS",
        "RED",
        "SELECTOR_ERROR",
        "TIMEOUT",
    }
)
_UNKNOWN_RESULTS = frozenset(
    {"", "CHALLENGED", "DEFERRED", "INCOMPLETE", "UNKNOWN", "UNCERTIFIED"}
)
_SELECTOR_FIELDS = (
    "selectors",
    "selected",
    "test_ids",
    "test_selectors",
    "oracle_ids",
    "f2p_ids",
    "gate_selectors",
    "selector",
    "test_selector",
    "nodeid",
)
_OBLIGATION_PREFIXES = ("oracle:", "unit:", "contract:")
_PATH_SELECTOR = re.compile(r"(?P<path>[^\s]+\.py)(?:::.*)?$")
_FAILED_SELECTOR = re.compile(
    r"(?m)^\s*FAILED\s+(?P<selector>[^\s]+\.py::[^\s]+)"
)


def _text(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    return str(value).strip() if value is not None else ""


def _normal(value: object) -> str:
    return _text(value).upper().replace("-", "_").replace(" ", "_")


def _mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        converted = as_dict()
        if isinstance(converted, Mapping):
            return converted
    return None


def _sequence(value: object) -> tuple[object, ...]:
    if value is None or isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(value, key=_text))
    return ()


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in (_text(item) for item in values) if value}))


def _clip(value: object, limit: int = MAX_FAILURE_TAIL_CHARS) -> str:
    text = _text(value)
    if len(text) <= limit:
        return text
    return text[-limit:]


def _clip_head(value: object, limit: int) -> str:
    """Normalize and keep the beginning of a diagnostic atom."""

    text = re.sub(r"\s+", " ", _text(value)).strip()
    limit = max(1, int(limit))
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool) or not _text(value):
        return None
    try:
        return int(_text(value))
    except (TypeError, ValueError):
        return None


def _canonical(value: object) -> JSONValue:
    """Convert the small set of receipt values used by this module to JSON."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            _text(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: _text(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical(item) for item in value]
    return _text(value)


def _digest(value: object, length: int = 16) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _kind(value: object) -> str:
    candidate = _normal(value).lower()
    if candidate in {"test_result", "suite_result", "pytest", "test"}:
        return "test"
    if candidate in {"artifact_evaluation", "artifact", "contract_evaluation", "evaluation"}:
        return "artifact"
    return candidate


def _result_class(record: Mapping[str, object]) -> str:
    for field_name in ("result_class", "outcome", "status", "result", "verdict"):
        value = record.get(field_name)
        if value is None or isinstance(value, Mapping):
            continue
        result = _normal(value)
        if result:
            return result
    return ""


def _state_for_result(value: object) -> ObligationState | None:
    result = _normal(value)
    if result in _GREEN_RESULTS:
        return ObligationState.SATISFIED
    if result in _MISSING_RESULTS:
        return ObligationState.MISSING
    if result in _UNKNOWN_RESULTS:
        return ObligationState.UNKNOWN
    if result in {"NA", "N_A", "NOT_APPLICABLE", "NOTAPPLICABLE"}:
        return ObligationState.NOT_APPLICABLE
    return None


def _is_selector(value: object) -> bool:
    text = _text(value)
    if not text or text.lower() in {"bare-run", "full", "full-selector", "all"}:
        return False
    return "::" in text or text.endswith(".py") or text.startswith("test_")


def _values(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        for name in ("selector", "nodeid", "id", "name", "test_id", "value", "ids", "tests", "items"):
            if name in value:
                return _values(value[name])
        return tuple(value.keys())
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        result: list[object] = []
        for item in value:
            result.extend(_values(item))
        return tuple(result)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(value, key=_text))
    return (value,)


def _extract_selectors(record: Mapping[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    for field_name in _SELECTOR_FIELDS:
        if field_name in record:
            values.extend(_text(item) for item in _values(record[field_name]))
    intent = _text(record.get("intent"))
    if _is_selector(intent):
        values.append(intent)
    for item in _values(record.get("command")):
        command_text = _text(item)
        candidates = command_text.split() if any(char.isspace() for char in command_text) else (command_text,)
        for candidate in candidates:
            candidate = candidate.strip("'\"")
            if _is_selector(candidate):
                values.append(candidate)
    return _dedupe(values)


def _path(value: str) -> str:
    candidate = value.strip().replace("\\", "/")
    return candidate[2:] if candidate.startswith("./") else candidate


def _paths_equivalent(left: str, right: str) -> bool:
    """Match repo-relative paths against pytest's temporary absolute paths."""

    normalized_left = _path(left)
    normalized_right = _path(right)
    return (
        normalized_left == normalized_right
        or normalized_left.endswith("/" + normalized_right)
        or normalized_right.endswith("/" + normalized_left)
    )


def _selector_matches(expected: str, observed: str, *, full_selector: bool = False) -> bool:
    # A bare selector has no file identity, so it must not match a
    # file-qualified receipt by suffix; build-unit selectors are qualified at
    # compile time from the runner's AST-derived in-world test IDs below.
    left = _path(expected)
    right = _path(observed)
    if left == right:
        return True
    if "::" in left and "::" in right:
        left_path, left_node = left.split("::", 1)
        right_path, right_node = right.split("::", 1)
        if left_node == right_node and _paths_equivalent(left_path, right_path):
            return True
    if not full_selector:
        return False
    match = _PATH_SELECTOR.match(right)
    if match and "::" in left and _paths_equivalent(
        left.split("::", 1)[0], match.group("path")
    ):
        return True
    return False


def _full_selector(record: Mapping[str, object], selectors: Sequence[str]) -> bool:
    value = record.get("full_selector", record.get("full_scope", record.get("is_full")))
    if isinstance(value, bool) and value:
        return True
    scope = _normal(record.get("scope", record.get("coverage")))
    if scope in {"FULL", "FULL_SUITE", "ALL", "COMPLETE"}:
        return True
    intent = _normal(record.get("intent"))
    if intent in {"FULL", "FULL_SELECTOR", "FULL_SUITE", "ALL"}:
        return True
    return any(item.endswith(".py") and "::" not in item for item in selectors)


def _failure_tail(record: Mapping[str, object]) -> str:
    for field_name in (
        "first_failing_atom",
        "failing_atom",
        "atom",
        "failing_tail",
        "failure_tail",
        "output_tail",
        "tail",
        "stderr",
        "error",
        "failure",
        "detail",
        "observation",
        "why",
    ):
        value = record.get(field_name)
        if isinstance(value, Mapping):
            for nested_name in (
                "tail",
                "message",
                "detail",
                "observation",
                "output",
                "stdout",
                "stderr",
                "why",
                "error",
            ):
                if nested_name in value and _text(value[nested_name]):
                    return _clip(value[nested_name])
        elif _text(value):
            return _clip(value)
    return ""


def _failed_selectors(record: Mapping[str, object], tail: str) -> tuple[str, ...]:
    """Extract explicit pytest failure node IDs from receipt evidence."""

    sources: list[str] = [tail]
    for field_name in (
        "first_failing_atom",
        "failing_atom",
        "atom",
        "failing_tail",
        "failure_tail",
        "output_tail",
        "tail",
        "stdout",
        "stderr",
        "output",
        "failure",
        "error",
        "detail",
        "observation",
        "why",
    ):
        value = record.get(field_name)
        if isinstance(value, Mapping):
            sources.extend(
                _text(value[nested_name])
                for nested_name in (
                    "tail",
                    "message",
                    "detail",
                    "observation",
                    "output",
                    "stdout",
                    "stderr",
                    "failure",
                    "error",
                    "why",
                )
                if _text(value.get(nested_name))
            )
        elif _text(value):
            sources.append(_text(value))
    selectors: list[str] = []
    for source in sources:
        for match in _FAILED_SELECTOR.finditer(source):
            selector = match.group("selector")
            path, node = selector.split("::", 1)
            selectors.append(f"{path}::{node.split('[', 1)[0]}")
    return _dedupe(selectors)


def _reason(record: Mapping[str, object]) -> str:
    for field_name in ("reason", "na_reason", "not_applicable_reason"):
        if _text(record.get(field_name)):
            return _clip(record[field_name], 240)
    return ""


def _evidence_ref(record: Mapping[str, object], index: int = 0) -> str:
    for field_name in ("receipt_id", "evidence_id", "fingerprint", "digest", "id"):
        if _text(record.get(field_name)):
            return _text(record[field_name])
    return f"receipt:{index}"


@dataclass(frozen=True)
class Obligation:
    """One deterministic obligation and its latest replayed evidence."""

    key: str
    kind: ObligationKind | str
    mandatory: bool = True
    selectors: tuple[str, ...] = ()
    state: ObligationState | str = ObligationState.UNKNOWN
    reason: str = ""
    failure_tail: str = ""
    evidence_refs: tuple[str, ...] = ()
    selector_states: tuple[tuple[str, ObligationState | str], ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    transitions: tuple[Record, ...] = ()
    last_satisfied_turn: int | None = None
    last_satisfied_evidence_ref: str = ""

    def __post_init__(self) -> None:
        key = _text(self.key)
        if not key:
            raise ValueError("obligation key must not be empty")
        try:
            kind = ObligationKind(self.kind)
        except ValueError as exc:
            raise ValueError(f"unknown obligation kind: {self.kind}") from exc
        try:
            state = ObligationState(self.state)
        except ValueError as exc:
            raise ValueError(f"unknown obligation state: {self.state}") from exc
        raw_selectors = (self.selectors,) if isinstance(self.selectors, str) else self.selectors
        raw_evidence_refs = (self.evidence_refs,) if isinstance(self.evidence_refs, str) else self.evidence_refs
        selectors = _dedupe(raw_selectors)
        evidence_refs = _dedupe(raw_evidence_refs)
        selector_states: list[tuple[str, ObligationState]] = []
        raw_selector_states = (
            self.selector_states.items()
            if isinstance(self.selector_states, Mapping)
            else self.selector_states
        )
        for selector, selector_state in raw_selector_states:
            try:
                coerced = ObligationState(selector_state)
            except ValueError as exc:
                raise ValueError(f"unknown selector state: {selector_state}") from exc
            selector_text = _text(selector)
            if selector_text:
                selector_states.append((selector_text, coerced))
        raw_metadata = self.metadata.items() if isinstance(self.metadata, Mapping) else self.metadata
        metadata = tuple(sorted((_text(key), _text(value)) for key, value in raw_metadata if _text(key)))
        raw_transitions = (
            (self.transitions,)
            if isinstance(self.transitions, Mapping)
            else self.transitions
        )
        transitions = tuple(
            dict(item) for item in raw_transitions if isinstance(item, Mapping)
        )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "mandatory", bool(self.mandatory))
        object.__setattr__(self, "selectors", selectors)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "reason", _clip(_text(self.reason), 240))
        object.__setattr__(self, "failure_tail", _clip(_text(self.failure_tail)))
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "selector_states", tuple(sorted(dict(selector_states).items())))
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "last_satisfied_turn", _optional_int(self.last_satisfied_turn))
        object.__setattr__(self, "last_satisfied_evidence_ref", _text(self.last_satisfied_evidence_ref))

    @property
    def gate_selectors(self) -> tuple[str, ...]:
        """Alias used by unit-plan callers."""

        return self.selectors

    @property
    def state_transitions(self) -> tuple[Record, ...]:
        """Compatibility alias for callers that use the full state name."""

        return self.transitions

    @property
    def gate_tests(self) -> tuple[str, ...]:
        """Compatibility alias for the planner's gate test naming."""

        return self.selectors

    @property
    def status(self) -> ObligationState:
        """Status alias for consumers that do not use state terminology."""

        return self.state

    @property
    def is_blocking(self) -> bool:
        return self.mandatory and (
            self.state in {ObligationState.UNKNOWN, ObligationState.MISSING}
            or (self.state is ObligationState.NOT_APPLICABLE and not self.reason)
        )

    def identity_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "kind": self.kind.value,
            "mandatory": self.mandatory,
            "selectors": list(self.selectors),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_dict(),
            "state": self.state.value,
            "reason": self.reason,
            "failure_tail": self.failure_tail,
            "evidence_refs": list(self.evidence_refs),
            "selector_states": {key: value.value for key, value in self.selector_states},
            "transitions": [dict(item) for item in self.transitions],
            "last_satisfied_turn": self.last_satisfied_turn,
            "last_satisfied_evidence_ref": self.last_satisfied_evidence_ref,
        }

    as_dict = to_dict


def _obligation_sort_key(obligation: Obligation) -> tuple[int, str]:
    return (_KIND_ORDER.get(obligation.kind.value, 99), obligation.key)


def reopen_dirty_obligations(graph: ObligationGraph, invalidation: object) -> ObligationGraph:
    """Reopen SATISFIED obligations whose evidence node became dirty."""
    dirty_refs = set(getattr(invalidation, "dirty_evidence_refs", ()) or ())
    if isinstance(invalidation, Mapping):
        dirty_refs.update(str(item) for item in invalidation.get("evidence_refs", ()) or ())
    if not dirty_refs:
        return graph
    changed: list[Obligation] = []
    transitions = list(graph.transitions)
    events = list(graph.events)
    for obligation in graph.obligations:
        if obligation.state is not ObligationState.SATISFIED:
            changed.append(obligation)
            continue
        hit = sorted(set(obligation.evidence_refs) & dirty_refs)
        if not hit:
            changed.append(obligation)
            continue
        transition = {
            "obligation_key": obligation.key,
            "from": ObligationState.SATISFIED.value,
            "to": ObligationState.UNKNOWN.value,
            "reason": "EVIDENCE_STALE",
            "evidence_ref": hit[0],
        }
        transitions.append(transition)
        events.append({"event": "OBLIGATION_REOPENED", **transition})
        changed.append(replace(
            obligation,
            state=ObligationState.UNKNOWN,
            reason="EVIDENCE_STALE",
            transitions=(*obligation.transitions, transition),
        ))
    graph.obligations = tuple(sorted(changed, key=_obligation_sort_key))
    graph.transitions = tuple(transitions)
    graph.events = tuple(events)
    return graph


@dataclass
class ObligationGraph:
    """Compiled task graph plus append-only graph lifecycle events."""

    task_id: str
    obligations: tuple[Obligation, ...]
    source_kind: str = "task"
    compile_gaps: tuple[str, ...] = ()
    graph_digest: str = ""
    events: tuple[Record, ...] = ()
    receipt_refs: tuple[str, ...] = ()
    transitions: tuple[Record, ...] = ()
    regression_notice_keys: tuple[str, ...] = ()
    last_edit_turn: int | None = None
    last_edit_fingerprint: str = ""

    def __post_init__(self) -> None:
        self.task_id = _text(self.task_id)
        self.source_kind = _text(self.source_kind) or "task"
        self.obligations = tuple(sorted(self.obligations, key=_obligation_sort_key))
        keys = [item.key for item in self.obligations]
        if len(keys) != len(set(keys)):
            raise ValueError("obligation keys must be unique")
        self.compile_gaps = _dedupe(self.compile_gaps)
        self.events = tuple(dict(item) for item in self.events if isinstance(item, Mapping))
        self.receipt_refs = _dedupe(self.receipt_refs)
        raw_transitions = (
            (self.transitions,)
            if isinstance(self.transitions, Mapping)
            else self.transitions
        )
        transitions = tuple(
            dict(item) for item in raw_transitions if isinstance(item, Mapping)
        )
        if not transitions:
            transitions = tuple(
                dict(item)
                for obligation in self.obligations
                for item in obligation.transitions
                if isinstance(item, Mapping)
            )
        self.transitions = transitions
        self.regression_notice_keys = _dedupe(self.regression_notice_keys)
        self.last_edit_turn = _optional_int(self.last_edit_turn)
        self.last_edit_fingerprint = _text(self.last_edit_fingerprint)
        if not self.graph_digest:
            self.graph_digest = _digest(
                {
                    "schema": SCHEMA,
                    "task_id": self.task_id,
                    "source_kind": self.source_kind,
                    "compile_gaps": self.compile_gaps,
                    "obligations": [item.identity_dict() for item in self.obligations],
                }
            )

    @property
    def digest(self) -> str:
        return self.graph_digest

    @property
    def blocker_keys(self) -> tuple[str, ...]:
        keys = [item.key for item in self.obligations if item.is_blocking]
        keys.extend(f"compile:{gap}" for gap in self.compile_gaps)
        return tuple(keys)

    @property
    def blockers(self) -> tuple[Obligation, ...]:
        return tuple(item for item in self.obligations if item.is_blocking)

    @property
    def mandatory_obligations(self) -> tuple[Obligation, ...]:
        return tuple(item for item in self.obligations if item.mandatory)

    @property
    def state_transitions(self) -> tuple[Record, ...]:
        """Compatibility alias for callers that use the full state name."""

        return self.transitions

    @property
    def regression_transitions(self) -> tuple[Record, ...]:
        return tuple(item for item in self.transitions if _is_regression_transition(item))

    @property
    def regressions(self) -> tuple[Record, ...]:
        return self.regression_transitions

    def get(self, key: str) -> Obligation | None:
        candidate = _text(key)
        return next((item for item in self.obligations if item.key == candidate), None)

    def by_kind(self, kind: ObligationKind | str) -> tuple[Obligation, ...]:
        candidate = ObligationKind(kind)
        return tuple(item for item in self.obligations if item.kind is candidate)

    def counts(self) -> dict[str, int]:
        counts = {state.value.lower(): 0 for state in ObligationState}
        for obligation in self.obligations:
            counts[obligation.state.value.lower()] += 1
        regression_count = len(self.regression_transitions)
        emitted_notice_keys = set(self.regression_notice_keys)
        pending_regression_count = len(
            {
                notice_key
                for notice_key, _ in _regression_notice_candidates(self)
                if notice_key not in emitted_notice_keys
            }
        )
        return {
            "total": len(self.obligations),
            "mandatory": sum(item.mandatory for item in self.obligations),
            **counts,
            "blocking": len(self.blocker_keys),
            "compile_gaps": len(self.compile_gaps),
            "transitions": len(self.transitions),
            "state_transitions": len(self.transitions),
            "regressions": regression_count,
            "regression_count": regression_count,
            "regression_transitions": regression_count,
            "regression_notices": len(self.regression_notice_keys),
            "regression_notices_emitted": len(self.regression_notice_keys),
            "regression_notices_pending": pending_regression_count,
            "pending_regression_notices": pending_regression_count,
        }

    def complete_enough(self) -> bool:
        return complete_enough(self)

    def coverage_receipt(self, *, new_gaps: int = 0) -> "CoverageReceipt":
        return coverage_receipt(self, new_gaps=new_gaps)

    def updated(self, receipts: Iterable[object]) -> "ObligationGraph":
        return update_obligation_graph(self, receipts)

    def update(self, receipts: Iterable[object]) -> "ObligationGraph":
        """Apply receipts in place and return this graph for fluent callers."""

        updated = update_obligation_graph(self, receipts)
        self.obligations = updated.obligations
        self.compile_gaps = updated.compile_gaps
        self.graph_digest = updated.graph_digest
        self.events = updated.events
        self.receipt_refs = updated.receipt_refs
        self.transitions = updated.transitions
        self.regression_notice_keys = updated.regression_notice_keys
        self.last_edit_turn = updated.last_edit_turn
        self.last_edit_fingerprint = updated.last_edit_fingerprint
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "task_id": self.task_id,
            "source_kind": self.source_kind,
            "graph_digest": self.graph_digest,
            "compile_gaps": list(self.compile_gaps),
            "obligations": [item.to_dict() for item in self.obligations],
            "counts": self.counts(),
            "complete_enough": self.complete_enough(),
            "blockers": list(self.blocker_keys),
            "receipt_refs": list(self.receipt_refs),
            "events": [dict(item) for item in self.events],
            "transitions": [dict(item) for item in self.transitions],
            "regression_notice_keys": list(self.regression_notice_keys),
            "last_edit_turn": self.last_edit_turn,
            "last_edit_fingerprint": self.last_edit_fingerprint,
        }

    as_dict = to_dict


def _task_id(task: Mapping[str, object]) -> str:
    for name in ("task_id", "id", "key", "name"):
        if _text(task.get(name)):
            return _text(task[name])
    if _text(task.get("sha")):
        return f"task:{_text(task['sha'])}"
    fallback = ":".join(
        _text(task.get(name))
        for name in ("repo", "rel", "fn")
        if _text(task.get(name))
    )
    if fallback:
        return f"task:{fallback}"
    return "task:unidentified"


def _task_is_creative(task: Mapping[str, object], contract: Mapping[str, object] | None) -> bool:
    track = _normal(task.get("track"))
    world = _normal(task.get("world"))
    task_id = _normal(task.get("task_id", task.get("id")))
    return (
        track == "CREATIVE"
        or world == "ARTIFACT"
        or task_id.startswith("CREATIVE:")
        or contract is not None and track == "CREATIVE"
    )


def _task_is_build(task: Mapping[str, object]) -> bool:
    repo = _normal(task.get("repo"))
    return (
        repo == "BUILD"
        or bool(task.get("is_build"))
        or bool(task.get("build_task"))
    )


def _source_kind(task: Mapping[str, object], creative: bool, build: bool, has_oracles: bool) -> str:
    if creative:
        return "creative"
    if build:
        return "build"
    if has_oracles:
        return "oracle"
    return "task"


def _oracle_ids(
    task: Mapping[str, object],
    *,
    oracle_ids: Sequence[object] | None,
    f2p_ids: Sequence[object] | None,
) -> tuple[str, ...]:
    values: list[object] = []
    if oracle_ids is None:
        values.extend(_values(task.get("oracle_ids", task.get("oracle_tests", task.get("oracles")))))
    else:
        values.extend(_values(oracle_ids))
    if f2p_ids is None:
        values.extend(_values(task.get("f2p_ids", task.get("f2p", task.get("fail_to_pass_ids")))))
    else:
        values.extend(_values(f2p_ids))
    normalized: list[str] = []
    for value in values:
        candidate = _text(value)
        for prefix in ("oracle:", "test:"):
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix) :]
        if candidate:
            normalized.append(candidate)
    return _dedupe(normalized)


def _build_test_oracle_data(task: Mapping[str, object]) -> tuple[str, tuple[str, ...]]:
    """Derive the runner file's function-level oracle IDs for a build row."""

    test_src = task.get("test_src")
    fn = _text(task.get("fn"))
    if not isinstance(test_src, str) or not fn:
        return "", ()
    runner_path = f"tests/test_{fn.strip('_')}_build.py"
    try:
        tree = ast.parse(test_src, filename=f"<{runner_path}>")
    except SyntaxError:
        return runner_path, ()
    names = _dedupe(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )
    return runner_path, tuple(f"{runner_path}::{name}" for name in names)


def _multifile_build_oracle_ids(
    task: Mapping[str, object], declared: Sequence[str],
) -> tuple[tuple[str, ...], bool]:
    """Derive functions only from declared public workspace test sources."""
    files = task.get("files")
    if not isinstance(files, Mapping):
        return (), False
    found: list[str] = []
    complete = bool(declared)
    for selector in declared:
        path = selector.split("::", 1)[0]
        payload = files.get(path)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("parent_src"), str):
            complete = False
            continue
        try:
            tree = ast.parse(payload["parent_src"], filename=path)
        except SyntaxError:
            complete = False
            continue
        before = len(found)
        names = [node.name for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name.startswith("test_")]
        for name in names:
            qualified = f"{path}::{name}"
            if selector == path or selector == qualified:
                found.append(qualified)
        complete = complete and len(found) > before
    return _dedupe(found), complete


def _oracle_metadata(test_id: str, *, build_runner_path: str) -> tuple[tuple[str, str], ...]:
    """Describe oracle granularity without turning a partial selector into a suite run."""
    if build_runner_path:
        return (
            ("granularity", "test_function"),
            ("oracle_file", build_runner_path),
            ("test_id", test_id),
        )
    if test_id.endswith(".py") and "::" not in test_id:
        # A non-build bare file is the corpus's only declared discriminator. This
        # is intentionally narrower than "any selector in the file": derived build
        # obligations above retain per-function granularity.
        return (
            ("granularity", "file"),
            ("oracle_file", test_id),
            ("test_id", test_id),
        )
    return (("test_id", test_id),)


def _unit_records(unit_plan: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(unit_plan, UnitPlan):
        return tuple(item.to_dict() for item in unit_plan.units)
    mapping = _mapping(unit_plan)
    values = mapping.get("units", ()) if mapping is not None else unit_plan
    records: list[Mapping[str, object]] = []
    if isinstance(values, Mapping):
        for name, item in values.items():
            converted = _mapping(item)
            if converted is None:
                continue
            named = dict(converted)
            named.setdefault("name", _text(name))
            records.append(named)
        return tuple(records)
    for item in _sequence(values):
        converted = _mapping(item)
        if converted is not None:
            records.append(converted)
    return tuple(records)


def _unit_name(record: Mapping[str, object]) -> str:
    for field_name in ("name", "symbol", "unit_name", "unit_id", "id"):
        value = _text(record.get(field_name))
        if value:
            if value.startswith("unit:"):
                value = value[5:]
            if value.endswith(":g0"):
                value = value[:-3]
            return value
    return ""


def _unit_gate_selectors(
    record: Mapping[str, object],
    *,
    runner_path: str = "",
    defined_selectors: Sequence[str] = (),
    qualify_bare: bool = False,
) -> tuple[str, ...]:
    values: list[object] = []
    for field_name in ("gate_selectors", "gate_tests", "selectors", "gates", "tests"):
        if field_name in record:
            values.extend(_values(record[field_name]))
    defined = set(defined_selectors)
    selectors = (_text(item) for item in values)
    def qualified_unit_selector(selector: str) -> str:
        matches = [value for value in defined_selectors if value.endswith("::" + selector)]
        return matches[0] if (qualify_bare and selector.startswith("test_")
                              and "::" not in selector and len(matches) == 1) else selector
    return _dedupe(
        (
            f"{runner_path}::{selector}"
            if runner_path
            and selector.startswith("test_")
            and "::" not in selector
            and f"{runner_path}::{selector}" in defined
            else qualified_unit_selector(selector)
        )
        for selector in selectors
    )


def _unit_metadata(
    record: Mapping[str, object], *, runner_path: str = ""
) -> tuple[tuple[str, str], ...]:
    metadata: dict[str, str] = {}
    for field_name in ("unit_id", "symbol", "kind", "depends_on", "handoff_rule"):
        value = record.get(field_name)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            value = ",".join(_text(item) for item in value)
        if _text(value):
            metadata[field_name] = _text(value)
    if runner_path:
        metadata["runner_path"] = runner_path
    return tuple(metadata.items())


def _contract_mapping(task: Mapping[str, object]) -> Mapping[str, object] | None:
    for field_name in ("contract", "artifact_contract"):
        value = _mapping(task.get(field_name))
        if value is not None:
            return value
    spec = _mapping(task.get("spec"))
    if spec is not None:
        return _mapping(spec.get("contract")) or spec
    return None


def _named_contract_values(value: object) -> tuple[tuple[str, bool], ...]:
    """Return deterministic (name, mandatory) pairs from a contract field."""

    pairs: list[tuple[str, bool]] = []
    if isinstance(value, Mapping):
        if any(name in value for name in ("name", "key", "id", "claim", "gate", "element")):
            name = next(
                (
                    _text(value.get(field_name))
                    for field_name in ("name", "key", "id", "claim", "gate", "element")
                    if _text(value.get(field_name))
                ),
                "",
            )
            mandatory = not bool(value.get("optional", False)) and value.get("required", True) is not False
            if name:
                pairs.append((name, mandatory))
        else:
            for key, item in value.items():
                item_mapping = _mapping(item)
                mandatory = True
                if item_mapping is not None:
                    mandatory = not bool(item_mapping.get("optional", False)) and item_mapping.get("required", True) is not False
                pairs.append((_text(key), mandatory))
        return tuple(sorted((name, mandatory) for name, mandatory in pairs if name))
    for item in _sequence(value):
        item_mapping = _mapping(item)
        if item_mapping is not None:
            pairs.extend(_named_contract_values(item_mapping))
        elif _text(item):
            pairs.append((_text(item), True))
    return tuple(sorted((name, mandatory) for name, mandatory in pairs if name))


def _contract_obligations(contract: Mapping[str, object]) -> tuple[Obligation, ...]:
    fields = (
        ("element_quality", "element"),
        ("elements", "element"),
        ("required_claims", "claim"),
        ("claims", "claim"),
        ("quality_gates", "gate"),
        ("gates", "gate"),
    )
    obligations: dict[str, Obligation] = {}
    for field_name, category in fields:
        if field_name not in contract:
            continue
        for name, mandatory in _named_contract_values(contract[field_name]):
            key = f"contract:{name}"
            # The public ID is the raw contract key. Category is preserved in
            # metadata so evidence can explain what that key represents.
            if key not in obligations:
                obligations[key] = Obligation(
                    key=key,
                    kind=ObligationKind.CONTRACT,
                    mandatory=mandatory,
                    metadata=(("contract_type", category),),
                )
    return tuple(obligations[key] for key in sorted(obligations))


def _compile_event(graph: ObligationGraph, event: str) -> Record:
    return {
        "schema": SCHEMA,
        "organ_id": ORGAN_ID,
        "event": event,
        "task_id": graph.task_id,
        "graph_digest": graph.graph_digest,
        "obligation_count": len(graph.obligations),
        "compile_gaps": list(graph.compile_gaps),
    }


def compile_obligation_graph(
    task: Mapping[str, object],
    *,
    oracle_ids: Sequence[object] | None = None,
    f2p_ids: Sequence[object] | None = None,
    unit_plan: object | None = None,
    include_units: bool | None = None,
    include_contract: bool | None = None,
) -> ObligationGraph:
    """Compile a stable graph from a task row, oracle IDs, plan, and contract.

    Creative rows default to contract obligations because their existing
    artifact receipts certify the contract lane. Passing an explicit unit plan
    always includes its units, including for a creative row.
    """

    if not isinstance(task, Mapping):
        raise TypeError("task must be a mapping")
    task_id = _task_id(task)
    contract = _contract_mapping(task)
    creative = _task_is_creative(task, contract)
    build = _task_is_build(task)
    compile_gaps: list[str] = []
    resolved_oracle_ids = _oracle_ids(task, oracle_ids=oracle_ids, f2p_ids=f2p_ids)
    build_runner_path = ""
    multifile_build = build and task.get("files") is not None
    if multifile_build:
        resolved_oracle_ids, complete = _multifile_build_oracle_ids(task, resolved_oracle_ids)
        if not complete:
            compile_gaps.append("multifile-public-test-functions-unavailable")
    elif build and isinstance(task.get("test_src"), str):
        build_runner_path, build_oracle_ids = _build_test_oracle_data(task)
        resolved_oracle_ids = build_oracle_ids
        if not build_oracle_ids:
            compile_gaps.append("build-test-functions-unavailable")
    obligations: list[Obligation] = [
        Obligation(
            key=f"oracle:{test_id}",
            kind=ObligationKind.ORACLE,
            selectors=(test_id,),
            metadata=_oracle_metadata(test_id, build_runner_path=(
                test_id.split("::", 1)[0] if multifile_build else build_runner_path)),
        )
        for test_id in resolved_oracle_ids
    ]
    embedded_plan = task.get("unit_plan", task.get("plan"))
    has_embedded_plan = embedded_plan is not None
    should_include_units = (
        unit_plan is not None or has_embedded_plan
        if include_units is None
        else include_units
    )
    if include_units is None and unit_plan is None and not has_embedded_plan:
        should_include_units = build and not creative
    if should_include_units:
        selected_plan = unit_plan if unit_plan is not None else embedded_plan
        if selected_plan is None:
            try:
                selected_plan = derive_unit_plan(task)
            except (AttributeError, KeyError, SyntaxError, TypeError, ValueError):
                compile_gaps.append("unit-plan-unavailable")
        records = _unit_records(selected_plan) if selected_plan is not None else ()
        if selected_plan is not None and not records:
            compile_gaps.append("unit-plan-empty")
        for record in records:
            name = _unit_name(record)
            if not name:
                compile_gaps.append("unit-name-missing")
                continue
            mandatory = record.get("required", True) is not False and not bool(record.get("optional", False))
            obligations.append(
                Obligation(
                    key=f"unit:{name}",
                    kind=ObligationKind.UNIT,
                    mandatory=mandatory,
                    selectors=_unit_gate_selectors(
                        record,
                        runner_path=build_runner_path,
                        defined_selectors=resolved_oracle_ids,
                        qualify_bare=multifile_build,
                    ),
                    metadata=_unit_metadata(record, runner_path=build_runner_path),
                )
            )

    should_include_contract = creative if include_contract is None else include_contract
    if should_include_contract:
        if contract is None:
            compile_gaps.append("contract-unavailable")
        else:
            contract_obligations = _contract_obligations(contract)
            if not contract_obligations:
                compile_gaps.append("contract-empty")
            obligations.extend(contract_obligations)

    source_kind = _source_kind(task, creative, build, bool(resolved_oracle_ids))
    graph = ObligationGraph(
        task_id=task_id,
        obligations=tuple(obligations),
        source_kind=source_kind,
        compile_gaps=tuple(compile_gaps),
    )
    graph.events = (_compile_event(graph, "OBLIGATION_GRAPH_COMPILED"),)
    return graph


build_obligation_graph = compile_obligation_graph
compile_graph = compile_obligation_graph
compile_obligations = compile_obligation_graph


def _copy_obligation(
    obligation: Obligation,
    *,
    state: ObligationState | None = None,
    reason: str | None = None,
    failure_tail: str | None = None,
    evidence_ref: str = "",
    selector_states: Mapping[str, ObligationState] | None = None,
) -> Obligation:
    refs = list(obligation.evidence_refs)
    if evidence_ref:
        refs.append(evidence_ref)
    selector_values = (
        dict(obligation.selector_states)
        if selector_states is None
        else dict(selector_states)
    )
    return replace(
        obligation,
        state=state if state is not None else obligation.state,
        reason=obligation.reason if reason is None else reason,
        failure_tail=obligation.failure_tail if failure_tail is None else failure_tail,
        evidence_refs=tuple(refs),
        selector_states=tuple((selector, value) for selector, value in selector_values.items()),
    )


def _set_state(
    obligations: dict[str, Obligation],
    key: str,
    state: ObligationState,
    *,
    reason: str = "",
    failure_tail: str = "",
    evidence_ref: str = "",
) -> None:
    obligation = obligations.get(key)
    if obligation is None:
        return
    next_reason = reason if state is not ObligationState.SATISFIED else ""
    next_tail = failure_tail if state is ObligationState.MISSING else ""
    obligations[key] = _copy_obligation(
        obligation,
        state=state,
        reason=next_reason,
        failure_tail=next_tail,
        evidence_ref=evidence_ref,
    )


def _resolve_obligation_keys(graph: ObligationGraph, raw: object, kind_hint: str = "") -> tuple[str, ...]:
    candidate = _text(raw)
    if not candidate:
        return ()
    keys = {item.key for item in graph.obligations}
    if candidate in keys:
        return (candidate,)
    if candidate.startswith(_OBLIGATION_PREFIXES):
        prefix, suffix = candidate.split(":", 1)
        exact = f"{prefix}:{suffix}"
        if exact in keys:
            return (exact,)
        qualified_suffix = suffix.rsplit(":", 1)[-1]
        qualified = f"{prefix}:{qualified_suffix}"
        if qualified in keys:
            return (qualified,)
    prefix = _normal(kind_hint).lower()
    if prefix in {"oracle", "unit", "contract"}:
        prefixed = f"{prefix}:{candidate}"
        if prefixed in keys:
            return (prefixed,)
    return tuple(sorted(key for key in keys if key.endswith(":" + candidate)))


def _iter_refs(value: object) -> tuple[tuple[str, Mapping[str, object] | None], ...]:
    if isinstance(value, Mapping):
        if any(field_name in value for field_name in ("key", "id", "name", "obligation", "obligation_id", "obligation_key")):
            name = next(
                (
                    _text(value.get(field_name))
                    for field_name in ("key", "id", "name", "obligation", "obligation_id", "obligation_key")
                    if _text(value.get(field_name))
                ),
                "",
            )
            return ((name, value),) if name else ()
        return tuple((_text(key), _mapping(item)) for key, item in value.items() if _text(key))
    if isinstance(value, (str, bytes)):
        return ((_text(value), None),) if _text(value) else ()
    result: list[tuple[str, Mapping[str, object] | None]] = []
    for item in _sequence(value):
        result.extend(_iter_refs(item))
    return tuple(result)


def _explicit_outcomes(
    record: Mapping[str, object],
    graph: ObligationGraph,
    result: str,
    tail: str,
    reason: str,
) -> tuple[tuple[str, ObligationState, str, str], ...]:
    outcomes: list[tuple[str, ObligationState, str, str]] = []
    default_state = _state_for_result(result) or ObligationState.UNKNOWN

    def add(value: object, state: ObligationState, hint: str = "") -> None:
        for raw, details in _iter_refs(value):
            local_state = state
            local_reason = reason
            local_tail = tail
            if details is not None:
                local_state = _state_for_result(
                    details.get("state", details.get("result", details.get("status")))
                ) or state
                local_reason = _text(details.get("reason", details.get("na_reason"))) or local_reason
                local_tail = _failure_tail(details) or local_tail
            for key in _resolve_obligation_keys(graph, raw, hint):
                outcomes.append((key, local_state, local_reason, local_tail))

    direct_fields = (
        ("obligation_keys", default_state, ""),
        ("obligation_ids", default_state, ""),
        ("satisfied_obligations", ObligationState.SATISFIED, ""),
        ("passed_obligations", ObligationState.SATISFIED, ""),
        ("missing_obligations", ObligationState.MISSING, ""),
        ("failed_obligations", ObligationState.MISSING, ""),
        ("unknown_obligations", ObligationState.UNKNOWN, ""),
        ("not_applicable", ObligationState.NOT_APPLICABLE, ""),
        ("not_applicable_obligations", ObligationState.NOT_APPLICABLE, ""),
        ("unit", default_state, "unit"),
        ("oracle", default_state, "oracle"),
        ("contract_element", default_state, "contract"),
        ("contract_claim", default_state, "contract"),
        ("contract_gate", default_state, "contract"),
        ("element", default_state, "contract"),
        ("claim", default_state, "contract"),
        ("gate", default_state, "contract"),
        ("contract_key", default_state, "contract"),
    )
    for field_name, field_state, hint in direct_fields:
        if field_name in record:
            add(record[field_name], field_state, hint)
    for field_name in ("not_applicable", "not_applicable_obligations"):
        value = record.get(field_name)
        if not isinstance(value, Mapping):
            continue
        for raw, detail in value.items():
            keys = _resolve_obligation_keys(graph, raw)
            detail_mapping = _mapping(detail)
            local_reason = _reason(detail_mapping) if detail_mapping is not None else _text(detail)
            for key in keys:
                outcomes.append((key, ObligationState.NOT_APPLICABLE, local_reason, ""))
    if "obligation" in record:
        add(record["obligation"], default_state)
    if isinstance(record.get("contract"), (str, bytes)):
        add(record["contract"], default_state, "contract")
    if "obligations" in record:
        for raw, details in _iter_refs(record["obligations"]):
            local_state = default_state
            local_reason = reason
            local_tail = tail
            if details is not None:
                local_state = _state_for_result(
                    details.get("state", details.get("result", details.get("status")))
                ) or local_state
                local_reason = _text(details.get("reason")) or local_reason
                local_tail = _failure_tail(details) or local_tail
            for key in _resolve_obligation_keys(graph, raw):
                outcomes.append((key, local_state, local_reason, local_tail))
    return tuple(outcomes)


def _named_results(value: object, default_state: ObligationState) -> tuple[tuple[str, ObligationState, str], ...]:
    result: list[tuple[str, ObligationState, str]] = []

    def add(name: object, details: Mapping[str, object] | None, fallback: ObligationState) -> None:
        candidate = _text(name)
        if not candidate:
            return
        state = fallback
        detail = ""
        if details is not None:
            if "passed" in details:
                state = ObligationState.SATISFIED if bool(details["passed"]) else ObligationState.MISSING
            state = _state_for_result(
                details.get("state", details.get("result", details.get("status")))
            ) or state
            detail = _failure_tail(details) if state is ObligationState.MISSING else _reason(details)
        result.append((candidate, state, detail))

    if isinstance(value, Mapping):
        if any(field_name in value for field_name in ("name", "key", "id", "claim", "gate", "element")):
            name = next(
                (
                    value.get(field_name)
                    for field_name in ("name", "key", "id", "claim", "gate", "element")
                    if _text(value.get(field_name))
                ),
                "",
            )
            add(name, value, default_state)
        else:
            for name, item in value.items():
                item_mapping = _mapping(item)
                fallback = (
                    ObligationState.SATISFIED
                    if item is True
                    else ObligationState.MISSING
                    if item is False
                    else default_state
                )
                add(name, item_mapping, fallback)
    else:
        for item in _sequence(value):
            item_mapping = _mapping(item)
            if item_mapping is not None:
                result.extend(_named_results(item_mapping, default_state))
            elif _text(item):
                add(item, None, default_state)
    return tuple(result)


def _contains_term(text: str, term: str) -> bool:
    candidate = _text(term).lower()
    if not candidate:
        return False
    return re.search(
        rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])",
        text.lower(),
    ) is not None


def _contract_test_outcomes(
    record: Mapping[str, object],
    graph: ObligationGraph,
    result: str,
    tail: str,
) -> tuple[tuple[str, ObligationState, str, str], ...]:
    """Bridge named phase-test selectors to their contract obligations.

    The phased desktop corpus exposes the quality-gate test as ``u7`` while
    the contract lane exposes ``css_tokens`` and its sibling gates.  A green
    u7 covers the complete quality-gate/claim check; a red u7 without a
    structured failure atom uses the measured css_tokens failure as its
    narrow fallback instead of marking every contract obligation missing.
    """

    kind = _kind(record.get("kind", record.get("type")))
    if kind not in {"test", "suite", "check", "artifact", "evaluation", "verdict", "done"}:
        return ()
    state = _state_for_result(result)
    if state not in {ObligationState.SATISFIED, ObligationState.MISSING}:
        return ()
    contract_obligations = tuple(
        item for item in graph.obligations if item.kind is ObligationKind.CONTRACT
    )
    if not contract_obligations:
        return ()
    searchable = " ".join(
        _text(record.get(field_name))
        for field_name in (
            "intent",
            "selector",
            "test_selector",
            "nodeid",
            "why",
            "detail",
            "failure",
            "error",
            "failing_tail",
        )
    )
    searchable = f"{searchable} {tail}".strip()
    diagnostic_searchable = " ".join(
        _text(record.get(field_name))
        for field_name in (
            "why",
            "detail",
            "failure",
            "error",
            "failing_tail",
            "tail",
            "output_tail",
        )
    )
    diagnostic_searchable = f"{diagnostic_searchable} {tail}".strip()
    why_tail = _clip(record.get("why"))
    normalized_tail = _clip(record.get("failing_tail"))
    diagnostic_tail = (
        normalized_tail
        if normalized_tail and normalized_tail != why_tail
        else _failure_tail(
            {
                field_name: record.get(field_name)
                for field_name in (
                    "failure_tail",
                    "output_tail",
                    "tail",
                    "stderr",
                    "error",
                    "failure",
                    "detail",
                )
            }
        )
    )
    explicit_searchable = " ".join(
        _text(record.get(field_name))
        for field_name in ("detail", "failure", "error", "tail", "output_tail")
    )
    explicit_searchable = f"{explicit_searchable} {diagnostic_tail}".strip()
    named = tuple(
        item.key
        for item in contract_obligations
        if _contains_term(diagnostic_searchable, item.key.split(":", 1)[1])
    )
    named_explicit = tuple(
        item.key
        for item in contract_obligations
        if _contains_term(explicit_searchable, item.key.split(":", 1)[1])
    )
    is_u7 = _contains_term(searchable, "u7")
    if is_u7 and state is ObligationState.SATISFIED:
        targets = tuple(
            item.key
            for item in contract_obligations
            if dict(item.metadata).get("contract_type") in {"gate", "claim"}
        )
    elif is_u7 and state is ObligationState.MISSING:
        css_tokens_key = next(
            (
                item.key
                for item in contract_obligations
                if item.key == "contract:css_tokens"
            ),
            "",
        )
        targets = named_explicit or ((css_tokens_key,) if css_tokens_key else ())
    else:
        targets = named
    if not targets:
        return ()
    unique_targets = tuple(dict.fromkeys(targets))
    outcomes: list[tuple[str, ObligationState, str, str]] = []
    for key in unique_targets:
        detail = ""
        if state is ObligationState.MISSING:
            detail = diagnostic_tail or (tail if tail != why_tail else "") or key.split(":", 1)[1]
        outcomes.append((key, state, "", detail))
    return tuple(outcomes)


def _contract_outcomes(
    record: Mapping[str, object],
    graph: ObligationGraph,
    result: str,
    tail: str,
) -> tuple[tuple[str, ObligationState, str, str], ...]:
    contract_keys = tuple(item.key for item in graph.obligations if item.kind is ObligationKind.CONTRACT)
    if not contract_keys:
        return ()
    outcomes: list[tuple[str, ObligationState, str, str]] = []

    for field_name in (
        "gates",
        "gate_results",
        "quality_gates",
        "elements",
        "element_results",
        "element_passed",
        "claims",
        "claim_results",
        "claim_passed",
    ):
        if field_name not in record:
            continue
        default = _state_for_result(result) or ObligationState.UNKNOWN
        for raw, state, detail in _named_results(record[field_name], default):
            for key in _resolve_obligation_keys(graph, raw, "contract"):
                outcomes.append(
                    (key, state, "", detail if state is ObligationState.MISSING else "")
                )

    nested_candidates: list[Mapping[str, object]] = []
    for field_name in ("contract", "evaluation", "artifact_evaluation", "contract_evaluation"):
        nested = _mapping(record.get(field_name))
        if nested is not None:
            nested_candidates.append(nested)
    for nested in nested_candidates:
        for field_name in ("element_passed", "elements", "gates", "claims"):
            if field_name in nested:
                default = _state_for_result(result) or ObligationState.UNKNOWN
                for raw, state, detail in _named_results(nested[field_name], default):
                    for key in _resolve_obligation_keys(graph, raw, "contract"):
                        outcomes.append(
                            (key, state, "", detail if state is ObligationState.MISSING else "")
                        )

    failures = record.get("failures")
    if failures is not None:
        failure_text = " ".join(_text(item) for item in _values(failures))
        for obligation in graph.obligations:
            raw_key = obligation.key.split(":", 1)[1]
            if raw_key and raw_key.lower() in failure_text.lower():
                outcomes.append(
                    (
                        obligation.key,
                        ObligationState.MISSING,
                        "",
                        _clip(failure_text) or tail,
                    )
                )

    kind = _kind(record.get("kind", record.get("type")))
    certified_flag = _normal(record.get("contract_certified")) in {"TRUE", "YES", "1"}
    certified = certified_flag or record.get("contract_passed") is True
    if kind in {"artifact", "contract", "evaluation"} and record.get("passed") is True:
        certified = True
    if kind in {"artifact", "verdict", "done", "contract", "evaluation"} and result in {
        "CERTIFIED",
        "ACCEPTED",
        "GREEN",
        "PASS",
        "PASSED",
        "SUCCESS",
    }:
        certified = True
    for field_name in ("evaluation", "artifact_evaluation", "contract_evaluation", "contract"):
        nested = _mapping(record.get(field_name))
        if nested is not None and nested.get("passed") is True:
            certified = True
    if certified:
        outcomes.extend((key, ObligationState.SATISFIED, "", "") for key in contract_keys)
    elif (
        kind in {"artifact", "contract", "evaluation"}
        and (result in _MISSING_RESULTS or record.get("passed") is False)
        and not outcomes
    ):
        outcomes.extend(
            (key, ObligationState.MISSING, "", tail or f"artifact reported {result}")
            for key in contract_keys
        )
    outcomes.extend(_contract_test_outcomes(record, graph, result, tail))
    return tuple(outcomes)


def convert_receipt(record: object, *, index: int = 0) -> Record:
    """Normalize raw tape records without losing their original fields."""

    converted = dict(_mapping(record) or {})
    result = _result_class(converted)
    selectors = _extract_selectors(converted)
    converted.update(
        {
            "schema": _text(converted.get("schema")) or SCHEMA,
            "kind": _kind(converted.get("kind", converted.get("type"))),
            "result_class": result,
            "selectors": list(selectors),
            "selector": selectors[0] if len(selectors) == 1 else _text(converted.get("selector")),
            "failing_tail": _failure_tail(converted),
            "full_selector": _full_selector(converted, selectors),
            "evidence_ref": _evidence_ref(converted, index),
        }
    )
    return converted


def convert_receipts(records: Iterable[object]) -> tuple[Record, ...]:
    return tuple(convert_receipt(record, index=index) for index, record in enumerate(records))


def load_receipts(path: str | Path) -> tuple[Record, ...]:
    """Load valid JSONL records in line order; malformed lines are reported."""

    records: list[Record] = []
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("obligation receipt read failed path=%s error=%s", source, type(exc).__name__)
        return ()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("obligation receipt parse failed path=%s line=%d", source, index + 1)
            continue
        if isinstance(value, Mapping):
            records.append(convert_receipt(value, index=len(records)))
    return tuple(records)


def _record_turn(record: Mapping[str, object], fallback: int | None = None) -> int | None:
    turn = _optional_int(record.get("turn"))
    if turn is not None:
        return turn
    return _optional_int(record.get("step")) if record.get("step") is not None else fallback


def _receipt_order(item: tuple[int, object]) -> tuple[int, int, str]:
    index, value = item
    record = _mapping(value) or {}
    turn = _record_turn(record)
    if turn is None:
        turn = 2**31 - 1
    return (turn, index, _evidence_ref(record, index))


def _apply_selector_observation(
    obligations: dict[str, Obligation],
    record: Mapping[str, object],
    *,
    state: ObligationState,
    tail: str,
    evidence_ref: str,
) -> None:
    selectors = _extract_selectors(record)
    full_selector = _full_selector(record, selectors)
    failed_selectors = _failed_selectors(record, tail) if state is ObligationState.MISSING else ()
    if not selectors and failed_selectors:
        selectors = _dedupe(
            failed.split("::", 1)[0]
            for failed in failed_selectors
            if "::" in failed
        )
        full_selector = True
    if not selectors and _normal(record.get("intent")) == "BARE_RUN":
        runner_paths = _dedupe(
            dict(obligation.metadata).get("oracle_file", "")
            for obligation in obligations.values()
            if obligation.kind is ObligationKind.ORACLE
            and dict(obligation.metadata).get("granularity") == "test_function"
        )
        if len(runner_paths) == 1:
            selectors = runner_paths
            full_selector = True

    def matches_oracle(obligation: Obligation, selector: str) -> bool:
        expected = obligation.key.split(":", 1)[1]
        if _selector_matches(expected, selector, full_selector=full_selector):
            return True
        metadata = dict(obligation.metadata)
        if metadata.get("granularity") != "file" or "::" in expected:
            return False
        # A corpus file obligation has no finer declared ID. A selector inside
        # that file is relevant evidence for that obligation; test_function
        # obligations deliberately do not enter this branch.
        return _paths_equivalent(expected, selector.split("::", 1)[0])

    for obligation in tuple(obligations.values()):
        if obligation.kind is ObligationKind.ORACLE:
            matched = any(
                (
                    matches_oracle(obligation, selector)
                    and (
                        not (
                            state is ObligationState.MISSING
                            and full_selector
                            and selector.endswith(".py")
                            and "::" not in selector
                            and "::" in obligation.key
                        )
                        or any(
                            "::" in failed
                            and _paths_equivalent(
                                failed.split("::", 1)[0], selector
                            )
                            and _selector_matches(
                                obligation.key.split(":", 1)[1],
                                failed,
                            )
                            for failed in failed_selectors
                        )
                    )
                )
                for selector in selectors
            )
        elif obligation.kind is ObligationKind.UNIT:
            matched = any(
                any(
                    not (
                        state is ObligationState.MISSING
                        and full_selector
                        and selector.endswith(".py")
                        and "::" not in selector
                    )
                    and _selector_matches(expected, selector, full_selector=full_selector)
                    for expected in obligation.selectors
                )
                for selector in selectors
            )
        else:
            matched = False
        if not matched:
            continue
        if obligation.kind is ObligationKind.UNIT and obligation.selectors:
            selector_states = dict(obligation.selector_states)
            for expected in obligation.selectors:
                if any(
                    not (
                        state is ObligationState.MISSING
                        and full_selector
                        and selector.endswith(".py")
                        and "::" not in selector
                    )
                    and _selector_matches(expected, selector, full_selector=full_selector)
                    for selector in selectors
                ):
                    selector_states[expected] = state
            obligations[obligation.key] = _copy_obligation(
                obligation,
                evidence_ref=evidence_ref,
                failure_tail=tail if state is ObligationState.MISSING else "",
                selector_states=selector_states,
            )
        else:
            obligations[obligation.key] = _copy_obligation(
                obligation,
                state=state,
                reason="",
                failure_tail=tail if state is ObligationState.MISSING else "",
                evidence_ref=evidence_ref,
            )


def _recompute_units(obligations: dict[str, Obligation]) -> None:
    for key, obligation in tuple(obligations.items()):
        if obligation.kind is not ObligationKind.UNIT or not obligation.selectors:
            continue
        states = dict(obligation.selector_states)
        observed = tuple(states.get(selector, ObligationState.UNKNOWN) for selector in obligation.selectors)
        if any(state is ObligationState.MISSING for state in observed):
            missing_selector = next(
                selector
                for selector in obligation.selectors
                if states.get(selector) is ObligationState.MISSING
            )
            tail = obligation.failure_tail or f"{missing_selector} reported MISSING"
            obligations[key] = replace(
                obligation,
                state=ObligationState.MISSING,
                reason="",
                failure_tail=tail,
            )
        elif all(state is ObligationState.SATISFIED for state in observed):
            obligations[key] = replace(
                obligation,
                state=ObligationState.SATISFIED,
                reason="",
                failure_tail="",
            )
        elif all(
            state in {ObligationState.SATISFIED, ObligationState.NOT_APPLICABLE}
            for state in observed
        ):
            obligations[key] = replace(
                obligation,
                state=ObligationState.SATISFIED,
                reason="all gate selectors satisfied or not applicable",
                failure_tail="",
            )
        else:
            obligations[key] = replace(
                obligation,
                state=ObligationState.UNKNOWN,
                reason="",
                failure_tail="",
            )


_EDIT_KINDS = frozenset(
    {
        "edit",
        "file_edit",
        "file_create",
        "file_write",
        "patch",
        "replace",
        "str_replace",
        "write",
    }
)
_EDIT_FAILURE_RESULTS = frozenset(
    {"ERROR", "FAILED", "FAIL", "NO_OP", "REJECTED", "TIMEOUT"}
)


def _edit_fingerprint(record: Mapping[str, object]) -> str:
    for field_name in (
        "edit_receipt_fingerprint",
        "edit_fingerprint",
        "fingerprint",
        "receipt_fingerprint",
        "digest",
        "edit_receipt",
        "receipt",
    ):
        value = record.get(field_name)
        if isinstance(value, Mapping):
            candidate = next(
                (
                    value.get(nested_name)
                    for nested_name in ("fingerprint", "receipt_fingerprint", "digest")
                    if _text(value.get(nested_name))
                ),
                "",
            )
        else:
            candidate = value
        candidate_text = _text(candidate)
        if candidate_text and not candidate_text.startswith("receipt:"):
            return candidate_text
    return ""


def _successful_edit_context(
    record: Mapping[str, object], index: int
) -> tuple[int | None, str] | None:
    kind = _kind(record.get("kind", record.get("type")))
    if kind not in _EDIT_KINDS and not kind.endswith("_edit"):
        return None
    if _result_class(record) in _EDIT_FAILURE_RESULTS:
        return None
    detail = _normal(record.get("detail"))
    if "NO_OP" in detail or detail.startswith(("ERROR", "FAILED", "REJECTED")):
        return None
    fingerprint = _edit_fingerprint(record)
    if not fingerprint:
        return None
    return (_record_turn(record, index), fingerprint)


def _is_regression_transition(record: Mapping[str, object]) -> bool:
    previous = _normal(
        record.get("from_state", record.get("previous_state", record.get("from")))
    )
    current = _normal(
        record.get("to_state", record.get("next_state", record.get("to")))
    )
    return previous == ObligationState.SATISFIED.value and current == ObligationState.MISSING.value


def _transition_notice_key(record: Mapping[str, object]) -> str:
    obligation = _text(record.get("obligation", record.get("key")))
    break_turn = _optional_int(record.get("break_turn", record.get("turn")))
    if not obligation or break_turn is None:
        return ""
    return json.dumps((obligation, break_turn), separators=(",", ":"))


def _regression_notice_candidates(
    graph: ObligationGraph,
) -> tuple[tuple[str, Record], ...]:
    candidates: list[tuple[str, Record]] = []
    seen: set[str] = set()
    for transition in graph.regression_transitions:
        notice_key = _transition_notice_key(transition)
        edit_turn = _optional_int(transition.get("edit_turn"))
        edit_fingerprint = _text(transition.get("edit_receipt_fingerprint")) or _text(
            transition.get("edit_fingerprint")
        )
        if not notice_key or edit_turn is None or not edit_fingerprint or notice_key in seen:
            continue
        seen.add(notice_key)
        candidates.append((notice_key, transition))
    return tuple(candidates)


def _render_regression_line(transition: Mapping[str, object]) -> str:
    obligation = _text(transition.get("obligation", transition.get("key")))
    edit_turn = _optional_int(transition.get("edit_turn"))
    edit_fingerprint = _text(transition.get("edit_receipt_fingerprint")) or _text(
        transition.get("edit_fingerprint")
    )
    if not obligation or edit_turn is None or not edit_fingerprint:
        return ""
    passing_value = transition.get("passing_turn")
    if passing_value is None:
        passing_value = transition.get("satisfied_turn")
    passing_turn = _optional_int(passing_value)
    passing_label = _clip_head(
        str(passing_turn) if passing_turn is not None else "?",
        24,
    )
    failure_value = transition.get("first_failing_tail")
    if not _text(failure_value):
        failure_value = transition.get("first_failing_atom")
    if not _text(failure_value):
        failure_value = transition.get("failure_tail")
    failure_tail = _clip_head(
        failure_value or "the obligation reported MISSING",
        MAX_REGRESSION_NOTICE_CHARS,
    ).rstrip(" .")
    obligation = _clip_head(obligation, 96)
    edit_label = _clip_head(str(edit_turn), 24)
    prefix = (
        f"REGRESSION: {obligation} was passing (turn {passing_label}) and your edit "
        f"at turn {edit_label} broke it: "
    )
    suffix = ". Restore it before continuing."
    available = MAX_REGRESSION_NOTICE_CHARS - len(prefix) - len(suffix)
    if available < 1:
        prefix = _clip_head(prefix, MAX_REGRESSION_NOTICE_CHARS - len(suffix) - 1)
        available = MAX_REGRESSION_NOTICE_CHARS - len(prefix) - len(suffix)
    atom = _clip_head(failure_tail, available).rstrip(" .") or "MISSING"
    return f"{prefix}{atom}{suffix}"[:MAX_REGRESSION_NOTICE_CHARS]


def render_regression_notice(graph: ObligationGraph) -> str:
    """Queue and return one bounded line for each newly observed regression."""

    if not isinstance(graph, ObligationGraph):
        raise TypeError("graph must be an ObligationGraph")
    emitted: list[str] = []
    notice_keys = set(graph.regression_notice_keys)
    for notice_key, transition in _regression_notice_candidates(graph):
        if notice_key in notice_keys:
            continue
        line = _render_regression_line(transition)
        if not line:
            continue
        notice_scope = (graph.task_id, notice_key)
        if notice_scope in _PENDING_REGRESSION_KEYS or notice_scope in _REGRESSION_NOTICE_STORE_KEYS:
            notice_keys.add(notice_key)
            continue
        _PENDING_REGRESSION_NOTICES.append((notice_scope, line))
        _PENDING_REGRESSION_KEYS.add(notice_scope)
        _REGRESSION_NOTICE_STORE.append({
            "id": f"regression-{len(_REGRESSION_NOTICE_STORE)}",
            "kind": "regression_notice",
            "task_id": graph.task_id,
            "notice_key": notice_key,
            "obligation_id": str(transition.get("obligation", transition.get("key", ""))),
            "step": transition.get("break_turn", transition.get("turn")),
            "gist": line,
            "line": line,
        })
        _REGRESSION_NOTICE_STORE_KEYS.add(notice_scope)
        notice_keys.add(notice_key)
        emitted.append(line)
    graph.regression_notice_keys = tuple(sorted(notice_keys))
    return "\n".join(emitted)


def pending_regression_notices() -> tuple[str, ...]:
    """Drain queued notices for the harness delivery chokepoint."""

    notices = tuple(line for _, line in _PENDING_REGRESSION_NOTICES)
    _PENDING_REGRESSION_NOTICES.clear()
    _PENDING_REGRESSION_KEYS.clear()
    return notices


def all_regression_notices() -> tuple[dict[str, object], ...]:
    """Return the lossless notice tape without changing pending delivery state."""
    return tuple(copy.deepcopy(row) for row in _REGRESSION_NOTICE_STORE)


drain_regression_notices = pending_regression_notices


def _state_transition_record(
    graph: ObligationGraph,
    previous: Obligation,
    current: Obligation,
    record: Mapping[str, object],
    *,
    index: int,
    result: str,
    tail: str,
    evidence_ref: str,
    edit_turn: int | None,
    edit_fingerprint: str,
) -> Record:
    turn = _record_turn(record, index)
    failure_tail = _clip(
        current.failure_tail
        or tail
        or f"{current.key} reported {result or current.state.value}"
    )
    return {
        "schema": SCHEMA,
        "organ_id": ORGAN_ID,
        "event": "OBLIGATION_STATE_TRANSITION",
        "task_id": graph.task_id,
        "obligation": current.key,
        "from_state": previous.state.value,
        "to_state": current.state.value,
        "previous_state": previous.state.value,
        "next_state": current.state.value,
        "turn": turn,
        "passing_turn": previous.last_satisfied_turn,
        "satisfied_turn": previous.last_satisfied_turn,
        "break_turn": turn,
        "receipt_ref": evidence_ref,
        "receipt_fingerprint": _text(record.get("fingerprint")),
        "edit_turn": edit_turn,
        "edit_fingerprint": edit_fingerprint,
        "edit_receipt_fingerprint": edit_fingerprint,
        "edit_receipt_ref": edit_fingerprint,
        "failure_tail": failure_tail,
        "first_failing_atom": failure_tail,
        "first_failing_tail": failure_tail,
        "graph_digest": graph.graph_digest,
    }


def _record_state_transitions(
    graph: ObligationGraph,
    before: Mapping[str, Obligation],
    obligations: dict[str, Obligation],
    transitions: list[Record],
    record: Mapping[str, object],
    *,
    index: int,
    result: str,
    tail: str,
    evidence_ref: str,
    edit_turn: int | None,
    edit_fingerprint: str,
) -> None:
    for key, previous in before.items():
        current = obligations.get(key)
        if current is None or current.state is previous.state:
            continue
        transition = _state_transition_record(
            graph,
            previous,
            current,
            record,
            index=index,
            result=result,
            tail=tail,
            evidence_ref=evidence_ref,
            edit_turn=edit_turn,
            edit_fingerprint=edit_fingerprint,
        )
        transitions.append(transition)
        if current.state is ObligationState.SATISFIED:
            current = replace(
                current,
                last_satisfied_turn=_record_turn(record, index),
                last_satisfied_evidence_ref=evidence_ref,
            )
        obligations[key] = replace(
            current,
            transitions=(*current.transitions, transition),
        )


def _update_graph_event(graph: ObligationGraph, event: str) -> Record:
    return {
        "schema": SCHEMA,
        "organ_id": ORGAN_ID,
        "event": event,
        "task_id": graph.task_id,
        "graph_digest": graph.graph_digest,
        "counts": graph.counts(),
        "blockers": list(graph.blocker_keys),
    }


def update_obligation_graph(graph: ObligationGraph, receipts: Iterable[object]) -> ObligationGraph:
    """Replay receipts in deterministic turn order and return a new graph."""

    if not isinstance(graph, ObligationGraph):
        raise TypeError("graph must be an ObligationGraph")
    if isinstance(receipts, Mapping) or isinstance(receipts, (str, bytes)):
        receipt_values: Iterable[object] = (receipts,)
    else:
        receipt_values = receipts
    ordered = sorted(enumerate(receipt_values), key=_receipt_order)
    obligations = {item.key: item for item in graph.obligations}
    receipt_refs = list(graph.receipt_refs)
    transitions = list(graph.transitions)
    last_edit_turn = graph.last_edit_turn
    last_edit_fingerprint = graph.last_edit_fingerprint
    for index, raw in ordered:
        record = convert_receipt(raw, index=index)
        before = dict(obligations)
        result = _result_class(record)
        state = _state_for_result(result) or ObligationState.UNKNOWN
        tail = _failure_tail(record)
        evidence_ref = _evidence_ref(record, index)
        prior_edit_turn = last_edit_turn
        prior_edit_fingerprint = last_edit_fingerprint
        if evidence_ref not in receipt_refs:
            receipt_refs.append(evidence_ref)
        for key, explicit_state, reason, explicit_tail in _explicit_outcomes(
            record, graph, result, tail, _reason(record)
        ):
            _set_state(
                obligations,
                key,
                explicit_state,
                reason=reason,
                failure_tail=explicit_tail or tail or f"{key} reported {result}",
                evidence_ref=evidence_ref,
            )
            if obligations[key].kind is ObligationKind.UNIT and obligations[key].selectors:
                selector_states = {
                    selector: explicit_state for selector in obligations[key].selectors
                }
                obligations[key] = _copy_obligation(
                    obligations[key],
                    selector_states=selector_states,
                    evidence_ref="",
                )
        if state is not ObligationState.NOT_APPLICABLE and (
            state is not ObligationState.UNKNOWN or _extract_selectors(record)
        ):
            _apply_selector_observation(
                obligations,
                record,
                state=state,
                tail=tail or f"{_text(record.get('selector')) or 'selector'} reported {result or 'UNKNOWN'}",
                evidence_ref=evidence_ref,
            )
        for key, contract_state, reason, contract_tail in _contract_outcomes(
            record, graph, result, tail
        ):
            _set_state(
                obligations,
                key,
                contract_state,
                reason=reason,
                failure_tail=contract_tail or tail or f"{key} reported {result}",
                evidence_ref=evidence_ref,
            )
        _recompute_units(obligations)
        _record_state_transitions(
            graph,
            before,
            obligations,
            transitions,
            record,
            index=index,
            result=result,
            tail=tail,
            evidence_ref=evidence_ref,
            edit_turn=prior_edit_turn,
            edit_fingerprint=prior_edit_fingerprint,
        )
        edit_context = _successful_edit_context(record, index)
        if edit_context is not None:
            last_edit_turn, last_edit_fingerprint = edit_context
    updated = ObligationGraph(
        task_id=graph.task_id,
        obligations=tuple(obligations.values()),
        source_kind=graph.source_kind,
        compile_gaps=graph.compile_gaps,
        graph_digest=graph.graph_digest,
        events=graph.events,
        receipt_refs=tuple(receipt_refs),
        transitions=tuple(transitions),
        regression_notice_keys=graph.regression_notice_keys,
        last_edit_turn=last_edit_turn,
        last_edit_fingerprint=last_edit_fingerprint,
    )
    new_transitions = transitions[len(graph.transitions) :]
    event = "OBLIGATION_GRAPH_CLOSED" if complete_enough(updated) else "OBLIGATION_GRAPH_BLOCKED"
    lifecycle_events: list[Record] = [
        _update_graph_event(updated, "OBLIGATION_GRAPH_UPDATED")
    ]
    if any(_is_regression_transition(item) for item in new_transitions):
        lifecycle_events.append(_update_graph_event(updated, "OBLIGATION_REGRESSION_DETECTED"))
    lifecycle_events.append(_update_graph_event(updated, event))
    updated.events = (
        *graph.events,
        *lifecycle_events,
    )
    return updated


apply_receipts = update_obligation_graph
replay_receipts = update_obligation_graph
update_from_receipts = update_obligation_graph


def complete_enough(graph: ObligationGraph) -> bool:
    """Return true only for a non-empty graph with no mandatory blockers."""

    if not isinstance(graph, ObligationGraph) or not graph.obligations:
        return False
    return not graph.blocker_keys


@dataclass(frozen=True)
class CoverageReceipt:
    """Bounded done-gate evidence with exact blocker IDs in structured form."""

    task_id: str
    graph_digest: str
    complete: bool
    counts: Mapping[str, int]
    blockers: tuple[str, ...] = ()
    blocker_details: tuple[str, ...] = ()
    new_gaps: int = 0
    schema: str = SCHEMA

    @property
    def total(self) -> int:
        return int(self.counts.get("total", 0))

    @property
    def satisfied(self) -> int:
        return int(self.counts.get("satisfied", 0))

    @property
    def not_applicable(self) -> int:
        return int(self.counts.get("not_applicable", 0))

    @property
    def unknown(self) -> int:
        return int(self.counts.get("unknown", 0))

    @property
    def missing(self) -> int:
        return int(self.counts.get("missing", 0))

    def to_dict(self) -> Record:
        return {
            "schema": self.schema,
            "kind": "coverage",
            "task_id": self.task_id,
            "graph_digest": self.graph_digest,
            "complete": self.complete,
            "counts": dict(self.counts),
            "blockers": list(self.blockers),
            "blocker_details": list(self.blocker_details),
            "new_gaps": self.new_gaps,
            "rendered": self.render(),
        }

    as_dict = to_dict

    def render(self, max_chars: int = MAX_COVERAGE_CHARS) -> str:
        """Render refusal-safe text, always bounded by max_chars."""

        limit = max(1, int(max_chars))
        status = "COMPLETE" if self.complete else "BLOCKED"
        counts = (
            f"total={self.total} satisfied={self.satisfied} "
            f"not_applicable={self.not_applicable} unknown={self.unknown} missing={self.missing}"
        )
        if self.new_gaps:
            counts += f" new_gaps={self.new_gaps}"
        prefix = f"COVERAGE {status} task={self.task_id} {counts}"
        if not self.blockers:
            return prefix[:limit]
        blocker_text = "; ".join(self.blocker_details or self.blockers)
        full = f"{prefix} blockers={blocker_text}"
        if len(full) <= limit:
            return full
        available = max(0, limit - len(prefix) - len(" blockers="))
        if available == 0:
            return prefix[:limit]
        return (prefix + " blockers=" + blocker_text[:available]).rstrip()


def _render_blockers(graph: ObligationGraph) -> tuple[str, ...]:
    def action_targets(obligation: Obligation) -> tuple[str, ...]:
        metadata = dict(obligation.metadata)
        if obligation.kind is ObligationKind.UNIT:
            runner_path = metadata.get("runner_path", "")
            return tuple(
                f"{runner_path}::{selector}"
                if runner_path and selector.startswith("test_") and "::" not in selector
                else selector
                for selector in obligation.selectors
            ) or (obligation.key,)
        if metadata.get("granularity") == "file":
            return (metadata.get("oracle_file") or obligation.key,)
        return obligation.selectors or (obligation.key,)

    result: list[str] = []
    for obligation in graph.blockers:
        detail = obligation.key
        if obligation.state is ObligationState.MISSING:
            tail = _clip(obligation.failure_tail, 140)
            detail += f" [MISSING: {tail}]" if tail else " [MISSING]"
        elif obligation.state is ObligationState.NOT_APPLICABLE:
            detail += " [NOT_APPLICABLE: reason required]"
        else:
            targets = ", ".join(action_targets(obligation))
            detail += f" [UNKNOWN: run GREEN on {targets}]"
        result.append(detail)
    result.extend(f"compile:{gap} [UNKNOWN]" for gap in graph.compile_gaps)
    return tuple(result)


def coverage_receipt(graph: ObligationGraph, *, new_gaps: int = 0) -> CoverageReceipt:
    return CoverageReceipt(
        task_id=graph.task_id,
        graph_digest=graph.graph_digest,
        complete=complete_enough(graph),
        counts=graph.counts(),
        blockers=graph.blocker_keys,
        blocker_details=_render_blockers(graph),
        new_gaps=max(0, int(new_gaps)),
    )


build_coverage_receipt = coverage_receipt


def render_coverage_receipt(graph: ObligationGraph, *, new_gaps: int = 0) -> str:
    return coverage_receipt(graph, new_gaps=new_gaps).render()


def done_gate_refusal(graph: ObligationGraph, *, new_gaps: int = 0) -> str:
    """Return the exact bounded refusal payload intended for a done gate."""

    return coverage_receipt(graph, new_gaps=new_gaps).render()


render_done_gate_refusal = done_gate_refusal


@dataclass(frozen=True)
class NewGapReceipt:
    """Stopping-rule evidence for newly discovered blockers in one pass."""

    pass_number: int
    graph_digest: str
    new_gaps: tuple[str, ...]
    total_gaps: int
    complete: bool

    @property
    def new_gap_count(self) -> int:
        return len(self.new_gaps)

    def to_dict(self) -> Record:
        return {
            "schema": SCHEMA,
            "kind": "new-gaps",
            "pass": self.pass_number,
            "graph_digest": self.graph_digest,
            "new_gaps": list(self.new_gaps),
            "new_gap_count": self.new_gap_count,
            "total_gaps": self.total_gaps,
            "complete": self.complete,
        }

    as_dict = to_dict


@dataclass
class NewGapCounter:
    """Track the set delta of blockers across repair passes."""

    previous: tuple[str, ...] = ()
    passes: tuple[NewGapReceipt, ...] = ()

    def observe(self, graph: ObligationGraph) -> NewGapReceipt:
        current = set(graph.blocker_keys)
        previous = set(self.previous)
        receipt = NewGapReceipt(
            pass_number=len(self.passes) + 1,
            graph_digest=graph.graph_digest,
            new_gaps=tuple(sorted(current - previous)),
            total_gaps=len(current),
            complete=complete_enough(graph),
        )
        self.previous = tuple(sorted(current))
        self.passes = (*self.passes, receipt)
        return receipt

    record = observe

    @property
    def latest(self) -> NewGapReceipt | None:
        return self.passes[-1] if self.passes else None

    def to_dict(self) -> Record:
        return {
            "schema": SCHEMA,
            "kind": "new-gaps-counter",
            "previous": list(self.previous),
            "passes": [item.to_dict() for item in self.passes],
        }


def new_gaps_per_pass(graphs: Iterable[ObligationGraph]) -> tuple[NewGapReceipt, ...]:
    counter = NewGapCounter()
    return tuple(counter.observe(graph) for graph in graphs)


def new_gap_keys(
    previous: ObligationGraph | Iterable[str],
    current: ObligationGraph | Iterable[str],
) -> tuple[str, ...]:
    previous_keys = set(
        previous.blocker_keys
        if isinstance(previous, ObligationGraph)
        else (_text(item) for item in previous)
    )
    current_keys = set(
        current.blocker_keys
        if isinstance(current, ObligationGraph)
        else (_text(item) for item in current)
    )
    return tuple(sorted(current_keys - previous_keys))


def count_new_gaps(
    previous: ObligationGraph | Iterable[str],
    current: ObligationGraph | Iterable[str],
) -> int:
    return len(new_gap_keys(previous, current))


@dataclass(frozen=True)
class ObligationCensus:
    """Census result plus the append-only receipt representation."""

    run_id: str
    status: str
    coverage: CoverageReceipt
    new_gaps: int = 0
    receipt: Record = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.coverage.complete

    def to_dict(self) -> Record:
        return {
            "schema": SCHEMA,
            "kind": "obligation-census",
            "run_id": self.run_id,
            "status": self.status,
            "coverage": self.coverage.to_dict(),
            "new_gaps": self.new_gaps,
            "receipt": dict(self.receipt),
        }

    as_dict = to_dict


def census(
    graph: ObligationGraph,
    *,
    run_id: str = "",
    gap_counter: NewGapCounter | None = None,
    receipt_sink: ReceiptSink | None = None,
) -> ObligationCensus:
    """Emit one deterministic obligation census receipt."""

    gap_receipt = gap_counter.observe(graph) if gap_counter is not None else None
    new_gap_count = gap_receipt.new_gap_count if gap_receipt is not None else 0
    coverage = coverage_receipt(graph, new_gaps=new_gap_count)
    effective_run_id = _text(run_id) or f"obligation:{graph.task_id}"
    status = "COMPLETE" if coverage.complete else "BLOCKED"
    receipt: Record = {
        "schema": SCHEMA,
        "kind": "census",
        "intent": "obligation-census",
        "organ_id": ORGAN_ID,
        "event": "OBLIGATION_CENSUS_EMITTED",
        "run_id": effective_run_id,
        "status": status,
        "result_class": status,
        "task_id": graph.task_id,
        "graph_digest": graph.graph_digest,
        "counts": dict(coverage.counts),
        "blockers": list(coverage.blockers),
        "blocker_details": list(coverage.blocker_details),
        "new_gaps": new_gap_count,
        "coverage": coverage.render(),
    }
    if receipt_sink is not None:
        try:
            receipt_sink(receipt)
        except Exception as exc:  # noqa: BLE001 - census must still return evidence
            logger.warning("obligation coverage check sink failed error=%s", type(exc).__name__)
            receipt["sink_error"] = type(exc).__name__
    return ObligationCensus(
        run_id=effective_run_id,
        status=status,
        coverage=coverage,
        new_gaps=new_gap_count,
        receipt=receipt,
    )


obligation_census = census
emit_census = census
run_census = census


def run_liveness_census(graph: ObligationGraph, *, run_id: str = "") -> object:
    """Run the shared liveness census over this organ's graph events."""

    effective_run_id = _text(run_id) or f"obligation:{graph.task_id}"
    return _run_liveness_census(
        Evidence.from_records(receipts=graph.events, run_id=effective_run_id),
        specs=(CENSUS_SPEC,),
        run_id=effective_run_id,
    )


def _census_trigger(evidence: Evidence) -> TriState:
    record = evidence.record
    event = _text(record.get("event", record.get("event_name")))
    organ = _text(record.get("organ_id", record.get("organ")))
    if organ == ORGAN_ID and event:
        return TriState.YES if event in CENSUS_EVENTS else TriState.NO
    if _text(record.get("schema")) == SCHEMA:
        return TriState.UNKNOWN
    return TriState.NO


CENSUS_EVENTS = frozenset(
    {
        "OBLIGATION_GRAPH_COMPILED",
        "OBLIGATION_GRAPH_UPDATED",
        "OBLIGATION_REGRESSION_DETECTED",
        "OBLIGATION_GRAPH_CLOSED",
        "OBLIGATION_GRAPH_BLOCKED",
        "OBLIGATION_CENSUS_EMITTED",
    }
)
CENSUS_SPEC = OrganCensusSpec(
    organ_id=ORGAN_ID,
    trigger=_census_trigger,
    fire_events=frozenset(CENSUS_EVENTS),
    suppression_events=frozenset({"OBLIGATION_GRAPH_NOOP"}),
    dedup_events=frozenset({"OBLIGATION_GRAPH_UPDATED"}),
    challenge_once_events=frozenset({"OBLIGATION_GRAPH_BLOCKED"}),
)
CENSUS_SPECS = (CENSUS_SPEC,)
census_spec = CENSUS_SPEC


__all__ = [
    "CENSUS_EVENTS",
    "CENSUS_SPEC",
    "CENSUS_SPECS",
    "CoverageReceipt",
    "MAX_PENDING_REGRESSION_NOTICES",
    "all_regression_notices",
    "MAX_REGRESSION_NOTICE_CHARS",
    "NewGapCounter",
    "NewGapReceipt",
    "ORGAN_ID",
    "Obligation",
    "ObligationCensus",
    "ObligationGraph",
    "ObligationKind",
    "ObligationState",
    "ObligationStatus",
    "SCHEMA",
    "State",
    "apply_receipts",
    "build_coverage_receipt",
    "build_obligation_graph",
    "census",
    "census_spec",
    "compile_graph",
    "compile_obligations",
    "compile_obligation_graph",
    "complete_enough",
    "convert_receipt",
    "convert_receipts",
    "count_new_gaps",
    "done_gate_refusal",
    "emit_census",
    "load_receipts",
    "new_gap_keys",
    "new_gaps_per_pass",
    "obligation_census",
    "pending_regression_notices",
    "render_coverage_receipt",
    "render_done_gate_refusal",
    "render_regression_notice",
    "reopen_dirty_obligations",
    "replay_receipts",
    "run_census",
    "run_liveness_census",
    "drain_regression_notices",
    "update_obligation_graph",
    "update_from_receipts",
]
