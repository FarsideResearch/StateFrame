"""Pure, receipt-backed pivot planning for the repair kernel.

The organ owns a small contingency graph.  It does not execute edits, reject
actions, or invent approaches: a receipt can only move the organ along the
graph, and a genuinely new receipt fingerprint is the only input allowed to
refresh that graph.  ``render_pivot_advisory`` is deliberately bounded so the
result can be projected into an existing situation-card options row.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import TypeAlias

from kernel_liveness import Evidence, OrganCensusSpec, TriState


SCHEMA = "poc.kernel_pivot/1"
MAX_VISIBLE_OPTIONS = 2
MAX_ADVISORY_CHARS = 400
TIER_PENALTIES = (0, 3, 10, 25)

JSONMapping: TypeAlias = Mapping[str, object]


class PivotError(ValueError):
    """Base error for malformed graph or receipt input."""


class ObjectiveTier(StrEnum):
    """Objective priority, from the primary oracle down to a probe."""

    PRIMARY_ORACLE = "primary_oracle"
    UNIT_GATE = "unit_gate"
    PROBE = "probe"

    PRIMARY = "primary_oracle"
    UNIT = "unit_gate"

    @property
    def rank(self) -> int:
        return {
            ObjectiveTier.PRIMARY_ORACLE: 0,
            ObjectiveTier.UNIT_GATE: 1,
            ObjectiveTier.PROBE: 2,
        }[self]


class PivotTier(IntEnum):
    """Blast-radius tier for one approach."""

    T0 = 0
    T1 = 1
    T2 = 2
    T3 = 3


def _text(value: object, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _canonical(value: object) -> object:
    if isinstance(value, (ObjectiveTier, PivotTier, IntEnum, StrEnum)):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [_canonical(item) for item in value]
        return sorted(
            values, key=lambda item: json.dumps(item, sort_keys=True, default=str)
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    to_dict = getattr(value, "to_dict", None)
    return _canonical(to_dict()) if callable(to_dict) else _text(value)


def canonical_json(value: object) -> str:
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def fingerprint(value: object) -> str:
    """Return a stable, non-sensitive receipt/graph fingerprint."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def derive_approach_id(
    *,
    scope: str,
    objective: ObjectiveTier | str,
    tier: PivotTier | int,
    strategy_class: str,
    seam_or_hop: str,
    target_oracle_atoms: Sequence[str] = (),
) -> str:
    """Derive a line-number-independent content address for an approach."""

    seed = {
        "scope": _text(scope, limit=160),
        "objective": _coerce_objective(objective).value,
        "tier": int(_coerce_pivot_tier(tier)),
        "strategy_class": _text(strategy_class, limit=120),
        "seam_or_hop": _text(seam_or_hop, limit=180),
        "target_oracle_atoms": tuple(
            sorted(
                {_text(item, limit=120) for item in target_oracle_atoms if _text(item)}
            )
        ),
    }
    if not all(seed[key] for key in ("scope", "strategy_class", "seam_or_hop")):
        raise PivotError(
            "scope, strategy_class, and seam_or_hop are required for approach identity"
        )
    return "approach:" + fingerprint(seed).removeprefix("sha256:")[:16]


def _coerce_objective(value: object) -> ObjectiveTier:
    if isinstance(value, ObjectiveTier):
        return value
    normalized = _text(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "primary": ObjectiveTier.PRIMARY_ORACLE,
        "oracle": ObjectiveTier.PRIMARY_ORACLE,
        "primary_oracle": ObjectiveTier.PRIMARY_ORACLE,
        "unit": ObjectiveTier.UNIT_GATE,
        "unit_gate": ObjectiveTier.UNIT_GATE,
        "gate": ObjectiveTier.UNIT_GATE,
        "probe": ObjectiveTier.PROBE,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise PivotError(f"unknown objective tier: {value!r}") from exc


def _coerce_pivot_tier(value: object) -> PivotTier:
    if isinstance(value, PivotTier):
        return value
    if isinstance(value, str):
        value = value.strip().upper().removeprefix("T")
    try:
        return PivotTier(int(value))
    except (TypeError, ValueError) as exc:
        raise PivotError(f"pivot tier must be T0, T1, T2, or T3: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class ApproachNode:
    """One graph node and its ordered fallback IDs."""

    approach_id: str
    label: str
    objective_tier: ObjectiveTier = ObjectiveTier.PRIMARY_ORACLE
    pivot_tier: PivotTier = PivotTier.T1
    fallback_ids: tuple[str, ...] = ()
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()
    demanded_atoms: tuple[str, ...] = ()
    dependencies_ready: bool = True
    coverage: float = 0.0
    cost_so_far: int = 0
    risk: float = 0.0

    def __post_init__(self) -> None:
        approach_id = _text(self.approach_id, limit=128)
        label = _text(self.label, limit=180)
        if not approach_id or not label:
            raise PivotError("approach_id and label are required")
        object.__setattr__(self, "approach_id", approach_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(
            self, "objective_tier", _coerce_objective(self.objective_tier)
        )
        object.__setattr__(self, "pivot_tier", _coerce_pivot_tier(self.pivot_tier))
        object.__setattr__(
            self,
            "fallback_ids",
            tuple(
                dict.fromkeys(
                    _text(item, limit=128) for item in self.fallback_ids if _text(item)
                )
            ),
        )
        object.__setattr__(self, "reason", _text(self.reason, limit=220))
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(
                dict.fromkeys(
                    _text(item, limit=160) for item in self.evidence_refs if _text(item)
                )
            ),
        )
        object.__setattr__(
            self,
            "demanded_atoms",
            tuple(
                dict.fromkeys(
                    _text(item, limit=120)
                    for item in self.demanded_atoms
                    if _text(item)
                )
            ),
        )
        if not isinstance(self.dependencies_ready, bool):
            object.__setattr__(
                self, "dependencies_ready", bool(self.dependencies_ready)
            )
        try:
            coverage = float(self.coverage)
            cost = int(self.cost_so_far)
            risk = float(self.risk)
        except (TypeError, ValueError) as exc:
            raise PivotError("coverage, cost_so_far, and risk must be numeric") from exc
        if not 0.0 <= coverage <= 1.0 or cost < 0 or risk < 0.0:
            raise PivotError("coverage must be 0..1 and cost/risk must be non-negative")
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "cost_so_far", cost)
        object.__setattr__(self, "risk", risk)

    @classmethod
    def from_mapping(cls, value: JSONMapping) -> "ApproachNode":
        fallbacks = value.get("fallback_ids", value.get("fallbacks", ()))
        if isinstance(fallbacks, str):
            fallbacks = (fallbacks,)
        evidence = value.get("evidence_refs", value.get("evidence", ()))
        atoms = value.get("demanded_atoms", value.get("oracle_atoms", ()))
        return cls(
            approach_id=str(value.get("approach_id", value.get("id", ""))),
            label=str(value.get("label", value.get("text", ""))),
            objective_tier=value.get(
                "objective_tier", value.get("objective", ObjectiveTier.PRIMARY_ORACLE)
            ),
            pivot_tier=value.get("pivot_tier", value.get("tier", PivotTier.T1)),
            fallback_ids=tuple(fallbacks) if isinstance(fallbacks, Sequence) else (),
            reason=str(value.get("reason", "")),
            evidence_refs=tuple(evidence)
            if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes))
            else (),
            demanded_atoms=tuple(atoms)
            if isinstance(atoms, Sequence) and not isinstance(atoms, (str, bytes))
            else (),
            dependencies_ready=bool(
                value.get("dependencies_ready", value.get("ready", True))
            ),
            coverage=value.get("coverage", 0.0),
            cost_so_far=value.get("cost_so_far", value.get("cost", 0)),
            risk=value.get("risk", 0.0),
        )

    @property
    def score(self) -> float:
        """Deterministic weight; graph order remains the primary tie-breaker."""

        return round(
            100.0 * self.coverage
            + 10.0 * min(len(self.evidence_refs), 3)
            + 2.0 * min(self.cost_so_far, 10)
            - TIER_PENALTIES[int(self.pivot_tier)]
            - self.risk,
            3,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "approach_id": self.approach_id,
            "label": self.label,
            "objective_tier": self.objective_tier.value,
            "pivot_tier": f"T{int(self.pivot_tier)}",
            "fallback_ids": list(self.fallback_ids),
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "demanded_atoms": list(self.demanded_atoms),
            "dependencies_ready": self.dependencies_ready,
            "coverage": self.coverage,
            "cost_so_far": self.cost_so_far,
            "risk": self.risk,
            "score": self.score,
        }


PivotNode = ApproachNode


@dataclass(frozen=True, slots=True)
class PivotEdge:
    """An ordered, reasoned contingency edge."""

    from_approach: str
    to_approach: str
    order: int
    reason: str = ""
    required_new_info: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.order) < 0:
            raise PivotError("edge order must be non-negative")
        object.__setattr__(self, "from_approach", _text(self.from_approach, limit=128))
        object.__setattr__(self, "to_approach", _text(self.to_approach, limit=128))
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "reason", _text(self.reason, limit=220))
        object.__setattr__(
            self,
            "required_new_info",
            tuple(
                _text(item, limit=160) for item in self.required_new_info if _text(item)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "from_approach": self.from_approach,
            "to_approach": self.to_approach,
            "order": self.order,
            "reason": self.reason,
            "required_new_info": list(self.required_new_info),
        }


@dataclass(frozen=True, slots=True)
class PivotGraph:
    """Immutable pre-graph materialized before receipt processing."""

    nodes: tuple[ApproachNode, ...]
    edges: tuple[PivotEdge, ...] = ()
    roots: Mapping[ObjectiveTier | str, str] = field(default_factory=dict)
    snapshot_id: str = ""
    _node_map: Mapping[str, ApproachNode] = field(init=False, repr=False, compare=False)
    _edge_map: Mapping[str, tuple[PivotEdge, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        raw_nodes = (
            self.nodes.values() if isinstance(self.nodes, Mapping) else self.nodes
        )
        normalized_nodes = tuple(
            node if isinstance(node, ApproachNode) else ApproachNode.from_mapping(node)
            for node in raw_nodes
        )
        by_id: dict[str, ApproachNode] = {}
        for node in normalized_nodes:
            if node.approach_id in by_id:
                raise PivotError(f"duplicate approach_id: {node.approach_id}")
            by_id[node.approach_id] = node
        raw_edges: list[PivotEdge] = list(self.edges)
        for node in normalized_nodes:
            raw_edges.extend(
                PivotEdge(node.approach_id, target, order, node.reason)
                for order, target in enumerate(node.fallback_ids)
            )
        edge_map: defaultdict[str, list[PivotEdge]] = defaultdict(list)
        seen_edges: set[tuple[str, str]] = set()
        for edge in raw_edges:
            if edge.from_approach not in by_id or edge.to_approach not in by_id:
                raise PivotError(
                    f"edge references unknown approach: {edge.from_approach}->{edge.to_approach}"
                )
            key = (edge.from_approach, edge.to_approach)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edge_map[edge.from_approach].append(edge)
        ordered_edges = {
            source: tuple(
                sorted(values, key=lambda edge: (edge.order, edge.to_approach))
            )
            for source, values in edge_map.items()
        }
        for source, values in ordered_edges.items():
            if len({edge.order for edge in values}) != len(values):
                raise PivotError(f"duplicate fallback order from {source}")
        normalized_roots: dict[ObjectiveTier, str] = {}
        raw_roots = self.roots or {}
        for key, value in raw_roots.items():
            objective = _coerce_objective(key)
            if value not in by_id:
                raise PivotError(f"root references unknown approach: {value}")
            normalized_roots[objective] = str(value)
        for objective in ObjectiveTier:
            if objective not in normalized_roots:
                candidates = [
                    node.approach_id
                    for node in normalized_nodes
                    if node.objective_tier is objective
                ]
                if candidates:
                    normalized_roots[objective] = sorted(candidates)[0]
        snapshot = self.snapshot_id or fingerprint(
            {
                "nodes": [node.to_dict() for node in normalized_nodes],
                "edges": [
                    edge.to_dict()
                    for values in ordered_edges.values()
                    for edge in values
                ],
            }
        )
        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(
            self,
            "edges",
            tuple(edge for values in ordered_edges.values() for edge in values),
        )
        object.__setattr__(self, "roots", MappingProxyType(normalized_roots))
        object.__setattr__(self, "snapshot_id", snapshot)
        object.__setattr__(self, "_node_map", MappingProxyType(by_id))
        object.__setattr__(self, "_edge_map", MappingProxyType(ordered_edges))

    @classmethod
    def declare(
        cls,
        *,
        primary_oracle: Sequence[ApproachNode],
        unit_gate: Sequence[ApproachNode] = (),
        probe: Sequence[ApproachNode] = (),
    ) -> "PivotGraph":
        """Materialize all objective-tier roots in one upfront declaration."""

        groups = (
            (ObjectiveTier.PRIMARY_ORACLE, tuple(primary_oracle)),
            (ObjectiveTier.UNIT_GATE, tuple(unit_gate)),
            (ObjectiveTier.PROBE, tuple(probe)),
        )
        nodes: list[ApproachNode] = []
        roots: dict[ObjectiveTier, str] = {}
        for objective, group in groups:
            if group:
                roots[objective] = group[0].approach_id
            nodes.extend(group)
        return cls(tuple(nodes), roots=roots)

    def node(self, approach_id: str) -> ApproachNode:
        try:
            return self._node_map[approach_id]
        except KeyError as exc:
            raise PivotError(f"unknown approach: {approach_id}") from exc

    def root(
        self, objective: ObjectiveTier | str = ObjectiveTier.PRIMARY_ORACLE
    ) -> ApproachNode:
        objective = _coerce_objective(objective)
        try:
            return self.node(self.roots[objective])
        except KeyError as exc:
            raise PivotError(
                f"no graph root for objective tier: {objective.value}"
            ) from exc

    def outgoing(self, approach_id: str) -> tuple[PivotEdge, ...]:
        self.node(approach_id)
        return self._edge_map.get(approach_id, ())

    def reachable(self, approach_id: str) -> tuple[ApproachNode, ...]:
        """Return reachable nodes in pre-graph order, cycle-safe."""

        seen: set[str] = {approach_id}
        result: list[ApproachNode] = []
        frontier = [edge.to_approach for edge in self.outgoing(approach_id)]
        while frontier:
            current = frontier.pop(0)
            if current in seen:
                continue
            seen.add(current)
            result.append(self.node(current))
            frontier.extend(edge.to_approach for edge in self.outgoing(current))
        return tuple(result)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "snapshot_id": self.snapshot_id,
            "roots": {key.value: value for key, value in self.roots.items()},
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @property
    def pregraph_digest(self) -> str:
        return self.snapshot_id


@dataclass(frozen=True, slots=True)
class PivotPolicy:
    """Evidence bars and output bounds for the organ."""

    intra_objective_failures: int = 1
    abandon_objective_failures: int = 2
    max_visible_options: int = MAX_VISIBLE_OPTIONS

    def __post_init__(self) -> None:
        for name in (
            "intra_objective_failures",
            "abandon_objective_failures",
            "max_visible_options",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise PivotError(f"{name} must be a positive integer")
        if int(self.abandon_objective_failures) <= int(self.intra_objective_failures):
            raise PivotError(
                "abandon_objective_failures must exceed intra_objective_failures"
            )
        object.__setattr__(
            self, "intra_objective_failures", int(self.intra_objective_failures)
        )
        object.__setattr__(
            self, "abandon_objective_failures", int(self.abandon_objective_failures)
        )
        object.__setattr__(
            self,
            "max_visible_options",
            min(int(self.max_visible_options), MAX_VISIBLE_OPTIONS),
        )


@dataclass(frozen=True, slots=True)
class PivotReceipt:
    """Minimum safe receipt accepted by the pivot organ."""

    fingerprint: str
    result_class: str
    approach_id: str = ""
    turn: int = 0
    kind: str = ""
    receipt_id: str = ""
    detail: str = ""
    new_information: bool = True

    @classmethod
    def from_mapping(
        cls, value: JSONMapping, *, default_approach: str = ""
    ) -> "PivotReceipt":
        if not isinstance(value, Mapping):
            raise PivotError("pivot receipt must be a mapping")
        result = _text(
            value.get("result_class", value.get("result", "")), limit=80
        ).upper()
        kind = _text(value.get("kind", ""), limit=80).lower()
        approach = _text(
            value.get("approach_id", value.get("active_approach_id", default_approach)),
            limit=128,
        )
        raw_fingerprint = _text(value.get("fingerprint", ""), limit=180)
        if not raw_fingerprint:
            raw_fingerprint = fingerprint(
                {
                    "kind": kind,
                    "result_class": result,
                    "approach_id": approach,
                    "intent": value.get("intent", ""),
                    "detail": value.get("detail", ""),
                }
            )
        try:
            turn = int(value.get("turn", value.get("step", 0)))
        except (TypeError, ValueError) as exc:
            raise PivotError("receipt turn must be an integer") from exc
        if turn < 0:
            raise PivotError("receipt turn must be non-negative")
        stable = {
            "fingerprint": raw_fingerprint,
            "result_class": result,
            "approach_id": approach,
            "turn": turn,
            "kind": kind,
        }
        return cls(
            fingerprint=raw_fingerprint,
            result_class=result,
            approach_id=approach,
            turn=turn,
            kind=kind,
            receipt_id=_text(value.get("receipt_id", ""), limit=180)
            or "receipt:" + fingerprint(stable).removeprefix("sha256:")[:24],
            detail=_text(value.get("detail", value.get("intent", "")), limit=200),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "kind": self.kind or "receipt",
            "receipt_id": self.receipt_id,
            "fingerprint": self.fingerprint,
            "result_class": self.result_class,
            "approach_id": self.approach_id,
            "turn": self.turn,
            "detail": self.detail,
            "new_information": self.new_information,
        }


AttemptReceipt = PivotReceipt


@dataclass(frozen=True, slots=True)
class PivotAdvisory:
    """Bounded model-facing nudge describing the next graph edge."""

    trigger: str
    active_approach: ApproachNode
    next_approach: ApproachNode
    chain: tuple[ApproachNode, ...]
    fingerprint: str
    required_failures: int
    distinct_failures: tuple[str, ...]
    banned_count: int
    reason: str = ""

    @property
    def next_approach_id(self) -> str:
        return self.next_approach.approach_id

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "trigger": self.trigger,
            "active_approach_id": self.active_approach.approach_id,
            "next_approach_id": self.next_approach.approach_id,
            "chain": [node.approach_id for node in self.chain],
            "fingerprint": self.fingerprint,
            "required_failures": self.required_failures,
            "distinct_failures": list(self.distinct_failures),
            "banned_count": self.banned_count,
            "reason": self.reason,
        }

    def render(self) -> str:
        return render_pivot_advisory(self)


@dataclass(frozen=True, slots=True)
class PivotObservation:
    receipt: PivotReceipt
    new_information: bool
    banned: bool
    advisory: PivotAdvisory | None
    event: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "event": self.event,
            "new_information": self.new_information,
            "banned": self.banned,
            "receipt": self.receipt.to_dict(),
            "advisory": self.advisory.to_dict() if self.advisory else None,
        }


@dataclass(frozen=True, slots=True)
class PivotCensus:
    fired: int
    suppressed: int
    banned: int
    digest: str
    events: tuple[Mapping[str, object], ...] = ()

    @property
    def N(self) -> int:
        return self.fired

    @property
    def M(self) -> int:
        return self.fired

    @property
    def S(self) -> int:
        return self.suppressed + self.banned

    @property
    def D(self) -> int:
        return self.banned

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "organ_id": "bolt.pivot",
            "fired": self.fired,
            "suppressed": self.suppressed,
            "banned": self.banned,
            "N": self.N,
            "M": self.M,
            "S": self.S,
            "D": self.D,
            "digest": self.digest,
        }


_FAILURE_RESULTS = frozenset(
    {"RED", "PRODUCT_RED", "FAILED", "FAILURE", "ERROR", "NO_HITS", "BLOCKED"}
)
_GREEN_RESULTS = frozenset({"GREEN", "PRODUCT_GREEN"})
_NON_PRODUCT_RESULTS = frozenset(
    {
        "NO_VERDICT",
        "SELECTOR_ERROR",
        "INFRA",
        "INFRA_FAILURE",
        "ENVIRONMENT",
        "UNKNOWN",
        "NO_TEST",
        "NO_TESTS",
        "NO_OP",
        "REJECTED",
    }
)


def _pivot_trigger(evidence: Evidence) -> TriState:
    event = _text(evidence.event.get("event", "")).upper()
    if event == "PIVOT_TRIGGERED":
        return TriState.YES
    if event in {
        "PIVOT_SUPPRESSED",
        "PIVOT_DEDUPED",
        "PIVOT_BANNED",
        "NO_LIVE_ALTERNATIVE",
    }:
        return TriState.NO
    return TriState.UNKNOWN if not evidence.complete else TriState.NO


CENSUS_SPEC = OrganCensusSpec(
    organ_id="bolt.pivot",
    trigger=_pivot_trigger,
    fire_events=frozenset(
        {"PIVOT_TRIGGERED", "PIVOT_ADVISORY_DELIVERED", "pivot_triggered"}
    ),
    suppression_events=frozenset(
        {"PIVOT_SUPPRESSED", "NO_LIVE_ALTERNATIVE", "pivot_suppressed"}
    ),
    dedup_events=frozenset(
        {"PIVOT_DEDUPED", "PREGRAPH_REORDER_UNCHANGED", "pivot_deduped"}
    ),
    challenge_once_events=frozenset(
        {"PIVOT_CHALLENGE_ONCE", "PIVOT_BANNED", "pivot_challenge_once"}
    ),
)
CENSUS_SPECS = (CENSUS_SPEC,)


class PivotOrgan:
    """Receipt reducer implementing pre-graph traversal and loop prevention."""

    def __init__(
        self,
        graph: PivotGraph,
        *,
        policy: PivotPolicy | None = None,
        active_approach_id: str | None = None,
    ) -> None:
        self.graph = graph
        self.policy = policy or PivotPolicy()
        self.active_approach_id = active_approach_id or graph.root().approach_id
        self.graph.node(self.active_approach_id)
        self._seen_fingerprints: set[str] = set()
        self._visited: set[tuple[str, str]] = set()
        self._banned: set[tuple[str, str]] = set()
        self._failed: defaultdict[str, set[str]] = defaultdict(set)
        self._emitted_pairs: set[tuple[str, str, str]] = set()
        self._events: list[dict[str, object]] = []
        self._counts: Counter[str] = Counter()
        self._last_receipt: PivotReceipt | None = None
        self._last_observation: PivotObservation | None = None
        self._graph_refreshes = 0
        self._emit(
            "PREGRAPH_MATERIALIZED",
            detail={"snapshot_id": graph.snapshot_id, "nodes": len(graph.nodes)},
        )

    @property
    def events(self) -> tuple[Mapping[str, object], ...]:
        return tuple(dict(event) for event in self._events)

    @property
    def visited(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._visited)

    @property
    def banned(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._banned)

    def _emit(
        self,
        event: str,
        *,
        receipt: PivotReceipt | None = None,
        approach_id: str = "",
        detail: object = None,
    ) -> None:
        payload: dict[str, object] = {
            "schema": SCHEMA,
            "kind": "pivot",
            "organ": "bolt.pivot",
            "event": event,
            "approach_id": approach_id
            or (receipt.approach_id if receipt else self.active_approach_id),
            "fingerprint": receipt.fingerprint if receipt else "",
            "turn": receipt.turn if receipt else 0,
            "detail": _canonical(detail if detail is not None else {}),
        }
        payload["receipt_id"] = (
            "pivot:" + fingerprint(payload).removeprefix("sha256:")[:24]
        )
        self._events.append(payload)

    def _candidate_chain(
        self, active_id: str, receipt_fingerprint: str
    ) -> tuple[ApproachNode, ...]:
        result: list[ApproachNode] = []
        seen: set[str] = {active_id}
        pending = [edge.to_approach for edge in self.graph.outgoing(active_id)]
        while pending and len(result) < self.policy.max_visible_options:
            approach_id = pending.pop(0)
            if approach_id in seen:
                continue
            seen.add(approach_id)
            node = self.graph.node(approach_id)
            if self._eligible(node, active_id, receipt_fingerprint):
                result.append(node)
            pending.extend(
                edge.to_approach for edge in self.graph.outgoing(approach_id)
            )
        return tuple(result)

    def _eligible(
        self, node: ApproachNode, active_id: str, receipt_fingerprint: str
    ) -> bool:
        if (
            not node.dependencies_ready
            or (node.approach_id, receipt_fingerprint) in self._visited
        ):
            return False
        active = self.graph.node(active_id)
        required = self.required_failures(active, node)
        if len(self._failed[active_id]) < required:
            return False
        if node.pivot_tier is PivotTier.T2 and not self._lower_tiers_exhausted(
            active_id
        ):
            return False
        if node.pivot_tier is PivotTier.T3 and not self._lower_tiers_exhausted(
            active_id, include_t2=True
        ):
            return False
        return True

    def _lower_tiers_exhausted(
        self, active_id: str, *, include_t2: bool = False
    ) -> bool:
        active = self.graph.node(active_id)
        limit = 3 if include_t2 else 2
        lower = [
            node
            for node in self.graph.reachable(active_id)
            if node.objective_tier is active.objective_tier
            and int(node.pivot_tier) < limit
        ]
        return not lower or all(
            node.approach_id in {item[0] for item in self._visited} for node in lower
        )

    def required_failures(
        self, active: ApproachNode | str, target: ApproachNode | str
    ) -> int:
        active_node = self.graph.node(active) if isinstance(active, str) else active
        target_node = self.graph.node(target) if isinstance(target, str) else target
        return (
            self.policy.abandon_objective_failures
            if target_node.objective_tier.rank > active_node.objective_tier.rank
            else self.policy.intra_objective_failures
        )

    def _next_advisory(
        self, receipt: PivotReceipt, *, trigger: str, force: bool = False
    ) -> PivotAdvisory | None:
        active = self.graph.node(receipt.approach_id)
        candidates = self._candidate_chain(active.approach_id, receipt.fingerprint)
        if not candidates:
            self._emit(
                "NO_LIVE_ALTERNATIVE", receipt=receipt, detail={"trigger": trigger}
            )
            return None
        next_node = candidates[0]
        pair = (active.approach_id, receipt.fingerprint, next_node.approach_id)
        if pair in self._emitted_pairs:
            self._emit(
                "PIVOT_DEDUPED", receipt=receipt, detail={"next": next_node.approach_id}
            )
            return None
        self._emitted_pairs.add(pair)
        failures = tuple(sorted(self._failed[active.approach_id]))
        advisory = PivotAdvisory(
            trigger=trigger,
            active_approach=active,
            next_approach=next_node,
            chain=(active, *candidates),
            fingerprint=receipt.fingerprint,
            required_failures=self.required_failures(active, next_node),
            distinct_failures=failures,
            banned_count=self._counts["banned"],
            reason=next_node.reason,
        )
        self._counts["fired"] += 1
        self._emit(
            "PIVOT_TRIGGERED",
            receipt=receipt,
            detail={"trigger": trigger, "next": next_node.approach_id, "force": force},
        )
        self._emit(
            "PIVOT_ADVISORY_DELIVERED",
            receipt=receipt,
            detail={"advisory_digest": fingerprint(advisory.to_dict())},
        )
        return advisory

    def observe(
        self, value: PivotReceipt | JSONMapping, *, approach_id: str | None = None
    ) -> PivotObservation:
        default = approach_id or self.active_approach_id
        receipt = (
            value
            if isinstance(value, PivotReceipt)
            else PivotReceipt.from_mapping(value, default_approach=default)
        )
        if approach_id:
            receipt = PivotReceipt(
                fingerprint=receipt.fingerprint,
                result_class=receipt.result_class,
                approach_id=approach_id,
                turn=receipt.turn,
                kind=receipt.kind,
                receipt_id=receipt.receipt_id,
                detail=receipt.detail,
                new_information=receipt.new_information,
            )
        self.graph.node(receipt.approach_id)
        self.active_approach_id = receipt.approach_id
        new_information = receipt.fingerprint not in self._seen_fingerprints
        if new_information:
            self._seen_fingerprints.add(receipt.fingerprint)
        pair = (receipt.approach_id, receipt.fingerprint)
        banned = pair in self._visited
        if banned:
            self._banned.add(pair)
            self._counts["banned"] += 1
            self._emit(
                "PIVOT_BANNED",
                receipt=receipt,
                detail={"reason": "approach already visited for fingerprint"},
            )
        else:
            self._visited.add(pair)
        if receipt.result_class in _FAILURE_RESULTS or banned:
            self._failed[receipt.approach_id].add(receipt.fingerprint)
        self._last_receipt = PivotReceipt(
            fingerprint=receipt.fingerprint,
            result_class=receipt.result_class,
            approach_id=receipt.approach_id,
            turn=receipt.turn,
            kind=receipt.kind,
            receipt_id=receipt.receipt_id,
            detail=receipt.detail,
            new_information=new_information,
        )
        advisory: PivotAdvisory | None = None
        trigger = ""
        first_observation = len(self._visited) == 1 and not banned
        if receipt.result_class in _GREEN_RESULTS:
            trigger = "GREEN_PROGRESS"
        elif banned:
            trigger = "LOOP_BAN"
            advisory = self._next_advisory(receipt, trigger=trigger, force=True)
        elif receipt.result_class in _NON_PRODUCT_RESULTS:
            trigger = "SUPPRESSED_EVIDENCE"
        else:
            trigger = "FAILURE_OBSERVED"
            if not first_observation:
                advisory = self._next_advisory(receipt, trigger=trigger)
        if advisory is None and trigger not in {"GREEN_PROGRESS"}:
            self._counts["suppressed"] += 1
            self._emit("PIVOT_SUPPRESSED", receipt=receipt, detail={"trigger": trigger})
        event = (
            "PIVOT_BANNED"
            if banned
            else ("PIVOT_TRIGGERED" if advisory else "PIVOT_SUPPRESSED")
        )
        observation = PivotObservation(
            self._last_receipt, new_information, banned, advisory, event
        )
        self._last_observation = observation
        return observation

    ingest_receipt = observe
    on_receipt = observe
    record_receipt = observe

    def activate(
        self, approach_id: str, *, fingerprint_value: str | None = None
    ) -> ApproachNode:
        node = self.graph.node(approach_id)
        if (
            fingerprint_value is not None
            and (approach_id, fingerprint_value) in self._visited
        ):
            self._banned.add((approach_id, fingerprint_value))
            self._counts["banned"] += 1
            self._emit(
                "PIVOT_BANNED",
                approach_id=approach_id,
                detail={
                    "reason": "activation repeats fingerprint",
                    "fingerprint": fingerprint_value,
                },
            )
        self.active_approach_id = approach_id
        return node

    select_approach = activate

    def refresh_graph(
        self, graph: PivotGraph, receipt: PivotReceipt | JSONMapping
    ) -> bool:
        """Accept a graph change only when the receipt adds new information."""

        incoming = (
            receipt
            if isinstance(receipt, PivotReceipt)
            else PivotReceipt.from_mapping(
                receipt, default_approach=self.active_approach_id
            )
        )
        if incoming.fingerprint in self._seen_fingerprints:
            self._emit(
                "PREGRAPH_REORDER_UNCHANGED",
                receipt=incoming,
                detail={"snapshot_id": self.graph.snapshot_id},
            )
            self._counts["suppressed"] += 1
            return False
        self._seen_fingerprints.add(incoming.fingerprint)
        graph.node(self.active_approach_id)
        self.graph = graph
        self._graph_refreshes += 1
        self._emit(
            "PREGRAPH_REFRESHED",
            receipt=incoming,
            detail={"snapshot_id": graph.snapshot_id},
        )
        return True

    def advisory(self) -> PivotAdvisory | None:
        """Return the last newly emitted card; repeated reads are suppressed."""

        return self._last_observation.advisory if self._last_observation else None

    def census(self) -> PivotCensus:
        payload = {
            "organ_id": "bolt.pivot",
            "fired": self._counts["fired"],
            "suppressed": self._counts["suppressed"],
            "banned": self._counts["banned"],
            "graph_refreshes": self._graph_refreshes,
        }
        return PivotCensus(
            fired=self._counts["fired"],
            suppressed=self._counts["suppressed"],
            banned=self._counts["banned"],
            digest=fingerprint(payload),
            events=self.events,
        )


def render_pivot_advisory(advisory: PivotAdvisory | None) -> str:
    """Render a compact nudge chain, never exceeding 400 characters."""

    if advisory is None:
        return ""
    chain = " -> ".join(
        f"{node.approach_id}:{_text(node.label, limit=48)}"
        for node in advisory.chain[:3]
    )
    evidence = ",".join(item[:12] for item in advisory.distinct_failures[:2]) or "none"
    text = (
        f"PIVOT {advisory.trigger}: {chain}. Next={advisory.next_approach.approach_id} "
        f"[T{int(advisory.next_approach.pivot_tier)}/{advisory.next_approach.objective_tier.value}]. "
        f"Evidence={evidence}; need={advisory.required_failures}; banned={advisory.banned_count}. "
        "Try next graph edge; justify any continuation."
    )
    return (
        text
        if len(text) <= MAX_ADVISORY_CHARS
        else text[: MAX_ADVISORY_CHARS - 1] + "…"
    )


def render_advisory(advisory: PivotAdvisory | None) -> str:
    """Compatibility alias for bounded advisory renderers."""

    return render_pivot_advisory(advisory)


__all__ = [
    "ApproachNode",
    "AttemptReceipt",
    "CENSUS_SPEC",
    "CENSUS_SPECS",
    "MAX_ADVISORY_CHARS",
    "MAX_VISIBLE_OPTIONS",
    "ObjectiveTier",
    "PivotAdvisory",
    "PivotCensus",
    "PivotEdge",
    "PivotError",
    "PivotGraph",
    "PivotNode",
    "PivotObservation",
    "PivotOrgan",
    "PivotPolicy",
    "PivotReceipt",
    "PivotTier",
    "SCHEMA",
    "canonical_json",
    "derive_approach_id",
    "fingerprint",
    "render_advisory",
    "render_pivot_advisory",
]
