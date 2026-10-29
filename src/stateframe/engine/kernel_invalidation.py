"""Receipt-backed dependency invalidation for the single-task bench.

The index is the live dependency graph for the loop.  Receipts are source
observations, edits invalidate indexed dependants, and the completion gate
admits only clean evidence.  Inputs remain mapping-shaped so old
``kernel_bolt`` receipts and newer typed records can share one index.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from kernel_liveness import Evidence, OrganCensusSpec, TriState, run_census


SCHEMA = "poc.kernel_invalidation/1"
ORGAN_ID = "kernel_invalidation"
MAX_NOTICE_CHARS = 240
MAX_DIRTY_NOTICES_PER_TURN = 2
REASON_SOURCE_EDIT = "source_edit"
REASON_CALL_GRAPH_DEPENDENCY = "call_graph_dependency"
REASON_EVIDENCE_STALE = "evidence_stale"
REASON_REOBSERVED = "fresh_reobservation"
COMPLETION_BLOCK_REASON = "STALE_EVIDENCE"

Record: TypeAlias = Mapping[str, object]
ReceiptSink: TypeAlias = Callable[[Mapping[str, object]], object]


class NodeKind(StrEnum):
    """Record families projected into the dependency index."""

    OBSERVATION = "observation"
    CLAIM = "claim"
    ADVISORY = "advisory"


class NodeState(StrEnum):
    """A dirty node needs fresh support; it is not thereby contradicted."""

    CLEAN = "CLEAN"
    DIRTY = "DIRTY"


@dataclass(frozen=True, slots=True)
class DependencyGeneration:
    """One language-neutral catalog dependency generation."""

    dependency_id: str
    generation: str
    kind: str = "catalog"

    def __post_init__(self) -> None:
        dependency_id = _text(self.dependency_id)
        generation = _text(self.generation)
        kind = _text(self.kind) or "catalog"
        if not dependency_id:
            raise ValueError("dependency_id is required")
        if not generation:
            raise ValueError("dependency generation is required")
        object.__setattr__(self, "dependency_id", dependency_id[:256])
        object.__setattr__(self, "generation", generation[:256])
        object.__setattr__(self, "kind", kind[:80])

    def to_dict(self) -> dict[str, str]:
        return {
            "dependency_id": self.dependency_id,
            "generation": self.generation,
            "kind": self.kind,
        }


def _text(value: object) -> str:
    return str(value or "").strip()


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path(value: object) -> str:
    result = _text(value).replace("\\", "/")
    result = re.sub(r"^[./]+", "", result)
    return result.rstrip(":")


_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9_]+"
)


def _paths(value: object) -> tuple[str, ...]:
    found: list[str] = []
    for match in _PATH_RE.finditer(_text(value)):
        candidate = _path(match.group(0))
        if candidate and candidate not in found:
            found.append(candidate)
    return tuple(found)


def _same_path(left: str, right: str, *, exact: bool = False) -> bool:
    # Workspace receipt identities preserve case, including on Linux workers.
    a, b = _path(left), _path(right)
    return bool(a and b and (a == b or (not exact and (a.endswith("/" + b) or b.endswith("/" + a)))))


def _receipt_id(record: Record) -> str:
    explicit = _text(record.get("receipt_id") or record.get("id"))
    if explicit:
        return explicit[:160]
    fingerprint = _text(record.get("fingerprint"))
    if fingerprint:
        return f"receipt:{fingerprint}"
    return "receipt:" + _digest(
        {
            "kind": _text(record.get("kind")),
            "turn": _int(record.get("turn")),
            "intent": _text(record.get("intent")),
        }
    )[:24]


def _revision(record: Record) -> int:
    return max(0, _int(record.get("workspace_revision", record.get("revision"))))


def _record_value(record: object) -> Mapping[str, object] | None:
    if isinstance(record, Mapping):
        return record
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return value if isinstance(value, Mapping) else None
    return None


def _is_test_path(path: str) -> bool:
    normalized = _path(path)
    parts = normalized.split("/")
    name = parts[-1] if parts else ""
    return name.startswith("test_") or name.endswith("_test.py") or "tests" in parts or "probes" in parts


def _is_check_observation(node: object) -> bool:
    """Only explicit check observations can veto completion.

    Source reads remain useful audit history after an edit, but they are not a
    failed verification.  Test receipts opt into this class through
    ``source_kind=\"check\"``; the legacy wording fallback keeps persisted
    green-test records compatible without broadening ordinary source reads.
    """
    source_kind = _text(getattr(node, "source_kind", "")).casefold()
    if source_kind in {"check", "test", "public_check", "verification"}:
        return True
    what = _text(getattr(node, "what", "")).casefold()
    return "test receipt" in what or what.startswith("check:")


def _result(record: Record) -> str:
    return _text(record.get("result_class") or record.get("result")).upper()


def _record_paths(record: Record, *, prefer_anchor: bool = False) -> tuple[str, ...]:
    if not prefer_anchor and "target_path" in record:
        # Executor identity must not be replaced or broadened by display prose.
        target = record["target_path"]
        path = _path(target) if isinstance(target, str) else ""
        return (path,) if path else ()
    fields = (
        ("anchor", "traceback", "traceback_anchor")
        if prefer_anchor
        else (
            "path",
            "target_path",
            "file",
            "file_path",
            "target",
            "intent",
            "anchor",
            "detail",
            "text",
            "line",
            "source_paths",
        )
    )
    found: list[str] = []
    for field_name in fields:
        for candidate in _paths(record.get(field_name)):
            if candidate not in found:
                found.append(candidate)
    return tuple(found)


def _as_strings(value: object) -> tuple[str, ...]:
    values: list[str] = []
    for item in _sequence(value):
        if isinstance(item, Mapping):
            item_value = item.get("id") or item.get("node_id") or item.get("path")
        else:
            item_value = item
        normalized = _text(item_value)
        if normalized and normalized not in values:
            values.append(normalized)
    return tuple(values)


def _catalog_dependency_values(value: object) -> tuple[DependencyGeneration, ...]:
    """Normalize CATC0/C1 dependency records without importing their owners."""

    if isinstance(value, Mapping):
        items: list[object] = []
        for dependency_id, generation in value.items():
            if isinstance(generation, Mapping):
                item = dict(generation)
                item.setdefault("dependency_id", dependency_id)
            else:
                item = {"dependency_id": dependency_id, "generation": generation}
            items.append(item)
    else:
        items = list(_sequence(value))

    result: list[DependencyGeneration] = []
    for item in items:
        if isinstance(item, DependencyGeneration):
            dependency = item
        else:
            if isinstance(item, Mapping):
                dependency_id: object = item.get("dependency_id") or item.get("entity_id") or item.get("id")
                entity = item.get("entity")
                if not dependency_id and isinstance(entity, Mapping):
                    dependency_id = entity.get("entity_id") or entity.get("id")
                generation: object = (
                    item.get("generation")
                    or item.get("revision")
                    or item.get("revision_or_digest")
                    or item.get("digest")
                )
                kind = item.get("kind", "catalog")
            else:
                dependency_id = getattr(item, "dependency_id", "") or getattr(item, "entity_id", "")
                entity = getattr(item, "entity", None)
                if not dependency_id:
                    dependency_id = getattr(entity, "entity_id", "")
                generation = (
                    getattr(item, "generation", "")
                    or getattr(item, "revision", "")
                    or getattr(item, "revision_or_digest", "")
                    or getattr(item, "digest", "")
                )
                kind = getattr(item, "kind", "catalog")
            dependency = DependencyGeneration(str(dependency_id or ""), str(generation or ""), str(kind or "catalog"))
        if dependency not in result:
            result.append(dependency)
    return tuple(result)


def _dependency_aliases(value: object) -> tuple[str, ...]:
    """Return identity aliases used by catalog entities and file observations."""

    raw = _text(value).replace("\\", "/")
    if not raw:
        return ()
    aliases = [raw]
    for marker in (":file:", "file:"):
        if marker in raw:
            suffix = raw.rsplit(marker, 1)[-1]
            if suffix:
                aliases.append(_path(suffix))
    normalized = _path(raw)
    if normalized:
        aliases.append(normalized)
    return tuple(dict.fromkeys(item for item in aliases if item))


def _dependency_matches_path(dependency_id: str, path: str) -> bool:
    wanted = set(_dependency_aliases(path))
    return bool(wanted.intersection(_dependency_aliases(dependency_id)))


@dataclass(frozen=True)
class ObservationNode:
    """One source observation keyed by normalized path and workspace revision."""

    path: str
    workspace_revision: int
    what: str = ""
    evidence_refs: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    digest: str = ""
    source_kind: str = "source"
    node_id: str = ""
    state: NodeState = NodeState.CLEAN
    dirty_turn: int | None = None
    dirty_reason: str = ""

    def __post_init__(self) -> None:
        normalized_path = _path(self.path)
        if not normalized_path:
            raise ValueError("observation path is required")
        revision = max(0, _int(self.workspace_revision))
        refs = tuple(dict.fromkeys(_text(item) for item in self.evidence_refs if _text(item)))
        dependencies = tuple(
            dict.fromkeys(_path(item) for item in self.dependencies if _path(item))
        )
        node_id = self.node_id or f"observation:{normalized_path}@{revision}"
        object.__setattr__(self, "path", normalized_path)
        object.__setattr__(self, "workspace_revision", revision)
        object.__setattr__(self, "what", _text(self.what)[:180] or normalized_path)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "digest", _text(self.digest))
        object.__setattr__(self, "source_kind", _text(self.source_kind) or "source")
        object.__setattr__(self, "node_id", _text(node_id))

    @property
    def dirty(self) -> bool:
        return self.state is NodeState.DIRTY

    @property
    def source_root_id(self) -> str:
        return "source:" + _digest(
            {
                "kind": self.source_kind,
                "path": self.path,
                "revision": self.workspace_revision,
                "digest": self.digest,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": NodeKind.OBSERVATION.value,
            "path": self.path,
            "workspace_revision": self.workspace_revision,
            "what": self.what,
            "evidence_refs": list(self.evidence_refs),
            "dependencies": list(self.dependencies),
            "digest": self.digest,
            "source_kind": self.source_kind,
            "source_root_id": self.source_root_id,
            "state": self.state.value,
            "dirty_turn": self.dirty_turn,
            "dirty_reason": self.dirty_reason,
        }


@dataclass(frozen=True)
class ClaimNode:
    """An explicit hypothesis or acceptance atom supported by evidence refs."""

    claim_id: str
    what: str = ""
    evidence_refs: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    workspace_revision: int = 0
    state: str = "SUPPORTED"
    node_state: NodeState = NodeState.CLEAN
    dirty_turn: int | None = None
    dirty_reason: str = ""
    prior_state: str = "SUPPORTED"
    dependency_generations: tuple[DependencyGeneration, ...] = ()

    def __post_init__(self) -> None:
        claim_id = _text(self.claim_id)
        if not claim_id:
            raise ValueError("claim_id is required")
        paths = tuple(dict.fromkeys(_path(item) for item in self.paths if _path(item)))
        refs = tuple(dict.fromkeys(_text(item) for item in self.evidence_refs if _text(item)))
        state = _text(self.state) or "SUPPORTED"
        object.__setattr__(self, "claim_id", claim_id[:180])
        object.__setattr__(self, "what", _text(self.what)[:180] or claim_id[:180])
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "workspace_revision", max(0, _int(self.workspace_revision)))
        object.__setattr__(self, "state", state[:80])
        object.__setattr__(self, "prior_state", _text(self.prior_state) or state)
        object.__setattr__(self, "dependency_generations", _catalog_dependency_values(self.dependency_generations))

    @property
    def node_id(self) -> str:
        return f"claim:{self.claim_id}"

    @property
    def dirty(self) -> bool:
        return self.node_state is NodeState.DIRTY

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": NodeKind.CLAIM.value,
            "claim_id": self.claim_id,
            "what": self.what,
            "evidence_refs": list(self.evidence_refs),
            "paths": list(self.paths),
            "workspace_revision": self.workspace_revision,
            "state": self.state,
            "node_state": self.node_state.value,
            "dirty_turn": self.dirty_turn,
            "dirty_reason": self.dirty_reason,
            "dependency_generations": [item.to_dict() for item in self.dependency_generations],
        }


@dataclass(frozen=True)
class AdvisoryNode:
    """A delivered organ advisory and the paths/claims it named."""

    advisory_id: str
    text: str
    targets: tuple[str, ...] = ()
    turn: int = 0
    delivered: bool = True
    node_state: NodeState = NodeState.CLEAN
    dirty_turn: int | None = None
    dirty_reason: str = ""

    def __post_init__(self) -> None:
        advisory_id = _text(self.advisory_id)
        if not advisory_id:
            raise ValueError("advisory_id is required")
        targets = tuple(dict.fromkeys(_text(item) for item in self.targets if _text(item)))
        object.__setattr__(self, "advisory_id", advisory_id[:180])
        object.__setattr__(self, "text", _text(self.text)[:300])
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "turn", max(0, _int(self.turn)))

    @property
    def node_id(self) -> str:
        return f"advisory:{self.advisory_id}"

    @property
    def dirty(self) -> bool:
        return self.node_state is NodeState.DIRTY

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": NodeKind.ADVISORY.value,
            "advisory_id": self.advisory_id,
            "text": self.text,
            "targets": list(self.targets),
            "turn": self.turn,
            "delivered": self.delivered,
            "node_state": self.node_state.value,
            "dirty_turn": self.dirty_turn,
            "dirty_reason": self.dirty_reason,
        }


@dataclass(frozen=True)
class Edge:
    """A mechanically-derived relation between projected records."""

    source_id: str
    relation: str
    target_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "relation": self.relation,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class InvalidationDecision:
    """One recorded decision, including suppressed and deduped outcomes."""

    event: str
    node_id: str
    path: str
    turn: int
    reason: str
    emitted: bool = False
    notice: str = ""
    source_root_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "organ_id": ORGAN_ID,
            "event": self.event,
            "node_id": self.node_id,
            "path": self.path,
            "turn": self.turn,
            "reason": self.reason,
            "emitted": self.emitted,
            "notice": self.notice,
            "source_root_ids": list(self.source_root_ids),
        }


@dataclass(frozen=True)
class InvalidationResult:
    """Bounded output from one receipt application."""

    path: str = ""
    turn: int = 0
    before_revision: int = 0
    after_revision: int = 0
    cause: str = ""
    dirtied_node_ids: tuple[str, ...] = ()
    notices: tuple[str, ...] = ()
    decisions: tuple[InvalidationDecision, ...] = ()

    @property
    def dirty(self) -> bool:
        return bool(self.dirtied_node_ids)

    @property
    def source_root_count(self) -> int:
        return len(
            {
                source_root
                for decision in self.decisions
                for source_root in decision.source_root_ids
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "path": self.path,
            "turn": self.turn,
            "before_revision": self.before_revision,
            "after_revision": self.after_revision,
            "cause": self.cause,
            "dirtied_node_ids": list(self.dirtied_node_ids),
            "notices": list(self.notices),
            "decisions": [item.to_dict() for item in self.decisions],
        }


def _observation_from_record(record: Record) -> ObservationNode | None:
    key = _mapping(record.get("key")) or record
    source = _mapping(record.get("source")) or record
    path = _path(key.get("path") or record.get("path"))
    if not path:
        return None
    subject = _text(key.get("subject_ref") or record.get("subject_ref") or record.get("predicate"))
    evidence = _text(source.get("evidence_ref") or record.get("evidence_ref"))
    digest = _text(source.get("content_digest") or record.get("content_digest") or record.get("digest"))
    dependencies = _as_strings(record.get("dependencies"))
    return ObservationNode(
        path=path,
        workspace_revision=_int(key.get("workspace_revision", record.get("revision"))),
        what=subject,
        evidence_refs=(evidence,) if evidence else (),
        dependencies=dependencies,
        digest=digest,
        source_kind="world_fact",
        node_id=_text(record.get("observation_id")) or "",
    )


def _claim_from_record(record: Record) -> ClaimNode | None:
    claim_id = _text(
        record.get("claim_id")
        or record.get("hypothesis_id")
        or record.get("id")
    )
    if not claim_id:
        return None
    refs: list[str] = list(_as_strings(record.get("evidence_refs")))
    for key in ("evidence_ref", "fact_id", "observed_fact_id"):
        value = _text(record.get(key))
        if value and value not in refs:
            refs.append(value)
    for item in _sequence(record.get("evidence")):
        if isinstance(item, Mapping):
            value = _text(item.get("evidence_ref") or item.get("fact_id") or item.get("id"))
            if value and value not in refs:
                refs.append(value)
    paths = _as_strings(record.get("paths"))
    if not paths:
        paths = _record_paths(record)
    return ClaimNode(
        claim_id=claim_id,
        what=_text(record.get("what") or record.get("text") or record.get("subject")),
        evidence_refs=tuple(refs),
        paths=tuple(paths),
        workspace_revision=_revision(record),
        state=_text(record.get("state") or record.get("status")) or "SUPPORTED",
    )


def _trigger(evidence: Evidence) -> TriState:
    event = _text(evidence.event.get("event")).upper()
    return TriState.YES if event.startswith("INVALIDATION_") else TriState.NO


CENSUS_SPECS = (
    OrganCensusSpec(
        organ_id=ORGAN_ID,
        trigger=_trigger,
        fire_events=frozenset(
            {
                "INVALIDATION_EDIT_SEEN",
                "INVALIDATION_NODE_DIRTY",
                "INVALIDATION_NOTICE_EMITTED",
                "INVALIDATION_REOBSERVED",
            }
        ),
        suppression_events=frozenset(
            {"INVALIDATION_NOTICE_SUPPRESSED", "INVALIDATION_NOTICE_DEDUPED"}
        ),
        dedup_events=frozenset({"INVALIDATION_NOTICE_DEDUPED"}),
        challenge_once_events=frozenset(),
    ),
)


class LedgerIndex:
    """Small deterministic dependency projection over existing bench records."""

    organ_id = ORGAN_ID

    def __init__(
        self,
        *,
        max_dirty_notices_per_turn: int = MAX_DIRTY_NOTICES_PER_TURN,
        receipt_sink: ReceiptSink | None = None,
    ) -> None:
        if max_dirty_notices_per_turn < 1:
            raise ValueError("max_dirty_notices_per_turn must be positive")
        self.max_dirty_notices_per_turn = max_dirty_notices_per_turn
        self.receipt_sink = receipt_sink
        self._observations: dict[str, ObservationNode] = {}
        self._claims: dict[str, ClaimNode] = {}
        self._advisories: dict[str, AdvisoryNode] = {}
        self._edges: set[Edge] = set()
        self._evidence_to_observation: dict[str, set[str]] = {}
        self._evidence_to_claims: dict[str, set[str]] = {}
        self._source_receipts: list[Record] = []
        self._receipts: list[dict[str, object]] = []
        self._noticed_by_turn: dict[int, set[str]] = {}
        self._last_edit: dict[str, tuple[int, int]] = {}
        # Reverse indexes make edit propagation proportional to the impacted
        # subgraph, not to every historical receipt in the run tape.
        self._path_to_observations: dict[str, set[str]] = {}
        self._dependency_to_observations: dict[str, set[str]] = {}
        self._path_to_claims: dict[str, set[str]] = {}
        self._target_to_advisories: dict[str, set[str]] = {}
        self._dependency_generations: dict[str, DependencyGeneration] = {}
        self._settled_dependency_generations: dict[str, str] = {}
        self._generation_to_claims: dict[str, set[str]] = {}
        self._call_graph: object | None = None
        self._call_graph_root = ""
        self._test_dependency_cache: dict[str, tuple[str, ...]] = {}
        self._graph_builds = 0
        self._propagation_candidates = 0
        # The selector consumes this explicit projection.  MicroGraph remains the
        # hypothesis graph authority; these maps are a bounded view of that same
        # node/edge set, never a second hypothesis store.
        self._projected_nodes: dict[str, Mapping[str, object]] = {}
        self._projected_edges: set[Edge] = set()
        self._hypothesis_postures: dict[str, object] = {}
        # DecisionGraph is attached lazily by the runtime or a lab caller.  It
        # is a projection on this graph, never a sibling invalidation authority.
        self._decision_graph: object | None = None

    @property
    def observations(self) -> tuple[ObservationNode, ...]:
        return tuple(self._observations.values())

    @property
    def claims(self) -> tuple[ClaimNode, ...]:
        return tuple(self._claims.values())

    @property
    def advisories(self) -> tuple[AdvisoryNode, ...]:
        return tuple(self._advisories.values())

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(sorted(self._edges, key=lambda item: (item.source_id, item.relation, item.target_id)))

    @property
    def graph_nodes(self) -> tuple[Mapping[str, object], ...]:
        """Dependency nodes plus the current MicroGraph projection."""
        direct: dict[str, Mapping[str, object]] = {}
        for node in (*self.observations, *self.claims, *self.advisories):
            direct[str(getattr(node, "node_id", ""))] = node.to_dict()
        direct.update(self._projected_nodes)
        decision_graph = self._decision_graph
        decision_nodes = getattr(decision_graph, "graph_nodes", ())
        if isinstance(decision_nodes, Sequence):
            for node in decision_nodes:
                if isinstance(node, Mapping):
                    node_id = _text(node.get("node_id") or node.get("decision_id"))
                    if node_id:
                        direct[node_id] = node
        return tuple(direct[key] for key in sorted(direct) if key)

    @property
    def graph_edges(self) -> tuple[Edge, ...]:
        """One selector graph: DEPGRAPH edges + projected MicroGraph edges."""
        decision_graph = self._decision_graph
        decision_edges = getattr(decision_graph, "graph_edges", ())
        projected_decision_edges = tuple(
            Edge(
                _text(getattr(edge, "source_id", "") or (edge.get("source_id") if isinstance(edge, Mapping) else "")),
                _text(getattr(edge, "relation", "") or (edge.get("relation") if isinstance(edge, Mapping) else "")),
                _text(getattr(edge, "target_id", "") or (edge.get("target_id") if isinstance(edge, Mapping) else "")),
            )
            for edge in decision_edges
            if (
                _text(getattr(edge, "source_id", "") or (edge.get("source_id") if isinstance(edge, Mapping) else ""))
                and _text(getattr(edge, "relation", "") or (edge.get("relation") if isinstance(edge, Mapping) else ""))
                and _text(getattr(edge, "target_id", "") or (edge.get("target_id") if isinstance(edge, Mapping) else ""))
            )
        ) if isinstance(decision_edges, Sequence) else ()
        return tuple(sorted((*self._edges, *self._projected_edges, *projected_decision_edges),
                            key=lambda item: (item.source_id, item.relation, item.target_id)))

    @property
    def hypothesis_postures(self) -> Mapping[str, object]:
        return dict(self._hypothesis_postures)

    @property
    def decision_graph(self) -> object | None:
        return self._decision_graph

    def attach_decision_graph(self, graph: object) -> None:
        """Attach the decision projection to this graph's selector surface."""
        if graph is self:
            raise ValueError("LedgerIndex cannot attach itself as a decision graph")
        self._decision_graph = graph

    @property
    def source_receipts(self) -> tuple[Record, ...]:
        return tuple(self._source_receipts)

    @property
    def receipts(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self._receipts)

    @property
    def dirty_nodes(self) -> tuple[object, ...]:
        return tuple(
            item
            for item in (*self.observations, *self.claims, *self.advisories)
            if getattr(item, "dirty", False)
        )

    @property
    def stale_check_nodes(self) -> tuple[object, ...]:
        """Dirty verification evidence; source/read dirtiness cannot veto done."""
        return tuple(item for item in self.dirty_nodes if _is_check_observation(item))

    @property
    def dirty_evidence_refs(self) -> tuple[str, ...]:
        """Evidence ids that cannot support completion while their node is dirty."""
        refs: list[str] = []
        for node in self.dirty_nodes:
            for ref in getattr(node, "evidence_refs", ()):
                if ref and ref not in refs:
                    refs.append(ref)
        return tuple(refs)

    @property
    def propagation_metrics(self) -> dict[str, int]:
        return {
            "graph_builds": self._graph_builds,
            "indexed_candidates": self._propagation_candidates,
            "dirty_nodes": len(self.dirty_nodes),
        }

    @property
    def current_revision(self) -> int:
        return max((revision for revision, _turn in self._last_edit.values()), default=0)

    def version_for(self, path: str) -> int | None:
        """Return current host revision for one path, independent of recency."""
        normalized = _path(path)
        if not normalized:
            return None
        revisions = [revision for candidate, (revision, _turn) in self._last_edit.items()
                     if _same_path(candidate, normalized)]
        revisions.extend(
            int(getattr(node, "workspace_revision", 0))
            for node in self._observations.values()
            if _same_path(getattr(node, "path", ""), normalized)
        )
        return max(revisions, default=0)

    @property
    def dependency_generations(self) -> Mapping[str, DependencyGeneration]:
        """Return current catalog generations, excluding historical revisions."""
        return dict(self._dependency_generations)

    def dependency_generation(self, dependency_id: str) -> DependencyGeneration | None:
        """Return the current generation for one catalog identity."""
        return self._dependency_generations.get(_text(dependency_id))

    def _claim_dependencies_current(self, claim: ClaimNode) -> bool:
        for dependency in claim.dependency_generations:
            current = self._dependency_generations.get(dependency.dependency_id)
            settled = self._settled_dependency_generations.get(dependency.dependency_id)
            if current is None or current.generation != dependency.generation:
                return False
            if settled != current.generation:
                return False
        return True

    def _refresh_claim_settled_generations(self, claim: ClaimNode) -> ClaimNode:
        """Advance only the dependency legs settled by an explicit host check."""

        updated = tuple(
            replace(dependency, generation=self._dependency_generations[dependency.dependency_id].generation)
            if (
                dependency.dependency_id in self._dependency_generations
                and self._settled_dependency_generations.get(dependency.dependency_id)
                == self._dependency_generations[dependency.dependency_id].generation
            )
            else dependency
            for dependency in claim.dependency_generations
        )
        if updated == claim.dependency_generations:
            return claim
        refreshed = replace(claim, dependency_generations=updated)
        self._claims[claim.node_id] = refreshed
        return refreshed

    def observe_dependency_generation(
        self,
        dependency_id: str,
        generation: str,
        *,
        kind: str = "catalog",
        turn: int = 0,
        reason: str = "catalog_dependency_changed",
    ) -> InvalidationResult:
        """Record a current generation and dirty claims bound to an older one."""

        dependency = DependencyGeneration(dependency_id, generation, kind)
        key = dependency.dependency_id
        previous = self._dependency_generations.get(key)
        self._dependency_generations[key] = dependency
        if previous is None:
            self._settled_dependency_generations.setdefault(key, dependency.generation)
            decision = InvalidationDecision(
                "INVALIDATION_DEPENDENCY_GENERATION_OBSERVED",
                "",
                key,
                max(0, _int(turn)),
                "initial_generation",
            )
            self._record(decision)
            return InvalidationResult(path=key, turn=max(0, _int(turn)), cause="initial_generation", decisions=(decision,))
        if previous.generation == dependency.generation:
            decision = InvalidationDecision(
                "INVALIDATION_DEPENDENCY_GENERATION_UNCHANGED",
                "",
                key,
                max(0, _int(turn)),
                "same_generation",
            )
            self._record(decision)
            return InvalidationResult(path=key, turn=max(0, _int(turn)), cause="same_generation", decisions=(decision,))

        decisions: list[InvalidationDecision] = [
            InvalidationDecision(
                "INVALIDATION_DEPENDENCY_GENERATION_CHANGED",
                "",
                key,
                max(0, _int(turn)),
                reason,
                source_root_ids=(f"catalog:{key}@{previous.generation}",),
            )
        ]
        self._record(decisions[0])
        for claim_id in sorted(self._generation_to_claims.get(key, ())):
            claim = self._claims.get(claim_id)
            if claim is None:
                continue
            dirty, notice = self._mark_dirty(
                claim.node_id,
                path=key,
                turn=max(0, _int(turn)),
                reason=reason,
                source_root_ids=(f"catalog:{key}@{previous.generation}",),
            )
            if dirty is not None:
                decisions.append(dirty)
            decisions.append(notice)
        return self._result(
            path=key,
            turn=max(0, _int(turn)),
            before_revision=0,
            after_revision=0,
            cause=reason,
            decisions=decisions,
        )

    def settle_dependency_generation(
        self,
        dependency_id: str,
        generation: str,
        *,
        kind: str = "catalog",
    ) -> bool:
        """Mark one dependency generation settled after an authoritative check."""

        dependency = DependencyGeneration(dependency_id, generation, kind)
        current = self._dependency_generations.get(dependency.dependency_id)
        if current is None or current.generation != dependency.generation:
            return False
        self._settled_dependency_generations[dependency.dependency_id] = dependency.generation
        return True

    def revalidate_claim(self, claim_id: str, *, turn: int = 0, reason: str = REASON_REOBSERVED) -> bool:
        """Clear a dirty catalog claim only after every bound basis is settled."""

        ident = _text(claim_id)
        claim = self._claims.get(f"claim:{ident}") or self._claims.get(ident)
        if claim is None:
            raise KeyError(ident)
        claim = self._refresh_claim_settled_generations(claim)
        if claim.dependency_generations and not self._claim_dependencies_current(claim):
            return False
        if not claim.dirty:
            return True
        self._claims[claim.node_id] = replace(
            claim,
            state=claim.prior_state,
            node_state=NodeState.CLEAN,
            dirty_turn=None,
            dirty_reason="",
        )
        decision = InvalidationDecision(
            "INVALIDATION_CLAIM_REVALIDATED",
            claim.node_id,
            "catalog",
            max(0, _int(turn)),
            reason,
        )
        self._record(decision)
        return True

    def project_micrograph(self, tracker: object | None) -> None:
        """Project the live HypothesisTracker MicroGraph into this graph.

        The tracker owns node/edge creation.  ``LedgerIndex`` owns invalidation
        and exposes the projection to the selector, which prevents three graph
        implementations from drifting while retaining the detreason API.
        """
        graph = getattr(tracker, "graph", None) if tracker is not None else None
        if graph is None and tracker is not None:
            consumer = getattr(tracker, "_consumer", None)
            graph = getattr(consumer, "graph", None)
        if graph is None:
            return
        projected_nodes: dict[str, Mapping[str, object]] = {}
        for node in getattr(graph, "nodes", ()):
            node_id = _text(getattr(node, "id", ""))
            if not node_id:
                continue
            projected_nodes[node_id] = {
                "node_id": node_id,
                "kind": _text(getattr(node, "kind", "")),
                "key": _text(getattr(node, "key", "")),
                "attrs": dict(getattr(node, "attrs", {}) or {}),
            }
        projected_edges: set[Edge] = set()
        for edge in getattr(graph, "edges", ()):
            source = _text(getattr(edge, "source", ""))
            target = _text(getattr(edge, "target", ""))
            relation = _text(getattr(edge, "kind", ""))
            if source and target and relation:
                projected_edges.add(Edge(source, relation, target))
        self._projected_nodes = projected_nodes
        self._projected_edges = projected_edges
        postures = getattr(tracker, "postures", None)
        if isinstance(postures, Mapping):
            self._hypothesis_postures = dict(postures)
        elif callable(postures):
            try:
                value = postures()
                self._hypothesis_postures = dict(value) if isinstance(value, Mapping) else {}
            except (TypeError, ValueError):
                self._hypothesis_postures = {}

    @property
    def path(self) -> str:
        for node in self.dirty_nodes:
            candidate = _text(getattr(node, "path", ""))
            if candidate:
                return candidate
        return "source"

    def configure_call_graph(self, graph_or_root: object) -> object | None:
        """Attach one repository call graph, building it at most once per root."""
        if graph_or_root is None:
            return None
        if not isinstance(graph_or_root, (str, Path)):
            self._call_graph = graph_or_root
            self._test_dependency_cache.clear()
            return graph_or_root
        root = str(Path(graph_or_root).resolve())
        if self._call_graph is not None and self._call_graph_root == root:
            return self._call_graph
        try:
            from mc_repo_intel_vendor.call_graph import build_graph
            self._call_graph = build_graph(root, use_cache=False, use_memo=True)
            self._call_graph_root = root
            self._graph_builds += 1
            self._test_dependency_cache.clear()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            # A missing optional graph must not erase direct-path invalidation.
            self._call_graph = None
            self._call_graph_root = root
        return self._call_graph

    def dependencies_for(self, path: str) -> tuple[str, ...]:
        """Return transitive source files called by a test file."""
        normalized = _path(path)
        if not normalized or not _is_test_path(normalized) or self._call_graph is None:
            return ()
        cached = self._test_dependency_cache.get(normalized)
        if cached is not None:
            return cached
        graph = self._call_graph
        dependencies: set[str] = set()
        defs = getattr(graph, "defs", {})
        for symbol, definition in defs.items():
            if _path(getattr(definition, "file", "")) != normalized:
                continue
            for ref in graph.find_callees(symbol, transitive=True):
                file_name = _path(getattr(ref, "file", ""))
                if file_name and file_name != normalized:
                    dependencies.add(file_name)
        result = tuple(sorted(dependencies))
        self._test_dependency_cache[normalized] = result
        return result

    def _record(self, decision: InvalidationDecision) -> None:
        payload = decision.to_dict()
        self._receipts.append(payload)
        if self.receipt_sink is not None:
            try:
                self.receipt_sink(payload)
            except Exception as exc:  # noqa: BLE001 - a sink cannot hide a decision
                self._receipts.append(
                    {
                        "schema": SCHEMA,
                        "organ_id": ORGAN_ID,
                        "event": "INVALIDATION_RECEIPT_SINK_ERROR",
                        "error": type(exc).__name__,
                    }
                )

    def _edge(self, source_id: str, relation: str, target_id: str) -> None:
        if source_id and target_id:
            self._edges.add(Edge(source_id, relation, target_id))

    def add_observation(
        self,
        observation: ObservationNode | Record | None = None,
        *,
        path: str = "",
        workspace_revision: int = 0,
        what: str = "",
        evidence_ref: str = "",
        dependencies: Sequence[str] = (),
        digest: str = "",
        source_kind: str = "source",
    ) -> ObservationNode:
        """Add or merge one world/source observation."""

        if isinstance(observation, ObservationNode):
            node = observation
        elif isinstance(observation, Mapping):
            node = _observation_from_record(observation)
            if node is None:
                raise ValueError("observation record needs a path")
        else:
            node = ObservationNode(
                path=path,
                workspace_revision=workspace_revision,
                what=what,
                evidence_refs=(evidence_ref,) if evidence_ref else (),
                dependencies=tuple(dependencies),
                digest=digest,
                source_kind=source_kind,
            )
        if _is_test_path(node.path) and not node.dependencies:
            derived = self.dependencies_for(node.path)
            if derived:
                node = replace(node, dependencies=derived)
        existing = self._observations.get(node.node_id)
        if existing is not None:
            refs = tuple(dict.fromkeys((*existing.evidence_refs, *node.evidence_refs)))
            dependencies = tuple(dict.fromkeys((*existing.dependencies, *node.dependencies)))
            node = replace(
                existing,
                evidence_refs=refs,
                dependencies=dependencies,
                digest=node.digest or existing.digest,
            )
            for old_path in (existing.path, *existing.dependencies):
                self._path_to_observations.get(old_path, set()).discard(existing.node_id)
        self._observations[node.node_id] = node
        self._path_to_observations.setdefault(node.path, set()).add(node.node_id)
        for ref in node.evidence_refs:
            self._evidence_to_observation.setdefault(ref, set()).add(node.node_id)
            self._edge(f"receipt:{ref}" if not ref.startswith("receipt:") else ref, "produces", node.node_id)
        for dependency in node.dependencies:
            self._dependency_to_observations.setdefault(dependency, set()).add(node.node_id)
            self._edge(node.node_id, "depends_on_path", f"path:{dependency}")
        self._bind_claims(node)
        return node

    def add_world_fact(self, fact: object) -> ObservationNode:
        """Project a ``WorldFact`` without importing its module at runtime."""

        if isinstance(fact, Mapping):
            record = fact
        elif hasattr(fact, "to_dict"):
            value = fact.to_dict()  # type: ignore[union-attr]
            if not isinstance(value, Mapping):
                raise TypeError("WorldFact.to_dict() must return a mapping")
            record = value
        else:
            record = {
                "path": getattr(fact, "path", ""),
                "workspace_revision": getattr(fact, "workspace_revision", 0),
                "subject_ref": getattr(fact, "subject_ref", ""),
                "fact_id": getattr(fact, "fact_id", ""),
                "source": {
                    "evidence_ref": getattr(fact, "evidence_ref", ""),
                    "content_digest": getattr(fact, "content_digest", ""),
                },
                "dependencies": getattr(fact, "dependencies", ()),
            }
        return self.add_observation(record)

    def _bind_claims(self, observation: ObservationNode) -> None:
        for claim in tuple(self._claims.values()):
            if set(claim.evidence_refs).intersection(observation.evidence_refs):
                self._edge(observation.node_id, "supports", claim.node_id)
            if claim.paths and any(_same_path(path, observation.path) for path in claim.paths):
                if claim.workspace_revision == observation.workspace_revision:
                    self._edge(observation.node_id, "supports", claim.node_id)

    def add_claim(
        self,
        claim: ClaimNode | Record | str,
        *,
        what: str = "",
        evidence_refs: Sequence[str] = (),
        evidence: Sequence[str] = (),
        paths: Sequence[str] = (),
        workspace_revision: int = 0,
        state: str = "SUPPORTED",
        dependency_generations: Mapping[str, object] | Sequence[object] = (),
        catalog_dependencies: Mapping[str, object] | Sequence[object] = (),
    ) -> ClaimNode:
        """Add one explicit claim; arbitrary model prose is never promoted."""

        if isinstance(claim, ClaimNode):
            node = claim
        elif isinstance(claim, Mapping):
            node = _claim_from_record(claim)
            if node is None:
                raise ValueError("claim record needs claim_id or hypothesis_id")
        else:
            raw_dependencies: object = dependency_generations
            if catalog_dependencies:
                if dependency_generations:
                    raise ValueError("provide one catalog dependency collection")
                raw_dependencies = catalog_dependencies
            node = ClaimNode(
                claim_id=claim,
                what=what,
                evidence_refs=tuple((*evidence_refs, *evidence)),
                paths=tuple(paths),
                workspace_revision=workspace_revision,
                state=state,
                dependency_generations=_catalog_dependency_values(raw_dependencies),
            )
        old = self._claims.get(node.node_id)
        if old is not None and old.dirty:
            node = replace(
                node,
                node_state=old.node_state,
                dirty_turn=old.dirty_turn,
                dirty_reason=old.dirty_reason,
                prior_state=old.prior_state,
                state=old.state,
                dependency_generations=old.dependency_generations or node.dependency_generations,
            )
        previous = self._claims.get(node.node_id)
        if previous is not None:
            for dependency in previous.dependency_generations:
                self._generation_to_claims.get(dependency.dependency_id, set()).discard(node.node_id)
        self._claims[node.node_id] = node
        for path in node.paths:
            self._path_to_claims.setdefault(path, set()).add(node.node_id)
        for ref in node.evidence_refs:
            self._evidence_to_claims.setdefault(ref, set()).add(node.node_id)
            for observation_id in self._evidence_to_observation.get(ref, ()):
                self._edge(observation_id, "supports", node.node_id)
        for dependency in node.dependency_generations:
            self._dependency_generations.setdefault(
                dependency.dependency_id,
                dependency,
            )
            self._settled_dependency_generations.setdefault(
                dependency.dependency_id,
                dependency.generation,
            )
            self._generation_to_claims.setdefault(dependency.dependency_id, set()).add(node.node_id)
            self._edge(
                f"catalog_dependency:{dependency.dependency_id}",
                "generates",
                node.node_id,
            )
        for observation in self._observations.values():
            if node.paths and any(_same_path(path, observation.path) for path in node.paths):
                if node.workspace_revision == observation.workspace_revision:
                    self._edge(observation.node_id, "supports", node.node_id)
        return node

    def add_catalog_claim(
        self,
        claim_id: str,
        *,
        dependency_generations: Mapping[str, object] | Sequence[object],
        what: str = "catalog assertion",
        evidence_refs: Sequence[str] = (),
        paths: Sequence[str] = (),
        workspace_revision: int = 0,
        applicability: str = "CURRENT",
    ) -> ClaimNode:
        """Bind one catalog assertion without granting it behavioral authority."""

        node = self.add_claim(
            claim_id,
            what=what,
            evidence_refs=evidence_refs,
            paths=paths,
            workspace_revision=workspace_revision,
            dependency_generations=dependency_generations,
        )
        if _text(applicability).upper() != "CURRENT" and not node.dirty:
            self._mark_dirty(
                node.node_id,
                path="catalog",
                turn=max(0, workspace_revision),
                reason="catalog_assertion_not_current",
            )
            node = self._claims[node.node_id]
        return node

    def add_advisory(
        self,
        advisory: AdvisoryNode | Record | str,
        *,
        text: str = "",
        targets: Sequence[str] = (),
        turn: int = 0,
        delivered: bool = True,
    ) -> AdvisoryNode:
        """Track only delivered/explicit advisories and their named targets."""

        if isinstance(advisory, AdvisoryNode):
            node = advisory
        elif isinstance(advisory, Mapping):
            advisory_id = _text(advisory.get("advisory_id") or advisory.get("id"))
            if not advisory_id:
                advisory_id = "advisory:" + _digest(advisory)[:20]
            raw_targets = advisory.get("targets")
            target_values = _as_strings(raw_targets)
            if isinstance(raw_targets, str):
                target_values = _paths(raw_targets)
            if not target_values:
                target_values = _record_paths(advisory)
            node = AdvisoryNode(
                advisory_id=advisory_id,
                text=_text(advisory.get("text") or advisory.get("detail") or advisory.get("line")),
                targets=target_values,
                turn=_int(advisory.get("turn")),
                delivered=bool(advisory.get("delivered", advisory.get("emitted", True))),
            )
        else:
            node = AdvisoryNode(advisory, text, tuple(targets), turn, delivered)
        self._advisories[node.node_id] = node
        for target in node.targets:
            normalized_target = _path(target)
            if normalized_target:
                self._target_to_advisories.setdefault(normalized_target, set()).add(node.node_id)
            for claim in self._claims.values():
                if target == claim.claim_id or target == claim.node_id:
                    self._edge(node.node_id, "targets", claim.node_id)
            for observation in self._observations.values():
                if _same_path(target, observation.path):
                    self._edge(node.node_id, "targets", observation.node_id)
        return node

    def _notice(
        self,
        node_id: str,
        path: str,
        what: str,
        turn: int,
        reason: str,
        source_root_ids: tuple[str, ...] = (),
    ) -> tuple[InvalidationDecision | None, InvalidationDecision]:
        noticed = self._noticed_by_turn.setdefault(turn, set())
        if node_id in noticed:
            decision = InvalidationDecision(
                "INVALIDATION_NOTICE_DEDUPED", node_id, path, turn, reason, False, "", source_root_ids
            )
            self._record(decision)
            return decision
        if len(noticed) >= self.max_dirty_notices_per_turn:
            decision = InvalidationDecision(
                "INVALIDATION_NOTICE_SUPPRESSED", node_id, path, turn, reason, False, "", source_root_ids
            )
            self._record(decision)
            return decision
        noticed.add(node_id)
        notice = (
            f"your belief about {path}:{what} predates your edit at turn {turn} "
            "— re-observe before relying on it"
        )[:MAX_NOTICE_CHARS]
        decision = InvalidationDecision(
            "INVALIDATION_NOTICE_EMITTED", node_id, path, turn, reason, True, notice, source_root_ids
        )
        self._record(decision)
        return decision

    def _mark_dirty(
        self,
        node_id: str,
        *,
        path: str,
        turn: int,
        reason: str,
        source_root_ids: tuple[str, ...] = (),
    ) -> tuple[InvalidationDecision | None, InvalidationDecision]:
        node = self._node(node_id)
        if node is None:
            raise KeyError(node_id)
        if getattr(node, "dirty", False):
            decision = InvalidationDecision(
                "INVALIDATION_NOTICE_DEDUPED",
                node_id,
                path,
                turn,
                reason,
                False,
                "",
                source_root_ids,
            )
            self._record(decision)
            return None, decision
        if isinstance(node, ObservationNode):
            updated: object = replace(
                node, state=NodeState.DIRTY, dirty_turn=turn, dirty_reason=reason
            )
            self._observations[node_id] = updated  # type: ignore[assignment]
        elif isinstance(node, ClaimNode):
            updated = replace(
                node,
                node_state=NodeState.DIRTY,
                dirty_turn=turn,
                dirty_reason=reason,
                prior_state=node.state,
                state=NodeState.DIRTY.value,
            )
            self._claims[node_id] = updated
        else:
            updated = replace(
                node, node_state=NodeState.DIRTY, dirty_turn=turn, dirty_reason=reason
            )
            self._advisories[node_id] = updated
        dirty = InvalidationDecision(
            "INVALIDATION_NODE_DIRTY", node_id, path, turn, reason, False, "", source_root_ids
        )
        self._record(dirty)
        decision_graph = self._decision_graph
        mark_item = getattr(decision_graph, "mark_item", None)
        if callable(mark_item):
            item_ids = (updated.node_id, *tuple(getattr(updated, "evidence_refs", ())))
            for item_id in dict.fromkeys(item_ids):
                try:
                    mark_item(item_id, state="STALE", reason=reason)
                except (TypeError, ValueError):
                    # The existing invalidation result remains authoritative;
                    # a projection adapter must not hide it or break the loop.
                    continue
        return dirty, self._notice(
            node_id, path, self._node_what(updated), turn, reason, source_root_ids
        )

    def _node(self, node_id: str) -> ObservationNode | ClaimNode | AdvisoryNode | None:
        return self._observations.get(node_id) or self._claims.get(node_id) or self._advisories.get(node_id)

    @staticmethod
    def _node_what(node: object) -> str:
        if isinstance(node, ObservationNode):
            return node.what
        if isinstance(node, ClaimNode):
            return node.what
        if isinstance(node, AdvisoryNode):
            return node.text or node.advisory_id
        return "unknown"

    def _result(
        self,
        *,
        path: str,
        turn: int,
        before_revision: int,
        after_revision: int,
        cause: str,
        decisions: Sequence[InvalidationDecision],
    ) -> InvalidationResult:
        emitted = tuple(item.notice for item in decisions if item.emitted and item.notice)
        dirtied = tuple(
            dict.fromkeys(
                item.node_id
                for item in decisions
                if item.event == "INVALIDATION_NODE_DIRTY"
            )
        )
        return InvalidationResult(
            path=path,
            turn=turn,
            before_revision=before_revision,
            after_revision=after_revision,
            cause=cause,
            dirtied_node_ids=dirtied,
            notices=emitted,
            decisions=tuple(decisions),
        )

    def _invalidate_path(
        self,
        path: str,
        *,
        before_revision: int,
        after_revision: int,
        turn: int,
        cause: str,
        exact: bool = False,
    ) -> InvalidationResult:
        normalized_path = _path(path)
        event = "INVALIDATION_EDIT_SEEN" if cause == "edit_receipt" else "INVALIDATION_PATH_EVALUATED"
        start = InvalidationDecision(event, "", normalized_path, turn, cause)
        self._record(start)
        decisions: list[InvalidationDecision] = [start]
        direct_ids = set(self._path_to_observations.get(normalized_path, ()))
        if not exact:
            direct_ids.update(
                node_id
                for candidate_path, node_ids in self._path_to_observations.items()
                if _same_path(candidate_path, normalized_path)
                for node_id in node_ids
            )
        dependency_ids = set(self._dependency_to_observations.get(normalized_path, ()))
        if not exact:
            dependency_ids.update(
                node_id
                for candidate_path, node_ids in self._dependency_to_observations.items()
                if _same_path(candidate_path, normalized_path)
                for node_id in node_ids
            )
        candidate_ids = direct_ids | dependency_ids
        eligible_observations = [
            self._observations[node_id]
            for node_id in candidate_ids
            if node_id in self._observations
            and not self._observations[node_id].dirty
            and self._observations[node_id].workspace_revision <= before_revision
        ]
        # A fresh observation supersedes older clean observations for the same
        # path. Historical nodes remain in the audit tape but do not create a
        # second propagation wave on every subsequent edit.
        latest_by_path: dict[str, ObservationNode] = {}
        for observation in eligible_observations:
            previous = latest_by_path.get(observation.path)
            if previous is None or observation.workspace_revision > previous.workspace_revision:
                latest_by_path[observation.path] = observation
        dirty_observations = list(latest_by_path.values())
        self._propagation_candidates += len(dirty_observations)
        source_roots = tuple(item.source_root_id for item in dirty_observations)
        for observation in dirty_observations:
            reason = (
                REASON_SOURCE_EDIT
                if observation.node_id in direct_ids
                else REASON_CALL_GRAPH_DEPENDENCY
            )
            dirty, notice = self._mark_dirty(
                observation.node_id,
                path=normalized_path,
                turn=turn,
                reason=reason,
                source_root_ids=(observation.source_root_id,),
            )
            if dirty is not None:
                decisions.append(dirty)
            decisions.append(notice)
        dirty_refs = {
            ref
            for item in dirty_observations
            for ref in item.evidence_refs
        }
        claim_ids = set(self._path_to_claims.get(normalized_path, ()))
        if not exact:
            claim_ids.update(
                claim_id
                for candidate_path, ids in self._path_to_claims.items()
                if _same_path(candidate_path, normalized_path)
                for claim_id in ids
            )
        claim_ids.update(
            claim_id
            for ref in dirty_refs
            for claim_id in self._evidence_to_claims.get(ref, ())
        )
        self._propagation_candidates += len(claim_ids)
        for claim_id in claim_ids:
            claim = self._claims.get(claim_id)
            if claim is None or claim.dirty:
                continue
            evidence_related = bool(set(claim.evidence_refs).intersection(dirty_refs))
            path_related = any(_same_path(item, normalized_path, exact=exact) for item in claim.paths)
            if evidence_related or path_related:
                dirty, notice = self._mark_dirty(
                    claim.node_id,
                    path=normalized_path,
                    turn=turn,
                    reason=cause,
                    source_root_ids=source_roots,
                )
                if dirty is not None:
                    decisions.append(dirty)
                decisions.append(notice)
        advisory_ids = set(self._target_to_advisories.get(normalized_path, ()))
        if not exact:
            advisory_ids.update(
                advisory_id
                for target, ids in self._target_to_advisories.items()
                if _same_path(target, normalized_path)
                for advisory_id in ids
            )
        self._propagation_candidates += len(advisory_ids)
        for advisory_id in advisory_ids:
            advisory = self._advisories.get(advisory_id)
            if advisory is not None and not advisory.dirty and any(_same_path(target, normalized_path, exact=exact) for target in advisory.targets):
                dirty, notice = self._mark_dirty(
                    advisory.node_id,
                    path=normalized_path,
                    turn=turn,
                    reason=cause,
                    source_root_ids=source_roots,
                )
                if dirty is not None:
                    decisions.append(dirty)
                decisions.append(notice)
        return self._result(
            path=normalized_path,
            turn=turn,
            before_revision=before_revision,
            after_revision=after_revision,
            cause=cause,
            decisions=decisions,
        )

    def _apply_red_test(self, record: Record, path: str) -> InvalidationResult:
        turn = _int(record.get("turn"))
        revision = _revision(record)
        start = InvalidationDecision(
            "INVALIDATION_RED_TEST_SEEN",
            "",
            path,
            turn,
            "red_test_traceback_anchor",
        )
        self._record(start)
        decisions: list[InvalidationDecision] = [start]
        last_edit = next(
            (
                value
                for edit_path, value in self._last_edit.items()
                if _same_path(edit_path, path)
            ),
            None,
        )
        if last_edit is None or last_edit[0] > revision:
            return self._result(
                path=path,
                turn=turn,
                before_revision=revision,
                after_revision=revision,
                cause="red_test_without_prior_edit",
                decisions=decisions,
            )
        for claim in tuple(self._claims.values()):
            observations = [
                observation
                for ref in claim.evidence_refs
                for observation_id in self._evidence_to_observation.get(ref, ())
                if (observation := self._observations.get(observation_id)) is not None
            ]
            path_observations = [
                item for item in observations if _same_path(item.path, path)
            ]
            path_claim = any(_same_path(item, path) for item in claim.paths)
            if not path_observations and not path_claim:
                continue
            if path_claim and any(
                not _same_path(item, path) for item in claim.paths
            ):
                continue
            if observations and (
                len(path_observations) != len(observations)
                or not all(item.workspace_revision < revision for item in observations)
            ):
                continue
            dirty, notice = self._mark_dirty(
                claim.node_id,
                path=path,
                turn=turn,
                reason="red_test_traceback_anchor",
                source_root_ids=tuple(item.source_root_id for item in path_observations),
            )
            if dirty is not None:
                decisions.append(dirty)
            decisions.append(notice)
        return self._result(
            path=path,
            turn=turn,
            before_revision=revision,
            after_revision=revision,
            cause="red_test_traceback_anchor",
            decisions=decisions,
        )

    def reobserve(
        self,
        path: str,
        workspace_revision: int,
        *,
        evidence_ref: str = "",
        what: str = "re-observed source",
        digest: str = "",
        dependencies: Sequence[str] = (),
        turn: int = 0,
        exact: bool = False,
        revalidated_claim_ids: Sequence[str] = (),
        source_kind: str = "reobserve",
    ) -> tuple[str, ...]:
        """Add fresh evidence and clear only explicitly revalidated claims.

        A source observation is not proof that every claim mentioning that
        source still holds.  The host must name claims it revalidated; source
        observations may still clear their own dirty observation and advisory
        nodes without silently restoring dependent claims.
        """

        observation = self.add_observation(
            path=path,
            workspace_revision=workspace_revision,
            evidence_ref=evidence_ref,
            what=what,
            digest=digest,
            dependencies=dependencies,
            source_kind=source_kind,
        )
        generation = _text(digest) or f"revision:{observation.workspace_revision}"
        for dependency_id, current in tuple(self._dependency_generations.items()):
            if _dependency_matches_path(dependency_id, observation.path):
                self.settle_dependency_generation(
                    dependency_id,
                    generation,
                    kind=current.kind,
                )
        cleared: list[str] = []
        revalidated = {
            _text(claim_id)
            for claim_id in _sequence(revalidated_claim_ids)
            if _text(claim_id)
        }
        for item in tuple(self._observations.values()):
            if item.dirty and _same_path(item.path, observation.path, exact=exact) and item.workspace_revision <= observation.workspace_revision:
                self._observations[item.node_id] = replace(
                    item, state=NodeState.CLEAN, dirty_turn=None, dirty_reason=""
                )
                cleared.append(item.node_id)
        for claim in tuple(self._claims.values()):
            if claim.claim_id not in revalidated:
                continue
            if claim.dependency_generations:
                claim = self._refresh_claim_settled_generations(claim)
                if self._claim_dependencies_current(claim):
                    if claim.dirty:
                        self._claims[claim.node_id] = replace(
                            claim,
                            state=claim.prior_state,
                            node_state=NodeState.CLEAN,
                            dirty_turn=None,
                            dirty_reason="",
                        )
                        cleared.append(claim.node_id)
                continue
            if claim.dirty and any(_same_path(item, observation.path, exact=exact) for item in claim.paths):
                self._claims[claim.node_id] = replace(
                    claim,
                    state=claim.prior_state,
                    node_state=NodeState.CLEAN,
                    dirty_turn=None,
                    dirty_reason="",
                )
                cleared.append(claim.node_id)
            elif claim.dirty and set(claim.evidence_refs).intersection(observation.evidence_refs):
                self._claims[claim.node_id] = replace(
                    claim,
                    state=claim.prior_state,
                    node_state=NodeState.CLEAN,
                    dirty_turn=None,
                    dirty_reason="",
                )
                cleared.append(claim.node_id)
        for advisory in tuple(self._advisories.values()):
            if advisory.dirty and any(_same_path(target, observation.path, exact=exact) for target in advisory.targets):
                self._advisories[advisory.node_id] = replace(
                    advisory, node_state=NodeState.CLEAN, dirty_turn=None, dirty_reason=""
                )
                cleared.append(advisory.node_id)
        decision = InvalidationDecision(
            "INVALIDATION_REOBSERVED",
            observation.node_id,
            observation.path,
            max(0, _int(turn)),
            "fresh_observation",
            False,
        )
        self._record(decision)
        return tuple(dict.fromkeys(cleared))

    def completion_gate(self) -> dict[str, object]:
        """Return the one authoritative clean-evidence decision for done/stage."""
        # A pre-edit read is stale historical context, not failed CHECK
        # evidence.  Only an unrevalidated check can produce DEPENDENCY_VETO.
        dirty = self.stale_check_nodes
        if not dirty:
            return {
                "admitted": True,
                "reason_code": "CLEAN_EVIDENCE",
                "dirty_node_ids": [],
                "evidence_refs": [],
            }
        first = dirty[0]
        path = _text(getattr(first, "path", ""))
        refs: list[str] = []
        for node in dirty:
            for ref in getattr(node, "evidence_refs", ()):
                if ref and ref not in refs:
                    refs.append(ref)
        observed_revision = _int(getattr(first, "workspace_revision", 0))
        current_revision = observed_revision
        if path in self._last_edit:
            current_revision = self._last_edit[path][0]
        elif path:
            for edit_path, (revision, _turn) in self._last_edit.items():
                if _same_path(edit_path, path):
                    current_revision = max(current_revision, revision)
        ref_text = refs[0] if refs else getattr(first, "node_id", "evidence")
        detail = (
            f"{COMPLETION_BLOCK_REASON}: {ref_text} was observed on "
            f"{path}@rev{observed_revision}, now rev{current_revision}; re-test before done"
        )
        return {
            "admitted": False,
            "reason_code": COMPLETION_BLOCK_REASON,
            "finding": detail[:MAX_NOTICE_CHARS],
            "next_action": "re-run the affected test or re-observe the edited file",
            "dirty_node_ids": [getattr(item, "node_id", "") for item in dirty],
            "evidence_refs": list(refs),
            "dirty_reasons": sorted({str(getattr(item, "dirty_reason", "")) for item in dirty}),
        }

    def observe_receipt(self, record: Record) -> InvalidationResult:
        """Fold one existing receipt and derive any invalidation mechanically."""

        self._source_receipts.append(record)
        kind = _text(record.get("kind")).lower()
        result = _result(record)
        turn = _int(record.get("turn"))
        revision = _revision(record)
        receipt_id = _receipt_id(record)
        if kind == "read":
            paths = _record_paths(record)
            if paths and result in {"OK", "HITS", "GREEN", "PASS"}:
                for observed_path in paths:
                    self.reobserve(
                        observed_path,
                        revision,
                        evidence_ref=receipt_id,
                        turn=turn,
                        dependencies=_as_strings(record.get("dependencies")),
                        digest=_text(record.get("digest") or record.get("content_digest")),
                        exact="target_path" in record,
                    )
            elif paths and result in {"MISSING", "NOT_FOUND"}:
                return self._invalidate_path(
                    paths[0],
                    before_revision=revision,
                    after_revision=revision,
                    turn=turn,
                    cause="missing_read_after_prior_observation",
                    exact="target_path" in record,
                )
            else:
                self._record(
                    InvalidationDecision("INVALIDATION_RECEIPT_IGNORED", "", "", turn, "read_without_path")
                )
            return InvalidationResult(turn=turn, cause="observation")
        if kind == "test" and result in {"GREEN", "PASS", "PASSED"}:
            paths = _record_paths(record, prefer_anchor=True) or _record_paths(record)
            if not paths:
                self._record(
                    InvalidationDecision("INVALIDATION_RECEIPT_IGNORED", "", "", turn, "green_test_without_path")
                )
                return InvalidationResult(turn=turn, cause="green_test_without_path")
            dependencies = _as_strings(record.get("dependencies"))
            for tested_path in paths:
                self.reobserve(
                    tested_path,
                    revision,
                    evidence_ref=receipt_id,
                    what="green test receipt",
                    dependencies=dependencies,
                    digest=_text(record.get("digest") or record.get("content_digest")),
                    turn=turn,
                    exact="target_path" in record,
                    source_kind="check",
                )
            return InvalidationResult(path=paths[0], turn=turn, cause="green_test")
        if kind == "edit" and result == "APPLIED":
            paths = _record_paths(record)
            if not paths:
                self._record(
                    InvalidationDecision("INVALIDATION_RECEIPT_IGNORED", "", "", turn, "edit_without_path")
                )
                return InvalidationResult(turn=turn, cause="edit_without_path")
            path = paths[0]
            after_revision = revision + 1
            self._last_edit[path] = (after_revision, turn)
            return self._invalidate_path(
                path,
                before_revision=revision,
                after_revision=after_revision,
                turn=turn,
                cause="edit_receipt",
                exact="target_path" in record,
            )
        if kind == "test" and result == "RED":
            paths = _record_paths(record, prefer_anchor=True)
            if paths:
                return self._apply_red_test(record, paths[0])
        if kind in {"organ_out", "advisory", "delivery"} and (
            record.get("emitted") is True or record.get("delivered") is True
        ):
            advisory = self.add_advisory(record)
            self._record(
                InvalidationDecision(
                    "INVALIDATION_ADVISORY_TRACKED",
                    advisory.node_id,
                    _path(advisory.targets[0]) if advisory.targets else "",
                    turn,
                    "delivered_advisory",
                )
            )
            return InvalidationResult(turn=turn, cause="delivered_advisory")
        self._record(
            InvalidationDecision("INVALIDATION_RECEIPT_IGNORED", "", "", turn, "not_an_invalidation_trigger")
        )
        return InvalidationResult(turn=turn, cause="not_an_invalidation_trigger")

    ingest_receipt = observe_receipt

    def census(self, *, run_id: str = "invalidation") -> object:
        """Run the existing post-task census and persist its receipt locally."""

        report = run_census(
            Evidence.from_records(receipts=self.receipts, run_id=run_id),
            specs=CENSUS_SPECS,
            run_id=run_id,
        )
        receipt = getattr(report, "receipt", None)
        if isinstance(receipt, Mapping):
            self._receipts.append(dict(receipt))
        return report


def _runtime_receipts(runtime: object) -> tuple[Mapping[str, object], ...]:
    """Collect host-owned receipt tapes without making a second receipt store."""
    values: list[Mapping[str, object]] = []
    seen: set[str] = set()
    owners = (runtime, getattr(runtime, "owner", None))
    attrs = (
        "receipts", "test_receipts", "probe_receipts", "fault_receipts",
        "completed_receipts", "localize_receipts",
    )
    for owner in owners:
        if owner is None:
            continue
        for attr in attrs:
            raw_values = getattr(owner, attr, ()) or ()
            if callable(raw_values):
                raw_values = raw_values()
            if isinstance(raw_values, Mapping):
                raw_values = (raw_values,)
            for raw in raw_values:
                record = _record_value(raw)
                if record is None:
                    continue
                ident = _receipt_id(record)
                if ident in seen:
                    continue
                seen.add(ident)
                values.append(record)
    return tuple(values)


def runtime_invalidation_index(runtime: object) -> LedgerIndex:
    """Return the live index owned by one runtime instance."""
    owner = getattr(runtime, "owner", None)
    owner_index = getattr(owner, "inval", None)
    index = owner_index if isinstance(owner_index, LedgerIndex) else getattr(runtime, "invalidation_index", None)
    if not isinstance(index, LedgerIndex):
        index = getattr(runtime, "_invalidation_index", None)
    if not isinstance(index, LedgerIndex):
        index = LedgerIndex()
    # KernelBolt.inval is the owner-side authority when the live loop has one.
    # Runtime aliases point to it; they must never silently create a sibling graph.
    setattr(runtime, "invalidation_index", index)
    setattr(runtime, "_invalidation_index", index)
    root = getattr(owner, "root", None)
    if root and getattr(index, "_call_graph", None) is None:
        index.configure_call_graph(root)
    if getattr(runtime, "decision_graph", None) is None:
        try:
            from kernel_decision_graph import DecisionGraph

            graph = DecisionGraph(
                ledger=index,
                recall=getattr(runtime, "recall_store", None),
                micrograph=getattr(runtime, "hypothesis_tracker", None),
            )
            setattr(runtime, "decision_graph", graph)
        except (ImportError, TypeError, ValueError) as exc:
            setattr(runtime, "_decision_graph_error", type(exc).__name__)
    else:
        index.attach_decision_graph(getattr(runtime, "decision_graph"))
    return index


def _refresh_runtime_consumers(runtime: object, index: LedgerIndex) -> None:
    """Reopen append-only consumers and mark stale reasoning/recall records."""
    owner = getattr(runtime, "owner", None)
    hypothesis_tracker = getattr(runtime, "hypothesis_tracker", None)
    if hypothesis_tracker is None:
        hypothesis_tracker = getattr(runtime, "_hypothesis_tracker", None)
    if hypothesis_tracker is None:
        hypothesis_tracker = getattr(owner, "hyp", None)
    # Project before the dirty early return: selector focus/pins must see new
    # hypothesis/evidence edges even on a clean turn.
    index.project_micrograph(hypothesis_tracker)
    dirty_refs = set(index.dirty_evidence_refs)
    if not dirty_refs:
        return
    requirements = getattr(runtime, "requirements_ledger", None)
    if requirements is None:
        requirements = getattr(runtime, "_requirements_ledger", None)
    if isinstance(requirements, Mapping):
        try:
            from kernel_requirements_ledger import reopen_dirty_requirements
            updated, notices = reopen_dirty_requirements(requirements, index)
            setattr(runtime, "requirements_ledger", updated)
            setattr(runtime, "_invalidation_requirement_notices", notices)
        except (ImportError, TypeError, ValueError):
            pass
    graph = getattr(runtime, "obligation_graph", None)
    if graph is None:
        graph = getattr(runtime, "_obligation_graph", None)
    if graph is not None:
        try:
            from kernel_obligations import reopen_dirty_obligations
            updated = reopen_dirty_obligations(graph, index)
            setattr(runtime, "obligation_graph", updated)
            setattr(runtime, "_obligation_graph", updated)
        except (ImportError, TypeError, ValueError):
            pass
    if hypothesis_tracker is not None:
        mark_path_stale = getattr(hypothesis_tracker, "mark_path_stale", None)
        if callable(mark_path_stale):
            for node in index.dirty_nodes:
                path = _text(getattr(node, "path", ""))
                if path:
                    mark_path_stale(
                        path,
                        reason=getattr(node, "dirty_reason", REASON_EVIDENCE_STALE),
                        evidence_refs=getattr(node, "evidence_refs", ()),
                    )
        for node in index.dirty_nodes:
            if isinstance(node, ClaimNode):
                mark_stale = getattr(hypothesis_tracker, "mark_stale", None)
                if callable(mark_stale):
                    mark_stale(node.claim_id, reason=node.dirty_reason, evidence_refs=node.evidence_refs)
    reasoning = getattr(runtime, "_f4_reasoning_ledger", None)
    if reasoning is not None:
        mark_stale = getattr(reasoning, "mark_evidence_stale", None)
        if callable(mark_stale):
            mark_stale(paths=tuple(_text(getattr(node, "path", "")) for node in index.dirty_nodes))
    recall = getattr(runtime, "recall_store", None)
    if recall is not None:
        mark_stale = getattr(recall, "mark_stale", None)
        if callable(mark_stale):
            for ref in dirty_refs:
                mark_stale(ref, reason=REASON_EVIDENCE_STALE)


def sync_runtime_invalidation(runtime: object) -> LedgerIndex:
    """Fold all new live receipts into the sole dependency graph."""
    index = runtime_invalidation_index(runtime)
    seen: set[str] = set(getattr(runtime, "_invalidation_seen_receipts", set()))
    for record in _runtime_receipts(runtime):
        ident = _receipt_id(record)
        if ident in seen:
            continue
        enriched = dict(record)
        owner = getattr(runtime, "owner", None)
        root = getattr(owner, "root", None)
        if root:
            for observed_path in _record_paths(enriched):
                candidate = (Path(root) / observed_path).resolve()
                try:
                    if candidate.is_file():
                        enriched.setdefault("digest", hashlib.sha256(candidate.read_bytes()).hexdigest())
                        break
                except OSError:
                    continue
        index.observe_receipt(enriched)
        seen.add(ident)
    setattr(runtime, "_invalidation_seen_receipts", seen)
    _refresh_runtime_consumers(runtime, index)
    decision_graph = getattr(runtime, "decision_graph", None)
    sync_graph = getattr(decision_graph, "sync_ledger", None)
    if callable(sync_graph):
        sync_graph()
    setattr(runtime, "_invalidation_gate", index.completion_gate())
    return index


def runtime_completion_gate(runtime: object) -> dict[str, object]:
    """Enforce clean dependency evidence and rebound decision bases."""
    index = sync_runtime_invalidation(runtime)
    decision_graph = getattr(runtime, "decision_graph", None)
    graph_gate = getattr(decision_graph, "action_gate", None)
    if callable(graph_gate):
        decision_finding = graph_gate("done")
        if isinstance(decision_finding, Mapping) and decision_finding.get("status") == "BLOCKED":
            return dict(decision_finding)
    finding = index.completion_gate()
    cache = getattr(runtime, "decision_cache", None)
    gate = getattr(cache, "action_gate", None)
    if callable(gate):
        decision_finding = gate("done")
        if decision_finding is not None:
            return decision_finding
    return finding


def replay_receipts(
    source: Iterable[Record] | str | Path,
    *,
    index: LedgerIndex | None = None,
) -> LedgerIndex:
    """Replay a JSONL receipt tape or mapping iterable into a ``LedgerIndex``."""

    target = index or LedgerIndex()
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8") as handle:
            records: Iterable[Record] = (
                json.loads(line)
                for line in handle
                if line.strip()
            )
    else:
        records = source
    for record in records:
        if isinstance(record, Mapping):
            target.observe_receipt(record)
    return target


replay_jsonl = replay_receipts


__all__ = [
    "AdvisoryNode",
    "CENSUS_SPECS",
    "COMPLETION_BLOCK_REASON",
    "ClaimNode",
    "Edge",
    "InvalidationDecision",
    "InvalidationResult",
    "LedgerIndex",
    "NodeKind",
    "NodeState",
    "ObservationNode",
    "REASON_CALL_GRAPH_DEPENDENCY",
    "REASON_EVIDENCE_STALE",
    "REASON_REOBSERVED",
    "REASON_SOURCE_EDIT",
    "SCHEMA",
    "runtime_completion_gate",
    "runtime_invalidation_index",
    "sync_runtime_invalidation",
    "replay_jsonl",
    "replay_receipts",
]
