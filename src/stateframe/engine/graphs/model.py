"""graphs/model.py — the shared fact format every layer emits.

One record shape for nodes, one for edges, one scan receipt, one freshness
stamp. Every field traces to a_graph_layers.md's "Shared layer fact format"
(line ~63) or a_graph_incremental.md's "Persistent graph records" (line ~25).

Deliberate simplifications from the source docs, stated rather than hidden:

  * ``snapshot_id`` here is the CONTENT TREE DIGEST of the scanned corpus, not a
    commit sha. The store is content-keyed (a rebuild of an unchanged tree hits
    the same directory), and a dirty working tree still gets an honest identity.
    The commit sha, when known, rides along as ``KnowledgeCut.history_watermark``.
  * There is no UUID entity registry and no lineage/tombstone table in v1. IDs
    are deterministic functions of (repo, kind, name) — so a RENAME mints a new
    ID rather than carrying the old one. That is a KNOWN GAP against
    a_graph_layers' identity-lineage requirement (line ~840), recorded in
    ``KNOWN_GAPS`` below rather than papered over: v1 is correct-but-coarse on
    rename, never silently-wrong.
  * ``confidence`` is an integer in BASIS POINTS (0..10000), never a float, so
    two builds of the same tree compare exactly.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

_POC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _POC not in sys.path:
    sys.path.insert(0, _POC)

from detreason.layer_registry import ALL_LAYERS, CORE_SIX, CanonicalLayer  # noqa: E402,F401

#: Confidence is basis points. 10000 == certain (a fact read directly off the
#: AST); anything inferred says so with a lower value and an evidence ref.
CONF_CERTAIN = 10000
CONF_LIKELY = 8000
CONF_WEAK = 5000

#: Resolution vocabulary, mirroring seam_graph/seam_adapter rather than
#: inventing a third one (#211).
RESOLVED = "resolved"
PARTIAL = "partial"
UNRESOLVED = "unresolved"
RESOLUTIONS = (RESOLVED, PARTIAL, UNRESOLVED)

#: Documented, deliberate gaps of graph-store v1 against the source designs.
#: Read by ``build.py`` into every receipt so a consumer can never mistake v1's
#: scope for the full design (rule #218: a scan certifies its own observation).
KNOWN_GAPS: Tuple[str, ...] = (
    "no-entity-lineage: a rename mints a new entity id (a_graph_layers line ~840)",
    "no-tombstones: deleted entities vanish from the next build rather than "
    "being tombstoned (a_graph_layers invariant 4)",
    "dataflow-intraprocedural-only: no interprocedural summaries or taint "
    "(a_graph_layers L2 C2/C3 modes)",
    "runtime-layer-absent: RUNTIME (deploy/ownership) is not built in v1",
    "history-sampled: co-change is computed over a bounded commit window, "
    "never whole history (a_graph_layers L5 C3)",
)


class Freshness(enum.Enum):
    """a_graph_incremental.md query-consistency states. Never inferred from the
    absence of a dirty flag — a layer that was not built is UNAVAILABLE, not
    CURRENT."""

    CURRENT = "current"
    STALE = "stale"
    REBUILDING = "rebuilding"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Evidence:
    """Where a fact came from. ``path:line`` mirrors seam_graph's Evidence."""

    path: str
    line: int = 0

    def __str__(self) -> str:
        return f"{self.path}:{self.line}"


@dataclass(frozen=True)
class Node:
    """A layer's attachment to a canonical entity.

    ``entity_id`` is ALWAYS a canonical ID from ``graphs.ids`` — layer-local
    concepts (an SSA value, a token posting) still carry the owning entity here,
    per a_graph_layers invariant 11.
    """

    layer: str
    entity_id: str
    kind: str
    evidence: Optional[Evidence] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> Tuple[str, str]:
        return (self.layer, self.entity_id)


@dataclass(frozen=True)
class Edge:
    """A typed, evidenced relation between two canonical entities."""

    layer: str
    src: str
    relation: str
    dst: str
    resolution: str = RESOLVED
    confidence: int = CONF_CERTAIN
    evidence: Optional[Evidence] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.resolution not in RESOLUTIONS:
            raise ValueError(
                f"Edge.resolution must be one of {RESOLUTIONS}, got {self.resolution!r}"
            )
        if not (0 <= self.confidence <= 10000):
            raise ValueError(
                f"Edge.confidence is basis points 0..10000, got {self.confidence!r}"
            )

    def edge_id(self) -> str:
        return f"{self.layer}|{self.src}|{self.relation}|{self.dst}"


@dataclass
class Receipt:
    """Rule #218: a layer build certifies its own observation before its verdict.

    A layer that scanned nothing, parsed nothing, or truncated is VISIBLE here.
    ``truncated`` and ``sampled`` are never silently applied — ``build.py``
    prints them and the store persists them.
    """

    layer: str
    files_seen: int = 0
    files_parsed: int = 0
    files_parse_failed: int = 0
    nodes: int = 0
    edges: int = 0
    unresolved_edges: int = 0
    partial_edges: int = 0
    truncated: bool = False
    sampled: bool = False
    sample_note: str = ""
    id_collisions: int = 0
    build_seconds: float = 0.0
    notes: Tuple[str, ...] = ()

    @property
    def is_vacuous(self) -> bool:
        """The silent-empty-run shape. Mirrors ``seam_adapter.CoverageSnapshot``."""
        return self.files_seen == 0 or (self.files_seen > 0 and self.files_parsed == 0)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["notes"] = list(self.notes)
        d["is_vacuous"] = self.is_vacuous
        return d


@dataclass(frozen=True)
class KnowledgeCut:
    """a_graph_layers.md line ~1006: every cross-layer query declares its cut.

    ``code_snapshot_id`` is the tree digest; the other fields say WHICH version of
    each extractor produced the layers being joined. A join across two different
    cuts is refused by ``graphs.join`` rather than silently mixed.
    """

    code_snapshot_id: str
    history_watermark: str = ""
    extractor_version: str = "graphs/v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerGraph:
    """One materialized layer: nodes + edges + its own receipt and freshness."""

    layer: str
    cut: KnowledgeCut
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    receipt: Receipt = None  # type: ignore[assignment]
    freshness: Freshness = Freshness.UNKNOWN

    def __post_init__(self) -> None:
        if self.receipt is None:
            self.receipt = Receipt(layer=self.layer)

    def finalize(self) -> "LayerGraph":
        """Stamp counts onto the receipt. Idempotent."""
        self.receipt.nodes = len(self.nodes)
        self.receipt.edges = len(self.edges)
        # Counted SEPARATELY: an unresolved edge is a fact we could not bind at
        # all; a partial edge is one bound at a coarser grain or without a proof.
        # Collapsing them into one number hides which kind of uncertainty a layer
        # carries, which is the whole thing the resolution vocabulary is for.
        self.receipt.unresolved_edges = sum(
            1 for e in self.edges if e.resolution == UNRESOLVED
        )
        self.receipt.partial_edges = sum(
            1 for e in self.edges if e.resolution == PARTIAL
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "cut": self.cut.to_dict(),
            "freshness": self.freshness.value,
            "receipt": self.receipt.to_dict(),
            "nodes": [_node_dict(n) for n in self.nodes],
            "edges": [_edge_dict(e) for e in self.edges],
        }


def _node_dict(n: Node) -> Dict[str, Any]:
    return {
        "layer": n.layer,
        "entity_id": n.entity_id,
        "kind": n.kind,
        "evidence": [n.evidence.path, n.evidence.line] if n.evidence else None,
        "attrs": n.attrs,
    }


def _edge_dict(e: Edge) -> Dict[str, Any]:
    return {
        "layer": e.layer,
        "src": e.src,
        "relation": e.relation,
        "dst": e.dst,
        "resolution": e.resolution,
        "confidence": e.confidence,
        "evidence": [e.evidence.path, e.evidence.line] if e.evidence else None,
        "attrs": e.attrs,
    }


def node_from_dict(d: Dict[str, Any]) -> Node:
    ev = d.get("evidence")
    return Node(
        layer=d["layer"],
        entity_id=d["entity_id"],
        kind=d["kind"],
        evidence=Evidence(ev[0], ev[1]) if ev else None,
        attrs=dict(d.get("attrs") or {}),
    )


def edge_from_dict(d: Dict[str, Any]) -> Edge:
    ev = d.get("evidence")
    return Edge(
        layer=d["layer"],
        src=d["src"],
        relation=d["relation"],
        dst=d["dst"],
        resolution=d.get("resolution", RESOLVED),
        confidence=d.get("confidence", CONF_CERTAIN),
        evidence=Evidence(ev[0], ev[1]) if ev else None,
        attrs=dict(d.get("attrs") or {}),
    )


def canonical_json(obj: Any) -> str:
    """Sorted-key, tight-separator JSON. The ONE serializer, so a digest over a
    layer is reproducible across runs and machines."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:16]


def tree_digest(files: Iterable[Tuple[str, str]]) -> str:
    """Content digest over ``(rel_path, source)`` pairs. Order-independent:
    pairs are sorted before hashing, so two walks of the same tree agree."""
    h = hashlib.sha256()
    for path, src in sorted(files):
        h.update(path.encode("utf-8"))
        h.update(b"\x00")
        h.update(hashlib.sha256(src.encode("utf-8", "replace")).digest())
        h.update(b"\x00")
    return h.hexdigest()[:16]


__all__ = [
    "ALL_LAYERS",
    "CORE_SIX",
    "CONF_CERTAIN",
    "CONF_LIKELY",
    "CONF_WEAK",
    "CanonicalLayer",
    "Edge",
    "Evidence",
    "Freshness",
    "KNOWN_GAPS",
    "KnowledgeCut",
    "LayerGraph",
    "Node",
    "PARTIAL",
    "RESOLUTIONS",
    "RESOLVED",
    "Receipt",
    "UNRESOLVED",
    "canonical_json",
    "digest_of",
    "edge_from_dict",
    "node_from_dict",
    "tree_digest",
]
