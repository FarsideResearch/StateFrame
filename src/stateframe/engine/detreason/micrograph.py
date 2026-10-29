"""Small fact graph built from tool receipts: a bounded, deterministic task micro-graph.

This module deliberately has no model, database, fixer, or ``get_sure`` import
on its critical path.  Callers supply structured observations and, later, an
authorized action set.  The graph owns canonical identity, bounded materialized
state, deterministic replay, and the small hypothesis/discrimination consumer.

The extractor contracts are intentionally adapter-shaped:

* :func:`traceback_signal_to_delta` consumes the plain dict returned by
  ``tracebacks.traceback_signal``.
* :func:`certificate_to_delta` consumes one fixer certificate row.
* :func:`scrubbed_symptom_to_delta` consumes already scrubbed symptom data or
  performs conservative local scrubbing with no unconstrained classifier.

Observed fixer certificate fields used by the certificate adapter were verified
against ``results/unified_gym.jsonl`` real rows on 2026-08-29 (for example row
13): top-level ``kind``, ``key``, ``fn``, ``config_fingerprint``, ``verdict``,
``margins``, ``stages``, ``fixed``, ``gold_contact``, ``wall_s``; nested stage
fields include ``class``, ``file``, ``function``, ``mutations``, and
``witness_replay`` counters.  Optional ``fix.line`` and ``fix.change`` are
accepted when a certificate supplies them, but are not required by the real
row contract.  The adapter only cites these observed fields and treats absent
fields as unknown.

The future ``detreason/get_sure.py`` action-selection seam is
:func:`select_get_sure_action`; it is advisory and returns the exact baseline
action on every ineligible/error path.  It does not edit or call ``get_sure``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence


SCHEMA = "stateframe.micrograph/1"
EXTRACTOR_VERSION = "1"


class EntityKind(str, Enum):
    FUNCTION = "function"
    VALUE = "value"
    LINE = "line"
    CLAIM = "claim"
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    SYMPTOM_TERM = "symptom-term"


class EdgeKind(str, Enum):
    CALLS = "calls"
    FLOWS_INTO = "flows-into"
    EXPLAINS = "explains"
    CONTRADICTS = "contradicts"
    DISCRIMINATES = "discriminates"
    SUSPECTS = "suspects"


NODE_KINDS = frozenset(item.value for item in EntityKind)
EDGE_KINDS = frozenset(item.value for item in EdgeKind)


def _as_kind(value: EntityKind | str) -> str:
    text = value.value if isinstance(value, Enum) else str(value)
    if text not in NODE_KINDS:
        raise ValueError(f"unsupported entity kind: {text!r}")
    return text


def _as_edge_kind(value: EdgeKind | str) -> str:
    text = value.value if isinstance(value, Enum) else str(value)
    if text not in EDGE_KINDS:
        raise ValueError(f"unsupported edge kind: {text!r}")
    return text


def _clean_text(value: object, limit: int = 256) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    return " ".join(text.split())[:limit]


def _canonical_key(kind: str, value: object) -> str:
    text = _clean_text(value, 512)
    if kind in {EntityKind.FUNCTION.value, EntityKind.LINE.value}:
        text = text.replace("\\", "/")
        text = re.sub(r"/{2,}", "/", text)
    if kind == EntityKind.SYMPTOM_TERM.value:
        text = text.casefold()
    return text


def _json_value(value: object) -> object:
    """Return bounded JSON-safe data without invoking arbitrary ``repr``."""
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float is not serializable")
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        items = [_json_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(items, key=_canonical_json)
        return items
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return _json_value(value.to_dict())
    return {"type": type(value).__name__}


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}:{_sha(value)}"


def _merge_values(left: object, right: object) -> object:
    """Commutative deterministic merge used for same-identity node upserts."""
    if left == right:
        return left
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        merged = {str(k): _json_value(v) for k, v in left.items()}
        for key, value in right.items():
            if key in merged:
                merged[str(key)] = _merge_values(merged[key], _json_value(value))
            else:
                merged[str(key)] = _json_value(value)
        return {key: merged[key] for key in sorted(merged)}
    values = []
    for value in (left, right):
        candidate = _json_value(value)
        if candidate not in values:
            values.append(candidate)
    return sorted(values, key=_canonical_json)


def _merge_mapping(left: Mapping[str, object], right: Mapping[str, object]) -> dict:
    merged = {str(k): _json_value(v) for k, v in left.items()}
    for key, value in right.items():
        key_text = str(key)
        if key_text in merged:
            merged[key_text] = _merge_values(merged[key_text], value)
        else:
            merged[key_text] = _json_value(value)
    return {key: merged[key] for key in sorted(merged)}


def _merge_provenance(left: Sequence[Mapping[str, object]], right: Sequence[Mapping[str, object]]) -> tuple:
    values = {_canonical_json(item): _json_value(item) for item in (*left, *right)}
    return tuple(values[key] for key in sorted(values))


def _coerce_provenance(value: object) -> tuple:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (_json_value(value),)
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(_json_value(item) for item in value if isinstance(item, Mapping))
    return ()


@dataclass(frozen=True)
class Node:
    """One immutable typed entity; identity is kind plus canonical key."""

    kind: EntityKind | str
    key: str
    attrs: Mapping[str, object] = field(default_factory=dict)
    provenance: tuple = ()
    id: str = ""

    def __post_init__(self) -> None:
        kind = _as_kind(self.kind)
        key = _canonical_key(kind, self.key)
        attrs = _json_value(self.attrs)
        if not isinstance(attrs, dict):
            raise TypeError("node attrs must be a mapping")
        provenance = _coerce_provenance(self.provenance)
        node_id = self.id or _stable_id("n", {"kind": kind, "key": key})
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "attrs", attrs)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "id", node_id)

    @property
    def priority(self) -> float:
        value = self.attrs.get("priority", 0)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0

    @property
    def pinned(self) -> bool:
        return bool(self.attrs.get("pinned", False))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "canonical_key": self.key,
            "attrs": _json_value(self.attrs),
            "provenance": _json_value(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Node":
        return cls(
            kind=str(value.get("kind", "")),
            key=str(value.get("canonical_key", value.get("key", ""))),
            attrs=value.get("attrs", {}) if isinstance(value.get("attrs", {}), Mapping) else {},
            provenance=value.get("provenance", ()),
            id=str(value.get("id", "")),
        )


@dataclass(frozen=True)
class Edge:
    """One immutable typed relation between two existing node identities."""

    kind: EdgeKind | str
    source: str
    target: str
    attrs: Mapping[str, object] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        kind = _as_edge_kind(self.kind)
        source = _clean_text(self.source, 256)
        target = _clean_text(self.target, 256)
        if not source or not target:
            raise ValueError("edge source and target are required")
        attrs = _json_value(self.attrs)
        if not isinstance(attrs, dict):
            raise TypeError("edge attrs must be a mapping")
        edge_id = self.id or _stable_id(
            "e", {"kind": kind, "source": source, "target": target, "attrs": attrs}
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "attrs", attrs)
        object.__setattr__(self, "id", edge_id)

    @property
    def priority(self) -> float:
        value = self.attrs.get("priority", 0)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0

    @property
    def pinned(self) -> bool:
        return bool(self.attrs.get("pinned", False))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "target": self.target,
            "attrs": _json_value(self.attrs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Edge":
        return cls(
            kind=str(value.get("kind", "")),
            source=str(value.get("source", "")),
            target=str(value.get("target", "")),
            attrs=value.get("attrs", {}) if isinstance(value.get("attrs", {}), Mapping) else {},
            id=str(value.get("id", "")),
        )


@dataclass(frozen=True)
class GraphDelta:
    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "warnings", tuple(_clean_text(item, 512) for item in self.warnings))

    def __add__(self, other: "GraphDelta") -> "GraphDelta":
        if not isinstance(other, GraphDelta):
            return NotImplemented
        return GraphDelta(self.nodes + other.nodes, self.edges + other.edges, self.warnings + other.warnings)


def make_node(kind: EntityKind | str, key: object, attrs: Mapping[str, object] | None = None, provenance: object = ()) -> Node:
    return Node(kind, _canonical_key(_as_kind(kind), key), attrs or {}, _coerce_provenance(provenance))


def make_edge(kind: EdgeKind | str, source: str, target: str, attrs: Mapping[str, object] | None = None) -> Edge:
    return Edge(kind, source, target, attrs or {})


node = make_node
edge = make_edge


_KIND_RETENTION = {
    EntityKind.CLAIM.value: 70,
    EntityKind.EVIDENCE.value: 80,
    EntityKind.FUNCTION.value: 60,
    EntityKind.HYPOTHESIS.value: 90,
    EntityKind.LINE.value: 50,
    EntityKind.SYMPTOM_TERM.value: 40,
    EntityKind.VALUE.value: 30,
}


class MicroGraph:
    """Bounded materialized graph with commutative/idempotent delta upserts."""

    def __init__(
        self,
        workspace_id: str = "",
        case_id: str = "",
        *,
        max_nodes: int = 256,
        max_edges: int = 1024,
        extractor_versions: Mapping[str, str] | None = None,
        created_at: str | None = None,
    ) -> None:
        if max_nodes < 1 or max_edges < 1:
            raise ValueError("graph bounds must be positive")
        self.workspace_id = _clean_text(workspace_id, 256)
        self.case_id = _clean_text(case_id, 256)
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)
        self.extractor_versions = dict(extractor_versions or {})
        self.created_at = _clean_text(created_at, 64) if created_at else None
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes[key] for key in sorted(self._nodes))

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(self._edges[key] for key in sorted(self._edges))

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def edge(self, edge_id: str) -> Edge | None:
        return self._edges.get(edge_id)

    def apply_delta(self, delta: GraphDelta) -> "MicroGraph":
        if not isinstance(delta, GraphDelta):
            raise TypeError("apply_delta expects GraphDelta")
        for incoming in delta.nodes:
            current = self._nodes.get(incoming.id)
            if current is None:
                self._nodes[incoming.id] = incoming
            else:
                self._nodes[incoming.id] = Node(
                    current.kind,
                    current.key,
                    _merge_mapping(current.attrs, incoming.attrs),
                    _merge_provenance(current.provenance, incoming.provenance),
                    current.id,
                )
        for incoming in delta.edges:
            if incoming.source not in self._nodes or incoming.target not in self._nodes:
                raise ValueError(f"edge references missing node: {incoming.id}")
            current = self._edges.get(incoming.id)
            if current is None:
                self._edges[incoming.id] = incoming
            else:
                self._edges[incoming.id] = Edge(
                    current.kind,
                    current.source,
                    current.target,
                    _merge_mapping(current.attrs, incoming.attrs),
                    current.id,
                )
        self._enforce_bounds()
        return self

    def apply(self, delta: GraphDelta) -> "MicroGraph":
        return self.apply_delta(delta)

    def add_node(self, value: Node) -> Node:
        self.apply_delta(GraphDelta((value,)))
        return value

    def add_edge(self, value: Edge) -> Edge:
        self.apply_delta(GraphDelta((), (value,)))
        return value

    def merge(self, *deltas: GraphDelta) -> "MicroGraph":
        for delta in deltas:
            self.apply_delta(delta)
        return self

    def _enforce_bounds(self) -> None:
        ranked_nodes = sorted(
            self._nodes.values(),
            key=lambda item: (
                0 if item.pinned else 1,
                -item.priority,
                -_KIND_RETENTION[item.kind],
                item.id,
            ),
        )
        kept_ids = {item.id for item in ranked_nodes[: self.max_nodes]}
        self._nodes = {key: value for key, value in self._nodes.items() if key in kept_ids}
        self._edges = {
            key: value
            for key, value in self._edges.items()
            if value.source in self._nodes and value.target in self._nodes
        }
        ranked_edges = sorted(
            self._edges.values(),
            key=lambda item: (0 if item.pinned else 1, -item.priority, item.kind, item.id),
        )
        kept_edges = {item.id for item in ranked_edges[: self.max_edges]}
        self._edges = {key: value for key, value in self._edges.items() if key in kept_edges}

    def _body_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "workspace_id": self.workspace_id,
            "case_id": self.case_id,
            "created_at": self.created_at,
            "extractor_versions": dict(sorted(self.extractor_versions.items())),
            "bounds": {"max_nodes": self.max_nodes, "max_edges": self.max_edges},
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
        }

    def digest(self) -> str:
        return f"sha256:{hashlib.sha256(_canonical_json(self._body_dict()).encode('utf-8')).hexdigest()}"

    @property
    def snapshot_id(self) -> str:
        return self.digest()

    def to_dict(self) -> dict:
        body = self._body_dict()
        body["snapshot_id"] = self.digest()
        return body

    def serialize(self) -> str:
        return _canonical_json(self.to_dict())

    def to_json(self) -> str:
        return self.serialize()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "MicroGraph":
        if value.get("schema") != SCHEMA:
            raise ValueError("unsupported micrograph schema")
        bounds = value.get("bounds", {})
        bounds_map = bounds if isinstance(bounds, Mapping) else {}
        graph = cls(
            str(value.get("workspace_id", "")),
            str(value.get("case_id", "")),
            max_nodes=int(bounds_map.get("max_nodes", 256)),
            max_edges=int(bounds_map.get("max_edges", 1024)),
            extractor_versions=value.get("extractor_versions", {})
            if isinstance(value.get("extractor_versions", {}), Mapping)
            else {},
            created_at=str(value.get("created_at")) if value.get("created_at") else None,
        )
        nodes = value.get("nodes", ())
        if not isinstance(nodes, (tuple, list)):
            raise ValueError("nodes must be a list")
        for raw_node in nodes:
            if not isinstance(raw_node, Mapping):
                raise ValueError("node entry must be an object")
            graph._nodes[Node.from_dict(raw_node).id] = Node.from_dict(raw_node)
        edges = value.get("edges", ())
        if not isinstance(edges, (tuple, list)):
            raise ValueError("edges must be a list")
        for raw_edge in edges:
            if not isinstance(raw_edge, Mapping):
                raise ValueError("edge entry must be an object")
            item = Edge.from_dict(raw_edge)
            if item.source not in graph._nodes or item.target not in graph._nodes:
                raise ValueError("edge references missing node")
            graph._edges[item.id] = item
        graph._enforce_bounds()
        expected = value.get("snapshot_id")
        if expected is not None and expected != graph.snapshot_id:
            raise ValueError("micrograph snapshot digest mismatch")
        return graph

    @classmethod
    def from_json(cls, value: str) -> "MicroGraph":
        parsed = json.loads(value)
        if not isinstance(parsed, Mapping):
            raise ValueError("micrograph JSON must contain an object")
        return cls.from_dict(parsed)

    @classmethod
    def deserialize(cls, value: str) -> "MicroGraph":
        return cls.from_json(value)


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _case_id(source: Mapping[str, object], explicit: str | None) -> str:
    return _clean_text(explicit if explicit is not None else source.get("case_id", "default"), 256) or "default"


def _source_hash(value: object, fallback: object) -> str:
    candidate = _get(value, "input_sha256") or _get(value, "source_hash") or fallback
    return _clean_text(candidate, 128)


def _bounded_attrs(value: object, keys: Iterable[str]) -> dict:
    result = {}
    for key in keys:
        candidate = _get(value, key, None)
        if candidate is not None:
            if isinstance(candidate, (str, int, bool, float)) or candidate is None:
                result[key] = _json_value(candidate)
            elif isinstance(candidate, (tuple, list, set, frozenset)):
                result[key] = [_json_value(item) for item in list(candidate)[:64]]
    return result


def _value_node(value: object, *, role: str = "value", provenance: object = ()) -> Node:
    safe = _json_value(value)
    key = _canonical_json(safe)
    return make_node(EntityKind.VALUE, key, {"role": role, "value": safe}, provenance)


def _claim_node(predicate: str, value: object, *, attrs: Mapping[str, object] | None = None, provenance: object = ()) -> Node:
    claim_attrs = {"predicate": predicate, "value": _json_value(value)}
    claim_attrs.update(attrs or {})
    return make_node(EntityKind.CLAIM, f"{predicate}={_canonical_json(value)}", claim_attrs, provenance)


def _evidence_node(source_kind: str, source_id: str, attrs: Mapping[str, object], provenance: object = ()) -> Node:
    return make_node(
        EntityKind.EVIDENCE,
        f"{source_kind}:{source_id}",
        {"source_kind": source_kind, "source_id": source_id, **dict(attrs)},
        provenance,
    )


def _wire_claim(nodes: list[Node], edges: list[Edge], source: Node, claim: Node) -> None:
    nodes.append(claim)
    edges.append(make_edge(EdgeKind.EXPLAINS, source.id, claim.id))
    value = _value_node(claim.attrs.get("value"), role="claim-value", provenance=claim.provenance)
    nodes.append(value)
    edges.append(make_edge(EdgeKind.FLOWS_INTO, claim.id, value.id))


def _anchor_nodes(anchor: Mapping[str, object], source: Node, nodes: list[Node], edges: list[Edge]) -> tuple[Node | None, Node | None]:
    repo_path = _clean_text(anchor.get("repo_path") or anchor.get("path") or "", 512)
    line_value = anchor.get("line")
    function_name = _clean_text(anchor.get("function") or "", 256)
    function = make_node(EntityKind.FUNCTION, _function_key(repo_path, function_name), {"name": function_name, "repo_path": repo_path, "anchor": anchor.get("anchor")}) if function_name else None
    line = None
    if repo_path and isinstance(line_value, int) and line_value > 0:
        line = make_node(EntityKind.LINE, f"{repo_path}:{line_value}", {"repo_path": repo_path, "line": line_value})
    if function:
        nodes.append(function)
        edges.append(make_edge(EdgeKind.SUSPECTS, source.id, function.id, {"authority": anchor.get("authority", "unknown")}))
    if line:
        nodes.append(line)
        edges.append(make_edge(EdgeKind.SUSPECTS, source.id, line.id, {"tier": anchor.get("tier")}))
        if function:
            edges.append(make_edge(EdgeKind.FLOWS_INTO, function.id, line.id))
    return function, line


def _function_key(repo_path: object, function_name: object) -> str:
    path = _clean_text(repo_path, 512)
    name = _clean_text(function_name, 256)
    return f"{path}::{name}" if path and name else name or path


def traceback_signal_to_delta(signal: Mapping[str, object], case_id: str | None = None) -> GraphDelta:
    """Convert the verified ``tracebacks.traceback_signal`` dict to a delta."""
    if not isinstance(signal, Mapping):
        raise TypeError("traceback signal must be a mapping")
    case = _case_id(signal, case_id)
    evidence_obj = signal.get("evidence")
    source_hash = _source_hash(signal, _sha({key: signal.get(key) for key in sorted(signal) if key != "evidence"}))
    source_id = f"{case}:{source_hash}"
    provenance = ({"source_kind": "traceback", "source_id": source_id, "extractor": "traceback_signal", "version": EXTRACTOR_VERSION, "source_hash": source_hash},)
    attrs = _bounded_attrs(
        signal,
        ("status", "confidence", "primary_anchor", "family", "families", "route_id", "exception_class", "tokens", "route_tokens", "bm25_allowed", "automatic_patch_allowed", "refusal_reason", "prohibited_assumptions", "repo_paths"),
    )
    evidence = _evidence_node("traceback", source_id, attrs, provenance)
    nodes = [evidence]
    edges: list[Edge] = []
    anchor_items = signal.get("anchors", ())
    if isinstance(anchor_items, (tuple, list)):
        for raw_anchor in anchor_items[:128]:
            if isinstance(raw_anchor, Mapping):
                _anchor_nodes(raw_anchor, evidence, nodes, edges)
    claim_specs = [
        ("traceback.status", signal.get("status")),
        ("traceback.family", signal.get("family")),
        ("traceback.route_id", signal.get("route_id")),
        ("traceback.exception_class", signal.get("exception_class")),
        ("traceback.primary_anchor", signal.get("primary_anchor")),
    ]
    for predicate, value in claim_specs:
        if value is not None:
            _wire_claim(nodes, edges, evidence, _claim_node(predicate, value, provenance=provenance))
    tokens = signal.get("tokens", ())
    if isinstance(tokens, (tuple, list, set, frozenset)):
        for token in sorted({_clean_text(item, 128).casefold() for item in tokens if item})[:128]:
            term = make_node(EntityKind.SYMPTOM_TERM, token, {"term": token, "grounded": True, "source": "traceback"}, provenance)
            nodes.append(term)
            edges.append(make_edge(EdgeKind.EXPLAINS, evidence.id, term.id))
    route_tokens = signal.get("route_tokens", ())
    if isinstance(route_tokens, (tuple, list, set, frozenset)):
        for token in sorted({_clean_text(item, 128).casefold() for item in route_tokens if item})[:128]:
            term = make_node(EntityKind.SYMPTOM_TERM, f"route:{token}", {"term": token, "grounded": False, "source": "traceback-route"}, provenance)
            nodes.append(term)
            edges.append(make_edge(EdgeKind.EXPLAINS, evidence.id, term.id, {"grounded": False}))
    if evidence_obj is not None:
        records = _get(evidence_obj, "tracebacks", ())
        if isinstance(records, (tuple, list)):
            previous_function: Node | None = None
            for record in records[:128]:
                frames = _get(record, "frames", ())
                if not isinstance(frames, (tuple, list)):
                    continue
                for frame in frames[:2048]:
                    path = _clean_text(_get(frame, "repo_path") or _get(frame, "file_normalized") or "", 512)
                    line_number = _get(frame, "line")
                    function_name = _clean_text(_get(frame, "function") or "", 256)
                    current_function = None
                    if function_name:
                        current_function = make_node(EntityKind.FUNCTION, _function_key(path, function_name), {"name": function_name, "repo_path": path}, provenance)
                        nodes.append(current_function)
                        edges.append(make_edge(EdgeKind.SUSPECTS, evidence.id, current_function.id, {"source": "traceback-frame"}))
                    if path and isinstance(line_number, int) and line_number > 0:
                        line_node = make_node(EntityKind.LINE, f"{path}:{line_number}", {"repo_path": path, "line": line_number}, provenance)
                        nodes.append(line_node)
                        edges.append(make_edge(EdgeKind.SUSPECTS, evidence.id, line_node.id, {"source": "traceback-frame"}))
                        if current_function:
                            edges.append(make_edge(EdgeKind.FLOWS_INTO, current_function.id, line_node.id))
                    if previous_function and current_function and previous_function.id != current_function.id:
                        edges.append(make_edge(EdgeKind.CALLS, previous_function.id, current_function.id))
                    if current_function:
                        previous_function = current_function
    return GraphDelta(tuple(nodes), tuple(edges), ())


traceback_to_graph_delta = traceback_signal_to_delta
extract_traceback_signal = traceback_signal_to_delta


_CERT_TOP_FIELDS = ("kind", "key", "fn", "config_fingerprint", "verdict", "fixed", "gold_contact", "wall_s", "run_at")
_CERT_STAGE_FIELDS = ("class", "class_by", "file", "file_by", "function", "function_by", "mutations", "line", "change")
_CERT_WITNESS_FIELDS = ("candidates", "survivors", "supported", "symptom_class", "eliminated")


def _certificate_payload(certificate: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("certificate", "cert"):
        nested = certificate.get(key)
        if isinstance(nested, Mapping):
            merged = dict(certificate)
            merged.update(nested)
            return merged
    return certificate


def certificate_to_delta(certificate: Mapping[str, object], case_id: str | None = None) -> GraphDelta:
    """Convert one fixer certificate row into evidence, claims, and suspects.

    ``results/unified_gym.jsonl`` is a row ledger rather than a nested cert
    envelope, so this accepts both that observed shape and a future wrapper.
    ``options`` and arbitrary model text are deliberately not copied.
    """
    if not isinstance(certificate, Mapping):
        raise TypeError("certificate must be a mapping")
    payload = _certificate_payload(certificate)
    case = _case_id(payload, case_id)
    source_hash = _sha(payload)
    source_id = f"{case}:{source_hash}"
    provenance = ({"source_kind": "fixer_certificate", "source_id": source_id, "extractor": "certificate", "version": EXTRACTOR_VERSION, "source_hash": source_hash},)
    verdict = payload.get("verdict")
    validity = str(payload.get("validity", "")) or ("invalid" if str(verdict).upper() == "INVALID" else "valid")
    attrs = {"validity": validity}
    for key in _CERT_TOP_FIELDS:
        if payload.get(key) is not None:
            attrs[key] = _json_value(payload.get(key))
    evidence = _evidence_node("fixer_certificate", source_id, attrs, provenance)
    nodes = [evidence]
    edges: list[Edge] = []
    claim_values = [(f"certificate.{key}", payload.get(key)) for key in _CERT_TOP_FIELDS if payload.get(key) is not None]
    margins = payload.get("margins")
    if isinstance(margins, (tuple, list)):
        claim_values.append(("certificate.margins.count", len(margins)))
        for index, margin in enumerate(margins[:64]):
            if isinstance(margin, Mapping):
                if margin.get("menu") is not None:
                    claim_values.append((f"certificate.margin.{index}.menu", margin.get("menu")))
                if margin.get("pick") is not None:
                    claim_values.append((f"certificate.margin.{index}.pick", _clean_text(margin.get("pick"), 256)))
                if margin.get("margin") is not None:
                    claim_values.append((f"certificate.margin.{index}.margin", margin.get("margin")))
    stages = payload.get("stages")
    function_nodes: list[Node] = []
    line_nodes: list[Node] = []
    if isinstance(stages, (tuple, list)):
        claim_values.append(("certificate.stages.count", len(stages)))
        for index, stage in enumerate(stages[:128]):
            if not isinstance(stage, Mapping):
                continue
            for key in _CERT_STAGE_FIELDS:
                if stage.get(key) is not None:
                    claim_values.append((f"certificate.stage.{index}.{key}", stage.get(key)))
            function_name = stage.get("function")
            file_name = stage.get("file")
            if function_name:
                function = make_node(EntityKind.FUNCTION, _function_key(file_name, function_name), {"name": _clean_text(function_name, 256), "repo_path": file_name or "", "source": "certificate.stage"}, provenance)
                nodes.append(function)
                function_nodes.append(function)
                edges.append(make_edge(EdgeKind.SUSPECTS, evidence.id, function.id, {"stage": index}))
            replay = stage.get("witness_replay")
            if isinstance(replay, Mapping):
                for key in _CERT_WITNESS_FIELDS:
                    if replay.get(key) is not None:
                        claim_values.append((f"certificate.stage.{index}.witness_replay.{key}", replay.get(key)))
            fix = stage.get("fix")
            if isinstance(fix, Mapping):
                for key in ("line", "change"):
                    if fix.get(key) is not None:
                        claim_values.append((f"certificate.stage.{index}.fix.{key}", fix.get(key)))
                line_number = fix.get("line")
                if isinstance(line_number, int) and line_number > 0 and file_name:
                    line = make_node(EntityKind.LINE, f"{file_name}:{line_number}", {"repo_path": file_name, "line": line_number, "source": "certificate.fix"}, provenance)
                    nodes.append(line)
                    line_nodes.append(line)
                    edges.append(make_edge(EdgeKind.SUSPECTS, evidence.id, line.id, {"stage": index}))
    for predicate, value in claim_values:
        if value is not None:
            _wire_claim(nodes, edges, evidence, _claim_node(predicate, value, provenance=provenance))
    for function in function_nodes:
        for line in line_nodes:
            if function.attrs.get("repo_path") == line.attrs.get("repo_path"):
                edges.append(make_edge(EdgeKind.FLOWS_INTO, function.id, line.id))
    return GraphDelta(tuple(nodes), tuple(edges), ())


fixer_certificate_to_delta = certificate_to_delta
extract_certificate = certificate_to_delta


_SYMPTOM_PHRASES = (
    ("missing user_id", "missing_identifier"),
    ("missing key", "missing_identifier"),
    ("lookup fails", "lookup_failure"),
    ("lookup failure", "lookup_failure"),
    ("timed out", "timeout"),
    ("timeout", "timeout"),
    ("permission denied", "permission_denied"),
    ("not authorized", "permission_denied"),
    ("schema mismatch", "schema_mismatch"),
    ("wrong type", "type_mismatch"),
    ("null", "null_value"),
)
_SECRET_RE = re.compile(r"(?i)(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|https?://\S+|\b[0-9a-f]{24,}\b|\b[0-9]{8,}\b)")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{2,}")
_STOP_WORDS = frozenset({"the", "and", "after", "with", "from", "this", "that", "request", "fails", "error"})


def _scrub(text: object) -> str:
    return _SECRET_RE.sub("<redacted>", _clean_text(text, 1024))


def _symptom_terms(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    known = {term for phrase, term in _SYMPTOM_PHRASES if phrase in lowered}
    words = {word.casefold() for word in _WORD_RE.findall(text) if word.casefold() not in _STOP_WORDS}
    return tuple(sorted(known | words))


def scrubbed_symptom_to_delta(symptom: Mapping[str, object] | str, case_id: str | None = None) -> GraphDelta:
    """Convert scrubbed symptom prose/fields into bounded term evidence."""
    if isinstance(symptom, str):
        source = {"text": symptom}
    elif isinstance(symptom, Mapping):
        source = symptom
    else:
        raise TypeError("symptom must be text or a mapping")
    case = _case_id(source, case_id)
    raw_text = _scrub(source.get("text", ""))
    structured_keys = ("code", "service", "severity", "expected", "actual", "polarity")
    structured = {key: _clean_text(source[key], 256) for key in structured_keys if source.get(key) is not None}
    source_terms = source.get("terms", source.get("tokens", source.get("symptom_terms", ())))
    if isinstance(source_terms, (tuple, list, set, frozenset)):
        terms = {_clean_text(item, 128).casefold() for item in source_terms if item}
        grounded = True
    else:
        terms = set()
        grounded = False
    terms.update(_symptom_terms(raw_text))
    if not terms:
        terms.add(f"unknown:{hashlib.sha256(raw_text.encode('utf-8')).hexdigest()[:24]}")
    source_id = f"{case}:{hashlib.sha256(_canonical_json({'structured': structured, 'terms': sorted(terms), 'text': raw_text}).encode('utf-8')).hexdigest()}"
    provenance = ({"source_kind": "symptom", "source_id": source_id, "extractor": "scrubbed_symptom", "version": EXTRACTOR_VERSION, "dictionary_version": "1", "source_hash": hashlib.sha256(raw_text.encode('utf-8')).hexdigest()},)
    evidence = _evidence_node("symptom", source_id, {"structured": structured, "text_fingerprint": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(), "term_count": len(terms)}, provenance)
    nodes = [evidence]
    edges: list[Edge] = []
    for key, value in sorted(structured.items()):
        _wire_claim(nodes, edges, evidence, _claim_node(f"symptom.{key}", value, attrs={"source": "structured"}, provenance=provenance))
    for term in sorted(terms)[:128]:
        is_known = term in {normalized for _, normalized in _SYMPTOM_PHRASES}
        term_node = make_node(EntityKind.SYMPTOM_TERM, term, {"term": term, "grounded": grounded or is_known, "dictionary_version": "1"}, provenance)
        nodes.append(term_node)
        edges.append(make_edge(EdgeKind.EXPLAINS, evidence.id, term_node.id, {"grounded": grounded or is_known}))
    return GraphDelta(tuple(nodes), tuple(edges), ())


symptom_to_graph_delta = scrubbed_symptom_to_delta
extract_scrubbed_symptom = scrubbed_symptom_to_delta


@dataclass(frozen=True)
class Predicate:
    predicate: str
    path: str = ""
    operator: str = "equals"
    value: object = None
    identifier: str = ""

    def __post_init__(self) -> None:
        predicate = _clean_text(self.predicate, 128)
        path = _clean_text(self.path, 256)
        operator = _clean_text(self.operator, 64) or "equals"
        identifier = _clean_text(self.identifier, 256)
        if not identifier:
            identifier = _canonical_json({"predicate": predicate, "path": path, "operator": operator, "value": self.value})
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "value", _json_value(self.value))
        object.__setattr__(self, "identifier", identifier)

    @classmethod
    def from_value(cls, value: object) -> "Predicate":
        if isinstance(value, Predicate):
            return value
        if isinstance(value, Mapping):
            return cls(str(value.get("predicate", value.get("name", ""))), str(value.get("path", "")), str(value.get("operator", value.get("op", "equals"))), value.get("value"), str(value.get("id", value.get("identifier", ""))))
        if isinstance(value, (tuple, list)):
            if len(value) == 2:
                return cls(str(value[0]), value=value[1])
            if len(value) >= 3:
                return cls(str(value[0]), str(value[1]), "equals", value[2])
        return cls(str(value))

    def to_dict(self) -> dict:
        return {"id": self.identifier, "predicate": self.predicate, "path": self.path, "operator": self.operator, "value": self.value}


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    claim: str = ""
    required: tuple = ()
    soft_predictions: tuple = ()
    prior_weight: float = 1.0
    discriminators: tuple = ()
    provenance: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypothesis_id", _clean_text(self.hypothesis_id, 256))
        object.__setattr__(self, "claim", _clean_text(self.claim, 512))
        object.__setattr__(self, "required", tuple(Predicate.from_value(item) for item in self.required))
        object.__setattr__(self, "soft_predictions", tuple(Predicate.from_value(item) for item in self.soft_predictions))
        weight = float(self.prior_weight) if isinstance(self.prior_weight, (int, float)) and math.isfinite(self.prior_weight) else 0.0
        object.__setattr__(self, "prior_weight", max(0.0, weight))
        object.__setattr__(self, "discriminators", tuple(_clean_text(item, 256) for item in self.discriminators))
        object.__setattr__(self, "provenance", _coerce_provenance(self.provenance))

    def to_dict(self) -> dict:
        return {"hypothesis_id": self.hypothesis_id, "claim": self.claim, "required": [item.to_dict() for item in self.required], "soft_predictions": [item.to_dict() for item in self.soft_predictions], "prior_weight": self.prior_weight, "discriminators": list(self.discriminators), "provenance": _json_value(self.provenance)}


HypothesisSpec = Hypothesis


@dataclass(frozen=True)
class ActionCandidate:
    action_id: str
    predictions: Mapping[str, object] = field(default_factory=dict)
    cost: float | None = 0.0
    risk: float | None = 0.0
    reliability: float | None = 1.0
    authorized: bool = True
    reachability: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _clean_text(self.action_id, 256))
        object.__setattr__(self, "predictions", dict(self.predictions))
        for field_name, default in (("cost", 0.0), ("risk", 0.0), ("reliability", 1.0), ("reachability", 1.0)):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                object.__setattr__(self, field_name, None)
            elif value is not None:
                object.__setattr__(self, field_name, float(value))

    @classmethod
    def from_value(cls, value: object) -> "ActionCandidate":
        if isinstance(value, ActionCandidate):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("action must be an ActionCandidate or mapping")
        predictions = value.get("predictions", value.get("expected_outcomes", {}))
        return cls(str(value.get("action_id", value.get("id", ""))), predictions if isinstance(predictions, Mapping) else {}, value.get("cost", 0.0), value.get("risk", 0.0), value.get("reliability", 1.0), bool(value.get("authorized", True)), value.get("reachability", 1.0))


ActionSpec = ActionCandidate


@dataclass(frozen=True)
class HypothesisPosture:
    hypothesis_id: str
    posture: str
    support_evidence_ids: tuple[str, ...] = ()
    contradiction_evidence_ids: tuple[str, ...] = ()
    unknown_predicates: tuple[str, ...] = ()
    conflict_predicates: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


LIVE = "live"
WEAKENED = "weakened"
ELIMINATED = "eliminated"
CONFIRMED = "confirmed"


def _hypothesis_node(hypothesis: Hypothesis) -> Node:
    return make_node(EntityKind.HYPOTHESIS, hypothesis.hypothesis_id, hypothesis.to_dict(), hypothesis.provenance)


def _resolve_hypothesis_id(value: str, hypotheses: Mapping[str, Hypothesis], graph: MicroGraph) -> str | None:
    if value in hypotheses:
        return value
    item = graph.node(value)
    if item and item.kind == EntityKind.HYPOTHESIS.value:
        return str(item.attrs.get("hypothesis_id", item.key))
    for hypothesis_id in hypotheses:
        if value == _hypothesis_node(hypotheses[hypothesis_id]).id:
            return hypothesis_id
    return None


def _fact_matches(predicate: Predicate, node_item: Node | None, edge_item: Edge) -> bool:
    attrs = dict(node_item.attrs if node_item else {})
    attrs.update({key: value for key, value in edge_item.attrs.items() if key not in attrs})
    if predicate.identifier in {str(attrs.get("predicate_id", "")), str(attrs.get("id", ""))}:
        return True
    if predicate.predicate and attrs.get("predicate") not in (None, predicate.predicate):
        return False
    if predicate.path and attrs.get("path") not in (None, predicate.path):
        return False
    if attrs.get("operator") not in (None, predicate.operator):
        return False
    if "value" in attrs and predicate.value is not None and attrs.get("value") != predicate.value:
        return False
    return True


class HypothesisGraphConsumer:
    """Recompute hypothesis postures from graph evidence edges, never arrival order."""

    def __init__(self, graph: MicroGraph | None = None, hypotheses: Iterable[Hypothesis | Mapping[str, object]] = ()) -> None:
        self.graph = graph or MicroGraph()
        self.hypotheses: dict[str, Hypothesis] = {}
        for item in self.graph.nodes:
            if item.kind == EntityKind.HYPOTHESIS.value:
                attrs = dict(item.attrs)
                hypothesis_id = str(attrs.get("hypothesis_id", item.key))
                self.hypotheses[hypothesis_id] = Hypothesis(
                    hypothesis_id,
                    str(attrs.get("claim", "")),
                    tuple(attrs.get("required", ())),
                    tuple(attrs.get("soft_predictions", ())),
                    attrs.get("prior_weight", 1.0),
                    tuple(attrs.get("discriminators", ())),
                    attrs.get("provenance", ()),
                )
        for item in hypotheses:
            self.add_hypothesis(item)

    def add_hypothesis(self, value: Hypothesis | Mapping[str, object]) -> Hypothesis:
        if isinstance(value, Hypothesis):
            hypothesis = value
        elif isinstance(value, Mapping):
            hypothesis = Hypothesis(str(value.get("hypothesis_id", value.get("id", ""))), str(value.get("claim", "")), tuple(value.get("required", ())), tuple(value.get("soft_predictions", ())), value.get("prior_weight", 1.0), tuple(value.get("discriminators", ())), value.get("provenance", ()))
        else:
            raise TypeError("hypothesis must be a Hypothesis or mapping")
        if not hypothesis.hypothesis_id:
            raise ValueError("hypothesis_id is required")
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        self.graph.add_node(_hypothesis_node(hypothesis))
        return hypothesis

    def add_evidence(self, value: Node) -> Node:
        if not isinstance(value, Node) or value.kind not in {EntityKind.EVIDENCE.value, EntityKind.CLAIM.value}:
            raise TypeError("evidence must be a claim/evidence Node")
        self.graph.add_node(value)
        return value

    def add_relation(self, value: Edge) -> Edge:
        if not isinstance(value, Edge):
            raise TypeError("relation must be an Edge")
        self.graph.add_edge(value)
        return value

    def support(self, hypothesis_id: str, evidence: Node | str, **attrs: object) -> Edge:
        return self._relate(EdgeKind.EXPLAINS.value, hypothesis_id, evidence, attrs)

    def contradict(self, hypothesis_id: str, evidence: Node | str, **attrs: object) -> Edge:
        return self._relate(EdgeKind.CONTRADICTS.value, evidence, hypothesis_id, attrs)

    def _relate(self, kind: str, left: str | Node, right: str | Node, attrs: Mapping[str, object]) -> Edge:
        left_id = left.id if isinstance(left, Node) else left
        right_id = right.id if isinstance(right, Node) else right
        hyp_left = _resolve_hypothesis_id(left_id, self.hypotheses, self.graph)
        hyp_right = _resolve_hypothesis_id(right_id, self.hypotheses, self.graph)
        left_id = _hypothesis_node(self.hypotheses[hyp_left]).id if hyp_left else left_id
        right_id = _hypothesis_node(self.hypotheses[hyp_right]).id if hyp_right else right_id
        if self.graph.node(left_id) is None and isinstance(left, Node):
            self.graph.add_node(left)
        if self.graph.node(right_id) is None and isinstance(right, Node):
            self.graph.add_node(right)
        return self.graph.add_edge(make_edge(kind, left_id, right_id, attrs))

    def update_postures(self) -> dict[str, HypothesisPosture]:
        result: dict[str, HypothesisPosture] = {}
        for hypothesis_id in sorted(self.hypotheses):
            hypothesis = self.hypotheses[hypothesis_id]
            support_ids: set[str] = set()
            contradiction_ids: set[str] = set()
            unknown = {item.identifier for item in hypothesis.required}
            conflicts: set[str] = set()
            fact_values: dict[str, set[tuple[str, str]]] = {}
            weakened_reasons: set[str] = set()
            hard_eliminated = False
            for relation in sorted(self.graph.edges, key=lambda item: item.id):
                left_hyp = _resolve_hypothesis_id(relation.source, self.hypotheses, self.graph)
                right_hyp = _resolve_hypothesis_id(relation.target, self.hypotheses, self.graph)
                if left_hyp != hypothesis_id and right_hyp != hypothesis_id:
                    continue
                other_id = relation.target if left_hyp == hypothesis_id else relation.source
                other = self.graph.node(other_id)
                if other is None or other.kind not in {EntityKind.EVIDENCE.value, EntityKind.CLAIM.value}:
                    continue
                validity = str(other.attrs.get("validity", relation.attrs.get("validity", "valid"))).casefold()
                if validity in {"invalid", "stale", "unknown"}:
                    continue
                evidence_id = other.id
                is_soft = bool(relation.attrs.get("soft", False))
                assessment = str(relation.attrs.get("assessment", "")).casefold()
                relation_kind = relation.kind
                matched = [item for item in hypothesis.required if _fact_matches(item, other, relation)]
                if relation_kind == EdgeKind.CONTRADICTS.value or assessment in {"contradiction", "refute", "refutes"}:
                    if is_soft:
                        weakened_reasons.add(f"soft contradiction:{evidence_id}")
                    else:
                        hard_eliminated = True
                    contradiction_ids.add(evidence_id)
                    continue
                if relation_kind != EdgeKind.EXPLAINS.value:
                    continue
                support_ids.add(evidence_id)
                for item in matched:
                    unknown.discard(item.identifier)
                if matched:
                    fact_key = matched[0].identifier
                    fact_value = (str(other.attrs.get("operator", "equals")), _canonical_json(other.attrs.get("value")))
                    fact_values.setdefault(fact_key, set()).add(fact_value)
                if relation.attrs.get("conflict") or other.attrs.get("conflict") or str(other.attrs.get("validity", "")).casefold() == "conflict":
                    conflicts.update(item.identifier for item in matched or hypothesis.required)
            for predicate_id, values in fact_values.items():
                if len(values) > 1:
                    conflicts.add(predicate_id)
            if hard_eliminated:
                posture = ELIMINATED
                reasons = ("valid hard contradiction",)
            elif conflicts:
                posture = WEAKENED
                reasons = ("conflicting evidence",)
            elif weakened_reasons:
                posture = WEAKENED
                reasons = tuple(sorted(weakened_reasons))
            else:
                posture = LIVE
                reasons = ("unknown remains possible",) if unknown else ("consistent evidence",)
            result[hypothesis_id] = HypothesisPosture(
                hypothesis_id=hypothesis_id,
                posture=posture,
                support_evidence_ids=tuple(sorted(support_ids)),
                contradiction_evidence_ids=tuple(sorted(contradiction_ids)),
                unknown_predicates=tuple(sorted(unknown)),
                conflict_predicates=tuple(sorted(conflicts)),
                reasons=tuple(sorted(reasons)),
            )
        return result

    postures = update_postures

    def live_hypotheses(self) -> tuple[Hypothesis, ...]:
        current = self.update_postures()
        return tuple(self.hypotheses[key] for key in sorted(self.hypotheses) if current[key].posture == LIVE)

    def weakened_hypotheses(self) -> tuple[Hypothesis, ...]:
        current = self.update_postures()
        return tuple(self.hypotheses[key] for key in sorted(self.hypotheses) if current[key].posture == WEAKENED)

    def eliminated_hypotheses(self) -> tuple[Hypothesis, ...]:
        current = self.update_postures()
        return tuple(self.hypotheses[key] for key in sorted(self.hypotheses) if current[key].posture == ELIMINATED)


@dataclass(frozen=True)
class ActionScore:
    action_id: str
    utility: float
    information_gain: float
    coverage: float
    split_count: int
    partitions: tuple[tuple[str, tuple[str, ...]], ...] = ()
    eligible: bool = True
    reason: str = ""


@dataclass(frozen=True)
class DiscriminationResult:
    status: str
    selected_action_id: str | None
    ranked: tuple[ActionScore, ...]
    live_hypothesis_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()
    abstain_reason: str | None = None

    @property
    def discrimination(self) -> float:
        for score in self.ranked:
            if score.action_id == self.selected_action_id:
                return score.information_gain
        return 0.0

    def to_dict(self) -> dict:
        return {"status": self.status, "selected_action_id": self.selected_action_id, "ranked": [_json_value(item.__dict__) for item in self.ranked], "live_hypothesis_ids": list(self.live_hypothesis_ids), "evidence_refs": list(self.evidence_refs), "abstain_reason": self.abstain_reason}


def _distribution(value: object) -> dict[str, float] | None:
    if isinstance(value, Mapping):
        if "outcomes" in value and isinstance(value["outcomes"], Mapping):
            value = value["outcomes"]
        result = {}
        for key, probability in value.items():
            if not isinstance(probability, (int, float)) or not math.isfinite(probability) or probability < 0:
                return None
            if probability:
                result[_clean_text(key, 128)] = float(probability)
    elif isinstance(value, (tuple, list, set, frozenset)):
        items = sorted(value, key=_canonical_json) if isinstance(value, (set, frozenset)) else list(value)
        if not items:
            return None
        result = defaultdict(float)
        for item in items:
            if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float)):
                result[_clean_text(item[0], 128)] += float(item[1])
            else:
                result[_clean_text(item, 128)] += 1.0
        result = dict(result)
    else:
        result = {_clean_text(value, 128): 1.0}
    total = sum(result.values())
    if total <= 0 or not math.isfinite(total):
        return None
    return {key: probability / total for key, probability in sorted(result.items())}


def _entropy(distribution: Mapping[str, float]) -> float:
    return -sum(value * math.log(value, 2) for value in distribution.values() if value > 0)


def _action_score(action: ActionCandidate, live: Sequence[Hypothesis]) -> ActionScore:
    live_ids = {item.hypothesis_id for item in live}
    parsed = {key: _distribution(value) for key, value in action.predictions.items() if key in live_ids}
    parsed = {key: value for key, value in parsed.items() if value is not None}
    weights = {item.hypothesis_id: item.prior_weight for item in live if item.hypothesis_id in parsed}
    parsed_weight = sum(weights.values())
    all_weight = sum(item.prior_weight for item in live)
    if parsed_weight <= 0 and parsed:
        weights = {key: 1.0 for key in parsed}
        parsed_weight = float(len(parsed))
    if all_weight <= 0:
        all_weight = float(len(live))
    coverage = sum(weights.values()) / all_weight if all_weight else 0.0
    if not action.authorized:
        return ActionScore(action.action_id, 0.0, 0.0, coverage, 0, (), False, "not authorized")
    if action.cost is None or action.risk is None or action.reliability is None or action.reachability is None or action.cost < 0 or not 0 <= action.risk <= 1 or not 0 <= action.reliability <= 1 or not 0 <= action.reachability <= 1:
        return ActionScore(action.action_id, 0.0, 0.0, coverage, 0, (), False, "invalid policy metadata")
    if len(parsed) < 2:
        return ActionScore(action.action_id, 0.0, 0.0, coverage, 0, (), False, "predictions cover fewer than two live hypotheses")
    normalized_weights = {key: value / parsed_weight for key, value in weights.items()}
    prior_entropy = _entropy(normalized_weights)
    outcome_mass: dict[str, float] = defaultdict(float)
    for hypothesis_id, distribution in parsed.items():
        for outcome, probability in distribution.items():
            outcome_mass[outcome] += normalized_weights[hypothesis_id] * probability
    posterior_entropy = 0.0
    for outcome, mass in outcome_mass.items():
        if mass <= 0:
            continue
        posterior = {}
        for hypothesis_id, distribution in parsed.items():
            contribution = normalized_weights[hypothesis_id] * distribution.get(outcome, 0.0)
            if contribution:
                posterior[hypothesis_id] = contribution / mass
        posterior_entropy += mass * _entropy(posterior)
    information_gain = max(0.0, prior_entropy - posterior_entropy)
    deterministic_labels = {hypothesis_id: next(iter(distribution)) for hypothesis_id, distribution in parsed.items() if len(distribution) == 1}
    partitions_map: dict[str, list[str]] = defaultdict(list)
    for hypothesis_id, outcome in deterministic_labels.items():
        partitions_map[outcome].append(hypothesis_id)
    partitions = tuple((outcome, tuple(sorted(ids))) for outcome, ids in sorted(partitions_map.items()))
    split_count = 0
    ids = sorted(parsed)
    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1 :]:
            left = parsed[left_id]
            right = parsed[right_id]
            tv = 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in set(left) | set(right))
            if tv > 1e-12:
                split_count += 1
    if information_gain <= 1e-12 or split_count == 0:
        return ActionScore(action.action_id, 0.0, information_gain, coverage, split_count, partitions, False, "predictions do not discriminate")
    utility = information_gain * coverage * action.reliability * action.reachability * (1.0 - action.risk) / (1.0 + action.cost)
    return ActionScore(action.action_id, utility, information_gain, coverage, split_count, partitions, True, "")


def discriminate(
    graph: MicroGraph | HypothesisGraphConsumer,
    active_hypotheses: Sequence[Hypothesis | Mapping[str, object] | str] | None,
    authorized_candidate_actions: Sequence[ActionCandidate | Mapping[str, object]],
    decision_policy: Mapping[str, object] | None = None,
) -> DiscriminationResult:
    """Rank injected authorized actions by deterministic information gain."""
    consumer = graph if isinstance(graph, HypothesisGraphConsumer) else HypothesisGraphConsumer(graph)
    if active_hypotheses is None:
        live = list(consumer.live_hypotheses())
    else:
        live = []
        current = consumer.update_postures()
        for value in active_hypotheses:
            if isinstance(value, str):
                item = consumer.hypotheses.get(value) or Hypothesis(value)
            elif isinstance(value, Hypothesis):
                item = value
            elif isinstance(value, Mapping):
                item = Hypothesis(str(value.get("hypothesis_id", value.get("id", ""))), str(value.get("claim", "")), tuple(value.get("required", ())), tuple(value.get("soft_predictions", ())), value.get("prior_weight", 1.0), tuple(value.get("discriminators", ())), value.get("provenance", ()))
            else:
                item = None
            if item and current.get(item.hypothesis_id, HypothesisPosture(item.hypothesis_id, LIVE)).posture == LIVE:
                live.append(item)
    live = sorted({item.hypothesis_id: item for item in live}.values(), key=lambda item: item.hypothesis_id)
    scores = [_action_score(ActionCandidate.from_value(value), live) for value in authorized_candidate_actions]
    policy = decision_policy or {}
    minimum_gain = policy.get("minimum_information_gain", 0.0)
    if not isinstance(minimum_gain, (int, float)) or not math.isfinite(minimum_gain):
        minimum_gain = 0.0
    ranked = tuple(sorted(scores, key=lambda item: (-item.utility, -item.information_gain, -item.coverage, item.action_id)))
    selected = next((item for item in ranked if item.eligible and item.information_gain >= minimum_gain), None)
    refs = tuple(sorted({evidence_id for hypothesis in live for evidence_id in consumer.update_postures().get(hypothesis.hypothesis_id, HypothesisPosture(hypothesis.hypothesis_id, LIVE)).support_evidence_ids}))
    if selected is None:
        reason = "no authorized action separates live hypotheses" if live else "no live hypotheses"
        return DiscriminationResult("abstain", None, ranked, tuple(item.hypothesis_id for item in live), refs, reason)
    return DiscriminationResult("selected", selected.action_id, ranked, tuple(item.hypothesis_id for item in live), refs, None)


discrimination_query = discriminate
rank_discriminating_actions = discriminate


@dataclass(frozen=True)
class GetSureSelection:
    """Serializable advisory result for the future ``get_sure`` adapter."""

    status: str
    selected_action_id: str | None
    source: str
    baseline_action_id: str | None
    graph_snapshot_id: str | None
    decision_confidence: float
    hypotheses_considered: tuple[str, ...]
    discrimination_score: float
    evidence_refs: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()
    fallback_reason: str | None = None

    def to_dict(self) -> dict:
        return _json_value(self.__dict__)


def select_get_sure_action(
    debug_context: Mapping[str, object],
    *,
    graph: MicroGraph | HypothesisGraphConsumer | None,
    hypotheses: Sequence[Hypothesis | Mapping[str, object] | str],
    authorized_actions: Sequence[ActionCandidate | Mapping[str, object]],
    baseline_action_id: str | None,
    feature_enabled: bool = True,
    policy: Mapping[str, object] | None = None,
    model: Callable[[object], object] | None = None,
) -> GetSureSelection:
    """Future ``detreason.get_sure`` selector seam, with fail-safe fallback.

    Contract for later integration: call this after obtaining the immutable
    pre-decision graph snapshot and the kernel's active hypotheses and already
    authorized candidate actions.  ``baseline_action_id`` must be exactly the
    ``U-ledger.next_action`` result.  If the feature is disabled, the graph is
    absent/stale/invalid, no action separates at least two live hypotheses, the
    selected ID is not in the authorized set, or any consumer exception occurs,
    return ``source="u-ledger"`` and the unchanged baseline.  This function
    never authorizes, executes, or mutates an action and never calls ``model``;
    any future model proposal must be an explicitly injected, validated callable
    outside this deterministic selection path.
    """
    context = debug_context if isinstance(debug_context, Mapping) else {}
    context_graph_id = context.get("graph_snapshot_id")
    baseline = baseline_action_id

    def fallback(reason: str) -> GetSureSelection:
        return GetSureSelection("fallback", baseline, "u-ledger", baseline, str(context_graph_id) if context_graph_id else None, 0.0, tuple(sorted(_hypothesis_ids(hypotheses))), 0.0, (), (reason,), reason)

    if not feature_enabled:
        return fallback("feature_disabled")
    if graph is None:
        return fallback("graph_unavailable")
    if isinstance(graph, HypothesisGraphConsumer):
        consumer = graph
    elif isinstance(graph, MicroGraph):
        consumer = HypothesisGraphConsumer(graph)
    else:
        return fallback("invalid_graph")
    try:
        result = discriminate(consumer, hypotheses, authorized_actions, policy)
        authorized_ids = {ActionCandidate.from_value(item).action_id for item in authorized_actions}
        if result.status != "selected":
            return fallback(result.abstain_reason or "discrimination_abstained")
        if result.selected_action_id not in authorized_ids:
            return fallback("action_not_authorized")
        selected_score = next(item for item in result.ranked if item.action_id == result.selected_action_id)
        snapshot_id = consumer.graph.snapshot_id
        if context_graph_id and str(context_graph_id) != snapshot_id:
            return fallback("stale_graph_snapshot")
        return GetSureSelection(
            "selected",
            result.selected_action_id,
            "hypothesis_graph",
            baseline,
            snapshot_id,
            min(1.0, max(0.0, selected_score.information_gain)),
            result.live_hypothesis_ids,
            selected_score.information_gain,
            result.evidence_refs,
            ("HIGH_HYPOTHESIS_SEPARATION",),
            None,
        )
    except Exception:
        return fallback("consumer_exception")


get_sure_action_selector = select_get_sure_action


def _hypothesis_ids(values: Sequence[Hypothesis | Mapping[str, object] | str]) -> tuple[str, ...]:
    result = []
    for value in values:
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, Hypothesis):
            result.append(value.hypothesis_id)
        elif isinstance(value, Mapping):
            result.append(str(value.get("hypothesis_id", value.get("id", ""))))
    return tuple(item for item in result if item)


__all__ = [
    "SCHEMA", "EXTRACTOR_VERSION", "EntityKind", "EdgeKind", "NODE_KINDS", "EDGE_KINDS",
    "Node", "Edge", "GraphDelta", "MicroGraph", "make_node", "make_edge", "node", "edge",
    "traceback_signal_to_delta", "traceback_to_graph_delta", "extract_traceback_signal",
    "certificate_to_delta", "fixer_certificate_to_delta", "extract_certificate",
    "scrubbed_symptom_to_delta", "symptom_to_graph_delta", "extract_scrubbed_symptom",
    "Predicate", "Hypothesis", "HypothesisSpec", "ActionCandidate", "ActionSpec",
    "HypothesisPosture", "HypothesisGraphConsumer", "LIVE", "WEAKENED", "ELIMINATED", "CONFIRMED",
    "ActionScore", "DiscriminationResult", "discriminate", "discrimination_query", "rank_discriminating_actions",
    "GetSureSelection", "select_get_sure_action", "get_sure_action_selector",
]
