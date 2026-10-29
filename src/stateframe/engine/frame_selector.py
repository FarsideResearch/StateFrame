"""Deterministic typed frame selection for the v46 state frame.

The selector deliberately has no language-model or prose relevance path.  Its
inputs are host-owned links, the projected invalidation graph, explicit
version bindings, and (optionally) host decision predicates.  ``LedgerIndex``
is the graph authority; the small adapter here keeps the scoring code usable by
the offline frame lab without importing the runtime.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field


LINK_FIELDS = (
    "requirements",
    "obligations",
    "targets",
    "files",
    "symbols",
    "hypotheses",
    "stages",
)
FOCUS_WEIGHTS = {
    "requirements": 4.0,
    "obligations": 4.0,
    "targets": 4.0,
    "symbols": 3.0,
    "files": 2.0,
    "hypotheses": 2.0,
    "stages": 1.0,
}
RELATION_WEIGHTS = {
    "explicit_binding": 1.0,
    "source": 1.0,
    "support": 1.0,
    "supports": 1.0,
    "explains": 1.0,
    "contradiction": 1.0,
    "contradicts": 1.0,
    "def": 0.8,
    "call": 0.8,
    "calls": 0.8,
    "reference": 0.8,
    "depends_on_path": 0.8,
    "produces": 0.8,
    "containment": 0.4,
    "stage": 0.4,
    "item_binding": 1.0,
}
DEFAULT_GRAPH_HOPS = 4
DEFAULT_GRAPH_DECAY = 0.65
DEFAULT_DISTRACTOR_PENALTY = 0.15
DEFAULT_UNANCHORED_CAP = 0.15
DEFAULT_HYPOTHESIS_CAP = 0.20
OMISSION_SCHEMA = "selector-omission/1"
OMISSION_REASONS = frozenset({
    "budget",
    "low_relevance_score",
    "stale",
    "distractor_cap",
    "duplicate",
})


@dataclass(frozen=True)
class SelectorConfig:
    """Bounded selector policy.  Values are intentionally small and explicit."""

    graph_hops: int = DEFAULT_GRAPH_HOPS
    graph_decay: float = DEFAULT_GRAPH_DECAY
    distractor_penalty: float = DEFAULT_DISTRACTOR_PENALTY
    unanchored_cap: float = DEFAULT_UNANCHORED_CAP
    hypothesis_cap: float = DEFAULT_HYPOTHESIS_CAP
    decision_bonus: float = 8.0


@dataclass(frozen=True)
class SelectorResult:
    items: tuple[dict[str, object], ...]
    omitted_count: int
    used_tokens: int
    pins: tuple[str, ...] = ()
    metrics: dict[str, object] = dataclass_field(default_factory=dict)
    omissions: tuple[dict[str, object], ...] = ()


def _omission(
    item: Mapping[str, object],
    reason: str,
    *,
    score: float | None = None,
    validity: str = "",
    cost: int | None = None,
    detail: str = "",
) -> dict[str, object]:
    """Create the bounded, host-readable explanation for one dropped item."""

    if reason not in OMISSION_REASONS:
        raise ValueError(f"unknown selector omission reason: {reason}")
    record: dict[str, object] = {
        "schema": OMISSION_SCHEMA,
        "item_id": _item_id(item),
        "candidate_id": _item_id(item),
        "reason": reason,
        "reason_code": reason,
        "omission_reason": reason,
    }
    if score is not None:
        record["score"] = float(score)
        record["selector_score"] = float(score)
    if validity:
        record["selector_validity"] = validity
    if cost is not None:
        record["token_cost"] = int(cost)
    if detail:
        record["detail"] = detail
    return record


def _values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        result: list[str] = []
        for nested in value.values():
            result.extend(_values(nested))
        return tuple(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for nested in value:
            result.extend(_values(nested))
        return tuple(result)
    text = str(value).strip()
    return (text,) if text else ()


def _normalise(value: object, *, path: bool = False) -> tuple[str, ...]:
    result: list[str] = []
    for item in _values(value):
        text = item.replace("\\", "/").strip() if path else " ".join(item.split())
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _explicit_named(
    value: object, names: set[str], *, depth: int = 0
) -> tuple[str, ...]:
    if depth > 3 or not isinstance(value, Mapping):
        return ()
    result: list[str] = []
    for key, raw in value.items():
        name = str(key).lower()
        if name in names:
            result.extend(_values(raw))
        elif isinstance(raw, Mapping):
            result.extend(_explicit_named(raw, names, depth=depth + 1))
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            for nested in raw:
                if isinstance(nested, Mapping):
                    result.extend(_explicit_named(nested, names, depth=depth + 1))
    return tuple(result)


def _direct_named(value: object, names: set[str]) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    result: list[str] = []
    for key, raw in value.items():
        if str(key).lower() in names:
            result.extend(_values(raw))
    return tuple(result)


def _typed_links(value: object) -> dict[str, tuple[str, ...]]:
    """Read only declared typed fields; never tokenize or scan prose."""

    source = value if isinstance(value, Mapping) else {}
    cached = source.get("_selector_links")
    if isinstance(cached, Mapping):
        return {field: tuple(_values(cached.get(field))) for field in LINK_FIELDS}
    links = source.get("links") if isinstance(source.get("links"), Mapping) else {}
    payload = (
        source.get("payload") if isinstance(source.get("payload"), Mapping) else {}
    )
    result: dict[str, tuple[str, ...]] = {}
    for field in LINK_FIELDS:
        raw = links.get(field, ()) if isinstance(links, Mapping) else ()
        if field == "targets":
            raw = (
                *_values(raw),
                *_direct_named(
                    payload,
                    {
                        "target",
                        "targets",
                        "target_id",
                        "target_ids",
                        "active_target",
                        "active_target_id",
                    },
                ),
            )
        result[field] = _normalise(raw, path=field == "files")
    # RecallLinks predates target links.  Preserve the explicit target on the
    # item without changing that persisted schema.
    if not result["targets"]:
        result["targets"] = _normalise(
            _direct_named(
                source,
                {
                    "target",
                    "targets",
                    "target_id",
                    "target_ids",
                    "active_target",
                    "active_target_id",
                },
            )
        )
    return result


def _focus_links(focus: object) -> dict[str, tuple[str, ...]]:
    result = _typed_links({"payload": focus})
    if isinstance(focus, Mapping):
        aliases = {
            "requirements": {
                "requirement",
                "requirements",
                "requirement_id",
                "requirement_ids",
                "requirement_ref",
                "requirement_refs",
                "ticket",
                "ticket_refs",
                "clause_id",
                "clause_ids",
            },
            "obligations": {
                "obligation",
                "obligations",
                "obligation_id",
                "obligation_ids",
                "focus_obligation",
                "open_requirement_ids",
                "blocking_requirement",
                "blocker_ids",
            },
            "targets": {
                "target",
                "targets",
                "target_id",
                "target_ids",
                "active_target",
                "active_target_id",
            },
            "files": {
                "path",
                "file",
                "filepath",
                "file_path",
                "source_path",
                "target_path",
                "location",
            },
            "symbols": {
                "symbol",
                "symbols",
                "fn",
                "function",
                "function_name",
                "entry_point",
            },
            "hypotheses": {
                "hypothesis",
                "hypotheses",
                "hypothesis_id",
                "hypothesis_ids",
                "open_hypothesis",
                "falsified_hypothesis",
                "falsified_hypotheses",
                "root_cause_hypothesis",
            },
            "stages": {
                "stage",
                "idealpath_stage",
                "idealpath",
                "phase",
                "current_stage",
            },
        }
        for field in LINK_FIELDS:
            direct = focus.get(field)
            nested = _explicit_named(focus, aliases[field])
            direct_values = (
                _normalise(direct, path=field == "files") if direct is not None else ()
            )
            nested_values = _normalise(nested, path=field == "files")
            result[field] = tuple(
                dict.fromkeys((*result[field], *direct_values, *nested_values))
            )
    return result


def _item_id(item: Mapping[str, object]) -> str:
    return str(item.get("id", ""))


def _status(item: Mapping[str, object]) -> str:
    payload = item.get("payload")
    payload_status = payload.get("status") if isinstance(payload, Mapping) else ""
    return str(item.get("status") or payload_status or "OPEN").upper()


def _cost(item: Mapping[str, object]) -> int:
    for key in ("token_cost", "render_token_cost"):
        value = item.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(1, int(value))
    payload = item.get("payload")
    if isinstance(payload, Mapping):
        for key in ("token_cost", "render_token_cost"):
            value = payload.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return max(1, int(value))
    try:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        encoded = repr(payload)
    return max(1, (len(encoded) + 3) // 4)


def _dependency_versions(item: Mapping[str, object]) -> dict[str, int]:
    result: dict[str, int] = {}
    candidates: list[object] = [
        item.get("dependency_versions"),
        item.get("binding_versions"),
    ]
    payload = item.get("payload")
    if isinstance(payload, Mapping):
        candidates.extend(
            (payload.get("dependency_versions"), payload.get("binding_versions"))
        )
        path = _direct_named(
            payload, {"path", "file", "filepath", "file_path", "target_path"}
        )
        revision = payload.get("workspace_revision", payload.get("revision"))
        if path and isinstance(revision, int):
            candidates.append({path[0]: revision})
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        for path, revision in candidate.items():
            if isinstance(revision, int) and revision >= 0:
                result[str(path).replace("\\", "/")] = revision
    return result


def _graph_value(value: object, *names: str) -> str:
    if isinstance(value, Mapping):
        for name in names:
            if value.get(name) not in (None, ""):
                return str(value[name])
        return ""
    for name in names:
        nested = getattr(value, name, None)
        if nested not in (None, ""):
            return str(nested)
    return ""


def _graph_nodes(ledger: object | None) -> tuple[object, ...]:
    if ledger is None:
        return ()
    value = getattr(ledger, "graph_nodes", ())
    return tuple(value) if isinstance(value, Sequence) else ()


def _graph_edges(ledger: object | None) -> tuple[object, ...]:
    if ledger is None:
        return ()
    value = getattr(ledger, "graph_edges", ())
    return tuple(value) if isinstance(value, Sequence) else ()


def _node_aliases(nodes: Sequence[object]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in nodes:
        node_id = _graph_value(node, "node_id", "id", "key")
        if not node_id:
            continue
        aliases[node_id] = node_id
        attrs = (
            node.get("attrs", {})
            if isinstance(node, Mapping)
            else getattr(node, "attrs", {})
        )
        if isinstance(attrs, Mapping):
            for key in (
                "hypothesis_id",
                "event_id",
                "evidence_id",
                "receipt_id",
                "claim_id",
                "path",
            ):
                for alias in _values(attrs.get(key)):
                    aliases[alias] = node_id
        for alias in _values(
            _graph_value(node, "hypothesis_id", "event_id", "evidence_id")
        ):
            aliases[alias] = node_id
    return aliases


def _adjacency(
    ledger: object | None, items: Sequence[Mapping[str, object]]
) -> tuple[dict[str, list[tuple[str, float, str]]], dict[str, int], dict[str, str]]:
    nodes = _graph_nodes(ledger)
    edges = _graph_edges(ledger)
    aliases = _node_aliases(nodes)
    adjacency: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        source = _graph_value(edge, "source_id", "source", "from")
        target = _graph_value(edge, "target_id", "target", "to")
        relation = _graph_value(edge, "relation", "kind", "type").lower()
        if not source or not target:
            continue
        weight = RELATION_WEIGHTS.get(relation, 0.4)
        adjacency[source].append((target, weight, relation))
        adjacency[target].append((source, weight, relation))
        degree[source] += 1
        degree[target] += 1
    for item in items:
        ident = _item_id(item)
        if not ident:
            continue
        typed = _typed_links(item)
        for field in LINK_FIELDS:
            for value in typed[field]:
                target = aliases.get(value, "")
                if not target:
                    continue
                adjacency[ident].append((target, 1.0, "item_binding"))
                adjacency[target].append((ident, 1.0, "item_binding"))
                degree[ident] += 1
                degree[target] += 1
    return dict(adjacency), dict(degree), aliases


def _graph_scores(
    focus: Mapping[str, tuple[str, ...]],
    ledger: object | None,
    items: Sequence[Mapping[str, object]],
    config: SelectorConfig,
) -> dict[str, float]:
    adjacency, degree, aliases = _adjacency(ledger, items)
    seeds = [aliases.get(value, value) for values in focus.values() for value in values]
    seeds = list(dict.fromkeys(seed for seed in seeds if seed))
    scores: dict[str, float] = {}
    candidate_ids = {_item_id(item) for item in items}
    for seed in seeds:
        frontier: deque[tuple[str, int, float]] = deque([(seed, 0, 1.0)])
        best: dict[tuple[str, int], float] = {(seed, 0): 1.0}
        while frontier:
            current, hops, score = frontier.popleft()
            if hops >= config.graph_hops:
                continue
            for target, edge_weight, _relation in adjacency.get(current, ()):
                next_hops = hops + 1
                next_score = score * edge_weight * config.graph_decay
                key = (target, next_hops)
                if next_score <= best.get(key, 0.0):
                    continue
                best[key] = next_score
                frontier.append((target, next_hops, next_score))
                candidate = target
                if candidate in candidate_ids:
                    scores[candidate] = max(
                        scores.get(candidate, 0.0),
                        next_score / math.sqrt(max(1, degree.get(candidate, 1))),
                    )
    return scores


def _ledger_version(ledger: object | None, path: str) -> int | None:
    if ledger is None:
        return None
    method = getattr(ledger, "version_for", None)
    if callable(method):
        try:
            value = method(path)
            return int(value) if isinstance(value, int) else None
        except (TypeError, ValueError):
            return None
    return None


def _validity(item: Mapping[str, object], ledger: object | None) -> str:
    if bool(item.get("stale")) or _status(item) in {"STALE", "INVALIDATED"}:
        return "INVALIDATED"
    dependencies = _dependency_versions(item)
    payload = item.get("payload")
    refs = set(
        _direct_named(
            payload,
            {
                "evidence_ref",
                "evidence_refs",
                "receipt_id",
                "receipt_ids",
                "evidence_id",
            },
        )
    )
    refs.add(_item_id(item))
    dirty_refs = (
        set(getattr(ledger, "dirty_evidence_refs", ())) if ledger is not None else set()
    )
    if refs.intersection(dirty_refs):
        return "INVALIDATED"
    observations = getattr(ledger, "observations", ()) if ledger is not None else ()
    item_paths = set(_typed_links(item)["files"])
    for observation in observations:
        path = str(getattr(observation, "path", "")).replace("\\", "/")
        if (
            not path
            or not item_paths
            or not any(
                path == candidate
                or path.endswith("/" + candidate)
                or candidate.endswith("/" + path)
                for candidate in item_paths
            )
        ):
            continue
        if bool(getattr(observation, "dirty", False)):
            return "INVALIDATED"
    if dependencies:
        for path, recorded in dependencies.items():
            current = _ledger_version(ledger, path)
            if current is None:
                return "UNKNOWN_BINDING"
            if current > recorded:
                return "INVALIDATED"
    return "CURRENT"


def _hypothesis_projection(
    ledger: object | None,
) -> tuple[set[str], set[str], set[str]]:
    open_ids: set[str] = set()
    falsified_ids: set[str] = set()
    evidence_ids: set[str] = set()
    if ledger is None:
        return open_ids, falsified_ids, evidence_ids
    postures = getattr(ledger, "hypothesis_postures", {})
    if isinstance(postures, Mapping):
        for ident, posture in postures.items():
            value = str(getattr(posture, "posture", posture)).lower()
            if value in {"eliminated", "falsified"}:
                falsified_ids.add(str(ident))
            else:
                open_ids.add(str(ident))
            for field in ("support_evidence_ids", "contradiction_evidence_ids"):
                evidence_ids.update(_values(getattr(posture, field, ())))
    return open_ids, falsified_ids, evidence_ids


def _mandatory_ids(
    items: Sequence[Mapping[str, object]], requested: set[str], ledger: object | None
) -> set[str]:
    open_ids, falsified_ids, evidence_ids = _hypothesis_projection(ledger)
    mandatory = set(requested)
    nodes = _graph_nodes(ledger)
    aliases = _node_aliases(nodes)
    graph_edges = _graph_edges(ledger)
    pinned_graph_ids: set[str] = set()
    for edge in graph_edges:
        source = _graph_value(edge, "source_id", "source", "from")
        target = _graph_value(edge, "target_id", "target", "to")
        relation = _graph_value(edge, "relation", "kind", "type").lower()
        source_aliases = {
            alias for alias, node_id in aliases.items() if node_id == source
        }
        target_aliases = {
            alias for alias, node_id in aliases.items() if node_id == target
        }
        if source_aliases.intersection(open_ids | falsified_ids) and relation in {
            "explains",
            "support",
            "supports",
            "supports",
            "contradicts",
            "contradiction",
        }:
            pinned_graph_ids.update(target_aliases)
        if target_aliases.intersection(open_ids | falsified_ids) and relation in {
            "contradicts",
            "contradiction",
        }:
            pinned_graph_ids.update(source_aliases)
    for item in items:
        ident = _item_id(item)
        typed = _typed_links(item)
        payload = item.get("payload")
        hyp_ids = set(typed["hypotheses"])
        hyp_ids.update(
            _direct_named(payload, {"hypothesis_id", "hypothesis_ids", "hypothesis"})
        )
        item_aliases = {ident, *hyp_ids}
        status = _status(item)
        if bool(item.get("pinned")) or status in {"FALSIFIED", "ELIMINATED"}:
            mandatory.add(ident)
        if hyp_ids.intersection(open_ids | falsified_ids) or item_aliases.intersection(
            pinned_graph_ids | evidence_ids
        ):
            mandatory.add(ident)
        if str(item.get("kind", "")).lower() == "hypothesis" and hyp_ids.intersection(
            open_ids | falsified_ids | {ident}
        ):
            mandatory.add(ident)
        if isinstance(payload, Mapping) and any(
            payload.get(key) is True
            for key in (
                "frozen_repro",
                "frozen_reproduction",
                "admitted_root_cause",
                "root_cause_admitted",
                "blocking",
            )
        ):
            mandatory.add(ident)
    return mandatory


def _decision_atoms(item: Mapping[str, object]) -> set[str]:
    payload = item.get("payload")
    atoms: list[str] = []
    for value in (
        item.get("assertion_atoms"),
        item.get("atoms"),
        payload.get("assertion_atoms") if isinstance(payload, Mapping) else None,
        payload.get("atoms") if isinstance(payload, Mapping) else None,
        payload.get("decision_atoms") if isinstance(payload, Mapping) else None,
    ):
        atoms.extend(_values(value))
    return set(atoms)


def _host_predicate_atoms(predicates: Sequence[object]) -> tuple[set[str], ...]:
    bundles: list[set[str]] = []
    for raw in predicates:
        if not isinstance(raw, Mapping):
            continue
        authority = str(raw.get("authority", raw.get("source", ""))).lower()
        if (
            authority not in {"host", "kernel", "host_predicate"}
            or raw.get("model_judged") is True
        ):
            continue
        raw_bundles = raw.get(
            "bundles", raw.get("required_atoms", raw.get("atoms", ()))
        )
        if isinstance(raw_bundles, Sequence) and not isinstance(
            raw_bundles, (str, bytes, bytearray)
        ):
            if raw_bundles and all(
                not isinstance(value, (Mapping, list, tuple, set))
                for value in raw_bundles
            ):
                bundles.append(set(_values(raw_bundles)))
            else:
                for bundle in raw_bundles:
                    atoms = set(
                        _values(
                            bundle.get("atoms", ())
                            if isinstance(bundle, Mapping)
                            else bundle
                        )
                    )
                    if atoms:
                        bundles.append(atoms)
    return tuple(bundle for bundle in bundles if bundle)


def _decision_value(item: Mapping[str, object], predicates: Sequence[object]) -> float:
    atoms = _decision_atoms(item)
    if not atoms:
        return 0.0
    bundles = _host_predicate_atoms(predicates)
    return float(sum(1 for bundle in bundles if bundle.intersection(atoms)))


def _overlap_score(
    item: Mapping[str, object], focus: Mapping[str, tuple[str, ...]]
) -> float:
    typed = _typed_links(item)
    score = 0.0
    for field, weight in FOCUS_WEIGHTS.items():
        left = set(typed[field])
        right = set(focus[field])
        if field == "files":
            score += weight * sum(
                1
                for value in left
                for target in right
                if value == target
                or value.endswith("/" + target)
                or target.endswith("/" + value)
            )
        else:
            score += weight * len(left.intersection(right))
    return score


def _unanchored(
    item: Mapping[str, object], overlap: float, graph: float, mandatory: bool
) -> bool:
    return not mandatory and overlap <= 0.0 and graph <= 0.0


def select_items(
    items: Sequence[Mapping[str, object]],
    focus: object = None,
    *,
    budget_tokens: int = 720,
    max_items: int = 8,
    ledger: object | None = None,
    requested_ids: Sequence[str] = (),
    decision_predicates: Sequence[object] = (),
    arm: str = "hybrid",
    historical_ids: Sequence[str] = (),
    config: SelectorConfig | None = None,
    kinds: Sequence[str] | None = None,
) -> SelectorResult:
    """Select a bounded frame from typed host records.

    ``historical`` and ``oracle`` exist only for the offline lab.  Production
    callers should use ``hybrid``; ``hybrid_no_decision`` is its explicit
    ablation.  No score reads ``gist``, payload text, or model output.
    """

    if budget_tokens < 1 or max_items < 1:
        raise ValueError("budget_tokens and max_items must be positive")
    policy = config or SelectorConfig()
    focus_links = _focus_links(focus)
    requested = {str(value) for value in requested_ids if str(value)}
    kind_filter = {str(value).lower() for value in kinds} if kinds is not None else None
    candidates: list[dict[str, object]] = []
    omissions: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for raw in items:
        if kind_filter is not None and str(raw.get("kind", "")).lower() not in kind_filter:
            continue
        item = dict(raw)
        ident = _item_id(item)
        if ident and ident in seen_ids:
            omissions.append(_omission(item, "duplicate", detail="duplicate candidate identity"))
            continue
        if ident:
            seen_ids.add(ident)
        candidates.append(item)
    for item in candidates:
        item["_selector_links"] = _typed_links(item)
    graph = _graph_scores(focus_links, ledger, candidates, policy)
    mandatory = _mandatory_ids(candidates, requested, ledger)
    historical = {str(value) for value in historical_ids}
    scored: list[dict[str, object]] = []
    for item in candidates:
        ident = _item_id(item)
        overlap = _overlap_score(item, focus_links)
        graph_score = graph.get(ident, 0.0)
        validity = _validity(item, ledger)
        required = ident in mandatory
        decision = _decision_value(item, decision_predicates)
        cost = _cost(item)
        if arm == "historical":
            score = 1.0 if ident in historical else -1.0
        elif arm == "recency":
            score = float(item.get("seq", 0) or 0)
        elif arm == "overlap":
            score = overlap
        elif arm == "graph":
            score = graph_score
        elif arm == "oracle":
            label = str(item.get("label", "UNKNOWN")).upper()
            score = {"NEEDED": 1000.0, "UNKNOWN": 0.0, "NOISE": -1000.0}.get(label, 0.0)
        else:
            score = 2.0 * overlap + 3.0 * graph_score
            if arm == "hybrid":
                score += policy.decision_bonus * decision
            if validity == "UNKNOWN_BINDING":
                score -= 2.0
            if validity == "INVALIDATED":
                score -= 100.0
            if _unanchored(item, overlap, graph_score, required):
                score -= policy.distractor_penalty * (cost / 128.0)
        item.update(
            {
                "selector_validity": validity,
                "selector_cost": cost,
                "selector_score": score,
            }
        )
        scored.append(
            {
                "item": item,
                "score": score,
                "overlap": overlap,
                "graph": graph_score,
                "decision": decision,
                "required": required,
                "validity": validity,
                "cost": cost,
                "unanchored": _unanchored(item, overlap, graph_score, required),
            }
        )

    def order(row: Mapping[str, object]) -> tuple[int, float, int, str]:
        item = row["item"]
        return (
            -int(bool(row["required"])),
            -float(row["score"]),
            -int(item.get("seq", 0) or 0),
            _item_id(item),
        )

    mandatory_rows = sorted((row for row in scored if bool(row["required"])), key=order)
    optional_rows = sorted(
        (row for row in scored if not bool(row["required"])), key=order
    )
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    used = 0
    pin_over_budget = 0
    for row in mandatory_rows:
        item = row["item"]
        ident = _item_id(item)
        if not ident or ident in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(ident)
        used += int(row["cost"])
        if used > budget_tokens:
            pin_over_budget += 1

    unanchored_tokens = sum(
        int(row["cost"])
        for row in mandatory_rows
        if _item_id(row["item"]) in selected_ids and bool(row["unanchored"])
    )
    hypothesis_tokens = sum(
        int(row["cost"])
        for row in mandatory_rows
        if _item_id(row["item"]) in selected_ids
        and str(row["item"].get("kind", "")).lower() == "hypothesis"
    )
    for row in optional_rows:
        item = row["item"]
        score = float(row["score"])
        if row["validity"] == "INVALIDATED":
            omissions.append(_omission(item, "stale", score=score,
                                        validity=str(row["validity"]), cost=int(row["cost"])))
            continue
        if arm in {"hybrid", "hybrid_no_decision"} and score <= 0.0:
            omissions.append(_omission(item, "low_relevance_score", score=score,
                                        validity=str(row["validity"]), cost=int(row["cost"])))
            continue
        if len(selected) >= max_items:
            omissions.append(_omission(item, "budget", score=score,
                                        validity=str(row["validity"]), cost=int(row["cost"]),
                                        detail="item cap"))
            continue
        if used + int(row["cost"]) > budget_tokens:
            omissions.append(_omission(item, "budget", score=score,
                                        validity=str(row["validity"]), cost=int(row["cost"])))
            continue
        if bool(row["unanchored"]):
            if (
                used
                and (unanchored_tokens + int(row["cost"]))
                / max(1, used + int(row["cost"]))
                > policy.unanchored_cap
            ):
                omissions.append(_omission(item, "distractor_cap", score=score,
                                            validity=str(row["validity"]), cost=int(row["cost"]),
                                            detail="unanchored token cap"))
                continue
        if (
            str(row["item"].get("kind", "")).lower() == "hypothesis"
            and used
            and (hypothesis_tokens + int(row["cost"])) / max(1, used + int(row["cost"]))
            > policy.hypothesis_cap
        ):
            omissions.append(_omission(item, "distractor_cap", score=score,
                                        validity=str(row["validity"]), cost=int(row["cost"]),
                                        detail="hypothesis token cap"))
            continue
        selected.append(item)
        selected_ids.add(_item_id(item))
        used += int(row["cost"])
        if bool(row["unanchored"]):
            unanchored_tokens += int(row["cost"])
        if str(row["item"].get("kind", "")).lower() == "hypothesis":
            hypothesis_tokens += int(row["cost"])

    metrics: dict[str, object] = {
        "arm": arm,
        "candidate_count": len(candidates) + sum(
            1 for row in omissions if row["reason"] == "duplicate"
        ),
        "selected_count": len(selected),
        "pin_count": len(mandatory_rows),
        "pin_over_budget": pin_over_budget,
        "unknown_binding_count": sum(
            1 for row in scored if row["validity"] == "UNKNOWN_BINDING"
        ),
        "invalidated_count": sum(
            1 for row in scored if row["validity"] == "INVALIDATED"
        ),
        "graph_edge_count": len(_graph_edges(ledger)),
        "decision_predicate_count": len(_host_predicate_atoms(decision_predicates)),
        "unanchored_tokens": unanchored_tokens,
        "hypothesis_tokens": hypothesis_tokens,
        "omission_count": len(omissions),
        "omission_reasons": {
            reason: sum(1 for row in omissions if row["reason"] == reason)
            for reason in sorted(OMISSION_REASONS)
        },
        "omissions": tuple(omissions),
    }
    for item in selected:
        item.pop("_selector_links", None)
    return SelectorResult(
        tuple(selected),
        len(omissions),
        used,
        tuple(
            _item_id(item) for item in mandatory_rows if _item_id(item) in selected_ids
        ),
        metrics,
        tuple(omissions),
    )


__all__ = [
    "DEFAULT_DISTRACTOR_PENALTY",
    "DEFAULT_HYPOTHESIS_CAP",
    "DEFAULT_UNANCHORED_CAP",
    "OMISSION_REASONS",
    "OMISSION_SCHEMA",
    "SelectorConfig",
    "SelectorResult",
    "select_items",
]
