"""Language-neutral orientation and bounded graph expansion over NAV1.

This module consumes ``NavigationSnapshot`` records and the existing graph
object.  It does not parse source, build a second index, or call provider
tools.  Every result names its immutable snapshot, exact source spans,
coverage, ambiguity, and any signed work frontier.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

try:
    from kernel_nav_snapshot import (
        NavigationError,
        NavigationSnapshot,
        ScopeDenied,
        SnapshotStale,
    )
except ImportError:  # pragma: no cover - package import path.
    from .kernel_nav_snapshot import (
        NavigationError,
        NavigationSnapshot,
        ScopeDenied,
        SnapshotStale,
    )


SCHEMA = "stateframe/nav-orientation/1"
FRONTIER_SCHEMA = "stateframe/nav-frontier/1"
MAX_FOCUS_FILES = 3
MAX_RANKED_ROWS = 8
MAX_GRAPH_DEPTH = 2
MAX_GRAPH_NODES = 200
MAX_GRAPH_EDGES = 1_000
DEFAULT_DEADLINE_MS = 300


@dataclass(frozen=True, slots=True)
class OrientationLimits:
    """Hard work and presentation limits established before graph traversal."""

    max_focus_files: int = MAX_FOCUS_FILES
    max_rows: int = MAX_RANKED_ROWS
    max_depth: int = MAX_GRAPH_DEPTH
    max_nodes: int = MAX_GRAPH_NODES
    max_edges: int = MAX_GRAPH_EDGES
    deadline_ms: int = DEFAULT_DEADLINE_MS
    continuation_ttl_seconds: int = 300

    def __post_init__(self) -> None:
        for name in ("max_focus_files", "max_rows", "max_depth", "max_nodes", "max_edges", "deadline_ms", "continuation_ttl_seconds"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise NavigationError(f"orientation limit {name} must be a positive integer")
        if self.max_depth > MAX_GRAPH_DEPTH or self.max_nodes > MAX_GRAPH_NODES or self.max_edges > MAX_GRAPH_EDGES:
            raise NavigationError("orientation graph limits exceed hard ceilings")


@dataclass(frozen=True, slots=True)
class OrientationContinuation:
    """Signed, snapshot/ACL/graph-bound frontier for a bounded expansion."""

    handle: str
    snapshot_id: str
    scope_id: str
    authorization_epoch: str
    phase: str
    focus: str
    graph_generation: str
    frontier: tuple[str, ...]
    expires_at: int
    visited: tuple[str, ...] = ()

    def payload(self) -> dict[str, object]:
        return {
            "schema": FRONTIER_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "scope_id": self.scope_id,
            "authorization_epoch": self.authorization_epoch,
            "phase": self.phase,
            "focus": self.focus,
            "graph_generation": self.graph_generation,
            "frontier": list(self.frontier),
            "expires_at": self.expires_at,
            "visited": list(self.visited),
        }


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _graph_generation(graph: object | None) -> str:
    if graph is None:
        return "graph:none"
    method = getattr(graph, "index_token", None)
    if callable(method):
        try:
            value = str(method())
            return value if value else "graph:unknown"
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return "graph:unknown"
    return "graph:unknown"


def _make_frontier(
    snapshot: NavigationSnapshot,
    *,
    phase: str,
    focus: str,
    graph_generation: str,
    frontier: Sequence[str],
    visited: Sequence[str],
    now: int,
    ttl: int,
) -> OrientationContinuation:
    payload = {
        "schema": FRONTIER_SCHEMA,
        "snapshot_id": snapshot.snapshot_id,
        "scope_id": snapshot.scope_id,
        "authorization_epoch": snapshot.authorization_epoch,
        "phase": phase,
        "focus": focus,
        "graph_generation": graph_generation,
        "frontier": list(frontier),
        "expires_at": now + ttl,
        "visited": list(visited),
    }
    encoded = base64.urlsafe_b64encode(_canonical(payload)).decode("ascii").rstrip("=")
    handle = "navfront:" + _digest(payload) + "." + encoded
    return OrientationContinuation(
        handle, snapshot.snapshot_id, snapshot.scope_id, snapshot.authorization_epoch,
        phase, focus, graph_generation, tuple(frontier), now + ttl, tuple(visited),
    )


def _decode_frontier(value: str) -> OrientationContinuation:
    if not isinstance(value, str) or not value.startswith("navfront:") or "." not in value:
        raise NavigationError("invalid orientation continuation")
    digest, encoded = value[len("navfront:"):].split(".", 1)
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NavigationError("invalid orientation continuation payload") from exc
    if not isinstance(payload, Mapping) or _digest(payload) != digest or payload.get("schema") != FRONTIER_SCHEMA:
        raise NavigationError("orientation continuation signature mismatch")
    frontier = payload.get("frontier")
    if not isinstance(frontier, list) or not all(isinstance(item, str) for item in frontier):
        raise NavigationError("orientation continuation frontier is invalid")
    visited = payload.get("visited", [])
    if not isinstance(visited, list) or not all(isinstance(item, str) for item in visited):
        raise NavigationError("orientation continuation visited set is invalid")
    try:
        return OrientationContinuation(
            value, str(payload["snapshot_id"]), str(payload["scope_id"]),
            str(payload["authorization_epoch"]), str(payload["phase"]), str(payload["focus"]),
            str(payload["graph_generation"]), tuple(frontier), int(payload["expires_at"]), tuple(visited),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NavigationError("orientation continuation fields are invalid") from exc


def _target_text(value: object) -> str:
    if isinstance(value, Mapping):
        for key in ("entity_id", "path", "handle"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return str(value or "").strip()


def _frontier_entry(entity: str, depth: int) -> str:
    return json.dumps([entity, depth], ensure_ascii=False, separators=(",", ":"))


def _frontier_state(value: str) -> tuple[str, int]:
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value, 0  # Compatibility with the first frontier encoding.
    if (
        isinstance(decoded, list)
        and len(decoded) == 2
        and isinstance(decoded[0], str)
        and type(decoded[1]) is int
        and decoded[1] >= 0
    ):
        return decoded[0], decoded[1]
    raise NavigationError("orientation continuation frontier entry is invalid")


def _outline_rows(snapshot: NavigationSnapshot, path: str) -> dict[str, object]:
    outline = snapshot.outline(path)
    if isinstance(outline.get("outline"), Mapping):
        # Vue retains the file-global records at the top level; nested block
        # data is supplementary and must not create a second source authority.
        outline = dict(outline)
    return outline


def _records(
    outline: Mapping[str, object], key: str, *, limit: int | None = None,
) -> tuple[Mapping[str, object], ...]:
    raw = outline.get(key, ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    result: list[Mapping[str, object]] = []
    for item in raw:
        if isinstance(item, Mapping):
            result.append(item)
        if limit is not None and len(result) >= limit:
            break
    return tuple(result)


def _line(record: Mapping[str, object]) -> int:
    value = record.get("line", record.get("start_line", 0))
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _end_line(record: Mapping[str, object]) -> int:
    value = record.get("end_line", record.get("line", 0))
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else _line(record)


def _row(path: str, record: Mapping[str, object], *, kind: str, rank_basis: str) -> dict[str, object]:
    return {
        "path": path,
        "kind": kind,
        "name": str(record.get("name") or record.get("expression") or record.get("module") or ""),
        "start_line": _line(record),
        "end_line": _end_line(record),
        "signature": str(record.get("signature") or record.get("source") or ""),
        "rank_basis": rank_basis,
    }


def _resolve_candidates(
    graph: object | None,
    query: str,
    *,
    allowed_paths: set[str] | None = None,
    max_inspected: int = MAX_GRAPH_NODES,
) -> tuple[Mapping[str, object], ...]:
    if graph is None or not query:
        return ()
    candidates: dict[str, Mapping[str, object]] = {}
    inspected = 0
    for namespace_name in ("defs", "variables"):
        namespace = getattr(graph, namespace_name, {})
        if not isinstance(namespace, Mapping):
            continue
        for key, value in namespace.items():
            if inspected >= max_inspected:
                return tuple(sorted(candidates.values(), key=lambda item: (
                    str(item.get("path", "")), int(item.get("start_line", 0)), str(item.get("entity_id", ""))
                )))
            inspected += 1
            entity_id = str(key)
            short = str(getattr(value, "short_name", "") or entity_id.rsplit("::", 1)[-1].split(".")[-1])
            qualname = str(getattr(value, "qualname", "") or "")
            if entity_id == query or qualname == query or short == query:
                path = str(getattr(value, "file", ""))
                if allowed_paths is not None and path not in allowed_paths:
                    continue
                candidates[entity_id] = {
                    "entity_id": entity_id,
                    "kind": str(getattr(value, "kind", "symbol")),
                    "path": path,
                    "start_line": int(getattr(value, "lineno", 0) or 0),
                    "end_line": int(getattr(value, "end_lineno", 0) or getattr(value, "lineno", 0) or 0),
                    "name": short,
                    "qualname": qualname or entity_id,
                }
    return tuple(sorted(candidates.values(), key=lambda item: (str(item.get("path", "")), int(item.get("start_line", 0)), str(item.get("entity_id", "")))))


def _adjacent(graph: object | None, entity_id: str, *, reverse: bool, limit: int) -> tuple[str, ...]:
    if graph is None or limit < 1:
        return ()
    adjacency = getattr(graph, "_rev" if reverse else "_edges", {})
    values = adjacency.get(entity_id, ()) if isinstance(adjacency, Mapping) else ()
    result: list[str] = []
    # Do not sort or materialize a fan-out before the cap. Graph owners expose
    # deterministic insertion order for this adapter; selected IDs are sorted.
    for value in values if isinstance(values, Iterable) and not isinstance(values, (str, bytes, bytearray)) else ():
        candidate = str(value)
        if candidate and candidate not in result:
            result.append(candidate)
        if len(result) >= limit:
            break
    return tuple(sorted(result))


def bounded_related(
    snapshot: NavigationSnapshot,
    graph: object,
    entity: str,
    relation: str,
    *,
    depth: int = 1,
    limits: OrientationLimits | None = None,
    continuation: str | OrientationContinuation | None = None,
    now: int | None = None,
) -> dict[str, object]:
    """Expand callers/callees with work caps applied before adjacency reads."""

    selected = limits or OrientationLimits()
    if depth < 1 or depth > selected.max_depth:
        raise NavigationError("related depth is outside the bounded range")
    if relation not in {"callers", "callees", "references", "impact", "lineage"}:
        raise NavigationError("unsupported related relation")
    started = time.monotonic()
    current_time = int(time.time()) if now is None else int(now)
    generation = _graph_generation(graph)
    focus = _target_text(entity)
    frontier = (focus,)
    if continuation is not None:
        token = _decode_frontier(continuation) if isinstance(continuation, str) else continuation
        if token.snapshot_id != snapshot.snapshot_id or token.scope_id != snapshot.scope_id:
            raise SnapshotStale("related continuation belongs to another snapshot")
        if token.authorization_epoch != snapshot.authorization_epoch:
            raise ScopeDenied("related continuation is no longer authorized")
        if token.graph_generation != generation or token.focus != focus:
            raise SnapshotStale("related continuation graph identity changed")
        if token.expires_at < current_time:
            raise SnapshotStale("related continuation expired")
        frontier = token.frontier
    queue: list[tuple[str, int]] = [_frontier_state(item) for item in frontier]
    seen: set[str] = set(token.visited) if continuation is not None else set()
    rows: list[dict[str, object]] = []
    edges_examined = 0
    deadline_hit = False
    while queue and len(seen) < selected.max_nodes and edges_examined < selected.max_edges:
        if (time.monotonic() - started) * 1000 >= selected.deadline_ms:
            deadline_hit = True
            break
        current, current_depth = queue.pop(0)
        if current in seen or current_depth > depth:
            continue
        seen.add(current)
        if current != focus:
            rows.append({"entity_id": current, "relation": relation, "depth": current_depth})
        if current_depth >= depth:
            continue
        reverse = relation == "callers"
        neighbors = _adjacent(graph, current, reverse=reverse, limit=selected.max_edges - edges_examined)
        for neighbor in neighbors:
            edges_examined += 1
            if neighbor not in seen:
                queue.append((neighbor, current_depth + 1))
            if edges_examined >= selected.max_edges:
                break
    unfinished = tuple(
        _frontier_entry(item, current_depth)
        for item, current_depth in queue[:selected.max_nodes]
    )
    truncated = bool(queue or len(seen) >= selected.max_nodes or edges_examined >= selected.max_edges or deadline_hit)
    next_token = None
    if truncated and unfinished:
        next_token = _make_frontier(
            snapshot, phase="related", focus=focus, graph_generation=generation,
            frontier=unfinished, visited=tuple(sorted(seen)), now=current_time,
            ttl=selected.continuation_ttl_seconds,
        )
    return {
        "schema": SCHEMA,
        "operation": "related",
        "relation": relation,
        "status": "PARTIAL" if truncated else ("FOUND" if rows else "EMPTY"),
        "snapshot_id": snapshot.snapshot_id,
        "rows": rows[:selected.max_rows],
        "counts": {"shown": min(len(rows), selected.max_rows), "total_relation": "lower_bound" if truncated else "exact"},
        "work": {
            "visited_nodes": len(seen), "examined_edges": edges_examined,
            "depth": depth, "deadline_hit": deadline_hit,
            "limits": {"max_nodes": selected.max_nodes, "max_edges": selected.max_edges, "max_depth": selected.max_depth, "deadline_ms": selected.deadline_ms},
        },
        "next_handle": next_token.handle if next_token else None,
    }


def _source_span(snapshot: NavigationSnapshot, path: str, start: int, end: int, role: str) -> dict[str, object] | None:
    if start < 1 or end < start:
        return None
    try:
        view = snapshot.evidence_view(path, start_line=start, end_line=end)
    except (NavigationError, OSError, ValueError):
        return None
    segment = view.segments[0] if view.segments else None
    if segment is None:
        return None
    return {
        "path": path,
        "start_line": start,
        "end_line": end,
        "span_id": segment.span_id,
        "body_sha256": segment.body_sha256,
        "source_sha256": snapshot.source_snapshot(path).source_sha256,
        "role": role,
    }


def orient(
    snapshot: NavigationSnapshot,
    phase: str,
    *,
    focus: object = None,
    question: str = "",
    graph: object | None = None,
    limits: OrientationLimits | None = None,
    continuation: str | OrientationContinuation | None = None,
    now: int | None = None,
) -> dict[str, object]:
    """Build a deterministic localize/edit/verify packet from one snapshot."""

    selected = limits or OrientationLimits()
    if phase not in {"localize", "edit", "verify"}:
        raise NavigationError("orientation phase is invalid")
    if not isinstance(question, str) or len(question) > 512:
        raise NavigationError("orientation question is invalid")
    started = time.monotonic()
    current_time = int(time.time()) if now is None else int(now)
    target = _target_text(focus)
    authorized_paths = {member.path for member in snapshot.members}
    candidates = _resolve_candidates(
        graph, target, allowed_paths=authorized_paths, max_inspected=selected.max_nodes,
    )
    ambiguity = "AMBIGUOUS" if len(candidates) > 1 else ("FOUND" if candidates else "UNKNOWN")
    focus_paths: list[str] = []
    if isinstance(focus, Mapping) and isinstance(focus.get("path"), str):
        focus_paths.append(focus["path"])
    for candidate in candidates:
        path = str(candidate.get("path", ""))
        if path and path not in focus_paths:
            focus_paths.append(path)
    if not focus_paths and question:
        tokens = {item.casefold() for item in question.replace("/", " ").replace(".", " ").split() if len(item) > 2}
        scored: list[tuple[int, str]] = []
        for member in snapshot.members:
            try:
                outline = snapshot.outline(member.path)
            except NavigationError:
                continue
            text = json.dumps(outline, ensure_ascii=False, sort_keys=True).casefold()
            score = sum(token in text for token in tokens)
            if score:
                scored.append((-score, member.path))
        focus_paths.extend(path for _score, path in sorted(scored)[:selected.max_focus_files])
    if not focus_paths and snapshot.members:
        focus_paths.append(snapshot.members[0].path)
    focus_paths = list(dict.fromkeys(focus_paths))[:selected.max_focus_files]

    definitions: list[dict[str, object]] = []
    imports: list[dict[str, object]] = []
    modules: list[dict[str, object]] = []
    types: list[dict[str, object]] = []
    spans: list[dict[str, object]] = []
    for path in focus_paths:
        if (time.monotonic() - started) * 1000 >= selected.deadline_ms:
            break
        try:
            outline = _outline_rows(snapshot, path)
        except NavigationError:
            continue
        for record in _records(outline, "imports", limit=selected.max_rows):
            imports.append({"path": path, **dict(record)})
        for record in _records(outline, "module_statements", limit=selected.max_rows):
            modules.append({"path": path, **dict(record)})
        for record in _records(outline, "types", limit=selected.max_rows):
            row = _row(path, record, kind="type", rank_basis="outline type")
            types.append(row)
            if len(types) < selected.max_rows:
                span = _source_span(snapshot, path, row["start_line"], row["end_line"], "type")
                if span is not None:
                    spans.append(span)
        for record in _records(outline, "functions", limit=selected.max_rows):
            row = _row(path, record, kind="definition", rank_basis="outline definition")
            definitions.append(row)
            if len(definitions) <= selected.max_rows:
                span = _source_span(snapshot, path, row["start_line"], row["end_line"], "definition")
                if span is not None:
                    spans.append(span)

    definitions.sort(key=lambda row: (str(row["path"]), int(row["start_line"]), str(row["name"])))
    types.sort(key=lambda row: (str(row["path"]), int(row["start_line"]), str(row["name"])))
    alternatives = [dict(item) for item in candidates[1:2]]
    related: dict[str, object] = {}
    if candidates and graph is not None:
        entity_id = str(candidates[0].get("entity_id", ""))
        related = {
            "callers": bounded_related(snapshot, graph, entity_id, "callers", depth=1, limits=selected, now=current_time),
            "callees": bounded_related(snapshot, graph, entity_id, "callees", depth=1, limits=selected, now=current_time),
        }
    coverage = snapshot.coverage.to_dict()
    coverage.update({
        "focus_files": len(focus_paths),
        "module_statements": {"shown": len(modules), "total_relation": "exact"},
        "ambiguity": ambiguity,
        "unknown": not bool(focus_paths),
    })
    graph_work = {
        "limits": {"max_nodes": selected.max_nodes, "max_edges": selected.max_edges, "max_depth": selected.max_depth, "deadline_ms": selected.deadline_ms},
        "bounded_before_traversal": True,
    }
    return {
        "schema": SCHEMA,
        "phase": phase,
        "snapshot_id": snapshot.snapshot_id,
        "scope_id": snapshot.scope_id,
        "focus": target,
        "focus_paths": focus_paths,
        "coverage": coverage,
        "ranked_definitions": definitions[:selected.max_rows],
        "imports": imports[:selected.max_rows],
        "module_statements": modules[:selected.max_rows],
        "nearest_types": types[:selected.max_rows],
        "direct_callers": related.get("callers", {"status": "UNKNOWN", "rows": []}),
        "direct_callees": related.get("callees", {"status": "UNKNOWN", "rows": []}),
        "retained_alternative": alternatives,
        "spans": spans[:selected.max_rows],
        "graph_work": graph_work,
        "unknown": [] if focus_paths else ["no authorized source matched the focus"],
        "orientation_digest": "orient:" + _digest((snapshot.snapshot_id, phase, target, question, focus_paths, definitions, modules)),
    }


__all__ = [
    "DEFAULT_DEADLINE_MS",
    "FRONTIER_SCHEMA",
    "MAX_GRAPH_DEPTH",
    "MAX_GRAPH_EDGES",
    "MAX_GRAPH_NODES",
    "OrientationContinuation",
    "OrientationLimits",
    "bounded_related",
    "orient",
]
