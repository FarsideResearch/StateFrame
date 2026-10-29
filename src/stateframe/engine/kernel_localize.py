"""Host-owned targetless localization for the IDEALPATH S2 boundary.

This module is deliberately read-only.  It builds a small public inventory,
reuses the vendored Mission Control graph, and returns typed records that can
be independently checked before a target is bound.  A ranked suspect is a
useful investigation handle, never proof of a root cause.
"""
from __future__ import annotations

import ast
import bisect
import hashlib
import json
import logging
import math
import os
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

logger = logging.getLogger(__name__)
_GRAPH_BUILD_LOCK = RLock()

SCHEMA = "stateframe-localize/1"
RECEIPT_SCHEMA = "stateframe-localize-receipt/1"
INDEX_SCHEMA = "stateframe-localize-index/1"
CHANNELS = ("literal", "lexical", "entrypoint", "state", "public_test")
DEFAULT_CHANNEL_LIMIT = 20
DEFAULT_PAGE_SIZE = 5
MAX_PAGE_SIZE = 20
MAX_QUERY_CHARS = 2400
MAX_SOURCE_BYTES = 512_000
MAX_FILES = 4000
MAX_GRAPH_DEPTH = 3
MAX_GRAPH_NODES = 96
MAX_GRAPH_EDGES = 192

_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", "coverage", "htmlcov", "evaluator", "fix", "backups",
    "backup", "private", "hidden", "gold", "grader", "graders", "fixture", "fixtures",
})
_SOURCE_SUFFIXES = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".sql", ".md",
})
_CODE_SUFFIXES = frozenset({".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte", ".sql"})
_TEST_PARTS = frozenset({"test", "tests", "spec", "specs", "probes"})
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "when", "then",
    "into", "have", "has", "are", "was", "were", "not", "does", "doesn",
    "should", "would", "could", "after", "before", "while", "there", "their",
    "they", "error", "issue", "bug", "fix", "file", "function", "method",
    "code", "value", "return", "expected", "actual", "public", "test",
})
_IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9])[_A-Za-z][_A-Za-z0-9]{2,}(?![A-Za-z0-9])")
_QUOTED_RE = re.compile(r"(['\"])(.{3,240}?)\1")
_ERROR_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9_]*(?:Error|Exception)|HTTP\s*\d{3}|\d{3}\s+[A-Za-z][^\n]{2,100})\b"
)
_PATH_RE = re.compile(r"/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._{}:-]+")
_ROUTE_RE = re.compile(
    r"(?P<kind>route|http_route|job|handler|event|consumer|producer|schedule|register|subscribe|enqueue|add_job|add_route)"
    r"[^\n]{0,180}", re.IGNORECASE,
)
_STATE_ASSIGN_RE = re.compile(
    r"(?:^|\s)(?:self\.|[A-Za-z_]\w*\[|[A-Za-z_]\w*\.)?[A-Za-z_]\w*(?:\]|\.)?\s*(?:\+=|-=|\*=|/=|:=|=)"
)


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text(value: object, limit: int = MAX_QUERY_CHARS) -> str:
    result = str(value or "").replace("\x00", " ").strip()
    return result[:limit]


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _tokens(value: object) -> tuple[str, ...]:
    raw = _text(value).casefold()
    tokens: list[str] = []
    for token in _IDENTIFIER_RE.findall(raw):
        pieces = re.sub(r"([a-z])([A-Z])", r"\1 \2", token).casefold().split()
        pieces = re.split(r"[_\-]+", " ".join(pieces))
        for piece in pieces:
            if len(piece) >= 3 and piece not in _STOPWORDS and piece not in tokens:
                tokens.append(piece)
    return tuple(tokens[:64])


def _is_test_path(path: str) -> bool:
    parts = tuple(part.casefold() for part in path.replace("\\", "/").split("/"))
    name = parts[-1] if parts else ""
    return bool(set(parts) & _TEST_PARTS) or name.startswith("test_") or name.endswith(("_test.py", ".spec.ts", ".test.ts"))


def _safe_rel(root: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return None


def _forbidden_relative(relative: str) -> bool:
    return any(part.casefold() in _SKIP_DIRS for part in relative.split("/"))


@dataclass(frozen=True)
class SourceFile:
    path: str
    text: str
    digest: str
    size: int
    is_test: bool
    parse_status: str


@dataclass(frozen=True)
class SourceInventory:
    root: Path
    files: tuple[SourceFile, ...]
    omitted: tuple[dict[str, object], ...]
    manifest_digest: str

    @property
    def parse_failures(self) -> int:
        return sum(item.parse_status == "PARSE_FAILED" for item in self.files)

    @property
    def source_bytes(self) -> int:
        return sum(item.size for item in self.files)

    def coverage(self) -> dict[str, object]:
        return {
            "files_seen": len(self.files),
            "files_omitted": len(self.omitted),
            "parse_failures": self.parse_failures,
            "oversized_files": sum(item.get("reason") == "OVERSIZED" for item in self.omitted),
            "source_bytes": self.source_bytes,
            "manifest_digest": self.manifest_digest,
            "scope_status": "COMPLETE" if not self.omitted and not self.parse_failures else "INCOMPLETE",
        }


def build_inventory(root: Path, *, max_files: int = MAX_FILES) -> SourceInventory:
    """Read only public source files and retain explicit incompleteness."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("localization workspace root is not a directory")
    if root.name.casefold() in _SKIP_DIRS:
        raise ValueError("localization refuses a forbidden workspace scope")
    files: list[SourceFile] = []
    omitted: list[dict[str, object]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if len(files) >= max_files:
            omitted.append({"path": "<remaining>", "reason": "FILE_LIMIT"})
            break
        if candidate.is_symlink() or not candidate.is_file():
            continue
        relative = _safe_rel(root, candidate)
        if not relative or _forbidden_relative(relative) or candidate.suffix.casefold() not in _SOURCE_SUFFIXES:
            continue
        try:
            size = candidate.stat().st_size
            if size > MAX_SOURCE_BYTES:
                omitted.append({"path": relative, "reason": "OVERSIZED", "size": size})
                continue
            data = candidate.read_bytes()
            text = data.decode("utf-8", errors="replace")
        except (OSError, UnicodeError) as error:
            omitted.append({"path": relative, "reason": "READ_FAILED", "error": type(error).__name__})
            continue
        parse_status = "NOT_APPLICABLE"
        if candidate.suffix.casefold() in {".py", ".pyi"}:
            try:
                ast.parse(text, filename=relative)
                parse_status = "PARSED"
            except (SyntaxError, ValueError, MemoryError, RecursionError):
                parse_status = "PARSE_FAILED"
        files.append(SourceFile(
            path=relative,
            text=text,
            digest=hashlib.sha256(data).hexdigest(),
            size=size,
            is_test=_is_test_path(relative),
            parse_status=parse_status,
        ))
    manifest = _digest({"files": [(item.path, item.digest) for item in files], "omitted": omitted})
    return SourceInventory(root, tuple(files), tuple(omitted), manifest)


def _query_anchors(brief: object, query: object = "") -> dict[str, tuple[str, ...]]:
    text = " ".join(value for value in (_text(brief), _text(query)) if value)
    quoted = [match.group(2).strip() for match in _QUOTED_RE.finditer(text)]
    errors = [match.group(0).strip() for match in _ERROR_RE.finditer(text)]
    paths = _PATH_RE.findall(text)
    terms = list(_tokens(text))
    for value in (*quoted, *errors, *paths):
        for token in _tokens(value):
            if token not in terms:
                terms.append(token)
    return {
        "literals": _unique(value.casefold() for value in (*quoted, *errors, *paths) if len(value) >= 3),
        "errors": _unique(value.casefold() for value in errors),
        "terms": tuple(terms[:64]),
        "paths": _unique(value.casefold() for value in paths),
    }


def _line_excerpt(line: str, limit: int = 220) -> str:
    value = " ".join(line.strip().split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _source_ref(source: SourceFile, line: int, kind: str, *, excerpt: str | None = None) -> dict[str, object]:
    if excerpt is None:
        lines = source.text.splitlines()
        excerpt = lines[line - 1] if lines and line <= len(lines) else ""
    return {
        "ref_id": f"localize-source:{source.digest[:16]}:{line}:{kind}",
        "kind": kind,
        "path": source.path,
        "line": line,
        "source_digest": "sha256:" + source.digest,
        "excerpt": _line_excerpt(excerpt),
    }


def _definition_index(graph: object) -> Mapping[str, tuple[object, ...]]:
    cached = getattr(graph, "_localize_definition_index", None)
    if isinstance(cached, Mapping):
        return cached
    grouped: dict[str, list[object]] = defaultdict(list)
    for definition in getattr(graph, "defs", {}).values():
        path = str(getattr(definition, "file", "") or "")
        if path:
            grouped[path].append(definition)
    indexed = {
        path: tuple(sorted(
            values,
            key=lambda item: (
                int(getattr(item, "lineno", 0) or 0),
                int(getattr(item, "end_lineno", 0) or 0),
            ),
        ))
        for path, values in grouped.items()
    }
    starts = {
        path: tuple(int(getattr(item, "lineno", 0) or 0) for item in values)
        for path, values in indexed.items()
    }
    try:
        setattr(graph, "_localize_definition_index", indexed)
        setattr(graph, "_localize_definition_starts", starts)
    except (AttributeError, TypeError):
        pass
    return indexed


def _symbol_for_line(graph: object, path: str, line: int) -> object | None:
    definitions = _definition_index(graph).get(path, ())
    starts = getattr(graph, "_localize_definition_starts", {}).get(path, ())
    index = bisect.bisect_right(starts, line) - 1
    while index >= 0:
        item = definitions[index]
        if line <= int(getattr(item, "end_lineno", 0) or 0):
            return item
        index -= 1
    return None


def _entity_id(path: str, line: int, symbol: object | None = None, prefix: str = "source") -> str:
    exact = str(getattr(symbol, "symbol", "") or "")
    return f"symbol:{exact}" if exact else f"{prefix}:{path}:{line}"


def _candidate(
    *,
    entity_id: str,
    kind: str,
    status: str,
    path: str,
    line: int,
    score: float,
    source: SourceFile | None = None,
    symbol: object | None = None,
    evidence: Sequence[Mapping[str, object]] = (),
    **extra: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "entity_id": entity_id,
        "kind": kind,
        "status": status,
        "path": path,
        "line": line,
        "score": round(float(score), 6),
        "evidence": [dict(item) for item in evidence[:8]],
    }
    exact_symbol = str(getattr(symbol, "symbol", "") or "")
    if exact_symbol:
        result.update({
            "symbol_id": exact_symbol,
            "qualname": str(getattr(symbol, "qualname", "") or ""),
            "symbol_kind": str(getattr(symbol, "kind", "") or ""),
        })
    if source is not None:
        result["source_digest"] = "sha256:" + source.digest
    for key, value in extra.items():
        if value not in (None, "", (), [], {}):
            result[key] = value
    return result


def _add_evidence(target: dict[str, object], values: Sequence[Mapping[str, object]]) -> None:
    current = target.setdefault("evidence", [])
    if not isinstance(current, list):
        return
    seen = {str(item.get("ref_id")) for item in current if isinstance(item, Mapping)}
    for value in values:
        ref_id = str(value.get("ref_id", ""))
        if ref_id and ref_id not in seen and len(current) < 8:
            current.append(dict(value))
            seen.add(ref_id)


def _merge_record(target: dict[str, dict[str, object]], row: dict[str, object]) -> None:
    ident = str(row.get("entity_id", ""))
    if not ident:
        return
    existing = target.get(ident)
    if existing is None:
        target[ident] = row
        return
    existing["score"] = round(max(float(existing.get("score", 0.0)), float(row.get("score", 0.0))), 6)
    existing["status"] = _merge_status(str(existing.get("status", "")), str(row.get("status", "")))
    kinds = list(existing.get("kinds", ())) if isinstance(existing.get("kinds"), Sequence) and not isinstance(existing.get("kinds"), str) else [str(existing.get("kind", ""))]
    for kind in (str(row.get("kind", "")), *(str(item) for item in row.get("kinds", ()) if isinstance(row.get("kinds"), Sequence) and not isinstance(row.get("kinds"), str))):
        if kind and kind not in kinds:
            kinds.append(kind)
    existing["kinds"] = kinds[:8]
    _add_evidence(existing, tuple(item for item in row.get("evidence", ()) if isinstance(item, Mapping)))
    for key in ("candidate_symbol_ids", "bindings", "next_actions"):
        values = list(existing.get(key, ())) if isinstance(existing.get(key), Sequence) and not isinstance(existing.get(key), str) else []
        raw = row.get(key, ())
        if isinstance(raw, Sequence) and not isinstance(raw, str):
            for value in raw:
                if value not in values:
                    values.append(value)
        if values:
            existing[key] = values[:16]


def _merge_status(left: str, right: str) -> str:
    priority = {"UNRESOLVED_BINDING": 4, "AMBIGUOUS": 3, "MISSING_SCOPE": 2, "RESOLVED": 1, "": 0}
    return right if priority.get(right, 0) > priority.get(left, 0) else left


def _rank_channel(rows: Mapping[str, Mapping[str, object]], limit: int = DEFAULT_CHANNEL_LIMIT) -> list[dict[str, object]]:
    result = [dict(row) for row in rows.values()]
    result.sort(key=lambda row: (-float(row.get("score", 0.0)), str(row.get("entity_id", ""))))
    return result[:max(1, min(DEFAULT_CHANNEL_LIMIT, int(limit)))]


def _page(rows: Sequence[Mapping[str, object]], *, page: int, page_size: int) -> dict[str, object]:
    size = max(1, min(MAX_PAGE_SIZE, int(page_size)))
    index = max(0, int(page)) * size
    selected = [dict(row) for row in rows[index:index + size]]
    next_cursor = f"p:{max(0, int(page)) + 1}" if index + size < len(rows) else None
    return {
        "status": "OK" if selected or not rows else "EXHAUSTED",
        "page": max(0, int(page)),
        "page_size": size,
        "total": len(rows),
        "shown": len(selected),
        "items": selected,
        "next_cursor": next_cursor,
        "exhausted": next_cursor is None,
    }


def _safe_graph_symbol_ids(graph: object, value: object) -> tuple[str, ...]:
    """Resolve exact IDs without using CodeGraph.find_definition()."""
    query = _text(value, 400)
    defs = getattr(graph, "defs", {})
    variables = getattr(graph, "variables", {})
    if query in defs or query in variables:
        return (query,)
    short = query.split("::")[-1].split(".")[-1]
    found = [str(symbol) for symbol, definition in defs.items() if str(getattr(definition, "short_name", "")) == short or str(getattr(definition, "qualname", "")) == query]
    return tuple(sorted(dict.fromkeys(found)))


def resolve_symbol(graph: object, query: object) -> dict[str, object]:
    """Return every exact match, explicitly marking short-name ambiguity."""
    ids = _safe_graph_symbol_ids(graph, query)
    if not ids:
        return {"status": "NOT_FOUND", "query": _text(query), "symbol_ids": []}
    status = "RESOLVED" if len(ids) == 1 else "AMBIGUOUS"
    return {"status": status, "query": _text(query), "symbol_ids": list(ids)}


def _walk(graph: object, seeds: Sequence[str], *, reverse: bool) -> tuple[dict[str, tuple[str, ...]], int, int]:
    adjacency = getattr(graph, "_rev" if reverse else "_edges", {})
    queue: deque[tuple[str, tuple[str, ...], int]] = deque((seed, (seed,), 0) for seed in sorted(set(seeds)))
    seen: dict[str, tuple[str, ...]] = {}
    edges = 0
    while queue and len(seen) < MAX_GRAPH_NODES and edges < MAX_GRAPH_EDGES:
        current, path, depth = queue.popleft()
        if current in seen:
            continue
        seen[current] = path
        if depth >= MAX_GRAPH_DEPTH:
            continue
        for neighbor in sorted(str(item) for item in adjacency.get(current, ())):
            edges += 1
            if neighbor not in seen and len(seen) + len(queue) < MAX_GRAPH_NODES:
                queue.append((neighbor, path + (neighbor,), depth + 1))
            if edges >= MAX_GRAPH_EDGES:
                break
    return seen, len(seen), edges


def _graph_record(graph: object, symbol_id: str, path: Sequence[str], *, intersection: bool, direction: str) -> dict[str, object] | None:
    definition = getattr(graph, "defs", {}).get(symbol_id)
    if definition is None:
        return None
    edge_evidence: list[dict[str, object]] = []
    calls = tuple(getattr(graph, "calls", ()))
    for left, right in zip(path, path[1:]):
        caller, callee = (left, right) if direction != "backward" else (right, left)
        confidence = 0.0
        strategy = "NAME_MATCH"
        edge_confidence = getattr(graph, "edge_confidence", None)
        edge_strategy = getattr(graph, "edge_strategy", None)
        if callable(edge_confidence):
            confidence = float(edge_confidence(caller, callee) or 0.0)
        if callable(edge_strategy):
            strategy = str(edge_strategy(caller, callee) or "NAME_MATCH")
        callsites = [
            {
                "caller": str(getattr(call, "caller", "")),
                "callee_short": str(getattr(call, "callee_short", "")),
                "line": int(getattr(call, "lineno", 0) or 0),
                "name_line": int(getattr(call, "name_line", 0) or 0),
                "confidence": round(float(getattr(call, "confidence", 0.0) or 0.0), 6),
                "strategy": str(getattr(call, "strategy", "NAME_MATCH") or "NAME_MATCH"),
            }
            for call in calls
            if str(getattr(call, "caller", "")) == caller
            and str(getattr(call, "callee_short", "")) == str(callee).split("::")[-1].split(".")[-1]
        ][:4]
        edge_evidence.append({
            "caller": caller,
            "callee": callee,
            "confidence": round(confidence, 6),
            "strategy": strategy,
            "callsites": callsites,
        })
    row = _candidate(
        entity_id=f"symbol:{symbol_id}",
        kind="CALL_GRAPH_INTERSECTION" if intersection else "CALL_GRAPH_NEIGHBOR",
        status="RESOLVED",
        path=str(getattr(definition, "file", "")),
        line=int(getattr(definition, "lineno", 0) or 0),
        score=42.0 if intersection else 16.0,
        symbol=definition,
        edge_path=list(path),
        edge_evidence=edge_evidence[:8],
        direction=direction,
        next_actions=["read_symbol", "walk_callers", "walk_callees"],
    )
    row["edge_evidence"] = edge_evidence[:8]
    return row


def _enclosing_defs(graph: object, source: SourceFile, line: int) -> Sequence[object]:
    return tuple(item for item in getattr(graph, "defs", {}).values() if getattr(item, "file", "") == source.path and int(getattr(item, "lineno", 0) or 0) <= line <= int(getattr(item, "end_lineno", 0) or 0))


def _text_match_score(line: str, anchors: Mapping[str, Sequence[str]]) -> float:
    folded = line.casefold()
    literals = anchors.get("literals", ())
    terms = anchors.get("terms", ())
    return float(sum(45 for literal in literals if literal and literal in folded) + sum(4 for term in terms if term in folded))


def _add_line_candidate(
    channel: dict[str, dict[str, object]],
    *,
    graph: object,
    source: SourceFile,
    line_number: int,
    kind: str,
    score: float,
    evidence_kind: str,
    status: str = "RESOLVED",
    excerpt: str | None = None,
    **extra: object,
) -> None:
    symbol = _symbol_for_line(graph, source.path, line_number)
    ident = _entity_id(source.path, line_number, symbol, prefix=kind.casefold())
    evidence = (_source_ref(source, line_number, evidence_kind, excerpt=excerpt),)
    row = _candidate(
        entity_id=ident, kind=kind, status=status, path=source.path, line=line_number,
        score=score, source=source, symbol=symbol, evidence=evidence, **extra,
    )
    _merge_record(channel, row)


def _literal_channel(inventory: SourceInventory, graph: object, anchors: Mapping[str, Sequence[str]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for source in inventory.files:
        for number, line in enumerate(source.text.splitlines(), 1):
            score = _text_match_score(line, anchors)
            if score <= 0:
                continue
            _add_line_candidate(result, graph=graph, source=source, line_number=number, kind="LITERAL_MATCH", score=score, evidence_kind="literal", excerpt=line)
    return result


def _lexical_channel(inventory: SourceInventory, graph: object, anchors: Mapping[str, Sequence[str]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    terms = tuple(anchors.get("terms", ()))
    by_short: dict[str, list[str]] = defaultdict(list)
    definitions = getattr(graph, "defs", {})
    for symbol_id, definition in definitions.items():
        if _is_test_path(str(getattr(definition, "file", "") or "")):
            continue
        short = str(getattr(definition, "short_name", "") or "").casefold()
        if short:
            by_short[short].append(str(symbol_id))
    for short in by_short:
        by_short[short].sort()
    for symbol_id, definition in definitions.items():
        if _is_test_path(str(getattr(definition, "file", "") or "")):
            continue
        short = str(getattr(definition, "short_name", "") or "").casefold()
        name = f"{getattr(definition, 'qualname', '')} {getattr(definition, 'file', '')}".casefold()
        matches = sum(12 for term in terms if term in name)
        if matches <= 0:
            continue
        short_matches = len(by_short.get(short, ()))
        status = "AMBIGUOUS" if short_matches > 1 else "RESOLVED"
        result[f"symbol:{symbol_id}"] = _candidate(
            entity_id=f"symbol:{symbol_id}", kind="LEXICAL_SYMBOL", status=status,
            path=str(getattr(definition, "file", "")), line=int(getattr(definition, "lineno", 0) or 0),
            score=matches, symbol=definition,
            evidence=({
                "ref_id": f"localize-symbol:{_digest(symbol_id)[:16]}",
                "kind": "lexical_symbol_match", "path": str(getattr(definition, "file", "")),
                "line": int(getattr(definition, "lineno", 0) or 0), "terms": [term for term in terms if term in name],
            },),
            candidate_symbol_ids=[str(item) for item in by_short.get(short, ())] if status == "AMBIGUOUS" else (),
            next_actions=["read_symbol", "who_calls", "what_it_calls"],
        )
    return result


def _entrypoint_channel(inventory: SourceInventory, graph: object, anchors: Mapping[str, Sequence[str]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    terms = tuple(anchors.get("terms", ()))
    for definition in getattr(graph, "defs", {}).values():
        if _is_test_path(str(getattr(definition, "file", "") or "")):
            continue
        if not bool(getattr(definition, "is_entrypoint", False)):
            continue
        name = f"{getattr(definition, 'qualname', '')} {getattr(definition, 'file', '')}".casefold()
        score = 22.0 + sum(5 for term in terms if term in name)
        row = _candidate(
            entity_id=f"symbol:{getattr(definition, 'symbol', '')}", kind="ENTRYPOINT", status="RESOLVED",
            path=str(getattr(definition, "file", "")), line=int(getattr(definition, "lineno", 0) or 0), score=score,
            symbol=definition, evidence=({
                "ref_id": f"localize-entry:{_digest(getattr(definition, 'symbol', ''))[:16]}",
                "kind": "entry_registration_or_entrypoint", "path": str(getattr(definition, "file", "")),
                "line": int(getattr(definition, "lineno", 0) or 0),
            },), next_actions=["read_entrypoint", "walk_forward"],
        )
        _merge_record(result, row)
    for source in inventory.files:
        if source.is_test:
            continue
        for number, line in enumerate(source.text.splitlines(), 1):
            match = _ROUTE_RE.search(line)
            if not match:
                continue
            score = 18.0 + sum(4 for term in terms if term in line.casefold())
            _add_line_candidate(result, graph=graph, source=source, line_number=number, kind="ENTRY_REGISTRATION", score=score, evidence_kind="entry_registration", excerpt=line, registration=match.group(0).strip()[:180], next_actions=["read_registration", "resolve_binding"])
    return result


def _state_channel(inventory: SourceInventory, graph: object, anchors: Mapping[str, Sequence[str]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    terms = tuple(anchors.get("terms", ()))
    for source in inventory.files:
        if source.is_test or Path(source.path).suffix.casefold() not in _CODE_SUFFIXES:
            continue
        for number, line in enumerate(source.text.splitlines(), 1):
            folded = line.casefold()
            term_hits = [term for term in terms if term in folded]
            if not term_hits:
                continue
            is_write = bool(_STATE_ASSIGN_RE.search(line)) or any(token in folded for token in (".update(", ".set(", "setdefault(", "append("))
            kind = "STATE_WRITER" if is_write else "STATE_READER"
            score = 14.0 + sum(5 for _ in term_hits) + (7.0 if is_write else 0.0)
            _add_line_candidate(result, graph=graph, source=source, line_number=number, kind=kind, score=score, evidence_kind="state_field_use", excerpt=line, state_terms=term_hits, next_actions=["walk_state_writers" if not is_write else "walk_state_readers"])
    return result


def _public_test_channel(inventory: SourceInventory, graph: object, anchors: Mapping[str, Sequence[str]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    terms = tuple(anchors.get("terms", ()))
    literals = tuple(anchors.get("literals", ()))
    for source in inventory.files:
        if not source.is_test:
            continue
        lines = source.text.splitlines()
        for number, line in enumerate(lines, 1):
            folded = line.casefold()
            hits = [term for term in terms if term in folded]
            hits.extend(literal for literal in literals if literal in folded)
            if not hits:
                continue
            symbol = _symbol_for_line(graph, source.path, number)
            ident = _entity_id(source.path, number, symbol, prefix="public_test")
            row = _candidate(
                entity_id=ident, kind="PUBLIC_TEST", status="RESOLVED", path=source.path,
                line=number, score=20.0 + 4.0 * len(set(hits)), source=source, symbol=symbol,
                evidence=(_source_ref(source, number, "public_test_match", excerpt=line),),
                matched_terms=list(dict.fromkeys(hits))[:16], next_actions=["read_test", "run_parent_test"],
            )
            _merge_record(result, row)
    return result


def _dynamic_bindings(inventory: SourceInventory, graph: object, anchors: Mapping[str, Sequence[str]]) -> list[dict[str, object]]:
    terms = tuple(anchors.get("terms", ()))
    rows: list[dict[str, object]] = []
    try:
        from mc_repo_intel_vendor.call_graph import dispatch_boundaries_in_file
    except ImportError:
        return rows
    for source in inventory.files:
        boundaries = dispatch_boundaries_in_file(source.text, source.path)
        for boundary in boundaries:
            line = int(getattr(boundary, "lineno", 0) or 0)
            text = f"{getattr(boundary, 'key_expr', '')} {getattr(boundary, 'note', '')}".casefold()
            if terms and not any(term in text for term in terms):
                continue
            symbol = _symbol_for_line(graph, source.path, line)
            rows.append(_candidate(
                entity_id=f"binding:{source.path}:{line}:{getattr(boundary, 'kind', '')}",
                kind="DYNAMIC_REGISTRATION", status="UNRESOLVED_BINDING", path=source.path, line=line,
                score=28.0 + sum(3 for term in terms if term in text), source=source, symbol=symbol,
                evidence=(_source_ref(source, line, "dynamic_registration"),),
                binding_kind=str(getattr(boundary, "kind", "")), key_expr=str(getattr(boundary, "key_expr", "")),
                note=str(getattr(boundary, "note", ""))[:240], next_actions=["search_registration", "resolve_runtime_binding"],
            ))
    return rows


def _graph_expansion(graph: object, channels: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, object]:
    trigger_ids = [str(row.get("symbol_id")) for name in ("entrypoint", "literal", "public_test") for row in channels.get(name, ()) if row.get("symbol_id") and row.get("status") == "RESOLVED"]
    outcome_ids = [str(row.get("symbol_id")) for name in ("state", "literal", "public_test") for row in channels.get(name, ()) if row.get("symbol_id") and row.get("status") == "RESOLVED"]
    forward, forward_nodes, forward_edges = _walk(graph, trigger_ids[:20], reverse=False) if trigger_ids else ({}, 0, 0)
    backward, backward_nodes, backward_edges = _walk(graph, outcome_ids[:20], reverse=True) if outcome_ids else ({}, 0, 0)
    intersection = sorted(set(forward).intersection(backward))
    neighbors = sorted(set(forward).union(backward) - set(intersection))
    records: list[dict[str, object]] = []
    for symbol_id in intersection:
        row = _graph_record(graph, symbol_id, forward[symbol_id], intersection=True, direction="forward_to_backward")
        if row is not None:
            row["backward_path"] = list(backward[symbol_id])
            records.append(row)
    for symbol_id in neighbors[:MAX_GRAPH_NODES]:
        direction = "forward" if symbol_id in forward else "backward"
        path = forward.get(symbol_id, backward.get(symbol_id, (symbol_id,)))
        row = _graph_record(graph, symbol_id, path, intersection=False, direction=direction)
        if row is not None:
            records.append(row)
    return {
        "trigger_seed_count": len(set(trigger_ids)), "outcome_seed_count": len(set(outcome_ids)),
        "forward_nodes": forward_nodes, "backward_nodes": backward_nodes,
        "forward_edges": forward_edges, "backward_edges": backward_edges,
        "intersection_ids": [f"symbol:{item}" for item in intersection],
        "bounded": True, "records": records,
    }


def _estimate_tokens(value: object) -> int:
    try:
        from kernel_context import token_estimate
        return int(token_estimate(value))
    except (ImportError, TypeError, ValueError):
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        return max(1, math.ceil(len(encoded) / 4))


class LocalizeIndex:
    """Cached public snapshot/index with deterministic refresh on content drift."""

    def __init__(
        self,
        root: Path,
        *,
        use_graph_cache: bool = False,
        use_graph_lsp: bool = False,
    ) -> None:
        self.root = root.resolve()
        self.use_graph_cache = bool(use_graph_cache)
        self.use_graph_lsp = bool(use_graph_lsp)
        self.inventory: SourceInventory | None = None
        self.graph: object | None = None
        self._manifest_digest = ""

    def refresh(self) -> None:
        inventory = build_inventory(self.root)
        if inventory.manifest_digest == self._manifest_digest and self.inventory is not None:
            return
        self.inventory = inventory
        self._manifest_digest = inventory.manifest_digest
        try:
            from mc_repo_intel_vendor.call_graph import build_graph
            with _GRAPH_BUILD_LOCK:
                previous_lsp = os.environ.get("CODE_GRAPH_LSP")
                if not self.use_graph_lsp:
                    os.environ["CODE_GRAPH_LSP"] = "off"
                try:
                    self.graph = build_graph(
                        str(self.root), use_cache=self.use_graph_cache, use_memo=True,
                    )
                finally:
                    if previous_lsp is None:
                        os.environ.pop("CODE_GRAPH_LSP", None)
                    else:
                        os.environ["CODE_GRAPH_LSP"] = previous_lsp
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("localization graph unavailable: %s", type(error).__name__)
            self.graph = _EmptyGraph()

    def search(
        self,
        brief: object,
        *,
        query: object = "",
        page: int = 0,
        page_size: int = DEFAULT_PAGE_SIZE,
        cursor: object = None,
        channel: str | None = None,
        max_per_channel: int = DEFAULT_CHANNEL_LIMIT,
    ) -> dict[str, object]:
        if cursor not in (None, ""):
            match = re.fullmatch(r"p:(\d{1,6})", _text(cursor, 32))
            if match is None:
                raise ValueError("invalid localization cursor")
            page = int(match.group(1))
        self.refresh()
        assert self.inventory is not None
        graph = self.graph or _EmptyGraph()
        anchors = _query_anchors(brief, query)
        raw_channels: dict[str, dict[str, dict[str, object]]] = {
            "literal": _literal_channel(self.inventory, graph, anchors),
            "lexical": _lexical_channel(self.inventory, graph, anchors),
            "entrypoint": _entrypoint_channel(self.inventory, graph, anchors),
            "state": _state_channel(self.inventory, graph, anchors),
            "public_test": _public_test_channel(self.inventory, graph, anchors),
        }
        ranked = {name: _rank_channel(rows, max_per_channel) for name, rows in raw_channels.items()}
        graph_info = _graph_expansion(graph, ranked)
        dynamic = _dynamic_bindings(self.inventory, graph, anchors)
        merged: dict[str, dict[str, object]] = {}
        for name, rows in ranked.items():
            for row in rows:
                row["channels"] = [name]
                _merge_record(merged, row)
        for row in graph_info["records"]:
            row["channels"] = ["graph"]
            _merge_record(merged, row)
        for row in dynamic:
            row["channels"] = ["dynamic_binding"]
            _merge_record(merged, row)
        graph_records = sorted(
            (dict(row) for row in graph_info["records"]),
            key=lambda row: (-float(row.get("score", 0.0)), str(row.get("entity_id", ""))),
        )[:MAX_PAGE_SIZE]
        all_suspects = sorted(merged.values(), key=lambda row: (-float(row.get("score", 0.0)), str(row.get("entity_id", ""))))
        for rank, row in enumerate(all_suspects, 1):
            row["rank"] = rank
            row["menu_label"] = f"{row.get('kind', 'SUSPECT')} {row.get('path', '')}:{row.get('line', '')}"
        selected_channels = CHANNELS if channel is None else (str(channel),)
        invalid = [name for name in selected_channels if name not in CHANNELS]
        if invalid:
            raise ValueError(f"unknown localization channel: {invalid[0]}")
        channel_pages = {name: _page(ranked[name], page=page, page_size=page_size) for name in selected_channels}
        suspect_page = _page(all_suspects, page=page, page_size=page_size)
        menu = [dict(row) for row in suspect_page["items"]]
        source_receipts = tuple(
            f"localize-source-scope:{item.path}:{item.digest}" for item in self.inventory.files
        )
        payload: dict[str, object] = {
            "schema": SCHEMA,
            "status": "SEARCHED",
            "snapshot_id": self.inventory.manifest_digest,
            "scope_digest": self.inventory.manifest_digest,
            "query": _text(query),
            "anchors": {key: list(value) for key, value in anchors.items()},
            "channels": channel_pages,
            "suspects": menu,
            "suspect_total": len(all_suspects),
            "suspect_page": suspect_page,
            "dynamic_bindings": dynamic[:DEFAULT_CHANNEL_LIMIT],
            "graph": {
                key: value for key, value in graph_info.items() if key != "records"
            } | {
                "limits": {
                    "max_depth": MAX_GRAPH_DEPTH,
                    "max_nodes": MAX_GRAPH_NODES,
                    "max_edges": MAX_GRAPH_EDGES,
                },
                "records": graph_records,
            },
            "frontier": [
                {"action_id": "page_channels", "purpose": "continue each retrieval channel", "next_cursor": page_value.get("next_cursor")}
                for page_value in channel_pages.values() if page_value.get("next_cursor")
            ] + [
                {"action_id": "resolve_runtime_binding", "purpose": "inspect dynamic registration evidence"}
                for _row in dynamic[:1]
            ],
            "coverage": self.inventory.coverage(),
            "source_receipt_ids": list(source_receipts[:64]),
        }
        payload["tokens"] = _estimate_tokens(payload)
        payload["receipt"] = {
            "schema": RECEIPT_SCHEMA,
            "receipt_id": "localize-search:" + _digest({"snapshot": self.inventory.manifest_digest, "query": _text(query), "page": page, "channel": channel})[7:31],
            "snapshot_id": self.inventory.manifest_digest,
            "scope_digest": self.inventory.manifest_digest,
            "source_receipt_ids": list(source_receipts[:64]),
            "channel_totals": {name: len(ranked[name]) for name in CHANNELS},
            "page": max(0, int(page)),
            "page_size": max(1, min(MAX_PAGE_SIZE, int(page_size))),
            "tokens": payload["tokens"],
        }
        return payload


class _EmptyGraph:
    defs: dict[str, object] = {}
    variables: dict[str, object] = {}
    _edges: dict[str, set[str]] = {}
    _rev: dict[str, set[str]] = {}


def localize(root: Path, brief: object, **kwargs: object) -> dict[str, object]:
    """One-shot convenience API used by tests and request qualification adapters."""
    return LocalizeIndex(root).search(brief, **kwargs)


def validate_binding(result: Mapping[str, object], *, candidate_id: object, path: object, symbol: object = "") -> dict[str, object]:
    """Validate a model-selected suspect against a prior host search receipt."""
    ident = _text(candidate_id, 300)
    expected_path = _text(path, 500).replace("\\", "/").lstrip("./")
    expected_symbol = _text(symbol, 400)
    candidates: list[Mapping[str, object]] = []
    raw = result.get("suspects", ())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        candidates = [item for item in raw if isinstance(item, Mapping)]
    candidate = next((item for item in candidates if str(item.get("entity_id", "")) == ident), None)
    if candidate is None:
        return {"accepted": False, "code": "UNKNOWN_LOCALIZATION_CANDIDATE", "candidate_id": ident}
    if str(candidate.get("status", "")) != "RESOLVED":
        return {"accepted": False, "code": str(candidate.get("status", "LOCALIZATION_NOT_RESOLVED")), "candidate_id": ident}
    actual_path = str(candidate.get("path", "")).replace("\\", "/").lstrip("./")
    actual_symbol = str(candidate.get("symbol_id", ""))
    if actual_path != expected_path or (expected_symbol and actual_symbol != expected_symbol):
        return {"accepted": False, "code": "LOCALIZATION_IDENTITY_MISMATCH", "candidate_id": ident}
    return {
        "accepted": True,
        "code": "LOCALIZATION_CANDIDATE_VALIDATED",
        "candidate_id": ident,
        "path": actual_path,
        "symbol_id": actual_symbol,
        "evidence_refs": [str(item.get("ref_id")) for item in candidate.get("evidence", ()) if isinstance(item, Mapping) and item.get("ref_id")],
    }


def result_token_count(result: Mapping[str, object]) -> int:
    raw = result.get("tokens")
    return int(raw) if isinstance(raw, int) else _estimate_tokens(result)


__all__ = [
    "CHANNELS", "DEFAULT_CHANNEL_LIMIT", "DEFAULT_PAGE_SIZE", "INDEX_SCHEMA", "LocalizeIndex",
    "RECEIPT_SCHEMA", "SCHEMA", "build_inventory", "localize", "resolve_symbol",
    "result_token_count", "validate_binding",
]
