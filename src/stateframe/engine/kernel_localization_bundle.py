"""Bounded public-source localization assembled from existing read-only owners."""
from __future__ import annotations

import hashlib
import json
import os
import time
from math import ceil
from collections.abc import Mapping, Sequence
from pathlib import Path

from kernel_public_localization import (
    build_parent_index,
    public_candidate_sites,
    public_identifiers,
    public_path,
    public_ticket,
)
from kernel_localization_spans import (
    MAX_ACQUISITION_BYTES, MAX_CONTEXT_BYTES, MAX_SCAN_BYTES, acquire_owner_spans,
    rank_owner_spans_by_words, ticket_words,
)


SCHEMA = "stateframe-localization-bundle/1"
MAX_CANDIDATES = 3
MAX_HOPS = 2
MAX_RELATIONS = 24
MAX_SOURCE_CHARS = 6000
MAX_EVIDENCE_TOKENS = 1500
MAX_GRAPH_FILES = 64
MAX_GRAPH_BYTES = 2 * 1024 * 1024
MAX_DECLARED_PATHS = 64
MAX_LARGE_SOURCE_BYTES = MAX_SOURCE_CHARS * 4
_PRIVATE_PARTS = frozenset({
    ".git", ".stateframe",
    "__pycache__", "node_modules", "dist", "build", "coverage", ".mission",
    "grader", "graders", "gold", "private", "hidden", "hidden_tests", "backups",
    "evaluator", "evaluators", "oracle", "sentinel", "reports",
})


def _digest(value: bytes | str | object) -> str:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _public_manifest(value: object) -> object:
    """Project manifest identity before it can affect indexing or cache keys."""
    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for key in sorted(value, key=lambda item: str(item)):
            public_key = public_ticket(str(key)).strip()
            if public_key.casefold() in {"private", "hidden", "evaluator", "evaluators", "oracle", "sentinel", "grader", "graders", "gold"}:
                continue
            public_value = _public_manifest(value[key])
            if public_key and public_value is not None:
                projected[public_key] = public_value
        return projected or "public-manifest"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = [_public_manifest(item) for item in value]
        return [item for item in values if item is not None]
    if isinstance(value, str):
        text = public_ticket(value).strip()
        return text or None
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return public_ticket(str(value)).strip() or None


def _token_count(value: object) -> int:
    text = str(value or "")
    return ceil(len(text) / 4) if text else 0


def _safe_paths(paths: Sequence[str], *, max_paths: int = MAX_DECLARED_PATHS) -> tuple[str, ...]:
    result: list[str] = []
    for value in paths:
        normalized = public_path(value)
        if normalized is None or any(part.casefold() in _PRIVATE_PARTS for part in normalized.split("/")):
            continue
        if normalized not in result:
            result.append(normalized)
    return tuple(sorted(result)[:max_paths])


def _source_map(
        root: Path | None,
        paths: Sequence[str],
        overrides: Mapping[str, bytes | str],
) -> tuple[dict[str, str], list[dict[str, object]]]:
    sources: dict[str, str] = {}
    omissions: list[dict[str, object]] = []
    for path in paths:
        if path in overrides:
            value = overrides[path]
            raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
            if len(raw) > MAX_LARGE_SOURCE_BYTES:
                omissions.append({"path": path, "kind": "source_deferred", "reason": "bounded_owner_span"})
            else:
                sources[path] = raw.decode("utf-8", errors="replace")
            continue
        if root is None:
            omissions.append({"path": path, "kind": "source_unavailable", "reason": "workspace_unavailable"})
            continue
        raw_candidate = root / path
        if raw_candidate.is_symlink():
            omissions.append({"path": path, "kind": "access_denied", "reason": "symlink"})
            continue
        candidate = raw_candidate.resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            omissions.append({"path": path, "kind": "access_denied", "reason": "outside_workspace"})
            continue
        path_parts: list[Path] = []
        cursor = raw_candidate
        while cursor != root and root in cursor.parents:
            path_parts.append(cursor)
            cursor = cursor.parent
        if any(item.is_symlink() for item in path_parts) or candidate.is_symlink():
            omissions.append({"path": path, "kind": "access_denied", "reason": "symlink"})
            continue
        try:
            if candidate.stat().st_size > MAX_LARGE_SOURCE_BYTES:
                omissions.append({"path": path, "kind": "source_deferred", "reason": "bounded_owner_span"})
                continue
        except OSError as error:
            omissions.append({"path": path, "kind": "source_unavailable", "reason": type(error).__name__})
            continue
        try:
            raw = candidate.read_bytes()
        except OSError as error:
            omissions.append({"path": path, "kind": "source_unavailable", "reason": type(error).__name__})
            continue
        sources[path] = raw.decode("utf-8", errors="replace")
    return sources, omissions


def _segment(source: str, path: str, line: int, end_line: int | None = None) -> tuple[dict[str, object], dict[str, object] | None]:
    lines = source.splitlines(keepends=True)
    start = max(1, int(line or 1))
    end = max(start, int(end_line or start))
    if start > len(lines):
        return {"path": path, "start_line": start, "end_line": start - 1, "content": ""}, {"kind": "span_out_of_range", "path": path}
    selected = "".join(lines[start - 1:end])
    omitted = None
    requested_end_line = min(end, len(lines))
    clipped = False
    if len(selected) > MAX_SOURCE_CHARS:
        selected = selected[:MAX_SOURCE_CHARS]
        clipped = True
        shown_lines = selected.count("\n") + (0 if selected.endswith("\n") else 1)
        shown_end_line = start + max(0, shown_lines - 1)
        omitted = {"kind": "segment_clipped", "path": path,
                   "start_line": start, "end_line": requested_end_line,
                   "shown_end_line": shown_end_line}
    else:
        shown_end_line = requested_end_line
    return {
        "path": path, "start_line": start, "end_line": shown_end_line,
        "requested_end_line": requested_end_line, "content": selected,
        "content_digest": _digest(selected), "clipped": clipped,
        "coverage": "PARTIAL" if clipped else "COMPLETE",
    }, omitted


def _bounded_repo(root: Path, *, deadline: float | None = None) -> tuple[bool, int, int, str | None]:
    count = 0
    total = 0
    try:
        pending = [root]
        while pending:
            directory = pending.pop()
            if deadline is not None and time.monotonic() >= deadline:
                return False, count, total, "deadline"
            child_directories: list[Path] = []
            with os.scandir(directory) as entries:
                for entry in entries:
                    if deadline is not None and time.monotonic() >= deadline:
                        return False, count, total, "deadline"
                    name = entry.name
                    if name.startswith(".") or name.casefold() in _PRIVATE_PARTS or entry.is_symlink():
                        continue
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        child_directories.append(path)
                        if len(child_directories) > MAX_GRAPH_FILES * 4:
                            return False, count, total, "budget"
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    relative = path.relative_to(root)
                    if any(part.casefold() in _PRIVATE_PARTS for part in relative.parts):
                        continue
                    count += 1
                    try:
                        total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
                    if count > MAX_GRAPH_FILES or total > MAX_GRAPH_BYTES:
                        return False, count, total, "budget"
            # Deterministic traversal keeps dependency enumeration stable while
            # retaining a bounded directory frontier.
            pending.extend(sorted(child_directories, key=lambda item: str(item), reverse=True))
    except OSError:
        return False, count, total, "unavailable"
    return True, count, total, None


_GRAPH_CODE_SUFFIXES = frozenset({
    ".py", ".pyi", ".go", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".java", ".kt", ".rs",
    ".rb", ".php", ".cs", ".scala", ".swift", ".c", ".h", ".cc", ".cpp", ".hpp",
})
MAX_SCOPE_SCAN_BYTES = 32 * 1024 * 1024


def _scoped_graph_files(
        root: Path, declared: Sequence[str], symbols: Sequence[str], *, deadline: float,
) -> tuple[list[str], dict[str, object]]:
    """Task-scoped graph input when the whole repo exceeds the graph budget.

    Scope = declared files plus code files that mention a candidate symbol as a
    whole word (its direct callers live there), within the SAME file/byte
    budget as the whole-repo graph.  The walk is bounded by a scan cap and the
    deadline.  Callers outside the scope are not claimed absent: the scope is
    reported, and the result is labelled task_scoped.
    """
    import re
    patterns = [re.compile(rb"(?<![A-Za-z0-9_$])" + re.escape(symbol.rsplit(".", 1)[-1].encode()) +
                           rb"(?![A-Za-z0-9_$])") for symbol in symbols if symbol.strip()]
    selected: list[str] = [path for path in declared if (root / path).is_file()]
    total = sum((root / path).stat().st_size for path in selected)
    scanned = 0
    truncated = False
    pending = [root]
    while pending and not truncated:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError:
            continue
        children: list[Path] = []
        for entry in entries:
            if time.monotonic() >= deadline or scanned > MAX_SCOPE_SCAN_BYTES:
                truncated = True
                break
            if entry.name.casefold() in _PRIVATE_PARTS or entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                # Hidden directories (worktrees, caches, VCS) hold copies or
                # tooling, never the task's own callers.
                if not entry.name.startswith("."):
                    children.append(Path(entry.path))
                continue
            relative = Path(entry.path).relative_to(root).as_posix()
            if (not entry.is_file(follow_symlinks=False)
                    or Path(relative).suffix.lower() not in _GRAPH_CODE_SUFFIXES
                    or relative in selected):
                continue
            try:
                size = entry.stat(follow_symlinks=False).st_size
                data = Path(entry.path).read_bytes() if size <= MAX_GRAPH_BYTES else b""
            except OSError:
                continue
            scanned += len(data)
            if not data or not any(pattern.search(data) for pattern in patterns):
                continue
            if len(selected) + 1 > MAX_GRAPH_FILES or total + size > MAX_GRAPH_BYTES:
                truncated = True
                break
            selected.append(relative)
            total += size
        pending.extend(sorted(children, key=lambda item: str(item), reverse=True))
    return selected, {"mode": "task_scoped", "files": len(selected), "bytes": total,
                      "scanned_bytes": scanned, "truncated": truncated}


def _public_graph_file(value: object) -> str | None:
    """Accept only repository-relative public graph file identities."""
    normalized = public_path(value)
    if normalized is None or any(part.casefold() in _PRIVATE_PARTS for part in normalized.split("/")):
        return None
    return normalized


def _graph_relations(
        root: Path | None,
        symbols: Sequence[str],
        *,
        allowed_paths: frozenset[str],
        deadline: float,
        declared_paths: Sequence[str] = (),
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    if root is None:
        return [], {"status": "UNAVAILABLE", "reason": "workspace_unavailable"}, [{"kind": "graph_unavailable", "reason": "workspace_unavailable"}]
    if time.monotonic() >= deadline:
        return [], {"status": "PARTIAL", "reason": "deadline"}, [{"kind": "graph_deadline"}]
    bounded, file_count, total_bytes, bound_reason = _bounded_repo(root, deadline=deadline)
    scope_info: dict[str, object] = {"mode": "repository"}
    graph_root = root
    mirror = None
    if not bounded:
        if bound_reason == "deadline":
            return [], {"status": "PARTIAL", "reason": "deadline", "files": file_count, "bytes": total_bytes}, [{"kind": "graph_deadline"}]
        # W8: a real repository exceeds the whole-repo budget (AHA: 2.9 MB),
        # so build the graph over a task-scoped mirror within the same budget.
        scoped, scope_info = _scoped_graph_files(root, declared_paths, symbols, deadline=deadline)
        if not scoped or time.monotonic() >= deadline:
            return [], {"status": "UNAVAILABLE", "reason": "graph_budget", "files": file_count, "bytes": total_bytes,
                        "scope": scope_info}, [{"kind": "graph_budget", "files": file_count, "bytes": total_bytes}]
        import shutil
        import tempfile
        mirror = tempfile.TemporaryDirectory(prefix="stateframe-scoped-graph-")
        graph_root = Path(mirror.name)
        for relative in scoped:
            target = graph_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
        file_count, total_bytes = int(scope_info["files"]), int(scope_info["bytes"])
    try:
        from mc_repo_intel_vendor.call_graph import build_graph
        # The mirror is disposable: no cache file is written into it or the workspace.
        graph = (build_graph(str(graph_root), use_cache=False, use_memo=False) if mirror is not None
                 else build_graph(str(root), use_cache=True, use_memo=True))
        if mirror is not None:
            mirror.cleanup()
        if time.monotonic() >= deadline:
            return [], {"status": "PARTIAL", "reason": "deadline", "files": file_count,
                        "bytes": total_bytes}, [{"kind": "graph_deadline"}]
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        if mirror is not None:
            mirror.cleanup()
        return [], {"status": "UNAVAILABLE", "reason": type(error).__name__}, [{"kind": "graph_parser", "reason": type(error).__name__}]
    relations: list[dict[str, object]] = []
    omissions: list[dict[str, object]] = []
    summary = graph.coverage_summary() if callable(getattr(graph, "coverage_summary", None)) else {}
    incomplete = sum(int(summary.get(key, 0)) for key in ("parse_partial", "skipped", "unknown"))
    graph_status = "PARTIAL" if incomplete else "FOUND"
    if incomplete:
        omissions.append({"kind": "graph_coverage", "incomplete_files": incomplete,
                          "coverage": dict(summary)})
    for symbol in symbols:
        if time.monotonic() >= deadline:
            omissions.append({"kind": "graph_deadline", "symbol": symbol})
            break
        candidates = graph.definition_candidates(symbol)
        if not candidates:
            graph_status = "PARTIAL"
            omissions.append({"kind": "graph_definition_unavailable", "symbol": symbol})
            continue
        public_definitions = 0
        for candidate in candidates[:MAX_CANDIDATES]:
            candidate_file = _public_graph_file(getattr(candidate, "file", ""))
            if candidate_file not in allowed_paths:
                continue
            public_definitions += 1
            callers, work = graph.find_callers_bounded(candidate.symbol, max_depth=MAX_HOPS, max_nodes=MAX_RELATIONS, max_edges=MAX_RELATIONS)
            if isinstance(work, Mapping) and work.get("truncated"):
                omissions.append({"kind": "graph_truncated", "symbol": symbol, "work": dict(work)})
            # W8: hop distance must be truthful.  Only depth-1 callers are
            # direct CALLS; deeper ancestors carry an upper bound, not a guess.
            direct, _direct_work = graph.find_callers_bounded(
                candidate.symbol, max_depth=1, max_nodes=MAX_RELATIONS, max_edges=MAX_RELATIONS)
            direct_keys = {(str(getattr(item, "file", "")), str(getattr(item, "symbol", item)))
                           for item in direct}
            public_relations = 0
            for relation in callers[:MAX_RELATIONS - len(relations)]:
                relation_file = _public_graph_file(getattr(relation, "file", ""))
                if relation_file is None:
                    continue
                public_relations += 1
                source_symbol = str(getattr(relation, "symbol", relation))
                is_direct = (str(getattr(relation, "file", "")), source_symbol) in direct_keys
                relations.append({"from": f"{relation_file}:{source_symbol}",
                                  "to": f"{candidate_file}:{candidate.symbol}",
                                  **({"relation": "CALLS", "distance": 1} if is_direct else
                                     {"relation": "CALLS_TRANSITIVE", "distance": None,
                                      "max_distance": MAX_HOPS})})
            if callers and not public_relations:
                graph_status = "PARTIAL"
                omissions.append({"kind": "graph_relation_scope", "symbol": symbol})
        if not public_definitions:
            graph_status = "PARTIAL"
            omissions.append({"kind": "graph_definition_scope", "symbol": symbol})
    if scope_info.get("mode") == "task_scoped":
        # Callers outside the scope are unobserved, not absent.
        graph_status = "PARTIAL"
        omissions.append({"kind": "graph_scope", "reason": "task_scoped", **scope_info})
    return relations[:MAX_RELATIONS], {"status": graph_status, "files": file_count,
                                       "bytes": total_bytes, "max_depth": MAX_HOPS,
                                       "scope": scope_info,
                                       "coverage": dict(summary)}, omissions


def _lexical_candidate_sites(
        ticket: str,
        sources: Mapping[str, str],
        indexed: Sequence[object],
) -> list[dict[str, object]]:
    """Add public lexical evidence without inventing a callable edit target."""

    identifiers = tuple(item.casefold() for item in public_identifiers(ticket))
    rows: list[dict[str, object]] = []
    for path, source in sources.items():
        for line_number, line in enumerate(source.splitlines(), 1):
            if line.lstrip().startswith(("#", "//", "/*", "*", "<!--", "--")):
                continue
            if not any(identifier and identifier in line.casefold() for identifier in identifiers):
                continue
            owner = next((item for item in indexed
                          if getattr(item, "path", None) == path
                          and int(getattr(item, "line", 0) or 0) == line_number), None)
            rows.append({
                "path": path,
                "symbol": str(getattr(owner, "name", "") or ""),
                "line": line_number,
                "kind": str(getattr(owner, "kind", "lexical") or "lexical"),
                "matched_identifiers": [identifier for identifier in identifiers if identifier in line.casefold()][:4],
                "selection_sources": ["public_ticket", "declared_source", "lexical_match"],
            })
            if len(rows) >= MAX_CANDIDATES:
                return rows
    return rows


def build_localization_bundle(
        ticket: str,
        declared_paths: Sequence[str],
        *,
        root: str | Path | None = None,
        source_overrides: Mapping[str, bytes | str] | None = None,
        public_manifest: object = "public-v1",
        deadline_ms: int = 250,
) -> dict[str, object]:
    """Build bounded candidates/segments/edges from public checkout material."""
    started = time.monotonic()
    deadline = started + max(1, min(int(deadline_ms), 5000)) / 1000
    workspace = Path(root).resolve() if root is not None else None
    manifest = _public_manifest(public_manifest)
    if workspace is not None and not workspace.is_dir():
        return {"schema": SCHEMA, "status": "UNAVAILABLE", "candidates": [], "source_segments": [],
                "edges": [], "coverage": {"status": "UNAVAILABLE"}, "omissions": [{"kind": "workspace"}],
                "dependency_digests": {"manifest": _digest(manifest)}}
    paths = _safe_paths(declared_paths)
    overrides: dict[str, bytes | str] = {}
    override_omissions: list[dict[str, object]] = []
    for key, value in (source_overrides or {}).items():
        normalized = public_path(key)
        if normalized is None:
            continue
        raw_size = len(value) if isinstance(value, bytes) else len(value.encode("utf-8")) if isinstance(value, str) else -1
        if raw_size < 0:
            override_omissions.append({"kind": "source_override_bound", "path": normalized})
            continue
        overrides[normalized] = value
    sources, omissions = _source_map(workspace, paths, overrides)
    large_paths = tuple(
        str(item["path"]) for item in omissions
        if item.get("kind") == "source_deferred" and str(item.get("path") or "")
    )
    omissions = [item for item in omissions if item.get("kind") != "source_deferred"]
    omissions.extend(override_omissions)
    override_paths = frozenset(path for path in paths if path in overrides)
    safe_ticket = public_ticket(ticket)
    site_result = public_candidate_sites(safe_ticket, paths, sources, max_candidates=MAX_CANDIDATES)
    candidates = list(site_result.get("candidates", ())) if isinstance(site_result.get("candidates"), list) else []
    indexed = build_parent_index(sources)
    lexical_candidates = _lexical_candidate_sites(safe_ticket, sources, indexed)
    for candidate in candidates:
        if any(item["path"] == candidate.get("path") and item["line"] == candidate.get("line")
               for item in lexical_candidates):
            candidate["selection_sources"] = list(dict.fromkeys(
                [*candidate.get("selection_sources", ()), "lexical_match"]
            ))
    large_span_results: dict[str, dict[str, object]] = {}
    large_candidates: list[dict[str, object]] = []
    for path in large_paths:
        if time.monotonic() >= deadline:
            omissions.append({"kind": "owner_span_deadline", "path": path})
            continue
        identifiers = public_identifiers(safe_ticket)
        if identifiers:
            result = acquire_owner_spans(
                workspace, path, identifiers,
                override=overrides.get(path), max_source_chars=MAX_SOURCE_CHARS,
                max_scan_bytes=MAX_SCAN_BYTES, max_context_bytes=MAX_CONTEXT_BYTES,
            )
        else:
            # W8: identifier-free tickets (the live AHA shape) rank whole-file
            # owners by ticket-word overlap, labelled and bounded.
            result = rank_owner_spans_by_words(
                workspace, path, ticket_words(safe_ticket),
                override=overrides.get(path), max_source_chars=MAX_SOURCE_CHARS,
            )
        large_span_results[path] = result
        large_candidates.extend(
            item for item in result.get("candidates", ())
            if isinstance(item, dict)
        )
        omissions.extend(item for item in result.get("omissions", ()) if isinstance(item, dict))
    if large_candidates:
        candidates.extend(large_candidates)
    if not candidates and lexical_candidates:
        candidates = lexical_candidates
        site_result = {**site_result, "status": "FOUND" if len(candidates) == 1 else "AMBIGUOUS"}
    if large_candidates:
        site_result = {
            **site_result,
            "status": "AMBIGUOUS" if len(candidates) > 1 else "FOUND",
            "candidates": candidates,
        }
    elif large_paths:
        fully_scanned = all(
            isinstance(result.get("coverage"), Mapping)
            and not bool(result["coverage"].get("scan_bound"))
            for result in large_span_results.values()
        )
        site_result = {
            **site_result,
            "status": "NOT_FOUND" if fully_scanned else "PARTIAL",
            "candidates": [],
        }
    # The shared outline/catalog are queried after source admission and before
    # rendering, so parser/cache misses become omission evidence rather than a
    # fabricated source body.
    outlines: dict[str, Mapping[str, object]] = {}
    catalog_coverage: dict[str, object] = {"status": "UNAVAILABLE", "reason": "not_queried"}
    if large_paths:
        catalog_coverage = {"status": "PARTIAL", "reason": "large_source_span_mode",
                            "paths": list(large_paths)}
        omissions.append({"kind": "catalog_scope", "paths": list(large_paths),
                          "reason": "owner_span_only"})
        catalog_edges = []
    elif workspace is not None and sources and time.monotonic() < deadline:
        try:
            from kernel_catalog_contracts import Namespace
            from kernel_repo_catalog import build_structural_index
            namespace = Namespace("public_" + _digest({
                "workspace": str(workspace), "manifest": manifest,
            })[7:39])
            catalog = build_structural_index(
                workspace, namespace=namespace, focus_paths=tuple(sources),
                source_overrides=sources, include_call_graph=False,
            )
            catalog_coverage = {"status": "FOUND", "stats": catalog.stats.as_dict()}
            outlines = {path: catalog.file_view(path) for path in sources}
            module_paths = {
                str(item.module_id): str(item.path) for item in catalog.modules
            }
            catalog_edges = []
            for dependency in catalog.dependencies:
                target_id = str(dependency.target_id)
                if _public_graph_file(target_id) is None:
                    continue
                catalog_edges.append({
                    "from": module_paths.get(str(dependency.source_id), str(dependency.source_id)),
                    "to": target_id,
                    "relation": str(dependency.kind),
                    "distance": int(dependency.distance),
                })
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            catalog_coverage = {"status": "UNAVAILABLE", "reason": type(error).__name__}
            omissions.append({"kind": "catalog", "reason": type(error).__name__})
            catalog_edges = []
    else:
        catalog_edges = []
    segments: list[dict[str, object]] = []
    for candidate in candidates[:MAX_CANDIDATES]:
        path = str(candidate.get("path"))
        acquired_segment = candidate.get("_segment")
        if isinstance(acquired_segment, dict):
            segment = dict(acquired_segment)
            used_tokens = _token_count("".join(str(item.get("content", "")) for item in segments))
            remaining_chars = max(0, (MAX_EVIDENCE_TOKENS - used_tokens) * 4)
            if remaining_chars < len(str(segment.get("content", ""))):
                segment["content"] = str(segment.get("content", ""))[:remaining_chars]
                segment["content_digest"] = _digest(str(segment["content"]))
                segment["clipped"] = True
                segment["coverage"] = "PARTIAL"
                segment["end_line"] = int(segment.get("start_line", 1)) + max(
                    0, str(segment["content"]).count("\n")
                    - (1 if str(segment["content"]).endswith("\n") else 0)
                )
                omissions.append({"kind": "evidence_budget", "path": path})
            segments.append(segment)
            if bool(segment.get("clipped")):
                omissions.append({"kind": "segment_clipped", "path": path,
                                  "start_line": segment.get("start_line"),
                                  "end_line": segment.get("requested_end_line", segment.get("end_line")),
                                  "shown_end_line": segment.get("end_line")})
            continue
        source = sources.get(path)
        if source is None:
            omissions.append({"kind": "candidate_source_missing", "path": path})
            continue
        start_line = int(candidate.get("line", 1))
        following = [item.line for item in indexed if item.path == path and item.line > start_line]
        end_line = min(following) - 1 if following else len(source.splitlines())
        segment, omission = _segment(source, path, start_line, end_line)
        used_tokens = _token_count("".join(str(item.get("content", "")) for item in segments))
        remaining_chars = max(0, (MAX_EVIDENCE_TOKENS - used_tokens) * 4)
        if remaining_chars < len(str(segment.get("content", ""))):
            segment["content"] = str(segment.get("content", ""))[:remaining_chars]
            segment["content_digest"] = _digest(str(segment["content"]))
            omission = {"kind": "evidence_budget", "path": path}
        segments.append(segment)
        if omission is not None:
            omissions.append(omission)
        if _token_count("".join(str(item.get("content", "")) for item in segments)) >= MAX_EVIDENCE_TOKENS:
            omissions.append({"kind": "evidence_budget"})
            break
    for candidate in candidates:
        candidate.setdefault("owner", str(candidate.get("symbol") or ""))
        candidate.setdefault("module", str(candidate.get("path") or ""))
        candidate.setdefault("type", candidate.get("kind"))
        candidate.pop("_segment", None)
    symbols = [str(item.get("symbol")) for item in candidates if str(item.get("symbol") or "").strip()]
    if override_paths:
        # The call graph is built from disk; override bytes are a different
        # snapshot, so no disk-graph edge may be presented beside them.
        edges = []
        graph_paths = sorted(override_paths)
        graph_coverage = {"status": "PARTIAL", "reason": "source_override_mode", "paths": graph_paths}
        graph_omissions = [{"kind": "graph_scope", "paths": graph_paths, "reason": "snapshot_override"}]
    else:
        # W8: a large declared file read from disk is the SAME snapshot the
        # graph is built from, so its owner keeps caller edges (bounded by the
        # graph's own repo budget) instead of losing graph evidence entirely.
        edges, graph_coverage, graph_omissions = _graph_relations(
            workspace, symbols, allowed_paths=frozenset(sources) | frozenset(large_paths),
            deadline=deadline, declared_paths=paths)
    edges = [*catalog_edges, *edges][:MAX_RELATIONS]
    omissions.extend(graph_omissions)
    if len(candidates) > MAX_CANDIDATES:
        omissions.append({"kind": "candidate_bound", "omitted": len(candidates) - MAX_CANDIDATES})
        candidates = candidates[:MAX_CANDIDATES]
    dependency_digests = {path: _digest(source) for path, source in sorted(sources.items())}
    large_source_coverage: dict[str, object] = {}
    large_source_identity: dict[str, object] = {}
    for path, result in large_span_results.items():
        identity = result.get("identity") if isinstance(result.get("identity"), Mapping) else {}
        digest = identity.get("content_digest") if isinstance(identity, Mapping) else None
        dependency_digests[path] = str(digest) if digest else "UNAVAILABLE"
        large_source_coverage[path] = result.get("coverage", {})
        large_source_identity[path] = dict(identity)
    if large_source_coverage:
        dependency_digests["source_spans"] = _digest(large_source_coverage)
    dependency_digests["dependencies"] = _digest(edges)
    dependency_digests["manifest"] = _digest(manifest)
    dependency_digests["workspace"] = _digest(str(workspace) if workspace is not None else "none")
    parser_status = "PARTIAL" if large_paths else "AVAILABLE"
    try:
        from mc_repo_intel_vendor.ts_dispatch import outline_source
        parsed_outlines = {path: outline_source(source, path) for path, source in sources.items()}
        if any(bool(item.get("degraded")) for item in parsed_outlines.values()):
            parser_status = "DEGRADED"
        elif not large_paths:
            parser_status = "AVAILABLE"
        if parser_status == "DEGRADED":
            omissions.append({"kind": "outline_degraded", "reason": "parser_or_grammar_capability"})
        elif parser_status == "PARTIAL":
            omissions.append({"kind": "outline_scope", "reason": "large_source_span_mode"})
        if not outlines:
            outlines = parsed_outlines
    except (ImportError, TypeError, ValueError, RuntimeError) as error:
        outlines = {}
        parser_status = "UNAVAILABLE"
        omissions.append({"kind": "outline_parser", "reason": type(error).__name__})
    coverage = {
        "status": "PARTIAL" if omissions or site_result.get("status") == "PARTIAL" else site_result.get("status", "UNAVAILABLE"),
        "parser": parser_status, "graph": graph_coverage,
        "candidate_count": len(candidates), "segment_count": len(segments),
        "evidence_tokens": _token_count("".join(str(item.get("content", "")) for item in segments)),
        "catalog": catalog_coverage,
        "candidate_status": site_result.get("status", "UNAVAILABLE"),
        "source_spans": {
            "status": "PARTIAL" if large_source_coverage else "NOT_USED",
            "max_scan_bytes": MAX_SCAN_BYTES,
            "max_context_bytes": MAX_CONTEXT_BYTES,
            "max_read_bytes": MAX_ACQUISITION_BYTES,
            "files": large_source_coverage,
            "identity": large_source_identity,
        },
    }
    status = str(site_result.get("status", "UNAVAILABLE"))
    if parser_status != "AVAILABLE" and status not in {"UNAVAILABLE", "NOT_FOUND", "PARTIAL", "AMBIGUOUS"}:
        status = "PARTIAL"
    if status == "FOUND" and omissions:
        status = "PARTIAL"
    if status == "UNAVAILABLE" and candidates:
        status = "PARTIAL"
    return {
        "schema": SCHEMA, "status": status, "candidates": candidates[:MAX_CANDIDATES],
        "source_segments": segments[:MAX_CANDIDATES], "edges": edges[:MAX_RELATIONS],
        "coverage": coverage, "omissions": omissions,
        "dependency_digests": dependency_digests,
        "public_manifest_digest": _digest(manifest),
        "bounds": {"candidates": MAX_CANDIDATES, "graph_depth": MAX_HOPS,
                   "relations": MAX_RELATIONS, "evidence_tokens": MAX_EVIDENCE_TOKENS,
                   "catalog": catalog_coverage},
    }


localize_public = build_localization_bundle


__all__ = ["SCHEMA", "build_localization_bundle", "localize_public"]
