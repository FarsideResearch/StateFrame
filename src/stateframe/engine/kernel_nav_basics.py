"""Bounded workspace navigation facts for the public ReAct surface.

The navigation tools are observation-only.  They resolve every requested scope
against the workspace root, never follow symlinks, prune generated/vendor trees,
and cap both recursive depth and rendered rows before returning model-visible
content.
"""
from __future__ import annotations

import difflib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import kernel_workspace_truth as _workspace_truth


SCHEMA = "navbasics-workspace-list/1"
ORIENTATION_SCHEMA = "navbasics-intake-orientation/1"
ORIENTATION_MARKER = "[NAVBASICS intake orientation]"
DEFAULT_DEPTH = 2
MAX_DEPTH = 4
DEFAULT_LIMIT = 80
MAX_LIMIT = 240
MAX_SUGGESTIONS = 5
MAX_SUGGESTION_CANDIDATES = 5000
MAX_ORIENTATION_ENTRIES = 40
MAX_ORIENTATION_SELECTORS = 8
MAX_ORIENTATION_CHARS = 6000
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".tox", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "dist", "build",
})


def _bounded_int(value: object, default: int, maximum: int, *, name: str) -> int:
    if value in (None, ""):
        return default
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if integer < 1:
        raise ValueError(f"{name} must be at least 1")
    return min(integer, maximum)


def _relative(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return relative or "."


def resolve_scope(root: Path, scope: object = None, *, journal: object = None) -> Path:
    """Resolve a repository-relative scope without permitting an escape."""
    root = Path(root).resolve()
    requested = root if scope in (None, "", ".") else Path(str(scope))
    candidate = requested if requested.is_absolute() else root / requested
    if candidate.is_symlink():
        raise ValueError("workspace navigation does not follow symlink scopes")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("workspace navigation scope must be inside the repository root")
    if not resolved.exists():
        if _workspace_truth.enabled():
            fact = _workspace_truth.exists(root, str(scope or "."), journal=journal)
            raise FileNotFoundError(_workspace_truth.render_fact(fact))
        raise FileNotFoundError(f"workspace navigation scope does not exist: {scope or '.'}")
    if resolved.is_symlink():
        raise ValueError("workspace navigation does not follow symlink scopes")
    return resolved


def _entry(path: Path, root: Path, depth: int) -> dict[str, object]:
    relative = _relative(root, path)
    if path.is_dir():
        size: int | None = None
        kind = "directory"
    else:
        kind = "file"
        try:
            size = path.stat().st_size
        except OSError:
            size = None
    return {"path": relative, "type": kind, "size": size, "depth": depth}


def list_workspace_files(
        root: Path,
        scope: object = None,
        *,
        depth: object = DEFAULT_DEPTH,
        limit: object = DEFAULT_LIMIT,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Return a deterministic, capped tree of files and directories.

    ``depth=1`` lists only direct children; deeper values include descendants.
    The cap applies to returned entries, and a ``truncated`` flag makes omission
    explicit instead of allowing a partial tree to look complete.
    """
    max_depth = _bounded_int(depth, DEFAULT_DEPTH, MAX_DEPTH, name="depth")
    max_entries = _bounded_int(limit, DEFAULT_LIMIT, MAX_LIMIT, name="limit")
    root = Path(root).resolve()
    start = resolve_scope(root, scope)
    if start.is_file():
        rows = (_entry(start, root, 0),)
        metadata = {
            "schema": SCHEMA, "scope": _relative(root, start), "depth": max_depth,
            "limit": max_entries, "shown": min(len(rows), max_entries),
            "truncated": len(rows) > max_entries,
        }
        return rows[:max_entries], metadata

    rows: list[dict[str, object]] = []
    pending: list[tuple[Path, int]] = [(start, 0)]
    while pending:
        current, current_depth = pending.pop()
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name.casefold())
        except OSError:
            continue
        next_directories: list[tuple[Path, int]] = []
        for child in children:
            if child.name in SKIP_DIRS or child.is_symlink():
                continue
            child_depth = current_depth + 1
            if child_depth > max_depth:
                continue
            if not (child.is_dir() or child.is_file()):
                continue
            rows.append(_entry(child, root, child_depth))
            if child.is_dir() and child_depth < max_depth:
                next_directories.append((child, child_depth))
            if len(rows) >= max_entries:
                metadata = {
                    "schema": SCHEMA, "scope": _relative(root, start),
                    "depth": max_depth, "limit": max_entries,
                    "shown": len(rows), "truncated": True,
                }
                return tuple(rows), metadata
        pending[0:0] = reversed(next_directories)
    metadata = {
        "schema": SCHEMA, "scope": _relative(root, start),
        "depth": max_depth, "limit": max_entries,
        "shown": len(rows), "truncated": False,
    }
    return tuple(rows), metadata


def render_file_tree(
        root: Path,
        scope: object = None,
        *,
        depth: object = DEFAULT_DEPTH,
        limit: object = DEFAULT_LIMIT,
) -> str:
    rows, metadata = list_workspace_files(root, scope, depth=depth, limit=limit)
    return json.dumps({**metadata, "entries": list(rows)}, sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))


def _candidate_paths(root: Path) -> Iterable[str]:
    yielded = 0
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories
            if name not in SKIP_DIRS and not (current_path / name).is_symlink()
        )
        for name in sorted(files):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                continue
            yield _relative(root, path)
            yielded += 1
            if yielded >= MAX_SUGGESTION_CANDIDATES:
                return


def closest_paths(root: Path, requested: object, *, limit: int = MAX_SUGGESTIONS) -> tuple[str, ...]:
    """Find same-basename or close fuzzy file paths, bounded to workspace data."""
    root = Path(root).resolve()
    raw = str(requested or "").strip().strip("`'\"")
    base = raw.split("::", 1)[0].strip("/")
    query_name = Path(base).name.casefold()
    query_path = base.casefold()
    if not query_name:
        return ()
    scored: list[tuple[float, str]] = []
    for candidate in _candidate_paths(root):
        name = Path(candidate).name.casefold()
        path_text = candidate.casefold()
        if name == query_name:
            score = 2.0
        else:
            score = max(
                difflib.SequenceMatcher(None, query_name, name).ratio(),
                difflib.SequenceMatcher(None, query_path, path_text).ratio(),
            )
        if score >= 0.48:
            scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(candidate for _score, candidate in scored[:max(1, min(limit, MAX_SUGGESTIONS))])


def missing_path_message(root: Path, requested: object, *, label: str = "read") -> str:
    suggestions = closest_paths(root, requested)
    rendered = ", ".join(suggestions) if suggestions else "none found; use list_files to inspect the workspace"
    if label == "selector":
        return f"[selector] MISSING: {requested}. Closest real paths: {rendered}."
    return f"MISSING: no such file: {requested}. Closest real paths: {rendered}."


def render_intake_orientation(root: Path, selectors: Sequence[object] = (), *,
                              repo_map: str = "") -> str:
    """Render one bounded host-fact orientation block for the first request."""
    rows, metadata = list_workspace_files(
        root, depth=DEFAULT_DEPTH, limit=MAX_ORIENTATION_ENTRIES)
    lines = [
        ORIENTATION_MARKER,
        f"schema={ORIENTATION_SCHEMA}",
        "Repository root: . (the workspace root; use repository-relative paths without a leading slash).",
        f"Top-level tree: depth={metadata['depth']} limit={metadata['limit']} "
        f"shown={metadata['shown']} truncated={str(metadata['truncated']).lower()}",
    ]
    for row in rows:
        suffix = "/" if row["type"] == "directory" else ""
        size = "unknown" if row["size"] is None else str(row["size"])
        lines.append(f"- {row['path']}{suffix} type={row['type']} size={size}")
    if repo_map:
        lines.extend(("Repository signatures (bounded, task-ranked):", str(repo_map)))
    clean_selectors = tuple(
        str(selector).strip().strip("`'\"")
        for selector in selectors
        if str(selector).strip()
    )[:MAX_ORIENTATION_SELECTORS]
    lines.append("Public checks live at: " + (
        ", ".join(clean_selectors) if clean_selectors else "none declared by the host for this task."
    ))
    if clean_selectors:
        lines.append("Run from the repository root exactly: " + " ; ".join(
            f"pytest {selector} (agent tool selector: run_tests(target={selector!r}))"
            for selector in clean_selectors
        ))
    else:
        lines.append("Run from the repository root exactly: no public selector was declared.")
    rendered = "\n".join(lines)
    return rendered[:MAX_ORIENTATION_CHARS]


def append_intake_orientation(
        messages: Sequence[Mapping[str, object]], orientation: str) -> list[dict[str, object]]:
    """Attach orientation once to the first user message, preserving history."""
    copied = [dict(message) for message in messages]
    if not orientation or any(ORIENTATION_MARKER in str(item.get("content", ""))
                              for item in copied if isinstance(item, Mapping)):
        return copied
    target = next((index for index, item in enumerate(copied)
                   if item.get("role") == "user"), None)
    if target is None:
        return copied
    content = str(copied[target].get("content", ""))
    copied[target]["content"] = f"{content}\n\n{orientation}" if content else orientation
    return copied


__all__ = [
    "DEFAULT_DEPTH", "DEFAULT_LIMIT", "MAX_DEPTH", "MAX_LIMIT", "ORIENTATION_MARKER",
    "SCHEMA", "append_intake_orientation", "closest_paths", "list_workspace_files",
    "missing_path_message", "render_file_tree", "render_intake_orientation", "resolve_scope",
]
