"""Bounded, revision-bound repository discovery pages.

Discovery is observation, so every model gets the same complete scan and truthful
continuation metadata.  The model policy can add a small navigation hint, but it
cannot change the rows, totals, or access to later pages.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import kernel_workspace_truth as _workspace_truth

_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".tox", ".venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".test-tmp", "node_modules", "dist", "build",
})
DEFAULT_PAGE_SIZE = 12
DEFAULT_DIRECTORY_PAGE_SIZE = 60
DEFAULT_LAYOUT_PAGE_SIZE = 80
MAX_PAGE_SIZE = 500


@dataclass(frozen=True)
class NavigationPage:
    text: str
    metadata: Mapping[str, object]


def _bounded_int(value: object, default: int, *, maximum: int = MAX_PAGE_SIZE) -> int:
    if value in (None, "", 0):
        return default
    try:
        return max(1, min(maximum, int(value)))
    except (TypeError, ValueError):
        return default


def page_size(args: object, *, default: int = DEFAULT_PAGE_SIZE) -> int:
    value = args.get("page_size") if isinstance(args, Mapping) else None
    return _bounded_int(value, default)


def _rooted(root: Path, scope: object, *, journal: object = None) -> Path:
    root = root.resolve()
    requested = root if scope in (None, "", ".") else Path(str(scope))
    candidate = requested if requested.is_absolute() else root / requested
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("navigation scope must be inside the repository root")
    if not resolved.exists():
        if _workspace_truth.enabled():
            fact = _workspace_truth.exists(root, str(scope or root), journal=journal)
            raise FileNotFoundError(_workspace_truth.render_fact(fact))
        raise FileNotFoundError(f"scope does not exist: {scope or root}")
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix() or "."


def _walk_files(root: Path, scope: object, *,
                journal: object = None) -> tuple[tuple[str, Path], ...]:
    root = root.resolve()
    start = _rooted(root, scope, journal=journal)
    if start.is_file():
        return ((_relative(root, start), start),)
    rows: list[tuple[str, Path]] = []
    for current, directories, files in os.walk(start, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories
            if name not in _SKIP_DIRS and not (current_path / name).is_symlink()
        )
        for name in sorted(files):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                continue
            rows.append((_relative(root, path), path))
    return tuple(rows)


def _read_text(path: Path) -> tuple[str | None, str]:
    try:
        raw = path.read_bytes()
    except OSError:
        return None, "unreadable"
    if b"\x00" in raw:
        return None, "non_text"
    try:
        return raw.decode("utf-8"), "text"
    except UnicodeDecodeError:
        return None, "non_text"


def _fingerprint(rows: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(list(rows), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _cursor(digest: str, offset: int) -> str:
    return f"nav1:{digest}:{offset}"


def _offset(cursor: object, digest: str) -> int:
    if cursor in (None, ""):
        return 0
    parts = str(cursor).split(":")
    if len(parts) != 3 or parts[0] != "nav1" or parts[1] != digest:
        raise ValueError("navigation continuation is stale; repeat the search")
    try:
        value = int(parts[2])
    except ValueError as error:
        raise ValueError("navigation continuation is invalid; repeat the search") from error
    if value < 0:
        raise ValueError("navigation continuation is invalid; repeat the search")
    return value


def _advice(policy: object | None, *, kind: str, size: int) -> str:
    if policy is None:
        return ""
    try:
        enabled = bool(policy.guidance_for("navigation"))
    except (AttributeError, TypeError, ValueError):
        enabled = False
    if not enabled:
        return ""
    return (f"\nAdvice: {kind} pages default to {size} rows; request page_size explicitly "
            "when you want a different bounded page.")


def _page_rows(rows: Sequence[Mapping[str, object]], *, cursor: object,
               size: int) -> tuple[list[Mapping[str, object]], dict[str, object]]:
    digest = _fingerprint(rows)
    start = _offset(cursor, digest)
    shown = list(rows[start:start + size])
    next_offset = start + len(shown)
    remaining = max(0, len(rows) - next_offset)
    next_cursor = _cursor(digest, next_offset) if remaining else None
    metadata = {
        "schema": "navigation-page/1",
        "result_set_digest": digest,
        "shown": len(shown),
        "total": len(rows),
        "offset": start,
        "remaining": remaining,
        "complete": next_cursor is None,
        "continuation": next_cursor,
    }
    return shown, metadata


def _page_text(kind: str, metadata: Mapping[str, object], rows: Sequence[str],
               *, advice: str = "", continuation: str = "") -> str:
    scan_status = ""
    if "scan_complete" in metadata:
        scan_status = f" scan_complete={str(metadata['scan_complete']).lower()}"
    header = (
        f"[{kind} page] shown={metadata['shown']} total={metadata['total']} "
        f"offset={metadata['offset']} complete={str(metadata['complete']).lower()}"
        f"{scan_status}"
    )
    from kernel_owner_first import coordinates_enabled
    if coordinates_enabled() and "mode" in metadata:
        header += (f" query_mode={metadata['mode']}"
                   f" result={metadata.get('result', 'UNKNOWN')}")
    if metadata["continuation"]:
        remaining = int(metadata["remaining"])
        next_line = f"{remaining} more, page with cursor={metadata['continuation']}"
    else:
        next_line = "0 more; complete=true"
    if metadata.get("scan_complete") is False:
        next_line += (f"; scan incomplete ({metadata.get('unreadable_files', 0)} unreadable files); "
                      "do not treat this page as repository-wide absence")
    lines = [header, next_line]
    if continuation:
        lines.append(continuation)
    lines.extend(rows)
    return "\n".join(lines) + advice


def search_repository(root: Path, pattern: str, scope: object = None, *,
                      mode: str | None = None, file_glob: str | None = None,
                      cursor: object = None, requested_page_size: object = None,
                      policy: object | None = None,
                      journal: object = None) -> NavigationPage:
    """Search all authorized UTF-8 text files or their paths, with a truthful page."""
    normalized_mode = str(mode or "regex").strip().casefold()
    if normalized_mode in {"literal", "text"}:
        normalized_mode = "literal"
    elif normalized_mode in {"path", "filename", "file"}:
        normalized_mode = "path"
    elif normalized_mode != "regex":
        raise ValueError("grep mode must be regex, literal, or path")
    page = _bounded_int(requested_page_size, DEFAULT_PAGE_SIZE)
    root = root.resolve()
    files = _walk_files(root, scope, journal=journal)
    query = str(pattern)
    rows: list[dict[str, object]] = []
    if normalized_mode == "path":
        for relative, _path in files:
            if file_glob and not (fnmatch.fnmatch(relative, file_glob)
                                  or fnmatch.fnmatch(Path(relative).name, file_glob)):
                continue
            if query and query not in relative:
                continue
            rows.append({"path": relative})
    else:
        expression = re.escape(query) if normalized_mode == "literal" else query
        try:
            matcher = re.compile(expression)
        except re.error as error:
            raise ValueError(f"invalid regex: {error}") from error
        unreadable_files = 0
        non_text_files = 0
        for relative, path in files:
            if file_glob and not (fnmatch.fnmatch(relative, file_glob)
                                  or fnmatch.fnmatch(path.name, file_glob)):
                continue
            text, status = _read_text(path)
            if text is None:
                unreadable_files += status == "unreadable"
                non_text_files += status == "non_text"
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if matcher.search(line):
                    rows.append({"path": relative, "line": line_number, "text": line})
    shown, metadata = _page_rows(rows, cursor=cursor, size=page)
    rendered = []
    if normalized_mode == "path":
        rendered = [str(row["path"]) for row in shown]
    else:
        rendered = [f"{row['path']}:{row['line']}:{row['text']}" for row in shown]
    metadata = {**metadata, "mode": normalized_mode, "pattern": query,
                "scope": str(scope or "."), "file_glob": file_glob,
                "page_size": page}
    if normalized_mode != "path":
        metadata.update({"scan_complete": unreadable_files == 0,
                         "unreadable_files": unreadable_files,
                         "non_text_files": non_text_files})
    from kernel_owner_first import coordinates_enabled
    if coordinates_enabled():
        metadata["result"] = (
            "HITS" if metadata["total"] else
            "NO_HITS" if metadata.get("scan_complete", True) else "PARTIAL"
        )
    return NavigationPage(
        _page_text("search", metadata, rendered,
                   advice=_advice(policy, kind="search", size=page)),
        metadata,
    )


def render_grep_result(
        page: NavigationPage, *, symbol: str | None = None,
        symbol_result: str | None = None,
        workspace_root: str | Path | None = None,
        revision: str | None = None, journal: object = None) -> str:
    """Render live literal-search and symbol-index evidence with distinct provenance."""
    from kernel_owner_first import coordinates_enabled
    if not symbol:
        return page.text
    if _workspace_truth.enabled():
        pattern = page.metadata.get("pattern")
        if page.metadata.get("mode") == "literal" and pattern == symbol:
            fact = _workspace_truth.literal_matches(
                workspace_root, symbol, scope=str(page.metadata.get("scope") or "."),
                revision=revision, journal=journal,
            )
            return (f"{page.text}\n[repository literal search | "
                    f"query_mode=literal source=live_workspace]\n"
                    f"{_workspace_truth.render_fact(fact)}")
        fact = _workspace_truth.symbol_defined(
            workspace_root, symbol, revision=revision, journal=journal,
        )
        answer = _workspace_truth.render_fact(fact)
        if fact.status == "indexed":
            status = "EXISTS"
        elif fact.status == "no_symbol_index_entry":
            status = "NO_SYMBOL_INDEX_ENTRY"
        else:
            status = "UNKNOWN"
    else:
        if not coordinates_enabled() or not symbol_result:
            return page.text
        answer = str(symbol_result)
        if answer.startswith(f"`{symbol}` EXISTS ("):
            status = "EXISTS"
        elif answer.startswith(f"`{symbol}`{_workspace_truth.LEGACY_SYMBOL_ABSENCE_PREFIX}"):
            status = "DOES_NOT_EXIST"
        elif answer.startswith(f"No symbol index entry for `{symbol}`"):
            status = "NO_SYMBOL_INDEX_ENTRY"
        else:
            status = "UNKNOWN"
    return (f"{page.text}\n[repository symbol lookup | source=symbol_registry "
            f"query_mode=identifier status={status} symbol=`{symbol}`]\n{answer}")


def directory_page(directory: Path, display: str, *, cursor: object = None,
                   requested_page_size: object = None,
                   policy: object | None = None) -> NavigationPage:
    """Render every child of a directory over stable, revision-bound pages."""
    page = _bounded_int(requested_page_size, DEFAULT_DIRECTORY_PAGE_SIZE)
    children = sorted(
        {child.name + ("/" if child.is_dir() else "") for child in directory.iterdir()}
    )
    rows = [{"name": name} for name in children]
    shown, metadata = _page_rows(rows, cursor=cursor, size=page)
    metadata = {**metadata, "path": display, "page_size": page}
    rendered = [str(row["name"]) for row in shown]
    text = (f"{display} is a DIRECTORY. Contents:\n"
            + _page_text("directory", metadata, rendered,
                         advice=_advice(policy, kind="directory", size=page)))
    return NavigationPage(text, metadata)


def workspace_layout(root: Path, *, policy: object | None = None) -> str:
    """Keep no-target orientation bounded while pointing at the path search page."""
    files = _walk_files(root, None)
    rows = [{"path": relative} for relative, _path in files]
    shown, metadata = _page_rows(rows, cursor=None, size=DEFAULT_LAYOUT_PAGE_SIZE)
    rendered = [f"- {row['path']}" for row in shown]
    body = _page_text("repository layout", metadata, rendered,
                      advice=_advice(policy, kind="layout", size=DEFAULT_LAYOUT_PAGE_SIZE))
    if metadata["continuation"]:
        body += ("\nDiscover omitted paths with grep using mode=path and the returned "
                 f"cursor={metadata['continuation']}")
    return "[Repository layout; no working source selected]\n" + body


__all__ = [
    "DEFAULT_DIRECTORY_PAGE_SIZE", "DEFAULT_LAYOUT_PAGE_SIZE", "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE", "NavigationPage", "directory_page", "page_size",
    "render_grep_result", "search_repository", "workspace_layout",
]
