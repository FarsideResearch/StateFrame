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
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from kernel_workspace_consumers import WorkspaceConsumers
    from kernel_workspace_tree import Entry, File

import kernel_workspace_truth as _workspace_truth

_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".tox", ".venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".test-tmp", "node_modules", "dist", "build",
})
DEFAULT_PAGE_SIZE = 12
DEFAULT_DIRECTORY_PAGE_SIZE = 60
DEFAULT_LAYOUT_PAGE_SIZE = 80
MAX_PAGE_SIZE = 500
_SEARCH_MAX_ENTRIES = 20_000
_SEARCH_MAX_FILE_BYTES = 16 << 20
_SEARCH_MAX_BYTES = 256 << 20
_SEARCH_CACHE_BYTES = 8 << 20
_SEARCH_RESULT_BYTES = 512 << 10
_SEARCH_FILE_RESULT_BYTES = 64 << 10


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


@dataclass(frozen=True, slots=True)
class _ParsedCursor:
    parts: tuple[str, ...] | None


def _parse_cursor(cursor: object) -> _ParsedCursor:
    if isinstance(cursor, _ParsedCursor):
        return cursor
    if cursor in (None, ""):
        return _ParsedCursor(None)
    return _ParsedCursor(tuple(str(cursor).split(":")))


def _offset_parts(parts: tuple[str, ...] | None, digest: str) -> int:
    if parts is None:
        return 0
    if len(parts) != 3 or parts[0] != "nav1" or parts[1] != digest:
        raise ValueError("navigation continuation is stale; repeat the search")
    try:
        value = int(parts[2])
    except ValueError as error:
        raise ValueError("navigation continuation is invalid; repeat the search") from error
    if value < 0:
        raise ValueError("navigation continuation is invalid; repeat the search")
    return value


def _offset(cursor: object, digest: str) -> int:
    parsed = cursor if isinstance(cursor, _ParsedCursor) else _parse_cursor(cursor)
    return _offset_parts(parsed.parts, digest)


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
    return shown, _page_metadata(digest, start, len(shown), len(rows))


def _cursor_offset_hint(parts: tuple[str, ...] | None) -> int:
    """Find a candidate page window without validating a cursor before the scan."""
    if parts is None or len(parts) != 3 or parts[0] != "nav1":
        return 0
    try:
        offset = int(parts[2])
    except ValueError:
        return 0
    return offset if offset >= 0 else 0


class _SearchPageAccumulator:
    """Hash every result row while retaining only the requested page window."""

    def __init__(self, cursor: object, size: int,
                 *, initial_page: list[dict[str, object]] | None = None):
        self._parsed_cursor = _parse_cursor(cursor)
        self.offset_hint = _cursor_offset_hint(self._parsed_cursor.parts)
        self.size = size
        self.rows = initial_page if initial_page is not None else []
        self.total = 0
        self._digest = hashlib.sha256()
        self._digest.update(b"[")
        self._serialized_chars = 1
        self._encoding_error: tuple[int, int, str, str] | None = None
        if initial_page is not None:
            for row in initial_page:
                self._record(row, retain=False)

    def _record(self, row: dict[str, object], *, retain: bool) -> None:
        separator = 1 if self.total else 0
        encoded_row = json.dumps(row, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"))
        if self._encoding_error is None:
            try:
                encoded = encoded_row.encode("utf-8")
            except UnicodeEncodeError as error:
                start = self._serialized_chars + separator + error.start
                end = self._serialized_chars + separator + error.end
                self._encoding_error = (start, end, error.reason,
                                        encoded_row[error.start:error.end])
            else:
                if separator:
                    self._digest.update(b",")
                self._digest.update(encoded)
        if retain and self.offset_hint <= self.total < self.offset_hint + self.size:
            self.rows.append(row)
        self._serialized_chars += separator + len(encoded_row)
        self.total += 1

    def add(self, row: dict[str, object]) -> None:
        self._record(row, retain=True)

    def finish(self) -> tuple[list[dict[str, object]], dict[str, object]]:
        self._digest.update(b"]")
        if self._encoding_error is not None:
            start, end, reason, invalid_text = self._encoding_error
            # Preserve the original error position without retaining the serialized result set.
            error_object = "\0" * start + invalid_text
            raise UnicodeEncodeError("utf-8", error_object, start, end, reason)
        digest = self._digest.hexdigest()[:24]
        start = _offset(self._parsed_cursor, digest)
        return self.rows, _page_metadata(digest, start, len(self.rows), self.total)


def _page_metadata(digest: str, start: int, shown: int, total: int) -> dict[str, object]:
    next_offset = start + shown
    remaining = max(0, total - next_offset)
    next_cursor = _cursor(digest, next_offset) if remaining else None
    return {
        "schema": "navigation-page/1",
        "result_set_digest": digest,
        "shown": shown,
        "total": total,
        "offset": start,
        "remaining": remaining,
        "complete": next_cursor is None,
        "continuation": next_cursor,
    }


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
        coverage = ", coverage limit reached" if metadata.get("coverage_limited") else ""
        next_line += (f"; scan incomplete ({metadata.get('unreadable_files', 0)} unreadable files"
                      f"{coverage}); do not treat this page as repository-wide absence")
    lines = [header, next_line]
    if continuation:
        lines.append(continuation)
    lines.extend(rows)
    return "\n".join(lines) + advice


_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class _SearchLRU(Generic[_Key, _Value]):
    def __init__(self, limit: int):
        self.limit = limit
        self.size = 0
        self.entries: OrderedDict[_Key, tuple[_Value, int]] = OrderedDict()

    def get(self, key: _Key) -> _Value | None:
        item = self.entries.get(key)
        if item is None:
            return None
        self.entries.move_to_end(key)
        return item[0]

    def put(self, key: _Key, value: _Value, weight: int) -> None:
        if weight > _SEARCH_CACHE_BYTES:
            return
        previous = self.entries.pop(key, None)
        self.size += weight - (previous[1] if previous else 0)
        self.entries[key] = value, weight
        while self.size > _SEARCH_CACHE_BYTES or len(self.entries) > self.limit:
            _, (_, removed) = self.entries.popitem(last=False)
            self.size -= removed


@dataclass(frozen=True, slots=True)
class _SearchResult:
    digest: str
    total: int
    unreadable: int
    non_text: int
    limited: bool
    rows: tuple[bytes, ...]
    all_rows: bool

    def page(self, start: int, size: int) -> tuple[list[Mapping[str, object]], dict[str, object]]:
        selected = self.rows[start:start + size] if self.all_rows else self.rows
        shown = [json.loads(row) for row in selected]
        return shown, _page_metadata(self.digest, start, len(shown), self.total)


_SearchParameters = tuple[str, str, str | None]
_FileKey = tuple[str, str, str, bytes]
_ResultKey = tuple[Path, bytes, _SearchParameters, tuple[int, int] | None]


class _SearchCache:
    def __init__(self):
        self.files = _SearchLRU[_FileKey, tuple[str, tuple[bytes, ...]]](_SEARCH_MAX_ENTRIES)
        self.results = _SearchLRU[_ResultKey, _SearchResult](24)

    def __reduce__(self):
        # A restored owner must derive search evidence under its own live pin.
        return type(self), ()


def _row(path: str, line: int | None = None, text: str = "") -> bytes:
    row: dict[str, object] = {"path": path}
    if line is not None:
        row.update(line=line, text=text)
    return json.dumps(row, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _match_rows(path: str, text: str, matcher: re.Pattern[str]) -> Iterator[bytes]:
    for line_number, line in enumerate(text.splitlines(), 1):
        if matcher.search(line):
            yield _row(path, line_number, line)


def _matches_file_glob(relative: str, file_glob: str | None) -> bool:
    return not file_glob or (fnmatch.fnmatch(relative, file_glob)
                             or fnmatch.fnmatch(relative.rsplit("/", 1)[-1], file_glob))


def _matcher(mode: str, query: str) -> re.Pattern[str] | None:
    if mode == "path":
        return None
    try:
        return re.compile(re.escape(query) if mode == "literal" else query)
    except re.error as error:
        raise ValueError(f"invalid regex: {error}") from error


class _SearchCoverageLimit(Exception):
    pass


def _pinned_files(prefix: str, entry: Entry) -> Iterator[tuple[str, File]]:
    from kernel_workspace_tree import Directory, File, children

    if isinstance(entry, File):
        yield prefix, entry
        return
    if not isinstance(entry, Directory):
        return
    pending = [(prefix, entry)]
    remaining = _SEARCH_MAX_ENTRIES
    while pending:
        parent, directory = pending.pop()
        files: list[tuple[str, File]] = []
        directories: list[tuple[str, Directory]] = []
        limited = False
        for name, child in children(directory):
            if remaining == 0:
                limited = True
                break
            remaining -= 1
            leaf = os.fsdecode(name)
            relative = f"{parent}/{leaf}" if parent else leaf
            if isinstance(child, File):
                files.append((relative, child))
            elif isinstance(child, Directory) and leaf not in _SKIP_DIRS:
                directories.append((relative, child))
        # Legacy walk uses decoded filename order and files before subdirectories.
        yield from sorted(files, key=lambda item: item[0])
        if limited:
            raise _SearchCoverageLimit
        pending.extend(sorted(directories, key=lambda item: item[0], reverse=True))


def _requested_offset(cursor: object) -> int:
    parsed = _parse_cursor(cursor)
    parts = parsed.parts
    if parts is None:
        return 0
    try:
        return _offset(parsed, parts[1])
    except (IndexError, ValueError):
        # Validate against the completed canonical digest before rendering.
        return 0


def _scan_pinned_search(
    root: Path, scope: str, entry: Entry, parameters: _SearchParameters,
    matcher: re.Pattern[str] | None, cache: _SearchCache, start: int, size: int,
) -> _SearchResult:
    mode, query, file_glob = parameters
    digest = hashlib.sha256(b"[")
    total = unreadable = non_text = scanned_bytes = 0
    limited = False
    page_rows: list[bytes] = []
    all_rows: list[bytes] | None = []
    all_weight = 0

    def append(row: bytes) -> None:
        nonlocal total, all_rows, all_weight
        if total:
            digest.update(b",")
        digest.update(row)
        if start <= total < start + size:
            page_rows.append(row)
        total += 1
        if all_rows is not None:
            all_weight += len(row) + 64
            if all_weight <= _SEARCH_RESULT_BYTES:
                all_rows.append(row)
            else:
                all_rows = None

    try:
        for relative, file in _pinned_files(scope, entry):
            if not _matches_file_glob(relative, file_glob):
                continue
            if matcher is None:
                if query in relative:
                    append(_row(relative))
                continue
            if file.size > _SEARCH_MAX_FILE_BYTES:
                limited = True
                continue
            scanned_bytes += file.size
            if scanned_bytes > _SEARCH_MAX_BYTES:
                limited = True
                break
            key = (mode, query, relative, file.digest)
            known = cache.files.get(key)
            if known is not None:
                status, hits = known
                for hit in hits:
                    append(hit)
                non_text += status == "non_text"
                continue
            text, status = _read_text(root / relative)
            if text is None:
                unreadable += status == "unreadable"
                non_text += status == "non_text"
                if status == "non_text":
                    cache.files.put(key, (status, ()), len(relative) * 4 + len(query) * 4 + 512)
                continue
            weight = len(relative) * 4 + len(query) * 4 + 512
            file_rows: list[bytes] = []
            for hit in _match_rows(relative, text, matcher):
                append(hit)
                weight += len(hit) + 64
                if weight <= _SEARCH_FILE_RESULT_BYTES:
                    file_rows.append(hit)
            if weight <= _SEARCH_FILE_RESULT_BYTES:
                cache.files.put(key, (status, tuple(file_rows)), weight)
    except _SearchCoverageLimit:
        limited = True
    digest.update(b"]")
    return _SearchResult(
        digest.hexdigest()[:24], total, unreadable, non_text, limited,
        tuple(all_rows if all_rows is not None else page_rows), all_rows is not None,
    )


def _pinned_search(
    consumers: WorkspaceConsumers, root: Path, scope: object,
    parameters: _SearchParameters, cursor: object, size: int, journal: object,
) -> _SearchResult | None:
    from kernel_request_auth_identity import _installation_home
    from kernel_workspace_state import WorkspaceUnavailable
    from kernel_workspace_tree import Directory, File, lookup

    boundary = "navigation-search"
    # Refuse bad scopes before opening a workspace observation.
    _rooted(root, scope, journal=journal)
    matcher = _matcher(parameters[0], parameters[1])
    try:
        with consumers.state.pin(boundary) as observation:
            start = _rooted(root, scope, journal=journal)
            home = _installation_home().resolve()
            if home.is_relative_to(root) and (home.is_relative_to(start) or start.is_relative_to(home)):
                # This workspace observation intentionally excludes the installation home.
                return None
            parts = start.relative_to(root).parts
            entry = lookup(observation.tree, tuple(map(os.fsencode, parts)))
            if not isinstance(entry, (Directory, File)):
                return None

            # The host binding owns this derived data, outside navigation's checkpoint globals.
            cache = vars(consumers).get("_navigation_search_cache")
            if not isinstance(cache, _SearchCache):
                cache = vars(consumers)["_navigation_search_cache"] = _SearchCache()
            key = (start, entry.digest, parameters, None)
            offset = _requested_offset(cursor)
            page_key = (*key[:3], (offset, size))
            cached = cache.results.get(key) or cache.results.get(page_key)
            result = cached
            if result is None:
                result = _scan_pinned_search(root, "/".join(parts), entry, parameters,
                                             matcher, cache, offset, size)
            if not observation.validate_for_use(boundary):
                vars(consumers).pop("_navigation_search_cache", None)
                raise WorkspaceUnavailable("search workspace observation changed")
            if cached is None and not result.unreadable:
                weight = sum(len(row) + 64 for row in result.rows) + len(parameters[1]) * 4 + 512
                cache.results.put(key if result.all_rows else page_key, result, weight)
            return result
    except WorkspaceUnavailable:
        return None


def search_repository(root: Path, pattern: str, scope: object = None, *,
                      mode: str | None = None, file_glob: str | None = None,
                      cursor: object = None, requested_page_size: object = None,
                      policy: object | None = None,
                      journal: object = None) -> NavigationPage:
    """Search authorized UTF-8 text files or paths with revision-bound pages."""
    normalized_mode = str(mode or "regex").strip().casefold()
    if normalized_mode in {"literal", "text"}:
        normalized_mode = "literal"
    elif normalized_mode in {"path", "filename", "file"}:
        normalized_mode = "path"
    elif normalized_mode != "regex":
        raise ValueError("grep mode must be regex, literal, or path")
    page = _bounded_int(requested_page_size, DEFAULT_PAGE_SIZE)
    root = root.resolve()
    query = str(pattern)
    pinned = None
    if os.environ.get("STATEFRAME_WORKSPACE_OBJECTS") == "on":
        from kernel_workspace_consumers import consumers_for

        consumers = consumers_for(root)
        if consumers is not None:
            pinned = _pinned_search(consumers, root, scope, (normalized_mode, query, file_glob),
                                    cursor, page, journal)

    if pinned is not None:
        shown, metadata = pinned.page(_offset(cursor, pinned.digest), page)
        unreadable_files, non_text_files = pinned.unreadable, pinned.non_text
        limited = pinned.limited
    else:
        files = _walk_files(root, scope, journal=journal)
        matcher = _matcher(normalized_mode, query)
        unreadable_files = non_text_files = 0
        limited = False

        def matching_rows() -> Iterator[dict[str, object]]:
            nonlocal unreadable_files, non_text_files
            for relative, path in files:
                if not _matches_file_glob(relative, file_glob):
                    continue
                if matcher is None:
                    if query in relative:
                        yield {"path": relative}
                    continue
                text, status = _read_text(path)
                if text is None:
                    unreadable_files += status == "unreadable"
                    non_text_files += status == "non_text"
                    continue
                for line_number, line in enumerate(text.splitlines(), 1):
                    if matcher.search(line):
                        yield {"path": relative, "line": line_number, "text": line}

        small_first_page = cursor is None or (type(cursor) is str and not cursor)
        rows = matching_rows()
        if small_first_page:
            initial_page = list(islice(rows, page + 1))
            if len(initial_page) <= page:
                shown, metadata = _page_rows(initial_page, cursor=cursor, size=page)
            else:
                overflow = initial_page.pop()
                page_rows = _SearchPageAccumulator(
                    cursor, page, initial_page=initial_page,
                )
                page_rows.add(overflow)
                for row in rows:
                    page_rows.add(row)
                shown, metadata = page_rows.finish()
        else:
            page_rows = _SearchPageAccumulator(cursor, page)
            for row in rows:
                page_rows.add(row)
            shown, metadata = page_rows.finish()

    if normalized_mode == "path":
        rendered = [str(row["path"]) for row in shown]
    else:
        rendered = [f"{row['path']}:{row['line']}:{row['text']}" for row in shown]
    metadata = {**metadata, "mode": normalized_mode, "pattern": query,
                "scope": str(scope or "."), "file_glob": file_glob,
                "page_size": page}
    if normalized_mode != "path" or limited:
        metadata.update({"scan_complete": unreadable_files == 0 and not limited,
                         "unreadable_files": unreadable_files,
                         "non_text_files": non_text_files})
    if limited:
        metadata["coverage_limited"] = True
    if not metadata["total"]:
        # Search evidence is limited to this query and scope. In particular,
        # literal alternation-looking text says nothing about either symbol.
        summary = (f"No hits for {normalized_mode} search: pattern={query!r} "
                   f"scope={metadata['scope']!r}")
        if file_glob:
            summary += f" file_glob={file_glob!r}"
        summary += "."
        if normalized_mode == "literal" and "|" in query:
            summary += " The '|' character was matched literally."
        rendered.append(summary)
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
        symbol_definitions: Sequence[Mapping[str, object]] | None = None,
        workspace_root: str | Path | None = None,
        revision: str | None = None, journal: object = None) -> str:
    """Append positive, located evidence; legacy symbol_result text is not proof."""
    from kernel_owner_first import coordinates_enabled
    if not symbol:
        return page.text
    if _workspace_truth.enabled():
        pattern = page.metadata.get("pattern")
        if (page.metadata.get("mode") == "literal" and pattern == symbol
                and page.metadata.get("total", 0) > 0):
            fact = _workspace_truth.literal_matches(
                workspace_root, symbol, scope=str(page.metadata.get("scope") or "."),
                revision=revision, journal=journal,
            )
            if fact.exists is True and fact.status == "matches" and fact.matches:
                return (f"{page.text}\n[repository literal search | "
                        f"query_mode=literal source=live_workspace]\n"
                        f"{_workspace_truth.render_fact(fact)}")
        fact = _workspace_truth.symbol_defined(
            workspace_root, symbol, revision=revision, journal=journal,
        )
        if not (fact.exists is True and fact.status == "indexed" and fact.definitions):
            return page.text
        answer = _workspace_truth.render_fact(fact)
    else:
        if not coordinates_enabled():
            return page.text
        if symbol_definitions is None:
            # Older callers supply only rendered text. Verify against the live
            # index instead of interpreting prose as presence/absence evidence.
            fact = _workspace_truth.symbol_defined(
                workspace_root, symbol, revision=revision, journal=journal,
            )
            if not (fact.exists is True and fact.status == "indexed" and fact.definitions):
                return page.text
            symbol_definitions = fact.definitions
        if not symbol_definitions:
            return page.text
        answer = _workspace_truth.render_legacy_symbol_answer(
            symbol, symbol_definitions, (), 0, 0, 0,
        )
    return (f"{page.text}\n[repository symbol lookup | source=symbol_registry "
            f"query_mode=identifier status=EXISTS symbol=`{symbol}`]\n{answer}")


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
