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
import signal
import stat
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from kernel_workspace_consumers import WorkspaceConsumers
    from kernel_workspace_tree import Entry, File

import kernel_workspace_truth as _workspace_truth
from kernel_tool_paths import is_display_only_path

_LINE_CPU_LIMIT_SECONDS = 10.0
# Lines long enough to span two ticks share one CPU budget per search call.
_CALL_CPU_LIMIT_SECONDS = 30.0
_LINE_TIMER_TICK_SECONDS = 0.1
_LINE_SCOPE = threading.local()
_THREAD_TIME = getattr(time, "thread_time", None)


class _RegexLineTimeout(TimeoutError): pass

def _restore_line_scope(state: dict[str, object]) -> None:
    state["flag"][0] = 0
    if state.get("armed"):
        try: signal.setitimer(state["timer"], 0); state["armed"] = False  # drain one pending tick
        except (OSError, RuntimeError, ValueError): return
    if state.get("signum") is not None: signal.signal(state["signum"], signal.SIG_DFL)

def _arm_line_scope(state: dict[str, object]) -> None:
    signum, timer = getattr(signal, "SIGVTALRM", None), getattr(signal, "ITIMER_VIRTUAL", None)
    getitimer, setitimer = getattr(signal, "getitimer", None), getattr(signal, "setitimer", None)
    if (threading.current_thread() is not threading.main_thread() or signum is None or timer is None
            or _THREAD_TIME is None or getitimer is None or setitimer is None
            or signal.getsignal(signum) is not signal.SIG_DFL or any(getitimer(timer))): return
    previous = signal.signal(signum, _line_tick)
    if previous is not signal.SIG_DFL: signal.signal(signum, previous); return
    state.update(signum=signum, timer=timer, armed=True)
    setitimer(timer, _LINE_TIMER_TICK_SECONDS, _LINE_TIMER_TICK_SECONDS)

def _line_match_session(function):
    @wraps(function)
    def run(*args, **kwargs):
        mode = str(kwargs.get("mode", "")).strip().casefold()
        if (getattr(_LINE_SCOPE, "state", None) is not None
                or mode in ("literal", "text", "path", "filename", "file")):
            return function(*args, **kwargs)
        state = {"flag": [0], "last_tick": None}; _LINE_SCOPE.state = state
        try:
            try: _arm_line_scope(state)
            except (OSError, RuntimeError, ValueError): _restore_line_scope(state)
            return function(*args, **kwargs)
        finally:
            try: _restore_line_scope(state)
            finally: del _LINE_SCOPE.state
    return run

def _line_tick(_signum: int, _frame) -> None:
    if ((state := getattr(_LINE_SCOPE, "state", None))
            and threading.current_thread() is threading.main_thread() and state["flag"][0]):
        key, previous, now = (state["pass_token"], state["flag"][0]), state["last_tick"], _THREAD_TIME()
        if previous is None or previous[:2] != key:
            state["last_tick"], state["seen"] = (*key, now), now
            return
        state["slow"] = state.get("slow", 0.0) + now - state.get("seen", now)
        state["seen"] = now
        state["spent"] = state["slow"] > _CALL_CPU_LIMIT_SECONDS
        if state["spent"] or now - previous[2] > _LINE_CPU_LIMIT_SECONDS: raise _RegexLineTimeout

def _undecided_note(lines: int, files: int) -> list[str]:
    notes = [f"{lines} line{'s' if lines != 1 else ''} not decided."] if lines else []
    if files:
        notes.append(f"{files} file{'s' if files != 1 else ''} not searched.")
    if _line_budget_spent():
        notes.append(f"Slow-pattern CPU limit reached ({_CALL_CPU_LIMIT_SECONDS:g} s in this call); "
                     "use a simpler pattern or a narrower scope.")
    return notes

def _line_budget_spent() -> bool:
    return bool((getattr(_LINE_SCOPE, "state", None) or {}).get("spent"))

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
_SEARCH_LITERAL_PREFILTER_BYTES = 8192


@dataclass(frozen=True)
class NavigationPage:
    text: str
    metadata: Mapping[str, object]
    # Digest of the returned result rows, excluding pagination and display headers.
    evidence_digest: str = ""


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
    if scope not in (None, "", ".") and is_display_only_path(str(scope)):
        raise ValueError("navigation scope must be inside the repository root")
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


def _open_search_directory(path: Path | str, parent: int | None = None) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)


def _reopen_search_directory(root_fd: int, parts: Sequence[str]) -> int:
    fd = root_fd
    try:
        for name in parts:
            child = _open_search_directory(name, fd)
            if fd != root_fd:
                os.close(fd)
            fd = child
        return fd if fd != root_fd else os.dup(root_fd)
    except BaseException:
        if fd != root_fd:
            os.close(fd)
        raise


def _fd_paths(root: Path, start: Path, errors: list[OSError]) -> Iterator[tuple[str, int]]:
    root_fd = -1
    try:
        root_fd = _open_search_directory(root)
        parts = start.relative_to(root).parts
        if parts:
            parent = _reopen_search_directory(root_fd, parts[:-1])
            try:
                mode = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False).st_mode
            finally:
                os.close(parent)
            if stat.S_ISREG(mode):
                yield "/".join(parts), root_fd
                return
            if stat.S_ISLNK(mode):
                raise OSError("search scope became a symlink")
            if not stat.S_ISDIR(mode):
                return
        pending = [parts]
        while pending:
            prefix = pending.pop()
            directories = []
            try:
                fd = _reopen_search_directory(root_fd, prefix)
                try:
                    for name in sorted(os.listdir(fd)):
                        try:
                            mode = os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode
                        except OSError as error:
                            errors.append(error)
                            continue
                        if stat.S_ISDIR(mode) and name not in _SKIP_DIRS:
                            directories.append((*prefix, name))
                        elif stat.S_ISREG(mode):
                            yield "/".join((*prefix, name)), root_fd
                finally:
                    os.close(fd)
            except OSError as error:
                errors.append(error)
            pending.extend(reversed(directories))
    except OSError as error:
        errors.append(error)
    finally:
        if root_fd >= 0:
            os.close(root_fd)


def _decode_search_text(raw: bytes) -> str:
    return raw.decode("utf-8")


def _read_text(relative: str, root_fd: int,
               literal_bytes: bytes | None = None) -> tuple[str | None, str]:
    dirname, _, name = relative.rpartition("/")
    parent = root_fd
    try:
        if dirname:
            parts = dirname.split("/")
            parent = _reopen_search_directory(root_fd, parts)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                return None, "unreadable"
            raw = os.read(fd, 8192) if info.st_size < 8192 else b""
            if len(raw) != info.st_size:
                with os.fdopen(fd, "rb", buffering=0, closefd=False) as stream:
                    raw += stream.read()
        finally:
            os.close(fd)
        if parent != root_fd:
            # Rewalk from the root after reading, while the original parent is still open.
            current = _reopen_search_directory(root_fd, parts)
            try:
                if not os.path.samestat(os.fstat(parent), os.fstat(current)):
                    return None, "unreadable"
            finally:
                os.close(current)
    except OSError:
        return None, "unreadable"
    finally:
        if parent != root_fd:
            os.close(parent)
    if b"\x00" in raw:
        return None, "non_text"
    if (literal_bytes is not None
            and (_SEARCH_LITERAL_PREFILTER_BYTES <= info.st_size <= _SEARCH_MAX_FILE_BYTES
                 or info.st_size >= 2 * _SEARCH_MAX_FILE_BYTES)
            and raw[:64].isascii()):
        if literal_bytes not in raw:
            if raw.isascii():
                return None, "text"
            try:
                _decode_search_text(raw)
            except UnicodeDecodeError:
                return None, "non_text"
            return None, "text"
        try:
            return _decode_search_text(raw), "text"
        except UnicodeDecodeError:
            return None, "non_text"
    try:
        return raw.decode("utf-8"), "text"
    except UnicodeDecodeError:
        return None, "non_text"


def _literal_bytes(query: str) -> bytes | None:
    try:
        return query.encode("utf-8")
    except UnicodeEncodeError:
        return None


def _fingerprint(rows: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(list(rows), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _undecodable_search_path(path: str) -> str:
    if not any("\ud800" <= c <= "\udfff" for c in path): return path
    return "//undecodable-name/" + "".join(
        f"\\x{ord(c) & 255:02x}" if "\ud800" <= c <= "\udfff" else c for c in path)


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
                if isinstance(path := row.get("path"), str) and path != (display := _undecodable_search_path(path)):
                    row["path"] = display
                    encoded_row = json.dumps(row, sort_keys=True, ensure_ascii=False,
                                             separators=(",", ":"))
                    encoded = encoded_row.encode("utf-8")
                else:
                    start = self._serialized_chars + separator + error.start
                    end = self._serialized_chars + separator + error.end
                    self._encoding_error = (start, end, error.reason,
                                            encoded_row[error.start:error.end])
            if self._encoding_error is None:
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
    undecided_lines: int
    unsearched: int
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


def _escape_search_name(name: str) -> str:
    return name if name.isascii() else name.encode("utf-8", "backslashreplace").decode("utf-8")


def _row(path: str, line: int | None = None, text: str = "") -> bytes:
    row: dict[str, object] = {"path": path}
    if line is not None:
        row.update(line=line, text=text)
    try: return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except UnicodeEncodeError:
        row["path"] = _undecodable_search_path(path)
        return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _matching_lines(lines: list[str], matcher: re.Pattern[str] | None,
                    undecided: list[int] | None, literal_query: str | None = None) -> Iterator[tuple[int, str]]:
    state = getattr(_LINE_SCOPE, "state", {})
    if literal_query is None: assert matcher is not None
    if literal_query is not None or not state.get("armed"):
        for number, line in enumerate(lines, 1):
            if literal_query in line if literal_query is not None else matcher.search(line):
                yield number, line
        return

    flag, offset, previous_token = state["flag"], 0, state.get("pass_token")
    state["pass_token"] = pass_token = object()
    try:
        while offset < len(lines):
            try:
                rest = lines if offset == 0 else islice(lines, offset, None)
                for number, line in enumerate(rest, offset + 1):
                    matched = False; flag[0] = number
                    matched = matcher.search(line)
                    if matched: flag[0] = 0; yield number, line
                return
            except _RegexLineTimeout:
                cut_line = flag[0]; flag[0] = 0
                if undecided is None: raise
                if matched: yield cut_line, lines[cut_line - 1]
                else: undecided[0] += 1
                offset = cut_line
                if state.get("spent"):
                    undecided[0] += len(lines) - offset
                    return
    finally:
        flag[0] = 0; state["pass_token"] = previous_token

def _match_rows(path: str, text: str, matcher: re.Pattern[str],
                literal_query: str | None = None,
                undecided: list[int] | None = None) -> Iterator[bytes]:
    for number, line in _matching_lines(text.splitlines(), matcher, undecided, literal_query):
        yield _row(path, number, line)


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
    literal_bytes = _literal_bytes(query) if mode == "literal" else None
    digest = hashlib.sha256(b"[")
    total = unreadable = non_text = scanned_bytes = undecided_lines = unsearched = 0
    limited = False
    page_rows: list[bytes] = []
    all_rows: list[bytes] | None = []
    all_weight = 0
    if matcher is not None:
        try: root_fd = _open_search_directory(root)
        except OSError: root_fd = -1

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
            if _line_budget_spent():
                limited = True
                unsearched += 1
                continue
            if root_fd < 0:
                text, status = None, "unreadable"
            else:
                text, status = _read_text(relative, root_fd, literal_bytes)
            if text is None:
                unreadable += status == "unreadable"
                non_text += status == "non_text"
                if status == "non_text":
                    cache.files.put(key, (status, ()), len(relative) * 4 + len(query) * 4 + 512)
                continue
            weight = len(relative) * 4 + len(query) * 4 + 512
            file_rows: list[bytes] = []
            file_undecided = [0] if mode == "regex" else None
            for hit in _match_rows(relative, text, matcher,
                                   query if mode == "literal" else None, file_undecided):
                append(hit)
                weight += len(hit) + 64
                if weight <= _SEARCH_FILE_RESULT_BYTES:
                    file_rows.append(hit)
            if file_undecided and file_undecided[0]:
                undecided_lines += file_undecided[0]
                weight = _SEARCH_FILE_RESULT_BYTES + 1
            if weight <= _SEARCH_FILE_RESULT_BYTES:
                cache.files.put(key, (status, tuple(file_rows)), weight)
    except _SearchCoverageLimit:
        limited = True
    finally:
        if matcher is not None and root_fd >= 0: os.close(root_fd)
    digest.update(b"]")
    return _SearchResult(
        digest.hexdigest()[:24], total, unreadable, non_text, limited or _line_budget_spent(), undecided_lines, unsearched,
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
    if (scoped := scope not in (None, "", ".")):
        _rooted(root, scope, journal=journal)
    matcher = _matcher(parameters[0], parameters[1])
    try:
        with consumers.state.pin(boundary) as observation:
            start = _rooted(root, scope, journal=journal) if scoped else root
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
            if (cached is None and not result.unreadable and not result.undecided_lines
                    and not result.unsearched):
                weight = sum(len(row) + 64 for row in result.rows) + len(parameters[1]) * 4 + 512
                cache.results.put(key if result.all_rows else page_key, result, weight)
            return result
    except WorkspaceUnavailable:
        return None


@_line_match_session
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
    literal_bytes = _literal_bytes(query) if normalized_mode == "literal" else None
    pinned = None
    safe_fds = (os.open in os.supports_dir_fd and os.stat in os.supports_dir_fd
                and os.listdir in os.supports_fd
                and all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")))
    if safe_fds and os.environ.get("STATEFRAME_WORKSPACE_OBJECTS") == "on":
        from kernel_workspace_consumers import consumers_for

        consumers = consumers_for(root)
        if consumers is not None:
            pinned = _pinned_search(consumers, root, scope, (normalized_mode, query, file_glob),
                                    cursor, page, journal)

    undecided_lines = [pinned.undecided_lines if pinned is not None else 0]
    unsearched = pinned.unsearched if pinned is not None else 0
    if pinned is not None:
        shown, metadata = pinned.page(_offset(cursor, pinned.digest), page)
        unreadable_files, non_text_files = pinned.unreadable, pinned.non_text
        limited = pinned.limited
    else:
        start = _rooted(root, scope, journal=journal) if safe_fds else None
        matcher = _matcher(normalized_mode, query)
        unreadable_files, non_text_files = int(not safe_fds), 0
        limited = False

        def matching_rows() -> Iterator[dict[str, object]]:
            nonlocal unreadable_files, non_text_files, unsearched
            if start is None:
                return
            errors: list[OSError] = []
            with closing(_fd_paths(root, start, errors)) as files:
                for relative, root_fd in files:
                    if not _matches_file_glob(relative, file_glob):
                        continue
                    if matcher is None:
                        if query in relative:
                            yield {"path": relative}
                        continue
                    if _line_budget_spent():
                        unsearched += 1
                        continue
                    text, status = _read_text(relative, root_fd, literal_bytes)
                    if text is None:
                        unreadable_files += status == "unreadable"
                        non_text_files += status == "non_text"
                        continue
                    for line_number, line in _matching_lines(
                            text.splitlines(), matcher, undecided_lines,
                            query if normalized_mode == "literal" else None):
                        yield {"path": relative, "line": line_number, "text": line}
            unreadable_files += sum(not isinstance(error, PermissionError) for error in errors)

        small_first_page = cursor is None or (type(cursor) is str and not cursor)
        with closing(matching_rows()) as rows:
            initial_page = list(islice(rows, page + 1)) if small_first_page else None
            if initial_page is not None and len(initial_page) <= page:
                try: shown, metadata = _page_rows(initial_page, cursor=cursor, size=page)
                except UnicodeEncodeError:
                    for row in initial_page: row["path"] = _undecodable_search_path(row["path"])
                    shown, metadata = _page_rows(initial_page, cursor=cursor, size=page)
            else:
                overflow = initial_page.pop() if initial_page is not None else None
                page_rows = _SearchPageAccumulator(cursor, page, initial_page=initial_page)
                if overflow is not None:
                    page_rows.add(overflow)
                for row in rows:
                    page_rows.add(row)
                shown, metadata = page_rows.finish()

    limited = limited or _line_budget_spent()
    if normalized_mode == "path":
        rendered = [str(row["path"]) for row in shown]
    else:
        rendered = [f"{row['path']}:{row['line']}:{row['text']}" for row in shown]
    metadata = {**metadata, "mode": normalized_mode, "pattern": _escape_search_name(query),
                "scope": _escape_search_name(str(scope or ".")),
                "file_glob": None if file_glob is None else _escape_search_name(file_glob),
                "page_size": page}
    if normalized_mode != "path" or limited or unreadable_files:
        metadata.update({"scan_complete": unreadable_files == 0 and not limited and not undecided_lines[0],
                         "unreadable_files": unreadable_files,
                         "non_text_files": non_text_files})
    if limited:
        metadata["coverage_limited"] = True
    if undecided_lines[0] or unsearched:
        if unsearched:
            metadata["unsearched_files"] = unsearched
        rendered.extend(_undecided_note(undecided_lines[0], unsearched))
    elif not metadata["total"]:
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
        "sha256:" + _fingerprint(shown),
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
    rows = [{"name": _undecodable_search_path(name)} for name in children]
    shown, metadata = _page_rows(rows, cursor=cursor, size=page)
    metadata = {**metadata, "path": display, "page_size": page}
    rendered = [str(row["name"]) for row in shown]
    text = (f"{display} is a DIRECTORY. Contents:\n"
            + _page_text("directory", metadata, rendered,
                         advice=_advice(policy, kind="directory", size=page)))
    return NavigationPage(text, metadata, "sha256:" + _fingerprint(shown))


def workspace_layout(root: Path, *, policy: object | None = None) -> str:
    """Keep no-target orientation bounded while pointing at the path search page."""
    files = _walk_files(root, None)
    rows = [{"path": _undecodable_search_path(relative)} for relative, _path in files]
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
