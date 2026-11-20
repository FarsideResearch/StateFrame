"""Repo map generator — compact AST-based signature map for a repository.

Generates per-file function/class/export signatures suitable for injection into
the model's system prompt prefix (~50-80 tokens per file). Injecting this as a stable
prefix enables prompt caching on all subsequent orchestrator calls for the same repo.

When STATEFRAME_HOME explicitly enables disk caching, maps are
keyed by repository root, a digest of map-relevant source files, repo_url, and
commit_sha. No shared cache location is selected by default.

Supports tree-sitter-language-pack (preferred) and tree-sitter-languages (legacy)
with a plain text fallback when neither is available.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

from kernel_obligation_assess import UnavailableEvidence

from .ts_dispatch import outline_source
from .ts_extract import is_tree_sitter_functional

logger = logging.getLogger(__name__)

# ── tree-sitter availability ──────────────────────────────────────────────────
_TREE_SITTER_AVAILABLE = False
_USES_LANGUAGE_PACK = False
_get_parser = None

try:
    from tree_sitter_language_pack import get_parser as _tslp_get_parser  # type: ignore[import]
    _get_parser = _tslp_get_parser
    _USES_LANGUAGE_PACK = True
    _TREE_SITTER_AVAILABLE = True
    logger.debug("repo_map: using tree-sitter-language-pack")
except ImportError:
    try:
        from tree_sitter_languages import get_parser as _tsl_get_parser  # type: ignore[import]
        _get_parser = _tsl_get_parser
        _TREE_SITTER_AVAILABLE = True
        logger.debug("repo_map: using tree-sitter-languages (legacy)")
    except ImportError:
        logger.warning(
            "repo_map: no tree-sitter parser available — using regex fallback. "
            "Install tree-sitter-language-pack for better results."
        )

# ── Configuration ─────────────────────────────────────────────────────────────
_MAX_FILES = 250            # cap to prevent repo map blowing up context
_MAX_SIGS_PER_FILE = 12     # max signatures extracted per file
_MAX_SIG_CHARS = 120        # max chars per individual signature line
_MAX_FILE_ENTRY_CHARS = 600 # max chars for a complete file entry in the map
_MAX_FILE_READ_BYTES = 200_000  # skip files larger than this

# Directories to skip when walking the repo
_SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".tox", "migrations", ".next", ".nuxt",
    "coverage", ".pytest_cache", ".mypy_cache", ".eggs",
    "grader", "graders", "gold", "private", "hidden", "hidden_tests", "backups",
})

# Extension → tree-sitter language name
_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".vue": "vue",
    ".kt": "kotlin",
    ".swift": "swift",
    ".cs": "c_sharp",
    ".cpp": "cpp",
    ".c": "c",
}


# ── Signature extraction ──────────────────────────────────────────────────────

def _extract_python_sigs(source_bytes: bytes) -> list[str]:
    """Extract function/class signatures from Python using tree-sitter."""
    sigs: list[str] = []
    parser = _get_parser("python")  # type: ignore[misc]
    tree = parser.parse(source_bytes)

    def walk(node, depth: int = 0) -> None:
        if len(sigs) >= _MAX_SIGS_PER_FILE:
            return
        if node.type in ("function_definition", "async_function_definition"):
            name_node = next((c for c in node.children if c.type == "identifier"), None)
            params_node = next((c for c in node.children if c.type == "parameters"), None)
            if name_node:
                name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                params = (
                    source_bytes[params_node.start_byte:params_node.end_byte].decode("utf-8", errors="replace")
                    if params_node else "()"
                )
                prefix = "  " * depth
                # Truncate long param lists
                if len(params) > 60:
                    params = params[:57] + "...)"
                async_prefix = "async " if node.type == "async_function_definition" or any(
                    child.type == "async" for child in node.children
                ) else ""
                sigs.append(f"{prefix}{async_prefix}def {name}{params}"[:_MAX_SIG_CHARS])
        elif node.type == "class_definition":
            name_node = next((c for c in node.children if c.type == "identifier"), None)
            if name_node:
                name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                prefix = "  " * depth
                sigs.append(f"{prefix}class {name}:"[:_MAX_SIG_CHARS])
                # Walk class body one level deep
                for child in node.children:
                    if child.type == "block":
                        for grandchild in child.children:
                            walk(grandchild, depth + 1)
                return
        for child in node.children:
            walk(child, depth)

    walk(tree.root_node)
    return sigs


def _extract_ts_js_sigs(source_bytes: bytes, lang: str) -> list[str]:
    """Extract export/function/class/interface signatures from TS/JS."""
    sigs: list[str] = []
    _TS_SIG_TYPES = {
        "function_declaration", "class_declaration",
        "interface_declaration", "type_alias_declaration",
        "export_statement", "method_definition",
        "lexical_declaration",  # const X = () => ...
    }
    parser = _get_parser(lang)  # type: ignore[misc]
    tree = parser.parse(source_bytes)

    def walk(node) -> None:
        if len(sigs) >= _MAX_SIGS_PER_FILE:
            return
        if node.type in _TS_SIG_TYPES:
            text = source_bytes[node.start_byte:min(node.start_byte + 150, node.end_byte)]
            first_line = text.decode("utf-8", errors="replace").split("\n")[0].strip()
            # Skip very short lines (likely just `{` or `}`)
            if len(first_line) > 8:
                sigs.append(first_line[:_MAX_SIG_CHARS])
                return  # don't recurse into export/class bodies
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return sigs


def _fallback_extract_sigs(source: str, ext: str) -> list[str]:
    """Line-scan fallback when tree-sitter is unavailable."""
    sigs: list[str] = []
    PY_STARTS = ("def ", "async def ", "class ")
    TS_STARTS = ("export ", "function ", "class ", "interface ", "type ", "const ", "async function ")

    for line in source.splitlines():
        stripped = line.strip()
        if ext == ".py":
            if any(stripped.startswith(k) for k in PY_STARTS):
                sigs.append(stripped[:_MAX_SIG_CHARS])
        elif ext in (".ts", ".tsx", ".js", ".jsx") and any(
            stripped.startswith(k) for k in TS_STARTS
        ):
            sigs.append(stripped[:_MAX_SIG_CHARS])
        if len(sigs) >= _MAX_SIGS_PER_FILE:
            break
    return sigs


_SCAN_EXTS = (".py", ".ts", ".tsx", ".js", ".jsx")
_TEXT_SCAN = "text-scan fallback"
_COMMIT_SHA_LOOKUP_FAILURE_REASON = "commit_sha_lookup_failed"


def _parser_available() -> bool:
    return _TREE_SITTER_AVAILABLE and (not _USES_LANGUAGE_PACK or is_tree_sitter_functional())


def _outline_sigs(source: str, rel_path: str) -> tuple[list[str], str, str | None]:
    """Signatures for languages outside py/ts/js from the source outline, the mode it used, and why it degraded.

    ``ts_dispatch.outline_source`` parses with tree-sitter when a grammar is installed and falls back to its
    lexical scan otherwise; the reason is None unless a parser was available but failed."""
    outline = outline_source(
        source, rel_path, allow_tree_sitter=not _USES_LANGUAGE_PACK or _parser_available(),
    )
    mode = "tree-sitter" if outline["parser"] == "tree-sitter" else _TEXT_SCAN
    reason = outline.get("degraded_reason")
    failure = (str(reason) if _parser_available() and mode != "tree-sitter"
               and reason not in (None, "unsupported-language") else None)
    records = [(record, True) for record in outline["types"]]
    records.extend((record, False) for record in outline["functions"])
    records.sort(key=lambda item: item[0]["line"])
    sigs = []
    for record, is_type in records:
        signature = str(record.get("signature", "")).split("{", 1)[0].strip()
        if outline["language"] == "go" and is_type and signature and not signature.startswith("type "):
            signature = f"type {signature}"
        if signature:
            sigs.append(signature[:_MAX_SIG_CHARS])
    return sigs[:_MAX_SIGS_PER_FILE], mode, failure


def _file_entry_with_mode(
    file_path: Path, repo_root: Path, *, source_bytes: bytes | None = None,
    observed: bool = False,
) -> tuple[str | None, str, str | None]:
    """Return a bounded entry, the mode that produced it, and the parser-failure reason (None if none).

    Python/TS/JS/Vue files keep their original rules exactly (files under 10 bytes and files with no
    signatures are left out), so maps of repos made only of those languages are unchanged. Every other
    supported language goes through the source outline and is always listed, "(no signatures)" if empty."""
    ext = file_path.suffix.lower()
    if ext not in _LANG_MAP:
        return None, "", None
    scanned = ext in _SCAN_EXTS or ext == ".vue"

    try:
        size = len(source_bytes) if observed and source_bytes is not None else file_path.stat().st_size
        if size > _MAX_FILE_READ_BYTES or (scanned and size < 10):
            return None, "", None
        if source_bytes is None:
            source_bytes = file_path.read_bytes()
    except OSError:
        return None, "", None

    source = source_bytes.decode("utf-8", errors="replace")
    rel_path = str(file_path.relative_to(repo_root))
    sigs: list[str] = []
    mode = _TEXT_SCAN
    failure: str | None = None

    if not scanned:
        try:
            sigs, mode, failure = _outline_sigs(source, rel_path)
        except Exception as exc:  # noqa: BLE001 - the map is fail-open: list the file, record why
            sigs, mode = [], _TEXT_SCAN
            failure = type(exc).__name__ if _parser_available() else None
    elif _TREE_SITTER_AVAILABLE and _get_parser is not None and ext != ".vue" and _parser_available():
        try:
            if ext == ".py":
                sigs = _extract_python_sigs(source_bytes)
            else:
                sigs = _extract_ts_js_sigs(source_bytes, "typescript" if ext in (".ts", ".tsx") else "javascript")
            mode = "tree-sitter"
        except Exception as exc:  # noqa: BLE001 - degrade this file to the text scan, and say so
            failure = type(exc).__name__
        if not sigs:
            # A parser that fails, or yields no nodes (e.g. a grammar ABI mismatch), degrades this file to
            # the text scan instead of losing it. A file only counts as text scan if the scan found something.
            sigs = _fallback_extract_sigs(source, ext)
            if sigs:
                mode = _TEXT_SCAN
    else:
        # Vue SFCs (script block scanned as JS) and every scanned file when no parser is installed.
        sigs = _fallback_extract_sigs(source, ".js" if ext == ".vue" else ext)

    if not sigs and scanned:
        return None, "", failure

    sig_block = "\n  ".join(sigs[:_MAX_SIGS_PER_FILE]) if sigs else "(no signatures)"
    entry = f"{rel_path}:\n  {sig_block}"
    return entry[:_MAX_FILE_ENTRY_CHARS], mode, failure


def _file_entry(file_path: Path, repo_root: Path) -> str | None:
    """Generate a compact signature entry for one file. Returns None if skipped."""
    return _file_entry_with_mode(file_path, repo_root)[0]


def _map_children(directory: Path, observation: IntakeMapObservation) -> list[os.DirEntry[str]]:
    observation.directory(directory)
    with os.scandir(directory) as entries:
        return sorted(entries, key=lambda entry: os.path.normcase(entry.name), reverse=True)


def _map_candidates(root: Path, observation: IntakeMapObservation) -> Iterator[Path]:
    """Yield files in path order, pruning excluded trees before listing them."""
    pending = _map_children(root, observation)
    while pending:
        entry = pending.pop()
        if entry.name in _SKIP_DIRS or entry.name.endswith(".egg-info"):
            continue
        if entry.is_dir(follow_symlinks=False):
            pending.extend(_map_children(Path(entry.path), observation))
            continue
        path = Path(entry.path)
        if path.suffix.lower() in _LANG_MAP and (
            path.is_file() if entry.is_symlink() else entry.is_file()
        ):
            yield path


# ── Map generation ────────────────────────────────────────────────────────────

class IntakeObservationUnavailable(Exception):
    """The eager map must handle this source or parser environment."""


class IntakeMapObservation:
    """Hash exactly the ordered sources visited by the capped text-scan map."""

    max_candidates = 4096
    max_bytes = 64 * 1024 * 1024

    def __init__(self, root: Path):
        self.root = root
        self.count = 0
        self.bytes_read = 0
        self.digest = hashlib.sha256()
        self._stamps: dict[Path, tuple[int, ...] | None] = {}

    @staticmethod
    def _stamp(info: os.stat_result) -> tuple[int, ...]:
        return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns)

    @classmethod
    def _path_stamp(cls, path: Path) -> tuple[int, ...] | None:
        try:
            return cls._stamp(path.lstat())
        except FileNotFoundError:
            return None

    def _remember(self, path: Path, stamp: tuple[int, ...] | None) -> None:
        if len(self._stamps) >= 2 * self.max_candidates:
            raise IntakeObservationUnavailable
        if self._stamps.setdefault(path, stamp) != stamp:
            raise IntakeObservationUnavailable

    def directory(self, path: Path) -> None:
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise IntakeObservationUnavailable
        self._remember(path, self._stamp(info))

    def missing(self, path: Path) -> None:
        self._remember(path, None)

    def current(self) -> bool:
        return all(self._path_stamp(path) == stamp for path, stamp in self._stamps.items())

    @staticmethod
    def policy() -> tuple | None:
        # Native grammars can fail per language, and the source outline can
        # notify a capability listener. Keep both on the ordinary parser path.
        if _parser_available():
            return None
        return (
            "text-scan/1", _TREE_SITTER_AVAILABLE, _USES_LANGUAGE_PACK,
            os.name, sorted(_SKIP_DIRS), sorted(_LANG_MAP), _SCAN_EXTS,
            _MAX_FILES, _MAX_SIGS_PER_FILE, _MAX_SIG_CHARS,
            _MAX_FILE_ENTRY_CHARS, _MAX_FILE_READ_BYTES,
        )

    def source(self, path: Path) -> bytes | None:
        ext = path.suffix.lower()
        if ext not in (*_SCAN_EXTS, ".vue") or self.count >= self.max_candidates:
            raise IntakeObservationUnavailable
        # Refuse symlinks and nonregular files before using their contents as
        # evidence. A length or stat change during the read is also ambiguous.
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise IntakeObservationUnavailable
            size = before.st_size
            data = None
            if 10 <= size <= _MAX_FILE_READ_BYTES:
                if self.bytes_read + size > self.max_bytes:
                    raise IntakeObservationUnavailable
                data = stream.read(size + 1)
                if len(data) != size:
                    raise IntakeObservationUnavailable
                self.bytes_read += size
            stamp = self._stamp(before)
            if stamp != self._path_stamp(path):
                raise IntakeObservationUnavailable
            self._remember(path, stamp)
        self.count += 1
        relative = str(path.relative_to(self.root)).encode("utf-8", errors="surrogatepass")
        self.digest.update(relative + b"\0")
        self.digest.update(b"skip" if data is None else b"read" + hashlib.sha256(data).digest())
        return data


def _generate_repo_map_with_stats(
    workspace_dir: str, *, cache: dict | None = None, fresh: dict | None = None,
    observation: IntakeMapObservation | None = None,
) -> tuple[str, dict]:
    """Walk the workspace; return the map text and the modes actually used.

    The header names the mode only as it really happened: "tree-sitter" when every listed file was parsed,
    "text-scan fallback" when none was, otherwise both with the fallback count, plus any parser failures."""
    stats: dict = {"files": 0, "tree_sitter": 0, "text_scan": 0, "parse_failures": 0, "failure_reasons": {}}
    root = Path(workspace_dir)
    if not root.exists():
        return "", stats

    file_entries: list[str] = []
    file_count = 0
    skipped = 0
    reasons: dict[str, int] = {}

    candidates = _map_candidates(root, observation) if observation is not None else sorted(root.rglob("*"))
    for file_path in candidates:
        if not file_path.is_file():
            continue

        # Skip excluded directories
        try:
            parts = file_path.relative_to(root).parts
        except ValueError:
            continue
        if any(p in _SKIP_DIRS or p.endswith(".egg-info") for p in parts):
            continue

        if cache is not None and fresh is not None:
            if file_path.is_symlink() or file_path.suffix.lower() not in _LANG_MAP:
                continue
            entry, mode, failure = _cached_map_entry(file_path, root, cache, fresh)
        elif observation is None:
            entry, mode, failure = _file_entry_with_mode(file_path, root)
        else:
            data = observation.source(file_path)
            if data is None:
                continue
            entry, mode, failure = _file_entry_with_mode(file_path, root, source_bytes=data, observed=True)
        if failure:
            reasons[failure] = reasons.get(failure, 0) + 1
        if entry:
            file_entries.append(entry)
            file_count += 1
            stats["tree_sitter" if mode == "tree-sitter" else "text_scan"] += 1
            if file_count >= _MAX_FILES:
                file_entries.append(f"[... map capped at {_MAX_FILES} files — additional files not shown]")
                break
        else:
            skipped += 1

    stats.update(files=file_count, parse_failures=sum(reasons.values()), failure_reasons=dict(sorted(reasons.items())))
    if stats["parse_failures"]:
        logger.warning("repo_map: %d parser failures during map generation (%s)", stats["parse_failures"],
                       ", ".join(f"{name}={count}" for name, count in stats["failure_reasons"].items()))
    if not file_entries:
        return "", stats

    fallback = stats["text_scan"]
    if fallback == file_count:
        mode = _TEXT_SCAN
    elif not fallback:
        mode = "tree-sitter"
    else:
        mode = f"tree-sitter; {fallback} {_TEXT_SCAN}"
    if stats["parse_failures"]:
        mode += f", {stats['parse_failures']} parse failures"
    header = f"## Repository Map ({file_count} files, {mode})\n"
    return header + "\n".join(file_entries), stats


def _cached_map_entry(file_path: Path, root: Path, cache: dict, fresh: dict) -> tuple:
    from .call_graph import _hash_bytes

    try:
        with file_path.open("rb") as source:
            data = source.read(_MAX_FILE_READ_BYTES + 1)
    except OSError:
        return None, "", None
    if len(data) > _MAX_FILE_READ_BYTES:
        return None, "", None
    relative = file_path.relative_to(root).as_posix()
    digest = _hash_bytes(data)
    row = cache.get(relative)
    if (isinstance(row, list) and len(row) == 4 and row[0] == digest
            and all(item is None or isinstance(item, str) for item in row[1:])
            and isinstance(row[2], str)):
        entry = tuple(row[1:])
    else:
        entry = _file_entry_with_mode(file_path, root, source_bytes=data)
    fresh[relative] = [digest, *entry]
    return entry


def background_map(root: Path) -> tuple[str, dict] | None:
    """Get the current map or schedule blob-keyed extraction off-thread."""
    from . import background
    from .call_graph import _atomic_write_text

    def work(publish, current):
        cache_path = background.cache_file(root, "map.json")
        parser = "tree-sitter" if _parser_available() else _TEXT_SCAN
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, RecursionError, UnicodeError, ValueError):
            cached = {}
        cache = {}
        if isinstance(cached, dict) and cached.get("parser") == parser:
            rows = cached.get("files")
            if isinstance(rows, dict):
                cache = rows
        fresh: dict = {}
        result = _generate_repo_map_with_stats(str(root), cache=cache, fresh=fresh)
        if current():
            try:
                _atomic_write_text(cache_path, json.dumps({"parser": parser, "files": fresh}))
            except OSError as error:
                logger.debug("Background repository map cache write failed: %s", type(error).__name__)
            publish(result)

    return background.request(root.absolute(), "map", work)


def _generate_repo_map(workspace_dir: str) -> str:
    """Walk the workspace and generate the full repo map text."""
    return _generate_repo_map_with_stats(workspace_dir)[0]


# ── Cache ─────────────────────────────────────────────────────────────────────

def _get_cache_dir() -> Path | None:
    cache_home = os.getenv("STATEFRAME_HOME")
    if not cache_home:
        return None
    from kernel_safe_paths import safe_expanduser

    cache_dir = safe_expanduser(Path(cache_home)) / "repo_map_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return cache_dir


# Bump when map content rules or cache identity change, so unsafe old entries are never reused.
_CACHE_FORMAT = "repo-map/3"


def _cache_snapshot_digest(workspace_dir: str) -> str | UnavailableEvidence:
    """Hash paths and contents of supported, non-excluded files that can affect a map."""
    from kernel_tree_digest import file_entry, tree_digest, unreadable_entry

    root = Path(workspace_dir)
    # One framed entry per record (see kernel_tree_digest.tree_digest): the root, then each file as
    # file / oversized / unreadable, so no variant can be confused with another's payload.
    entries: list[list[object]] = [["root", str(root.resolve())]]

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        try:
            parts = file_path.relative_to(root).parts
        except ValueError:
            return UnavailableEvidence("REPO_MAP_CACHE_SNAPSHOT_PATH_OUTSIDE_ROOT")
        if any(part in _SKIP_DIRS or part.endswith(".egg-info") for part in parts):
            continue
        if file_path.suffix.lower() not in _LANG_MAP:
            continue

        relative = file_path.relative_to(root).as_posix()
        try:
            size = file_path.stat().st_size
            if size > _MAX_FILE_READ_BYTES:
                # Oversized files are omitted from the map; size still detects a transition below the cap.
                entries.append(["oversized", relative, size])
                continue
            content = file_path.read_bytes()
        except OSError:
            entries.append(unreadable_entry(relative))
            continue
        entries.append(file_entry(relative, content))

    return tree_digest("repo-map-cache-snapshot/1", entries)


def _make_cache_key(
    repo_url: str,
    commit_sha: str,
    workspace_dir: str | None = None,
    snapshot_digest: str | None = None,
) -> str | UnavailableEvidence:
    """Hash repository identity and snapshot. Empty commit_sha → never cache."""
    if not commit_sha or workspace_dir is None:
        return ""
    resolved_root = str(Path(workspace_dir).resolve())
    snapshot = snapshot_digest or _cache_snapshot_digest(workspace_dir)
    if not isinstance(snapshot, str):
        return snapshot
    fields = (_CACHE_FORMAT, repo_url, commit_sha, resolved_root, snapshot)
    digest = hashlib.sha256()
    for field in fields:
        encoded = os.fsencode(field)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()[:32]


def _load_cached_map(cache_key: str, cache_dir: Path | None = None) -> str | None:
    if not cache_key or cache_dir is None:
        return None
    cache_file = cache_dir / f"{cache_key}.txt"
    if not cache_file.exists():
        return None
    try:
        return cache_file.read_text(encoding="utf-8")
    except OSError:
        return None


def _save_cached_map(cache_key: str, repo_map: str, cache_dir: Path | None = None) -> None:
    """Write cache atomically via temp-file + rename to prevent partial reads."""
    if not cache_key or cache_dir is None:
        return
    cache_file = cache_dir / f"{cache_key}.txt"
    tmp_file: Path | None = None
    try:
        # One private temp file per writer: concurrent writers of the same key never share
        # an open file, so the rename publishes only a fully written map.
        fd, tmp_name = tempfile.mkstemp(prefix=f".{cache_key}.", suffix=".tmp", dir=cache_dir)
        tmp_file = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(repo_map)
        tmp_file.replace(cache_file)  # atomic on POSIX; best-effort on Windows
    except OSError as exc:
        logger.warning("repo_map: failed to write cache file %s: %s", cache_file, exc)
        if tmp_file is not None:
            try:
                tmp_file.unlink(missing_ok=True)
            except OSError:
                pass


# ── Public API ────────────────────────────────────────────────────────────────

def build_repo_map_with_stats(
    workspace_dir: str,
    repo_url: str = "",
    commit_sha: str = "",
    *,
    use_cache: bool = True,
) -> tuple[str, dict]:
    """Build the repo map and report the modes actually used (see ``_generate_repo_map_with_stats``).

    A cache hit returns ``{"source": "cache"}`` as its stats: the counts belong to the build that wrote it.
    Caching is skipped when disabled, when commit_sha is empty, or when no explicit cache home is configured.
    Never raises: returns ("", stats) on any failure."""
    if not workspace_dir:
        return "", {"source": "none"}

    cache_dir = _get_cache_dir() if use_cache and commit_sha else None
    cache_key = ""
    cache_identity_reason_code: str | None = None
    if cache_dir is not None:
        try:
            cache_identity = _make_cache_key(repo_url, commit_sha, workspace_dir=workspace_dir)
            if isinstance(cache_identity, str):
                cache_key = cache_identity
            else:
                cache_identity_reason_code = cache_identity.reason_code
        except (OSError, RuntimeError) as exc:
            logger.debug("repo_map: cache identity unavailable: %s", type(exc).__name__)
    cached = _load_cached_map(cache_key, cache_dir)
    if cached:
        logger.debug("repo_map: cache hit (key=%s)", cache_key)
        return cached, {"source": "cache"}

    try:
        repo_map, stats = _generate_repo_map_with_stats(workspace_dir)
    except Exception as exc:  # noqa: BLE001 - repo-map generation remains fail-open at intake.
        logger.warning("repo_map: generation failed: %s", exc)
        failure_stats = {"source": "error", "error": type(exc).__name__}
        if cache_identity_reason_code is not None:
            failure_stats["cache_identity_reason_code"] = cache_identity_reason_code
        return "", failure_stats

    if repo_map and cache_key:
        _save_cached_map(cache_key, repo_map, cache_dir)

    char_count = len(repo_map)
    approx_tokens = char_count // 4
    logger.info(
        "repo_map: generated %d chars (~%d tokens) for %s",
        char_count, approx_tokens, workspace_dir,
    )
    result_stats = {"source": "generated", **stats}
    if cache_identity_reason_code is not None:
        result_stats["cache_identity_reason_code"] = cache_identity_reason_code
    return repo_map, result_stats


def build_repo_map(
    workspace_dir: str,
    repo_url: str = "",
    commit_sha: str = "",
    *,
    use_cache: bool = True,
) -> str:
    """Build and return a compact text repo map for the workspace.

    Uses a root-and-content keyed cache only when explicitly enabled and a
    STATEFRAME_HOME is configured. Never raises — returns ""
    on any failure.
    """
    return build_repo_map_with_stats(
        workspace_dir, repo_url, commit_sha, use_cache=use_cache,
    )[0]


def _record_commit_sha_lookup_failure(error: BaseException, journal: object | None) -> None:
    """Record safe failure metadata without rendering the exception.

    Keep this metadata handling aligned with ``react_repair._record_intake_orientation_failure``.
    """
    error_type = type(error)
    exc_type = "<unknown>"
    try:
        candidate_name = type.__dict__["__name__"].__get__(error_type, type)
        if type(candidate_name) is str:
            exc_type = "<invalid>"
            if re.fullmatch(r"^[A-Za-z_][A-Za-z0-9_]{0,99}$", candidate_name) is not None:
                exc_type = candidate_name
    except BaseException:  # noqa: BLE001 - this guards only the built-in type slot descriptor
        pass
    errno_value: int | None = None
    try:
        error_mro = type.__dict__["__mro__"].__get__(error_type, type)
        if any(base is OSError for base in error_mro):
            # Bypass an OSError subclass's potentially hostile errno property.
            candidate = OSError.errno.__get__(error)
            if type(candidate) is int:
                errno_value = candidate
    except BaseException:  # noqa: BLE001 - diagnostics must not replace the lookup failure
        errno_value = None

    try:
        logger.warning(
            "repo_map: commit SHA lookup failed: exc_type=%s errno=%s",
            exc_type, errno_value,
        )
    except Exception:  # noqa: BLE001 - logging must not replace the lookup result
        pass

    selected_journal = journal
    if selected_journal is None:
        try:
            from kernel_pilot import ACTIVE_RUNTIME
            runtime = ACTIVE_RUNTIME.get()
            selected_journal = getattr(runtime, "journal", None)
        except (ImportError, LookupError):
            selected_journal = None
    try:
        emit = getattr(selected_journal, "emit", None)
        if callable(emit):
            emit(
                "repo_map_commit_sha_unavailable",
                reason=_COMMIT_SHA_LOOKUP_FAILURE_REASON,
                exc_type=exc_type,
                errno=errno_value,
            )
    except Exception:  # noqa: BLE001 - ordinary journal failures must not replace the lookup result
        pass


def get_current_commit_sha(workspace_dir: str, *, journal: object | None = None) -> str:
    """Get HEAD commit SHA for cache key computation; expected lookup failures return ''."""
    from pathlib import Path

    from kernel_git_view import GitObjectError, legacy_git_query
    try:
        result = legacy_git_query(Path(workspace_dir), ["rev-parse", "HEAD"])
    except (OSError, ValueError, GitObjectError) as error:
        _record_commit_sha_lookup_failure(error, journal)
        return ""
    return result.strip()[:40] if result else ""
