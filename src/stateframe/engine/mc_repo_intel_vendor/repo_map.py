"""Repo map generator — compact AST-based signature map for a repository.

Generates per-file function/class/export signatures suitable for injection into
the model's system prompt prefix (~50-80 tokens per file). Injecting this as a stable
prefix enables prompt caching on all subsequent orchestrator calls for the same repo.

Caches maps by (repo_url, commit_sha) to a file in LEAN_CTX_HOME/repo_map_cache/.
Regenerates automatically when commit_sha changes (new commits = new cache key).

Supports tree-sitter-language-pack (preferred) and tree-sitter-languages (legacy)
with a plain text fallback when neither is available.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from .ts_dispatch import outline_source

logger = logging.getLogger(__name__)

# ── tree-sitter availability ──────────────────────────────────────────────────
_TREE_SITTER_AVAILABLE = False
_get_parser = None

try:
    from tree_sitter_language_pack import get_parser as _tslp_get_parser  # type: ignore[import]
    _get_parser = _tslp_get_parser
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
    "coverage", ".pytest_cache", ".mypy_cache", ".eggs", ".mission",
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
        elif ext in (".ts", ".tsx", ".js", ".jsx"):
            if any(stripped.startswith(k) for k in TS_STARTS):
                sigs.append(stripped[:_MAX_SIG_CHARS])
        if len(sigs) >= _MAX_SIGS_PER_FILE:
            break
    return sigs


_SCAN_EXTS = (".py", ".ts", ".tsx", ".js", ".jsx")
_TEXT_SCAN = "text-scan fallback"


def _outline_sigs(source: str, rel_path: str) -> tuple[list[str], str, Optional[str]]:
    """Signatures for languages outside py/ts/js from the source outline, the mode it used, and why it degraded.

    ``ts_dispatch.outline_source`` parses with tree-sitter when a grammar is installed and falls back to its
    lexical scan otherwise; the reason is None unless a parser was available but failed."""
    outline = outline_source(source, rel_path)
    mode = "tree-sitter" if outline["parser"] == "tree-sitter" else _TEXT_SCAN
    reason = outline.get("degraded_reason")
    failure = (str(reason) if _TREE_SITTER_AVAILABLE and mode != "tree-sitter"
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


def _file_entry_with_mode(file_path: Path, repo_root: Path) -> tuple[Optional[str], str, Optional[str]]:
    """Return a bounded entry, the mode that produced it, and the parser-failure reason (None if none).

    Python/TS/JS/Vue files keep their original rules exactly (files under 10 bytes and files with no
    signatures are left out), so maps of repos made only of those languages are unchanged. Every other
    supported language goes through the source outline and is always listed, "(no signatures)" if empty."""
    ext = file_path.suffix.lower()
    if ext not in _LANG_MAP:
        return None, "", None
    scanned = ext in _SCAN_EXTS or ext == ".vue"

    try:
        stat = file_path.stat()
        if stat.st_size > _MAX_FILE_READ_BYTES or (scanned and stat.st_size < 10):
            return None, "", None
        source_bytes = file_path.read_bytes()
    except OSError:
        return None, "", None

    source = source_bytes.decode("utf-8", errors="replace")
    rel_path = str(file_path.relative_to(repo_root))
    sigs: list[str] = []
    mode = _TEXT_SCAN
    failure: Optional[str] = None

    if not scanned:
        try:
            sigs, mode, failure = _outline_sigs(source, rel_path)
        except Exception as exc:  # noqa: BLE001 - the map is fail-open: list the file, record why
            sigs, mode = [], _TEXT_SCAN
            failure = type(exc).__name__ if _TREE_SITTER_AVAILABLE else None
    elif _TREE_SITTER_AVAILABLE and _get_parser is not None and ext != ".vue":
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


def _file_entry(file_path: Path, repo_root: Path) -> Optional[str]:
    """Generate a compact signature entry for one file. Returns None if skipped."""
    return _file_entry_with_mode(file_path, repo_root)[0]


# ── Map generation ────────────────────────────────────────────────────────────

def _generate_repo_map_with_stats(workspace_dir: str) -> tuple[str, dict]:
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

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue

        # Skip excluded directories
        try:
            parts = file_path.relative_to(root).parts
        except ValueError:
            continue
        if any(p in _SKIP_DIRS or p.endswith(".egg-info") for p in parts):
            continue

        entry, mode, failure = _file_entry_with_mode(file_path, root)
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


def _generate_repo_map(workspace_dir: str) -> str:
    """Walk the workspace and generate the full repo map text."""
    return _generate_repo_map_with_stats(workspace_dir)[0]


# ── Cache ─────────────────────────────────────────────────────────────────────

def _get_cache_dir() -> Path:
    lean_ctx_home = os.getenv("LEAN_CTX_HOME", "/workspace/.lean-ctx")
    cache_dir = Path(lean_ctx_home) / "repo_map_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return cache_dir


# Bump when map content rules change, so maps cached by an older build are not reused.
_CACHE_FORMAT = "repo-map/2"


def _make_cache_key(repo_url: str, commit_sha: str) -> str:
    """Cache key is hash of repo_url + commit_sha. Empty commit_sha → never cache."""
    if not commit_sha:
        return ""
    raw = f"{_CACHE_FORMAT}:{repo_url}:{commit_sha}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _load_cached_map(cache_key: str) -> Optional[str]:
    if not cache_key:
        return None
    cache_file = _get_cache_dir() / f"{cache_key}.txt"
    if not cache_file.exists():
        return None
    try:
        return cache_file.read_text(encoding="utf-8")
    except OSError:
        return None


def _save_cached_map(cache_key: str, repo_map: str) -> None:
    """Write cache atomically via temp-file + rename to prevent partial reads."""
    if not cache_key:
        return
    cache_file = _get_cache_dir() / f"{cache_key}.txt"
    tmp_file = cache_file.with_suffix(".tmp")
    try:
        tmp_file.write_text(repo_map, encoding="utf-8")
        tmp_file.replace(cache_file)  # atomic on POSIX; best-effort on Windows
    except OSError as exc:
        logger.warning("repo_map: failed to write cache file %s: %s", cache_file, exc)
        try:
            tmp_file.unlink(missing_ok=True)
        except OSError:
            pass


# ── Public API ────────────────────────────────────────────────────────────────

def build_repo_map_with_stats(
    workspace_dir: str,
    repo_url: str = "",
    commit_sha: str = "",
) -> tuple[str, dict]:
    """Build the repo map and report the modes actually used (see ``_generate_repo_map_with_stats``).

    A cache hit returns ``{"source": "cache"}`` as its stats: the counts belong to the build that wrote it.
    Never raises: returns ("", stats) on any failure."""
    if not workspace_dir:
        return "", {"source": "none"}

    cache_key = _make_cache_key(repo_url, commit_sha)
    cached = _load_cached_map(cache_key)
    if cached:
        logger.debug("repo_map: cache hit (key=%s)", cache_key)
        return cached, {"source": "cache"}

    try:
        repo_map, stats = _generate_repo_map_with_stats(workspace_dir)
    except Exception as exc:
        logger.warning("repo_map: generation failed: %s", exc)
        return "", {"source": "error", "error": type(exc).__name__}

    if repo_map and cache_key:
        _save_cached_map(cache_key, repo_map)

    char_count = len(repo_map)
    approx_tokens = char_count // 4
    logger.info(
        "repo_map: generated %d chars (~%d tokens) for %s",
        char_count, approx_tokens, workspace_dir,
    )
    return repo_map, {"source": "generated", **stats}


def build_repo_map(
    workspace_dir: str,
    repo_url: str = "",
    commit_sha: str = "",
) -> str:
    """Build and return a compact text repo map for the workspace.

    Checks cache first (keyed on repo_url + commit_sha). Regenerates if cache
    misses or commit_sha is empty. Never raises — returns "" on any failure.
    """
    return build_repo_map_with_stats(workspace_dir, repo_url, commit_sha)[0]


def get_current_commit_sha(workspace_dir: str) -> str:
    """Get HEAD commit SHA for cache key computation. Returns '' on any failure."""
    from pathlib import Path
    from kernel_git_view import legacy_git_query
    result = legacy_git_query(Path(workspace_dir), ["rev-parse", "HEAD"])
    return result.strip()[:40] if result else ""
