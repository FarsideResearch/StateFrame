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
    try:
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
                    sigs.append(f"{prefix}def {name}{params}"[:_MAX_SIG_CHARS])
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
    except Exception as exc:
        logger.debug("repo_map: tree-sitter python extraction failed: %s", exc)
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
    try:
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
    except Exception as exc:
        logger.debug("repo_map: tree-sitter %s extraction failed: %s", lang, exc)
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


def _file_entry(file_path: Path, repo_root: Path) -> Optional[str]:
    """Generate a compact signature entry for one file. Returns None if skipped."""
    ext = file_path.suffix.lower()
    if ext not in _LANG_MAP:
        return None

    try:
        stat = file_path.stat()
        if stat.st_size > _MAX_FILE_READ_BYTES or stat.st_size < 10:
            return None
        source_bytes = file_path.read_bytes()
    except OSError:
        return None

    sigs: list[str] = []

    if _TREE_SITTER_AVAILABLE and _get_parser is not None:
        try:
            if ext == ".py":
                sigs = _extract_python_sigs(source_bytes)
            elif ext in (".ts", ".tsx"):
                sigs = _extract_ts_js_sigs(source_bytes, "typescript")
            elif ext in (".js", ".jsx"):
                sigs = _extract_ts_js_sigs(source_bytes, "javascript")
            elif ext == ".vue":
                # Vue SFCs: extract from <script> block using JavaScript parser
                # Full vue grammar support varies by tree-sitter package version
                sigs = _fallback_extract_sigs(source_bytes.decode("utf-8", errors="replace"), ".js")
            else:
                # Other languages: use text-scan fallback (Go/Rust/Java etc.)
                sigs = _fallback_extract_sigs(
                    source_bytes.decode("utf-8", errors="replace"), ext
                )
        except Exception:
            # Parser failure — degrade to text scan rather than losing the file entirely
            sigs = _fallback_extract_sigs(
                source_bytes.decode("utf-8", errors="replace"), ext
            )
        if not sigs:
            # Some installed tree-sitter packages expose a parser but fail to
            # produce nodes with the vendored grammar ABI. Preserve the map's
            # fail-open contract instead of dropping every affected file.
            sigs = _fallback_extract_sigs(
                source_bytes.decode("utf-8", errors="replace"), ext
            )
    else:
        sigs = _fallback_extract_sigs(source_bytes.decode("utf-8", errors="replace"), ext)

    if not sigs:
        return None

    rel_path = str(file_path.relative_to(repo_root))
    sig_block = "\n  ".join(sigs[:_MAX_SIGS_PER_FILE])
    entry = f"{rel_path}:\n  {sig_block}"
    return entry[:_MAX_FILE_ENTRY_CHARS]


# ── Map generation ────────────────────────────────────────────────────────────

def _generate_repo_map(workspace_dir: str) -> str:
    """Walk the workspace and generate the full repo map text."""
    root = Path(workspace_dir)
    if not root.exists():
        return ""

    file_entries: list[str] = []
    file_count = 0
    skipped = 0

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

        entry = _file_entry(file_path, root)
        if entry:
            file_entries.append(entry)
            file_count += 1
            if file_count >= _MAX_FILES:
                file_entries.append(f"[... map capped at {_MAX_FILES} files — additional files not shown]")
                break
        else:
            skipped += 1

    if not file_entries:
        return ""

    mode = "tree-sitter" if _TREE_SITTER_AVAILABLE else "text-scan fallback"
    header = f"## Repository Map ({file_count} files, {mode})\n"
    return header + "\n".join(file_entries)


# ── Cache ─────────────────────────────────────────────────────────────────────

def _get_cache_dir() -> Path:
    lean_ctx_home = os.getenv("LEAN_CTX_HOME", "/workspace/.lean-ctx")
    cache_dir = Path(lean_ctx_home) / "repo_map_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return cache_dir


def _make_cache_key(repo_url: str, commit_sha: str) -> str:
    """Cache key is hash of repo_url + commit_sha. Empty commit_sha → never cache."""
    if not commit_sha:
        return ""
    raw = f"{repo_url}:{commit_sha}"
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

def build_repo_map(
    workspace_dir: str,
    repo_url: str = "",
    commit_sha: str = "",
) -> str:
    """Build and return a compact text repo map for the workspace.

    Checks cache first (keyed on repo_url + commit_sha). Regenerates if cache
    misses or commit_sha is empty. Never raises — returns "" on any failure.
    """
    if not workspace_dir:
        return ""

    cache_key = _make_cache_key(repo_url, commit_sha)
    cached = _load_cached_map(cache_key)
    if cached:
        logger.debug("repo_map: cache hit (key=%s)", cache_key)
        return cached

    try:
        repo_map = _generate_repo_map(workspace_dir)
    except Exception as exc:
        logger.warning("repo_map: generation failed: %s", exc)
        return ""

    if repo_map and cache_key:
        _save_cached_map(cache_key, repo_map)

    char_count = len(repo_map)
    approx_tokens = char_count // 4
    logger.info(
        "repo_map: generated %d chars (~%d tokens) for %s",
        char_count, approx_tokens, workspace_dir,
    )
    return repo_map


def get_current_commit_sha(workspace_dir: str) -> str:
    """Get HEAD commit SHA for cache key computation. Returns '' on any failure."""
    from pathlib import Path
    from kernel_git_view import legacy_git_query
    result = legacy_git_query(Path(workspace_dir), ["rev-parse", "HEAD"])
    return result.strip()[:40] if result else ""
