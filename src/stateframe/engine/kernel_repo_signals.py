"""Host-computed repository signals.

The package is deliberately a presentation-only sidecar.  It reads the current
workspace and public task observations, then returns bounded suggestions.  It
does not run tests, edit files, add obligations, or participate in completion
reduction.  History is a separate, explicitly fenced input: only commits that
are ancestors of the supplied task base commit are ever inspected.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path


MODE_ENV = "STATEFRAME_REPO_SIGNALS"
SCHEMA = "repo-signals/1"
HISTORY_UNAVAILABLE = "unavailable"
_VALID_MODES = frozenset({"off", "on"})
_SOURCE_SUFFIXES = frozenset({
    ".c", ".cc", ".cpp", ".go", ".h", ".hpp", ".java", ".js", ".jsx",
    ".mjs", ".php", ".py", ".rb", ".rs", ".sql", ".swift", ".ts", ".tsx",
    ".vue",
})
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".bzr", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".venv", "__pycache__", "build", "dist",
    "node_modules", "vendor",
})
_STOPWORDS = frozenset({
    "about", "after", "again", "also", "and", "are", "but", "can", "change",
    "check", "create", "does", "for", "from", "fix", "have", "into", "that",
    "the", "this", "with", "when", "where", "which", "will", "would", "file",
    "files", "path", "paths", "test", "tests", "using", "use", "used",
})
_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9_.-]+)?"
    r"(?::\d+(?::\d+)?)?"
)
_SYMBOL_RE = re.compile(
    r"(?:`([^`]+)`|\b(?:class|function|method|symbol|fixture|handler|module)\s+"
    r"([A-Za-z_][A-Za-z0-9_.]*)|\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*))"
)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)
_HISTORY_WORDS_RE = re.compile(r"\b(?:fix|fixed|bug|bugfix|regression|repair|correct)\b", re.I)


def mode() -> str:
    """Return the strictly validated feature mode."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in _VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


repo_signals_mode = mode


def _normal_path(value: object) -> str:
    text = str(value or "").replace("\\", "/").strip()
    if text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def _relative_path(root: Path, value: object) -> str:
    text = _normal_path(value)
    if not text:
        return ""
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            return ""
    return Path(text).as_posix()


def _safe_root(root: Path | str) -> Path:
    return Path(root).expanduser().resolve()


def _source_paths(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    paths: list[Path] = []
    try:
        iterator = root.rglob("*")
        for path in iterator:
            if not path.is_file() or path.is_symlink():
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if any(part in _SKIP_DIRS or part.startswith(".") for part in relative.parts):
                continue
            if path.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            paths.append(path)
    except (OSError, RuntimeError):
        return paths
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def _workspace_paths(root: Path) -> list[Path]:
    """Return files for W0/Wt, including docs/configuration beyond source code."""
    if not root.exists() or not root.is_dir():
        return []
    paths: list[Path] = []
    try:
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if any(part in _SKIP_DIRS or part.startswith(".") and part != relative.name
                   for part in relative.parts[:-1]):
                continue
            paths.append(path)
    except (OSError, RuntimeError):
        return paths
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def _read_source(path: Path) -> str:
    try:
        # Source indexing is bounded so a generated/binary-looking file cannot
        # consume the request budget or make signal generation unbounded.
        return path.read_text(encoding="utf-8")[:200_000]
    except (OSError, UnicodeError):
        return ""


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(131_072), b""):
                digest.update(chunk)
    except (OSError, IOError):
        return "missing"
    return digest.hexdigest()


def _run_git(cwd: Path, args: Sequence[str], *, timeout: float = 3.0) -> str | None:
    """Run a read-only Git query with user config and pager effects disabled."""
    from kernel_git_view import legacy_git_query
    return legacy_git_query(cwd, [str(arg) for arg in args], timeout=timeout)


def _git_root(root: Path) -> Path | None:
    raw = _run_git(root, ("rev-parse", "--show-toplevel"))
    if not raw:
        return None
    try:
        repository = Path(raw.strip()).resolve()
        root.resolve().relative_to(repository)
    except (OSError, ValueError):
        return None
    return repository


def _task_prefix(root: Path, repository: Path) -> str:
    try:
        relative = root.resolve().relative_to(repository.resolve()).as_posix()
        return "" if relative == "." else relative
    except (OSError, ValueError):
        return ""


def _status_paths(root: Path, repository: Path | None) -> list[str]:
    if repository is None:
        return []
    prefix = _task_prefix(root, repository)
    output = _run_git(
        repository,
        ("status", "--short", "--untracked-files=all", "--", prefix or "."),
    )
    if not output:
        return []
    paths: list[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        raw = line[3:].strip()
        if " -> " in raw:
            raw = raw.rsplit(" -> ", 1)[-1]
        raw = raw.strip('"')
        if prefix:
            prefix_slash = prefix.rstrip("/") + "/"
            if raw == prefix:
                relative = ""
            elif raw.startswith(prefix_slash):
                relative = raw[len(prefix_slash):]
            else:
                continue
        else:
            relative = raw
        if relative:
            paths.append(Path(relative).as_posix())
    return sorted(set(paths))


def capture_task_start(root: Path | str) -> dict[str, object]:
    """Capture W0 before task edits; dirty-at-start status remains separate."""
    path = _safe_root(root)
    repository = _git_root(path)
    files: dict[str, str] = {}
    for source in _workspace_paths(path):
        relative = source.relative_to(path).as_posix()
        files[relative] = _file_digest(source)
    dirty = _status_paths(path, repository)
    return {
        "schema": "repo-snapshot/1",
        "files": files,
        "dirty_at_capture": dirty,
        "git_repository": str(repository) if repository is not None else None,
        "revision": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


snapshot_workspace = capture_task_start


def _snapshot_files(snapshot: Mapping[str, object] | None) -> dict[str, str]:
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("files"), Mapping):
        return {}
    return {str(key): str(value) for key, value in snapshot["files"].items()}


def task_delta(w0: Mapping[str, object], wt: Mapping[str, object]) -> dict[str, object]:
    """Diff W0 and Wt, with initial dirty paths never folded into task work."""
    before = _snapshot_files(w0)
    after = _snapshot_files(wt)
    added = sorted(set(after) - set(before))
    deleted = sorted(set(before) - set(after))
    modified = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    changed = sorted((*added, *modified, *deleted))
    pre_existing = sorted({
        str(path) for path in (w0.get("dirty_at_capture", ()) if isinstance(w0, Mapping) else ())
        if str(path)
    })
    return {
        "w0": {"file_count": len(before), "revision": w0.get("revision") if isinstance(w0, Mapping) else None},
        "wt": {"file_count": len(after), "revision": wt.get("revision") if isinstance(wt, Mapping) else None},
        "added": added,
        "modified": modified,
        "deleted": deleted,
        "changed_since_task_start": changed,
        "pre_existing_dirty": pre_existing,
        "pre_existing_dirty_separate": True,
        "task_changes": [
            {"path": path, "kind": ("added" if path in added else "deleted" if path in deleted else "modified"),
             "pre_existing_dirty": path in pre_existing}
            for path in changed
        ],
    }


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _brief(context: object) -> str:
    if isinstance(context, str):
        return context
    if not isinstance(context, Mapping):
        return ""
    for key in ("brief", "task_brief", "description", "symptom", "msg", "ticket"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    row = context.get("task_row")
    if isinstance(row, Mapping):
        return _brief(row)
    return ""


def _context(task: object, frame: object) -> dict[str, object]:
    result: dict[str, object] = {}
    if isinstance(task, str):
        result["brief"] = task
    elif isinstance(task, Mapping):
        result.update(copy.deepcopy(dict(task)))
    if isinstance(frame, Mapping):
        for key, value in frame.items():
            result.setdefault(str(key), value)
        row = frame.get("task_row")
        if isinstance(row, Mapping):
            for key, value in row.items():
                result.setdefault(str(key), value)
    return result


def _values(value: object) -> list[object]:
    if isinstance(value, Mapping):
        result: list[object] = []
        for nested in value.values():
            result.extend(_values(nested))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for nested in value:
            result.extend(_values(nested))
        return result
    return [value]


def _task_paths(root: Path, context: Mapping[str, object], target_path: object = None) -> list[str]:
    candidates: list[str] = []
    if target_path:
        candidates.append(_relative_path(root, target_path))
    for key in ("target_path", "path", "file", "target_files", "files", "product_files",
                "affected_files", "source_files"):
        value = context.get(key)
        for item in _values(value):
            if isinstance(item, Mapping):
                item = item.get("path", item.get("file", item.get("target_path", "")))
            if isinstance(item, str) and ("/" in item or Path(item).suffix):
                relative = _relative_path(root, item)
                if relative:
                    candidates.append(relative)
    return list(dict.fromkeys(candidates))


def _task_symbols(context: Mapping[str, object]) -> list[str]:
    text = _brief(context)
    values: list[str] = []
    for match in _SYMBOL_RE.finditer(text):
        value = next((item for item in match.groups() if item), "")
        value = value.strip("`'\".,:;()[]")
        if value and value not in values:
            values.append(value)
    for key in ("symbol", "target_symbol", "function", "fn", "edited_symbols"):
        for item in _values(context.get(key)):
            if isinstance(item, str):
                value = item.strip("`'\".,:;()[]")
                if value and value not in values:
                    values.append(value)
    return values


def _all_text(context: Mapping[str, object], receipts: object = None) -> str:
    pieces = [_brief(context)]
    for key in ("last_action", "recent_test_feedback", "test_receipts", "receipts",
                "failure", "stack", "traceback", "failure_receipts", "observations"):
        value = context.get(key)
        if value is not None:
            pieces.extend(str(item) for item in _values(value))
    if receipts is not None:
        pieces.extend(str(item) for item in _values(receipts))
    return "\n".join(pieces)


def _paths_from_text(text: str) -> list[str]:
    paths: list[str] = []
    for match in _PATH_RE.finditer(text):
        value = match.group(0).split(":", 1)[0].strip("`'\"()[]{}.,;")
        if value and value not in paths:
            paths.append(Path(value).as_posix())
    return paths


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS}


def _source_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _read_source(path)
        for path in _source_paths(root)[:1000]
    }


def rank_fusion(channels: Mapping[str, object], *, weights: Mapping[str, float] | None = None,
                k: float = 60.0) -> list[dict[str, object]]:
    """Fuse ranked channel lists using ``sum(weight / (k + rank))``.

    Channel names, candidate paths, and ties are sorted before the calculation
    where ordering is not evidence.  This keeps repeated projections stable.
    """
    if k < 0:
        raise ValueError("rank-fusion k must be nonnegative")
    configured = {"exact": 4.0, "stack_failure": 3.0, "lexical": 2.0,
                  "static_reference": 1.5, "co_change": 1.0}
    if weights:
        configured.update({str(key): float(value) for key, value in weights.items()})
    scores: defaultdict[str, float] = defaultdict(float)
    ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for original_channel in sorted(channels, key=str):
        channel = str(original_channel)
        raw = channels[original_channel]
        if isinstance(raw, Mapping):
            rows = [str(path) for path, _score in sorted(
                raw.items(), key=lambda item: (-float(item[1]), str(item[0])))]
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            rows = [str(path) for path in raw]
        else:
            rows = []
        unique: list[str] = []
        for path in rows:
            path = _normal_path(path)
            if path and path not in unique:
                unique.append(path)
        for rank, path in enumerate(unique, 1):
            ranks[path][channel] = rank
            scores[path] += configured.get(channel, 1.0) / (float(k) + rank)
    result = []
    for path in sorted(scores, key=lambda item: (-scores[item], item)):
        result.append({
            "path": path,
            "score": round(scores[path], 10),
            "channel_ranks": {key: ranks[path][key] for key in sorted(ranks[path])},
            "channels": sorted(ranks[path]),
        })
    return result


def _localization(root: Path, context: Mapping[str, object], sources: Mapping[str, str],
                  target_path: object, receipts: object) -> tuple[dict[str, object], list[str], list[str]]:
    paths = _task_paths(root, context, target_path)
    symbols = _task_symbols(context)
    combined = _all_text(context, receipts)
    stack_parts: list[str] = []
    for key in ("last_action", "recent_test_feedback", "test_receipts", "receipts",
                "failure", "stack", "traceback", "failure_receipts", "observations"):
        if context.get(key) is not None:
            stack_parts.extend(str(item) for item in _values(context[key]))
    if receipts is not None:
        stack_parts.extend(str(item) for item in _values(receipts))
    stack_paths = _paths_from_text("\n".join(stack_parts))
    task_tokens = _tokens(combined)
    exact: list[str] = []
    stack: list[str] = []
    lexical_scores: dict[str, float] = {}
    static_scores: dict[str, float] = {}
    for path in sources:
        path_tokens = _tokens(path.replace("/", " "))
        text = sources[path]
        if path in paths or Path(path).name in {Path(item).name for item in paths}:
            exact.append(path)
        if any(symbol and re.search(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])", text)
               for symbol in symbols):
            exact.append(path)
        if path in stack_paths or Path(path).name in {Path(item).name for item in stack_paths}:
            stack.append(path)
        lexical = len(path_tokens & task_tokens)
        for symbol in symbols:
            if symbol.lower() in text.lower():
                lexical += 1
        if lexical:
            lexical_scores[path] = float(lexical)
        static = 0.0
        for symbol in symbols:
            if re.search(rf"\b{re.escape(symbol.split('.')[-1])}\b", text):
                static += 1.0
        for item in paths:
            stem = Path(item).stem
            if stem and stem in text:
                static += 0.5
        if static:
            static_scores[path] = static
    channels: dict[str, object] = {
        "exact": sorted(set(exact)),
        "lexical": [path for path, _score in sorted(lexical_scores.items(), key=lambda item: (-item[1], item[0]))],
        "stack_failure": sorted(set(stack)),
        "static_reference": [path for path, _score in sorted(static_scores.items(), key=lambda item: (-item[1], item[0]))],
    }
    fused = rank_fusion(channels)
    # A ticket with little text should still get a deterministic inspection
    # shortlist from the current source tree, never from hidden/oracle data.
    seen = {row["path"] for row in fused}
    for path in sorted(sources):
        if path not in seen:
            fused.append({"path": path, "score": 0.0, "channel_ranks": {}, "channels": []})
        if len(fused) >= 50:
            break
    ranked_candidates = copy.deepcopy(fused[:50])
    shortlist = fused[:8]
    labels = {
        "exact": "exact ticket path/symbol",
        "lexical": "lexical task terms",
        "stack_failure": "stack/failure receipt",
        "static_reference": "static reference",
    }
    for row in shortlist:
        row["reasons"] = [labels[channel] for channel in row["channels"]]
        if not row["reasons"]:
            row["reasons"] = ["workspace candidate"]
        row["suggestion_only"] = True
        row["label"] = "suggestion, not root cause"
    return {"shortlist": shortlist, "ranked_candidates": ranked_candidates,
            "channel_candidates": channels,
            "label": "suggestion, not root cause"}, paths, symbols


def _test_path(path: str, text: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name.lower()
    return (name.startswith("test_") or name.endswith("_test.py") or name.endswith(".test.js")
            or any(part.lower() in {"test", "tests", "spec", "specs"} for part in parts)
            or bool(re.search(r"\b(?:describe|it|pytest|unittest|assert)\b", text)))


def _observed_rows(context: Mapping[str, object], receipts: object) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for value in (context.get("coverage"), context.get("observed_coverage"),
                  context.get("coverage_receipts"), context.get("test_links"),
                  context.get("test_receipts"), receipts):
        if isinstance(value, Mapping):
            rows.append(value)
            rows.extend(item for item in value.values() if isinstance(item, Mapping))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            rows.extend(item for item in value if isinstance(item, Mapping))
    return rows


def _row_paths(row: Mapping[str, object]) -> list[str]:
    paths: list[str] = []
    for key in ("test", "test_path", "path", "target", "target_path", "file", "covered_file"):
        value = row.get(key)
        if isinstance(value, str) and ("/" in value or Path(value).suffix):
            paths.append(_normal_path(value).split("::", 1)[0])
    for key in ("selectors", "covered_files", "files", "coverage_paths"):
        value = row.get(key)
        for item in _values(value):
            if isinstance(item, str) and ("/" in item or Path(item).suffix):
                paths.append(_normal_path(item).split("::", 1)[0])
    return list(dict.fromkeys(paths))


def mark_test_links_stale(links: Sequence[Mapping[str, object]], *,
                          current_revisions: Mapping[str, object] | None = None,
                          changed_paths: Sequence[str] = (),
                          current_revision: object = None) -> list[dict[str, object]]:
    """Refresh link freshness without upgrading an inferred link to coverage."""
    revisions = {_normal_path(key): str(value) for key, value in (current_revisions or {}).items()}
    changed = {_normal_path(path) for path in changed_paths}
    refreshed: list[dict[str, object]] = []
    for row in links:
        item = copy.deepcopy(dict(row))
        target = _normal_path(item.get("target"))
        observed = item.get("observed_revision", item.get("revision"))
        stale = bool(item.get("stale", False))
        if target in changed:
            stale = True
        if observed is not None and target in revisions and str(observed) != revisions[target]:
            stale = True
        if (current_revision is not None and observed is not None
                and str(observed) != str(current_revision)):
            stale = True
        item["stale"] = stale
        item["status"] = "stale_after_edit" if stale else "current"
        refreshed.append(item)
    return refreshed


def _test_links(root: Path, context: Mapping[str, object], sources: Mapping[str, str],
                shortlist: Sequence[Mapping[str, object]], receipts: object,
                delta: Mapping[str, object], history: Mapping[str, object]) -> list[dict[str, object]]:
    tests = {path: text for path, text in sources.items() if _test_path(path, text)}
    observed_rows = _observed_rows(context, receipts)
    changed = delta.get("changed_since_task_start", ())
    co_change = history.get("co_change") if isinstance(history.get("co_change"), Mapping) else {}
    co_change_paths = {
        str(row.get("path")) for row in co_change.get("companions", ())
        if isinstance(row, Mapping) and row.get("path")
    }
    links: list[dict[str, object]] = []
    priority = {"observed_coverage": 0, "import": 1, "name": 2, "co_change": 3}
    for target_row in shortlist[:8]:
        target = str(target_row.get("path", ""))
        if not target:
            continue
        if target in sources and _test_path(target, sources[target]):
            continue
        target_stem = Path(target).stem.lower()
        module = target.replace("/", ".").rsplit(".", 1)[0]
        for test, text in tests.items():
            test_name = test.lower()
            basis: str | None = None
            evidence = ""
            observed_revision = None
            for row in observed_rows:
                row_paths = {_normal_path(item) for item in _row_paths(row)}
                if test in row_paths and (target in row_paths or not any("/" in item for item in row_paths)):
                    basis = "observed_coverage"
                    evidence = "host observation names this test and target"
                    observed_revision = row.get("revision", row.get("file_revision"))
                    break
            if basis is None and (f"from {module} " in text or f"import {module}" in text
                                  or re.search(rf"\b{re.escape(Path(target).stem)}\b", text)):
                basis = "import"
                evidence = "test source imports or references target module"
            if basis is None and target_stem and target_stem in test_name:
                basis = "name"
                evidence = "test name matches target path"
            if basis is None and test in co_change_paths:
                basis = "co_change"
                evidence = "permitted ancestor history co-change (support threshold met)"
            if basis is None:
                continue
            current_revision = _file_digest(root / target) if (root / target).is_file() else "missing"
            link = {
                "test": test,
                "target": target,
                "basis": basis,
                "evidence": evidence,
                "claim": "observed coverage" if basis == "observed_coverage" else f"{basis} relationship",
                "stale": False,
                "status": "current",
            }
            if observed_revision is not None:
                link["observed_revision"] = str(observed_revision)
            if basis == "observed_coverage" and target in changed:
                link["stale"] = True
                link["status"] = "stale_after_edit"
            link["current_revision"] = current_revision
            links.append(link)
            if test in co_change_paths and basis != "co_change":
                co_link = dict(link)
                co_link.update({
                    "basis": "co_change",
                    "evidence": "permitted ancestor history co-change (support threshold met)",
                    "claim": "co-change relationship",
                })
                links.append(co_link)
    unique = {(row["test"], row["target"], row["basis"]): row for row in links}
    return sorted(unique.values(), key=lambda row: (
        priority.get(str(row.get("basis")), 9), str(row.get("test")), str(row.get("target"))))[:16]


def _defined_symbols(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, TypeError):
        return set(re.findall(r"\b(?:class|def|function)\s+([A-Za-z_]\w*)", text))
    return {node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def _neighbors(context: Mapping[str, object], sources: Mapping[str, str],
               shortlist: Sequence[Mapping[str, object]], symbols: Sequence[str]) -> dict[str, object]:
    target_paths = {str(row.get("path")) for row in shortlist[:8] if row.get("path")}
    names = {item.split(".")[-1] for item in symbols if item}
    for path in target_paths:
        names.update(_defined_symbols(sources.get(path, "")))
    rows: list[dict[str, object]] = []
    for path, text in sources.items():
        if path in target_paths:
            continue
        matched = sorted(name for name in names if name and re.search(rf"\b{re.escape(name)}\b", text))
        if not matched:
            continue
        caller = any(re.search(rf"\b{re.escape(name)}\s*\(", text) for name in matched)
        rows.append({
            "path": path,
            "kind": "caller" if caller else "reference",
            "basis": "static_reference",
            "symbols": matched[:4],
            "suggestion_only": True,
        })
    rows.sort(key=lambda row: (0 if row["kind"] == "caller" else 1, row["path"]))
    shown = rows[:8]
    return {"neighbors": shown, "omitted_count": max(0, len(rows) - len(shown)),
            "basis": "static_reference", "suggestion_only": True}


def _base_commit(context: Mapping[str, object], explicit: object = None) -> str | None:
    if explicit:
        return str(explicit)
    for key in ("base_commit", "task_base_commit", "base_revision", "task_base_sha", "base_sha"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("STATEFRAME_REPO_BASE_COMMIT", "STATEFRAME_TASK_BASE_COMMIT"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    for key in ("git", "repository", "repo"):
        value = context.get(key)
        if isinstance(value, Mapping):
            found = _base_commit(value)
            if found:
                return found
    return None


def _history_scope(root: Path, base_commit: object) -> dict[str, object]:
    repository = _git_root(root)
    if repository is None:
        return {"status": HISTORY_UNAVAILABLE, "reason": "no_git_repository",
                "allowed_commits": [], "repository": None, "base_commit": None}
    raw_base = str(base_commit or "").strip()
    if not raw_base:
        return {"status": HISTORY_UNAVAILABLE, "reason": "base_commit_required",
                "allowed_commits": [], "repository": str(repository), "base_commit": None}
    # Symbolic moving names are intentionally refused.  A task must supply its
    # immutable base object; this also prevents accidentally inspecting the
    # real repository's current tip.
    if raw_base.lower() in {"head", "main", "master", "origin/main", "@"} or not _COMMIT_RE.fullmatch(raw_base):
        return {"status": HISTORY_UNAVAILABLE, "reason": "base_commit_not_immutable",
                "allowed_commits": [], "repository": str(repository), "base_commit": None}
    resolved = _run_git(repository, ("rev-parse", "--verify", f"{raw_base}^{{commit}}"))
    if not resolved:
        return {"status": HISTORY_UNAVAILABLE, "reason": "base_commit_not_found",
                "allowed_commits": [], "repository": str(repository), "base_commit": None}
    full_base = resolved.strip().splitlines()[0]
    output = _run_git(repository, ("rev-list", full_base))
    allowed = [item.strip() for item in (output or "").splitlines()
               if _COMMIT_RE.fullmatch(item.strip())]
    if full_base not in allowed:
        allowed.insert(0, full_base)
    allowed = list(dict.fromkeys(allowed))
    return {"status": "available", "reason": "ancestor_set_only", "repository": str(repository),
            "base_commit": full_base, "allowed_commits": allowed}


def allowed_commit_set(root: Path | str, base_commit: str) -> tuple[str, ...]:
    """Return exactly the immutable ancestor set used by history signals."""
    return tuple(_history_scope(_safe_root(root), base_commit).get("allowed_commits", ()))


def _history_paths(scope: Mapping[str, object], root: Path, commit: str) -> list[str]:
    repository = Path(str(scope.get("repository")))
    prefix = _task_prefix(root, repository)
    output = _run_git(repository, ("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit))
    result: list[str] = []
    for raw in (output or "").splitlines():
        path = raw.strip()
        if not path:
            continue
        if prefix:
            prefix_slash = prefix.rstrip("/") + "/"
            if not path.startswith(prefix_slash):
                continue
            path = path[len(prefix_slash):]
        result.append(Path(path).as_posix())
    return sorted(set(result))


def _commit_subject(scope: Mapping[str, object], commit: str) -> str:
    output = _run_git(Path(str(scope.get("repository"))), ("show", "-s", "--format=%s", commit))
    return (output or "").strip().splitlines()[0][:240] if output else ""


def _co_change(scope: Mapping[str, object], root: Path, targets: Sequence[str]) -> dict[str, object]:
    allowed = [str(item) for item in scope.get("allowed_commits", ())]
    target_set = set(targets)
    commit_paths = {commit: set(_history_paths(scope, root, commit)) for commit in allowed}
    relevant = [set(paths) for paths in commit_paths.values() if paths]
    target_commits = [paths for paths in relevant if paths & target_set]
    n_a = len(target_commits)
    total = len(relevant)
    counts: Counter[str] = Counter(path for paths in target_commits for path in paths - target_set)
    all_counts: Counter[str] = Counter(path for paths in relevant for path in paths)
    companions: list[dict[str, object]] = []
    for path, support in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if support < 3 or n_a <= 0 or total <= 0:
            continue
        confidence = support / n_a
        prevalence = all_counts[path] / total
        lift = confidence / prevalence if prevalence else 0.0
        companions.append({
            "path": path,
            "support": support,
            "confidence": round(confidence, 6),
            "lift": round(lift, 6),
            "direction": "co-change_same_commit",
            "label": "inspect, not required edit",
            "suggestion_only": True,
        })
    return {"status": "available", "support_threshold": 3, "n_target_commits": n_a,
            "history_commit_count": total, "companions": companions[:3],
            "omitted_count": max(0, len(companions) - 3),
            "statistics_scope": "permitted ancestors of base only"}


def _evolution(scope: Mapping[str, object], root: Path, targets: Sequence[str]) -> dict[str, object]:
    if not targets:
        return {"status": HISTORY_UNAVAILABLE, "reason": "no_target_path",
                "causal_status": "not_established"}
    repository = Path(str(scope.get("repository")))
    base = str(scope.get("base_commit"))
    path = targets[0]
    prefix = _task_prefix(root, repository)
    repo_path = f"{prefix}/{path}" if prefix else path
    blame = _run_git(repository, ("blame", "--line-porcelain", base, "--", repo_path), timeout=5.0)
    if not blame:
        return {"status": HISTORY_UNAVAILABLE, "reason": "path_not_in_base",
                "causal_status": "not_established"}
    allowed = set(str(item) for item in scope.get("allowed_commits", ()))
    counts: Counter[str] = Counter()
    for line in blame.splitlines():
        match = re.match(r"^([0-9a-f]{7,64})\s+", line, re.I)
        if match and match.group(1) in allowed:
            counts[match.group(1)] += 1
    rows = [{"commit": commit, "lines": count, "subject": _commit_subject(scope, commit)}
            for commit, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]]
    relevant = rows[0] if rows else None
    return {"status": "available", "path": path, "range_blame": rows,
            "relevant_ancestor_commit": relevant,
            "causal_status": "not_established"}


def _prior_fix(scope: Mapping[str, object], root: Path, targets: Sequence[str]) -> dict[str, object]:
    repository = Path(str(scope.get("repository")))
    prefix = _task_prefix(root, repository)
    allowed = [str(item) for item in scope.get("allowed_commits", ())]
    for commit in allowed:
        subject = _commit_subject(scope, commit)
        changed = _history_paths(scope, root, commit)
        if not any(path in targets for path in changed) or not _HISTORY_WORDS_RE.search(subject):
            continue
        path = next(path for path in changed if path in targets)
        repo_path = f"{prefix}/{path}" if prefix else path
        hunk = _run_git(repository, ("show", "--format=fuller", "--no-ext-diff", "--unified=3",
                                     commit, "--", repo_path), timeout=5.0)
        if not hunk:
            continue
        return {"status": "available", "commit": commit, "path": path,
                "subject": subject, "hunk": hunk[:2400],
                "applicability": "unverified", "allowed_commit": True,
                "suggestion_only": True}
    return {"status": HISTORY_UNAVAILABLE, "reason": "no_permitted_prior_fix",
            "applicability": "unverified"}


def _history_signals(root: Path, context: Mapping[str, object], targets: Sequence[str],
                     explicit_base: object = None) -> dict[str, object]:
    scope = _history_scope(root, explicit_base or _base_commit(context))
    base = {key: copy.deepcopy(value) for key, value in scope.items()
            if key in {"status", "reason", "repository", "base_commit", "allowed_commits"}}
    if scope.get("status") != "available":
        base.update({
            "co_change": {"status": HISTORY_UNAVAILABLE, "reason": scope.get("reason"), "companions": []},
            "evolution": {"status": HISTORY_UNAVAILABLE, "reason": scope.get("reason"),
                           "causal_status": "not_established"},
            "prior_fix": {"status": HISTORY_UNAVAILABLE, "reason": scope.get("reason"),
                          "applicability": "unverified"},
        })
        return base
    base["co_change"] = _co_change(scope, root, targets)
    base["evolution"] = _evolution(scope, root, targets)
    base["prior_fix"] = _prior_fix(scope, root, targets)
    return base


def _compact(value: object, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def compute_repo_signals(root: Path | str, *, task: object = None, frame: object = None,
                         w0: Mapping[str, object] | None = None, target_path: object = None,
                         receipts: object = None, base_commit: object = None) -> dict[str, object]:
    """Compute the bounded v1 signal object from public workspace observations."""
    workspace = _safe_root(root)
    context = _context(task, frame)
    if receipts is not None and "receipts" not in context:
        context["receipts"] = receipts
    start = copy.deepcopy(dict(w0)) if isinstance(w0, Mapping) else capture_task_start(workspace)
    current = capture_task_start(workspace)
    delta = task_delta(start, current)
    sources = _source_inventory(workspace)
    localization, task_paths, symbols = _localization(workspace, context, sources, target_path, receipts)
    history_targets = list(dict.fromkeys(
        task_paths or [str(row.get("path")) for row in localization["shortlist"][:1]
                      if row.get("path")]
    ))
    history = _history_signals(workspace, context, history_targets, base_commit)
    links = _test_links(workspace, context, sources, localization["shortlist"], receipts, delta, history)
    links = mark_test_links_stale(
        links,
        current_revisions=current.get("files", {}) if isinstance(current.get("files"), Mapping) else {},
        changed_paths=delta.get("changed_since_task_start", ()),
    )
    neighbors = _neighbors(context, sources, localization["shortlist"], symbols)
    return {
        "schema": SCHEMA,
        "authority": {"kind": "suggestion_only", "may_block": False,
                       "creates_requirements": False, "creates_counterexamples": False},
        "task_delta": delta,
        "localization": localization,
        "test_links": links,
        "structural_neighbors": neighbors,
        "history": history,
        "retrieval_scope": "current source plus permitted ancestor history only",
        "expanded_action": "expand_repo_signals",
    }


def _render_history(history: Mapping[str, object], *, expanded: bool) -> list[str]:
    co_change = history.get("co_change") if isinstance(history.get("co_change"), Mapping) else {}
    companions = co_change.get("companions", ()) if isinstance(co_change, Mapping) else ()
    if companions:
        details = "; ".join(
            f"{row.get('path')} s={row.get('support')} c={row.get('confidence')} "
            f"lift={row.get('lift')} dir={row.get('direction')}"
            for row in companions[:3] if isinstance(row, Mapping))
        co_line = f"available {details} (inspect, not required edit)"
    else:
        co_line = f"{co_change.get('status', HISTORY_UNAVAILABLE)}"
        if co_change.get("reason"):
            co_line += f" ({co_change['reason']})"
        co_line += " (inspect, not required edit)"
    evolution = history.get("evolution") if isinstance(history.get("evolution"), Mapping) else {}
    prior = history.get("prior_fix") if isinstance(history.get("prior_fix"), Mapping) else {}
    ancestor = evolution.get("relevant_ancestor_commit")
    evolution_detail = f" ancestor={ancestor.get('commit')}" if isinstance(ancestor, Mapping) else ""
    prior_detail = (f" commit={prior.get('commit')} path={prior.get('path')}"
                    if prior.get("status") == "available" else "")
    lines = [f"co-change: {co_line}",
             f"evolution: {evolution.get('status', HISTORY_UNAVAILABLE)} "
             f"causal_status={evolution.get('causal_status', 'not_established')}{evolution_detail}",
             f"prior fix: {prior.get('status', HISTORY_UNAVAILABLE)} "
             f"applicability={prior.get('applicability', 'unverified')}{prior_detail}" ]
    if prior.get("hunk") and not expanded:
        lines.append("  retrieved prior-fix hunk: "
                     + _compact(prior["hunk"], 260)
                     + " (applicability unverified)")
    if expanded and evolution.get("relevant_ancestor_commit"):
        row = evolution["relevant_ancestor_commit"]
        lines.append(f"  ancestor: {row.get('commit')} {_compact(row.get('subject'), 120)}")
    if expanded and prior.get("hunk"):
        lines.append("  retrieved hunk (applicability unverified): " + _compact(prior["hunk"], 1000))
    return lines


def render_repo_signals(signals: Mapping[str, object], *, expanded: bool = False,
                        max_tokens: int = 560) -> str:
    """Render a compact block; expansion is only selected by an explicit caller."""
    if not isinstance(signals, Mapping):
        raise TypeError("repository signals must be a mapping")
    delta = signals.get("task_delta") if isinstance(signals.get("task_delta"), Mapping) else {}
    localization = signals.get("localization") if isinstance(signals.get("localization"), Mapping) else {}
    links = signals.get("test_links") if isinstance(signals.get("test_links"), Sequence) else ()
    neighbors = signals.get("structural_neighbors") if isinstance(signals.get("structural_neighbors"), Mapping) else {}
    lines = [
        "[Repository signals v1 | suggestions only]",
        "authority: suggestions only; suggestion, not root cause",
        (f"1 task delta W0->Wt: +{len(delta.get('added', ()))}, "
         f"~{len(delta.get('modified', ()))}, -{len(delta.get('deleted', ()))}, "
         f"pre-existing dirty kept separate={len(delta.get('pre_existing_dirty', ())) }"),
        "2 localization (suggestion, not root cause):",
    ]
    shortlist = localization.get("shortlist", ())
    shown = shortlist if expanded else shortlist[:8]
    for index, row in enumerate(shown, 1):
        if not isinstance(row, Mapping):
            continue
        reasons = ",".join(str(item) for item in row.get("reasons", ()))
        lines.append(f"  {index}. {row.get('path')} [{reasons}] score={row.get('score', 0)}")
    if not shortlist:
        lines.append("  unavailable (no current source candidates)")
    lines.append("3 test links (basis typed; observed coverage only when observed):")
    for row in list(links)[:12 if expanded else 6]:
        if isinstance(row, Mapping):
            lines.append(f"  {row.get('test')} -> {row.get('target')} basis={row.get('basis')} "
                         f"status={row.get('status', 'current')}")
    if not links:
        lines.append("  none observed")
    lines.append(f"4 structural neighbors (basis={neighbors.get('basis', 'static_reference')}, "
                 f"omitted={neighbors.get('omitted_count', 0)}):")
    for row in list(neighbors.get("neighbors", ()))[:8 if expanded else 5]:
        if isinstance(row, Mapping):
            lines.append(f"  {row.get('kind')} {row.get('path')} basis={row.get('basis')}")
    if not neighbors.get("neighbors"):
        lines.append("  none observed")
    lines.extend(f"{index} {line}" for index, line in enumerate(
        _render_history(signals.get("history", {}), expanded=expanded), 5))
    lines.append("action=expand_repo_signals (explicit action only)")
    rendered = "\n".join(lines)
    if expanded or max_tokens <= 0:
        return rendered
    # Keep the compact projection within its stated normal budget while
    # retaining its header and authority label if a pathological fixture has
    # unusually long paths.
    words = rendered.split()
    if len(words) <= max_tokens:
        return rendered
    clipped = " ".join(words[:max_tokens - 8])
    return clipped + " ... [expand_repo_signals for full suggestions]"


def expand_repo_signals(signals: Mapping[str, object]) -> str:
    return render_repo_signals(signals, expanded=True)


def append_repo_signals(messages: Sequence[dict], signals: Mapping[str, object] | None,
                        *, provider_request: bool = True, expanded: bool = False) -> list[dict]:
    """Append only an advisory user block; OFF and non-provider paths copy unchanged."""
    result = copy.deepcopy(list(messages))
    if not provider_request or mode() != "on" or not isinstance(signals, Mapping):
        return result
    result.append({"role": "user", "content": render_repo_signals(signals, expanded=expanded)})
    return result


def apply_repo_signals_to_authority(frame: Mapping[str, object], signals: Mapping[str, object] | None) -> dict[str, object]:
    """Compatibility boundary proving that suggestions cannot alter authority."""
    return copy.deepcopy(dict(frame))


# Descriptive aliases keep the small package usable by offline probes without
# making the renderer's internal helpers part of the authority path.
compute_task_delta = task_delta
fuse_channels = rank_fusion
history_scope = _history_scope


__all__ = [
    "MODE_ENV", "SCHEMA", "allowed_commit_set", "append_repo_signals",
    "apply_repo_signals_to_authority", "capture_task_start", "compute_repo_signals",
    "compute_task_delta", "enabled", "expand_repo_signals", "fuse_channels",
    "history_scope", "mark_test_links_stale", "mode", "rank_fusion",
    "render_repo_signals", "repo_signals_mode", "snapshot_workspace", "task_delta",
]
