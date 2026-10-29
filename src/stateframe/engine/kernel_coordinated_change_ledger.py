"""v22's retained coordinated-change capability.

The requirements ledger is the storage authority.  ``UnifiedChangeLedger`` is
only an adapter/projection over its ``coordination`` extension; it does not
own a second completion ledger.  Planned artifacts, where-used observations,
compatibility coverage and batch transactions are children or evidence in the
same aggregate and readiness is reduced once from that aggregate.

The batch form is an extension of ``str_replace``.  It supports retained
staging and one publish receipt, while scalar editing remains on its v21 rail.
Capability checks are host truth and do not depend on the guidance switch.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
import threading
from collections.abc import Iterable, Mapping, Sequence
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import kernel_parse_cache as _parse_cache
import kernel_workspace_truth as _workspace_truth
from kernel_measurement import profiled_function_span
import kernel_turn0_persistence as _turn0


LEDGER_ENV = "STATEFRAME_TOKENLAB_COORDINATED_LEDGER"
ATOMIC_APPLY_ENV = "STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY"
GUIDANCE_ENV = "STATEFRAME_TOKENLAB_COORDINATION_GUIDANCE"
SCHEMA = "requirements-ledger-coordination/1"
APPLY_SCHEMA = "stateframe-v22-change-set/2"
IMPORT_SCHEMA = "stateframe-v22-import-failure/1"
COMPAT_SCHEMA = "stateframe-v22-compatibility-check/1"
LEDGER_EVENT = "tokenlab_v22_unified_ledger"
APPLY_EVENT = "tokenlab_v22_change_set"
IMPORT_EVENT = "tokenlab_v22_import_failure"
COMPAT_EVENT = "tokenlab_v22_compatibility_check"

STATUS_PENDING = "pending"
STATUS_APPLIED = "applied"
STATUS_VERIFIED = "verified"
EDIT_PROPOSED = "proposed"
EDIT_PENDING = "pending"
EDIT_STAGED = "staged"
EDIT_APPLIED = "applied"
EDIT_SUPERSEDED = "superseded"
EDIT_STATES = frozenset({EDIT_PROPOSED, EDIT_PENDING, EDIT_STAGED,
                         EDIT_APPLIED, EDIT_SUPERSEDED})
VERIFY_UNCHECKED = "unchecked"
VERIFY_PASS = "pass"
VERIFY_FAIL = "fail"
VERIFY_UNKNOWN = "unknown"
VERIFY_STALE = "stale"
VERIFY_STATES = frozenset({VERIFY_UNCHECKED, VERIFY_PASS, VERIFY_FAIL,
                           VERIFY_UNKNOWN, VERIFY_STALE})
STATUSES = (STATUS_PENDING, STATUS_APPLIED, STATUS_VERIFIED)

SYMBOL_RENAME = "symbol_rename"
SYMBOL_MOVE = "symbol_move"
REFERENCE_UPDATE = "reference_update"
PRESERVE_CALLABLE = "preserve_callable"
PRESERVE_INPUT = "preserve_input"
PLANNED_ARTIFACT = "planned_artifact"
COMPATIBILITY_ALIAS = PRESERVE_CALLABLE
COMPATIBILITY_INPUT = PRESERVE_INPUT
IMPORT_REPAIR = "import_repair"

MAX_RENDER_TOKENS = 320
TARGET_RENDER_TOKENS = 260

_FORBIDDEN_PARTS = frozenset({
    ".git", ".hypothesis", "__pycache__", "backups", "backup", "private",
    "hidden", "hidden_tests", "near_miss", "gold", "oracle",
})
_TEXT_SUFFIXES = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".vue", ".html", ".htm",
    ".css", ".scss", ".json", ".md", ".txt", ".yaml", ".yml", ".go", ".rs",
    ".java", ".php",
})
_SYMBOL = r"[A-Za-z_][A-Za-z0-9_]*"
_MISSING_IMPORT = re.compile(
    r"cannot import name [\"'](?P<symbol>[A-Za-z_]\w*)[\"']\s+from\s+"
    r"[\"'](?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)[\"']",
    re.IGNORECASE,
)


class CoordinatedChangeError(ValueError):
    """Raised for a malformed or unsafe coordination request."""


class BatchValidationError(CoordinatedChangeError):
    """Raised only by callers that request fail-fast validation."""


def _switch(name: str, *, aliases: Sequence[str] = ()) -> str:
    values = [os.environ.get(candidate) for candidate in (name, *aliases)]
    value = next((item for item in values if item is not None), "off")
    normalized = str(value).strip().lower() or "off"
    if normalized not in {"off", "on"}:
        raise ValueError(f"{name} must be off or on")
    return normalized


def ledger_mode(environment: Mapping[str, str] | None = None) -> str:
    values = environment if environment is not None else os.environ
    value = str(values.get(LEDGER_ENV, "off")).strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{LEDGER_ENV} must be off or on")
    return value


def atomic_apply_mode(environment: Mapping[str, str] | None = None) -> str:
    values = environment if environment is not None else os.environ
    value = str(values.get(ATOMIC_APPLY_ENV, "off")).strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{ATOMIC_APPLY_ENV} must be off or on")
    return value


def guidance_mode(environment: Mapping[str, str] | None = None) -> str:
    values = environment if environment is not None else os.environ
    value = str(values.get(GUIDANCE_ENV, "off")).strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{GUIDANCE_ENV} must be off or on")
    return value


def enabled(environment: Mapping[str, str] | None = None) -> bool:
    return ledger_mode(environment) == "on"


def atomic_enabled(environment: Mapping[str, str] | None = None) -> bool:
    # This is a capability, not model guidance.  The v22 vertical gate admits
    # the truth mechanism; the legacy alias is retained for env manifests.
    return ledger_mode(environment) == "on" or atomic_apply_mode(environment) == "on"


def guidance_enabled(environment: Mapping[str, str] | None = None,
                     runtime: object | None = None) -> bool:
    from kernel_model_policy import resolve_model_policy
    if runtime is not None:
        policy = getattr(runtime, "model_policy", None)
        if policy is None:
            policy = resolve_model_policy(getattr(runtime, "model_name", None))
    else:
        policy = resolve_model_policy(environ=environment)
    return policy.guidance_for("coordination")


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return _sha_bytes(value.encode("utf-8"))


def _safe_relative(root: Path, raw: object, *, allow_missing: bool = False,
                   journal: object = None) -> tuple[Path, str]:
    text = str(raw or "").replace("\\", "/").strip()
    if not text or "\x00" in text or Path(text).is_absolute():
        raise CoordinatedChangeError("path is empty")
    raw_path = root / text
    # A symlinked target or parent would let a batch escape after validation.
    check = raw_path
    while check != root and not check.exists():
        check = check.parent
    if check.is_symlink():
        raise CoordinatedChangeError("symlink paths are not admissible")
    candidate = raw_path.resolve()
    root_resolved = root.resolve()
    try:
        relative = candidate.relative_to(root_resolved)
    except ValueError as error:
        raise CoordinatedChangeError("path is outside the workspace") from error
    if any(part in _FORBIDDEN_PARTS or part.startswith(".") for part in relative.parts):
        raise CoordinatedChangeError("path is protected or private")
    if not allow_missing and not candidate.is_file():
        if _workspace_truth.enabled():
            fact = _workspace_truth.exists(root_resolved, relative.as_posix(), journal=journal)
            detail = (
                _workspace_truth.render_fact(fact)
                if fact.status != "exists"
                else f"Path {relative.as_posix()} exists in the live workspace but is not a file."
            )
            raise CoordinatedChangeError(detail)
        raise CoordinatedChangeError(f"file does not exist: {relative.as_posix()}")
    if candidate.is_file() and candidate.stat().st_nlink > 1:
        raise CoordinatedChangeError("hard-linked files are not admissible")
    return candidate, relative.as_posix()


def _allowed_path(path: Path) -> bool:
    return not any(part in _FORBIDDEN_PARTS or part.startswith(".")
                   for part in path.parts)


@dataclass(frozen=True)
class DefinitionSite:
    symbol: str
    path: str
    line: int
    kind: str
    module: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol, "path": self.path, "line": self.line,
            "kind": self.kind, "module": self.module,
        }


@dataclass(frozen=True)
class UsageSite:
    symbol: str
    path: str
    line: int
    kind: str
    context: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol, "path": self.path, "line": self.line,
            "kind": self.kind, "context": self.context,
        }


@dataclass(frozen=True)
class ScanResult:
    definitions: tuple[DefinitionSite, ...]
    importers: Mapping[str, tuple[UsageSite, ...]]
    usages: Mapping[str, tuple[UsageSite, ...]]
    files: tuple[str, ...]

    def defining_file(self, symbol: str, preferred: str = "") -> str:
        matches = [row for row in self.definitions if row.symbol == symbol]
        if preferred and any(row.path == preferred for row in matches):
            return preferred
        return matches[0].path if matches else preferred

    def sites_for(self, symbols: Iterable[str]) -> tuple[UsageSite, ...]:
        wanted = {str(item) for item in symbols if str(item)}
        rows: list[UsageSite] = []
        for symbol in wanted:
            rows.extend(self.usages.get(symbol, ()))
        return _dedupe_sites(rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "files": list(self.files),
            "definitions": [row.to_dict() for row in self.definitions],
            "importers": {
                key: [row.to_dict() for row in rows]
                for key, rows in sorted(self.importers.items())
            },
            "usages": {
                key: [row.to_dict() for row in rows]
                for key, rows in sorted(self.usages.items())
            },
        }


def _dedupe_sites(rows: Iterable[UsageSite]) -> tuple[UsageSite, ...]:
    result: list[UsageSite] = []
    seen: set[tuple[str, str, int, str]] = set()
    for row in rows:
        key = (row.symbol, row.path, row.line, row.kind)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return tuple(sorted(result, key=lambda item: (item.path, item.line, item.symbol, item.kind)))


def _iter_files(root: Path, overlays: Mapping[str, str] | None = None) -> tuple[str, ...]:
    values = set(str(key).replace("\\", "/") for key in (overlays or {}))
    pending: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
    while pending:
        directory, parent_parts = pending.pop()
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                parts = (*parent_parts, entry.name)
                if any(part in _FORBIDDEN_PARTS or part.startswith(".") for part in parts):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append((Path(entry.path), parts))
                    elif (entry.is_file(follow_symlinks=True)
                          and Path(entry.name).suffix.lower() in _TEXT_SUFFIXES):
                        values.add("/".join(parts))
                except OSError:
                    continue
    return tuple(sorted(values))


def _module_for_path(path: str) -> str:
    value = path[:-3] if path.endswith(".py") else path
    if value.endswith("/__init__"):
        value = value[:-9]
    return value.replace("/", ".")


def _source_for(root: Path, relative: str, overlays: Mapping[str, str] | None) -> str | None:
    if overlays and relative in overlays:
        return overlays[relative]
    try:
        return (root / relative).read_text(encoding="utf-8", errors="surrogateescape")
    except (OSError, UnicodeError):
        return None


def scan_repository(
    root: str | Path,
    *,
    overlays: Mapping[str, str] | None = None,
    task_id: str | None = None,
    journal: object | None = None,
) -> ScanResult:
    """Statically locate definitions, importers, and usages without importing code."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise CoordinatedChangeError(f"scan root is not a directory: {root}")
    files = _iter_files(root_path, overlays)
    cache_key = _repository_cache_key(root_path, files, overlays)
    with _REPOSITORY_CACHE_LOCK:
        cached = _SCAN_RESULT_CACHE.get(cache_key)
        if cached is not None:
            _SCAN_RESULT_CACHE.move_to_end(cache_key)
            return _copy_scan_result(cached)

    try:
        persist_context, disk_result = _load_persisted_scan_result(
            root_path, files, overlays=overlays, task_id=task_id, journal=journal)
    except Exception:
        # Persistence must never turn an optional disk hit into a failed scan.
        persist_context, disk_result = None, None
        _turn0.emit_cache_bypass(journal, name="ccl_scan", reason="load_failed")
    if disk_result is not None:
        _remember_scan_result(cache_key, disk_result)
        return _copy_scan_result(disk_result)

    with _parse_cache.gc_paused():  # L112: millions of acyclic sites; see gc_paused
        result = _scan_repository(root_path, overlays=overlays, files=files)
    _remember_scan_result(cache_key, result)
    if persist_context is not None:
        try:
            _save_persisted_scan_result(
                persist_context, files, result, journal=journal, root=root_path)
        except Exception:
            _turn0.emit_cache_bypass(journal, name="ccl_scan", reason="write_failed")
    return result


def _load_persisted_scan_result(
    root: Path,
    files: tuple[str, ...],
    *,
    overlays: Mapping[str, str] | None,
    task_id: str | None,
    journal: object | None,
) -> tuple[_turn0.CacheContext | None, ScanResult | None]:
    if not _turn0.enabled():
        return None, None
    if overlays:
        _turn0.emit_cache_outcome(journal, name="ccl_scan", outcome="miss")
        _turn0.emit_cache_bypass(journal, name="ccl_scan", reason="overlay")
        return None, None
    module_version = _turn0.implementation_version(
        "coordinated-change-scan/1", Path(__file__))
    context, reason = _turn0.make_context(
        root,
        task_id=task_id or "",
        name="ccl_scan",
        module_version=module_version or "",
        walked_paths=files,
    )
    if context is None:
        _turn0.emit_cache_outcome(journal, name="ccl_scan", outcome="miss")
        _turn0.emit_cache_bypass(
            journal, name="ccl_scan", reason=reason or "context_unavailable")
        return None, None
    payload, reason = _turn0.read_payload(context)
    if reason not in {None, "missing"}:
        _turn0.emit_cache_bypass(journal, name="ccl_scan", reason=reason)
    if payload is None or reason is not None:
        _turn0.emit_cache_outcome(journal, name="ccl_scan", outcome="miss")
        return context, None
    try:
        result = _scan_result_from_payload(payload)
    except (TypeError, ValueError, KeyError):
        _turn0.emit_cache_outcome(journal, name="ccl_scan", outcome="miss")
        _turn0.emit_cache_bypass(journal, name="ccl_scan", reason="payload_invalid")
        return context, None
    current_files = _iter_files(root)
    if (result.files != files
            or not _turn0.sources_unchanged(context, current_files)):
        _turn0.emit_cache_outcome(journal, name="ccl_scan", outcome="miss")
        _turn0.emit_cache_bypass(journal, name="ccl_scan", reason="source_changed")
        return context, None
    _turn0.emit_cache_outcome(journal, name="ccl_scan", outcome="hit")
    return context, result


def _save_persisted_scan_result(
    context: _turn0.CacheContext,
    files: tuple[str, ...],
    result: ScanResult,
    *,
    journal: object | None,
    root: Path,
) -> None:
    current_files = _iter_files(root)
    if (result.files != files
            or not _turn0.sources_unchanged(context, current_files)):
        _turn0.emit_cache_bypass(journal, name="ccl_scan", reason="source_changed")
        return
    reason = _turn0.write_payload(context, result.to_dict())
    if reason:
        _turn0.emit_cache_bypass(journal, name="ccl_scan", reason=reason)


def _remember_scan_result(cache_key: tuple[object, ...], result: ScanResult) -> None:
    with _REPOSITORY_CACHE_LOCK:
        for previous_key in tuple(_SCAN_RESULT_CACHE):
            if previous_key[0] == cache_key[0] and previous_key != cache_key:
                del _SCAN_RESULT_CACHE[previous_key]
        _SCAN_RESULT_CACHE[cache_key] = _copy_scan_result(result)
        _SCAN_RESULT_CACHE.move_to_end(cache_key)
        while len(_SCAN_RESULT_CACHE) > _REPOSITORY_CACHE_LIMIT:
            _SCAN_RESULT_CACHE.popitem(last=False)


def _scan_result_from_payload(value: object) -> ScanResult:
    if not isinstance(value, Mapping) or set(value) != {
            "files", "definitions", "importers", "usages"}:
        raise ValueError("scan result payload is malformed")
    raw_files = value.get("files")
    raw_definitions = value.get("definitions")
    raw_importers = value.get("importers")
    raw_usages = value.get("usages")
    if (not isinstance(raw_files, list) or not isinstance(raw_definitions, list)
            or not isinstance(raw_importers, Mapping) or not isinstance(raw_usages, Mapping)):
        raise ValueError("scan result payload has invalid field types")
    if any(not isinstance(path, str) for path in raw_files):
        raise ValueError("scan result file path is malformed")
    files = tuple(raw_files)
    if len(set(files)) != len(files):
        raise ValueError("scan result file list contains duplicates")
    file_set = set(files)

    def definition(row: object) -> DefinitionSite:
        if not isinstance(row, Mapping) or set(row) != {
                "symbol", "path", "line", "kind", "module"}:
            raise ValueError("definition row is malformed")
        if (not isinstance(row["symbol"], str) or not isinstance(row["path"], str)
                or type(row["line"]) is not int or not isinstance(row["kind"], str)
                or not isinstance(row["module"], str) or row["path"] not in file_set):
            raise ValueError("definition row has invalid fields")
        return DefinitionSite(
            row["symbol"], row["path"], row["line"], row["kind"], row["module"])

    def usage(row: object) -> UsageSite:
        if not isinstance(row, Mapping) or set(row) != {
                "symbol", "path", "line", "kind", "context"}:
            raise ValueError("usage row is malformed")
        if (not isinstance(row["symbol"], str) or not isinstance(row["path"], str)
                or type(row["line"]) is not int or not isinstance(row["kind"], str)
                or not isinstance(row["context"], str) or row["path"] not in file_set):
            raise ValueError("usage row has invalid fields")
        return UsageSite(
            row["symbol"], row["path"], row["line"], row["kind"], row["context"])

    definitions = tuple(definition(row) for row in raw_definitions)

    def usage_map(raw: Mapping[str, object]) -> dict[str, tuple[UsageSite, ...]]:
        result: dict[str, tuple[UsageSite, ...]] = {}
        for symbol, rows in raw.items():
            if not isinstance(symbol, str) or not isinstance(rows, list):
                raise ValueError("scan result usage list is malformed")
            result[symbol] = tuple(usage(row) for row in rows)
        return result

    importers = usage_map(raw_importers)
    usages = usage_map(raw_usages)
    return ScanResult(definitions, importers, usages, files)


_REPOSITORY_CACHE_LIMIT = 2
_REPOSITORY_CACHE_LOCK = threading.RLock()
_SCAN_RESULT_CACHE: OrderedDict[tuple[object, ...], ScanResult] = OrderedDict()
_TREE_REVISION_CACHE: OrderedDict[tuple[object, ...], str] = OrderedDict()


def _stat_signature(path: Path) -> tuple[int, int, int, int, int, int] | None:
    try:
        current = path.stat()
    except OSError:
        return None
    return (
        int(current.st_dev), int(current.st_ino), int(current.st_mode),
        int(current.st_size), int(current.st_mtime_ns), int(current.st_ctime_ns),
    )


def _repository_cache_key(root: Path, files: Sequence[str],
                          overlays: Mapping[str, str] | None) -> tuple[object, ...]:
    overlay_paths = set(overlays or {})
    file_signature = tuple(
        (relative, None if relative in overlay_paths else _stat_signature(root / relative))
        for relative in files
    )
    overlay_signature = tuple(sorted(
        (str(relative), _sha_text(str(source))) for relative, source in (overlays or {}).items()
    ))
    return (root.as_posix(), file_signature, overlay_signature)


def _copy_scan_result(result: ScanResult) -> ScanResult:
    return ScanResult(
        result.definitions,
        {key: tuple(rows) for key, rows in result.importers.items()},
        {key: tuple(rows) for key, rows in result.usages.items()},
        result.files,
    )


def _scan_repository(root: str | Path, *, overlays: Mapping[str, str] | None = None,
                     files: tuple[str, ...] | None = None) -> ScanResult:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise CoordinatedChangeError(f"scan root is not a directory: {root}")
    definitions: list[DefinitionSite] = []
    files = files if files is not None else _iter_files(root_path, overlays)

    # L112: each file's contribution is a pure function of (relative, module, source)
    # and is memoized by content, so a post-edit refresh rescans only edited files.
    # Sites are deduplicated and sorted PER FILE: the dedupe key includes the path,
    # so no two files can collide, and ``files`` is sorted like ``item.path``; the
    # concatenation in file order is therefore exactly the former global
    # ``_dedupe_sites`` result, and dict insertion order is unchanged.
    usage_parts: dict[str, list[tuple[UsageSite, ...]]] = {}
    importer_parts: dict[str, list[tuple[UsageSite, ...]]] = {}
    for relative in files:
        source = _source_for(root_path, relative, overlays)
        if source is None:
            continue
        module = _module_for_path(relative)
        try:
            if relative.endswith(".py"):
                scanned = _parse_cache.derived(
                    ("ccl.scan_py", relative, module), source, relative,
                    lambda tree, relative=relative, module=module, source=source:
                        _file_scan(_python_rows(tree, relative, module), source, relative))
            else:
                scanned = _parse_cache.derived_source(
                    ("ccl.scan_text", relative), source, relative,
                    lambda text, relative=relative: _file_scan((), text, relative))
        except SyntaxError:
            continue
        file_definitions, file_importers, file_usages = scanned
        definitions.extend(file_definitions)
        for symbol, sites in file_importers:
            importer_parts.setdefault(symbol, []).append(sites)
        for symbol, sites in file_usages:
            usage_parts.setdefault(symbol, []).append(sites)

    return ScanResult(
        tuple(sorted(definitions, key=lambda item: (item.path, item.line, item.symbol))),
        {key: _concat_sites(parts) for key, parts in importer_parts.items()},
        {key: _concat_sites(parts) for key, parts in usage_parts.items()},
        files,
    )


_SYMBOL_TOKEN = re.compile(rf"\b({_SYMBOL})\b")
_FileScan = tuple[tuple["DefinitionSite", ...],
                  tuple[tuple[str, tuple["UsageSite", ...]], ...],
                  tuple[tuple[str, tuple["UsageSite", ...]], ...]]


def _dedupe_file_sites(sites: list[UsageSite]) -> tuple[UsageSite, ...]:
    # A single site is already deduplicated and sorted (most symbols, per file).
    return (sites[0],) if len(sites) == 1 else _dedupe_sites(sites)


def _concat_sites(parts: list[tuple[UsageSite, ...]]) -> tuple[UsageSite, ...]:
    return parts[0] if len(parts) == 1 else tuple(site for part in parts for site in part)


def _python_rows(tree: ast.AST, relative: str, module: str) -> list[tuple[object, ...]]:
    """One Python file's AST rows, in the exact emission order of the former inline scan."""
    rows: list[tuple[object, ...]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            rows.append(("def", DefinitionSite(
                node.name, relative, int(getattr(node, "lineno", 0)),
                type(node).__name__.removesuffix("Def").lower(), module)))
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    symbol = str(alias.asname or alias.name.rsplit(".", 1)[-1])
                    rows.append(("import", UsageSite(symbol, relative, int(node.lineno), "import", module)))
            else:
                for alias in node.names:
                    symbol = str(alias.asname or alias.name.split(".", 1)[0])
                    rows.append(("import", UsageSite(symbol, relative, int(node.lineno), "import", module)))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    rows.append(("def", DefinitionSite(
                        target.id, relative, int(getattr(node, "lineno", 0)),
                        "alias" if isinstance(node.value, (ast.Name, ast.Attribute)) else "assignment",
                        module)))
    return rows


def _file_scan(rows: Sequence[tuple[object, ...]], source: str, relative: str) -> _FileScan:
    """Group one file's rows (AST rows, then per-line text identifiers) by symbol."""
    definitions: list[DefinitionSite] = []
    usages: dict[str, list[UsageSite]] = {}
    importers: dict[str, list[UsageSite]] = {}
    for kind, site in rows:
        if kind == "def":
            definitions.append(site)
        else:
            usages.setdefault(site.symbol, []).append(site)
            importers.setdefault(site.symbol, []).append(site)
    for line_number, line in enumerate(source.splitlines(), 1):
        context = None
        for match in _SYMBOL_TOKEN.finditer(line):
            if context is None:
                context = line.strip()[:180]
            symbol = match.group(1)
            usages.setdefault(symbol, []).append(UsageSite(symbol, relative, line_number, "text", context))
    return (tuple(definitions),
            tuple((symbol, _dedupe_file_sites(sites)) for symbol, sites in importers.items()),
            tuple((symbol, _dedupe_file_sites(sites)) for symbol, sites in usages.items()))


def _symbol_list(text: str) -> list[str]:
    return [value for value in re.findall(rf"\b{_SYMBOL}\b", text)
            if value.casefold() not in {"and", "or"}]


def _mapping_pair(value: object) -> tuple[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    old = next((str(value.get(key, "")).strip()
                for key in ("old", "old_symbol", "removed", "legacy", "before")
                if str(value.get(key, "")).strip()), "")
    new = next((str(value.get(key, "")).strip()
                for key in ("new", "new_symbol", "added", "canonical", "after")
                if str(value.get(key, "")).strip()), "")
    if re.fullmatch(_SYMBOL, old) and re.fullmatch(_SYMBOL, new) and old != new:
        return old, new
    return None


def _diff_pairs(diff: object) -> list[tuple[str, str, str]]:
    """Read symbol pairs from a public static diff without naming a domain."""
    if not isinstance(diff, str):
        return []
    removed: list[tuple[str, str]] = []
    added: list[tuple[str, str]] = []
    definition = re.compile(
        r"\b(?:async\s+def|def|class)\s+(?P<name>" + _SYMBOL + r")|"
        r"^\s*(?P<assignment>" + _SYMBOL + r")\s*=")
    for line in diff.splitlines():
        marker = line[:1]
        match = definition.search(line[1:] if marker in {"-", "+"} else line)
        if not match:
            continue
        symbol = match.group("name") or match.group("assignment")
        row = (symbol, line.strip())
        if marker == "-" and not line.startswith("---"):
            removed.append(row)
        elif marker == "+" and not line.startswith("+++"):
            added.append(row)
    return [(old[0], new[0], "static_diff")
            for old, new in zip(removed, added) if old[0] != new[0]]


def _plan_pairs(plan: object) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    if isinstance(plan, Mapping):
        containers = [plan.get(key) for key in (
            "changes", "renames", "items", "diff", "static_diff", "patch")]
        for container in containers:
            if isinstance(container, str):
                rows.extend(_diff_pairs(container))
            elif isinstance(container, Mapping):
                pair = _mapping_pair(container)
                if pair:
                    rows.append((*pair, "explicit_plan"))
                else:
                    rows.extend(_plan_pairs(container))
            elif isinstance(container, Sequence) and not isinstance(container, (bytes, bytearray)):
                for item in container:
                    pair = _mapping_pair(item)
                    if pair:
                        rows.append((*pair, "explicit_plan"))
                    elif isinstance(item, str):
                        rows.extend(_diff_pairs(item))
        pair = _mapping_pair(plan)
        if pair:
            rows.append((*pair, "explicit_plan"))
        removed = plan.get("removed")
        added = plan.get("added")
        if (isinstance(removed, Sequence) and isinstance(added, Sequence)
                and not isinstance(removed, (str, bytes, bytearray))
                and not isinstance(added, (str, bytes, bytearray))):
            for old, new in zip(removed, added):
                if re.fullmatch(_SYMBOL, str(old)) and re.fullmatch(_SYMBOL, str(new)):
                    rows.append((str(old), str(new), "static_diff"))
    elif isinstance(plan, Sequence) and not isinstance(plan, (str, bytes, bytearray)):
        for item in plan:
            pair = _mapping_pair(item)
            if pair:
                rows.append((*pair, "explicit_plan"))
            elif isinstance(item, str):
                rows.extend(_diff_pairs(item))
    return rows


def _rename_pairs(ticket: str, plan: object = None, *,
                  grounded_symbols: Iterable[str] = ()) -> list[tuple[str, str, str]]:
    """Return planned pairs and explicit, workspace-grounded ticket renames."""
    rows: list[tuple[str, str, str]] = []
    rows.extend(_plan_pairs(plan))

    # Public prose is not rename authority. Ticket text must quote both symbols
    # and each symbol must be an actual code definition in this workspace.
    grounded = {str(symbol) for symbol in grounded_symbols}
    quoted_symbol = rf'''(?:`({_SYMBOL})`|'({_SYMBOL})'|"({_SYMBOL})")'''
    patterns = (
        re.compile(
            rf"\b(?:rename|renamed|change)\s+{quoted_symbol}\s*(?:to|->)\s*{quoted_symbol}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"{quoted_symbol}\s*(?:is\s+now|becomes|->|→)\s*{quoted_symbol}",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(ticket):
            groups = match.groups()
            old = next(value for value in groups[:3] if value is not None)
            new = next(value for value in groups[3:6] if value is not None)
            # Ground the OLD name only: a genuine rename's new name usually does not exist yet.
            if old in grounded and new != old:
                rows.append((old, new, "ticket_text"))
    # Explicit grouped form ("new names are a, b (old: x, y)") stays supported, under the same grounding.
    grouped = re.search(
        r"new\s+names?\s+are\s+(?P<new>[^.\n]+?)\s*\(\s*old\s*:\s*(?P<old>[^)]+)\)",
        ticket, re.IGNORECASE)
    if grouped:
        new_names = _symbol_list(grouped.group("new"))
        old_names = _symbol_list(grouped.group("old"))
        if len(new_names) == len(old_names):
            rows.extend((old, new, "ticket_name_pair") for old, new in zip(old_names, new_names)
                        if old in grounded and new != old)

    result: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for old, new, source in rows:
        key = (old, new)
        if old != new and key not in seen:
            seen.add(key)
            result.append((old, new, source))
    return result


def _word_pairs(ticket: str) -> dict[str, str]:
    patterns = (
        re.compile(rf"\b(?:word|term|concept)\s+[`'\"]?({_SYMBOL})[`'\"]?\s*"
                   rf"(?:to|is\s+now|becomes|→)\s*[`'\"]?({_SYMBOL})[`'\"]?", re.I),
        re.compile(rf"[`'\"]?({_SYMBOL})[`'\"]?\s*(?:is\s+now|becomes|→)\s*"
                   rf"[`'\"]?({_SYMBOL})[`'\"]?", re.I),
    )
    result: dict[str, str] = {}
    for pattern in patterns:
        for match in pattern.finditer(ticket):
            old, new = match.group(1), match.group(2)
            if old != new:
                result.setdefault(old, new)
    return result


def _fixture_pair(key: object, fixture: object) -> tuple[str, str] | None:
    text = str(key or "")
    match = re.search(r"(?:^|:)\s*([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)\s*$", text)
    if match:
        return match.group(1), match.group(2)
    if isinstance(fixture, Mapping):
        relation = fixture.get("relation")
        for candidate in (fixture, relation):
            pair = _mapping_pair(candidate)
            if pair:
                return pair
            if isinstance(candidate, Mapping):
                old = next((str(candidate.get(name, "")).strip() for name in (
                    "old_field", "legacy_field", "input_field")
                            if str(candidate.get(name, "")).strip()), "")
                new = next((str(candidate.get(name, "")).strip() for name in (
                    "new_field", "canonical_field", "output_field")
                            if str(candidate.get(name, "")).strip()), "")
                if re.fullmatch(_SYMBOL, old) and re.fullmatch(_SYMBOL, new) and old != new:
                    return old, new
    return None


def _fixture_for(fixture_map: Mapping[str, object] | None, *, item_id: str,
                 old: str, new: str, kind: str) -> Mapping[str, object] | None:
    if not isinstance(fixture_map, Mapping):
        return None
    direct = fixture_map.get(item_id)
    if isinstance(direct, Mapping):
        return direct
    candidates: list[Mapping[str, object]] = []
    for key, value in fixture_map.items():
        if not isinstance(value, Mapping):
            continue
        pair = _fixture_pair(key, value)
        if pair == (old, new):
            return value
        fixture_kind = str(value.get("kind", "")).casefold()
        if (fixture_kind == kind.casefold() or
                (kind.casefold() != "input" and fixture_kind != "input")):
            candidates.append(value)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _input_pairs(ticket: str, pairs: Sequence[tuple[str, str, str]],
                 fixture_map: Mapping[str, object] | None = None) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    mentions = [match.group(1) for match in re.finditer(
        rf"\b(?:still\s+send|send|inputs?\s+that\s+still\s+send)\s+[`'\"]?({_SYMBOL})",
        ticket, re.IGNORECASE)]
    words = _word_pairs(ticket)
    for old, new, provenance in pairs:
        if old in mentions:
            rows.append((old, new, "ticket_input"))
    for old in mentions:
        if any(row[0] == old for row in rows):
            continue
        prefix, separator, suffix = old.rpartition("_")
        new_prefix = words.get(prefix, "")
        if separator and new_prefix:
            rows.append((old, f"{new_prefix}_{suffix}", "ticket_input"))
    if isinstance(fixture_map, Mapping):
        for key, fixture in fixture_map.items():
            pair = _fixture_pair(key, fixture)
            if pair and not any(row[:2] == pair for row in rows):
                rows.append((*pair, "public_fixture"))
    result: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for old, new, provenance in rows:
        key = (old, new)
        if key not in seen and old != new:
            seen.add(key)
            result.append((old, new, provenance))
    return result


def _ticket_requires_compatibility(ticket: str) -> bool:
    return bool(re.search(
        r"\b(?:compatib|alias|old\s+(?:names?|callers?)|still\s+send|keep\s+\w+\s+working|during\s+the\s+migration)\w*",
        ticket, re.IGNORECASE))


def _path_for_symbol(scan: ScanResult, symbol: str, preferred: str = "") -> str:
    return scan.defining_file(symbol, preferred)


def _module_from_file(path: str) -> str:
    return _module_for_path(path)


def _fixture_symbol(fixture: Mapping[str, object] | None) -> str:
    if not isinstance(fixture, Mapping):
        return ""
    value = next((fixture.get(key) for key in ("symbol", "target", "entry_point")
                  if isinstance(fixture.get(key), str) and fixture.get(key)), "")
    text = str(value)
    return text.rsplit(":", 1)[-1].rsplit(".", 1)[-1] if text else ""


def _input_target(scan: ScanResult, pairs: Sequence[tuple[str, str, str]],
                  field: str, fixture: Mapping[str, object] | None,
                  legacy_field: str = "") -> str:
    explicit = _fixture_symbol(fixture)
    if explicit and any(row.symbol == explicit for row in scan.definitions):
        return explicit
    excluded = {field, legacy_field, *(old for old, _, _ in pairs)}
    definitions = [row for row in scan.definitions
                   if row.kind in {"function", "asyncfunction"}
                   and row.symbol not in excluded and not row.symbol.startswith("__")]
    scored: list[tuple[int, int, str]] = []
    for definition in definitions:
        score = sum(row.path == definition.path for row in scan.usages.get(field, ()))
        if any(new == definition.symbol for _, new, _ in pairs):
            score += 2
        if score:
            scored.append((score, definition.line, definition.symbol))
    if scored:
        return max(scored)[2]
    return next((new for _, new, _ in pairs
                 if any(row.symbol == new for row in definitions)), "")


def _compat_call(old: str, new: str, *, path: str, input_field: str = "",
                 target_symbol: str = "", fixture: Mapping[str, object] | None = None,
                 baseline_symbol: str = "") -> dict[str, object]:
    kind = str(fixture.get("kind", "")) if isinstance(fixture, Mapping) else ""
    if not kind:
        kind = "class" if old[:1].isupper() else "function"
    relation: dict[str, object] = {
        "kind": "observable_equivalence",
        "legacy": old,
        "canonical": new,
    }
    if input_field:
        symbol = target_symbol or new
        relation["input"] = {"legacy": input_field, "canonical": new}
        return {
            "kind": "input",
            "module": _module_from_file(path), "symbol": symbol,
            "call": f"{symbol}(<host fixture containing {input_field}>)",
            "baseline_symbol": baseline_symbol, "fixture_required": True,
            "relation": relation,
        }
    return {
        "kind": kind, "module": _module_from_file(path), "symbol": old,
        "call": f"{old}(<host fixture>)", "fixture_required": True,
        "relation": relation,
    }


def _import_owner_witnesses(sites: Sequence[UsageSite], defining_module: str) -> list[dict[str, object]]:
    """Describe real modules that import the changed symbol for later execution."""
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in sites:
        if row.kind != "import":
            continue
        module = str(row.context or _module_from_file(row.path)).strip()
        if not module or module == defining_module or module in seen:
            continue
        seen.add(module)
        result.append({"module": module, "path": row.path, "line": row.line,
                       "imported_symbol": row.symbol})
    return result


@dataclass
class ChangeItem:
    item_id: str
    kind: str
    old: str
    new: str
    defining_file: str
    importers: tuple[UsageSite, ...] = ()
    usage_sites: tuple[UsageSite, ...] = ()
    requirement_ids: tuple[str, ...] = ()
    planned_artifact_ids: tuple[str, ...] = ()
    compatibility: dict[str, object] | None = None
    status: str = STATUS_PENDING
    revision: str = ""
    applied_revision: str | None = None
    verified_revision: str | None = None
    status_reason: str = "waiting for host observation"
    checkable: bool = False
    stale: bool = False
    edit_progress: str = EDIT_PROPOSED
    verification: str = VERIFY_UNCHECKED
    producers: tuple[str, ...] = ()
    consumers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise CoordinatedChangeError(f"unknown coordination status: {self.status}")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.item_id, "kind": self.kind, "old": self.old, "new": self.new,
            "defining_file": self.defining_file,
            "importers": [row.to_dict() for row in self.importers],
            "usage_sites": [row.to_dict() for row in self.usage_sites],
            "requirement_ids": list(self.requirement_ids),
            "planned_artifact_ids": list(self.planned_artifact_ids),
            "compatibility": copy.deepcopy(self.compatibility), "status": self.status,
            "revision": self.revision, "applied_revision": self.applied_revision,
            "verified_revision": self.verified_revision, "status_reason": self.status_reason,
            "checkable": self.checkable, "stale": self.stale,
            "edit_progress": self.edit_progress, "verification": self.verification,
            "producers": list(self.producers), "consumers": list(self.consumers),
        }


class CoordinatedChangeLedger:
    """One typed coordination view bridged to v21 host-owned ledgers."""

    def __init__(self, *, root: str | Path, ticket: str, scan: ScanResult,
                 items: Sequence[ChangeItem], revision: str = "",
                 requirements_ledger: Mapping[str, object] | None = None,
                 planned_artifacts: object = None, obligation_ledger: object = None,
                 journal: object | None = None, task_id: str = ""):
        self.root = Path(root).resolve()
        self.task_id = task_id if isinstance(task_id, str) else ""
        self.ticket = ticket
        self.ticket_digest = _sha_text(ticket)
        self.scan = scan
        self.items: list[ChangeItem] = list(items)
        self.revision = revision or tree_revision(self.root)
        # Keep the host mapping itself as storage authority.  Deep copies are
        # limited to receipts and rendered snapshots below.
        self.requirements_ledger = (requirements_ledger
                                    if isinstance(requirements_ledger, dict)
                                    else dict(requirements_ledger)
                                    if isinstance(requirements_ledger, Mapping) else None)
        if self.requirements_ledger is None:
            self.requirements_ledger = {
                "schema_version": "requirements-ledger/1",
                "requirements": [], "obligations": [],
            }
        coordination = self.requirements_ledger.setdefault("coordination", {})
        if not isinstance(coordination, dict):
            raise CoordinatedChangeError("requirements ledger coordination must be a mapping")
        coordination.setdefault("schema_version", SCHEMA)
        coordination.setdefault("work_children", [])
        coordination.setdefault("change_sets", [])
        coordination.setdefault("decision_basis", {
            "published_revision": self.revision, "stage_ref": None,
            "evidence_watermark": 0, "holds": [], "next_action": None,
        })
        self.planned_artifacts = planned_artifacts
        self.obligation_ledger = obligation_ledger
        self.journal = journal
        self.broken: list[str] = []
        self.last_batch_id: str | None = None
        self._rendered_revision: str | None = None

    @classmethod
    def from_ticket(cls, root: str | Path, ticket: str, *, plan: object = None,
                    compatibility_fixtures: Mapping[str, object] | None = None,
                    requirements_ledger: Mapping[str, object] | None = None,
                    planned_artifacts: object = None, obligation_ledger: object = None,
                    journal: object | None = None,
                    revision: str = "", task_id: str = "") -> "CoordinatedChangeLedger":
        if not isinstance(ticket, str) or not ticket.strip():
            raise CoordinatedChangeError("ticket text is required")
        scan = scan_repository(root, task_id=task_id, journal=journal)
        grounded_symbols = {definition.symbol for definition in scan.definitions}
        pairs = _rename_pairs(ticket, plan, grounded_symbols=grounded_symbols)
        input_pairs = _input_pairs(ticket, pairs, compatibility_fixtures)
        compatibility = _ticket_requires_compatibility(ticket) or bool(input_pairs)
        req_ids = _requirement_ids(requirements_ledger)
        planned_ids = _planned_ids(planned_artifacts)
        items: list[ChangeItem] = []
        used: set[str] = set()

        def add(item: ChangeItem) -> None:
            if item.item_id not in used:
                used.add(item.item_id)
                items.append(item)

        for old, new, provenance in pairs:
            path = _path_for_symbol(scan, new) or _path_for_symbol(scan, old)
            sites = scan.sites_for((old, new))
            imports = tuple(row for row in sites if row.kind == "import")
            related_req = tuple(item for item in req_ids
                                if _requirement_mentions(requirements_ledger, item, old, new))
            item_id = f"rename:{old}->{new}"
            add(ChangeItem(item_id, SYMBOL_RENAME, old, new, path,
                           imports, sites, related_req, planned_ids,
                           status_reason=f"proposed by {provenance}"))
            if compatibility:
                alias_id = f"alias:{old}->{new}"
                alias_path = _path_for_symbol(scan, old) or path
                fixture = _fixture_for(compatibility_fixtures, item_id=alias_id,
                                       old=old, new=new, kind="function")
                check = _compat_call(old, new, path=alias_path, fixture=fixture)
                check["import_owners"] = _import_owner_witnesses(
                    imports, str(check.get("module", "")))
                if fixture is not None:
                    check["fixture"] = copy.deepcopy(dict(fixture))
                add(ChangeItem(
                    alias_id, COMPATIBILITY_ALIAS, old, new, alias_path,
                    tuple(row for row in sites if row.kind == "import"), sites,
                    related_req, planned_ids, compatibility=check, checkable=bool(alias_path),
                    status_reason="ticket requires old callers/imports to keep working"))
        for old, new, provenance in input_pairs:
            fixture = _fixture_for(compatibility_fixtures,
                                   item_id=f"input:{old}->{new}", old=old, new=new,
                                   kind="input")
            target_symbol = _input_target(scan, pairs, new, fixture, old)
            baseline_symbol = next((pair_old for pair_old, pair_new, _ in pairs
                                    if pair_new == target_symbol), "")
            path = _path_for_symbol(scan, target_symbol)
            sites = scan.sites_for((old, new))
            related_req = tuple(item for item in req_ids
                                if _requirement_mentions(requirements_ledger, item, old, new))
            check = _compat_call(old, new, path=path, input_field=old,
                                 target_symbol=target_symbol, fixture=fixture,
                                 baseline_symbol=baseline_symbol)
            check["import_owners"] = _import_owner_witnesses(
                tuple(row for row in sites if row.kind == "import"),
                str(check.get("module", "")))
            if fixture is not None:
                check["fixture"] = copy.deepcopy(dict(fixture))
            add(ChangeItem(
                f"input:{old}->{new}", COMPATIBILITY_INPUT, old, new, path,
                tuple(row for row in sites if row.kind == "import"), sites,
                related_req, planned_ids, compatibility=check, checkable=bool(path),
                status_reason=f"{provenance} requires the legacy input to remain equivalent"))
        return cls(root=root, ticket=ticket, scan=scan, items=items, revision=revision,
                   requirements_ledger=requirements_ledger,
                   planned_artifacts=planned_artifacts, obligation_ledger=obligation_ledger,
                   journal=journal, task_id=task_id)

    def item(self, item_id: str) -> ChangeItem:
        for item in self.items:
            if item.item_id == item_id:
                return item
        raise KeyError(item_id)

    def current_item(self) -> ChangeItem | None:
        return next((item for item in self.items if item.status != STATUS_VERIFIED), None)

    def remaining(self) -> tuple[ChangeItem, ...]:
        return tuple(item for item in self.items if item.status != STATUS_VERIFIED)

    def refresh_scan(self, *, revision: str = "") -> None:
        current = revision or tree_revision(self.root)
        if current != self.revision:
            for item in self.items:
                if item.status != STATUS_PENDING:
                    item.status = STATUS_PENDING
                    item.stale = True
                    item.applied_revision = None
                    item.verified_revision = None
                    item.status_reason = "revision changed; host evidence must be re-run"
                    item.edit_progress = EDIT_PENDING
                    item.verification = VERIFY_STALE
            self.revision = current
        self.scan = scan_repository(
            self.root, task_id=self.task_id, journal=self.journal)

    def record_batch(self, result: "BatchResult") -> None:
        if result.status != "APPLIED":
            return
        self.last_batch_id = result.batch_id
        touched = set(result.changed_paths)
        for item in self.items:
            related = {item.defining_file, *(row.path for row in item.importers),
                       *(row.path for row in item.usage_sites)} - {""}
            if touched.intersection(related):
                item.status = STATUS_APPLIED
                item.edit_progress = EDIT_APPLIED
                item.verification = VERIFY_STALE
                item.stale = False
                item.applied_revision = result.revision
                item.status_reason = f"host applied batch {result.batch_id}"
        self.revision = result.revision
        self.refresh_scan(revision=result.revision)
        self._sync_coordination(stage_ref=result.batch_id, published_revision=result.revision)
        for row in self.requirements_ledger["coordination"].get("change_sets", ()):
            if isinstance(row, dict) and row.get("id") == result.batch_id:
                row.update({"status": "applied", "revision": result.revision,
                            "changed_paths": list(result.changed_paths)})
        self._emit(LEDGER_EVENT, action="batch_applied", batch_id=result.batch_id,
                   item_statuses=self._status_counts(), revision=self.revision)

    def record_stage(self, result: "BatchResult") -> None:
        if result.status != "STAGED":
            return
        touched = set(result.changed_paths)
        for item in self.items:
            related = {item.defining_file, *(row.path for row in item.importers),
                       *(row.path for row in item.usage_sites)} - {""}
            if touched.intersection(related):
                item.edit_progress = EDIT_STAGED
                item.verification = VERIFY_STALE
                item.status_reason = f"retained staged generation {result.batch_id}"
        coordination = self.requirements_ledger["coordination"]
        coordination["change_sets"] = [
            row for row in coordination.get("change_sets", [])
            if not (isinstance(row, Mapping) and row.get("id") == result.batch_id)
        ]
        coordination["change_sets"].append({
            "id": result.batch_id, "revision": result.stage_revision or result.revision,
            "changed_paths": list(result.changed_paths), "status": "staged",
        })
        self._sync_coordination(stage_ref=result.batch_id,
                                published_revision=result.stage_revision or result.revision)
        self._emit(LEDGER_EVENT, action="batch_staged", batch_id=result.batch_id,
                   item_statuses=self._status_counts(), revision=self.revision)

    def record_verification(self, item_id: str, *, passed: bool, evidence: object = None,
                            revision: str = "") -> ChangeItem:
        item = self.item(item_id)
        current = revision or tree_revision(self.root)
        if passed and item.checkable:
            item.status = STATUS_VERIFIED
            item.edit_progress = EDIT_APPLIED
            item.verification = VERIFY_PASS
            item.verified_revision = current
            item.status_reason = "host compatibility check passed"
        else:
            item.status = STATUS_PENDING if item.stale else STATUS_APPLIED
            item.verification = VERIFY_FAIL if passed is False else VERIFY_UNKNOWN
            item.status_reason = "host check did not pass or is not checkable"
        item.stale = False
        self.revision = current
        self._sync_coordination(stage_ref=self.last_batch_id, published_revision=current)
        self._emit(COMPAT_EVENT, item_id=item.item_id, checkable=item.checkable,
                   passed=bool(passed), evidence=copy.deepcopy(evidence), revision=current)
        return item

    def record_import_failure(self, failure: Mapping[str, object] | str) -> ChangeItem:
        row = _failure_parts(failure)
        symbol = row["symbol"]
        importer = row["importer"]
        defining = row["defining_file"]
        item_id = f"import:{importer}:{symbol}"
        existing = next((item for item in self.items if item.item_id == item_id), None)
        if existing is None:
            existing = ChangeItem(
                item_id, IMPORT_REPAIR, symbol, symbol, defining,
                usage_sites=(UsageSite(symbol, importer, 0, "import", row["message"]),),
                status_reason="host smoke found a missing import", checkable=False)
            self.items.append(existing)
        existing.status = STATUS_PENDING
        existing.status_reason = row["message"]
        message = row["message"]
        if message not in self.broken:
            self.broken.append(message)
        self._emit(IMPORT_EVENT, missing_symbol=symbol, defining_file=defining,
                   importer_file=importer, message=message)
        return existing

    def checkable_compatibility(self) -> tuple[ChangeItem, ...]:
        return tuple(item for item in self.items if item.kind in {
            COMPATIBILITY_ALIAS, COMPATIBILITY_INPUT} and item.checkable)

    def coordination_evidence(self) -> dict[str, object]:
        """Return holds for the host readiness authority; do not reduce readiness here."""
        holds: list[str] = []
        for item in self.items:
            if item.edit_progress != EDIT_APPLIED:
                holds.append(f"{item.item_id}:edit:{item.edit_progress}")
            if item.verification != VERIFY_PASS:
                holds.append(f"{item.item_id}:verify:{item.verification}")
        reason = ("coordination scope is empty; task completion is assessed separately"
                  if not self.items else
                  "all tracked coordination obligations applied and verified" if not holds else holds[0])
        return {"holds": holds[:8],
                "reason": reason, "revision": self.revision,
                "next_action": None if not holds else "apply or verify the first open child"}

    def _sync_coordination(self, *, stage_ref: str | None,
                           published_revision: str) -> None:
        coordination = self.requirements_ledger["coordination"]
        coordination["work_children"] = [item.to_dict() for item in self.items]
        if stage_ref:
            changes = coordination.setdefault("change_sets", [])
            if not any(isinstance(row, Mapping) and row.get("id") == stage_ref for row in changes):
                changes.append({"id": stage_ref, "revision": published_revision,
                                "changed_paths": [], "status": "applied"})
        basis = coordination.setdefault("decision_basis", {})
        basis["published_revision"] = published_revision
        basis["stage_ref"] = stage_ref
        evidence = self.coordination_evidence()
        basis["holds"] = list(evidence["holds"])
        basis["next_action"] = evidence["next_action"]

    def _status_counts(self) -> dict[str, int]:
        return {status: sum(item.status == status for item in self.items) for status in STATUSES}

    def _emit(self, event: str, **fields: object) -> None:
        _emit(self.journal, event, **fields)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA, "ticket_digest": self.ticket_digest,
            "revision": self.revision, "items": [item.to_dict() for item in self.items],
            "current_item": self.current_item().item_id if self.current_item() else None,
            "remaining": [item.item_id for item in self.remaining()],
            "broken": list(self.broken),
            "requirement_ids": sorted({ref for item in self.items for ref in item.requirement_ids}),
            "planned_artifact_ids": sorted({ref for item in self.items for ref in item.planned_artifact_ids}),
            "obligation_ids": list(_obligation_ids(self.obligation_ledger)),
            "requirements_ledger": copy.deepcopy(self.requirements_ledger),
            "coordination_evidence": self.coordination_evidence(),
        }

    def render(self, *, max_tokens: int = MAX_RENDER_TOKENS) -> str:
        current = self.current_item()
        withheld = int(getattr(self, "_render_withheld_count", 0) or 0)
        if current is None and not self.items and withheld:
            current_line = "CURRENT: coordination entries withheld by product-path scope"
        elif current is None and not self.items:
            current_line = "CURRENT: coordination scope empty"
        elif current is None:
            current_line = "CURRENT: all tracked coordination items applied and verified"
        else:
            transition = f"{current.old}->{current.new}" if current.old != current.new else current.old
            current_line = (f"CURRENT: {current.item_id} {current.kind} {transition}"
                            f" @ {current.defining_file or 'unlocated'} [{current.status}]")
        remaining = self.remaining()
        groups: dict[str, list[str]] = {}
        for item in remaining:
            groups.setdefault(item.kind, []).append(item.item_id)
        grouped = "; ".join(
            f"{kind}={','.join(ids[:4])}" for kind, ids in groups.items()) or "none"
        strongest = "failure detail suppressed by product-path scope" if (
            withheld or getattr(self, "_render_failure_suppressed", False)
        ) else (self.broken[-1] if self.broken else "none observed")
        evidence = self.coordination_evidence()
        readiness_holds = bool(evidence["holds"]) or withheld > 0
        readiness_reason = (
            "one or more coordination entries were withheld by product-path scope"
            if withheld else evidence["reason"])
        next_action = (
            "review withheld coordination entries through the host ledger"
            if withheld else evidence["next_action"]
            or "no coordination action pending; task completion authority is separate")
        withheld_line = f"\nWITHHELD: {withheld} out-of-scope entries" if withheld else ""
        text = ("[COORDINATED CHANGE LEDGER]\n" + current_line +
                f"\nREMAINING: {len(remaining)} grouped {grouped}" + withheld_line +
                f"\nSTRONGEST FAILURE: {strongest}\n"
                f"READINESS: evidence {'HOLD' if readiness_holds else 'CLEAR'} {readiness_reason}\n"
                f"NEXT: {next_action}")
        words = text.split()
        return text if len(words) <= max_tokens else " ".join(words[:max_tokens]) + " ... [bounded]"


# Public name describes the ownership boundary. The legacy
# spelling remains an import-compatible adapter for the v22 probe harness.
UnifiedChangeLedger = CoordinatedChangeLedger


def _requirement_ids(ledger: Mapping[str, object] | None) -> tuple[str, ...]:
    if not isinstance(ledger, Mapping):
        return ()
    return tuple(str(row.get("id")) for row in ledger.get("requirements", ())
                 if isinstance(row, Mapping) and row.get("id"))


def _requirement_mentions(ledger: Mapping[str, object] | None, ident: str,
                          old: str, new: str) -> bool:
    if not isinstance(ledger, Mapping):
        return False
    for row in ledger.get("requirements", ()):
        if isinstance(row, Mapping) and str(row.get("id")) == ident:
            text = str(row.get("exact_clause", ""))
            return old in text or new in text
    return False


def _planned_ids(ledger: object) -> tuple[str, ...]:
    if ledger is None:
        return ()
    records = getattr(ledger, "records", None)
    if records is None:
        records = getattr(ledger, "artifacts", None)
    if isinstance(records, Mapping):
        return tuple(sorted(str(key) for key in records))
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
        result = []
        for row in records:
            value = row.get("id") if isinstance(row, Mapping) else getattr(row, "id", None)
            if value:
                result.append(str(value))
        return tuple(sorted(result))
    return ()


def _obligation_ids(ledger: object) -> tuple[str, ...]:
    if ledger is None:
        return ()
    if isinstance(ledger, Mapping):
        rows = ledger.get("obligations", ledger.get("requirements", ()))
    else:
        rows = getattr(ledger, "obligations", ())
    if isinstance(rows, Mapping):
        rows = rows.values()
    if not isinstance(rows, Sequence) and not isinstance(rows, Iterable):
        return ()
    result = []
    for row in rows:
        value = row.get("id") if isinstance(row, Mapping) else getattr(row, "id", None)
        if value:
            result.append(str(value))
    return tuple(sorted(set(result)))


def tree_revision(root: str | Path) -> str:
    root_path = Path(root).resolve()
    files = _iter_files(root_path)
    cache_key = _repository_cache_key(root_path, files, None)
    with _REPOSITORY_CACHE_LOCK:
        cached = _TREE_REVISION_CACHE.get(cache_key)
        if cached is not None:
            _TREE_REVISION_CACHE.move_to_end(cache_key)
            return cached
    revision = _compute_tree_revision(root_path, files)
    with _REPOSITORY_CACHE_LOCK:
        for previous_key in tuple(_TREE_REVISION_CACHE):
            if previous_key[0] == cache_key[0] and previous_key != cache_key:
                del _TREE_REVISION_CACHE[previous_key]
        _TREE_REVISION_CACHE[cache_key] = revision
        _TREE_REVISION_CACHE.move_to_end(cache_key)
        while len(_TREE_REVISION_CACHE) > _REPOSITORY_CACHE_LIMIT:
            _TREE_REVISION_CACHE.popitem(last=False)
    return revision


def _compute_tree_revision(root_path: Path, files: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        path = root_path / relative
        try:
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
        except OSError:
            continue
    return "sha256:" + digest.hexdigest()


def _emit(journal: object | None, event: str, **fields: object) -> None:
    emit = getattr(journal, "emit", None)
    if callable(emit):
        schemas = {
            LEDGER_EVENT: SCHEMA,
            APPLY_EVENT: APPLY_SCHEMA,
            IMPORT_EVENT: IMPORT_SCHEMA,
            COMPAT_EVENT: COMPAT_SCHEMA,
        }
        emit(event, producer="kernel_coordinated_change_ledger",
             event_schema=schemas.get(event, APPLY_SCHEMA),
             dispatch=False, **fields)


def _failure_parts(failure: Mapping[str, object] | str) -> dict[str, str]:
    if isinstance(failure, Mapping):
        symbol = str(failure.get("missing_symbol", failure.get("symbol", ""))).strip()
        importer = str(failure.get("importer_file", failure.get("importer", ""))).strip()
        defining = str(failure.get("defining_file", "")).strip()
        message = str(failure.get("message", "")).strip()
        if not message:
            message = f"missing {symbol}; define {defining}, imported by {importer}"
        return {"symbol": symbol, "importer": importer, "defining_file": defining,
                "message": message}
    text = str(failure)
    match = _MISSING_IMPORT.search(text)
    if match:
        symbol = match.group("symbol")
        module = match.group("module")
        return {
            "symbol": symbol, "importer": "unknown importer",
            "defining_file": module.replace(".", "/") + ".py",
            "message": text,
        }
    return {"symbol": "unknown", "importer": "unknown importer",
            "defining_file": "unknown file", "message": text}


@dataclass(frozen=True)
class BatchEdit:
    path: str
    old: str
    new: str
    expected_revision: str = ""
    operation: str = "replace"
    target_ref: Mapping[str, object] | None = None
    destination_ref: Mapping[str, object] | None = None
    order: int | None = None
    dependency_digest: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "BatchEdit":
        operation = str(value.get("op", value.get("operation", "replace")) or "replace")
        raw_target = value.get("target_ref", value.get("site_ref"))
        target = dict(raw_target) if isinstance(raw_target, Mapping) else None
        path = str(value.get("path", "") or (target or {}).get("path", "")).strip()
        old = value.get("old", value.get("old_str"))
        new = value.get("new", value.get("new_str"))
        if not path or (operation == "replace" and raw_target is None and not isinstance(old, str)):
            raise CoordinatedChangeError("each batch item needs path and typed target or old/new strings")
        if not isinstance(new, str):
            new = ""
        raw_destination = value.get("destination_ref", value.get("destination"))
        destination = dict(raw_destination) if isinstance(raw_destination, Mapping) else None
        order = value.get("order")
        if order is not None:
            try:
                order = int(order)
            except (TypeError, ValueError) as error:
                raise CoordinatedChangeError("batch order must be an integer") from error
        return cls(path, old if isinstance(old, str) else "", new,
                   str(value.get("expected_revision", "") or ""), operation,
                   target, destination, order,
                   str(value.get("dependency_digest", "") or ""))

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"path": self.path, "old": self.old, "new": self.new,
                                    "expected_revision": self.expected_revision,
                                    "op": self.operation}
        if self.target_ref is not None:
            value["target_ref"] = dict(self.target_ref)
        if self.destination_ref is not None:
            value["destination_ref"] = dict(self.destination_ref)
        if self.order is not None:
            value["order"] = self.order
        if self.dependency_digest:
            value["dependency_digest"] = self.dependency_digest
        return value


@dataclass(frozen=True)
class BatchFailure:
    item_index: int
    path: str
    code: str
    message: str
    missing_symbol: str = ""
    defining_file: str = ""
    importer_file: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "item_index": self.item_index, "path": self.path, "code": self.code,
            "message": self.message, "missing_symbol": self.missing_symbol,
            "defining_file": self.defining_file, "importer_file": self.importer_file,
        }


@dataclass(frozen=True)
class BatchValidation:
    valid: bool
    candidates: Mapping[str, str] = field(default_factory=dict)
    original: Mapping[str, bytes] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    failures: tuple[BatchFailure, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid, "paths": sorted(self.candidates),
            "warnings": list(self.warnings), "failures": [row.to_dict() for row in self.failures],
        }


@dataclass(frozen=True)
class BatchResult:
    status: str
    batch_id: str
    revision: str
    changed_paths: tuple[str, ...] = ()
    item_statuses: tuple[str, ...] = ()
    failures: tuple[BatchFailure, ...] = ()
    warnings: tuple[str, ...] = ()
    stage_revision: str | None = None
    change_set_ref: str | None = None
    recovery_required: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": APPLY_SCHEMA, "status": self.status, "batch_id": self.batch_id,
            "revision": self.revision, "changed_paths": list(self.changed_paths),
            "item_statuses": list(self.item_statuses),
            "failures": [row.to_dict() for row in self.failures],
            "warnings": list(self.warnings),
            "stage_revision": self.stage_revision, "change_set_ref": self.change_set_ref,
            "recovery_required": self.recovery_required,
        }


_STAGED_BATCHES: dict[str, dict[str, object]] = {}


def _stage_directory(root: Path, batch_id: str) -> Path:
    digest = _digest({"root": str(root), "batch": batch_id})[:24]
    return Path(tempfile.gettempdir()) / "stateframe-v22-stages" / digest


def _persist_stage(root: Path, batch_id: str, edits: Sequence[object],
                   validation: BatchValidation, stage_revision: str) -> None:
    generation = _stage_directory(root, batch_id)
    generation.mkdir(parents=True, exist_ok=True)
    candidates_root = generation / "candidate"
    for relative, source in validation.candidates.items():
        target = candidates_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8", errors="surrogateescape")
    manifest = {
        "schema": APPLY_SCHEMA, "batch_id": batch_id, "root": str(root),
        "stage_revision": stage_revision,
        "edits": [item.to_dict() for item in _coerce_edits(edits)],
        "changed_paths": sorted(validation.candidates),
        "original_digests": {path: _sha_bytes(value)
                             for path, value in validation.original.items()},
    }
    temporary = generation / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False),
                         encoding="utf-8")
    os.replace(temporary, generation / "manifest.json")


def _load_staged_generation(root: Path, batch_id: str, *,
                            journal: object = None) -> dict[str, object] | None:
    generation = _stage_directory(root, batch_id)
    manifest_path = generation / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (manifest.get("schema") != APPLY_SCHEMA or manifest.get("batch_id") != batch_id
            or manifest.get("root") != str(root)):
        return None
    edits = manifest.get("edits")
    digests = manifest.get("original_digests")
    if not isinstance(edits, list) or not isinstance(digests, Mapping):
        return None
    result: list[dict[str, object]] = []
    for row in edits:
        if not isinstance(row, Mapping):
            return None
        copy_row = dict(row)
        path = str(copy_row.get("path", ""))
        expected = digests.get(path)
        if not isinstance(expected, str):
            return None
        copy_row["expected_revision"] = expected
        result.append(copy_row)
    candidates: dict[str, str] = {}
    for raw_path in manifest.get("changed_paths", ()):
        relative = str(raw_path)
        try:
            _safe_relative(root, relative, journal=journal)
            candidate_path = generation / "candidate" / relative
            candidates[relative] = candidate_path.read_text(
                encoding="utf-8", errors="surrogateescape")
        except (CoordinatedChangeError, OSError, UnicodeError):
            return None
    return {"edits": result, "candidates": candidates,
            "stage_revision": str(manifest.get("stage_revision", "")),
            "original_digests": dict(digests)}


def _coerce_edits(edits: Sequence[object]) -> tuple[BatchEdit, ...]:
    if isinstance(edits, (str, bytes, bytearray)) or not isinstance(edits, Sequence):
        raise CoordinatedChangeError("edits must be a list")
    result = tuple(BatchEdit.from_mapping(item) for item in edits
                   if isinstance(item, Mapping))
    if len(result) != len(edits):
        raise CoordinatedChangeError("every batch item must be an object")
    if not result:
        raise CoordinatedChangeError("batch must contain at least one edit")
    return result


def _validate_file(source: str, relative: str) -> str | None:
    suffix = Path(relative).suffix.lower()
    if suffix == ".py":
        try:
            ast.parse(source, filename=relative)
        except SyntaxError as error:
            return f"Python syntax error at line {error.lineno}: {error.msg}"
        return None
    if suffix in {".json"}:
        try:
            json.loads(source)
        except json.JSONDecodeError as error:
            return f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        return None
    try:
        from kernel_tokenlab_nonpy_edits import applies, validate
        if applies(Path(relative)):
            validate(source, Path(relative))
    except ImportError:
        return None
    except ValueError as error:
        return str(error)
    return None


def _resolve_from(current_module: str, module: str | None, level: int) -> str:
    package = current_module.split(".")[:-1]
    if level:
        base = package[:max(0, len(package) - level + 1)]
    else:
        base = []
    if module:
        base.extend(module.split("."))
    return ".".join(part for part in base if part)


def _python_import_failures(root: Path, overlays: Mapping[str, str],
                            edit_indexes: Mapping[str, int],
                            changed_paths: Iterable[str] = ()) -> tuple[BatchFailure, ...]:
    """Check only import/export edges touched by this candidate overlay.

    The starter tree is intentionally not treated as globally healthy.  A
    pre-existing unrelated import failure is evidence for the host's repair
    ledger, not a reason to reject this change set.
    """
    changed = set(changed_paths)
    files = [path for path in _iter_files(root, overlays) if path.endswith(".py")]
    modules: dict[str, tuple[str, set[str]]] = {}
    trees: dict[str, ast.Module] = {}
    for relative in files:
        source = _source_for(root, relative, overlays)
        if source is None:
            continue
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError:
            continue
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                names.update(target.id for target in targets if isinstance(target, ast.Name))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name != "*":
                        names.add(str(alias.asname or alias.name.rsplit(".", 1)[-1]))
        module = _module_for_path(relative)
        modules[module] = (relative, names)
        trees[relative] = tree
    failures: list[BatchFailure] = []
    for relative, tree in trees.items():
        current_module = _module_for_path(relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target_module = _resolve_from(current_module, node.module, node.level)
            target = modules.get(target_module)
            if target is None or (relative not in changed and target[0] not in changed):
                continue
            for alias in node.names:
                if alias.name == "*" or alias.name in target[1]:
                    continue
                symbol = alias.name
                importer = relative
                defining = target[0]
                index = min(edit_indexes.get(importer, edit_indexes.get(defining, 0)), 2**31 - 1)
                failures.append(BatchFailure(
                    index, importer, "MISSING_SYMBOL",
                    f"missing symbol {symbol}; {defining} must define {symbol}, "
                    f"imported by {importer}", symbol, defining, importer))
    unique: list[BatchFailure] = []
    seen: set[tuple[str, str, str]] = set()
    for row in failures:
        key = (row.path, row.missing_symbol, row.defining_file)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return tuple(unique)


def _typed_plan_paths(plan: object) -> tuple[str, ...]:
    paths: set[str] = set()
    if not isinstance(plan, Mapping):
        return ()
    edits = plan.get("edits", ())
    if not isinstance(edits, Sequence) or isinstance(edits, (str, bytes)):
        return ()
    for item in edits:
        if not isinstance(item, Mapping):
            continue
        for key in ("path",):
            raw = item.get(key)
            if isinstance(raw, str) and raw:
                paths.add(raw.replace("\\", "/").lstrip("./"))
        for key in ("target_ref", "site_ref", "destination_ref", "destination"):
            value = item.get(key)
            if isinstance(value, Mapping) and isinstance(value.get("path"), str):
                paths.add(str(value["path"]).replace("\\", "/").lstrip("./"))
    return tuple(sorted(paths))


def _typed_authority(root: Path, plan: Mapping[str, object], *, journal: object = None):
    from kernel_revision_bound_edit import EditAuthority, EditPlan

    paths = _typed_plan_paths(plan)
    sources: dict[str, bytes] = {}
    for relative in paths:
        path, normalized = _safe_relative(root, relative, journal=journal)
        if path.is_symlink():
            raise CoordinatedChangeError(f"symlink target is not editable: {normalized}")
        if not path.is_file():
            if _workspace_truth.enabled():
                fact = _workspace_truth.exists(root, normalized, journal=journal)
                detail = (
                    _workspace_truth.render_fact(fact)
                    if fact.status != "exists"
                    else f"Path {normalized} exists in the live workspace but is not a file."
                )
                raise CoordinatedChangeError(detail)
            raise CoordinatedChangeError(f"typed edit source is missing: {normalized}")
        sources[normalized] = path.read_bytes()
    edit_plan = EditPlan.from_mapping(plan)
    return EditPlan, EditAuthority(
        root=root, sources=sources, base_revision=edit_plan.base_revision,
        authority_ref=edit_plan.authority_ref, allowed_paths=paths, journal=journal,
    ), edit_plan


def validate_typed_plan(root: str | Path, plan: Mapping[str, object], *,
                        journal: object = None) -> BatchValidation:
    """Validate a host-anchored plan without mutating the checkout."""
    root_path = Path(root).resolve()
    try:
        _unused, authority, edit_plan = _typed_authority(root_path, plan, journal=journal)
        from kernel_revision_bound_edit import CandidateTransaction, validate_source

        def final_validator(candidate: Mapping[str, bytes]) -> tuple[str, ...]:
            failures: list[str] = []
            for relative, source in candidate.items():
                result = validate_source(relative, source,
                                         baseline=authority.sources.get(relative))
                if result.syntax_status == "FAIL":
                    failures.extend(f"{relative}: {message}" for message in result.diagnostics)
            return tuple(failures)

        transaction = CandidateTransaction(authority)
        generation = transaction.prepare(edit_plan, validator=final_validator)
        originals = dict(authority.sources)
        candidates = {path: source.decode("utf-8", errors="surrogateescape")
                      for path, source in generation.sources.items()
                      if source != originals.get(path)}
        failures = tuple(BatchFailure(0, "", "TYPED_PLAN", message)
                         for message in generation.diagnostics)
        warnings: list[str] = []
        for relative, source in generation.sources.items():
            result = validate_source(relative, source, baseline=authority.sources.get(relative))
            if result.syntax_status in {"UNKNOWN", "UNRUNNABLE"}:
                warnings.append(f"{relative}: syntax status {result.syntax_status}")
        return BatchValidation(not failures, candidates, originals, tuple(warnings), failures)
    except (CoordinatedChangeError, ValueError, TypeError) as error:
        failure = BatchFailure(0, "", "TYPED_PLAN", str(error))
        return BatchValidation(False, failures=(failure,))


_TYPED_TRANSACTIONS: dict[str, object] = {}


def apply_typed_plan(root: str | Path, plan: Mapping[str, object], *,
                     publish: bool | None = None, journal: object = None) -> BatchResult:
    """Stage or pointer-publish a typed plan; never materialize partial files."""
    root_path = Path(root).resolve()
    from kernel_revision_bound_edit import CandidateTransaction

    try:
        _unused, authority, edit_plan = _typed_authority(root_path, plan, journal=journal)
        identity = str(plan.get("change_set", "") or "")
        transaction = _TYPED_TRANSACTIONS.get(identity) if identity else None
        if not isinstance(transaction, CandidateTransaction):
            transaction = CandidateTransaction(authority)
            identity = identity or ("typed:" + hashlib.sha256(
                json.dumps(edit_plan.to_dict(), sort_keys=True).encode()).hexdigest()[:16])
            _TYPED_TRANSACTIONS[identity] = transaction
        generation = transaction.prepare(edit_plan)
        do_publish = edit_plan.publish if publish is None else publish
        if not do_publish:
            return BatchResult("STAGED" if not generation.diagnostics else "REJECTED", identity,
                               authority.source_revision, tuple(sorted(generation.sources)),
                               tuple("STAGED" for _ in edit_plan.edits),
                               failures=tuple(BatchFailure(0, "", "TYPED_PLAN", item)
                                              for item in generation.diagnostics),
                               stage_revision=authority.source_revision, change_set_ref=identity)
        receipt = transaction.publish(current_sources=authority.sources)
        status = "APPLIED" if receipt.published else "REJECTED"
        failures = tuple(BatchFailure(0, "", receipt.status.value, item)
                         for item in receipt.diagnostics)
        return BatchResult(status, identity, receipt.base_revision,
                           tuple(sorted(generation.sources)),
                           tuple("APPLIED" if receipt.published else "REJECTED" for _ in edit_plan.edits),
                           failures=failures, stage_revision=authority.source_revision,
                           change_set_ref=receipt.generation_id,
                           recovery_required=receipt.status.value == "UNKNOWN_PUBLISHED")
    except (CoordinatedChangeError, ValueError, TypeError) as error:
        failure = BatchFailure(0, "", "TYPED_PLAN", str(error))
        return BatchResult("REJECTED", str(plan.get("change_set", "typed:invalid")),
                           tree_revision(root_path), failures=(failure,))


def validate_batch(root: str | Path, edits: Sequence[object], *,
                   protected_paths: Iterable[object] = (),
                   plan: Mapping[str, object] | None = None,
                   journal: object = None) -> BatchValidation:
    """Preflight every edit and only the affected Python import/export edges."""
    if plan is not None:
        return validate_typed_plan(root, plan, journal=journal)
    root_path = Path(root).resolve()
    coerced = _coerce_edits(edits)
    protected = {str(Path(str(item)).as_posix()) for item in protected_paths}
    candidates: dict[str, str] = {}
    originals: dict[str, bytes] = {}
    warnings: list[str] = []
    failures: list[BatchFailure] = []
    edit_indexes: dict[str, int] = {}
    for index, edit in enumerate(coerced):
        try:
            path, relative = _safe_relative(root_path, edit.path, journal=journal)
        except CoordinatedChangeError as error:
            failures.append(BatchFailure(index, edit.path, "PATH", str(error)))
            continue
        if relative in protected or path.name.startswith("test_") and "tests" in path.parts:
            failures.append(BatchFailure(index, relative, "PROTECTED", "target is protected"))
            continue
        if edit.old == edit.new:
            failures.append(BatchFailure(index, relative, "NO_OP", "old and new text are identical"))
            continue
        edit_indexes.setdefault(relative, index)
        if relative not in originals:
            try:
                originals[relative] = path.read_bytes()
            except OSError as error:
                failures.append(BatchFailure(index, relative, "READ", f"cannot read file: {error}"))
                continue
            candidates[relative] = originals[relative].decode("utf-8", errors="surrogateescape")
        current = candidates[relative]
        expected = edit.expected_revision.strip()
        if expected:
            actual = _sha_bytes(originals[relative])
            if expected not in {actual, actual.removeprefix("sha256:")}:
                failures.append(BatchFailure(
                    index, relative, "STALE_REVISION",
                    f"stale revision for {relative}: expected {expected}, current {actual}"))
                continue
        count = current.count(edit.old)
        if count != 1:
            failures.append(BatchFailure(
                index, relative, "MATCH", f"old text matches {count} places in {relative}; expected exactly one"))
            continue
        candidates[relative] = current.replace(edit.old, edit.new, 1)
    # Validate final per-file overlays only.  Ordered edits may intentionally
    # pass through an invalid intermediate text while ending in valid source.
    for relative, candidate in candidates.items():
        reason = _validate_file(candidate, relative)
        if reason:
            index = edit_indexes.get(relative, 0)
            failures.append(BatchFailure(index, relative, "SYNTAX", reason))
            continue
        if relative.endswith((".html", ".htm")):
            try:
                from kernel_tokenlab_nonpy_edits import validate
                warning = validate(candidate, Path(relative))
            except (ImportError, ValueError):
                warning = None
            if warning:
                warnings.append(f"{relative}: {warning}")
    failures.extend(_python_import_failures(root_path, candidates, edit_indexes, candidates))
    return BatchValidation(not failures, candidates, originals, tuple(warnings), tuple(failures))


def apply_batch(root: str | Path, edits: Sequence[object], *,
                journal: object | None = None,
                ledger: CoordinatedChangeLedger | None = None,
                protected_paths: Iterable[object] = (),
                publish: bool = True,
                change_set: str | None = None,
                plan: Mapping[str, object] | None = None) -> BatchResult:
    """Validate one retained generation and optionally publish it.

    Preflight is all-or-nothing.  A publish failure is reported as recovery
    required; this path never restores an old file over a partially committed
    generation and never emits fake per-file completion receipts.
    """
    if plan is not None:
        result = apply_typed_plan(root, plan, publish=publish, journal=journal)
        _emit(journal, APPLY_EVENT, action="typed-plan", status=result.status,
              batch_id=result.batch_id, changed_paths=list(result.changed_paths),
              failures=[item.to_dict() for item in result.failures])
        return result
    try:
        batch_identity = [dict(item) if isinstance(item, Mapping) else repr(item) for item in edits]
    except TypeError:
        batch_identity = repr(edits)
    batch_id = change_set or ("batch:" + _digest(batch_identity)[:16])
    root_path = Path(root).resolve()
    if not atomic_enabled():
        return BatchResult("DISABLED", batch_id, tree_revision(root_path))
    persisted_generation: dict[str, object] | None = None
    if not edits:
        staged = _STAGED_BATCHES.get(batch_id)
        if staged is None:
            loaded = _load_staged_generation(root_path, batch_id, journal=journal)
            if loaded is not None:
                staged = {"root": str(root_path), "edits": tuple(loaded["edits"]),
                          "candidates": loaded["candidates"],
                          "original_digests": loaded["original_digests"],
                          "stage_revision": loaded["stage_revision"]}
                _STAGED_BATCHES[batch_id] = staged
        if publish and staged is not None:
            edits = staged["edits"]  # type: ignore[assignment]
            if isinstance(staged.get("candidates"), Mapping):
                persisted_generation = staged
        else:
            failure = BatchFailure(0, "", "REQUEST", "an edit list is required")
            return BatchResult("REJECTED", batch_id, tree_revision(root_path), failures=(failure,))
    try:
        validation = validate_batch(
            root_path, edits, protected_paths=protected_paths, journal=journal,
        )
    except (CoordinatedChangeError, TypeError, ValueError) as error:
        failure = BatchFailure(0, "", "REQUEST", str(error))
        result = BatchResult("REJECTED", batch_id, tree_revision(root_path), failures=(failure,))
        _emit(journal, APPLY_EVENT, action="preflight", status=result.status,
              batch_id=batch_id, failures=[failure.to_dict()])
        return result
    if persisted_generation is not None and validation.valid:
        persisted_candidates = dict(persisted_generation["candidates"])
        if persisted_candidates != dict(validation.candidates):
            failure = BatchFailure(0, "", "STALE_STAGE",
                                   "retained candidate differs from its ordered edit manifest")
            validation = BatchValidation(False, failures=(failure,))
        else:
            # Publish the retained bytes, not a newly recomputed candidate.
            validation = BatchValidation(True, persisted_candidates, validation.original,
                                         validation.warnings, validation.failures)
    if not validation.valid:
        result = BatchResult("REJECTED", batch_id, tree_revision(root_path),
                             failures=validation.failures, warnings=validation.warnings)
        _emit(journal, APPLY_EVENT, action="preflight", status=result.status,
              batch_id=batch_id, failures=[row.to_dict() for row in validation.failures],
              warnings=list(validation.warnings))
        return result
    # Re-read all pre-images after validation.  A concurrent edit is a stale
    # transaction, never a reason to overwrite another agent's work.
    for relative, original in validation.original.items():
        current = (root_path / relative).read_bytes()
        if current != original:
            failure = BatchFailure(0, relative, "CONCURRENT_CHANGE",
                                   f"workspace changed during preflight: {relative}")
            result = BatchResult("REJECTED", batch_id, tree_revision(root_path), failures=(failure,))
            _emit(journal, APPLY_EVENT, action="preflight", status=result.status,
                  batch_id=batch_id, failures=[failure.to_dict()])
            return result
    stage_revision = tree_revision(root_path)
    if not publish:
        _persist_stage(root_path, batch_id, edits, validation, stage_revision)
        _STAGED_BATCHES[batch_id] = {
            "root": str(root_path), "edits": tuple(edits),
            "candidates": dict(validation.candidates), "original": dict(validation.original),
            "stage_revision": stage_revision,
        }
        result = BatchResult(
            "STAGED", batch_id, stage_revision, tuple(sorted(validation.candidates)),
            tuple("STAGED" for _ in _coerce_edits(edits)), warnings=validation.warnings,
            stage_revision=stage_revision, change_set_ref=batch_id)
        _emit(journal, APPLY_EVENT, action="stage", status=result.status, batch_id=batch_id,
              changed_paths=list(result.changed_paths), stage_revision=stage_revision,
              warnings=list(result.warnings))
        if ledger is not None:
            ledger.record_stage(result)
        return result
    temporary_paths: dict[str, Path] = {}
    replaced: list[str] = []
    try:
        for relative, source in validation.candidates.items():
            target = root_path / relative
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.",
                                             suffix=".candidate", delete=False) as handle:
                temporary_paths[relative] = Path(handle.name)
                handle.write(source.encode("utf-8", errors="surrogateescape"))
                handle.flush()
                os.fsync(handle.fileno())
        for relative, temporary in temporary_paths.items():
            os.replace(temporary, root_path / relative)
            replaced.append(relative)
        for relative, source in validation.candidates.items():
            expected = source.encode("utf-8", errors="surrogateescape")
            if (root_path / relative).read_bytes() != expected:
                raise OSError(f"post-publish verification failed for {relative}")
    except Exception as error:  # noqa: BLE001 - surface recovery, never hide it.
        for temporary in temporary_paths.values():
            if temporary.exists():
                temporary.unlink()
        failure = BatchFailure(0, replaced[-1] if replaced else "", "RECOVERY_REQUIRED",
                               f"atomic publish failed: {type(error).__name__}: {error}")
        result = BatchResult("REJECTED", batch_id, tree_revision(root_path),
                             failures=(failure,),
                             stage_revision=stage_revision, change_set_ref=batch_id,
                             recovery_required=bool(replaced))
        _emit(journal, APPLY_EVENT, action="publish", status=result.status, batch_id=batch_id,
              failures=[failure.to_dict()],
              recovery_required=bool(replaced))
        return result
    revision = tree_revision(root_path)
    result = BatchResult(
        "APPLIED", batch_id, revision, tuple(sorted(validation.candidates)),
        tuple("APPLIED" if edit.old != edit.new else "NO_OP" for edit in _coerce_edits(edits)),
        warnings=validation.warnings, stage_revision=stage_revision,
        change_set_ref=batch_id)
    if ledger is not None:
        ledger.record_batch(result)
    _emit(journal, APPLY_EVENT, action="publish", status=result.status, batch_id=batch_id,
          changed_paths=list(result.changed_paths), item_statuses=list(result.item_statuses),
          revision=revision, warnings=list(result.warnings))
    _STAGED_BATCHES.pop(batch_id, None)
    return result


def atomic_apply(root: str | Path, arguments: Mapping[str, object], **kwargs: object) -> BatchResult:
    if not atomic_enabled():
        return BatchResult("DISABLED", "batch:disabled", tree_revision(root))
    typed_plan = arguments.get("plan") if isinstance(arguments, Mapping) else None
    if isinstance(typed_plan, Mapping):
        return apply_typed_plan(root, typed_plan,
                                publish=bool(arguments.get("publish", typed_plan.get("publish", False))),
                                journal=kwargs.get("journal"))
    if isinstance(arguments, Mapping) and isinstance(arguments.get("edits"), Sequence):
        raw_edits = arguments.get("edits")
        if any(isinstance(item, Mapping) and (
                item.get("op") in {"insert_before", "move"}
                or isinstance(item.get("target_ref", item.get("site_ref")), Mapping))
                for item in raw_edits if isinstance(raw_edits, Sequence)):
            plan = {
                "schema": "revision-bound-edit-plan/1",
                "base_revision": arguments.get("base_revision", ""),
                "authority_ref": arguments.get("authority_ref", ""),
                "inventory_ref": arguments.get("inventory_ref", "batch-inventory"),
                "edits": list(raw_edits),
                "check_refs": arguments.get("check_refs", ()),
                "publish": bool(arguments.get("publish", False)),
            }
            return apply_typed_plan(root, plan, journal=kwargs.get("journal"))
    edits = arguments.get("edits") if isinstance(arguments, Mapping) else None
    publish = bool(arguments.get("publish", True)) if isinstance(arguments, Mapping) else True
    change_set = str(arguments.get("change_set", "") or "") if isinstance(arguments, Mapping) else ""
    if edits == [] and publish and change_set:
        return apply_batch(root, (), publish=True, change_set=change_set, **kwargs)
    if not isinstance(edits, Sequence) or isinstance(edits, (str, bytes, bytearray)):
        failure = BatchFailure(0, "", "REQUEST", "atomic_apply requires an edits list")
        return BatchResult("REJECTED", "batch:invalid", tree_revision(root), failures=(failure,))
    return apply_batch(root, edits, publish=publish, change_set=change_set or None, **kwargs)


def augment_str_replace_tool(tools: Sequence[dict]) -> list[dict]:
    """Extend the existing ``str_replace`` schema without adding a new tool.

    The scalar branch is copied byte-for-byte in shape; the batch branch is an
    opt-in union.  When capability is off the original list object is returned
    unchanged so the v21 surface remains byte-identical.
    """
    if not atomic_enabled():
        return list(tools)
    result = copy.deepcopy(list(tools))
    from kernel_revision_bound_edit import site_ref_property
    batch = {
        "type": "object", "properties": {
            "edits": {"type": "array", "items": {"type": "object", "properties": {
                "path": {"type": "string"}, "old": {"type": "string"},
                "new": {"type": "string"}, "expected_revision": {"type": "string"},
                "target_ref": site_ref_property(),
                "site_ref": site_ref_property(),
                "destination_ref": site_ref_property(),
                "op": {"type": "string", "enum": ["replace", "insert_before", "move"]},
                "order": {"type": "integer"},
            }, "oneOf": [
                {"required": ["path", "old", "new"]},
                {"required": ["target_ref", "op", "new"]},
                {"required": ["target_ref", "destination_ref", "op"]},
            ], "additionalProperties": False}},
            "change_set": {"type": "string"}, "stage_revision": {"type": "string"},
            "base_revision": {"type": "string"},
            "authority_ref": {"type": "string"},
            "inventory_ref": {"type": "string"},
            "check_refs": {"type": "array", "items": {"type": "string"}},
            "publish": {"type": "boolean"},
        }, "required": ["edits"], "additionalProperties": False,
    }
    for tool in result:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if not isinstance(function, dict) or function.get("name") != "str_replace":
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        scalar = copy.deepcopy(dict(parameters))
        function["description"] = str(function.get("description", "")) + (
            " Batch form accepts ordered edits and publishes one parent receipt; "
            "publish=false retains a staged generation.")
        function["parameters"] = {"oneOf": [scalar, batch]}
        break
    return result


def apply_readiness_to_decision(runtime: object, decision: object) -> object:
    """Add the unified ledger hold to the existing readiness authority snapshot."""
    if not enabled():
        return decision
    ledger = getattr(runtime, "coordinated_change_ledger", None)
    if ledger is None:
        render_runtime_ledger(runtime)
        ledger = getattr(runtime, "coordinated_change_ledger", None)
    if ledger is None:
        return decision
    evidence = ledger.coordination_evidence()
    if not evidence.get("holds"):
        return decision
    blockers = tuple(getattr(decision, "blockers", ()) or ())
    marker = {"check_id": "coordination-ledger", "id": "coordination-ledger",
              "counterexample_authority": False, "pending": True,
              "reason": str(evidence["reason"]), "holds": list(evidence["holds"])}
    advisory = list(getattr(decision, "advisory", ()) or ())
    if not any(str(row.get("check_id", row.get("id", ""))) == marker["check_id"]
               for row in advisory if isinstance(row, Mapping)):
        advisory.append(marker)
    dispositions = dict(getattr(decision, "dispositions", {}) or {})
    dispositions["done"] = "deny"
    # Coordination adds its own completion hold, but it cannot grant a tool
    # that an upstream readiness or post-green authority denied.
    digest = _digest({"previous": getattr(decision, "candidate_digest", ""),
                      "coordination": evidence})
    updated = dataclasses.replace(decision, blockers=blockers,
                                  advisory=tuple(advisory), dispositions=dispositions,
                                  candidate_digest=digest)
    _emit(getattr(runtime, "journal", None), LEDGER_EVENT,
          action="evidence_projection", coordination_evidence=evidence)
    return updated


def guidance_menu(runtime: object, selected: Sequence[dict]) -> list[dict]:
    """Project ordering/menu advice only when the independent guidance switch is on."""
    result = list(selected)
    if not guidance_enabled(runtime=runtime):
        return result
    ledger = getattr(runtime, "coordinated_change_ledger", None)
    if ledger is None:
        return result
    evidence = ledger.coordination_evidence()
    if not evidence.get("holds"):
        return result
    result = [tool for tool in result
              if str(tool.get("function", {}).get("name", "")) != "done"]
    next_action = str(evidence.get("next_action") or "")
    if next_action:
        result.sort(key=lambda tool: 0 if tool.get("function", {}).get("name") == next_action else 1)
    _emit(getattr(runtime, "journal", None), LEDGER_EVENT,
          action="guidance_menu", remaining=len(ledger.remaining()), next_action=next_action)
    return result


def record_runtime_batch_receipt(runtime: object, receipt: dict) -> None:
    """Record one parent batch receipt and aggregate downstream observations."""
    status = str(receipt.get("result_class", receipt.get("status", ""))).upper()
    if not atomic_enabled() or status not in {"APPLIED", "STAGED"}:
        return
    ledger = getattr(runtime, "coordinated_change_ledger", None)
    if ledger is None:
        return
    changed = tuple(str(item) for item in receipt.get("changed_paths", ())
                    if str(item).strip())
    batch = BatchResult(status, str(receipt.get("change_set_ref", receipt.get("batch_id", ""))),
                        str(receipt.get("revision", tree_revision(ledger.root))), changed,
                        tuple(str(item) for item in receipt.get("item_statuses", ())),
                        stage_revision=str(receipt.get("stage_revision", "") or "") or None,
                        change_set_ref=str(receipt.get("change_set_ref", "") or "") or None)
    if status == "STAGED":
        ledger.record_stage(batch)
    else:
        ledger.record_batch(batch)
    receipt["coordination"] = {"schema": APPLY_SCHEMA, "change_set_ref": batch.batch_id,
                                "changed_paths": list(changed), "parent": True}
    if status == "APPLIED":
        receipt.setdefault("ripple_preview", {"mode": "aggregate", "changed_paths": list(changed)})
        receipt.setdefault("multidef_observation", {
            "mode": "aggregate", "changed_paths": list(changed), "child_receipts": False})
    try:
        if status != "APPLIED":
            return
        from kernel_tokenlab_stall_breaker import note_coordinated_progress
        note_coordinated_progress(runtime, batch.batch_id, changed)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    _emit(getattr(runtime, "journal", None), APPLY_EVENT, action="parent_receipt",
          status="APPLIED", batch_id=batch.batch_id, changed_paths=list(changed))


def _import_module(root: Path, module_name: str) -> ModuleType:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(module_name)


@contextmanager
def _fresh_imports(root: Path, module_name: str = ""):
    before = set(sys.modules)
    try:
        yield
    finally:
        prefixes = {root.name}
        if module_name:
            prefixes.add(module_name.split(".", 1)[0])
        for name in list(sys.modules):
            if name not in before and name.split(".", 1)[0] in prefixes:
                sys.modules.pop(name, None)


def _rename_fixture_keys(value: object, old: str, new: str) -> object:
    if isinstance(value, Mapping):
        return {new if key == old else key: _rename_fixture_keys(child, old, new)
                for key, child in value.items()}
    if isinstance(value, list):
        return [_rename_fixture_keys(child, old, new) for child in value]
    if isinstance(value, tuple):
        return tuple(_rename_fixture_keys(child, old, new) for child in value)
    return copy.deepcopy(value)


def _observable(value: object, selectors: object = None) -> object:
    if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes, bytearray)):
        selected: dict[str, object] = {}
        for selector in selectors:
            path = str(selector)
            current = value
            for part in path.split("."):
                if isinstance(current, Mapping):
                    current = current.get(part)
                else:
                    current = getattr(current, part, None)
            selected[path] = _observable(current)
        return selected
    if isinstance(value, Mapping):
        return {str(key): _observable(child) for key, child in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [_observable(child) for child in value]
    if hasattr(value, "__dict__"):
        return _observable(vars(value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _nonempty_observable(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value) and any(_nonempty_observable(child) for child in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value) and any(_nonempty_observable(child) for child in value)
    return True


def _call_target(target: object, fixture: Mapping[str, object]) -> object:
    if not callable(target):
        raise TypeError("compatibility target is not callable")
    call = fixture
    calls = fixture.get("calls")
    if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes, bytearray)) and calls:
        first = calls[0]
        if isinstance(first, Mapping):
            call = first
    kwargs = call.get("kwargs", {})
    if not isinstance(kwargs, Mapping):
        kwargs = {}
    args = call.get("args", ())
    if isinstance(args, Mapping):
        return target(args, **dict(kwargs))
    if not isinstance(args, Sequence) or isinstance(args, (str, bytes, bytearray)):
        args = ()
    return target(*list(args), **dict(kwargs))


def _run_check_call(root: Path, item: ChangeItem, *, symbol: str = "",
                    fixture: Mapping[str, object] | None = None,
                    module_name: str | None = None) -> tuple[bool, object, str]:
    check = item.compatibility or {}
    selected_fixture = fixture or check.get("fixture")
    if not isinstance(selected_fixture, Mapping):
        return False, None, "explicit host compatibility fixture is missing"
    module_name = str(module_name or check.get("module", ""))
    selected_symbol = symbol or str(check.get("symbol", item.old))
    try:
        with _fresh_imports(root, module_name):
            module = _import_module(root, module_name)
            target = getattr(module, selected_symbol)
            value = _call_target(target, selected_fixture)
            observed = _observable(value, selected_fixture.get("observe"))
        return True, observed, f"{selected_symbol} produced an observable result"
    except Exception as error:  # noqa: BLE001 - result is evidence, not a crash.
        return False, None, f"{type(error).__name__}: {error}"


def _relation_fixtures(item: ChangeItem) -> tuple[Mapping[str, object], Mapping[str, object]] | None:
    check = item.compatibility or {}
    fixture = check.get("fixture")
    relation = check.get("relation")
    if not isinstance(fixture, Mapping) or not isinstance(relation, Mapping) or not _nonempty_observable(fixture):
        return None
    if item.kind == COMPATIBILITY_INPUT:
        canonical = _rename_fixture_keys(fixture, item.old, item.new)
        if not isinstance(canonical, Mapping):
            return None
        return fixture, canonical
    return fixture, fixture


def run_compatibility_check(root: str | Path, item: ChangeItem,
                            *, starter_root: str | Path | None = None,
                            journal: object | None = None) -> dict[str, object]:
    """Run one host-fixture witness and require one observable relation."""
    if item.kind not in {COMPATIBILITY_ALIAS, COMPATIBILITY_INPUT}:
        raise CoordinatedChangeError("item is not a compatibility check")
    if not item.checkable:
        result = {"status": "UNAVAILABLE", "gating": False,
                  "reason": "defining file or call target was not located"}
    elif starter_root is None:
        result = {"status": "UNKNOWN", "gating": False,
                  "reason": "a valid starter witness is required for preservation"}
    elif _relation_fixtures(item) is None:
        result = {"status": "UNKNOWN", "gating": False,
                  "reason": "a sound compatibility relation and fixture are required"}
    else:
        relation_fixtures = _relation_fixtures(item)
        assert relation_fixtures is not None
        legacy_fixture, canonical_fixture = relation_fixtures
        check = item.compatibility or {}
        legacy_symbol = str(check.get("symbol", item.old))
        canonical_symbol = (str(check.get("symbol", item.new))
                            if item.kind == COMPATIBILITY_INPUT else item.new)
        legacy_passed, legacy_value, legacy_detail = _run_check_call(
            Path(root).resolve(), item, symbol=legacy_symbol, fixture=legacy_fixture)
        canonical_passed, canonical_value, canonical_detail = _run_check_call(
            Path(root).resolve(), item, symbol=canonical_symbol, fixture=canonical_fixture)
        relation_passed = (legacy_passed and canonical_passed and
                           _nonempty_observable(legacy_value)
                           and _nonempty_observable(canonical_value)
                           and legacy_value == canonical_value)
        owner_witnesses: list[dict[str, object]] = []
        owners = check.get("import_owners", ())
        if isinstance(owners, Sequence) and not isinstance(owners, (str, bytes, bytearray)):
            for owner in owners:
                if not isinstance(owner, Mapping) or not str(owner.get("module", "")).strip():
                    continue
                module_name = str(owner["module"])
                owner_legacy_symbol = (legacy_symbol if item.kind == COMPATIBILITY_ALIAS
                                       else str(check.get("symbol", item.new)))
                owner_canonical_symbol = (canonical_symbol if item.kind == COMPATIBILITY_ALIAS
                                          else str(check.get("symbol", item.new)))
                owner_legacy_passed, owner_legacy_value, owner_legacy_detail = _run_check_call(
                    Path(root).resolve(), item, symbol=owner_legacy_symbol,
                    fixture=legacy_fixture, module_name=module_name)
                owner_canonical_passed, owner_canonical_value, owner_canonical_detail = _run_check_call(
                    Path(root).resolve(), item, symbol=owner_canonical_symbol,
                    fixture=canonical_fixture, module_name=module_name)
                owner_passed = (owner_legacy_passed and owner_canonical_passed
                                and _nonempty_observable(owner_legacy_value)
                                and _nonempty_observable(owner_canonical_value)
                                and owner_legacy_value == owner_canonical_value)
                owner_witnesses.append({
                    "module": module_name, "path": owner.get("path"),
                    "legacy": {"symbol": owner_legacy_symbol, "passed": owner_legacy_passed,
                                "detail": owner_legacy_detail, "observable": owner_legacy_value},
                    "canonical": {"symbol": owner_canonical_symbol, "passed": owner_canonical_passed,
                                   "detail": owner_canonical_detail, "observable": owner_canonical_value},
                    "passed": owner_passed,
                })
            relation_passed = relation_passed and all(row["passed"] for row in owner_witnesses)
        result = {
            "status": "PASS" if relation_passed else "FAIL",
            "gating": relation_passed,
            "detail": {"legacy": legacy_detail, "canonical": canonical_detail},
            "relation": copy.deepcopy(check.get("relation")),
            "call": check.get("call", ""),
            "import_owner_witnesses": owner_witnesses,
        }
        baseline_symbol = str(check.get("baseline_symbol", ""))
        starter_symbol = baseline_symbol or (legacy_symbol if item.kind == COMPATIBILITY_ALIAS else "")
        if starter_symbol:
            starter_passed, starter_value, starter_detail = _run_check_call(
                Path(starter_root).resolve(), item, symbol=starter_symbol,
                fixture=legacy_fixture)
            result["starter_detail"] = starter_detail
            result["starter_observable"] = starter_value
        else:
            starter_passed = False
            result["reason"] = "starter legacy entry could not be derived"
        if not starter_passed:
            result["gating"] = False
            result["reason"] = "starter fails the same compatibility witness"
        elif not relation_passed:
            result["reason"] = "legacy and canonical calls did not produce the same observable result"
    _emit(journal, COMPAT_EVENT, item_id=item.item_id, item_kind=item.kind,
          status=result.get("status"), gating=bool(result.get("gating")),
          call=result.get("call", ""), detail=result.get("detail", ""))
    return result


def extract_public_journal(path: str | Path) -> dict[str, object]:
    """Extract only public task/source text from a crawl journal."""
    journal_path = Path(path)
    from kernel_measurement import resolve_journal_blob_refs

    if any(part.casefold() in _FORBIDDEN_PARTS for part in journal_path.parts):
        raise CoordinatedChangeError("refusing a private or prohibited journal path")
    ticket = ""
    sources: dict[str, str] = {}
    blob_cache: dict[str, object] = {}
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            row = resolve_journal_blob_refs(row, journal_path, blob_cache)
        except (OSError, ValueError) as error:
            raise CoordinatedChangeError("journal contains an invalid blob reference") from error
        for text in _string_values(row):
            if (not ticket and "Task:" in text and
                    re.search(r"\brename\w*\b|\bis\s+now\b|->|→|\bold\s+names?\b",
                              text, re.IGNORECASE)):
                ticket = text.split("Task:", 1)[1].split("--- user ---", 1)[0].strip()
            for match in re.finditer(r"\[Current source: ([^\]]+)\]\n(.*?)(?=\n(?:---|\[)|\Z)", text, re.S):
                source_path = match.group(1).strip().replace("\\", "/")
                if any(part.casefold() in _FORBIDDEN_PARTS for part in Path(source_path).parts):
                    continue
                if source_path and not source_path.startswith("/"):
                    sources[source_path] = match.group(2).rstrip("\n") + "\n"
    return {"ticket": ticket, "sources": sources, "path": str(journal_path)}


def _string_values(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for child in value:
            yield from _string_values(child)


def replay_journal(path: str | Path) -> dict[str, object]:
    extracted = extract_public_journal(path)
    with tempfile.TemporaryDirectory(prefix="stateframe-v22-replay-") as temporary:
        root = Path(temporary)
        for relative, source in dict(extracted["sources"]).items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        ticket = str(extracted["ticket"])
        ledger = CoordinatedChangeLedger.from_ticket(root, ticket)
        return {
            "ticket": ticket, "source_count": len(extracted["sources"]),
            "rename_items": [item.item_id for item in ledger.items if item.kind == SYMBOL_RENAME],
            "compatibility_items": [item.item_id for item in ledger.items
                                    if item.kind in {COMPATIBILITY_ALIAS, COMPATIBILITY_INPUT}],
            "input_checkable": any(item.kind == COMPATIBILITY_INPUT and item.checkable
                                    for item in ledger.items),
            "ledger": ledger.to_dict(),
        }


def _failure_target_from_smoke(result: Mapping[str, object]) -> tuple[dict[str, str], ...]:
    failures = result.get("failures", ())
    rows: list[dict[str, str]] = []
    if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes, bytearray)):
        return ()
    for failure in failures:
        if isinstance(failure, Mapping):
            importer = str(failure.get("path", failure.get("importer_file", "")))
            error = str(failure.get("error", failure.get("message", "")))
        elif isinstance(failure, Sequence) and len(failure) >= 2:
            importer, error = str(failure[0]), str(failure[1])
        else:
            continue
        match = _MISSING_IMPORT.search(error)
        if not match:
            continue
        rows.append({
            "missing_symbol": match.group("symbol"),
            "module": match.group("module"),
            "defining_file": match.group("module").replace(".", "/") + ".py",
            "importer_file": importer,
        })
    return tuple(rows)


_RENDER_PATH_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
    r"|(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]+\.(?:py|js|ts|tsx|jsx|vue|go|rs)"
)


def _render_scope_projection(ledger: CoordinatedChangeLedger,
                             allowed_paths: Sequence[str]) -> tuple[CoordinatedChangeLedger, int, int]:
    """Copy only task-scoped coordination rows for model rendering."""
    allowed = {str(path).replace("\\", "/") for path in allowed_paths}
    projection = copy.copy(ledger)
    safe_items = []
    dropped = 0
    for item in ledger.items:
        defining = str(getattr(item, "defining_file", "") or "").replace("\\", "/")
        usage_sites = getattr(item, "usage_sites", ())
        usage_paths = [
            str(getattr(site, "path", "") or "").replace("\\", "/")
            for site in usage_sites if getattr(site, "path", None)
        ]
        fields = (
            str(getattr(item, "item_id", "")),
            str(getattr(item, "old", "")),
            str(getattr(item, "new", "")),
            defining,
            *usage_paths,
        )
        tokens = {
            token.replace("\\", "/") for value in fields
            for token in _RENDER_PATH_TOKEN.findall(value)
        }
        if (not defining or defining not in allowed
                or any(path not in allowed for path in usage_paths)
                or not tokens.issubset(allowed)):
            dropped += 1
            continue
        safe_items.append(item)
    projection.items = safe_items
    suppressed_failures = len(ledger.broken)
    projection.broken = []
    projection._render_withheld_count = dropped
    projection._render_failure_suppressed = bool(suppressed_failures)
    return projection, dropped, suppressed_failures


@profiled_function_span("tokenlab_v22_unified_ledger")
def render_runtime_ledger(runtime: object) -> str | None:
    """Build and render one compact runtime view at the existing sidecar seam."""
    try:
        from kernel_host_cache import memo_runtime_sidecar_text, sidecar_cache_enabled
    except Exception:
        return _render_runtime_ledger_uncached(runtime)
    if not sidecar_cache_enabled():
        return _render_runtime_ledger_uncached(runtime)
    try:
        # Smoke results can change without either prescribed key axis, so this
        # render must stay on the current path whenever one is present.
        if getattr(runtime, "_ship_tree_smoke_result", None) is not None:
            return _render_runtime_ledger_uncached(runtime)
    except Exception:
        return _render_runtime_ledger_uncached(runtime)
    return memo_runtime_sidecar_text(
        runtime, "LEDGER_RENDER", lambda: _render_runtime_ledger_uncached(runtime))


def _render_runtime_ledger_uncached(runtime: object) -> str | None:
    """Current uncached renderer; keep this as the cache's fail-open path."""
    if ledger_mode() != "on":
        return None
    ledger = getattr(runtime, "coordinated_change_ledger", None)
    if ledger is None:
        runtime_owner = getattr(runtime, "owner", None)
        row = getattr(runtime_owner, "task_row", {})
        if not isinstance(row, Mapping):
            return None
        ticket = next((row.get(key) for key in (
            "_bridge_ordinary_brief", "public_brief", "ticket_text", "brief", "msg")
                       if isinstance(row.get(key), str) and row.get(key).strip()), "")
        if not ticket:
            return None
        requirements = getattr(runtime, "requirements_ledger", None)
        if requirements is None:
            try:
                from kernel_requirements_ledger import build_ledger
                requirements = build_ledger(row, root=getattr(runtime_owner, "root", Path.cwd()))
            except (ImportError, CoordinatedChangeError, OSError, UnicodeError, ValueError):
                requirements = None
        planned = getattr(runtime, "planned_artifacts", None)
        if planned is None:
            continuation = getattr(runtime, "continuation", None)
            engine = getattr(continuation, "engine", None)
            planned = getattr(engine, "planned_artifacts", None)
        compatibility_fixtures: dict[str, object] = {}
        fixture = next((row.get(key) for key in (
            "input_fixture", "public_fixture", "fixture")
                        if isinstance(row.get(key), Mapping)), None)
        if isinstance(fixture, Mapping):
            compatibility_fixtures["public_fixture"] = {
                "kind": "input", "args": [[dict(fixture)]],
            }
        try:
            ledger = CoordinatedChangeLedger.from_ticket(
                getattr(runtime_owner, "root", Path.cwd()), str(ticket),
                compatibility_fixtures=compatibility_fixtures,
                task_id=str(row.get("task_id") or ""),
                requirements_ledger=requirements,
                planned_artifacts=planned,
                obligation_ledger=getattr(runtime, "where_used_ledger", None),
                journal=getattr(runtime, "journal", None))
        except (CoordinatedChangeError, OSError, UnicodeError):
            return None
        runtime.coordinated_change_ledger = ledger
    if not guidance_enabled():
        return None
    smoke = getattr(runtime, "_ship_tree_smoke_result", None)
    if smoke is not None:
        for failure in _failure_target_from_smoke(smoke.as_dict() if hasattr(smoke, "as_dict") else smoke):
            ledger.record_import_failure(failure)
    try:
        from kernel_request_source_window import editable_product_paths
        allowed_paths = editable_product_paths(runtime)
    except (ImportError, OSError, TypeError, ValueError):
        allowed_paths = ()
    projection, dropped_entries, suppressed_failures = _render_scope_projection(
        ledger, allowed_paths)
    rendered = projection.render()
    if getattr(runtime, "journal", None) is not None:
        _emit(runtime.journal, LEDGER_EVENT, action="render", item_id=(
            projection.current_item().item_id if projection.current_item() else None),
            remaining=len(projection.remaining()),
            dropped_entries=dropped_entries,
            suppressed_failure_details=suppressed_failures,
            revision=ledger.revision)
    return rendered


__all__ = [
    "ATOMIC_APPLY_ENV", "APPLY_EVENT", "APPLY_SCHEMA", "BatchEdit", "BatchFailure",
    "BatchResult", "BatchValidation", "ChangeItem", "COMPATIBILITY_ALIAS",
    "COMPATIBILITY_INPUT", "COMPAT_SCHEMA", "CoordinatedChangeError",
    "CoordinatedChangeLedger", "UnifiedChangeLedger", "DefinitionSite", "IMPORT_EVENT", "IMPORT_REPAIR",
    "IMPORT_SCHEMA", "LEDGER_ENV", "LEDGER_EVENT", "SCHEMA",
    "STATUS_APPLIED", "STATUS_PENDING", "STATUS_VERIFIED", "SYMBOL_RENAME",
    "apply_batch", "atomic_apply", "atomic_apply_mode", "atomic_enabled", "enabled",
    "augment_str_replace_tool", "apply_readiness_to_decision", "guidance_menu",
    "guidance_enabled", "guidance_mode", "record_runtime_batch_receipt",
    "extract_public_journal", "ledger_mode", "replay_journal", "render_runtime_ledger",
    "run_compatibility_check", "scan_repository", "tree_revision",
    "validate_batch",
]
