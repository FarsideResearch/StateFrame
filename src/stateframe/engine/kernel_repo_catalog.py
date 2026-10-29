"""Language-neutral structural catalog and external cache (CATC2).

This adapter consumes the shared OUTLINEGENERIC API and CATC0 handles.  It does
not execute source, import repository code, call a provider, or write inside the
    analysed repository. The cache contains immutable source bytes for ordinary
    span reads. Current-worktree relation rendering uses ``read_content``, which
    rechecks live bytes against this index generation.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import logging
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeVar

from kernel_catalog_contracts import (
    ContentHandle,
    CoverageRef,
    CoverageStatus,
    Namespace,
)
from mc_repo_intel_vendor.call_graph import build_graph, enforce_repo_cache_cap, repo_cache_file
from mc_repo_intel_vendor.ts_dispatch import outline_source
from kernel_measurement import profiled_function_span

logger = logging.getLogger(__name__)

SCHEMA = "catalog-structural-index/1"
CACHE_VERSION = 1
MAX_FILES = 512
MAX_OVERLAY_FILES = 32
MAX_INDEX_FILES = MAX_FILES + MAX_OVERLAY_FILES
MAX_NEIGHBORS = 32
MAX_CACHED_BODY_BYTES = 2 * 1024 * 1024
MAX_SCAN_ENTRIES = 32_768
_CACHE_FILENAME = ".stateframe_catalog_structural.json"
_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9._:/-]+")
_ASSIGNMENT_RE = re.compile(
    r"(?:^|\b)(?:const|let|var|static|final|readonly|shared|val|mut)?\s*"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$.-]*)\s*(?::[^=\n]+)?(?<![=!<>])=(?!=)"
)
_ENV_RE = re.compile(r"\b(?:getenv|get|setdefault)\s*\(\s*[\"'](?P<key>[A-Za-z_][A-Za-z0-9_]*)")
_SOURCE_SUFFIXES = frozenset({
    ".py", ".go", ".rs", ".ts", ".mts", ".cts", ".tsx", ".js", ".mjs", ".cjs", ".jsx",
    ".java", ".c", ".h", ".cc", ".cp", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".cs",
    ".rb", ".php", ".vue", ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".env", ".properties", ".xml", ".sql",
})
_SOURCE_FILENAMES = frozenset({"Dockerfile", "Makefile", "CMakeLists.txt", "go.mod", "Cargo.toml"})
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", "env", "dist",
    "build", ".tox", "coverage", ".pytest_cache", ".mypy_cache", ".eggs", ".mission",
    "grader", "graders", "gold", "private", "hidden", "hidden_tests", "backups",
})


class StructuralIndexError(ValueError):
    """Raised when a structural index request cannot be admitted safely."""


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_id(value: object, prefix: str) -> str:
    text = _ID_SAFE_RE.sub("_", str(value or "").replace("\\", "/")).strip("._/") or "unknown"
    return f"{prefix}:{text}"


def _namespace_project(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "repo"
    if not text[0].isalpha():
        text = "repo_" + text
    return text[:240]


def _relative(root: Path, path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        candidate = candidate.resolve()
        try:
            candidate = candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise StructuralIndexError("source path escapes repository root") from exc
    normalized = candidate.as_posix()
    if not normalized or normalized == "." or ".." in Path(normalized).parts:
        raise StructuralIndexError("source path must be a normalized relative path")
    if _ignored_scan_path(Path(normalized).parts):
        raise StructuralIndexError("source path is outside the public structural scope")
    return normalized


def _ignored_scan_path(parts: Sequence[str]) -> bool:
    return any(part.casefold() in _SKIP_DIRS or part.startswith(".") for part in parts)


def _candidate_name(path: Path) -> bool:
    return path.name in _SOURCE_FILENAMES or path.suffix.casefold() in _SOURCE_SUFFIXES


def _candidate(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and _candidate_name(path)


@dataclass(frozen=True, slots=True)
class _ScanInventory:
    source_paths: tuple[str, ...]
    entries: tuple[tuple[str, str], ...] | None


def _scan_inventory(root: Path) -> _ScanInventory:
    """Enumerate only source candidates and directories the catalog descends into."""
    source_paths: list[str] = []
    entries: list[tuple[str, str]] = []
    stack: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath())]
    complete = True
    within_limit = True
    entry_count = 0
    while stack:
        directory, relative_directory = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda item: item.name)
        except OSError:
            complete = False
            continue
        for child in children:
            relative = relative_directory / child.name
            if _ignored_scan_path(relative.parts):
                continue
            path = Path(child.path)
            try:
                mode = child.stat(follow_symlinks=False).st_mode
            except OSError:
                complete = False
                continue
            if stat.S_ISDIR(mode):
                kind = "directory"
                stack.append((path, relative))
            elif stat.S_ISREG(mode) and _candidate_name(path):
                # Reuse the catalog's candidate predicate so validation and indexing
                # cannot silently drift to separate source-file rules.
                if not _candidate(path):
                    complete = False
                    continue
                kind = "source"
                source_paths.append(relative.as_posix())
            else:
                continue
            entry_count += 1
            if entry_count > MAX_SCAN_ENTRIES:
                within_limit = False
                entries.clear()
            elif within_limit:
                entries.append((relative.as_posix(), kind))
    exact_entries = (
        tuple(sorted(entries)) if complete and within_limit else None
    )
    return _ScanInventory(tuple(sorted(source_paths)), exact_entries)


def scan_entry_set(root: Path) -> tuple[tuple[str, str], ...] | None:
    """Return the bounded source/directory inventory used to validate a catalog scan."""
    return _scan_inventory(Path(root)).entries


@dataclass(frozen=True, slots=True)
class _ScanResult:
    paths: tuple[str, ...]
    candidate_files: int | None
    omitted_files: int
    priority_files_omitted: int
    scope: str
    scan_entries: tuple[tuple[str, str], ...] | None = None


def _scan(
    root: Path,
    *,
    focus_paths: Sequence[str] = (),
    priority_paths: Sequence[str] = (),
    max_files: int = MAX_FILES,
    capture_scan_entries: bool = False,
) -> _ScanResult:
    limit = max(1, min(int(max_files), MAX_FILES))
    if focus_paths:
        if priority_paths:
            raise StructuralIndexError("focus and priority paths cannot be combined")
        values = tuple(sorted({_relative(root, item) for item in focus_paths}))
        for value in values:
            if not _candidate(root / value):
                raise StructuralIndexError(f"focus path is not a supported regular source file: {value}")
        return _ScanResult(values, None, 0, 0, "focused")

    priorities: set[str] = set()
    for item in priority_paths:
        try:
            value = _relative(root, item)
        except StructuralIndexError:
            continue
        if _candidate(root / value):
            priorities.add(value)

    inventory = _scan_inventory(root)
    found = list(inventory.source_paths)

    if len(found) > limit:
        canonical = tuple(sorted(found))
        base = canonical[:limit]
        base_set = set(base)
        overlay = tuple(sorted(priorities.intersection(found).difference(base_set)))
        if len(overlay) > MAX_OVERLAY_FILES or len(base) + len(overlay) > MAX_INDEX_FILES:
            # Preserve the legacy priority-first result when a request exceeds the
            # explicit overlay bound; callers can report the fallback reason.
            ordered = sorted(priorities) + [path for path in canonical if path not in priorities]
            selected = tuple(ordered[:limit])
            scope = "overlay_limit"
        else:
            selected = tuple(sorted(base_set.union(overlay)))
            scope = "truncated_overlay" if overlay else "truncated"
    else:
        selected = tuple(sorted(found))
        scope = "full"
    omitted = max(0, len(found) - len(selected))
    return _ScanResult(
        selected,
        len(found),
        omitted,
        len(priorities.difference(selected)),
        scope,
        inventory.entries if capture_scan_entries else None,
    )


@functools.lru_cache(maxsize=64)
def _line_offsets(body: bytes) -> tuple[int, ...]:
    """Cumulative byte offsets of splitlines(keepends=True): offsets[i] = start of line i+1.

    Host-time fix (r2AC): the structural index called _line_range once per span and
    re-split + re-summed the whole file each time (quadratic; ~68s per request on
    benchmark). bytes cache their hash, so this per-body table is computed once. Output is
    byte-identical to the original sum-over-splitlines computation.
    """
    offsets = [0]
    for line in body.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return tuple(offsets)


def _line_count(body: bytes) -> int:
    return max(1, (len(_line_offsets(body)) - 1) or (1 if body else 0))


def _line_range(body: bytes, start_line: int, end_line: int) -> tuple[int, int]:
    if start_line < 1 or end_line < start_line:
        raise StructuralIndexError("span line range is invalid")
    offsets = _line_offsets(body)
    line_total = len(offsets) - 1
    if not line_total and start_line == end_line == 1:
        return 0, 0
    if end_line > line_total:
        raise StructuralIndexError("span line range is outside source content")
    return offsets[start_line - 1], offsets[end_line]


def _record_lines(record: Mapping[str, object]) -> tuple[int, int]:
    start = record.get("line", 1)
    end = record.get("end_line", start)
    return (int(start) if isinstance(start, int) else 1, int(end) if isinstance(end, int) else int(start))


def _coverage(content: ContentHandle, outline: Mapping[str, object], *, kind: str = "structural_index") -> CoverageRef:
    degraded = bool(outline.get("degraded"))
    status = CoverageStatus.PARTIAL if degraded else CoverageStatus.COMPLETE
    return CoverageRef(
        f"coverage:{kind}:{content.content_id}",
        kind,
        status,
        inspected=content.byte_length,
        expected=content.byte_length,
    )


@dataclass(frozen=True, slots=True)
class SpanRecord:
    span_id: str
    content_id: str
    path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    byte_start: int
    byte_end: int
    body_digest: str
    coverage: CoverageRef

    def as_dict(self) -> dict[str, object]:
        return {
            "span_id": self.span_id, "content_id": self.content_id, "path": self.path,
            "kind": self.kind, "name": self.name, "start_line": self.start_line,
            "end_line": self.end_line, "byte_start": self.byte_start, "byte_end": self.byte_end,
            "body_digest": self.body_digest, "coverage": self.coverage.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SpanRecord":
        coverage = value.get("coverage")
        if not isinstance(coverage, Mapping):
            raise StructuralIndexError("cached span coverage is malformed")
        return cls(
            str(value["span_id"]), str(value["content_id"]), str(value["path"]), str(value["kind"]),
            str(value["name"]), int(value["start_line"]), int(value["end_line"]),
            int(value["byte_start"]), int(value["byte_end"]), str(value["body_digest"]),
            CoverageRef.from_dict(coverage),
        )


@dataclass(frozen=True, slots=True)
class ModuleRecord:
    module_id: str
    path: str
    language: str
    parser: str
    degraded: bool
    content: ContentHandle
    coverage: CoverageRef
    imports: tuple[str, ...] = ()
    statements: tuple[str, ...] = ()
    symbol_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "module_id": self.module_id, "path": self.path, "language": self.language,
            "parser": self.parser, "degraded": self.degraded, "content": self.content.to_dict(),
            "coverage": self.coverage.to_dict(), "imports": list(self.imports),
            "statements": list(self.statements), "symbol_ids": list(self.symbol_ids),
        }


@dataclass(frozen=True, slots=True)
class TypeRecord:
    type_id: str
    module_id: str
    name: str
    kind: str
    span_id: str
    coverage: CoverageRef
    attributes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "type_id": self.type_id, "module_id": self.module_id, "name": self.name,
            "kind": self.kind, "span_id": self.span_id, "coverage": self.coverage.to_dict(),
            "attributes": list(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class ConfigRecord:
    config_id: str
    module_id: str
    key: str
    span_id: str
    declaration_kind: str
    coverage: CoverageRef

    def as_dict(self) -> dict[str, object]:
        return {
            "config_id": self.config_id, "module_id": self.module_id, "key": self.key,
            "span_id": self.span_id, "declaration_kind": self.declaration_kind,
            "coverage": self.coverage.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DependencyRecord:
    source_id: str
    target_id: str
    relation: str
    distance: int
    coverage: CoverageRef

    def as_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id, "target_id": self.target_id, "relation": self.relation,
            "distance": self.distance, "coverage": self.coverage.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class NeighborhoodResult:
    source_id: str
    records: tuple[DependencyRecord, ...]
    truncated: bool
    coverage: CoverageRef

    def as_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id, "records": [item.as_dict() for item in self.records],
            "truncated": self.truncated, "coverage": self.coverage.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class IndexStats:
    source_reads: int
    cache_hits: int
    cache_misses: int
    rebuilt_files: tuple[str, ...]
    neighborhoods_truncated: int
    candidate_files: int | None = None
    indexed_files: int = 0
    omitted_files: int = 0
    priority_files_omitted: int = 0
    scan_scope: str = "full"

    def as_dict(self) -> dict[str, object]:
        return {
            "source_reads": self.source_reads, "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses, "rebuilt_files": list(self.rebuilt_files),
            "neighborhoods_truncated": self.neighborhoods_truncated,
            "candidate_files": self.candidate_files, "indexed_files": self.indexed_files,
            "omitted_files": self.omitted_files,
            "priority_files_omitted": self.priority_files_omitted,
            "scan_scope": self.scan_scope,
        }


@dataclass(frozen=True, slots=True)
class SpanRead:
    span: SpanRecord
    content: ContentHandle
    body: bytes
    from_cache: bool


class StructuralIndex:
    """Bounded records plus exact content handles for one index generation."""

    def __init__(
        self,
        *,
        root: Path,
        namespace: Namespace,
        modules: Sequence[ModuleRecord],
        configs: Sequence[ConfigRecord],
        types: Sequence[TypeRecord],
        spans: Sequence[SpanRecord],
        dependencies: Sequence[DependencyRecord],
        coverage: Mapping[str, CoverageRef],
        contents: Mapping[str, ContentHandle],
        bodies: Mapping[str, bytes],
        outlines: Mapping[str, Mapping[str, object]],
        cache_path: Path,
        stats: IndexStats,
        scan_entries: Sequence[tuple[str, str]] | None = None,
        scan_stable: bool = False,
    ) -> None:
        self.root = root
        self.namespace = namespace
        self.modules = tuple(modules)
        self.configs = tuple(configs)
        self.types = tuple(types)
        self.spans = tuple(spans)
        self.dependencies = tuple(dependencies)
        self.coverage = dict(coverage)
        self.contents = dict(contents)
        self._bodies = dict(bodies)
        self.outlines = {key: dict(value) for key, value in outlines.items()}
        self.cache_path = cache_path
        self.stats = stats
        self.scan_entries = tuple(scan_entries) if scan_entries is not None else None
        self.scan_stable = bool(scan_stable)
        self._spans_by_id = {item.span_id: item for item in self.spans}

    def module(self, path: str) -> ModuleRecord | None:
        normalized = _relative(self.root, path)
        return next((item for item in self.modules if item.path == normalized), None)

    def file_view(self, path: str) -> dict[str, object]:
        module = self.module(path)
        if module is None:
            raise StructuralIndexError(f"source file is not indexed: {path}")
        module_spans = tuple(item for item in self.spans if item.path == module.path)
        type_rows = tuple(item for item in self.types if item.module_id == module.module_id)
        config_rows = tuple(item for item in self.configs if item.module_id == module.module_id)
        dependency_rows = tuple(item for item in self.dependencies if item.source_id == module.module_id)
        view = dict(self.outlines.get(module.path, {}))
        view.update({
            "schema": "mc-outline/1", "catalog_schema": SCHEMA,
            "content_handle": module.content.to_dict(), "coverage_handle": module.coverage.to_dict(),
            "module": module.as_dict(),
            "catalog": {
                "types": [item.as_dict() for item in type_rows],
                "configs": [item.as_dict() for item in config_rows],
                "spans": [item.as_dict() for item in module_spans],
                "dependencies": [item.as_dict() for item in dependency_rows],
                "stats": self.stats.as_dict(),
            },
        })
        return view

    def neighbors(self, source_id: str, *, limit: int = MAX_NEIGHBORS) -> NeighborhoodResult:
        bound = max(1, min(int(limit), MAX_NEIGHBORS))
        rows = tuple(item for item in self.dependencies if item.source_id == source_id)
        coverage = rows[0].coverage if rows else CoverageRef(
            f"coverage:neighborhood:{_digest_json(source_id)}", "dependency_neighborhood", CoverageStatus.UNKNOWN,
        )
        return NeighborhoodResult(source_id, rows[:bound], len(rows) > bound, coverage)

    def read_span(self, span_id: str) -> SpanRead:
        try:
            span = self._spans_by_id[span_id]
        except KeyError as exc:
            raise StructuralIndexError(f"unknown source span: {span_id}") from exc
        content = self.contents[span.content_id]
        body = self._bodies.get(content.content_id)
        from_cache = body is not None
        if body is None:
            target = self.root / span.path
            try:
                body = target.read_bytes()
            except OSError as exc:
                raise StructuralIndexError(f"cached source body is unavailable: {span.path}") from exc
            if _digest_bytes(body) != content.content_digest:
                raise StructuralIndexError("source changed since index generation; refresh is required")
        exact = body[span.byte_start : span.byte_end]
        if _digest_bytes(exact) != span.body_digest:
            raise StructuralIndexError("source span digest does not match its content handle")
        return SpanRead(span, content, exact, from_cache)

    def read_content(self, path: str, *, max_bytes: int | None = None) -> tuple[ContentHandle, bytes]:
        """Return one exact, digest-checked file body from this index generation."""
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise StructuralIndexError("source read limit must be a nonnegative integer")
        normalized = _relative(self.root, path)
        module = self.module(normalized)
        if module is None:
            raise StructuralIndexError(f"source file is not indexed: {path}")
        if max_bytes is not None and module.content.byte_length > max_bytes:
            raise StructuralIndexError("indexed source exceeds read limit")
        target = self.root / normalized
        if target.is_symlink():
            raise StructuralIndexError("source file is a symlink")
        resolved = target.resolve()
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError as exc:
            raise StructuralIndexError("source file escapes repository root") from exc
        try:
            if max_bytes is None:
                body = resolved.read_bytes()
            else:
                with resolved.open("rb") as stream:
                    body = stream.read(max_bytes + 1)
        except OSError as exc:
            raise StructuralIndexError(f"source body is unavailable: {normalized}") from exc
        if max_bytes is not None and len(body) > max_bytes:
            raise StructuralIndexError("current source exceeds read limit")
        if _digest_bytes(body) != module.content.content_digest:
            raise StructuralIndexError("source changed since index generation; refresh is required")
        return module.content, body

    def read_source_span(self, path: str, start_line: int, end_line: int) -> SpanRead:
        normalized = _relative(self.root, path)
        candidates = tuple(item for item in self.spans if item.path == normalized
                           and item.start_line == start_line and item.end_line == end_line)
        if not candidates:
            raise StructuralIndexError("requested exact source span is not indexed")
        return self.read_span(candidates[0].span_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA, "namespace": self.namespace.to_dict(),
            "modules": [item.as_dict() for item in self.modules],
            "types": [item.as_dict() for item in self.types],
            "configs": [item.as_dict() for item in self.configs],
            "spans": [item.as_dict() for item in self.spans],
            "dependencies": [item.as_dict() for item in self.dependencies],
            "coverage": {key: value.to_dict() for key, value in sorted(self.coverage.items())},
            "stats": self.stats.as_dict(), "cache_path": str(self.cache_path),
        }


def _span(
    *,
    body: bytes,
    content: ContentHandle,
    coverage: CoverageRef,
    path: str,
    kind: str,
    name: str,
    start_line: int,
    end_line: int,
) -> SpanRecord:
    byte_start, byte_end = _line_range(body, start_line, end_line)
    identity = {"content": content.content_id, "kind": kind, "name": name,
                "start_line": start_line, "end_line": end_line}
    span_id = "span:" + _digest_json(identity)
    return SpanRecord(
        span_id, content.content_id, path, kind, name, start_line, end_line,
        byte_start, byte_end, _digest_bytes(body[byte_start:byte_end]), coverage,
    )


def _outline_span_records(
    *,
    body: bytes,
    content: ContentHandle,
    coverage: CoverageRef,
    path: str,
    outline: Mapping[str, object],
) -> tuple[tuple[SpanRecord, ...], tuple[TypeRecord, ...], tuple[ConfigRecord, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    module_id = _safe_id(f"{content.namespace.workspace_id}:{path}", "module")
    spans: list[SpanRecord] = [_span(
        body=body, content=content, coverage=coverage, path=path, kind="module", name=path,
        start_line=1, end_line=_line_count(body),
    )]
    types: list[TypeRecord] = []
    configs: list[ConfigRecord] = []
    symbol_ids: list[str] = []
    imports: list[str] = []
    statements: list[str] = []

    def records(key: str) -> tuple[Mapping[str, object], ...]:
        raw = outline.get(key, ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            return ()
        return tuple(item for item in raw if isinstance(item, Mapping))

    for item in records("imports"):
        module = item.get("module")
        names = item.get("names")
        values = [str(module)] if module not in (None, "") else []
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes, bytearray)):
            values.extend(str(value) for value in names if str(value).strip())
        imports.extend(value for value in values if value not in imports)
        start, end = _record_lines(item)
        spans.append(_span(body=body, content=content, coverage=coverage, path=path,
                           kind="import", name="|".join(values) or "import", start_line=start, end_line=end))

    for key, span_kind in (("types", "type"), ("functions", "function"), ("module_statements", "module_statement")):
        for item in records(key):
            name = str(item.get("name") or item.get("expression") or item.get("kind") or span_kind)
            start, end = _record_lines(item)
            current = _span(body=body, content=content, coverage=coverage, path=path,
                            kind=span_kind, name=name, start_line=start, end_line=end)
            spans.append(current)
            if key == "types":
                type_id = _safe_id(f"{module_id}:{name}:{start}", "type")
                raw_methods = item.get("methods")
                attributes = tuple(
                    str(method.get("name")) for method in raw_methods
                    if isinstance(method, Mapping) and method.get("name") not in (None, "")
                ) if isinstance(raw_methods, Sequence) and not isinstance(raw_methods, (str, bytes, bytearray)) else ()
                types.append(TypeRecord(type_id, module_id, name, str(item.get("kind") or "type"), current.span_id, coverage, attributes))
                symbol_ids.append(type_id)
            elif key == "functions":
                symbol_ids.append(_safe_id(f"{module_id}:{name}:{start}", "symbol"))
            else:
                statements.append(current.span_id)
                source = str(item.get("source") or "")
                for match in _ASSIGNMENT_RE.finditer(source):
                    key_name = match.group("name")
                    config_id = _safe_id(f"{module_id}:{key_name}:{start}", "config")
                    configs.append(ConfigRecord(config_id, module_id, key_name, current.span_id, "assignment", coverage))
                for match in _ENV_RE.finditer(source):
                    key_name = match.group("key")
                    config_id = _safe_id(f"{module_id}:{key_name}:{start}", "config")
                    configs.append(ConfigRecord(config_id, module_id, key_name, current.span_id, "environment", coverage))

    # Config formats are not all parser-backed, but key locations can still be
    # represented without pretending an unsupported parser was complete.
    text = body.decode(content.encoding, errors="replace")
    for line_number, line in enumerate(text.splitlines(), 1):
        for match in re.finditer(r"^\s*[\"']?([A-Za-z_][A-Za-z0-9_.-]*)[\"']?\s*[:=]", line):
            key_name = match.group(1)
            if any(item.key == key_name and item.module_id == module_id for item in configs):
                continue
            current = _span(body=body, content=content, coverage=coverage, path=path,
                            kind="config", name=key_name, start_line=line_number, end_line=line_number)
            spans.append(current)
            configs.append(ConfigRecord(_safe_id(f"{module_id}:{key_name}:{line_number}", "config"), module_id,
                                        key_name, current.span_id, "config", coverage))
    return tuple(spans), tuple(types), tuple(configs), tuple(imports), tuple(statements), tuple(symbol_ids)


def _cache_fallback_bases() -> tuple[Path, ...]:
    candidates: list[Path] = []
    try:
        candidates.append(Path.home() / ".cache" / "stateframe" / "repo-intel")
    except (OSError, RuntimeError):
        pass
    try:
        candidates.append(Path(tempfile.gettempdir()) / "stateframe" / "repo-intel")
    except (OSError, RuntimeError):
        pass
    candidates.append(Path("/private/tmp/stateframe/repo-intel"))

    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            path = candidate.expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if path not in seen:
            seen.add(path)
            resolved.append(path)
    return tuple(resolved)


def _external_cache_target(base: Path, root: Path, key: str) -> Path | None:
    try:
        base.relative_to(root)
    except ValueError:
        pass
    else:
        return None
    target = base / key
    target.mkdir(parents=True, exist_ok=True)
    try:
        os.utime(target, None)
    except OSError:
        pass
    enforce_repo_cache_cap(base, keep=target)
    return target / _CACHE_FILENAME


def _cache_path(root: Path) -> Path:
    resolved_root = Path(root).resolve()
    try:
        return repo_cache_file(resolved_root, filename=_CACHE_FILENAME)
    except (OSError, RuntimeError, ValueError) as initial_error:
        key = hashlib.sha256(str(resolved_root).encode("utf-8")).hexdigest()[:16]
        last_error: BaseException = initial_error
        for base in _cache_fallback_bases():
            try:
                fallback = _external_cache_target(base, resolved_root, key)
                if fallback is None:
                    continue
                logger.debug(
                    "structural cache selected fallback after %s",
                    type(initial_error).__name__,
                )
                return fallback
            except (OSError, RuntimeError, ValueError) as fallback_error:
                last_error = fallback_error

        raise StructuralIndexError("external structural cache directory is unavailable") from last_error


def _load_cache(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, Mapping) or raw.get("schema") != SCHEMA or raw.get("version") != CACHE_VERSION:
        return {}
    files = raw.get("files")
    return dict(files) if isinstance(files, Mapping) else {}


def _cached_entry(value: object, *, body: bytes, namespace: Namespace, path: str) -> tuple[ContentHandle, CoverageRef, Mapping[str, object], bytes] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        content_raw = value.get("content")
        coverage_raw = value.get("coverage")
        outline = value.get("outline")
        encoded = value.get("body_base64")
        if not isinstance(content_raw, Mapping) or not isinstance(coverage_raw, Mapping) or not isinstance(outline, Mapping):
            return None
        content = ContentHandle.from_dict(content_raw)
        if content.namespace != namespace or content.source_ref != path or content.content_digest != _digest_bytes(body):
            return None
        coverage = CoverageRef.from_dict(coverage_raw)
        if not isinstance(encoded, str):
            return None
        cached_body = base64.b64decode(encoded, validate=True)
        if cached_body != body:
            return None
        return content, coverage, outline, cached_body
    except (TypeError, ValueError, KeyError, StructuralIndexError):
        return None


def _save_cache(path: Path, files: Mapping[str, object]) -> None:
    payload = {"schema": SCHEMA, "version": CACHE_VERSION, "files": dict(files)}
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        logger.debug("structural cache unavailable at %s: %s", path, type(exc).__name__)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _graph_dependencies(root: Path, modules: Sequence[ModuleRecord], coverage: Mapping[str, CoverageRef]) -> tuple[DependencyRecord, ...]:
    try:
        graph = build_graph(str(root), use_cache=True, use_memo=True)
    except Exception as exc:  # noqa: BLE001 - graph is advisory, outline remains authoritative
        logger.debug("structural dependency graph unavailable: %s", type(exc).__name__)
        return ()
    records: list[DependencyRecord] = []
    for source_id in sorted(getattr(graph, "defs", {})):
        try:
            refs = graph.find_callees(source_id)
        except Exception:  # noqa: BLE001 - one malformed edge cannot drop index records
            continue
        source_path = source_id.split("::", 1)[0]
        source_coverage = coverage.get(source_path)
        if source_coverage is None:
            continue
        for ref in list(refs)[:MAX_NEIGHBORS]:
            target_id = str(getattr(ref, "symbol", ref))
            records.append(DependencyRecord(source_id, target_id, "CALLS", 1, source_coverage))
    return tuple(records)


@profiled_function_span("structural_index_build")
def build_structural_index(
    root: Path,
    *,
    namespace: Namespace | None = None,
    focus_paths: Sequence[str] = (),
    priority_paths: Sequence[str] = (),
    max_files: int = MAX_FILES,
    source_overrides: Mapping[str, bytes | str] | None = None,
    include_call_graph: bool = True,
    capture_scan_entries: bool = False,
) -> StructuralIndex:
    """Build or refresh a bounded language-neutral index for ``root``."""

    resolved = Path(root).resolve()
    if not resolved.is_dir() or resolved.name.casefold() in _SKIP_DIRS:
        raise StructuralIndexError("repository root is not a public directory")
    active_namespace = namespace or Namespace("workspace:repo-intel", _namespace_project(resolved.name))
    scan = _scan(
        resolved, focus_paths=focus_paths, priority_paths=priority_paths, max_files=max_files,
        capture_scan_entries=capture_scan_entries,
    )
    paths = scan.paths
    overrides = {str(key).replace("\\", "/"): value for key, value in (source_overrides or {}).items()}
    cache_path = _cache_path(resolved)
    persisted = _load_cache(cache_path) if not overrides else {}
    cache_files: dict[str, object] = dict(persisted)
    modules: list[ModuleRecord] = []
    configs: list[ConfigRecord] = []
    types: list[TypeRecord] = []
    spans: list[SpanRecord] = []
    contents: dict[str, ContentHandle] = {}
    bodies: dict[str, bytes] = {}
    outlines: dict[str, Mapping[str, object]] = {}
    coverage: dict[str, CoverageRef] = {}
    cache_hits = 0
    cache_misses = 0
    rebuilt: list[str] = []
    source_reads = 0

    for relpath in paths:
        override = overrides.get(relpath)
        if override is None:
            try:
                body = (resolved / relpath).read_bytes()
            except OSError as exc:
                logger.debug("structural source read failed for %s: %s", relpath, type(exc).__name__)
                continue
            source_reads += 1
        else:
            body = override.encode("utf-8") if isinstance(override, str) else bytes(override)
        cached = _cached_entry(persisted.get(relpath), body=body, namespace=active_namespace, path=relpath)
        if cached is not None:
            content, file_coverage, outline, cached_body = cached
            cache_hits += 1
            body = cached_body
        else:
            cache_misses += 1
            rebuilt.append(relpath)
            content = ContentHandle.capture(body, namespace=active_namespace, source_ref=relpath)
            try:
                outline = outline_source(body.decode(content.encoding, errors="replace"), relpath)
            except Exception as exc:  # noqa: BLE001 - shared outline is non-throwing by contract
                outline = {"schema": "mc-outline/1", "file": relpath, "language": "unknown",
                           "parser": "unavailable", "degraded": True, "degraded_reason": type(exc).__name__}
            file_coverage = _coverage(content, outline)
            if not overrides and len(body) <= MAX_CACHED_BODY_BYTES:
                cache_files[relpath] = {
                    "content": content.to_dict(), "coverage": file_coverage.to_dict(),
                    "outline": dict(outline), "body_base64": base64.b64encode(body).decode("ascii"),
                }
        module_id = _safe_id(f"{active_namespace.workspace_id}:{relpath}", "module")
        file_spans, file_types, file_configs, imports, statements, symbol_ids = _outline_span_records(
            body=body, content=content, coverage=file_coverage, path=relpath, outline=outline,
        )
        modules.append(ModuleRecord(
            module_id, relpath, str(outline.get("language") or "unknown"), str(outline.get("parser") or "unknown"),
            bool(outline.get("degraded")), content, file_coverage, imports, statements, symbol_ids,
        ))
        contents[content.content_id] = content
        bodies[content.content_id] = body
        coverage[relpath] = file_coverage
        outlines[relpath] = outline
        spans.extend(file_spans)
        types.extend(file_types)
        configs.extend(file_configs)

    if not overrides:
        _save_cache(cache_path, cache_files)
    dependencies: list[DependencyRecord] = []
    for module in modules:
        dependency_coverage = CoverageRef(
            f"coverage:module-neighborhood:{module.content.content_id}", "dependency_neighborhood",
            module.coverage.status,
        )
        for imported in module.imports:
            dependencies.append(DependencyRecord(module.module_id, _safe_id(imported, "import"), "IMPORTS", 1, dependency_coverage))
    if include_call_graph and not focus_paths:
        dependencies.extend(_graph_dependencies(resolved, modules, coverage))
    grouped: dict[str, list[DependencyRecord]] = {}
    for item in dependencies:
        grouped.setdefault(item.source_id, []).append(item)
    neighborhoods_truncated = sum(len(rows) > MAX_NEIGHBORS for rows in grouped.values())
    dependencies = [item for source in sorted(grouped) for item in grouped[source][:MAX_NEIGHBORS]]
    indexed_paths = {module.path for module in modules}
    scan_stable = bool(
        capture_scan_entries
        and scan.scan_entries is not None
        and indexed_paths == set(paths)
        and scan_entry_set(resolved) == scan.scan_entries
    )
    return StructuralIndex(
        root=resolved, namespace=active_namespace, modules=modules, configs=configs, types=types,
        spans=spans, dependencies=dependencies, coverage=coverage, contents=contents, bodies=bodies,
        outlines=outlines,
        cache_path=cache_path,
        stats=IndexStats(
            source_reads, cache_hits, cache_misses, tuple(rebuilt), neighborhoods_truncated,
            candidate_files=scan.candidate_files, indexed_files=len(modules),
            omitted_files=scan.omitted_files,
            priority_files_omitted=scan.priority_files_omitted,
            scan_scope=scan.scope,
        ),
        scan_entries=scan.scan_entries,
        scan_stable=scan_stable,
    )


def merge_structural_index(previous: StructuralIndex, delta: StructuralIndex) -> StructuralIndex:
    """Replace changed file records while preserving the unchanged index generation."""
    if previous.root != delta.root or previous.namespace.to_dict() != delta.namespace.to_dict():
        raise StructuralIndexError("incremental structural indexes have different scopes")
    previous_modules = {module.path: module for module in previous.modules}
    delta_modules = {module.path: module for module in delta.modules}
    changed_paths = set(delta_modules)
    if not changed_paths or not changed_paths.issubset(previous_modules):
        raise StructuralIndexError("incremental structural index changed the selected path set")
    changed_module_ids = {previous_modules[path].module_id for path in changed_paths}
    if any(delta_modules[path].module_id != previous_modules[path].module_id for path in changed_paths):
        raise StructuralIndexError("incremental structural index changed a module identity")

    modules = tuple(delta_modules.get(module.path, module) for module in previous.modules)

    def merge_by_module(old_rows, new_rows, row_key, module_key):
        old_grouped: dict[str, list[object]] = {}
        new_grouped: dict[str, list[object]] = {}
        for row in old_rows:
            old_grouped.setdefault(row_key(row), []).append(row)
        for row in new_rows:
            new_grouped.setdefault(row_key(row), []).append(row)
        result = []
        for module in modules:
            selected = new_grouped if module.path in changed_paths else old_grouped
            result.extend(selected.get(module_key(module), ()))
        return tuple(result)

    configs = merge_by_module(
        previous.configs, delta.configs, lambda row: row.module_id, lambda module: module.module_id,
    )
    types = merge_by_module(
        previous.types, delta.types, lambda row: row.module_id, lambda module: module.module_id,
    )
    spans = merge_by_module(previous.spans, delta.spans, lambda row: row.path, lambda module: module.path)

    old_dependencies: dict[str, list[DependencyRecord]] = {}
    new_dependencies: dict[str, list[DependencyRecord]] = {}
    for row in previous.dependencies:
        old_dependencies.setdefault(row.source_id, []).append(row)
    for row in delta.dependencies:
        new_dependencies.setdefault(row.source_id, []).append(row)
    source_ids = set(old_dependencies) | set(new_dependencies) | changed_module_ids
    dependencies = tuple(
        row
        for source_id in sorted(source_ids)
        for row in (new_dependencies if source_id in changed_module_ids else old_dependencies).get(
            source_id, ()
        )
    )

    coverage = dict(previous.coverage)
    coverage.update(delta.coverage)
    contents = dict(previous.contents)
    bodies = dict(previous._bodies)
    outlines = dict(previous.outlines)
    for path in changed_paths:
        old_content_id = previous_modules[path].content.content_id
        contents.pop(old_content_id, None)
        bodies.pop(old_content_id, None)
        outlines.pop(path, None)
    contents.update(delta.contents)
    bodies.update(delta._bodies)
    outlines.update(delta.outlines)
    stats = IndexStats(
        previous.stats.source_reads + delta.stats.source_reads,
        previous.stats.cache_hits + delta.stats.cache_hits,
        previous.stats.cache_misses + delta.stats.cache_misses,
        tuple(dict.fromkeys((*previous.stats.rebuilt_files, *delta.stats.rebuilt_files))),
        previous.stats.neighborhoods_truncated,
        candidate_files=previous.stats.candidate_files,
        indexed_files=len(modules),
        omitted_files=previous.stats.omitted_files,
        priority_files_omitted=previous.stats.priority_files_omitted,
        scan_scope=previous.stats.scan_scope,
    )
    return StructuralIndex(
        root=previous.root, namespace=previous.namespace, modules=modules, configs=configs,
        types=types, spans=spans, dependencies=dependencies, coverage=coverage,
        contents=contents, bodies=bodies, outlines=outlines,
        cache_path=previous.cache_path, stats=stats,
        scan_entries=previous.scan_entries,
        scan_stable=previous.scan_stable,
    )


def subset_structural_index(
    index: StructuralIndex,
    paths: Sequence[str],
    *,
    candidate_files: int | None,
    omitted_files: int,
    priority_files_omitted: int = 0,
    scan_scope: str = "focused",
    keep_scan_inventory: bool = False,
) -> StructuralIndex:
    """Return a path subset while retaining exact records and optional scan authority."""
    selected_paths = set(paths)
    modules = tuple(module for module in index.modules if module.path in selected_paths)
    module_ids = {module.module_id for module in modules}
    content_ids = {module.content.content_id for module in modules}
    configs = tuple(row for row in index.configs if row.module_id in module_ids)
    types = tuple(row for row in index.types if row.module_id in module_ids)
    spans = tuple(row for row in index.spans if row.path in selected_paths)
    dependencies = tuple(row for row in index.dependencies if row.source_id in module_ids)
    grouped: dict[str, list[DependencyRecord]] = {}
    for row in dependencies:
        grouped.setdefault(row.source_id, []).append(row)
    neighborhoods_truncated = sum(len(rows) > MAX_NEIGHBORS for rows in grouped.values())
    rebuilt_paths = set(index.stats.rebuilt_files)
    rebuilt = tuple(module.path for module in modules if module.path in rebuilt_paths)
    stats = IndexStats(
        source_reads=len(modules),
        cache_hits=max(0, len(modules) - len(rebuilt)),
        cache_misses=len(rebuilt),
        rebuilt_files=rebuilt,
        neighborhoods_truncated=neighborhoods_truncated,
        candidate_files=candidate_files,
        indexed_files=len(modules),
        omitted_files=max(0, omitted_files),
        priority_files_omitted=max(0, priority_files_omitted),
        scan_scope=scan_scope,
    )
    return StructuralIndex(
        root=index.root,
        namespace=index.namespace,
        modules=modules,
        configs=configs,
        types=types,
        spans=spans,
        dependencies=dependencies,
        coverage={path: value for path, value in index.coverage.items() if path in selected_paths},
        contents={key: value for key, value in index.contents.items() if key in content_ids},
        bodies={key: value for key, value in index._bodies.items() if key in content_ids},
        outlines={key: value for key, value in index.outlines.items() if key in selected_paths},
        cache_path=index.cache_path,
        stats=stats,
        scan_entries=index.scan_entries if keep_scan_inventory else None,
        scan_stable=index.scan_stable if keep_scan_inventory else False,
    )

_Row = TypeVar("_Row")


def _ordered_module_rows(
    modules: Sequence[ModuleRecord],
    rows: Sequence[_Row],
    owner: Callable[[_Row], str],
    module_key: Callable[[ModuleRecord], str],
) -> tuple[_Row, ...]:
    grouped: dict[str, list[_Row]] = {}
    for row in rows:
        grouped.setdefault(owner(row), []).append(row)
    return tuple(
        row for module in modules for row in grouped.get(module_key(module), ())
    )


def _ordered_dependencies(
    rows: Sequence[DependencyRecord],
    module_ids: set[str],
) -> tuple[tuple[DependencyRecord, ...], int]:
    grouped: dict[str, list[DependencyRecord]] = {}
    for row in rows:
        if row.source_id in module_ids:
            grouped.setdefault(row.source_id, []).append(row)
    truncated = sum(len(items) > MAX_NEIGHBORS for items in grouped.values())
    ordered = tuple(
        row for source_id in sorted(grouped)
        for row in grouped[source_id][:MAX_NEIGHBORS]
    )
    return ordered, truncated


def _combined_scan_scope(
    base: StructuralIndex,
    additions: Sequence[str],
    priority_paths: Sequence[str],
) -> tuple[int | None, int, int, str]:
    candidate_files = base.stats.candidate_files
    omitted_files = base.stats.omitted_files
    priority_files_omitted = base.stats.priority_files_omitted
    scan_scope = base.stats.scan_scope
    if type(candidate_files) is not int or base.scan_entries is None:
        return candidate_files, omitted_files, priority_files_omitted, scan_scope

    candidates = tuple(sorted(
        path for path, kind in base.scan_entries if kind == "source"
    ))
    base_count = max(0, candidate_files - base.stats.omitted_files)
    selected = set(candidates[:base_count])
    added_paths = set(additions)
    if not added_paths.issubset(candidates):
        raise StructuralIndexError("overlay path is outside the stable source inventory")
    selected.update(added_paths)

    requested = _requested_candidates(base.root, priority_paths, set(candidates))
    omitted_files = max(0, base.stats.omitted_files - len(added_paths))
    priority_files_omitted = len(requested.difference(selected))
    scan_scope = (
        "full" if omitted_files == 0
        else "truncated_overlay" if additions
        else "truncated"
    )
    return candidate_files, omitted_files, priority_files_omitted, scan_scope


def _requested_candidates(
    root: Path,
    priority_paths: Sequence[str],
    candidates: set[str],
) -> set[str]:
    requested: set[str] = set()
    for path in priority_paths:
        try:
            normalized = _relative(root, path)
        except StructuralIndexError:
            continue
        if normalized in candidates:
            requested.add(normalized)
    return requested


def _combined_scan_stats(
    base: StructuralIndex,
    overlay: StructuralIndex | None,
    modules: Sequence[ModuleRecord],
    additions: Sequence[str],
    priority_paths: Sequence[str],
    neighborhoods_truncated: int,
) -> IndexStats:
    candidate_files, omitted_files, priority_files_omitted, scan_scope = (
        _combined_scan_scope(base, additions, priority_paths)
    )
    rebuilt_paths = set(base.stats.rebuilt_files)
    if overlay is not None:
        rebuilt_paths.update(overlay.stats.rebuilt_files)
    rebuilt = tuple(module.path for module in modules if module.path in rebuilt_paths)
    return IndexStats(
        source_reads=base.stats.source_reads + (overlay.stats.source_reads if overlay else 0),
        cache_hits=base.stats.cache_hits + (overlay.stats.cache_hits if overlay else 0),
        cache_misses=base.stats.cache_misses + (overlay.stats.cache_misses if overlay else 0),
        rebuilt_files=rebuilt,
        neighborhoods_truncated=neighborhoods_truncated,
        candidate_files=candidate_files,
        indexed_files=len(modules),
        omitted_files=omitted_files,
        priority_files_omitted=priority_files_omitted,
        scan_scope=scan_scope,
    )


def combine_structural_indexes(
    base: StructuralIndex,
    overlay: StructuralIndex | None,
    *,
    overlay_paths: Sequence[str] = (),
    priority_paths: Sequence[str] = (),
) -> StructuralIndex:
    """Combine a stable base with bounded focus files in canonical path order."""
    if overlay is not None and (
        base.root != overlay.root or base.namespace.to_dict() != overlay.namespace.to_dict()
    ):
        raise StructuralIndexError("overlay structural index has a different scope")
    additions = tuple(sorted({_relative(base.root, path) for path in overlay_paths}))
    base_paths = {module.path for module in base.modules}
    if (
        len(base_paths) > MAX_FILES
        or len(additions) > MAX_OVERLAY_FILES
        or len(base_paths) + len(additions) > MAX_INDEX_FILES
    ):
        raise StructuralIndexError("combined structural index exceeds its file limit")
    if base_paths.intersection(additions):
        raise StructuralIndexError("overlay structural index overlaps its base")
    overlay_modules = tuple(overlay.modules) if overlay is not None else ()
    addition_set = set(additions)
    if {module.path for module in overlay_modules} != addition_set:
        raise StructuralIndexError("overlay structural index does not match requested paths")

    modules = tuple(sorted((*base.modules, *overlay_modules), key=lambda module: module.path))
    module_ids = {module.module_id for module in modules}
    configs = _ordered_module_rows(
        modules,
        (*base.configs, *(overlay.configs if overlay is not None else ())),
        lambda row: row.module_id,
        lambda module: module.module_id,
    )
    types = _ordered_module_rows(
        modules,
        (*base.types, *(overlay.types if overlay is not None else ())),
        lambda row: row.module_id,
        lambda module: module.module_id,
    )
    spans = _ordered_module_rows(
        modules,
        (*base.spans, *(overlay.spans if overlay is not None else ())),
        lambda row: row.path,
        lambda module: module.path,
    )
    dependencies, neighborhoods_truncated = _ordered_dependencies(
        (*base.dependencies, *(overlay.dependencies if overlay is not None else ())),
        module_ids,
    )

    content_ids = {module.content.content_id for module in modules}
    contents = dict(base.contents)
    bodies = dict(base._bodies)
    outlines = dict(base.outlines)
    coverage = dict(base.coverage)
    if overlay is not None:
        contents.update(overlay.contents)
        bodies.update(overlay._bodies)
        outlines.update(overlay.outlines)
        coverage.update(overlay.coverage)
    contents = {key: value for key, value in contents.items() if key in content_ids}
    bodies = {key: value for key, value in bodies.items() if key in content_ids}
    stats = _combined_scan_stats(
        base, overlay, modules, additions, priority_paths, neighborhoods_truncated,
    )
    return StructuralIndex(
        root=base.root,
        namespace=base.namespace,
        modules=modules,
        configs=configs,
        types=types,
        spans=spans,
        dependencies=dependencies,
        coverage=coverage,
        contents=contents,
        bodies=bodies,
        outlines=outlines,
        cache_path=base.cache_path,
        stats=stats,
        scan_entries=base.scan_entries,
        scan_stable=base.scan_stable,
    )


__all__ = [
    "CACHE_VERSION", "ConfigRecord", "DependencyRecord", "IndexStats", "MAX_INDEX_FILES",
    "MAX_NEIGHBORS", "MAX_OVERLAY_FILES", "MAX_SCAN_ENTRIES",
    "ModuleRecord", "NeighborhoodResult", "SCHEMA", "SpanRead", "SpanRecord", "StructuralIndex",
    "StructuralIndexError", "TypeRecord", "build_structural_index", "combine_structural_indexes",
    "merge_structural_index", "scan_entry_set", "subset_structural_index",
]
