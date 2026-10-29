"""Host-owned K1-K3 knowledge packet/index adapter.

The adapter refreshes from the existing language-neutral structural catalog and
projects bounded evidence into FRAMEBUILD.  It never executes repository code,
executes documentation instructions, calls a model, or treats a missing/stale
packet as permission to invent a fact.  Python graph/AST queries are isolated
to :class:`PythonAnalysisAdapter`; every other language uses ``mc-outline/1``
and explicit degraded/unknown coverage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

try:  # Flat v46 loader.
    from kernel_catalog_contracts import ContentHandle, Namespace
    from kernel_knowledge_contracts import (
        Applicability,
        canonical_digest,
        CoverageSnapshot,
        CoverageState,
        DependencyKind,
        DocumentationCandidate,
        DocumentationQuery,
        DocumentationResolution,
        DocumentationStatus,
        EvidenceLevel,
        IdiomQuery,
        IdiomSelection,
        IdiomSpec,
        IdiomStatus,
        KnowledgeContractError,
        KnowledgePacket,
        KnowledgeRecord,
        RecordKind,
        SourceSpanHandle,
        TaskKnowledgeView,
        Validity,
        ValidationCost,
    )
    from kernel_repo_catalog import StructuralIndex, StructuralIndexError, build_structural_index
    from kernel_repo_intel import host_where_defined, import_construction_fact, intake_orientation
except ImportError:  # Package importer.
    from .kernel_catalog_contracts import ContentHandle, Namespace
    from .kernel_knowledge_contracts import (
        Applicability,
        canonical_digest,
        CoverageSnapshot,
        CoverageState,
        DependencyKind,
        DocumentationCandidate,
        DocumentationQuery,
        DocumentationResolution,
        DocumentationStatus,
        EvidenceLevel,
        IdiomQuery,
        IdiomSelection,
        IdiomSpec,
        IdiomStatus,
        KnowledgeContractError,
        KnowledgePacket,
        KnowledgeRecord,
        RecordKind,
        SourceSpanHandle,
        TaskKnowledgeView,
        Validity,
        ValidationCost,
    )
    from .kernel_repo_catalog import StructuralIndex, StructuralIndexError, build_structural_index
    from .kernel_repo_intel import host_where_defined, import_construction_fact, intake_orientation

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = frozenset({
    ".py", ".go", ".rs", ".ts", ".mts", ".cts", ".tsx", ".js", ".mjs", ".cjs", ".jsx",
    ".java", ".c", ".h", ".cc", ".cp", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".cs",
    ".rb", ".php", ".vue", ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".env", ".properties", ".xml", ".sql",
})
_SOURCE_FILENAMES = frozenset({"Dockerfile", "Makefile", "CMakeLists.txt", "go.mod", "Cargo.toml"})
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", "env", "dist",
    "build", ".tox", "coverage", ".pytest_cache", ".mypy_cache", ".eggs", ".mission",
    ".stateframe", "grader", "graders", "gold", "private", "hidden", "hidden_tests", "backups",
})
_DOC_FORMATS = frozenset({"markdown", "text", "json"})
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_MIN_VIEW_BYTES = 4_096  # Envelope + two exact contradiction handles is the smallest useful view.
_EVIDENCE_RANK = {
    EvidenceLevel.CURRENT_USAGE: 0,
    EvidenceLevel.VERIFIED_PATTERN: 1,
    EvidenceLevel.UNVERIFIED: 2,
}


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _line_bounds(body: bytes, start_line: int, end_line: int) -> tuple[int, int]:
    lines = body.splitlines(keepends=True)
    if not lines and start_line == end_line == 1:
        return 0, 0
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        raise KnowledgeContractError("requested source span is outside current bytes")
    byte_start = sum(len(item) for item in lines[: start_line - 1])
    byte_end = byte_start + sum(len(item) for item in lines[start_line - 1 : end_line])
    return byte_start, byte_end


def _relative(root: Path, value: str | Path) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise KnowledgeContractError("path escapes repository root") from exc
    normalized = candidate.as_posix().lstrip("./")
    if not normalized or normalized == "." or ".." in Path(normalized).parts:
        raise KnowledgeContractError("path must be a normalized relative path")
    if any(part.casefold() in _SKIP_DIRS for part in Path(normalized).parts):
        raise KnowledgeContractError("path is outside public scope")
    return normalized


def _is_supported(path: str) -> bool:
    name = Path(path).name
    return name in _SOURCE_FILENAMES or Path(path).suffix.casefold() in _SUPPORTED_SUFFIXES


def _public_files(root: Path) -> tuple[str, ...]:
    """Walk without following directory or file symlinks."""

    found: list[str] = []
    for directory, directories, files in os.walk(root, followlinks=False):
        current = Path(directory)
        directories[:] = sorted(
            name for name in directories
            if name.casefold() not in _SKIP_DIRS and not (current / name).is_symlink()
        )
        for name in sorted(files):
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                found.append(path.relative_to(root).as_posix())
            except ValueError:
                continue
    return tuple(found)


def _root_fingerprint(root: Path) -> str:
    entries: list[dict[str, str]] = []
    for relpath in _public_files(root):
        try:
            body = (root / relpath).read_bytes()
        except (OSError, UnicodeError):
            continue
        entries.append({"path": relpath, "digest": _sha256_bytes(body)})
    return canonical_digest(entries)


def _coverage_from_catalog(value: object, *, reason: str | None = None) -> CoverageSnapshot:
    status = str(getattr(value, "status", "unknown")).upper()
    try:
        selected = CoverageState(status)
    except ValueError:
        selected = CoverageState.UNKNOWN
    coverage_id = str(getattr(value, "coverage_id", "coverage:knowledge:unknown"))
    kind = str(getattr(value, "kind", "structural_index"))
    return CoverageSnapshot(
        coverage_id=coverage_id,
        kind=kind,
        status=selected,
        inspected=getattr(value, "inspected", None),
        expected=getattr(value, "expected", None),
        reason=reason if selected is CoverageState.UNKNOWN else None,
    )


def _record_id(kind: str, identity: object) -> str:
    encoded_identity = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return f"{kind.lower()}:{hashlib.sha256(encoded_identity).hexdigest()[:32]}"


def _module_span(index: StructuralIndex, path: str) -> tuple[object, bytes]:
    module = index.module(path)
    if module is None:
        raise KnowledgeContractError(f"source file is not indexed: {path}")
    span = next((item for item in index.spans if item.path == module.path and item.kind == "module"), None)
    if span is None:
        raise KnowledgeContractError(f"module span is unavailable: {path}")
    read = index.read_span(span.span_id)
    return module, read.body


def _handle(index: StructuralIndex, path: str, start_line: int, end_line: int) -> SourceSpanHandle:
    module, body = _module_span(index, path)
    start, end = _line_bounds(body, start_line, end_line)
    exact = body[start:end]
    span_id = _record_id("span", {
        "source": module.content.content_id, "start_line": start_line, "end_line": end_line,
    })
    return SourceSpanHandle(
        source_ref=path,
        source_digest=module.content.content_digest,
        content_id=module.content.content_id,
        span_id=span_id,
        start_line=start_line,
        end_line=end_line,
        byte_start=start,
        byte_end=end,
        span_digest=_sha256_bytes(exact),
        coverage_ref=module.coverage.coverage_id,
    )


def _standalone_handle(root: Path, path: str, body: bytes, *, reason: str) -> SourceSpanHandle:
    namespace = Namespace("workspace:knowledge", root.name or "repo")
    content = ContentHandle.capture(body, namespace=namespace, source_ref=path)
    start, end = _line_bounds(body, 1, max(1, len(body.splitlines())))
    return SourceSpanHandle(
        source_ref=path,
        source_digest=content.content_digest,
        content_id=content.content_id,
        span_id=_record_id("span", {"content": content.content_id, "start": 1, "end": 1}),
        start_line=1,
        end_line=max(1, len(body.splitlines())),
        byte_start=start,
        byte_end=end,
        span_digest=_sha256_bytes(body[start:end]),
        coverage_ref=f"coverage:knowledge:{reason}",
    )


def _module_records(structural: StructuralIndex) -> list[KnowledgeRecord]:
    records: list[KnowledgeRecord] = []
    for module in structural.modules:
        path = module.path
        handle = _handle(structural, path, 1, max(1, next(
            (item.end_line for item in structural.spans if item.path == path and item.kind == "module"), 1
        )))
        coverage = _coverage_from_catalog(module.coverage, reason="shared outline is degraded")
        applicability = Applicability.CURRENT if coverage.status is CoverageState.COMPLETE else Applicability.UNKNOWN
        validity = Validity.VALID if coverage.status is CoverageState.COMPLETE else Validity.UNKNOWN
        outline = dict(structural.outlines.get(path, {}))
        records.append(KnowledgeRecord(
            record_id=_record_id("fact", {"subject": f"module:{path}", "digest": module.content.content_digest}),
            kind=RecordKind.FACT,
            subject=f"module:{path}",
            value={
                "schema": "mc-outline/1",
                "path": path,
                "language": module.language,
                "parser": module.parser,
                "degraded": module.degraded,
                "outline": outline,
            },
            source_refs=(handle,),
            dependency_kinds=(DependencyKind.SOURCE, DependencyKind.STRUCTURAL),
            applicability=applicability,
            validity=validity,
            coverage=coverage,
            structural_relation="module",
            language=module.language or "unknown",
        ))
        for span in structural.spans:
            if span.path != path or span.kind == "module":
                continue
            records.append(KnowledgeRecord(
                record_id=_record_id("fact", {"span": span.span_id, "kind": span.kind, "name": span.name}),
                kind=RecordKind.FACT,
                subject=f"symbol:{path}:{span.name}",
                value={"kind": span.kind, "name": span.name, "language": module.language},
                source_refs=(_handle(structural, path, span.start_line, span.end_line),),
                dependency_kinds=(DependencyKind.SOURCE, DependencyKind.STRUCTURAL),
                applicability=applicability,
                validity=validity,
                coverage=coverage,
                structural_relation=span.kind,
                language=module.language or "unknown",
            ))
    modules_by_id = {module.module_id: module for module in structural.modules}
    for dependency in structural.dependencies:
        module = modules_by_id.get(dependency.source_id)
        if module is None:
            continue
        path = module.path
        end_line = max(1, next((item.end_line for item in structural.spans if item.path == path and item.kind == "module"), 1))
        records.append(KnowledgeRecord(
            record_id=_record_id("fact", {"source": dependency.source_id, "target": dependency.target_id, "kind": dependency.relation}),
            kind=RecordKind.FACT,
            subject=f"dependency:{path}:{dependency.target_id}",
            value={"source": path, "target": dependency.target_id, "kind": dependency.relation},
            source_refs=(_handle(structural, path, 1, end_line),),
            dependency_kinds=(DependencyKind.SOURCE, DependencyKind.STRUCTURAL),
            applicability=Applicability.CURRENT if not module.degraded else Applicability.UNKNOWN,
            validity=Validity.VALID if not module.degraded else Validity.UNKNOWN,
            coverage=_coverage_from_catalog(dependency.coverage, reason="dependency outline is degraded"),
            structural_relation="dependency",
            language=module.language or "unknown",
        ))
    return records


class PythonAnalysisAdapter:
    """The only adapter allowed to use Python import/AST repository analysis."""

    def records(self, root: Path, structural: StructuralIndex, symbols: Sequence[str]) -> tuple[KnowledgeRecord, ...]:
        records: list[KnowledgeRecord] = []
        python_modules = {module.path: module for module in structural.modules if module.path.endswith(".py")}
        for symbol in tuple(dict.fromkeys(str(item) for item in symbols if str(item).isidentifier())):
            try:
                result = host_where_defined(root, symbol)
            except (OSError, TypeError, ValueError, RuntimeError, RecursionError):
                continue
            if not isinstance(result, Mapping) or result.get("status") != "FOUND":
                continue
            path = self._path_in(result)
            module = python_modules.get(path) if path else next(iter(python_modules.values()), None)
            if module is None:
                continue
            end_line = max(1, next((item.end_line for item in structural.spans if item.path == module.path and item.kind == "module"), 1))
            records.append(KnowledgeRecord(
                record_id=_record_id("fact", {"action": "where_defined", "symbol": symbol, "digest": module.content.content_digest}),
                kind=RecordKind.FACT,
                subject=f"definition:{symbol}",
                value={"adapter": "python-import-ast", "action": "where_defined", "result": dict(result)},
                source_refs=(_handle(structural, module.path, 1, end_line),),
                dependency_kinds=(DependencyKind.SOURCE, DependencyKind.STRUCTURAL),
                applicability=Applicability.CURRENT,
                validity=Validity.VALID,
                coverage=_coverage_from_catalog(module.coverage),
                structural_relation="definition",
                language="python",
            ))
        for path in sorted(python_modules):
            if not symbols:
                continue
            try:
                result = import_construction_fact(root, path, symbols)
            except (OSError, TypeError, ValueError, RuntimeError, RecursionError):
                continue
            if not isinstance(result, Mapping):
                continue
            module = python_modules[path]
            end_line = max(1, next((item.end_line for item in structural.spans if item.path == path and item.kind == "module"), 1))
            records.append(KnowledgeRecord(
                record_id=_record_id("fact", {"action": "import_construction", "path": path, "value": dict(result)}),
                kind=RecordKind.FACT,
                subject=f"construction:{path}",
                value={"adapter": "python-import-ast", "action": "import_construction", "result": dict(result)},
                source_refs=(_handle(structural, path, 1, end_line),),
                dependency_kinds=(DependencyKind.SOURCE, DependencyKind.STRUCTURAL),
                applicability=Applicability.CURRENT,
                validity=Validity.VALID,
                coverage=_coverage_from_catalog(module.coverage),
                structural_relation="construction",
                language="python",
            ))
        return tuple(records)

    def orientation(self, root: Path, *, target_path: str = "", symbol: str = "") -> str:
        """Expose existing intake orientation without making it a fact source."""

        return intake_orientation(root, target_path=target_path, symbol=symbol)

    @staticmethod
    def _path_in(value: object) -> str | None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"path", "source_path", "file"} and isinstance(item, str) and item.endswith(".py"):
                    return item.replace("\\", "/")
                found = PythonAnalysisAdapter._path_in(item)
                if found:
                    return found
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                found = PythonAnalysisAdapter._path_in(item)
                if found:
                    return found
        return None


class KnowledgeIndex:
    """One refreshable host index with negative lookup invalidation."""

    def __init__(
        self,
        root: Path,
        *,
        focus_paths: Sequence[str] = (),
        focus_symbols: Sequence[str] = (),
        idiom_specs: Sequence[IdiomSpec] = (),
        max_files: int = 512,
    ) -> None:
        root_path = Path(root)
        if root_path.is_symlink() or not root_path.is_dir():
            raise KnowledgeContractError("knowledge root must be a non-symlink directory")
        self.root = root_path.resolve()
        self.focus_paths = tuple(dict.fromkeys(str(item).replace("\\", "/") for item in focus_paths))
        self.focus_symbols = tuple(dict.fromkeys(str(item) for item in focus_symbols))
        self.idiom_specs = tuple(idiom_specs)
        self.max_files = max(1, min(int(max_files), 512))
        self.generation = ""
        self.structural: StructuralIndex | None = None
        self.records: tuple[KnowledgeRecord, ...] = ()
        self.stats: Mapping[str, object] = {}
        self._negative: dict[str, str] = {}
        self.refresh()

    def refresh(self) -> "KnowledgeIndex":
        fingerprint = _root_fingerprint(self.root)
        supported: list[str] = []
        unsupported: list[str] = []
        candidates = self.focus_paths or _public_files(self.root)
        for value in candidates:
            try:
                relpath = _relative(self.root, value)
            except KnowledgeContractError:
                continue
            target = self.root / relpath
            if target.is_symlink() or not target.is_file():
                continue
            (supported if _is_supported(relpath) else unsupported).append(relpath)
        try:
            structural = build_structural_index(
                self.root,
                focus_paths=tuple(sorted(set(supported))) if supported else (),
                max_files=self.max_files,
                include_call_graph=False,
            )
        except StructuralIndexError as exc:
            logger.warning("knowledge structural refresh degraded: %s", type(exc).__name__)
            structural = build_structural_index(self.root, max_files=self.max_files, include_call_graph=False)
        records = _module_records(structural)
        records.extend(PythonAnalysisAdapter().records(self.root, structural, self.focus_symbols))
        for relpath in sorted(set(unsupported)):
            target = self.root / relpath
            try:
                body = target.read_bytes()
            except (OSError, UnicodeError):
                continue
            handle = _standalone_handle(self.root, relpath, body, reason="unsupported-format")
            coverage = CoverageSnapshot(
                coverage_id=handle.coverage_ref,
                kind="structural_index",
                status=CoverageState.UNKNOWN,
                reason="unsupported language or format",
            )
            records.append(KnowledgeRecord(
                record_id=_record_id("fact", {"path": relpath, "digest": handle.source_digest}),
                kind=RecordKind.FACT,
                subject=f"module:{relpath}",
                value={
                    "schema": "mc-outline/1", "path": relpath, "language": "UNKNOWN",
                    "parser": "unsupported", "degraded": True, "degraded_reason": "unsupported format",
                },
                source_refs=(handle,),
                dependency_kinds=(DependencyKind.SOURCE, DependencyKind.STRUCTURAL),
                applicability=Applicability.UNKNOWN,
                validity=Validity.UNKNOWN,
                coverage=coverage,
                structural_relation="UNKNOWN",
                language="UNKNOWN",
            ))
        records.extend(self._idiom_records(structural))
        self.structural = structural
        self.records = tuple(sorted(records, key=lambda item: item.record_id))
        self.generation = fingerprint
        self.stats = structural.stats.as_dict()
        self._negative.clear()
        return self

    def ensure_current(self) -> None:
        if _root_fingerprint(self.root) != self.generation:
            self.refresh()

    def packet(self, extra_records: Sequence[KnowledgeRecord] = ()) -> KnowledgePacket:
        self.ensure_current()
        combined = tuple(self.records) + tuple(extra_records)
        coverage = CoverageSnapshot(
            coverage_id="coverage:knowledge-index:" + self.generation.split(":", 1)[-1][:24],
            kind="knowledge_index",
            status=CoverageState.COMPLETE if self.records else CoverageState.UNKNOWN,
            inspected=len(self.records),
            expected=len(self.records),
            reason=None if self.records else "no eligible current records",
        )
        return KnowledgePacket.create(self.generation, combined, coverage)

    def lookup(self, subject: str) -> tuple[KnowledgeRecord, ...]:
        self.ensure_current()
        key = str(subject)
        if self._negative.get(key) == self.generation:
            return ()
        rows = tuple(record for record in self.records if record.subject == key)
        if not rows:
            self._negative[key] = self.generation
        return rows

    def _idiom_records(self, structural: StructuralIndex) -> list[KnowledgeRecord]:
        records: list[KnowledgeRecord] = []
        for spec in self.idiom_specs:
            try:
                path = _relative(self.root, spec.source_ref)
                module = structural.module(path)
                if module is None or not _is_supported(path):
                    continue
                handle = _handle(structural, path, spec.start_line, spec.end_line)
                read = structural.read_span(next(item.span_id for item in structural.spans if item.path == path and item.kind == "module"))
                lines = read.body.splitlines(keepends=True)
                source_text = b"".join(lines[spec.start_line - 1:spec.end_line]).decode("utf-8", errors="replace")
                identifiers = tuple(spec.identifiers) or tuple(sorted(set(_IDENTIFIER_RE.findall(source_text))))
                if any(identifier not in set(_IDENTIFIER_RE.findall(source_text)) for identifier in spec.identifiers):
                    continue
                coverage = _coverage_from_catalog(module.coverage, reason="idiom source outline is degraded")
                records.append(KnowledgeRecord(
                    record_id=_record_id("idiom", {
                        "path": path, "start": spec.start_line, "end": spec.end_line,
                        "relation": spec.structural_relation, "digest": handle.span_digest,
                        "preconditions": dict(spec.preconditions), "transformation": spec.transformation,
                        "evidence_level": spec.evidence_level.value,
                        "identifiers": list(spec.identifiers),
                    }),
                    kind=RecordKind.IDIOM,
                    subject=f"idiom:{spec.structural_relation}:{path}",
                    value={
                        "source_text": source_text,
                        "transformation": spec.transformation,
                        "identifiers": list(identifiers),
                    },
                    source_refs=(handle,),
                    dependency_kinds=(DependencyKind.SOURCE, DependencyKind.STRUCTURAL, DependencyKind.IDIOM),
                    applicability=Applicability.CURRENT,
                    validity=Validity.VALID,
                    coverage=coverage,
                    structural_relation=spec.structural_relation,
                    preconditions=dict(spec.preconditions),
                    evidence_level=spec.evidence_level,
                    language=module.language or "unknown",
                ))
            except (KnowledgeContractError, KeyError, StopIteration, StructuralIndexError):
                continue
        return records


class DocumentationResolver:
    """Resolve only exact feature/version/runtime matches from admitted sources."""

    def __init__(
        self,
        *,
        local_roots: Sequence[Path] = (),
        local_candidates: Sequence[DocumentationCandidate] = (),
        official_snapshots: Sequence[DocumentationCandidate] = (),
    ) -> None:
        self.local_roots = tuple(self._root(path) for path in local_roots)
        self.local_candidates = tuple(local_candidates)
        self.official_snapshots = tuple(official_snapshots)

    @staticmethod
    def _root(path: Path) -> Path:
        value = Path(path)
        if value.is_symlink() or not value.is_dir():
            raise KnowledgeContractError("documentation allowlist root must be a non-symlink directory")
        return value.resolve()

    def _admitted(self, candidate: DocumentationCandidate) -> bool:
        if candidate.origin == "official":
            return candidate.authorized
        if candidate.origin != "local" or not candidate.authorized or not candidate.local_path:
            return False
        path = Path(candidate.local_path)
        if path.is_symlink() or not path.is_file():
            return False
        try:
            resolved = path.resolve()
        except OSError:
            return False
        return any(_is_within(resolved, root) for root in self.local_roots)

    def resolve(self, query: DocumentationQuery) -> DocumentationResolution:
        admitted = tuple(candidate for candidate in (*self.local_candidates, *self.official_snapshots)
                         if self._admitted(candidate))
        matching = tuple(candidate for candidate in admitted if (
            candidate.feature == query.feature and candidate.version == query.version and candidate.runtime == query.runtime
        ))
        supported = tuple(candidate for candidate in matching if candidate.format in _DOC_FORMATS)
        if not supported:
            if matching:
                return DocumentationResolution(DocumentationStatus.UNSUPPORTED_FORMAT, query, matching,
                                               "eligible metadata matched, but format is unsupported")
            return DocumentationResolution(DocumentationStatus.NO_ELIGIBLE_SOURCE, query, (),
                                           "no authorized source matches feature, version, and runtime")
        if len({candidate.content_digest for candidate in supported}) > 1:
            return DocumentationResolution(DocumentationStatus.CONFLICT, query, supported,
                                           "authorized sources disagree for the exact version-bound query")
        return DocumentationResolution(DocumentationStatus.FOUND, query, supported,
                                       "one or more equivalent authorized sources matched")

    def records(self, resolution: DocumentationResolution) -> tuple[KnowledgeRecord, ...]:
        if resolution.status not in {DocumentationStatus.FOUND, DocumentationStatus.CONFLICT}:
            return ()
        records: list[KnowledgeRecord] = []
        for candidate in resolution.candidates:
            body = candidate.content.encode("utf-8")
            handle = _standalone_handle(Path(candidate.local_path).parent if candidate.local_path else Path("."),
                                        candidate.source_ref, body, reason="documentation")
            coverage = CoverageSnapshot(
                coverage_id=handle.coverage_ref,
                kind="documentation",
                status=CoverageState.COMPLETE,
                inspected=len(body),
                expected=len(body),
            )
            records.append(KnowledgeRecord(
                record_id=f"api-doc:{candidate.source_id}",
                kind=RecordKind.API_DOC,
                subject=f"api:{resolution.query.feature}",
                value={
                    "feature": resolution.query.feature,
                    "version": resolution.query.version,
                    "runtime": resolution.query.runtime,
                    "format": candidate.format,
                    "documentation": candidate.content,
                    "instruction_as_data": True,
                    "source_digest": candidate.content_digest,
                    "origin": candidate.origin,
                },
                source_refs=(handle,),
                dependency_kinds=(DependencyKind.DOCUMENTATION, DependencyKind.VERSION, DependencyKind.FEATURE, DependencyKind.RUNTIME),
                applicability=Applicability.CONFLICT if resolution.status is DocumentationStatus.CONFLICT else Applicability.CURRENT,
                validity=Validity.VALID,
                coverage=coverage,
                language="documentation",
            ))
        return tuple(records)


def select_idiom(index: KnowledgeIndex, query: IdiomQuery) -> IdiomSelection:
    index.ensure_current()
    candidates = [record for record in index.records if record.kind is RecordKind.IDIOM]
    candidates = [record for record in candidates if record.structural_relation == query.structural_relation]
    candidates = [record for record in candidates if (
        query.language is None or record.language.casefold() == query.language.casefold()
    )]
    candidates = [record for record in candidates if all(record.preconditions.get(key) == value
                                                        for key, value in query.preconditions.items())]
    matched: list[KnowledgeRecord] = []
    for record in candidates:
        raw_identifiers = record.value.get("identifiers", ()) if isinstance(record.value, Mapping) else ()
        identifiers = set(str(item) for item in raw_identifiers) if isinstance(raw_identifiers, Sequence) else set()
        if not set(query.identifiers).issubset(identifiers):
            continue
        matched.append(record)
    candidates = matched
    if not candidates:
        return IdiomSelection(IdiomStatus.NO_MATCH, reason="no current idiom satisfies relation, preconditions, and identifiers")
    selected = sorted(candidates, key=lambda item: (
        _EVIDENCE_RANK.get(item.evidence_level or EvidenceLevel.UNVERIFIED, 9), item.record_id
    ))[0]
    return IdiomSelection(IdiomStatus.MATCH, selected, "deterministic current candidate")


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def source_delivery_status(content: object, *, path: str, source: str) -> dict[str, object]:
    """Validate an existing bounded source window, otherwise stay fail-open."""

    try:
        try:
            from kernel_request_source_window import verify_window, verify_window_projection
            from kernel_request_evidence_closure import source_revision
        except ImportError:
            from .kernel_request_source_window import verify_window, verify_window_projection
            from .kernel_request_evidence_closure import source_revision
        if verify_window(content, path=path, source=source):
            return {"mode": "bounded-current-window", "verified": True, "path": path, "source_revision": source_revision(source)}
        if verify_window_projection(content, path=path):
            return {"mode": "bounded-projected-window", "verified": True, "path": path, "source_revision": source_revision(source)}
    except (OSError, TypeError, ValueError, UnicodeError):
        pass
    return {
        "mode": "ordinary-source-acquisition",
        "verified": False,
        "path": path,
        "reason": "bounded source window is missing or stale; acquire current source normally",
    }


def _view_cost(index: KnowledgeIndex, record_count: int, documentation_count: int, knowledge_bytes: int) -> ValidationCost:
    source_bytes = sum(int(getattr(content, "byte_length", 0)) for content in (index.structural.contents.values() if index.structural else ()))
    return ValidationCost(
        files_scanned=len(index.structural.modules) if index.structural else 0,
        source_bytes=source_bytes,
        records_checked=record_count,
        documentation_candidates=documentation_count,
        knowledge_bytes=knowledge_bytes,
        estimated_input_tokens=max(1, (knowledge_bytes + 3) // 4),
    )


def build_task_view(
    index: KnowledgeIndex,
    *,
    extra_records: Sequence[KnowledgeRecord] = (),
    max_bytes: int = 8 * 1024,
    packet: KnowledgePacket | None = None,
    source_window: object | None = None,
    source_path: str = "",
    source_text: str = "",
) -> TaskKnowledgeView:
    """Project current packet records into a bounded, contradiction-safe view."""

    index.ensure_current()
    selected_packet = packet or index.packet(extra_records)
    stale = selected_packet.generation != index.generation
    if stale or not selected_packet.records:
        cost = _view_cost(index, 0, len(extra_records), 0)
        delivery = source_delivery_status(source_window, path=source_path, source=source_text) if source_window is not None else {
            "mode": "ordinary-source-acquisition", "verified": False, "reason": "no bounded source window supplied",
        }
        return TaskKnowledgeView(None, None, "FALLBACK", (), (),
                                 {"status": CoverageState.UNKNOWN.value, "reason": "stale or missing knowledge packet"},
                                 (), tuple(record.record_id for record in selected_packet.records), False, True,
                                 "stale or missing packet; ordinary source acquisition remains authoritative", cost, delivery)

    record_map = {record.record_id: record.to_dict() for record in selected_packet.records}
    contradiction_map = {
        item.contradiction_id: {
            "schema": item.SCHEMA,
            "contradiction_id": item.contradiction_id,
            "subject": item.subject,
            "record_ids": list(item.record_ids),
            # The full packet retains content/coverage IDs.  The bounded view
            # keeps the locator and both digests needed to reacquire bytes.
            "source_refs": [
                {
                    "source_ref": ref.source_ref,
                    "source_digest": ref.source_digest,
                    "span_id": ref.span_id,
                    "start_line": ref.start_line,
                    "end_line": ref.end_line,
                    "byte_start": ref.byte_start,
                    "byte_end": ref.byte_end,
                    "span_digest": ref.span_digest,
                }
                for ref in item.source_refs
            ],
            "reason": item.reason,
        }
        for item in selected_packet.contradictions
    }
    contradiction_ids = {
        record_id for item in selected_packet.contradictions for record_id in item.record_ids
    }
    ordered_ids = sorted(contradiction_ids) + [record_id for record_id in sorted(record_map) if record_id not in contradiction_ids]
    limit = max(_MIN_VIEW_BYTES, int(max_bytes))
    delivery = source_delivery_status(source_window, path=source_path, source=source_text) if source_window is not None else {
        "mode": "ordinary-source-acquisition", "verified": False, "reason": "no bounded source window supplied",
    }

    def compact_record(record: Mapping[str, object]) -> dict[str, object]:
        compact = dict(record)
        refs = record.get("source_refs", ())
        compact["source_refs"] = [
            {
                key: ref[key]
                for key in ("source_ref", "source_digest", "span_id", "start_line", "end_line", "byte_start", "byte_end", "span_digest")
                if isinstance(ref, Mapping) and key in ref
            }
            for ref in refs if isinstance(refs, Sequence) and isinstance(ref, Mapping)
        ]
        return compact

    def make_view(ids: Sequence[str], clipped: bool, cost_bytes: int = 0) -> TaskKnowledgeView:
        records = tuple(compact_record(record_map[item]) if clipped else record_map[item] for item in ids)
        omitted = tuple(record_id for record_id in sorted(record_map) if record_id not in ids)
        return TaskKnowledgeView(
            selected_packet.packet_id,
            selected_packet.packet_sha256,
            "CURRENT",
            records,
            tuple(contradiction_map.values()),
            selected_packet.coverage.to_dict(),
            tuple(ids),
            omitted,
            clipped,
            False,
            None,
            _view_cost(index, len(ids), len(extra_records), cost_bytes),
            delivery,
        )

    chosen: list[str] = []
    for record_id in ordered_ids:
        candidate = make_view((*chosen, record_id), True)
        encoded = json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) <= limit or not chosen:
            chosen.append(record_id)
    while True:
        provisional = make_view(chosen, len(chosen) < len(record_map))
        provisional_bytes = len(json.dumps(provisional.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        view = make_view(chosen, len(chosen) < len(record_map), provisional_bytes)
        encoded = json.dumps(view.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) <= limit:
            return view
        removable = [record_id for record_id in reversed(chosen) if record_id not in contradiction_ids]
        if not removable:
            removable = list(reversed(chosen))
        if not removable:
            # The contradiction envelope itself is larger than the caller's
            # requested budget; keep it intact rather than dropping contrary
            # evidence.  The contract's minimum envelope is explicit above.
            return view
        chosen.remove(removable[0])


def prepare_request_with_knowledge(
    frame: object,
    view: TaskKnowledgeView,
    *,
    model: str = "gpt-4o-mini",
    tools: Sequence[Mapping[str, object]] | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
    public_feedback: str = "",
    profile: object | None = None,
    registry: object | None = None,
    journal: object | None = None,
) -> object:
    """Consume the view through FRAMEBUILD's final serializer and bind it."""

    from kernel_optional_engine_module import optional_module
    framebuild = optional_module("STATEFRAME_FRAMEBUILD_MODULE")
    if framebuild is None:
        raise ImportError("frame builder extension is not configured")
    prepare_request = framebuild.prepare_request
    effective_profile = profile
    if effective_profile is None:
        frame_profile = getattr(frame, "profile", None)
        if frame_profile is not None and getattr(frame_profile, "name", "") == "decision":
            # Decision frames normally reserve 2K input tokens.  A bounded
            # knowledge view is an explicit source artifact, so retain the
            # decision action surface while giving its serialized evidence a
            # measured finite ceiling instead of refusing current source.
            from dataclasses import replace
            effective_profile = replace(
                frame_profile,
                input_token_ceiling=max(int(getattr(frame_profile, "input_token_ceiling", 2_000)), 12_000),
                wire_byte_ceiling=max(int(getattr(frame_profile, "wire_byte_ceiling", 12 * 1024)), 48 * 1024),
            )
    prepared = prepare_request(
        frame,
        model=model,
        tools=tools,
        temperature=temperature,
        seed=seed,
        public_feedback=public_feedback,
        profile=effective_profile,
        registry=registry,
        payload_overrides={"knowledge_packet": view.to_dict()},
    )
    if hasattr(prepared, "request_bytes"):
        bind_prepared_request(prepared, view, journal=journal)
    return prepared


def bind_prepared_request(prepared: object, view: TaskKnowledgeView, *, journal: object | None = None) -> dict[str, object]:
    """Verify IDs/digests against finalized bytes, then emit existing journal evidence."""

    request_bytes = getattr(prepared, "request_bytes", None)
    request_sha = getattr(prepared, "request_sha256", None)
    payload = getattr(prepared, "payload", None)
    if not isinstance(request_bytes, bytes) or not isinstance(request_sha, str) or not isinstance(payload, Mapping):
        raise KnowledgeContractError("prepared request is missing final wire fields")
    actual_sha = hashlib.sha256(request_bytes).hexdigest()
    if actual_sha != request_sha:
        raise KnowledgeContractError("PreparedRequest request digest does not match bytes")
    try:
        wire = json.loads(request_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KnowledgeContractError("PreparedRequest bytes are not canonical JSON") from exc
    if not isinstance(wire, Mapping) or wire.get("knowledge_packet") != view.to_dict():
        raise KnowledgeContractError("knowledge packet is not bound to finalized request bytes")
    binding = {
        "event_schema": "knowledge-wire-binding/1",
        "packet_id": view.packet_id,
        "packet_sha256": view.packet_sha256,
        "request_sha256": request_sha,
        "request_bytes": len(request_bytes),
        "status": view.status,
        "record_count": len(view.records),
        "contradiction_count": len(view.contradictions),
        "fallback_required": view.fallback_required,
        "estimated_input_tokens": view.cost.estimated_input_tokens,
    }
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("knowledge_packet_bound", producer="kernel_knowledge_substrate", dispatch=False, **binding)
    return binding


__all__ = [
    "DocumentationResolver", "KnowledgeIndex", "PythonAnalysisAdapter", "build_task_view",
    "bind_prepared_request", "prepare_request_with_knowledge", "select_idiom", "source_delivery_status",
]
