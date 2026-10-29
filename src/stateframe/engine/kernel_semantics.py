"""Receipt-to-semantic-evidence bridge for the local repair kernel.

This module is deliberately a read-only conversion surface:

    structured receipt + source snapshot -> SemanticEvidence -> bounded advisory

It does not import application code, execute source, run a command, create a
patch, or decide whether a hypothesis is correct.  Source facts are extracted
with :mod:`ast`; observations are accepted only from structured receipt fields.
The four producers are the small, task-family witnesses identified in
``sol_answers/l19_presolve.md``:

* ACK lifecycle predecessor and lookup-key witness;
* F2 namespace/authority differential matrix;
* watchdog launched-population denominator witness;
* RSS redirect-boundary witness.

The module is intentionally self-contained.  That keeps import cycles away
from ``kernel_bolt`` and gives the orchestrator a safe ``trace_cb`` and
``organ_out`` seam.  Every organ arms visibly, emits turn-stamped local trace
records, and degrades to a permanent no-op after an infrastructure failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from typing import Literal, TypeAlias, cast

from kernel_liveness import Evidence, OrganCensusSpec, TriState

try:
    import ast as _ast
    _AST_IMPORT_ERROR: BaseException | None = None
except Exception as _ast_exc:  # pragma: no cover - standard library failure control
    _ast = None
    _AST_IMPORT_ERROR = _ast_exc


SCHEMA = "poc.semantic_evidence/1"
INPUT_SCHEMA = "poc.semantic_inputs/1"
ADVISORY_SCHEMA = "poc.evidence_gap_advisory/1"
PRODUCER_VERSION = "1"
MAX_ADVISORY_MISSING = 3

TraceCallback: TypeAlias = Callable[[str, str, str], None]
OutputHook: TypeAlias = Callable[[str, str | None], str | None]
ReceiptMapping: TypeAlias = Mapping[str, object]
ReceiptInput: TypeAlias = ReceiptMapping | Sequence[ReceiptMapping]
SourceInput: TypeAlias = "SourceSnapshot | SourceBundle | Mapping[str, object] | Sequence[object] | str | None"


class FactState(str, Enum):
    """Three-valued state used for every source/receipt proposition."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class EvidenceStatus(str, Enum):
    """Aggregate state of a producer's required facts."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class EvidenceCode(str, Enum):
    """Stable, greppable names for the four families' obligations."""

    ACK_VERDICT_LOOKUP_KEYS = "ACK_VERDICT_LOOKUP_KEYS"
    ACK_PROPOSAL_PREDECESSOR = "ACK_PROPOSAL_PREDECESSOR"
    ACK_SETUP_SHAPE = "ACK_SETUP_SHAPE"

    DB_NAMESPACE_CASE_MATRIX = "DB_NAMESPACE_CASE_MATRIX"
    DB_BACKEND_HOST_PORT_AUTHORITY = "DB_BACKEND_HOST_PORT_AUTHORITY"

    WATCHDOG_POPULATION_WITNESS = "WATCHDOG_POPULATION_WITNESS"
    WATCHDOG_429_EXCLUSION = "WATCHDOG_429_EXCLUSION"
    WATCHDOG_THRESHOLD_BOUNDARY = "WATCHDOG_THRESHOLD_BOUNDARY"

    RSS_INITIAL_VALIDATION = "RSS_INITIAL_VALIDATION"
    RSS_PER_HOP_VALIDATION = "RSS_PER_HOP_VALIDATION"
    RSS_RELATIVE_REDIRECT_RESOLUTION = "RSS_RELATIVE_REDIRECT_RESOLUTION"
    RSS_REDIRECT_BOUND = "RSS_REDIRECT_BOUND"
    RSS_REDIRECT_HOP_WITNESS = "RSS_REDIRECT_HOP_WITNESS"


class _ProducerName(str, Enum):
    ACK = "ack_lifecycle"
    DB = "namespace_matrix"
    WATCHDOG = "watchdog_population"
    RSS = "redirect_boundary"


ACK_REQUIRED = (
    EvidenceCode.ACK_VERDICT_LOOKUP_KEYS,
    EvidenceCode.ACK_PROPOSAL_PREDECESSOR,
    EvidenceCode.ACK_SETUP_SHAPE,
)
DB_REQUIRED = (
    EvidenceCode.DB_NAMESPACE_CASE_MATRIX,
    EvidenceCode.DB_BACKEND_HOST_PORT_AUTHORITY,
)
WATCHDOG_REQUIRED = (
    EvidenceCode.WATCHDOG_POPULATION_WITNESS,
    EvidenceCode.WATCHDOG_429_EXCLUSION,
    EvidenceCode.WATCHDOG_THRESHOLD_BOUNDARY,
)
RSS_REQUIRED = (
    EvidenceCode.RSS_INITIAL_VALIDATION,
    EvidenceCode.RSS_PER_HOP_VALIDATION,
    EvidenceCode.RSS_RELATIVE_REDIRECT_RESOLUTION,
    EvidenceCode.RSS_REDIRECT_BOUND,
)


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _short_hash(value: object, length: int = 20) -> str:
    return _sha256(value)[:length]


def _canonical_value(value: object) -> object:
    """Return JSON-safe data with deterministic ordering and no object reprs."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical_value(item) for item in value), key=str)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__name__}


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            _canonical_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        # A malformed receipt must not make the organ control the repair loop.
        return json.dumps({"type": type(value).__name__}, sort_keys=True)


def _clip(value: object, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _safe_path(value: object, root: str | None = None) -> str:
    """Normalize a path into a repo-relative, non-absolute identity."""

    text = str(value or "").replace("\\", "/").strip()
    if not text:
        return ""

    absolute = text.startswith("/") or bool(re.match(r"^[A-Za-z]:/", text))
    if root and (text.startswith("/") or re.match(r"^[A-Za-z]:/", text)):
        try:
            candidate = os.path.relpath(text, root).replace("\\", "/")
            if candidate and not candidate.startswith("../") and candidate != "..":
                text = candidate
                absolute = False
        except (OSError, ValueError):
            pass

    # Strip known workspace roots when absolute paths arrive from a local probe.
    known_root = False
    for marker in ("local-swarm-poc/", "mission-control/"):
        if marker in text:
            text = text.split(marker, 1)[1]
            known_root = True
            break
    if absolute and not known_root:
        return ""

    parts = [part for part in text.split("/") if part not in ("", ".", "..")]
    if not parts:
        return ""
    # Keeping a bounded suffix prevents an absolute temporary path from leaking
    # into evidence while preserving the useful source path for diagnostics.
    if len(parts) > 12:
        parts = parts[-12:]
    return "/".join(parts)


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _state(value: object) -> FactState:
    if isinstance(value, FactState):
        return value
    if isinstance(value, bool):
        return FactState.TRUE if value else FactState.FALSE
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "supported", "accepted", "valid", "1"}:
            return FactState.TRUE
        if lowered in {"false", "no", "contradicted", "rejected", "invalid", "0"}:
            return FactState.FALSE
    return FactState.UNKNOWN


def _status(states: Mapping[EvidenceCode, FactState]) -> EvidenceStatus:
    if any(value is FactState.FALSE for value in states.values()):
        return EvidenceStatus.CONTRADICTED
    if states and all(value is FactState.TRUE for value in states.values()):
        return EvidenceStatus.SUPPORTED
    return EvidenceStatus.UNKNOWN


def _dedupe_strings(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _clip(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


@dataclass(frozen=True)
class AnchorRef:
    path: str
    line: int | None = None
    symbol: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "symbol": self.symbol,
        }


@dataclass(frozen=True)
class ProducerRef:
    name: str
    version: str = PRODUCER_VERSION

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class SubjectRef:
    path: str
    symbol: str
    kind: str = "function"

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "symbol": self.symbol, "kind": self.kind}


@dataclass(frozen=True)
class PredicateRef:
    name: str
    expression: str
    subject: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "expression": self.expression,
            "subject": self.subject,
        }


@dataclass(frozen=True)
class SourceRef:
    path: str
    symbol: str = ""
    start_line: int | None = None
    end_line: int | None = None
    fact: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "fact": self.fact,
        }


@dataclass(frozen=True)
class ReceiptRef:
    receipt_id: str
    sequence: int
    kind: str
    result_class: str
    turn: int | None
    workspace_revision: int
    anchor: AnchorRef | None = None
    test_run_id: str | None = None
    environment_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "sequence": self.sequence,
            "kind": self.kind,
            "result_class": self.result_class,
            "turn": self.turn,
            "workspace_revision": self.workspace_revision,
            "anchor": self.anchor.to_dict() if self.anchor else None,
            "test_run_id": self.test_run_id,
            "environment_id": self.environment_id,
        }


@dataclass(frozen=True)
class SourceFileRef:
    path: str
    digest: str
    symbols: tuple[str, ...] = ()
    spans: tuple[SourceRef, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "digest": self.digest,
            "symbols": self.symbols,
            "spans": tuple(span.to_dict() for span in self.spans),
        }


@dataclass(frozen=True)
class SourceBundle:
    generation: int
    workspace_revision: int
    tree_digest: str
    files: tuple[SourceFileRef, ...]
    truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "workspace_revision": self.workspace_revision,
            "tree_digest": self.tree_digest,
            "files": tuple(file.to_dict() for file in self.files),
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class SourceFileInput:
    path: str
    content: str


@dataclass(frozen=True)
class SourceSnapshot:
    """Bounded source input accepted by the bridge.

    ``files`` may be a mapping of repo-relative paths to source text, a single
    source string, or a sequence of ``SourceFileInput``/mapping records.  The
    source bodies are used only during the local AST pass and are never stored
    in a ``SemanticEvidence`` record.
    """

    files: Mapping[str, str] | Sequence[SourceFileInput] | str = field(default_factory=dict)
    generation: int = 0
    workspace_revision: int = 0
    truncated: bool = False
    root: str | None = None

    def as_bundle(self) -> SourceBundle:
        normalized = _source_file_texts(self.files, self.root)
        refs = tuple(
            SourceFileRef(path=path, digest=hashlib.sha256(text.encode("utf-8")).hexdigest())
            for path, text in sorted(normalized.items())
        )
        tree_digest = _sha256(
            {"files": [(ref.path, ref.digest) for ref in refs], "truncated": self.truncated}
        )
        return SourceBundle(
            generation=self.generation,
            workspace_revision=self.workspace_revision,
            tree_digest=tree_digest,
            files=refs,
            truncated=self.truncated,
        )


@dataclass(frozen=True)
class EvidenceRequirement:
    code: EvidenceCode
    state: FactState
    provenance: tuple[SourceRef | ReceiptRef, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "state": self.state.value,
            "provenance": tuple(
                item.to_dict() if hasattr(item, "to_dict") else _canonical_value(item)
                for item in self.provenance
            ),
        }


@dataclass(frozen=True)
class AckFacts:
    call_path: tuple[str, ...] = ()
    lookup_keys: tuple[str, ...] = ()
    verdict_lookup_only: FactState = FactState.UNKNOWN
    proposal_predecessor: FactState = FactState.UNKNOWN
    setup_shape: str = "unknown"
    proposal_write_path: FactState = FactState.UNKNOWN

    @property
    def required_key(self) -> tuple[str, ...]:
        return self.lookup_keys

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "AckFacts",
            "call_path": self.call_path,
            "lookup_keys": self.lookup_keys,
            "verdict_lookup_only": self.verdict_lookup_only.value,
            "proposal_predecessor": self.proposal_predecessor.value,
            "setup_shape": self.setup_shape,
            "proposal_write_path": self.proposal_write_path.value,
        }


@dataclass(frozen=True)
class NamespaceCase:
    label: str
    database: str
    backend: str
    host: str
    port: int
    expected_accept: bool
    source_accept: FactState
    observed_accept: FactState = FactState.UNKNOWN

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "database": self.database,
            "backend": self.backend,
            "host": self.host,
            "port": self.port,
            "expected_accept": self.expected_accept,
            "source_accept": self.source_accept.value,
            "observed_accept": self.observed_accept.value,
        }


@dataclass(frozen=True)
class NamespaceFacts:
    matrix: tuple[NamespaceCase, ...] = ()
    namespace_role: str = "name"
    authority_roles: tuple[str, ...] = ("backend", "host", "port")
    base_name: str = "mission_control_test_f2"
    port: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "NamespaceFacts",
            "matrix": tuple(case.to_dict() for case in self.matrix),
            "namespace_role": self.namespace_role,
            "authority_roles": self.authority_roles,
            "base_name": self.base_name,
            "port": self.port,
        }


@dataclass(frozen=True)
class PopulationControl:
    name: str
    raw: int
    launched: int
    eligible: int
    dispatch_corpses: int
    eligible_failures: int
    excluded_429: int
    rate_milli: int | None
    trigger: bool | None
    provenance: str = "CONTROL_SIMULATION"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "raw": self.raw,
            "launched": self.launched,
            "eligible": self.eligible,
            "dispatch_corpses": self.dispatch_corpses,
            "eligible_failures": self.eligible_failures,
            "excluded_429": self.excluded_429,
            "rate_milli": self.rate_milli,
            "trigger": self.trigger,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class PopulationFacts:
    raw: int | None = None
    launched: int | None = None
    eligible: int | None = None
    never_started: int | None = None
    zero_attempt: int | None = None
    dispatch_corpses: int | None = None
    eligible_failures: int | None = None
    excluded_429: int | None = None
    rate_milli: int | None = None
    minimum_population: int | None = None
    threshold_milli: int | None = None
    trigger: bool | None = None
    controls: tuple[PopulationControl, ...] = ()
    source_population_predicate: FactState = FactState.UNKNOWN

    @property
    def denominator(self) -> int | None:
        return self.eligible

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "PopulationFacts",
            "raw": self.raw,
            "launched": self.launched,
            "eligible": self.eligible,
            "never_started": self.never_started,
            "zero_attempt": self.zero_attempt,
            "dispatch_corpses": self.dispatch_corpses,
            "eligible_failures": self.eligible_failures,
            "excluded_429": self.excluded_429,
            "rate_milli": self.rate_milli,
            "minimum_population": self.minimum_population,
            "threshold_milli": self.threshold_milli,
            "trigger": self.trigger,
            "controls": tuple(control.to_dict() for control in self.controls),
            "source_population_predicate": self.source_population_predicate.value,
        }


@dataclass(frozen=True)
class RedirectHop:
    index: int
    url: str
    validated: FactState = FactState.UNKNOWN
    fetched: FactState = FactState.UNKNOWN
    unsafe: FactState = FactState.UNKNOWN
    relative: FactState = FactState.UNKNOWN

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "url": self.url,
            "validated": self.validated.value,
            "fetched": self.fetched.value,
            "unsafe": self.unsafe.value,
            "relative": self.relative.value,
        }


@dataclass(frozen=True)
class RedirectFacts:
    initial_validation: FactState = FactState.UNKNOWN
    per_hop_validation: FactState = FactState.UNKNOWN
    relative_resolution: FactState = FactState.UNKNOWN
    redirect_bound: FactState = FactState.UNKNOWN
    max_hops: int | None = None
    client_redirect_ownership: str = "unknown"
    hop_witness: FactState = FactState.UNKNOWN
    hops: tuple[RedirectHop, ...] = ()
    sibling_manual_head: FactState = FactState.UNKNOWN

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "RedirectFacts",
            "initial_validation": self.initial_validation.value,
            "per_hop_validation": self.per_hop_validation.value,
            "relative_resolution": self.relative_resolution.value,
            "redirect_bound": self.redirect_bound.value,
            "max_hops": self.max_hops,
            "client_redirect_ownership": self.client_redirect_ownership,
            "hop_witness": self.hop_witness.value,
            "hops": tuple(hop.to_dict() for hop in self.hops),
            "sibling_manual_head": self.sibling_manual_head.value,
        }


Payload: TypeAlias = AckFacts | NamespaceFacts | PopulationFacts | RedirectFacts


@dataclass(frozen=True)
class SemanticEvidence:
    """Immutable, content-addressed semantic projection."""

    schema: Literal["poc.semantic_evidence/1"]
    evidence_id: str
    producer: ProducerRef
    task_id: str
    frontier_hypothesis_id: str | None
    subject: SubjectRef
    trigger: ReceiptRef
    source: SourceBundle
    predicate: PredicateRef
    payload: Payload
    status: EvidenceStatus
    required: tuple[EvidenceCode, ...]
    missing: tuple[EvidenceCode, ...]
    limitations: tuple[str, ...]
    input_digest: str
    requirements: tuple[EvidenceRequirement, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "evidence_id": self.evidence_id,
            "producer": self.producer.to_dict(),
            "task_id": self.task_id,
            "frontier_hypothesis_id": self.frontier_hypothesis_id,
            "subject": self.subject.to_dict(),
            "trigger": self.trigger.to_dict(),
            "source": self.source.to_dict(),
            "predicate": self.predicate.to_dict(),
            "payload": self.payload.to_dict(),
            "status": self.status.value,
            "required": tuple(code.value for code in self.required),
            "missing": tuple(code.value for code in self.missing),
            "limitations": self.limitations,
            "input_digest": self.input_digest,
            "requirements": tuple(item.to_dict() for item in self.requirements),
        }

    @property
    def source_digest(self) -> str:
        return self.source.tree_digest


@dataclass(frozen=True)
class EvidenceGapAdvisory:
    schema: Literal["poc.evidence_gap_advisory/1"]
    frontier_hypothesis_id: str
    producer: str
    source_digest: str
    missing: tuple[EvidenceCode, ...]
    mode: Literal["ADVISORY_ONLY"]
    challenge_key: str
    subject: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "frontier_hypothesis_id": self.frontier_hypothesis_id,
            "producer": self.producer,
            "source_digest": self.source_digest,
            "missing": tuple(code.value for code in self.missing),
            "mode": self.mode,
            "challenge_key": self.challenge_key,
            "subject": self.subject,
        }

    def render(self) -> str:
        missing = ",".join(code.value for code in self.missing)
        target = self.subject or self.frontier_hypothesis_id
        return (
            f"[kernel] SEMANTIC_EVIDENCE_GAP evidence gap for {target}: "
            f"missing={missing}; advisory only"
        )


def _source_file_texts(
    files: Mapping[str, str] | Sequence[SourceFileInput] | str,
    root: str | None = None,
) -> dict[str, str]:
    if isinstance(files, str):
        return {"source.py": files}
    if isinstance(files, Mapping):
        result: dict[str, str] = {}
        for path, content in files.items():
            if isinstance(content, str):
                safe = _safe_path(path, root)
                if safe:
                    result[safe] = content
        return result
    result = {}
    for item in files:
        if isinstance(item, SourceFileInput):
            safe = _safe_path(item.path, root)
            if safe:
                result[safe] = item.content
        elif isinstance(item, Mapping):
            path = item.get("path") or item.get("file")
            content = item.get("content")
            if isinstance(path, str) and isinstance(content, str):
                safe = _safe_path(path, root)
                if safe:
                    result[safe] = content
    return result


def _coerce_snapshot(source: SourceInput) -> SourceSnapshot:
    if isinstance(source, SourceSnapshot):
        return source
    if isinstance(source, SourceBundle):
        # Bundles intentionally have no bodies.  The AST pass will remain UNKNOWN.
        return SourceSnapshot(
            files={},
            generation=source.generation,
            workspace_revision=source.workspace_revision,
            truncated=source.truncated,
        )
    if isinstance(source, str):
        return SourceSnapshot(files=source)
    if isinstance(source, Mapping):
        if isinstance(source.get("files"), (Mapping, list, tuple)):
            raw_files = source.get("files")
            files: Mapping[str, str] | Sequence[SourceFileInput]
            if isinstance(raw_files, Mapping):
                files = {
                    str(path): content
                    for path, content in raw_files.items()
                    if isinstance(content, str)
                }
            else:
                converted: list[SourceFileInput] = []
                for item in raw_files:
                    if isinstance(item, SourceFileInput):
                        converted.append(item)
                    elif (
                        isinstance(item, Mapping)
                        and isinstance(item.get("path"), str)
                        and isinstance(item.get("content"), str)
                    ):
                        converted.append(SourceFileInput(item["path"], item["content"]))
                files = tuple(converted)
            return SourceSnapshot(
                files=files,
                generation=_int_or_none(source.get("generation")) or 0,
                workspace_revision=_int_or_none(source.get("workspace_revision")) or 0,
                truncated=bool(source.get("truncated", False)),
                root=str(source.get("root")) if source.get("root") else None,
            )
        path = source.get("path") or source.get("file") or "source.py"
        content = source.get("content") or source.get("source") or ""
        return SourceSnapshot(files=(SourceFileInput(str(path), str(content)),))
    if isinstance(source, Sequence):
        converted = []
        for item in source:
            if isinstance(item, SourceFileInput):
                converted.append(item)
            elif (
                isinstance(item, Mapping)
                and isinstance(item.get("path"), str)
                and isinstance(item.get("content"), str)
            ):
                converted.append(SourceFileInput(item["path"], item["content"]))
        return SourceSnapshot(files=tuple(converted))
    return SourceSnapshot()


@dataclass(frozen=True)
class _FunctionInfo:
    path: str
    name: str
    node: object
    start_line: int
    end_line: int

    def source_ref(self, fact: str = "") -> SourceRef:
        return SourceRef(
            path=self.path,
            symbol=self.name,
            start_line=self.start_line,
            end_line=self.end_line,
            fact=fact,
        )


class _SourceContext:
    """AST-only index for one immutable source snapshot."""

    def __init__(self, source: SourceInput):
        snapshot = _coerce_snapshot(source)
        self.snapshot = snapshot
        self.texts = _source_file_texts(snapshot.files, snapshot.root)
        self.errors: list[str] = []
        self.functions: list[_FunctionInfo] = []
        self.trees: dict[str, object] = {}
        supplied_bundle = source if isinstance(source, SourceBundle) else None
        self.bundle = supplied_bundle or snapshot.as_bundle()

        if _AST_IMPORT_ERROR is not None or _ast is None:
            self.errors.append("ast_import_failed")
            return

        if supplied_bundle is not None:
            # A metadata-only bundle is still a valid source identity, but it has
            # no bodies to parse.  Preserve its digest and bounded file refs.
            return

        for path, text in sorted(self.texts.items()):
            try:
                tree = _ast.parse(text, filename=path, type_comments=False)
            except SyntaxError:
                self.errors.append(f"syntax_error:{path}")
                continue
            except Exception as exc:  # noqa: BLE001 - source input is untrusted
                self.errors.append(f"parse_error:{path}:{type(exc).__name__}")
                continue
            self.trees[path] = tree
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    self.functions.append(
                        _FunctionInfo(
                            path=path,
                            name=node.name,
                            node=node,
                            start_line=int(getattr(node, "lineno", 0) or 0),
                            end_line=int(getattr(node, "end_lineno", getattr(node, "lineno", 0)) or 0),
                        )
                    )

        by_path: dict[str, list[_FunctionInfo]] = {}
        for function in self.functions:
            by_path.setdefault(function.path, []).append(function)
        refs: list[SourceFileRef] = []
        for ref in self.bundle.files:
            funcs = sorted(
                by_path.get(ref.path, []),
                key=lambda function: (function.start_line, function.name),
            )
            refs.append(
                replace(
                    ref,
                    symbols=tuple(function.name for function in funcs),
                    spans=tuple(function.source_ref("function_definition") for function in funcs),
                )
            )
        self.bundle = replace(self.bundle, files=tuple(refs))

    def find(self, name: str, path_hint: str = "") -> tuple[_FunctionInfo, ...]:
        safe_hint = _safe_path(path_hint)
        candidates = [function for function in self.functions if function.name == name]
        if safe_hint:
            exact = [
                function
                for function in candidates
                if function.path == safe_hint or function.path.endswith("/" + safe_hint)
            ]
            if exact:
                candidates = exact
        return tuple(sorted(candidates, key=lambda item: (item.path, item.start_line, item.name)))

    def function_text(self, function: _FunctionInfo) -> str:
        text = self.texts.get(function.path, "")
        if _ast is None:
            return ""
        try:
            return _ast.get_source_segment(text, function.node) or ""
        except Exception:  # noqa: BLE001 - diagnostic only
            return ""

    def reachable(self, root: _FunctionInfo | None, max_depth: int = 2) -> tuple[_FunctionInfo, ...]:
        if root is None:
            return ()
        result: list[_FunctionInfo] = []
        queue: list[tuple[_FunctionInfo, int]] = [(root, 0)]
        seen: set[tuple[str, int, str]] = set()
        by_name: dict[str, tuple[_FunctionInfo, ...]] = {}
        for function in self.functions:
            by_name.setdefault(function.name, tuple())
        for name in tuple(by_name):
            by_name[name] = self.find(name)

        while queue:
            function, depth = queue.pop(0)
            key = (function.path, function.start_line, function.name)
            if key in seen:
                continue
            seen.add(key)
            result.append(function)
            if depth >= max_depth:
                continue
            for called in sorted(_called_names(function.node)):
                for child in by_name.get(called, ()):
                    queue.append((child, depth + 1))
        return tuple(sorted(result, key=lambda item: (item.path, item.start_line, item.name)))


def _literal(node: object) -> object | None:
    if _ast is None:
        return None
    if isinstance(node, _ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return node.value
    if isinstance(node, _ast.UnaryOp) and isinstance(node.op, (_ast.USub, _ast.UAdd)):
        value = _literal(node.operand)
        if isinstance(value, (int, float)):
            return -value if isinstance(node.op, _ast.USub) else value
    if isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Add):
        left = _literal(node.left)
        right = _literal(node.right)
        if isinstance(left, (str, int, float)) and isinstance(right, type(left)):
            return left + right
    if isinstance(node, (_ast.Tuple, _ast.List, _ast.Set)):
        values = tuple(_literal(element) for element in node.elts)
        if all(value is not None for value in values):
            if isinstance(node, _ast.Set):
                return frozenset(values)
            return values if isinstance(node, _ast.Tuple) else list(values)
    if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) and node.func.id in {
        "frozenset",
        "set",
        "tuple",
        "list",
    } and len(node.args) == 1 and not node.keywords:
        value = _literal(node.args[0])
        if isinstance(value, (tuple, list, frozenset)):
            if node.func.id == "frozenset":
                return frozenset(value)
            if node.func.id == "set":
                return frozenset(value)
            if node.func.id == "tuple":
                return tuple(value)
            return list(value)
    return None


def _called_names(node: object) -> set[str]:
    if _ast is None:
        return set()
    result: set[str] = set()
    for child in _ast.walk(node):
        if not isinstance(child, _ast.Call):
            continue
        func = child.func
        if isinstance(func, _ast.Name):
            result.add(func.id)
        elif isinstance(func, _ast.Attribute):
            result.add(func.attr)
    return result


def _ast_text(node: object, ctx: _SourceContext) -> str:
    if _ast is None:
        return ""
    path = ""
    if isinstance(node, _ast.AST):
        for function in ctx.functions:
            if function.node is node:
                path = function.path
                break
        if not path:
            line = int(getattr(node, "lineno", 0) or 0)
            candidates = [
                function
                for function in ctx.functions
                if function.start_line <= line <= function.end_line
            ]
            if candidates:
                path = min(
                    candidates,
                    key=lambda function: (
                        function.end_line - function.start_line,
                        function.start_line,
                        function.name,
                    ),
                ).path
    text = ctx.texts.get(path, "")
    try:
        return _ast.get_source_segment(text, node) or ""
    except Exception:  # noqa: BLE001
        return ""


def _assignment_literals(ctx: _SourceContext, names: Iterable[str]) -> dict[str, object]:
    wanted = set(names)
    result: dict[str, object] = {}
    if _ast is None:
        return result
    for tree in ctx.trees.values():
        for node in _ast.walk(tree):
            targets: list[object] = []
            value: object | None = None
            if isinstance(node, _ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, _ast.AnnAssign):
                targets = [node.target]
                value = node.value
            if value is None:
                continue
            literal = _literal(value)
            if literal is None:
                continue
            for target in targets:
                if isinstance(target, _ast.Name) and target.id in wanted:
                    result[target.id] = literal
    return result


def _anchor_from_value(value: object) -> AnchorRef | None:
    if isinstance(value, AnchorRef):
        return value
    if isinstance(value, Mapping):
        path = _safe_path(value.get("path") or value.get("file"))
        symbol = str(value.get("symbol") or value.get("function") or value.get("fn") or "")
        line = _int_or_none(value.get("line") or value.get("lineno"))
        return AnchorRef(path, line, symbol) if path or symbol else None
    text = str(value or "").strip()
    if not text:
        return None
    bits = text.split(":", 2)
    path = _safe_path(bits[0])
    line = _int_or_none(bits[1]) if len(bits) >= 2 else None
    symbol = bits[2].strip() if len(bits) >= 3 else ""
    return AnchorRef(path, line, symbol) if path or symbol else None


def _as_mapping(value: object) -> dict[str, object]:
    return {str(key): item for key, item in value.items()} if isinstance(value, Mapping) else {}


def _nested_mapping(mapping: Mapping[str, object], *keys: str) -> dict[str, object]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return _as_mapping(value)
    return {}


def _semantic_receipt(raw: Mapping[str, object]) -> dict[str, object]:
    """Drop model/time/diagnostic fields from semantic identity."""

    ignored = {"ts", "timestamp", "why", "detail", "raw", "model_prose", "trace"}

    def clean(value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): clean(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                if str(key).lower() not in ignored
            }
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted((clean(item) for item in value), key=str)
        return _canonical_value(value)

    cleaned = clean(raw)
    return cast(dict[str, object], cleaned if isinstance(cleaned, dict) else {})


@dataclass(frozen=True)
class _NormalizedReceipt:
    raw: Mapping[str, object]
    semantic: Mapping[str, object]
    ref: ReceiptRef


def _normalize_receipt(raw: Mapping[str, object], sequence: int) -> _NormalizedReceipt:
    semantic_inputs = _nested_mapping(raw, "semantic_inputs", "semantic", "inputs")
    test = _nested_mapping(semantic_inputs, "test")
    environment = _nested_mapping(semantic_inputs, "environment", "env")
    result_raw = raw.get("result_class") or raw.get("result") or test.get("result_class")
    result = str(result_raw or "").strip().upper()
    kind = str(raw.get("kind") or "").strip().lower()
    turn = _int_or_none(raw.get("turn") or raw.get("step"))
    revision = _int_or_none(raw.get("workspace_revision") or raw.get("revision")) or 0
    anchor = _anchor_from_value(raw.get("anchor") or raw.get("primary_anchor"))
    rid_value = raw.get("receipt_id") or raw.get("id")
    if rid_value:
        receipt_id = str(rid_value)
    else:
        receipt_id = "r:" + _short_hash(_semantic_receipt(raw), 24)
    test_run = raw.get("test_run_id") or test.get("test_run_id")
    environment_id = raw.get("environment_id") or environment.get("id") or environment.get("environment_id")
    return _NormalizedReceipt(
        raw=raw,
        semantic=_semantic_receipt(raw),
        ref=ReceiptRef(
            receipt_id=receipt_id,
            sequence=_int_or_none(raw.get("sequence") or raw.get("seq")) or sequence,
            kind=kind,
            result_class=result,
            turn=turn,
            workspace_revision=revision,
            anchor=anchor,
            test_run_id=str(test_run) if test_run is not None else None,
            environment_id=str(environment_id) if environment_id is not None else None,
        ),
    )


def _receipt_list(receipts: ReceiptInput) -> tuple[_NormalizedReceipt, ...]:
    raw_items: list[Mapping[str, object]] = []
    if isinstance(receipts, Mapping):
        raw_items.append(receipts)
    elif isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes)):
        raw_items.extend(item for item in receipts if isinstance(item, Mapping))
    normalized = [_normalize_receipt(item, index) for index, item in enumerate(raw_items, 1)]
    return tuple(
        sorted(
            normalized,
            key=lambda item: (
                item.ref.sequence,
                item.ref.turn if item.ref.turn is not None else -1,
                item.ref.receipt_id,
            ),
        )
    )


_NON_VERDICTS = {
    "GREEN",
    "SELECTOR_ERROR",
    "NO_TESTS",
    "NO_VERDICT",
    "INFRA_FAILURE",
    "USAGE_ERROR",
    "CHALLENGED",
    "REFUSED",
    "REJECTED",
    "NO_OP",
    "FAILED_EDIT",
}


def _is_admissible_measurement(receipt: _NormalizedReceipt) -> bool:
    result = receipt.ref.result_class
    if result in _NON_VERDICTS:
        return False
    raw = receipt.raw
    semantic = _nested_mapping(raw, "semantic_inputs", "semantic", "inputs")
    test = _nested_mapping(semantic, "test")
    exit_code = _int_or_none(raw.get("exit_code") or test.get("exit_code"))
    if exit_code is not None and exit_code != 1:
        return False
    admissibility = str(raw.get("admissibility") or test.get("admissibility") or "").lower()
    if admissibility and admissibility not in {
        "product_red",
        "product_failure",
        "product",
        "red",
        "admissible",
    }:
        return False
    if result in {"RED", "PRODUCT_RED"}:
        return True
    if result == "FAILED" and receipt.ref.kind in {"test", "observation", "witness", "semantic"}:
        return True
    return False


def _section(receipt: _NormalizedReceipt, *names: str) -> dict[str, object]:
    raw_semantic = _nested_mapping(receipt.raw, "semantic_inputs", "semantic", "inputs")
    for name in names:
        found = _nested_mapping(raw_semantic, name)
        if found:
            return found
    return raw_semantic


def _explicit_producer(receipt: _NormalizedReceipt) -> str:
    raw_semantic = _nested_mapping(receipt.raw, "semantic_inputs", "semantic", "inputs")
    return str(receipt.raw.get("producer") or raw_semantic.get("producer") or "").strip().lower()


def _anchor_matches(receipt: _NormalizedReceipt, targets: Sequence[str]) -> bool:
    anchor = receipt.ref.anchor
    if not anchor:
        return False
    symbol = anchor.symbol.rsplit(".", 1)[-1]
    return symbol in targets


def _latest_trigger(
    receipts: ReceiptInput,
    targets: Sequence[str],
    sections: Sequence[str],
    producer_name: str,
) -> _NormalizedReceipt | None:
    candidates = _receipt_list(receipts)
    explicit_aliases = {producer_name.lower(), producer_name.replace("_", "-").lower()}
    for receipt in reversed(candidates):
        if not _is_admissible_measurement(receipt):
            continue
        explicit = _explicit_producer(receipt)
        raw_semantic = _nested_mapping(receipt.raw, "semantic_inputs", "semantic", "inputs")
        section_present = any(
            section in raw_semantic and raw_semantic.get(section) is not None
            for section in sections
        )
        # A direct producer call may use a generic anchored RED.  The bridge
        # router applies the stricter qualified-symbol selection before calling it.
        if explicit and explicit not in explicit_aliases:
            continue
        if explicit or _anchor_matches(receipt, targets) or section_present:
            return receipt
    return None


def _subject_for(
    receipt: _NormalizedReceipt,
    ctx: _SourceContext,
    target_names: Sequence[str],
) -> tuple[SubjectRef, _FunctionInfo | None]:
    anchor = receipt.ref.anchor
    if anchor and (anchor.path or anchor.symbol):
        symbol = anchor.symbol or (target_names[0] if target_names else "")
        return SubjectRef(anchor.path, symbol), (ctx.find(symbol, anchor.path) or (None,))[0]
    for name in target_names:
        found = ctx.find(name)
        if found:
            function = found[0]
            return SubjectRef(function.path, function.name), function
    symbol = target_names[0] if target_names else "unknown"
    return SubjectRef("", symbol), None


def _source_bundle_for(ctx: _SourceContext, receipt: _NormalizedReceipt) -> SourceBundle:
    revision = max(ctx.bundle.workspace_revision, receipt.ref.workspace_revision)
    return replace(ctx.bundle, workspace_revision=revision)


def _make_evidence(
    *,
    producer: str,
    task_id: str,
    frontier_hypothesis_id: str | None,
    receipt: _NormalizedReceipt,
    ctx: _SourceContext,
    subject: SubjectRef,
    predicate: PredicateRef,
    payload: Payload,
    required: tuple[EvidenceCode, ...],
    states: Mapping[EvidenceCode, FactState],
    provenance: Mapping[EvidenceCode, tuple[SourceRef | ReceiptRef, ...]],
    limitations: Iterable[object],
) -> SemanticEvidence:
    status = _status(states)
    missing = tuple(code for code in required if states.get(code, FactState.UNKNOWN) is FactState.UNKNOWN)
    requirements = tuple(
        EvidenceRequirement(
            code=code,
            state=states.get(code, FactState.UNKNOWN),
            provenance=provenance.get(code, (receipt.ref,)),
        )
        for code in required
    )
    source = _source_bundle_for(ctx, receipt)
    core = {
        "producer": {"name": producer, "version": PRODUCER_VERSION},
        "task_id": task_id,
        "frontier_hypothesis_id": frontier_hypothesis_id,
        "receipt": receipt.semantic,
        "receipt_ref": receipt.ref.to_dict(),
        "source_digest": source.tree_digest,
        "predicate": predicate.to_dict(),
        "payload": payload.to_dict(),
        "required": [code.value for code in required],
        "states": {code.value: states.get(code, FactState.UNKNOWN).value for code in required},
    }
    input_digest = _sha256(core)
    evidence_id = "se:" + _short_hash({**core, "status": status.value, "missing": missing}, 24)
    return SemanticEvidence(
        schema=SCHEMA,
        evidence_id=evidence_id,
        producer=ProducerRef(producer, PRODUCER_VERSION),
        task_id=task_id,
        frontier_hypothesis_id=frontier_hypothesis_id,
        subject=subject,
        trigger=receipt.ref,
        source=source,
        predicate=predicate,
        payload=payload,
        status=status,
        required=required,
        missing=missing,
        limitations=_dedupe_strings(limitations),
        input_digest=input_digest,
        requirements=requirements,
    )


class _Producer:
    """Shared fail-open/wiring behavior for a semantic witness."""

    NAME = "semantic"
    TARGETS: tuple[str, ...] = ()
    REQUIRED: tuple[EvidenceCode, ...] = ()
    SECTION_NAMES: tuple[str, ...] = ()

    def __init__(self, task_id: str = "", *, trace_cb: TraceCallback | None = None):
        self.task_id = str(task_id or "task")
        self.trace: list[dict[str, object]] = []
        self.status = "ARMED"
        self.degraded = False
        self._trace_cb = trace_cb
        self._record(
            "SEMANTIC_PRODUCER_ARMED",
            None,
            f"producer={self.NAME}/1 required={','.join(code.value for code in self.REQUIRED)}",
        )
        if _AST_IMPORT_ERROR is not None or _ast is None:
            self._degrade(_AST_IMPORT_ERROR or RuntimeError("ast import unavailable"), None)

    def _record(self, event: str, turn: int | None, detail: object) -> None:
        entry = {
            "cylinder": "semantic",
            "event": event,
            "detail": _clip(detail, 240),
            "turn": turn,
        }
        self.trace.append(entry)
        callback = self._trace_cb
        if callback is None:
            return
        try:
            callback("semantic", event, str(entry["detail"]))
        except Exception as exc:  # noqa: BLE001 - observability cannot control work
            print(f"[kernel-wiring] semantic trace failed: {type(exc).__name__}")
            self._trace_cb = None

    def _degrade(self, exc: BaseException, turn: int | None) -> None:
        if not self.degraded:
            self.degraded = True
            self.status = "DISARMED"
            print(f"[kernel-wiring] semantic {self.NAME} degraded: {type(exc).__name__}: {_clip(exc, 120)}")
            self._record("SEMANTIC_PRODUCER_DISARMED", turn, type(exc).__name__)

    def _trigger(self, receipts: ReceiptInput) -> _NormalizedReceipt | None:
        return _latest_trigger(receipts, self.TARGETS, self.SECTION_NAMES, self.NAME)

    def produce(
        self,
        receipts: ReceiptInput,
        source: SourceInput,
        *,
        task_id: str | None = None,
        frontier_hypothesis_id: str | None = None,
    ) -> SemanticEvidence | None:
        if self.degraded:
            return None
        if _AST_IMPORT_ERROR is not None or _ast is None:
            self._degrade(_AST_IMPORT_ERROR or RuntimeError("ast import unavailable"), None)
            return None
        receipt = self._trigger(receipts)
        if receipt is None:
            return None
        self._record(
            "SEMANTIC_TRIGGER",
            receipt.ref.turn,
            f"producer={self.NAME} result={receipt.ref.result_class or 'UNKNOWN'}",
        )
        try:
            evidence = self._derive(
                receipt,
                source,
                task_id=str(task_id or receipt.raw.get("task_id") or self.task_id),
                frontier_hypothesis_id=frontier_hypothesis_id
                or str(receipt.raw.get("frontier_hypothesis_id") or "")
                or None,
            )
            if evidence is not None:
                self._record(
                    "SEMANTIC_EVIDENCE_DERIVED",
                    receipt.ref.turn,
                    f"SEMANTIC_EVIDENCE producer={self.NAME} id={evidence.evidence_id} "
                    f"status={evidence.status.value} missing={','.join(code.value for code in evidence.missing) or '-'}",
                )
            return evidence
        except Exception as exc:  # noqa: BLE001 - organ must never break the loop
            self._degrade(exc, receipt.ref.turn)
            return None

    derive = produce
    build = produce

    def __call__(self, receipts: ReceiptInput, source: SourceInput, **kwargs: object) -> SemanticEvidence | None:
        return self.produce(receipts, source, **cast(dict[str, object], kwargs))

    def _derive(
        self,
        receipt: _NormalizedReceipt,
        source: SourceInput,
        *,
        task_id: str,
        frontier_hypothesis_id: str | None,
    ) -> SemanticEvidence | None:
        raise NotImplementedError


def _call_sites(function: _FunctionInfo | None, names: set[str]) -> tuple[object, ...]:
    if function is None or _ast is None:
        return ()
    return tuple(
        node
        for node in _ast.walk(function.node)
        if isinstance(node, _ast.Call)
        and (
            (isinstance(node.func, _ast.Name) and node.func.id in names)
            or (isinstance(node.func, _ast.Attribute) and node.func.attr in names)
        )
    )


def _bool_from_mapping(mapping: Mapping[str, object], *keys: str) -> FactState:
    for key in keys:
        if key in mapping:
            return _state(mapping[key])
    return FactState.UNKNOWN


def _setup_shape(mapping: Mapping[str, object]) -> str:
    value = mapping.get("setup_shape") or mapping.get("setup")
    if isinstance(value, str) and value.strip().lower() in {"preseeded", "bare_orm", "unknown"}:
        return value.strip().lower()
    if _state(mapping.get("preseeded") or mapping.get("proposal_preseeded")) is FactState.TRUE:
        return "preseeded"
    if _state(mapping.get("bare_orm") or mapping.get("bare_decision")) is FactState.TRUE:
        return "bare_orm"
    return "unknown"


class AckLifecycleProducer(_Producer):
    NAME = _ProducerName.ACK.value
    TARGETS = ("_ack_decision", "ack_decision")
    REQUIRED = ACK_REQUIRED
    SECTION_NAMES = ("ack", "ack_lifecycle", "lifecycle")

    def _derive(
        self,
        receipt: _NormalizedReceipt,
        source: SourceInput,
        *,
        task_id: str,
        frontier_hypothesis_id: str | None,
    ) -> SemanticEvidence:
        ctx = _SourceContext(source)
        subject, ack_function = _subject_for(receipt, ctx, self.TARGETS)
        if ack_function is None:
            found = ctx.find("_ack_decision")
            ack_function = found[0] if found else None
        reachable = ctx.reachable(ack_function, max_depth=2)
        verdict_calls: list[tuple[_FunctionInfo, object]] = []
        proposal_calls: list[tuple[_FunctionInfo, object]] = []
        for function in reachable:
            for call in _call_sites(function, {"record_verdict"}):
                verdict_calls.append((function, call))
            for call in _call_sites(function, {"record_proposal", "_write_decision_ledger"}):
                proposal_calls.append((function, call))

        lookup_keys = ("workspace_id", "source", "source_ref_id")
        observed_lookup_keys: tuple[str, ...] = ()
        key_state = FactState.UNKNOWN
        if verdict_calls:
            observed: set[str] = set()
            for _function, call in verdict_calls:
                for keyword in call.keywords:
                    if keyword.arg in {"workspace_id", "source", "source_ref_id", "ledger_id"}:
                        observed.add(keyword.arg)
            observed_lookup_keys = tuple(
                key for key in lookup_keys if key in observed
            ) + (("ledger_id",) if "ledger_id" in observed and "source_ref_id" not in observed else ())
            if set(lookup_keys).issubset(observed) or "ledger_id" in observed:
                key_state = FactState.TRUE
            else:
                # Positional/expanded arguments and omitted fields are not an
                # explicit contradiction; the bridge cannot recover their
                # semantic names without executing application code.
                key_state = FactState.UNKNOWN
        elif ack_function is not None:
            key_state = FactState.FALSE

        record_verdict_functions = ctx.find("record_verdict")
        lookup_only = FactState.UNKNOWN
        for function in record_verdict_functions:
            text = ctx.function_text(function)
            has_lookup = bool(re.search(r"\.first\s*\(", text))
            has_missing_return = bool(re.search(r"(?:if\s+(?:not\s+)?row|row\s+is\s+None).*?return\s+None", text, re.S))
            has_proposal_write = "record_proposal" in text or "upsert" in text.lower() or "db.add(" in text
            if has_lookup and has_missing_return and not has_proposal_write:
                lookup_only = FactState.TRUE
            elif has_proposal_write:
                lookup_only = FactState.FALSE

        if lookup_only is FactState.FALSE:
            key_state = FactState.FALSE
        elif lookup_only is FactState.UNKNOWN and key_state is FactState.TRUE:
            key_state = FactState.UNKNOWN

        ack_inputs = _section(receipt, "ack", "ack_lifecycle", "lifecycle")
        setup_shape = _setup_shape(ack_inputs)
        explicit_predecessor = _bool_from_mapping(
            ack_inputs,
            "proposal_predecessor",
            "proposal_exists",
            "proposal_row_exists",
        )
        if proposal_calls:
            proposal_state = FactState.TRUE
        elif setup_shape == "preseeded":
            # A fixture row is not a route predecessor.  This is the masking
            # control called out by l19_presolve.md and is never upgraded.
            proposal_state = FactState.FALSE
        elif explicit_predecessor is not FactState.UNKNOWN:
            proposal_state = explicit_predecessor
        elif setup_shape == "bare_orm":
            proposal_state = FactState.FALSE
        else:
            proposal_state = FactState.UNKNOWN

        setup_state = FactState.TRUE if setup_shape in {"preseeded", "bare_orm"} else FactState.UNKNOWN
        call_path = tuple(
            sorted(
                f"{function.name}:{getattr(call, 'lineno', 0)} -> record_verdict"
                for function, call in verdict_calls
            )
        )
        proposal_path_state = FactState.TRUE if proposal_calls else FactState.UNKNOWN
        payload = AckFacts(
            call_path=call_path,
            lookup_keys=observed_lookup_keys,
            verdict_lookup_only=lookup_only,
            proposal_predecessor=proposal_state,
            setup_shape=setup_shape,
            proposal_write_path=proposal_path_state,
        )
        states = {
            EvidenceCode.ACK_VERDICT_LOOKUP_KEYS: key_state,
            EvidenceCode.ACK_PROPOSAL_PREDECESSOR: proposal_state,
            EvidenceCode.ACK_SETUP_SHAPE: setup_state,
        }
        refs = {
            EvidenceCode.ACK_VERDICT_LOOKUP_KEYS: tuple(
                function.source_ref("record_verdict_call") for function, _call in verdict_calls
            ),
            EvidenceCode.ACK_PROPOSAL_PREDECESSOR: tuple(
                function.source_ref("proposal_predecessor") for function, _call in proposal_calls
            ) or (receipt.ref,),
            EvidenceCode.ACK_SETUP_SHAPE: (receipt.ref,),
        }
        limitations: list[str] = list(ctx.errors)
        if setup_shape == "preseeded":
            limitations.append("preseeded_ledger_row_does_not_prove_predecessor")
        if proposal_state is FactState.UNKNOWN:
            limitations.append("receipt_declaration_missing:ACK_PROPOSAL_PREDECESSOR")
        if setup_state is FactState.UNKNOWN:
            limitations.append("receipt_declaration_missing:ACK_SETUP_SHAPE")
        predicate = PredicateRef(
            name="ack_lifecycle",
            expression="decision -> proposal predecessor -> human verdict",
            subject=f"{subject.path}:{subject.symbol}",
        )
        return _make_evidence(
            producer=self.NAME,
            task_id=task_id,
            frontier_hypothesis_id=frontier_hypothesis_id,
            receipt=receipt,
            ctx=ctx,
            subject=subject,
            predicate=predicate,
            payload=payload,
            required=self.REQUIRED,
            states=states,
            provenance=refs,
            limitations=limitations,
        )


def _ast_ref_name(node: object) -> str:
    if _ast is None:
        return ""
    if isinstance(node, _ast.Name):
        return node.id
    if isinstance(node, _ast.Attribute):
        return node.attr
    if isinstance(node, _ast.Call):
        return _ast_ref_name(node.func)
    return ""


_NAMESPACE_DATABASE_REFS = frozenset({"database_name", "database"})
_NAMESPACE_DATABASE_CONSTANTS = frozenset({"F2_DATABASE_NAME_PREFIX", "F2_DATABASE_NAME"})
_NAMESPACE_AUTHORITY_COMPONENTS = ("backend", "host", "port", "dbname")
_NAMESPACE_CONTRACT_HOSTS = frozenset({"localhost", "127.0.0.1"})
_NAMESPACE_UNRESOLVED = object()


@dataclass(frozen=True)
class _NamespaceGuard:
    component: str
    kind: str
    accepted_values: tuple[object, ...]
    state: FactState
    start_line: int
    end_line: int
    branch_lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class _NamespaceCaseMeasurement:
    label: str
    state: FactState
    name_state: FactState
    guards: tuple[_NamespaceGuard, ...] = ()


@dataclass(frozen=True)
class _NamespaceAnalysis:
    prefix: str
    port: int | None
    namespace_state: FactState
    authority_state: FactState
    backend_state: FactState
    host_state: FactState
    port_state: FactState
    dbname_state: FactState
    guards: tuple[_NamespaceGuard, ...]


def _namespace_suffix_kind(node: object, prefix: str) -> str:
    if _ast is None:
        return ""
    if isinstance(node, _ast.Name) and node.id in _NAMESPACE_DATABASE_CONSTANTS:
        return "bare"
    literal = _literal(node)
    if literal == prefix:
        return "bare"
    if literal == prefix + "_":
        return "delimited"
    if isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Add):
        if (
            isinstance(node.left, _ast.Name)
            and node.left.id in _NAMESPACE_DATABASE_CONSTANTS
            and _literal(node.right) == "_"
        ) or (
            _literal(node.left) == prefix
            and _literal(node.right) == "_"
        ):
            return "delimited"
    if isinstance(node, _ast.JoinedStr):
        has_prefix = any(
            isinstance(value, _ast.FormattedValue)
            and isinstance(value.value, _ast.Name)
            and value.value.id in _NAMESPACE_DATABASE_CONSTANTS
            for value in node.values
        )
        has_delimiter = any(
            isinstance(value, _ast.Constant) and isinstance(value.value, str) and "_" in value.value
            for value in node.values
        )
        if has_prefix and has_delimiter:
            return "delimited"
    return ""


def _namespace_component_for_refs(refs: Iterable[str]) -> str:
    values = set(refs)
    if values & _NAMESPACE_DATABASE_REFS:
        return "dbname"
    if values & {"backend_name", "backend", "get_backend_name"}:
        return "backend"
    if "host" in values:
        return "host"
    if "port" in values:
        return "port"
    return ""


def _namespace_component_for_compare(node: object) -> str:
    if _ast is None or not isinstance(node, _ast.Compare):
        return ""
    refs = (_ast_ref_name(node.left), *(_ast_ref_name(item) for item in node.comparators))
    return _namespace_component_for_refs(refs)


def _namespace_component_for_suffix(node: object) -> str:
    if _ast is None or not isinstance(node, _ast.Call) or not isinstance(node.func, _ast.Attribute):
        return ""
    if node.func.attr != "startswith":
        return ""
    return _namespace_component_for_refs((_ast_ref_name(node.func.value),))


def _namespace_guard_node(node: object) -> bool:
    return bool(_namespace_component_for_compare(node) or _namespace_component_for_suffix(node))


def _namespace_condition_nodes(condition: object) -> tuple[object, ...]:
    if _ast is None:
        return ()
    return tuple(
        node
        for node in _ast.walk(condition)
        if isinstance(node, (_ast.Compare, _ast.Call)) and _namespace_guard_node(node)
    )


def _namespace_branch_lines(function: _FunctionInfo | None) -> dict[int, tuple[int, ...]]:
    if function is None or _ast is None:
        return {}
    aliases: dict[str, tuple[int, ...]] = {}
    for node in _ast.walk(function.node):
        if isinstance(node, _ast.Assign):
            targets = tuple(target for target in node.targets if isinstance(target, _ast.Name))
            value = node.value
        elif isinstance(node, _ast.AnnAssign):
            targets = (node.target,) if isinstance(node.target, _ast.Name) else ()
            value = node.value
        else:
            continue
        if value is None:
            continue
        candidate_ids = tuple(id(item) for item in _namespace_condition_nodes(value))
        if not candidate_ids:
            continue
        for target in targets:
            aliases[target.id] = candidate_ids

    branches: dict[int, set[int]] = {}
    for node in _ast.walk(function.node):
        if not isinstance(node, _ast.If):
            continue
        candidates = _namespace_condition_nodes(node.test)
        if not candidates:
            alias_name = ""
            if isinstance(node.test, _ast.Name):
                alias_name = node.test.id
            elif isinstance(node.test, _ast.UnaryOp) and isinstance(node.test.op, _ast.Not) and isinstance(node.test.operand, _ast.Name):
                alias_name = node.test.operand.id
            candidate_ids = aliases.get(alias_name, ())
        else:
            candidate_ids = tuple(id(item) for item in candidates)
        branch_line = int(getattr(node, "lineno", 0) or 0)
        if not branch_line:
            continue
        for candidate_id in candidate_ids:
            branches.setdefault(candidate_id, set()).add(branch_line)
    return {node_id: tuple(sorted(lines)) for node_id, lines in branches.items()}


def _namespace_resolve(node: object, constants: Mapping[str, object]) -> object:
    literal = _literal(node)
    if literal is not None:
        return literal
    if _ast is not None and isinstance(node, _ast.Name) and node.id in constants:
        return constants[node.id]
    return _NAMESPACE_UNRESOLVED


def _namespace_compare_other(node: object, component: str) -> object:
    if _ast is None or not isinstance(node, _ast.Compare) or not node.comparators:
        return _NAMESPACE_UNRESOLVED
    left_ref = _ast_ref_name(node.left)
    if _namespace_component_for_refs((left_ref,)) == component:
        return node.comparators[0]
    for comparator in node.comparators:
        if _namespace_component_for_refs((_ast_ref_name(comparator),)) == component:
            return node.left
    return _NAMESPACE_UNRESOLVED


def _namespace_line(node: object) -> int:
    return int(getattr(node, "lineno", 0) or 0)


def _namespace_end_line(node: object) -> int:
    return int(getattr(node, "end_lineno", _namespace_line(node)) or _namespace_line(node))


def _namespace_guard(
    *,
    component: str,
    kind: str,
    accepted_values: tuple[object, ...],
    state: FactState,
    node: object,
    branch_lines: Mapping[int, tuple[int, ...]],
) -> _NamespaceGuard:
    return _NamespaceGuard(
        component=component,
        kind=kind,
        accepted_values=accepted_values,
        state=state,
        start_line=_namespace_line(node),
        end_line=_namespace_end_line(node),
        branch_lines=branch_lines.get(id(node), ()),
    )


def _namespace_compare_guard(
    node: object,
    constants: Mapping[str, object],
    prefix: str,
    branch_lines: Mapping[int, tuple[int, ...]],
) -> _NamespaceGuard | None:
    if _ast is None or not isinstance(node, _ast.Compare) or not node.ops:
        return None
    component = _namespace_component_for_compare(node)
    if not component:
        return None
    operator = node.ops[0]
    other = _namespace_compare_other(node, component)
    value = _namespace_resolve(other, constants)
    if value is _NAMESPACE_UNRESOLVED:
        return _namespace_guard(
            component=component,
            kind="comparison",
            accepted_values=(),
            state=FactState.UNKNOWN,
            node=node,
            branch_lines=branch_lines,
        )

    equality = isinstance(operator, (_ast.Eq, _ast.NotEq))
    membership = isinstance(operator, (_ast.In, _ast.NotIn))
    if component == "dbname":
        if not equality or not isinstance(value, str):
            return _namespace_guard(
                component=component,
                kind="comparison",
                accepted_values=(),
                state=FactState.UNKNOWN,
                node=node,
                branch_lines=branch_lines,
            )
        if value == prefix:
            accepted = ("base",)
            kind = "exact_name"
        elif value.startswith(prefix + "_"):
            accepted = ("delimited",)
            kind = "exact_delimited_name"
        else:
            accepted = ()
            kind = "wrong_name"
        return _namespace_guard(
            component=component,
            kind=kind,
            accepted_values=accepted,
            state=FactState.TRUE,
            node=node,
            branch_lines=branch_lines,
        )

    if component == "backend":
        if equality and isinstance(value, str):
            accepted = (value,) if value == "postgresql" else ()
            return _namespace_guard(
                component=component,
                kind="backend_name",
                accepted_values=accepted,
                state=FactState.TRUE,
                node=node,
                branch_lines=branch_lines,
            )
        if membership and isinstance(value, (tuple, list, frozenset)) and all(isinstance(item, str) for item in value):
            return _namespace_guard(
                component=component,
                kind="backend_membership",
                accepted_values=tuple(sorted(value)),
                state=FactState.TRUE,
                node=node,
                branch_lines=branch_lines,
            )

    if component == "host":
        if membership and isinstance(value, (tuple, list, frozenset)) and all(isinstance(item, str) for item in value):
            return _namespace_guard(
                component=component,
                kind="host_membership",
                accepted_values=tuple(sorted(value)),
                state=FactState.TRUE,
                node=node,
                branch_lines=branch_lines,
            )
        if equality and isinstance(value, str):
            return _namespace_guard(
                component=component,
                kind="host_equality",
                accepted_values=(value,),
                state=FactState.TRUE,
                node=node,
                branch_lines=branch_lines,
            )

    if component == "port" and equality and isinstance(value, int) and not isinstance(value, bool):
        return _namespace_guard(
            component=component,
            kind="port_equality",
            accepted_values=(value,),
            state=FactState.TRUE,
            node=node,
            branch_lines=branch_lines,
        )

    return _namespace_guard(
        component=component,
        kind="comparison",
        accepted_values=(),
        state=FactState.UNKNOWN,
        node=node,
        branch_lines=branch_lines,
    )


def _namespace_suffix_guard(
    node: object,
    prefix: str,
    branch_lines: Mapping[int, tuple[int, ...]],
) -> _NamespaceGuard | None:
    if _ast is None or not isinstance(node, _ast.Call) or _namespace_component_for_suffix(node) != "dbname" or not node.args:
        return None
    kind = _namespace_suffix_kind(node.args[0], prefix)
    if kind == "delimited":
        state = FactState.TRUE
        accepted = ("delimited",)
        guard_kind = "delimited_suffix"
    elif kind == "bare":
        state = FactState.TRUE
        accepted = ()
        guard_kind = "bare_suffix"
    else:
        state = FactState.UNKNOWN
        accepted = ()
        guard_kind = "dynamic_suffix"
    return _namespace_guard(
        component="dbname",
        kind=guard_kind,
        accepted_values=accepted,
        state=state,
        node=node,
        branch_lines=branch_lines,
    )


def _namespace_guards(
    function: _FunctionInfo | None,
    constants: Mapping[str, object],
    prefix: str,
) -> tuple[_NamespaceGuard, ...]:
    if function is None or _ast is None:
        return ()
    branch_lines = _namespace_branch_lines(function)
    guards: list[_NamespaceGuard] = []
    nodes = sorted(
        _ast.walk(function.node),
        key=lambda node: (_namespace_line(node), int(getattr(node, "col_offset", 0) or 0), type(node).__name__),
    )
    for node in nodes:
        guard = _namespace_compare_guard(node, constants, prefix, branch_lines)
        if guard is None:
            guard = _namespace_suffix_guard(node, prefix, branch_lines)
        if guard is not None:
            guards.append(guard)
    return tuple(guards)


def _namespace_name_coverage(guards: Sequence[_NamespaceGuard]) -> FactState:
    relevant = tuple(guard for guard in guards if guard.component == "dbname")
    if not relevant:
        return FactState.UNKNOWN
    has_base = any(guard.state is FactState.TRUE and "base" in guard.accepted_values for guard in relevant)
    has_delimited = any(guard.state is FactState.TRUE and "delimited" in guard.accepted_values for guard in relevant)
    if has_base and has_delimited and not any(guard.state is FactState.UNKNOWN for guard in relevant):
        return FactState.TRUE
    if any(guard.state is FactState.UNKNOWN for guard in relevant):
        return FactState.UNKNOWN
    return FactState.FALSE


def _namespace_guard_supports_component(
    guard: _NamespaceGuard,
    component: str,
    *,
    port: int | None,
) -> bool:
    if guard.state is not FactState.TRUE:
        return False
    if component == "dbname":
        return bool(set(guard.accepted_values) & {"base", "delimited"})
    if component == "backend":
        return bool(guard.accepted_values) and set(guard.accepted_values).issubset({"postgresql"})
    if component == "host":
        return bool(guard.accepted_values) and set(guard.accepted_values).issubset(_NAMESPACE_CONTRACT_HOSTS)
    if component == "port":
        return port is not None and guard.accepted_values == (port,)
    return False


def _namespace_component_coverage(
    guards: Sequence[_NamespaceGuard],
    component: str,
    *,
    port: int | None,
) -> FactState:
    relevant = tuple(guard for guard in guards if guard.component == component)
    if not relevant:
        return FactState.FALSE
    if component == "dbname":
        relevant = tuple(guard for guard in relevant if guard.kind != "bare_suffix")
        if not relevant:
            return FactState.UNKNOWN if any(guard.state is FactState.UNKNOWN for guard in guards if guard.component == component) else FactState.FALSE
    if any(not _namespace_guard_supports_component(guard, component, port=port) for guard in relevant if guard.state is not FactState.UNKNOWN):
        return FactState.FALSE
    if any(guard.state is FactState.UNKNOWN for guard in relevant):
        return FactState.UNKNOWN
    return FactState.TRUE


def _namespace_name_case(
    guards: Sequence[_NamespaceGuard],
    label: str,
) -> tuple[FactState, tuple[_NamespaceGuard, ...]]:
    relevant = tuple(guard for guard in guards if guard.component == "dbname")
    if not relevant:
        return FactState.UNKNOWN, ()
    accepted = tuple(
        guard
        for guard in relevant
        if guard.state is FactState.TRUE and label in guard.accepted_values
    )
    if accepted and not any(guard.state is FactState.UNKNOWN for guard in relevant):
        return FactState.TRUE, accepted
    if any(guard.state is FactState.UNKNOWN for guard in relevant):
        return FactState.UNKNOWN, relevant
    return FactState.FALSE, relevant


def _namespace_component_case(
    guards: Sequence[_NamespaceGuard],
    component: str,
    value: object,
) -> tuple[FactState, tuple[_NamespaceGuard, ...]]:
    relevant = tuple(guard for guard in guards if guard.component == component)
    if not relevant:
        return FactState.UNKNOWN, ()
    rejected = tuple(
        guard
        for guard in relevant
        if guard.state is FactState.FALSE
        or (guard.state is FactState.TRUE and value not in guard.accepted_values)
    )
    if rejected:
        return FactState.FALSE, rejected
    unknown = tuple(guard for guard in relevant if guard.state is FactState.UNKNOWN)
    if unknown:
        return FactState.UNKNOWN, unknown
    return FactState.TRUE, relevant


def _namespace_case_measurement(
    guards: Sequence[_NamespaceGuard],
    *,
    label: str,
    backend: str,
    host: str,
    port: int,
) -> _NamespaceCaseMeasurement:
    name_label = label if label in {"base", "delimited_suffix", "undelimited_suffix"} else "base"
    name_category = {
        "base": "base",
        "delimited_suffix": "delimited",
        "undelimited_suffix": "undelimited",
    }[name_label]
    name_state, name_guards = _namespace_name_case(guards, name_category)
    authority_results: list[tuple[str, FactState, tuple[_NamespaceGuard, ...]]] = []
    for component, value in (("backend", backend), ("host", host), ("port", port)):
        state, component_guards = _namespace_component_case(guards, component, value)
        authority_results.append((component, state, component_guards))
    authority_states = tuple(state for _component, state, _guards in authority_results)
    authority_guards = tuple(
        guard
        for _component, _state, component_guards in authority_results
        for guard in component_guards
    )
    state = _status_as_fact((name_state, *authority_states))
    if state is FactState.TRUE:
        selected = (*name_guards, *authority_guards)
    elif state is FactState.FALSE:
        selected = tuple(
            guard for guard in name_guards if name_state is FactState.FALSE
        ) + tuple(
            guard
            for _component, component_state, component_guards in authority_results
            if component_state is FactState.FALSE
            for guard in component_guards
        )
    else:
        selected = (*name_guards, *authority_guards)
    return _NamespaceCaseMeasurement(
        label=label,
        state=state,
        name_state=name_state,
        guards=_namespace_unique_guards(selected),
    )


def _namespace_analysis(ctx: _SourceContext, function: _FunctionInfo | None) -> _NamespaceAnalysis:
    constants = _assignment_literals(
        ctx,
        {"F2_DATABASE_NAME_PREFIX", "F2_DATABASE_NAME", "F2_DATABASE_PORT", "_LOCAL_HOSTS"},
    )
    prefix_value = constants.get("F2_DATABASE_NAME_PREFIX")
    if not isinstance(prefix_value, str):
        prefix_value = constants.get("F2_DATABASE_NAME")
    prefix = str(prefix_value) if isinstance(prefix_value, str) else "mission_control_test_f2"
    port_value = constants.get("F2_DATABASE_PORT")
    port = int(port_value) if isinstance(port_value, int) and not isinstance(port_value, bool) else None
    if function is None:
        return _NamespaceAnalysis(
            prefix=prefix,
            port=port,
            namespace_state=FactState.UNKNOWN,
            authority_state=FactState.UNKNOWN,
            backend_state=FactState.UNKNOWN,
            host_state=FactState.UNKNOWN,
            port_state=FactState.UNKNOWN,
            dbname_state=FactState.UNKNOWN,
            guards=(),
        )
    guards = _namespace_guards(function, constants, prefix)
    backend_state = _namespace_component_coverage(guards, "backend", port=port)
    host_state = _namespace_component_coverage(guards, "host", port=port)
    port_state = _namespace_component_coverage(guards, "port", port=port)
    dbname_state = _namespace_component_coverage(guards, "dbname", port=port)
    return _NamespaceAnalysis(
        prefix=prefix,
        port=port,
        namespace_state=_namespace_name_coverage(guards),
        authority_state=_status_as_fact((backend_state, host_state, port_state, dbname_state)),
        backend_state=backend_state,
        host_state=host_state,
        port_state=port_state,
        dbname_state=dbname_state,
        guards=guards,
    )


def _namespace_name_features(function: _FunctionInfo | None, prefix: str) -> FactState:
    constants = {
        "F2_DATABASE_NAME_PREFIX": prefix,
        "F2_DATABASE_NAME": prefix,
    }
    return _namespace_name_coverage(_namespace_guards(function, constants, prefix))


def _namespace_features(ctx: _SourceContext, function: _FunctionInfo | None) -> tuple[FactState, FactState, FactState, FactState, str, int | None]:
    analysis = _namespace_analysis(ctx, function)
    return (
        analysis.namespace_state,
        analysis.authority_state,
        analysis.backend_state,
        analysis.host_state,
        analysis.prefix,
        analysis.port,
    )


def _status_as_fact(states: Iterable[FactState]) -> FactState:
    values = tuple(states)
    if any(value is FactState.FALSE for value in values):
        return FactState.FALSE
    if values and all(value is FactState.TRUE for value in values):
        return FactState.TRUE
    return FactState.UNKNOWN


def _observed_case_state(section: Mapping[str, object], label: str) -> FactState:
    matrix = section.get("matrix")
    if isinstance(matrix, Sequence) and not isinstance(matrix, (str, bytes)):
        for item in matrix:
            if not isinstance(item, Mapping):
                continue
            item_label = str(item.get("label") or item.get("case") or "").lower()
            if item_label == label.lower():
                return _state(item.get("accepted") if "accepted" in item else item.get("observed_accept"))
    return FactState.UNKNOWN


def _namespace_state_word(state: FactState) -> str:
    if state is FactState.TRUE:
        return "ACCEPTS"
    if state is FactState.FALSE:
        return "REJECTS"
    return "UNKNOWN"


def _namespace_coverage_word(state: FactState) -> str:
    if state is FactState.TRUE:
        return "COMPLETE"
    if state is FactState.FALSE:
        return "INCOMPLETE"
    return "UNDECIDABLE"


def _namespace_outcome_word(state: FactState) -> str:
    if state is FactState.TRUE:
        return "ACCEPTED"
    if state is FactState.FALSE:
        return "REJECTED"
    return "UNKNOWN"


def _namespace_contract_word(expected: bool) -> str:
    return "ACCEPT" if expected else "REJECT"


def _namespace_display_label(label: str) -> str:
    return {
        "delimited_suffix": "delimited-suffix",
        "undelimited_suffix": "undelimited-suffix",
        "remote_host": "remote-host",
        "wrong_port": "wrong-port",
        "non_postgres": "non-postgres",
    }.get(label, label)


def _namespace_guard_span_text(guards: Sequence[_NamespaceGuard]) -> str:
    spans: list[str] = []
    seen: set[tuple[int, int]] = set()
    for guard in sorted(guards, key=lambda item: (item.start_line, item.end_line, item.component, item.kind)):
        key = (guard.start_line, guard.end_line)
        if key in seen or guard.start_line <= 0:
            continue
        seen.add(key)
        if guard.start_line == guard.end_line:
            spans.append(f"L{guard.start_line}")
        else:
            spans.append(f"L{guard.start_line}-L{guard.end_line}")
    return ",".join(spans) or "NONE"


def _namespace_branch_span_text(guards: Sequence[_NamespaceGuard]) -> str:
    lines = sorted({line for guard in guards for line in guard.branch_lines if line > 0})
    return ",".join(f"L{line}" for line in lines) or "NONE"


def _namespace_unique_guards(guards: Iterable[_NamespaceGuard]) -> tuple[_NamespaceGuard, ...]:
    result: list[_NamespaceGuard] = []
    seen: set[tuple[str, str, int, int]] = set()
    for guard in guards:
        key = (guard.component, guard.kind, guard.start_line, guard.end_line)
        if key in seen:
            continue
        seen.add(key)
        result.append(guard)
    return tuple(result)


def _namespace_contract_state(
    cases: Sequence[NamespaceCase],
    states: Sequence[FactState] | None = None,
) -> FactState:
    mismatch = False
    unknown = False
    for index, case in enumerate(cases):
        source_state = case.source_accept if states is None else states[index]
        expected = FactState.TRUE if case.expected_accept else FactState.FALSE
        if source_state is FactState.UNKNOWN:
            unknown = True
        elif source_state is not expected:
            mismatch = True
    if mismatch:
        return FactState.FALSE
    return FactState.UNKNOWN if unknown else FactState.TRUE


def _namespace_case_provenance(
    function: _FunctionInfo | None,
    cases: Sequence[NamespaceCase],
    measurements: Sequence[_NamespaceCaseMeasurement],
    fallback: ReceiptRef,
) -> tuple[SourceRef | ReceiptRef, ...]:
    if function is None:
        return (fallback,)
    summary = "; ".join(
        f"{_namespace_display_label(case.label)}({_namespace_guard_span_text(measurement.guards)})="
        f"{_namespace_state_word(measurement.state)}"
        for case, measurement in zip(cases, measurements)
    ) or "none=UNKNOWN"
    refs: list[SourceRef | ReceiptRef] = [
        function.source_ref(f"namespace cases: {summary}"),
    ]
    for case, measurement in zip(cases, measurements):
        guards = measurement.guards
        guard_span = _namespace_guard_span_text(guards)
        branch_span = _namespace_branch_span_text(guards)
        fact = (
            f"{_namespace_display_label(case.label)} case: "
            f"{_namespace_outcome_word(measurement.state)} by guard {guard_span}"
            f" (branch {branch_span}) — contract says {_namespace_contract_word(case.expected_accept)}"
        )
        start_line = min((guard.start_line for guard in guards), default=function.start_line)
        end_line = max((guard.end_line for guard in guards), default=function.end_line)
        refs.append(
            SourceRef(
                path=function.path,
                symbol=function.name,
                start_line=start_line,
                end_line=end_line,
                fact=fact,
            )
        )
    return tuple(refs)


def _namespace_authority_provenance(
    function: _FunctionInfo | None,
    analysis: _NamespaceAnalysis,
    fallback: ReceiptRef,
) -> tuple[SourceRef | ReceiptRef, ...]:
    if function is None:
        return (fallback,)
    component_states = {
        "backend": analysis.backend_state,
        "host": analysis.host_state,
        "port": analysis.port_state,
        "dbname": analysis.dbname_state,
    }
    parts: list[str] = []
    refs: list[SourceRef | ReceiptRef] = []
    for component in _NAMESPACE_AUTHORITY_COMPONENTS:
        guards = tuple(guard for guard in analysis.guards if guard.component == component)
        state = component_states[component]
        span = _namespace_guard_span_text(guards)
        compared = (
            "COMPARED"
            if state is FactState.TRUE
            else "NOT_COMPARED"
            if state is FactState.FALSE
            else "UNKNOWN"
        )
        parts.append(f"{component}={compared}[{span}]")
        start_line = min((guard.start_line for guard in guards), default=function.start_line)
        end_line = max((guard.end_line for guard in guards), default=function.end_line)
        guard_details = ",".join(
            f"{guard.kind}@{_namespace_guard_span_text((guard,))}" for guard in guards
        ) or "NONE"
        refs.append(
            SourceRef(
                path=function.path,
                symbol=function.name,
                start_line=start_line,
                end_line=end_line,
                fact=(
                    f"authority component={component}; compared={state.value}; "
                    f"guards={guard_details}; contract=REQUIRED"
                ),
            )
        )
    summary = "; ".join(parts) or "none=UNKNOWN"
    refs.insert(
        0,
        function.source_ref(
            f"authority components: {summary}; "
            f"contract requires {','.join(_NAMESPACE_AUTHORITY_COMPONENTS)}; "
            f"coverage={_namespace_coverage_word(analysis.authority_state)}"
        ),
    )
    return tuple(refs)


class NamespaceMatrixProducer(_Producer):
    NAME = _ProducerName.DB.value
    TARGETS = ("require_isolated_f2_database",)
    REQUIRED = DB_REQUIRED
    SECTION_NAMES = ("namespace", "database", "environment")

    def _derive(
        self,
        receipt: _NormalizedReceipt,
        source: SourceInput,
        *,
        task_id: str,
        frontier_hypothesis_id: str | None,
    ) -> SemanticEvidence:
        ctx = _SourceContext(source)
        subject, function = _subject_for(receipt, ctx, self.TARGETS)
        if function is None:
            found = ctx.find(self.TARGETS[0])
            function = found[0] if found else None
        analysis = _namespace_analysis(ctx, function)
        section = _section(receipt, "namespace", "database", "environment")
        actual_port = analysis.port or 55432
        controls = (
            ("base", analysis.prefix, "postgresql", "localhost", actual_port, True),
            ("delimited_suffix", f"{analysis.prefix}_copy", "postgresql", "localhost", actual_port, True),
            ("undelimited_suffix", f"{analysis.prefix}x", "postgresql", "localhost", actual_port, False),
            ("remote_host", analysis.prefix, "postgresql", "203.0.113.8", actual_port, False),
            ("wrong_port", analysis.prefix, "postgresql", "localhost", 55433, False),
            ("non_postgres", analysis.prefix, "sqlite", "localhost", actual_port, False),
        )
        cases: list[NamespaceCase] = []
        measurements: list[_NamespaceCaseMeasurement] = []
        for label, database, backend, host, case_port, expected in controls:
            measurement = _namespace_case_measurement(
                analysis.guards,
                label=label,
                backend=backend,
                host=host,
                port=case_port,
            )
            measurements.append(measurement)
            cases.append(
                NamespaceCase(
                    label=label,
                    database=database,
                    backend=backend,
                    host=host,
                    port=case_port,
                    expected_accept=expected,
                    source_accept=measurement.state,
                    observed_accept=_observed_case_state(section, label),
                )
            )

        namespace_check = _status_as_fact(
            (
                analysis.namespace_state,
                _namespace_contract_state(
                    cases[:3],
                    [measurement.name_state for measurement in measurements[:3]],
                ),
            )
        )
        authority_check = _status_as_fact(
            (analysis.authority_state, _namespace_contract_state(cases[3:]))
        )

        payload = NamespaceFacts(
            matrix=tuple(cases),
            namespace_role="name",
            authority_roles=("backend", "host", "port"),
            base_name=analysis.prefix,
            port=actual_port,
        )
        states = {
            EvidenceCode.DB_NAMESPACE_CASE_MATRIX: namespace_check,
            EvidenceCode.DB_BACKEND_HOST_PORT_AUTHORITY: authority_check,
        }
        limitations: list[str] = list(ctx.errors)
        if function is None:
            limitations.append("target_function_not_found")
        if any(case.observed_accept is FactState.UNKNOWN for case in cases):
            limitations.append("runtime_namespace_acceptance_not_observed")
        predicate = PredicateRef(
            name="f2_namespace_authority",
            expression="base|delimited suffix accepted; undelimited suffix and authority mismatches rejected",
            subject=f"{subject.path}:{subject.symbol}",
        )
        return _make_evidence(
            producer=self.NAME,
            task_id=task_id,
            frontier_hypothesis_id=frontier_hypothesis_id,
            receipt=receipt,
            ctx=ctx,
            subject=subject,
            predicate=predicate,
            payload=payload,
            required=self.REQUIRED,
            states=states,
            provenance={
                EvidenceCode.DB_NAMESPACE_CASE_MATRIX: _namespace_case_provenance(
                    function,
                    cases,
                    measurements,
                    receipt.ref,
                ),
                EvidenceCode.DB_BACKEND_HOST_PORT_AUTHORITY: _namespace_authority_provenance(
                    function,
                    analysis,
                    receipt.ref,
                ),
            },
            limitations=limitations,
        )


def _row_value(row: object, key: str) -> tuple[bool, object | None]:
    if isinstance(row, Mapping):
        return key in row, row.get(key)
    try:
        return hasattr(row, key), getattr(row, key, None)
    except Exception:  # noqa: BLE001 - receipt adapters are fail-closed
        return False, None


def _population_rows(receipt: _NormalizedReceipt) -> tuple[object, ...] | None:
    section = _section(receipt, "population", "watchdog", "tasks")
    for value in (
        section.get("rows"),
        section.get("tasks"),
        section.get("recent_tasks"),
        receipt.raw.get("recent_tasks"),
    ):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return tuple(value)
    return None


def _population_counts(receipt: _NormalizedReceipt) -> dict[str, int | None] | None:
    rows = _population_rows(receipt)
    if rows is not None:
        result: dict[str, int | None] = {
            "raw": len(rows),
            "launched": 0,
            "eligible": 0,
            "never_started": 0,
            "zero_attempt": 0,
            "dispatch_corpses": 0,
            "eligible_failures": 0,
            "excluded_429": 0,
        }
        for row in rows:
            has_started, started = _row_value(row, "started_at")
            has_attempts, attempts_value = _row_value(row, "attempts")
            has_error, error_value = _row_value(row, "error")
            has_status, status_value = _row_value(row, "status")
            attempts = _int_or_none(attempts_value)
            if not (has_started and has_attempts and has_error and has_status and attempts is not None):
                return {key: None for key in result}
            if started is None:
                result["never_started"] = (result["never_started"] or 0) + 1
            if attempts <= 0:
                result["zero_attempt"] = (result["zero_attempt"] or 0) + 1
            error = str(error_value or "")
            dispatch_corpse = error.startswith("Turn dispatch failed")
            if dispatch_corpse:
                result["dispatch_corpses"] = (result["dispatch_corpses"] or 0) + 1
            launched = started is not None and attempts > 0
            if launched:
                result["launched"] = (result["launched"] or 0) + 1
            eligible = launched and not dispatch_corpse
            if eligible:
                result["eligible"] = (result["eligible"] or 0) + 1
                failed = str(status_value or "").lower() == "failed"
                if failed and "429" in error:
                    result["excluded_429"] = (result["excluded_429"] or 0) + 1
                elif failed:
                    result["eligible_failures"] = (result["eligible_failures"] or 0) + 1
        return result

    section = _section(receipt, "population", "watchdog", "tasks")
    if not section:
        return None
    names = (
        "raw",
        "launched",
        "eligible",
        "never_started",
        "zero_attempt",
        "dispatch_corpses",
        "eligible_failures",
        "excluded_429",
    )
    if not any(name in section for name in names):
        return None
    result = {name: _int_or_none(section.get(name)) for name in names}
    if result["eligible"] is None:
        result["eligible"] = result["launched"]
    return result


def _rate_and_trigger(
    counts: Mapping[str, int | None], minimum: int | None, threshold_milli: int | None
) -> tuple[int | None, bool | None]:
    eligible = counts.get("eligible")
    failures = counts.get("eligible_failures")
    if eligible is None or failures is None or minimum is None or threshold_milli is None:
        return None, None
    if eligible == 0:
        return 0, False
    rate_milli = failures * 1000 // eligible
    return rate_milli, eligible >= minimum and failures * 1000 >= threshold_milli * eligible


def _control_population() -> tuple[PopulationControl, ...]:
    controls: list[PopulationControl] = []
    corpse_counts = {
        "raw": 5,
        "launched": 0,
        "eligible": 0,
        "dispatch_corpses": 5,
        "eligible_failures": 0,
        "excluded_429": 0,
    }
    rate, trigger = _rate_and_trigger(corpse_counts, 5, 600)
    controls.append(PopulationControl("five_dispatch_corpses", rate_milli=rate, trigger=trigger, **corpse_counts))
    failure_counts = {
        "raw": 5,
        "launched": 5,
        "eligible": 5,
        "dispatch_corpses": 0,
        "eligible_failures": 5,
        "excluded_429": 0,
    }
    rate, trigger = _rate_and_trigger(failure_counts, 5, 600)
    controls.append(PopulationControl("five_launched_non_429_failures", rate_milli=rate, trigger=trigger, **failure_counts))
    return tuple(controls)


def _watchdog_compare(function: _FunctionInfo | None, attribute: str, operator: type[object], expected: object) -> bool:
    if function is None or _ast is None:
        return False
    for node in _ast.walk(function.node):
        if not isinstance(node, _ast.Compare):
            continue
        values = (node.left, *node.comparators)
        if not any(isinstance(value, _ast.Attribute) and value.attr == attribute for value in values):
            continue
        for current_operator, comparator in zip(node.ops, node.comparators):
            if not isinstance(current_operator, operator):
                continue
            if expected is None and isinstance(comparator, _ast.Constant) and comparator.value is None:
                return True
            literal = _literal(comparator)
            if literal == expected:
                return True
            if isinstance(current_operator, (_ast.In, _ast.NotIn)) and isinstance(literal, (tuple, list, frozenset)):
                if expected in literal:
                    return True
    return False


def _watchdog_len_assignment(function: _FunctionInfo | None, name: str) -> bool:
    if function is None or _ast is None:
        return False
    for node in _ast.walk(function.node):
        value: object | None = None
        targets: tuple[object, ...] = ()
        if isinstance(node, _ast.Assign):
            value = node.value
            targets = tuple(node.targets)
        elif isinstance(node, _ast.AnnAssign):
            value = node.value
            targets = (node.target,)
        if not isinstance(value, _ast.Call) or not isinstance(value.func, _ast.Name) or value.func.id != "len":
            continue
        if not value.args or not isinstance(value.args[0], _ast.Name) or value.args[0].id != name:
            continue
        if any(isinstance(target, _ast.Name) and target.id in {"total", "denominator"} for target in targets):
            return True
    return False


def _watchdog_dispatch_exclusion(function: _FunctionInfo | None) -> FactState:
    if function is None or _ast is None:
        return FactState.UNKNOWN
    saw_prefix_check = False
    for node in _ast.walk(function.node):
        if not isinstance(node, _ast.Call) or not isinstance(node.func, _ast.Attribute) or node.func.attr != "startswith":
            continue
        if not node.args or _literal(node.args[0]) != "Turn dispatch failed":
            continue
        saw_prefix_check = True
        for parent in _ast.walk(function.node):
            if isinstance(parent, _ast.UnaryOp) and isinstance(parent.op, _ast.Not) and any(
                child is node for child in _ast.walk(parent.operand)
            ):
                return FactState.TRUE
    return FactState.FALSE if saw_prefix_check else FactState.UNKNOWN


def _watchdog_429_exclusion(function: _FunctionInfo | None) -> FactState:
    if function is None or _ast is None:
        return FactState.UNKNOWN
    saw_inclusion = False
    for node in _ast.walk(function.node):
        if not isinstance(node, _ast.Compare):
            continue
        for current_operator, comparator in zip(node.ops, node.comparators):
            literals = (_literal(node.left), _literal(comparator))
            if "429" not in literals:
                continue
            if isinstance(current_operator, _ast.NotIn):
                return FactState.TRUE
            if isinstance(current_operator, _ast.In):
                saw_inclusion = True
    return FactState.FALSE if saw_inclusion else FactState.UNKNOWN


@dataclass(frozen=True)
class _WatchdogFlow:
    """Static lineage for a value derived from the watchdog population."""

    roots: tuple[str, ...] = ()
    filter_state: FactState = FactState.UNKNOWN
    filter_terms: tuple[str, ...] = ()
    exclusion_state: FactState = FactState.FALSE
    exclusion_lines: tuple[int, ...] = ()
    dynamic: bool = False
    status_predicate: bool = False


@dataclass(frozen=True)
class _WatchdogAggregate:
    """One aggregate expression and the facts statically reaching it."""

    label: str
    iterable: str
    kind: str
    start_line: int
    end_line: int
    filter_state: FactState
    filter_terms: tuple[str, ...]
    exclusion_state: FactState
    exclusion_lines: tuple[int, ...]
    dynamic: bool
    status_predicate: bool
    scalar: bool

    @property
    def failure_relevant(self) -> bool:
        label = self.label.lower()
        return self.status_predicate or any(
            token in label for token in ("failure", "failed", "error", "numerator")
        )


@dataclass(frozen=True)
class _WatchdogThreshold:
    """A threshold comparison and the population aggregates it reaches."""

    target: str
    marker: str
    operator: str
    start_line: int
    end_line: int
    reached: tuple[str, ...]
    state: FactState
    value: object | None = None


@dataclass(frozen=True)
class _WatchdogAstFacts:
    population_state: FactState
    exclusion_state: FactState
    threshold_state: FactState
    population_provenance: tuple[SourceRef, ...]
    exclusion_provenance: tuple[SourceRef, ...]
    threshold_provenance: tuple[SourceRef, ...]
    exclusion_dynamic: bool = False


_WATCHDOG_AGGREGATE_CALLS = frozenset({"all", "any", "len", "max", "min", "sum"})
_WATCHDOG_COLLECTION_CALLS = frozenset({"enumerate", "list", "reversed", "set", "sorted", "tuple"})
_WATCHDOG_COMPREHENSIONS = () if _ast is None else (
    _ast.DictComp,
    _ast.GeneratorExp,
    _ast.ListComp,
    _ast.SetComp,
)
_WATCHDOG_POPULATION_TERMS = frozenset({"started_at", "attempts", "dispatch_corpse"})
_WATCHDOG_THRESHOLD_CONSTANTS = frozenset({
    "_AUTO_PAUSE_FAILURE_RATE_MIN_TASKS",
    "_AUTO_PAUSE_FAILURE_RATE_THRESHOLD",
})
_WATCHDOG_DYNAMIC_DEPENDENCY = "__dynamic_dependency__"
_WATCHDOG_PURE_DEPENDENCY_CALLS = frozenset({"bool", "float", "int", "round", "str"})


def _watchdog_line(node: object) -> int:
    return int(getattr(node, "lineno", 0) or 0)


def _watchdog_end_line(node: object) -> int:
    return int(getattr(node, "end_lineno", _watchdog_line(node)) or _watchdog_line(node))


def _watchdog_unique(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _watchdog_is_429_literal(value: object) -> bool:
    return (isinstance(value, str) and value == "429") or (isinstance(value, int) and not isinstance(value, bool) and value == 429)


def _watchdog_parameter_names(function: _FunctionInfo | None) -> tuple[str, ...]:
    if function is None or _ast is None:
        return ()
    arguments = getattr(function.node, "args", None)
    if arguments is None:
        return ()
    raw_args = (
        tuple(getattr(arguments, "posonlyargs", ()))
        + tuple(getattr(arguments, "args", ()))
        + tuple(getattr(arguments, "kwonlyargs", ()))
    )
    names = [str(getattr(argument, "arg", "")) for argument in raw_args]
    for argument in (getattr(arguments, "vararg", None), getattr(arguments, "kwarg", None)):
        name = str(getattr(argument, "arg", "")) if argument is not None else ""
        if name:
            names.append(name)
    return tuple(name for name in names if name)


def _watchdog_call_name(node: object) -> str:
    if _ast is None:
        return ""
    if isinstance(node, _ast.Name):
        return node.id
    if isinstance(node, _ast.Attribute):
        return node.attr
    return ""


def _watchdog_is_comprehension(node: object) -> bool:
    return _ast is not None and isinstance(node, _WATCHDOG_COMPREHENSIONS)


def _watchdog_is_aggregate_call(node: object) -> bool:
    if _ast is None or not isinstance(node, _ast.Call):
        return False
    return _watchdog_call_name(node.func) in _WATCHDOG_AGGREGATE_CALLS or (
        isinstance(node.func, _ast.Attribute) and node.func.attr == "count"
    )


def _watchdog_target_names(node: object) -> tuple[str, ...]:
    if _ast is None:
        return ()
    if isinstance(node, _ast.Name):
        return (node.id,)
    if isinstance(node, (_ast.List, _ast.Tuple)):
        return _watchdog_unique(
            name
            for element in node.elts
            for name in _watchdog_target_names(element)
        )
    return ()


def _watchdog_assignment_parts(node: object) -> tuple[object | None, tuple[str, ...]]:
    if _ast is None:
        return None, ()
    if isinstance(node, _ast.Assign):
        return node.value, _watchdog_unique(
            name
            for target in node.targets
            for name in _watchdog_target_names(target)
        )
    if isinstance(node, _ast.AnnAssign):
        return node.value, _watchdog_target_names(node.target)
    return None, ()


def _watchdog_population_roots(function: _FunctionInfo | None) -> tuple[str, ...]:
    if function is None or _ast is None:
        return ()
    parameters = _watchdog_parameter_names(function)
    parameter_set = set(parameters)
    used_as_iterable: set[str] = set()
    for node in _ast.walk(function.node):
        iterable_nodes: tuple[object, ...] = ()
        if isinstance(node, _WATCHDOG_COMPREHENSIONS):
            iterable_nodes = tuple(generator.iter for generator in node.generators)
        elif isinstance(node, (_ast.For, _ast.AsyncFor)):
            iterable_nodes = (node.iter,)
        elif _watchdog_is_aggregate_call(node):
            iterable_nodes = tuple(node.args)
            if isinstance(node.func, _ast.Attribute):
                iterable_nodes += (node.func.value,)
        for iterable in iterable_nodes:
            if isinstance(iterable, _ast.Name) and iterable.id in parameter_set:
                used_as_iterable.add(iterable.id)

    named_population = tuple(
        name
        for name in parameters
        if name in used_as_iterable
        and any(token in name.lower() for token in ("task", "population", "row", "item"))
    )
    if named_population:
        return named_population
    parameter_roots = tuple(name for name in parameters if name in used_as_iterable)
    if parameter_roots:
        return parameter_roots
    return tuple(
        name
        for name in parameters
        if any(token in name.lower() for token in ("task", "population", "row", "item"))
    )


def _watchdog_names_in(node: object, names: set[str]) -> tuple[str, ...]:
    if _ast is None:
        return ()
    return _watchdog_unique(
        child.id
        for child in _ast.walk(node)
        if isinstance(child, _ast.Name) and child.id in names
    )


def _watchdog_merge_filter(left: FactState, right: FactState) -> FactState:
    if left is FactState.UNKNOWN or right is FactState.UNKNOWN:
        return FactState.UNKNOWN
    if left is FactState.TRUE or right is FactState.TRUE:
        return FactState.TRUE
    return FactState.FALSE


def _watchdog_merge_exclusion(left: FactState, right: FactState) -> FactState:
    if left is FactState.UNKNOWN or right is FactState.UNKNOWN:
        return FactState.UNKNOWN
    if left is FactState.FALSE:
        return right
    if right is FactState.FALSE:
        return left
    return FactState.TRUE


def _watchdog_merge_flows(flows: Sequence[_WatchdogFlow]) -> _WatchdogFlow | None:
    if not flows:
        return None
    roots: list[str] = []
    filter_state = FactState.FALSE
    exclusion_state = FactState.FALSE
    terms: list[str] = []
    exclusion_lines: list[int] = []
    dynamic = False
    status_predicate = False
    for flow in flows:
        roots.extend(flow.roots)
        filter_state = _watchdog_merge_filter(filter_state, flow.filter_state)
        exclusion_state = _watchdog_merge_exclusion(exclusion_state, flow.exclusion_state)
        terms.extend(flow.filter_terms)
        exclusion_lines.extend(flow.exclusion_lines)
        dynamic = dynamic or flow.dynamic
        status_predicate = status_predicate or flow.status_predicate
    return _WatchdogFlow(
        roots=_watchdog_unique(roots),
        filter_state=filter_state,
        filter_terms=_watchdog_unique(terms),
        exclusion_state=exclusion_state,
        exclusion_lines=tuple(sorted(set(exclusion_lines))),
        dynamic=dynamic,
        status_predicate=status_predicate,
    )


def _watchdog_429_signal(condition: object) -> tuple[FactState, tuple[int, ...]]:
    """Classify literal 429 comparisons, including an enclosing ``not``."""

    if _ast is None:
        return FactState.UNKNOWN, ()
    positive = False
    negative = False
    lines: list[int] = []

    def visit(node: object, negated: bool = False) -> None:
        nonlocal positive, negative
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.Not):
            visit(node.operand, not negated)
            return
        if isinstance(node, _ast.BoolOp):
            for value in node.values:
                visit(value, negated)
            return
        if not isinstance(node, _ast.Compare):
            for child in _ast.iter_child_nodes(node):
                visit(child, negated)
            return
        for operator, comparator in zip(node.ops, node.comparators):
            left_literal = _literal(node.left)
            right_literal = _literal(comparator)
            if not _watchdog_is_429_literal(left_literal) and not _watchdog_is_429_literal(right_literal):
                continue
            is_negative = isinstance(operator, _ast.NotIn)
            if negated:
                is_negative = not is_negative
            lines.append(_watchdog_line(node))
            if is_negative:
                negative = True
            else:
                positive = True

    visit(condition)
    if not positive and not negative:
        return FactState.FALSE, ()
    if positive and negative:
        return FactState.UNKNOWN, tuple(sorted(set(lines)))
    if negative:
        return FactState.TRUE, tuple(sorted(set(lines)))
    return FactState.FALSE, tuple(sorted(set(lines)))


def _watchdog_dynamic_condition(condition: object) -> bool:
    if _ast is None:
        return True
    for node in _ast.walk(condition):
        if not isinstance(node, _ast.Call):
            continue
        if isinstance(node.func, _ast.Attribute) and node.func.attr == "startswith":
            if node.args and isinstance(_literal(node.args[0]), str):
                continue
        if isinstance(node.func, _ast.Name) and node.func.id in {"all", "any", "bool", "len"}:
            continue
        return True
    return False


def _watchdog_condition_terms(condition: object) -> tuple[str, ...]:
    if _ast is None:
        return ()
    terms: list[str] = []
    for node in _ast.walk(condition):
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute):
            if node.func.attr == "startswith" and node.args and _literal(node.args[0]) == "Turn dispatch failed":
                terms.append("dispatch_corpse")
        if not isinstance(node, _ast.Compare):
            continue
        attributes = {
            child.attr
            for child in _ast.walk(node)
            if isinstance(child, _ast.Attribute)
        }
        pairs = tuple(zip(node.ops, node.comparators))
        if "started_at" in attributes and any(
            isinstance(operator, _ast.IsNot) and (
                (_literal(comparator) is None and isinstance(comparator, _ast.Constant) and comparator.value is None)
            )
            for operator, comparator in pairs
        ):
            terms.append("started_at")
        if "attempts" in attributes and any(
            isinstance(operator, _ast.Gt) and _literal(comparator) == 0
            for operator, comparator in pairs
        ):
            terms.append("attempts")
        if "status" in attributes and any(
            isinstance(operator, _ast.Eq) and _literal(comparator) == "failed"
            for operator, comparator in pairs
        ):
            terms.append("status_failed")
    exclusion, _lines = _watchdog_429_signal(condition)
    if exclusion is FactState.TRUE:
        terms.append("429_exclusion")
    elif exclusion is FactState.FALSE and _watchdog_names_in(condition, {"error", "status"}):
        # Keep a positive 429 check visible in provenance without treating it as
        # one of the launch/attempt population predicates.
        if any(
            isinstance(node, _ast.Compare)
            and any(_watchdog_is_429_literal(_literal(comparator)) for comparator in node.comparators)
            for node in _ast.walk(condition)
        ):
            terms.append("429_inclusion")
    elif exclusion is FactState.UNKNOWN:
        terms.append("429_dynamic")
    return _watchdog_unique(terms)


def _watchdog_filter_analysis(
    conditions: Sequence[object],
) -> tuple[FactState, tuple[str, ...], FactState, tuple[int, ...], bool, bool]:
    if not conditions:
        return FactState.FALSE, (), FactState.FALSE, (), False, False
    terms: list[str] = []
    exclusion_state = FactState.FALSE
    exclusion_lines: list[int] = []
    dynamic = False
    status_predicate = False
    for condition in conditions:
        dynamic = dynamic or _watchdog_dynamic_condition(condition)
        condition_terms = _watchdog_condition_terms(condition)
        terms.extend(condition_terms)
        status_predicate = status_predicate or "status_failed" in condition_terms
        signal, lines = _watchdog_429_signal(condition)
        exclusion_state = _watchdog_merge_exclusion(exclusion_state, signal)
        exclusion_lines.extend(lines)
    if dynamic:
        exclusion_state = FactState.UNKNOWN
    return (
        FactState.UNKNOWN if dynamic else FactState.TRUE,
        _watchdog_unique(terms),
        exclusion_state,
        tuple(sorted(set(exclusion_lines))),
        dynamic,
        status_predicate,
    )


def _watchdog_flow_for_expr(
    expression: object,
    roots: tuple[str, ...],
    flows: Mapping[str, _WatchdogFlow],
) -> _WatchdogFlow | None:
    if _ast is None:
        return None
    if isinstance(expression, _ast.Name):
        if expression.id in roots:
            return _WatchdogFlow(roots=(expression.id,), filter_state=FactState.FALSE)
        return flows.get(expression.id)
    if _watchdog_is_comprehension(expression):
        return _watchdog_flow_for_comprehension(expression, roots, flows)
    if isinstance(expression, _ast.Starred):
        return _watchdog_flow_for_expr(expression.value, roots, flows)
    if isinstance(expression, _ast.Call):
        name = _watchdog_call_name(expression.func)
        arguments = list(expression.args) + [keyword.value for keyword in expression.keywords]
        argument_flows = [
            flow
            for argument in arguments
            for flow in (_watchdog_flow_for_expr(argument, roots, flows),)
            if flow is not None
        ]
        if argument_flows and name in _WATCHDOG_COLLECTION_CALLS:
            return _watchdog_merge_flows(argument_flows)
        if _watchdog_names_in(expression, set(roots)):
            base = _watchdog_merge_flows(argument_flows)
            return _WatchdogFlow(
                roots=base.roots if base else roots,
                filter_state=FactState.UNKNOWN,
                filter_terms=base.filter_terms if base else (),
                exclusion_state=FactState.UNKNOWN,
                exclusion_lines=base.exclusion_lines if base else (),
                dynamic=True,
                status_predicate=base.status_predicate if base else False,
            )
        return None
    if isinstance(expression, _ast.Subscript):
        base = _watchdog_flow_for_expr(expression.value, roots, flows)
        if base is None:
            return None
        return replace(base, filter_state=_watchdog_merge_filter(base.filter_state, FactState.TRUE))
    if isinstance(expression, (_ast.List, _ast.Set, _ast.Tuple)):
        child_flows = [
            flow
            for element in expression.elts
            for flow in (_watchdog_flow_for_expr(element, roots, flows),)
            if flow is not None
        ]
        return _watchdog_merge_flows(child_flows)
    return None


def _watchdog_flow_for_comprehension(
    expression: object,
    roots: tuple[str, ...],
    flows: Mapping[str, _WatchdogFlow],
) -> _WatchdogFlow | None:
    if _ast is None or not _watchdog_is_comprehension(expression):
        return None
    iterable_flows: list[_WatchdogFlow] = []
    conditions: list[object] = []
    for generator in expression.generators:
        flow = _watchdog_flow_for_expr(generator.iter, roots, flows)
        if flow is None and _watchdog_names_in(generator.iter, set(roots)):
            flow = _WatchdogFlow(roots=_watchdog_names_in(generator.iter, set(roots)), filter_state=FactState.UNKNOWN, dynamic=True)
        if flow is not None:
            iterable_flows.append(flow)
        conditions.extend(generator.ifs)
    base = _watchdog_merge_flows(iterable_flows)
    if base is None:
        return None
    filter_state, terms, exclusion, lines, dynamic, status_predicate = _watchdog_filter_analysis(conditions)
    return _WatchdogFlow(
        roots=base.roots,
        filter_state=_watchdog_merge_filter(base.filter_state, filter_state),
        filter_terms=_watchdog_unique(base.filter_terms + terms),
        exclusion_state=_watchdog_merge_exclusion(base.exclusion_state, exclusion),
        exclusion_lines=tuple(sorted(set(base.exclusion_lines + lines))),
        dynamic=base.dynamic or dynamic,
        status_predicate=base.status_predicate or status_predicate,
    )


def _watchdog_flow_for_aggregate(
    call: object,
    roots: tuple[str, ...],
    flows: Mapping[str, _WatchdogFlow],
) -> _WatchdogFlow | None:
    if _ast is None or not isinstance(call, _ast.Call) or not _watchdog_is_aggregate_call(call):
        return None
    if isinstance(call.func, _ast.Attribute) and call.func.attr == "count":
        return _watchdog_flow_for_expr(call.func.value, roots, flows)
    arguments = list(call.args) + [keyword.value for keyword in call.keywords]
    if not arguments:
        return None
    candidate = arguments[0]
    if _watchdog_is_comprehension(candidate):
        return _watchdog_flow_for_comprehension(candidate, roots, flows)
    return _watchdog_flow_for_expr(candidate, roots, flows)


def _watchdog_aggregate_analysis(
    function: _FunctionInfo | None,
) -> tuple[
    tuple[str, ...],
    tuple[_WatchdogAggregate, ...],
    Mapping[str, tuple[str, ...]],
    Mapping[int, str],
]:
    if function is None or _ast is None:
        return (), (), {}, {}
    roots = _watchdog_population_roots(function)
    if not roots:
        return (), (), {}, {}
    flows: dict[str, _WatchdogFlow] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    node_labels: dict[int, str] = {}
    aggregates: list[_WatchdogAggregate] = []
    aggregate_nodes: dict[int, object] = {}
    assigned_comprehensions: set[int] = set()
    dependency_assignments: list[tuple[str, object]] = []

    def unique_label(base: str) -> str:
        labels = {aggregate.label for aggregate in aggregates}
        if base not in labels:
            return base
        index = 2
        while f"{base}#{index}" in labels:
            index += 1
        return f"{base}#{index}"

    def add_aggregate(label: str, expression: object, flow: _WatchdogFlow, kind: str) -> str:
        actual_label = unique_label(label or kind)
        aggregates.append(
            _WatchdogAggregate(
                label=actual_label,
                iterable=", ".join(flow.roots),
                kind=kind,
                start_line=_watchdog_line(expression),
                end_line=_watchdog_end_line(expression),
                filter_state=flow.filter_state,
                filter_terms=flow.filter_terms,
                exclusion_state=flow.exclusion_state,
                exclusion_lines=flow.exclusion_lines,
                dynamic=flow.dynamic,
                status_predicate=flow.status_predicate,
                scalar=kind not in {"dictcomp", "genexpr", "listcomp", "setcomp"},
            )
        )
        node_labels[id(expression)] = actual_label
        aggregate_nodes[id(expression)] = expression
        return actual_label

    def dependencies_for(expression: object) -> tuple[str, ...]:
        if isinstance(expression, _ast.Name):
            return dependencies.get(expression.id, ())
        direct = node_labels.get(id(expression))
        if direct:
            return (direct,)
        values: list[str] = []
        for child in _ast.iter_child_nodes(expression):
            values.extend(dependencies_for(child))
        return _watchdog_unique(values)

    def dependency_is_dynamic(expression: object) -> bool:
        if not dependencies_for(expression):
            return False
        if isinstance(expression, _ast.Name):
            return _WATCHDOG_DYNAMIC_DEPENDENCY in dependencies.get(expression.id, ())
        if node_labels.get(id(expression)):
            return False
        if isinstance(expression, _ast.Call):
            children = tuple(expression.args) + tuple(keyword.value for keyword in expression.keywords)
            child_dynamic = any(dependency_is_dynamic(child) for child in children)
            if child_dynamic:
                return True
            if _watchdog_call_name(expression.func) in _WATCHDOG_PURE_DEPENDENCY_CALLS:
                return False
            return True
        return any(dependency_is_dynamic(child) for child in _ast.iter_child_nodes(expression))

    assignment_nodes = sorted(
        (
            node
            for node in _ast.walk(function.node)
            if isinstance(node, (_ast.Assign, _ast.AnnAssign))
        ),
        key=lambda node: (_watchdog_line(node), int(getattr(node, "col_offset", 0) or 0)),
    )
    for assignment in assignment_nodes:
        value, targets = _watchdog_assignment_parts(assignment)
        if value is None:
            continue
        dependency_assignments.extend((target, value) for target in targets)
        if _watchdog_is_comprehension(value):
            flow = _watchdog_flow_for_comprehension(value, roots, flows)
            if flow is not None:
                assigned_comprehensions.add(id(value))
                for target in targets or ("comprehension",):
                    label = add_aggregate(target, value, flow, type(value).__name__.lower())
                    if target != "comprehension":
                        flows[target] = flow
                        dependencies[target] = (label,)
            continue
        if _watchdog_is_aggregate_call(value):
            flow = _watchdog_flow_for_aggregate(value, roots, flows)
            if flow is not None:
                label = add_aggregate(targets[0] if targets else _watchdog_call_name(value.func), value, flow, _watchdog_call_name(value.func))
                for target in targets:
                    dependencies[target] = (label,)
            continue
        for target in targets:
            flow = _watchdog_flow_for_expr(value, roots, flows)
            if flow is not None:
                flows[target] = flow
            else:
                flows.pop(target, None)
            target_dependencies = dependencies_for(value)
            if target_dependencies:
                dependencies[target] = target_dependencies
            else:
                dependencies.pop(target, None)

    all_calls = sorted(
        (node for node in _ast.walk(function.node) if _watchdog_is_aggregate_call(node)),
        key=lambda node: (_watchdog_line(node), int(getattr(node, "col_offset", 0) or 0)),
    )
    for call in all_calls:
        if id(call) in node_labels:
            continue
        flow = _watchdog_flow_for_aggregate(call, roots, flows)
        if flow is not None:
            add_aggregate(_watchdog_call_name(call.func), call, flow, _watchdog_call_name(call.func))

    consumed_comprehensions: set[int] = set(assigned_comprehensions)
    for call in aggregate_nodes.values():
        for child in _ast.walk(call):
            if _watchdog_is_comprehension(child):
                consumed_comprehensions.add(id(child))
    all_comprehensions = sorted(
        (node for node in _ast.walk(function.node) if _watchdog_is_comprehension(node)),
        key=lambda node: (_watchdog_line(node), int(getattr(node, "col_offset", 0) or 0)),
    )
    for comprehension in all_comprehensions:
        if id(comprehension) in consumed_comprehensions:
            continue
        flow = _watchdog_flow_for_comprehension(comprehension, roots, flows)
        if flow is not None:
            add_aggregate(
                f"comprehension@L{_watchdog_line(comprehension)}",
                comprehension,
                flow,
                type(comprehension).__name__.lower(),
            )

    # Inline aggregates such as ``len(failures) / total`` are indexed only after
    # their source alias has been seen.  Rebuild dependencies once every aggregate
    # node has a stable label so threshold reach is complete and deterministic.
    dependencies.clear()
    for target, value in dependency_assignments:
        label = node_labels.get(id(value))
        if label and (_watchdog_is_comprehension(value) or _watchdog_is_aggregate_call(value)):
            dependencies[target] = (label,)
            continue
        if dependency_is_dynamic(value):
            dependencies[target] = (_WATCHDOG_DYNAMIC_DEPENDENCY,)
            continue
        target_dependencies = dependencies_for(value)
        if target_dependencies:
            dependencies[target] = target_dependencies

    aggregates.sort(key=lambda aggregate: (aggregate.start_line, aggregate.label, aggregate.kind))
    return tuple(roots), tuple(aggregates), dependencies, node_labels


def _watchdog_filter_label(state: FactState) -> str:
    if state is FactState.TRUE:
        return "FILTERED"
    if state is FactState.FALSE:
        return "UNFILTERED"
    return "UNDECIDABLE"


def _watchdog_population_coverage(aggregate: _WatchdogAggregate) -> str:
    terms = set(aggregate.filter_terms)
    if _WATCHDOG_POPULATION_TERMS.issubset(terms):
        return "FULL"
    if terms.intersection(_WATCHDOG_POPULATION_TERMS):
        return "PARTIAL"
    return "NONE"


def _watchdog_aggregate_ref(
    function: _FunctionInfo,
    aggregate: _WatchdogAggregate,
    fact: str,
) -> SourceRef:
    return SourceRef(
        path=function.path,
        symbol=function.name,
        start_line=aggregate.start_line,
        end_line=aggregate.end_line,
        fact=fact,
    )


def _watchdog_population_provenance(
    function: _FunctionInfo,
    roots: tuple[str, ...],
    aggregates: Sequence[_WatchdogAggregate],
) -> tuple[SourceRef, ...]:
    iterable = ", ".join(roots) or "population"
    summary = ", ".join(
        f"{aggregate.label}(L{aggregate.start_line})={_watchdog_filter_label(aggregate.filter_state)}"
        for aggregate in aggregates
    ) or "none=UNDECIDABLE"
    refs = [
        SourceRef(
            path=function.path,
            symbol=function.name,
            start_line=function.start_line,
            end_line=function.end_line,
            fact=f"aggregates over {iterable}: {summary}",
        )
    ]
    refs.extend(
        _watchdog_aggregate_ref(
            function,
            aggregate,
            "; ".join(
                (
                    f"aggregate={aggregate.label}",
                    f"iterable={aggregate.iterable}",
                    f"population_filter={_watchdog_filter_label(aggregate.filter_state)}",
                    f"population_predicate={_watchdog_population_coverage(aggregate)}",
                    f"predicates={','.join(aggregate.filter_terms) or 'NONE'}",
                )
            ),
        )
        for aggregate in aggregates
    )
    return tuple(refs)


def _watchdog_429_label(aggregate: _WatchdogAggregate) -> str:
    if not aggregate.failure_relevant:
        return "NOT_APPLICABLE"
    if aggregate.exclusion_state is FactState.TRUE:
        return "EXCLUDED"
    if aggregate.exclusion_state is FactState.FALSE:
        return "NOT_EXCLUDED"
    return "UNDECIDABLE"


def _watchdog_429_provenance(
    function: _FunctionInfo,
    roots: tuple[str, ...],
    aggregates: Sequence[_WatchdogAggregate],
) -> tuple[SourceRef, ...]:
    iterable = ", ".join(roots) or "population"
    summary = ", ".join(
        f"{aggregate.label}(L{aggregate.start_line})={_watchdog_429_label(aggregate)}"
        for aggregate in aggregates
    ) or "none=UNDECIDABLE"
    refs: list[SourceRef] = [
        SourceRef(
            path=function.path,
            symbol=function.name,
            start_line=function.start_line,
            end_line=function.end_line,
            fact=f"429 exclusion over {iterable}: {summary}",
        )
    ]
    for aggregate in aggregates:
        label = _watchdog_429_label(aggregate)
        check_span = ",".join(f"L{line}" for line in aggregate.exclusion_lines) or "NONE"
        refs.append(
            _watchdog_aggregate_ref(
                function,
                aggregate,
                f"aggregate={aggregate.label}; iterable={aggregate.iterable}; 429={label}; check_span={check_span}",
            )
        )
        for line in aggregate.exclusion_lines:
            refs.append(
                SourceRef(
                    path=function.path,
                    symbol=function.name,
                    start_line=line,
                    end_line=line,
                    fact=f"aggregate={aggregate.label}; 429={label}; exclusion_check_span=L{line}",
                )
            )
    return tuple(refs)


def _watchdog_threshold_operator(operator: object) -> str:
    if _ast is None:
        return "?"
    names = {
        _ast.Eq: "==",
        _ast.Gt: ">",
        _ast.GtE: ">=",
        _ast.Lt: "<",
        _ast.LtE: "<=",
        _ast.NotEq: "!=",
    }
    for operator_type, symbol in names.items():
        if isinstance(operator, operator_type):
            return symbol
    return type(operator).__name__


def _watchdog_threshold_marker(
    expression: object,
    constants: Mapping[str, object],
) -> tuple[str, bool, object | None] | None:
    if _ast is None:
        return None
    if isinstance(expression, _ast.Name):
        name = expression.id
        if name in _WATCHDOG_THRESHOLD_CONSTANTS or any(
            token in name.upper() for token in ("THRESHOLD", "MIN_TASK", "MIN_POPULATION")
        ):
            return name, name in constants, constants.get(name)
        return None
    literal = _literal(expression)
    if isinstance(literal, (int, float)) and not isinstance(literal, bool):
        return f"literal:{literal}", True, literal
    return None


def _watchdog_threshold_target_hint(expression: object) -> bool:
    if _ast is None:
        return False
    if isinstance(expression, _ast.Name):
        lowered = expression.id.lower()
        return any(token in lowered for token in ("total", "denominator", "rate", "failure", "eligible", "count"))
    return _watchdog_is_aggregate_call(expression)


def _watchdog_threshold_analysis(
    function: _FunctionInfo | None,
    roots: tuple[str, ...],
    aggregates: Sequence[_WatchdogAggregate],
    dependencies: Mapping[str, tuple[str, ...]],
    node_labels: Mapping[int, str],
    constants: Mapping[str, object],
) -> tuple[FactState, tuple[_WatchdogThreshold, ...]]:
    if function is None or _ast is None:
        return FactState.UNKNOWN, ()

    def dependencies_for(expression: object) -> tuple[str, ...]:
        if isinstance(expression, _ast.Name):
            return dependencies.get(expression.id, ())
        direct = node_labels.get(id(expression))
        if direct:
            return (direct,)
        return _watchdog_unique(
            dependency
            for child in _ast.iter_child_nodes(expression)
            for dependency in dependencies_for(child)
        )

    aggregate_by_label = {aggregate.label: aggregate for aggregate in aggregates}
    observations: list[_WatchdogThreshold] = []
    for compare in sorted(
        (node for node in _ast.walk(function.node) if isinstance(node, _ast.Compare)),
        key=lambda node: (_watchdog_line(node), int(getattr(node, "col_offset", 0) or 0)),
    ):
        for operator, comparator in zip(compare.ops, compare.comparators):
            left = compare.left
            right = comparator
            left_dependencies = dependencies_for(left)
            right_dependencies = dependencies_for(right)
            left_marker = _watchdog_threshold_marker(left, constants)
            right_marker = _watchdog_threshold_marker(right, constants)
            target: object | None = None
            marker: tuple[str, bool, object | None] | None = None
            if left_dependencies and right_marker:
                target, marker = left, right_marker
            elif right_dependencies and left_marker:
                target, marker = right, left_marker
            elif right_marker and _watchdog_threshold_target_hint(left):
                target, marker = left, right_marker
            elif left_marker and _watchdog_threshold_target_hint(right):
                target, marker = right, left_marker
            if target is None or marker is None:
                continue
            reached = _watchdog_unique(
                label for label in dependencies_for(target) if label in aggregate_by_label
            )
            reached_dynamic = any(aggregate_by_label[label].dynamic for label in reached)
            target_label = (
                target.id
                if isinstance(target, _ast.Name)
                else node_labels.get(id(target), _watchdog_call_name(getattr(target, "func", None)))
            )
            state = FactState.TRUE if marker[1] and reached and not reached_dynamic else FactState.UNKNOWN
            observations.append(
                _WatchdogThreshold(
                    target=target_label or "aggregate",
                    marker=marker[0],
                    operator=_watchdog_threshold_operator(operator),
                    start_line=_watchdog_line(compare),
                    end_line=_watchdog_end_line(compare),
                    reached=reached,
                    state=state,
                    value=marker[2],
                )
            )

    scalar_aggregates = tuple(aggregate for aggregate in aggregates if aggregate.scalar)
    if not scalar_aggregates:
        return FactState.UNKNOWN, tuple(observations)
    if not observations:
        return FactState.FALSE, ()
    if any(observation.state is FactState.UNKNOWN for observation in observations):
        return FactState.UNKNOWN, tuple(observations)
    expected_values: Mapping[str, object] = {
        "_AUTO_PAUSE_FAILURE_RATE_MIN_TASKS": 5,
        "_AUTO_PAUSE_FAILURE_RATE_THRESHOLD": 0.6,
    }
    for observation in observations:
        expected = expected_values.get(observation.marker)
        if expected is not None and observation.value != expected:
            return FactState.FALSE, tuple(observations)
    reached = {label for observation in observations for label in observation.reached}
    if not all(aggregate.label in reached for aggregate in scalar_aggregates):
        return FactState.FALSE, tuple(observations)
    return FactState.TRUE, tuple(observations)


def _watchdog_threshold_provenance(
    function: _FunctionInfo,
    aggregates: Sequence[_WatchdogAggregate],
    observations: Sequence[_WatchdogThreshold],
) -> tuple[SourceRef, ...]:
    aggregate_by_label = {aggregate.label: aggregate for aggregate in aggregates}
    if observations:
        summary_parts = []
        for observation in observations:
            reached = ",".join(
                f"{label}(L{aggregate_by_label[label].start_line})"
                for label in observation.reached
                if label in aggregate_by_label
            ) or "NONE"
            summary_parts.append(
                f"{observation.target}(L{observation.start_line}){observation.operator}{observation.marker}"
                f" -> {reached}"
            )
        summary = "; ".join(summary_parts)
    else:
        summary = "none=NOT_FOUND"
    refs: list[SourceRef] = [
        SourceRef(
            path=function.path,
            symbol=function.name,
            start_line=function.start_line,
            end_line=function.end_line,
            fact=f"threshold boundaries: {summary}",
        )
    ]
    for observation in observations:
        reached = ",".join(observation.reached) or "NONE"
        refs.append(
            SourceRef(
                path=function.path,
                symbol=function.name,
                start_line=observation.start_line,
                end_line=observation.end_line,
                fact=(
                    f"threshold={observation.marker}; target={observation.target}; "
                    f"operator={observation.operator}; reaches={reached}; "
                    f"boundary={'REACHED' if observation.state is FactState.TRUE else 'UNDECIDABLE'}; "
                    f"value={observation.value!r}"
                ),
            )
        )
        for label in observation.reached:
            aggregate = aggregate_by_label.get(label)
            if aggregate is None:
                continue
            refs.append(
                _watchdog_aggregate_ref(
                    function,
                    aggregate,
                    f"aggregate={label}; threshold={observation.marker}; reach=REACHED; boundary_span=L{observation.start_line}",
                )
            )
    reached_labels = {label for observation in observations for label in observation.reached}
    for aggregate in aggregates:
        if aggregate.scalar and aggregate.label not in reached_labels:
            refs.append(
                _watchdog_aggregate_ref(
                    function,
                    aggregate,
                    f"aggregate={aggregate.label}; threshold_boundary=NOT_REACHED",
                )
            )
    return tuple(refs)


def _watchdog_ast_measurement(
    function: _FunctionInfo | None,
    constants: Mapping[str, object],
) -> _WatchdogAstFacts:
    if function is None or _ast is None:
        return _WatchdogAstFacts(
            population_state=FactState.UNKNOWN,
            exclusion_state=FactState.UNKNOWN,
            threshold_state=FactState.UNKNOWN,
            population_provenance=(),
            exclusion_provenance=(),
            threshold_provenance=(),
            exclusion_dynamic=False,
        )
    roots, aggregates, dependencies, node_labels = _watchdog_aggregate_analysis(function)
    population_state = FactState.UNKNOWN
    if aggregates:
        filter_states = tuple(aggregate.filter_state for aggregate in aggregates)
        if all(state is FactState.TRUE for state in filter_states):
            population_state = FactState.TRUE
        elif all(state is FactState.FALSE for state in filter_states):
            population_state = FactState.FALSE
        elif not any(state is FactState.UNKNOWN for state in filter_states):
            population_state = FactState.FALSE
    relevant = tuple(aggregate for aggregate in aggregates if aggregate.failure_relevant)
    exclusion_dynamic = any(
        aggregate.exclusion_state is FactState.UNKNOWN and aggregate.dynamic
        for aggregate in relevant
    )
    if not relevant:
        exclusion_state = FactState.UNKNOWN if exclusion_dynamic else FactState.FALSE
    elif any(aggregate.exclusion_state is FactState.UNKNOWN for aggregate in relevant):
        exclusion_state = FactState.UNKNOWN
    elif all(aggregate.exclusion_state is FactState.TRUE for aggregate in relevant):
        exclusion_state = FactState.TRUE
    else:
        exclusion_state = FactState.FALSE
    threshold_state, observations = _watchdog_threshold_analysis(
        function,
        roots,
        aggregates,
        dependencies,
        node_labels,
        constants,
    )
    return _WatchdogAstFacts(
        population_state=population_state,
        exclusion_state=exclusion_state,
        threshold_state=threshold_state,
        population_provenance=_watchdog_population_provenance(function, roots, aggregates),
        exclusion_provenance=_watchdog_429_provenance(function, roots, aggregates),
        threshold_provenance=_watchdog_threshold_provenance(function, aggregates, observations),
        exclusion_dynamic=exclusion_dynamic,
    )


def _watchdog_source_features(function: _FunctionInfo | None) -> tuple[bool, bool, FactState, bool, bool]:
    if function is None or _ast is None:
        return False, False, FactState.UNKNOWN, False, False
    has_started = _watchdog_compare(function, "started_at", _ast.IsNot, None)
    has_attempts = _watchdog_compare(function, "attempts", _ast.Gt, 0)
    exclusion = _watchdog_dispatch_exclusion(function)
    has_status = _watchdog_compare(function, "status", _ast.Eq, "failed")
    return (
        has_started,
        has_attempts,
        exclusion,
        has_status,
        _watchdog_len_assignment(function, "launched"),
    )


class WatchdogPopulationProducer(_Producer):
    NAME = _ProducerName.WATCHDOG.value
    TARGETS = ("_watchdog_auto_pause_trigger", "watchdog_auto_pause_trigger")
    REQUIRED = WATCHDOG_REQUIRED
    SECTION_NAMES = ("population", "watchdog", "tasks")

    def _derive(
        self,
        receipt: _NormalizedReceipt,
        source: SourceInput,
        *,
        task_id: str,
        frontier_hypothesis_id: str | None,
    ) -> SemanticEvidence:
        ctx = _SourceContext(source)
        subject, function = _subject_for(receipt, ctx, self.TARGETS)
        if function is None:
            found = ctx.find(self.TARGETS[0])
            function = found[0] if found else None
        constants = _assignment_literals(
            ctx,
            {"_AUTO_PAUSE_FAILURE_RATE_MIN_TASKS", "_AUTO_PAUSE_FAILURE_RATE_THRESHOLD"},
        )
        minimum_value = constants.get("_AUTO_PAUSE_FAILURE_RATE_MIN_TASKS")
        minimum = int(minimum_value) if isinstance(minimum_value, int) and not isinstance(minimum_value, bool) else None
        threshold_value = constants.get("_AUTO_PAUSE_FAILURE_RATE_THRESHOLD")
        threshold_milli: int | None = None
        if isinstance(threshold_value, (int, float)) and not isinstance(threshold_value, bool):
            threshold_milli = int(Fraction(str(threshold_value)) * 1000)
        has_started, has_attempts, dispatch_exclusion, has_failed_status, good_denominator = _watchdog_source_features(function)
        bad_denominator = _watchdog_len_assignment(function, "recent_tasks")
        if bad_denominator:
            population_source_state = FactState.FALSE
        elif has_started and has_attempts and dispatch_exclusion is FactState.TRUE and has_failed_status and good_denominator:
            population_source_state = FactState.TRUE
        else:
            population_source_state = FactState.UNKNOWN

        counts = _population_counts(receipt)
        if minimum is None:
            minimum = 5
        if threshold_milli is None:
            threshold_milli = 600
        rate_milli, trigger = _rate_and_trigger(counts or {}, minimum, threshold_milli) if counts else (None, None)

        ast_facts = _watchdog_ast_measurement(function, constants)
        exclusion_state = ast_facts.exclusion_state
        if (
            exclusion_state is FactState.UNKNOWN
            and not ast_facts.exclusion_dynamic
            and counts
            and (counts.get("excluded_429") or 0) > 0
        ):
            exclusion_state = FactState.FALSE

        population_state = ast_facts.population_state
        if not has_failed_status:
            # A failure aggregate without a statically identifiable failed-status
            # predicate cannot be classified as a watchdog population measurement.
            population_state = FactState.UNKNOWN
        if counts is None and population_state is FactState.TRUE:
            population_state = FactState.UNKNOWN
        threshold_state = ast_facts.threshold_state
        if counts is None and threshold_state is FactState.TRUE:
            threshold_state = FactState.UNKNOWN
        payload = PopulationFacts(
            raw=counts.get("raw") if counts else None,
            launched=counts.get("launched") if counts else None,
            eligible=counts.get("eligible") if counts else None,
            never_started=counts.get("never_started") if counts else None,
            zero_attempt=counts.get("zero_attempt") if counts else None,
            dispatch_corpses=counts.get("dispatch_corpses") if counts else None,
            eligible_failures=counts.get("eligible_failures") if counts else None,
            excluded_429=counts.get("excluded_429") if counts else None,
            rate_milli=rate_milli,
            minimum_population=minimum,
            threshold_milli=threshold_milli,
            trigger=trigger,
            controls=_control_population(),
            source_population_predicate=population_source_state,
        )
        states = {
            EvidenceCode.WATCHDOG_POPULATION_WITNESS: population_state,
            EvidenceCode.WATCHDOG_429_EXCLUSION: exclusion_state,
            EvidenceCode.WATCHDOG_THRESHOLD_BOUNDARY: threshold_state,
        }
        function_ref = (function.source_ref("watchdog_population"),) if function else (receipt.ref,)
        limitations: list[str] = list(ctx.errors)
        limitations.append("control_simulations_are_not_runtime_observations")
        if counts is None:
            limitations.append("receipt_population_rows_missing")
        if counts and counts.get("raw") is None:
            limitations.append("receipt_population_fields_incomplete")
        predicate = PredicateRef(
            name="watchdog_failure_population",
            expression="started_at != None and attempts > 0 and not dispatch-failure corpse",
            subject=f"{subject.path}:{subject.symbol}",
        )
        return _make_evidence(
            producer=self.NAME,
            task_id=task_id,
            frontier_hypothesis_id=frontier_hypothesis_id,
            receipt=receipt,
            ctx=ctx,
            subject=subject,
            predicate=predicate,
            payload=payload,
            required=self.REQUIRED,
            states=states,
            provenance={
                EvidenceCode.WATCHDOG_POPULATION_WITNESS: function_ref + ast_facts.population_provenance + (receipt.ref,),
                EvidenceCode.WATCHDOG_429_EXCLUSION: function_ref + ast_facts.exclusion_provenance,
                EvidenceCode.WATCHDOG_THRESHOLD_BOUNDARY: function_ref + ast_facts.threshold_provenance,
            },
            limitations=limitations,
        )


def _request_calls(function: _FunctionInfo | None) -> tuple[object, ...]:
    if function is None or _ast is None:
        return ()
    result: list[object] = []
    for node in _ast.walk(function.node):
        if not isinstance(node, _ast.Call) or not isinstance(node.func, _ast.Attribute):
            continue
        if node.func.attr in {"get", "head"}:
            result.append(node)
    return tuple(sorted(result, key=lambda item: int(getattr(item, "lineno", 0) or 0)))


def _validate_calls(node: object) -> tuple[object, ...]:
    if _ast is None:
        return ()
    return tuple(
        child
        for child in _ast.walk(node)
        if isinstance(child, _ast.Call)
        and (
            (isinstance(child.func, _ast.Name) and child.func.id in {"_validate", "validate_external_url"})
            or (isinstance(child.func, _ast.Attribute) and child.func.attr in {"_validate", "validate_external_url"})
        )
    )


def _loop_nodes(function: _FunctionInfo | None) -> tuple[object, ...]:
    if function is None or _ast is None:
        return ()
    return tuple(node for node in _ast.walk(function.node) if isinstance(node, (_ast.For, _ast.While)))


def _constant_ints(node: object) -> tuple[int, ...]:
    if _ast is None:
        return ()
    values: list[int] = []
    for child in _ast.walk(node):
        literal = _literal(child)
        if isinstance(literal, int) and not isinstance(literal, bool):
            values.append(literal)
    return tuple(values)


def _hop_witness(section: Mapping[str, object]) -> tuple[FactState, tuple[RedirectHop, ...]]:
    raw_hops = section.get("redirect_hops") or section.get("hops")
    if not isinstance(raw_hops, Sequence) or isinstance(raw_hops, (str, bytes)):
        return FactState.UNKNOWN, ()
    hops: list[RedirectHop] = []
    all_known = True
    valid = True
    for index, raw in enumerate(raw_hops, 1):
        if not isinstance(raw, Mapping):
            all_known = False
            continue
        validated = _state(raw.get("validated"))
        fetched = _state(raw.get("fetched") if "fetched" in raw else raw.get("requested"))
        unsafe = _state(raw.get("unsafe"))
        relative = _state(raw.get("relative"))
        if validated is FactState.UNKNOWN or fetched is FactState.UNKNOWN:
            all_known = False
        if unsafe is FactState.TRUE and fetched is FactState.TRUE:
            valid = False
        hops.append(
            RedirectHop(
                index=_int_or_none(raw.get("index")) or index,
                url=_clip(raw.get("url") or "", 240),
                validated=validated,
                fetched=fetched,
                unsafe=unsafe,
                relative=relative,
            )
        )
    if not all_known:
        return FactState.UNKNOWN, tuple(hops)
    return (FactState.TRUE if valid else FactState.FALSE), tuple(hops)


class RedirectBoundaryProducer(_Producer):
    NAME = _ProducerName.RSS.value
    TARGETS = ("fetch_rss_feed",)
    REQUIRED = RSS_REQUIRED
    SECTION_NAMES = ("redirect", "rss", "redirects")

    def _derive(
        self,
        receipt: _NormalizedReceipt,
        source: SourceInput,
        *,
        task_id: str,
        frontier_hypothesis_id: str | None,
    ) -> SemanticEvidence:
        ctx = _SourceContext(source)
        subject, function = _subject_for(receipt, ctx, self.TARGETS)
        if function is None:
            found = ctx.find(self.TARGETS[0])
            function = found[0] if found else None
        requests = _request_calls(function)
        validations = _validate_calls(function.node if function else None)
        loops = _loop_nodes(function)
        text = ctx.function_text(function) if function else ""
        first_request_line = min((int(getattr(call, "lineno", 0) or 0) for call in requests), default=None)
        initial_validation = FactState.UNKNOWN
        if first_request_line is not None:
            initial_validation = _state(
                any(int(getattr(call, "lineno", 0) or 0) < first_request_line for call in validations)
            )
        elif function is not None:
            initial_validation = FactState.FALSE if not validations else FactState.UNKNOWN

        follow_true = bool(re.search(r"follow_redirects\s*=\s*True", text))
        follow_false = bool(re.search(r"follow_redirects\s*=\s*False", text))
        loop_has_request = any(any(call in tuple(_ast.walk(loop)) for call in requests) for loop in loops) if _ast else False
        loop_texts = [_ast_text(loop, ctx) for loop in loops]
        has_location = "Location" in text or "location" in text
        has_urljoin = bool(re.search(r"\burljoin\s*\(", text))
        validation_in_loop = any(_validate_calls(loop) for loop in loops)
        if follow_true:
            per_hop = FactState.FALSE
            ownership = "client_managed"
        elif follow_false and loop_has_request and validation_in_loop and has_location:
            per_hop = FactState.TRUE
            ownership = "manual"
        elif follow_false and not loop_has_request:
            per_hop = FactState.FALSE
            ownership = "disabled_without_manual_walk"
        elif function is None:
            per_hop = FactState.UNKNOWN
            ownership = "unknown"
        else:
            per_hop = FactState.UNKNOWN
            ownership = "unknown"

        if follow_true:
            relative = FactState.FALSE
        elif has_urljoin and per_hop is FactState.TRUE:
            relative = FactState.TRUE
        elif function is not None and (follow_false or loop_has_request):
            relative = FactState.FALSE
        else:
            relative = FactState.UNKNOWN

        constants = _assignment_literals(ctx, {"_MAX_FEED_REDIRECTS"})
        max_hops_value = constants.get("_MAX_FEED_REDIRECTS")
        max_hops = int(max_hops_value) if isinstance(max_hops_value, int) and not isinstance(max_hops_value, bool) else None
        bounded_loop = False
        unbounded_loop = False
        for loop, loop_text in zip(loops, loop_texts):
            if isinstance(loop, _ast.For) and isinstance(loop.iter, _ast.Call) and isinstance(loop.iter.func, _ast.Name) and loop.iter.func.id == "range":
                ints = _constant_ints(loop.iter)
                if max_hops is not None and (
                    max_hops in ints or "_MAX_FEED_REDIRECTS" in _ast_text(loop, ctx)
                ):
                    bounded_loop = True
                elif any(value in {5, 6} for value in ints):
                    bounded_loop = True
            if isinstance(loop, _ast.While):
                unbounded_loop = isinstance(loop.test, _ast.Constant) and loop.test.value is True
            if "while True" in loop_text or "for _ in itertools" in loop_text:
                unbounded_loop = True
        if bounded_loop and max_hops is not None:
            bound_state = FactState.TRUE if max_hops == 5 else FactState.FALSE
        elif unbounded_loop or (loop_has_request and not bounded_loop):
            bound_state = FactState.FALSE
        elif follow_true or follow_false:
            bound_state = FactState.FALSE
        else:
            bound_state = FactState.UNKNOWN

        sibling = ctx.find("_ssrf_safe_head")
        sibling_text = " ".join(ctx.function_text(item) for item in sibling)
        sibling_manual = _state(
            bool(
                re.search(r"follow_redirects\s*=\s*False", sibling_text)
                and "urljoin" in sibling_text
                and "validate_external_url" in sibling_text
            )
        ) if sibling else FactState.UNKNOWN
        section = _section(receipt, "redirect", "rss", "redirects")
        hop_state, hops = _hop_witness(section)
        payload = RedirectFacts(
            initial_validation=initial_validation,
            per_hop_validation=per_hop,
            relative_resolution=relative,
            redirect_bound=bound_state,
            max_hops=max_hops,
            client_redirect_ownership=ownership,
            hop_witness=hop_state,
            hops=hops,
            sibling_manual_head=sibling_manual,
        )
        states = {
            EvidenceCode.RSS_INITIAL_VALIDATION: initial_validation,
            EvidenceCode.RSS_PER_HOP_VALIDATION: per_hop,
            EvidenceCode.RSS_RELATIVE_REDIRECT_RESOLUTION: relative,
            EvidenceCode.RSS_REDIRECT_BOUND: bound_state,
        }
        function_ref = (function.source_ref("redirect_boundary"),) if function else (receipt.ref,)
        limitations: list[str] = list(ctx.errors)
        if hop_state is FactState.UNKNOWN:
            limitations.append("RSS_REDIRECT_HOP_WITNESS_unknown_without_structured_receipt")
        if sibling_manual is FactState.UNKNOWN:
            limitations.append("manual_head_prior_art_not_in_snapshot")
        predicate = PredicateRef(
            name="ssrf_redirect_boundary",
            expression="validate initial URL -> one hop -> resolve Location -> validate next hop -> bounded repeat",
            subject=f"{subject.path}:{subject.symbol}",
        )
        return _make_evidence(
            producer=self.NAME,
            task_id=task_id,
            frontier_hypothesis_id=frontier_hypothesis_id,
            receipt=receipt,
            ctx=ctx,
            subject=subject,
            predicate=predicate,
            payload=payload,
            required=self.REQUIRED,
            states=states,
            provenance={
                EvidenceCode.RSS_INITIAL_VALIDATION: function_ref,
                EvidenceCode.RSS_PER_HOP_VALIDATION: function_ref,
                EvidenceCode.RSS_RELATIVE_REDIRECT_RESOLUTION: function_ref,
                EvidenceCode.RSS_REDIRECT_BOUND: function_ref,
            },
            limitations=limitations,
        )


Producer: TypeAlias = AckLifecycleProducer | NamespaceMatrixProducer | WatchdogPopulationProducer | RedirectBoundaryProducer


@dataclass(frozen=True)
class _Frontier:
    hypothesis_id: str
    status: str
    path: str
    symbol: str
    producer: str = ""


def _frontier_items(frontier: object) -> tuple[Mapping[str, object], ...]:
    if hasattr(frontier, "frontier_refs"):
        try:
            value = frontier.frontier_refs()
        except Exception as exc:  # noqa: BLE001
            print(f"[kernel-wiring] semantic frontier read failed: {type(exc).__name__}")
            return ()
    elif isinstance(frontier, Mapping):
        value = frontier.get("frontier") or frontier.get("hypotheses") or [frontier]
    else:
        value = frontier
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _normalize_frontier(item: Mapping[str, object]) -> _Frontier | None:
    subject = item.get("subject")
    subject_map = _as_mapping(subject)
    path = _safe_path(item.get("path") or item.get("file") or subject_map.get("path"))
    symbol = str(item.get("symbol") or item.get("fn") or subject_map.get("symbol") or "")
    hid = str(item.get("hypothesis_id") or item.get("id") or "").strip()
    status = str(item.get("status") or item.get("posture") or "").upper()
    producer = str(item.get("producer") or "").lower()
    if not hid or status not in {"LIVE", "UNKNOWN"}:
        return None
    return _Frontier(hid, status, path, symbol, producer)


def _matching_frontier(
    frontier: object,
    evidence: Sequence[SemanticEvidence],
    source: SourceInput | None,
) -> tuple[_Frontier, SemanticEvidence] | None:
    items = tuple(
        sorted(
            (item for raw in _frontier_items(frontier) if (item := _normalize_frontier(raw)) is not None),
            key=lambda item: (item.hypothesis_id, item.path, item.symbol),
        )
    )
    if not items:
        return None
    source_bundle = _SourceContext(source).bundle if source is not None else None
    source_paths = {file.path for file in source_bundle.files} if source_bundle else set()
    for current in items:
        if source_paths and current.path and current.path not in source_paths:
            continue
        candidates = [
            item
            for item in evidence
            if item.subject.path == current.path
            and item.subject.symbol == current.symbol
            and (not current.producer or item.producer.name == current.producer)
        ]
        if not candidates and not current.path and current.symbol:
            candidates = [item for item in evidence if item.subject.symbol == current.symbol]
        if candidates:
            selected = sorted(candidates, key=lambda item: item.evidence_id)[-1]
            return current, selected
    return None


class SemanticEvidenceConsumer:
    """Render at most one missing-evidence advisory per source frontier."""

    NAME = "semantic_evidence"

    def __init__(
        self,
        task_id: str = "",
        *,
        trace_cb: TraceCallback | None = None,
        organ_out: OutputHook | None = None,
    ) -> None:
        self.task_id = str(task_id or "task")
        self.trace: list[dict[str, object]] = []
        self.status = "ARMED"
        self.degraded = False
        self._trace_cb = trace_cb
        self._organ_out = organ_out
        self._challenged: set[str] = set()
        self.last_advisory: EvidenceGapAdvisory | None = None
        self.last_decision: dict[str, object] = {
            "event": None,
            "would_fire": False,
            "emitted": False,
        }
        self._record(
            "SEMANTIC_CONSUMER_ARMED",
            None,
            "consumer=semantic_evidence/1 mode=ADVISORY_ONLY max_missing=3",
        )

    def _record(self, event: str, turn: int | None, detail: object) -> None:
        entry = {
            "cylinder": "semantic",
            "event": event,
            "detail": _clip(detail, 240),
            "turn": turn,
        }
        self.trace.append(entry)
        callback = self._trace_cb
        if callback is None:
            return
        try:
            callback("semantic", event, str(entry["detail"]))
        except Exception as exc:  # noqa: BLE001
            print(f"[kernel-wiring] semantic trace failed: {type(exc).__name__}")
            self._trace_cb = None

    def advisory_for(
        self,
        frontier: object,
        evidence: SemanticEvidence | Sequence[SemanticEvidence],
        *,
        source: SourceInput | None = None,
    ) -> EvidenceGapAdvisory | None:
        records = (evidence,) if isinstance(evidence, SemanticEvidence) else tuple(
            item for item in evidence if isinstance(item, SemanticEvidence)
        )
        matched = _matching_frontier(frontier, records, source)
        if matched is None:
            return None
        current, record = matched
        if record.status is not EvidenceStatus.UNKNOWN or not record.missing:
            self._record(
                "SEMANTIC_SUPPRESSED",
                record.trigger.turn,
                f"hypothesis={current.hypothesis_id} status={record.status.value} missing=0",
            )
            return None
        missing = tuple(record.missing[:MAX_ADVISORY_MISSING])
        source_digest = record.source.tree_digest
        challenge_key = f"{current.hypothesis_id}|{record.producer.name}/{record.producer.version}|{source_digest}"
        if challenge_key in self._challenged:
            self._record(
                "SEMANTIC_DEDUPED",
                record.trigger.turn,
                f"hypothesis={current.hypothesis_id} producer={record.producer.name}",
            )
            self._record(
                "SEMANTIC_CHALLENGE_ONCE",
                record.trigger.turn,
                f"hypothesis={current.hypothesis_id} producer={record.producer.name}",
            )
            return None
        self._challenged.add(challenge_key)
        subject = f"{record.subject.path}:{record.subject.symbol}".strip(":")
        advisory = EvidenceGapAdvisory(
            schema=ADVISORY_SCHEMA,
            frontier_hypothesis_id=current.hypothesis_id,
            producer=record.producer.name,
            source_digest=source_digest,
            missing=missing,
            mode="ADVISORY_ONLY",
            challenge_key=challenge_key,
            subject=subject,
        )
        self.last_advisory = advisory
        self._record(
            "SEMANTIC_EVIDENCE_GAP",
            record.trigger.turn,
            f"SEMANTIC_EVIDENCE_GAP hypothesis={current.hypothesis_id} "
            f"missing={','.join(code.value for code in missing)}",
        )
        return advisory

    def emit(
        self,
        frontier: object,
        evidence: SemanticEvidence | Sequence[SemanticEvidence],
        *,
        source: SourceInput | None = None,
        organ_out: OutputHook | None = None,
    ) -> str | None:
        if self.degraded:
            return None
        try:
            advisory = self.advisory_for(frontier, evidence, source=source)
            if advisory is None:
                return None
            line = advisory.render()
            hook = organ_out or self._organ_out
            if hook is None:
                self.last_decision = {"event": "SEMANTIC_EVIDENCE_GAP", "would_fire": True, "emitted": True}
                return line
            try:
                emitted = hook(self.NAME, line)
            except Exception as exc:  # noqa: BLE001 - safe no-op on wiring failure
                print(f"[kernel-wiring] semantic organ_out failed: {type(exc).__name__}")
                self.last_decision = {"event": "SEMANTIC_EVIDENCE_GAP", "would_fire": True, "emitted": False}
                return None
            self.last_decision = {
                "event": "SEMANTIC_EVIDENCE_GAP",
                "would_fire": True,
                "emitted": emitted is not None,
            }
            return emitted
        except Exception as exc:  # noqa: BLE001
            self.status = "DISARMED"
            self.degraded = True
            print(f"[kernel-wiring] semantic consumer degraded: {type(exc).__name__}: {_clip(exc, 120)}")
            self._record("SEMANTIC_CONSUMER_DISARMED", None, type(exc).__name__)
            return None

    def consume(self, frontier: object, evidence: SemanticEvidence | Sequence[SemanticEvidence], **kwargs: object) -> str | None:
        return self.emit(frontier, evidence, **cast(dict[str, object], kwargs))

    render = emit

    def challenge_once(self, advisory: EvidenceGapAdvisory) -> bool:
        """Record whether a challenge key is new without rendering anything."""

        key = advisory.challenge_key
        if key in self._challenged:
            return False
        self._challenged.add(key)
        return True


_QUALIFIED_TARGETS: dict[tuple[str, str], str] = {
    ("backend/app/routers/helm_human_work.py", "_ack_decision"): _ProducerName.ACK.value,
    ("app/routers/helm_human_work.py", "_ack_decision"): _ProducerName.ACK.value,
    ("backend/app/services/probe_deck/f2/database_guard.py", "require_isolated_f2_database"): _ProducerName.DB.value,
    ("app/services/probe_deck/f2/database_guard.py", "require_isolated_f2_database"): _ProducerName.DB.value,
    ("backend/app/background_workers.py", "_watchdog_auto_pause_trigger"): _ProducerName.WATCHDOG.value,
    ("background_workers.py", "_watchdog_auto_pause_trigger"): _ProducerName.WATCHDOG.value,
    ("backend/app/services/feed_core/fetch.py", "fetch_rss_feed"): _ProducerName.RSS.value,
    ("app/services/feed_core/fetch.py", "fetch_rss_feed"): _ProducerName.RSS.value,
}


class SemanticBridge:
    """Orchestrator-friendly facade over the four independent producers."""

    NAME = "semantic"

    def __init__(
        self,
        task_id: str = "",
        *,
        trace_cb: TraceCallback | None = None,
        organ_out: OutputHook | None = None,
    ) -> None:
        self.task_id = str(task_id or "task")
        self.trace: list[dict[str, object]] = []
        self.status = "ARMED"
        self._trace_cb = trace_cb
        self._organ_out = organ_out
        self._record(
            "SEMANTIC_BRIDGE_ARMED",
            None,
            "schema=poc.semantic_evidence/1 producers=4 mode=receipt+source_ast",
        )
        self.producers: dict[str, Producer] = {
            _ProducerName.ACK.value: AckLifecycleProducer(self.task_id, trace_cb=trace_cb),
            _ProducerName.DB.value: NamespaceMatrixProducer(self.task_id, trace_cb=trace_cb),
            _ProducerName.WATCHDOG.value: WatchdogPopulationProducer(self.task_id, trace_cb=trace_cb),
            _ProducerName.RSS.value: RedirectBoundaryProducer(self.task_id, trace_cb=trace_cb),
        }
        self.consumer = SemanticEvidenceConsumer(
            self.task_id,
            trace_cb=trace_cb,
            organ_out=organ_out,
        )

    def _record(self, event: str, turn: int | None, detail: object) -> None:
        entry = {"cylinder": "semantic", "event": event, "detail": _clip(detail, 240), "turn": turn}
        self.trace.append(entry)
        if self._trace_cb is not None:
            try:
                self._trace_cb("semantic", event, str(entry["detail"]))
            except Exception as exc:  # noqa: BLE001
                print(f"[kernel-wiring] semantic trace failed: {type(exc).__name__}")
                self._trace_cb = None

    def producer(self, name: str) -> Producer | None:
        normalized = str(name or "").lower().strip()
        aliases = {
            "ack": _ProducerName.ACK.value,
            "watchdog": _ProducerName.WATCHDOG.value,
            "rss": _ProducerName.RSS.value,
            "redirect": _ProducerName.RSS.value,
            "db": _ProducerName.DB.value,
            "namespace": _ProducerName.DB.value,
        }
        return self.producers.get(aliases.get(normalized, normalized))

    def derive(
        self,
        receipts: ReceiptInput,
        source: SourceInput,
        *,
        producer: str | Producer | None = None,
        task_id: str | None = None,
        frontier_hypothesis_id: str | None = None,
    ) -> SemanticEvidence | None:
        selected: Producer | None
        if isinstance(producer, str):
            selected = self.producer(producer)
        elif producer is not None:
            selected = producer
        else:
            selected = self._route(receipts)
        if selected is None:
            return None
        try:
            return selected.produce(
                receipts,
                source,
                task_id=task_id,
                frontier_hypothesis_id=frontier_hypothesis_id,
            )
        except Exception as exc:  # noqa: BLE001
            self.status = "DEGRADED"
            print(f"[kernel-wiring] semantic bridge derive failed: {type(exc).__name__}")
            return None

    def derive_all(
        self,
        receipts: ReceiptInput,
        source: SourceInput,
        *,
        task_id: str | None = None,
        frontier_hypothesis_id: str | None = None,
    ) -> tuple[SemanticEvidence, ...]:
        result: list[SemanticEvidence] = []
        for name in (
            _ProducerName.ACK.value,
            _ProducerName.DB.value,
            _ProducerName.WATCHDOG.value,
            _ProducerName.RSS.value,
        ):
            evidence = self.derive(
                receipts,
                source,
                producer=name,
                task_id=task_id,
                frontier_hypothesis_id=frontier_hypothesis_id,
            )
            if evidence is not None:
                result.append(evidence)
        return tuple(result)

    def _route(self, receipts: ReceiptInput) -> Producer | None:
        for receipt in _receipt_list(receipts):
            explicit = _explicit_producer(receipt)
            if explicit:
                selected = self.producer(explicit)
                if selected is not None:
                    return selected
            anchor = receipt.ref.anchor
            if not anchor:
                continue
            key = (_safe_path(anchor.path), anchor.symbol)
            name = _QUALIFIED_TARGETS.get(key)
            if name:
                return self.producer(name)
        return None

    def consume(
        self,
        frontier: object,
        evidence: SemanticEvidence | Sequence[SemanticEvidence],
        *,
        source: SourceInput | None = None,
        organ_out: OutputHook | None = None,
    ) -> str | None:
        return self.consumer.emit(frontier, evidence, source=source, organ_out=organ_out)

    def process(
        self,
        receipts: ReceiptInput,
        source: SourceInput,
        frontier: object,
        *,
        task_id: str | None = None,
        frontier_hypothesis_id: str | None = None,
        organ_out: OutputHook | None = None,
    ) -> tuple[tuple[SemanticEvidence, ...], str | None]:
        evidence = self.derive_all(
            receipts,
            source,
            task_id=task_id,
            frontier_hypothesis_id=frontier_hypothesis_id,
        )
        return evidence, self.consume(frontier, evidence, source=source, organ_out=organ_out)


# Public aliases make the seam easy to discover without creating competing schemas.
SemanticEvidenceEngine = SemanticBridge
AckProducer = AckLifecycleProducer
DatabaseNamespaceProducer = NamespaceMatrixProducer
WatchdogProducer = WatchdogPopulationProducer
RSSProducer = RedirectBoundaryProducer


def _semantic_trigger(evidence: Evidence) -> TriState:
    """Recognize a receipt that the semantic bridge would inspect as a trigger."""

    record = evidence.event
    if evidence.source == "trace":
        event = str(record.get("event") or "").strip().lower()
        if not event:
            return TriState.UNKNOWN
        return TriState.YES if event == "semantic_trigger" else TriState.NO
    if evidence.source != "receipt":
        return TriState.NO

    normalized = _normalize_receipt(record, 1)
    explicit = _explicit_producer(normalized)
    aliases = {
        "ack",
        "ack_lifecycle",
        "namespace",
        "namespace_matrix",
        "db",
        "watchdog",
        "watchdog_population",
        "rss",
        "redirect",
        "redirect_boundary",
    }
    raw_semantic = _nested_mapping(record, "semantic_inputs", "semantic", "inputs")
    section_names = {
        "ack",
        "ack_lifecycle",
        "lifecycle",
        "namespace",
        "database",
        "environment",
        "population",
        "watchdog",
        "tasks",
        "redirect",
        "rss",
        "redirects",
    }
    has_anchor = any(
        _anchor_matches(normalized, producer.TARGETS)
        for producer in (
            AckLifecycleProducer,
            NamespaceMatrixProducer,
            WatchdogPopulationProducer,
            RedirectBoundaryProducer,
        )
    )
    has_section = any(
        section in raw_semantic and raw_semantic.get(section) is not None
        for section in section_names
    )
    if explicit and explicit not in aliases:
        return TriState.NO
    if not (explicit or has_anchor or has_section):
        return TriState.NO

    test = _nested_mapping(raw_semantic, "test")
    result = record.get("result_class") or record.get("result") or test.get("result_class")
    if not str(result or "").strip():
        return TriState.UNKNOWN
    return TriState.YES if _is_admissible_measurement(normalized) else TriState.NO


CENSUS_SPEC = OrganCensusSpec(
    organ_id="semantic",
    trigger=_semantic_trigger,
    fire_events=frozenset({"SEMANTIC_EVIDENCE_DERIVED", "SEMANTIC_EVIDENCE_GAP"}),
    suppression_events=frozenset({"SEMANTIC_SUPPRESSED"}),
    dedup_events=frozenset({"SEMANTIC_DEDUPED"}),
    challenge_once_events=frozenset({"SEMANTIC_CHALLENGE_ONCE"}),
)
CENSUS_SPECS = (CENSUS_SPEC,)


__all__ = [
    "ACK_REQUIRED",
    "ADVISORY_SCHEMA",
    "CENSUS_SPEC",
    "CENSUS_SPECS",
    "AckFacts",
    "AckLifecycleProducer",
    "AckProducer",
    "AnchorRef",
    "DB_REQUIRED",
    "DatabaseNamespaceProducer",
    "EvidenceCode",
    "EvidenceGapAdvisory",
    "EvidenceRequirement",
    "EvidenceStatus",
    "FactState",
    "INPUT_SCHEMA",
    "NamespaceCase",
    "NamespaceFacts",
    "NamespaceMatrixProducer",
    "PopulationControl",
    "PopulationFacts",
    "PRODUCER_VERSION",
    "ProducerRef",
    "PredicateRef",
    "RSS_REQUIRED",
    "RSSProducer",
    "ReceiptRef",
    "RedirectBoundaryProducer",
    "RedirectFacts",
    "RedirectHop",
    "SCHEMA",
    "SemanticBridge",
    "SemanticEvidence",
    "SemanticEvidenceConsumer",
    "SemanticEvidenceEngine",
    "SourceBundle",
    "SourceFileInput",
    "SourceFileRef",
    "SourceRef",
    "SourceSnapshot",
    "SubjectRef",
    "WATCHDOG_REQUIRED",
    "WatchdogPopulationProducer",
    "WatchdogProducer",
]
