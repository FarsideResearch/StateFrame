"""Typed contracts for the host-owned K1-K3 knowledge substrate.

This module owns packet shapes and canonical identities only.  Repository reads,
documentation admission, idiom selection, and FRAMEBUILD integration belong to
``kernel_knowledge_substrate``.  A packet is an evidence projection, not a
second source-of-truth store: every record carries an exact source handle or a
typed unknown/degraded coverage state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


SCHEMA = "knowledge-substrate/1"
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:/-]{0,255}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class KnowledgeContractError(ValueError):
    """Raised when a value cannot cross the packet contract boundary."""


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class RecordKind(_ValueEnum):
    FACT = "FACT"
    API_DOC = "API_DOC"
    IDIOM = "IDIOM"


class DependencyKind(_ValueEnum):
    SOURCE = "SOURCE"
    STRUCTURAL = "STRUCTURAL"
    VERSION = "VERSION"
    FEATURE = "FEATURE"
    RUNTIME = "RUNTIME"
    DOCUMENTATION = "DOCUMENTATION"
    IDIOM = "IDIOM"


class Applicability(_ValueEnum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Validity(_ValueEnum):
    VALID = "VALID"
    STALE = "STALE"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class CoverageState(_ValueEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class EvidenceLevel(_ValueEnum):
    CURRENT_USAGE = "CURRENT_USAGE"
    VERIFIED_PATTERN = "VERIFIED_PATTERN"
    UNVERIFIED = "UNVERIFIED"


class DocumentationStatus(_ValueEnum):
    FOUND = "FOUND"
    CONFLICT = "CONFLICT"
    NO_ELIGIBLE_SOURCE = "NO_ELIGIBLE_SOURCE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"


class IdiomStatus(_ValueEnum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise KnowledgeContractError("value must be JSON serializable") from exc


def canonical_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise KnowledgeContractError(f"{label} must be a bounded identifier")
    return value


def _text(value: object, label: str, maximum: int = 16_384) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeContractError(f"{label} must be non-empty text")
    if len(value) > maximum or "\x00" in value:
        raise KnowledgeContractError(f"{label} is invalid or too large")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise KnowledgeContractError(f"{label} must be a sha256 digest")
    return value


def _enum(value: object, enum_type: type[_ValueEnum], label: str) -> _ValueEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise KnowledgeContractError(f"{label} is not a supported value") from exc


def _tuple_text(values: Sequence[object], label: str, *, identifiers: bool = False) -> tuple[str, ...]:
    result = tuple(_text(value, label, 256) for value in values)
    if identifiers and any(not _ID_RE.fullmatch(value) for value in result):
        raise KnowledgeContractError(f"{label} contains an invalid identifier")
    if len(set(result)) != len(result):
        raise KnowledgeContractError(f"{label} contains duplicates")
    return result


@dataclass(frozen=True, slots=True)
class CoverageSnapshot:
    """Coverage copied from an existing owner or explicitly marked unknown."""

    coverage_id: str
    kind: str
    status: CoverageState
    inspected: int | None = None
    expected: int | None = None
    reason: str | None = None

    SCHEMA: ClassVar[str] = "knowledge-coverage/1"

    def __post_init__(self) -> None:
        _id(self.coverage_id, "coverage_id")
        _id(self.kind, "coverage.kind")
        if not isinstance(self.status, CoverageState):
            object.__setattr__(self, "status", _enum(self.status, CoverageState, "coverage.status"))
        for value, label in ((self.inspected, "coverage.inspected"), (self.expected, "coverage.expected")):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise KnowledgeContractError(f"{label} must be a non-negative integer")
        if self.inspected is not None and self.expected is not None and self.inspected > self.expected:
            raise KnowledgeContractError("coverage.inspected cannot exceed coverage.expected")
        if self.status is CoverageState.UNKNOWN and not self.reason:
            raise KnowledgeContractError("unknown coverage requires a reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "coverage_id": self.coverage_id,
            "kind": self.kind,
            "status": self.status.value,
            "inspected": self.inspected,
            "expected": self.expected,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SourceSpanHandle:
    """Exact locator binding a record to current source bytes."""

    source_ref: str
    source_digest: str
    content_id: str
    span_id: str
    start_line: int
    end_line: int
    byte_start: int
    byte_end: int
    span_digest: str
    coverage_ref: str

    SCHEMA: ClassVar[str] = "knowledge-source-span/1"

    def __post_init__(self) -> None:
        _text(self.source_ref, "source_ref", 2048)
        if ".." in self.source_ref.replace("\\", "/").split("/"):
            raise KnowledgeContractError("source_ref escapes its root")
        _digest(self.source_digest, "source_digest")
        _id(self.content_id, "content_id")
        _id(self.span_id, "span_id")
        _id(self.coverage_ref, "coverage_ref")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise KnowledgeContractError("source line range is invalid")
        if self.byte_start < 0 or self.byte_end < self.byte_start:
            raise KnowledgeContractError("source byte range is invalid")
        _digest(self.span_digest, "span_digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "source_ref": self.source_ref,
            "source_digest": self.source_digest,
            "content_id": self.content_id,
            "span_id": self.span_id,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "span_digest": self.span_digest,
            "coverage_ref": self.coverage_ref,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeRecord:
    record_id: str
    kind: RecordKind
    subject: str
    value: object
    source_refs: tuple[SourceSpanHandle, ...]
    dependency_kinds: tuple[DependencyKind, ...]
    applicability: Applicability
    validity: Validity
    coverage: CoverageSnapshot
    contradiction_refs: tuple[str, ...] = ()
    counterevidence_refs: tuple[str, ...] = ()
    structural_relation: str | None = None
    preconditions: Mapping[str, object] = field(default_factory=dict)
    evidence_level: EvidenceLevel | None = None
    language: str = "unknown"

    SCHEMA: ClassVar[str] = "knowledge-record/1"

    def __post_init__(self) -> None:
        _id(self.record_id, "record_id")
        if not isinstance(self.kind, RecordKind):
            object.__setattr__(self, "kind", _enum(self.kind, RecordKind, "record.kind"))
        _text(self.subject, "record.subject", 1024)
        _canonical(self.value)
        refs = tuple(self.source_refs)
        if not refs or any(not isinstance(ref, SourceSpanHandle) for ref in refs):
            raise KnowledgeContractError("record.source_refs must contain exact source handles")
        object.__setattr__(self, "source_refs", refs)
        deps = tuple(self.dependency_kinds)
        if any(not isinstance(dep, DependencyKind) for dep in deps):
            object.__setattr__(self, "dependency_kinds", tuple(_enum(dep, DependencyKind, "dependency") for dep in deps))
        if not isinstance(self.applicability, Applicability):
            object.__setattr__(self, "applicability", _enum(self.applicability, Applicability, "record.applicability"))
        if not isinstance(self.validity, Validity):
            object.__setattr__(self, "validity", _enum(self.validity, Validity, "record.validity"))
        if not isinstance(self.coverage, CoverageSnapshot):
            raise KnowledgeContractError("record.coverage must be CoverageSnapshot")
        object.__setattr__(self, "contradiction_refs", _tuple_text(self.contradiction_refs, "contradiction_refs", identifiers=True))
        object.__setattr__(self, "counterevidence_refs", _tuple_text(self.counterevidence_refs, "counterevidence_refs", identifiers=True))
        if self.structural_relation is not None:
            _text(self.structural_relation, "structural_relation", 512)
        _canonical(self.preconditions)
        if not isinstance(self.preconditions, Mapping):
            raise KnowledgeContractError("record.preconditions must be an object")
        object.__setattr__(self, "preconditions", copy.deepcopy(dict(self.preconditions)))
        if self.evidence_level is not None and not isinstance(self.evidence_level, EvidenceLevel):
            object.__setattr__(self, "evidence_level", _enum(self.evidence_level, EvidenceLevel, "evidence_level"))
        _text(self.language, "record.language", 128)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "record_id": self.record_id,
            "kind": self.kind.value,
            "subject": self.subject,
            "value": copy.deepcopy(self.value),
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "dependency_kinds": [dep.value for dep in self.dependency_kinds],
            "applicability": self.applicability.value,
            "validity": self.validity.value,
            "coverage": self.coverage.to_dict(),
            "contradiction_refs": list(self.contradiction_refs),
            "counterevidence_refs": list(self.counterevidence_refs),
            "structural_relation": self.structural_relation,
            "preconditions": copy.deepcopy(dict(self.preconditions)),
            "evidence_level": self.evidence_level.value if self.evidence_level else None,
            "language": self.language,
        }


@dataclass(frozen=True, slots=True)
class Contradiction:
    contradiction_id: str
    subject: str
    record_ids: tuple[str, ...]
    source_refs: tuple[SourceSpanHandle, ...]
    reason: str

    SCHEMA: ClassVar[str] = "knowledge-contradiction/1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "contradiction_id": self.contradiction_id,
            "subject": self.subject,
            "record_ids": list(self.record_ids),
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class KnowledgePacket:
    packet_id: str
    packet_sha256: str
    generation: str
    records: tuple[KnowledgeRecord, ...]
    contradictions: tuple[Contradiction, ...]
    coverage: CoverageSnapshot
    fallback_required: bool = False
    fallback_reason: str | None = None

    SCHEMA: ClassVar[str] = "knowledge-packet/1"

    @classmethod
    def create(
        cls,
        generation: str,
        records: Sequence[KnowledgeRecord],
        coverage: CoverageSnapshot,
        *,
        fallback_required: bool = False,
        fallback_reason: str | None = None,
    ) -> "KnowledgePacket":
        ordered = tuple(sorted(records, key=lambda item: item.record_id))
        if len({record.record_id for record in ordered}) != len(ordered):
            raise KnowledgeContractError("packet records must have unique record IDs")
        by_subject: dict[str, list[KnowledgeRecord]] = {}
        for record in ordered:
            by_subject.setdefault(record.subject, []).append(record)
        contradictions: list[Contradiction] = []
        for subject, rows in sorted(by_subject.items()):
            distinct = {canonical_digest(row.value) for row in rows}
            if len(distinct) < 2:
                continue
            refs = tuple(ref for row in rows for ref in row.source_refs)
            identity = {"subject": subject, "record_ids": [row.record_id for row in rows]}
            contradictions.append(Contradiction(
                "contradiction:" + hashlib.sha256(_canonical(identity)).hexdigest()[:32],
                subject,
                tuple(row.record_id for row in rows),
                refs,
                "records for one subject carry different values",
            ))
        body = {
            "schema": cls.SCHEMA,
            "generation": generation,
            "records": [row.to_dict() for row in ordered],
            "contradictions": [item.to_dict() for item in contradictions],
            "coverage": coverage.to_dict(),
            "fallback_required": fallback_required,
            "fallback_reason": fallback_reason,
        }
        packet_sha = canonical_digest(body)
        packet_id = "packet:" + packet_sha.split(":", 1)[1][:32]
        return cls(packet_id, packet_sha, generation, ordered, tuple(contradictions), coverage,
                   fallback_required, fallback_reason)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "packet_id": self.packet_id,
            "packet_sha256": self.packet_sha256,
            "generation": self.generation,
            "records": [row.to_dict() for row in self.records],
            "contradictions": [item.to_dict() for item in self.contradictions],
            "coverage": self.coverage.to_dict(),
            "fallback_required": self.fallback_required,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True, slots=True)
class DocumentationCandidate:
    source_id: str
    source_ref: str
    content: str
    version: str
    feature: str
    runtime: str
    format: str
    origin: str
    authorized: bool = False
    local_path: str | None = None

    def __post_init__(self) -> None:
        _id(self.source_id, "documentation.source_id")
        _text(self.source_ref, "documentation.source_ref", 2048)
        _text(self.content, "documentation.content", 64 * 1024)
        _text(self.version, "documentation.version", 256)
        _text(self.feature, "documentation.feature", 512)
        _text(self.runtime, "documentation.runtime", 256)
        _text(self.format, "documentation.format", 64)
        _text(self.origin, "documentation.origin", 64)

    @property
    def content_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DocumentationQuery:
    feature: str
    version: str
    runtime: str


@dataclass(frozen=True, slots=True)
class DocumentationResolution:
    status: DocumentationStatus
    query: DocumentationQuery
    candidates: tuple[DocumentationCandidate, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class IdiomSpec:
    source_ref: str
    start_line: int
    end_line: int
    structural_relation: str
    preconditions: Mapping[str, object]
    transformation: str
    evidence_level: EvidenceLevel = EvidenceLevel.CURRENT_USAGE
    identifiers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IdiomQuery:
    structural_relation: str
    preconditions: Mapping[str, object]
    identifiers: tuple[str, ...] = ()
    language: str | None = None


@dataclass(frozen=True, slots=True)
class IdiomSelection:
    status: IdiomStatus
    candidate: KnowledgeRecord | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ValidationCost:
    files_scanned: int
    source_bytes: int
    records_checked: int
    documentation_candidates: int
    knowledge_bytes: int
    estimated_input_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "files_scanned": self.files_scanned,
            "source_bytes": self.source_bytes,
            "records_checked": self.records_checked,
            "documentation_candidates": self.documentation_candidates,
            "knowledge_bytes": self.knowledge_bytes,
            "estimated_input_tokens": self.estimated_input_tokens,
        }


@dataclass(frozen=True, slots=True)
class TaskKnowledgeView:
    packet_id: str | None
    packet_sha256: str | None
    status: str
    records: tuple[Mapping[str, object], ...]
    contradictions: tuple[Mapping[str, object], ...]
    coverage: Mapping[str, object]
    included_record_ids: tuple[str, ...]
    omitted_record_ids: tuple[str, ...]
    clipped: bool
    fallback_required: bool
    fallback_reason: str | None
    cost: ValidationCost
    source_delivery: Mapping[str, object] = field(default_factory=dict)

    SCHEMA: ClassVar[str] = "knowledge-task-view/1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "packet_id": self.packet_id,
            "packet_sha256": self.packet_sha256,
            "status": self.status,
            "records": [copy.deepcopy(dict(item)) for item in self.records],
            "contradictions": [copy.deepcopy(dict(item)) for item in self.contradictions],
            "coverage": copy.deepcopy(dict(self.coverage)),
            "included_record_ids": list(self.included_record_ids),
            "omitted_record_ids": list(self.omitted_record_ids),
            "clipped": self.clipped,
            "fallback_required": self.fallback_required,
            "fallback_reason": self.fallback_reason,
            "cost": self.cost.to_dict(),
            "source_delivery": copy.deepcopy(dict(self.source_delivery)),
        }


__all__ = [
    "Applicability", "canonical_digest", "Contradiction", "CoverageSnapshot", "CoverageState",
    "DependencyKind", "DocumentationCandidate", "DocumentationQuery", "DocumentationResolution",
    "DocumentationStatus", "EvidenceLevel", "IdiomQuery", "IdiomSelection", "IdiomSpec",
    "IdiomStatus", "KnowledgeContractError", "KnowledgePacket", "KnowledgeRecord", "RecordKind",
    "SCHEMA", "SourceSpanHandle", "TaskKnowledgeView", "Validity", "ValidationCost",
]
