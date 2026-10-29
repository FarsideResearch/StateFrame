"""EC01 evidence identities, views, omissions, and delivery manifests.

This is a pure contract boundary.  It owns no files, databases, catalog
projection, source parser, provider call, or prompt renderer.  A caller gives
it one immutable byte snapshot; the module validates deterministic views over
that snapshot and the exact bytes of a certified delivery.

The catalog axes are imported from CATC0.  The StateStore schema identifiers
are referenced for cross-owner identity but this module never opens a store or
creates a competing event/blob authority.
"""

from __future__ import annotations

import base64
import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import ClassVar

try:  # Supports the flat v46 loader and package imports.
    from kernel_catalog_contracts import (
        ApplicabilityState,
        Availability,
        Confidence,
        ContractError as CatalogContractError,
        CoverageRef,
        CoverageStatus,
    )
except (ImportError, TypeError):  # pragma: no cover - isolated/older interpreter path.
    from dataclasses import dataclass as _compat_dataclass

    class _CompatEnum(str, Enum):
        def __str__(self) -> str:
            return self.value

    class ApplicabilityState(_CompatEnum):
        CURRENT = "current"
        STALE = "stale"
        UNKNOWN = "unknown"
        CONFLICT = "conflict"

    class Availability(_CompatEnum):
        PRESENT = "present"
        ABSENT = "absent"
        UNAVAILABLE = "unavailable"
        DENIED = "denied"
        UNSUPPORTED = "unsupported"

    class Confidence(_CompatEnum):
        OBSERVED = "observed"
        INFERRED = "inferred"
        ASSUMED = "assumed"

    class CoverageStatus(_CompatEnum):
        COMPLETE = "complete"
        PARTIAL = "partial"
        UNKNOWN = "unknown"

    class CatalogContractError(ValueError):
        pass

    @_compat_dataclass(frozen=True)
    class CoverageRef:
        coverage_id: str
        kind: str
        status: CoverageStatus
        inspected: int | None = None
        expected: int | None = None

try:  # Keep this module usable when only src_v46 is on sys.path.
    from tools.sciledger.state_store import (
        SNAPSHOT_SCHEMA as STATE_STORE_SNAPSHOT_SCHEMA,
        STATE_SCHEMA as STATE_STORE_SCHEMA,
    )
except (ImportError, TypeError):  # pragma: no cover - isolated/older interpreter path.
    STATE_STORE_SCHEMA = "sciledger/state-store/1"
    STATE_STORE_SNAPSHOT_SCHEMA = "sciledger/state-store-snapshot/1"


EVIDENCE_SNAPSHOT_SCHEMA = "stateframe/evidence-snapshot/1"
EVIDENCE_SPAN_SCHEMA = "stateframe/evidence-span/1"
EVIDENCE_COVERAGE_SCHEMA = "stateframe/evidence-coverage/1"
EVIDENCE_SELECTOR_SCHEMA = "stateframe/evidence-selector/1"
EVIDENCE_FACT_SCHEMA = "stateframe/evidence-fact/1"
EVIDENCE_OMISSION_SCHEMA = "stateframe/evidence-omission/1"
EVIDENCE_VIEW_SCHEMA = "stateframe/evidence-view/1"
EVIDENCE_DELIVERY_SCHEMA = "stateframe/evidence-delivery/1"
EVIDENCE_FIXTURE_SCHEMA = "stateframe/evidence-fixture/1"
OUTLINE_SCHEMA = "mc-outline/1"

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:/-]{0,255}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_CANONICAL_BYTES = 256 * 1024


class EvidenceContractError(CatalogContractError):
    """Raised when an evidence record cannot be admitted safely."""


class UnknownFieldError(EvidenceContractError):
    """Raised when a closed evidence record has an undeclared field."""


class UnknownVersionError(EvidenceContractError):
    """Raised when a record belongs to an unsupported evidence schema."""


class EvidenceVisibility(str, Enum):
    """Visibility labels ordered from least to most restrictive."""

    PUBLIC = "public"
    WORKSPACE = "workspace"
    OPERATOR_ONLY = "operator_only"
    EVALUATOR_RESTRICTED = "evaluator_restricted"

    @property
    def level(self) -> int:
        return {
            EvidenceVisibility.PUBLIC: 0,
            EvidenceVisibility.WORKSPACE: 1,
            EvidenceVisibility.OPERATOR_ONLY: 2,
            EvidenceVisibility.EVALUATOR_RESTRICTED: 3,
        }[self]


class SelectorMode(str, Enum):
    SIGNATURES = "signatures"
    MAP = "map"
    SPAN = "span"
    DIFF_SINCE_LAST_SEEN = "diff-since-last-seen"
    FULL = "full"


class OmissionReason(str, Enum):
    BUDGET = "budget"
    NOT_SELECTED = "not_selected"
    NOT_AUTHORIZED = "not_authorized"
    PARTIAL_CAPTURE = "partial_capture"
    UNKNOWN = "unknown"
    STALE = "stale"
    BASELINE_UNAVAILABLE = "baseline_unavailable"
    PARSER_FAILURE = "parser_failure"


def _text(value: object, label: str, *, max_length: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceContractError(f"{label} must be a non-empty string")
    if len(value) > max_length:
        raise EvidenceContractError(f"{label} exceeds {max_length} characters")
    if "\x00" in value:
        raise EvidenceContractError(f"{label} contains a NUL character")
    return value


def _id(value: object, label: str) -> str:
    result = _text(value, label, max_length=256)
    if not _ID_RE.fullmatch(result):
        raise EvidenceContractError(f"{label} must be a valid identifier")
    return result


def _optional_id(value: object, label: str) -> str | None:
    return None if value is None else _id(value, label)


def _nonnegative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceContractError(f"{label} must be an integer >= 0")
    return value


def _positive(value: object, label: str) -> int:
    result = _nonnegative(value, label)
    if result < 1:
        raise EvidenceContractError(f"{label} must be an integer >= 1")
    return result


def _digest(value: object, label: str) -> str:
    result = _text(value, label, max_length=71)
    if not _DIGEST_RE.fullmatch(result):
        raise EvidenceContractError(f"{label} must be a lowercase SHA-256 digest")
    return result


def digest_bytes(value: bytes) -> str:
    """Hash exact bytes without text-mode newline or encoding conversion."""

    if not isinstance(value, bytes):
        raise TypeError("digest_bytes requires bytes")
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _json_value(value: object, label: str = "value", *, depth: int = 0) -> object:
    if depth > 12:
        raise EvidenceContractError(f"{label} exceeds the JSON depth limit")
    if value is None or isinstance(value, (str, int, bool)):
        if isinstance(value, str) and len(value) > 4096:
            raise EvidenceContractError(f"{label} contains oversized text")
        return copy.deepcopy(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise EvidenceContractError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise EvidenceContractError(f"{label} object keys must be strings")
            result[key] = _json_value(nested, f"{label}.{key}", depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, f"{label}[]", depth=depth + 1) for item in value]
    raise EvidenceContractError(f"{label} contains unsupported type {type(value).__name__}")


def _canonical(value: object) -> bytes:
    encoded = json.dumps(
        _json_value(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_CANONICAL_BYTES:
        raise EvidenceContractError("canonical evidence record exceeds size limit")
    return encoded


def digest_json(value: object) -> str:
    return digest_bytes(_canonical(value))


def _closed(value: object, required: set[str], optional: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise EvidenceContractError(f"{label} must be an object")
    result = dict(value)
    if any(not isinstance(key, str) for key in result):
        raise EvidenceContractError(f"{label} keys must be strings")
    unknown = sorted(set(result) - required - optional)
    if unknown:
        raise UnknownFieldError(f"{label} has unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(result))
    if missing:
        raise EvidenceContractError(f"{label} is missing field(s): {', '.join(missing)}")
    return result


def _schema(data: Mapping[str, object], expected: str, label: str) -> None:
    if data.get("schema") != expected:
        raise UnknownVersionError(f"{label}.schema must be {expected}")


def _enum(value: object, enum_type: type[Enum], label: str) -> Enum:
    try:
        return enum_type(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(str(item.value) for item in enum_type)
        raise EvidenceContractError(f"{label} must be one of: {allowed}") from exc


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EvidenceContractError(f"{label} must be an array")
    return tuple(value)


def _unique_ids(values: object, label: str) -> tuple[str, ...]:
    result = tuple(_id(value, f"{label}[]") for value in _sequence(values, label))
    if len(set(result)) != len(result):
        raise EvidenceContractError(f"{label} must not contain duplicates")
    return result


def _opaque_handle(value: object, label: str) -> str:
    result = _id(value, label)
    if not result.startswith("expand:"):
        raise EvidenceContractError(f"{label} must be a host-issued expansion handle")
    return result


@dataclass(frozen=True)
class SourceSnapshot:
    """One immutable source observation whose digest covers exact bytes."""

    snapshot_id: str
    store_id: str
    namespace_id: str
    repo_id: str
    worktree_id: str
    source_ref: str
    source_revision: str
    origin_receipt_id: str
    body: bytes
    encoding: str = "utf-8"
    visibility: EvidenceVisibility = EvidenceVisibility.PUBLIC

    SCHEMA: ClassVar[str] = EVIDENCE_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        _id(self.snapshot_id, "snapshot_id")
        _id(self.store_id, "store_id")
        _id(self.namespace_id, "namespace_id")
        _id(self.repo_id, "repo_id")
        _id(self.worktree_id, "worktree_id")
        _text(self.source_ref, "source_ref", max_length=2048)
        _id(self.source_revision, "source_revision")
        _id(self.origin_receipt_id, "origin_receipt_id")
        if not isinstance(self.body, bytes):
            raise EvidenceContractError("snapshot.body must be immutable bytes")
        _text(self.encoding, "encoding", max_length=128)
        if not isinstance(self.visibility, EvidenceVisibility):
            object.__setattr__(self, "visibility", _enum(self.visibility, EvidenceVisibility, "visibility"))

    @property
    def source_sha256(self) -> str:
        return digest_bytes(self.body)

    def computed_id(self) -> str:
        identity = {
            "schema": self.SCHEMA,
            "store_id": self.store_id,
            "namespace_id": self.namespace_id,
            "repo_id": self.repo_id,
            "worktree_id": self.worktree_id,
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
            "origin_receipt_id": self.origin_receipt_id,
            "source_sha256": self.source_sha256,
            "encoding": self.encoding,
            "visibility": self.visibility.value,
        }
        return "snapshot:" + hashlib.sha256(_canonical(identity)).hexdigest()

    def validate_integrity(self) -> None:
        if self.snapshot_id != self.computed_id():
            raise EvidenceContractError("snapshot_id does not match immutable source identity")

    @classmethod
    def capture(
        cls,
        body: bytes | bytearray,
        *,
        store_id: str,
        namespace_id: str,
        repo_id: str,
        worktree_id: str,
        source_ref: str,
        source_revision: str,
        origin_receipt_id: str,
        encoding: str = "utf-8",
        visibility: EvidenceVisibility = EvidenceVisibility.PUBLIC,
    ) -> "SourceSnapshot":
        raw = bytes(body)
        normalized_visibility = visibility if isinstance(visibility, EvidenceVisibility) else _enum(visibility, EvidenceVisibility, "visibility")
        identity = {
            "schema": cls.SCHEMA,
            "store_id": store_id,
            "namespace_id": namespace_id,
            "repo_id": repo_id,
            "worktree_id": worktree_id,
            "source_ref": source_ref,
            "source_revision": source_revision,
            "origin_receipt_id": origin_receipt_id,
            "source_sha256": digest_bytes(raw),
            "encoding": encoding,
            "visibility": normalized_visibility.value,
        }
        snapshot_id = "snapshot:" + hashlib.sha256(_canonical(identity)).hexdigest()
        return cls(snapshot_id, store_id, namespace_id, repo_id, worktree_id, source_ref,
                   source_revision, origin_receipt_id, raw, encoding, normalized_visibility)

    @classmethod
    def from_dict(cls, value: object) -> "SourceSnapshot":
        data = _closed(value, {
            "schema", "snapshot_id", "store_id", "namespace_id", "repo_id", "worktree_id",
            "source_ref", "source_revision", "origin_receipt_id", "body_base64", "source_sha256",
        }, {"encoding", "visibility"}, "source_snapshot")
        _schema(data, cls.SCHEMA, "source_snapshot")
        try:
            body = base64.b64decode(data["body_base64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise EvidenceContractError("source_snapshot.body_base64 is invalid") from exc
        snapshot = cls(data["snapshot_id"], data["store_id"], data["namespace_id"], data["repo_id"],
                       data["worktree_id"], data["source_ref"], data["source_revision"],
                       data["origin_receipt_id"], body, data.get("encoding", "utf-8"),
                       _enum(data.get("visibility", EvidenceVisibility.PUBLIC.value), EvidenceVisibility, "visibility"))
        if snapshot.source_sha256 != data["source_sha256"]:
            raise EvidenceContractError("source_snapshot.source_sha256 does not match body")
        snapshot.validate_integrity()
        return snapshot

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "snapshot_id": self.snapshot_id,
            "store_id": self.store_id,
            "namespace_id": self.namespace_id,
            "repo_id": self.repo_id,
            "worktree_id": self.worktree_id,
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
            "origin_receipt_id": self.origin_receipt_id,
            "body_base64": base64.b64encode(self.body).decode("ascii"),
            "source_sha256": self.source_sha256,
            "encoding": self.encoding,
            "visibility": self.visibility.value,
        }

    def _line_starts(self) -> tuple[int, ...]:
        starts = [0]
        for position, byte in enumerate(self.body):
            if byte == 0x0A and position + 1 < len(self.body):
                starts.append(position + 1)
        return tuple(starts)

    @property
    def line_count(self) -> int:
        return len(self._line_starts())

    def byte_range_for_lines(self, start_line: int, end_line: int) -> tuple[int, int]:
        start = _positive(start_line, "start_line")
        end = _positive(end_line, "end_line")
        if end < start or end > self.line_count:
            raise EvidenceContractError("line range is outside the immutable snapshot")
        starts = self._line_starts()
        byte_start = starts[start - 1]
        byte_end = len(self.body) if end == self.line_count else starts[end]
        return byte_start, byte_end

    def span(self, start_line: int, end_line: int, *, anchor_id: str | None = None) -> "EvidenceSpan":
        byte_start, byte_end = self.byte_range_for_lines(start_line, end_line)
        span_id = anchor_id or EvidenceSpan.make_id(
            self.source_sha256, self.source_ref, byte_start, byte_end,
        )
        return EvidenceSpan(
            span_id, byte_start, byte_end, start_line, end_line,
            digest_bytes(self.body[byte_start:byte_end]),
        )


@dataclass(frozen=True)
class EvidenceSpan:
    """An exact half-open byte interval plus one-based inclusive line range."""

    span_id: str
    byte_start: int
    byte_end: int
    start_line: int
    end_line: int
    body_sha256: str

    SCHEMA: ClassVar[str] = EVIDENCE_SPAN_SCHEMA

    def __post_init__(self) -> None:
        _id(self.span_id, "span_id")
        start = _nonnegative(self.byte_start, "byte_start")
        end = _nonnegative(self.byte_end, "byte_end")
        if end <= start:
            raise EvidenceContractError("span byte interval must be non-empty and half-open")
        line_start = _positive(self.start_line, "start_line")
        line_end = _positive(self.end_line, "end_line")
        if line_end < line_start:
            raise EvidenceContractError("span line interval must be ordered")
        _digest(self.body_sha256, "body_sha256")

    @staticmethod
    def make_id(source_sha256: str, source_ref: str, byte_start: int, byte_end: int) -> str:
        return "span:" + hashlib.sha256(_canonical({
            "source_sha256": source_sha256, "source_ref": source_ref,
            "byte_start": byte_start, "byte_end": byte_end,
        })).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceSpan":
        data = _closed(value, {"schema", "span_id", "byte_start", "byte_end", "start_line", "end_line", "body_sha256"}, set(), "evidence_span")
        _schema(data, cls.SCHEMA, "evidence_span")
        return cls(data["span_id"], data["byte_start"], data["byte_end"], data["start_line"], data["end_line"], data["body_sha256"])  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "span_id": self.span_id, "byte_start": self.byte_start,
                "byte_end": self.byte_end, "start_line": self.start_line, "end_line": self.end_line,
                "body_sha256": self.body_sha256}


@dataclass(frozen=True)
class EvidenceCoverage:
    """Catalog-aligned coverage with an explicitly named completeness universe."""

    coverage_id: str
    kind: str
    status: CoverageStatus
    universe: str
    inspected: int | None = None
    expected: int | None = None
    unknown_reason: str | None = None

    SCHEMA: ClassVar[str] = EVIDENCE_COVERAGE_SCHEMA

    def __post_init__(self) -> None:
        _id(self.coverage_id, "coverage_id")
        _id(self.kind, "coverage.kind")
        _text(self.universe, "coverage.universe", max_length=2048)
        if not isinstance(self.status, CoverageStatus):
            object.__setattr__(self, "status", _enum(self.status, CoverageStatus, "coverage.status"))
        if self.inspected is not None:
            _nonnegative(self.inspected, "coverage.inspected")
        if self.expected is not None:
            _nonnegative(self.expected, "coverage.expected")
        if self.inspected is not None and self.expected is not None and self.inspected > self.expected:
            raise EvidenceContractError("coverage.inspected cannot exceed coverage.expected")
        if self.status is CoverageStatus.UNKNOWN and self.unknown_reason is None:
            raise EvidenceContractError("unknown coverage requires unknown_reason")
        if self.unknown_reason is not None:
            _text(self.unknown_reason, "coverage.unknown_reason", max_length=512)

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceCoverage":
        data = _closed(value, {"schema", "coverage_id", "kind", "status", "universe"},
                       {"inspected", "expected", "unknown_reason"}, "evidence_coverage")
        _schema(data, cls.SCHEMA, "evidence_coverage")
        return cls(data["coverage_id"], data["kind"], _enum(data["status"], CoverageStatus, "coverage.status"),
                   data["universe"], data.get("inspected"), data.get("expected"), data.get("unknown_reason"))  # type: ignore[arg-type]

    def catalog_ref(self) -> CoverageRef:
        return CoverageRef(self.coverage_id, self.kind, self.status, self.inspected, self.expected)

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "coverage_id": self.coverage_id, "kind": self.kind,
                "status": self.status.value, "universe": self.universe, "inspected": self.inspected,
                "expected": self.expected, "unknown_reason": self.unknown_reason}


@dataclass(frozen=True)
class EvidenceSelector:
    mode: SelectorMode
    symbol_id: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    baseline_view_id: str | None = None
    cursor: str | None = None

    SCHEMA: ClassVar[str] = EVIDENCE_SELECTOR_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.mode, SelectorMode):
            object.__setattr__(self, "mode", _enum(self.mode, SelectorMode, "selector.mode"))
        object.__setattr__(self, "symbol_id", _optional_id(self.symbol_id, "selector.symbol_id"))
        if self.start_line is not None:
            _positive(self.start_line, "selector.start_line")
        if self.end_line is not None:
            _positive(self.end_line, "selector.end_line")
        if self.start_line is not None and self.end_line is not None and self.end_line < self.start_line:
            raise EvidenceContractError("selector line range is reversed")
        object.__setattr__(self, "baseline_view_id", _optional_id(self.baseline_view_id, "selector.baseline_view_id"))
        if self.cursor is not None:
            _id(self.cursor, "selector.cursor")
        if self.mode is SelectorMode.DIFF_SINCE_LAST_SEEN and not self.baseline_view_id:
            raise EvidenceContractError("diff-since-last-seen requires baseline_view_id")

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceSelector":
        data = _closed(value, {"schema", "mode"}, {"symbol_id", "start_line", "end_line", "baseline_view_id", "cursor"}, "evidence_selector")
        _schema(data, cls.SCHEMA, "evidence_selector")
        return cls(_enum(data["mode"], SelectorMode, "selector.mode"), data.get("symbol_id"), data.get("start_line"),
                   data.get("end_line"), data.get("baseline_view_id"), data.get("cursor"))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "mode": self.mode.value, "symbol_id": self.symbol_id,
                "start_line": self.start_line, "end_line": self.end_line,
                "baseline_view_id": self.baseline_view_id, "cursor": self.cursor}


@dataclass(frozen=True)
class EvidenceDependency:
    dependency_id: str
    revision_or_digest: str
    coverage_ref: str | None = None

    def __post_init__(self) -> None:
        _id(self.dependency_id, "dependency_id")
        _id(self.revision_or_digest, "revision_or_digest")
        object.__setattr__(self, "coverage_ref", _optional_id(self.coverage_ref, "dependency.coverage_ref"))

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceDependency":
        data = _closed(value, {"dependency_id", "revision_or_digest"}, {"coverage_ref"}, "evidence_dependency")
        return cls(data["dependency_id"], data["revision_or_digest"], data.get("coverage_ref"))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"dependency_id": self.dependency_id, "revision_or_digest": self.revision_or_digest,
                "coverage_ref": self.coverage_ref}


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    kind: str
    value: object
    support_refs: tuple[str, ...]
    derivation_kind: Confidence

    SCHEMA: ClassVar[str] = EVIDENCE_FACT_SCHEMA

    def __post_init__(self) -> None:
        _id(self.fact_id, "fact_id")
        _id(self.kind, "fact.kind")
        object.__setattr__(self, "value", _json_value(self.value, "fact.value"))
        object.__setattr__(self, "support_refs", _unique_ids(self.support_refs, "fact.support_refs"))
        if not self.support_refs:
            raise EvidenceContractError("fact.support_refs must name at least one receipt or span")
        if not isinstance(self.derivation_kind, Confidence):
            object.__setattr__(self, "derivation_kind", _enum(self.derivation_kind, Confidence, "fact.derivation_kind"))

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceFact":
        data = _closed(value, {"schema", "fact_id", "kind", "value", "support_refs", "derivation_kind"}, set(), "evidence_fact")
        _schema(data, cls.SCHEMA, "evidence_fact")
        return cls(data["fact_id"], data["kind"], data["value"], _unique_ids(data["support_refs"], "fact.support_refs"),  # type: ignore[arg-type]
                   _enum(data["derivation_kind"], Confidence, "fact.derivation_kind"))

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "fact_id": self.fact_id, "kind": self.kind,
                "value": copy.deepcopy(self.value), "support_refs": list(self.support_refs),
                "derivation_kind": self.derivation_kind.value}


@dataclass(frozen=True)
class EvidenceOmission:
    omission_id: str
    range_or_item: str
    reason: OmissionReason
    expansion_handle: str | None = None

    SCHEMA: ClassVar[str] = EVIDENCE_OMISSION_SCHEMA

    def __post_init__(self) -> None:
        _id(self.omission_id, "omission_id")
        _text(self.range_or_item, "omission.range_or_item", max_length=2048)
        if not isinstance(self.reason, OmissionReason):
            object.__setattr__(self, "reason", _enum(self.reason, OmissionReason, "omission.reason"))
        if self.expansion_handle is not None:
            object.__setattr__(self, "expansion_handle", _opaque_handle(self.expansion_handle, "omission.expansion_handle"))

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceOmission":
        data = _closed(value, {"schema", "omission_id", "range_or_item", "reason"}, {"expansion_handle"}, "evidence_omission")
        _schema(data, cls.SCHEMA, "evidence_omission")
        return cls(data["omission_id"], data["range_or_item"], _enum(data["reason"], OmissionReason, "omission.reason"), data.get("expansion_handle"))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "omission_id": self.omission_id, "range_or_item": self.range_or_item,
                "reason": self.reason.value, "expansion_handle": self.expansion_handle}


@dataclass(frozen=True)
class EvidenceView:
    """Deterministic projection over one source snapshot."""

    view_id: str
    transform_version: str
    store_id: str
    namespace_id: str
    repo_id: str
    worktree_id: str
    snapshot_id: str
    source_ref: str
    source_sha256: str
    origin_receipt_id: str
    source_revision: str
    dependency_vector: tuple[EvidenceDependency, ...]
    snapshot_seq: int
    selector: EvidenceSelector
    segments: tuple[EvidenceSpan, ...]
    facts: tuple[EvidenceFact, ...]
    availability: Availability
    coverage: EvidenceCoverage
    applicability: ApplicabilityState
    visibility: EvidenceVisibility
    omitted: tuple[EvidenceOmission, ...]
    protected_anchor_ids: tuple[str, ...]
    view_sha256: str
    expansion_handle: str | None = None

    SCHEMA: ClassVar[str] = EVIDENCE_VIEW_SCHEMA

    def __post_init__(self) -> None:
        _id(self.view_id, "view_id")
        _id(self.transform_version, "transform_version")
        _id(self.store_id, "store_id")
        _id(self.namespace_id, "namespace_id")
        _id(self.repo_id, "repo_id")
        _id(self.worktree_id, "worktree_id")
        _id(self.snapshot_id, "snapshot_id")
        _text(self.source_ref, "source_ref", max_length=2048)
        _digest(self.source_sha256, "source_sha256")
        _id(self.origin_receipt_id, "origin_receipt_id")
        _id(self.source_revision, "source_revision")
        _nonnegative(self.snapshot_seq, "snapshot_seq")
        if not isinstance(self.selector, EvidenceSelector):
            raise EvidenceContractError("view.selector must be EvidenceSelector")
        object.__setattr__(self, "dependency_vector", tuple(self.dependency_vector))
        if any(not isinstance(item, EvidenceDependency) for item in self.dependency_vector):
            raise EvidenceContractError("view.dependency_vector contains an invalid item")
        if len({item.dependency_id for item in self.dependency_vector}) != len(self.dependency_vector):
            raise EvidenceContractError("view.dependency_vector must not repeat dependency IDs")
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "facts", tuple(self.facts))
        object.__setattr__(self, "omitted", tuple(self.omitted))
        if any(not isinstance(item, EvidenceSpan) for item in self.segments):
            raise EvidenceContractError("view.segments contains an invalid item")
        if any(not isinstance(item, EvidenceFact) for item in self.facts):
            raise EvidenceContractError("view.facts contains an invalid item")
        if any(not isinstance(item, EvidenceOmission) for item in self.omitted):
            raise EvidenceContractError("view.omitted contains an invalid item")
        if not isinstance(self.availability, Availability):
            object.__setattr__(self, "availability", _enum(self.availability, Availability, "view.availability"))
        if not isinstance(self.coverage, EvidenceCoverage):
            raise EvidenceContractError("view.coverage must be EvidenceCoverage")
        if not isinstance(self.applicability, ApplicabilityState):
            object.__setattr__(self, "applicability", _enum(self.applicability, ApplicabilityState, "view.applicability"))
        if not isinstance(self.visibility, EvidenceVisibility):
            object.__setattr__(self, "visibility", _enum(self.visibility, EvidenceVisibility, "view.visibility"))
        object.__setattr__(self, "protected_anchor_ids", _unique_ids(self.protected_anchor_ids, "protected_anchor_ids"))
        _digest(self.view_sha256, "view_sha256")
        if self.expansion_handle is not None:
            object.__setattr__(self, "expansion_handle", _opaque_handle(self.expansion_handle, "view.expansion_handle"))
        ids = self.item_ids
        if len(ids) != len(set(ids)):
            raise EvidenceContractError("view item IDs must be unique across segments, facts, and omissions")
        if not set(self.protected_anchor_ids).issubset(set(ids)):
            raise EvidenceContractError("every protected anchor must identify a segment or typed fact")
        previous_end = -1
        for segment in self.segments:
            if segment.byte_start < previous_end:
                raise EvidenceContractError("view segments must be ordered and non-overlapping")
            previous_end = segment.byte_end

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(item.span_id for item in self.segments) + tuple(item.fact_id for item in self.facts) + tuple(item.omission_id for item in self.omitted)

    def _without_digest(self) -> dict[str, object]:
        value = self.to_dict()
        value["view_sha256"] = None
        return value

    def computed_digest(self) -> str:
        return digest_json(self._without_digest())

    def validate_integrity(self) -> None:
        if self.view_sha256 != self.computed_digest():
            raise EvidenceContractError("view_sha256 does not match the closed view record")

    def _validate_snapshot_binding(self, snapshot: SourceSnapshot) -> None:
        if not isinstance(snapshot, SourceSnapshot):
            raise EvidenceContractError("view validation requires a SourceSnapshot")
        snapshot.validate_integrity()
        if self.visibility.level < snapshot.visibility.level:
            raise EvidenceContractError("view visibility widens the source snapshot visibility")
        expected = {
            "store_id": snapshot.store_id, "namespace_id": snapshot.namespace_id,
            "repo_id": snapshot.repo_id, "worktree_id": snapshot.worktree_id,
            "snapshot_id": snapshot.snapshot_id, "source_ref": snapshot.source_ref,
            "source_sha256": snapshot.source_sha256, "origin_receipt_id": snapshot.origin_receipt_id,
            "source_revision": snapshot.source_revision,
        }
        actual = {key: getattr(self, key) for key in expected}
        if actual != expected:
            raise EvidenceContractError("view is bound to a different source snapshot")
        for segment in self.segments:
            if segment.byte_end > len(snapshot.body):
                raise EvidenceContractError("segment extends past source snapshot")
            if snapshot.body[segment.byte_start:segment.byte_end] and digest_bytes(snapshot.body[segment.byte_start:segment.byte_end]) != segment.body_sha256:
                raise EvidenceContractError("segment body digest differs from source snapshot")
            expected_range = snapshot.byte_range_for_lines(segment.start_line, segment.end_line)
            if expected_range != (segment.byte_start, segment.byte_end):
                raise EvidenceContractError("segment line and byte coordinates disagree")
    def validate_against(self, snapshot: SourceSnapshot) -> None:
        self._validate_snapshot_binding(snapshot)
        self.validate_integrity()

    @classmethod
    def create(
        cls,
        snapshot: SourceSnapshot,
        *,
        transform_version: str,
        snapshot_seq: int,
        selector: EvidenceSelector,
        coverage: EvidenceCoverage,
        segments: Sequence[EvidenceSpan] = (),
        facts: Sequence[EvidenceFact] = (),
        availability: Availability = Availability.PRESENT,
        applicability: ApplicabilityState = ApplicabilityState.CURRENT,
        visibility: EvidenceVisibility | None = None,
        omitted: Sequence[EvidenceOmission] = (),
        protected_anchor_ids: Sequence[str] = (),
        dependency_vector: Sequence[EvidenceDependency] = (),
        expansion_handle: str | None = None,
    ) -> "EvidenceView":
        if not isinstance(snapshot, SourceSnapshot):
            raise EvidenceContractError("view creation requires a SourceSnapshot")
        view_visibility = visibility or snapshot.visibility
        provisional = cls(
            "view:pending", transform_version, snapshot.store_id, snapshot.namespace_id,
            snapshot.repo_id, snapshot.worktree_id, snapshot.snapshot_id, snapshot.source_ref,
            snapshot.source_sha256, snapshot.origin_receipt_id, snapshot.source_revision,
            tuple(dependency_vector),
            snapshot_seq, selector, tuple(segments), tuple(facts), availability, coverage,
            applicability, view_visibility, tuple(omitted), tuple(protected_anchor_ids),
            "sha256:" + "0" * 64, expansion_handle,
        )
        provisional._validate_snapshot_binding(snapshot)
        view_id = "view:" + hashlib.sha256(_canonical(provisional._without_digest())).hexdigest()
        with_id = cls(
            view_id, provisional.transform_version, provisional.store_id, provisional.namespace_id,
            provisional.repo_id, provisional.worktree_id, provisional.snapshot_id, provisional.source_ref,
            provisional.source_sha256, provisional.origin_receipt_id, provisional.source_revision,
            provisional.dependency_vector, provisional.snapshot_seq, provisional.selector,
            provisional.segments, provisional.facts, provisional.availability, provisional.coverage,
            provisional.applicability, provisional.visibility, provisional.omitted,
            provisional.protected_anchor_ids, "sha256:" + "0" * 64, provisional.expansion_handle,
        )
        view_sha256 = with_id.computed_digest()
        result = cls(
            with_id.view_id, with_id.transform_version, with_id.store_id, with_id.namespace_id,
            with_id.repo_id, with_id.worktree_id, with_id.snapshot_id, with_id.source_ref,
            with_id.source_sha256, with_id.origin_receipt_id, with_id.source_revision,
            with_id.dependency_vector, with_id.snapshot_seq, with_id.selector, with_id.segments,
            with_id.facts, with_id.availability, with_id.coverage, with_id.applicability,
            with_id.visibility, with_id.omitted, with_id.protected_anchor_ids, view_sha256,
            with_id.expansion_handle,
        )
        result.validate_against(snapshot)
        return result

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceView":
        required = {"schema", "view_id", "transform_version", "store_id", "namespace_id", "repo_id", "worktree_id", "snapshot_id", "source_ref", "source_sha256", "origin_receipt_id", "source_revision", "dependency_vector", "snapshot_seq", "selector", "segments", "facts", "availability", "coverage", "applicability", "visibility", "omitted", "protected_anchor_ids", "view_sha256"}
        data = _closed(value, required, {"expansion_handle"}, "evidence_view")
        _schema(data, cls.SCHEMA, "evidence_view")
        result = cls(
            data["view_id"], data["transform_version"], data["store_id"], data["namespace_id"],
            data["repo_id"], data["worktree_id"], data["snapshot_id"], data["source_ref"],
            data["source_sha256"], data["origin_receipt_id"], data["source_revision"],
            tuple(EvidenceDependency.from_dict(item) for item in _sequence(data["dependency_vector"], "view.dependency_vector")),
            data["snapshot_seq"], EvidenceSelector.from_dict(data["selector"]),
            tuple(EvidenceSpan.from_dict(item) for item in _sequence(data["segments"], "view.segments")),
            tuple(EvidenceFact.from_dict(item) for item in _sequence(data["facts"], "view.facts")),
            _enum(data["availability"], Availability, "view.availability"),
            EvidenceCoverage.from_dict(data["coverage"]),
            _enum(data["applicability"], ApplicabilityState, "view.applicability"),
            _enum(data["visibility"], EvidenceVisibility, "view.visibility"),
            tuple(EvidenceOmission.from_dict(item) for item in _sequence(data["omitted"], "view.omitted")),
            _unique_ids(data["protected_anchor_ids"], "protected_anchor_ids"), data["view_sha256"], data.get("expansion_handle"),
        )
        result.validate_integrity()
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA, "view_id": self.view_id, "transform_version": self.transform_version,
            "store_id": self.store_id, "namespace_id": self.namespace_id, "repo_id": self.repo_id,
            "worktree_id": self.worktree_id, "snapshot_id": self.snapshot_id, "source_ref": self.source_ref,
            "source_sha256": self.source_sha256, "origin_receipt_id": self.origin_receipt_id,
            "source_revision": self.source_revision,
            "dependency_vector": [item.to_dict() for item in self.dependency_vector],
            "snapshot_seq": self.snapshot_seq, "selector": self.selector.to_dict(),
            "segments": [item.to_dict() for item in self.segments],
            "facts": [item.to_dict() for item in self.facts], "availability": self.availability.value,
            "coverage": self.coverage.to_dict(), "applicability": self.applicability.value,
            "visibility": self.visibility.value, "omitted": [item.to_dict() for item in self.omitted],
            "protected_anchor_ids": list(self.protected_anchor_ids), "view_sha256": self.view_sha256,
            "expansion_handle": self.expansion_handle,
        }


@dataclass(frozen=True)
class OracleSpan:
    anchor_id: str
    byte_start: int
    byte_end: int
    start_line: int
    end_line: int
    body_sha256: str

    @classmethod
    def from_dict(cls, value: object) -> "OracleSpan":
        data = _closed(value, {"anchor_id", "byte_start", "byte_end", "start_line", "end_line", "body_sha256"}, set(), "oracle_span")
        return cls(data["anchor_id"], data["byte_start"], data["byte_end"], data["start_line"], data["end_line"], data["body_sha256"])  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"anchor_id": self.anchor_id, "byte_start": self.byte_start, "byte_end": self.byte_end,
                "start_line": self.start_line, "end_line": self.end_line, "body_sha256": self.body_sha256}


@dataclass(frozen=True)
class FixtureOracle:
    """Independently authored expected protected slices for public fixtures."""

    fixture_id: str
    source_ref: str
    source_sha256: str
    visibility: EvidenceVisibility
    protected_slices: tuple[OracleSpan, ...]

    SCHEMA: ClassVar[str] = EVIDENCE_FIXTURE_SCHEMA

    def __post_init__(self) -> None:
        _id(self.fixture_id, "fixture_id")
        _text(self.source_ref, "fixture.source_ref", max_length=2048)
        _digest(self.source_sha256, "fixture.source_sha256")
        if not isinstance(self.visibility, EvidenceVisibility):
            object.__setattr__(self, "visibility", _enum(self.visibility, EvidenceVisibility, "fixture.visibility"))
        object.__setattr__(self, "protected_slices", tuple(self.protected_slices))
        if not self.protected_slices or any(not isinstance(item, OracleSpan) for item in self.protected_slices):
            raise EvidenceContractError("fixture must contain protected oracle slices")

    @classmethod
    def from_dict(cls, value: object) -> "FixtureOracle":
        data = _closed(value, {"schema", "fixture_id", "source_ref", "source_sha256", "visibility", "protected_slices"}, set(), "fixture_oracle")
        _schema(data, cls.SCHEMA, "fixture_oracle")
        return cls(data["fixture_id"], data["source_ref"], data["source_sha256"],
                   _enum(data["visibility"], EvidenceVisibility, "fixture.visibility"),
                   tuple(OracleSpan.from_dict(item) for item in _sequence(data["protected_slices"], "fixture.protected_slices")))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "fixture_id": self.fixture_id, "source_ref": self.source_ref,
                "source_sha256": self.source_sha256, "visibility": self.visibility.value,
                "protected_slices": [item.to_dict() for item in self.protected_slices]}

    def validate_snapshot(self, snapshot: SourceSnapshot) -> None:
        snapshot.validate_integrity()
        if snapshot.source_ref != self.source_ref or snapshot.source_sha256 != self.source_sha256:
            raise EvidenceContractError("fixture oracle does not match source snapshot identity")
        if snapshot.visibility.level < self.visibility.level:
            raise EvidenceContractError("fixture oracle visibility widens source visibility")
        for expected in self.protected_slices:
            start = _nonnegative(expected.byte_start, "oracle.byte_start")
            end = _nonnegative(expected.byte_end, "oracle.byte_end")
            if end <= start or end > len(snapshot.body):
                raise EvidenceContractError("oracle slice is outside source snapshot")
            if digest_bytes(snapshot.body[start:end]) != expected.body_sha256:
                raise EvidenceContractError(f"oracle slice digest mismatch: {expected.anchor_id}")
            if snapshot.byte_range_for_lines(expected.start_line, expected.end_line) != (start, end):
                raise EvidenceContractError(f"oracle slice coordinates mismatch: {expected.anchor_id}")

    def expected_segments(self) -> tuple[EvidenceSpan, ...]:
        return tuple(EvidenceSpan(item.anchor_id, item.byte_start, item.byte_end, item.start_line,
                                  item.end_line, item.body_sha256) for item in self.protected_slices)


@dataclass(frozen=True)
class DeliveryManifest:
    """Certified delivery facts; selection, dispatch, response, and use stay separate."""

    delivery_id: str
    request_id: str
    view_id: str
    store_id: str
    namespace_id: str
    visibility: EvidenceVisibility
    selected_item_ids: tuple[str, ...]
    sent_item_ids: tuple[str, ...]
    response_received: bool | None
    consumed_item_ids: tuple[str, ...]
    coverage: EvidenceCoverage
    wire_sha256: str | None
    response_id: str | None = None

    SCHEMA: ClassVar[str] = EVIDENCE_DELIVERY_SCHEMA

    def __post_init__(self) -> None:
        _id(self.delivery_id, "delivery_id")
        _id(self.request_id, "request_id")
        _id(self.view_id, "view_id")
        _id(self.store_id, "store_id")
        _id(self.namespace_id, "namespace_id")
        if not isinstance(self.visibility, EvidenceVisibility):
            object.__setattr__(self, "visibility", _enum(self.visibility, EvidenceVisibility, "delivery.visibility"))
        selected = _unique_ids(self.selected_item_ids, "selected_item_ids")
        sent = _unique_ids(self.sent_item_ids, "sent_item_ids")
        consumed = _unique_ids(self.consumed_item_ids, "consumed_item_ids")
        object.__setattr__(self, "selected_item_ids", selected)
        object.__setattr__(self, "sent_item_ids", sent)
        object.__setattr__(self, "consumed_item_ids", consumed)
        if not set(sent).issubset(set(selected)):
            raise EvidenceContractError("sent items must be selected items")
        if not set(consumed).issubset(set(sent)):
            raise EvidenceContractError("consumed items must be sent items")
        if type(self.response_received) not in (bool, type(None)):
            raise EvidenceContractError("response_received must be true, false, or unknown")
        if self.response_received is True and self.response_id is None:
            raise EvidenceContractError("a received response requires response_id")
        if self.response_received is not True and self.response_id is not None:
            raise EvidenceContractError("response_id requires response_received=true")
        if consumed and self.response_received is not True:
            raise EvidenceContractError("consumed items require a received response")
        if self.response_id is not None:
            object.__setattr__(self, "response_id", _id(self.response_id, "response_id"))
        if self.wire_sha256 is not None:
            object.__setattr__(self, "wire_sha256", _digest(self.wire_sha256, "wire_sha256"))
        if sent and self.wire_sha256 is None:
            raise EvidenceContractError("sent delivery requires certified wire bytes")
        if not isinstance(self.coverage, EvidenceCoverage):
            raise EvidenceContractError("delivery.coverage must be EvidenceCoverage")

    @classmethod
    def certify(
        cls,
        view: EvidenceView,
        snapshot: SourceSnapshot,
        *,
        request_id: str,
        selected_item_ids: Sequence[str],
        sent_item_ids: Sequence[str],
        wire_bytes: bytes,
        response_received: bool | None = None,
        response_id: str | None = None,
        consumed_item_ids: Sequence[str] = (),
        visibility: EvidenceVisibility | None = None,
        coverage: EvidenceCoverage | None = None,
    ) -> "DeliveryManifest":
        view.validate_against(snapshot)
        if not isinstance(wire_bytes, bytes):
            raise EvidenceContractError("certified wire body must be bytes")
        selected = tuple(selected_item_ids)
        sent = tuple(sent_item_ids)
        protected = set(view.protected_anchor_ids)
        view_items = set(view.item_ids)
        if not set(selected).issubset(view_items):
            raise EvidenceContractError("selected delivery item is absent from the evidence view")
        if not protected.issubset(set(selected)):
            raise EvidenceContractError("all protected anchors must be selected")
        if not protected.issubset(set(sent)):
            raise EvidenceContractError("all protected anchors must be sent")
        manifest_coverage = coverage or view.coverage
        delivery_visibility = visibility or view.visibility
        if delivery_visibility.level < view.visibility.level:
            raise EvidenceContractError("delivery visibility widens the evidence view")
        delivery_id = "delivery:" + hashlib.sha256(_canonical({
            "request_id": request_id, "view_id": view.view_id,
            "wire_sha256": digest_bytes(wire_bytes), "namespace_id": view.namespace_id,
        })).hexdigest()
        result = cls(delivery_id, request_id, view.view_id, view.store_id, view.namespace_id,
                     delivery_visibility, selected, sent, response_received, tuple(consumed_item_ids),
                     manifest_coverage, digest_bytes(wire_bytes), response_id)
        result.verify_wire(view, snapshot, wire_bytes)
        return result

    def verify_wire(self, view: EvidenceView, snapshot: SourceSnapshot, wire_bytes: bytes) -> None:
        view.validate_against(snapshot)
        if self.view_id != view.view_id or self.store_id != view.store_id or self.namespace_id != view.namespace_id:
            raise EvidenceContractError("delivery manifest is bound to a different view scope")
        if self.visibility.level < view.visibility.level:
            raise EvidenceContractError("delivery visibility widens the evidence view")
        if self.wire_sha256 != digest_bytes(wire_bytes):
            raise EvidenceContractError("wire bytes differ from certified delivery digest")
        items = {item.span_id: item for item in view.segments}
        items.update({item.fact_id: item for item in view.facts})
        items.update({item.omission_id: item for item in view.omitted})
        for item_id in self.sent_item_ids:
            if item_id not in items:
                raise EvidenceContractError(f"delivery references an item absent from view: {item_id}")
            item = items[item_id]
            if isinstance(item, EvidenceSpan):
                literal = snapshot.body[item.byte_start:item.byte_end]
            elif isinstance(item, EvidenceFact):
                literal = _canonical(item.to_dict())
            else:
                literal = _canonical(item.to_dict())
            if literal not in wire_bytes:
                raise EvidenceContractError(f"sent item is absent from certified wire bytes: {item_id}")

    @classmethod
    def from_dict(cls, value: object) -> "DeliveryManifest":
        data = _closed(value, {"schema", "delivery_id", "request_id", "view_id", "store_id", "namespace_id", "visibility", "selected_item_ids", "sent_item_ids", "response_received", "consumed_item_ids", "coverage", "wire_sha256"}, {"response_id"}, "delivery_manifest")
        _schema(data, cls.SCHEMA, "delivery_manifest")
        return cls(data["delivery_id"], data["request_id"], data["view_id"], data["store_id"], data["namespace_id"],
                   _enum(data["visibility"], EvidenceVisibility, "delivery.visibility"),
                   _unique_ids(data["selected_item_ids"], "selected_item_ids"),
                   _unique_ids(data["sent_item_ids"], "sent_item_ids"), data["response_received"],
                   _unique_ids(data["consumed_item_ids"], "consumed_item_ids"),
                   EvidenceCoverage.from_dict(data["coverage"]), data["wire_sha256"], data.get("response_id"))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "delivery_id": self.delivery_id, "request_id": self.request_id,
                "view_id": self.view_id, "store_id": self.store_id, "namespace_id": self.namespace_id,
                "visibility": self.visibility.value, "selected_item_ids": list(self.selected_item_ids),
                "sent_item_ids": list(self.sent_item_ids), "response_received": self.response_received,
                "consumed_item_ids": list(self.consumed_item_ids), "coverage": self.coverage.to_dict(),
                "wire_sha256": self.wire_sha256, "response_id": self.response_id}


def expansion_handle(snapshot: SourceSnapshot, principal_id: str, selector: EvidenceSelector) -> str:
    """Mint an opaque, scoped handle; no caller-supplied filesystem root is accepted."""

    principal = _id(principal_id, "principal_id")
    return "expand:" + hashlib.sha256(_canonical({
        "snapshot_id": snapshot.snapshot_id, "namespace_id": snapshot.namespace_id,
        "principal_id": principal, "selector": selector.to_dict(),
    })).hexdigest()


def outline_for_snapshot(snapshot: SourceSnapshot) -> dict[str, object]:
    """Use OUTLINEGENERIC's shared language-neutral outline API for navigation."""

    try:
        text = snapshot.body.decode(snapshot.encoding)
    except (LookupError, UnicodeDecodeError):
        return {
            "schema": OUTLINE_SCHEMA, "file": snapshot.source_ref, "language": "unknown",
            "detection": "fallback", "parser": "unavailable", "degraded": True,
            "degraded_reason": "unsupported-encoding",
        }
    try:
        try:
            from mc_repo_intel_vendor.ts_dispatch import outline_source
        except ImportError:  # pragma: no cover - package import path.
            from .mc_repo_intel_vendor.ts_dispatch import outline_source
        result = outline_source(text, snapshot.source_ref)
    except Exception as exc:  # noqa: BLE001 - navigation must not authorize edits.
        return {
            "schema": OUTLINE_SCHEMA, "file": snapshot.source_ref, "language": "unknown",
            "detection": "fallback", "parser": "unavailable", "degraded": True,
            "degraded_reason": type(exc).__name__,
        }
    if result.get("schema") != OUTLINE_SCHEMA:
        raise EvidenceContractError("OUTLINEGENERIC returned an unsupported schema")
    return result


__all__ = [
    "ApplicabilityState", "Availability", "Confidence", "DeliveryManifest",
    "EVIDENCE_COVERAGE_SCHEMA", "EVIDENCE_DELIVERY_SCHEMA", "EVIDENCE_FACT_SCHEMA",
    "EVIDENCE_FIXTURE_SCHEMA", "EVIDENCE_OMISSION_SCHEMA", "EVIDENCE_SELECTOR_SCHEMA",
    "EVIDENCE_SNAPSHOT_SCHEMA", "EVIDENCE_SPAN_SCHEMA", "EVIDENCE_VIEW_SCHEMA",
    "EvidenceContractError", "EvidenceCoverage", "EvidenceDependency", "EvidenceFact",
    "EvidenceOmission", "EvidenceSelector", "EvidenceSpan", "EvidenceView",
    "EvidenceVisibility", "FixtureOracle", "OmissionReason", "OracleSpan", "OUTLINE_SCHEMA",
    "SelectorMode", "SourceSnapshot", "STATE_STORE_SCHEMA", "STATE_STORE_SNAPSHOT_SCHEMA",
    "CoverageStatus", "digest_bytes", "digest_json", "expansion_handle", "outline_for_snapshot",
]
