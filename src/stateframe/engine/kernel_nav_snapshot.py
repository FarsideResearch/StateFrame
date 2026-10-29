"""Shared NAV1 snapshot, coverage, and evidence boundary.

This module is deliberately an adapter over existing owners.  It does not
build a symbol index, persist a graph, or admit catalog receipts.  A
``NavigationSnapshot`` captures the authorized source membership and exact
source bytes for one read-only navigation turn.  Graph queries consume the
existing ``CodeGraph`` object, while catalog queries consume the immutable
``CatalogState`` projection.

The source parser is never selected by file suffix in this module.  All
structural facts come from ``outline_for_snapshot`` and therefore use the
language-neutral OUTLINEGENERIC API for Python, Go, Rust, TypeScript,
JavaScript, Java, PHP, Vue, and the other supported languages.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
from typing import ClassVar

try:  # Flat v46 imports are used by the release loader and focused tests.
    from kernel_catalog import CatalogState
    from kernel_evidence_contracts import (
        EvidenceCoverage,
        EvidenceOmission,
        EvidenceSelector,
        EvidenceView,
        EvidenceVisibility,
        OmissionReason,
        SelectorMode,
        SourceSnapshot,
        CoverageStatus,
        digest_bytes,
        expansion_handle,
        outline_for_snapshot,
    )
    from kernel_repo_intel import FORBIDDEN_PARTS
except ImportError:  # pragma: no cover - package import path.
    from .kernel_catalog import CatalogState
    from .kernel_evidence_contracts import (
        EvidenceCoverage,
        EvidenceOmission,
        EvidenceSelector,
        EvidenceView,
        EvidenceVisibility,
        OmissionReason,
        SelectorMode,
        SourceSnapshot,
        CoverageStatus,
        digest_bytes,
        expansion_handle,
        outline_for_snapshot,
    )
    from .kernel_repo_intel import FORBIDDEN_PARTS


SCHEMA = "stateframe/nav-snapshot/1"
RESULT_SCHEMA = "stateframe/nav-result/1"
CONTINUATION_SCHEMA = "stateframe/nav-continuation/1"
CACHE_KEY_SCHEMA = "stateframe/nav-cache-key/1"
PARSER_VERSION = "mc-outline/1"
MAX_QUERY_CHARS = 240
MAX_ROWS = 24
DEFAULT_MAX_FILES = 4096
DEFAULT_MAX_FILE_BYTES = 4 * 1024 * 1024

SOURCE_SUFFIXES = frozenset({
    ".c", ".cc", ".cpp", ".cxx", ".cs", ".h", ".hpp", ".java", ".js",
    ".jsx", ".php", ".py", ".pyi", ".rb", ".rs", ".ts", ".tsx", ".vue",
    ".go",
})
DEFAULT_DENIED_PARTS = frozenset(set(FORBIDDEN_PARTS) | {
    ".git", ".hg", ".svn", ".hypothesis", ".pytest_cache", "__pycache__",
    "node_modules", ".venv", "venv",
})
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:/-]{0,255}$")


class NavigationError(ValueError):
    """Base error for an invalid or unavailable navigation contract."""


class ScopeDenied(NavigationError):
    """The requested path or graph row is outside the authorized universe."""


class EvidenceUnavailable(NavigationError):
    """A source member exists but exact bytes are not available."""


class SnapshotStale(NavigationError):
    """A snapshot, graph, catalog, or continuation no longer matches."""


class ResolutionStatus(str, Enum):
    FOUND = "FOUND"
    EMPTY = "EMPTY"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    PARTIAL = "PARTIAL"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"
    REFUSED = "REFUSED"


class CoverageCompleteness(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"
    STALE = "STALE"


class MemberStatus(str, Enum):
    INDEXED = "INDEXED"
    PARTIAL = "PARTIAL"
    UNREADABLE = "UNREADABLE"
    OVERSIZED = "OVERSIZED"


def _text(value: object, label: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NavigationError(f"{label} must be a non-empty string")
    if len(value) > maximum or "\x00" in value:
        raise NavigationError(f"{label} is invalid or too long")
    return value


def _identifier(value: object, label: str, *, maximum: int = 256) -> str:
    text = _text(value, label, maximum=maximum)
    if not _ID_RE.fullmatch(text):
        raise NavigationError(f"{label} is not a valid identifier")
    return text


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _hex_digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _normalise_relative(value: object, label: str = "path") -> str:
    text = _text(value, label, maximum=2048).replace("\\", "/")
    if text.startswith("/"):
        raise ScopeDenied(f"{label} is outside the authorized workspace")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ScopeDenied(f"{label} is not a normalized repository path")
    return "/".join(parts)


def _safe_default_id(prefix: str, name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._/-]+", "_", name).strip("_") or "root"
    return _identifier(f"{prefix}:{cleaned}", "generated identifier")


def _copy_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return copy.deepcopy(dict(value))


@dataclass(frozen=True, slots=True)
class SnapshotCoverage:
    """Coverage without denied names or a denied-item count."""

    status: CoverageCompleteness
    scanned_files: int
    failed_files: int = 0
    denied_boundary: bool = False
    truncated: bool = False
    reasons: tuple[str, ...] = ()

    SCHEMA: ClassVar[str] = "stateframe/nav-coverage/1"

    def __post_init__(self) -> None:
        if not isinstance(self.status, CoverageCompleteness):
            object.__setattr__(self, "status", CoverageCompleteness(self.status))
        for label, value in (("scanned_files", self.scanned_files),
                             ("failed_files", self.failed_files)):
            if type(value) is not int or value < 0:
                raise NavigationError(f"coverage.{label} must be >= 0")
        normalized = tuple(sorted({str(item) for item in self.reasons if str(item)}))
        object.__setattr__(self, "reasons", normalized)

    @property
    def complete(self) -> bool:
        return self.status is CoverageCompleteness.COMPLETE

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "status": self.status.value,
            "scanned_files": self.scanned_files,
            "failed_files": self.failed_files,
            "denied_boundary": self.denied_boundary,
            "truncated": self.truncated,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class SourceMember:
    """One admitted source path, without storing a second symbol index."""

    path: str
    status: MemberStatus
    source_sha256: str | None
    byte_length: int
    language: str
    outline_sha256: str | None = None
    reason: str | None = None

    SCHEMA: ClassVar[str] = "stateframe/nav-source-member/1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalise_relative(self.path))
        if not isinstance(self.status, MemberStatus):
            object.__setattr__(self, "status", MemberStatus(self.status))
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise NavigationError("source member byte_length must be >= 0")
        _text(self.language, "source member language", maximum=64)
        if self.source_sha256 is not None:
            _identifier(self.source_sha256, "source member digest")
        if self.outline_sha256 is not None:
            _identifier(self.outline_sha256, "outline digest")
        if self.status in {MemberStatus.INDEXED, MemberStatus.PARTIAL} and not self.source_sha256:
            raise NavigationError("readable source members require a source digest")
        if self.reason is not None:
            _text(self.reason, "source member reason", maximum=256)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "path": self.path,
            "status": self.status.value,
            "source_sha256": self.source_sha256,
            "byte_length": self.byte_length,
            "language": self.language,
            "outline_sha256": self.outline_sha256,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """A member plus its immutable EC01 source evidence and outline."""

    member: SourceMember
    source_snapshot: SourceSnapshot | None
    outline_json: str | None

    def __post_init__(self) -> None:
        if self.source_snapshot is None:
            if self.outline_json is not None:
                raise NavigationError("unreadable source cannot have an outline")
            return
        if self.source_snapshot.source_ref != self.member.path:
            raise NavigationError("source evidence path differs from source member")
        if self.member.source_sha256 != self.source_snapshot.source_sha256:
            raise NavigationError("source member and evidence digest differ")
        if self.outline_json is not None:
            try:
                parsed = json.loads(self.outline_json)
            except (TypeError, ValueError) as error:
                raise NavigationError("source outline is not canonical JSON") from error
            if not isinstance(parsed, Mapping):
                raise NavigationError("source outline must be an object")

    def outline(self) -> dict[str, object] | None:
        if self.outline_json is None:
            return None
        return _copy_mapping(json.loads(self.outline_json))


@dataclass(frozen=True, slots=True)
class QueryCoverage:
    """Query coverage that cannot disclose excluded or denied item counts."""

    predicate: str
    status: CoverageCompleteness
    scanned_files: int
    graph_source_match: bool | None = None
    gap_reason: str | None = None

    SCHEMA: ClassVar[str] = "stateframe/nav-query-coverage/1"

    def __post_init__(self) -> None:
        _text(self.predicate, "coverage predicate", maximum=256)
        if not isinstance(self.status, CoverageCompleteness):
            object.__setattr__(self, "status", CoverageCompleteness(self.status))
        if type(self.scanned_files) is not int or self.scanned_files < 0:
            raise NavigationError("query coverage scanned_files must be >= 0")
        if self.gap_reason is not None:
            _text(self.gap_reason, "coverage gap reason", maximum=256)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "predicate": self.predicate,
            "status": self.status.value,
            "scanned_files": self.scanned_files,
            "graph_source_match": self.graph_source_match,
            "gap_reason": self.gap_reason,
        }


@dataclass(frozen=True, slots=True)
class Continuation:
    """Opaque, snapshot- and authorization-bound result continuation."""

    handle: str
    snapshot_id: str
    scope_id: str
    authorization_epoch: str
    operation: str
    query: str
    relation: str
    page_size: int
    cursor: int
    graph_generation: str

    SCHEMA: ClassVar[str] = CONTINUATION_SCHEMA

    def __post_init__(self) -> None:
        handle = _text(self.handle, "continuation handle", maximum=16384)
        if not re.fullmatch(r"navcont:[A-Za-z0-9._:-]+", handle):
            raise NavigationError("continuation handle is not a valid opaque token")
        _identifier(self.snapshot_id, "continuation snapshot_id")
        _identifier(self.scope_id, "continuation scope_id")
        _identifier(self.authorization_epoch, "continuation authorization_epoch")
        _text(self.operation, "continuation operation", maximum=64)
        _text(self.query, "continuation query", maximum=MAX_QUERY_CHARS)
        _text(self.relation, "continuation relation", maximum=64)
        if type(self.page_size) is not int or not 1 <= self.page_size <= MAX_ROWS:
            raise NavigationError("continuation page_size is outside the bounded range")
        if type(self.cursor) is not int or self.cursor < 0:
            raise NavigationError("continuation cursor must be >= 0")
        _identifier(self.graph_generation, "continuation graph_generation")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "snapshot_id": self.snapshot_id,
            "scope_id": self.scope_id,
            "authorization_epoch": self.authorization_epoch,
            "operation": self.operation,
            "query": self.query,
            "relation": self.relation,
            "page_size": self.page_size,
            "cursor": self.cursor,
            "graph_generation": self.graph_generation,
        }

    def to_dict(self) -> dict[str, object]:
        value = self.payload()
        value["handle"] = self.handle
        return value


@dataclass(frozen=True, slots=True)
class NavigationResult:
    """Stable, bounded graph result with explicit snapshot and coverage identity."""

    result_id: str
    snapshot_id: str
    operation: str
    query: str
    status: ResolutionStatus
    rows: tuple[dict[str, object], ...]
    shown: int
    total: int | None
    total_relation: str
    coverage: QueryCoverage
    next_continuation: Continuation | None = None

    SCHEMA: ClassVar[str] = RESULT_SCHEMA

    def __post_init__(self) -> None:
        _identifier(self.result_id, "result_id")
        _identifier(self.snapshot_id, "result snapshot_id")
        _text(self.operation, "result operation", maximum=64)
        _text(self.query, "result query", maximum=MAX_QUERY_CHARS)
        if not isinstance(self.status, ResolutionStatus):
            object.__setattr__(self, "status", ResolutionStatus(self.status))
        rows = tuple(_copy_mapping(row) for row in self.rows)
        object.__setattr__(self, "rows", rows)
        if self.shown != len(rows) or self.shown < 0:
            raise NavigationError("result shown count does not match rows")
        if self.total is not None and (type(self.total) is not int or self.total < self.shown):
            raise NavigationError("result total is invalid")
        if self.total_relation not in {"exact", "lower_bound", "unknown"}:
            raise NavigationError("result total_relation is invalid")
        if self.next_continuation is not None and self.next_continuation.snapshot_id != self.snapshot_id:
            raise NavigationError("result continuation is bound to another snapshot")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "result_id": self.result_id,
            "snapshot_id": self.snapshot_id,
            "operation": self.operation,
            "query": self.query,
            "status": self.status.value,
            "rows": [_copy_mapping(row) for row in self.rows],
            "counts": {
                "shown": self.shown,
                "total": self.total,
                "total_relation": self.total_relation,
            },
            "coverage": self.coverage.to_dict(),
            "next_handle": self.next_continuation.handle if self.next_continuation else None,
        }


@dataclass(frozen=True, slots=True)
class CatalogResult:
    """Catalog A query joined to the same source snapshot identity."""

    snapshot_id: str
    catalog_watermark: int
    catalog_basis_digest: str
    result: object

    SCHEMA: ClassVar[str] = "stateframe/nav-catalog-result/1"

    def to_dict(self) -> dict[str, object]:
        renderer = getattr(self.result, "to_dict", None)
        value = renderer() if callable(renderer) else copy.deepcopy(self.result)
        return {
            "schema": self.SCHEMA,
            "snapshot_id": self.snapshot_id,
            "catalog_watermark": self.catalog_watermark,
            "catalog_basis_digest": self.catalog_basis_digest,
            "result": value,
        }


@dataclass(frozen=True, slots=True)
class NavigationCacheKey:
    """One identity for both memory and disk caches."""

    snapshot_id: str
    authorization_epoch: str
    policy_id: str
    source_ref: str
    source_sha256: str
    parser_version: str
    view: str
    query_digest: str

    SCHEMA: ClassVar[str] = CACHE_KEY_SCHEMA

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": self.SCHEMA,
            "snapshot_id": self.snapshot_id,
            "authorization_epoch": self.authorization_epoch,
            "policy_id": self.policy_id,
            "source_ref": self.source_ref,
            "source_sha256": self.source_sha256,
            "parser_version": self.parser_version,
            "view": self.view,
            "query_digest": self.query_digest,
        }

    @property
    def key(self) -> str:
        return "navcache:" + _hex_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class NavigationSnapshot:
    """Immutable authorized source/evidence snapshot for one nav query set."""

    snapshot_id: str
    root_id: str
    scope_id: str
    repo_id: str
    worktree_id: str
    authorization_epoch: str
    policy_id: str
    tree_membership_generation: str
    index_generation: str
    parser_version: str
    catalog_watermark: int
    catalog_basis_digest: str
    coverage: SnapshotCoverage
    members: tuple[SourceMember, ...]
    records: tuple[SourceRecord, ...]
    allowed_prefixes: tuple[str, ...] = ()
    denied_parts: tuple[str, ...] = ()
    source_suffixes: tuple[str, ...] = ()
    max_files: int = DEFAULT_MAX_FILES
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    policy_version: str = "nav-policy/1"
    _root_path: Path | None = field(default=None, repr=False, compare=False)
    _catalog: object | None = field(default=None, repr=False, compare=False)

    SCHEMA: ClassVar[str] = SCHEMA

    def __post_init__(self) -> None:
        _identifier(self.snapshot_id, "snapshot_id")
        for label in ("root_id", "scope_id", "repo_id", "worktree_id",
                      "authorization_epoch", "policy_id", "tree_membership_generation",
                      "index_generation", "catalog_basis_digest"):
            _identifier(getattr(self, label), label)
        _text(self.parser_version, "parser_version", maximum=128)
        if type(self.catalog_watermark) is not int or self.catalog_watermark < 0:
            raise NavigationError("catalog_watermark must be >= 0")
        members = tuple(self.members)
        records = tuple(self.records)
        if len({item.path for item in members}) != len(members):
            raise NavigationError("snapshot members must have unique paths")
        if {item.member.path for item in records} != {item.path for item in members}:
            raise NavigationError("snapshot records and members differ")
        object.__setattr__(self, "members", tuple(sorted(members, key=lambda item: item.path)))
        object.__setattr__(self, "records", tuple(sorted(records, key=lambda item: item.member.path)))
        if self.snapshot_id != "snapshot:pending" and self.snapshot_id != self._computed_snapshot_id():
            raise NavigationError("snapshot_id does not match the snapshot identity")

    @classmethod
    def capture(
        cls,
        root: str | Path,
        *,
        root_id: str | None = None,
        scope_id: str | None = None,
        repo_id: str | None = None,
        worktree_id: str | None = None,
        authorization_epoch: str = "auth:initial",
        allowed_prefixes: Sequence[str] = (),
        denied_parts: Iterable[str] = DEFAULT_DENIED_PARTS,
        source_suffixes: Iterable[str] = SOURCE_SUFFIXES,
        max_files: int = DEFAULT_MAX_FILES,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        policy_version: str = "nav-policy/1",
        parser_version: str = PARSER_VERSION,
        catalog: CatalogState | object | None = None,
    ) -> "NavigationSnapshot":
        """Capture one bounded tree and exact source-byte observation.

        The walk never follows a symlink.  Denied paths are not returned as
        members and their count is intentionally not part of the contract.
        """
        supplied_root = Path(root)
        if supplied_root.is_symlink():
            raise ScopeDenied("navigation root cannot be a symlink")
        resolved_root = supplied_root.resolve()
        if not resolved_root.is_dir():
            raise NavigationError("navigation root is not a directory")
        if type(max_files) is not int or max_files < 1:
            raise NavigationError("max_files must be >= 1")
        if type(max_file_bytes) is not int or max_file_bytes < 1:
            raise NavigationError("max_file_bytes must be >= 1")
        normalized_prefixes = tuple(sorted({_normalise_relative(item, "allowed prefix")
                                            for item in allowed_prefixes}))
        normalized_denied = tuple(sorted({str(item).casefold() for item in denied_parts if str(item)}))
        normalized_suffixes = tuple(sorted({
            str(item).casefold() if str(item).startswith(".") else "." + str(item).casefold()
            for item in source_suffixes
        }))
        rid = _identifier(root_id or _safe_default_id("workspace", resolved_root.name), "root_id")
        sid = _identifier(scope_id or rid, "scope_id")
        repo = _identifier(repo_id or _safe_default_id("repo", resolved_root.name), "repo_id")
        worktree = _identifier(worktree_id or _safe_default_id("worktree", resolved_root.name), "worktree_id")
        auth = _identifier(authorization_epoch, "authorization_epoch")
        policy = "policy:" + _hex_digest({
            "version": policy_version,
            "allowed_prefixes": normalized_prefixes,
            "denied_parts": normalized_denied,
            "source_suffixes": normalized_suffixes,
            "max_files": max_files,
            "max_file_bytes": max_file_bytes,
        })
        records, scanned, failed, denied, truncated = _scan_source_tree(
            resolved_root, scope_id=sid, repo_id=repo, worktree_id=worktree,
            authorization_epoch=auth, allowed_prefixes=normalized_prefixes,
            denied_parts=frozenset(normalized_denied), source_suffixes=frozenset(normalized_suffixes),
            max_files=max_files, max_file_bytes=max_file_bytes,
        )
        members = tuple(item.member for item in records)
        partial_reasons: list[str] = []
        if failed:
            partial_reasons.append("unreadable_or_oversized_source")
        if denied:
            partial_reasons.append("denied_or_symlink_boundary")
        if truncated:
            partial_reasons.append("source_file_limit")
        if any(item.member.status is MemberStatus.PARTIAL for item in records):
            partial_reasons.append("degraded_outline")
        coverage_status = CoverageCompleteness.PARTIAL if partial_reasons else CoverageCompleteness.COMPLETE
        coverage = SnapshotCoverage(
            coverage_status, scanned, failed, denied, truncated, tuple(partial_reasons)
        )
        tree_generation = "tree:" + _hex_digest({
            "policy_id": policy,
            "members": [(item.path, item.status.value) for item in members],
        })
        index_generation = "index:" + _hex_digest({
            "parser_version": parser_version,
            "members": [item.to_dict() for item in members],
        })
        watermark, basis = _catalog_identity(catalog)
        provisional = cls(
            "snapshot:pending", rid, sid, repo, worktree, auth, policy,
            tree_generation, index_generation, parser_version, watermark, basis,
            coverage, members, records, normalized_prefixes, normalized_denied,
            normalized_suffixes, max_files, max_file_bytes, policy_version,
            resolved_root, catalog,
        )
        snapshot_id = provisional._computed_snapshot_id()
        return cls(
            snapshot_id, rid, sid, repo, worktree, auth, policy, tree_generation,
            index_generation, parser_version, watermark, basis, coverage, members,
            records, normalized_prefixes, normalized_denied, normalized_suffixes,
            max_files, max_file_bytes, policy_version, resolved_root, catalog,
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "root_id": self.root_id,
            "scope_id": self.scope_id,
            "repo_id": self.repo_id,
            "worktree_id": self.worktree_id,
            "authorization_epoch": self.authorization_epoch,
            "policy_id": self.policy_id,
            "tree_membership_generation": self.tree_membership_generation,
            "index_generation": self.index_generation,
            "parser_version": self.parser_version,
            "catalog_watermark": self.catalog_watermark,
            "catalog_basis_digest": self.catalog_basis_digest,
            "coverage": self.coverage.to_dict(),
            "members": [item.to_dict() for item in self.members],
        }

    def _computed_snapshot_id(self) -> str:
        return "snapshot:" + _hex_digest(self._identity_payload())

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_payload(),
            "snapshot_id": self.snapshot_id,
            "allowed_prefixes": list(self.allowed_prefixes),
            "denied_parts": list(self.denied_parts),
            "source_suffixes": list(self.source_suffixes),
            "max_files": self.max_files,
            "max_file_bytes": self.max_file_bytes,
            "policy_version": self.policy_version,
            "sources": [
                {
                    "path": record.member.path,
                    "source_snapshot": record.source_snapshot.to_dict()
                    if record.source_snapshot is not None else None,
                    "outline": record.outline(),
                }
                for record in self.records
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> "NavigationSnapshot":
        if not isinstance(value, Mapping) or value.get("schema") != cls.SCHEMA:
            raise NavigationError("unsupported navigation snapshot schema")
        members_raw = value.get("members")
        if not isinstance(members_raw, Sequence) or isinstance(members_raw, (str, bytes, bytearray)):
            raise NavigationError("snapshot members must be an array")
        members = tuple(_member_from_dict(item) for item in members_raw)
        sources_raw = value.get("sources")
        if not isinstance(sources_raw, Sequence) or isinstance(sources_raw, (str, bytes, bytearray)):
            raise NavigationError("snapshot sources must be an array")
        source_by_path: dict[str, SourceRecord] = {}
        for item in sources_raw:
            if not isinstance(item, Mapping):
                raise NavigationError("snapshot source record must be an object")
            path = _normalise_relative(item.get("path"), "source path")
            if path in source_by_path:
                raise NavigationError("snapshot source records must have unique paths")
            raw_source = item.get("source_snapshot")
            source = SourceSnapshot.from_dict(raw_source) if raw_source is not None else None
            outline_value = item.get("outline")
            outline_json = None if outline_value is None else json.dumps(
                outline_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            member = next((candidate for candidate in members if candidate.path == path), None)
            if member is None:
                raise NavigationError("source record has no matching member")
            source_by_path[path] = SourceRecord(member, source, outline_json)
        records = tuple(source_by_path[item.path] for item in members if item.path in source_by_path)
        if len(records) != len(members):
            raise NavigationError("snapshot source records are incomplete")
        coverage_raw = value.get("coverage")
        if not isinstance(coverage_raw, Mapping):
            raise NavigationError("snapshot coverage must be an object")
        coverage = SnapshotCoverage(
            CoverageCompleteness(coverage_raw.get("status")),
            coverage_raw.get("scanned_files", 0), coverage_raw.get("failed_files", 0),
            bool(coverage_raw.get("denied_boundary", False)),
            bool(coverage_raw.get("truncated", False)),
            tuple(str(item) for item in coverage_raw.get("reasons", ())),
        )
        result = cls(
            str(value.get("snapshot_id")), str(value.get("root_id")), str(value.get("scope_id")),
            str(value.get("repo_id")), str(value.get("worktree_id")),
            str(value.get("authorization_epoch")), str(value.get("policy_id")),
            str(value.get("tree_membership_generation")), str(value.get("index_generation")),
            str(value.get("parser_version")), int(value.get("catalog_watermark", 0)),
            str(value.get("catalog_basis_digest")), coverage, members, records,
            tuple(str(item) for item in value.get("allowed_prefixes", ())),
            tuple(str(item) for item in value.get("denied_parts", ())),
            tuple(str(item) for item in value.get("source_suffixes", ())),
            int(value.get("max_files", DEFAULT_MAX_FILES)),
            int(value.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES)),
            str(value.get("policy_version", "nav-policy/1")), None, None,
        )
        return result

    def _record(self, path: str) -> SourceRecord:
        normalized = _normalise_relative(path)
        for record in self.records:
            if record.member.path == normalized:
                return record
        # Do not distinguish a missing path from a denied path to callers.
        raise ScopeDenied("source path is outside the authorized snapshot")

    def source_snapshot(self, path: str) -> SourceSnapshot:
        record = self._record(path)
        if record.source_snapshot is None:
            raise EvidenceUnavailable("source bytes are unavailable for this member")
        return record.source_snapshot

    def outline(self, path: str) -> dict[str, object]:
        record = self._record(path)
        outline = record.outline()
        if outline is None:
            raise EvidenceUnavailable("source outline is unavailable for this member")
        return outline

    def evidence_view(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        principal_id: str = "principal:nav",
        snapshot_seq: int | None = None,
        visibility: EvidenceVisibility = EvidenceVisibility.PUBLIC,
    ) -> EvidenceView:
        """Return an EC01 view whose bytes are bound to this NAV1 snapshot."""
        source = self.source_snapshot(path)
        final_line = source.line_count if end_line is None else end_line
        selector = EvidenceSelector(SelectorMode.SPAN, start_line=start_line, end_line=final_line)
        coverage = EvidenceCoverage(
            "coverage:nav-source-" + _hex_digest((self.snapshot_id, source.source_ref))[:32],
            "source_read", CoverageStatus.COMPLETE,
            f"{self.scope_id}:{source.source_ref}", inspected=1, expected=1,
        )
        handle = expansion_handle(source, principal_id, selector)
        if not source.body:
            omission = EvidenceOmission(
                "omission:empty-source-" + _hex_digest(source.source_ref)[:32],
                source.source_ref, OmissionReason.UNKNOWN, handle,
            )
            return EvidenceView.create(
                source,
                transform_version=f"nav1:{self.parser_version}",
                snapshot_seq=self.catalog_watermark if snapshot_seq is None else snapshot_seq,
                selector=selector, coverage=coverage, visibility=visibility,
                omitted=(omission,), expansion_handle=handle,
            )
        span = source.span(start_line, final_line)
        return EvidenceView.create(
            source,
            transform_version=f"nav1:{self.parser_version}",
            snapshot_seq=self.catalog_watermark if snapshot_seq is None else snapshot_seq,
            selector=selector,
            coverage=coverage,
            segments=(span,),
            visibility=visibility,
            protected_anchor_ids=(span.span_id,),
            expansion_handle=handle,
        )

    def cache_key(self, path: str, *, view: str, query: object = ()) -> NavigationCacheKey:
        source = self.source_snapshot(path)
        return NavigationCacheKey(
            self.snapshot_id, self.authorization_epoch, self.policy_id, source.source_ref,
            source.source_sha256, self.parser_version, _text(view, "cache view", maximum=128),
            "query:" + _hex_digest(query),
        )

    def catalog_lookup(
        self,
        catalog: CatalogState | object,
        entity_id: str,
        predicates: Sequence[str] = (),
        *,
        mode: str = "current",
        token_budget: int = 512,
        cursor: int = 0,
    ) -> CatalogResult:
        """Read Catalog A without copying its assertions into NAV1."""
        watermark, basis = _catalog_identity(catalog)
        if (watermark, basis) != (self.catalog_watermark, self.catalog_basis_digest):
            raise SnapshotStale("catalog projection is newer or differs from this snapshot")
        lookup = getattr(catalog, "lookup", None)
        if not callable(lookup):
            raise NavigationError("catalog projection does not expose lookup")
        result = lookup(entity_id, predicates, mode=mode, token_budget=token_budget, cursor=cursor)
        return CatalogResult(self.snapshot_id, watermark, basis, result)

    def is_current(self, root: str | Path | None = None) -> bool:
        if root is None:
            if self._root_path is None:
                return True
            root = self._root_path
        refreshed = self.capture(
            root,
            root_id=self.root_id, scope_id=self.scope_id, repo_id=self.repo_id,
            worktree_id=self.worktree_id, authorization_epoch=self.authorization_epoch,
            allowed_prefixes=self.allowed_prefixes, denied_parts=self.denied_parts,
            source_suffixes=self.source_suffixes, max_files=self.max_files,
            max_file_bytes=self.max_file_bytes, policy_version=self.policy_version,
            parser_version=self.parser_version, catalog=self._catalog,
        )
        return refreshed.snapshot_id == self.snapshot_id

    def require_current(self, root: str | Path | None = None) -> None:
        if not self.is_current(root):
            raise SnapshotStale("source membership or bytes changed after snapshot capture")

    def resolve_symbol(
        self,
        graph: object,
        query: str,
        *,
        relation: str = "definition",
        page_size: int = MAX_ROWS,
        continuation: str | Continuation | None = None,
    ) -> NavigationResult:
        """Query the existing graph with strict exact/ambiguous resolution."""
        normalized = _text(query, "symbol query", maximum=MAX_QUERY_CHARS).strip()
        if not 1 <= page_size <= MAX_ROWS:
            raise NavigationError("page_size is outside the bounded range")
        if relation not in {"definition", "callers", "callees"}:
            raise NavigationError("unsupported graph relation")
        graph_generation = _graph_generation(graph)
        cursor = 0
        if continuation is not None:
            token = _decode_continuation(continuation) if isinstance(continuation, str) else continuation
            self._validate_continuation(token, normalized, relation, page_size, graph_generation)
            cursor = token.cursor
        coverage = self._graph_coverage(graph)
        if coverage.status is CoverageCompleteness.STALE:
            return self._result("graph", normalized, relation, ResolutionStatus.STALE, (), coverage,
                                page_size, cursor, graph_generation)
        try:
            candidates = _strict_candidates(graph, normalized, self)
        except (AttributeError, TypeError, ValueError):
            return self._result("graph", normalized, ResolutionStatus.UNAVAILABLE, (), coverage, page_size, cursor, graph_generation)
        if not candidates:
            suggestions = _suggestions(graph, normalized, self)
            rows = tuple(suggestions)
            return self._result("graph", normalized, relation, ResolutionStatus.NOT_FOUND, rows, coverage,
                                page_size, cursor, graph_generation, total=len(rows), total_relation="unknown")
        if len(candidates) > 1:
            return self._result("graph", normalized, relation, ResolutionStatus.AMBIGUOUS, candidates, coverage,
                                page_size, cursor, graph_generation, total=len(candidates), total_relation="exact")
        target = candidates[0]
        if relation == "definition":
            rows = (target,)
        else:
            method = getattr(graph, "find_callers" if relation == "callers" else "find_callees", None)
            if not callable(method):
                return self._result("graph", normalized, relation, ResolutionStatus.UNAVAILABLE, (), coverage,
                                    page_size, cursor, graph_generation)
            try:
                refs = method(str(target["entity_id"]))
                rows = tuple(_graph_row(item) for item in refs)
                rows = tuple(item for item in rows if self._row_in_scope(item))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                return self._result("graph", normalized, relation, ResolutionStatus.UNAVAILABLE, (), coverage,
                                    page_size, cursor, graph_generation)
        status = ResolutionStatus.FOUND if rows else ResolutionStatus.EMPTY
        return self._result("graph", normalized, relation, status, rows, coverage, page_size, cursor,
                            graph_generation, total=len(rows), total_relation="exact")

    def _graph_coverage(self, graph: object) -> QueryCoverage:
        expected = {
            item.path: item.source_sha256.removeprefix("sha256:")
            for item in self.members if item.source_sha256
        }
        actual = getattr(graph, "file_hashes", None)
        if not isinstance(actual, Mapping):
            return QueryCoverage("repo-symbol-resolution", CoverageCompleteness.UNKNOWN,
                                 len(expected), None, "graph source hashes unavailable")
        if any(str(actual.get(path, "")) != digest for path, digest in expected.items()):
            return QueryCoverage("repo-symbol-resolution", CoverageCompleteness.STALE,
                                 len(expected), False, "graph source bytes differ from snapshot")
        if any(path not in expected for path in actual):
            return QueryCoverage("repo-symbol-resolution", CoverageCompleteness.STALE,
                                 len(expected), False, "graph includes a source outside snapshot")
        incomplete = getattr(graph, "incomplete_files", None)
        has_incomplete = False
        if callable(incomplete):
            try:
                has_incomplete = bool(tuple(incomplete()))
            except (TypeError, ValueError, RuntimeError):
                has_incomplete = True
        if self.coverage.status is not CoverageCompleteness.COMPLETE or has_incomplete:
            return QueryCoverage("repo-symbol-resolution", CoverageCompleteness.PARTIAL,
                                 len(expected), True, "source or graph coverage is partial")
        return QueryCoverage("repo-symbol-resolution", CoverageCompleteness.COMPLETE,
                             len(expected), True)

    def _row_in_scope(self, row: Mapping[str, object]) -> bool:
        path = row.get("file") or row.get("locator")
        if not isinstance(path, str):
            return False
        try:
            normalized = _normalise_relative(path.split(":", 1)[0], "graph path")
        except NavigationError:
            return False
        return any(item.path == normalized for item in self.members)

    def _validate_continuation(
        self, token: Continuation, query: str, relation: str, page_size: int, graph_generation: str,
    ) -> None:
        if token.snapshot_id != self.snapshot_id or token.scope_id != self.scope_id:
            raise SnapshotStale("continuation belongs to another source snapshot")
        if token.authorization_epoch != self.authorization_epoch:
            raise ScopeDenied("continuation authorization epoch is no longer valid")
        if token.query != query or token.relation != relation or token.page_size != page_size:
            raise NavigationError("continuation query does not match the requested expansion")
        if token.graph_generation != graph_generation:
            raise SnapshotStale("continuation graph generation is no longer current")

    def _result(
        self,
        operation: str,
        query: str,
        relation: str,
        status: ResolutionStatus,
        rows: Sequence[Mapping[str, object]],
        coverage: QueryCoverage,
        page_size: int,
        cursor: int,
        graph_generation: str,
        *,
        total: int | None = None,
        total_relation: str = "unknown",
    ) -> NavigationResult:
        all_rows = tuple(_copy_mapping(row) for row in rows)
        page = all_rows[cursor:cursor + page_size]
        next_token = None
        if cursor + page_size < len(all_rows):
            next_token = _make_continuation(
                self, operation, query, relation, page_size, cursor + page_size, graph_generation,
            )
        if total is None:
            total = len(all_rows) if not next_token else None
        if next_token is not None:
            total_relation = "lower_bound" if total is not None else "unknown"
        payload = {
            "schema": RESULT_SCHEMA, "snapshot_id": self.snapshot_id,
            "operation": operation, "query": query, "status": status.value,
            "rows": page, "coverage": coverage.to_dict(), "cursor": cursor,
            "next": next_token.handle if next_token else None,
        }
        return NavigationResult(
            "result:" + _hex_digest(payload), self.snapshot_id, operation, query, status,
            page, len(page), total, total_relation, coverage, next_token,
        )


def _catalog_identity(catalog: object | None) -> tuple[int, str]:
    if catalog is None:
        return 0, "catalog:empty"
    watermark = getattr(catalog, "watermark", None)
    basis = getattr(catalog, "basis_digest", None)
    if type(watermark) is not int or watermark < 0 or not callable(basis):
        raise NavigationError("catalog projection lacks an immutable watermark/basis API")
    digest = str(basis()).strip()
    if re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        digest = "sha256:" + digest.lower()
    return watermark, _identifier(digest, "catalog basis digest")


def _scan_source_tree(
    root: Path,
    *,
    scope_id: str,
    repo_id: str,
    worktree_id: str,
    authorization_epoch: str,
    allowed_prefixes: tuple[str, ...],
    denied_parts: frozenset[str],
    source_suffixes: frozenset[str],
    max_files: int,
    max_file_bytes: int,
) -> tuple[tuple[SourceRecord, ...], int, int, bool, bool]:
    records: list[SourceRecord] = []
    scanned = failed = 0
    denied = truncated = False

    def allowed(relative: str) -> bool:
        return not allowed_prefixes or any(
            relative == prefix or relative.startswith(prefix + "/")
            for prefix in allowed_prefixes
        )

    def walk(directory: Path, parent: str = "") -> None:
        nonlocal scanned, failed, denied, truncated
        if truncated:
            return
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError:
            failed += 1
            return
        for entry in entries:
            relative = f"{parent}/{entry.name}" if parent else entry.name
            parts = relative.split("/")
            if any(part.casefold() in denied_parts for part in parts):
                continue
            if not allowed(relative) and entry.is_dir(follow_symlinks=False):
                continue
            try:
                if entry.is_symlink():
                    denied = True
                    continue
                if entry.is_dir(follow_symlinks=False):
                    walk(Path(entry.path), relative)
                    continue
                if not entry.is_file(follow_symlinks=False) or not allowed(relative):
                    continue
                if Path(entry.name).suffix.casefold() not in source_suffixes:
                    continue
                scanned += 1
                if scanned > max_files:
                    scanned -= 1
                    truncated = True
                    return
                stat = entry.stat(follow_symlinks=False)
                if stat.st_size > max_file_bytes:
                    failed += 1
                    records.append(SourceRecord(
                        SourceMember(relative, MemberStatus.OVERSIZED, None, stat.st_size, "unknown",
                                     reason="file-byte-limit"), None, None
                    ))
                    continue
                raw = Path(entry.path).read_bytes()
            except OSError:
                failed += 1
                records.append(SourceRecord(
                    SourceMember(relative, MemberStatus.UNREADABLE, None, 0, "unknown",
                                 reason="source-read-failed"), None, None
                ))
                continue
            source = SourceSnapshot.capture(
                raw, store_id="store:nav", namespace_id=scope_id, repo_id=repo_id,
                worktree_id=worktree_id, source_ref=relative,
                source_revision=digest_bytes(raw),
                origin_receipt_id="receipt:nav:" + _hex_digest((relative, digest_bytes(raw))),
                visibility=EvidenceVisibility.PUBLIC,
            )
            try:
                outline = outline_for_snapshot(source)
            except Exception as error:  # noqa: BLE001 - parse failure is evidence, not a crash.
                outline = {
                    "schema": "mc-outline/1", "file": relative, "language": "unknown",
                    "detection": "fallback", "parser": "unavailable", "degraded": True,
                    "degraded_reason": type(error).__name__,
                }
            outline_json = json.dumps(outline, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            degraded = bool(outline.get("degraded"))
            status = MemberStatus.PARTIAL if degraded else MemberStatus.INDEXED
            records.append(SourceRecord(
                SourceMember(relative, status, source.source_sha256, len(raw),
                             str(outline.get("language", "unknown")), digest_bytes(outline_json.encode("utf-8")),
                             reason=str(outline.get("degraded_reason")) if degraded else None),
                source, outline_json,
            ))

    walk(root)
    return tuple(records), scanned, failed, denied, truncated


def _member_from_dict(value: object) -> SourceMember:
    if not isinstance(value, Mapping) or value.get("schema") != SourceMember.SCHEMA:
        raise NavigationError("invalid source member")
    return SourceMember(
        str(value.get("path")), MemberStatus(value.get("status")), value.get("source_sha256"),
        int(value.get("byte_length", 0)), str(value.get("language")), value.get("outline_sha256"),
        value.get("reason"),
    )


def _graph_generation(graph: object) -> str:
    method = getattr(graph, "index_token", None)
    if callable(method):
        value = method()
        return _identifier(str(value), "graph generation")
    return "graph:unknown"


def _graph_row(value: object) -> dict[str, object]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        raw = value.to_dict()
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raw = {
            key: getattr(value, key)
            for key in ("symbol", "file", "qualname", "lineno", "kind")
            if hasattr(value, key)
        }
    symbol = str(raw.get("symbol") or raw.get("entity_id") or "")
    file = str(raw.get("file") or "")
    row = {
        "entity_id": symbol,
        "kind": str(raw.get("kind") or "unknown"),
        "locator": f"{file}:{raw.get('lineno', 0)}",
        "file": file,
        "lineno": int(raw.get("lineno", 0) or 0),
        "qualname": str(raw.get("qualname") or symbol),
    }
    return row


def _strict_candidates(graph: object, query: str, snapshot: NavigationSnapshot) -> tuple[dict[str, object], ...]:
    candidates: dict[str, dict[str, object]] = {}
    namespaces = []
    for name in ("defs", "variables"):
        value = getattr(graph, name, {})
        if isinstance(value, Mapping):
            namespaces.append(value)
    exact_id = any(query in namespace for namespace in namespaces)
    for namespace in namespaces:
        for key, value in namespace.items():
            row = _graph_row(value)
            entity_id = str(key or row["entity_id"])
            row["entity_id"] = entity_id
            if not snapshot._row_in_scope(row):
                continue
            short = str(getattr(value, "short_name", "") or entity_id.rsplit("::", 1)[-1].split(".")[-1])
            qualname = str(row.get("qualname") or "")
            matches = entity_id == query or qualname == query
            if not exact_id and query == short:
                matches = True
            if matches:
                candidates[entity_id] = row
    return tuple(sorted(candidates.values(), key=lambda row: (
        str(row.get("file", "")), int(row.get("lineno", 0) or 0), str(row["entity_id"])
    )))


def _suggestions(graph: object, query: str, snapshot: NavigationSnapshot) -> list[dict[str, object]]:
    search = getattr(graph, "search_names", None)
    if not callable(search):
        return []
    try:
        values = search(query, limit=MAX_ROWS)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return []
    result = []
    for value in values:
        row = _graph_row(value)
        if snapshot._row_in_scope(row):
            row["evidence_role"] = "suggestion"
            result.append(row)
    return result[:MAX_ROWS]


def _make_continuation(
    snapshot: NavigationSnapshot, operation: str, query: str, relation: str,
    page_size: int, cursor: int, graph_generation: str,
) -> Continuation:
    payload = {
        "schema": CONTINUATION_SCHEMA, "snapshot_id": snapshot.snapshot_id,
        "scope_id": snapshot.scope_id, "authorization_epoch": snapshot.authorization_epoch,
        "operation": operation, "query": query, "relation": relation,
        "page_size": page_size, "cursor": cursor, "graph_generation": graph_generation,
    }
    encoded = base64.urlsafe_b64encode(_canonical(payload)).decode("ascii").rstrip("=")
    handle = "navcont:" + _hex_digest(payload) + "." + encoded
    return Continuation(handle, snapshot.snapshot_id, snapshot.scope_id,
                        snapshot.authorization_epoch, operation, query, relation,
                        page_size, cursor, graph_generation)


def _decode_continuation(value: str) -> Continuation:
    text = _text(value, "continuation handle", maximum=16384)
    if not text.startswith("navcont:") or "." not in text:
        raise NavigationError("invalid continuation handle")
    digest, encoded = text[len("navcont:"):].split(".", 1)
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise NavigationError("invalid continuation payload") from error
    if not isinstance(payload, Mapping) or _hex_digest(payload) != digest:
        raise NavigationError("continuation identity digest mismatch")
    try:
        return Continuation(
            text, str(payload.get("snapshot_id")), str(payload.get("scope_id")),
            str(payload.get("authorization_epoch")), str(payload.get("operation")),
            str(payload.get("query")), str(payload.get("relation")),
            int(payload.get("page_size", 0)), int(payload.get("cursor", -1)),
            str(payload.get("graph_generation")),
        )
    except (TypeError, ValueError) as error:
        raise NavigationError("continuation fields are invalid") from error


__all__ = [
    "CACHE_KEY_SCHEMA", "CONTINUATION_SCHEMA", "CatalogResult", "Continuation",
    "CoverageCompleteness", "EvidenceUnavailable", "MemberStatus", "NavigationCacheKey",
    "NavigationError", "NavigationResult", "NavigationSnapshot", "QueryCoverage",
    "RESULT_SCHEMA", "SCHEMA", "ScopeDenied", "SnapshotCoverage", "SnapshotStale",
    "SourceMember", "SourceRecord", "_decode_continuation",
]
