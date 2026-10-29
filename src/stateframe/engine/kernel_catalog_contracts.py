"""Pure C0 contracts for the Catalog A projection.

This module owns shapes and authority boundaries only.  It does not subscribe to
the runtime journal, read files, call providers, persist data, or render a
prompt.  A :class:`CatalogRegistry` is an in-memory admission index that makes
workspace scope, registered predicates, coverage, and reference integrity
explicit before a later projection lane is attached.

The module deliberately keeps confidence separate from freshness.  ``observed``
means a host receipt was observed; it does not mean the observation is still
current or that an observed interpretation is true.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

try:  # Supports both the v46 flat-module loader and package imports.
    from kernel_host_facts import HostFact
except ImportError:  # pragma: no cover - exercised only by package importers.
    from .kernel_host_facts import HostFact


SCHEMA = "catalog-contracts/1"
ENTITY_SCHEMA = "catalog-entity/1"
NAMESPACE_SCHEMA = "catalog-namespace/1"
SCOPE_SCHEMA = "catalog-scope/1"
CONTENT_SCHEMA = "catalog-content/1"
COVERAGE_SCHEMA = "catalog-coverage/1"
PROVENANCE_SCHEMA = "catalog-provenance/1"
FRESHNESS_SCHEMA = "catalog-freshness/1"
RECALL_REF_SCHEMA = "catalog-recall-ref/1"
SCILEDGER_REF_SCHEMA = "catalog-sciledger-ref/1"
APPLICABILITY_SCHEMA = "catalog-applicability/1"

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:/-]{0,255}$")
_MAX_JSON_BYTES = 64 * 1024


class ContractError(ValueError):
    """Base error for a value that cannot enter the C0 contract boundary."""


class UnknownFieldError(ContractError):
    """Raised when a closed record contains an undeclared field."""


class UnknownVersionError(ContractError):
    """Raised when a record belongs to an unsupported schema version."""


class ReferenceError(ContractError):
    """Raised for a dangling, foreign-workspace, or malformed reference."""


class IntegrityError(ContractError):
    """Raised when an immutable identity is reused with contradictory data."""


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Confidence(_StringEnum):
    """Epistemic provenance, intentionally independent of freshness."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    ASSUMED = "assumed"


class CoverageStatus(_StringEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ApplicabilityState(_StringEnum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class Availability(_StringEnum):
    PRESENT = "present"
    ABSENT = "absent"
    UNAVAILABLE = "unavailable"
    DENIED = "denied"
    UNSUPPORTED = "unsupported"


def _ensure_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or not _ID_RE.fullmatch(value):
        raise ContractError(f"{label} must be a non-empty identifier")
    return value


def _ensure_text(value: object, label: str, *, max_length: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")
    if len(value) > max_length:
        raise ContractError(f"{label} exceeds {max_length} characters")
    if "\x00" in value:
        raise ContractError(f"{label} contains a NUL character")
    return value


def _ensure_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{label} must be an integer >= 0")
    return value


def _ensure_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError(f"{label} must be an integer >= 1")
    return value


def _ensure_finite_number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ContractError(f"{label} must be a finite number >= {minimum}")
    return result


def _json_value(value: object, label: str = "value") -> object:
    """Validate and copy a JSON-shaped value without accepting executable objects."""

    if value is None or isinstance(value, (str, int, bool)):
        return copy.deepcopy(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{label} object keys must be strings")
            result[key] = _json_value(nested, f"{label}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, f"{label}[]") for item in value]
    raise ContractError(f"{label} contains unsupported type {type(value).__name__}")


def _canonical(value: object) -> bytes:
    encoded = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_JSON_BYTES:
        raise ContractError(f"canonical value exceeds {_MAX_JSON_BYTES} bytes")
    return encoded


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _closed(value: object, required: set[str], optional: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    result = dict(value)
    if any(not isinstance(key, str) for key in result):
        raise ContractError(f"{label} keys must be strings")
    unknown = sorted(set(result) - required - optional)
    if unknown:
        raise UnknownFieldError(f"{label} has unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(result))
    if missing:
        raise ContractError(f"{label} is missing required field(s): {', '.join(missing)}")
    return result


def _schema(data: Mapping[str, object], expected: str, label: str) -> None:
    actual = data.get("schema")
    if actual != expected:
        raise UnknownVersionError(f"{label}.schema must be {expected}")


def _string_tuple(value: object, label: str, *, identifiers: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractError(f"{label} must be an array")
    result = tuple(
        (_ensure_id(item, f"{label}[]") if identifiers else _ensure_text(item, f"{label}[]", max_length=512))
        for item in value
    )
    if len(set(result)) != len(result):
        raise ContractError(f"{label} must not contain duplicates")
    return result


def _enum(value: object, enum_type: type[_StringEnum], label: str) -> _StringEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ContractError(f"{label} must be one of: {allowed}") from exc


@dataclass(frozen=True, slots=True)
class Namespace:
    workspace_id: str
    project_id: str | None = None
    resource_root_id: str | None = None

    SCHEMA: ClassVar[str] = NAMESPACE_SCHEMA

    def __post_init__(self) -> None:
        _ensure_id(self.workspace_id, "namespace.workspace_id")
        if self.project_id is not None:
            _ensure_id(self.project_id, "namespace.project_id")
        if self.resource_root_id is not None:
            _ensure_id(self.resource_root_id, "namespace.resource_root_id")

    @classmethod
    def from_dict(cls, value: object) -> "Namespace":
        data = _closed(value, {"schema", "workspace_id"}, {"project_id", "resource_root_id"}, "namespace")
        _schema(data, cls.SCHEMA, "namespace")
        return cls(data["workspace_id"], data.get("project_id"), data.get("resource_root_id"))

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "workspace_id": self.workspace_id,
                "project_id": self.project_id, "resource_root_id": self.resource_root_id}


@dataclass(frozen=True, slots=True)
class EntityKindSpec:
    kind: str
    locator_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _ensure_id(self.kind, "entity_kind.kind")
        object.__setattr__(self, "locator_fields",
                           _string_tuple(self.locator_fields, "entity_kind.locator_fields", identifiers=True))


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    predicate: str
    subject_kinds: tuple[str, ...] = ()
    value_types: tuple[str, ...] = ()
    requires_coverage: bool = True

    def __post_init__(self) -> None:
        _ensure_id(self.predicate, "predicate.predicate")
        object.__setattr__(self, "subject_kinds",
                           _string_tuple(self.subject_kinds, "predicate.subject_kinds", identifiers=True))
        object.__setattr__(self, "value_types",
                           _string_tuple(self.value_types, "predicate.value_types", identifiers=True))
        allowed = {"null", "boolean", "number", "string", "array", "object"}
        if not set(self.value_types).issubset(allowed):
            raise ContractError("predicate.value_types contains an unsupported JSON type")
        if not isinstance(self.requires_coverage, bool):
            raise ContractError("predicate.requires_coverage must be boolean")


@dataclass(frozen=True, slots=True)
class CoverageSpec:
    kind: str
    statuses: tuple[CoverageStatus, ...] = tuple(CoverageStatus)

    def __post_init__(self) -> None:
        _ensure_id(self.kind, "coverage.kind")
        statuses = tuple(self.statuses)
        if not statuses or len(set(statuses)) != len(statuses):
            raise ContractError("coverage.statuses must be non-empty and unique")
        if any(not isinstance(status, CoverageStatus) for status in statuses):
            raise ContractError("coverage.statuses must contain CoverageStatus values")
        object.__setattr__(self, "statuses", statuses)


class EntityRegistry:
    """Registered entity kinds plus immutable identity reuse protection."""

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = _ensure_id(workspace_id, "entity_registry.workspace_id")
        self._kinds: dict[str, EntityKindSpec] = {}
        self._identities: dict[str, str] = {}

    def register_kind(self, spec: EntityKindSpec) -> None:
        previous = self._kinds.get(spec.kind)
        if previous is not None and previous != spec:
            raise IntegrityError(f"entity kind reused with different definition: {spec.kind}")
        self._kinds[spec.kind] = spec

    def require_kind(self, kind: str) -> EntityKindSpec:
        try:
            return self._kinds[kind]
        except KeyError as exc:
            raise ReferenceError(f"unregistered entity kind: {kind}") from exc

    def validate(self, entity: "EntityRef") -> None:
        if not isinstance(entity, EntityRef):
            raise ContractError("entity registry requires an EntityRef")
        if entity.namespace.workspace_id != self.workspace_id:
            raise ReferenceError("entity belongs to a different workspace")
        self.require_kind(entity.kind)
        identity = entity.identity_digest()
        previous = self._identities.get(entity.entity_id)
        if previous is not None and previous != identity:
            raise IntegrityError(f"entity identity reused contradictorily: {entity.entity_id}")

    def register(self, entity: "EntityRef") -> None:
        self.validate(entity)
        identity = entity.identity_digest()
        self._identities[entity.entity_id] = identity

    def contains(self, entity_id: str) -> bool:
        return entity_id in self._identities


class PredicateRegistry:
    """Explicit predicate definitions; no implicit predicate admission."""

    def __init__(self) -> None:
        self._predicates: dict[str, PredicateSpec] = {}

    def register(self, spec: PredicateSpec) -> None:
        previous = self._predicates.get(spec.predicate)
        if previous is not None and previous != spec:
            raise IntegrityError(f"predicate reused with different definition: {spec.predicate}")
        self._predicates[spec.predicate] = spec

    def require(self, predicate: str) -> PredicateSpec:
        try:
            return self._predicates[predicate]
        except KeyError as exc:
            raise ReferenceError(f"unregistered predicate: {predicate}") from exc


class CoverageRegistry:
    """Coverage kinds and their allowed COMPLETE/PARTIAL/UNKNOWN states."""

    def __init__(self) -> None:
        self._coverage: dict[str, CoverageSpec] = {}

    def register(self, spec: CoverageSpec) -> None:
        previous = self._coverage.get(spec.kind)
        if previous is not None and previous != spec:
            raise IntegrityError(f"coverage kind reused with different definition: {spec.kind}")
        self._coverage[spec.kind] = spec

    def require(self, kind: str, status: CoverageStatus) -> CoverageSpec:
        try:
            spec = self._coverage[kind]
        except KeyError as exc:
            raise ReferenceError(f"unregistered coverage kind: {kind}") from exc
        if status not in spec.statuses:
            raise ContractError(f"coverage status {status} is not allowed for {kind}")
        return spec


@dataclass(frozen=True, slots=True)
class EntityRef:
    entity_id: str
    kind: str
    namespace: Namespace
    locator: object
    generation: str | None = None
    aliases: tuple[str, ...] = ()
    owner_ref: str | None = None

    SCHEMA: ClassVar[str] = ENTITY_SCHEMA

    def __post_init__(self) -> None:
        _ensure_id(self.entity_id, "entity.entity_id")
        _ensure_id(self.kind, "entity.kind")
        if not isinstance(self.namespace, Namespace):
            raise ContractError("entity.namespace must be Namespace")
        object.__setattr__(self, "locator", _json_value(self.locator, "entity.locator"))
        object.__setattr__(self, "aliases", _string_tuple(self.aliases, "entity.aliases", identifiers=True))
        if self.generation is not None:
            _ensure_id(self.generation, "entity.generation")
        if self.owner_ref is not None:
            _ensure_id(self.owner_ref, "entity.owner_ref")

    def identity_digest(self) -> str:
        return _digest({"entity_id": self.entity_id, "kind": self.kind,
                        "namespace": self.namespace.to_dict(), "locator": self.locator})

    @classmethod
    def from_dict(cls, value: object) -> "EntityRef":
        data = _closed(value, {"schema", "entity_id", "kind", "namespace", "locator"},
                       {"generation", "aliases", "owner_ref"}, "entity")
        _schema(data, cls.SCHEMA, "entity")
        return cls(data["entity_id"], data["kind"], Namespace.from_dict(data["namespace"]),
                   data["locator"], data.get("generation"), tuple(data.get("aliases", ())),
                   data.get("owner_ref"))

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "entity_id": self.entity_id, "kind": self.kind,
                "namespace": self.namespace.to_dict(), "locator": copy.deepcopy(self.locator),
                "generation": self.generation, "aliases": list(self.aliases), "owner_ref": self.owner_ref}


@dataclass(frozen=True, slots=True)
class ScopeRef:
    namespace: Namespace
    scope_id: str = "workspace"
    selector: Mapping[str, object] = field(default_factory=dict)

    SCHEMA: ClassVar[str] = SCOPE_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, Namespace):
            raise ContractError("scope.namespace must be Namespace")
        _ensure_id(self.scope_id, "scope.scope_id")
        object.__setattr__(self, "selector", _json_value(self.selector, "scope.selector"))
        if not isinstance(self.selector, dict):
            raise ContractError("scope.selector must be an object")

    @classmethod
    def from_dict(cls, value: object) -> "ScopeRef":
        data = _closed(value, {"schema", "namespace"}, {"scope_id", "selector"}, "scope")
        _schema(data, cls.SCHEMA, "scope")
        return cls(Namespace.from_dict(data["namespace"]), data.get("scope_id", "workspace"),
                   data.get("selector", {}))

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "namespace": self.namespace.to_dict(),
                "scope_id": self.scope_id, "selector": copy.deepcopy(dict(self.selector))}


@dataclass(frozen=True, slots=True)
class ContentHandle:
    """Stable identity for exact bytes observed by a catalog producer.

    The handle deliberately carries no body.  Bodies belong to the existing
    evidence/cache owner; this contract only binds source identity, digest,
    encoding, and namespace so structural records cannot outlive their bytes
    unnoticed.
    """

    content_id: str
    namespace: Namespace
    source_ref: str
    content_digest: str
    byte_length: int
    encoding: str = "utf-8"

    SCHEMA: ClassVar[str] = CONTENT_SCHEMA

    def __post_init__(self) -> None:
        _ensure_id(self.content_id, "content.content_id")
        if not isinstance(self.namespace, Namespace):
            raise ContractError("content.namespace must be Namespace")
        _ensure_text(self.source_ref, "content.source_ref", max_length=2048)
        _ensure_text(self.content_digest, "content.content_digest", max_length=128)
        _ensure_nonnegative_int(self.byte_length, "content.byte_length")
        _ensure_text(self.encoding, "content.encoding", max_length=128)

    @classmethod
    def capture(
        cls,
        body: bytes | bytearray,
        *,
        namespace: Namespace,
        source_ref: str,
        encoding: str = "utf-8",
    ) -> "ContentHandle":
        raw = bytes(body)
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        identity = {
            "schema": cls.SCHEMA,
            "namespace": namespace.to_dict(),
            "source_ref": source_ref,
            "content_digest": digest,
            "byte_length": len(raw),
            "encoding": encoding,
        }
        content_id = "content:" + hashlib.sha256(_canonical(identity)).hexdigest()
        return cls(content_id, namespace, source_ref, digest, len(raw), encoding)

    @classmethod
    def from_dict(cls, value: object) -> "ContentHandle":
        data = _closed(
            value,
            {"schema", "content_id", "namespace", "source_ref", "content_digest", "byte_length"},
            {"encoding"},
            "content",
        )
        _schema(data, cls.SCHEMA, "content")
        return cls(
            data["content_id"],
            Namespace.from_dict(data["namespace"]),
            data["source_ref"],
            data["content_digest"],
            data["byte_length"],
            data.get("encoding", "utf-8"),
        )  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "content_id": self.content_id,
            "namespace": self.namespace.to_dict(),
            "source_ref": self.source_ref,
            "content_digest": self.content_digest,
            "byte_length": self.byte_length,
            "encoding": self.encoding,
        }

    def key(self) -> tuple[str, str, str]:
        return self.namespace.workspace_id, self.source_ref, self.content_digest


@dataclass(frozen=True, slots=True)
class CoverageRef:
    coverage_id: str
    kind: str
    status: CoverageStatus
    inspected: int | None = None
    expected: int | None = None

    SCHEMA: ClassVar[str] = COVERAGE_SCHEMA

    def __post_init__(self) -> None:
        _ensure_id(self.coverage_id, "coverage.coverage_id")
        _ensure_id(self.kind, "coverage.kind")
        if not isinstance(self.status, CoverageStatus):
            raise ContractError("coverage.status must be CoverageStatus")
        if self.inspected is not None:
            _ensure_nonnegative_int(self.inspected, "coverage.inspected")
        if self.expected is not None:
            _ensure_nonnegative_int(self.expected, "coverage.expected")
        if self.inspected is not None and self.expected is not None and self.inspected > self.expected:
            raise ContractError("coverage.inspected cannot exceed coverage.expected")

    @classmethod
    def from_dict(cls, value: object) -> "CoverageRef":
        data = _closed(value, {"schema", "coverage_id", "kind", "status"}, {"inspected", "expected"}, "coverage")
        _schema(data, cls.SCHEMA, "coverage")
        return cls(data["coverage_id"], data["kind"], _enum(data["status"], CoverageStatus, "coverage.status"),
                   data.get("inspected"), data.get("expected"))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "coverage_id": self.coverage_id, "kind": self.kind,
                "status": self.status.value, "inspected": self.inspected, "expected": self.expected}


@dataclass(frozen=True, slots=True)
class Freshness:
    observed_at: str
    host_epoch: str
    observed_monotonic: float
    ttl_seconds: float | None = None
    source_revision: str | None = None
    state: ApplicabilityState = ApplicabilityState.CURRENT

    SCHEMA: ClassVar[str] = FRESHNESS_SCHEMA

    def __post_init__(self) -> None:
        _ensure_text(self.observed_at, "freshness.observed_at", max_length=128)
        _ensure_id(self.host_epoch, "freshness.host_epoch")
        object.__setattr__(self, "observed_monotonic", _ensure_finite_number(
            self.observed_monotonic, "freshness.observed_monotonic"))
        if self.ttl_seconds is not None:
            object.__setattr__(self, "ttl_seconds", _ensure_finite_number(self.ttl_seconds, "freshness.ttl_seconds"))
        if self.source_revision is not None:
            _ensure_id(self.source_revision, "freshness.source_revision")
        if not isinstance(self.state, ApplicabilityState):
            raise ContractError("freshness.state must be ApplicabilityState")

    def evaluate(self, *, now_monotonic: float, host_epoch: str) -> ApplicabilityState:
        _ensure_finite_number(now_monotonic, "now_monotonic")
        _ensure_id(host_epoch, "host_epoch")
        if self.state is not ApplicabilityState.CURRENT:
            return self.state
        if host_epoch != self.host_epoch:
            return ApplicabilityState.STALE
        if self.ttl_seconds is not None and now_monotonic > self.observed_monotonic + self.ttl_seconds:
            return ApplicabilityState.STALE
        return ApplicabilityState.CURRENT

    @classmethod
    def from_dict(cls, value: object) -> "Freshness":
        data = _closed(value, {"schema", "observed_at", "host_epoch", "observed_monotonic"},
                       {"ttl_seconds", "source_revision", "state"}, "freshness")
        _schema(data, cls.SCHEMA, "freshness")
        return cls(data["observed_at"], data["host_epoch"], data["observed_monotonic"],
                   data.get("ttl_seconds"), data.get("source_revision"),
                   _enum(data.get("state", ApplicabilityState.CURRENT.value), ApplicabilityState, "freshness.state"))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "observed_at": self.observed_at, "host_epoch": self.host_epoch,
                "observed_monotonic": self.observed_monotonic, "ttl_seconds": self.ttl_seconds,
                "source_revision": self.source_revision, "state": self.state.value}


@dataclass(frozen=True, slots=True)
class RecallRef:
    workspace_id: str
    item_id: str

    SCHEMA: ClassVar[str] = RECALL_REF_SCHEMA

    def __post_init__(self) -> None:
        _ensure_id(self.workspace_id, "recall.workspace_id")
        _ensure_id(self.item_id, "recall.item_id")

    @classmethod
    def from_dict(cls, value: object) -> "RecallRef":
        data = _closed(value, {"schema", "workspace_id", "item_id"}, set(), "recall_ref")
        _schema(data, cls.SCHEMA, "recall_ref")
        return cls(data["workspace_id"], data["item_id"])

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "workspace_id": self.workspace_id, "item_id": self.item_id}


@dataclass(frozen=True, slots=True)
class SciledgerRef:
    workspace_id: str
    claim_ref: str
    revision: int | None = None

    SCHEMA: ClassVar[str] = SCILEDGER_REF_SCHEMA

    def __post_init__(self) -> None:
        _ensure_id(self.workspace_id, "sciledger.workspace_id")
        _ensure_id(self.claim_ref, "sciledger.claim_ref")
        if self.revision is not None:
            _ensure_positive_int(self.revision, "sciledger.revision")

    @classmethod
    def from_dict(cls, value: object) -> "SciledgerRef":
        data = _closed(value, {"schema", "workspace_id", "claim_ref"}, {"revision"}, "sciledger_ref")
        _schema(data, cls.SCHEMA, "sciledger_ref")
        return cls(data["workspace_id"], data["claim_ref"], data.get("revision"))

    @classmethod
    def from_value(cls, value: object, *, workspace_id: str) -> "SciledgerRef":
        """Adapt ClaimSpec/ClaimState-shaped values without importing the store."""

        if isinstance(value, cls):
            result = value
        elif isinstance(value, str):
            result = cls(workspace_id, value)
        elif isinstance(value, Mapping):
            allowed = {"schema", "workspace_id", "claim_ref", "claim_id", "revision"}
            unknown = set(value) - allowed
            if unknown:
                raise UnknownFieldError(f"sciledger value has unknown field(s): {', '.join(sorted(unknown))}")
            if value.get("schema") is not None and value["schema"] != "sciledger/claim/1":
                raise UnknownVersionError("unsupported sciledger reference schema")
            claim_ref = value.get("claim_ref", value.get("claim_id"))
            if claim_ref is None:
                raise ReferenceError("sciledger value requires claim_ref or claim_id")
            if value.get("claim_ref") is not None and value.get("claim_id") is not None and value["claim_ref"] != value["claim_id"]:
                raise IntegrityError("sciledger claim_ref and claim_id disagree")
            result = cls(value.get("workspace_id", workspace_id), claim_ref, value.get("revision"))
        else:
            claim_ref = getattr(value, "claim_ref", getattr(value, "claim_id", None))
            if claim_ref is None:
                raise ReferenceError("sciledger value requires claim_ref or claim_id")
            result = cls(getattr(value, "workspace_id", workspace_id), claim_ref,
                         getattr(value, "revision", None))
        if result.workspace_id != workspace_id:
            raise ReferenceError("sciledger reference belongs to a different workspace")
        return result

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "workspace_id": self.workspace_id,
                "claim_ref": self.claim_ref, "revision": self.revision}

    def key(self) -> tuple[str, str, int | None]:
        return self.workspace_id, self.claim_ref, self.revision


@dataclass(frozen=True, slots=True)
class DependencyRef:
    kind: str
    entity: EntityRef
    revision_or_digest: str
    coverage: CoverageRef | None = None

    def __post_init__(self) -> None:
        _ensure_id(self.kind, "dependency.kind")
        if not isinstance(self.entity, EntityRef):
            raise ContractError("dependency.entity must be EntityRef")
        if self.coverage is not None and not isinstance(self.coverage, CoverageRef):
            raise ContractError("dependency.coverage must be CoverageRef or None")
        _ensure_text(self.revision_or_digest, "dependency.revision_or_digest", max_length=256)

    @classmethod
    def from_dict(cls, value: object) -> "DependencyRef":
        data = _closed(value, {"kind", "entity", "revision_or_digest"}, {"coverage"}, "dependency")
        return cls(data["kind"], EntityRef.from_dict(data["entity"]), data["revision_or_digest"],
                   None if data.get("coverage") is None else CoverageRef.from_dict(data["coverage"]))

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "entity": self.entity.to_dict(),
                "revision_or_digest": self.revision_or_digest,
                "coverage": None if self.coverage is None else self.coverage.to_dict()}


@dataclass(frozen=True, slots=True)
class Provenance:
    producer_id: str
    producer_version: str
    observed_at: str
    admitted_seq: int
    observed_step: int | None = None
    receipt_refs: tuple[str, ...] = ()
    source_span_refs: tuple[str, ...] = ()
    depends_on: tuple[DependencyRef, ...] = ()
    derivation_ref: str | None = None
    supersedes_ref: str | None = None
    visibility: str = "workspace"
    trust_domain: str = "host"

    SCHEMA: ClassVar[str] = PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        _ensure_id(self.producer_id, "provenance.producer_id")
        _ensure_text(self.producer_version, "provenance.producer_version", max_length=128)
        _ensure_text(self.observed_at, "provenance.observed_at", max_length=128)
        _ensure_nonnegative_int(self.admitted_seq, "provenance.admitted_seq")
        if self.observed_step is not None:
            _ensure_nonnegative_int(self.observed_step, "provenance.observed_step")
        object.__setattr__(self, "receipt_refs", _string_tuple(self.receipt_refs, "provenance.receipt_refs"))
        object.__setattr__(self, "source_span_refs", _string_tuple(self.source_span_refs, "provenance.source_span_refs"))
        if any(not isinstance(item, DependencyRef) for item in self.depends_on):
            raise ContractError("provenance.depends_on must contain DependencyRef values")
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        if self.derivation_ref is not None:
            _ensure_id(self.derivation_ref, "provenance.derivation_ref")
        if self.supersedes_ref is not None:
            _ensure_id(self.supersedes_ref, "provenance.supersedes_ref")
        _ensure_id(self.visibility, "provenance.visibility")
        _ensure_id(self.trust_domain, "provenance.trust_domain")

    @classmethod
    def from_dict(cls, value: object) -> "Provenance":
        data = _closed(value, {"schema", "producer_id", "producer_version", "observed_at", "admitted_seq"},
                       {"observed_step", "receipt_refs", "source_span_refs", "depends_on", "derivation_ref", "supersedes_ref", "visibility", "trust_domain"},
                       "provenance")
        _schema(data, cls.SCHEMA, "provenance")
        return cls(data["producer_id"], data["producer_version"], data["observed_at"], data["admitted_seq"],
                   data.get("observed_step"), tuple(data.get("receipt_refs", ())), tuple(data.get("source_span_refs", ())),
                   tuple(DependencyRef.from_dict(item) for item in data.get("depends_on", ())),
                   data.get("derivation_ref"), data.get("supersedes_ref"), data.get("visibility", "workspace"),
                   data.get("trust_domain", "host"))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "producer_id": self.producer_id, "producer_version": self.producer_version,
                "observed_at": self.observed_at, "admitted_seq": self.admitted_seq, "observed_step": self.observed_step,
                "receipt_refs": list(self.receipt_refs), "source_span_refs": list(self.source_span_refs),
                "depends_on": [item.to_dict() for item in self.depends_on], "derivation_ref": self.derivation_ref,
                "supersedes_ref": self.supersedes_ref, "visibility": self.visibility, "trust_domain": self.trust_domain}


@dataclass(frozen=True, slots=True)
class AssertionEnvelope:
    schema: str
    assertion_id: str
    revision: int
    subject: EntityRef
    predicate: str
    value: object
    scope: ScopeRef
    coverage: CoverageRef
    provenance: Provenance
    freshness: Freshness
    confidence: Confidence
    recall_refs: tuple[RecallRef, ...] = ()
    sciledger_refs: tuple[SciledgerRef, ...] = ()

    SCHEMA: ClassVar[str] = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != self.SCHEMA:
            raise UnknownVersionError(f"assertion.schema must be {self.SCHEMA}")
        _ensure_id(self.assertion_id, "assertion.assertion_id")
        _ensure_positive_int(self.revision, "assertion.revision")
        if not isinstance(self.subject, EntityRef):
            raise ContractError("assertion.subject must be EntityRef")
        _ensure_id(self.predicate, "assertion.predicate")
        if not isinstance(self.scope, ScopeRef):
            raise ContractError("assertion.scope must be ScopeRef")
        if not isinstance(self.coverage, CoverageRef):
            raise ContractError("assertion.coverage must be CoverageRef")
        if not isinstance(self.provenance, Provenance):
            raise ContractError("assertion.provenance must be Provenance")
        if not isinstance(self.freshness, Freshness):
            raise ContractError("assertion.freshness must be Freshness")
        object.__setattr__(self, "value", _json_value(self.value, "assertion.value"))
        if not isinstance(self.confidence, Confidence):
            raise ContractError("assertion.confidence must be Confidence")
        if any(not isinstance(item, RecallRef) for item in self.recall_refs):
            raise ContractError("assertion.recall_refs must contain RecallRef values")
        if any(not isinstance(item, SciledgerRef) for item in self.sciledger_refs):
            raise ContractError("assertion.sciledger_refs must contain SciledgerRef values")
        object.__setattr__(self, "recall_refs", tuple(self.recall_refs))
        object.__setattr__(self, "sciledger_refs", tuple(self.sciledger_refs))
        if len(set(self.recall_refs)) != len(self.recall_refs):
            raise ContractError("assertion.recall_refs must not contain duplicates")
        if len(set(ref.key() for ref in self.sciledger_refs)) != len(self.sciledger_refs):
            raise ContractError("assertion.sciledger_refs must not contain duplicates")

    @classmethod
    def create(cls, *, revision: int, subject: EntityRef, predicate: str, value: object,
               scope: ScopeRef, coverage: CoverageRef, provenance: Provenance,
               freshness: Freshness, confidence: Confidence,
               recall_refs: Sequence[RecallRef] = (),
               sciledger_refs: Sequence[SciledgerRef] = ()) -> "AssertionEnvelope":
        body = {
            "schema": cls.SCHEMA, "revision": revision, "subject": subject.to_dict(),
            "predicate": predicate, "value": value, "scope": scope.to_dict(),
            "coverage": coverage.to_dict(), "provenance": provenance.to_dict(),
            "freshness": freshness.to_dict(), "confidence": confidence.value,
            "recall_refs": [item.to_dict() for item in recall_refs],
            "sciledger_refs": [item.to_dict() for item in sciledger_refs],
        }
        return cls(cls.SCHEMA, "assertion:" + hashlib.sha256(_canonical(body)).hexdigest(), revision,
                   subject, predicate, value, scope, coverage, provenance, freshness, confidence,
                   tuple(recall_refs), tuple(sciledger_refs))

    @classmethod
    def from_dict(cls, value: object) -> "AssertionEnvelope":
        required = {"schema", "assertion_id", "revision", "subject", "predicate", "value", "scope",
                    "coverage", "provenance", "freshness", "confidence", "recall_refs", "sciledger_refs"}
        data = _closed(value, required, set(), "assertion")
        _schema(data, cls.SCHEMA, "assertion")
        return cls(data["schema"], data["assertion_id"], data["revision"], EntityRef.from_dict(data["subject"]),
                   data["predicate"], data["value"], ScopeRef.from_dict(data["scope"]), CoverageRef.from_dict(data["coverage"]),
                   Provenance.from_dict(data["provenance"]), Freshness.from_dict(data["freshness"]),
                   _enum(data["confidence"], Confidence, "assertion.confidence"),
                   tuple(RecallRef.from_dict(item) for item in data["recall_refs"]),
                   tuple(SciledgerRef.from_dict(item) for item in data["sciledger_refs"]))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "assertion_id": self.assertion_id, "revision": self.revision,
                "subject": self.subject.to_dict(), "predicate": self.predicate,
                "value": copy.deepcopy(self.value), "scope": self.scope.to_dict(),
                "coverage": self.coverage.to_dict(), "provenance": self.provenance.to_dict(),
                "freshness": self.freshness.to_dict(), "confidence": self.confidence.value,
                "recall_refs": [item.to_dict() for item in self.recall_refs],
                "sciledger_refs": [item.to_dict() for item in self.sciledger_refs]}

    def key(self) -> tuple[str, str, str]:
        return self.subject.entity_id, self.predicate, _digest(self.scope.to_dict())

    def applicability(self, *, now_monotonic: float, host_epoch: str,
                      as_of_seq: int | None = None) -> "Applicability":
        state = self.freshness.evaluate(now_monotonic=now_monotonic, host_epoch=host_epoch)
        reason = None if state is ApplicabilityState.CURRENT else f"freshness:{state.value}"
        if as_of_seq is not None:
            _ensure_nonnegative_int(as_of_seq, "as_of_seq")
            if as_of_seq < self.provenance.admitted_seq:
                state = ApplicabilityState.UNKNOWN
                reason = "assertion admitted after requested watermark"
        return Applicability(self.assertion_id, as_of_seq if as_of_seq is not None else self.provenance.admitted_seq,
                             state, Availability.PRESENT, self.coverage.status, self.coverage.coverage_id,
                             None, reason, (), None)


@dataclass(frozen=True, slots=True)
class Applicability:
    assertion_id: str
    as_of_seq: int
    state: ApplicabilityState
    availability: Availability
    coverage: CoverageStatus
    coverage_ref: str
    valid_until: float | None = None
    stale_reason: str | None = None
    invalidated_by: tuple[str, ...] = ()
    owner_state_ref: str | None = None

    SCHEMA: ClassVar[str] = APPLICABILITY_SCHEMA

    def __post_init__(self) -> None:
        _ensure_id(self.assertion_id, "applicability.assertion_id")
        _ensure_nonnegative_int(self.as_of_seq, "applicability.as_of_seq")
        _ensure_id(self.coverage_ref, "applicability.coverage_ref")
        if not isinstance(self.state, ApplicabilityState) or not isinstance(self.availability, Availability):
            raise ContractError("applicability enum fields are invalid")
        if not isinstance(self.coverage, CoverageStatus):
            raise ContractError("applicability.coverage is invalid")
        object.__setattr__(self, "invalidated_by", _string_tuple(self.invalidated_by, "applicability.invalidated_by", identifiers=True))
        if self.stale_reason is not None:
            _ensure_text(self.stale_reason, "applicability.stale_reason", max_length=512)
        if self.owner_state_ref is not None:
            _ensure_id(self.owner_state_ref, "applicability.owner_state_ref")
        if self.valid_until is not None:
            _ensure_finite_number(self.valid_until, "applicability.valid_until")

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.SCHEMA, "assertion_id": self.assertion_id, "as_of_seq": self.as_of_seq,
                "state": self.state.value, "availability": self.availability.value,
                "coverage": self.coverage.value, "coverage_ref": self.coverage_ref,
                "valid_until": self.valid_until, "stale_reason": self.stale_reason,
                "invalidated_by": list(self.invalidated_by), "owner_state_ref": self.owner_state_ref}


class CatalogRegistry:
    """C0 authority map and in-memory integrity index.

    The registry accepts only host-constructed envelopes.  It does not claim
    that an envelope's value is true; it records whether its references and
    declared contract are internally admissible.
    """

    def __init__(self, namespace: Namespace) -> None:
        if not isinstance(namespace, Namespace):
            raise ContractError("catalog registry requires a Namespace")
        self.namespace = namespace
        self.entities = EntityRegistry(namespace.workspace_id)
        self.predicates = PredicateRegistry()
        self.coverage = CoverageRegistry()
        self._recall_refs: set[tuple[str, str]] = set()
        self._sciledger_refs: set[tuple[str, str, int | None]] = set()
        self._assertions: dict[str, AssertionEnvelope] = {}
        self._superseded_by: dict[str, str] = {}

    @classmethod
    def default(cls, namespace: Namespace) -> "CatalogRegistry":
        registry = cls(namespace)
        for kind in ("artifact", "symbol", "configuration", "environment", "tool", "test",
                     "observation", "hypothesis", "decision", "claim", "task", "person",
                     "requirement", "source", "dataset", "workspace"):
            registry.entities.register_kind(EntityKindSpec(kind))
        for predicate in (
            "exists", "status", "content_digest", "defines", "contains", "calls", "references",
            "depends_on", "supports", "contradicts", "requires", "read_complete", "search_coverage",
            "execution_status", "installed", "enabled", "offered", "dispatched", "fired", "succeeded",
        ):
            registry.predicates.register(PredicateSpec(predicate))
        for kind in ("host_fact", "source_read", "search", "execution", "sciledger", "unknown"):
            registry.coverage.register(CoverageSpec(kind))
        return registry

    def register_entity_kind(self, spec: EntityKindSpec) -> None:
        self.entities.register_kind(spec)

    def register_predicate(self, spec: PredicateSpec) -> None:
        self.predicates.register(spec)

    def register_coverage(self, spec: CoverageSpec) -> None:
        self.coverage.register(spec)

    def register_entity(self, entity: EntityRef) -> None:
        self.entities.register(entity)

    def register_recall(self, reference: RecallRef) -> None:
        self._require_workspace(reference.workspace_id, "recall")
        self._recall_refs.add((reference.workspace_id, reference.item_id))

    def register_sciledger(self, reference: SciledgerRef) -> None:
        self._require_workspace(reference.workspace_id, "sciledger")
        self._sciledger_refs.add(reference.key())

    def _require_workspace(self, workspace_id: str, label: str) -> None:
        if workspace_id != self.namespace.workspace_id:
            raise ReferenceError(f"{label} reference belongs to a different workspace")

    def admit(self, assertion: AssertionEnvelope) -> AssertionEnvelope:
        self._require_workspace(assertion.subject.namespace.workspace_id, "subject")
        self._require_workspace(assertion.scope.namespace.workspace_id, "scope")
        self.entities.validate(assertion.subject)
        predicate = self.predicates.require(assertion.predicate)
        if predicate.subject_kinds and assertion.subject.kind not in predicate.subject_kinds:
            raise ContractError(f"predicate {assertion.predicate} does not accept {assertion.subject.kind}")
        if predicate.value_types:
            value_type = _value_type(assertion.value)
            if value_type not in predicate.value_types:
                raise ContractError(f"predicate {assertion.predicate} rejects JSON type {value_type}")
        self.coverage.require(assertion.coverage.kind, assertion.coverage.status)
        for reference in assertion.recall_refs:
            self._require_workspace(reference.workspace_id, "recall")
            if (reference.workspace_id, reference.item_id) not in self._recall_refs:
                raise ReferenceError(f"dangling RECALL id: {reference.item_id}")
        for reference in assertion.sciledger_refs:
            self._require_workspace(reference.workspace_id, "sciledger")
            if reference.key() not in self._sciledger_refs:
                raise ReferenceError(f"dangling sciledger ref: {reference.claim_ref}")
        for dependency in assertion.provenance.depends_on:
            self._require_workspace(dependency.entity.namespace.workspace_id, "dependency")
            if not self.entities.contains(dependency.entity.entity_id):
                raise ReferenceError(f"dangling dependency entity: {dependency.entity.entity_id}")
            if dependency.coverage is not None:
                self.coverage.require(dependency.coverage.kind, dependency.coverage.status)
        existing = self._assertions.get(assertion.assertion_id)
        if existing is not None and existing.to_dict() != assertion.to_dict():
            raise IntegrityError(f"assertion identity reused with different content: {assertion.assertion_id}")
        for linked in (assertion.provenance.derivation_ref, assertion.provenance.supersedes_ref):
            if linked is not None and linked not in self._assertions:
                raise ReferenceError(f"dangling assertion reference: {linked}")
        if assertion.provenance.supersedes_ref is not None:
            previous = self._assertions[assertion.provenance.supersedes_ref]
            if previous.key() != assertion.key():
                raise IntegrityError("superseded assertion does not have the same subject/predicate/scope")
            self._superseded_by[previous.assertion_id] = assertion.assertion_id
        self.entities.register(assertion.subject)
        self._assertions[assertion.assertion_id] = assertion
        return assertion

    def get(self, assertion_id: str) -> AssertionEnvelope | None:
        return self._assertions.get(assertion_id)

    def applicability(self, assertion: AssertionEnvelope, *, now_monotonic: float,
                      host_epoch: str, as_of_seq: int | None = None) -> Applicability:
        if self._assertions.get(assertion.assertion_id) != assertion:
            raise ReferenceError("assertion is not admitted by this registry")
        result = assertion.applicability(now_monotonic=now_monotonic, host_epoch=host_epoch,
                                         as_of_seq=as_of_seq)
        if assertion.assertion_id in self._superseded_by:
            return Applicability(assertion.assertion_id, result.as_of_seq, ApplicabilityState.STALE,
                                 result.availability, result.coverage, result.coverage_ref,
                                 stale_reason="superseded by " + self._superseded_by[assertion.assertion_id])
        if result.state is ApplicabilityState.CURRENT:
            peers = [item for item in self._assertions.values()
                     if item.assertion_id != assertion.assertion_id and item.key() == assertion.key()
                     and item.assertion_id not in self._superseded_by]
            if any(_digest(item.value) != _digest(assertion.value) for item in peers):
                return Applicability(assertion.assertion_id, result.as_of_seq, ApplicabilityState.CONFLICT,
                                     result.availability, result.coverage, result.coverage_ref,
                                     stale_reason="contradictory current assertion", invalidated_by=tuple(item.assertion_id for item in peers))
        return result


def _value_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "array"
    return "object"


def recall_id_to_ref(item_id: str, namespace: Namespace) -> RecallRef:
    """Adapt an existing RECALL ``item_id`` without touching RecallStore."""

    return RecallRef(namespace.workspace_id, item_id)


def sciledger_value_to_ref(value: object, namespace: Namespace) -> SciledgerRef:
    """Adapt ClaimSpec/ClaimState-shaped values or a claim ID to a C0 ref."""

    return SciledgerRef.from_value(value, workspace_id=namespace.workspace_id)


def assertion_from_host_fact(
    fact: HostFact,
    *,
    namespace: Namespace,
    admitted_seq: int,
    observed_at: str,
    host_epoch: str,
    observed_monotonic: float,
    recall_ids: Sequence[str] = (),
    predicate: str | None = None,
) -> AssertionEnvelope:
    """Adapt a verified ``HostFact`` into an observed catalog assertion.

    HostFact evidence IDs remain receipt references.  They are not silently
    reclassified as RECALL IDs; callers must explicitly provide ``recall_ids``
    when a receipt was also indexed by RECALL.
    """

    if not isinstance(fact, HostFact):
        raise ContractError("assertion_from_host_fact requires a HostFact instance")
    state = {
        "CURRENT": ApplicabilityState.CURRENT,
        "STALE": ApplicabilityState.STALE,
        "CONFLICT": ApplicabilityState.CONFLICT,
    }[fact.status]
    entity = EntityRef(
        entity_id="host:" + fact.subject,
        kind="observation",
        namespace=namespace,
        locator={"subject": fact.subject, "fact_id": fact.fact_id},
        generation=fact.revision or fact.dependency_fingerprint,
    )
    return AssertionEnvelope.create(
        revision=1,
        subject=entity,
        predicate=predicate or fact.predicate,
        value=fact.value,
        scope=ScopeRef(namespace, "host-fact", {"kind": fact.kind, "status": fact.status}),
        coverage=CoverageRef("hostfact:" + fact.fact_id, "host_fact", CoverageStatus.COMPLETE),
        provenance=Provenance(
            producer_id="kernel_host_facts",
            producer_version="stateframe-host-facts/1",
            observed_at=observed_at,
            admitted_seq=admitted_seq,
            receipt_refs=fact.evidence_ids,
            trust_domain="host",
        ),
        freshness=Freshness(observed_at, host_epoch, observed_monotonic,
                            source_revision=fact.revision, state=state),
        confidence=Confidence.OBSERVED,
        recall_refs=tuple(recall_id_to_ref(item, namespace) for item in recall_ids),
    )


host_fact_to_assertion = assertion_from_host_fact
recall_ref_from_id = recall_id_to_ref
sciledger_ref_from_value = sciledger_value_to_ref
ContentRef = ContentHandle
CoverageHandle = CoverageRef


__all__ = [
    "Applicability", "ApplicabilityState", "AssertionEnvelope", "Availability", "CatalogRegistry",
    "Confidence", "ContentHandle", "ContentRef", "ContractError", "CoverageHandle", "CoverageRef",
    "CoverageSpec", "CoverageStatus", "CoverageRegistry",
    "DependencyRef", "EntityKindSpec", "EntityRef", "EntityRegistry", "Freshness", "IntegrityError",
    "Namespace", "PredicateRegistry", "PredicateSpec", "Provenance", "RecallRef", "ReferenceError",
    "SciledgerRef", "ScopeRef", "UnknownFieldError", "UnknownVersionError", "assertion_from_host_fact",
    "host_fact_to_assertion", "recall_id_to_ref", "recall_ref_from_id", "sciledger_ref_from_value",
    "sciledger_value_to_ref",
]
