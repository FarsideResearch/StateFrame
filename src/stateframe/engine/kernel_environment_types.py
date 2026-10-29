"""Typed, versioned facts used by the v29 environment/check boundary.

This module contains value objects only.  It deliberately does not probe the
host, execute recipes, or decide which owner may mutate a resource.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, TypeAlias


SCHEMA = "stateframe-environment-model/1"

ResourceKind: TypeAlias = Literal[
    "runtime", "dependency_set", "process", "service", "database",
    "generated_artifact", "configuration", "fixture_set",
]
Verification: TypeAlias = Literal["verified", "declared", "unverified", "contradicted"]
Freshness: TypeAlias = Literal["fresh", "stale", "unknown"]
Availability: TypeAlias = Literal[
    "present", "absent", "unknown", "unsupported", "denied", "error",
]
Coverage: TypeAlias = Literal["complete", "partial"]
PredicateOperator: TypeAlias = Literal["equals", "not_equals", "contains", "set_equals"]
FreshnessPolicy: TypeAlias = Literal["current", "historical", "live_at_acceptance"]
BindingMode: TypeAlias = Literal["historical_execution", "live_at_acceptance"]
PredicateStatus: TypeAlias = Literal["current", "stale", "unknown"]

_RESOURCE_KINDS = frozenset({
    "runtime", "dependency_set", "process", "service", "database",
    "generated_artifact", "configuration", "fixture_set",
})
_VERIFICATIONS = frozenset({"verified", "declared", "unverified", "contradicted"})
_FRESHNESS = frozenset({"fresh", "stale", "unknown"})
_AVAILABILITY = frozenset({"present", "absent", "unknown", "unsupported", "denied", "error"})
_COVERAGE = frozenset({"complete", "partial"})
_OPERATORS = frozenset({"equals", "not_equals", "contains", "set_equals"})
_FRESHNESS_POLICIES = frozenset({"current", "historical", "live_at_acceptance"})
_BINDING_MODES = frozenset({"historical_execution", "live_at_acceptance"})


class EnvironmentModelError(ValueError):
    """Raised when a host-provided environment value is malformed."""


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EnvironmentModelError(f"{field} must be a non-empty string")
    return value.strip()


def _require_enum(value: object, field: str, allowed: frozenset[str]) -> str:
    text = _require_text(value, field)
    if text not in allowed:
        raise EnvironmentModelError(f"invalid {field}: {text}")
    return text


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def stable_digest(value: object) -> str:
    """Return a public, non-secret identity for a typed value."""
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _public_value(facet: str, value: object) -> object:
    if any(token in facet.casefold() for token in ("secret", "password", "token", "credential", "dsn")):
        return "<redacted>"
    return value


ResourceRef: TypeAlias = "ResourceKey | str"


def _resource_dict(value: ResourceRef) -> object:
    return value.to_dict() if isinstance(value, ResourceKey) else _require_text(value, "resource_ref")


@dataclass(frozen=True, slots=True)
class ResourceKey:
    tenant_scope: str
    workspace_id: str
    environment_id: str
    namespace_id: str
    resource_kind: ResourceKind
    opaque_resource_id: str

    def __post_init__(self) -> None:
        for field in ("tenant_scope", "workspace_id", "environment_id", "namespace_id", "opaque_resource_id"):
            _require_text(getattr(self, field), field)
        _require_enum(self.resource_kind, "resource_kind", _RESOURCE_KINDS)

    @property
    def identity(self) -> str:
        return stable_digest(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "tenant_scope": self.tenant_scope,
            "workspace_id": self.workspace_id,
            "environment_id": self.environment_id,
            "namespace_id": self.namespace_id,
            "resource_kind": self.resource_kind,
            "opaque_resource_id": self.opaque_resource_id,
        }


@dataclass(frozen=True, slots=True)
class VersionRef:
    owner: str
    resource_ref: ResourceRef
    generation: int
    facet: str
    digest: str

    def __post_init__(self) -> None:
        _require_text(self.owner, "owner")
        if not isinstance(self.resource_ref, (ResourceKey, str)):
            raise EnvironmentModelError("resource_ref must be a ResourceKey or string")
        if isinstance(self.resource_ref, str):
            _require_text(self.resource_ref, "resource_ref")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 1:
            raise EnvironmentModelError("generation must be a positive integer")
        _require_text(self.facet, "facet")
        _require_text(self.digest, "digest")

    def sort_key(self) -> tuple[str, str, int, str, str]:
        resource = self.resource_ref.identity if isinstance(self.resource_ref, ResourceKey) else self.resource_ref
        return (self.owner, resource, self.generation, self.facet, self.digest)

    def to_dict(self) -> dict[str, object]:
        return {
            "owner": self.owner,
            "resource_ref": _resource_dict(self.resource_ref),
            "generation": self.generation,
            "facet": self.facet,
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class Provenance:
    kind: Literal["host_observed", "repo_declared", "user_declared", "model_proposed", "observed_command"]
    producer_id: str
    producer_version: str
    source_ref: str
    source_digest: str
    span: tuple[int, int] | None = None
    receipt_ref: str | None = None

    def __post_init__(self) -> None:
        _require_enum(self.kind, "provenance kind", frozenset({
            "host_observed", "repo_declared", "user_declared", "model_proposed", "observed_command",
        }))
        for field in ("producer_id", "producer_version", "source_ref", "source_digest"):
            _require_text(getattr(self, field), field)
        if self.span is not None:
            if (len(self.span) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in self.span)
                    or self.span[1] < self.span[0]):
                raise EnvironmentModelError("provenance span must be an ordered non-negative pair")
        if self.receipt_ref is not None:
            _require_text(self.receipt_ref, "receipt_ref")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "source_ref": self.source_ref,
            "source_digest": self.source_digest,
            "span": list(self.span) if self.span is not None else None,
            "receipt_ref": self.receipt_ref,
        }


@dataclass(frozen=True, slots=True)
class ResourceObservation:
    resource_ref: ResourceRef
    host_epoch: str
    producer_seq: int
    sample_window: tuple[float, float]
    monotonic_expiry: float | None
    dependency_refs: tuple[VersionRef, ...]
    provenance: Provenance
    verification: Verification
    freshness: Freshness
    availability: Availability
    coverage: Coverage
    version: VersionRef | None = None
    value: object = None
    evidence_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resource_ref, (ResourceKey, str)):
            raise EnvironmentModelError("resource_ref must be a ResourceKey or string")
        if isinstance(self.resource_ref, str):
            _require_text(self.resource_ref, "resource_ref")
        if self.version is not None and not isinstance(self.version, VersionRef):
            raise EnvironmentModelError("version must be a VersionRef")
        if any(not isinstance(item, VersionRef) for item in self.dependency_refs):
            raise EnvironmentModelError("dependency_refs must contain VersionRef values")
        _require_text(self.host_epoch, "host_epoch")
        if isinstance(self.producer_seq, bool) or not isinstance(self.producer_seq, int) or self.producer_seq < 0:
            raise EnvironmentModelError("producer_seq must be a non-negative integer")
        if (len(self.sample_window) != 2 or any(not isinstance(item, (int, float)) or isinstance(item, bool)
                                                or not math.isfinite(float(item)) for item in self.sample_window)
                or self.sample_window[1] < self.sample_window[0]):
            raise EnvironmentModelError("sample_window must contain finite ordered timestamps")
        if self.monotonic_expiry is not None and (
                not isinstance(self.monotonic_expiry, (int, float))
                or isinstance(self.monotonic_expiry, bool)
                or not math.isfinite(float(self.monotonic_expiry))):
            raise EnvironmentModelError("monotonic_expiry must be finite")
        _require_enum(self.verification, "verification", _VERIFICATIONS)
        _require_enum(self.freshness, "freshness", _FRESHNESS)
        _require_enum(self.availability, "availability", _AVAILABILITY)
        _require_enum(self.coverage, "coverage", _COVERAGE)
        if self.evidence_ref is not None:
            _require_text(self.evidence_ref, "evidence_ref")

    @property
    def identity(self) -> str:
        return stable_digest({
            "resource_ref": _resource_dict(self.resource_ref),
            "host_epoch": self.host_epoch,
            "producer_seq": self.producer_seq,
            "version": self.version.to_dict() if self.version else None,
            "value": self.value,
        })

    def to_dict(self) -> dict[str, object]:
        return {
            "resource_ref": _resource_dict(self.resource_ref),
            "host_epoch": self.host_epoch,
            "producer_seq": self.producer_seq,
            "sample_window": list(self.sample_window),
            "monotonic_expiry": self.monotonic_expiry,
            "dependency_refs": [item.to_dict() for item in self.dependency_refs],
            "provenance": self.provenance.to_dict(),
            "verification": self.verification,
            "freshness": self.freshness,
            "availability": self.availability,
            "coverage": self.coverage,
            "version": self.version.to_dict() if self.version else None,
            "value": self.value,
            "evidence_ref": self.evidence_ref,
        }


@dataclass(frozen=True, slots=True)
class ResourcePredicate:
    predicate_id: str
    resource_ref: ResourceRef
    facet: str
    operator: PredicateOperator
    expected: object = None
    expected_ref: VersionRef | None = None
    freshness_policy: FreshnessPolicy = "current"
    coverage_required: Coverage = "complete"
    verifier_descriptor_ref: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.predicate_id, "predicate_id")
        if not isinstance(self.resource_ref, (ResourceKey, str)):
            raise EnvironmentModelError("resource_ref must be a ResourceKey or string")
        if isinstance(self.resource_ref, str):
            _require_text(self.resource_ref, "resource_ref")
        _require_text(self.facet, "facet")
        _require_enum(self.operator, "operator", _OPERATORS)
        _require_enum(self.freshness_policy, "freshness_policy", _FRESHNESS_POLICIES)
        _require_enum(self.coverage_required, "coverage_required", _COVERAGE)
        if self.expected_ref is not None and not isinstance(self.expected_ref, VersionRef):
            raise EnvironmentModelError("expected_ref must be a VersionRef")
        if self.verifier_descriptor_ref is not None:
            _require_text(self.verifier_descriptor_ref, "verifier_descriptor_ref")
        if self.expected_ref is None and self.expected is None:
            raise EnvironmentModelError("predicate requires expected or expected_ref")

    def to_dict(self) -> dict[str, object]:
        return {
            "predicate_id": self.predicate_id,
            "resource_ref": _resource_dict(self.resource_ref),
            "facet": self.facet,
            "operator": self.operator,
            "expected": self.expected,
            "expected_ref": self.expected_ref.to_dict() if self.expected_ref else None,
            "freshness_policy": self.freshness_policy,
            "coverage_required": self.coverage_required,
            "verifier_descriptor_ref": self.verifier_descriptor_ref,
        }


@dataclass(frozen=True, slots=True)
class RequirementBinding:
    check_id: str
    contract_version: str
    predicate_refs: tuple[str, ...]
    mode: BindingMode = "historical_execution"

    def __post_init__(self) -> None:
        _require_text(self.check_id, "check_id")
        _require_text(self.contract_version, "contract_version")
        if not self.predicate_refs or any(not isinstance(item, str) or not item.strip() for item in self.predicate_refs):
            raise EnvironmentModelError("requirement binding needs predicate refs")
        if len(set(self.predicate_refs)) != len(self.predicate_refs):
            raise EnvironmentModelError("requirement binding predicate refs must be unique")
        _require_enum(self.mode, "binding mode", _BINDING_MODES)

    @property
    def version(self) -> str:
        return self.contract_version

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "contract_version": self.contract_version,
            "predicate_refs": list(self.predicate_refs),
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentInputVector:
    version_refs: tuple[VersionRef, ...]
    predicate_refs: tuple[str, ...]
    observation_refs: tuple[str, ...]
    input_coverage: Coverage
    descriptor_digest: str
    fence_token: str

    def __post_init__(self) -> None:
        _require_enum(self.input_coverage, "input_coverage", _COVERAGE)
        _require_text(self.descriptor_digest, "descriptor_digest")
        _require_text(self.fence_token, "fence_token")
        if any(not isinstance(item, VersionRef) for item in self.version_refs):
            raise EnvironmentModelError("version_refs must contain VersionRef values")
        if any(not isinstance(item, str) or not item.strip() for item in (*self.predicate_refs, *self.observation_refs)):
            raise EnvironmentModelError("input refs must be non-empty strings")

    @property
    def digest(self) -> str:
        return stable_digest(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "version_refs": [item.to_dict() for item in sorted(self.version_refs, key=VersionRef.sort_key)],
            "predicate_refs": list(self.predicate_refs),
            "observation_refs": list(self.observation_refs),
            "input_coverage": self.input_coverage,
            "descriptor_digest": self.descriptor_digest,
            "fence_token": self.fence_token,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class ResourceFact:
    resource_ref: ResourceRef
    facet: str
    version: VersionRef | None
    value: object
    availability: Availability = "present"
    freshness: Freshness = "fresh"
    coverage: Coverage = "complete"
    verification: Verification = "verified"
    evidence_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resource_ref, (ResourceKey, str)):
            raise EnvironmentModelError("resource_ref must be a ResourceKey or string")
        if isinstance(self.resource_ref, str):
            _require_text(self.resource_ref, "resource_ref")
        _require_text(self.facet, "facet")
        if self.version is not None and not isinstance(self.version, VersionRef):
            raise EnvironmentModelError("version must be a VersionRef")
        for field, allowed in (("availability", _AVAILABILITY), ("freshness", _FRESHNESS),
                               ("coverage", _COVERAGE), ("verification", _VERIFICATIONS)):
            _require_enum(getattr(self, field), field, allowed)
        if self.evidence_ref is not None:
            _require_text(self.evidence_ref, "evidence_ref")

    def key(self) -> tuple[str, str]:
        ref = self.resource_ref.identity if isinstance(self.resource_ref, ResourceKey) else self.resource_ref
        return ref, self.facet

    def to_dict(self) -> dict[str, object]:
        return {
            "resource_ref": _resource_dict(self.resource_ref),
            "facet": self.facet,
            "version": self.version.to_dict() if self.version else None,
            "value": _public_value(self.facet, self.value),
            "availability": self.availability,
            "freshness": self.freshness,
            "coverage": self.coverage,
            "verification": self.verification,
            "evidence_ref": self.evidence_ref,
        }


@dataclass(frozen=True, slots=True)
class PredicateEvaluation:
    predicate_id: str
    status: PredicateStatus
    reason: str
    evidence_ref: str | None = None
    version: VersionRef | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "predicate_id": self.predicate_id,
            "status": self.status,
            "reason": self.reason,
            "evidence_ref": self.evidence_ref,
            "version": self.version.to_dict() if self.version else None,
        }


# Descriptive aliases keep the public phase-1 vocabulary discoverable without
# duplicating schemas.
EnvironmentObservation = ResourceObservation
PredicateResult = PredicateEvaluation


__all__ = [
    "Availability", "BindingMode", "Coverage", "EnvironmentInputVector",
    "EnvironmentModelError", "EnvironmentObservation", "Freshness", "FreshnessPolicy",
    "PredicateEvaluation", "PredicateOperator", "PredicateResult", "PredicateStatus",
    "Provenance", "RequirementBinding", "ResourceFact", "ResourceKey", "ResourceKind",
    "ResourceObservation", "ResourcePredicate", "ResourceRef", "SCHEMA", "Verification",
    "VersionRef", "stable_digest",
]
