"""Predicate evaluation and check applicability over versioned environment facts."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

from kernel_environment_types import (
    EnvironmentInputVector,
    PredicateEvaluation,
    RequirementBinding,
    ResourceFact,
    ResourcePredicate,
    ResourceRef,
    stable_digest,
)


Applicability: TypeAlias = Literal["current", "stale", "unknown", "mixed_inputs"]


def _ref_key(value: ResourceRef) -> str:
    return value.identity if hasattr(value, "identity") else str(value)


def _fact_lookup(
    predicate: ResourcePredicate,
    facts: Mapping[object, ResourceFact] | Iterable[ResourceFact],
) -> ResourceFact | None:
    if isinstance(facts, Mapping):
        direct = facts.get(predicate.predicate_id)
        if isinstance(direct, ResourceFact):
            return direct
        direct = facts.get((_ref_key(predicate.resource_ref), predicate.facet))
        if isinstance(direct, ResourceFact):
            return direct
        direct = facts.get((predicate.resource_ref, predicate.facet))
        if isinstance(direct, ResourceFact):
            return direct
        values = facts.values()
    else:
        values = facts
    for fact in values:
        if isinstance(fact, ResourceFact) and fact.key() == (_ref_key(predicate.resource_ref), predicate.facet):
            return fact
    return None


def _actual_value(fact: ResourceFact) -> object:
    return fact.value if fact.value is not None else (fact.version.digest if fact.version else None)


def _expected_value(predicate: ResourcePredicate) -> object:
    return predicate.expected_ref.digest if predicate.expected_ref is not None else predicate.expected


def _matches(predicate: ResourcePredicate, actual: object) -> bool:
    expected = _expected_value(predicate)
    if predicate.operator == "equals":
        return actual == expected
    if predicate.operator == "not_equals":
        return actual != expected
    if predicate.operator == "contains":
        try:
            return expected in actual  # type: ignore[operator]
        except (TypeError, AttributeError):
            return False
    if predicate.operator == "set_equals":
        try:
            return set(actual) == set(expected)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            return False
    return False


def evaluate_predicate(
    predicate: ResourcePredicate,
    facts: Mapping[object, ResourceFact] | Iterable[ResourceFact],
) -> PredicateEvaluation:
    """Evaluate one predicate without probing or mutating the host."""
    fact = _fact_lookup(predicate, facts)
    if fact is None:
        return PredicateEvaluation(predicate.predicate_id, "unknown", "no observation")
    if fact.availability != "present":
        return PredicateEvaluation(predicate.predicate_id, "unknown", f"availability={fact.availability}", fact.evidence_ref, fact.version)
    if fact.coverage != predicate.coverage_required:
        return PredicateEvaluation(predicate.predicate_id, "unknown", f"coverage={fact.coverage}", fact.evidence_ref, fact.version)
    if fact.verification != "verified":
        return PredicateEvaluation(predicate.predicate_id, "unknown", f"verification={fact.verification}", fact.evidence_ref, fact.version)
    if not fact.evidence_ref:
        return PredicateEvaluation(predicate.predicate_id, "unknown", "missing evidence", None, fact.version)
    if predicate.freshness_policy in {"current", "live_at_acceptance"} and fact.freshness != "fresh":
        return PredicateEvaluation(predicate.predicate_id, "stale", f"freshness={fact.freshness}", fact.evidence_ref, fact.version)
    if not _matches(predicate, _actual_value(fact)):
        return PredicateEvaluation(predicate.predicate_id, "stale", "version mismatch", fact.evidence_ref, fact.version)
    return PredicateEvaluation(predicate.predicate_id, "current", "verified match", fact.evidence_ref, fact.version)


@dataclass(frozen=True, slots=True)
class ApplicabilityResult:
    status: Applicability
    check_id: str
    evaluations: tuple[PredicateEvaluation, ...]
    current_vector: EnvironmentInputVector | None
    reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.status == "current"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-check-applicability/1",
            "status": self.status,
            "check_id": self.check_id,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "current_vector": self.current_vector.to_dict() if self.current_vector else None,
            "reasons": list(self.reasons),
        }


def descriptor_digest(descriptor: object) -> str:
    raw_predicates = getattr(descriptor, "environment_predicates", ())
    predicates = tuple(
        item.to_dict() if hasattr(item, "to_dict") else str(item)
        for item in raw_predicates
    )
    payload = {
        "check_id": getattr(descriptor, "check_id", None),
        "runner": getattr(descriptor, "runner", None),
        "argv": tuple(getattr(descriptor, "argv", ())),
        "cwd": getattr(descriptor, "cwd", None),
        "kind": getattr(descriptor, "kind", None),
        "adapter": getattr(descriptor, "adapter", None),
        "environment_predicates": predicates,
    }
    return stable_digest(payload)


def capture_input_vector(
    binding: RequirementBinding,
    predicates: Mapping[str, ResourcePredicate],
    facts: Mapping[object, ResourceFact] | Iterable[ResourceFact],
    *, descriptor: object | None = None, fence_token: str = "fence:unknown",
) -> EnvironmentInputVector:
    evaluations = tuple(evaluate_predicate(predicates[ref], facts) for ref in binding.predicate_refs if ref in predicates)
    versions = tuple(item.version for item in evaluations if item.version is not None)
    coverage = "complete" if len(evaluations) == len(binding.predicate_refs) else "partial"
    if any(item.status != "current" for item in evaluations):
        coverage = "partial"
    return EnvironmentInputVector(
        version_refs=tuple(versions),
        predicate_refs=binding.predicate_refs,
        observation_refs=tuple(item.evidence_ref for item in evaluations if item.evidence_ref),
        input_coverage=coverage,
        descriptor_digest=descriptor_digest(descriptor) if descriptor is not None else "descriptor:unknown",
        fence_token=fence_token,
    )


class EnvironmentLedger:
    """A projection of facts owned by EnvironmentState, suitable for checks."""

    def __init__(self) -> None:
        self._facts: dict[tuple[str, str], ResourceFact] = {}
        self._predicates: dict[str, ResourcePredicate] = {}
        self._bindings: dict[str, RequirementBinding] = {}

    def register_predicate(self, predicate: ResourcePredicate) -> None:
        self._predicates[predicate.predicate_id] = predicate

    def bind_check(self, binding: RequirementBinding) -> None:
        missing = [item for item in binding.predicate_refs if item not in self._predicates]
        if missing:
            raise ValueError("unknown predicate refs: " + ",".join(missing))
        self._bindings[binding.check_id] = binding

    def put_fact(self, fact: ResourceFact) -> None:
        key = fact.key()
        prior = self._facts.get(key)
        if prior is not None and prior.version is not None and fact.version is not None:
            if fact.version.generation < prior.version.generation:
                raise ValueError("resource generation regressed")
        self._facts[key] = fact

    def fact(self, resource_ref: ResourceRef, facet: str) -> ResourceFact | None:
        return self._facts.get((_ref_key(resource_ref), facet))

    def facts(self) -> tuple[ResourceFact, ...]:
        return tuple(self._facts[key] for key in sorted(self._facts))

    def predicates(self) -> tuple[ResourcePredicate, ...]:
        return tuple(self._predicates[key] for key in sorted(self._predicates))

    def binding(self, check_id: str) -> RequirementBinding | None:
        return self._bindings.get(check_id)

    def capture(self, check_id: str, *, descriptor: object | None = None,
                fence_token: str = "fence:unknown") -> EnvironmentInputVector:
        binding = self._bindings.get(check_id)
        if binding is None:
            raise ValueError("unknown check binding: " + check_id)
        return capture_input_vector(binding, self._predicates, self._facts, descriptor=descriptor, fence_token=fence_token)

    def evaluate(self, check_id: str) -> tuple[PredicateEvaluation, ...]:
        binding = self._bindings.get(check_id)
        if binding is None:
            raise ValueError("unknown check binding: " + check_id)
        return tuple(evaluate_predicate(self._predicates[item], self._facts) for item in binding.predicate_refs)

    def applicability(
        self, check_id: str, *, receipt_vector: EnvironmentInputVector | None = None,
        descriptor: object | None = None, fence_token: str = "fence:unknown",
    ) -> ApplicabilityResult:
        binding = self._bindings.get(check_id)
        if binding is None:
            return ApplicabilityResult("unknown", check_id, (), None, ("unknown check binding",))
        evaluations = self.evaluate(check_id)
        current_vector = self.capture(check_id, descriptor=descriptor, fence_token=fence_token)
        reasons = tuple(f"{item.predicate_id}:{item.reason}" for item in evaluations if item.status != "current")
        if any(item.status == "unknown" for item in evaluations):
            status: Applicability = "unknown"
        elif any(item.status == "stale" for item in evaluations):
            status = "stale"
        elif receipt_vector is None:
            status = "unknown"
            reasons = (*reasons, "receipt has no captured environment input vector")
        elif receipt_vector.input_coverage != "complete" or current_vector.input_coverage != "complete":
            status = "mixed_inputs"
            reasons = (*reasons, "required environment input coverage is incomplete")
        elif receipt_vector.version_refs != current_vector.version_refs or receipt_vector.digest != current_vector.digest:
            status = "stale"
            reasons = (*reasons, "captured environment input vector differs")
        else:
            status = "current"
        if binding.mode == "live_at_acceptance" and any(item.status != "current" for item in evaluations):
            status = "unknown" if any(item.status == "unknown" for item in evaluations) else "stale"
        return ApplicabilityResult(status, check_id, evaluations, current_vector, reasons)


def applicability_for_receipt(
    receipt: object, ledger: EnvironmentLedger, *, descriptor: object | None = None,
    fence_token: str = "fence:unknown",
) -> ApplicabilityResult:
    check_id = str(getattr(receipt, "check_id", ""))
    vector = getattr(receipt, "environment_input_vector", None)
    if not isinstance(vector, EnvironmentInputVector):
        raw = getattr(receipt, "as_dict", lambda: {})()
        if isinstance(raw, Mapping):
            vector = raw.get("environment_input_vector")
    return ledger.applicability(check_id, receipt_vector=vector, descriptor=descriptor, fence_token=fence_token)


def annotate_receipt(receipt: object, result: ApplicabilityResult) -> object:
    """Attach applicability to a dataclass receipt without changing execution status."""
    if hasattr(receipt, "applicability") and hasattr(receipt, "environment_input_vector"):
        return replace(receipt, applicability=result.status)
    return receipt


__all__ = [
    "Applicability", "ApplicabilityResult", "EnvironmentLedger", "annotate_receipt",
    "applicability_for_receipt", "capture_input_vector", "descriptor_digest",
    "evaluate_predicate",
]
