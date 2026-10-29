"""Evidence-bound setup recipe contracts and their small in-memory registry."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

from kernel_environment_types import (
    Coverage,
    EnvironmentModelError,
    Provenance,
    ResourceRef,
    stable_digest,
)


AdmissionState: TypeAlias = Literal["candidate", "admitted", "rejected"]
AttemptState: TypeAlias = Literal[
    "reserved", "dispatched", "running", "verifying", "satisfied",
    "failed", "cancelled", "outcome_unknown",
]
Idempotency: TypeAlias = Literal[
    "pure_probe", "convergent", "keyed", "non_idempotent", "unknown",
]

_ADMISSION = frozenset({"candidate", "admitted", "rejected"})
_ATTEMPT = frozenset({
    "reserved", "dispatched", "running", "verifying", "satisfied",
    "failed", "cancelled", "outcome_unknown",
})
_IDEMPOTENCY = frozenset({"pure_probe", "convergent", "keyed", "non_idempotent", "unknown"})


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EnvironmentModelError(f"{field} must be a non-empty string")
    return value.strip()


def _enum(value: object, field: str, allowed: frozenset[str]) -> str:
    text = _text(value, field)
    if text not in allowed:
        raise EnvironmentModelError(f"invalid {field}: {text}")
    return text


def _refs(values: Sequence[str], field: str) -> tuple[str, ...]:
    normalized = tuple(_text(value, field) for value in values)
    if len(set(normalized)) != len(normalized):
        raise EnvironmentModelError(f"{field} must not contain duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class SetupRecipe:
    recipe_id: str
    version: str
    environment_scope: str
    provenance: tuple[Provenance, ...]
    command_descriptor_ref: str
    input_refs: tuple[ResourceRef, ...] = ()
    output_resource_facets: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    ensures: tuple[str, ...] = ()
    effects: Mapping[str, tuple[str, ...]] = ()
    idempotency: Idempotency = "unknown"
    retry_reconcile_descriptor: str | None = None
    cost: Mapping[str, float] = ()
    timeout: float = 300.0
    cancellation_policy: str = "bounded"
    lifetime: str = "operation"
    admission_state: AdmissionState = "candidate"
    coverage: Coverage = "complete"
    source_digest: str | None = None

    def __post_init__(self) -> None:
        for field in ("recipe_id", "version", "environment_scope", "command_descriptor_ref", "cancellation_policy", "lifetime"):
            _text(getattr(self, field), field)
        if any(not isinstance(item, Provenance) for item in self.provenance):
            raise EnvironmentModelError("recipe provenance must contain Provenance values")
        _refs(self.output_resource_facets, "output_resource_facets")
        _refs(self.requires, "requires")
        _refs(self.ensures, "ensures")
        _enum(self.idempotency, "idempotency", _IDEMPOTENCY)
        _enum(self.admission_state, "admission_state", _ADMISSION)
        if self.coverage not in {"complete", "partial"}:
            raise EnvironmentModelError("recipe coverage must be complete or partial")
        if isinstance(self.timeout, bool) or self.timeout <= 0:
            raise EnvironmentModelError("recipe timeout must be positive")
        if self.retry_reconcile_descriptor is not None:
            _text(self.retry_reconcile_descriptor, "retry_reconcile_descriptor")
        if self.source_digest is not None:
            _text(self.source_digest, "source_digest")
        for key, value in dict(self.cost).items():
            _text(key, "cost key")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise EnvironmentModelError("recipe cost values must be non-negative numbers")

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict(include_digest=False))

    def admitted(self) -> "SetupRecipe":
        if self.admission_state == "rejected":
            raise EnvironmentModelError("a rejected recipe cannot be admitted")
        return replace(self, admission_state="admitted")

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload = {
            "schema": "stateframe-setup-recipe/1",
            "recipe_id": self.recipe_id,
            "version": self.version,
            "environment_scope": self.environment_scope,
            "provenance": [item.to_dict() for item in self.provenance],
            "command_descriptor_ref": self.command_descriptor_ref,
            "input_refs": [item.to_dict() if hasattr(item, "to_dict") else item for item in self.input_refs],
            "output_resource_facets": list(self.output_resource_facets),
            "requires": list(self.requires),
            "ensures": list(self.ensures),
            "effects": {str(key): list(value) for key, value in dict(self.effects).items()},
            "idempotency": self.idempotency,
            "retry_reconcile_descriptor": self.retry_reconcile_descriptor,
            "cost": dict(self.cost),
            "timeout": self.timeout,
            "cancellation_policy": self.cancellation_policy,
            "lifetime": self.lifetime,
            "admission_state": self.admission_state,
            "coverage": self.coverage,
            "source_digest": self.source_digest,
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class SetupAttempt:
    attempt_id: str
    recipe_digest: str
    idempotency_key: str
    scope: str
    execution_epoch: str
    captured_inputs: tuple[str, ...]
    reservations: tuple[str, ...]
    durable_intent_ref: str
    state: AttemptState = "reserved"
    execution_receipt_ref: str | None = None
    postcondition_receipt_refs: tuple[str, ...] = ()
    output_versions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("attempt_id", "recipe_digest", "idempotency_key", "scope", "execution_epoch", "durable_intent_ref"):
            _text(getattr(self, field), field)
        _enum(self.state, "attempt state", _ATTEMPT)
        for field in ("captured_inputs", "reservations", "postcondition_receipt_refs", "output_versions"):
            _refs(getattr(self, field), field)
        if self.execution_receipt_ref is not None:
            _text(self.execution_receipt_ref, "execution_receipt_ref")

    def transition(self, state: AttemptState, *, receipt_ref: str | None = None) -> "SetupAttempt":
        _enum(state, "attempt state", _ATTEMPT)
        if self.state in {"satisfied", "failed", "cancelled"}:
            raise EnvironmentModelError("terminal setup attempt cannot transition")
        if state == "satisfied" and receipt_ref is None and self.execution_receipt_ref is None:
            raise EnvironmentModelError("satisfied attempt requires an execution receipt")
        return replace(self, state=state, execution_receipt_ref=receipt_ref or self.execution_receipt_ref)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-setup-attempt/1",
            "attempt_id": self.attempt_id,
            "recipe_digest": self.recipe_digest,
            "idempotency_key": self.idempotency_key,
            "scope": self.scope,
            "execution_epoch": self.execution_epoch,
            "captured_inputs": list(self.captured_inputs),
            "reservations": list(self.reservations),
            "durable_intent_ref": self.durable_intent_ref,
            "state": self.state,
            "execution_receipt_ref": self.execution_receipt_ref,
            "postcondition_receipt_refs": list(self.postcondition_receipt_refs),
            "output_versions": list(self.output_versions),
        }


@dataclass(frozen=True, slots=True)
class RecipeCandidate:
    recipe: SetupRecipe
    source_ref: str
    source_digest: str
    source_span: tuple[int, int] | None = None
    unresolved: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.source_ref, "source_ref")
        _text(self.source_digest, "source_digest")
        _refs(self.unresolved, "unresolved")

    @property
    def admitted(self) -> bool:
        return self.recipe.admission_state == "admitted" and not self.unresolved


class RecipeRegistry:
    """Host-owned registry; discovery can add candidates but never execute them."""

    def __init__(self, recipes: Iterable[SetupRecipe] = ()) -> None:
        self._recipes: dict[str, SetupRecipe] = {}
        for recipe in recipes:
            self.register(recipe)

    def register(self, recipe: SetupRecipe) -> None:
        prior = self._recipes.get(recipe.recipe_id)
        if prior is not None and prior.digest != recipe.digest:
            raise EnvironmentModelError(f"recipe id already has a different digest: {recipe.recipe_id}")
        self._recipes[recipe.recipe_id] = recipe

    def get(self, recipe_id: str) -> SetupRecipe | None:
        return self._recipes.get(recipe_id)

    def all(self) -> tuple[SetupRecipe, ...]:
        return tuple(self._recipes[key] for key in sorted(self._recipes))

    def admit(self, recipe_id: str) -> SetupRecipe:
        recipe = self._recipes.get(recipe_id)
        if recipe is None:
            raise EnvironmentModelError(f"unknown recipe: {recipe_id}")
        admitted = recipe.admitted()
        self._recipes[recipe_id] = admitted
        return admitted

    def validate_acyclic(self) -> None:
        """Reject prerequisite cycles before a plan can be admitted."""
        by_requirement = {
            requirement: recipe.recipe_id
            for recipe in self._recipes.values()
            for requirement in recipe.ensures
        }
        graph = {
            recipe.recipe_id: {
                by_requirement[requirement]
                for requirement in recipe.requires
                if requirement in by_requirement and by_requirement[requirement] != recipe.recipe_id
            }
            for recipe in self._recipes.values()
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(recipe_id: str, path: tuple[str, ...]) -> None:
            if recipe_id in visiting:
                cycle = " -> ".join((*path, recipe_id))
                raise EnvironmentModelError(f"recipe prerequisite cycle: {cycle}")
            if recipe_id in visited:
                return
            visiting.add(recipe_id)
            for dependency in sorted(graph.get(recipe_id, ())):
                visit(dependency, (*path, recipe_id))
            visiting.remove(recipe_id)
            visited.add(recipe_id)

        for recipe_id in sorted(graph):
            visit(recipe_id, ())


EnvironmentRecipeRegistry = RecipeRegistry


__all__ = [
    "AdmissionState", "AttemptState", "EnvironmentRecipeRegistry", "Idempotency",
    "RecipeCandidate", "RecipeRegistry", "SetupAttempt", "SetupRecipe",
]
