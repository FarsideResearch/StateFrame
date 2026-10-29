"""Declarative language/runtime adapters and host-owned S3 force planning.

Adapters describe isolated setup and negotiated observation capabilities.  A
later host runner executes the plan; this module never guesses a direct call,
starts a process, contacts a provider, or substitutes a task-specific recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from kernel_observation_contract import CallableIdentity, ObservationStatus


class AdapterId(str, Enum):
    PYTHON_SUBPROCESS = "python-subprocess"
    GO_PACKAGE_TEST = "go-package-test"
    GO_BINARY = "go-binary"
    RUST_TEST = "rust-test"
    RUST_BINARY = "rust-binary"
    TYPESCRIPT_MODULE = "typescript-module-process"
    JAVASCRIPT_MODULE = "javascript-module-process"
    JAVA_FRESH_JVM = "java-fresh-jvm"
    JAVA_FRESH_CLASSLOADER = "java-fresh-class-loader"
    PHP_CLI = "php-cli"
    VUE_LIFECYCLE = "vue-lifecycle-fixture"


class PlanState(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"


# These are host capabilities, not task IDs.  A concrete adapter may negotiate
# them while the later runner supplies the real product-bound observation.
OBSERVATION_CAPABILITIES = (
    "construction_site_observation",
    "logical_event_trace",
    "fresh_process_module_cache",
    "ready_boundary_assignment",
    "independent_persistence_readback",
)


@dataclass(frozen=True)
class AdapterDescriptor:
    adapter_id: AdapterId
    language: str
    runtime: str
    capabilities: tuple[str, ...]
    setup_recipe_ids: tuple[str, ...]
    isolation: str


@dataclass(frozen=True)
class AdapterPlan:
    state: PlanState
    status: ObservationStatus
    adapter_id: str
    target: CallableIdentity | None
    fixture_id: str
    setup_recipe_ids: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    negotiated_capabilities: tuple[str, ...]
    dependency_ids: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "status": self.status.value,
            "adapter_id": self.adapter_id,
            "target": self.target.as_dict() if self.target else None,
            "fixture_id": self.fixture_id,
            "setup_recipe_ids": list(self.setup_recipe_ids),
            "required_capabilities": list(self.required_capabilities),
            "negotiated_capabilities": list(self.negotiated_capabilities),
            "dependency_ids": list(self.dependency_ids),
            "missing": list(self.missing),
            "reason": self.reason,
        }


def _default_descriptors() -> tuple[AdapterDescriptor, ...]:
    common = (
        "serialized_observation", "mock_external_services", "independent_readback",
        *OBSERVATION_CAPABILITIES,
    )
    return (
        AdapterDescriptor(AdapterId.PYTHON_SUBPROCESS, "python", "subprocess", common + ("controlled_clock",), ("fresh-process", "json-receipt"), "fresh-process"),
        AdapterDescriptor(AdapterId.GO_PACKAGE_TEST, "go", "package-test", common, ("go-test-process", "json-receipt"), "fresh-process"),
        AdapterDescriptor(AdapterId.GO_BINARY, "go", "binary", common, ("go-build", "go-binary-process", "json-receipt"), "fresh-process"),
        AdapterDescriptor(AdapterId.RUST_TEST, "rust", "test", common, ("cargo-test-process", "json-receipt"), "fresh-process"),
        AdapterDescriptor(AdapterId.RUST_BINARY, "rust", "binary", common, ("cargo-build", "rust-binary-process", "json-receipt"), "fresh-process"),
        AdapterDescriptor(AdapterId.TYPESCRIPT_MODULE, "typescript", "module-process", common, ("ts-module-process", "json-receipt"), "fresh-process"),
        AdapterDescriptor(AdapterId.JAVASCRIPT_MODULE, "javascript", "module-process", common, ("js-module-process", "json-receipt"), "fresh-process"),
        AdapterDescriptor(AdapterId.JAVA_FRESH_JVM, "java", "fresh-jvm", common, ("fresh-jvm", "json-receipt"), "fresh-jvm"),
        AdapterDescriptor(AdapterId.JAVA_FRESH_CLASSLOADER, "java", "fresh-class-loader", common, ("fresh-jvm", "fresh-class-loader", "json-receipt"), "fresh-class-loader"),
        AdapterDescriptor(AdapterId.PHP_CLI, "php", "cli", common, ("php-cli-process", "json-receipt"), "fresh-process"),
        AdapterDescriptor(AdapterId.VUE_LIFECYCLE, "vue", "lifecycle-fixture", common + ("controlled_timers",), ("vue-fixture", "controlled-timers", "json-receipt"), "isolated-fixture"),
    )


class AdapterCatalog:
    """Host-owned lookup with no task-specific routing."""

    def __init__(self, descriptors: Sequence[AdapterDescriptor] = ()) -> None:
        source = tuple(descriptors) or _default_descriptors()
        self._by_key = {(item.language, item.runtime): item for item in source}

    def resolve(self, language: str, runtime: str) -> AdapterDescriptor | None:
        return self._by_key.get((language.lower().strip(), runtime.lower().strip()))

    def prepare(
        self,
        *,
        target: CallableIdentity | None,
        fixture_id: str,
        dependency_ids: Sequence[str],
        language: str,
        runtime: str,
        required_capabilities: Sequence[str] = (),
    ) -> AdapterPlan:
        descriptor = self.resolve(language, runtime)
        required = tuple(dict.fromkeys(str(value) for value in required_capabilities if str(value).strip()))
        dependencies = tuple(dict.fromkeys(str(value) for value in dependency_ids if str(value).strip()))
        if descriptor is None:
            return AdapterPlan(
                PlanState.BLOCKED,
                ObservationStatus.UNRUNNABLE,
                f"unsupported:{language}/{runtime}",
                target,
                fixture_id,
                (),
                required,
                (),
                dependencies,
                ("adapter",),
                "unsupported language/runtime adapter",
            )
        missing: list[str] = []
        if target is None:
            missing.append("target")
        if not isinstance(fixture_id, str) or not fixture_id.strip():
            missing.append("fixture")
        if not dependencies:
            missing.append("dependency")
        if target is not None and target.language.lower() != descriptor.language:
            missing.append("target-language")
        if target is not None and target.runtime.lower() != descriptor.runtime:
            missing.append("target-runtime")
        missing.extend(sorted(set(required) - set(descriptor.capabilities)))
        if missing:
            return AdapterPlan(
                PlanState.BLOCKED,
                ObservationStatus.UNRUNNABLE,
                descriptor.adapter_id.value,
                target,
                fixture_id,
                (),
                required,
                (),
                dependencies,
                tuple(missing),
                "target, fixture, dependency, or observation capability unavailable",
            )
        return AdapterPlan(
            PlanState.READY,
            ObservationStatus.UNCOVERED,
            descriptor.adapter_id.value,
            target,
            fixture_id,
            descriptor.setup_recipe_ids,
            required,
            tuple(sorted(set(required) & set(descriptor.capabilities))),
            dependencies,
            reason=f"{descriptor.isolation} adapter setup negotiated; execution evidence is pending",
        )


@dataclass(frozen=True)
class HostForceResult:
    """Typed result for a future S3 bridge; blocked is a first-class result."""

    state: PlanState
    status: ObservationStatus
    plan: AdapterPlan | None
    missing: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "status": self.status.value,
            "plan": self.plan.as_dict() if self.plan else None,
            "missing": list(self.missing),
            "reason": self.reason,
        }


def build_host_owned_force(
    *,
    target: CallableIdentity | None,
    fixture_id: str,
    dependency_ids: Sequence[str],
    language: str,
    runtime: str,
    required_capabilities: Sequence[str] = ("serialized_observation",),
    catalog: AdapterCatalog | None = None,
) -> HostForceResult:
    """Negotiate a typed adapter plan; never fall back to an untyped call."""

    plan = (catalog or AdapterCatalog()).prepare(
        target=target,
        fixture_id=fixture_id,
        dependency_ids=dependency_ids,
        language=language,
        runtime=runtime,
        required_capabilities=required_capabilities,
    )
    if plan.state is PlanState.BLOCKED:
        return HostForceResult(PlanState.BLOCKED, ObservationStatus.UNRUNNABLE, plan, plan.missing, plan.reason)
    return HostForceResult(PlanState.READY, ObservationStatus.UNCOVERED, plan, reason="host-owned adapter plan is ready; execution evidence is pending")


def prepare_observation_handoff(
    *,
    target: CallableIdentity | None,
    fixture_id: str,
    dependency_ids: Sequence[str],
    language: str,
    runtime: str,
    capabilities: Sequence[str],
    catalog: AdapterCatalog | None = None,
) -> HostForceResult:
    """Expose a generic, language-neutral handoff for real host execution.

    This only negotiates the execution contract.  It does not fabricate a
    client, clock, event, persistence read, or task-specific observation.
    """
    required = tuple(dict.fromkeys((*capabilities, "serialized_observation")))
    return build_host_owned_force(
        target=target,
        fixture_id=fixture_id,
        dependency_ids=dependency_ids,
        language=language,
        runtime=runtime,
        required_capabilities=required,
        catalog=catalog,
    )


__all__ = [
    "AdapterCatalog", "AdapterDescriptor", "AdapterId", "AdapterPlan", "HostForceResult", "PlanState",
    "OBSERVATION_CAPABILITIES", "build_host_owned_force", "prepare_observation_handoff",
]
