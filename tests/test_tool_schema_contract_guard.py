"""Keep every model-visible tool shape accepted by its step contract."""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from unittest.mock import patch


def _schema_variants(schema: Mapping[str, object]) -> list[tuple[frozenset[str], frozenset[str], frozenset[str]]]:
    """Return top-level property/required shapes for the tool schema's alternatives."""
    raw_properties = schema.get("properties")
    properties = frozenset(str(name) for name in raw_properties) if isinstance(raw_properties, Mapping) else frozenset()
    raw_required = schema.get("required")
    required = (frozenset(str(name) for name in raw_required)
                if isinstance(raw_required, Sequence) and not isinstance(raw_required, (str, bytes, bytearray))
                else frozenset())
    variants = [(properties, required, frozenset())]

    for keyword in ("oneOf", "anyOf"):
        raw_branches = schema.get(keyword)
        if not isinstance(raw_branches, Sequence) or isinstance(raw_branches, (str, bytes, bytearray)):
            continue
        choices = [branch for branch in raw_branches if isinstance(branch, Mapping)]
        if not choices:
            continue
        expanded = []
        for base_properties, base_required, distinguishing in variants:
            for branch in choices:
                for child_properties, child_required, child_distinguishing in _schema_variants(branch):
                    expanded.append((
                        base_properties | child_properties,
                        base_required | child_required,
                        distinguishing | child_distinguishing | (child_properties - properties),
                    ))
        variants = expanded

    raw_all = schema.get("allOf")
    if isinstance(raw_all, Sequence) and not isinstance(raw_all, (str, bytes, bytearray)):
        for branch in raw_all:
            if not isinstance(branch, Mapping):
                continue
            expanded = []
            for base_properties, base_required, distinguishing in variants:
                for child_properties, child_required, child_distinguishing in _schema_variants(branch):
                    expanded.append((
                        base_properties | child_properties,
                        base_required | child_required,
                        distinguishing | child_distinguishing | (child_properties - properties),
                    ))
            variants = expanded
    return variants


def _schema_violations(
    name: str, advertised: Mapping[str, object], contract: Mapping[str, object],
) -> list[str]:
    contract_variants = _schema_variants(contract)
    accepted_properties = frozenset().union(*(properties for properties, _required, _distinct in contract_variants))
    root_required = _schema_variants({"required": advertised.get("required", ())})[0][1]
    violations: set[str] = set()
    for properties, required, distinguishing in _schema_variants(advertised):
        violations.update(f"{name}.{property_name}: property rejected by step contract"
                          for property_name in properties - accepted_properties)
        if contract_variants and not any(contract_required <= required
                                         for _contract_properties, contract_required, _contract_distinct
                                         in contract_variants):
            missing = frozenset().union(*(contract_required - required
                                          for _contract_properties, contract_required, _contract_distinct
                                          in contract_variants))
            label = min(required - root_required or distinguishing or properties or missing,
                        default="<required>")
            violations.add(
                f"{name}.{label}: advertised shape cannot satisfy required step-contract fields "
                + ", ".join(sorted(missing))
            )
    return sorted(violations)


def _tool_parts(tool: object) -> tuple[str, Mapping[str, object]] | None:
    if not isinstance(tool, Mapping):
        return None
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return None
    name = str(function.get("name") or "")
    parameters = function.get("parameters")
    if not name or not isinstance(parameters, Mapping):
        return None
    return name, parameters


def _renderable_tool_schemas() -> list[tuple[dict[str, object], dict[str, object]]]:
    """Collect product menus, all shipped profiles, and their schema-bearing branches."""
    import kernel_completion_owner as completion_owner
    import kernel_agent_tools
    from kernel_agent_schemas import BUILTIN_SCHEMAS
    from kernel_coordinated_change_ledger import ATOMIC_APPLY_ENV
    from kernel_git_commit import registered_schemas
    from kernel_shell_tool import SHELL_SCHEMA
    from probes import react_repair

    profile_directory = Path(__file__).parents[1] / "src" / "stateframe" / "profiles"
    profile_paths = sorted(profile_directory.glob("*.json"))
    assert profile_paths, f"no shipped profiles found under {profile_directory}"

    schemas: list[tuple[dict[str, object], dict[str, object]]] = []

    dynamic_tools = [*BUILTIN_SCHEMAS, *registered_schemas()]

    class ProfileToolRegistry:
        """The run-scoped source uses each registered provider schema as its contract."""

        custom: dict[str, object] = {}
        custom_schemas: list[dict[str, object]] = []

        def __init__(self) -> None:
            self.names = frozenset(
                str(tool["function"]["name"])
                for tool in dynamic_tools
                if isinstance(tool.get("function"), Mapping)
            )

        def offered(self, name: str) -> bool:
            return name in self.names

        def schemas(self) -> list[dict[str, object]]:
            return copy.deepcopy(dynamic_tools)

    def add_schema(tool: object, contract: Mapping[str, object] | None = None) -> None:
        parts = _tool_parts(tool)
        if parts is None:
            return
        _name, parameters = parts
        accepted = contract if contract is not None else completion_owner._spec_for_tool(tool).schema
        schemas.append((copy.deepcopy(dict(tool)), copy.deepcopy(dict(accepted))))

    for tool in dynamic_tools:
        parts = _tool_parts(tool)
        if parts is not None:
            add_schema(tool, parts[1])
    add_schema(SHELL_SCHEMA)
    for spec in completion_owner.TOOL_SPECS.values():
        add_schema(spec.advertised(), spec.schema)
    canonicalize = completion_owner.canonicalize_tool_surface

    def capture_surface(tools: object, *, runtime: object | None = None,
                        probe_modes: Sequence[str] | None = None) -> list[dict[str, object]]:
        if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)):
            for tool in tools:
                add_schema(tool)
        rendered = canonicalize(tools, runtime=runtime, probe_modes=probe_modes)
        for tool in rendered:
            add_schema(tool)
        return rendered

    with patch.object(completion_owner, "canonicalize_tool_surface", capture_surface):
        for profile_path in profile_paths:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile_switches = profile.get("switches", {})
            assert isinstance(profile_switches, Mapping), f"invalid switch map in {profile_path.name}"
            profile_environment = {str(key): str(value) for key, value in profile_switches.items()}
            for atomic_apply in ("off", "on"):
                profile_environment[ATOMIC_APPLY_ENV] = atomic_apply
                with patch.dict(os.environ, profile_environment, clear=False):
                    for registry in (None, ProfileToolRegistry()):
                        with patch.object(kernel_agent_tools, "_active", registry):
                            for probe_gate_on, observation_probe_on, dependency_fact_on in product((False, True), repeat=3):
                                menu = react_repair._tools_for_probe_modes(
                                    probe_gate_on=probe_gate_on,
                                    observation_probe_on=observation_probe_on,
                                    dependency_fact_on=dependency_fact_on,
                                )
                                for tool in menu:
                                    add_schema(tool)
                                wrong_test_menu = react_repair._done_tool_surface(menu, {
                                    "test_wrong_wire": True, "protect_tests": "off", "runner_kind": "pytest",
                                })
                                for tool in wrong_test_menu:
                                    add_schema(tool)
    return schemas


def _audit_renderable_tool_schemas(
    schemas: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
) -> tuple[int, int, list[str]]:
    names: set[str] = set()
    properties: set[tuple[str, str]] = set()
    violations: set[str] = set()
    for tool, contract in schemas:
        parts = _tool_parts(tool)
        if parts is None:
            continue
        name, advertised = parts
        names.add(name)
        for variant_properties, _required, _distinguishing in _schema_variants(advertised):
            properties.update((name, property_name) for property_name in variant_properties)
        violations.update(_schema_violations(name, advertised, contract))
    return len(names), len(properties), sorted(violations)


def test_every_renderable_tool_schema_matches_its_step_contract() -> None:
    schemas = _renderable_tool_schemas()
    tool_count, property_count, violations = _audit_renderable_tool_schemas(schemas)
    print(f"tool-schema guard: tools={tool_count} properties={property_count} violations={len(violations)}")

    assert tool_count >= 25, f"tool-schema guard scanned too few tools: {tool_count}"
    assert not violations, "tool-schema guard violations: " + "; ".join(violations)


def test_tool_schema_guard_rejects_a_planted_unknown_property() -> None:
    advertised = {"type": "object", "properties": {"planted": {"type": "string"}}, "required": ["planted"]}
    contract = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    violations = _schema_violations("synthetic_tool", advertised, contract)

    assert len(violations) == 1
    assert violations[0].startswith("synthetic_tool.planted:")
