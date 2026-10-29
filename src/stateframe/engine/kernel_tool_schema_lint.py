"""Validate generated tool schemas before sending them to a provider.

Reject unsupported or contradictory schema shapes locally so invalid tool
declarations cannot reach the request boundary.
"""
from __future__ import annotations

from collections.abc import Mapping

HARNESS_CONTRACT = "HARNESS_CONTRACT"
_COMBINATORS = ("oneOf", "anyOf", "allOf")
_NAME_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions"})
# Literal data, not sub-schemas: never descend into these values.
_DATA_KEYS = frozenset({"enum", "const", "default", "examples", "description", "title"})


class ToolSchemaContractError(ValueError):
    """A tool surface the provider cannot compile; terminal, never retried."""

    outcome = HARNESS_CONTRACT
    failure_origin = "harness"
    harness_contract = True
    transport_retryable = False
    transport_class = "contract"

    def __init__(self, problems: list[str]) -> None:
        super().__init__("tool schema contract violation: " + "; ".join(problems))
        self.problems = list(problems)


def _escape(token: object) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def _walk(node: object, pointer: str, problems: list[str]) -> None:
    if isinstance(node, Mapping):
        if "enum" in node:
            enum = node["enum"]
            if not isinstance(enum, list) or not enum:
                problems.append(f"{pointer}/enum: enum must be a non-empty array")
        if "$ref" in node:
            problems.append(f"{pointer}/$ref: $ref is not supported on the tool wire")
        for key in _COMBINATORS:
            if key in node and (not isinstance(node[key], list) or not node[key]):
                problems.append(f"{pointer}/{key}: {key} must be a non-empty array")
        low, high = node.get("minItems"), node.get("maxItems")
        if type(low) is int and type(high) is int and low > high:
            problems.append(f"{pointer}: minItems {low} exceeds maxItems {high}")
        for key, value in node.items():
            child = f"{pointer}/{_escape(key)}"
            if key in _DATA_KEYS:
                continue
            if key in _NAME_MAPS and isinstance(value, Mapping):
                # A name->schema map: its keys are property names, not schema
                # keywords (a property literally named "enum" is legal).
                for name, schema in value.items():
                    _walk(schema, f"{child}/{_escape(name)}", problems)
            else:
                _walk(value, child, problems)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, f"{pointer}/{index}", problems)


def lint_tools(tools: object) -> list[str]:
    """Return ``"<tool>:<json-pointer>: reason"`` strings; ``[]`` when clean.

    ``None``/absent tools are clean (plain chat requests carry no tools).
    """
    if tools is None:
        return []
    if not isinstance(tools, list):
        return ["<tools>: tools must be a list"]
    problems: list[str] = []
    for index, tool in enumerate(tools):
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if not isinstance(function, Mapping):
            problems.append(f"<tool {index}>: tool lacks a function object")
            continue
        name = function.get("name")
        label = name if isinstance(name, str) and name else f"<tool {index}>"
        if label.startswith("<"):
            problems.append(f"{label}: function name must be a non-empty string")
        parameters = function.get("parameters")
        if parameters is None:
            continue
        found: list[str] = []
        _walk(parameters, "", found)
        problems.extend(f"{label}:{item}" for item in found)
    return problems


def assert_tools_clean(tools: object) -> None:
    problems = lint_tools(tools)
    if problems:
        raise ToolSchemaContractError(problems)
