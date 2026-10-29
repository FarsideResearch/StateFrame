"""R45's opt-in single-carrier tool-schema reduction.

The decision cache and continuity modules own their fields and consumers.  This
module owns only the cross-cutting R45 rule: when enabled, those optional
sidecars remain advertised on one stable tool instead of every tool.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping


MODE_ENV = "STATEFRAME_SCHEMA_SINGLE_CARRIER"
CARRIER_TOOL_NAME = "done"
OPTIONAL_FIELDS = ("memory", "decision_delta")
EVENT = "tool_schema_diet_applied"
# STATEFRAME_TOKENLAB_SIDECAR_OFF: no carrier at all -- the sidecars are stripped from EVERY tool
# (they cost 2.3-4.3 KB of `done` schema per request and were populated in 0 of 425 tool calls across 9 r46
# frame journals).  Strict off|on, default off (byte-identical).  Own event: tool_sidecar_stripped, emitted by
# the runtime at the confirmed-render point on the FINAL tool list of a provider request.
SIDECAR_OFF_ENV = "STATEFRAME_TOKENLAB_SIDECAR_OFF"
SIDECAR_OFF_EVENT = "tool_sidecar_stripped"


def sidecar_off_mode() -> str:
    value = str(os.environ.get(SIDECAR_OFF_ENV, "off") or "off").strip().lower()
    if value in {"", "0", "off"}:
        return "off"
    if value in {"1", "on"}:
        return "on"
    raise ValueError(f"{SIDECAR_OFF_ENV} must be off or on")


def sidecar_off_enabled() -> bool:
    return sidecar_off_mode() == "on"


def _carries_sidecar(tool: object) -> bool:
    function = tool.get("function") if isinstance(tool, Mapping) else None
    parameters = function.get("parameters") if isinstance(function, Mapping) else None
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    return isinstance(properties, Mapping) and any(name in properties for name in OPTIONAL_FIELDS)


def emit_sidecar_stripped(journal: object, tools: object, *, request_id: object = None) -> bool:
    """Own event, fired only when the switch is on AND no tool on the final list carries a sidecar."""
    if journal is None or not sidecar_off_enabled() or not isinstance(tools, list) or not tools:
        return False
    if any(_carries_sidecar(tool) for tool in tools):
        return False
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return False
    try:
        emit(SIDECAR_OFF_EVENT, producer="kernel_tool_schema_diet", request_id=request_id,
             fields=list(OPTIONAL_FIELDS), tools=[_tool_name(tool) for tool in tools], dispatch=False)
    except Exception:  # noqa: BLE001 - telemetry must not change request behavior
        return False
    return True


def mode() -> str:
    """Return the strict default-off switch value."""
    value = str(os.environ.get(MODE_ENV, "off") or "off").strip().lower()
    if value in {"", "0", "off"}:
        return "off"
    if value in {"1", "on"}:
        return "on"
    raise ValueError(f"{MODE_ENV} must be off or on")


def enabled() -> bool:
    return mode() == "on"


def _tool_name(tool: object) -> str | None:
    if not isinstance(tool, Mapping):
        return None
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    return name if isinstance(name, str) else None


def _remove_optional_fields(parameters: dict[str, object]) -> None:
    properties = parameters.get("properties")
    if isinstance(properties, dict):
        for field_name in OPTIONAL_FIELDS:
            properties.pop(field_name, None)
    required = parameters.get("required")
    if isinstance(required, list):
        parameters["required"] = [
            name for name in required
            if name not in OPTIONAL_FIELDS
        ]


def filter_tools(tools: object) -> object:
    """Keep optional sidecar properties on ``done`` only when this arm is on.

    STATEFRAME_TOKENLAB_SIDECAR_OFF on: no carrier, the sidecars leave every tool."""
    sidecar_off = sidecar_off_enabled()
    if (not enabled() and not sidecar_off) or not isinstance(tools, list):
        return tools
    carrier = None if sidecar_off else CARRIER_TOOL_NAME
    result = copy.deepcopy(tools)
    for tool in result:
        if (carrier is not None and _tool_name(tool) == carrier) or not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            _remove_optional_fields(parameters)
    return result


def note_applied(journal: object, tools: object) -> bool:
    """Journal the owning event once when the reduced surface is actually present."""
    if journal is None or not enabled() or getattr(journal, "_schema_single_carrier_applied", False):
        return False
    if not isinstance(tools, list):
        return False
    counts = {field_name: 0 for field_name in OPTIONAL_FIELDS}
    available_tools: list[str] = []
    for tool in tools:
        name = _tool_name(tool)
        if name is None:
            continue
        available_tools.append(name)
        function = tool.get("function") if isinstance(tool, Mapping) else None
        parameters = function.get("parameters") if isinstance(function, Mapping) else None
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        if isinstance(properties, Mapping):
            for field_name in OPTIONAL_FIELDS:
                if field_name in properties:
                    counts[field_name] += 1
    retained_fields = [field_name for field_name, count in counts.items() if count]
    if not retained_fields or CARRIER_TOOL_NAME not in available_tools:
        return False
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return False
    try:
        emit(
            EVENT,
            carrier=CARRIER_TOOL_NAME,
            available_tools=available_tools,
            retained_fields=retained_fields,
            field_counts=counts,
            dispatch=False,
        )
        setattr(journal, "_schema_single_carrier_applied", True)
    except Exception:  # noqa: BLE001 - telemetry must not change request behavior
        return False
    return True


# Descriptive aliases keep callers at the schema-surface boundary readable.
tool_surface = filter_tools
surface = filter_tools


__all__ = [
    "CARRIER_TOOL_NAME",
    "EVENT",
    "MODE_ENV",
    "OPTIONAL_FIELDS",
    "enabled",
    "filter_tools",
    "mode",
    "note_applied",
    "SIDECAR_OFF_ENV",
    "SIDECAR_OFF_EVENT",
    "emit_sidecar_stripped",
    "sidecar_off_enabled",
    "sidecar_off_mode",
    "surface",
    "tool_surface",
]
