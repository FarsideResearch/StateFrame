"""Menu surface for the handle-addressed edit protocol tools.

Under ``STATEFRAME_EDIT_PROTOCOL=handles`` the legacy text editors are hidden
and ``edit_source``/``rewrite_source`` are appended. The schemas are static:
handle ids are request-scoped and only ever appear in the current-source map,
never in a tool schema (that would change the menu bytes every request).
Under ``legacy_safe`` the input list is returned as the same object.
"""
from __future__ import annotations

import copy

from kernel_edit_protocol_switch import handles_enabled
from kernel_edit_tool_names import EDIT_SOURCE_TOOL, REWRITE_SOURCE_TOOL

EDIT_SOURCE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": EDIT_SOURCE_TOOL,
        "description": (
            "Apply 1-4 text edits to ONE file. Each edit is {handle, old_text, new_text}: "
            "handle is a span handle from the current source map, old_text must occur "
            "exactly once inside that handle's span, and new_text replaces it. Edits must "
            "not overlap. The batch is applied atomically or refused as a whole, with every "
            "problem reported at once."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "handle": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                        },
                        "required": ["handle", "old_text", "new_text"],
                    },
                },
            },
            "required": ["edits"],
        },
    },
}

REWRITE_SOURCE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": REWRITE_SOURCE_TOOL,
        "description": (
            "Replace the whole region covered by a REWRITE handle from the current source "
            "map with new_text. The handle must be a REWRITE handle; span handles are "
            "refused. Use edit_source for bounded changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["handle", "new_text"],
        },
    },
}

# Legacy text editors the protocol replaces; observation and file-lifecycle
# tools stay on the menu.
HIDDEN_LEGACY_TOOLS: frozenset[str] = frozenset({"edit", "str_replace", "insert_before"})


def _tool_name(tool: object) -> object:
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    return function.get("name") if isinstance(function, dict) else None


def tool_surface(tools: list[dict]) -> list[dict]:
    """Return the menu for the active edit protocol; never mutates ``tools``."""
    if not handles_enabled():
        return tools
    kept = [tool for tool in tools if _tool_name(tool) not in HIDDEN_LEGACY_TOOLS]
    return kept + [copy.deepcopy(EDIT_SOURCE_SCHEMA), copy.deepcopy(REWRITE_SOURCE_SCHEMA)]


__all__ = ["EDIT_SOURCE_SCHEMA", "HIDDEN_LEGACY_TOOLS", "REWRITE_SOURCE_SCHEMA", "tool_surface"]
