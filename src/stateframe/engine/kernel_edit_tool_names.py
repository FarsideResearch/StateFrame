"""Shared edit-tool name constants for the edit protocol.

Governors, locate policy and menu authorities classify tool calls by name.
Those names were historically hardcoded as literal sets in each module, so a
new edit tool (``edit_source``/``rewrite_source``) could be misclassified as a
non-edit action by any list that was not updated. This module is the single
source of truth those lists derive from; ``tests/test_edit_tool_names.py``
scans the engine for literal lists and fails when one drifts from it.

Lists are wired onto these constants per module as the edit protocol lands;
the drift test keeps a strict xfail for every list not yet wired.
"""
from __future__ import annotations

# Legacy edit/write tool names and the aliases the engine's classifiers accept
# from hosts (edit_file/write_file/create_file/... are host-side spellings of
# the same mutations). ``shell`` is deliberately absent: it can mutate files
# but it is not an edit tool and the edit protocol never admits it as one.
LEGACY_EDIT_TOOL_NAMES: frozenset[str] = frozenset({
    "edit",
    "str_replace",
    "file_create",
    "file_delete",
    "file_move",
    "replace_module",
    "write",
    "replace",
    "patch",
    "apply_patch",
    "insert_before",
    "insert",
    "batch_edit",
    "edit_file",
    "file_edit",
    "write_file",
    "file_write",
    "create_file",
    "delete_file",
    "rename_file",
    "delete",
    "move",
    "restore_candidate",
})

# Handle-addressed edit tools introduced by the edit protocol.
EDIT_SOURCE_TOOL = "edit_source"
REWRITE_SOURCE_TOOL = "rewrite_source"
PROTOCOL_EDIT_TOOL_NAMES: frozenset[str] = frozenset({EDIT_SOURCE_TOOL, REWRITE_SOURCE_TOOL})

# Every tool name that counts as an edit for classification purposes.
EDIT_TOOL_NAMES: frozenset[str] = LEGACY_EDIT_TOOL_NAMES | PROTOCOL_EDIT_TOOL_NAMES

# Edit tools that replace a whole file (or an unbounded region) rather than a
# bounded span. The stale-edit repair policy and the destructive guard key on
# this set; ``rewrite_source`` must be classified destructive from day one.
DESTRUCTIVE_EDIT_TOOLS: frozenset[str] = frozenset({"edit", REWRITE_SOURCE_TOOL})


def is_edit_tool(name: object) -> bool:
    """True when ``name`` is an edit tool under the shared classification."""
    return isinstance(name, str) and name in EDIT_TOOL_NAMES


def is_destructive_edit_tool(name: object) -> bool:
    """True when ``name`` is an edit tool that rewrites more than a bounded span."""
    return isinstance(name, str) and name in DESTRUCTIVE_EDIT_TOOLS


__all__ = [
    "DESTRUCTIVE_EDIT_TOOLS",
    "EDIT_SOURCE_TOOL",
    "EDIT_TOOL_NAMES",
    "LEGACY_EDIT_TOOL_NAMES",
    "PROTOCOL_EDIT_TOOL_NAMES",
    "REWRITE_SOURCE_TOOL",
    "is_destructive_edit_tool",
    "is_edit_tool",
]
