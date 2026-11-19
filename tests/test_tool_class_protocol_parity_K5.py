"""Protocol edit-tool parity for the K5 edit-detection literal lists.

Each list here classifies what the model already did or asked (edit counting,
receipt/ledger recognition, loop and rejection tracking, post-edit refresh).
When a run uses the protocol menu, the protocol edit tools (``edit_source``,
``rewrite_source``) must be counted alongside the legacy editors, or those
edits silently go unrecognised.

Each case pins a list by ``file:line`` in the 0.7.79 source and its exact legacy
membership. The check is: the list still names every legacy
member, and it names exactly the protocol tools on top of them.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from stateframe.engine import kernel_edit_tool_names as names

ENGINE = pathlib.Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
SENTINEL = "str_replace"


def _effective_members(node: ast.AST) -> frozenset[str] | None:
    """Return the string members of a literal container, expanding the protocol splice."""
    if not isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        return None
    members: set[str] = set()
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            members.add(elt.value)
        elif (isinstance(elt, ast.Starred) and isinstance(elt.value, ast.Name)
              and elt.value.id == "PROTOCOL_EDIT_TOOL_NAMES"):
            members |= names.PROTOCOL_EDIT_TOOL_NAMES
        else:
            return None
    return frozenset(members)


def _lists_at(rel: str, line: int) -> list[frozenset[str]]:
    tree = ast.parse((ENGINE / rel).read_text(encoding="utf-8"), filename=rel)
    found = []
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != line:
            continue
        members = _effective_members(node)
        if members is not None and SENTINEL in members:
            found.append(members)
    return found


# (file, line after this change, original legacy members)
MIGRATED_LISTS = [
    ("kernel_edit_churn_budget.py", 313,
     {"edit", "file_create", "replace", "replace_module", "str_replace", "write"}),
    ("kernel_groundtruth.py", 375, {"edit", "str_replace"}),
    ("kernel_groundtruth.py", 708, {"edit", "str_replace", "test"}),
    ("kernel_groundtruth.py", 713, {"edit", "str_replace", "test"}),
    ("kernel_intent_carry.py", 31,
     {"edit", "file_create", "replace", "replace_module", "str_replace", "write"}),
    ("kernel_post_edit_refresh.py", 11,
     {"edit", "file_create", "file_delete", "file_move", "replace", "replace_module",
      "shell_effect", "str_replace", "write"}),
    ("kernel_preedit_coverage_runtime.py", 20,
     {"edit", "file_create", "replace", "replace_module", "str_replace", "write"}),
    ("kernel_rejected_candidate.py", 13, {"edit", "file_create", "str_replace"}),
    ("kernel_rejection_template_restrict.py", 763,
     {"edit", "file_create", "insert_before", "replace", "replace_module", "str_replace", "write"}),
    ("kernel_rejection_template_restrict.py", 814,
     {"edit", "file_create", "insert_before", "replace", "replace_module", "str_replace", "write"}),
    ("kernel_stateframe_action_ledger.py", 107, {"edit", "edit_file", "str_replace"}),
    ("kernel_stateframe_action_ledger.py", 418, {"edit", "edit_file", "str_replace"}),
    ("kernel_stateframe_action_ledger.py", 479, {"edit", "edit_file", "str_replace"}),
    ("kernel_stateframe_action_ledger.py", 1163,
     {"edit", "edit_file", "file_create", "read", "str_replace"}),
    ("kernel_stateframe_action_ledger.py", 1212, {"edit", "edit_file", "str_replace"}),
    ("kernel_stateframe_action_ledger.py", 1415,
     {"edit", "edit_file", "file_create", "str_replace"}),
    ("kernel_stateframe_f4.py", 60,
     {"apply_patch", "edit", "edit_file", "file_delete", "file_move", "shell", "str_replace",
      "write_file"}),
    ("kernel_stateframe_thinking_router.py", 225,
     {"edit", "file_create", "replace_module", "str_replace"}),
    ("kernel_stateframe_v2_s1.py", 36, {"edit", "file_create", "str_replace"}),
    ("kernel_stateframe_v2_s1.py", 280,
     {"edit", "file_create", "probe", "read", "run_tests", "str_replace", "test"}),
    ("kernel_working_set_snapshot.py", 500,
     {"edit", "file_create", "file_delete", "file_move", "insert_before", "str_replace"}),
]


def test_migrated_list_count_is_pinned() -> None:
    assert len(MIGRATED_LISTS) == 21


@pytest.mark.parametrize(
    ("rel", "line", "legacy"),
    [pytest.param(rel, line, frozenset(legacy), id=f"{rel}:{line}")
     for rel, line, legacy in MIGRATED_LISTS],
)
def test_detect_list_counts_protocol_edit_tools(rel: str, line: int, legacy: set[str]) -> None:
    lists = _lists_at(rel, line)
    assert lists, f"no str_replace literal list at {rel}:{line}"
    members = lists[0]
    assert names.EDIT_SOURCE_TOOL in members, f"{rel}:{line} does not count edit_source"
    assert names.REWRITE_SOURCE_TOOL in members, f"{rel}:{line} does not count rewrite_source"
    assert frozenset(legacy) <= members, f"{rel}:{line} dropped legacy members"
    assert members - frozenset(legacy) == names.PROTOCOL_EDIT_TOOL_NAMES, (
        f"{rel}:{line} has unexpected members beyond legacy + protocol")
