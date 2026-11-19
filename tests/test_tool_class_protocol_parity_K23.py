"""Protocol edit-tool parity for the K23 edit-detection lists.

Each list here classifies what the model already did (edit counting, applied
product records, loop breakers, rejected-edit notes). Those lists must also
recognize the protocol edit tools (``edit_source``/``rewrite_source``), or a
protocol edit is silently counted as a non-edit. Every list below must:

* name both protocol tools (via the ``*PROTOCOL_EDIT_TOOL_NAMES`` splice), and
* keep every original legacy member.

Lists are located by their enclosing scope (module-level name or function
name) and matched by their exact legacy membership, so an unrelated literal in
the same file cannot satisfy the check. OFFER lists (menus, schemas, dispatch
gates) are intentionally absent: they must not change in this chunk.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from stateframe.engine import kernel_edit_tool_names as names

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"
_SPLICES = {"PROTOCOL_EDIT_TOOL_NAMES": names.PROTOCOL_EDIT_TOOL_NAMES,
            "EDIT_TOOL_NAMES": names.EDIT_TOOL_NAMES}

# (file relative to engine, enclosing scope name, exact legacy members)
MIGRATED_LISTS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("detreason/attempt_memory.py", "_EDIT_TOOLS", frozenset({
        "apply_patch", "edit", "file_create", "file_write", "patch", "replace",
        "str_replace", "write_file"})),
    ("kernel_completion_authority.py", "_observed_effects", frozenset({
        "edit", "file_create", "file_delete", "file_move", "replace", "replace_module",
        "shell", "str_replace", "write"})),
    ("kernel_completion_owner.py", "_STALL_RULE_EDIT_TOOLS", frozenset({"edit", "str_replace"})),
    ("kernel_completion_owner.py", "_progress_key", frozenset({
        "edit", "file_create", "file_delete", "file_move", "replace", "replace_module",
        "shell", "str_replace", "write"})),
    ("kernel_completion_policy.py", "_PRODUCT_EDIT_TOOLS", frozenset({
        "edit", "str_replace", "write", "file_create", "replace", "replace_module"})),
    ("probes/stateframe_shadow.py", "_EDIT_TOOLS", frozenset({
        "edit", "str_replace", "file_create"})),
    ("probes/react_repair.py", "_LOOP_BREAKING_COMMANDS", frozenset({
        "edit", "str_replace", "file_create", "insert_before", "replace_module", "write",
        "test", "run_tests", "authored_check", "done", "work", "abstain"})),
    ("probes/react_repair.py", "_render_contract_rejection", frozenset({
        "edit", "str_replace", "file_create", "insert_before"})),
    ("probes/react_repair.py", "work", frozenset({
        "edit", "file_create", "file_delete", "file_move", "shell", "str_replace",
        "insert_before", "replace_module"})),
    ("probes/react_repair.py", "work", frozenset({
        "edit", "str_replace", "file_create", "insert_before"})),
)


def _members(node: ast.AST) -> frozenset[str] | None:
    """String members of a set/tuple/list literal, resolving protocol splices.

    Returns None when the node is not a pure tool-name literal.
    """
    if not isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        return None
    if not all(isinstance(e, (ast.Constant, ast.Starred)) for e in node.elts):
        return None
    out: set[str] = set()
    for e in node.elts:
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            out.add(e.value)
        elif (isinstance(e, ast.Starred) and isinstance(e.value, ast.Name)
              and e.value.id in _SPLICES):
            out |= set(_SPLICES[e.value.id])
    return frozenset(out)


def _scope_nodes(tree: ast.Module, scope: str) -> list[ast.AST]:
    """AST nodes for a module-level assignment or a (possibly nested) function."""
    found: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == scope for t in node.targets):
            found.append(node)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == scope:
            found.append(node)
    return found


def _matching_literals(rel: str, scope: str, legacy: frozenset[str]) -> list[frozenset[str]]:
    tree = ast.parse((ENGINE / rel).read_text(encoding="utf-8"), filename=rel)
    matches: list[frozenset[str]] = []
    for root in _scope_nodes(tree, scope):
        for node in ast.walk(root):
            members = _members(node)
            if members is None:
                continue
            if members - names.PROTOCOL_EDIT_TOOL_NAMES == legacy:
                matches.append(members)
    return matches


@pytest.mark.parametrize(
    ("rel", "scope", "legacy"), MIGRATED_LISTS,
    ids=[f"{rel}:{scope}:{i}" for i, (rel, scope, _legacy) in enumerate(MIGRATED_LISTS)])
def test_migrated_edit_detection_list_names_protocol_tools(
        rel: str, scope: str, legacy: frozenset[str]) -> None:
    matches = _matching_literals(rel, scope, legacy)
    assert matches, f"{rel} ({scope}): no literal with legacy members {sorted(legacy)}"
    for members in matches:
        assert names.EDIT_SOURCE_TOOL in members, f"{rel} ({scope}) missing edit_source"
        assert names.REWRITE_SOURCE_TOOL in members, f"{rel} ({scope}) missing rewrite_source"
        assert legacy <= members, f"{rel} ({scope}) dropped legacy members"


def test_scope_parser_finds_splice_as_protocol_members() -> None:
    """The splice resolver must count ``*PROTOCOL_EDIT_TOOL_NAMES`` as protocol tools."""
    node = ast.parse('{"edit", *PROTOCOL_EDIT_TOOL_NAMES}', mode="eval").body
    assert _members(node) == frozenset({"edit", "edit_source", "rewrite_source"})


def test_attempt_memory_counts_protocol_edits_as_edits() -> None:
    """Behavior: a rewrite_source receipt is classified as an edit, not an unknown action."""
    from stateframe.engine.detreason import attempt_memory as am

    assert am._action_kind("edit_source") == "edit"
    assert am._action_kind("rewrite_source") == "edit"
    assert am._receipt_kind({"intent": "rewrite_source app.py"}) == "edit"
