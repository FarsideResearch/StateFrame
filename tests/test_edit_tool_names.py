"""Edit-tool name drift guard.

The engine hardcodes tool-name literal lists (``{"edit", "str_replace", ...}``)
in many modules. Each one is a place where a new edit tool can be
misclassified as a non-edit action. This test scans the engine sources for
every literal list that names ``str_replace`` and checks it against the
shared constants in ``kernel_edit_tool_names``:

* every edit-shaped member must be a known edit tool name (drift guard);
* every list must name ``edit_source`` (strict xfail until the list is wired
  onto the shared constant; the xfail id carries ``file:line``).
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from stateframe.engine import kernel_edit_tool_names as names

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"
SENTINEL = "str_replace"

# Tool names that legitimately share a literal list with edit tools without
# being edits (menu floors, repair menus, observation tools, receipts). A name
# outside this set and outside EDIT_TOOL_NAMES is unclassified drift.
NON_EDIT_TOOL_NAMES: frozenset[str] = frozenset({
    "abstain", "authored_check", "contract_fact", "decide", "dependency_fact",
    "dispute", "dispute_check", "done", "edit_receipt", "fact", "file_list",
    "git", "grep", "job_kill", "job_output", "job_status", "job_stop",
    "job_wait", "launch_job", "list_dir", "list_files", "localization_frontier",
    "localize", "new_contract", "new_contract_fact", "note_retrieve",
    "note_update", "parallel", "predict", "probe", "pytest", "question_gate",
    "read", "read_check_receipt", "read_file", "read_ticket", "recall",
    "recall_evidence", "references", "reproduce", "requirement_table",
    "run_tests", "shell", "shell_effect", "skill_load", "subframe_blocked",
    "test", "test_run", "typed_test_writer", "verification", "web_fetch",
    "where_defined", "who_calls", "work", "work_update",
})


def _literal_tool_lists() -> list[tuple[str, int, frozenset[str]]]:
    """Every set/tuple/list literal of strings in the engine naming str_replace."""
    found: list[tuple[str, int, frozenset[str]]] = []
    for path in sorted(ENGINE.rglob("*.py")):
        if path.name == "kernel_edit_tool_names.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Set, ast.Tuple, ast.List)):
                continue
            # Starred elements splice another list in; the literal members are
            # still a tool-name list and still have to be classified.
            if not all(isinstance(e, (ast.Constant, ast.Starred)) for e in node.elts):
                continue
            members = set(
                e.value for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str))
            # A splice of the shared constant names the protocol tools.
            for e in node.elts:
                if (isinstance(e, ast.Starred) and isinstance(e.value, ast.Name)
                        and e.value.id in {"PROTOCOL_EDIT_TOOL_NAMES", "EDIT_TOOL_NAMES"}):
                    members |= set(names.PROTOCOL_EDIT_TOOL_NAMES)
            members = frozenset(members)
            if SENTINEL in members:
                rel = path.relative_to(ROOT).as_posix()
                found.append((rel, node.lineno, members))
    return found


LITERAL_LISTS = _literal_tool_lists()


def _list_id(entry: tuple[str, int, frozenset[str]]) -> str:
    return f"{entry[0]}:{entry[1]}"


def test_scan_finds_the_known_literal_lists() -> None:
    """The scanner sees the engine; an empty scan would make every check vacuous."""
    assert len(LITERAL_LISTS) >= 100
    seen = {rel for rel, _line, _members in LITERAL_LISTS}
    assert "src/stateframe/engine/kernel_readiness_authority.py" in seen
    assert "src/stateframe/engine/kernel_continuation_menu_authority.py" in seen


def test_constants_are_consistent() -> None:
    assert names.PROTOCOL_EDIT_TOOL_NAMES == {"edit_source", "rewrite_source"}
    assert names.PROTOCOL_EDIT_TOOL_NAMES <= names.EDIT_TOOL_NAMES
    assert names.LEGACY_EDIT_TOOL_NAMES <= names.EDIT_TOOL_NAMES
    assert {"edit", "str_replace"} <= names.LEGACY_EDIT_TOOL_NAMES
    assert names.DESTRUCTIVE_EDIT_TOOLS <= names.EDIT_TOOL_NAMES
    assert "rewrite_source" in names.DESTRUCTIVE_EDIT_TOOLS
    assert "edit_source" not in names.DESTRUCTIVE_EDIT_TOOLS
    assert "str_replace" not in names.DESTRUCTIVE_EDIT_TOOLS
    assert not (names.EDIT_TOOL_NAMES & NON_EDIT_TOOL_NAMES)
    assert "shell" not in names.EDIT_TOOL_NAMES


def test_helpers_follow_the_constants() -> None:
    assert names.is_edit_tool("edit_source")
    assert names.is_edit_tool("str_replace")
    assert not names.is_edit_tool("read")
    assert not names.is_edit_tool(None)
    assert names.is_destructive_edit_tool("rewrite_source")
    assert names.is_destructive_edit_tool("edit")
    assert not names.is_destructive_edit_tool("edit_source")
    assert not names.is_destructive_edit_tool(42)


@pytest.mark.parametrize("entry", LITERAL_LISTS, ids=_list_id)
def test_literal_list_edit_members_are_known_edit_tools(
        entry: tuple[str, int, frozenset[str]]) -> None:
    """Every edit-shaped name in a literal list is a subset of EDIT_TOOL_NAMES."""
    rel, line, members = entry
    edit_shaped = members - NON_EDIT_TOOL_NAMES
    unknown = sorted(edit_shaped - names.EDIT_TOOL_NAMES)
    assert not unknown, (
        f"{rel}:{line} names tools outside EDIT_TOOL_NAMES: {unknown}; add them to "
        "kernel_edit_tool_names (edit tools) or NON_EDIT_TOOL_NAMES (not edits)")


# Lists that name only legacy editors on purpose (the protocol menu hides
# exactly these); they must never name the protocol tools.
LEGACY_ONLY_LISTS: frozenset[str] = frozenset({
    "src/stateframe/engine/kernel_edit_source_tool_surface.py",
})


def test_legacy_only_lists_never_name_protocol_tools() -> None:
    entries = [entry for entry in LITERAL_LISTS if entry[0] in LEGACY_ONLY_LISTS]
    assert {entry[0] for entry in entries} == LEGACY_ONLY_LISTS
    for _rel, _line, members in entries:
        assert not members & names.PROTOCOL_EDIT_TOOL_NAMES


@pytest.mark.parametrize("entry", [
    pytest.param(
        entry, id=_list_id(entry),
        marks=() if names.EDIT_SOURCE_TOOL in entry[2] else pytest.mark.xfail(
            strict=True,
            reason=f"{_list_id(entry)} hardcodes edit tools without "
                   f"{names.EDIT_SOURCE_TOOL}; derive it from EDIT_TOOL_NAMES"))
    for entry in LITERAL_LISTS if entry[0] not in LEGACY_ONLY_LISTS
])
def test_literal_list_names_edit_source(entry: tuple[str, int, frozenset[str]]) -> None:
    """Each literal edit-tool list derives from the constant (names edit_source)."""
    rel, line, members = entry
    assert names.EDIT_SOURCE_TOOL in members, f"{rel}:{line} is missing edit_source"
