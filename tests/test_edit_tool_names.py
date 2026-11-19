"""Edit-tool name drift guard.

The engine hardcodes tool-name literal lists (``{"edit", "str_replace", ...}``)
in many modules. Each one is a place where a new edit tool can be
misclassified as a non-edit action. This test scans the engine sources for
every literal list that names ``str_replace`` and checks it against the
shared constants in ``kernel_edit_tool_names``:

* every edit-shaped member must be a known edit tool name (drift guard);
* every detection list must name ``edit_source`` (strict xfail until the list
  is wired onto the shared constant; the xfail id carries ``file:line``);
* reviewed offer and wire lists retain their existing members.
"""
from __future__ import annotations

import ast
import pathlib
from typing import NamedTuple

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


OFFER_REVIEWED: dict[str, str] = {
    'src/stateframe/engine/kernel_completion_owner.py:_STALL_RULE_TOOLS': 'Default stall recovery menu and order.',
    'src/stateframe/engine/kernel_completion_owner.py:_DECLARED_REPAIR_TOOLS': 'Declared repair menu and order.',
    'src/stateframe/engine/kernel_completion_owner.py:_common_sense_decision': "Chooses the current or full tool surface for the governor's menu projection.",
    'src/stateframe/engine/kernel_completion_owner.py:_reserved_terminal_menu_names': 'Admits the existing reserved submit menu shapes.',
    'src/stateframe/engine/kernel_completion_owner.py:_expanded_zero_edit_submit_menu': 'Projects the reserved submit menu in its existing order.',
    'src/stateframe/engine/kernel_completion_owner.py:_preserve_zero_edit_terminal_menu': 'Retains a reserved terminal menu only for the existing menu shapes.',
    'src/stateframe/engine/kernel_completion_owner.py:stall_open_work_frame': 'No-progress stall recovery menu rendered into the frame.',
    'src/stateframe/engine/kernel_completion_owner.py:legal_next_actions.add': 'Admission rule for offering legacy repair actions after a green check.',
    'src/stateframe/engine/kernel_completion_owner.py:legal_next_actions': 'Fixed menu of open-work actions offered after a green check.',
    'src/stateframe/engine/kernel_completion_owner.py:align_tools_to_completion_route': 'Keeps specific legacy repair tools on the projected wire menu.',
    'src/stateframe/engine/kernel_completion_owner.py:_rejection_template_tool_admissible': 'Rejected-edit recovery admits the single str_replace schema.',
    'src/stateframe/engine/kernel_completion_owner.py:_rejection_menu_admission': 'Required-context lookup is tied to the str_replace-only rejection menu.',
    'src/stateframe/engine/kernel_completion_owner.py:_terminal_tool_menu_impl': 'Fixed terminal menu and order for runs without host checks.',
    'src/stateframe/engine/kernel_stateframe_runtime.py:_lean_no_host_recovery_tools': 'Orders the frozen no-host recovery menu.',
    'src/stateframe/engine/kernel_stateframe_runtime.py:StateFrameRuntime._get_sure_menu': 'Controls when predict is added to the menu.',
    'src/stateframe/engine/kernel_stateframe_runtime.py:StateFrameRuntime._run_host_repair_checks': 'Supplies the fallback repair packet menu.',
    'src/stateframe/engine/kernel_stateframe_runtime.py:StateFrameRuntime._host_closure_for_done': 'Builds the repair packet fallback and boundary menus.',
    'src/stateframe/engine/kernel_stateframe_runtime.py:StateFrameRuntime._terminal_tool_menu': 'Selects default host-closure tools to offer.',
    'src/stateframe/engine/kernel_readiness_authority.py:REPAIR_TOOLS': 'Defines the legacy repair menu and its rendered names.',
    'src/stateframe/engine/kernel_readiness_authority.py:ReadinessDecision.next_recovery_action': 'Orders the next actions named in readiness output.',
    'src/stateframe/engine/kernel_readiness_authority.py:project_menu': 'Orders tools restored to the post-green menu.',
    'src/stateframe/engine/kernel_readiness_authority.py:menu_violations': 'Validates allowed and required post-green menu entries.',
    'src/stateframe/engine/kernel_done_readiness_authority.py:_candidate_recoveries': 'Selects and orders generic recovery actions shown on refusal.',
    'src/stateframe/engine/kernel_done_readiness_authority.py:_unchanged_candidate_refusal': 'Filters recovery actions shown for an unchanged candidate.',
    'src/stateframe/engine/kernel_done_is_submit_authority.py:_REQUIRED_CANDIDATE_TOOLS': 'Defines the ordered candidate tool floor.',
    'src/stateframe/engine/kernel_done_is_submit_authority.py:_OWNERLESS_PROGRESS_TOOLS': 'Selects the recovery actions named in the ownerless refusal.',
    'src/stateframe/engine/kernel_capability_floor.py:CAPABILITY_GROUPS': 'Keeps the existing capability floor and its tool order.',
    'src/stateframe/engine/kernel_capability_floor.py:SCOPE_TURN_PREFERRED_TOOLS': 'Pins the preferred order and receipt for scoped edit turns.',
    'src/stateframe/engine/kernel_convergence_observer.py:_route': 'Keeps both ordered recovery suggestions in the model-facing notice.',
    'src/stateframe/engine/kernel_frame_budget.py:_EDIT_TOOLS': 'Limits the schemas offered by each frame-budget gear.',
    'src/stateframe/engine/kernel_frame_budget.py:CONTRACT_TOOLS': 'Pins which provider schemas bypass compaction.',
    'src/stateframe/engine/kernel_request_budget.py:_CORE_TOOLS': 'Pins schema priority during provider request trimming.',
    'src/stateframe/engine/kernel_request_budget.py:_runtime_context': "Pins the ordered editor choices for the native request's protected schema.",
    'src/stateframe/engine/kernel_request_tool_surface.py:_REQUEST_TOOL_FLOOR': 'Pins the names and order of the request-final capability floor.',
    'src/stateframe/engine/kernel_route_budget.py:_next_action': 'Pins the next-action preference rendered in the route-budget block.',
    'src/stateframe/engine/kernel_stateframe_get_sure.py:REPAIR_TOOLS': 'Keeps the set of tools that get-sure can add back to the repair menu.',
    'src/stateframe/engine/kernel_tokenlab_stall_breaker.py:StallWireSignal.next_actions': 'Pins the next-action names recorded by the wire signal.',
    'src/stateframe/engine/kernel_tokenlab_stall_breaker.py:observe_closed_action': 'Pins the advertised next actions recorded in the probe-suppression receipt.',
    'src/stateframe/engine/kernel_tokenlab_stall_breaker.py:_project_stallwire_menu': 'Pins the fallback editor menu and its next-action receipt.',
    'src/stateframe/engine/kernel_unit_policy.py:_BASE_TOOLS': 'Keeps the subframe menu and dispatch allowlist fixed.',
    'src/stateframe/engine/kernel_boundary_recovery.py:CAPABILITY_FLOOR': 'Names the schemas restored by the boundary-recovery menu.',
    'src/stateframe/engine/kernel_clause_review.py:next_actions': 'Sets the order of suggested repair actions.',
    'src/stateframe/engine/kernel_cutoff_rule.py:CORE_MENU': 'Defines the exact ordered cutoff menu.',
    'src/stateframe/engine/kernel_green_lock_repair.py:_REPAIR_MENU_EDIT_ACTIONS': 'Names the editors the repair menu can restore.',
    'src/stateframe/engine/kernel_green_menu_open_obligation.py:EDIT_TOOLS': 'Names the editors the open-counterexample menu can restore.',
    'src/stateframe/engine/kernel_idealpath.py:PRODUCT_WRITE_ACTIONS': 'Declares the stage S5 product-write affordances.',
    'src/stateframe/engine/kernel_lean.py:PLAIN_TOOL_NAMES': 'Declares the tools wired into lean requests.',
    'src/stateframe/engine/kernel_mutate_projection.py:AUTHORING_EFFECT_TOOLS': 'Defines the exclusions shared by the authoring menu and its dispatch contract.',
    'src/stateframe/engine/kernel_plain_host.py:PlainHost.SUPPORTED': 'Names the plain host handlers paired with the baseline schemas.',
    'src/stateframe/engine/kernel_postgreen_no_veto.py:defer_readiness': 'Names the readiness dispositions published for the delegated menu.',
    'src/stateframe/engine/kernel_premature_rule.py:MENU': 'Defines the fallback selection and ordered menu printed in the frame.',
    'src/stateframe/engine/kernel_refusal_compact.py:_refresh_action': 'Sets the priority of refresh actions named in refusal text.',
    'src/stateframe/engine/kernel_rejection_template_restrict.py:READ_POINTER_REQUIRED_TOOLS': 'Defines the required schemas in the read-pointer repair menu.',
    'src/stateframe/engine/kernel_stateframe_hstate_plus.py:_menu_names': 'Defines the fixed fallback menu names.',
    'src/stateframe/engine/kernel_working_notebook.py:NOTE_CAPABLE_TOOLS': 'Names the tool schemas that receive the note sidecar.',
}

# A few functions deliberately contain two separate legacy menu literals.
REVIEWED_OFFER_COUNTS: dict[str, int] = {
    "src/stateframe/engine/kernel_stateframe_runtime.py:StateFrameRuntime._host_closure_for_done": 2,
    "src/stateframe/engine/kernel_readiness_authority.py:menu_violations": 2,
    "src/stateframe/engine/kernel_done_readiness_authority.py:_candidate_recoveries": 2,
    "src/stateframe/engine/kernel_convergence_observer.py:_route": 2,
    "src/stateframe/engine/kernel_tokenlab_stall_breaker.py:_project_stallwire_menu": 2,
}


class ToolList(NamedTuple):
    path: str
    line: int
    members: frozenset[str]
    symbol: str

    @property
    def site(self) -> str:
        return f"{self.path}:{self.symbol}"


def _list_symbol(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Return the owning function, class, or declared module/class constant."""
    scopes: list[str] = []
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scopes.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            parent = parents.get(node)
            if not isinstance(parent, (ast.Module, ast.ClassDef)):
                continue
            target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else (
                node.target if isinstance(node, ast.AnnAssign) else None)
            if isinstance(target, ast.Name):
                scopes.append(target.id)
    return ".".join(reversed(scopes))


def _literal_tool_lists() -> list[ToolList]:
    """Find literal tool lists containing str_replace, with reviewed owners."""
    found: list[ToolList] = []
    reviewed_paths = {key.split(":", 1)[0] for key in OFFER_REVIEWED}
    shared_names = {"PROTOCOL_EDIT_TOOL_NAMES", "EDIT_TOOL_NAMES"}
    for path in sorted(ENGINE.rglob("*.py")):
        if path.name == "kernel_edit_tool_names.py":
            continue
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = ({child: parent for parent in ast.walk(tree)
                    for child in ast.iter_child_nodes(parent)} if rel in reviewed_paths else {})
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Set, ast.Tuple, ast.List)):
                continue
            # Starred members splice shared edit names into otherwise literal menus.
            if not all(isinstance(item, (ast.Constant, ast.Starred)) for item in node.elts):
                continue
            members = {
                item.value for item in node.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            for item in node.elts:
                if not isinstance(item, ast.Starred):
                    continue
                value = item.value
                if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                        and value.func.id == "sorted" and len(value.args) == 1):
                    value = value.args[0]
                if isinstance(value, ast.Name) and value.id in shared_names:
                    members.update(names.PROTOCOL_EDIT_TOOL_NAMES)
            if SENTINEL in members:
                symbol = _list_symbol(node, parents) if parents else ""
                found.append(ToolList(rel, node.lineno, frozenset(members), symbol))
    return found


LITERAL_LISTS = _literal_tool_lists()


def _list_id(entry: ToolList) -> str:
    return f"{entry.path}:{entry.line}"


def test_scan_finds_the_known_literal_lists() -> None:
    """The scanner sees the engine; an empty scan would make every check vacuous."""
    assert len(LITERAL_LISTS) >= 100
    seen = {entry.path for entry in LITERAL_LISTS}
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
def test_literal_list_edit_members_are_known_edit_tools(entry: ToolList) -> None:
    """Every edit-shaped name in a literal list is a subset of EDIT_TOOL_NAMES."""
    edit_shaped = entry.members - NON_EDIT_TOOL_NAMES
    unknown = sorted(edit_shaped - names.EDIT_TOOL_NAMES)
    assert not unknown, (
        f"{entry.path}:{entry.line} names tools outside EDIT_TOOL_NAMES: {unknown}; add them to "
        "kernel_edit_tool_names (edit tools) or NON_EDIT_TOOL_NAMES (not edits)")


# Lists that name only legacy editors on purpose (the protocol menu hides
# exactly these); they must never name the protocol tools.
LEGACY_ONLY_LISTS: frozenset[str] = frozenset({
    "src/stateframe/engine/kernel_edit_source_tool_surface.py",
})


def test_legacy_only_lists_never_name_protocol_tools() -> None:
    entries = [entry for entry in LITERAL_LISTS if entry.path in LEGACY_ONLY_LISTS]
    assert {entry.path for entry in entries} == LEGACY_ONLY_LISTS
    for entry in entries:
        assert not entry.members & names.PROTOCOL_EDIT_TOOL_NAMES


def _reviewed_offer_lists() -> dict[str, list[ToolList]]:
    reviewed = {key: [] for key in OFFER_REVIEWED}
    for entry in LITERAL_LISTS:
        if entry.site in reviewed:
            reviewed[entry.site].append(entry)
    return reviewed


REVIEWED_OFFER_LISTS = _reviewed_offer_lists()


@pytest.mark.parametrize("key", OFFER_REVIEWED)
def test_reviewed_offer_lists_stay_legacy_only(key: str) -> None:
    entries = REVIEWED_OFFER_LISTS[key]
    expected_count = REVIEWED_OFFER_COUNTS.get(key, 1)
    assert len(entries) == expected_count, OFFER_REVIEWED[key]
    for entry in entries:
        assert not entry.members & names.PROTOCOL_EDIT_TOOL_NAMES, OFFER_REVIEWED[key]


@pytest.mark.parametrize("entry", [
    pytest.param(
        entry, id=_list_id(entry),
        marks=() if names.EDIT_SOURCE_TOOL in entry.members else pytest.mark.xfail(
            strict=True,
            reason=f"{_list_id(entry)} hardcodes edit tools without "
                   f"{names.EDIT_SOURCE_TOOL}; derive it from EDIT_TOOL_NAMES"))
    for entry in LITERAL_LISTS
    if entry.path not in LEGACY_ONLY_LISTS and entry.site not in OFFER_REVIEWED
])
def test_literal_list_names_edit_source(entry: ToolList) -> None:
    """Each detection list derives from the constant (names edit_source)."""
    assert names.EDIT_SOURCE_TOOL in entry.members, (
        f"{entry.path}:{entry.line} is missing edit_source")
