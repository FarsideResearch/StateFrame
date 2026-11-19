"""Protocol edit-tool parity for the K4 edit-detection lists.

These lists classify what the model already did: governor counts, pending-effect
and ledger checks, and recognition of an edit call or action line. Each one must
count the protocol tools (``edit_source``, ``rewrite_source``) as edits while
keeping every legacy member. Lists are located by file plus their exact legacy
member set, so moving lines inside a file does not break the check.
"""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from stateframe.engine import kernel_edit_tool_names as names
from stateframe.engine.kernel_bolt import KernelBolt
from stateframe.engine.kernel_cog_governor import _is_productive
from stateframe.engine.kernel_step_contract import is_edit_action_line

ENGINE = pathlib.Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
SPLICE_NAMES = frozenset({"PROTOCOL_EDIT_TOOL_NAMES", "EDIT_TOOL_NAMES"})

# (engine file, exact legacy members before the protocol splice, module-level name or None)
K4_DETECT_LISTS: tuple[tuple[str, frozenset[str], str | None], ...] = (
    ("kernel_bolt.py", frozenset({
        "delete",
        "done",
        "edit",
        "file_create",
        "move",
        "str_replace",
    }), None),
    ("kernel_clause_review.py", frozenset({
        "edit",
        "file_create",
        "replace_module",
        "str_replace",
        "write",
    }), None),
    ("kernel_cog_governor.py", frozenset({
        "edit",
        "file_create",
        "str_replace",
    }), '_EDIT_KINDS'),
    ("kernel_cog_governor.py", frozenset({
        "contract_fact",
        "edit",
        "fact",
        "file_create",
        "new_contract",
        "new_contract_fact",
        "pytest",
        "str_replace",
        "test",
        "test_run",
    }), '_PRODUCT_KINDS'),
    ("kernel_continuation_evidence.py", frozenset({
        "edit",
        "insert_before",
        "replace",
        "replace_module",
        "str_replace",
        "write",
    }), None),
    ("kernel_done_is_submit_authority.py", frozenset({
        "edit",
        "file_create",
        "file_delete",
        "file_move",
        "str_replace",
    }), '_MUTATION_TOOLS'),
    ("kernel_fanout_continuation.py", frozenset({
        "apply_patch",
        "create_file",
        "delete",
        "delete_file",
        "edit",
        "edit_file",
        "file_create",
        "file_delete",
        "file_move",
        "insert_before",
        "move",
        "patch",
        "rename_file",
        "replace",
        "replace_module",
        "str_replace",
        "write",
        "write_file",
    }), '_EDIT_TOOLS'),
    ("kernel_step_contract.py", frozenset({
        "edit",
        "file_create",
        "insert_before",
        "replace_module",
        "str_replace",
    }), None),
    ("kernel_tokenlab_stall_breaker.py", frozenset({
        "edit",
        "file_create",
        "replace",
        "replace_module",
        "str_replace",
        "write",
    }), '_STALLWIRE_EDIT_TOOLS'),
    ("kernel_tokens_row6_runtime.py", frozenset({
        "edit",
        "file_create",
        "replace",
        "replace_module",
        "str_replace",
        "write",
    }), '_EDIT_TOOLS'),
    ("kernel_tokens_row6_state.py", frozenset({
        "edit",
        "file_create",
        "replace",
        "replace_module",
        "str_replace",
        "write",
    }), '_EDIT_TOOLS'),

)
_IDS = [f"{rel}:{attr or 'inline'}:{index}" for index, (rel, _legacy, attr) in enumerate(K4_DETECT_LISTS)]


def _find_lists(path: pathlib.Path, legacy: frozenset[str]) -> list[ast.AST]:
    """Set/tuple/list literals whose string members are exactly ``legacy``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Set, ast.Tuple, ast.List)):
            continue
        if not all(isinstance(e, (ast.Constant, ast.Starred)) for e in node.elts):
            continue
        consts = frozenset(e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str))
        if consts == legacy:
            found.append(node)
    return found


@pytest.mark.parametrize("rel, legacy, _attr", K4_DETECT_LISTS, ids=_IDS)
def test_list_splices_protocol_edit_tools(rel: str, legacy: frozenset[str], _attr: str | None) -> None:
    """Each K4 detection list keeps its legacy members and splices the protocol constant."""
    nodes = _find_lists(ENGINE / rel, legacy)
    assert len(nodes) == 1, f"{rel}: expected one list with the legacy members, found {len(nodes)}"
    node = nodes[0]
    spliced = {e.value.id for e in node.elts if isinstance(e, ast.Starred) and isinstance(e.value, ast.Name)}
    assert spliced & SPLICE_NAMES, f"{rel}:{node.lineno} does not splice PROTOCOL_EDIT_TOOL_NAMES"
    members = legacy | names.PROTOCOL_EDIT_TOOL_NAMES
    assert names.EDIT_SOURCE_TOOL in members
    assert names.REWRITE_SOURCE_TOOL in members
    assert legacy <= members


@pytest.mark.parametrize(
    "rel, attr",
    [(rel, attr) for rel, _legacy, attr in K4_DETECT_LISTS if attr is not None],
    ids=lambda value: str(value),
)
def test_named_constant_counts_protocol_edit_tools(rel: str, attr: str) -> None:
    """Module-level detection sets evaluate to a value that includes both protocol tools."""
    module = importlib.import_module(f"stateframe.engine.{rel[:-len('.py')]}")
    value = getattr(module, attr)
    assert names.EDIT_SOURCE_TOOL in value
    assert names.REWRITE_SOURCE_TOOL in value


def test_step_contract_recognizes_protocol_edit_action_lines() -> None:
    assert is_edit_action_line("edit_source src/app.py")
    assert is_edit_action_line("rewrite_source src/app.py")
    assert not is_edit_action_line("read src/app.py")


def test_bolt_keeps_invalidation_notices_off_protocol_edit_receipts() -> None:
    """Edit-class actions return before touching instance state, so no bolt state is needed."""
    bolt = object.__new__(KernelBolt)
    for action in ("edit", "edit_source", "rewrite_source"):
        assert bolt.take_invalidation_notice(action) is None


def test_governor_counts_protocol_edits_as_product_work() -> None:
    assert _is_productive({"kind": "edit_source", "result": "APPLIED"})
    assert _is_productive({"kind": "rewrite_source", "result": "APPLIED"})
