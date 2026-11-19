"""Protocol edit calls count as repair actions in completion policy."""
from types import SimpleNamespace

import pytest

import kernel_completion_policy as completion
import kernel_done_challenge_once as challenge
import kernel_readiness_authority as readiness


@pytest.mark.parametrize("tool", ["str_replace", "edit_source", "rewrite_source"])
def test_readiness_recognizes_edit_calls(tool: str) -> None:
    call = {"function": {"name": tool}}

    assert call["function"]["name"] in readiness.EDIT_TOOLS


@pytest.mark.parametrize(
    ("tool", "can_repair"),
    [("str_replace", True), ("edit_source", True), ("rewrite_source", True), ("run_tests", False)],
)
def test_done_challenge_recognizes_repair_actions(tool: str, can_repair: bool) -> None:
    runtime = SimpleNamespace(
        stateframe_repair_blocks_done_mode="on",
        _readiness_decision=SimpleNamespace(repair_blocks_done_active=True),
    )
    menu = ("done", tool)

    assert challenge.repair_hold(runtime, menu) is can_repair
    assert challenge.has_qualifying_action(menu) is can_repair


@pytest.mark.parametrize(
    ("tool", "action"),
    [("str_replace", "challenge"), ("edit_source", "challenge"),
     ("rewrite_source", "challenge"), ("run_tests", "stop")],
)
def test_unified_stop_challenges_when_menu_can_repair(tool: str, action: str) -> None:
    runtime = SimpleNamespace(
        completion_menu=("done", tool),
        binding=lambda: {"artifact": "candidate"},
    )

    decision = completion.evaluate(runtime, reason_codes=("ORACLE_TESTS_RED",))

    assert decision.repair_on_menu is (action == "challenge")
    assert decision.action == action
