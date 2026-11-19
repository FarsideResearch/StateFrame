"""Protocol edits reach the same engine detectors as legacy span edits."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import kernel_aha10_honest_loop as honest_loop
import kernel_claim_grounding as claim_grounding
import kernel_continuation_menu_authority as continuation_menu
import kernel_green_lock as green_lock
import kernel_green_lock_repair as green_lock_repair
import kernel_green_menu_open_obligation as open_obligation
import kernel_greenlock_menu_authority as greenlock_menu
import kernel_greenlock_terminal_fixes as terminal_fixes
import kernel_information_policy as information_policy
import kernel_mutate_projection as mutate
import kernel_postgreen_no_evidence as no_evidence
import kernel_postgreen_no_veto as no_veto
import kernel_qgates as qgates
import kernel_tool_surface as tool_surface


@pytest.fixture(params=["str_replace", "edit_source", "rewrite_source"])
def edit_tool(request):
    return request.param


def _schema(name):
    return {"type": "function", "function": {"name": name}}


def test_honest_loop_detects_the_available_write_route(edit_tool):
    assert honest_loop.capability_routes([_schema(edit_tool)])["write"] == edit_tool


def test_honest_loop_prefers_legacy_route_when_both_are_available():
    tools = [_schema("edit_source"), _schema("rewrite_source"), _schema("str_replace")]
    assert honest_loop.capability_routes(tools)["write"] == "str_replace"


def test_claim_grounding_rejects_ungrounded_edit_targets(tmp_path, edit_tool):
    grounding = claim_grounding.ClaimGrounding(tmp_path)
    result = grounding.inspect_tool(edit_tool, {"path": "missing.py"})
    assert result["receipt"]["code"] == "UNGROUNDED_CLAIM"


def test_green_lock_counts_applied_product_receipts(edit_tool):
    owner = SimpleNamespace(_receipts=[{
        "tool": edit_tool, "result": "APPLIED", "path": "app.py",
    }])
    lock = green_lock.GreenLock(SimpleNamespace(owner=owner), mode="off")
    assert lock._has_applied_product_edit()


def test_greenlock_menu_has_one_edit_identity(edit_tool):
    assert greenlock_menu.canonical_action(edit_tool) == "edit"


def test_greenlock_still_checks_the_exact_protocol_runner_capability(monkeypatch):
    monkeypatch.setattr(tool_surface, "dispatch_admissible", lambda name, runtime: name == "edit")
    lock = SimpleNamespace(runtime=SimpleNamespace())
    decision = greenlock_menu.decide(lock, "edit_source", for_menu=True)
    assert not decision.allowed and decision.rule == "runner_capability_unavailable"


def test_greenlock_readiness_floor_keeps_edit_capabilities(monkeypatch, edit_tool):
    monkeypatch.setattr(continuation_menu, "enabled", lambda runtime: True)
    runtime = SimpleNamespace(
        stateframe_readiness_authority_mode="on",
        stateframe_hard_budget_only_terminals_mode="on",
        continuation=SimpleNamespace(engine=SimpleNamespace(active_intention_id=lambda: "root")),
    )
    lock = SimpleNamespace(
        runtime=runtime,
        terminal_engaged=False,
        _readiness_snapshot=lambda: SimpleNamespace(dispositions={edit_tool: "deny"}),
    )
    decision = greenlock_menu.Decision(edit_tool, True, "admitted")
    assert greenlock_menu._intersect_readiness(lock, decision) == decision
    assert runtime._floor_kept_pending == {edit_tool}


def test_repair_recognizes_an_edit_linked_to_a_counterexample(monkeypatch, edit_tool):
    monkeypatch.setattr(green_lock_repair, "readiness_repair_counterexamples",
                        lambda runtime, frame=None: ({"targets": {"paths": ["app.py"]}},))
    assert green_lock_repair.action_linked_to_counterexample(edit_tool, {"path": "app.py"})
    assert not green_lock_repair.action_linked_to_counterexample(edit_tool, {"path": "other.py"})


def test_green_menu_watchdog_counts_an_offered_edit(monkeypatch, edit_tool):
    monkeypatch.setattr(open_obligation, "enabled", lambda runtime: True)
    monkeypatch.setattr(open_obligation, "counterexample_ids", lambda runtime: ("open-check",))
    runtime = SimpleNamespace()
    open_obligation.observe_model_menu(runtime, [_schema("done")])
    open_obligation.observe_model_menu(runtime, [_schema(edit_tool)])
    assert runtime._green_menu_open_obligation_missing_edit == 0


def test_counterexample_menu_projections_keep_existing_restore_sets(monkeypatch):
    monkeypatch.setattr(open_obligation, "open_counterexamples",
                        lambda runtime, frame=None: ("open-check",))
    monkeypatch.setattr(green_lock_repair, "enabled", lambda: True)
    monkeypatch.setattr(green_lock_repair, "readiness_repair_counterexamples",
                        lambda runtime, frame: ("open-check",))
    current = [_schema("done")]
    recovery = [_schema("edit_source"), _schema("str_replace"), _schema("run_tests")]
    expected = current + recovery[1:]
    assert open_obligation.ensure_open_menu(current, recovery) == expected
    assert green_lock_repair.repair_terminal_menu(current, recovery) == expected


def test_first_edit_rule_recognizes_product_edits(edit_tool):
    lock = SimpleNamespace(
        mode="terminal",
        runtime=SimpleNamespace(stateframe_greenlock_first_edit_mode="on"),
        terminal_engaged=True,
        _repair_blocks_done_refusal=lambda: None,
        _workspace_path=lambda path: path,
    )
    assert terminal_fixes.menu_rule(
        lock, edit_tool, {"path": "app.py"}, for_menu=False,
    ) == terminal_fixes.RULE_FIRST_EDIT


def test_no_red_terminal_settles_an_unexecuted_edit(monkeypatch, edit_tool):
    monkeypatch.setenv(terminal_fixes.NO_RED_TERMINAL_ENV, "on")
    effect = SimpleNamespace(tool=edit_tool, read_only=False)
    record = {"result": "REFUSED", "before_binding": "r1", "after_binding": "r1"}
    assert terminal_fixes.settle_unexecuted_edit(effect, record, "r1") == "not_applied"


def test_information_policy_checks_edit_command_paths(tmp_path, edit_tool):
    policy = information_policy.InformationPolicy({"_bridge_capture": True})
    assert policy.command_allowed(tmp_path, f"{edit_tool} app.py")
    assert not policy.command_allowed(tmp_path, f"{edit_tool} ../outside.py")


def test_mutation_authoring_keeps_wire_exclusions_in_sync_with_dispatch():
    runtime = SimpleNamespace(pass_gates_vector=lambda tools: SimpleNamespace(
        recoveries=({"tool": "str_replace"}, {"tool": "probe"}),
    ))
    tools = [_schema("mutate"), _schema("str_replace"), _schema("probe")]
    offered = mutate.authoring_tools(tools, runtime)
    assert offered == [_schema("mutate"), _schema("probe")]
    assert [tool for tool in tools if mutate.authoring_dispatch_allowed(
        tool["function"]["name"], runtime, tools,
    )] == offered


def test_postgreen_evidence_gate_detects_redundant_edits(monkeypatch, edit_tool):
    monkeypatch.setattr(no_evidence, "clean_current_revision", lambda lock: True)
    lock = SimpleNamespace(runtime=SimpleNamespace(stateframe_postgreen_no_evidence_mode="on"))
    assert no_evidence.decide(lock, edit_tool) == (False, no_evidence.RULE)


def test_postgreen_no_veto_admits_product_edits(edit_tool):
    lock = SimpleNamespace(
        mode="terminal",
        terminal_engaged=True,
        runtime=SimpleNamespace(stateframe_postgreen_no_veto_mode="on"),
    )
    verdict = no_veto.decide(lock, edit_tool, {"path": "app.py"})
    assert verdict is not None and verdict.allowed and verdict.rule == no_veto.RULE_ADMITTED


def test_question_gate_blocks_an_edit_before_first_edit_gate(monkeypatch, edit_tool):
    monkeypatch.setenv(qgates.FLAG, "on")
    reply = qgates.guard_action(SimpleNamespace(), edit_tool, {"path": "app.py"})
    assert reply is not None
    assert json.loads(reply)["gate"] == "before_first_product_edit"


def test_question_gate_counts_an_edit_observation(monkeypatch, edit_tool):
    monkeypatch.setenv(qgates.FLAG, "on")
    runtime = SimpleNamespace()
    qgates.after_observation(runtime, f"{edit_tool} app.py", "applied")
    assert qgates.state(runtime)["belief_path_writes"] == {"app.py": 1}
