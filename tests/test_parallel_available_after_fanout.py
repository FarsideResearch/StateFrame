"""DR-H: a completed parallel fanout retires only its own plan, never the tool.

A run that has finished one fanout must still offer `parallel` on the final menu,
accept a genuinely new plan, and refuse a re-dispatch of the identical completed plan.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

import kernel_completion_owner as completion_owner
import kernel_done_is_submit_authority as done_authority
import kernel_request_tool_surface as request_surface
import kernel_stable_tool_head as head
from probes.react_repair import _parallel_result_is_terminal

PLAN_A = {"tasks": [{"id": "csv", "owns": ["src/csv_reader.py"], "check": "tests/test_csv.py"}]}
PLAN_A_REORDERED = {"tasks": [{"check": "tests/test_csv.py", "owns": ["src/csv_reader.py"], "id": "csv"}]}
PLAN_B = {"tasks": [{"id": "json", "owns": ["src/json_reader.py"], "check": "tests/test_json.py"}]}


def _tool(name: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": name,
        "parameters": {"type": "object", "properties": {}}}}


def _names(tools: list[dict]) -> list[str]:
    return [tool["function"]["name"] for tool in tools]


def _completed_fanout() -> SimpleNamespace:
    return SimpleNamespace(
        admission=SimpleNamespace(status="ADMITTED"),
        outcomes=(SimpleNamespace(status="COMMITTED"),),
        final_check="PASS",
        integration="VERIFIED",
    )


@pytest.fixture
def surface_env(monkeypatch):
    monkeypatch.setenv(head.MODE_ENV, "on")
    monkeypatch.setenv("STATEFRAME_TOOL_MENU_MEMO", "off")
    monkeypatch.setenv("STATEFRAME_PARALLEL_DROP_TOOL_AFTER_TERMINAL", "on")
    monkeypatch.setattr(done_authority, "enabled", lambda _runtime: False)
    monkeypatch.setattr(done_authority, "candidate_exists", lambda _runtime: False)
    monkeypatch.setattr(completion_owner, "bind_final_tool_surface", lambda *a, **k: True)


def _author_menu(runtime: object, request_id: str, current: list[dict]) -> list[dict]:
    body = {"messages": [{"role": "user", "content": "task"}],
            "tools": copy.deepcopy(current)}
    surface = request_surface.author_final_tool_surface(
        SimpleNamespace(), runtime, body, request_id=request_id, bind_current_surface=True)
    assert surface is not None
    return body["tools"]


def test_parallel_stays_on_final_menu_after_completed_fanout(surface_env):
    # The production dispatcher marks the run with this flag after a verified fanout.
    runtime = SimpleNamespace(_parallel_parent_terminal=True)
    menu = [_tool("read"), _tool("edit"), _tool("parallel")]
    head.remember_offerable_tools(runtime, menu)

    assert "parallel" in _names(_author_menu(runtime, "request-2", menu))


def test_new_plan_dispatches_and_identical_completed_plan_is_refused(surface_env):
    from kernel_parallel_retirement import refusal_for, retire_completed_plan

    runtime = SimpleNamespace()
    terminal = _parallel_result_is_terminal(_completed_fanout())
    assert retire_completed_plan(runtime, PLAN_A, terminal, enabled=True)

    assert refusal_for(runtime, PLAN_B, enabled=True) is None
    refusal = refusal_for(runtime, PLAN_A, enabled=True)
    assert refusal is not None and refusal.startswith("REFUSED PLAN_ALREADY_COMPLETED")
    assert refusal_for(runtime, PLAN_A_REORDERED, enabled=True) == refusal


def test_incomplete_fanout_does_not_retire_its_plan(surface_env):
    from kernel_parallel_retirement import refusal_for, retire_completed_plan

    runtime = SimpleNamespace()
    unverified = SimpleNamespace(
        admission=SimpleNamespace(status="ADMITTED"),
        outcomes=(SimpleNamespace(status="COMMITTED_UNVERIFIED"),),
        final_check="PASS", integration="UNVERIFIED")
    terminal = _parallel_result_is_terminal(unverified)
    assert not retire_completed_plan(runtime, PLAN_A, terminal, enabled=True)
    assert refusal_for(runtime, PLAN_A, enabled=True) is None


def test_retirement_is_scoped_to_one_run(surface_env):
    from kernel_parallel_retirement import refusal_for, retire_completed_plan

    first_run, second_run = SimpleNamespace(), SimpleNamespace()
    retire_completed_plan(first_run, PLAN_A, True, enabled=True)
    assert refusal_for(first_run, PLAN_A, enabled=True) is not None
    assert refusal_for(second_run, PLAN_A, enabled=True) is None


def test_retirement_switch_off_disables_refusal_and_invalid_value_fails_loud(monkeypatch):
    from kernel_parallel_retirement import refusal_for, retire_completed_plan
    from probes.react_repair import _parallel_plan_retirement_enabled

    runtime = SimpleNamespace()
    monkeypatch.setenv("STATEFRAME_PARALLEL_DROP_TOOL_AFTER_TERMINAL", "off")
    switch = _parallel_plan_retirement_enabled()
    assert switch is False
    assert not retire_completed_plan(runtime, PLAN_A, True, enabled=switch)
    assert refusal_for(runtime, PLAN_A, enabled=switch) is None

    monkeypatch.setenv("STATEFRAME_PARALLEL_DROP_TOOL_AFTER_TERMINAL", "maybe")
    with pytest.raises(ValueError):
        _parallel_plan_retirement_enabled()
