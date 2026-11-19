"""Behavioral parity between legacy and handle-addressed edit calls."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import kernel_convergence_observer as convergence
import kernel_frame_routes as routes
from probes.react_repair import parse_dsml_tool_calls


@pytest.fixture(params=["str_replace", "edit_source", "rewrite_source"])
def editor(request):
    return request.param


def test_convergence_resets_repeat_streak_after_an_applied_edit(editor):
    runtime = SimpleNamespace(stateframe_convergence_observer_mode="on")
    applied = {"tool": editor, "status": "APPLIED", "byte_changed": True}

    assert convergence.observe_closed_action(runtime, editor, "", receipt=applied) is None
    assert convergence.observe_closed_action(runtime, editor, "", receipt=applied) is None
    assert convergence.decision(runtime) is None


def test_convergence_retains_an_alternate_edit_route(editor):
    runtime = SimpleNamespace(stateframe_convergence_observer_mode="on")
    refused = {"tool": "edit", "status": "REJECTED"}
    convergence.observe_closed_action(runtime, "edit", "", receipt=refused)
    decision = convergence.observe_closed_action(runtime, "edit", "", receipt=refused)
    assert decision is not None
    assert decision.clearing_route == ("str_replace", "edit", "read_check_receipt")

    tools = [{"function": {"name": name}} for name in ("edit", editor, "done")]
    assert convergence.apply_menu(runtime, tools) == tools[1:]


def test_convergence_uses_edit_repair_order_for_rejected_edits(editor):
    runtime = SimpleNamespace(stateframe_convergence_observer_mode="on")
    rejected = {"tool": editor, "status": "REJECTED"}
    convergence.observe_closed_action(runtime, editor, "", receipt=rejected)
    decision = convergence.observe_closed_action(runtime, editor, "", receipt=rejected)

    assert decision is not None
    assert decision.clearing_route == ("str_replace", "edit", "read_check_receipt")


def test_routing_escalates_edits_in_mixed_calls(editor, monkeypatch):
    endpoint = "http://127.0.0.1:8080/v1"
    reader = routes.Route("read", "run", None, {"endpoint": endpoint, "model": "reader"})
    monkeypatch.setattr(routes, "_roles", {"edit": {"endpoint": endpoint, "model": "editor"}})
    calls = [{"function": {"name": name, "arguments": "{}"}}
             for name in ("read", editor, "done")]

    assert routes.classify_calls(calls) == "edit"
    assert routes.escalation_target(reader, calls) == "edit"


@pytest.mark.parametrize("bare", [False, True])
def test_r1_parser_recognizes_enabled_edit_calls(monkeypatch, editor, bare):
    monkeypatch.setenv("STATEFRAME_EDIT_PROTOCOL", "handles")
    arguments = {
        "str_replace": {"path": "pricing.py", "old_str": "25", "new_str": "30"},
        "edit_source": {"edits": [{"handle": "q1h1", "old_text": "25", "new_text": "30"}]},
        "rewrite_source": {"handle": "q1h1", "new_text": "minimum = 30"},
    }[editor]
    reply = (f"{editor}\n{json.dumps(arguments)}" if bare else
             json.dumps({"function": editor, **arguments}))

    calls = parse_dsml_tool_calls(reply)

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == editor
    assert json.loads(calls[0]["function"]["arguments"]) == arguments


@pytest.mark.parametrize("name", ["edit_source", "rewrite_source"])
def test_r1_parser_keeps_protocol_calls_disabled_by_default(monkeypatch, name):
    monkeypatch.delenv("STATEFRAME_EDIT_PROTOCOL", raising=False)

    assert parse_dsml_tool_calls(json.dumps({"function": name})) == []
