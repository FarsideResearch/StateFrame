"""Completion-owner fallback boundaries preserve contracts and typed gaps."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

import kernel_agent_tools
import kernel_completion_owner as owner
from kernel_completion_owner import RequestActionContractError
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_request_seal import OwnerPresentation, RequestSealError, freeze


def _journal():
    events = []
    journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    return SimpleNamespace(journal=journal), events


def _tool(name: str) -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def _raise(error: BaseException):
    def raised(*_args, **_kwargs):
        raise error
    return raised


def _bind_inputs():
    tool = _tool("read")
    contract = {
        "contract_kind": owner.LEAN_REQUEST_CONTRACT_KIND,
        "route": "lean",
        "request_id": "request-1",
        "admitted_tool_schemas": [tool],
        "advertised_actions": ["read"],
        "owner_blocks": {},
    }
    contract["decision_digest"] = owner._contract_decision_digest(contract)
    capture = SimpleNamespace(request_action_contract=contract)
    surface = {
        "request_id": "request-1",
        "tools": [tool],
        "names": ["read"],
        "tool_schema_digest": owner._digest([tool]),
    }
    return capture, surface


def _terminal_menu_inputs(monkeypatch, error):
    import kernel_done_is_submit_authority as done_authority
    import kernel_greenlock_menu_authority as menu_authority

    monkeypatch.setattr(done_authority, "enabled", lambda _runtime: True)
    monkeypatch.setattr(done_authority, "candidate_exists", lambda _runtime: True)
    monkeypatch.setattr(
        done_authority, "project_candidate_tools",
        lambda _runtime, tools, *, full_tools: list(tools),
    )
    monkeypatch.setattr(menu_authority, "finalize_wire_menu_authority", _raise(error))
    runtime, events = _journal()
    return runtime, [_tool("done")], events


def _invoke_boundary(monkeypatch, boundary, error):
    runtime, events = _journal()
    if boundary == "route_alignment":
        monkeypatch.setattr(owner, "completion_route_decision", _raise(error))
        return lambda: owner.align_tools_to_completion_route(runtime, [_tool("edit")]), events
    if boundary == "final_surface_binding":
        capture, surface = _bind_inputs()
        monkeypatch.setattr(owner, "_request_contract_revision_rebind_mode", lambda *_args: "off")
        monkeypatch.setattr(owner, "_contract_revision", _raise(error))
        return lambda: owner.bind_final_tool_surface(
            runtime, capture, surface, messages=[],
        ), events
    if boundary == "terminal_menu_authority":
        runtime, tools, events = _terminal_menu_inputs(monkeypatch, error)
        return lambda: owner.terminal_tool_menu(runtime, tools), events
    raise AssertionError(f"unknown boundary: {boundary}")


@pytest.mark.parametrize(
    "boundary", ["route_alignment", "final_surface_binding", "terminal_menu_authority"],
)
def test_request_contract_errors_propagate_at_each_boundary(monkeypatch, boundary):
    error = RequestActionContractError("contract must remain visible")
    invoke, events = _invoke_boundary(monkeypatch, boundary, error)

    with pytest.raises(RequestActionContractError, match="contract must remain visible"):
        invoke()

    assert events == []


@pytest.mark.parametrize("error_type", [MemoryError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "boundary", ["route_alignment", "final_surface_binding", "terminal_menu_authority"],
)
def test_process_control_exceptions_propagate_at_each_boundary(
        monkeypatch, boundary, error_type):
    invoke, events = _invoke_boundary(monkeypatch, boundary, error_type("must propagate"))

    with pytest.raises(error_type, match="must propagate"):
        invoke()

    assert events == []


@pytest.mark.parametrize(
    ("failure", "error_type", "reason_code"),
    [
        ("binding_key", KeyError, "COMPLETION_ROUTE_ALIGNMENT_UNAVAILABLE"),
        ("tool_schema", StopIteration, "COMPLETION_ROUTE_ALIGNMENT_UNAVAILABLE"),
        ("owner_state", OverflowError, "COMPLETION_ROUTE_ALIGNMENT_UNAVAILABLE"),
    ],
)
def test_route_alignment_records_real_unavailable_failures_once(
        monkeypatch, failure, error_type, reason_code):
    runtime_base, events = _journal()
    journal = runtime_base.journal
    if failure == "binding_key":
        binding_runtime = SimpleNamespace(
            projector_kind="basic",
            _binding_cache=None,
            _binding_epoch=0,
            _host_step_budget2_cache_enabled=lambda: False,
            target=None,
            journal=SimpleNamespace(manifest={}),
        )
        runtime = SimpleNamespace(
            journal=journal,
            binding=lambda: StateFrameRuntime.binding(binding_runtime),
        )
        def trigger(current):
            return owner._revision(current)
    elif failure == "tool_schema":
        malformed_host_tools = SimpleNamespace(
            custom=frozenset({"private_custom"}), custom_schemas=(), names=frozenset(),
        )
        monkeypatch.setattr(kernel_agent_tools, "service", lambda: malformed_host_tools)
        custom_tool = _tool("private_custom")
        runtime = SimpleNamespace(journal=journal)
        def trigger(_current):
            return owner.validate_tool_arguments(
                [custom_tool], "private_custom", {},
            )
    else:
        runtime = SimpleNamespace(
            journal=journal,
            _completion_owner_state={
                **owner.initial_state(), "done_only_offer_count": float("inf"),
            },
        )
        def trigger(current):
            return owner._state(current)

    if failure == "tool_schema":
        monkeypatch.setattr(
            owner, "completion_route_decision",
            lambda current, *_args, **_kwargs: trigger(current),
        )
    original = [_tool("edit")]

    assert owner.align_tools_to_completion_route(runtime, original) == original
    alignment_events = [event for event in events if event[0] == "completion_owner_unavailable"]
    assert alignment_events == [("completion_owner_unavailable", {
        "producer": "kernel_completion_owner",
        "event_schema": owner.SCHEMA,
        "dispatch": False,
        "operation": "completion_route_alignment",
        "reason_code": reason_code,
        "reason": f"completion route alignment unavailable: {error_type.__name__}",
        "error_type": error_type.__name__,
    })]


def test_final_surface_revision_rebind_records_unavailability_and_keeps_legacy_contract(monkeypatch):
    runtime, events = _journal()
    capture, surface = _bind_inputs()
    monkeypatch.setattr(owner, "_request_contract_revision_rebind_mode", lambda *_args: "off")
    monkeypatch.setattr(owner, "_contract_revision", _raise(RuntimeError("private detail")))

    assert owner.bind_final_tool_surface(runtime, capture, surface, messages=[])
    assert events == [("completion_owner_unavailable", {
        "producer": "kernel_completion_owner",
        "event_schema": owner.SCHEMA,
        "dispatch": False,
        "operation": "final_surface_revision_rebind",
        "reason_code": "FINAL_SURFACE_REVISION_REBIND_UNAVAILABLE",
        "reason": "final-surface revision rebind unavailable: RuntimeError",
        "error_type": "RuntimeError",
    })]
    assert "private detail" not in repr(events)


def test_final_surface_rebinds_owner_presentation_with_its_contract():
    runtime, _events = _journal()
    capture, surface = _bind_inputs()
    original_contract = capture.request_action_contract
    presentation = OwnerPresentation(
        hashlib.sha256(freeze(original_contract)).hexdigest(), None, None,
    )
    capture.owner_presentation = presentation

    assert owner.bind_final_tool_surface(runtime, capture, surface, messages=[])

    assert capture.request_action_contract != original_contract
    capture.owner_presentation.validate(capture.request_action_contract)
    with pytest.raises(RequestSealError, match="another contract"):
        presentation.validate(capture.request_action_contract)


@pytest.mark.parametrize("changed_field", [
    "budget_wire_digest", "evidence_pin_wire_digest",
])
def test_owner_presentation_rebind_refuses_changed_owner_proof(changed_field):
    previous = {
        "budget_wire_digest": owner._digest("saved limits"),
        "evidence_pin_wire_digest": owner._digest(["saved pin"]),
    }
    presentation = OwnerPresentation(
        hashlib.sha256(freeze(previous)).hexdigest(), "saved limits", "saved pin",
    )
    current = dict(previous)
    current[changed_field] = owner._digest("different owner proof")

    with pytest.raises(RequestSealError, match="differ.* from rebound contract"):
        presentation.rebind(previous, current)
    presentation.validate(previous)


@pytest.mark.parametrize("removed_field", [
    "budget_wire_digest", "evidence_pin_wire_digest",
])
def test_owner_presentation_rebind_refuses_removed_owner_proof(removed_field):
    previous = {
        "budget_wire_digest": owner._digest("saved limits"),
        "evidence_pin_wire_digest": owner._digest(["saved pin"]),
    }
    presentation = OwnerPresentation(
        hashlib.sha256(freeze(previous)).hexdigest(), "saved limits", "saved pin",
    )
    current = dict(previous)
    del current[removed_field]

    with pytest.raises(RequestSealError, match="removed from rebound contract"):
        presentation.rebind(previous, current)
    presentation.validate(previous)


def test_owner_presentation_rebind_refuses_unrelated_previous_contract():
    presentation = OwnerPresentation(
        hashlib.sha256(freeze({"route": "edit"})).hexdigest(), None, None,
    )

    with pytest.raises(RequestSealError, match="another contract"):
        presentation.rebind({"route": "submit"}, {"route": "submit"})


def test_final_surface_binding_records_typed_unavailability_once(monkeypatch):
    runtime, events = _journal()
    capture, surface = _bind_inputs()
    monkeypatch.setattr(owner, "_verify_contract_integrity", _raise(KeyError("private detail")))

    assert not owner.bind_final_tool_surface(runtime, capture, surface, messages=[])
    assert events == [("completion_owner_unavailable", {
        "producer": "kernel_completion_owner",
        "event_schema": owner.SCHEMA,
        "dispatch": False,
        "operation": "final_tool_surface_binding",
        "reason_code": "FINAL_TOOL_SURFACE_BINDING_UNAVAILABLE",
        "reason": "final tool-surface binding unavailable: KeyError",
        "error_type": "KeyError",
    })]
    assert "private detail" not in repr(events)


def test_terminal_menu_authority_records_unavailability_and_preserves_menu(monkeypatch):
    error = RuntimeError("private detail")
    runtime, tools, events = _terminal_menu_inputs(monkeypatch, error)

    assert owner.terminal_tool_menu(runtime, tools) == tools
    assert events == [("completion_owner_unavailable", {
        "producer": "kernel_completion_owner",
        "event_schema": owner.SCHEMA,
        "dispatch": False,
        "operation": "terminal_menu_authority",
        "reason_code": "TERMINAL_MENU_AUTHORITY_UNAVAILABLE",
        "reason": "terminal menu authority unavailable: RuntimeError",
        "error_type": "RuntimeError",
    })]
    assert "private detail" not in repr(events)
