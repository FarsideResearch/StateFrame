"""Proposal side-calls cross the real dialect boundary without reasoning.

This release has one configurable Decide2SideCallContract, not the newer
side-call registry or forced-ask feature. Exercise the forced-ask tool envelope
through that existing contract without introducing either feature.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import kernel_context_capacity
import kernel_measurement
import kernel_provider_dialect
from kernel_decide2 import DecisionRequest, build_frame, build_micro_request
from kernel_launch_identity_cost import ProviderBoundaryContractError, enforce_provider_body
from kernel_measurement import _finalize_provider_refit
from kernel_request_kind import RequestKindError, bind_auxiliary, guard_physical_send
from kernel_v46_mechanism_wiring import Decide2SideCallContract, _invoke_decide2_provider


TOOL_NAMES = ("decide2_answer", "forced_ask_judge")


def request(contract):
    decision = DecisionRequest(
        decision_id=contract.decision_id, question_kind="done",
        question="Is the task done?", options=("yes", "no", "ABSTAIN"),
        trigger_reason="strict transport regression",
    )
    body = build_micro_request(build_frame(decision), rung="L1", max_tokens=128)
    body["tools"][0]["function"]["name"] = contract.tool_name
    body["tool_choice"]["function"]["name"] = contract.tool_name
    return body


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
@pytest.mark.parametrize("dialect,solver_effort", [
    ("openai-strict", "none"), ("llama.cpp", None), ("llama.cpp", "high"),
])
@pytest.mark.parametrize("attempt_limit", [1, 2])
def test_captured_side_call_sends_explicit_reasoning_off(
        tmp_path, monkeypatch, tool_name, dialect, solver_effort, attempt_limit):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0")
    if solver_effort is None:
        monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("REACT_REASONING_EFFORT", solver_effort)
    monkeypatch.setattr(kernel_provider_dialect, "capabilities_for", lambda *args, **kwargs:
                        kernel_provider_dialect.ProviderCapabilities(dialect=dialect, slots_status=404))
    monkeypatch.setattr(kernel_context_capacity, "_get_json", lambda *args: None)
    contract = Decide2SideCallContract("decision", tool_name=tool_name)
    body = request(contract)
    original = copy.deepcopy(body)
    wires = []

    def fake_http(capture, wire, endpoint, key, **kwargs):
        sent = json.loads(wire)
        assert sent["reasoning_effort"] == "none"
        if dialect == "openai-strict":
            assert "chat_template_kwargs" not in sent
            assert "reasoning_mode" not in sent
            assert "reasoning_budget_tokens" not in sent
            # The unchanged validator consumes the proved canonical view;
            # this local view must never become the physical wire.
            canonical = {**sent, "chat_template_kwargs": {"enable_thinking": False}}
        else:
            canonical = sent
        contract.validate(canonical)
        wires.append(wire)
        if attempt_limit > 1 and len(wires) == 1:
            raise ConnectionResetError("transient pre-response failure")
        return json.dumps({"choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "{}",
        }}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}).encode()

    monkeypatch.setattr(kernel_measurement, "_request_bytes_for_capture", fake_http)
    journal = kernel_measurement.AuditJournal(tmp_path, "strict-side-call", {})
    capture = kernel_measurement.RequestCapture(journal, 0, "snapshot", call_purpose="decide2")
    bind_auxiliary(capture, "decide2")

    def provider(body, *, side_call_contract):
        assert side_call_contract is contract
        return capture.send(body, "https://provider.invalid/v1/chat/completions", "",
                            side_call_contract=side_call_contract,
                            transport_attempt_limit=attempt_limit)

    reply = _invoke_decide2_provider(provider, body, contract)
    assert reply["message"]["content"] == "{}"
    assert body == original
    assert len(wires) == attempt_limit
    assert all(wire == wires[0] for wire in wires)
    assert json.loads(wires[0])["tools"][0]["function"]["name"] == tool_name
    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    submitted = [row for row in events if row["kind"] == "request_submitted"]
    assert [row["transport_attempt"] for row in submitted] == list(range(1, attempt_limit + 1))
    assert len({row["request_id"] for row in submitted}) == 1
    if attempt_limit > 1:
        assert submitted[1]["payload_delta"]["identical_wire"] is True
        assert any(row["kind"] == "transport_error" and row["retry_scheduled"] for row in events)
    assert any(row["kind"] == "response_received" for row in events)


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
@pytest.mark.parametrize("effort", [None, "", "high", "minimal", "invalid"])
def test_strict_side_call_without_confirmed_off_support_never_dispatches(
        tmp_path, monkeypatch, tool_name, effort):
    if effort is None:
        monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("REACT_REASONING_EFFORT", effort)
    monkeypatch.setattr(kernel_provider_dialect, "capabilities_for", lambda *args, **kwargs:
                        kernel_provider_dialect.ProviderCapabilities(slots_status=404))

    def reject_http(*args, **kwargs):
        pytest.fail("unproven reasoning controls reached the provider")

    monkeypatch.setattr(kernel_measurement, "_request_bytes_for_capture", reject_http)
    contract = Decide2SideCallContract("decision", tool_name=tool_name)
    body = request(contract)
    original = copy.deepcopy(body)
    journal = kernel_measurement.AuditJournal(tmp_path, "unproven-side-call", {})
    capture = kernel_measurement.RequestCapture(journal, 0, "snapshot", call_purpose="decide2")
    bind_auxiliary(capture, "decide2")
    with pytest.raises(ProviderBoundaryContractError, match="REACT_REASONING_EFFORT=none"):
        capture.send(body, "https://provider.invalid/v1/chat/completions", "",
                     side_call_contract=contract, transport_attempt_limit=2)
    assert body == original
    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert not any(row["kind"] == "request_submitted" for row in events)


def strict_admission(contract):
    capture = SimpleNamespace(qualified_request=None, _wire_truth_side_call_contract=contract)
    bind_auxiliary(capture, "decide2")
    body = request(contract)
    for key in ("chat_template_kwargs", "reasoning_mode", "reasoning_budget_tokens", "rung", "decision_id"):
        body.pop(key)
    return capture, body


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
@pytest.mark.parametrize("effort", [None, "low", "off", False, 0])
def test_strict_physical_admission_requires_explicit_none(tool_name, effort):
    capture, body = strict_admission(Decide2SideCallContract("decision", tool_name=tool_name))
    if effort is None:
        body.pop("reasoning_effort")
    else:
        body["reasoning_effort"] = effort
    with pytest.raises(ValueError, match="reasoning must be explicitly disabled"):
        guard_physical_send(capture, None, body, json.dumps(body).encode(), dialect="openai-strict")


@pytest.mark.parametrize("corruption,error,reason", [
    ("missing_contract", RequestKindError, "auxiliary_decide2_contract_missing"),
    ("wrong_tool", ValueError, "dedicated answer tool"),
    ("extra_tool", ValueError, "exactly one dedicated tool"),
    ("llama_extension", ProviderBoundaryContractError, "unknown OpenAI Chat Completions request keys"),
])
def test_strict_physical_admission_retains_contract_and_schema_guards(corruption, error, reason):
    contract = Decide2SideCallContract("decision", tool_name="forced_ask_judge")
    capture, body = strict_admission(contract)
    if corruption == "missing_contract":
        capture._wire_truth_side_call_contract = None
    elif corruption == "wrong_tool":
        body["tools"][0]["function"]["name"] = "decide2_answer"
    elif corruption == "extra_tool":
        body["tools"].append(copy.deepcopy(body["tools"][0]))
    else:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    with pytest.raises(error, match=reason):
        guard_physical_send(capture, None, body, json.dumps(body).encode(), dialect="openai-strict")


def test_wire_conversion_does_not_relax_canonical_contract():
    contract = Decide2SideCallContract("decision", tool_name="forced_ask_judge")
    capture, body = strict_admission(contract)
    with pytest.raises(ValueError, match="reasoning must be explicitly disabled"):
        contract.validate(body)
    with pytest.raises(ValueError, match="reasoning must be explicitly disabled"):
        guard_physical_send(capture, None, body, json.dumps(body).encode(), dialect="llama.cpp")


def test_strict_side_call_policy_survives_repeated_conversion_and_refit(monkeypatch):
    monkeypatch.setenv("REACT_REASONING_EFFORT", " NONE ")
    body = request(Decide2SideCallContract("decision", tool_name="forced_ask_judge"))
    strict = enforce_provider_body(body, dialect="openai-strict", side_call=True)
    assert strict["reasoning_effort"] == "none"
    assert enforce_provider_body(strict, dialect="openai-strict", side_call=True) == strict
    assert _finalize_provider_refit(strict, profile=None, dialect="openai-strict", side_call=True) == strict
    monkeypatch.setenv("REACT_REASONING_EFFORT", "high")
    assert enforce_provider_body(body, dialect="openai-strict")["reasoning_effort"] == "high"
    with pytest.raises(ProviderBoundaryContractError, match="REACT_REASONING_EFFORT=none"):
        enforce_provider_body(strict, dialect="openai-strict", side_call=True)
    with pytest.raises(ProviderBoundaryContractError, match="REACT_REASONING_EFFORT=none"):
        _finalize_provider_refit(strict, profile=None, dialect="openai-strict", side_call=True)
