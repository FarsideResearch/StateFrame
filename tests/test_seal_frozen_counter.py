"""The allocation and final proof must use one request measurement authority."""
from __future__ import annotations

import copy
import json
import math
import time
from types import SimpleNamespace

import pytest

from kernel_frame_allocation_capture import FrozenRequestCounter, final_request_tokens
from kernel_frame_request_allocation import FrameRequestAllocationError
from kernel_final_wire_budget import reconcile_frozen_budget
from kernel_request_diet import RequestTokenMeter


def _body(size=1000):
    return {"messages": [{"role": "user", "content": "x" * size}],
            "tools": [], "max_tokens": 24000}


def _capture(body, meter, *, method, tokens):
    if method == "tokenize":
        from kernel_frame_allocation_capture import _remember_exact_pair
        _remember_exact_pair(body, meter, tokens)
    return SimpleNamespace(
        request_id="seal-r1", request_action_contract={"admitted_tool_schemas": []},
        request_budget_final_wire={}, _frame_candidate_inputs={"token_ceiling": 33344},
        _frame_candidate_counter=FrozenRequestCounter.capture(
            body, meter, tokens, method, "seal-r1"),
        _frame_candidate_fallback_fit=method == "calibrated_bytes",
    )


def _proof(monkeypatch, body, meter, capture):
    monkeypatch.setattr("kernel_request_budget._request_token_meter",
                        lambda *_args, **_kwargs: meter)
    return reconcile_frozen_budget(capture, SimpleNamespace(), body)


def test_prefit_menu_binding_defers_final_proof_until_after_fit(monkeypatch):
    import kernel_completion_owner
    import kernel_done_is_submit_authority
    import kernel_final_wire_budget
    import kernel_read_lockout
    import kernel_request_tool_surface
    from kernel_wire_identity import next_wire_id

    monkeypatch.setenv("STATEFRAME_TOOL_MENU_MEMO", "off")
    monkeypatch.setenv("STATEFRAME_STABLE_TOOL_HEAD", "off")
    monkeypatch.setenv("STATEFRAME_REQUEST_TOOL_FLOOR", "off")
    monkeypatch.delenv("SF_LAB_DETFIX3", raising=False)
    monkeypatch.setattr(kernel_done_is_submit_authority, "enabled", lambda _runtime: False)
    monkeypatch.setattr(kernel_done_is_submit_authority, "candidate_exists", lambda _runtime: False)
    monkeypatch.setattr(kernel_read_lockout, "filter_tools", lambda _runtime, tools: list(tools))
    monkeypatch.setattr(kernel_completion_owner, "bind_final_tool_surface",
                        lambda *_args, **_kwargs: True)

    proofs = []

    class Journal:
        def __init__(self):
            self._sequence = 0
            self.events = []

        def emit(self, kind, **fields):
            self._sequence += 1
            self.events.append({"kind": kind, **fields})

    def counted_proof(capture, _runtime, body):
        status = "HOST_BLOCKED" if len(body["messages"][0]["content"]) > 5000 else "READY"
        receipt = {
            "status": status,
            "final_validation_failures": (
                ["request_tokens_exceed_max_prompt_tokens"] if status == "HOST_BLOCKED" else []),
            "tool_schema_digest": capture.final_tool_surface["tool_schema_digest"],
            "tool_names": capture.final_tool_surface["names"],
        }
        capture.request_budget_final_wire = receipt
        proofs.append(status)
        capture.journal.emit("request_budget_final_wire", status=status,
                             phase="frozen_transport_revalidation")
        return receipt

    monkeypatch.setattr(kernel_final_wire_budget, "reconcile_frozen_budget", counted_proof)
    tool = {"type": "function", "function": {"name": "read", "description": "Read a file",
            "parameters": {"type": "object", "properties": {}}}}

    def bind(*, final_fit_pending):
        capture = SimpleNamespace(request_id="seal-r1", request_action_contract={},
                                  request_budget_final_wire={"status": "READY"},
                                  journal=Journal())
        runtime = SimpleNamespace()
        body = {"messages": [{"role": "user", "content": "x" * 10000}],
                "tools": [copy.deepcopy(tool)]}
        surface = kernel_request_tool_surface.author_final_tool_surface(
            capture, runtime, body, request_id=capture.request_id,
            bind_current_surface=True, final_fit_pending=final_fit_pending)
        assert surface is not None
        return capture, runtime, body

    # Planted control: the old pre-fit call emits a rejected final proof.
    control, _runtime, _body_before_fit = bind(final_fit_pending=False)
    assert control.request_budget_final_wire["status"] == "HOST_BLOCKED"
    assert proofs == ["HOST_BLOCKED"]

    capture, runtime, body = bind(final_fit_pending=True)
    assert proofs == ["HOST_BLOCKED"]
    assert capture.request_budget_final_wire["status"] == "READY"
    assert capture.journal.events == [{
        "kind": "request_budget_final_wire", "producer": "kernel_request_budget",
        "request_id": "seal-r1", "step": None, "dispatch": False,
        "phase": "pre_fit_tool_surface", "status": "PRE_FIT_DEFERRED",
        "tool_names": ["read"],
        "tool_schema_digest": capture.final_tool_surface["tool_schema_digest"],
    }]
    # Planted skipped-event control: the next provider-visible ID changes.
    expected_next_id = next_wire_id(capture.journal, "request", 2)
    skipped = Journal()
    with pytest.raises(AssertionError):
        assert next_wire_id(skipped, "request", 2) == expected_next_id
    body["messages"][0]["content"] = "x" * 1000
    assert counted_proof(capture, runtime, body)["status"] == "READY"
    assert proofs == ["HOST_BLOCKED", "READY"]


@pytest.mark.parametrize("layout", ["off", "on"])
def test_exact_allocation_survives_tokenizer_failure(monkeypatch, layout):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", layout)
    body = _body(80000)
    meter = RequestTokenMeter(tokenizer=lambda _text: 20000)
    meter.bind_request("seal-r1", "seal-r1")
    tokens, method = meter.measure(body)
    assert method == "tokenize"
    capture = _capture(body, meter, method=method, tokens=tokens)
    monkeypatch.setattr(meter, "_availability", "unavailable")
    # Control: the live cold ratio would reject this unchanged fitted body.
    assert meter.measure(body)[0] > 33344
    receipt = _proof(monkeypatch, body, meter, capture)
    assert receipt["status"] == "READY"
    assert receipt["measured_prompt_tokens"] == tokens
    assert receipt["measure_method"] == "tokenize"


@pytest.mark.parametrize("layout", ["off", "on"])
def test_fallback_allocation_stays_calibrated_after_tokenizer_recovers(monkeypatch, layout):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", layout)
    body = _body(50000)
    meter = RequestTokenMeter(tokenizer=lambda _text: 20000)
    meter.bind_request("seal-r1", "seal-r1")
    monkeypatch.setattr(meter, "_availability", "unavailable")
    from kernel_frame_allocation_capture import calibrated_estimate
    tokens = calibrated_estimate(body, meter)
    capture = _capture(body, meter, method="calibrated_bytes", tokens=tokens)
    monkeypatch.setattr(meter, "_availability", "available")
    assert meter.measure(body)[0] != tokens  # Control: a live recount drifts.
    receipt = _proof(monkeypatch, body, meter, capture)
    assert receipt["status"] == "READY"
    assert receipt["measured_prompt_tokens"] == tokens
    assert receipt["measure_method"] == "calibrated_bytes"


def test_changed_exact_body_uses_frozen_delta_when_tokenizer_fails(monkeypatch):
    body = _body(50000)
    meter = RequestTokenMeter(tokenizer=lambda _text: 20000)
    meter.bind_request("seal-r1", "seal-r1")
    tokens, method = meter.measure(body)
    capture = _capture(body, meter, method=method, tokens=tokens)
    monkeypatch.setattr(meter, "_availability", "unavailable")
    changed = copy.deepcopy(body)
    changed["messages"][0]["content"] += "x" * 1700
    count, frozen_method = final_request_tokens(changed, meter, capture)
    assert frozen_method == "tokenize"
    assert count == tokens + math.ceil(1700 / capture._frame_candidate_counter.ratio)
    assert meter.measure(changed)[0] > count


def test_oversize_rejection_sentinel_freezes_only_after_trimming():
    oversized = _body(600000)
    meter = RequestTokenMeter(tokenizer=lambda _text: 100)
    meter.bind_request("seal-r1", "seal-r1")
    bound, method = meter.measure(oversized)
    assert method == "verified_byte_bound"  # Control: no exact allocation exists yet.
    capture = _capture(oversized, meter, method=method, tokens=bound)
    trimmed = _body(1000)
    tokens, method = final_request_tokens(trimmed, meter, capture)
    assert (tokens, method) == (meter.measure(trimmed)[0], "tokenize")
    assert capture._frame_candidate_counter.method == "tokenize"


@pytest.mark.parametrize("allocation_method", ["tokenize", "calibrated_bytes"])
def test_opt_in_layout_uses_frozen_counter_across_flip(monkeypatch, allocation_method):
    from test_final_cache_layout import _body as layout_body
    from kernel_cache_layout_candidate import render_final_cache_candidate

    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda receipt: receipt["status"] == "COMPLETE")
    body = layout_body()
    if allocation_method == "tokenize":
        # Cold 2.46 byte fallback would reject the canonical fitted request.
        body["messages"][3]["content"] += "x" * 78300
    meter = RequestTokenMeter(tokenizer=lambda _text: 1000)
    meter.bind_request("seal-r1", "seal-r1")
    if allocation_method == "tokenize":
        tokens, _ = meter.measure(body)
        capture = _capture(body, meter, method=allocation_method, tokens=tokens)
        monkeypatch.setattr(meter, "_availability", "unavailable")
    else:
        monkeypatch.setattr(meter, "_availability", "unavailable")
        from kernel_frame_allocation_capture import calibrated_estimate
        capture = _capture(body, meter, method=allocation_method,
                           tokens=calibrated_estimate(body, meter))
        monkeypatch.setattr(meter, "_availability", "available")
    capture.journal = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    inputs = {"token_ceiling": 33344, "byte_ceiling": 82000,
              "source_plan": {}, "sources": {}, "receipt": None}
    result = render_final_cache_candidate(capture, None, body, meter=meter, inputs=inputs)
    assert result is body  # A new projection needs a live exact count.
    assert _proof(monkeypatch, result, meter, capture)["status"] == "READY"
    assert capture.request_budget_final_wire["measure_method"] == allocation_method


def test_layout_exact_count_is_carried_through_later_outage(monkeypatch):
    from test_final_cache_layout import _body as layout_body
    from kernel_cache_layout_candidate import render_final_cache_candidate

    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda _receipt: True)
    body = layout_body()
    body["messages"][3]["content"] += "x" * 78300
    meter = RequestTokenMeter(tokenizer=lambda _text: 20000)
    meter.bind_request("seal-r1", "seal-r1")
    tokens, method = meter.measure(body)
    capture = _capture(body, meter, method=method, tokens=tokens)
    projected = render_final_cache_candidate(
        capture, None, body, meter=meter,
        inputs={"token_ceiling": 33344, "byte_ceiling": 82000,
                "source_plan": {}, "sources": {}, "receipt": None})
    assert projected is not body
    projected_tokens = capture._frame_candidate_counter.tokens
    monkeypatch.setattr(meter, "_availability", "unavailable")
    assert meter.measure(projected)[0] > 33344
    receipt = _proof(monkeypatch, projected, meter, capture)
    assert receipt["status"] == "READY"
    assert receipt["measured_prompt_tokens"] == projected_tokens


def test_permanent_outage_growth_is_rejected_by_layout_before_proof(monkeypatch):
    from test_final_cache_layout import Meter, _body as layout_body
    from kernel_cache_layout_candidate import render_final_cache_candidate

    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda _receipt: True)
    body = layout_body()
    pad = 78921 - len(json.dumps(body, sort_keys=True, ensure_ascii=False))
    assert pad > 0
    body["messages"][3]["content"] += "x" * pad
    meter = RequestTokenMeter(disabled=True)
    meter.bind_request("seal-r1", "seal-r1")
    from kernel_frame_allocation_capture import calibrated_estimate
    tokens = calibrated_estimate(body, meter)
    assert tokens == 33230
    assert meter._calibration_samples == 0
    assert meter._fallback_ratio()[0] == 2.46
    raw = render_final_cache_candidate(
        SimpleNamespace(request_source_final_delivery=None), None, body,
        meter=Meter(), inputs={"token_ceiling": 100000, "byte_ceiling": 100000,
                              "source_plan": {}, "sources": {}, "receipt": None})
    assert raw is not body
    assert calibrated_estimate(raw, meter) > 33344
    capture = _capture(body, meter, method="calibrated_bytes", tokens=tokens)
    capture.journal = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    inputs = {"token_ceiling": 33344, "byte_ceiling": 82000,
              "source_plan": {}, "sources": {}, "receipt": None}
    projected = render_final_cache_candidate(capture, None, body, meter=meter, inputs=inputs)
    assert projected is body
    assert _proof(monkeypatch, projected, meter, capture)["status"] == "READY"


@pytest.mark.parametrize("layout", ["off", "on"])
def test_permanent_outage_post_fit_growth_recovers_before_submission(monkeypatch, tmp_path,
                                                                    layout):
    from kernel_final_wire_allocation import _recover_post_fit_growth
    from kernel_frame_allocation_capture import calibrated_estimate

    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", layout)
    body = _body(0)
    body["messages"][0]["content"] = "x" * (
        78871 - len(json.dumps(body, sort_keys=True, ensure_ascii=False)))
    assert len(json.dumps(body).encode()) == 78871
    meter = RequestTokenMeter(disabled=True)
    meter.bind_request("seal-r1", "seal-r1")
    tokens = calibrated_estimate(body, meter)
    assert tokens == 33210 and meter._calibration_samples == 0
    assert meter._fallback_ratio()[0] == 2.46
    grown = copy.deepcopy(body)
    grown["messages"][0]["content"] += "x" * 1690
    assert len(json.dumps(grown).encode()) == 80561
    assert calibrated_estimate(grown, meter) > 33344

    # Planted control: sending the changed body directly fails its final proof.
    control = _capture(body, meter, method="calibrated_bytes", tokens=tokens)
    assert "request_tokens_exceed_max_prompt_tokens" in _proof(
        monkeypatch, grown, meter, control)["final_validation_failures"]

    capture = _capture(body, meter, method="calibrated_bytes", tokens=tokens)
    capture.request_source_final_delivery = {"status": "COMPLETE"}
    fitted_counter = capture._frame_candidate_counter
    fitted_inputs = {"token_ceiling": 33344, "byte_ceiling": 82000,
                     "receipt": {"status": "COMPLETE"}}
    capture._frame_candidate_inputs = copy.deepcopy(fitted_inputs)
    recovered, changed = _recover_post_fit_growth(
        capture, SimpleNamespace(), grown, body, fitted_counter, fitted_inputs,
        capture.request_source_final_delivery, meter=meter, endpoint=None, key="")
    assert changed and recovered == body
    assert _proof(monkeypatch, recovered, meter, capture)["status"] == "READY"

    # The recovered wire reaches the ordinary transport submission and model
    # response path. Keep the endpoint in-process and its recorder isolated.
    import kernel_measurement

    monkeypatch.setattr("kernel_bounded_http.request_bytes", lambda *_args, **_kwargs:
                        b'{"choices":[{"message":{"role":"assistant","content":"done"}}]}')
    journal = kernel_measurement.AuditJournal(tmp_path, "seal-growth", {})
    transport = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    response, _submission, _started = transport._send_transport_with_retries(
        recovered, json.dumps(recovered).encode(), "http://127.0.0.1:1/v1/chat/completions",
        "", admission=None, socket_timeout=2, deadline=time.monotonic() + 10,
        wall_timeout=2, backend_identity={}, request_extensions={},
        transport_attempt_limit=1, frozen_retry_only=True)
    assert response["choices"][0]["message"]["content"] == "done"
    records = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert len([row for row in records if row["kind"] == "request_submitted"]) == 1


def test_late_rewrite_refits_with_the_frozen_counter(monkeypatch):
    from kernel_final_wire_allocation import _recover_post_fit_growth
    from kernel_frame_allocation_capture import (
        calibrated_estimate, carry_final_request_tokens,
    )

    body = _body(0)
    body["messages"][0]["content"] = "x" * (
        78871 - len(json.dumps(body, sort_keys=True, ensure_ascii=False)))
    meter = RequestTokenMeter(disabled=True)
    meter.bind_request("seal-r1", "seal-r1")
    count = calibrated_estimate(body, meter)
    capture = _capture(body, meter, method="calibrated_bytes", tokens=count)
    capture.request_source_final_delivery = {"status": "COMPLETE"}
    inputs = {"token_ceiling": 33344, "byte_ceiling": 82000,
              "receipt": {"status": "COMPLETE"}}
    capture._frame_candidate_inputs = copy.deepcopy(inputs)
    grown = copy.deepcopy(body)
    grown["messages"][0]["content"] += "x" * 1690
    assert "request_tokens_exceed_max_prompt_tokens" in _proof(
        monkeypatch, grown, meter, capture)["final_validation_failures"]

    def add_late_bytes(_capture, value, **_kwargs):
        value["messages"][0]["content"] += "x" * 1690

    refits = []

    def trim_changed_body(_context, candidate, value, *, frame_counter):
        assert candidate._frame_candidate_counter.method == "calibrated_bytes"
        refits.append(frame_counter)
        trimmed = copy.deepcopy(value)
        trimmed["messages"][0]["content"] = trimmed["messages"][0]["content"][:-4000]
        tokens, _ = final_request_tokens(trimmed, meter, candidate)
        carry_final_request_tokens(trimmed, candidate, tokens)
        return trimmed

    monkeypatch.setattr("kernel_context_capacity.enforce_context_capacity", add_late_bytes)
    monkeypatch.setattr("kernel_frame_allocation_capture.finalize_frame_candidate",
                        trim_changed_body)
    recovered, changed = _recover_post_fit_growth(
        capture, SimpleNamespace(), grown, body, capture._frame_candidate_counter,
        inputs, capture.request_source_final_delivery,
        meter=meter, endpoint=None, key="")
    assert changed and len(refits) == 1
    assert recovered != body
    assert _proof(monkeypatch, recovered, meter, capture)["status"] == "READY"


def test_recovered_exact_count_refuses_real_calibrated_undercount(monkeypatch):
    body = _body(50000)
    meter = RequestTokenMeter(tokenizer=lambda _text: 34000)
    meter.bind_request("seal-r1", "seal-r1")
    monkeypatch.setattr(meter, "_availability", "unavailable")
    from kernel_frame_allocation_capture import calibrated_estimate
    capture = _capture(body, meter, method="calibrated_bytes",
                       tokens=calibrated_estimate(body, meter))
    monkeypatch.setattr(meter, "_availability", "available")
    receipt = _proof(monkeypatch, body, meter, capture)
    assert receipt["status"] == "HOST_BLOCKED"
    assert "request_tokens_exceed_max_prompt_tokens" in receipt["final_validation_failures"]


def test_unchanged_exact_body_keeps_wire_and_count(monkeypatch):
    body = _body(1000)
    wire = json.dumps(body).encode()
    meter = RequestTokenMeter(tokenizer=lambda _text: 100)
    meter.bind_request("seal-r1", "seal-r1")
    tokens, method = meter.measure(body)
    capture = _capture(body, meter, method=method, tokens=tokens)
    receipt = _proof(monkeypatch, body, meter, capture)
    assert json.dumps(body).encode() == wire
    assert receipt["request_bytes"] == len(wire)
    assert receipt["measured_prompt_tokens"] == tokens


def test_frozen_count_refuses_provider_identity_drift(monkeypatch):
    body = _body()
    meter = RequestTokenMeter(tokenizer=lambda _text: 100)
    meter.bind_request("seal-r1", "seal-r1")
    tokens, method = meter.measure(body)
    capture = _capture(body, meter, method=method, tokens=tokens)
    changed = copy.deepcopy(body)
    changed["model"] = "different-provider"
    with pytest.raises(FrameRequestAllocationError, match="identity changed"):
        final_request_tokens(changed, meter, capture)
