"""Main 0.4.35 request contracts remain intact with unavailable evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest


ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
sys.path.insert(0, str(ENGINE))


@pytest.mark.parametrize("reason", ["BINDING_UNAVAILABLE", "PUBLIC_CHECK_UNAVAILABLE"])
def test_qualified_builder_stops_on_unavailable_wire_facts(monkeypatch, reason):
    import kernel_final_wire_allocation as allocation
    import kernel_request_candidate
    import kernel_request_budget
    import kernel_request_kind
    import kernel_wire_truth_gate
    from kernel_request_seal import RequestSealError
    from kernel_wire_truth_facts import UnavailableFactEvidence

    monkeypatch.setattr(kernel_request_kind, "classify", lambda *_args: True)
    monkeypatch.setattr(kernel_request_kind, "validate_adapter", lambda *_args: None)
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kernel_request_candidate, "detached_context",
                        lambda *_args: (SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(kernel_wire_truth_gate, "capture_context",
                        lambda *_args, **_kwargs: UnavailableFactEvidence(reason))
    capture = SimpleNamespace(qualified_request=None)

    with pytest.raises(RequestSealError, match=reason):
        allocation.build_qualified_request(
            capture, SimpleNamespace(), {"messages": [], "tools": []},
            endpoint="http://127.0.0.1:1/v1/chat/completions", key="",
        )
    assert capture.qualified_request is None


@pytest.mark.parametrize("error_type", [
    "owner_contract", "wire_fact_unavailable",
])
def test_capacity_refit_preserves_host_contract_error(monkeypatch, tmp_path, error_type):
    import kernel_bounded_http
    import kernel_capacity_retry
    import kernel_measurement
    from kernel_completion_owner import RequestActionContractError
    from kernel_wire_truth_facts import WireTruthFactUnavailable

    error = kernel_bounded_http.BoundedTransportError(
        error_type="HTTPError", http_status=400, relay_error=None,
        server_error=json.dumps({"error": {
            "code": "context_length_exceeded",
            "message": "maximum context length is 32768 tokens",
        }}),
    )
    sends = []

    def reject(*_args, **_kwargs):
        sends.append(1)
        raise error

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", reject)
    refit_error = (RequestActionContractError("owner contract changed")
                   if error_type == "owner_contract" else
                   WireTruthFactUnavailable("wire truth facts unavailable: BINDING_UNAVAILABLE"))
    monkeypatch.setattr(
        kernel_capacity_retry, "reseal_capacity_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(refit_error),
    )
    body = {"model": "scripted", "messages": [{"role": "user", "content": "hi"}],
            "tools": [], "max_tokens": 100}
    journal = kernel_measurement.AuditJournal(tmp_path, "owner-contract", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    capture._capacity_retry_original_body = body.copy()
    capture._request_measurement = SimpleNamespace(prompt_tokens=100, output_tokens=100)

    with pytest.raises(type(refit_error), match=str(refit_error)):
        capture._send_transport_with_retries(
            body, json.dumps(body).encode(), "http://127.0.0.1:1/v1/chat/completions", "",
            admission=None, socket_timeout=2, deadline=time.monotonic() + 120,
            wall_timeout=2, backend_identity={}, request_extensions={},
            transport_attempt_limit=2,
        )
    assert sends == [1]
    assert all(json.loads(line)["kind"] != "capacity_retry_refit_failed"
               for line in journal.path.read_text().splitlines())


def test_transport_retry_stops_when_wire_facts_become_unavailable(monkeypatch, tmp_path):
    import kernel_bounded_http
    import kernel_measurement
    import kernel_request_reprepare
    import kernel_wire_truth_gate
    from kernel_request_seal import RequestSealError
    from kernel_wire_truth_facts import UnavailableFactEvidence

    monkeypatch.setenv("STATEFRAME_PER_REQUEST_TIMEOUT_S", "2")

    sends = []

    def timeout(*_args, **_kwargs):
        sends.append(1)
        raise kernel_bounded_http.BoundedTransportError(
            error_type="TimeoutError", http_status=None, relay_error=None)

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", timeout)
    monkeypatch.setattr(kernel_request_reprepare, "qualify_source",
                        lambda _capture, _runtime, _preparation, _body, wire: wire)
    monkeypatch.setattr(kernel_wire_truth_gate, "capture_context",
                        lambda *_args, **_kwargs: UnavailableFactEvidence("BINDING_UNAVAILABLE"))
    body = {"model": "scripted", "messages": [{"role": "user", "content": "hi"}],
            "tools": [], "max_tokens": 100}
    journal = kernel_measurement.AuditJournal(tmp_path, "retry-wire-facts", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    monkeypatch.setattr(capture, "_request_capture_pre_wire_extensions",
                        lambda *_args, **_kwargs: None)

    with pytest.raises(RequestSealError, match="BINDING_UNAVAILABLE"):
        capture._send_transport_with_retries(
            body, json.dumps(body).encode(), "http://127.0.0.1:1/v1/chat/completions", "",
            admission=None, socket_timeout=2, deadline=time.monotonic() + 120,
            wall_timeout=2, backend_identity={}, request_extensions={},
            transport_attempt_limit=2,
        )
    assert sends == [1]


def test_transport_retry_does_not_record_owner_contract_as_disagreement(monkeypatch, tmp_path):
    import kernel_bounded_http
    import kernel_measurement
    import kernel_one_snapshot_preparation
    import kernel_request_reprepare
    from kernel_completion_owner import RequestActionContractError

    monkeypatch.setenv("STATEFRAME_PER_REQUEST_TIMEOUT_S", "2")
    monkeypatch.setattr(kernel_one_snapshot_preparation, "enabled", lambda: True)
    recorded = []
    monkeypatch.setattr(kernel_one_snapshot_preparation, "record_disagreement",
                        lambda *_args, **_kwargs: recorded.append(1) or True)
    sends = []

    def timeout(*_args, **_kwargs):
        sends.append(1)
        raise kernel_bounded_http.BoundedTransportError(
            error_type="TimeoutError", http_status=None, relay_error=None)

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", timeout)
    monkeypatch.setattr(kernel_request_reprepare, "qualify_source",
                        lambda _capture, _runtime, _preparation, _body, wire: wire)
    body = {"model": "scripted", "messages": [{"role": "user", "content": "hi"}],
            "tools": [], "max_tokens": 100}
    wire = json.dumps(body).encode()
    journal = kernel_measurement.AuditJournal(tmp_path, "retry-owner-contract", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    capture._one_snapshot_first_wire = wire

    def owner_contract_error(*_args, **_kwargs):
        raise RequestActionContractError("owner contract changed on retry")

    monkeypatch.setattr(capture, "_request_capture_pre_wire_extensions", owner_contract_error)

    with pytest.raises(RequestActionContractError, match="owner contract changed on retry"):
        capture._send_transport_with_retries(
            body, wire, "http://127.0.0.1:1/v1/chat/completions", "",
            admission=None, socket_timeout=2, deadline=time.monotonic() + 120,
            wall_timeout=2, backend_identity={}, request_extensions={},
            transport_attempt_limit=2,
        )
    assert sends == [1]
    assert recorded == []


def test_lean_preparation_stops_with_existing_capture_on_fact_gap(monkeypatch):
    import kernel_stateframe_runtime as runtime_module
    import kernel_wire_truth_gate
    from kernel_wire_truth_facts import UnavailableFactEvidence

    capture = SimpleNamespace(request_id="lean-request")
    runtime = SimpleNamespace(system_outcome=None)
    monkeypatch.setattr(kernel_wire_truth_gate, "capture_context",
                        lambda *_args, **_kwargs: UnavailableFactEvidence("BINDING_UNAVAILABLE"))

    result = runtime_module._lean_wire_truth_context(capture, runtime, [], [])

    assert isinstance(result, runtime_module.PreparedRequest)
    assert result.capture is capture
    assert result.messages == ()
    assert result.overflow == {
        "kind": "REQUEST_BUILDER_UNAVAILABLE", "reason_code": "BINDING_UNAVAILABLE",
    }
    assert runtime.system_outcome == "request_builder_unavailable"
