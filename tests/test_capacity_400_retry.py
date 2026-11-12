"""Capacity HTTP 400 classification, deterministic ceilings, and retry receipts."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator, ValidationError
from _timeouts import scaled

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"
sys.path.insert(0, str(ENGINE))

from kernel_bounded_http import capacity_400_reason, transport_class  # noqa: E402
from kernel_capacity_retry import (  # noqa: E402
    apply_learned_limit, compact_tool_prose, learned_context_limit,
    learned_request_limits, learn_context_limit,
    reported_context_limit, retry_limits,
)
from test_capacity_400_acceptance import _run_case  # noqa: E402


FIXTURES = ROOT / "tests" / "fixtures" / "capacity_400"


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda path: path.stem)
def test_provider_fixture_is_capacity_400(path: Path) -> None:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    body = json.dumps(fixture["body"])
    assert capacity_400_reason(400, body) is not None
    assert transport_class(400, body) == "capacity"


@pytest.mark.parametrize("body", [
    '{"error":{"code":"context_length_exceeded"}}',
    '{"error":{"message":"maximum context length is 32768 tokens"}}',
])
def test_bare_capacity_strings(body: str) -> None:
    assert capacity_400_reason(400, body) is not None


@pytest.mark.parametrize("body", [
    '{"error":{"message":"The length of parameter name is invalid"}}',
    '{"error":{"message":"Unsupported parameter: max_length"}}',
    '{"error":{"message":"String length must be <= 1024"}}',
    '{"error":{"message":"Your output is too long"}}',
    '{"error":{"code":"unsupported_parameter","message":"context_length is not supported"}}',
    '{"error":{"code":"unsupported_parameter","message":"context_length_exceeded is not supported"}}',
])
def test_non_capacity_400_length_errors_stay_contract(body: str) -> None:
    assert capacity_400_reason(400, body) is None
    assert transport_class(400, body) == "contract"


def test_capacity_marker_on_other_status_does_not_enable_400_retry() -> None:
    assert capacity_400_reason(422, '{"error":{"code":"context_length_exceeded"}}') is None


def test_schema_keyword_mention_is_not_capacity() -> None:
    body = ('{"error":{"message":"Invalid response_format property: maximum '
            'context length is not a valid schema keyword"}}')
    assert capacity_400_reason(400, body) is None
    assert transport_class(400, body) == "contract"
    conflicting = ('{"error":{"code":"context_length_exceeded",'
                   '"message":"Invalid response_format property: maximum '
                   'context length is not a valid schema keyword"}}')
    assert capacity_400_reason(400, conflicting) is None


def test_reported_provider_limits_and_deterministic_shrink() -> None:
    messages = {
        "openai": ("maximum context length is 32768 tokens", 32768),
        "vllm": ("maximum context length is 16384 tokens", 16384),
        "anthropic": ("prompt is too long: 209375 tokens > 200000 maximum", 200000),
    }
    for message, expected in messages.values():
        assert reported_context_limit(message) == expected
    assert reported_context_limit("request exceeds the available context size") is None
    assert retry_limits(32_000, 10_000, None) == retry_limits(32_000, 10_000, None)
    assert retry_limits(32_000, 10_000, None).max_request_bytes == 28_000
    assert retry_limits(32_000, 10_000, None).max_prompt_tokens == 8_750
    assert retry_limits(32_000, 10_000, 8_000).max_prompt_tokens == 7_000
    with pytest.raises(ValueError):
        retry_limits(1, 10_000, None)


def test_reported_total_context_is_split_before_fitting() -> None:
    limits = retry_limits(40_000, 30_000, 32_768, 24_000)
    assert limits.max_output_tokens == 7_168
    assert limits.max_prompt_tokens + limits.max_output_tokens <= 32_768 * 7 // 8
    learned = learned_request_limits(32_768, 24_000)
    assert learned.max_prompt_tokens + learned.max_output_tokens == 32_768 * 7 // 8


def test_of_form_capacity_error_extracts_and_enforces_total_limit() -> None:
    message = "maximum context length of 8192 tokens"
    body = json.dumps({"error": {"message": message}})
    assert capacity_400_reason(400, body) == "maximum_context_length"
    assert transport_class(400, body) == "capacity"
    assert reported_context_limit(body) == 8192
    limits = retry_limits(40_000, 15_000, reported_context_limit(body), 24_000)
    assert limits.max_output_tokens is not None
    assert limits.max_prompt_tokens + limits.max_output_tokens <= 8192 * 7 // 8


def test_retry_compacts_prose_without_changing_callable_schema() -> None:
    body = {"tools": [{"type": "function", "function": {
        "name": "edit", "description": "A careful editing tool. " * 12,
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to edit."},
        }, "required": ["path"]},
    }}]}
    compact_tool_prose(body)
    function = body["tools"][0]["function"]
    assert function["name"] == "edit"
    assert function["parameters"]["properties"]["path"]["type"] == "string"
    assert function["parameters"]["required"] == ["path"]
    assert "description" not in function["parameters"]["properties"]["path"]
    assert 0 < len(function["description"]) <= 80


def test_retry_compaction_preserves_required_description_property() -> None:
    parameters = {"type": "object", "description": "Root prose", "properties": {
        "description": {"type": "string", "description": "Argument prose"},
        "nested": {"type": "object", "properties": {
            "description": {"type": "integer", "description": "Nested prose"},
        }, "required": ["description"]},
    }, "required": ["description"]}
    body = {"tools": [{"type": "function", "function": {
        "name": "edit", "description": "Long tool prose", "parameters": parameters,
    }}]}
    compact_tool_prose(body)
    compacted = body["tools"][0]["function"]["parameters"]
    assert compacted["required"] == ["description"]
    assert compacted["properties"]["description"] == {"type": "string"}
    assert compacted["properties"]["nested"]["properties"]["description"] == {
        "type": "integer"}
    Draft202012Validator.check_schema(compacted)
    Draft202012Validator(compacted).validate({"description": "valid"})
    with pytest.raises(ValidationError):
        Draft202012Validator(compacted).validate({})


def test_endpoint_limit_is_durable_and_scoped(tmp_path: Path) -> None:
    from kernel_measurement import AuditJournal

    journal = AuditJournal(tmp_path, "learn-limit", {})
    first = "http://127.0.0.1:8000/v1/chat/completions"
    second = "http://127.0.0.1:8001/v1/chat/completions"
    learn_context_limit(journal, first, "model-a", "key-a", 32_768)
    learn_context_limit(journal, first, "model-a", "key-a", 65_536)
    assert learned_context_limit(journal, first, "model-a", "key-a") == 32_768
    assert learned_context_limit(journal, second, "model-a", "key-a") is None
    assert learned_context_limit(journal, first, "model-b", "key-a") is None
    assert learned_context_limit(journal, first, "model-a", "key-b") is None
    del journal._capacity_context_limits
    assert learned_context_limit(journal, first, "model-a", "key-a") == 32_768
    assert learned_context_limit(journal, first, "model-b", "key-a") is None
    assert learned_context_limit(journal, first, "model-a", "key-b") is None
    events = [event for event in journal.path.read_text().splitlines()
              if json.loads(event)["kind"] == "capacity_context_limit_learned"]
    assert len(events) == 1
    assert "key-a" not in events[0]
    assert "model-a" not in events[0]


def test_later_request_uses_learned_endpoint_ceiling(tmp_path: Path) -> None:
    from kernel_measurement import AuditJournal, RequestCapture

    journal = AuditJournal(tmp_path, "learn-later", {})
    endpoint = "http://127.0.0.1:8000/v1/chat/completions"
    learn_context_limit(journal, endpoint, "model-a", "key-a", 32_768)
    capture = RequestCapture(journal, 2, "snapshot")
    body = {"model": "model-a", "max_tokens": 24_000}
    apply_learned_limit(capture, body, endpoint, "key-a")
    limits = capture._capacity_retry_limits
    assert body["max_tokens"] == limits.max_output_tokens
    assert limits.max_prompt_tokens + body["max_tokens"] == 32_768 * 7 // 8
    other = RequestCapture(journal, 3, "snapshot")
    unchanged = {"model": "model-a", "max_tokens": 24_000}
    apply_learned_limit(other, unchanged,
                        "http://127.0.0.1:8001/v1/chat/completions", "key-a")
    assert unchanged["max_tokens"] == 24_000
    assert not hasattr(other, "_capacity_retry_limits")


def test_no_rejection_keeps_body_and_credential_identity_untouched(
    monkeypatch, tmp_path: Path
) -> None:
    import kernel_request_auth_identity
    from kernel_measurement import AuditJournal, RequestCapture

    def unexpected_binding(_key):
        raise AssertionError("ordinary requests need no credential identity lookup")

    monkeypatch.setattr(kernel_request_auth_identity, "auth_binding_for_key",
                        unexpected_binding)
    journal = AuditJournal(tmp_path, "no-capacity-rejection", {})
    capture = RequestCapture(journal, 1, "snapshot")
    body = {"model": "model-a", "max_tokens": 24_000}
    apply_learned_limit(capture, body,
                        "http://127.0.0.1:8000/v1/chat/completions", "key-a")
    assert body == {"model": "model-a", "max_tokens": 24_000}
    assert not hasattr(capture, "_capacity_retry_limits")


def test_model_router_limit_lookup_uses_effective_destination(monkeypatch, tmp_path: Path) -> None:
    import kernel_capacity_retry
    from kernel_measurement import AuditJournal, RequestCapture

    base = "http://127.0.0.1:8000/v1/chat/completions"
    routed = "http://127.0.0.1:8001/v1/chat/completions"
    journal = AuditJournal(tmp_path, "routed-limit", {})
    learn_context_limit(journal, base, "base-model", "base-key", 32_768)
    learn_context_limit(journal, routed, "routed-model", "routed-key", 8_192)
    capture = RequestCapture(journal, 2, "snapshot")

    class Router:
        def prepare_call(self, _capture, body, _endpoint, _key):
            return SimpleNamespace(body={**body, "model": "routed-model"},
                                   endpoint=routed, key="routed-key")

        def validate_final_request(self, _capture, _body):
            pass

    capture.model_router = Router()

    class LookupObserved(Exception):
        pass

    original_apply = kernel_capacity_retry.apply_learned_limit

    def observe(_capture, body, endpoint, key):
        assert (endpoint, body["model"], key) == (routed, "routed-model", "routed-key")
        original_apply(_capture, body, endpoint, key)
        assert _capture._capacity_retry_limits.provider_limit_tokens == 8_192
        assert body["max_tokens"] == 1_792
        raise LookupObserved

    monkeypatch.setattr(kernel_capacity_retry, "apply_learned_limit", observe)
    with pytest.raises(LookupObserved):
        capture._send_once({"model": "base-model", "messages": [], "tools": [],
                            "max_tokens": 24_000}, base, "base-key")


def test_continuation_replay_uses_retried_wire_for_notice(monkeypatch) -> None:
    import kernel_step_contract
    from kernel_continuation_restart import reconcile_environment_notice_tail

    original = {"messages": [{"role": "user", "content": "original"}]}
    retried = {"messages": [{"role": "user", "content": "retried"}]}
    wire = json.dumps(retried)
    sha = "sha256:" + hashlib.sha256(wire.encode()).hexdigest()
    seen = []
    monkeypatch.setattr(kernel_step_contract, "commit_environment_notice_delivery",
                        lambda _runtime, messages: seen.append(messages) or True)
    tail = [
        {"sequence": 1, "kind": "request_final_wire_prepared", "request_id": "req",
         "payload_sha256": "sha256:" + hashlib.sha256(json.dumps(original).encode()).hexdigest(),
         "messages": original["messages"]},
        {"sequence": 2, "kind": "request_submitted", "request_id": "req",
         "event_id": "sent-1", "payload": original,
         "payload_utf8": json.dumps(original)},
        {"sequence": 3, "kind": "request_final_wire_prepared", "request_id": "req",
         "payload_sha256": sha, "messages": retried["messages"]},
        {"sequence": 4, "kind": "request_submitted", "request_id": "req",
         "event_id": "sent-2", "payload": retried, "payload_utf8": wire,
         "payload_hash": sha},
        {"sequence": 5, "kind": "response_received", "request_id": "req",
         "event_id": "response-2", "transport_attempt_id": "sent-2"},
        {"sequence": 6, "kind": "native_batch_received", "request_id": "req",
         "origin_response_event_id": "response-2"},
    ]
    runtime = SimpleNamespace(_evictfix_environment_notice_pending={"pending": True})
    assert reconcile_environment_notice_tail(runtime, tail, 0)
    assert seen == [retried["messages"]]


def test_explicit_one_attempt_limit_blocks_capacity_retry(monkeypatch, tmp_path: Path) -> None:
    import kernel_bounded_http
    import kernel_measurement
    import kernel_capacity_retry

    error = kernel_bounded_http.BoundedTransportError(
        error_type="HTTPError", http_status=400, relay_error=None,
        server_error='{"error":{"code":"context_length_exceeded",'
                     '"message":"maximum context length is 32768 tokens"}}',
    )
    attempts = []

    def reject(*_args, **_kwargs):
        attempts.append(1)
        raise error

    def forbidden_refit(*_args, **_kwargs):
        raise AssertionError("attempt limit must prevent capacity refit")

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", reject)
    monkeypatch.setattr(kernel_capacity_retry, "reseal_capacity_retry", forbidden_refit)
    body = {"model": "scripted", "messages": [{"role": "user", "content": "hi"}],
            "tools": [], "max_tokens": 100}
    wire = json.dumps(body).encode()
    journal = kernel_measurement.AuditJournal(tmp_path, "attempt-limit", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    capture._capacity_retry_original_body = body.copy()
    capture._request_measurement = SimpleNamespace(prompt_tokens=100, output_tokens=100)
    with pytest.raises(kernel_measurement.HarnessTransportError):
        capture._send_transport_with_retries(
            body, wire, "http://127.0.0.1:1/v1/chat/completions", "",
            admission=None, socket_timeout=scaled(2), deadline=time.monotonic() + scaled(10),
            wall_timeout=scaled(2), backend_identity={}, request_extensions={},
            transport_attempt_limit=1)
    assert attempts == [1]
    assert capture._capacity_retry_count == 0


@pytest.fixture(scope="module")
def capacity_retry_case(tmp_path_factory: pytest.TempPathFactory):
    error = {"error": {"code": "context_length_exceeded",
                       "message": "maximum context length is 32768 tokens"}}
    return _run_case(tmp_path_factory.mktemp("capacity-retry-regression"), (error,))


@pytest.mark.host_sandbox
def test_capacity_retry_journal_records_actual_shrink(capacity_retry_case) -> None:
    case = capacity_retry_case
    assert case.exit_code == 0, case.output
    assert len(case.requests) == 2
    events = [event for event in case.events if event.get("kind") == "capacity_retry"]
    assert len(events) == 1, events
    event = events[0]
    assert event["reason"] == "context_length_exceeded"
    assert event["provider_limit_tokens"] == 32768
    assert event["before_bytes"] == len(case.requests[0])
    assert event["after_bytes"] == len(case.requests[1])
    assert event["shrink_bytes"] == len(case.requests[0]) - len(case.requests[1]) > 0
    assert event["shrink_tokens"] > 0
    assert event["after_output_tokens"] < event["before_output_tokens"]
    assert event["after_bytes"] <= event["before_bytes"] * 7 // 8
    assert (event["after_prompt_tokens"] + event["after_output_tokens"]
            <= event["provider_limit_tokens"] * 7 // 8)
    assert event["retry_scheduled"] is True


@pytest.mark.host_sandbox
def test_resume_reconstructs_capacity_retry_wire(capacity_retry_case) -> None:
    from _journal import measurement_journal_paths
    from kernel_measurement import resolve_journal_blob_refs
    from kernel_native_prefix_delivery import submitted_payload

    case = capacity_retry_case
    assert case.exit_code == 0, case.output
    submitted = [event for event in case.events if event.get("kind") == "request_submitted"]
    assert len(submitted) == 2
    journal_paths = measurement_journal_paths(case.summary["run_dir"])
    assert len(journal_paths) == 1
    retry = resolve_journal_blob_refs(submitted[1], journal_paths[0])
    wire = case.requests[1]
    assert retry["payload_utf8"].encode() == wire
    assert retry["payload_hash"] == "sha256:" + hashlib.sha256(wire).hexdigest()
    assert retry["payload"] == json.loads(wire)
    assert "payload_ref" not in retry and "payload_delta" not in retry

    class ReplayedEvidence:
        def event(self, event_id: str, kind: str):
            return next(row for row in case.events
                        if row.get("event_id") == event_id and row.get("kind") == kind)

    assert submitted_payload(ReplayedEvidence(), retry) == json.loads(wire)


@pytest.mark.host_sandbox
def test_non_capacity_400_mentioning_length_is_not_retried(tmp_path: Path) -> None:
    error = {"error": {"code": "invalid_request_error",
                       "message": "The length of parameter name is invalid"}}
    case = _run_case(tmp_path, (error,))
    assert len(case.requests) == 1, case.output
    assert case.summary["engine_result"] == "HARNESS_CONTRACT"
    assert not any(event.get("kind") == "capacity_retry" for event in case.events)
