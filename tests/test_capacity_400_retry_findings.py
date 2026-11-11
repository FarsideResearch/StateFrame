"""Regressions for provider limit extraction and ordinary retry scheduling."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pytest
from _timeouts import scaled


ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
sys.path.insert(0, str(ENGINE))

import kernel_bounded_http  # noqa: E402
import kernel_measurement  # noqa: E402
from kernel_capacity_retry import (  # noqa: E402
    apply_learned_limit, learn_context_limit, learned_context_limit,
    reported_context_limit, retry_limits,
)


ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"


def test_limit_ignores_other_error_fields_and_next_request_remains_usable(tmp_path: Path) -> None:
    body = json.dumps({"error": {
        "code": "context_length_exceeded",
        "message": "maximum context length is 32768 tokens",
        "details": "maximum context length is 1 tokens",
    }})
    assert kernel_bounded_http.capacity_400_reason(400, body) is not None
    assert reported_context_limit(body) == 32_768
    journal = kernel_measurement.AuditJournal(tmp_path, "message-only", {})
    learn_context_limit(journal, ENDPOINT, "model", "key", reported_context_limit(body))
    capture = kernel_measurement.RequestCapture(journal, 2, "snapshot")
    next_body = {"model": "model", "max_tokens": 24_000}
    apply_learned_limit(capture, next_body, ENDPOINT, "key")
    assert capture._capacity_retry_limits.provider_limit_tokens == 32_768
    assert next_body["max_tokens"] > 0


def test_details_only_number_does_not_create_a_learned_limit(tmp_path: Path) -> None:
    body = json.dumps({"error": {
        "code": "context_length_exceeded",
        "message": "request exceeds the available context size",
        "details": "maximum context length is 1 tokens",
    }})
    assert kernel_bounded_http.capacity_400_reason(400, body) is not None
    limit = reported_context_limit(body)
    assert limit is None
    journal = kernel_measurement.AuditJournal(tmp_path, "details-only", {})
    learn_context_limit(journal, ENDPOINT, "model", "key", limit)
    assert learned_context_limit(journal, ENDPOINT, "model", "key") is None
    capture = kernel_measurement.RequestCapture(journal, 2, "snapshot")
    next_body = {"model": "model", "max_tokens": 24_000}
    apply_learned_limit(capture, next_body, ENDPOINT, "key")
    assert next_body["max_tokens"] == 24_000
    assert not hasattr(capture, "_capacity_retry_limits")


def test_tiny_reported_ceiling_is_journaled_but_never_applied(tmp_path: Path) -> None:
    body = json.dumps({"error": {
        "code": "context_length_exceeded",
        "message": "maximum context length is 1 tokens",
    }})
    journal = kernel_measurement.AuditJournal(tmp_path, "tiny-limit", {})
    limit = reported_context_limit(body)
    assert limit == 1
    learn_context_limit(journal, ENDPOINT, "model", "key", limit)
    assert learned_context_limit(journal, ENDPOINT, "model", "key") is None
    assert retry_limits(32_000, 10_000, limit, 24_000).provider_limit_tokens is None
    capture = kernel_measurement.RequestCapture(journal, 2, "snapshot")
    next_body = {"model": "model", "max_tokens": 24_000}
    apply_learned_limit(capture, next_body, ENDPOINT, "key")
    assert next_body["max_tokens"] == 24_000
    assert not hasattr(capture, "_capacity_retry_limits")
    ignored = [json.loads(line) for line in journal.path.read_text().splitlines()
               if json.loads(line)["kind"] == "capacity_context_limit_ignored"]
    assert len(ignored) == 1
    assert ignored[0]["reason"] == "below_minimum_usable_context_tokens"
    assert ignored[0]["minimum_usable_context_tokens"] == 1_024


def test_ordinary_timeout_retry_still_calls_sleep_zero(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STATEFRAME_PER_REQUEST_TIMEOUT_S", "2")
    error = kernel_bounded_http.BoundedTransportError(
        error_type="TimeoutError", http_status=None, relay_error=None)
    sends: list[int] = []
    sleeps: list[float] = []

    def timeout(*_args: object, **_kwargs: object) -> bytes:
        sends.append(1)
        raise error

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", timeout)
    monkeypatch.setattr(kernel_measurement.time, "sleep", sleeps.append)
    body = {"model": "scripted", "messages": [{"role": "user", "content": "hi"}],
            "tools": [], "max_tokens": 100}
    wire = json.dumps(body).encode()
    journal = kernel_measurement.AuditJournal(tmp_path, "timeout-sleep", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    with pytest.raises(kernel_measurement.HarnessTransportError):
        capture._send_transport_with_retries(
            body, wire, ENDPOINT, "", admission=None, socket_timeout=scaled(2),
            deadline=time.monotonic() + scaled(10), wall_timeout=scaled(2),
            backend_identity={}, request_extensions={}, transport_attempt_limit=2)
    assert sends == [1, 1]
    assert sleeps == [0.0]
