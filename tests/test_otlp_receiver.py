"""Tests for the loopback OTLP/HTTP JSON receiver and its review helpers."""
from __future__ import annotations

import copy
import errno
import http.client
import json
import socket
from urllib.parse import urlsplit

import pytest

from tests.otlp_receiver import (
    MAX_REPLY_BYTES,
    OVERSIZED_REPLY_BYTES,
    OTLPReceiver,
    ResponseStep,
    assert_capture_clean,
    assert_no_duplicate_spans,
    assert_valid_traces,
    find_duplicate_spans,
    planted_capture_leaks,
    validate_traces,
)


TRACE_ID = "a13f9c7e5d4b3210a13f9c7e5d4b3210"
SPAN_ID = "b13f9c7e5d4b3210"
PARENT_SPAN_ID = "c13f9c7e5d4b3210"

TRACE_GOLDEN = {
    "resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "test-runner"}}]},
        "scopeSpans": [{
            "scope": {"name": "local.agent", "version": "0.1"},
            "spans": [{
                "traceId": TRACE_ID,
                "spanId": SPAN_ID,
                "parentSpanId": PARENT_SPAN_ID,
                "name": "run",
                "kind": 1,
                "startTimeUnixNano": "1730000000000000000",
                "endTimeUnixNano": "1730000000000000042",
                "attributes": [{"key": "run.outcome", "value": {"stringValue": "success"}}],
                "events": [{
                    "timeUnixNano": "1730000000000000021",
                    "name": "tool.completed",
                    "attributes": [{"key": "tool.outcome", "value": {"stringValue": "ok"}}],
                }],
                "links": [{
                    "traceId": TRACE_ID,
                    "spanId": PARENT_SPAN_ID,
                    "flags": 1,
                }],
                "status": {"message": "", "code": 1},
            }],
        }],
    }],
}

_HTTP_ERRORS = (400, 401, 413, 429)
_RETRY_AFTER_VALUES = (None, "17", "Wed, 21 Oct 2037 07:28:00 GMT")
_PRIVATE_SECRET = "synthetic-secret-marker-6f117a"
_PRIVATE_SOURCE = "synthetic-source-marker-DoNotExport-42"


@pytest.fixture
def receiver_factory():
    receivers: list[OTLPReceiver] = []

    def make(*args, **kwargs) -> OTLPReceiver:
        receiver = OTLPReceiver(*args, **kwargs).start()
        receivers.append(receiver)
        return receiver

    yield make

    failures: list[str] = []
    for receiver in reversed(receivers):
        try:
            receiver.close()
            if (not receiver.closed or not receiver.socket_closed or not receiver.refused_socket_closed
                    or not receiver.thread_stopped):
                failures.append("receiver server resources remained open")
            if receiver.active_handlers:
                failures.append("receiver request handlers remained active")
        except Exception as error:
            failures.append(type(error).__name__)
    if failures:
        pytest.fail("receiver cleanup failed: " + ", ".join(failures))


def _post(url: str, body: bytes = b"{}", *, headers: dict[str, str] | None = None, timeout: float = 0.5):
    parsed = urlsplit(url)
    assert parsed.hostname is not None and parsed.port is not None
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    try:
        connection.request(
            "POST",
            parsed.path or "/",
            body=body,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        response = connection.getresponse()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        response_body = response.read()
        return response.status, response_headers, response_body
    finally:
        connection.close()


def test_trace_golden_vector_matches_strict_json_mapping():
    assert_valid_traces(TRACE_GOLDEN)

    trace_wire = json.dumps(TRACE_GOLDEN, separators=(",", ":"), ensure_ascii=True)
    assert '"traceId":"' + TRACE_ID + '"' in trace_wire
    assert '"startTimeUnixNano":"1730000000000000000"' in trace_wire
    assert '"kind":1' in trace_wire and '"code":1' in trace_wire


def test_validators_reject_bad_ids_numeric_int64_and_string_enums_without_echoing_values():
    invalid = copy.deepcopy(TRACE_GOLDEN)
    span = invalid["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    span["traceId"] = _PRIVATE_SECRET
    span["startTimeUnixNano"] = 1
    span["kind"] = "SPAN_KIND_INTERNAL"

    errors = validate_traces(invalid)
    joined = " ".join(errors)
    assert any("identifier must be 32 hexadecimal characters" in error for error in errors)
    assert any("int64 must be a decimal string" in error for error in errors)
    assert any("expected bounded integer" in error for error in errors)
    assert _PRIVATE_SECRET not in joined


def test_validator_rejects_snake_case_and_unknown_trace_fields():
    invalid = copy.deepcopy(TRACE_GOLDEN)
    span = invalid["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    span["dropped_attributes_count"] = 0
    span["unrecognizedField"] = "synthetic marker"

    errors = validate_traces(invalid)
    assert any("key is not lowerCamelCase" in error for error in errors)
    assert any("unknown key" in error for error in errors)


def test_receiver_captures_path_headers_decoded_json_and_arrival_order(receiver_factory):
    receiver = receiver_factory(("accept", "accept"))
    first_status, _, _ = _post(receiver.url + "/v1/traces", json.dumps(TRACE_GOLDEN).encode())
    second_status, _, _ = _post(receiver.url + "/v1/alternate-traces", json.dumps(TRACE_GOLDEN).encode())

    assert (first_status, second_status) == (200, 200)
    requests = receiver.requests
    assert [request.arrival_order for request in requests] == [1, 2]
    assert [request.path for request in requests] == ["/v1/traces", "/v1/alternate-traces"]
    assert [request.headers["content-type"] for request in requests] == ["application/json"] * 2
    assert requests[0].decoded_json == TRACE_GOLDEN
    assert requests[1].decoded_json == TRACE_GOLDEN


def test_receiver_records_invalid_incoming_json_without_echoing_request_bytes(receiver_factory):
    receiver = receiver_factory(("accept",))
    assert _post(receiver.url + "/v1/traces", b"{")[0] == 200
    request = receiver.requests[0]
    assert request.decoded_json is None
    assert request.decode_error == "invalid JSON"
    assert request.raw_body == b"{"


def test_partial_success_response_uses_trace_int64_count(receiver_factory):
    receiver = receiver_factory(("partial_success",))
    status, _, body = _post(receiver.url + "/v1/traces")
    assert status == 200
    assert json.loads(body)["partialSuccess"] == {"rejectedSpans": "1", "errorMessage": "rejected"}


@pytest.mark.parametrize(("status", "retry_after"), [
    (status, retry_after)
    for status in _HTTP_ERRORS
    for retry_after in _RETRY_AFTER_VALUES
])
def test_retryable_and_auth_http_status_modes(status, retry_after, receiver_factory):
    mode = f"http_{status}"
    receiver = receiver_factory((ResponseStep(mode, retry_after=retry_after),))
    actual_status, headers, body = _post(receiver.url + "/v1/traces")

    assert actual_status == status
    assert json.loads(body) == {"error": "collector rejected request"}
    if retry_after is None:
        assert "retry-after" not in headers
    else:
        assert headers["retry-after"] == retry_after


@pytest.mark.parametrize("status", (500, 502, 503, 504))
def test_server_error_modes_are_bounded(status, receiver_factory):
    receiver = receiver_factory((f"http_{status}",))
    actual_status, _, _ = _post(receiver.url + "/v1/alternate-traces")
    assert actual_status == status


@pytest.mark.parametrize(("mode", "expected_body"), (
    ("malformed_json", b'{"partialSuccess":'),
    ("empty_body", b""),
))
def test_malformed_and_empty_reply_modes(mode, expected_body, receiver_factory):
    receiver = receiver_factory((mode,))
    status, headers, body = _post(receiver.url + "/v1/traces")
    assert status == 200
    assert headers["content-length"] == str(len(expected_body))
    assert body == expected_body


def test_same_origin_redirect_is_followable_and_records_each_arrival(receiver_factory):
    receiver = receiver_factory(("redirect_same_origin", "accept"))
    status, headers, _ = _post(receiver.url + "/v1/traces")
    location = headers["location"]
    assert status == 307
    assert urlsplit(location).netloc == urlsplit(receiver.url).netloc
    assert _post(location)[0] == 200
    assert [record.path for record in receiver.requests] == ["/v1/traces", "/v1/redirected"]


def test_cross_origin_redirect_uses_a_second_loopback_collector(receiver_factory):
    target = receiver_factory(("accept",))
    receiver = receiver_factory((ResponseStep("redirect_cross_origin", target_url=target.url + "/v1/alternate-traces"),))
    status, headers, _ = _post(receiver.url + "/v1/traces")
    location = headers["location"]
    assert status == 307
    assert urlsplit(location).netloc != urlsplit(receiver.url).netloc
    assert _post(location)[0] == 200
    assert receiver.requests[0].path == "/v1/traces"
    assert target.requests[0].path == "/v1/alternate-traces"


def test_redirect_target_cannot_escape_loopback():
    with pytest.raises(ValueError, match="loopback"):
        ResponseStep("redirect_cross_origin", target_url="https://collector.invalid/v1/traces")


def test_hostile_reply_controls_cannot_exceed_helper_bounds():
    with pytest.raises(ValueError, match="half a second"):
        ResponseStep("slow_drip", delay_seconds=0.6)
    with pytest.raises(ValueError, match="bounded test range"):
        ResponseStep("oversized_reply", oversized_bytes=MAX_REPLY_BYTES + 1)
    with pytest.raises(ValueError, match="two seconds"):
        OTLPReceiver(hang_seconds=2.1)


def test_hang_before_headers_times_out_and_can_be_released(receiver_factory):
    receiver = receiver_factory(("hang_before_headers",), hang_seconds=1.0)
    connection = http.client.HTTPConnection("127.0.0.1", urlsplit(receiver.url).port, timeout=0.1)
    try:
        connection.request("POST", "/v1/traces", body=b"{}", headers={"Content-Type": "application/json"})
        assert receiver.wait_for_scripted_mode(timeout=0.5)
        with pytest.raises((TimeoutError, socket.timeout)):
            connection.getresponse()
    finally:
        connection.close()
        receiver.release_hangs()
    assert receiver.wait_for_handlers(timeout=1.0)


def test_hang_mid_body_times_out_after_headers_and_can_be_released(receiver_factory):
    receiver = receiver_factory(("hang_mid_body",), hang_seconds=1.0)
    connection = http.client.HTTPConnection("127.0.0.1", urlsplit(receiver.url).port, timeout=0.1)
    try:
        connection.request("POST", "/v1/traces", body=b"{}", headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        assert receiver.wait_for_scripted_mode(timeout=0.5)
        with pytest.raises((TimeoutError, socket.timeout)):
            response.read()
    finally:
        connection.close()
        receiver.release_hangs()
    assert receiver.wait_for_handlers(timeout=1.0)


def test_slow_drip_stalls_mid_reply_and_can_be_released(receiver_factory):
    receiver = receiver_factory((ResponseStep("slow_drip", delay_seconds=0.4),))
    connection = http.client.HTTPConnection("127.0.0.1", urlsplit(receiver.url).port, timeout=0.1)
    try:
        connection.request("POST", "/v1/alternate-traces", body=b"{}", headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        assert receiver.wait_for_scripted_mode(timeout=0.5)
        with pytest.raises((TimeoutError, socket.timeout)):
            response.read()
    finally:
        connection.close()
        receiver.release_hangs()
    assert receiver.wait_for_handlers(timeout=1.0)


def test_connection_reset_aborts_the_response(receiver_factory):
    receiver = receiver_factory(("connection_reset",))
    connection = http.client.HTTPConnection("127.0.0.1", urlsplit(receiver.url).port, timeout=0.5)
    try:
        connection.request("POST", "/v1/traces", body=b"{}", headers={"Content-Type": "application/json"})
        with pytest.raises((OSError, http.client.HTTPException)):
            connection.getresponse()
    finally:
        connection.close()


def test_oversized_reply_advertises_bounded_over_limit_body(receiver_factory):
    receiver = receiver_factory(("oversized_reply",))
    connection = http.client.HTTPConnection("127.0.0.1", urlsplit(receiver.url).port, timeout=0.5)
    try:
        connection.request("POST", "/v1/alternate-traces", body=b"{}", headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        assert int(response.getheader("Content-Length")) == OVERSIZED_REPLY_BYTES
    finally:
        connection.close()


def test_closed_port_refuses_connection_and_has_no_open_socket(receiver_factory):
    receiver = receiver_factory(("accept",))
    refused = urlsplit(receiver.refused_url)
    assert receiver.refused_socket_closed
    connection = http.client.HTTPConnection(refused.hostname, refused.port, timeout=0.5)
    try:
        with pytest.raises(OSError) as failure:
            connection.request("POST", refused.path, body=b"{}")
        assert failure.value.errno == errno.ECONNREFUSED
    finally:
        connection.close()


def test_duplicate_delivery_mode_is_found_by_span_identity(receiver_factory):
    receiver = receiver_factory(("duplicate_delivery",))
    assert _post(receiver.url + "/v1/traces", json.dumps(TRACE_GOLDEN).encode())[0] == 200
    assert _post(receiver.url + "/v1/traces", json.dumps(TRACE_GOLDEN).encode())[0] == 200

    requests = receiver.requests
    assert [request.arrival_order for request in requests] == [1, 2, 3]
    assert requests[1].duplicate_of == 1
    duplicates = find_duplicate_spans(requests)
    assert len(duplicates) == 2
    assert duplicates[0].trace_id == TRACE_ID
    assert duplicates[0].span_id == SPAN_ID
    assert (duplicates[0].first_arrival_order, duplicates[0].duplicate_arrival_order) == (1, 2)
    assert (duplicates[1].first_arrival_order, duplicates[1].duplicate_arrival_order) == (1, 3)
    with pytest.raises(AssertionError, match="duplicate span"):
        assert_no_duplicate_spans(requests)


def test_planted_secret_and_source_scanner_covers_headers_and_decoded_body(receiver_factory):
    receiver = receiver_factory(("accept",))
    payload = copy.deepcopy(TRACE_GOLDEN)
    span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    span["attributes"].append({"key": "synthetic.marker", "value": {"stringValue": _PRIVATE_SOURCE}})
    _post(
        receiver.url + "/v1/traces",
        json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + _PRIVATE_SECRET},
    )

    captured = receiver.requests
    assert planted_capture_leaks(captured, (_PRIVATE_SECRET, _PRIVATE_SOURCE)) == (0, 1)
    assert planted_capture_leaks({"wire": _PRIVATE_SECRET.encode("utf-8")}, (_PRIVATE_SECRET,)) == (0,)
    assert_capture_clean(captured, ("marker-that-is-not-present",))


def test_capture_scanner_detects_json_escaped_plants_without_echoing_them():
    marker = 'private"marker\\tail'
    escaped = json.dumps(marker)[1:-1]
    assert planted_capture_leaks({"captured": escaped}, (marker,)) == (0,)
