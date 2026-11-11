"""Transient pre-response failures keep one logical request and a bounded retry budget."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import struct
import threading
import time

import pytest

from _timeouts import scaled
from stateframe import runner
from stateframe.engine import kernel_bounded_http, kernel_measurement


REPLY = b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
BODY = {"messages": [{"role": "user", "content": "hello"}], "tools": []}
WIRE = json.dumps(BODY).encode()


@contextmanager
def _server(*, drops: int, partial_body: bool = False, status: int = 200,
            reset_first: bool = False):
    received: list[bytes] = []
    connections: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            received.append(self.rfile.read(int(self.headers["Content-Length"])))
            if len(received) <= drops:
                if partial_body:
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\nx")
                    self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            body = REPLY if status == 200 else b'{"error":"invalid request"}'
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class Server(ThreadingHTTPServer):
        def process_request(self, request, client_address):
            connections.append(client_address[1])
            if reset_first and len(connections) == 1:
                request.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                   struct.pack("ii", 1, 0))
                request.close()
                return
            super().process_request(request, client_address)

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
               received, connections)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=scaled(5))


def _events(journal):
    return [json.loads(line) for line in journal.path.read_text().splitlines()]


def _send(tmp_path, endpoint, *, deadline):
    journal = kernel_measurement.AuditJournal(tmp_path, "transport-recovery", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    try:
        result = capture._send_transport_with_retries(
            BODY, WIRE, endpoint, "", admission=None, socket_timeout=scaled(3),
            deadline=deadline, wall_timeout=scaled(3), backend_identity={},
            request_extensions={}, frozen_retry_only=True)
        return result, _events(journal)
    except kernel_measurement.HarnessTransportError as error:
        error.events = _events(journal)
        raise


@pytest.mark.host_sandbox
def test_connection_drop_retries_same_request_and_journals_second_attempt(
        monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.01")
    with _server(drops=0, reset_first=True) as (endpoint, received, connections):
        (response, _, _), events = _send(
            tmp_path, endpoint, deadline=time.monotonic() + scaled(10))
    assert response["choices"][0]["message"]["content"] == "ok"
    assert received == [WIRE]
    assert len(connections) == 2
    submitted = [event for event in events if event["kind"] == "request_submitted"]
    assert [event["transport_attempt"] for event in submitted] == [1, 2]
    assert len({event["request_id"] for event in submitted}) == 1
    assert submitted[1]["payload_delta"]["identical_wire"] is True
    assert [event["retry_scheduled"] for event in events
            if event["kind"] == "transport_error"] == [True]


@pytest.mark.host_sandbox
def test_connection_drop_exhausts_bounded_attempts(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.01")
    with _server(drops=3) as (endpoint, received, _connections):
        with pytest.raises(kernel_measurement.HarnessTransportError) as caught:
            _send(tmp_path, endpoint, deadline=time.monotonic() + scaled(10))
    assert caught.value.attempts == 2
    assert received == [WIRE, WIRE]
    errors = [event for event in caught.value.events if event["kind"] == "transport_error"]
    assert [event["transport_attempt"] for event in errors] == [1, 2]
    assert errors[-1]["retry_blocked_reason"] == "attempt_limit"


@pytest.mark.host_sandbox
def test_connection_drop_cannot_retry_past_run_deadline(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", str(scaled(2)))
    with _server(drops=1) as (endpoint, received, _connections):
        with pytest.raises(kernel_measurement.HarnessTransportError) as caught:
            _send(tmp_path, endpoint, deadline=time.monotonic() + scaled(1))
    assert received == [WIRE]
    errors = [event for event in caught.value.events if event["kind"] == "transport_error"]
    assert len(errors) == 1
    assert errors[0]["retry_blocked_reason"] == "deadline"


@pytest.mark.host_sandbox
@pytest.mark.parametrize("partial_body,status", [(True, 200), (False, 400)])
def test_response_bytes_and_http_400_are_never_retried(
        monkeypatch, tmp_path, partial_body, status):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.01")
    with _server(drops=1 if partial_body else 0,
                 partial_body=partial_body, status=status) as (endpoint, received, _connections):
        with pytest.raises(kernel_measurement.HarnessTransportError) as caught:
            _send(tmp_path, endpoint, deadline=time.monotonic() + scaled(10))
    assert received == [WIRE]
    assert len([event for event in caught.value.events
                if event["kind"] == "transport_error"]) == 1


def test_transport_retry_switches_are_forwarded_and_bounded(monkeypatch):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.25")
    selected = runner.resolve_opt_in_switches()
    assert selected["STATEFRAME_TRANSPORT_MAX_ATTEMPTS"] == "4"
    assert selected["STATEFRAME_TRANSPORT_BACKOFF_S"] == "0.25"
    assert kernel_measurement._transport_retry_settings(3) == (4, 0.25)


@pytest.mark.parametrize("attempts,backoff", [
    ("0", "1"), ("6", "1"), ("bad", "1"),
    ("3", "-1"), ("3", "nan"), ("3", "inf"), ("3", "61"),
])
def test_invalid_transport_retry_switches_fail_closed(monkeypatch, attempts, backoff):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", attempts)
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", backoff)
    with pytest.raises(runner.RunError):
        runner.resolve_opt_in_switches()
    with pytest.raises(ValueError):
        kernel_measurement._transport_retry_settings(3)


@pytest.mark.parametrize("pre_response,response_bytes,retryable", [
    (True, False, True), (False, True, False), (False, False, False),
    (True, True, False),
])
def test_timeout_retry_requires_positive_no_response_evidence(
        pre_response, response_bytes, retryable):
    diagnostic = json.dumps({
        "error_type": "TimeoutError", "pre_response_failure": pre_response,
        "response_bytes_received": response_bytes,
    }).encode()
    error = kernel_bounded_http._child_transport_error(diagnostic, "", "http://127.0.0.1:1")
    assert kernel_measurement._transport_retryable(
        error, elapsed_ns=3_000_000_000, http_status=None, relay_error=None,
        error_type="TimeoutError") is retryable


def test_partial_response_timeout_does_not_use_timeout_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_PER_REQUEST_TIMEOUT_S", "1")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0")
    sends = []

    def send(_capture, wire, *_args, **_kwargs):
        sends.append(wire)
        error = TimeoutError("response body timed out")
        error.response_bytes_received = True
        raise error

    monkeypatch.setattr(kernel_measurement, "_request_bytes_for_capture", send)
    journal = kernel_measurement.AuditJournal(tmp_path, "partial-timeout", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    with pytest.raises(kernel_measurement.HarnessTransportError):
        capture._send_transport_with_retries(
            BODY, WIRE, "http://127.0.0.1:1/v1/chat/completions", "",
            admission=None, socket_timeout=scaled(3),
            deadline=time.monotonic() + scaled(10), wall_timeout=scaled(3),
            backend_identity={}, request_extensions={}, frozen_retry_only=True)
    assert sends == [WIRE]
