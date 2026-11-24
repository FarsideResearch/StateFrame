"""Provider 429 retries and the host timers around inter-send waits."""
from __future__ import annotations

from contextlib import contextmanager
import datetime
import email.utils
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import kernel_bounded_http

from stateframe import _child, runner
from stateframe.engine import kernel_measurement
from stateframe.replay_tape import Replayer, build as build_replay_tape, pending_request


REPLY = b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
BODY = {"messages": [{"role": "user", "content": "hello"}], "tools": []}
WIRE = json.dumps(BODY).encode()


@contextmanager
def _server(*, statuses: tuple[int, ...] | dict[str, tuple[int, ...]],
            retry_after: str | None = None, bodies: tuple[bytes, ...] | None = None,
            delays: tuple[float, ...] = (), resets: frozenset[int] = frozenset()):
    received: list[bytes] = []
    sent_at: list[float] = []
    body_counts: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            request_body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append(request_body)
            sent_at.append(time.monotonic())
            request_obj = json.loads(request_body)
            key = str(request_obj["messages"][0]["content"])
            body_counts[key] = body_counts.get(key, 0) + 1
            sequence = statuses.get(key, (500,)) if isinstance(statuses, dict) else statuses
            status = sequence[min(body_counts[key] - 1, len(sequence) - 1)]
            if len(received) in resets:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            if len(received) <= len(delays) and delays[len(received) - 1] > 0:
                time.sleep(delays[len(received) - 1])
            body = (bodies[min(len(received) - 1, len(bodies) - 1)] if bodies is not None
                    else REPLY if status == 200 else b'{"error":{"message":"busy"}}')
            self.send_response(status)
            if retry_after is not None and status in {429, 503}:
                self.send_header("Retry-After", retry_after)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
               received, sent_at)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _events(journal):
    return [json.loads(line) for line in journal.path.read_text().splitlines()]


def _send(tmp_path, endpoint, *, deadline, socket_timeout=0.05,
          body=BODY, wire=WIRE, cancel_requested=False):
    journal = kernel_measurement.AuditJournal(tmp_path, "provider-rate-limit", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    capture.cancel_requested = cancel_requested
    try:
        response = capture._send_transport_with_retries(
            body, wire, endpoint, "", admission=None, socket_timeout=socket_timeout,
            deadline=deadline, wall_timeout=max(1, socket_timeout), backend_identity={},
            request_extensions={}, frozen_retry_only=True)
        return response, _events(journal)
    except KeyboardInterrupt as error:
        error.events = _events(journal)
        raise
    except kernel_measurement.HarnessTransportError as error:
        error.events = _events(journal)
        raise


@pytest.mark.host_sandbox
def test_inter_send_wait_longer_than_response_stall_window_is_not_a_stall(
        monkeypatch, tmp_path):
    """A real retry wait exceeds the per-response silence limit without arming it in between."""
    monkeypatch.setattr(kernel_bounded_http, "send_stall_seconds", lambda: 0.05)
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.2")
    with _server(statuses=(503, 200)) as (endpoint, received, sent_at):
        (response, _, _), events = _send(
            tmp_path, endpoint, deadline=time.monotonic() + 60)

    assert response["choices"][0]["message"]["content"] == "ok"
    assert received == [WIRE, WIRE]
    assert sent_at[1] - sent_at[0] > 0.15
    assert not [event for event in events if event["kind"] == "send_stall"]
    assert [event["kind"] for event in events if event["kind"] == "transport_error"] == [
        "transport_error"]


@pytest.mark.host_sandbox
@pytest.mark.parametrize("header_kind", ["delta", "date"])
def test_429_retry_after_waits_before_resending_same_bytes(
        monkeypatch, tmp_path, header_kind):
    monkeypatch.delenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", raising=False)
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.01")
    if header_kind == "delta":
        retry_after, minimum_wait = "0.25", 0.20
    else:
        target = (datetime.datetime.now(datetime.timezone.utc)
                  + datetime.timedelta(seconds=4)).replace(microsecond=0)
        retry_after, minimum_wait = email.utils.format_datetime(target, usegmt=True), 0.0
    with _server(statuses=(429, 200), retry_after=retry_after) as (endpoint, received, sent_at):
        (response, _, _), events = _send(
            tmp_path, endpoint, deadline=time.monotonic() + 60)

    assert response["choices"][0]["message"]["content"] == "ok"
    assert received == [WIRE, WIRE]
    assert sent_at[1] - sent_at[0] >= minimum_wait
    if header_kind == "date":
        resent_at = (datetime.datetime.now(datetime.timezone.utc)
                     - datetime.timedelta(seconds=time.monotonic() - sent_at[1]))
        assert resent_at >= target - datetime.timedelta(seconds=0.2)
    limits = [event for event in events if event["kind"] == "transport_error"]
    assert len(limits) == 1
    assert limits[0]["submission_state"] == "rejected"
    assert limits[0]["usage_state"] == "not_incurred"
    assert limits[0]["retry_blocked_reason"] is None
    waits = [event for event in events if event["kind"] == "provider_rate_limited"]
    assert len(waits) == 1
    assert waits[0]["retry_delay_seconds"] >= minimum_wait
    assert waits[0]["retry_after_s"] >= minimum_wait
    assert "Retry-After" not in json.dumps(events)


@pytest.mark.host_sandbox
def test_ten_429_responses_retry_past_default_attempt_count(monkeypatch, tmp_path):
    monkeypatch.delenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", raising=False)
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.001")
    with _server(statuses=(429,) * 10 + (200,)) as (endpoint, received, _sent_at):
        (response, _, _), events = _send(
            tmp_path, endpoint, deadline=time.monotonic() + 60)

    assert response["choices"][0]["message"]["content"] == "ok"
    assert received == [WIRE] * 11
    errors = [event for event in events if event["kind"] == "transport_error"]
    assert len(errors) == 10
    assert [event["ordinary_attempt_count"] for event in errors] == [0] * 10
    assert len([event for event in events if event["kind"] == "provider_rate_limited"]) == 10


@pytest.mark.host_sandbox
@pytest.mark.parametrize("attempt_cap", [1, 2])
def test_explicit_attempt_cap_bounds_429_retries(monkeypatch, tmp_path, attempt_cap):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", str(attempt_cap))
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.001")
    with _server(statuses=(429,) * 4) as (endpoint, received, _sent_at):
        with pytest.raises(kernel_measurement.ProviderRateLimitError) as caught:
            _send(tmp_path, endpoint, deadline=time.monotonic() + 60)

    assert caught.value.attempts == attempt_cap
    assert str(caught.value) == "provider rate limit (HTTP 429)"
    assert received == [WIRE] * attempt_cap
    errors = [event for event in caught.value.events if event["kind"] == "transport_error"]
    assert errors[-1]["retry_blocked_reason"] == "attempt_limit"
    assert errors[-1]["submission_state"] == "rejected"
    assert errors[-1]["usage_state"] == "not_incurred"
    assert all(event["retry_blocked_reason"] != "ambiguous_submission" for event in errors)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("typed_field", ["type", "code"])
def test_typed_quota_429_stops_but_free_text_quota_retries(
        monkeypatch, tmp_path, typed_field):
    monkeypatch.delenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", raising=False)
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.001")
    quota = json.dumps({"error": {
        typed_field: "insufficient_quota", "message": "plan exhausted",
    }}).encode()
    with _server(statuses=(429,), bodies=(quota,)) as (endpoint, received, _sent_at):
        with pytest.raises(kernel_measurement.ProviderRateLimitError) as caught:
            _send(tmp_path / "quota", endpoint, deadline=time.monotonic() + 60)
    assert str(caught.value) == "provider quota exhausted (HTTP 429)"
    assert len(received) == 1
    assert caught.value.events[-1]["kind"] != "provider_rate_limited"
    assert any(event.get("retry_blocked_reason") == "provider_quota_exhausted"
               for event in caught.value.events)

    free_text = b'{"error":{"message":"quota exceeded; try later"}}'
    with _server(statuses=(429, 200), bodies=(free_text, REPLY)) as (endpoint, received, _sent_at):
        (response, _, _), events = _send(
            tmp_path / "free-text", endpoint, deadline=time.monotonic() + 60)
    assert response["choices"][0]["message"]["content"] == "ok"
    assert received == [WIRE, WIRE]
    assert len([event for event in events if event["kind"] == "provider_rate_limited"]) == 1


@pytest.mark.host_sandbox
@pytest.mark.parametrize("header", ["not-a-date", "-1", "NaN", "Infinity", None])
def test_malformed_negative_and_nan_retry_after_use_existing_backoff(
        monkeypatch, tmp_path, header: str | None):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.001")
    with _server(statuses=(429, 200), retry_after=header) as (endpoint, received, _sent_at):
        (response, _, _), events = _send(
            tmp_path, endpoint, deadline=time.monotonic() + 60)
    assert response["choices"][0]["message"]["content"] == "ok"
    assert received == [WIRE, WIRE]
    error = next(event for event in events if event["kind"] == "transport_error")
    assert error["retry_after_s"] is None
    assert "Retry-After" not in json.dumps(events)


@pytest.mark.host_sandbox
def test_finite_huge_retry_after_stops_at_explicit_queue_wait_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "30")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.001")
    header = "100000000000000000000"
    with _server(statuses=(429,), retry_after=header) as (endpoint, received, _sent_at):
        with pytest.raises(kernel_measurement.ProviderRateLimitError) as caught:
            _send(tmp_path, endpoint, deadline=time.monotonic() + 60)
    assert received == [WIRE]
    error = next(event for event in caught.value.events if event["kind"] == "transport_error")
    assert error["retry_after_s"] == 1e20
    assert error["retry_blocked_reason"] == "queue_wait_cap"
    assert header not in json.dumps(caught.value.events)


@pytest.mark.host_sandbox
def test_deadline_shorter_than_retry_after_stops_without_sleep(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.001")
    with _server(statuses=(429, 200), retry_after="60") as (endpoint, received, _sent_at):
        started = time.monotonic()
        with pytest.raises(kernel_measurement.ProviderRateLimitError) as caught:
            _send(tmp_path, endpoint, deadline=started + 30)
    assert received == [WIRE]
    assert time.monotonic() - started < 30
    assert next(event for event in caught.value.events if event["kind"] == "transport_error")[
        "retry_blocked_reason"] == "deadline"


@pytest.mark.host_sandbox
def test_cancel_interrupts_a_huge_retry_after_and_wait_is_rendered(monkeypatch, tmp_path):
    monkeypatch.delenv("STATEFRAME_QUEUE_WAIT_MAX_S", raising=False)
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.001")
    cancelled = threading.Event()
    timer = threading.Timer(0.04, cancelled.set)
    timer.start()
    try:
        with _server(statuses=(429, 200), retry_after="100000000000000000000") as (
                endpoint, received, _sent_at):
            with pytest.raises(KeyboardInterrupt) as caught:
                _send(tmp_path, endpoint, deadline=None,
                      cancel_requested=cancelled)
    finally:
        timer.cancel()
    assert received == [WIRE]
    wait = next(event for event in caught.value.events
                if event["kind"] == "provider_rate_limited")
    line = runner._render_event(wait)
    assert "HTTP 429" in line and "1e+20 seconds" in line
    assert "Retry-After" not in json.dumps(caught.value.events)


@pytest.mark.host_sandbox
def test_429_after_response_bytes_is_terminal(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.001")
    sends = []

    def partial(_capture, wire, *_args, **_kwargs):
        sends.append(wire)
        error = kernel_bounded_http.BoundedTransportError(
            error_type="PartialResponseError", http_status=429, relay_error=None,
            response_bytes_received=True, retry_after_s=0.001)
        raise error

    monkeypatch.setattr(kernel_measurement, "_request_bytes_for_capture", partial)
    with pytest.raises(kernel_measurement.ProviderRateLimitError) as caught:
        _send(tmp_path, "http://127.0.0.1:1/v1/chat/completions",
              deadline=time.monotonic() + 60)
    assert sends == [WIRE]
    error = next(event for event in caught.value.events if event["kind"] == "transport_error")
    assert error["retry_blocked_reason"] == "response_bytes_received"
    assert error["submission_state"] == "rejected"
    assert error["usage_state"] == "not_incurred"


@pytest.mark.host_sandbox
@pytest.mark.parametrize("status,description", [
    (400, "request rejected"), (401, "authentication rejected"),
    (403, "access denied"), (404, "model or endpoint not found"),
])
def test_other_provider_4xx_stays_terminal(monkeypatch, tmp_path, status, description):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", "3")
    with _server(statuses=(status,) * 4) as (endpoint, received, _sent_at):
        with pytest.raises(kernel_measurement.ProviderHTTPError) as caught:
            _send(tmp_path, endpoint, deadline=time.monotonic() + 60)
    assert received == [WIRE]
    assert caught.value.http_status == status
    assert description in str(caught.value)


@pytest.mark.host_sandbox
def test_non429_error_keeps_the_existing_typed_error_event_shape(monkeypatch, tmp_path):
    body = json.dumps({"error": {
        "type": "invalid_request_error", "code": "invalid_grammar",
        "message": "grammar rejected",
    }}).encode()
    with _server(statuses=(400,), bodies=(body,)) as (endpoint, received, _sent_at):
        with pytest.raises(kernel_measurement.ProviderHTTPError) as caught:
            _send(tmp_path, endpoint, deadline=time.monotonic() + 60)

    assert received == [WIRE]
    error = next(event for event in caught.value.events if event["kind"] == "transport_error")
    assert error["server_error"] == {
        "message": "grammar rejected", "type": "invalid_request_error",
    }


@pytest.mark.host_sandbox
@pytest.mark.parametrize("has_header", [False, True])
def test_fast_503_retries_and_honors_retry_after(monkeypatch, tmp_path, has_header):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.01")
    retry_after = "0.2" if has_header else None
    with _server(statuses=(503, 200), retry_after=retry_after) as (
            endpoint, received, sent_at):
        (response, _, _), events = _send(
            tmp_path, endpoint, deadline=time.monotonic() + 60)
    assert response["choices"][0]["message"]["content"] == "ok"
    assert received == [WIRE, WIRE]
    assert sent_at[1] - sent_at[0] >= (0.16 if has_header else 0.005)
    if has_header:
        error = next(event for event in events if event["kind"] == "transport_error")
        assert error["retry_after_s"] == 0.2


@pytest.mark.host_sandbox
def test_slow_503_with_retry_after_stays_terminal(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.01")
    with _server(statuses=(503, 200), retry_after="0.01", delays=(2.1,)) as (
            endpoint, received, _sent_at):
        with pytest.raises(kernel_measurement.HarnessTransportError) as caught:
            _send(tmp_path, endpoint, deadline=time.monotonic() + 5, socket_timeout=3)
    assert len(received) == 1
    assert caught.value.http_status == 503


@pytest.mark.host_sandbox
def test_429_then_connection_resets_keep_ordinary_attempt_budget(monkeypatch, tmp_path):
    monkeypatch.delenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", raising=False)
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0")
    with _server(statuses=(429, 200, 200, 200), resets=frozenset({2, 3, 4})) as (
            endpoint, received, _sent_at):
        with pytest.raises(kernel_measurement.HarnessTransportError) as caught:
            _send(tmp_path, endpoint, deadline=time.monotonic() + 60)
    assert received == [WIRE] * 4
    errors = [event for event in caught.value.events if event["kind"] == "transport_error"]
    assert [event["http_status"] for event in errors] == [429, None, None, None]
    assert [event["ordinary_attempt_count"] for event in errors] == [0, 1, 2, 3]
    assert [event["retry_scheduled"] for event in errors] == [True, True, True, False]


@pytest.mark.host_sandbox
def test_two_parallel_lanes_retry_429_independently(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.01")
    bodies = [
        {"messages": [{"role": "user", "content": name}], "tools": []}
        for name in ("lane-1", "lane-2")
    ]
    wires = [json.dumps(body).encode() for body in bodies]
    statuses = {"lane-1": (429, 200), "lane-2": (429, 200)}
    with _server(statuses=statuses) as (endpoint, received, _sent_at):
        with ThreadPoolExecutor(max_workers=2) as lanes:
            results = list(lanes.map(
                lambda item: _send(tmp_path / item[0], endpoint,
                                   deadline=time.monotonic() + 5,
                                   body=item[1], wire=item[2]),
                zip(("lane-1", "lane-2"), bodies, wires)))
    assert [result[0][0]["choices"][0]["message"]["content"] for result in results] == [
        "ok", "ok"]
    assert sorted(received) == sorted([wires[0], wires[0], wires[1], wires[1]])
    assert all(len([event for event in result[1]
                    if event["kind"] == "provider_rate_limited"]) == 1
               for result in results)


@pytest.mark.host_sandbox
def test_kill_during_429_wait_resumes_with_a_live_send(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.001")
    original_request_bytes = kernel_bounded_http.request_bytes
    original_host_state = _child.HOST_STATE
    monkeypatch.setattr(_child, "HOST_STATE", dict(original_host_state))
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    with _server(statuses=(429, 200), retry_after="0.5") as (endpoint, received, _sent_at):
        _child.install_wire_recorder(attempt, endpoint)
        monkeypatch.setattr(kernel_measurement, "_wait_interruptibly",
                            lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt) as caught:
            _send(attempt / "journal", endpoint, deadline=time.monotonic() + 60)
        errors = [event for event in caught.value.events if event["kind"] == "transport_error"]
        assert errors[-1]["submission_state"] == "rejected"
        assert errors[-1]["usage_state"] == "not_incurred"
        assert errors[-1]["retry_scheduled"] is True
        assert errors[-1]["retry_blocked_reason"] is None
        assert not any(event.get("retry_blocked_reason") == "ambiguous_submission"
                       for event in errors)

        tape = build_replay_tape(attempt / "wire", attempt / "sends.jsonl")
        pending = pending_request(attempt / "wire")
        assert tape == [] and pending is None
        kernel_bounded_http.request_bytes = original_request_bytes
        _child.install_wire_recorder(resumed, endpoint, Replayer(tape, pending=pending))
        response = kernel_bounded_http.request_bytes(
            WIRE, endpoint, "", deadline=time.monotonic() + 5, socket_timeout=0.05)
        kernel_bounded_http.request_bytes = original_request_bytes
        _child.HOST_STATE = original_host_state

    assert json.loads(response)["choices"][0]["message"]["content"] == "ok"
    assert received == [WIRE, WIRE]


def test_rate_limit_event_is_a_known_resume_event_kind():
    import kernel_continuation_restart as restart

    source = Path(restart.__file__).read_text(encoding="utf-8")
    assert " provider_rate_limited " in source


def test_child_shows_each_rate_limit_wait_as_it_starts(monkeypatch, capsys):
    import kernel_measurement as engine_measurement

    monkeypatch.setattr(engine_measurement, "RATE_LIMIT_NOTICE", None)
    _child.install_rate_limit_notice()
    engine_measurement.RATE_LIMIT_NOTICE(http_status=429, retry_delay_seconds=40.0)
    line = capsys.readouterr().out.strip()
    assert line.startswith("[sf] ")
    assert runner._render_event(json.loads(line[len("[sf] "):])) == (
        "provider rate limited (HTTP 429); retrying in 40 seconds")


def test_retry_loop_calls_the_notice_before_it_waits():
    source = Path(kernel_measurement.__file__).read_text(encoding="utf-8")
    notice = source.index("RATE_LIMIT_NOTICE(http_status=429")
    assert notice < source.index("_wait_interruptibly(self, delay)", notice)


@pytest.mark.host_sandbox
def test_ttft_retries_do_not_use_the_users_attempt_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.001")
    monkeypatch.setattr(kernel_measurement, "_ttft_retry_delay_seconds", lambda _index: 0.0)
    ttft = json.dumps({"error": {"label": "relay_ttft_timeout", "message": "slow"}}).encode()
    busy = b'{"error":{"message":"busy"}}'
    with _server(statuses=(504, 504, 503, 200), bodies=(ttft, ttft, busy, REPLY)) as (
            endpoint, received, _sent_at):
        (response, _, _), events = _send(
            tmp_path, endpoint, deadline=time.monotonic() + 30, socket_timeout=5)

    assert response["choices"][0]["message"]["content"] == "ok"
    assert received == [WIRE] * 4
    errors = [event for event in events if event["kind"] == "transport_error"]
    assert [event["retry_budget"] for event in errors] == ["ttft", "ttft", "ordinary_transport"]
    assert errors[-1]["ordinary_attempt_count"] == 1


@pytest.mark.host_sandbox
def test_users_cap_counts_429_and_ordinary_attempts_together(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_TRANSPORT_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_S", "0.001")
    with _server(statuses=(429, 503, 200)) as (endpoint, received, _sent_at):
        with pytest.raises(kernel_measurement.HarnessTransportError):
            _send(tmp_path, endpoint, deadline=time.monotonic() + 30, socket_timeout=5)
    assert received == [WIRE] * 2


def test_context_limit_429_takes_one_refit_then_stops(tmp_path):
    from test_provider_4xx_typed import _run_cli

    body = {"error": {"code": "context_length_exceeded",
                      "message": "maximum context length is 32768 tokens"}}
    exit_code, summary, events, output, requests = _run_cli(
        tmp_path, (429,) * 6, error_bodies=(body,) * 6)

    assert exit_code == 2, output
    assert len(requests) == 2 and requests[0] != requests[1], output
    assert len([row for row in events if row.get("kind") == "capacity_retry"]) == 1
    assert not [row for row in events if row.get("kind") == "provider_rate_limited"]
    errors = [row for row in events if row.get("kind") == "transport_error"]
    assert [row["retry_blocked_reason"] for row in errors] == [None, "capacity_retry_exhausted"]
    assert {row["submission_state"] for row in errors} == {"rejected"}


@pytest.mark.parametrize("message", [
    "Rate limit reached on tokens per min (TPM): Limit 30000, Used 29000, Requested 2500.",
    "Request too large for model on tokens per min (TPM): Limit 30000, Requested 45000.",
    "You exceeded your current quota, please check your plan and billing details.",
    "This request would exceed your organization's rate limit of 40,000 input tokens per minute.",
    "Too many tokens, please wait before trying again.",
])
def test_rate_or_quota_prose_is_not_a_context_limit(message):
    body = json.dumps({"error": {"code": "rate_limit_exceeded", "message": message}})
    assert kernel_bounded_http.capacity_400_reason(400, body) is None
    assert not kernel_measurement._provider_quota_exhausted(body)
