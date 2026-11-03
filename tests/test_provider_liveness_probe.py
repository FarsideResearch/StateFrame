"""In-flight provider probes must distinguish HTTP replies from lost transport."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from stateframe.engine import kernel_bounded_http
from stateframe.engine.probes import react_repair
from stateframe import runner


WIRE = b'{"model":"scripted","messages":[{"role":"user","content":"exact bytes"}]}'
REPLY = b'{"choices":[{"message":{"content":"done"}}]}'
LIVENESS_S = 0.15


@contextmanager
def provider(*, slots_status: int = 401, hold_post: bool = True,
             post_delay: float = 0.45, stream_stall: bool = False,
             trickle_headers: bool = False):
    posts: list[tuple[str, bytes, str | None]] = []
    probes: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            posts.append((self.path, body, self.headers.get("Authorization")))
            if trickle_headers:
                try:
                    self.wfile.write(b"HTTP/1.1 200 ")
                    self.wfile.flush()
                    for _ in range(8):
                        time.sleep(post_delay / 8)
                        self.wfile.write(b" ")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if hold_post and not stream_stall:
                time.sleep(post_delay)
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(REPLY)))
                self.end_headers()
                if stream_stall:
                    self.wfile.write(REPLY[:1])
                    self.wfile.flush()
                    time.sleep(post_delay)
                    self.wfile.write(REPLY[1:])
                else:
                    self.wfile.write(REPLY)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):  # noqa: N802
            probes.append((self.path, self.headers.get("Authorization")))
            status = slots_status if self.path == "/slots" else 503
            if self.path == "/health" and not hold_post:
                status = 200
            body = b"ok" if status == 200 else b"busy"
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", posts, probes
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.host_sandbox
@pytest.mark.parametrize(("slots_status", "key"), [(401, "route-key"), (501, "")])
def test_busy_server_http_probe_response_keeps_single_post(monkeypatch, slots_status, key):
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", str(LIVENESS_S))
    with provider(slots_status=slots_status) as (endpoint, posts, probes):
        reply = kernel_bounded_http.request_bytes(
            WIRE, endpoint, key, deadline=time.monotonic() + 5,
            socket_timeout=3)
        assert reply == REPLY
        assert posts == [("/v1/chat/completions", WIRE, f"Bearer {key}")]
        assert probes
        assert all(path in {"/slots", "/health"} for path, _ in probes)
        assert all(auth == (f"Bearer {key}" if key else None) for _, auth in probes)


@pytest.mark.host_sandbox
def test_transport_probe_failures_still_raise_liveness_timeout(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", str(LIVENESS_S))
    with provider() as (endpoint, posts, probes):
        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            dead_port = unused.getsockname()[1]
        monkeypatch.setattr(kernel_bounded_http, "_provider_probe_origin",
                            lambda _endpoint: ("http", "127.0.0.1", dead_port))
        with pytest.raises(kernel_bounded_http.ProviderLivenessTimeout):
            kernel_bounded_http.request_bytes(
                WIRE, endpoint, "", deadline=time.monotonic() + 5,
                socket_timeout=3)
        assert len(posts) <= 1
        assert all(post == ("/v1/chat/completions", WIRE, "Bearer ") for post in posts)
        assert probes == []


@pytest.mark.host_sandbox
def test_routed_port_receives_probes_and_default_port_does_not(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", str(LIVENESS_S))
    with provider(hold_post=False) as (_default, default_posts, default_probes):
        with provider(slots_status=501) as (routed, routed_posts, routed_probes):
            reply = kernel_bounded_http.request_bytes(
                WIRE, routed, "route-key", deadline=time.monotonic() + 5,
                socket_timeout=3,
                route_meta={"role": "read"})
            assert reply == REPLY
            assert routed_posts == [("/v1/chat/completions", WIRE, "Bearer route-key")]
            assert routed_probes
            assert default_posts == default_probes == []


@pytest.mark.host_sandbox
def test_recovery_uses_failed_request_root_and_authorization(monkeypatch):
    with provider(hold_post=False) as (default, _default_posts, default_probes):
        with provider(hold_post=False) as (routed, _routed_posts, routed_probes):
            monkeypatch.setattr(react_repair, "API", default)
            capture = SimpleNamespace(_provider_probe_destination=(routed, "route-key"))
            endpoint, key = react_repair._failed_request_destination(capture)
            assert react_repair._await_endpoint(endpoint, key, max_wait=1)
            assert routed_probes == [("/health", "Bearer route-key")]
            assert default_probes == []


@pytest.mark.host_sandbox
def test_empty_key_model_header_matches_5140116_behavior():
    with provider(hold_post=False) as (endpoint, posts, _probes):
        assert kernel_bounded_http.request_bytes(
            WIRE, endpoint, "", deadline=time.monotonic() + 2,
            socket_timeout=1) == REPLY
    assert posts == [("/v1/chat/completions", WIRE, "Bearer ")]


@pytest.mark.host_sandbox
def test_probe_redirect_never_sends_key_to_second_origin(monkeypatch):
    received: list[str | None] = []

    class Destination(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            received.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

    destination = ThreadingHTTPServer(("127.0.0.1", 0), Destination)
    destination_thread = threading.Thread(target=destination.serve_forever, daemon=True)
    destination_thread.start()

    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{destination.server_port}/health")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    source = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    source_thread = threading.Thread(target=source.serve_forever, daemon=True)
    source_thread.start()
    try:
        endpoint = f"http://127.0.0.1:{source.server_port}/v1/chat/completions"
        assert kernel_bounded_http._probe_liveness(endpoint, "secret-key", 1) == 302
        assert not react_repair._await_endpoint(endpoint, "secret-key", max_wait=0.25)
        assert received == []
    finally:
        source.shutdown()
        source.server_close()
        source_thread.join(timeout=5)
        destination.shutdown()
        destination.server_close()
        destination_thread.join(timeout=5)


@pytest.mark.host_sandbox
def test_trickling_probe_headers_cannot_pass_hard_deadline(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", "4")
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "3")

    class Trickle(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            try:
                self.wfile.write(b"HTTP/1.1 200 ")
                self.wfile.flush()
                for _ in range(20):
                    time.sleep(0.1)
                    self.wfile.write(b" ")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(2)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Trickle)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            kernel_bounded_http.request_bytes(
                WIRE, f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                "", deadline=started + 0.35, run_deadline=started + 1,
                socket_timeout=0.35)
        assert time.monotonic() - started < 1.5
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.host_sandbox
def test_recovery_trickling_health_body_obeys_wait_deadline():
    class Trickle(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"o")
            self.wfile.flush()
            time.sleep(1)
            try:
                self.wfile.write(b"k")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Trickle)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        assert not react_repair._await_endpoint(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            "secret-key", max_wait=0.3)
        assert time.monotonic() - started < 0.7
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_probe_process_start_failure_is_a_failed_probe(monkeypatch):
    def fail_start(*_args, **_kwargs):
        raise OSError("process limit")

    monkeypatch.setattr(kernel_bounded_http.subprocess, "Popen", fail_start)
    assert kernel_bounded_http._probe_liveness(
        "http://localhost:1/v1/chat/completions", "key", 0.5) is None


@pytest.mark.host_sandbox
def test_busy_queue_waits_past_request_cap_without_resending(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", "0.5")
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "5")
    waits = []
    with provider(post_delay=2.2) as (endpoint, posts, probes):
        started = time.monotonic()
        reply = kernel_bounded_http.request_bytes(
            WIRE, endpoint, "route-key", deadline=started + 1,
            run_deadline=started + 6, socket_timeout=1,
            record_queue_wait=lambda *values: waits.append(values))
        assert reply == REPLY
        assert posts == [("/v1/chat/completions", WIRE, "Bearer route-key")]
        assert probes
        assert len(waits) == 1
        assert waits[0][0] >= 1
        assert waits[0][1] == 401
        assert waits[0][2].startswith("http://127.0.0.1:")
        assert "route-key" not in waits[0][2]


@pytest.mark.host_sandbox
@pytest.mark.parametrize("caller", ["ask", "ask_tools"])
def test_outer_future_allows_held_queue_response(monkeypatch, caller):
    import kernel_measurement

    monkeypatch.setenv("REACT_REQUEST_TIMEOUT_S", "0.2")
    monkeypatch.setenv("STATEFRAME_PER_REQUEST_TIMEOUT_S", "0.3")
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", "0.6")
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "3")
    # Compress the historical 120-second cushion to exercise the same boundary quickly.
    monkeypatch.setattr(kernel_measurement, "request_timeout_budget_seconds", lambda: 0.4)
    with provider(post_delay=1.2) as (endpoint, posts, probes):
        monkeypatch.setattr(react_repair, "API", endpoint)

        class Capture:
            journal = None

            def send_with_frozen_timeout_retry(self, body, destination, key):
                started = time.monotonic()
                response = kernel_bounded_http.request_bytes(
                    json.dumps(body).encode(), destination, key,
                    deadline=started + 0.3, run_deadline=started + 3,
                    socket_timeout=0.3)
                return json.loads(response)["choices"][0]

        token = kernel_measurement.ACTIVE_REQUEST.set(Capture())
        try:
            messages = [{"role": "user", "content": "go"}]
            if caller == "ask":
                assert react_repair.ask(messages) == "done"
            else:
                assert react_repair.ask_tools(messages, tools=[]) == ("done", [])
        finally:
            kernel_measurement.ACTIVE_REQUEST.reset(token)
        assert len(posts) == 1
        assert probes


def test_future_budget_keeps_legacy_floor_and_respects_route_deadline(monkeypatch):
    import kernel_measurement

    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "3")
    monkeypatch.setattr(kernel_measurement, "request_timeout_budget_seconds", lambda: 0.4)
    assert kernel_measurement.request_queue_budget_seconds(0.3) == 3
    assert react_repair._transport_future_timeout(SimpleNamespace(), 0.3, True) >= 3
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "0.1")
    assert kernel_measurement.request_queue_budget_seconds(0.3) == 0.6
    assert react_repair._transport_future_timeout(SimpleNamespace(), 0.3, True) >= 0.6
    monkeypatch.setattr(react_repair._frame_routes, "ROUTES_ACTIVE", True)
    routed = SimpleNamespace(route_deadline=time.monotonic() + 0.2)
    assert 0 < react_repair._transport_future_timeout(routed, 0.3, True) <= 0.2


@pytest.mark.host_sandbox
def test_dead_probe_does_not_extend_request_cap(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", "4")
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "5")
    with provider(post_delay=2.2) as (endpoint, posts, _probes):
        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            dead_port = unused.getsockname()[1]
        monkeypatch.setattr(kernel_bounded_http, "_provider_probe_origin",
                            lambda _endpoint: ("http", "127.0.0.1", dead_port))
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            kernel_bounded_http.request_bytes(
                WIRE, endpoint, "", deadline=started + 1,
                run_deadline=started + 6, socket_timeout=1)
        assert time.monotonic() - started < 2
        assert len(posts) <= 1


@pytest.mark.host_sandbox
def test_hosted_probe_answer_cannot_extend_past_queue_cap(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", "0.5")
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "2")
    waits = []
    with provider(slots_status=404, post_delay=4) as (endpoint, posts, probes):
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            kernel_bounded_http.request_bytes(
                WIRE, endpoint, "", deadline=started + 0.5,
                run_deadline=started + 6, socket_timeout=0.5,
                record_queue_wait=lambda *values: waits.append(values))
        elapsed = time.monotonic() - started
        assert 1.8 <= elapsed < 3
        assert len(posts) == 1
        assert probes
        assert len(waits) == 1
        assert waits[0][0] >= 1.4
        assert waits[0][1] == 404


@pytest.mark.host_sandbox
def test_streaming_response_still_obeys_original_request_cap(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", "0.5")
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "5")
    with provider(post_delay=2.2, stream_stall=True) as (endpoint, posts, _probes):
        started = time.monotonic()
        with pytest.raises((TimeoutError, kernel_bounded_http.BoundedTransportError)):
            kernel_bounded_http.request_bytes(
                WIRE, endpoint, "", deadline=started + 1,
                run_deadline=started + 6, socket_timeout=1)
        assert time.monotonic() - started < 2
        assert len(posts) == 1


@pytest.mark.host_sandbox
def test_partial_response_headers_cannot_extend_original_cap(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", "0.5")
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "5")
    with provider(post_delay=2.4, trickle_headers=True) as (endpoint, posts, _probes):
        started = time.monotonic()
        with pytest.raises((TimeoutError, kernel_bounded_http.BoundedTransportError)):
            kernel_bounded_http.request_bytes(
                WIRE, endpoint, "", deadline=started + 1,
                run_deadline=started + 6, socket_timeout=1)
        assert time.monotonic() - started < 1.8
        assert len(posts) == 1


@pytest.mark.host_sandbox
def test_run_deadline_bounds_healthy_queue_wait(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", "0.5")
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "20")
    # The request deadline leaves room for child startup and the POST on a loaded host (the
    # body must be on the wire before it to earn queue wait); the reply is held past the run
    # deadline, which is what must end the wait.
    with provider(post_delay=10) as (endpoint, posts, probes):
        started = time.monotonic()
        run_deadline = started + 4
        with pytest.raises(TimeoutError):
            kernel_bounded_http.request_bytes(
                WIRE, endpoint, "", deadline=started + 3,
                run_deadline=run_deadline, socket_timeout=0.5)
        assert time.monotonic() - run_deadline < 0.5
        assert len(posts) == 1
        assert probes


@pytest.mark.host_sandbox
def test_queue_cap_bounds_even_a_longer_request_timeout(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", "0.5")
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "1.2")
    with provider(post_delay=4) as (endpoint, posts, probes):
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            kernel_bounded_http.request_bytes(
                WIRE, endpoint, "", deadline=started + 3,
                run_deadline=started + 6, socket_timeout=3)
        assert time.monotonic() - started < 1.8
        assert len(posts) == 1
        assert probes


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad"])
def test_queue_wait_max_requires_finite_positive_value(monkeypatch, value):
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", value)
    with pytest.raises(ValueError, match="STATEFRAME_QUEUE_WAIT_MAX_S"):
        kernel_bounded_http.queue_wait_max_seconds()


def test_queue_wait_warning_is_printed_once(tmp_path):
    events = tmp_path / "worker" / "measurement-runs" / "one" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text("\n".join(json.dumps(event) for event in (
        {"kind": "provider_queue_wait", "seconds_waited": 1.25},
        {"kind": "provider_queue_wait", "seconds_waited": 2.5},
    )) + "\n")
    lines = []
    runner._print_queue_wait_warning(tmp_path, lines.append)
    assert lines == ["warning: the model server queued requests for up to 2.5 s; "
                     "consider fewer parallel runs"]


@pytest.mark.parametrize("value", ["0", "nan", "inf", "bad"])
def test_runner_rejects_invalid_queue_wait_override(monkeypatch, value):
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", value)
    with pytest.raises(runner.RunError, match="STATEFRAME_QUEUE_WAIT_MAX_S"):
        runner.resolve_opt_in_switches()
