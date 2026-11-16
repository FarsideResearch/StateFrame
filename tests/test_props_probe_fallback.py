"""A transient /props failure is retried, and a persistent one falls back loudly.

The stub answers each probe path from a script: "hang" holds the request until the
test releases it (so the client times out), a (status, body) pair is a real answer.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import kernel_provider_dialect as dialect
from kernel_provider_dialect import LLAMA_CPP, OPENAI_STRICT, capabilities_for, origin_of

LLAMA_PROPS = {
    "default_generation_settings": {"n_ctx": 65536},
    "model_path": "local-model.gguf",
}
LLAMA_SLOTS = [{"id": 0, "n_ctx": 65536, "is_processing": False}]
PLAIN_404 = (404, {"error": "not found"})
KEY = "test-key-not-a-secret"


class _Stub:
    def __init__(self, plans: dict[str, list[object]]) -> None:
        self._plans = {path: list(items) for path, items in plans.items()}
        self.seen: list[str] = []
        self.release = threading.Event()
        self._lock = threading.Lock()

    def next_action(self, path: str) -> object:
        with self._lock:
            self.seen.append(path)
            queue = self._plans.get(path)
            if not queue:
                return PLAIN_404
            # The last scripted action repeats, so "hang" can stay in force.
            return queue.pop(0) if len(queue) > 1 else queue[0]


@contextmanager
def _stub(plans: dict[str, list[object]]) -> Iterator[tuple[str, _Stub]]:
    stub = _Stub(plans)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            action = stub.next_action(self.path)
            if action == "hang":
                stub.release.wait(timeout=30)
                return  # the client has already given up on this probe
            status, body = action  # type: ignore[misc]
            payload = json.dumps(body).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", stub
    finally:
        stub.release.set()
        server.shutdown()
        server.server_close()


def _closed_port_endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


@pytest.mark.host_sandbox
def test_slow_props_retries_then_detects_llama():
    plans = {"/props": ["hang", "hang", (200, LLAMA_PROPS)], "/slots": [(200, LLAMA_SLOTS)]}
    with _stub(plans) as (endpoint, stub):
        caps = capabilities_for(
            endpoint, KEY, deadline=time.monotonic() + 30, cache={}, environ={},
        )
    assert caps.dialect == LLAMA_CPP, caps.reason
    # Tokenization needs the /props proof; slot discovery alone must not stand in for it.
    assert caps.reason == "proven by /props", caps.reason
    assert caps.props_status == 200
    assert stub.seen.count("/props") == 3
    assert caps.probe_degradation is None


@pytest.mark.host_sandbox
def test_props_unreachable_falls_back_loudly(capsys):
    plans = {"/props": ["hang"], "/slots": ["hang"]}
    with _stub(plans) as (endpoint, _stub_state):
        caps = capabilities_for(
            endpoint, KEY, deadline=time.monotonic() + 5.5, cache={}, environ={},
        )
        origin = origin_of(endpoint)
    assert caps.dialect == OPENAI_STRICT
    assert getattr(caps, "probe_degradation", None) is not None, "fallback was not marked degraded"
    event = dialect.probe_degradation_event(caps, origin)
    assert event is not None
    assert set(event) == {
        "code", "endpoint_origin", "attempts", "last_error_type", "elapsed_ms", "dialect", "tokenizer",
    }
    assert event["code"] == dialect.PROBE_DEGRADED_CODE == "props_probe_unreachable"
    assert event["endpoint_origin"] == origin
    assert event["attempts"] >= 2
    assert event["dialect"] == OPENAI_STRICT
    assert event["tokenizer"] is False
    assert type(event["elapsed_ms"]) is int and event["elapsed_ms"] > 0
    err = capsys.readouterr().err
    warnings = [line for line in err.splitlines() if line.startswith("WARNING: /props probe")]
    assert len(warnings) == 1, err
    assert origin in warnings[0]


@pytest.mark.host_sandbox
def test_definite_404_is_not_retried_and_stays_silent(capsys):
    plans = {"/props": [PLAIN_404], "/slots": [PLAIN_404]}
    with _stub(plans) as (endpoint, stub):
        caps = capabilities_for(
            endpoint, KEY, deadline=time.monotonic() + 10, cache={}, environ={},
        )
    assert caps.dialect == OPENAI_STRICT
    assert caps.reason == "openai-strict because /props was 404 and /slots was unproven"
    assert caps.props_status == 404
    assert stub.seen.count("/props") == 1
    assert getattr(caps, "probe_degradation", None) is None
    assert dialect.probe_degradation_event(caps, origin_of(endpoint)) is None
    assert capsys.readouterr().err == ""


@pytest.mark.host_sandbox
def test_retries_respect_deadline():
    plans = {"/props": ["hang"], "/slots": ["hang"]}
    with _stub(plans) as (endpoint, stub):
        started = time.monotonic()
        deadline = started + 1.2
        caps = capabilities_for(endpoint, KEY, deadline=deadline, cache={}, environ={})
        elapsed = time.monotonic() - started
    assert caps.dialect == OPENAI_STRICT
    assert stub.seen.count("/props") >= 1
    # The probe may not outlive the deadline, plus a small allowance for child spawn.
    assert elapsed < 1.2 + 0.8, elapsed
    assert time.monotonic() >= deadline - 0.05


@pytest.mark.host_sandbox
def test_event_has_no_error_message_text(capsys):
    endpoint = _closed_port_endpoint()
    caps = capabilities_for(
        endpoint, KEY, deadline=time.monotonic() + 15, cache={}, environ={},
    )
    event = dialect.probe_degradation_event(caps, origin_of(endpoint))
    assert event is not None
    last_error = event["last_error_type"]
    assert isinstance(last_error, str) and last_error.isidentifier()
    rendered = repr(event) + capsys.readouterr().err
    assert KEY not in rendered
    assert "/props" not in repr(event)
    assert "/v1" not in rendered
    assert "refused" not in rendered.lower()
    assert "127.0.0.1" in event["endpoint_origin"]
    for value in event.values():
        if isinstance(value, str):
            assert value.count("127.0.0.1") <= 1
