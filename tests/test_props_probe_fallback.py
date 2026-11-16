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
    # The exception class name is allowed; its message text (errno, host, port) is not.
    assert "connection refused" not in rendered.lower()
    assert "Errno" not in rendered
    assert "127.0.0.1" in event["endpoint_origin"]
    for value in event.values():
        if isinstance(value, str):
            assert value.count("127.0.0.1") <= 1


def _degraded_caps(**overrides: object) -> dialect.ProviderCapabilities:
    fields: dict[str, object] = {
        "dialect": OPENAI_STRICT,
        "props_status": 503,
        "slots_status": 503,
        "probe_degradation": dialect.ProbeDegradation(
            attempts=4, last_error_type="HTTPServerError", elapsed_ms=1810),
    }
    fields.update(overrides)
    return dialect.ProviderCapabilities(**fields)  # type: ignore[arg-type]


class _FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def write(self, kind: str, **fields: object) -> None:
        self.events.append((kind, fields))


@pytest.fixture
def run_cache() -> Iterator[None]:
    """The engine's run-wide cache is module state; leave it empty afterwards."""
    try:
        yield
    finally:
        dialect.seed_run_capability_cache({})


@pytest.mark.host_sandbox
def test_5xx_props_is_retried_and_degradation_survives_host_seed(run_cache):
    plans = {"/props": [(503, {"error": "loading"})], "/slots": [(503, {"error": "loading"})]}
    with _stub(plans) as (endpoint, stub):
        origin = origin_of(endpoint)
        host = capabilities_for(endpoint, KEY, deadline=time.monotonic() + 20, cache={}, environ={})
        assert stub.seen.count("/props") == 4, stub.seen
        assert host.probe_degradation is not None
        assert host.probe_degradation.last_error_type == "HTTPServerError"
        record = host.to_record()
        assert record["probe_degradation"] == {
            "attempts": 4, "last_error_type": "HTTPServerError",
            "elapsed_ms": host.probe_degradation.elapsed_ms,
        }
        # The host ships its proof to the engine child, which must not re-probe.
        dialect.seed_run_capability_cache(json.loads(json.dumps({origin: record})))
        before = len(stub.seen)
        engine = capabilities_for(endpoint, KEY, cache=dialect.RUN_CAPABILITY_CACHE, environ={})
        assert len(stub.seen) == before
        assert engine == host
        assert engine.probe_degradation == host.probe_degradation


def test_seed_round_trip_keeps_degradation_and_drops_malformed(run_cache):
    origin = "http://127.0.0.1:1"
    record = _degraded_caps().to_record()
    dialect.seed_run_capability_cache({origin: record})
    assert dialect.RUN_CAPABILITY_CACHE[origin] == _degraded_caps()

    clean = dialect.ProviderCapabilities(
        dialect=OPENAI_STRICT, props_status=404, slots_status=404,
        reason="openai-strict because /props was 404 and /slots was unproven")
    dialect.seed_run_capability_cache({origin: clean.to_record()})
    assert dialect.RUN_CAPABILITY_CACHE[origin] == clean
    assert dialect.RUN_CAPABILITY_CACHE[origin].probe_degradation is None

    malformed = [
        {"attempts": 0, "last_error_type": "TimeoutError", "elapsed_ms": 5},
        {"attempts": True, "last_error_type": "TimeoutError", "elapsed_ms": 5},
        {"attempts": 2, "last_error_type": "Connection refused by host", "elapsed_ms": 5},
        {"attempts": 2, "last_error_type": "", "elapsed_ms": 5},
        {"attempts": 2, "last_error_type": "TimeoutError", "elapsed_ms": -1},
        {"attempts": 2, "last_error_type": "TimeoutError", "elapsed_ms": 1.5},
        {"attempts": 2, "last_error_type": "TimeoutError"},
        {"attempts": 2, "last_error_type": 7, "elapsed_ms": 5},
        "degraded",
        [4, "TimeoutError", 5],
    ]
    for value in malformed:
        dialect.seed_run_capability_cache({origin: {**record, "probe_degradation": value}})
        seeded = dialect.RUN_CAPABILITY_CACHE[origin]
        assert seeded.dialect == OPENAI_STRICT and seeded.props_status == 503, value
        assert seeded.probe_degradation is None, value

    # A record from a release that never carried the field still seeds.
    dialect.seed_run_capability_cache({origin: {k: v for k, v in record.items() if k != "probe_degradation"}})
    assert dialect.RUN_CAPABILITY_CACHE[origin].probe_degradation is None
    assert dialect.RUN_CAPABILITY_CACHE[origin].props_status == 503


def test_seeded_degradation_journals_exactly_one_event_on_request_path(run_cache, capsys):
    from kernel_plain_host import _PlainHTTPProvider

    origin = "http://127.0.0.1:1"
    dialect.seed_run_capability_cache({origin: _degraded_caps().to_record()})

    # The engine request path: one dialect journal entry per origin per run.
    events = []
    for _request in range(3):
        caps = capabilities_for(origin + "/v1/chat/completions", KEY, cache=dialect.RUN_CAPABILITY_CACHE)
        if dialect.mark_dialect_journaled(origin):
            degraded = dialect.probe_degradation_event(caps, origin)
            if degraded is not None:
                events.append(degraded)
    assert len(events) == 1, events
    assert events[0]["code"] == dialect.PROBE_DEGRADED_CODE
    assert events[0]["last_error_type"] == "HTTPServerError"
    assert events[0]["attempts"] == 4

    # The plain host loop journals through the same once-per-origin guard.
    journal = _FakeJournal()
    provider = _PlainHTTPProvider(origin + "/v1/chat/completions", KEY, journal=journal, environ={})
    provider.capability_cache = dict(dialect.RUN_CAPABILITY_CACHE)
    for _request in range(2):
        provider.prepare_body({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    degraded_events = [fields for kind, fields in journal.events if kind == dialect.PROBE_DEGRADED_EVENT]
    assert len(degraded_events) == 1, journal.events
    assert degraded_events[0]["endpoint_origin"] == origin
    assert degraded_events[0]["dispatch"] is False
    assert sum(1 for kind, _fields in journal.events if kind == "provider_dialect") == 1
    err = capsys.readouterr().err
    assert len([line for line in err.splitlines() if line.startswith("WARNING: /props probe")]) == 2


def test_runner_preflight_warns_once_per_degraded_origin(capsys):
    from stateframe import runner

    degraded_origin = "http://127.0.0.1:1"
    warnings = runner._provider_preflight_warnings({
        degraded_origin: _degraded_caps(),
        "http://127.0.0.1:2": dialect.ProviderCapabilities(
            dialect=OPENAI_STRICT, props_status=404, slots_status=404),
        "http://127.0.0.1:3": dialect.ProviderCapabilities(
            dialect=LLAMA_CPP, slots=True, props_status=200, slots_status=200),
    })
    assert len(warnings) == 1, warnings
    assert warnings[0].startswith("WARNING: /props probe to " + degraded_origin)
    assert "4 attempt(s) (HTTPServerError)" in warnings[0]
    assert "openai-strict" in warnings[0]
    assert capsys.readouterr().err == ""
    assert runner._provider_preflight_warnings({}) == []


@pytest.mark.host_sandbox
def test_last_error_type_is_the_probe_exception_class():
    endpoint = _closed_port_endpoint()
    caps = capabilities_for(endpoint, KEY, deadline=time.monotonic() + 15, cache={}, environ={})
    assert caps.probe_degradation is not None
    assert caps.probe_degradation.last_error_type == "ConnectionRefusedError"
