"""RED coverage for bounded retries of temporary capability-probe failures."""
from __future__ import annotations

import json
import math
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from _fault_contract import isolated_reason_log
import kernel_provider_dialect as dialect


PROPS = {"default_generation_settings": {"n_ctx": 4096}}
SLOTS = [{"id": 0, "n_ctx": 4096, "is_processing": False}]
_RECOVERY_AFTER_S = 20.0

_Response = tuple[int, object]
_Action = _Response | str | Callable[[], _Response | str]


@dataclass(frozen=True)
class _Request:
    requestline: str
    path: str
    headers: tuple[tuple[str, str], ...]
    body_length: int


class _Script:
    def __init__(self, plans: dict[str, list[_Action]]) -> None:
        self._plans = {path: list(actions) for path, actions in plans.items()}
        self._lock = threading.Lock()
        self.requests: list[_Request] = []
        self.release_hangs = threading.Event()

    def record(self, request: _Request) -> None:
        with self._lock:
            self.requests.append(request)

    def next_action(self, path: str) -> _Action:
        with self._lock:
            actions = self._plans.get(path)
            if not actions:
                return 404, None
            return actions.pop(0) if len(actions) > 1 else actions[0]


@contextmanager
def _loopback(plans: dict[str, list[_Action]]) -> Iterator[tuple[str, _Script]]:
    script = _Script(plans)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            raw_length = self.headers.get("Content-Length", "0")
            try:
                body_length = int(raw_length)
            except ValueError:
                body_length = -1
            script.record(_Request(
                requestline=self.requestline,
                path=self.path,
                headers=tuple(self.headers.raw_items()),
                body_length=body_length,
            ))
            action = script.next_action(self.path)
            if callable(action):
                action = action()
            if action == "drop":
                self.close_connection = True
                self.connection.close()
                return
            if action == "hang":
                script.release_hangs.wait()
                return
            status, payload = action
            response_body = json.dumps(payload).encode("utf-8") if status == 200 else b""
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", script
    finally:
        script.release_hangs.set()
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)


class _PauseClock:
    def __init__(self) -> None:
        self.delays: list[float] = []
        self._elapsed = 0.0
        self._lock = threading.Lock()
        self.before_pause: Callable[[], None] | None = None

    def sleep(self, delay: float) -> None:
        if self.before_pause is not None:
            self.before_pause()
        with self._lock:
            self.delays.append(delay)
            self._elapsed += delay

    @property
    def elapsed(self) -> float:
        with self._lock:
            return self._elapsed


@pytest.fixture
def pause_clock(monkeypatch: pytest.MonkeyPatch) -> _PauseClock:
    clock = _PauseClock()
    global_sleep = dialect.time.sleep
    monkeypatch.setattr(dialect, "_sleep", clock.sleep)
    assert dialect.time.sleep is global_sleep
    return clock


def _safe_request_log(script: _Script) -> list[tuple[str, tuple[tuple[str, str], ...], int]]:
    normalized: list[tuple[str, tuple[tuple[str, str], ...], int]] = []
    for request in script.requests:
        headers = tuple(
            (name, "<loopback>" if name.casefold() == "host" else value)
            for name, value in request.headers
        )
        normalized.append((request.requestline, headers, request.body_length))
    return normalized


def _expected_request_log(paths: list[str]) -> list[tuple[str, tuple[tuple[str, str], ...], int]]:
    headers = (
        ("Host", "<loopback>"),
        ("Accept-Encoding", "identity"),
        ("Accept", "application/json"),
    )
    return [(f"GET {path} HTTP/1.1", headers, 0) for path in paths]


def _paths(script: _Script) -> list[str]:
    return [request.path for request in script.requests]


def _track_probe_attempts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    real_probe = dialect.bounded_json_probe
    attempts: list[str] = []

    def tracked_probe(
        endpoint: str,
        key: str,
        path: str,
        *,
        deadline: float | None = None,
    ) -> tuple[int, object] | None:
        attempts.append(path)
        if deadline is None:
            return real_probe(endpoint, key, path)
        return real_probe(endpoint, key, path, deadline=deadline)

    monkeypatch.setattr(dialect, "bounded_json_probe", tracked_probe)
    return attempts


@pytest.mark.host_sandbox
def test_service_fault_recovers_after_twenty_fake_seconds_with_one_wait_notice(
    pause_clock: _PauseClock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def loading_response() -> _Response:
        return (200, PROPS) if pause_clock.elapsed >= _RECOVERY_AFTER_S else (503, None)

    with _loopback({"/props": [loading_response]}) as (endpoint, script):
        endpoint = endpoint.replace("http://", "http://probe-user:probe-pass@")
        capabilities = dialect.capabilities_for(
            endpoint, "", cache={}, require_slots=False, environ={},
        )

    continuation = capabilities.continuation
    assert continuation is True
    assert capabilities.probe_degradation is None
    assert pause_clock.elapsed >= _RECOVERY_AFTER_S
    assert pause_clock.delays == list(dialect.PROBE_RETRY_DELAYS[:5])
    assert _paths(script) == ["/props"] * (len(pause_clock.delays) + 1)
    assert _safe_request_log(script) == _expected_request_log(["/props"] * 6)
    waiting_lines = [
        line for line in capsys.readouterr().err.splitlines() if line.startswith("INFO: /")
    ]
    assert waiting_lines == ["INFO: /props probe received HTTP 503; waiting 8s before retry"]
    assert "probe-user:probe-pass" not in waiting_lines[0]


@pytest.mark.host_sandbox
@pytest.mark.parametrize("status", [500, 501, 502, 504])
def test_other_server_faults_match_the_pinned_base_and_keep_the_short_schedule(
    pause_clock: _PauseClock,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    attempts = _track_probe_attempts(monkeypatch)
    paths = ["/props"] * 4 + ["/slots"] * 4
    with _loopback({"/props": [(status, None)], "/slots": [(status, None)]}) as (endpoint, script):
        capabilities = dialect.capabilities_for(endpoint, "", cache={}, environ={})

    # Stable fallback fields match base 6063b3c49949ddacc61976bb262edfed5323bd52.
    degradation = capabilities.probe_degradation
    assert degradation is not None
    assert capabilities.to_record() == {
        "dialect": dialect.OPENAI_STRICT,
        "slots": False,
        "slot_ids": [],
        "reason": "openai-strict because /props was not llama.cpp and /slots was unproven",
        "props_status": status,
        "slots_status": status,
        "probe_degradation": {
            "attempts": 4,
            "last_error_type": "HTTPServerError",
            "elapsed_ms": degradation.elapsed_ms,
        },
    }
    assert attempts == paths
    assert pause_clock.delays == [0.25, 0.5, 1.0] * 2
    assert _safe_request_log(script) == _expected_request_log(paths)
    assert not [line for line in capsys.readouterr().err.splitlines() if line.startswith("INFO: /")]


@pytest.mark.host_sandbox
def test_only_the_unsettled_path_retries_and_is_not_cached_early(
    pause_clock: _PauseClock,
) -> None:
    cache: dict[str, dialect.ProviderCapabilities] = {}

    def loading_slots() -> _Response:
        return (200, SLOTS) if pause_clock.elapsed >= _RECOVERY_AFTER_S else (503, None)

    with _loopback({"/props": [(404, None)], "/slots": [loading_slots]}) as (endpoint, script):
        origin = dialect.origin_of(endpoint)

        def cache_stays_unsettled() -> None:
            assert origin not in cache

        pause_clock.before_pause = cache_stays_unsettled
        capabilities = dialect.capabilities_for(
            endpoint, "", cache=cache, environ={},
        )

    continuation = capabilities.continuation
    assert continuation is True
    assert capabilities.slots is True
    assert _paths(script)[0] == "/props"
    assert _paths(script).count("/props") == 1
    assert _paths(script).count("/slots") == len(pause_clock.delays) + 1
    assert cache[origin] == capabilities


@pytest.mark.host_sandbox
@pytest.mark.parametrize("status", [503, 429])
def test_exhaustion_keeps_the_strict_fallback_and_emits_one_warning(
    pause_clock: _PauseClock,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    status: int,
) -> None:
    cache: dict[str, dialect.ProviderCapabilities] = {}
    with _loopback({"/props": [(status, None)], "/slots": [(status, None)]}) as (endpoint, script):
        endpoint = endpoint.replace("http://", "http://probe-user:probe-pass@")
        with isolated_reason_log(monkeypatch, tmp_path) as harness:
            capabilities = dialect.capabilities_for(
                endpoint, "", cache=cache, environ={},
            )
            event = dialect.probe_degradation_event(capabilities, dialect.origin_of(endpoint))
            rows = harness.rows()
        stderr = capsys.readouterr().err
        warnings = [
            line for line in stderr.splitlines()
            if line.startswith("WARNING: /props probe")
        ]
        waiting_lines = [line for line in stderr.splitlines() if line.startswith("INFO: /")]

    schedule = tuple(dialect.PROBE_RETRY_DELAYS)
    assert capabilities.dialect == dialect.OPENAI_STRICT
    assert capabilities.reason.startswith("openai-strict because /props was ")
    assert capabilities.props_status == status
    assert capabilities.slots is False
    # With slots required, both paths settle and the strict result is cached, as before.
    assert len(cache) == 1
    assert event is not None
    assert event["attempts"] == len(schedule) + 1
    assert capabilities.slots_status == status
    assert len(pause_clock.delays) == len(schedule) * 2
    assert pause_clock.delays == [*schedule, *schedule]
    assert all(math.isfinite(delay) and delay > 0 for delay in pause_clock.delays)
    assert _paths(script) == ["/props"] * (len(schedule) + 1) + ["/slots"] * (len(schedule) + 1)
    assert len(warnings) == 1
    assert sum(pause_clock.delays) == pytest.approx(180.0, abs=10.0)
    paths = ["/props"] * (len(schedule) + 1) + ["/slots"] * (len(schedule) + 1)
    assert _safe_request_log(script) == _expected_request_log(paths)
    assert len(waiting_lines) == 1
    assert f"HTTP {status}" in waiting_lines[0] and "waiting" in waiting_lines[0]
    assert "probe-user:probe-pass" not in waiting_lines[0]
    assert len(rows) == 1
    assert rows[0]["fields"]["cause_code"] == "ProbePropsUnreachable:props_probe_unreachable"
    expected_error = "HTTPRateLimit" if status == 429 else "HTTPServerError"
    assert capabilities.probe_degradation.last_error_type == expected_error
    assert event["last_error_type"] == expected_error
    assert expected_error in warnings[0]
    if status == 429:
        assert rows[0]["fields"]["handler_id"] == dialect.PROBE_FAULT_HANDLER
        assert rows[0]["fields"]["exception_message"].startswith("HTTPRateLimit after 8 attempt(s)")


@pytest.mark.host_sandbox
def test_connection_refusal_keeps_the_base_attempt_count_and_pause_budget(
    pause_clock: _PauseClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = _track_probe_attempts(monkeypatch)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}/v1"

    capabilities = dialect.capabilities_for(
        endpoint, "", cache={}, require_slots=False, environ={},
    )

    assert capabilities.dialect == dialect.OPENAI_STRICT
    assert capabilities.probe_degradation is not None
    assert capabilities.probe_degradation.last_error_type == "ConnectionRefusedError"
    assert capabilities.probe_degradation.attempts == 4
    assert attempts == ["/props"] * 4
    assert pause_clock.delays == [0.25, 0.5, 1.0]
    assert sum(pause_clock.delays) == pytest.approx(1.75)


@pytest.mark.host_sandbox
def test_connection_reset_keeps_the_base_attempt_count_and_request_log(
    pause_clock: _PauseClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = _track_probe_attempts(monkeypatch)
    with _loopback({"/props": ["drop"]}) as (endpoint, script):
        capabilities = dialect.capabilities_for(
            endpoint, "", cache={}, require_slots=False, environ={},
        )

    assert capabilities.probe_degradation is not None
    assert capabilities.probe_degradation.attempts == 4
    assert attempts == ["/props"] * 4
    assert pause_clock.delays == [0.25, 0.5, 1.0]
    assert _safe_request_log(script) == _expected_request_log(["/props"] * 4)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("status", [503, 429])
def test_first_server_fault_gets_its_full_schedule_after_a_dropped_connection(
    pause_clock: _PauseClock,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    attempts = _track_probe_attempts(monkeypatch)
    with _loopback({"/props": ["drop", (status, None)]}) as (endpoint, script):
        capabilities = dialect.capabilities_for(
            endpoint, "", cache={}, require_slots=False, environ={},
        )

    assert capabilities.props_status == status
    assert capabilities.probe_degradation is not None
    assert pause_clock.delays == [0.25, *dialect.PROBE_RETRY_DELAYS]
    assert attempts == ["/props"] * (len(pause_clock.delays) + 1)
    assert len(script.requests) == len(pause_clock.delays) + 1


@pytest.mark.host_sandbox
@pytest.mark.parametrize(
    ("status", "payload"),
    [(401, None), (403, None), (404, None), (200, {"models": []})],
)
def test_definite_answers_keep_the_base_probe_bytes_and_attempts(
    pause_clock: _PauseClock,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    payload: object,
) -> None:
    attempts = _track_probe_attempts(monkeypatch)
    with _loopback({"/props": [(status, payload)], "/slots": [(404, None)]}) as (endpoint, script):
        capabilities = dialect.capabilities_for(
            endpoint, "", cache={}, environ={},
        )

    assert capabilities.dialect == dialect.OPENAI_STRICT
    assert capabilities.probe_degradation is None
    assert pause_clock.delays == []
    assert attempts == ["/props", "/slots"]
    assert _safe_request_log(script) == _expected_request_log(["/props", "/slots"])


@pytest.mark.host_sandbox
def test_flapping_service_and_rate_limit_answers_retry_until_definite(
    pause_clock: _PauseClock,
) -> None:
    plan: list[_Action] = [
        (503, None),
        (429, None),
        (503, None),
        (200, PROPS),
    ]
    with _loopback({"/props": plan}) as (endpoint, script):
        capabilities = dialect.capabilities_for(
            endpoint, "", cache={}, require_slots=False, environ={},
        )

    continuation = capabilities.continuation
    assert continuation is True
    assert _paths(script) == ["/props", "/props", "/props", "/props"]
    assert len(pause_clock.delays) == 3


@pytest.mark.host_sandbox
def test_listening_socket_timeout_keeps_the_short_base_attempt_budget(
    pause_clock: _PauseClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_probe = dialect.bounded_json_probe
    attempts: list[str] = []

    def short_probe_timeout(
        endpoint: str,
        key: str,
        path: str,
        *,
        deadline: float | None = None,
    ) -> tuple[int, object] | None:
        attempts.append(path)
        return real_probe(endpoint, key, path, timeout=0.5, deadline=deadline)

    monkeypatch.setattr(dialect, "bounded_json_probe", short_probe_timeout)

    def timeout_until_recovered() -> _Response | str:
        return (200, PROPS) if pause_clock.elapsed >= _RECOVERY_AFTER_S else "hang"

    with _loopback({"/props": [timeout_until_recovered]}) as (endpoint, script):
        capabilities = dialect.capabilities_for(
            endpoint, "", cache={}, require_slots=False, environ={},
        )

    assert capabilities.dialect == dialect.OPENAI_STRICT
    assert capabilities.probe_degradation is not None
    assert capabilities.probe_degradation.attempts == 4
    assert attempts == ["/props"] * 4
    assert pause_clock.delays == [0.25, 0.5, 1.0]
    assert pause_clock.elapsed < _RECOVERY_AFTER_S
    assert _safe_request_log(script) == _expected_request_log(["/props"] * 4)


@pytest.mark.host_sandbox
def test_two_origins_keep_independent_capability_results() -> None:
    cache: dict[str, dialect.ProviderCapabilities] = {}
    with (
        _loopback({"/props": [(404, None)], "/slots": [(404, None)]})
        as (strict_endpoint, strict_script),
        _loopback({"/props": [(200, PROPS)], "/slots": [(200, SLOTS)]})
        as (native_endpoint, native_script),
    ):
        strict = dialect.capabilities_for(strict_endpoint, "", cache=cache, environ={})
        native = dialect.capabilities_for(native_endpoint, "", cache=cache, environ={})
        assert dialect.capabilities_for(strict_endpoint, "", cache=cache, environ={}) == strict
        assert dialect.capabilities_for(native_endpoint, "", cache=cache, environ={}) == native

    assert strict.dialect == dialect.OPENAI_STRICT
    native_continuation = native.continuation
    assert native_continuation is True
    assert _paths(strict_script) == ["/props", "/slots"]
    assert _paths(native_script) == ["/props", "/slots"]


@pytest.mark.host_sandbox
def test_healthy_probe_request_log_keeps_base_bytes_and_order(
    pause_clock: _PauseClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = _track_probe_attempts(monkeypatch)
    cache: dict[str, dialect.ProviderCapabilities] = {}
    with _loopback({"/props": [(200, PROPS)], "/slots": [(200, SLOTS)]}) as (endpoint, script):
        first = dialect.capabilities_for(endpoint, "", cache=cache, environ={})
        second = dialect.capabilities_for(endpoint, "", cache=cache, environ={})

    first_continuation = first.continuation
    assert first_continuation is True
    assert second == first
    assert pause_clock.delays == []
    assert attempts == ["/props", "/slots"]
    assert _safe_request_log(script) == _expected_request_log(["/props", "/slots"])
