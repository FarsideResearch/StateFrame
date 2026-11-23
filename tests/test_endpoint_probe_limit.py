"""The models probe reports a bounded response overflow instead of a JSON parse failure."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stateframe.endpoint import MAX_JSON_BYTES, Endpoint, EndpointError


LIMIT = 128
LIMIT_ENV = "STATEFRAME_ENDPOINT_MAX_JSON_BYTES"


def _payload(size: int) -> bytes:
    prefix, suffix = b'{"data":[{"id":"served"}],"padding":"', b'"}'
    assert size >= len(prefix) + len(suffix)
    return prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix


class _Body:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read1(self, _size):
        return self.chunks.pop(0) if self.chunks else b""


class _Opener:
    def __init__(self, *outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def open(self, request, timeout):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@contextmanager
def _server(body: bytes, *, mode: str = "length", status: int = 200):
    seen = []
    wire_body = body

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            seen.append(self.path)
            self.send_response(status)
            if mode == "chunked":
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(len(wire_body)))
            self.end_headers()
            try:
                if mode == "chunked":
                    for start in range(0, len(wire_body), 19):
                        chunk = wire_body[start:start + 19]
                        self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self.wfile.write(wire_body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", seen
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("mode", ["length", "chunked"])
def test_body_exactly_at_limit_is_accepted_without_an_extra_request(monkeypatch, mode):
    monkeypatch.setenv(LIMIT_ENV, str(LIMIT))
    with _server(_payload(LIMIT), mode=mode) as (url, seen):
        result = Endpoint(url, "served", timeout_s=2).probe()
    assert result["model_found"] is True
    assert seen == ["/v1/models"]


def test_operator_can_raise_the_default_limit(monkeypatch):
    monkeypatch.setenv(LIMIT_ENV, str(MAX_JSON_BYTES + 1))
    with _server(_payload(MAX_JSON_BYTES + 1)) as (url, seen):
        result = Endpoint(url, "served", timeout_s=5).probe()
    assert result["model_found"] is True
    assert seen == ["/v1/models"]


@pytest.mark.parametrize("mode", ["length", "chunked"])
def test_oversized_body_is_typed_and_names_only_its_path_and_limit(monkeypatch, mode):
    monkeypatch.setenv(LIMIT_ENV, str(LIMIT))
    with _server(_payload(LIMIT + 1), mode=mode) as (url, seen):
        with pytest.raises(EndpointError, match=r"GET /v1/models.*128-byte limit") as caught:
            Endpoint(url, "served", timeout_s=2).probe()
    assert type(caught.value).__name__ == "EndpointProbeLimitError"
    assert "127.0.0.1" not in str(caught.value)
    assert "padding" not in str(caught.value)
    assert seen == ["/v1/models"]


def test_oversized_optional_slots_probe_returns_none_with_one_plain_diagnostic(monkeypatch, capsys):
    monkeypatch.setenv(LIMIT_ENV, str(LIMIT))
    with _server(_payload(LIMIT + 1)) as (url, seen):
        result = Endpoint(url, "served", timeout_s=2).get_slots()
    assert result is None
    assert seen == ["/slots"]
    diagnostic = capsys.readouterr().err
    assert diagnostic == (
        "stateframe: warning: optional endpoint GET /slots response exceeds "
        "128-byte limit; ignoring probe\n")
    assert "padding" not in diagnostic


@pytest.mark.parametrize("value", ["0", "-2", "many", "2.5"])
def test_invalid_limit_is_explicit_and_refuses_before_the_request(monkeypatch, value):
    monkeypatch.setenv(LIMIT_ENV, value)
    with _server(_payload(LIMIT)) as (url, seen):
        with pytest.raises(EndpointError, match=rf"{LIMIT_ENV}.*positive integer"):
            Endpoint(url, "served", timeout_s=2).probe()
    assert seen == []


def test_invalid_limit_is_not_hidden_by_the_optional_slots_probe(monkeypatch):
    monkeypatch.setenv(LIMIT_ENV, "0")
    with _server(_payload(LIMIT)) as (url, seen):
        with pytest.raises(EndpointError, match=rf"{LIMIT_ENV}.*positive integer"):
            Endpoint(url, "served", timeout_s=2).get_slots()
    assert seen == []


def test_invalid_limit_is_reported_once_before_startup_requests(monkeypatch, capsys, tmp_path):
    from stateframe import cli

    monkeypatch.setenv(LIMIT_ENV, "0")
    with _server(_payload(LIMIT)) as (url, seen):
        code = cli.main(["run", "--repo", str(tmp_path), "--task", "probe endpoint",
                         "--endpoint", url, "--model", "served"])
    assert code == 2
    assert capsys.readouterr().err == (
        f"stateframe: error: {LIMIT_ENV} must be a positive integer\n")
    assert seen == []


def test_limit_switch_is_host_only_and_not_reported_as_unknown(monkeypatch):
    from stateframe import runner
    from stateframe.engine.kernel_release_switch_registry import release_stateframe_switches

    assert LIMIT_ENV in runner.SHELL_SETTINGS
    assert LIMIT_ENV not in runner.ignored_shell_settings({LIMIT_ENV: str(LIMIT)})
    switches, _digest = release_stateframe_switches()
    assert LIMIT_ENV not in switches


def test_slow_body_still_hits_the_probe_deadline_and_retries_once(monkeypatch):
    from stateframe import endpoint as endpoint_module

    monkeypatch.setenv(LIMIT_ENV, "8")
    clock = [0.0]
    monkeypatch.setattr(endpoint_module.time, "monotonic", lambda: clock[0])

    class Trickle(_Body):
        def read1(self, size):
            clock[0] += 0.1
            return super().read1(size)

    first, second = Trickle([b"["] * 100), Trickle([b"["] * 100)
    with pytest.raises(TimeoutError, match="probe deadline"):
        Endpoint("http://127.0.0.1:1/v1", "served", timeout_s=0.3)._read_json(
            _Opener(first, second), "http://127.0.0.1:1/v1/models")
    assert 96 <= len(first.chunks) <= 97 and 96 <= len(second.chunks) <= 97


def test_error_status_keeps_its_existing_message_and_is_not_retried(monkeypatch):
    monkeypatch.setenv(LIMIT_ENV, str(LIMIT))
    with _server(_payload(LIMIT + 1), status=503) as (url, seen):
        with pytest.raises(EndpointError, match="HTTP 503 on GET /models"):
            Endpoint(url, "served", timeout_s=2).probe()
    assert seen == ["/v1/models"]


def test_run_start_reports_the_models_probe_overflow(monkeypatch, capsys, tmp_path):
    from stateframe import cli

    def probe_then_return(options, printer):
        options.endpoint.probe()
        return {"exit_code": 0}

    monkeypatch.setattr(cli, "run", probe_then_return)
    monkeypatch.setattr(cli, "_print_summary", lambda _summary: None)
    with _server(_payload(MAX_JSON_BYTES + 1)) as (url, seen):
        code = cli.main(["run", "--repo", str(tmp_path), "--task", "probe endpoint",
                         "--endpoint", url, "--model", "served", "--check-cmd", "true"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err == (
        f"stateframe: error: endpoint GET /v1/models response exceeds {MAX_JSON_BYTES}-byte limit "
        f"({LIMIT_ENV})\n")
    assert seen == ["/v1/models"]
