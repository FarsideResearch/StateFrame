"""The startup probe retries once when a model server stalls past its timeout."""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stateframe.endpoint import Endpoint, EndpointError


@contextmanager
def _server(stalls: int, *, stall_seconds: float = 1.0, status: int = 200):
    """A local server whose first ``stalls`` GETs hang past the client timeout before answering."""
    seen = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            with lock:  # the handler threads race; the count decides which request stalls
                seen.append(self.path)
                stall = len(seen) <= stalls
            if stall:
                time.sleep(stall_seconds)
            if self.path == "/v1/models":
                body = json.dumps({"data": [{"id": "served"}]}).encode()
            else:
                body = json.dumps([{"id": 0, "n_ctx": 65536}]).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client already gave up on this stalled attempt

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", seen
    finally:
        server.shutdown()
        server.server_close()


def test_probe_retries_once_after_a_stall():
    with _server(stalls=1) as (url, seen):
        result = Endpoint(url, "served", timeout_s=0.3).probe()
    assert result["model_found"] is True
    assert seen == ["/v1/models", "/v1/models"]


def test_probe_gives_up_after_one_retry():
    with _server(stalls=2) as (url, seen):
        with pytest.raises(EndpointError, match="unreachable.*TimeoutError"):
            Endpoint(url, "served", timeout_s=0.3).probe()
    assert seen == ["/v1/models", "/v1/models"]


def test_probe_does_not_retry_an_answer():
    with _server(stalls=0, status=500) as (url, seen):
        with pytest.raises(EndpointError, match="HTTP 500"):
            Endpoint(url, "served", timeout_s=0.3).probe()
    assert seen == ["/v1/models"]


class _Opener:
    """Stands in for urllib's opener: plays a fixed list of outcomes and records each request."""

    def __init__(self, *outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def open(self, request, timeout):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Body:
    def __init__(self, chunks, *, delay=0.0):
        self.chunks, self.delay = list(chunks), delay

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read1(self, _size):
        time.sleep(self.delay)
        return self.chunks.pop(0) if self.chunks else b""


@pytest.mark.parametrize("error", [
    urllib.error.URLError(ConnectionRefusedError(61, "refused")),
    urllib.error.URLError(socket.gaierror(8, "nodename nor servname provided")),
    urllib.error.HTTPError("http://127.0.0.1:1/v1/models", 503, "busy", {}, None),
])
def test_only_timeouts_are_retried(error):
    opener = _Opener(error, _Body([b"[]"]))
    with pytest.raises(type(error)):
        Endpoint("http://127.0.0.1:1/v1", "served", timeout_s=1)._read_json(opener, "http://127.0.0.1:1/v1/models")
    assert len(opener.requests) == 1


def test_wrapped_timeout_is_retried_and_reads_the_key_again(monkeypatch):
    # urllib wraps a connect timeout as URLError(reason=TimeoutError); the retry builds a new request,
    # so a key rotated between attempts is the one sent.
    monkeypatch.setenv("PROBE_KEY", "first")
    endpoint = Endpoint("http://127.0.0.1:1/v1", "served", api_key_env="PROBE_KEY", timeout_s=1)

    class RotatingOpener(_Opener):
        def open(self, request, timeout):
            if not self.requests:
                monkeypatch.setenv("PROBE_KEY", "second")
            return super().open(request, timeout)

    opener = RotatingOpener(urllib.error.URLError(TimeoutError("timed out")), _Body([b'{"data": []}']))
    assert endpoint._read_json(opener, "http://127.0.0.1:1/v1/models") == {"data": []}
    assert [request.get_header("Authorization") for request in opener.requests] == ["Bearer first",
                                                                                     "Bearer second"]


def test_a_trickling_response_cannot_outlive_its_deadline(monkeypatch):
    # Each chunk arrives within the socket timeout, but the attempt as a whole must still end. A fake
    # clock advances 0.1 s per chunk, so the outcome does not depend on how loaded the machine is.
    from stateframe import endpoint as endpoint_module

    clock = [0.0]
    monkeypatch.setattr(endpoint_module.time, "monotonic", lambda: clock[0])

    class Trickle(_Body):
        def read1(self, size):
            clock[0] += 0.1
            return super().read1(size)

    first, second = Trickle([b"["] * 1000), Trickle([b"["] * 1000)
    with pytest.raises(TimeoutError, match="probe deadline"):
        Endpoint("http://127.0.0.1:1/v1", "served", timeout_s=0.3)._read_json(
            _Opener(first, second), "http://127.0.0.1:1/v1/models")
    # Both attempts stopped after about timeout_s worth of chunks, far short of the full body.
    assert 1000 - len(first.chunks) <= 4 and 1000 - len(second.chunks) <= 4


def test_slot_discovery_retries_once_after_a_stall():
    with _server(stalls=1) as (url, seen):
        slots = Endpoint(url, "served", timeout_s=0.3).get_slots()
    assert slots == [{"id": 0, "n_ctx": 65536}]
    assert seen == ["/slots", "/slots"]
