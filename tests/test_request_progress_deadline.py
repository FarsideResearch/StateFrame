"""A model request is stopped only by no progress or by a cap the user set.

Elapsed time alone must never end a streaming response. A deadline of None means no per-request
elapsed cap. Socket idle bounds and provider liveness still apply, and a run deadline still wins.
"""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
import time

import pytest

from _timeouts import scaled
from stateframe.engine import kernel_bounded_http


WIRE = b'{"model":"scripted","messages":[{"role":"user","content":"progress"}]}'
REPLY = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()
LIVENESS_S = 10.0


@contextmanager
def server(*, first_byte_delay: float = 0.0, chunks: int = 0, chunk_interval: float = 0.0,
           silent_after_first: float = 0.0):
    """Serve REPLY as Content-Length JSON.

    chunks/chunk_interval: stream the body in `chunks` pieces, one every `chunk_interval` seconds.
    silent_after_first: send the first piece, then stay silent for this many seconds.
    first_byte_delay: sleep before writing anything.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            time.sleep(first_byte_delay)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(REPLY)))
            self.end_headers()
            try:
                if silent_after_first:
                    piece = max(1, len(REPLY) // 4)
                    self.wfile.write(REPLY[:piece])
                    self.wfile.flush()
                    time.sleep(silent_after_first)
                    self.wfile.write(REPLY[piece:])
                elif chunks:
                    step = max(1, -(-len(REPLY) // chunks))
                    for start in range(0, len(REPLY), step):
                        self.wfile.write(REPLY[start:start + step])
                        self.wfile.flush()
                        time.sleep(chunk_interval)
                else:
                    self.wfile.write(REPLY)
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=scaled(5))


@pytest.fixture(autouse=True)
def _liveness_env(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", str(LIVENESS_S))
    monkeypatch.delenv("STATEFRAME_QUEUE_WAIT_MAX_S", raising=False)


@pytest.mark.host_sandbox
def test_no_deadline_streaming_body_past_three_seconds_completes():
    # a: elapsed time alone never stops a request that keeps delivering bytes.
    with server(chunks=15, chunk_interval=0.2) as endpoint:
        reply = kernel_bounded_http.request_bytes(
            WIRE, endpoint, "", deadline=None, socket_timeout=1.0)
    assert reply == REPLY


@pytest.mark.host_sandbox
def test_finite_deadline_renews_on_progress_during_body_read():
    # b: the request deadline bounds the wait for progress, and each chunk renews it (L5 fix).
    with server(chunks=15, chunk_interval=0.2) as endpoint:
        reply = kernel_bounded_http.request_bytes(
            WIRE, endpoint, "", deadline=time.monotonic() + 1.0, socket_timeout=1.0)
    assert reply == REPLY


@pytest.mark.host_sandbox
def test_no_deadline_stalled_body_raises_stall_not_elapsed():
    # c: a silent socket is a stall: no bytes for socket_timeout seconds.
    started = time.monotonic()
    with server(silent_after_first=3.0) as endpoint:
        with pytest.raises(TimeoutError, match="stalled"):
            kernel_bounded_http.request_bytes(
                WIRE, endpoint, "", deadline=None, socket_timeout=0.5)
    assert time.monotonic() - started < scaled(2.0)


@pytest.mark.host_sandbox
def test_run_deadline_still_caps_streaming_body():
    # d: a user/run cap wins over progress renewal.
    with server(chunks=15, chunk_interval=0.2) as endpoint:
        with pytest.raises(TimeoutError, match="absolute deadline"):
            kernel_bounded_http.request_bytes(
                WIRE, endpoint, "", deadline=None, socket_timeout=1.0,
                run_deadline=time.monotonic() + 1.5)


@pytest.mark.host_sandbox
def test_no_deadline_slow_first_byte_is_not_an_elapsed_timeout():
    # e: first-byte waiting is liveness's job, not elapsed time. Probes do not fire inside 1.5 s.
    with server(first_byte_delay=1.5) as endpoint:
        reply = kernel_bounded_http.request_bytes(
            WIRE, endpoint, "", deadline=None, socket_timeout=0.5)
    assert reply == REPLY


@pytest.mark.host_sandbox
def test_no_deadline_large_upload_completes():
    # f: request-body send paths must stay inf-safe (no inf reaches socket.settimeout).
    big_wire = json.dumps({"model": "scripted", "messages": [
        {"role": "user", "content": secrets.token_hex(1024 * 1024)}]}).encode()
    with server() as endpoint:
        reply = kernel_bounded_http.request_bytes(
            big_wire, endpoint, "", deadline=None, socket_timeout=2.0)
    assert reply == REPLY
