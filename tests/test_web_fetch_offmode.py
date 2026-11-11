"""Ordinary web fetch preserves its early text-content refusal behavior."""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stateframe.engine import kernel_web_fetch as web


_BINARY_BODY = b"%PDF-1.7 binary payload"


class FetchHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/stalled-binary":
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", "4096")
            self.end_headers()
            self.server.stalled_headers_sent.set()
            self.server.release_stalled_body.wait()
            try:
                self.wfile.write(b"x" * 4096)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        body = b"%PDF-1.7 oversized binary payload" if self.path == "/oversized-binary" else _BINARY_BODY
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


class FetchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), FetchHandler)
        self.stalled_headers_sent = threading.Event()
        self.release_stalled_body = threading.Event()


@pytest.fixture
def fetch_server():
    server = FetchServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.release_stalled_body.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def loopback(*_args: object, **_kwargs: object) -> list[tuple[int, int, int, str, tuple[str, int]]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]


def policy(port: int, *, deadline: float = 20.0, max_body: int = 2 << 20,
           max_decoded: int = 8 << 20) -> web.WebPolicy:
    return web.WebPolicy(("fetch.test",), max_body=max_body, max_decoded=max_decoded,
                         deadline=deadline, allowed_ports=(port,),
                         allow_private_for_tests=True)


def refusal() -> web.WebResult:
    return web.WebResult(
        False, None, None, None, "", 0,
        "content type is not readable text: application/pdf")


@pytest.mark.host_sandbox
def test_fetch_rejects_non_text_before_reading_stalled_body(fetch_server):
    url = f"http://fetch.test:{fetch_server.server_port}/stalled-binary"
    result_ready = threading.Event()
    results: list[web.WebResult] = []
    errors: list[Exception] = []

    def fetch() -> None:
        try:
            results.append(web.fetch(url, policy(fetch_server.server_port, deadline=15),
                                     resolver=loopback))
        except Exception as exc:
            errors.append(exc)
        finally:
            result_ready.set()

    worker = threading.Thread(target=fetch, daemon=True)
    worker.start()
    try:
        headers_sent = fetch_server.stalled_headers_sent.wait(timeout=2)
        returned_while_body_stalled = result_ready.wait(timeout=1.5)
    finally:
        fetch_server.release_stalled_body.set()
        worker.join(timeout=2)

    assert headers_sent
    assert returned_while_body_stalled, "fetch waited for a non-text response body"
    assert not errors, f"fetch raised unexpectedly: {errors}"
    assert results == [refusal()]


@pytest.mark.host_sandbox
def test_fetch_oversized_binary_keeps_v0426_content_type_refusal(fetch_server):
    url = f"http://fetch.test:{fetch_server.server_port}/oversized-binary"
    result = web.fetch(url, policy(fetch_server.server_port, max_body=4, max_decoded=32),
                       resolver=loopback)

    assert result == refusal()


@pytest.mark.host_sandbox
def test_fetch_raw_still_returns_non_text_body(fetch_server):
    url = f"http://fetch.test:{fetch_server.server_port}/binary"
    result = web.fetch_raw(url, policy(fetch_server.server_port), max_bytes=1024,
                           resolver=loopback)

    assert result.error is None
    assert result.status == 200
    assert result.headers["content-type"] == "application/pdf"
    assert result.body == _BINARY_BODY
