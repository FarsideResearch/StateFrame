"""Context-size probe reads whole /slots and /props bodies up to a fixed cap; larger bodies are refused typed."""
import io
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import kernel_context_capacity as capacity
import kernel_reason_log as rl

LINE = re.compile(r"^\[stateframe\] (WARN|ERROR|INFO) (?P<component>[a-z_]+): .+", re.M)
PAD = "x" * (1100 * 1024)  # just over 1 MiB, the old truncation point


def _slots_body() -> bytes:
    return json.dumps([{"id": 0, "n_ctx": 65536, "pad": PAD}]).encode()


def _props_body() -> bytes:
    return json.dumps({"default_generation_settings": {"n_ctx": 65536}, "pad": PAD}).encode()


@pytest.fixture(autouse=True)
def stream(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    capacity._cache.clear()
    out = io.StringIO()
    rl.configure(level="INFO", stream=out)
    yield out
    capacity._cache.clear()
    rl.reset_for_tests()


@pytest.fixture
def server():
    routes = {"/slots": _slots_body(), "/props": _props_body()}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 -- http.server naming
            body = routes.get(self.path)
            if body is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep test output quiet
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _lines(out, component):
    return [m.group(0) for m in LINE.finditer(out.getvalue()) if m.group("component") == component]


def test_oversized_slots_body_is_read_whole_under_default_cap(server):
    assert capacity.probe_n_ctx(server) == 65536


def test_body_over_cap_is_refused_typed_and_not_parsed(server, monkeypatch, stream):
    monkeypatch.setattr(capacity, "_PROBE_MAX_BYTES", 1024 * 1024)
    assert capacity.probe_n_ctx(server) is None
    lines = _lines(stream, "context_capacity")
    for path in ("/slots", "/props"):
        assert any("context_probe_body_over_cap" in line and f"GET {path} " in line
                   and "1048576 bytes" in line for line in lines), stream.getvalue()
