"""Model HTTP requests must ignore inherited and system proxy settings."""

import ast
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from stateframe import _child
from stateframe.engine import kernel_bounded_http
from stateframe.engine.kernel_plain_http import AbortiveHTTPTransport
from stateframe.engine.kernel_request_diet import _http_tokenize


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def servers():
    target_requests = []
    proxy_requests = []

    class Target(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            target_requests.append((self.path, body, dict(self.headers)))
            reply = b'{"tokens":[1,2]}' if self.path == "/tokenize" else b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def do_GET(self):  # noqa: N802
            target_requests.append((self.path, None, dict(self.headers)))
            reply = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _reject(self):
            length = int(self.headers.get("Content-Length", "0"))
            proxy_requests.append((self.path, self.rfile.read(length), dict(self.headers)))
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):  # noqa: N802
            self._reject()

        def do_GET(self):  # noqa: N802
            self._reject()

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    threads = [threading.Thread(target=server.serve_forever, daemon=True)
               for server in (target, proxy)]
    for thread in threads:
        thread.start()
    try:
        yield (f"http://127.0.0.1:{target.server_port}", target_requests,
               f"http://127.0.0.1:{proxy.server_port}", proxy_requests)
    finally:
        for server in (target, proxy):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)


def _set_proxy(monkeypatch, proxy_url):
    for name in ("HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(name, proxy_url)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.host_sandbox
def test_bounded_http_ignores_proxy(monkeypatch, servers):
    target_url, target_requests, proxy_url, proxy_requests = servers
    _set_proxy(monkeypatch, proxy_url)
    wire = b'{"model":"scripted","messages":[]}'
    assert kernel_bounded_http.request_bytes(
        wire, target_url + "/v1/chat/completions", "secret",
        deadline=time.monotonic() + 10, socket_timeout=5) == b'{"ok":true}'
    assert [(path, body) for path, body, _ in target_requests] == [
        ("/v1/chat/completions", wire)]
    assert target_requests[0][2]["Authorization"] == "Bearer secret"
    assert proxy_requests == []


@pytest.mark.host_sandbox
def test_tokenize_ignores_proxy(monkeypatch, servers):
    target_url, target_requests, proxy_url, proxy_requests = servers
    _set_proxy(monkeypatch, proxy_url)
    assert _http_tokenize("hello", target_url + "/tokenize") == [1, 2]
    assert len(target_requests) == 1
    assert json.loads(target_requests[0][1]) == {
        "content": "hello", "add_special": False, "parse_special": True}
    assert proxy_requests == []


@pytest.mark.host_sandbox
def test_plain_http_ignores_proxy(monkeypatch, servers):
    target_url, target_requests, proxy_url, proxy_requests = servers
    _set_proxy(monkeypatch, proxy_url)
    tracker = AbortiveHTTPTransport()
    tracker_id = tracker.begin()
    try:
        request = urllib.request.Request(target_url + "/v1/chat/completions", data=b"body")
        with tracker.opener(tracker_id).open(request, timeout=5) as response:
            assert response.read() == b'{"ok":true}'
    finally:
        tracker.finish(tracker_id)
        tracker.release(tracker_id)
    assert [(path, body) for path, body, _ in target_requests] == [
        ("/v1/chat/completions", b"body")]
    assert proxy_requests == []


@pytest.mark.host_sandbox
def test_engine_child_global_opener_ignores_proxy(monkeypatch, servers):
    target_url, target_requests, proxy_url, proxy_requests = servers
    _set_proxy(monkeypatch, proxy_url)
    original = urllib.request._opener
    try:
        _child._install_direct_opener()
        with urllib.request.urlopen(target_url + "/slots", timeout=5) as response:
            assert response.read() == b'{"ok":true}'
    finally:
        urllib.request._opener = original
    assert [path for path, _, _ in target_requests] == ["/slots"]
    assert proxy_requests == []


def test_engine_child_installs_opener_after_environment():
    tree = ast.parse((ROOT / "src/stateframe/_child.py").read_text())
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node.value.func.id for node in main.body
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name)]
    index = calls.index("install_environment")
    assert calls[index + 1] == "_install_direct_opener"


def _call_name(node):
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def test_every_build_opener_explicitly_disables_proxy():
    offenders = []
    for path in (ROOT / "src/stateframe").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if _call_name(node) != "build_opener":
                continue
            direct = any(
                _call_name(arg) == "ProxyHandler"
                and len(arg.args) == 1
                and isinstance(arg.args[0], ast.Dict) and not arg.args[0].keys
                for arg in node.args)
            if not direct:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, "build_opener without ProxyHandler({}): " + ", ".join(offenders)
