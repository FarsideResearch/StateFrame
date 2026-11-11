"""Model-server transports must attach their configured bearer credential."""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from _class_guard_scanners import ALLOWLIST, scan_auth_requests
from _timeouts import scaled


pytestmark = pytest.mark.host_sandbox

_KEY = "auth-class-guard-test-key"
_BODY = {"messages": [{"role": "user", "content": "auth guard"}], "tools": []}
_AUTH_SCAN_FLOOR = 20  # The current 19 keyed sites plus the user-web-fetch exception.


class _KeyedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


class _KeyedHandler(BaseHTTPRequestHandler):
    def _reply(self, status: int, payload: object) -> None:
        if isinstance(payload, bytes):
            body, content_type = payload, "text/plain"
        else:
            body = json.dumps(payload).encode("utf-8")
            content_type = "application/json"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _dispatch(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        authorization = self.headers.get("Authorization")
        self.server.requests.append((self.command, self.path, authorization))
        if authorization != self.server.required_authorization:
            self._reply(401, {"error": "authorization required"})
            return

        path = urlsplit(self.path).path
        if self.command == "GET" and path == "/slots":
            self._reply(200, self.server.slots_payload)
        elif self.command == "GET" and path == "/props":
            self._reply(200, {"default_generation_settings": {"n_ctx": 8192}})
        elif self.command == "GET" and path == "/health":
            self._reply(200, b"ok")
        elif self.command == "GET" and path == "/v1/models":
            self._reply(200, {"data": [{"id": "guard-model"}]})
        elif self.command == "POST" and path == "/apply-template":
            self._reply(200, {"prompt": "rendered auth guard prompt"})
        elif self.command == "POST" and path == "/tokenize":
            self._reply(200, {"tokens": [1, 2, 3]})
        elif self.command == "POST" and path == "/v1/chat/completions":
            self._reply(200, {"choices": [{"message": {"content": '{"answer":"yes"}'}}]})
        elif self.command == "POST" and path.startswith("/slots/"):
            self._reply(200, {})
        else:
            self._reply(404, {"error": "unknown route"})

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _start_server(required_authorization: str = f"Bearer {_KEY}"):
    server = _KeyedHTTPServer(("127.0.0.1", 0), _KeyedHandler)
    server.required_authorization = required_authorization
    server.requests = []
    server.slots_payload = [{"id": 0, "is_processing": True}]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    return server, thread, base


def _stop_server(server: _KeyedHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=scaled(5))
    assert not thread.is_alive(), "auth guard HTTP server did not stop"


def test_auth_source_scan_classifies_every_outbound_http_site() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src/stateframe").rglob("*.py"))
    report = scan_auth_requests(sources, root=root)
    print(report.receipt)

    assert re.fullmatch(
        r"auth-guard scan: scanned=\d+ keyed=\d+ allowlisted=\d+ violations=\d+",
        report.receipt,
    )
    assert report.scanned == report.keyed + report.allowlisted + len(report.violations)
    assert report.scanned >= _AUTH_SCAN_FLOOR
    assert report.allowlisted == len(ALLOWLIST)
    assert all(reason.strip() and "\n" not in reason for reason in ALLOWLIST.values())
    assert not report.violations, "unclassified outbound HTTP sites:\n" + "\n".join(report.violations)


def test_keyed_public_model_server_entrypoints_send_the_configured_key(monkeypatch) -> None:
    from stateframe.endpoint import Endpoint
    import capability_capacity_preflight
    import decode_depth_per_arm
    import kernel_context_capacity
    import kernel_kv_chunk_reuse
    import kernel_template_conformance
    import long_horizon_decode_logging
    from detreason.kernel import Atom, LlamaSensor
    from fixer import _post
    from kernel_bounded_http import provider_health_ok, request_bytes
    from kernel_plain_host import http_provider
    from kernel_request_diet import RequestTokenMeter
    from swarm.client2 import ask

    server, thread, base = _start_server()
    endpoint_url = base + "/v1/chat/completions"
    monkeypatch.setenv("AUTH_CLASS_GUARD_KEY", _KEY)
    monkeypatch.setenv("LLM_API_KEY", _KEY)
    monkeypatch.setenv("LLAMA_CPP_TOKENIZE_URL", base + "/tokenize")
    monkeypatch.setenv("LONG_HORIZON_TELEMETRY", "on")
    monkeypatch.setenv("LONG_HORIZON_DECODE_DEPTH_PER_ARM", "on")
    try:
        endpoint = Endpoint.from_args(base + "/v1", "guard-model", "AUTH_CLASS_GUARD_KEY")
        assert endpoint.get_slots() == server.slots_payload
        assert endpoint.probe()["reachable"] is True

        monkeypatch.setattr(kernel_context_capacity, "_cache", {})
        assert kernel_context_capacity.probe_n_ctx(endpoint_url, _KEY) == 8192

        assert provider_health_ok(endpoint_url, _KEY, time.monotonic() + scaled(5))
        assert provider_health_ok(
            f"http://localhost:{server.server_port}/v1/chat/completions",
            _KEY,
            time.monotonic() + scaled(5),
        )

        meter = RequestTokenMeter(
            endpoint=base + "/tokenize", template_overhead_tokens=0,
            auth_key=_KEY, generation_endpoint=endpoint_url,
        )
        assert meter.measure(_BODY)[1] == "tokenize"

        rendered = kernel_template_conformance.render_endpoint_request(
            _BODY, endpoint_url, _KEY,
        )
        assert rendered["token_ids"] == [1, 2, 3]

        long_capture = type("Capture", (), {})()
        assert long_horizon_decode_logging.observe_slots(
            long_capture, endpoint_url, key=_KEY,
        ) == 1
        arm_capture = type("Capture", (), {})()
        assert decode_depth_per_arm.observe_slots(arm_capture, endpoint_url, key=_KEY) == 1

        assert kernel_kv_chunk_reuse.slot_action(
            endpoint_url, 0, "erase", None, _KEY,
        ) == {}

        server.slots_payload = [{"id": 0, "n_ctx": 8192, "is_processing": True}]

        class Journal:
            def emit(self, _kind, **_fields):
                return None

        class Capture:
            def __init__(self):
                self.journal = Journal()
                self.request_id = "auth-class-guard"

            def send(self, body, endpoint, key, *, deadline=None):
                return "original send"

        capability_capacity_preflight.install(
            {"capacity_preflight": {"context_limit": 8192}}, Capture,
        )
        assert Capture().send(
            {"max_tokens": 64, "id_slot": 0}, endpoint_url, _KEY,
        ) == "original send"

        bounded_response = request_bytes(
            json.dumps(_BODY).encode(), endpoint_url, _KEY,
            deadline=time.monotonic() + scaled(10),
            socket_timeout=scaled(3), liveness_timeout=scaled(3),
        )
        assert json.loads(bounded_response)["choices"]

        plain_response = http_provider(endpoint_url, _KEY)(_BODY, scaled(5))
        assert plain_response["choices"]
        assert _post(_BODY, endpoint_url, timeout=scaled(5))["choices"]
        assert ask("guard", url=endpoint_url, timeout=scaled(5)).ok
        assert LlamaSensor(endpoint_url, margin_probe=False, timeout=scaled(5)).answer(
            Atom("a", "supported")).value == "TRUE"

        expected_paths = {
            "/v1/models", "/slots", "/props", "/health", "/tokenize",
            "/apply-template", "/v1/chat/completions", "/slots/0?action=erase",
        }
        recorded_paths = {path for _method, path, _authorization in server.requests}
        assert expected_paths <= recorded_paths
        assert server.requests
        assert all(authorization == f"Bearer {_KEY}"
                   for _method, _path, authorization in server.requests), server.requests
    finally:
        _stop_server(server, thread)


def test_source_scanner_rejects_a_planted_keyless_urlopen(tmp_path: Path) -> None:
    planted = tmp_path / "planted_keyless.py"
    planted.write_text(
        "import urllib.request\n"
        "def call():\n"
        "    urllib.request.urlopen('http://127.0.0.1:8000/v1/chat/completions')\n",
        encoding="utf-8",
    )
    report = scan_auth_requests([planted], root=tmp_path)
    assert report.scanned == 1
    assert report.keyed == 0
    assert report.allowlisted == 0
    assert len(report.violations) == 1
    assert report.receipt == (
        "auth-guard scan: scanned=1 keyed=0 allowlisted=0 violations=1")


@pytest.mark.parametrize("source, expected_kind", [
    ("from urllib.request import Request as R\n"
     "def call():\n    return R('http://127.0.0.1:8000/v1/chat/completions')\n", "Request"),
    ("import urllib.request as u\n"
     "def call():\n    return u.urlopen('http://127.0.0.1:8000/v1/chat/completions')\n", "urlopen"),
    ("import urllib.request as u\n"
     "def call():\n    send = u.urlopen\n"
     "    return send('http://127.0.0.1:8000/v1/chat/completions')\n", "urlopen"),
    ("import urllib.request as u\n"
     "def call():\n    client = u.build_opener()\n"
     "    return client.open('http://127.0.0.1:8000/v1/chat/completions')\n", "open"),
    ("import urllib.request\n"
     "def make_request():\n    return urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions')\n"
     "def call():\n    return urllib.request.urlopen(make_request())\n", "urlopen"),
    ("import http.client as hc\n"
     "def call():\n    connection = hc.HTTPConnection('127.0.0.1')\n"
     "    connection.request('GET', '/v1/models')\n", "request"),
    ("from http.client import HTTPConnection as H\n"
     "def call():\n    connection = H('127.0.0.1')\n"
     "    connection.request('GET', '/v1/models')\n", "request"),
    ("import subprocess as sp\n"
     "def call():\n    command = ['curl', 'http://127.0.0.1:8000/v1/models']\n"
     "    sp.run(command)\n", "curl"),
    ("import subprocess as sp\n"
     "def command():\n    return ['curl', 'http://127.0.0.1:8000/v1/models']\n"
     "def call():\n    sp.run(command())\n", "curl"),
    ("import urllib.request\n"
     "def auth_headers():\n    return {}\n"
     "def call():\n    return urllib.request.Request('http://127.0.0.1:8000/v1/models', "
     "headers=auth_headers())\n", "Request"),
    ("import urllib.request\n"
     "def call(key):\n    headers = {}\n"
     "    request = urllib.request.Request('http://127.0.0.1:8000/v1/models', headers=headers)\n"
     "    headers['Authorization'] = 'Bearer ' + key\n"
     "    return urllib.request.urlopen(request)\n", "Request"),
])
def test_auth_scanner_rejects_keyless_transport_variants(
        tmp_path: Path, source: str, expected_kind: str) -> None:
    planted = tmp_path / "keyless_variant.py"
    planted.write_text(source, encoding="utf-8")
    report = scan_auth_requests([planted], root=tmp_path)
    assert report.violations, report.receipt
    assert any(expected_kind in violation for violation in report.violations)


def test_auth_scanner_resolves_imported_request_alias(tmp_path: Path) -> None:
    planted = tmp_path / "keyed_alias.py"
    planted.write_text(
        "from urllib.request import Request as R, urlopen as open_url\n"
        "def call(key):\n"
        "    request = R('http://127.0.0.1:8000/v1/models', "
        "headers={'Authorization': 'Bearer ' + key})\n"
        "    return open_url(request)\n",
        encoding="utf-8",
    )
    report = scan_auth_requests([planted], root=tmp_path)
    assert report.receipt == "auth-guard scan: scanned=1 keyed=1 allowlisted=0 violations=0"


def test_auth_scanner_requires_review_of_new_http_libraries(tmp_path: Path) -> None:
    planted = tmp_path / "untracked_transport.py"
    planted.write_text("import httpx\n", encoding="utf-8")
    report = scan_auth_requests([planted], root=tmp_path)
    assert report.scanned == 1
    assert report.violations == (
        "untracked_transport.py:<import>:1 (unsupported HTTP library: httpx)",
    )


def test_bounded_transport_keeps_the_blank_bearer_header() -> None:
    from kernel_bounded_http import request_bytes

    server, thread, base = _start_server(required_authorization="Bearer ")
    try:
        response = request_bytes(
            b'{"messages":[]}', base + "/v1/chat/completions", "",
            deadline=time.monotonic() + scaled(10),
            socket_timeout=scaled(3), liveness_timeout=scaled(3),
        )
        assert json.loads(response)["choices"]
        chat_requests = [record for record in server.requests
                         if record[1] == "/v1/chat/completions"]
        assert chat_requests
        assert all(record[2] == "Bearer " for record in chat_requests)
    finally:
        _stop_server(server, thread)


def test_endpoint_without_a_key_omits_the_authorization_header() -> None:
    from stateframe.endpoint import Endpoint

    server, thread, base = _start_server(required_authorization=None)
    try:
        endpoint = Endpoint.from_args(base + "/v1", "guard-model", None)
        assert endpoint.get_slots() == server.slots_payload
        assert endpoint.probe()["reachable"] is True
        assert server.requests
        assert all(authorization is None for _method, _path, authorization in server.requests)
    finally:
        _stop_server(server, thread)
