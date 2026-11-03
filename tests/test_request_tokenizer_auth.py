"""Request tokenizer authentication stays scoped to the generation origin."""
import sys
import threading
import json
from types import SimpleNamespace
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import kernel_request_diet
from kernel_request_diet import RequestTokenMeter, _http_tokenize, resolve_tokenize_url
from stateframe.scripted_endpoint import Script, serve


pytestmark = pytest.mark.host_sandbox

_BODY = {"messages": [{"role": "user", "content": "tokenizer auth regression"}], "tools": []}
_KEY = "tokenize-test-key"


def _server(tokenize):
    script = Script({"steps": [], "tokenize": tokenize})
    server = serve(script)
    base = f"http://127.0.0.1:{server.server_port}"
    return script, server, base


def _measure_with_request_key(tokenize_url, generation_endpoint, key, monkeypatch):
    from kernel_request_budget import _request_token_meter

    monkeypatch.setitem(
        sys.modules,
        "probes.react_repair",
        SimpleNamespace(KEY=key, API=generation_endpoint),
    )
    capture = SimpleNamespace(
        _tokenize_endpoint=tokenize_url,
        request_id="tokenizer-auth-test",
        journal=None,
    )
    return _request_token_meter(None, capture).measure(_BODY)


@pytest.fixture
def fallback_events(monkeypatch):
    from stateframe import _child

    events = []
    monkeypatch.setattr(_child, "emit", lambda kind, **fields: events.append((kind, fields)))
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    # Register the method with monkeypatch before install_token_count_warning wraps it.
    monkeypatch.setattr(RequestTokenMeter, "measure", RequestTokenMeter.measure)
    _child.install_token_count_warning()
    return events


def _assert_fallback_reason(events, reason):
    unavailable = [(kind, fields) for kind, fields in events if kind.endswith("_unavailable")]
    assert unavailable
    assert all(isinstance(fields.get("reason"), str) and fields["reason"]
               for _kind, fields in unavailable)
    assert events == [("token_counting_unavailable", {"reason": reason})]


def test_keyed_same_origin_tokenizer_is_measured_and_receives_key(monkeypatch):
    script, server, base = _server({"mode": "ok", "require_key": _KEY})
    try:
        count, method = _measure_with_request_key(
            base + "/tokenize", base + "/v1/chat/completions", _KEY, monkeypatch,
        )
        assert count > 0
        assert method == "tokenize"
        assert script.tokenize_authorization_present == [True]
        assert script.tokenize_authorized == [True]
    finally:
        server.shutdown()
        server.server_close()


def test_meter_constructor_auth_survives_a_thread_handoff():
    script, server, base = _server({"mode": "ok", "require_key": _KEY})
    try:
        meter = RequestTokenMeter(
            endpoint=base + "/tokenize",
            template_overhead_tokens=0,
            auth_key=_KEY,
            generation_endpoint=base + "/v1/chat/completions",
        )
        result = []

        def measure():
            try:
                result.append(("ok", meter.measure(_BODY)))
            except Exception as error:  # captured so the worker thread cannot hide the failure
                result.append(("error", error))

        thread = threading.Thread(target=measure)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), "tokenizer measurement did not finish"
        assert result and result[0][0] == "ok", result
        count, method = result[0][1]
        assert count > 0
        assert method == "tokenize"
        assert script.tokenize_authorization_present == [True]
        assert script.tokenize_authorized == [True]
    finally:
        server.shutdown()
        server.server_close()


def test_routed_meter_uses_route_specific_generation_credentials(monkeypatch):
    from kernel_frame_routes import Route
    from kernel_request_budget import _request_token_meter

    script, server, base = _server({"mode": "ok", "require_key": _KEY})
    try:
        monkeypatch.setenv("SF_ROUTE_TOKENIZER_KEY", _KEY)
        monkeypatch.setattr(kernel_request_diet._frame_routes, "ROUTES_ACTIVE", True)
        capture = SimpleNamespace(
            frame_route=Route(
                role="edit",
                scope="request",
                unit_class=None,
                settings={
                    "endpoint": base + "/v1",
                    "model": "scripted",
                    "api_key_env": "SF_ROUTE_TOKENIZER_KEY",
                    "allow_remote": False,
                },
            ),
            _tokenize_endpoint=base + "/tokenize",
            request_id="routed-tokenizer-auth-test",
            journal=None,
        )
        count, method = _request_token_meter(None, capture).measure(_BODY)
        assert count > 0
        assert method == "tokenize"
        assert script.tokenize_authorization_present == [True]
        assert script.tokenize_authorized == [True]
    finally:
        server.shutdown()
        server.server_close()


def test_explicit_different_origin_tokenizer_does_not_receive_generation_key(monkeypatch):
    generation_script, generation_server, generation_base = _server({})
    tokenizer_script, tokenizer_server, tokenizer_base = _server({"mode": "ok"})
    try:
        monkeypatch.setenv("LLAMA_CPP_TOKENIZE_URL", tokenizer_base + "/tokenize")
        tokenize_url = resolve_tokenize_url()
        assert tokenize_url == tokenizer_base + "/tokenize"
        count, method = _measure_with_request_key(
            tokenize_url, generation_base + "/v1/chat/completions", _KEY, monkeypatch,
        )
        assert count > 0
        assert method == "tokenize"
        assert generation_script.tokenize_authorization_present == []
        assert tokenizer_script.tokenize_authorization_present == [False]
    finally:
        tokenizer_server.shutdown()
        tokenizer_server.server_close()
        generation_server.shutdown()
        generation_server.server_close()


def test_same_origin_redirect_strips_key_before_cross_origin_tokenizer(monkeypatch, fallback_events):
    external_authorization = []

    class ExternalHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            external_authorization.append(self.headers.get("Authorization"))
            payload = json.dumps({"tokens": [1]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            return

    external_server = ThreadingHTTPServer(("127.0.0.1", 0), ExternalHandler)
    external_thread = threading.Thread(target=external_server.serve_forever, daemon=True)
    external_thread.start()
    external_base = f"http://127.0.0.1:{external_server.server_port}"
    redirect_authorization = []

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            redirect_authorization.append(self.headers.get("Authorization"))
            self.send_response(303)
            self.send_header("Location", external_base + "/tokenize")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, _format, *_args):
            return

    redirect_server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = threading.Thread(target=redirect_server.serve_forever, daemon=True)
    thread.start()
    redirect_base = f"http://127.0.0.1:{redirect_server.server_port}"
    try:
        count, method = _measure_with_request_key(
            redirect_base + "/tokenize", redirect_base + "/v1/chat/completions", _KEY, monkeypatch,
        )
        assert count > 0
        assert method == "tokenize"
        assert redirect_authorization == [f"Bearer {_KEY}"]
        assert external_authorization == [None]
        assert not fallback_events
    finally:
        redirect_server.shutdown()
        redirect_server.server_close()
        external_server.shutdown()
        external_server.server_close()


def test_wrong_or_missing_key_reports_unauthorized_tokenizer_fallback(
        monkeypatch, fallback_events):
    from stateframe import _child

    script, server, base = _server({"mode": "ok", "require_key": _KEY})
    try:
        for key in ("wrong-tokenizer-key", None):
            _child.HOST_STATE["token_warning"]["done"] = False
            fallback_events.clear()
            _count, method = _measure_with_request_key(
                base + "/tokenize", base + "/v1/chat/completions", key, monkeypatch,
            )
            assert method == "calibrated_bytes"
            _assert_fallback_reason(fallback_events, "unauthorized")
        assert script.tokenize_authorized
        assert not any(script.tokenize_authorized)
        assert script.tokenize_authorization_present.count(False) > 0
    finally:
        server.shutdown()
        server.server_close()


def test_forbidden_tokenizer_response_reports_unauthorized(fallback_events):
    _script, server, base = _server({"mode": "fail", "status": 403})
    try:
        count, method = RequestTokenMeter(
            endpoint=base + "/tokenize", template_overhead_tokens=0,
        ).measure(_BODY)
        assert count > 0
        assert method == "calibrated_bytes"
        _assert_fallback_reason(fallback_events, "unauthorized")
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("status", [404, 503])
def test_other_http_errors_report_http_error(fallback_events, status):
    _script, server, base = _server({"mode": "fail", "status": status})
    try:
        count, method = RequestTokenMeter(
            endpoint=base + "/tokenize", template_overhead_tokens=0,
        ).measure(_BODY)
        assert count > 0
        assert method == "calibrated_bytes"
        _assert_fallback_reason(fallback_events, "http_error")
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("cause", [ConnectionRefusedError, ConnectionResetError])
def test_connection_errors_report_connection(monkeypatch, fallback_events, cause):
    def fail(*_args, **_kwargs):
        raise urllib.error.URLError(cause("tokenizer connection failed"))

    monkeypatch.setattr(kernel_request_diet, "_http_tokenize", fail)
    count, method = RequestTokenMeter(
        endpoint="http://127.0.0.1:1/tokenize", template_overhead_tokens=0,
    ).measure(_BODY)
    assert count > 0
    assert method == "calibrated_bytes"
    _assert_fallback_reason(fallback_events, "connection")


@pytest.mark.parametrize("wrapped", [False, True])
def test_timeout_errors_report_timeout(monkeypatch, fallback_events, wrapped):
    def timeout(*_args, **_kwargs):
        error = TimeoutError("tokenizer timed out")
        if wrapped:
            raise urllib.error.URLError(error)
        raise error

    monkeypatch.setattr(kernel_request_diet, "_http_tokenize", timeout)
    count, method = RequestTokenMeter(
        endpoint="http://127.0.0.1:1/tokenize", template_overhead_tokens=0,
    ).measure(_BODY)
    assert count > 0
    assert method == "calibrated_bytes"
    _assert_fallback_reason(fallback_events, "timeout")


@pytest.mark.parametrize("mode", ["bad_json", "no_tokens"])
def test_invalid_tokenizer_response_reports_bad_response(fallback_events, mode):
    _script, server, base = _server({"mode": mode})
    try:
        count, method = RequestTokenMeter(
            endpoint=base + "/tokenize", template_overhead_tokens=0,
        ).measure(_BODY)
        assert count > 0
        assert method == "calibrated_bytes"
        _assert_fallback_reason(fallback_events, "bad_response")
    finally:
        server.shutdown()
        server.server_close()


def test_missing_tokenizer_route_reports_no_url(fallback_events):
    count, method = RequestTokenMeter(template_overhead_tokens=0).measure(_BODY)
    assert count > 0
    assert method == "calibrated_bytes"
    _assert_fallback_reason(fallback_events, "no_url")


def test_disabled_tokenizer_reports_disabled_without_calling_route(fallback_events):
    script, server, base = _server({"mode": "ok"})
    try:
        meter = RequestTokenMeter(
            endpoint=base + "/tokenize", template_overhead_tokens=0, disabled=True,
        )
        count, method = meter.measure(_BODY)
        assert count > 0
        assert method == "calibrated_bytes"
        assert script.tokenize_calls == 0
        _assert_fallback_reason(fallback_events, "disabled")
    finally:
        server.shutdown()
        server.server_close()


def test_no_key_tokenizer_keeps_original_request_bytes_and_behavior():
    script, server, base = _server({"mode": "ok", "record_request_bytes": True})
    try:
        assert _http_tokenize("hello", base + "/tokenize") == [0]
        assert script.tokenize_request_bytes == [
            b'{"content": "hello", "add_special": false, "parse_special": true}'
        ]
        assert script.tokenize_authorization_present == [False]
        assert script.tokenize_authorized == [True]
    finally:
        server.shutdown()
        server.server_close()


def test_bound_tokenizer_credentials_are_not_serialized_to_checkpoints():
    from kernel_checkpoint import DROP_ATTRS, _runtime_state

    meter = RequestTokenMeter(endpoint="http://127.0.0.1:8080/tokenize")
    meter.bind_tokenizer_auth(_KEY, "http://127.0.0.1:8080/v1/chat/completions")
    reduce_args = _runtime_state(
        meter, DROP_ATTRS[("kernel_request_diet", "RequestTokenMeter")],
    )
    state = reduce_args[2]
    assert "_tokenizer_auth_key" not in state
    assert "_generation_endpoint" not in state
