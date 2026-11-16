"""Request tokenizer authentication stays scoped to the generation origin."""
import io
import json
import os
import sys
import threading
import time
import types
from types import SimpleNamespace
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from _timeouts import scaled
import kernel_checkpoint
import kernel_measurement
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
        thread.join(timeout=scaled(5))
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
    import ssl

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
        def do_GET(self):  # noqa: N802
            payload = b'{"default_generation_settings":{"n_ctx":65536},"model_path":"mock-llama"}'
            self.send_response(200 if self.path == "/props" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

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
        contexts = 0
        original_factory = ssl._create_default_https_context

        def counted_context(*args, **kwargs):
            nonlocal contexts
            contexts += 1
            return original_factory(*args, **kwargs)

        monkeypatch.setattr(ssl, "_create_default_https_context", counted_context)
        generation_endpoint = redirect_base + "/v1/chat/completions"
        meter = RequestTokenMeter(
            endpoint="https://context.example/tokenize",
            template_overhead_tokens=0,
            auth_key=_KEY,
            generation_endpoint=generation_endpoint,
        )
        cached_context = meter._get_http_ssl_context()
        previous_opener = None
        for content in ("first tokenizer request", "second tokenizer request"):
            opener = meter._get_http_opener()
            assert opener is not None and opener is not previous_opener
            previous_opener = opener
            https_handler = next(
                handler for handler in opener.handlers
                if isinstance(handler, kernel_request_diet._TokenizeHTTPSHandler)
            )
            assert https_handler._context_provider() is cached_context
            assert kernel_request_diet._http_tokenize(
                content,
                redirect_base + "/tokenize",
                auth_key=_KEY,
                generation_endpoint=generation_endpoint,
                opener=opener,
            ) == [1]

        assert contexts == 1
        assert redirect_authorization == [f"Bearer {_KEY}", f"Bearer {_KEY}"]
        assert external_authorization == [None, None]
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


class _CaptureJournal:
    task_id = "checkpoint-secret-test"
    manifest = {"run_identity_seed": "31" * 16}

    def __init__(self):
        self.events = []

    def request_records(self, _request_id):
        return ()

    def emit(self, kind, **fields):
        self.events.append((kind, fields))


@pytest.fixture
def registered_checkpoint_key():
    previous = kernel_checkpoint._SECRETS.copy()
    key = "STATEFRAME_CHECKPOINT_TEST_SECRET_123"
    kernel_checkpoint.register_secret(key)
    try:
        yield key
    finally:
        kernel_checkpoint._SECRETS[:] = previous


def _request_capture_with_key(key):
    from kernel_measurement import RequestCapture

    capture = RequestCapture(
        _CaptureJournal(), 1, "snapshot", endpoint_profile={"profile": "test"},
    )
    capture._provider_probe_destination = ("http://127.0.0.1:8080/v1", key)
    return capture


@pytest.fixture
def model_router_test_config():
    from kernel_stateframe_model_router import EndpointConfig, ModelRouterConfig

    return ModelRouterConfig(
        endpoints=(EndpointConfig(
            name="router-moe",
            model_id="test/router-moe",
            role="moe",
            url="https://router-provider.example/v1/chat/completions",
            api_key_env="STATEFRAME_ROUTER_PROVIDER_KEY",
            context_tokens=8192,
        ),),
        default_endpoint="router-moe",
        lease_actions=2,
    )


def _prepare_capture_with_model_router(capture, config, *, mode, endpoint, key):
    from kernel_stateframe_model_router import ModelRouter

    router = ModelRouter(
        mode=mode, config=config, base_model="primary-model", replay_enabled=False,
    )
    prepared = router.prepare_call(capture, _BODY, endpoint, key)
    capture._provider_probe_destination = (prepared.endpoint, prepared.key)
    return prepared


def _record_physical_send_provenance(capture, endpoint):
    from kernel_measurement import _record_probe_destination_provenance

    _record_probe_destination_provenance(capture, endpoint)


def _checkpoint_context():
    return {"framecells": {}, "framefuncs": {}, "cvars": {},
            "hostfns": {}, "participants": {}}


def _checkpoint_dump(value):
    stream = io.BytesIO()
    writer = kernel_checkpoint._StateWriter(stream)
    kernel_checkpoint.dump_to(writer, value, _checkpoint_context())
    writer.finish()
    stored = stream.getvalue()
    return kernel_checkpoint.decode_state(stored, {
        "state_codec": kernel_checkpoint.STATE_CODEC,
        "state_stored_bytes": writer.stored_count,
        "state_stored_sha256": writer.stored_digest.hexdigest(),
        "state_bytes": writer.count,
        "state_sha256": writer.digest.hexdigest(),
    })


def test_request_capture_probe_key_is_absent_from_checkpoint_dump(registered_checkpoint_key):
    capture = _request_capture_with_key(registered_checkpoint_key)

    dumped = _checkpoint_dump({"capture": capture})

    assert registered_checkpoint_key.encode() not in dumped


def test_restored_request_capture_without_recorded_provenance_refuses_probe_by_name(
        registered_checkpoint_key, monkeypatch):
    from kernel_checkpoint import DROP_ATTRS, _Unpickler
    from probes import react_repair

    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "CUSTOM_KEY")
    monkeypatch.setenv("CUSTOM_KEY", registered_checkpoint_key)
    monkeypatch.setenv("OTHER_KEY", "other-run-key")
    capture = _request_capture_with_key(registered_checkpoint_key)
    dumped = _checkpoint_dump({"capture": capture})
    restored = _Unpickler(io.BytesIO(dumped), _checkpoint_context()).load()["capture"]
    endpoint = "http://127.0.0.1:8091/v1"
    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "OTHER_KEY")
    monkeypatch.setattr(react_repair, "API", endpoint)
    monkeypatch.setattr(react_repair, "KEY", "stale-primary-key")
    probe_calls = []
    monkeypatch.setattr(
        react_repair, "_await_endpoint",
        lambda probe_endpoint, probe_key: probe_calls.append((probe_endpoint, probe_key)) or True,
    )

    assert "_provider_probe_destination" in DROP_ATTRS[(
        "kernel_measurement", "RequestCapture")]
    assert "_provider_probe_destination" not in vars(restored)
    assert "_provider_probe_provenance" not in vars(restored)
    assert react_repair._await_failed_request_destination(restored) is False
    assert probe_calls == []
    assert "_provider_probe_provenance" not in vars(restored)
    assert restored.journal.events[-1][0] == "provider_probe_skipped"
    assert restored.journal.events[-1][1]["reason"] == "probe_provenance_missing"


def test_restored_routed_capture_uses_recorded_live_provenance(
        registered_checkpoint_key, monkeypatch, model_router_test_config):
    from kernel_frame_routes import Route
    from kernel_checkpoint import _Unpickler
    from probes import react_repair

    capture = _request_capture_with_key(registered_checkpoint_key)
    route_base = "http://127.0.0.1:8092/v1"
    endpoint = route_base + "/chat/completions"
    capture.frame_route = Route(
        role="read", scope="run", unit_class=None,
        settings={
            "endpoint": route_base, "model": "route-model",
            "api_key_env": "STATEFRAME_ROUTE_TEST_KEY",
        },
    )
    monkeypatch.setattr(react_repair._frame_routes, "ROUTES_ACTIVE", True)
    _prepare_capture_with_model_router(
        capture, model_router_test_config, mode="off",
        endpoint="http://127.0.0.1:8093/v1", key="default-route-key",
    )
    _record_physical_send_provenance(capture, endpoint)
    dumped = _checkpoint_dump({"capture": capture})
    restored = _Unpickler(io.BytesIO(dumped), _checkpoint_context()).load()["capture"]
    monkeypatch.setattr(react_repair._frame_routes, "ROUTES_ACTIVE", False)
    monkeypatch.setattr(react_repair, "API", "http://127.0.0.1:8093/v1")
    monkeypatch.setattr(react_repair, "KEY", "default-route-key")
    monkeypatch.setenv("STATEFRAME_ROUTE_TEST_KEY", registered_checkpoint_key)

    assert react_repair._failed_request_destination(restored) == (
        endpoint, registered_checkpoint_key)


def test_restored_frame_route_probe_uses_rotated_live_credential(
        registered_checkpoint_key, monkeypatch, model_router_test_config):
    from kernel_checkpoint import _Unpickler
    from kernel_frame_routes import Route
    from probes import react_repair

    capture = _request_capture_with_key(registered_checkpoint_key)
    route_base = "http://127.0.0.1:8099/v1"
    endpoint = route_base + "/chat/completions"
    capture.frame_route = Route(
        role="read", scope="run", unit_class=None,
        settings={
            "endpoint": route_base, "model": "route-model",
            "api_key_env": "STATEFRAME_ROUTE_TEST_KEY",
        },
    )
    monkeypatch.setattr(react_repair._frame_routes, "ROUTES_ACTIVE", True)
    monkeypatch.setenv("STATEFRAME_ROUTE_TEST_KEY", registered_checkpoint_key)
    _prepare_capture_with_model_router(
        capture, model_router_test_config, mode="off",
        endpoint="http://127.0.0.1:8093/v1", key="primary-key-before-restore",
    )
    capture._provider_probe_destination = (endpoint, registered_checkpoint_key)
    _record_physical_send_provenance(capture, endpoint)
    dumped = _checkpoint_dump({"capture": capture})
    restored = _Unpickler(io.BytesIO(dumped), _checkpoint_context()).load()["capture"]

    rotated_key = "rotated-frame-route-key"
    monkeypatch.setenv("STATEFRAME_ROUTE_TEST_KEY", rotated_key)
    monkeypatch.setattr(react_repair._frame_routes, "ROUTES_ACTIVE", False)
    monkeypatch.setattr(react_repair, "API", "http://127.0.0.1:8996/v1")
    monkeypatch.setattr(react_repair, "KEY", "wrong-primary-key")

    assert react_repair._failed_request_destination(restored) == (endpoint, rotated_key)


def test_restored_frame_route_probe_skips_when_live_credential_is_unset(
        registered_checkpoint_key, monkeypatch, model_router_test_config):
    from kernel_checkpoint import _Unpickler
    from kernel_frame_routes import Route
    from probes import react_repair

    capture = _request_capture_with_key(registered_checkpoint_key)
    route_base = "http://127.0.0.1:8100/v1"
    endpoint = route_base + "/chat/completions"
    capture.frame_route = Route(
        role="read", scope="run", unit_class=None,
        settings={
            "endpoint": route_base, "model": "route-model",
            "api_key_env": "STATEFRAME_ROUTE_TEST_KEY",
        },
    )
    monkeypatch.setattr(react_repair._frame_routes, "ROUTES_ACTIVE", True)
    monkeypatch.setenv("STATEFRAME_ROUTE_TEST_KEY", registered_checkpoint_key)
    _prepare_capture_with_model_router(
        capture, model_router_test_config, mode="off",
        endpoint="http://127.0.0.1:8093/v1", key="primary-key-before-restore",
    )
    capture._provider_probe_destination = (endpoint, registered_checkpoint_key)
    _record_physical_send_provenance(capture, endpoint)
    dumped = _checkpoint_dump({"capture": capture})
    restored = _Unpickler(io.BytesIO(dumped), _checkpoint_context()).load()["capture"]

    monkeypatch.delenv("STATEFRAME_ROUTE_TEST_KEY", raising=False)
    monkeypatch.setattr(react_repair._frame_routes, "ROUTES_ACTIVE", False)
    monkeypatch.setattr(react_repair, "API", "http://127.0.0.1:8995/v1")
    monkeypatch.setattr(react_repair, "KEY", "available-primary-key")
    probe_calls = []
    monkeypatch.setattr(
        react_repair, "_await_endpoint",
        lambda probe_endpoint, probe_key: probe_calls.append((probe_endpoint, probe_key)) or True,
    )

    assert react_repair._await_failed_request_destination(restored) is False
    assert probe_calls == []
    assert restored.journal.events[-1][0] == "provider_probe_skipped"
    assert restored.journal.events[-1][1]["reason"] == "missing_live_credential"


def test_restored_primary_probe_uses_rotated_live_credential(
        registered_checkpoint_key, monkeypatch, model_router_test_config):
    from kernel_checkpoint import _Unpickler
    from probes import react_repair

    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "DEEPINFRA_API_KEY")
    endpoint = "http://127.0.0.1:8094/v1"
    monkeypatch.setenv("DEEPINFRA_API_KEY", registered_checkpoint_key)
    capture = _request_capture_with_key(registered_checkpoint_key)
    _prepare_capture_with_model_router(
        capture, model_router_test_config, mode="off", endpoint=endpoint,
        key=registered_checkpoint_key,
    )
    _record_physical_send_provenance(capture, endpoint)
    dumped = _checkpoint_dump({"capture": capture})
    restored = _Unpickler(io.BytesIO(dumped), _checkpoint_context()).load()["capture"]

    rotated_key = "rotated-primary-probe-key"
    monkeypatch.setenv("DEEPINFRA_API_KEY", rotated_key)
    monkeypatch.setattr(react_repair, "API", "http://127.0.0.1:8999/v1")
    monkeypatch.setattr(react_repair, "KEY", registered_checkpoint_key)

    assert react_repair._failed_request_destination(restored) == (endpoint, rotated_key)


def test_restored_primary_probe_skips_when_live_credential_is_unset(
        registered_checkpoint_key, monkeypatch, model_router_test_config):
    from kernel_checkpoint import _Unpickler
    from probes import react_repair

    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "DEEPINFRA_API_KEY")
    endpoint = "http://127.0.0.1:8095/v1"
    monkeypatch.setenv("DEEPINFRA_API_KEY", registered_checkpoint_key)
    capture = _request_capture_with_key(registered_checkpoint_key)
    _prepare_capture_with_model_router(
        capture, model_router_test_config, mode="off", endpoint=endpoint,
        key=registered_checkpoint_key,
    )
    _record_physical_send_provenance(capture, endpoint)
    dumped = _checkpoint_dump({"capture": capture})
    restored = _Unpickler(io.BytesIO(dumped), _checkpoint_context()).load()["capture"]

    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.setattr(react_repair, "API", endpoint)
    monkeypatch.setattr(react_repair, "KEY", registered_checkpoint_key)
    probe_calls = []
    monkeypatch.setattr(
        react_repair, "_await_endpoint",
        lambda probe_endpoint, probe_key: probe_calls.append((probe_endpoint, probe_key)) or True,
    )

    assert react_repair._await_failed_request_destination(restored) is False
    assert probe_calls == []
    assert restored.journal.events[-1][0] == "provider_probe_skipped"
    assert restored.journal.events[-1][1]["reason"] == "missing_live_credential"


def test_model_router_probe_restores_provider_endpoint_and_live_credential(
        registered_checkpoint_key, monkeypatch, model_router_test_config):
    from kernel_checkpoint import _Unpickler
    from probes import react_repair

    monkeypatch.setattr(react_repair._frame_routes, "ROUTES_ACTIVE", False)
    monkeypatch.setenv("STATEFRAME_ROUTER_PROVIDER_KEY", registered_checkpoint_key)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "wrong-primary-provider-key")
    capture = _request_capture_with_key(registered_checkpoint_key)
    prepared = _prepare_capture_with_model_router(
        capture, model_router_test_config, mode="on",
        endpoint="http://127.0.0.1:8096/v1", key="primary-live-key",
    )
    _record_physical_send_provenance(capture, prepared.endpoint)
    dumped = _checkpoint_dump({"capture": capture})
    restored = _Unpickler(io.BytesIO(dumped), _checkpoint_context()).load()["capture"]

    live_router_key = "rotated-router-probe-key"
    monkeypatch.setenv("STATEFRAME_ROUTER_PROVIDER_KEY", live_router_key)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "another-wrong-primary-key")
    monkeypatch.setattr(react_repair, "API", "http://127.0.0.1:8998/v1")
    monkeypatch.setattr(react_repair, "KEY", "wrong-primary-key")

    assert prepared.endpoint == "https://router-provider.example/v1/chat/completions"
    assert react_repair._failed_request_destination(restored) == (
        prepared.endpoint, live_router_key)


def test_physical_send_provenance_tracks_model_router_over_active_frame_route(
        registered_checkpoint_key, monkeypatch, model_router_test_config):
    from kernel_frame_routes import Route
    from probes import react_repair

    monkeypatch.setattr(react_repair._frame_routes, "ROUTES_ACTIVE", True)
    monkeypatch.setenv("STATEFRAME_ROUTER_PROVIDER_KEY", registered_checkpoint_key)
    capture = _request_capture_with_key(registered_checkpoint_key)
    route_base = "http://127.0.0.1:8090/v1"
    capture.frame_route = Route(
        role="read", scope="run", unit_class=None,
        settings={
            "endpoint": route_base, "model": "route-model",
            "api_key_env": "STATEFRAME_ROUTE_TEST_KEY",
        },
    )
    prepared = _prepare_capture_with_model_router(
        capture, model_router_test_config, mode="on",
        endpoint=route_base, key="primary-live-key",
    )

    _record_physical_send_provenance(capture, prepared.endpoint)

    assert capture._provider_probe_provenance["source"] == "model_router"
    assert capture._provider_probe_provenance["endpoint"] == prepared.endpoint
    assert capture._provider_probe_provenance["credential_env"] == (
        "STATEFRAME_ROUTER_PROVIDER_KEY")


def test_restored_model_router_probe_fails_closed_when_router_cannot_be_resolved(
        registered_checkpoint_key, monkeypatch, model_router_test_config):
    from kernel_checkpoint import _Unpickler
    from probes import react_repair

    monkeypatch.setattr(react_repair._frame_routes, "ROUTES_ACTIVE", False)
    monkeypatch.setenv("STATEFRAME_ROUTER_PROVIDER_KEY", registered_checkpoint_key)
    capture = _request_capture_with_key(registered_checkpoint_key)
    prepared = _prepare_capture_with_model_router(
        capture, model_router_test_config, mode="on",
        endpoint="http://127.0.0.1:8098/v1", key="primary-live-key",
    )
    _record_physical_send_provenance(capture, prepared.endpoint)
    dumped = _checkpoint_dump({"capture": capture})
    restored = _Unpickler(io.BytesIO(dumped), _checkpoint_context()).load()["capture"]
    del restored.model_router_decision

    monkeypatch.setenv("DEEPINFRA_API_KEY", "available-primary-key")
    monkeypatch.setattr(react_repair, "API", "http://127.0.0.1:8997/v1")
    monkeypatch.setattr(react_repair, "KEY", "available-primary-key")
    probe_calls = []
    monkeypatch.setattr(
        react_repair, "_await_endpoint",
        lambda probe_endpoint, probe_key: probe_calls.append((probe_endpoint, probe_key)) or True,
    )

    assert react_repair._await_failed_request_destination(restored) is False
    assert probe_calls == []
    assert restored.journal.events[-1][0] == "provider_probe_skipped"
    assert restored.journal.events[-1][1]["reason"] == "router_destination_unresolved"


def _undropped_secret_fields(owners, key):
    secret = key.encode()

    def contains(value, seen):
        if isinstance(value, str):
            return key in value
        if isinstance(value, (bytes, bytearray)):
            return secret in value
        if id(value) in seen:
            return False
        seen.add(id(value))
        if isinstance(value, dict):
            return any(contains(k, seen) or contains(v, seen)
                       for k, v in value.items())
        if isinstance(value, (tuple, list, set, frozenset)):
            return any(contains(item, seen) for item in value)
        if isinstance(value, types.ModuleType) or isinstance(value, type):
            return False
        if isinstance(value, types.MethodType):
            return (contains(value.__self__, seen)
                    or contains(value.__func__, seen))
        if isinstance(value, types.FunctionType):
            for cell in value.__closure__ or ():
                try:
                    if contains(cell.cell_contents, seen):
                        return True
                except ValueError:  # an empty closure cell has no value to inspect
                    continue
        try:
            attributes = vars(value)
        except TypeError:
            attributes = {}
        if any(contains(item, seen) for item in attributes.values()):
            return True
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            try:
                item = object.__getattribute__(value, name)
            except (AttributeError, TypeError):
                continue
            if contains(item, seen):
                return True
        return False

    paths = []
    for owner in owners:
        owner_key = (type(owner).__module__, type(owner).__qualname__)
        dropped = kernel_checkpoint.DROP_ATTRS.get(owner_key, frozenset())
        for name, value in vars(owner).items():
            if contains(value, set()) and name not in dropped:
                paths.append(f"{owner_key[1]}.{name}")
    return paths


def test_capture_and_meter_registered_secrets_are_checkpoint_dropped(
        registered_checkpoint_key):
    from kernel_request_diet import RequestTokenMeter

    capture = _request_capture_with_key(registered_checkpoint_key)
    meter = RequestTokenMeter(endpoint="http://127.0.0.1:8080/tokenize")
    meter.bind_tokenizer_auth(registered_checkpoint_key, "http://127.0.0.1:8080/v1")

    assert _undropped_secret_fields((capture, meter), registered_checkpoint_key) == []
    capture._future_secret_field = SimpleNamespace(credential=registered_checkpoint_key)
    assert _undropped_secret_fields((capture, meter), registered_checkpoint_key) == [
        "RequestCapture._future_secret_field"]
    del capture._future_secret_field

    def tokenizer_factory(secret):
        def tokenizer(_text):
            return secret
        return tokenizer

    meter._tokenizer = tokenizer_factory(registered_checkpoint_key)
    assert _undropped_secret_fields((capture, meter), registered_checkpoint_key) == [
        "RequestTokenMeter._tokenizer"]


def test_checkpoint_secret_guard_still_rejects_unfiltered_values(registered_checkpoint_key):
    with pytest.raises(kernel_checkpoint.CheckpointError,
                       match="a registered secret would be written"):
        _checkpoint_dump({"unfiltered": registered_checkpoint_key})


@pytest.mark.skipif(not hasattr(os, "fork"), reason="async writer requires fork")
def test_parent_writer_warning_includes_redacted_child_error(
        tmp_path, registered_checkpoint_key, capsys):
    old_state = kernel_checkpoint._STATE.copy()
    run = tmp_path / "run"
    pending = run / "checkpoints" / ".0001-1.tmp"
    run.mkdir()
    pending.mkdir(parents=True)
    kernel_checkpoint._STATE.clear()
    kernel_checkpoint._STATE.update(failed=2, consecutive_failed=2, disabled=False)

    def fail_write():
        raise kernel_checkpoint.CheckpointError(
            f"refused credential {registered_checkpoint_key}")

    try:
        kernel_checkpoint._fork_writer(
            run, 1, run / "checkpoints" / "0001-1", fail_write, pending,
        )
        deadline = time.monotonic() + 10
        while not kernel_checkpoint._reap():
            assert time.monotonic() < deadline, "background writer was not reaped"
            time.sleep(0.01)
        warning = capsys.readouterr().err
        assert "checkpoint refused" in warning
        assert registered_checkpoint_key not in warning
        events = [json.loads(line) for line in
                  (run / "checkpoints/events.jsonl").read_text().splitlines()]
        event = next(row for row in reversed(events)
                     if row.get("event") == "checkpoint_failed")
        assert event["failure_reason"] == "checkpoint refused"
        assert registered_checkpoint_key not in event["error"]
    finally:
        kernel_checkpoint._STATE.clear()
        kernel_checkpoint._STATE.update(old_state)


def _inherited_primary_router_config():
    from kernel_stateframe_model_router import EndpointConfig, ModelRouterConfig

    return ModelRouterConfig(
        endpoints=(EndpointConfig(
            name="router-inherits-primary",
            model_id="test/router-inherits-primary",
            role="moe",
            url="https://inherited-router.example/v1/chat/completions",
            inherit_api_key=True,
        ),),
        default_endpoint="router-inherits-primary",
        lease_actions=2,
    )


def _prepare_capture_with_inherited_router(capture, key):
    from kernel_stateframe_model_router import ModelRouter

    incoming_endpoint = "http://127.0.0.1:8110/v1/chat/completions"
    router = ModelRouter(
        mode="on", config=_inherited_primary_router_config(),
        base_model="primary-model", replay_enabled=False,
    )
    prepared = router.prepare_call(capture, _BODY, incoming_endpoint, key)
    capture._provider_probe_destination = (prepared.endpoint, prepared.key)
    return prepared


def _restore_request_capture(capture):
    from kernel_checkpoint import _Unpickler

    dumped = _checkpoint_dump({"capture": capture})
    restored = _Unpickler(io.BytesIO(dumped), _checkpoint_context()).load()["capture"]
    return restored, dumped


def _custom_primary_capture(key, monkeypatch, endpoint):
    from kernel_checkpoint import _Unpickler

    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "CUSTOM_KEY")
    capture = _request_capture_with_key(key)
    _prepare_capture_with_model_router(
        capture, _inherited_primary_router_config(), mode="off",
        endpoint=endpoint, key=key,
    )
    _record_physical_send_provenance(capture, endpoint)
    dumped = _checkpoint_dump({"capture": capture})
    restored = _Unpickler(io.BytesIO(dumped), _checkpoint_context()).load()["capture"]
    return restored, dumped


def test_child_carries_primary_api_key_env_name_without_value(monkeypatch):
    from stateframe import _child

    monkeypatch.setenv("CUSTOM_KEY", "child-primary-secret")
    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "stale-name")

    _child._install_primary_api_key_env({"api_key_env": "CUSTOM_KEY"})
    assert os.environ["STATEFRAME_PRIMARY_API_KEY_ENV"] == "CUSTOM_KEY"
    assert "child-primary-secret" not in os.environ["STATEFRAME_PRIMARY_API_KEY_ENV"]

    _child._install_primary_api_key_env({})
    assert "STATEFRAME_PRIMARY_API_KEY_ENV" not in os.environ


def test_restored_primary_probe_uses_custom_env_name_and_rotated_live_key(
        registered_checkpoint_key, monkeypatch):
    from probes import react_repair

    monkeypatch.setenv("CUSTOM_KEY", registered_checkpoint_key)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "wrong-primary-provider-key")
    endpoint = "http://127.0.0.1:8111/v1"
    restored, dumped = _custom_primary_capture(
        registered_checkpoint_key, monkeypatch, endpoint,
    )
    monkeypatch.setenv("CUSTOM_KEY", "rotated-custom-primary-key")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "rotated-wrong-primary-key")
    monkeypatch.setattr(react_repair, "API", "http://127.0.0.1:8991/v1")
    monkeypatch.setattr(react_repair, "KEY", "stale-module-key")

    assert restored._provider_probe_provenance["credential_env"] == "CUSTOM_KEY"
    assert registered_checkpoint_key.encode() not in dumped
    assert react_repair._failed_request_destination(restored) == (
        endpoint, "rotated-custom-primary-key")


def test_restored_primary_probe_skips_unset_custom_env_even_with_default_key_present(
        registered_checkpoint_key, monkeypatch):
    from probes import react_repair

    monkeypatch.setenv("CUSTOM_KEY", registered_checkpoint_key)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "wrong-primary-provider-key")
    restored, _dumped = _custom_primary_capture(
        registered_checkpoint_key, monkeypatch, "http://127.0.0.1:8112/v1",
    )
    monkeypatch.delenv("CUSTOM_KEY", raising=False)
    probe_calls = []
    monkeypatch.setattr(
        react_repair, "_await_endpoint",
        lambda probe_endpoint, probe_key: probe_calls.append((probe_endpoint, probe_key)) or True,
    )

    assert react_repair._await_failed_request_destination(restored) is False
    assert probe_calls == []
    assert restored.journal.events[-1][0] == "provider_probe_skipped"
    assert restored.journal.events[-1][1]["reason"] == "missing_live_credential"


def test_restored_primary_probe_refuses_name_mismatched_to_current_run(
        registered_checkpoint_key, monkeypatch):
    from probes import react_repair

    monkeypatch.setenv("CUSTOM_KEY", registered_checkpoint_key)
    monkeypatch.setenv("OTHER_KEY", "other-run-key")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "wrong-primary-provider-key")
    restored, _dumped = _custom_primary_capture(
        registered_checkpoint_key, monkeypatch, "http://127.0.0.1:8113/v1",
    )
    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "OTHER_KEY")
    probe_calls = []
    monkeypatch.setattr(
        react_repair, "_await_endpoint",
        lambda probe_endpoint, probe_key: probe_calls.append((probe_endpoint, probe_key)) or True,
    )

    assert react_repair._await_failed_request_destination(restored) is False
    assert probe_calls == []
    assert restored.journal.events[-1][0] == "provider_probe_skipped"
    assert restored.journal.events[-1][1]["reason"] == "primary_credential_env_mismatch"


def test_restored_primary_probe_refuses_when_run_env_name_is_unavailable(
        registered_checkpoint_key, monkeypatch):
    from probes import react_repair

    monkeypatch.setenv("CUSTOM_KEY", registered_checkpoint_key)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "wrong-primary-provider-key")
    restored, _dumped = _custom_primary_capture(
        registered_checkpoint_key, monkeypatch, "http://127.0.0.1:8115/v1",
    )
    monkeypatch.delenv("STATEFRAME_PRIMARY_API_KEY_ENV", raising=False)
    probe_calls = []
    monkeypatch.setattr(
        react_repair, "_await_endpoint",
        lambda probe_endpoint, probe_key: probe_calls.append((probe_endpoint, probe_key)) or True,
    )

    assert react_repair._await_failed_request_destination(restored) is False
    assert probe_calls == []
    assert restored.journal.events[-1][0] == "provider_probe_skipped"
    assert restored.journal.events[-1][1]["reason"] == (
        "primary_run_credential_env_unavailable")


def test_restored_primary_probe_refuses_invalid_recorded_env_name(
        registered_checkpoint_key, monkeypatch):
    from probes import react_repair

    monkeypatch.setenv("CUSTOM_KEY", registered_checkpoint_key)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "wrong-primary-provider-key")
    restored, _dumped = _custom_primary_capture(
        registered_checkpoint_key, monkeypatch, "http://127.0.0.1:8114/v1",
    )
    restored._provider_probe_provenance["credential_env"] = "CUSTOM-KEY"
    probe_calls = []
    monkeypatch.setattr(
        react_repair, "_await_endpoint",
        lambda probe_endpoint, probe_key: probe_calls.append((probe_endpoint, probe_key)) or True,
    )

    assert react_repair._await_failed_request_destination(restored) is False
    assert probe_calls == []
    assert restored.journal.events[-1][0] == "provider_probe_skipped"
    assert restored.journal.events[-1][1]["reason"] == "invalid_credential_env_name"


def test_restored_primary_probe_refuses_missing_recorded_env_name(
        registered_checkpoint_key, monkeypatch):
    from probes import react_repair

    monkeypatch.setenv("CUSTOM_KEY", registered_checkpoint_key)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "wrong-primary-provider-key")
    restored, _dumped = _custom_primary_capture(
        registered_checkpoint_key, monkeypatch, "http://127.0.0.1:8116/v1",
    )
    restored._provider_probe_provenance["credential_env"] = None
    probe_calls = []
    monkeypatch.setattr(
        react_repair, "_await_endpoint",
        lambda probe_endpoint, probe_key: probe_calls.append((probe_endpoint, probe_key)) or True,
    )

    assert react_repair._await_failed_request_destination(restored) is False
    assert probe_calls == []
    assert restored.journal.events[-1][0] == "provider_probe_skipped"
    assert restored.journal.events[-1][1]["reason"] == "primary_credential_env_missing"


def test_restored_inherited_router_probe_without_recorded_provenance_refuses(
        registered_checkpoint_key, monkeypatch):
    from kernel_checkpoint import _Unpickler
    from probes import react_repair

    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "CUSTOM_KEY")
    monkeypatch.setenv("CUSTOM_KEY", registered_checkpoint_key)
    monkeypatch.setenv("OTHER_KEY", "other-run-key")
    capture = _request_capture_with_key(registered_checkpoint_key)
    _prepare_capture_with_inherited_router(capture, registered_checkpoint_key)
    capture.__dict__.pop("_provider_probe_provenance", None)
    dumped = _checkpoint_dump({"capture": capture})
    restored = _Unpickler(io.BytesIO(dumped), _checkpoint_context()).load()["capture"]
    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "OTHER_KEY")
    probe_calls = []
    monkeypatch.setattr(
        react_repair, "_await_endpoint",
        lambda probe_endpoint, probe_key: probe_calls.append((probe_endpoint, probe_key)) or True,
    )

    assert react_repair._await_failed_request_destination(restored) is False
    assert probe_calls == []
    assert "_provider_probe_provenance" not in vars(restored)
    assert restored.journal.events[-1][0] == "provider_probe_skipped"
    assert restored.journal.events[-1][1]["reason"] == "probe_provenance_missing"


def test_model_router_probe_provenance_is_credential_free_in_checkpoint(
        registered_checkpoint_key, monkeypatch, model_router_test_config):
    from kernel_checkpoint import _Unpickler

    monkeypatch.setenv("STATEFRAME_ROUTER_PROVIDER_KEY", registered_checkpoint_key)
    capture = _request_capture_with_key(registered_checkpoint_key)
    _prepare_capture_with_model_router(
        capture, model_router_test_config, mode="on",
        endpoint="http://127.0.0.1:8097/v1", key=registered_checkpoint_key,
    )

    provenance = capture._provider_probe_provenance
    dumped = _checkpoint_dump({"capture": capture})
    restored = _Unpickler(io.BytesIO(dumped), _checkpoint_context()).load()["capture"]

    assert provenance["source"] == "model_router"
    assert provenance["provider_id"] == "router-moe"
    assert provenance["credential_env"] == "STATEFRAME_ROUTER_PROVIDER_KEY"
    assert registered_checkpoint_key.encode() not in dumped
    assert restored._provider_probe_provenance == provenance


def test_restored_inherited_router_probe_uses_recorded_custom_primary_env_and_rotation(
        registered_checkpoint_key, monkeypatch):
    from probes import react_repair

    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "CUSTOM_KEY")
    monkeypatch.setenv("CUSTOM_KEY", registered_checkpoint_key)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "wrong-primary-provider-key")
    capture = _request_capture_with_key(registered_checkpoint_key)
    prepared = _prepare_capture_with_inherited_router(capture, registered_checkpoint_key)
    restored, dumped = _restore_request_capture(capture)
    monkeypatch.setenv("CUSTOM_KEY", "rotated-inherited-primary-key")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "rotated-wrong-primary-key")
    monkeypatch.setattr(react_repair, "API", "http://127.0.0.1:8992/v1")
    monkeypatch.setattr(react_repair, "KEY", "stale-module-key")

    assert capture._provider_probe_provenance["credential_env"] == "CUSTOM_KEY"
    assert restored._provider_probe_provenance["credential_env"] == "CUSTOM_KEY"
    assert registered_checkpoint_key.encode() not in dumped
    assert react_repair._failed_request_destination(restored) == (
        prepared.endpoint, "rotated-inherited-primary-key")


def test_restored_inherited_router_probe_refuses_unset_recorded_primary_key(
        registered_checkpoint_key, monkeypatch):
    from probes import react_repair

    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "CUSTOM_KEY")
    monkeypatch.setenv("CUSTOM_KEY", registered_checkpoint_key)
    capture = _request_capture_with_key(registered_checkpoint_key)
    _prepare_capture_with_inherited_router(capture, registered_checkpoint_key)
    restored, _dumped = _restore_request_capture(capture)
    monkeypatch.delenv("CUSTOM_KEY", raising=False)
    probe_calls = []
    monkeypatch.setattr(
        react_repair, "_await_endpoint",
        lambda probe_endpoint, probe_key: probe_calls.append((probe_endpoint, probe_key)) or True,
    )

    assert react_repair._await_failed_request_destination(restored) is False
    assert probe_calls == []
    assert restored.journal.events[-1][0] == "provider_probe_skipped"
    assert restored.journal.events[-1][1]["reason"] == "missing_live_credential"


def test_restored_inherited_router_probe_refuses_primary_env_name_mismatch(
        registered_checkpoint_key, monkeypatch):
    from probes import react_repair

    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "CUSTOM_KEY")
    monkeypatch.setenv("CUSTOM_KEY", registered_checkpoint_key)
    monkeypatch.setenv("OTHER_KEY", "other-run-key")
    capture = _request_capture_with_key(registered_checkpoint_key)
    _prepare_capture_with_inherited_router(capture, registered_checkpoint_key)
    restored, _dumped = _restore_request_capture(capture)
    monkeypatch.setenv("STATEFRAME_PRIMARY_API_KEY_ENV", "OTHER_KEY")
    probe_calls = []
    monkeypatch.setattr(
        react_repair, "_await_endpoint",
        lambda probe_endpoint, probe_key: probe_calls.append((probe_endpoint, probe_key)) or True,
    )

    assert react_repair._await_failed_request_destination(restored) is False
    assert probe_calls == []
    assert restored.journal.events[-1][0] == "provider_probe_skipped"
    assert restored.journal.events[-1][1]["reason"] == (
        "router_primary_credential_env_mismatch")


def test_probe_skip_journal_failure_is_logged_without_exception_secrets(caplog):
    from probes import react_repair

    def broken_emit(*_args, **_kwargs):
        raise RuntimeError("private-provider-key-must-not-escape")

    capture = SimpleNamespace(journal=SimpleNamespace(emit=broken_emit), request_id="probe")
    react_repair._record_probe_destination_skip(capture, "missing_live_credential")
    assert capture._provider_probe_unavailable_reason == "missing_live_credential"
    assert "provider probe skip event could not be recorded" in caplog.text
    assert "private-provider-key-must-not-escape" not in caplog.text
