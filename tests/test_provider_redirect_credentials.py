"""Provider redirects stay on their original HTTP origin."""
from __future__ import annotations

import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
from types import SimpleNamespace

import pytest

from stateframe import _child, endpoint
from stateframe.engine import (
    kernel_bounded_http, kernel_measurement, kernel_plain_http, kernel_redirect_policy,
)
import kernel_request_diet


_KEY = "provider-secret-6b2c"
ROOT = Path(__file__).resolve().parents[1]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        path = self.path.split("?", 1)[0]
        self.server.state["requests"].append({
            "server": self.server.label,
            "path": self.path,
            "method": self.command,
            "body": body,
            "authorization": self.headers.get("Authorization"),
            "x-api-key": self.headers.get("X-Api-Key"),
            "api-key": self.headers.get("Api-Key"),
            "cookie": self.headers.get("Cookie"),
            "custom": self.headers.get("X-Custom-Credential"),
        })
        if path in self.server.state.get("redirects", {}):
            location = self.server.state["redirects"][path]
            status = self.server.state.get("status", 302)
        elif self.path.startswith("/cross"):
            location = self.server.state["target"] + "/received?api_key=" + _KEY
            status = self.server.state.get("status", 302)
        elif self.path.startswith("/userinfo"):
            location = self.server.state["target"].replace(
                "http://", f"http://user:{_KEY}@", 1) + f"/private/path?key={_KEY}"
            status = 302
        elif self.path == "/chain/start":
            location = self.server.state["source"] + "/chain/middle"
            status = 302
        elif self.path == "/chain/middle":
            location = self.server.state["target"] + "/received?key=" + _KEY
            status = 302
        elif self.path == "/protocol/start":
            location = "//" + self.server.state["target"].removeprefix("http://") + "/received"
            status = 302
        elif self.path == "/same/start":
            location = self.server.state["source"] + "/same/final"
            status = 302
        elif self.path == "/dot/start":
            location = self.server.state["dot_target"]
            status = 302
        elif self.path == "/relative/start":
            location = "/relative/final"
            status = 302
        else:
            location = None
            status = 200
        body = self.server.state.get("success_body", b"ok")
        self.send_response(status)
        if location is not None:
            self.send_header("Location", location)
            body = b""
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    do_GET = _handle
    do_POST = _handle


class _Servers:
    def __enter__(self):
        self.state = {"requests": []}
        self.source = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.target = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.source.label = "source"
        self.target.label = "target"
        self.source.state = self.target.state = self.state
        self.state["source"] = f"http://127.0.0.1:{self.source.server_port}"
        self.state["target"] = f"http://127.0.0.1:{self.target.server_port}"
        self.threads = [threading.Thread(target=server.serve_forever, daemon=True)
                        for server in (self.source, self.target)]
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *_args) -> None:
        for server in (self.source, self.target):
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=5)
            assert not thread.is_alive()


def _opener_cases():
    yield "direct provider opener", lambda: kernel_redirect_policy.build_credential_scoped_opener(
        proxy_mode="direct")
    yield "endpoint probe opener", lambda: kernel_redirect_policy.build_credential_scoped_opener(
        proxy_mode="direct", handlers=(endpoint._LocalRedirects(allow_remote=True),))
    yield "routed provider opener", lambda: kernel_redirect_policy.build_credential_scoped_opener(
        proxy_mode="direct", handlers=(kernel_bounded_http._RoutedRedirects(allow_remote=True),))
    yield "tokenizer opener", kernel_request_diet._build_tokenize_opener

    def plain_opener():
        transport = kernel_plain_http.AbortiveHTTPTransport()
        tracker = transport.begin()
        return transport.opener(tracker), lambda: (transport.finish(tracker), transport.release(tracker))

    def bounded_opener():
        tracker = kernel_bounded_http._AbortiveConnectionTracker()
        tracker.request_deadline = time.monotonic() + 20
        tracker.hard_deadline = tracker.request_deadline
        tracker.socket_timeout = 5
        tracker.send_stall_seconds = 5
        opener = kernel_redirect_policy.build_credential_scoped_opener(
            proxy_mode="direct", handlers=(
                kernel_bounded_http._AbortiveHTTPHandler(tracker),
                kernel_bounded_http._AbortiveHTTPSHandler(tracker),
            ))
        return opener, tracker.finish_all

    def wrap(factory):
        def create():
            value = factory()
            if isinstance(value, tuple):
                return value
            return value, lambda: None
        return create

    yield "plain provider opener", wrap(plain_opener)
    yield "bounded chat opener", bounded_opener


def _no_environment_proxy(monkeypatch) -> None:
    for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setattr(urllib.request, "_opener", None)


@pytest.mark.parametrize("status", (302, 307))
@pytest.mark.parametrize("method", ("GET", "POST"))
def test_provider_urlopen_refuses_cross_origin_redirect_before_contact(
        monkeypatch, status, method) -> None:
    _no_environment_proxy(monkeypatch)
    with _Servers() as servers:
        servers.state["status"] = status
        request = urllib.request.Request(
            servers.state["source"] + "/cross/start", data=b"provider body" if method == "POST" else None,
            headers={"Authorization": "Bearer " + _KEY})
        with pytest.raises(kernel_redirect_policy.ProviderRedirectRefused) as caught:
            kernel_redirect_policy.provider_urlopen(request, timeout=2)
        assert [row["method"] for row in servers.state["requests"] if row["server"] == "source"] == [method]
        assert not any(row["server"] == "target" for row in servers.state["requests"])
        assert _KEY not in str(caught.value) and _KEY not in repr(caught.value)


def test_provider_urlopen_keeps_same_origin_redirect_and_key(monkeypatch) -> None:
    _no_environment_proxy(monkeypatch)
    with _Servers() as servers:
        request = urllib.request.Request(
            servers.state["source"] + "/same/start", headers={"Authorization": "Bearer " + _KEY})
        with kernel_redirect_policy.provider_urlopen(request, timeout=2) as response:
            assert response.read() == b"ok"
        final = next(row for row in servers.state["requests"] if row["path"] == "/same/final")
        assert final["authorization"] == "Bearer " + _KEY
        assert not any(row["server"] == "target" for row in servers.state["requests"])


def test_provider_urlopen_honors_injected_urlopen(monkeypatch) -> None:
    calls = []

    def injected(request, *, timeout):
        calls.append((request, timeout))
        return object()

    monkeypatch.setattr(urllib.request, "urlopen", injected)
    request = urllib.request.Request("http://127.0.0.1:1/provider")
    assert kernel_redirect_policy.provider_urlopen(request, timeout=7) is not None
    assert calls == [(request, 7)]


def test_provider_urlopen_ignores_foreign_installed_opener(monkeypatch) -> None:
    _no_environment_proxy(monkeypatch)
    monkeypatch.setattr(urllib.request, "_opener", urllib.request.build_opener())
    with _Servers() as servers:
        servers.state["redirects"] = {
            "/foreign/start": servers.state["target"] + "/received",
        }
        request = urllib.request.Request(
            servers.state["source"] + "/foreign/start", headers={"X-Api-Key": _KEY})
        try:
            with kernel_redirect_policy.provider_urlopen(request, timeout=2) as response:
                response.read()
        except kernel_redirect_policy.ProviderRedirectRefused:
            pass
        assert [row["path"] for row in servers.state["requests"]
                if row["server"] == "source"] == ["/foreign/start"]
        assert not any(row["server"] == "target" for row in servers.state["requests"])


_URL_OPEN_SITES = (
    "decode-depth slots", "long-horizon slots", "endpoint models", "endpoint slots",
    "provider epoch", "template sidecar", "detreason model", "fixer model",
    "swarm model", "KV slot action", "react ask_tools", "react ask",
)


def _call_urlopen_site(caller, servers, monkeypatch):
    source = servers.state["source"]
    if caller == "decode-depth slots":
        from stateframe.engine import decode_depth_per_arm as module
        monkeypatch.setattr(module, "enabled", lambda: True)
        return module.observe_slots(SimpleNamespace(), source, key=_KEY)
    if caller == "long-horizon slots":
        from stateframe.engine import long_horizon_decode_logging as module
        monkeypatch.setattr(module, "enabled", lambda: True)
        return module.observe_slots(SimpleNamespace(), source, key=_KEY)
    if caller in {"endpoint models", "endpoint slots", "provider epoch"}:
        monkeypatch.setenv("STATEFRAME_REDIRECT_TEST_KEY", _KEY)
        selected = endpoint.Endpoint(source, "local-model", "STATEFRAME_REDIRECT_TEST_KEY", 2)
        if caller == "endpoint models":
            return selected.probe()
        if caller == "endpoint slots":
            return selected.get_slots()
        from stateframe import send1_probe_cache
        import stateframe
        monkeypatch.setattr(stateframe, "_startup_optimization_enabled", lambda: True)
        monkeypatch.setattr(send1_probe_cache, "host_fingerprint", lambda: "test-host")
        return selected.provider_epoch("read")
    if caller == "template sidecar":
        from stateframe.engine import kernel_template_conformance as module
        return module._post_json(source + "/cross/start", {"model": "local"}, _KEY,
                                 deadline=time.monotonic() + 5)
    if caller == "detreason model":
        import kernel_launch_identity_cost as identity
        import kernel_provider_dialect as dialect
        monkeypatch.setattr(dialect, "capabilities_for",
                            lambda *_args, **_kwargs: SimpleNamespace(dialect="llama.cpp"))
        monkeypatch.setattr(identity, "reasoning_profile", lambda: "default")
        monkeypatch.setattr(identity, "enforce_provider_body", lambda body, **_kwargs: body)
        from detreason.kernel import LlamaSensor
        return LlamaSensor(source + "/cross/start", api_key=_KEY, margin_probe=False)._call("p", 1)
    if caller == "fixer model":
        import kernel_launch_identity_cost as identity
        import kernel_provider_dialect as dialect
        monkeypatch.setenv("LLM_API_KEY", _KEY)
        monkeypatch.setattr(dialect, "capabilities_for",
                            lambda *_args, **_kwargs: SimpleNamespace(dialect="llama.cpp"))
        monkeypatch.setattr(identity, "reasoning_profile", lambda: "default")
        monkeypatch.setattr(identity, "enforce_provider_body", lambda body, **_kwargs: body)
        import fixer
        return fixer._post({"messages": []}, source + "/cross/start", timeout=2)
    if caller == "swarm model":
        from swarm.client2 import ask
        return ask("p", url=source + "/cross/start", timeout=2, api_key=_KEY)
    if caller in {"react ask_tools", "react ask"}:
        import kernel_provider_dialect as dialect
        from probes import react_repair as module
        monkeypatch.setattr(dialect, "enforce_selected_endpoint_body",
                            lambda body, *_args, **_kwargs: body)
        monkeypatch.setattr(module, "API", source + "/cross/start")
        monkeypatch.setattr(module, "KEY", _KEY)
        monkeypatch.setattr(module, "MODEL", "local-model")
        messages = [{"role": "user", "content": "Say ok."}]
        if caller == "react ask_tools":
            return module.ask_tools(messages, mt=128, temp=0, tools=[])
        return module.ask(messages, mt=128, temp=0)
    if caller == "KV slot action":
        import kernel_provider_dialect as dialect
        monkeypatch.setattr(dialect, "require_feature", lambda *_args, **_kwargs: None)
        from stateframe.engine import kernel_kv_chunk_reuse as module
        return module.slot_action(source, 0, "save", "state.bin", _KEY, timeout=2,
                                  capabilities=object())
    raise AssertionError(caller)


@pytest.mark.parametrize("caller", _URL_OPEN_SITES)
@pytest.mark.parametrize("status", (302, 307))
def test_each_urlopen_call_site_refuses_cross_origin(monkeypatch, caller, status) -> None:
    _no_environment_proxy(monkeypatch)
    with _Servers() as servers:
        servers.state["status"] = status
        path = {"decode-depth slots": "/slots", "long-horizon slots": "/slots",
                "endpoint models": "/v1/models", "endpoint slots": "/slots",
                "provider epoch": "/v1/provider_epoch", "template sidecar": "/cross/start",
                "detreason model": "/cross/start", "fixer model": "/cross/start",
                "swarm model": "/cross/start", "KV slot action": "/slots/0",
                "react ask_tools": "/cross/start", "react ask": "/cross/start"}[caller]
        servers.state["redirects"] = {
            path: servers.state["target"] + "/received?api_key=" + _KEY}
        error = None
        result = None
        try:
            result = _call_urlopen_site(caller, servers, monkeypatch)
        except (kernel_redirect_policy.ProviderRedirectRefused, endpoint.EndpointError) as caught:
            error = caught
        optional = caller in {"decode-depth slots", "long-horizon slots", "endpoint slots", "provider epoch"}
        assert (result is None and error is None) if optional else error is not None, caller
        assert [row["path"].split("?", 1)[0] for row in servers.state["requests"]
                if row["server"] == "source"] == [path], caller
        assert not any(row["server"] == "target" for row in servers.state["requests"]), caller
        if error is not None:
            assert _KEY not in str(error) and _KEY not in repr(error), caller


@pytest.mark.parametrize("caller", _URL_OPEN_SITES)
def test_each_urlopen_call_site_keeps_same_origin_redirect(monkeypatch, caller) -> None:
    _no_environment_proxy(monkeypatch)
    with _Servers() as servers:
        path = {"decode-depth slots": "/slots", "long-horizon slots": "/slots",
                "endpoint models": "/v1/models", "endpoint slots": "/slots",
                "provider epoch": "/v1/provider_epoch", "template sidecar": "/cross/start",
                "detreason model": "/cross/start", "fixer model": "/cross/start",
                "swarm model": "/cross/start", "KV slot action": "/slots/0",
                "react ask_tools": "/cross/start", "react ask": "/cross/start"}[caller]
        servers.state["redirects"] = {path: servers.state["source"] + "/same/final"}
        servers.state["success_body"] = b'{"choices":[{"message":{"content":"ok"}}]}'
        _call_urlopen_site(caller, servers, monkeypatch)
        rows = servers.state["requests"]
        source = [row for row in rows if row["server"] == "source"]
        expected = [path] if caller == "provider epoch" else [path, "/same/final"]
        assert [row["path"].split("?", 1)[0] for row in source] == expected
        assert source[-1]["authorization"] == "Bearer " + _KEY, caller
        assert not any(row["server"] == "target" for row in rows), caller


@pytest.mark.parametrize("status", (302, 307))
def test_context_capacity_probe_returns_unavailable_without_following_redirect(status) -> None:
    import kernel_context_capacity as capacity

    capacity._cache.clear()
    with _Servers() as servers:
        servers.state["status"] = status
        servers.state["redirects"] = {
            "/slots": servers.state["target"] + "/received?api_key=" + _KEY,
            "/props": servers.state["target"] + "/received?api_key=" + _KEY,
        }
        assert capacity.probe_n_ctx(servers.state["source"], _KEY) is None
        source = [row for row in servers.state["requests"] if row["server"] == "source"]
        assert [row["path"] for row in source] == ["/slots", "/props"]
        assert source[0]["authorization"] == "Bearer " + _KEY
        assert not any(row["path"].startswith("/received") for row in servers.state["requests"])
        assert not any(row["server"] == "target" for row in servers.state["requests"])
    capacity._cache.clear()


@pytest.mark.parametrize("status", (302, 307))
def test_context_capacity_probe_does_not_follow_same_origin_redirect(status) -> None:
    import kernel_context_capacity as capacity

    capacity._cache.clear()
    with _Servers() as servers:
        servers.state["status"] = status
        servers.state["redirects"] = {"/slots": servers.state["source"] + "/same/final"}
        assert capacity.probe_n_ctx(servers.state["source"], _KEY) is None
        rows = [row for row in servers.state["requests"] if row["server"] == "source"]
        assert [row["path"] for row in rows] == ["/slots", "/props"]
        assert rows[0]["authorization"] == "Bearer " + _KEY
        assert not any(row["path"] == "/same/final" for row in rows)
        assert not any(row["server"] == "target" for row in servers.state["requests"])
    capacity._cache.clear()


@pytest.mark.parametrize("caller,opener_factory", tuple(_opener_cases()))
@pytest.mark.parametrize("status", (301, 302, 303, 307, 308))
@pytest.mark.parametrize("method", ("GET", "POST"))
def test_provider_openers_refuse_cross_origin_before_target_contact(
        caller, opener_factory, status, method) -> None:
    with _Servers() as servers:
        servers.state["status"] = status
        result = opener_factory()
        opener, cleanup = result if isinstance(result, tuple) else (result, lambda: None)
        request = urllib.request.Request(
            servers.state["source"] + "/cross/start?token=" + _KEY,
            data=b"provider request" if method == "POST" else None, headers={
                "Authorization": "Bearer " + _KEY,
                "X-Api-Key": _KEY,
                "Api-Key": _KEY,
                "Cookie": "session=" + _KEY,
                "X-Custom-Credential": _KEY,
            })
        error = None
        try:
            with opener.open(request, timeout=2) as response:
                response.read()
        except Exception as caught:  # the production refusal crosses several transport types
            error = caught
        finally:
            cleanup()

        target_requests = [row for row in servers.state["requests"] if row["server"] == "target"]
        assert target_requests == [], f"{caller} contacted the redirected origin: {target_requests!r}"
        assert error is not None, f"{caller} did not refuse the redirected origin"
        assert type(error).__name__ == "ProviderRedirectRefused", (
            f"{caller} returned an untyped redirect refusal: {type(error).__name__}: {error}"
        )
        rendered = f"{error!s} {error!r}"
        assert "127.0.0.1" in rendered
        assert [row["method"] for row in servers.state["requests"] if row["server"] == "source"] == [method]
        for secret in (_KEY, "/received", "api_key", "user:"):
            assert secret not in rendered


@pytest.mark.parametrize("caller,opener_factory", tuple(_opener_cases()))
def test_provider_openers_refuse_cross_origin_after_same_origin_hop(caller, opener_factory) -> None:
    with _Servers() as servers:
        result = opener_factory()
        opener, cleanup = result if isinstance(result, tuple) else (result, lambda: None)
        request = urllib.request.Request(
            servers.state["source"] + "/chain/start", headers={"Authorization": "Bearer " + _KEY})
        try:
            with pytest.raises(Exception) as caught:
                opener.open(request, timeout=2)
        finally:
            cleanup()
        assert type(caught.value).__name__ == "ProviderRedirectRefused"
        assert [row["path"] for row in servers.state["requests"] if row["server"] == "source"] == [
            "/chain/start", "/chain/middle"]
        assert not any(row["server"] == "target" for row in servers.state["requests"])


@pytest.mark.parametrize("caller,opener_factory", tuple(_opener_cases()))
@pytest.mark.parametrize("path", ("/same/start", "/relative/start"))
def test_provider_openers_keep_same_origin_redirects(caller, opener_factory, path) -> None:
    with _Servers() as servers:
        result = opener_factory()
        opener, cleanup = result if isinstance(result, tuple) else (result, lambda: None)
        request = urllib.request.Request(
            servers.state["source"] + path, headers={"Authorization": "Bearer " + _KEY})
        try:
            with opener.open(request, timeout=2) as response:
                assert response.read() == b"ok"
        finally:
            cleanup()
        final = next(row for row in servers.state["requests"]
                     if row["server"] == "source" and row["path"].endswith("final"))
        assert final["authorization"] == "Bearer " + _KEY, caller
        assert not any(row["server"] == "target" for row in servers.state["requests"])


def test_provider_openers_refuse_userinfo_redirect_without_rendering_url_secrets() -> None:
    opener = kernel_redirect_policy.build_credential_scoped_opener(proxy_mode="direct")
    with _Servers() as servers:
        request = urllib.request.Request(
            servers.state["source"] + "/userinfo/start", headers={"Authorization": "Bearer " + _KEY})
        with pytest.raises(Exception) as caught:
            opener.open(request, timeout=2)
        error = caught.value
        assert type(error).__name__ == "ProviderRedirectRefused"
        assert not any(row["server"] == "target" for row in servers.state["requests"])
        text = f"{error!s} {error!r}"
        for secret in (_KEY, "user", "/private/path", "?key="):
            assert secret not in text


def test_no_key_same_origin_redirect_remains_usable() -> None:
    opener = kernel_redirect_policy.build_credential_scoped_opener(proxy_mode="direct")
    with _Servers() as servers:
        with opener.open(servers.state["source"] + "/same/start", timeout=2) as response:
            assert response.read() == b"ok"
        assert any(row["path"] == "/same/final" for row in servers.state["requests"])


def test_no_key_cross_origin_redirect_is_refused_before_target_contact() -> None:
    opener = kernel_redirect_policy.build_credential_scoped_opener(proxy_mode="direct")
    with _Servers() as servers:
        with pytest.raises(Exception) as caught:
            opener.open(servers.state["source"] + "/cross/start", timeout=2)
        assert type(caught.value).__name__ == "ProviderRedirectRefused"
        assert not any(row["server"] == "target" for row in servers.state["requests"])


@pytest.mark.parametrize(("source", "target"), [
    ("http://example.test/start", "https://example.test/next"),
    ("https://example.test/start", "http://example.test/next"),
    ("http://example.test/start", "http://example.test:81/next"),
])
def test_scheme_and_port_changes_are_refused(source, target) -> None:
    handler = kernel_redirect_policy._scoped_redirect_handler(
        urllib.request.HTTPRedirectHandler())
    request = urllib.request.Request(source, headers={"Authorization": "Bearer " + _KEY})
    with pytest.raises(kernel_redirect_policy.ProviderRedirectRefused) as caught:
        handler.redirect_request(request, None, 302, "Found", {"Location": target}, target)
    assert _KEY not in str(caught.value) and target.split(":", 1)[0] not in str(caught.value)


def test_normalized_origins_cover_trailing_dot_ipv6_and_explicit_defaults() -> None:
    origin = kernel_redirect_policy._origin
    assert origin("http://example.test.") == origin("http://example.test:80")
    assert origin("https://example.test") == origin("https://example.test:443")
    assert origin("http://[::1]") == origin("http://[0:0:0:0:0:0:0:1]:80")


def test_trailing_dot_same_origin_redirect_is_kept() -> None:
    handler = kernel_redirect_policy._scoped_redirect_handler(
        urllib.request.HTTPRedirectHandler())
    request = urllib.request.Request(
        "http://example.test/start", headers={"Authorization": "Bearer " + _KEY})
    redirected = handler.redirect_request(
        request, None, 302, "Found", {}, "http://example.test.:80/next")
    assert redirected is not None
    assert redirected.get_header("Authorization") == "Bearer " + _KEY


def test_provider_urlopen_no_key_cross_origin_redirect_is_refused(monkeypatch) -> None:
    _no_environment_proxy(monkeypatch)
    with _Servers() as servers:
        with pytest.raises(kernel_redirect_policy.ProviderRedirectRefused):
            kernel_redirect_policy.provider_urlopen(servers.state["source"] + "/cross/start", timeout=2)
        assert not any(row["server"] == "target" for row in servers.state["requests"])


def test_protocol_relative_cross_origin_redirect_is_refused() -> None:
    opener = kernel_redirect_policy.build_credential_scoped_opener(proxy_mode="direct")
    with _Servers() as servers:
        with pytest.raises(Exception) as caught:
            opener.open(servers.state["source"] + "/protocol/start", timeout=2)
        assert type(caught.value).__name__ == "ProviderRedirectRefused"
        assert not any(row["server"] == "target" for row in servers.state["requests"])


def test_engine_installed_urlopen_refuses_cross_origin(monkeypatch) -> None:
    previous = urllib.request._opener
    try:
        _child._install_direct_opener()
        with _Servers() as servers:
            request = urllib.request.Request(
                servers.state["source"] + "/cross/start", headers={"Authorization": "Bearer " + _KEY})
            with pytest.raises(Exception) as caught:
                urllib.request.urlopen(request, timeout=2)
            assert type(caught.value).__name__ == "ProviderRedirectRefused"
            assert not any(row["server"] == "target" for row in servers.state["requests"])
    finally:
        urllib.request._opener = previous


@pytest.mark.parametrize("target", ("http://example.test/next", "file:///private/path"))
def test_scheme_change_and_non_http_targets_are_typed_refusals(target) -> None:
    handler = kernel_redirect_policy._scoped_redirect_handler(
        urllib.request.HTTPRedirectHandler())
    request = urllib.request.Request("https://example.test/start", headers={"X-Api-Key": _KEY})
    with pytest.raises(Exception) as caught:
        handler.redirect_request(request, None, 302, "Found", {"Location": target}, target)
    assert type(caught.value).__name__ == "ProviderRedirectRefused"
    assert _KEY not in str(caught.value)
    assert "/private/path" not in str(caught.value)


def test_successful_provider_request_body_and_headers_are_unchanged() -> None:
    opener = kernel_redirect_policy.build_credential_scoped_opener(proxy_mode="direct")
    body = b'{"model":"local","messages":[]}'
    with _Servers() as servers:
        request = urllib.request.Request(
            servers.state["source"] + "/plain", data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + _KEY})
        with opener.open(request, timeout=2) as response:
            assert response.read() == b"ok"
        row, = servers.state["requests"]
        assert row["method"] == "POST"
        assert row["body"] == body
        assert row["authorization"] == "Bearer " + _KEY
        assert not any(item["server"] == "target" for item in servers.state["requests"])


def test_provider_open_call_sites_use_only_the_shared_redirect_policy() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "stateframe"
    offenders = []
    for path in source.rglob("*.py"):
        if path.name == "kernel_redirect_policy.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = function.id if isinstance(function, ast.Name) else (
                function.attr if isinstance(function, ast.Attribute) else "")
            if name in {"urlopen", "OpenerDirector"}:
                if name == "OpenerDirector" and path.relative_to(source).as_posix() == "endpoint.py":
                    # provider_epoch uses a bare director with only HTTPHandler, so it cannot redirect.
                    continue
                offenders.append(f"{path.relative_to(source)}:{node.lineno}:{name}")
    assert offenders == []


@pytest.mark.parametrize(("target", "host"), [
    ("http://bad\r\nhost/path", "unknown"),
    ("http://example.test/private/path", "example.test"),
    ("http://example.test/path?key=secret", "example.test"),
    ("http://user:secret@example.test/path", "unknown"),
])
def test_redirect_refusal_reports_only_a_safe_target_host(target, host) -> None:
    error = kernel_redirect_policy.ProviderRedirectRefused(target)
    assert error.target_host == host
    assert target not in str(error)


def test_bounded_child_refusal_is_typed_and_safe_in_stderr_and_error_projection() -> None:
    wire = b'{"messages":[]}'
    with _Servers() as servers:
        deadline = time.monotonic() + 10
        packet = json.dumps({
            "wire": wire.decode(),
            "endpoint": servers.state["source"] + "/cross/start?key=" + _KEY,
            "key": _KEY,
            "deadline": deadline,
            "request_deadline": deadline,
            "socket_timeout": 3,
            "send_stall_seconds": kernel_bounded_http.send_stall_seconds(),
            "report_body_sent": True,
            "gzip": False,
        }).encode()
        child = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(kernel_bounded_http.__file__).resolve()),
             "--http-child"],
            input=packet, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=12, check=False)
        target_requests = [row for row in servers.state["requests"] if row["server"] == "target"]

    assert child.returncode != 0
    assert target_requests == []
    diagnostic = child.stderr.decode("utf-8", errors="replace")
    assert _KEY not in diagnostic
    child_record = json.loads(diagnostic.splitlines()[-1])
    assert child_record["error_type"] == "ProviderRedirectRefused"
    assert child_record["redirect_target_host"] == "127.0.0.1"
    error = kernel_bounded_http._child_transport_error(
        child.stderr, _KEY, servers.state["source"] + "/cross/start")
    assert isinstance(error, kernel_redirect_policy.ProviderRedirectRefused)
    assert _KEY not in str(error) and _KEY not in repr(error)
    assert "/cross" not in str(error) and "?key=" not in str(error)
    projected_detail = kernel_bounded_http.safe_transport_detail(error, _KEY)
    assert projected_detail == str(error)
    assert _KEY not in projected_detail


def test_engine_transport_summary_and_journal_omit_credentials(tmp_path) -> None:
    body = {"messages": [{"role": "user", "content": "local"}], "tools": []}
    wire = json.dumps(body).encode()
    journal = kernel_measurement.AuditJournal(tmp_path, "redirect-refusal", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    with _Servers() as servers:
        with pytest.raises(kernel_measurement.HarnessTransportError) as caught:
            capture._send_transport_with_retries(
                body, wire, servers.state["source"] + "/cross/start?key=" + _KEY, _KEY,
                admission=None, socket_timeout=3, deadline=time.monotonic() + 10,
                wall_timeout=3, backend_identity={}, request_extensions={},
                frozen_retry_only=True)
    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    errors = [event for event in events if event.get("kind") == "transport_error"]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "ProviderRedirectRefused"
    assert errors[0]["retry_scheduled"] is False
    assert caught.value.error_type == "ProviderRedirectRefused"
    assert _KEY not in str(caught.value)
    assert _KEY not in json.dumps(events)


@pytest.mark.parametrize(("source", "target", "same"), [
    ("http://EXAMPLE.com/start", "http://example.com.:80/next", True),
    ("https://[::1]/start", "https://[0:0:0:0:0:0:0:1]:443/next", True),
    ("http://localhost/start", "http://127.0.0.1/next", False),
    ("https://example.com/start", "http://example.com/next", False),
    ("https://example.com:8443/start", "https://example.com:8444/next", False),
])
def test_origin_normalization(source, target, same) -> None:
    assert (kernel_redirect_policy._origin(source) == kernel_redirect_policy._origin(target)) is same


def test_refusal_classifier_recognizes_a_separately_loaded_policy_copy() -> None:
    from runpy import run_path

    duplicate = run_path(str(Path(kernel_redirect_policy.__file__).resolve()))
    error = duplicate["ProviderRedirectRefused"]("https://other.example.test/secret?key=hidden")
    assert kernel_redirect_policy.is_provider_redirect_refused(error)


@pytest.mark.host_sandbox
def test_endpoint_slots_hide_redirect_refusal_and_probe_keeps_endpoint_error() -> None:
    from stateframe.endpoint import Endpoint, EndpointError

    with _Servers() as servers:
        servers.state["redirects"] = {
            "/slots": servers.state["target"] + "/received?api_key=" + _KEY,
            "/v1/models": servers.state["target"] + "/received?api_key=" + _KEY,
        }
        endpoint = Endpoint(servers.state["source"] + "/v1", "local-model")
        assert endpoint.get_slots() is None
        with pytest.raises(EndpointError) as caught:
            endpoint.probe()

    assert "provider redirect refused for target host 127.0.0.1" in str(caught.value)
    assert _KEY not in str(caught.value)
    assert "/received" not in str(caught.value)
    assert not any(row["server"] == "target" for row in servers.state["requests"])


@pytest.mark.host_sandbox
def test_tokenizer_refusal_falls_back_after_one_authenticated_post(monkeypatch) -> None:
    import kernel_provider_dialect

    monkeypatch.setattr(kernel_provider_dialect, "capabilities_for", lambda *_a, **_k: object())
    monkeypatch.setattr(kernel_provider_dialect, "require_feature", lambda *_a, **_k: None)
    with _Servers() as servers:
        servers.state["status"] = 307
        servers.state["redirects"] = {
            "/tokenize": servers.state["target"] + "/received?api_key=" + _KEY,
        }
        source = servers.state["source"]
        meter = kernel_request_diet.RequestTokenMeter(
            endpoint=source + "/tokenize", template_overhead_tokens=0,
            auth_key=_KEY, generation_endpoint=source + "/v1/chat/completions",
        )
        count, method = meter.measure({"messages": [{"role": "user", "content": "local"}]})
        posts = [row for row in servers.state["requests"]
                 if row["server"] == "source" and row["path"].startswith("/tokenize")]
        assert method == "calibrated_bytes" and count > 0
        assert len(posts) == 1 and posts[0]["method"] == "POST"
        assert posts[0]["authorization"] == "Bearer " + _KEY
        assert not any(row["server"] == "target" for row in servers.state["requests"])


def test_redirect_callers_import_with_only_the_source_package() -> None:
    code = (
        "import stateframe.endpoint, stateframe.engine.swarm.client2, "
        "stateframe.engine.detreason.kernel, "
        "stateframe.engine.kernel_template_conformance, stateframe.engine.fixer"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.pop("PYTHONHOME", None)
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")


@pytest.mark.parametrize("call", ("text", "subframe"))
def test_tokenizer_refusal_stops_the_other_count_paths_after_one_attempt(monkeypatch, call) -> None:
    meter = kernel_request_diet.RequestTokenMeter(
        tokenizer=lambda content: [1], template_overhead_tokens=0)
    attempts = []

    def refuse(*_args, **_kwargs):
        attempts.append(1)
        raise kernel_redirect_policy.ProviderRedirectRefused("https://other.example.test/x")

    if call == "text":
        meter._availability = "available"
        monkeypatch.setattr(meter, "_count_tokens", refuse)
        assert meter.count_text_exact("local") is None
    else:
        meter._tokenizer = refuse
        count, method = meter.measure_subframe_cap(
            {"messages": [{"role": "user", "content": "local"}]}, tokens_per_byte=0.5)
        assert method == "calibrated_bytes" and count > 0
    assert attempts == [1]


@pytest.mark.parametrize("host", ("example.test", "127.0.0.1", "::1"))
def test_bounded_parent_keeps_the_refused_host(host) -> None:
    import kernel_bounded_http

    record = {"error_type": "ProviderRedirectRefused", "redirect_target_host": host,
              "response_bytes_received": True}
    error = kernel_bounded_http._child_transport_error(
        json.dumps(record).encode(), _KEY, "http://127.0.0.1/v1")
    assert kernel_redirect_policy.is_provider_redirect_refused(error)
    assert (error.target_host, error.response_bytes_received) == (host, True)


def test_bounded_parent_does_not_repeat_a_key_inside_the_refused_host() -> None:
    import kernel_bounded_http

    record = {"error_type": "ProviderRedirectRefused",
              "redirect_target_host": "sample-key.example.test"}
    error = kernel_bounded_http._child_transport_error(
        json.dumps(record).encode(), "Sample-Key", "http://127.0.0.1/v1")
    assert error.target_host == "unknown"
    assert "sample-key" not in str(error)


def test_subframe_refusal_after_the_deadline_is_a_timeout(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(kernel_request_diet.time, "monotonic", lambda: clock[0])

    def refuse(_content):
        clock[0] = 2.0
        raise kernel_redirect_policy.ProviderRedirectRefused("https://other.example.test/x")

    meter = kernel_request_diet.RequestTokenMeter(tokenizer=refuse, template_overhead_tokens=0)
    meter._deadline = 1.0
    with pytest.raises(TimeoutError):
        meter.measure_subframe_cap(
            {"messages": [{"role": "user", "content": "local"}]}, tokens_per_byte=0.5)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("kind", ("tokenizer", "scoped"))
@pytest.mark.parametrize("target", (
    "file:///etc/hostname", "ftp://127.0.0.1:9/x", "data:text/plain,x", "https://127.0.0.1:9/x"))
def test_redirect_to_another_scheme_or_port_is_refused(monkeypatch, kind, target) -> None:
    import urllib.error

    _no_environment_proxy(monkeypatch)
    opened = []
    for name in ("FileHandler.file_open", "FTPHandler.ftp_open", "DataHandler.data_open",
                 "HTTPSHandler.https_open"):
        owner, method = name.split(".")
        monkeypatch.setattr(getattr(urllib.request, owner), method,
                            lambda *_args, _name=name: opened.append(_name))
    monkeypatch.setattr(kernel_request_diet._TokenizeHTTPSHandler, "https_open",
                        lambda *_args: opened.append("tokenizer https"))
    opener = (kernel_request_diet._build_tokenize_opener() if kind == "tokenizer"
              else kernel_redirect_policy.build_credential_scoped_opener(proxy_mode="direct"))
    with _Servers() as servers:
        servers.state["redirects"] = {"/start": target}
        request = urllib.request.Request(
            servers.state["source"] + "/start", headers={"Authorization": "Bearer " + _KEY})
        with pytest.raises((urllib.error.HTTPError, ValueError)) as caught:
            opener.open(request, timeout=5)
    assert opened == []
    assert (kernel_redirect_policy.is_provider_redirect_refused(caught.value)
            or isinstance(caught.value, urllib.error.HTTPError))
    assert _KEY not in str(caught.value)
