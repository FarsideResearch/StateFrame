from __future__ import annotations

import gzip
import email.utils
import http.server
import os
import socket
import ssl
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from stateframe.otel.transport import TransportSettings, _backoff, destination_id, send
from stateframe.otel.types import (
    AttrKey,
    SafeAttr,
    SafeSpan,
    Signal,
    SignalBatch,
    ResultKind,
    Reason,
)
from tests.otlp_receiver import OTLPReceiver, ResponseStep, assert_valid_traces


KEY = bytes(range(32))
TRACE_ID = bytes.fromhex("00112233445566778899aabbccddeeff")
RESOURCE_ID = bytes.fromhex("ffeeddccbbaa99887766554433221100")
SPAN_ID = bytes.fromhex("0102030405060708")
EPOCH = 1_730_000_000_000_000_000


def _batch() -> SignalBatch:
    attrs = tuple(sorted((
        SafeAttr(AttrKey.RUN_ID, "10112233445566778899aabbccddeeff"),
        SafeAttr(AttrKey.ATTEMPT, 1),
        SafeAttr(AttrKey.SOURCE, "record"),
        SafeAttr(AttrKey.TIMING, "publication"),
        SafeAttr(AttrKey.COMPLETE, True),
    ), key=lambda item: str(item.key)))
    span = SafeSpan(
        "stateframe.invocation", TRACE_ID, SPAN_ID, None, EPOCH, EPOCH,
        attrs, (), (), 0, 0, 0, False,
    )
    return SignalBatch(Signal.TRACES, RESOURCE_ID, (span,))


def _settings(endpoint: str, **overrides) -> TransportSettings:
    values = {
        "endpoint": endpoint,
        "protocol": "http/json",
        "headers": (),
        "compression": "none",
        "timeout_ms": 1_000,
        "ca": None,
        "cert": None,
        "key_file": None,
    }
    values.update(overrides)
    return TransportSettings(**values)


def _deadline(seconds: float = 2.0) -> int:
    return time.monotonic_ns() + int(seconds * 1_000_000_000)


def _no_sleep(monkeypatch):
    monkeypatch.setattr("stateframe.otel.transport.time.sleep", lambda _seconds: None)


def test_retry_jitter_is_bounded_and_stable_per_payload():
    delays = [_backoff(attempt, b"stable-payload-a") for attempt in range(5)]
    assert delays == [_backoff(attempt, b"stable-payload-a") for attempt in range(5)]
    assert all(0 <= delay < min(5.0, 0.25 * (2 ** attempt))
               for attempt, delay in enumerate(delays))
    assert _backoff(0, b"stable-payload-a") != _backoff(0, b"stable-payload-b")


def _resolve_test_collector(monkeypatch):
    original = socket.getaddrinfo

    def resolve(host, port, *args, **kwargs):
        if host == "collector.test":
            return original("127.0.0.1", port, *args, **kwargs)
        return original(host, port, *args, **kwargs)

    monkeypatch.setattr("stateframe.otel.transport.socket.getaddrinfo", resolve)


@pytest.mark.parametrize(
    ("mode", "expected_requests", "expected_kind"),
    [
        ("http_400", 1, "permanent"),
        ("http_401", 1, "permanent"),
        ("http_413", 1, "permanent"),
        ("http_429", 2, "accepted"),
        ("http_500", 1, "permanent"),
        ("http_502", 2, "accepted"),
        ("http_503", 2, "accepted"),
        ("http_504", 2, "accepted"),
    ],
)
def test_retry_status_matrix(mode: str, expected_requests: int, expected_kind: str, monkeypatch):
    _no_sleep(monkeypatch)
    with OTLPReceiver([ResponseStep(mode, retry_after="0"), "accept"]) as receiver:
        result = send(_batch(), _settings(receiver.url + "/v1/traces"), _deadline())
        assert result.kind.value == expected_kind
        assert len(receiver.requests) == expected_requests
        if mode == "http_413":
            assert result.reason == Reason.PAYLOAD_REJECTED
            assert len(receiver.requests) == 1


@pytest.mark.parametrize("retry_after", ["0.01", None])
def test_retry_after_seconds_and_http_date(retry_after: str | None, monkeypatch):
    _no_sleep(monkeypatch)
    delays: list[float] = []
    monkeypatch.setattr("stateframe.otel.transport.time.sleep", delays.append)
    if retry_after is None:
        retry_after = email.utils.formatdate(time.time() + 1, usegmt=True)
    with OTLPReceiver([ResponseStep("http_503", retry_after=retry_after), "accept"]) as receiver:
        result = send(_batch(), _settings(receiver.url + "/v1/traces", timeout_ms=5_000), _deadline(6.0))
        assert result.kind == ResultKind.ACCEPTED
        assert len(receiver.requests) == 2
    assert delays
    if retry_after == "0.01":
        assert delays[0] >= 0.01


def test_export_timeout_not_reset_per_attempt(monkeypatch):
    _no_sleep(monkeypatch)
    clock = [10_000_000_000]
    monkeypatch.setattr("stateframe.otel.transport.time.monotonic_ns", lambda: clock[0])
    monkeypatch.setattr(
        "stateframe.otel.transport.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + int(seconds * 1_000_000_000)),
    )
    with OTLPReceiver([ResponseStep("http_503", retry_after="2"), "accept"]) as receiver:
        result = send(_batch(), _settings(receiver.url + "/v1/traces", timeout_ms=5_000), clock[0] + 500_000_000)
        assert result.kind == ResultKind.RETRY
        assert result.may_have_delivered
        assert len(receiver.requests) == 1


def test_gzip_request_and_bounded_response_roundtrip():
    with OTLPReceiver() as receiver:
        result = send(
            _batch(),
            _settings(receiver.url + "/v1/traces", compression="gzip"),
            _deadline(),
        )
        assert result.kind == ResultKind.ACCEPTED
        request = receiver.requests[0]
        assert request.headers["content-encoding"] == "gzip"
        assert int(request.headers["content-length"]) == len(request.raw_body)
        assert_valid_traces(__import__("json").loads(gzip.decompress(request.raw_body)))


class _GzipReplyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_POST(self):  # noqa: N802
        size = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(size)
        body = gzip.compress(b"{}" + b"x" * (70 * 1024), mtime=0)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass


def test_response_gzip_bomb_is_bounded():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _GzipReplyHandler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=False)
    thread.start()
    try:
        port = server.server_address[1]
        result = send(_batch(), _settings(f"http://127.0.0.1:{port}/v1/traces"), _deadline())
        assert result.kind == ResultKind.PERMANENT
        assert result.reason == Reason.BAD_RESPONSE
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_redirect_never_forwards_credentials(monkeypatch):
    _resolve_test_collector(monkeypatch)
    class _PassthroughTLS:
        check_hostname = True
        verify_mode = ssl.CERT_REQUIRED

        def load_verify_locations(self, **_kwargs):
            return None

        def wrap_socket(self, sock, *, server_hostname):
            assert server_hostname == "collector.test"
            return sock

    monkeypatch.setattr("stateframe.otel.transport.ssl.create_default_context", lambda **_kwargs: _PassthroughTLS())
    with OTLPReceiver() as target:
        with OTLPReceiver(
            ["redirect_cross_origin"],
            cross_origin_url=target.url + "/v1/traces",
        ) as source:
            parsed = __import__("urllib.parse").parse.urlsplit(source.url)
            endpoint = f"https://collector.test:{parsed.port}/v1/traces"
            settings = _settings(endpoint, headers=(("Authorization", "Bearer private-header-marker"),))
            result = send(_batch(), settings, _deadline())
            assert result.kind == ResultKind.PERMANENT
            assert result.reason == Reason.REDIRECT
            assert len(source.requests) == 1
            assert target.requests == ()
            assert "private-header-marker" not in repr(result)


@pytest.mark.parametrize("fail_tls", [False, True])
def test_tls_hostname_ca_mtls_never_downgrades(tmp_path: Path, monkeypatch, fail_tls: bool):
    _resolve_test_collector(monkeypatch)
    ca, cert, key_file = (tmp_path / name for name in ("ca.pem", "client.pem", "client.key"))
    for path in (ca, cert, key_file):
        path.write_text("fixture", encoding="ascii")
        path.chmod(0o600)

    contexts = []

    class _TLSContext:
        check_hostname = True
        verify_mode = ssl.CERT_REQUIRED

        def __init__(self):
            self.ca = None
            self.client_pair = None
            self.host = None

        def load_verify_locations(self, *, cafile):
            self.ca = cafile

        def load_cert_chain(self, certfile, keyfile):
            self.client_pair = (certfile, keyfile)

        def wrap_socket(self, sock, *, server_hostname):
            self.host = server_hostname
            if fail_tls:
                raise ssl.SSLCertVerificationError("fixture failure marker")
            return sock

    def create_context(**_kwargs):
        context = _TLSContext()
        contexts.append(context)
        return context

    monkeypatch.setattr("stateframe.otel.transport.ssl.create_default_context", create_context)
    with OTLPReceiver() as receiver:
        port = __import__("urllib.parse").parse.urlsplit(receiver.url).port
        settings = _settings(
            f"https://collector.test:{port}/v1/traces",
            ca=ca,
            cert=cert,
            key_file=key_file,
        )
        result = send(_batch(), settings, _deadline())

    context = contexts[0]
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.ca == str(ca)
    assert context.client_pair == (str(cert), str(key_file))
    assert context.host == "collector.test"
    if fail_tls:
        assert result.reason == Reason.TLS_FAILED
        assert receiver.requests == ()
    else:
        assert result.kind == ResultKind.ACCEPTED
        assert len(receiver.requests) == 1


def test_auth_and_413_use_bounded_health_and_no_splitting(monkeypatch):
    for status, reason in (("http_401", Reason.AUTH_FAILED), ("http_413", Reason.PAYLOAD_REJECTED)):
        with OTLPReceiver([status, "accept"]) as receiver:
            result = send(_batch(), _settings(receiver.url + "/v1/traces"), _deadline())
            assert result.kind == ResultKind.PERMANENT
            assert result.reason == reason
            assert len(receiver.requests) == 1
            assert "collector rejected request" not in repr(result)


def test_proxy_and_global_opener_cannot_interpose(monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: pytest.fail("global opener used"))
    with OTLPReceiver() as receiver:
        result = send(_batch(), _settings(receiver.url + "/v1/traces"), _deadline())
        assert result.kind == ResultKind.ACCEPTED
        assert len(receiver.requests) == 1


def test_destination_authority_and_dns_address_pinning(monkeypatch):
    _no_sleep(monkeypatch)
    calls: list[tuple[str, int]] = []
    class _NoopTLSContext:
        check_hostname = True
        verify_mode = ssl.CERT_REQUIRED

        def wrap_socket(self, sock, *, server_hostname):
            self.server_hostname = server_hostname
            return sock

    monkeypatch.setattr("stateframe.otel.transport.ssl.create_default_context", _NoopTLSContext)

    def resolve(host, port, *_args, **_kwargs):
        calls.append((host, port))
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port))]

    monkeypatch.setattr("stateframe.otel.transport.socket.getaddrinfo", resolve)
    with OTLPReceiver(["http_503", "accept"]) as receiver:
        port = __import__("urllib.parse").parse.urlsplit(receiver.url).port
        settings = _settings(f"https://collector.test:{port}/v1/traces")
        result = send(_batch(), settings, _deadline())
        assert result.kind == ResultKind.ACCEPTED
        assert calls == [("collector.test", port)]
        assert len(receiver.requests) == 2
        assert all(request.headers["host"] == f"collector.test:{port}" for request in receiver.requests)

    calls.clear()

    def resolve_mixed(host, port, *_args, **_kwargs):
        calls.append((host, port))
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("169.254.169.254", port)),
        ]

    monkeypatch.setattr("stateframe.otel.transport.socket.getaddrinfo", resolve_mixed)
    with OTLPReceiver() as receiver:
        port = __import__("urllib.parse").parse.urlsplit(receiver.url).port
        result = send(_batch(), _settings(f"https://collector.test:{port}/v1/traces"), _deadline())
        assert result.reason == Reason.DNS_FAILED
        assert receiver.requests == ()


def test_destination_fingerprint_is_secret_safe_and_configuration_bound(tmp_path: Path):
    marker = "private-destination-token-marker"
    path = tmp_path / "client.key"
    path.write_text("private key marker", encoding="ascii")
    path.chmod(0o600)
    first = _settings(
        "https://collector.test/v1/traces",
        headers=(("Authorization", marker),),
        cert=path,
        key_file=path,
    )
    same = _settings(
        "https://collector.test/v1/traces",
        headers=(("Authorization", marker),),
        cert=path,
        key_file=path,
    )
    changed = _settings(
        "https://collector.test/v1/alternate",
        headers=(("Authorization", marker),),
        cert=path,
        key_file=path,
    )
    assert destination_id(KEY, first) == destination_id(KEY, same)
    assert destination_id(KEY, first) != destination_id(KEY, changed)
    assert len(destination_id(KEY, first)) == 16
    assert marker not in repr(first)
    assert str(path) not in repr(first)


def test_throttle_storm_jitter_and_aggregate_bounds(monkeypatch):
    from stateframe.otel.transport import MAX_RETRIES

    delays = []
    monkeypatch.setattr("stateframe.otel.transport.time.sleep", delays.append)
    with OTLPReceiver([ResponseStep("http_429", retry_after="0.01")] * (MAX_RETRIES + 1)) as receiver:
        outcome = send(_batch(), _settings(receiver.url + "/v1/traces"), _deadline())
        assert outcome.kind is ResultKind.RETRY
        assert outcome.reason is Reason.THROTTLED
        assert len(receiver.requests) == MAX_RETRIES
        assert len(delays) == MAX_RETRIES - 1
        assert all(0.01 <= delay < 1.0 for delay in delays)
        assert len(set(delays)) > 1
        assert len({request.raw_body for request in receiver.requests}) == 1
