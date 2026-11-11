"""Raw web fetches retain the existing web observation behavior."""

import gzip
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stateframe.engine import kernel_web_fetch as web


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/plain")
            self.end_headers()
            return
        if self.path == "/private-redirect":
            self.send_response(302)
            self.send_header("Location", "http://private.test/secret")
            self.end_headers()
            return
        if self.path == "/gzip":
            body = gzip.compress(b"compressed page")
            headers = {"Content-Type": "text/plain; charset=utf-8",
                       "Content-Encoding": "gzip"}
        elif self.path == "/gzip-large":
            body = gzip.compress(b"A" * 5000)
            headers = {"Content-Type": "text/plain; charset=utf-8",
                       "Content-Encoding": "gzip"}
        elif self.path == "/html":
            body = b"<h1>Guide</h1><script>secret</script><p>Read <a href='/docs'>docs</a>.</p>"
            headers = {"Content-Type": "text/html; charset=utf-8"}
        elif self.path == "/large":
            body = b"0123456789"
            headers = {"Content-Type": "text/plain"}
        elif self.path == "/binary-large":
            body = b"%PDF-1.7 plus bytes"
            headers = {"Content-Type": "application/pdf"}
        elif self.path == "/binary-encoding":
            body = b"opaque"
            headers = {"Content-Type": "application/pdf", "Content-Encoding": "br"}
        else:
            body = b"plain page"
            headers = {"Content-Type": "text/plain; charset=utf-8"}
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def fetch_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def loopback(*_args, **_kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]


def policy(port, **kwargs):
    return web.WebPolicy(("fetch.test",), allowed_ports=(port,),
                         allow_private_for_tests=True, **kwargs)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("path", ["/plain", "/redirect", "/gzip", "/html"])
def test_fetch_keeps_golden_output_while_raw_exposes_body(fetch_server, path):
    url = f"http://fetch.test:{fetch_server}{path}"
    result = web.fetch(url, policy(fetch_server), resolver=loopback)
    raw = web.fetch_raw(url, policy(fetch_server), accept="text/plain",
                        max_bytes=1 << 20, resolver=loopback)

    if path == "/plain":
        assert result == web.WebResult(
            True, 200, url, "text/plain",
            f"UNTRUSTED EXTERNAL CONTENT from {url}\nplain page\n"
            "[10 decoded bytes, 10 transferred bytes]\nEND UNTRUSTED EXTERNAL CONTENT",
            10, None)
        assert raw.body == b"plain page"
    elif path == "/redirect":
        final_url = f"http://fetch.test:{fetch_server}/plain"
        assert result == web.WebResult(
            True, 200, final_url, "text/plain",
            f"UNTRUSTED EXTERNAL CONTENT from {final_url}\nplain page\n"
            "[10 decoded bytes, 10 transferred bytes]\nEND UNTRUSTED EXTERNAL CONTENT",
            10, None)
        assert raw.final_url == final_url
        assert raw.redirects == (final_url,)
        assert raw.body == b"plain page"
    elif path == "/gzip":
        assert result.ok and result.text.startswith(
            f"UNTRUSTED EXTERNAL CONTENT from {url}\ncompressed page\n")
        assert result.text.endswith("[15 decoded bytes, 35 transferred bytes]\n"
                                     "END UNTRUSTED EXTERNAL CONTENT")
        assert raw.body == b"compressed page"
        assert raw.headers["content-encoding"] == "gzip"
    else:
        assert result.ok
        assert result.text == (f"UNTRUSTED EXTERNAL CONTENT from {url}\n"
                               "Guide\nRead docs [http://fetch.test:"
                               f"{fetch_server}/docs].\n"
                               "[74 decoded bytes, 74 transferred bytes]\n"
                               "END UNTRUSTED EXTERNAL CONTENT")
        assert raw.body.startswith(b"<h1>Guide</h1>")
        assert b"secret" in raw.body


@pytest.mark.host_sandbox
def test_raw_caps_and_private_address_refusal_preserve_fetch_result(fetch_server):
    url = f"http://fetch.test:{fetch_server}/large"
    policy_value = policy(fetch_server, max_body=4, max_decoded=100)
    raw = web.fetch_raw(url, policy_value, max_bytes=4, resolver=loopback)
    assert raw.error == "response_too_large"
    assert raw.body == b""
    result = web.fetch(url, policy_value, resolver=loopback)
    assert result == web.WebResult(
        False, None, None, None, "", 0, "response body exceeds 4 bytes")

    compressed_url = f"http://fetch.test:{fetch_server}/gzip-large"
    compressed_policy = policy(fetch_server, max_body=10000, max_decoded=1000)
    compressed = web.fetch_raw(compressed_url, compressed_policy, max_bytes=10000,
                               resolver=loopback)
    assert compressed.error == "decoded_response_too_large"
    assert compressed.body == b""
    decoded = web.fetch(compressed_url, compressed_policy, resolver=loopback)
    assert decoded == web.WebResult(
        False, None, None, None, "", 0, "decoded response exceeds 1000 bytes")

    binary_policy = policy(fetch_server, max_body=4, max_decoded=100)
    binary = web.fetch(f"http://fetch.test:{fetch_server}/binary-large", binary_policy,
                       resolver=loopback)
    assert binary == web.WebResult(
        False, None, None, None, "", 0,
        "content type is not readable text: application/pdf")
    encoding = web.fetch(f"http://fetch.test:{fetch_server}/binary-encoding",
                         policy(fetch_server), resolver=loopback)
    assert encoding == web.WebResult(
        False, None, None, None, "", 0,
        "content type is not readable text: application/pdf")

    def private_resolver(*_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 80))]

    private = web.fetch_raw("http://private.test/", web.WebPolicy(()),
                             max_bytes=100, resolver=private_resolver)
    assert private.error == "non_public_address"
    denied = web.fetch("http://private.test/", web.WebPolicy(()),
                       resolver=private_resolver)
    assert denied == web.WebResult(
        False, None, None, None, "", 0,
        "DNS returned a non-public address: 10.0.0.1")


@pytest.mark.host_sandbox
def test_raw_redirect_revalidates_private_destination_and_preserves_refusal(fetch_server):
    url = f"http://fetch.test:{fetch_server}/private-redirect"
    redirect_policy = web.WebPolicy((), allowed_ports=(fetch_server,),
                                    allow_private_for_tests=True)

    def resolver(host, port, *, type):
        address = "127.0.0.1" if host == "fetch.test" else "10.0.0.1"
        return loopback(host, port, type=type) if address == "127.0.0.1" else [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", (address, port))]

    raw = web.fetch_raw(url, redirect_policy, max_bytes=1024, resolver=resolver)
    assert raw.error == "non_public_address"
    assert raw.redirects == ("http://private.test/secret",)
    denied = web.fetch(url, redirect_policy, resolver=resolver)
    assert denied == web.WebResult(
        False, None, None, None, "", 0,
        "DNS returned a non-public address: 10.0.0.1")


@pytest.mark.host_sandbox
def test_raw_fetch_ignores_proxy_environment(monkeypatch, fetch_server):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    url = f"http://fetch.test:{fetch_server}/plain"
    raw = web.fetch_raw(url, policy(fetch_server), max_bytes=100, resolver=loopback)
    assert raw.error is None
    assert raw.body == b"plain page"
