import gzip
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stateframe.engine.kernel_web_fetch import WebPolicy, fetch, handle_web_call, public_address


@pytest.mark.parametrize("address,expected", [
    ("127.0.0.1", False), ("127.255.255.255", False), ("0.0.0.0", False),
    ("10.0.0.1", False), ("10.255.255.255", False), ("172.16.0.1", False),
    ("172.31.255.255", False), ("192.168.0.1", False), ("192.168.255.255", False),
    ("169.254.169.254", False), ("169.254.0.1", False), ("100.64.0.1", False),
    ("100.127.255.255", False), ("224.0.0.1", False), ("239.255.255.255", False),
    ("240.0.0.1", False), ("255.255.255.255", False), ("192.0.0.1", False),
    ("198.18.0.1", False), ("198.51.100.1", False), ("203.0.113.1", False),
    ("8.8.8.8", True), ("1.1.1.1", True), ("9.9.9.9", True),
    ("::", False), ("::1", False), ("fe80::1", False), ("fc00::1", False),
    ("fd00:ec2::254", False), ("ff02::1", False), ("2001:db8::1", False),
    ("::ffff:127.0.0.1", False), ("::ffff:8.8.8.8", True),
    ("2002:0a00:0001::", False), ("2002:0808:0808::", False),
    ("64:ff9b::a9fe:a9fe", False), ("64:ff9b::808:808", False),
    ("64:ff9b:1::7f00:1", False), ("2001:4860:4860::8888", True),
    ("0x7f.1", False), ("2130706433", False), ("[::ffff:127.0.0.1]", False),
    ("2001::1", False), ("fec0::1", False), ("192.0.2.1", False),
])
def test_address_classification(address, expected):
    assert public_address(address) is expected


def resolution(*ips):
    def resolver(host, port, *, type):
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM,
                 0, "", (ip, port)) for ip in ips]
    return resolver


def test_mixed_dns_refused_before_connection():
    result = fetch("http://example.com/", WebPolicy(()), resolver=resolution("8.8.8.8", "127.0.0.1"))
    assert not result.ok and "non-public" in result.refusal


def test_dns_resolution_obeys_total_deadline():
    def slow_resolver(host, port, *, type):
        time.sleep(0.2)
        return resolution("8.8.8.8")(host, port, type=type)

    start = time.monotonic()
    result = fetch("http://example.com/", WebPolicy((), deadline=0.02), resolver=slow_resolver)
    assert not result.ok and "deadline" in result.refusal
    assert time.monotonic() - start < 0.1


def test_userinfo_ports_schemes_and_allow_list():
    cases = ["http://user:pw@example.com/", "ftp://example.com/x", "http://example.com:81/",
             "http://example.com.evil.test/"]
    policy = WebPolicy(("example.com",))
    for url in cases:
        result = fetch(url, policy, resolver=resolution("8.8.8.8"))
        assert not result.ok
    result = fetch("http://127.0.0.1/", WebPolicy(()), resolver=resolution("127.0.0.1"))
    assert not result.ok and "non-public" in result.refusal


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/slow-headers":
            try:
                self.request.sendall(b"HTTP/1.1 200 OK\r\n")
                for byte in b"Content-Type: text/plain\r\n\r\n":
                    self.request.sendall(bytes([byte]))
                    time.sleep(0.2)
            except BrokenPipeError:
                pass
            return
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://private.test/secret")
            self.end_headers()
            return
        if self.path == "/gzip":
            body = gzip.compress(b"A" * 100_000)
            content_type = "text/plain"
            extra = {"Content-Encoding": "gzip"}
        elif self.path == "/html":
            body = (b"<h1>Guide</h1><script>ignore me</script><p>Read <a href='/docs'>docs</a>.</p>"
                    b"<style>hide</style><h2>Next</h2>")
            content_type = "text/html; charset=utf-8"
            extra = {}
        elif self.path == "/slow":
            body = b"slow"
            content_type = "text/plain"
            extra = {}
        else:
            body = b"%PDF-1.7"
            content_type = "application/pdf"
            extra = {}
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in extra.items():
            self.send_header(key, value)
        self.end_headers()
        if self.path == "/slow":
            for byte in body:
                try:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(0.2)
                except BrokenPipeError:
                    break
        else:
            self.wfile.write(body)

    def log_message(self, *_):
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


@pytest.mark.host_sandbox
def test_redirect_revalidates_private_destination(server):
    def resolver(host, port, *, type):
        address = "127.0.0.1" if host == "public.test" else "10.0.0.1"
        return resolution(address)(host, port, type=type)
    # The test hook permits loopback transport only. Redirected private DNS is
    # rejected before attempting a second connection.
    policy = WebPolicy((), allowed_ports=(server,), allow_private_for_tests=True)
    result = fetch(f"http://public.test:{server}/redirect", policy, resolver=resolver)
    assert not result.ok and "non-public" in result.refusal


@pytest.mark.host_sandbox
def test_gzip_bomb_binary_and_html(server):
    policy = WebPolicy(("public.test",), max_decoded=1000, allowed_ports=(server,),
                       allow_private_for_tests=True)
    url = f"http://public.test:{server}"
    result = fetch(url + "/gzip", policy, resolver=resolution("127.0.0.1"))
    assert not result.ok and "decoded response exceeds" in result.refusal
    result = fetch(url + "/binary", policy, resolver=resolution("127.0.0.1"))
    assert not result.ok and "content type" in result.refusal
    result = fetch(url + "/html", policy, resolver=resolution("127.0.0.1"))
    assert result.ok and "Guide" in result.text and "Next" in result.text
    assert f"docs [{url}/docs]" in result.text
    assert "ignore me" not in result.text and "hide" not in result.text
    assert result.text.startswith("UNTRUSTED EXTERNAL CONTENT")
    assert result.text.endswith("END UNTRUSTED EXTERNAL CONTENT")


@pytest.mark.host_sandbox
def test_total_deadline_catches_slow_drip(server):
    policy = WebPolicy(("public.test",), deadline=0.25, allowed_ports=(server,),
                       allow_private_for_tests=True)
    start = time.monotonic()
    result = fetch(f"http://public.test:{server}/slow", policy,
                   resolver=resolution("127.0.0.1"))
    assert not result.ok and time.monotonic() - start < 0.8
    start = time.monotonic()
    result = fetch(f"http://public.test:{server}/slow-headers", policy,
                   resolver=resolution("127.0.0.1"))
    assert not result.ok and time.monotonic() - start < 0.8


def test_handler_refusal():
    text, fields = handle_web_call(WebPolicy(()), {"url": 5})
    assert not fields["ok"] and "refused" in text
