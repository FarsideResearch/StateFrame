"""Real bounded HTTPS child: local CA trust and certificate rejection."""

import json
import shutil
import ssl
import subprocess
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stateframe import _child
from stateframe.engine import kernel_bounded_http as http
from _timeouts import scaled


BODY = b'{"tls":"verified"}'
WIRE = b'{"model":"local","messages":[]}'
pytestmark = pytest.mark.host_sandbox


def _openssl(*args: str) -> None:
    subprocess.run(
        ["openssl", *args], check=True, capture_output=True, text=True, timeout=scaled(20))


# Every extension is explicit: Python 3.13+ default contexts set VERIFY_X509_STRICT, which needs
# Subject/Authority Key Identifiers, and LibreSSL (macOS /usr/bin/openssl) does not add them by default.
_CA_CONFIG = """[req]
distinguished_name = dn
x509_extensions = v3_ca
prompt = no
[dn]
CN = {name}
[v3_ca]
basicConstraints = critical,CA:TRUE
keyUsage = critical,keyCertSign,cRLSign
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always
"""
_LEAF_EXTENSIONS = """basicConstraints = CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = {san}
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer
"""


def _key_identifiers(cert) -> tuple[bool, bool]:
    text = subprocess.run(["openssl", "x509", "-in", str(cert), "-noout", "-text"],
                          check=True, capture_output=True, text=True, timeout=scaled(20)).stdout
    return "Subject Key Identifier" in text, "Authority Key Identifier" in text


def _make_ca(directory, name: str):
    key, cert, config = (directory / f"{name}.{suffix}" for suffix in ("key", "pem", "cnf"))
    config.write_text(_CA_CONFIG.format(name=name))
    _openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-sha256",
             "-config", str(config), "-keyout", str(key), "-out", str(cert))
    return key, cert


def _make_server_cert(directory, name: str, ca_key, ca_cert, san: str):
    key, csr, cert = (directory / f"{name}.{suffix}" for suffix in ("key", "csr", "pem"))
    ext = directory / f"{name}.ext"
    ext.write_text(_LEAF_EXTENSIONS.format(san=san))
    _openssl("req", "-new", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN={name}",
             "-keyout", str(key), "-out", str(csr))
    _openssl("x509", "-req", "-sha256", "-in", str(csr), "-CA", str(ca_cert),
             "-CAkey", str(ca_key), "-set_serial", "1", "-days", "1",
             "-extfile", str(ext), "-out", str(cert))
    return key, cert


@pytest.fixture(scope="module")
def certificates(tmp_path_factory):
    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI is required for local HTTPS certificate tests")
    directory = tmp_path_factory.mktemp("bounded-https")
    trusted_key, trusted_ca = _make_ca(directory, "trusted-ca")
    untrusted_key, untrusted_ca = _make_ca(directory, "untrusted-ca")
    good = _make_server_cert(directory, "good", trusted_key, trusted_ca,
                             "IP:127.0.0.1,DNS:localhost")
    wrong_host = _make_server_cert(directory, "wrong-host", trusted_key, trusted_ca,
                                   "DNS:wrong.example")
    untrusted = _make_server_cert(directory, "untrusted", untrusted_key, untrusted_ca,
                                  "IP:127.0.0.1,DNS:localhost")
    # Self-check: a CLI that drops key identifiers must fail here, not as a confusing TLS error.
    for cert in (trusted_ca, untrusted_ca, good[1], wrong_host[1], untrusted[1]):
        assert _key_identifiers(cert) == (True, True), f"{cert.name} lacks SKI/AKI"
    return trusted_ca, good, wrong_host, untrusted


@pytest.fixture
def https_server(request, certificates):
    _, good, wrong_host, untrusted = certificates
    key, cert = {"good": good, "wrong_host": wrong_host,
                 "untrusted": untrusted}[request.param]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert, keyfile=key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://127.0.0.1:{server.server_port}/v1/chat/completions"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=scaled(5))


def _send(endpoint: str):
    return http.request_bytes(WIRE, endpoint, "", deadline=time.monotonic() + scaled(10),
                              socket_timeout=scaled(5))


def test_default_handler_effective_context_verifies_host_and_ca():
    """The context handed to the connection is our own verifying one, on every Python."""
    import urllib.request

    handler = http._AbortiveHTTPSHandler(http._AbortiveConnectionTracker())
    seen = {}

    def capture(factory, request, **kwargs):
        seen.update(kwargs)
        return None

    handler.do_open = capture
    handler.https_open(urllib.request.Request("https://localhost/v1/chat/completions"))
    assert set(seen) == {"context"}  # no check_hostname: removed from http.client in 3.12
    context = seen["context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED


@pytest.mark.parametrize("https_server", ["good"], indirect=True)
def test_trusted_matching_certificate_returns_exact_bytes(monkeypatch, certificates, https_server):
    monkeypatch.setenv("SSL_CERT_FILE", str(certificates[0]))
    assert _send(https_server) == BODY


@pytest.mark.parametrize("https_server, expected", [
    ("wrong_host", "mismatch"), ("untrusted", "certificate"),
], indirect=["https_server"])
def test_invalid_certificate_is_refused(monkeypatch, certificates, https_server, expected):
    monkeypatch.setenv("SSL_CERT_FILE", str(certificates[0]))
    with pytest.raises(http.BoundedTransportError) as failure:
        _send(https_server)
    assert failure.value.error_type == "URLError"
    assert failure.value.error_detail is not None
    assert expected in failure.value.error_detail.lower()
    assert "https://" not in failure.value.error_detail
    assert failure.value.error_detail in str(failure.value)


def test_transport_detail_redacts_urls_credentials_queries_and_length():
    raw = ("connection to https://alice:pw@example.com/path?access_token=hidden "
           "failed; user:secret@host query?key=secret Authorization: Bearer bearer-secret "
           "Authorization: Basic basic-secret api_key=abc12345 key=known-secret " + "x" * 500)
    detail = http.safe_transport_detail(urllib.error.URLError(raw), "known-secret")
    assert detail is not None and len(detail) <= 300
    for secret in ("example.com", "alice", "pw", "hidden", "secret", "bearer-secret",
                   "basic-secret",
                   "abc12345", "known-secret", "access_token=", "?key="):
        assert secret not in detail
    assert "[redacted URL]" in detail


def test_parent_resanitizes_child_detail_with_endpoint_key():
    diagnostic = json.dumps({"error_type": "URLError", "error_detail":
                             "failed https://alice:pw@host.example/path?key=bad "
                             "Authorization: Basic abc123 key=known-secret"}).encode()
    error = http._child_transport_error(diagnostic, "known-secret", "https://host.example")
    assert isinstance(error, http.BoundedTransportError)
    assert error.error_detail is not None
    for secret in ("host.example", "alice", "pw", "?key=", "abc123", "known-secret"):
        assert secret not in str(error)


@pytest.mark.parametrize("https_server", ["wrong_host"], indirect=True)
def test_verification_reason_reaches_wire_error_and_send_event(
        monkeypatch, certificates, https_server, tmp_path, capsys):
    import kernel_bounded_http as child_http

    monkeypatch.setenv("SSL_CERT_FILE", str(certificates[0]))
    monkeypatch.setattr(child_http, "request_bytes", child_http.request_bytes)
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    original = child_http.request_bytes
    try:
        _child.install_wire_recorder(tmp_path, https_server)
        with pytest.raises(child_http.BoundedTransportError) as failure:
            child_http.request_bytes(WIRE, https_server, "", deadline=time.monotonic() + scaled(10),
                                     socket_timeout=scaled(5))
    finally:
        child_http.request_bytes = original
    detail = failure.value.error_detail
    assert detail is not None and "mismatch" in detail.lower()
    assert (tmp_path / "wire/0001.error.txt").read_text() == f"BoundedTransportError: {detail}\n"
    events = [json.loads(line.removeprefix("[sf] "))
              for line in capsys.readouterr().out.splitlines()
              if line.startswith("[sf] ")]
    assert any(event.get("kind") == "send_error" and event.get("detail") == detail
               for event in events)


def test_plain_transport_https_handler_uses_its_own_verifying_context():
    """kernel_plain_http had the same removed-attribute read; it must hand over only its own context."""
    import urllib.request

    import kernel_plain_http as plain

    handler = plain._TrackedHTTPSHandler(plain.AbortiveHTTPTransport(), 1)
    seen = {}

    def capture(factory, request, **kwargs):
        seen.update(kwargs)
        return None

    handler.do_open = capture
    handler.https_open(urllib.request.Request("https://localhost/v1/chat/completions"))
    assert set(seen) == {"context"}
    assert seen["context"].check_hostname is True
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED


def test_constructing_handlers_loads_no_tls_context(monkeypatch):
    """A plain-http send must not pay a CA-store load (~80 ms on 3.11 under load) for an https handler."""
    import kernel_plain_http as plain

    built = []
    real = ssl.create_default_context
    monkeypatch.setattr(ssl, "create_default_context", lambda *a, **k: built.append(1) or real(*a, **k))
    http._AbortiveHTTPSHandler(http._AbortiveConnectionTracker())
    plain._TrackedHTTPSHandler(plain.AbortiveHTTPTransport(), 1)
    assert built == []
