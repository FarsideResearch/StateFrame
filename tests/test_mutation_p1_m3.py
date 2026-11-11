"""Broker transport regressions for IPv6-only DNS answers."""

import io
import json
import socket
import threading
from types import SimpleNamespace

import pytest

from stateframe.engine.kernel_exec_capability import ExecLimits
from stateframe.engine.kernel_mutate_exec import _broker_config_roots, _fixed_env_capability
from stateframe.engine.sf_mutate_fetch import run_once


def _request(url: str) -> io.BytesIO:
    return io.BytesIO(json.dumps({
        "url": url, "accept": "text/plain", "max_bytes": 1024,
        "timeout_s": 3, "domains": ["example.test"],
    }).encode())


def _response(data: bytes) -> tuple[dict, bytes]:
    header, payload = data.split(b"\n", 1)
    signature, metadata_size, body_size = header.split()
    assert signature == b"SFMB1"
    metadata_length, body_length = int(metadata_size), int(body_size)
    metadata = json.loads(payload[:metadata_length])
    body = payload[metadata_length:]
    assert len(body) == body_length
    return metadata, body


def test_broker_fetch_ipv6_only_answer_connects_without_second_dns_lookup(monkeypatch):
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as listener:
        listener.bind(("::1", 0))
        listener.listen(1)
        listener.settimeout(3)
        port = listener.getsockname()[1]

        def serve() -> None:
            try:
                client, _ = listener.accept()
            except socket.timeout:
                return
            with client:
                client.settimeout(3)
                client.recv(4096)
                client.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n"
                               b"Content-Type: text/plain\r\nConnection: close\r\n\r\nhello")

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        def only_domain(host, requested_port, *_args, **_kwargs):
            if host != "example.test":
                raise socket.gaierror("numeric address was resolved again")
            return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "",
                     ("::1", requested_port, 0, 0))]

        monkeypatch.setattr(socket, "getaddrinfo", only_domain)
        output = io.BytesIO()
        assert run_once(_request(f"http://example.test:{port}/"), output,
                        allow_private_for_tests=True, resolver=only_domain) == 0
        metadata, body = _response(output.getvalue())
        thread.join(timeout=3)
        assert metadata["status"] == 200, metadata
        assert metadata["error"] is None, metadata
        assert body == b"hello"


@pytest.mark.parametrize("address", ["::1", "fe80::1", "fd00::1"])
def test_broker_fetch_refuses_nonpublic_ipv6(address):
    def private_answer(_host, port, **_kwargs):
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "",
                 (address, port, 0, 0))]

    output = io.BytesIO()
    assert run_once(_request("https://example.test/"), output,
                    resolver=private_answer) == 0
    metadata, body = _response(output.getvalue())
    assert metadata["status"] is None
    assert metadata["error"] == "non_public_address"
    assert body == b""


def test_broker_ca_mounts_include_hash_symlink_targets(tmp_path, monkeypatch):
    certs = tmp_path / "certs"
    certs.mkdir()
    targets = tmp_path / "trust" / "vendor"
    targets.mkdir(parents=True)
    certificate = targets / "root.crt"
    certificate.write_text("certificate")
    (certs / "12345678.0").symlink_to(certificate)
    monkeypatch.setattr("stateframe.engine.kernel_mutate_exec.ssl.get_default_verify_paths",
                        lambda: SimpleNamespace(cafile=None, capath=str(certs),
                                                openssl_cafile=str(tmp_path / "missing.pem"),
                                                openssl_capath=None))

    roots = _broker_config_roots()

    assert certs in roots
    assert certificate in roots
    assert certs / "12345678.0" not in roots
    assert tmp_path / "missing.pem" not in roots


def test_broker_ca_mounts_include_effective_and_compiled_paths_only(tmp_path, monkeypatch):
    effective_file = tmp_path / "effective.pem"
    compiled_file = tmp_path / "compiled.pem"
    env_file = tmp_path / "env.pem"
    effective_dir = tmp_path / "effective-certs"
    compiled_dir = tmp_path / "compiled-certs"
    env_dir = tmp_path / "env-certs"
    second_env_dir = tmp_path / "other-env-certs"
    for path in (effective_file, compiled_file, env_file):
        path.write_text("certificate")
    for path in (effective_dir, compiled_dir, env_dir, second_env_dir):
        path.mkdir()
    monkeypatch.setattr("stateframe.engine.kernel_mutate_exec.ssl.get_default_verify_paths",
                        lambda: SimpleNamespace(cafile=str(effective_file), capath=str(effective_dir),
                                                openssl_cafile=str(compiled_file),
                                                openssl_capath=str(compiled_dir)))
    monkeypatch.setenv("SSL_CERT_FILE", str(env_file))
    monkeypatch.setenv("SSL_CERT_DIR", str(env_dir))

    roots = _broker_config_roots()

    assert {effective_file, compiled_file, effective_dir, compiled_dir}.issubset(roots)
    assert {env_file, env_dir, second_env_dir}.isdisjoint(roots)


def test_broker_ignores_ambient_ca_env_even_if_path_is_admitted(tmp_path, monkeypatch):
    certificate = tmp_path / "custom.pem"
    certificate.write_text("certificate")
    monkeypatch.setenv("SSL_CERT_FILE", str(certificate))

    broker = _fixed_env_capability("mutation_broker", tmp_path, (certificate,),
                                   "full", ExecLimits())
    unadmitted = _fixed_env_capability("mutation_broker", tmp_path, (),
                                      "full", ExecLimits())
    payload = _fixed_env_capability("mutation", tmp_path, (), "none", ExecLimits())

    assert broker.env == {"LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
                          "HOME": "/nonexistent", "TZ": "UTC"}
    assert broker.write_roots == ()
    assert "SSL_CERT_FILE" not in unadmitted.env
    assert "SSL_CERT_FILE" not in payload.env
