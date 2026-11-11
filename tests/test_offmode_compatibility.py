"""Preserve ordinary source admission and fetch refusals while keeping broker pinning."""
from __future__ import annotations

import io
import json
import socket
from types import SimpleNamespace

import pytest

import kernel_mutate_runtime as mutations
from kernel_request_source_window import bound_runtime_sources
from stateframe.engine import kernel_web_fetch as web
from stateframe.engine import sf_mutate_fetch as broker


@pytest.mark.parametrize("level", [None, "off", "prebuilt", "authored"])
@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("editable", [False, True])
def test_new_output_closure_requires_empty_sources_or_active_mutations(
        tmp_path, monkeypatch, level, installed, editable):
    if level is None:
        monkeypatch.delenv("STATEFRAME_MUTATIONS", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_MUTATIONS", level)
    monkeypatch.setattr(mutations, "_active", object() if installed else None)
    (tmp_path / "source.py").write_text("value = 1\n")
    runtime = SimpleNamespace(
        target=tmp_path / "output/result.txt",
        owner=SimpleNamespace(root=tmp_path, task_row={
            "msg": "Create output/result.txt.", "new_file_roots": ["output"],
            "editable_paths": ["source.py"] if editable else [],
        }),
    )

    result = bound_runtime_sources([], runtime, max_request_bytes=4096)

    assert result == {"changed": False, "windowed": 0, "paths": ()}
    expected = not editable or (level in {"prebuilt", "authored"} and installed)
    assert hasattr(runtime, "_request_source_closure") is expected
    if expected:
        assert runtime._request_source_closure["new_output_target"] == "output/result.txt"


def test_disabled_source_window_never_imports_mutation_runtime(tmp_path, monkeypatch):
    import builtins

    monkeypatch.delenv("STATEFRAME_MUTATIONS", raising=False)
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "kernel_mutate_runtime"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    runtime = SimpleNamespace(
        target=tmp_path / "output/result.txt",
        owner=SimpleNamespace(root=tmp_path, task_row={
            "msg": "Create output/result.txt.", "new_file_roots": ["output"],
        }),
    )
    bound_runtime_sources([], runtime, max_request_bytes=4096)
    assert runtime._request_source_closure["new_output_target"] == "output/result.txt"


@pytest.mark.parametrize("level", ("off", "prebuilt", "authored"))
def test_new_output_closure_propagates_unavailable_editable_paths(tmp_path, monkeypatch, level):
    import kernel_request_source_window as window
    from kernel_obligation_assess import UnavailableEvidence

    monkeypatch.setenv("STATEFRAME_MUTATIONS", level)
    gap = UnavailableEvidence("REQUEST_SOURCE_EDITABLE_PATHS_UNAVAILABLE")
    monkeypatch.setattr(window, "editable_product_paths", lambda _runtime: gap)
    runtime = SimpleNamespace(
        target=tmp_path / "output/result.txt",
        owner=SimpleNamespace(root=tmp_path, task_row={
            "msg": "Create output/result.txt.", "new_file_roots": ["output"],
        }),
    )

    assert bound_runtime_sources([], runtime, max_request_bytes=4096) is gap
    assert not hasattr(runtime, "_request_source_closure")


def _resolver(_host, port, **_kwargs):
    return [(family, socket.SOCK_STREAM, 0, "", (address, port))
            for family, address in ((socket.AF_INET6, "2001:4860:4860::8888"),
                                    (socket.AF_INET, "1.1.1.1"))]


def _sockets(monkeypatch, *, succeed=False, error=OSError("first address refused")):
    attempts, closed = [], []

    class PinnedSocket:
        def __init__(self, family, kind):
            assert kind == socket.SOCK_STREAM
            self.family = family

        def settimeout(self, timeout):
            self.timeout = timeout

        def connect(self, address):
            attempts.append((self.family, address, self.timeout))
            if not succeed or len(attempts) == 1:
                if len(attempts) == 1:
                    raise error
                raise OSError("second address refused")

        def close(self):
            closed.append(self.family)

    def forbidden(*_args, **_kwargs):
        pytest.fail("pinned connections must not resolve the hostname again")

    monkeypatch.setattr(web.socket, "socket", PinnedSocket)
    monkeypatch.setattr(web.socket, "create_connection", forbidden)
    monkeypatch.setattr(web.socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(web.time, "monotonic", lambda: 100.0)
    return attempts, closed


@pytest.mark.parametrize("scheme", ("http", "https"))
@pytest.mark.parametrize("level", ("off", "authored"))
@pytest.mark.parametrize("error", (OSError("first address refused"), TimeoutError()))
def test_ordinary_fetch_keeps_first_ip_and_exact_refusal(monkeypatch, scheme, level, error):
    monkeypatch.setenv("STATEFRAME_MUTATIONS", level)
    attempts, closed = _sockets(monkeypatch, error=error)
    result = web.fetch(f"{scheme}://fetch.test/", web.WebPolicy((), deadline=20),
                       resolver=_resolver)
    assert attempts == [(socket.AF_INET6, ("2001:4860:4860::8888",
                                          443 if scheme == "https" else 80), 20.0)]
    assert closed == [socket.AF_INET6]
    assert result == web.WebResult(False, None, None, None, "", 0,
                                   str(error) or type(error).__name__)


def test_broker_entry_alone_retries_and_keeps_last_refusal(monkeypatch):
    attempts, closed = _sockets(monkeypatch)
    request = {"url": "http://fetch.test/", "accept": None, "max_bytes": 1024,
               "timeout_s": 20, "domains": None}
    output = io.BytesIO()
    assert broker.run_once(io.BytesIO(json.dumps(request).encode()), output,
                           resolver=_resolver) == 0
    expected = [(socket.AF_INET6, ("2001:4860:4860::8888", 80), 10.0),
                (socket.AF_INET, ("1.1.1.1", 80), 20.0)]
    assert attempts == expected
    assert closed == [socket.AF_INET6, socket.AF_INET]
    assert b'"error":"transport_error"' in output.getvalue()
    attempts.clear()
    raw = web.fetch_raw(request["url"], web.WebPolicy(()), max_bytes=1024,
                        broker_role=True, resolver=_resolver)
    assert raw.refusal == "second address refused"
    assert attempts == expected
    attempts.clear()
    raw = web.fetch_raw(request["url"], web.WebPolicy(()), max_bytes=1024,
                        resolver=_resolver)
    assert raw.refusal == "first address refused"
    assert attempts == [(socket.AF_INET6, ("2001:4860:4860::8888", 80), 20.0)]


@pytest.mark.parametrize("connection_class", (web._PinnedHTTP, web._PinnedHTTPS))
def test_pinned_retry_reaches_second_vetted_family(monkeypatch, connection_class):
    attempts, closed = _sockets(monkeypatch, succeed=True)
    conn = connection_class("fetch.test", 443,
                            [(socket.AF_INET6, "2001:4860:4860::8888"),
                             (socket.AF_INET, "1.1.1.1")], 120.0)
    connected = conn._connect_socket()
    assert connected.family == socket.AF_INET
    assert attempts == [(socket.AF_INET6, ("2001:4860:4860::8888", 443), 10.0),
                        (socket.AF_INET, ("1.1.1.1", 443), 20.0)]
    assert closed == [socket.AF_INET6]
    connected.close()


@pytest.mark.parametrize("broker_role", (False, True))
def test_all_dns_answers_are_vetted_before_any_connection(monkeypatch, broker_role):
    attempts, _closed = _sockets(monkeypatch)

    def mixed_answers(host, port, **kwargs):
        return _resolver(host, port, **kwargs) + [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", port))]

    raw = web.fetch_raw("http://fetch.test/", web.WebPolicy(()), max_bytes=1024,
                        broker_role=broker_role, resolver=mixed_answers)
    assert raw.refusal == "DNS returned a non-public address: 127.0.0.1"
    assert attempts == []


@pytest.mark.parametrize("candidate_body,expected", (
    (b'{"a":1,"b":2}', 0),
    (b'{"b":2,"a":1}', 1),
))
def test_parity_command_compares_raw_bytes_and_exercises_resume(
        tmp_path, monkeypatch, candidate_body, expected):
    from tools import compare_mutation_off_requests as parity

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    calls = []

    def capture(tree, python, launcher, example, repo, run_dir, *, stable_tool_head, resume):
        calls.append((tree, example, resume))
        run_dir.mkdir()
        body = b'{"a":1,"b":2}' if tree == baseline else candidate_body
        return body, [["read"]], {}

    monkeypatch.setattr(parity, "capture", capture)
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(parity.sys, "argv", ["compare", "--baseline", str(baseline),
        "--candidate", str(candidate), "--work-dir", str(tmp_path / "captures")])
    assert parity.main() == expected
    assert calls == [(tree, example, resume)
                     for example, resume in [(name, False) for name in parity.EXAMPLES]
                     + [("discount_bug", True)]
                     for tree in (baseline, candidate)]


def test_capture_environment_prevents_source_tree_bytecode_writes(tmp_path, monkeypatch):
    from tools import compare_mutation_off_requests as parity

    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    assert parity.source_env(tmp_path, stable_tool_head="on")["PYTHONDONTWRITEBYTECODE"] == "1"
