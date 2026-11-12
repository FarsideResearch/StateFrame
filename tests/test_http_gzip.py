"""Request compression compatibility with ordinary OpenAI-compatible servers."""

import gzip
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from _timeouts import scaled

from stateframe.engine import kernel_bounded_http
from stateframe.engine.kernel_plain_host import _PlainHTTPProvider, encode_request_body
from stateframe import _child
from stateframe.replay_tape import Replayer, build as build_replay_tape
from stateframe.scripted_endpoint import Script, make_handler
from engine_run import run_monitored


ROOT = Path(__file__).resolve().parents[1]
WIRE = json.dumps({"model": "scripted", "messages": [
    {"role": "user", "content": "x" * 40_000}]}).encode()


@pytest.fixture
def transport_server():
    seen = []
    accepts_gzip = {"value": False}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            encoding = self.headers.get("Content-Encoding")
            seen.append((encoding, body, self.headers.get("X-Request-Body-SHA256")))
            if encoding is None and accepts_gzip.get("trickle_identity_headers"):
                self.wfile.write(b"HTTP/1.1 200 ")
                self.wfile.flush()
                time.sleep(1.5)
                self.close_connection = True
                return
            if encoding is None and accepts_gzip.get("close_on_identity"):
                self.close_connection = True
                return
            if encoding == "gzip" and not accepts_gzip["value"]:
                status, reply = 415, b'{"error":"unsupported encoding"}'
            elif encoding is None and accepts_gzip.get("reject_identity") and len(seen) <= 2:
                status, reply = 415, b'{"error":"unsupported identity"}'
            else:
                status, reply = 200, b'{"ok":true}'
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", seen, accepts_gzip
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=scaled(5))


@pytest.mark.host_sandbox
def test_default_large_request_to_server_refusing_gzip(monkeypatch, transport_server):
    # The old default compressed this 40 KB request and received HTTP 415.
    monkeypatch.delenv("STATEFRAME_HTTP_GZIP", raising=False)
    endpoint, seen, _ = transport_server
    reply = kernel_bounded_http.request_bytes(
        WIRE, endpoint, "", deadline=time.monotonic() + scaled(10), socket_timeout=scaled(5))
    assert reply == b'{"ok":true}'
    assert seen == [(None, WIRE, hashlib.sha256(WIRE).hexdigest())]


@pytest.mark.host_sandbox
def test_gzip_415_resends_same_identity_once(monkeypatch, transport_server):
    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "on")
    endpoint, seen, _ = transport_server
    reply = kernel_bounded_http.request_bytes(
        WIRE, endpoint, "", deadline=time.monotonic() + scaled(10), socket_timeout=scaled(5))
    assert reply == b'{"ok":true}'
    assert [entry[0] for entry in seen] == ["gzip", None]
    assert gzip.decompress(seen[0][1]) == seen[1][1] == WIRE
    assert [entry[2] for entry in seen] == [hashlib.sha256(WIRE).hexdigest()] * 2


@pytest.mark.host_sandbox
def test_gzip_415_identity_queue_wait_uses_resend_first_byte(monkeypatch):
    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "on")
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", "0.6")
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "3")
    seen: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            encoding = self.headers.get("Content-Encoding")
            seen.append(encoding)
            if encoding == "gzip":
                status, body = 415, b"unsupported"
            else:
                time.sleep(1.2)
                status, body = 200, b'{"ok":true}'
            try:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        waits = []
        reply = kernel_bounded_http.request_bytes(
            WIRE, f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            "", deadline=started + 0.5, run_deadline=started + 3,
            socket_timeout=0.5, record_queue_wait=lambda *values: waits.append(values))
        assert reply == b'{"ok":true}'
        assert seen == ["gzip", None]
        assert len(waits) == 1
        assert waits[0][0] >= 0.6
        assert waits[0][1] == 401
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=scaled(5))


@pytest.mark.host_sandbox
def test_gzip_415_identity_partial_headers_keep_original_deadline(
        monkeypatch, transport_server):
    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "on")
    monkeypatch.setenv("STATEFRAME_PROVIDER_LIVENESS_S", "0.6")
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "3")
    endpoint, seen, settings = transport_server
    settings["trickle_identity_headers"] = True
    started = time.monotonic()
    with pytest.raises((TimeoutError, kernel_bounded_http.BoundedTransportError)):
        kernel_bounded_http.request_bytes(
            WIRE, endpoint, "", deadline=started + 0.65,
            run_deadline=started + 3, socket_timeout=0.65)
    assert time.monotonic() - started < 1.2
    assert [entry[0] for entry in seen] == ["gzip", None]


@pytest.mark.host_sandbox
def test_identity_415_is_not_resent(monkeypatch, transport_server):
    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "off")
    endpoint, seen, settings = transport_server
    settings["reject_identity"] = True
    with pytest.raises(kernel_bounded_http.BoundedTransportError) as failure:
        kernel_bounded_http.request_bytes(
            WIRE, endpoint, "", deadline=time.monotonic() + scaled(10), socket_timeout=scaled(5))
    assert failure.value.http_status == 415
    assert [entry[0] for entry in seen] == [None]


@pytest.mark.host_sandbox
def test_gzip_then_identity_415_stops_after_one_resend(monkeypatch, transport_server):
    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "on")
    endpoint, seen, settings = transport_server
    settings["reject_identity"] = True
    with pytest.raises(kernel_bounded_http.BoundedTransportError) as failure:
        kernel_bounded_http.request_bytes(
            WIRE, endpoint, "", deadline=time.monotonic() + scaled(10), socket_timeout=scaled(5))
    assert failure.value.http_status == 415
    assert [entry[0] for entry in seen] == ["gzip", None]
    assert gzip.decompress(seen[0][1]) == seen[1][1] == WIRE


@pytest.mark.host_sandbox
def test_plain_provider_gzip_415_resends_identity(monkeypatch, transport_server):
    monkeypatch.setenv("STATEFRAME_PLAIN_GZIP", "on")
    endpoint, seen, _ = transport_server
    # The fake server is not llama.cpp, so the provider sends strict OpenAI chat bodies; use one so the body stays above the gzip threshold.
    reply = _PlainHTTPProvider(endpoint, "")(
        {"messages": [{"role": "user", "content": "x" * 40_000}]}, timeout=scaled(5))
    assert reply == {"ok": True}
    assert [entry[0] for entry in seen] == ["gzip", None]
    assert gzip.decompress(seen[0][1]) == seen[1][1]
    assert seen[0][2] == seen[1][2] == hashlib.sha256(seen[1][1]).hexdigest()


@pytest.mark.host_sandbox
def test_wire_recorder_captures_both_physical_attempts(monkeypatch, tmp_path, transport_server):
    import kernel_bounded_http as child_http

    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "on")
    endpoint, seen, _ = transport_server
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(child_http, "request_bytes", child_http.request_bytes)
    _child.install_wire_recorder(tmp_path, endpoint)
    assert child_http.request_bytes(
        WIRE, endpoint, "", deadline=time.monotonic() + scaled(10),
        socket_timeout=scaled(5)) == b'{"ok":true}'
    record = json.loads((tmp_path / "sends.jsonl").read_text())
    assert [attempt["content_encoding"] for attempt in record["transport_attempts"]] == ["gzip", None]
    assert [attempt["http_status"] for attempt in record["transport_attempts"]] == [415, 200]
    assert [attempt["body_sha256"] for attempt in record["transport_attempts"]] == [
        hashlib.sha256(entry[1]).hexdigest() for entry in seen]
    assert record["request_sha256"] == hashlib.sha256(WIRE).hexdigest()
    assert (tmp_path / "wire/0001.request.json").read_bytes() == WIRE
    assert (tmp_path / "wire/0001.response.json").read_bytes() == b'{"ok":true}'


@pytest.mark.host_sandbox
def test_wire_recorder_keeps_refused_resend_evidence_on_failure(monkeypatch, tmp_path, transport_server):
    import kernel_bounded_http as child_http

    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "on")
    endpoint, seen, settings = transport_server
    settings["reject_identity"] = True
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(child_http, "request_bytes", child_http.request_bytes)
    _child.install_wire_recorder(tmp_path, endpoint)
    with pytest.raises(child_http.BoundedTransportError) as failure:
        child_http.request_bytes(
            WIRE, endpoint, "", deadline=time.monotonic() + scaled(10), socket_timeout=scaled(5))
    assert failure.value.http_status == 415
    assert [entry[0] for entry in seen] == ["gzip", None]
    record = json.loads((tmp_path / "sends.jsonl").read_text())
    assert [attempt["http_status"] for attempt in record["transport_attempts"]] == [415, 415]
    assert record["request_sha256"] == hashlib.sha256(WIRE).hexdigest()
    assert (tmp_path / "wire/0001.error.txt").read_text() == (
        f"BoundedTransportError: {failure.value.error_detail}\n")
    assert build_replay_tape(tmp_path / "wire", tmp_path / "sends.jsonl") == []


@pytest.mark.host_sandbox
def test_wire_recorder_keeps_resend_when_server_closes_without_response(
        monkeypatch, tmp_path, transport_server):
    import kernel_bounded_http as child_http

    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "on")
    endpoint, seen, settings = transport_server
    settings["close_on_identity"] = True
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(child_http, "request_bytes", child_http.request_bytes)
    _child.install_wire_recorder(tmp_path, endpoint)
    with pytest.raises(child_http.BoundedTransportError) as failure:
        child_http.request_bytes(
            WIRE, endpoint, "", deadline=time.monotonic() + scaled(10), socket_timeout=scaled(5))

    assert [entry[0] for entry in seen] == ["gzip", None]
    assert gzip.decompress(seen[0][1]) == seen[1][1] == WIRE
    assert [entry[2] for entry in seen] == [hashlib.sha256(WIRE).hexdigest()] * 2
    assert failure.value.error_type == "RemoteDisconnected"
    assert failure.value.http_status is None
    record = json.loads((tmp_path / "sends.jsonl").read_text())
    attempts = record["transport_attempts"]
    assert [attempt["http_status"] for attempt in attempts] == [415, None]
    assert [attempt["content_encoding"] for attempt in attempts] == ["gzip", None]
    assert [attempt["body_bytes"] for attempt in attempts] == [len(entry[1]) for entry in seen]
    assert [attempt["body_sha256"] for attempt in attempts] == [
        hashlib.sha256(entry[1]).hexdigest() for entry in seen]
    assert record["error"] == "BoundedTransportError"
    assert (tmp_path / "wire/0001.error.txt").read_text() == (
        f"BoundedTransportError: {failure.value.error_detail}\n")


@pytest.mark.host_sandbox
def test_wire_recorder_normal_record_keeps_022_shape(monkeypatch, tmp_path, transport_server):
    from types import SimpleNamespace

    import kernel_bounded_http as child_http
    from kernel_measurement import ACTIVE_REQUEST

    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "off")
    endpoint, _, _ = transport_server
    original = child_http.request_bytes
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(child_http, "request_bytes", original)
    token = ACTIVE_REQUEST.set(SimpleNamespace(
        _final_cache_layout_record={"final_layout": "plain", "stable_wire": "off"}))
    try:
        _child.install_wire_recorder(tmp_path, endpoint)
        assert child_http.request_bytes(
            WIRE, endpoint, "", deadline=time.monotonic() + scaled(10),
            socket_timeout=scaled(5)) == b'{"ok":true}'
    finally:
        ACTIVE_REQUEST.reset(token)
    line = (tmp_path / "sends.jsonl").read_text()
    record = json.loads(line)
    _assert_normal_send_keeps_022_shape(line)
    assert "final_layout" not in record
    assert "final_layout_reason" not in record
    assert "stable_wire" not in record
    assert "stable_wire_reason" not in record

    import kernel_cache_layout_candidate as candidate

    monkeypatch.setattr(candidate, "_REPLAY_LAYOUT_DECISION_PROVIDER", None, raising=False)
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", None, raising=False)
    monkeypatch.setattr(child_http, "request_bytes", original)
    replay_wire = tmp_path / "replay-wire"
    replay_wire.mkdir()
    (replay_wire / "0001.request.json").write_bytes(
        (tmp_path / "wire/0001.request.json").read_bytes())
    (replay_wire / "0001.response.json").write_text(
        json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}))
    tape = build_replay_tape(replay_wire, tmp_path / "sends.jsonl")
    resume_run = tmp_path / "resume-run"
    resume_run.mkdir()
    _child.install_wire_recorder(resume_run, endpoint, Replayer(tape))
    assert candidate.replay_final_layout_decision() == "plain"


def _assert_normal_send_keeps_022_shape(line: str) -> None:
    record = json.loads(line)
    expected = {"send": 1, "request_bytes": len(WIRE),
                "request_sha256": hashlib.sha256(WIRE).hexdigest(),
                "messages": 1, "seconds": record["seconds"],
                "prompt_tokens": None, "completion_tokens": None,
                "cached_tokens": None, "actions": []}
    assert line == json.dumps(expected) + "\n"


def test_022_send_shape_pin_rejects_planted_plain_layout_key():
    record = {"send": 1, "request_bytes": len(WIRE),
              "request_sha256": hashlib.sha256(WIRE).hexdigest(),
              "messages": 1, "seconds": 0.25,
              "prompt_tokens": None, "completion_tokens": None,
              "cached_tokens": None, "actions": []}
    _assert_normal_send_keeps_022_shape(json.dumps(record) + "\n")
    planted = {**record, "final_layout": "plain"}
    with pytest.raises(AssertionError):
        _assert_normal_send_keeps_022_shape(json.dumps(planted) + "\n")
    planted_stable_wire = {**record, "stable_wire": "off"}
    with pytest.raises(AssertionError):
        _assert_normal_send_keeps_022_shape(json.dumps(planted_stable_wire) + "\n")


@pytest.mark.host_sandbox
def test_wire_recorder_keeps_failed_open_layout_reason(monkeypatch, tmp_path, transport_server):
    from types import SimpleNamespace

    import kernel_bounded_http as child_http
    from kernel_measurement import ACTIVE_REQUEST

    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "off")
    endpoint, _, _ = transport_server
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(child_http, "request_bytes", child_http.request_bytes)
    token = ACTIVE_REQUEST.set(SimpleNamespace(_final_cache_layout_record={
        "final_layout": "failed_open", "final_layout_reason": "exceeds_measured_envelope"}))
    try:
        _child.install_wire_recorder(tmp_path, endpoint)
        assert child_http.request_bytes(
            WIRE, endpoint, "", deadline=time.monotonic() + scaled(10),
            socket_timeout=scaled(5)) == b'{"ok":true}'
    finally:
        ACTIVE_REQUEST.reset(token)
    record = json.loads((tmp_path / "sends.jsonl").read_text())
    assert record["final_layout"] == "failed_open"
    assert record["final_layout_reason"] == "exceeds_measured_envelope"


@pytest.mark.host_sandbox
def test_gzip_accepting_server_keeps_compressed_send(monkeypatch, transport_server):
    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "on")
    endpoint, seen, accepts_gzip = transport_server
    accepts_gzip["value"] = True
    reply = kernel_bounded_http.request_bytes(
        WIRE, endpoint, "", deadline=time.monotonic() + scaled(10), socket_timeout=scaled(5))
    assert reply == b'{"ok":true}'
    assert len(seen) == 1
    assert seen[0][0] == "gzip"
    assert gzip.decompress(seen[0][1]) == WIRE
    assert seen[0][2] == hashlib.sha256(WIRE).hexdigest()


def test_shipped_profiles_and_code_defaults_disable_request_gzip():
    # Profile values must agree with both code defaults; neither is an implicit opt-in.
    profiles = list((ROOT / "src/stateframe/profiles").glob("*.json"))
    assert profiles
    for path in profiles:
        switches = json.loads(path.read_text())["switches"]
        assert switches["STATEFRAME_HTTP_GZIP"] == "off", path
        assert switches["STATEFRAME_PLAIN_GZIP"] == "off", path
    assert kernel_bounded_http.request_gzip_enabled({}) is False
    identity, headers = encode_request_body({"content": "x" * 40_000}, {})
    assert len(identity) >= 32 * 1024
    assert "Content-Encoding" not in headers
    compressed, opt_in_headers = encode_request_body(
        {"content": "x" * 40_000}, {"STATEFRAME_PLAIN_GZIP": "on"})
    assert opt_in_headers["Content-Encoding"] == "gzip"
    assert gzip.decompress(compressed) == identity
    with pytest.raises(ValueError, match="must be 'on' or 'off'"):
        kernel_bounded_http.request_gzip_enabled({"STATEFRAME_HTTP_GZIP": "invalid"})
    with pytest.raises(ValueError, match="must be 'on' or 'off'"):
        encode_request_body({"content": "x" * 40_000}, {"STATEFRAME_PLAIN_GZIP": "invalid"})


@pytest.mark.host_sandbox
def test_default_profile_cli_run_sends_large_request_without_gzip(tmp_path):
    # A real scripted run catches profile overrides that direct transport tests cannot.
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg/value.py").write_text("VALUE = 1\n")
    (repo / "tests/test_value.py").write_text(
        "from pkg.value import VALUE\ndef test_value():\n    assert VALUE == 1\n")
    script = Script({"model": "scripted", "steps": [
        {"tool": "run_tests", "arguments": {"target": "tests/test_value.py"}},
        {"tool": "done", "arguments": {"summary": "Checks passed."}},
    ]})
    seen = []
    scripted_handler = make_handler(script)

    class RejectGzip(scripted_handler):
        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") == "/v1/chat/completions":
                seen.append((self.headers.get("Content-Encoding"),
                             int(self.headers["Content-Length"])))
                if self.headers.get("Content-Encoding") == "gzip":
                    self.rfile.read(int(self.headers["Content-Length"]))
                    self._send(415, {"error": "unsupported encoding"})
                    return
            super().do_POST()

    server = ThreadingHTTPServer(("127.0.0.1", 0), RejectGzip)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = os.environ.copy()
    env.pop("STATEFRAME_HTTP_GZIP", None)
    env.pop("STATEFRAME_PLAIN_GZIP", None)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(ROOT / "src/stateframe/engine")))
    run_dir = tmp_path / "run"
    try:
        result = run_monitored([
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
            # The task is capped at 4096 characters; system, tools and frame already carry about 30 KB,
            # so a task near the cap puts the first request over the 32 KiB compression threshold.
            "--task", "Check this repository. " + "x" * 4_000,
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
            "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", sys.executable, "--max-turns", "4", "--quiet",
        ], env=env, capture_output=True, text=True, expected_seconds=scaled(240))
        if (result.returncode != 0 and not seen
                and "sandbox-exec: sandbox_apply: Operation not permitted" in result.stderr):
            pytest.skip("macOS sandbox-exec unavailable in this nested sandbox")
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        assert json.loads((run_dir / "summary.json").read_text())["succeeded"]
        script.assert_consumed()
        assert seen and seen[0][1] >= 32 * 1024
        assert all(encoding is None for encoding, _ in seen)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=scaled(5))
