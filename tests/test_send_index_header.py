"""The optional send-index echo labels each physical chat POST without touching the request body.

A transport retry re-posts the same engine send, so a server that numbers arrivals drifts away from
the engine's own send numbers. With ``STATEFRAME_SEND_INDEX_HEADER=on`` every recorded chat POST
carries its engine send number and its attempt number in two request headers; off (the default)
the request is unchanged.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import kernel_bounded_http
import pytest

from _timeouts import scaled

from stateframe import _child, runner
from stateframe._child import install_wire_recorder

SWITCH = "STATEFRAME_SEND_INDEX_HEADER"
INDEX_HEADER = "X-SF-Send-Index"
ATTEMPT_HEADER = "X-SF-Send-Attempt"
REPLY = b'{"usage":{"prompt_tokens":7,"completion_tokens":3},"choices":[]}'
# Above the gzip threshold, so STATEFRAME_HTTP_GZIP=on makes the first attempt a gzip body.
WIRES = [json.dumps({"model": "scripted", "messages": [
    {"role": "user", "content": letter * 40_000}]}).encode() for letter in "xy"]


@pytest.fixture
def arrivals():
    """A server that refuses gzip once per send (HTTP 415) and logs every arriving POST."""
    seen: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            encoding = self.headers.get("Content-Encoding")
            seen.append({
                "index": self.headers.get(INDEX_HEADER), "attempt": self.headers.get(ATTEMPT_HEADER),
                "encoding": encoding, "body": body,
                "body_sha256": self.headers.get("X-Request-Body-SHA256"),
                "headers": sorted(name.lower() for name in self.headers.keys()),
            })
            status, reply = (415, b'{"error":"unsupported encoding"}') if encoding == "gzip" else (200, REPLY)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=scaled(5))


def _recorded_sends(monkeypatch, run: Path, endpoint: str) -> list[dict[str, object]]:
    """Send both wires through a freshly installed wire recorder; return the sends.jsonl rows."""
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    real_transport = kernel_bounded_http.request_bytes
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", real_transport)
    run.mkdir(parents=True)
    install_wire_recorder(run, endpoint)
    try:
        for wire in WIRES:
            assert kernel_bounded_http.request_bytes(
                wire, endpoint, "", deadline=time.monotonic() + scaled(20),
                socket_timeout=scaled(5)) == REPLY
    finally:
        kernel_bounded_http.request_bytes = real_transport
    return [json.loads(line) for line in (run / "sends.jsonl").read_text().splitlines()]


def _identity(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{key: value for key, value in row.items() if key != "seconds"} for row in rows]


@pytest.mark.host_sandbox
def test_off_by_default_sends_no_index_headers(tmp_path, monkeypatch, arrivals):
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.delenv("STATEFRAME_HTTP_GZIP", raising=False)
    endpoint, seen = arrivals
    _recorded_sends(monkeypatch, tmp_path / "run", endpoint)
    assert [(entry["index"], entry["attempt"]) for entry in seen] == [(None, None)] * 2
    assert [entry["body"] for entry in seen] == WIRES


@pytest.mark.host_sandbox
def test_on_labels_every_physical_attempt_with_engine_send_and_attempt(tmp_path, monkeypatch, arrivals):
    monkeypatch.setenv(SWITCH, "on")
    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "on")
    endpoint, seen = arrivals
    rows = _recorded_sends(monkeypatch, tmp_path / "run", endpoint)
    # Four arrivals, two engine sends: the retry keeps its send number and advances the attempt.
    assert [(entry["index"], entry["attempt"]) for entry in seen] == [
        ("1", "1"), ("1", "2"), ("2", "1"), ("2", "2")]
    assert [entry["encoding"] for entry in seen] == ["gzip", None, "gzip", None]
    assert [row["send"] for row in rows] == [1, 2]
    for send, wire in enumerate(WIRES, start=1):
        first, second = seen[2 * send - 2], seen[2 * send - 1]
        assert gzip.decompress(first["body"]) == second["body"] == wire
        assert first["body_sha256"] == second["body_sha256"] == hashlib.sha256(wire).hexdigest()
        assert (tmp_path / "run" / "wire" / f"{send:04d}.request.json").read_bytes() == wire


@pytest.mark.host_sandbox
def test_on_changes_headers_only_never_body_or_recorded_identity(tmp_path, monkeypatch, arrivals):
    endpoint, seen = arrivals
    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "on")
    monkeypatch.delenv(SWITCH, raising=False)
    off_rows = _recorded_sends(monkeypatch, tmp_path / "off", endpoint)
    off_seen = list(seen)
    seen.clear()
    monkeypatch.setenv(SWITCH, "on")
    on_rows = _recorded_sends(monkeypatch, tmp_path / "on", endpoint)
    assert _identity(on_rows) == _identity(off_rows)
    assert [entry["body"] for entry in seen] == [entry["body"] for entry in off_seen]
    assert [entry["body_sha256"] for entry in seen] == [entry["body_sha256"] for entry in off_seen]
    extra = {INDEX_HEADER.lower(), ATTEMPT_HEADER.lower()}
    for on_entry, off_entry in zip(seen, off_seen, strict=True):
        assert set(on_entry["headers"]) - set(off_entry["headers"]) == extra
        assert set(off_entry["headers"]) - set(on_entry["headers"]) == set()
    for name in ("0001.request.json", "0002.request.json", "0001.response.json", "0002.response.json"):
        assert (tmp_path / "on" / "wire" / name).read_bytes() == (tmp_path / "off" / "wire" / name).read_bytes()
    assert sorted(path.name for path in (tmp_path / "on" / "wire").iterdir()) == sorted(
        path.name for path in (tmp_path / "off" / "wire").iterdir())


@pytest.mark.host_sandbox
def test_unrecorded_send_has_no_index_even_when_on(monkeypatch, arrivals):
    monkeypatch.setenv(SWITCH, "on")
    monkeypatch.delenv("STATEFRAME_HTTP_GZIP", raising=False)
    endpoint, seen = arrivals
    assert kernel_bounded_http.request_bytes(
        WIRES[0], endpoint, "", deadline=time.monotonic() + scaled(20),
        socket_timeout=scaled(5)) == REPLY
    assert [(entry["index"], entry["attempt"]) for entry in seen] == [(None, None)]


@pytest.mark.parametrize("value, expected", [
    (None, False), ("off", False), (" OFF ", False), ("on", True), ("On", True)])
def test_switch_values(value, expected):
    environ = {} if value is None else {SWITCH: value}
    assert kernel_bounded_http.send_index_header_enabled(environ) is expected


@pytest.mark.parametrize("value", ["", "1", "true", "yes", "auto"])
def test_switch_fails_closed_on_unknown_value(value):
    with pytest.raises(ValueError, match=SWITCH):
        kernel_bounded_http.send_index_header_enabled({SWITCH: value})


def test_recorder_refuses_unknown_switch_value_before_any_send(tmp_path, monkeypatch):
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", kernel_bounded_http.request_bytes)
    monkeypatch.setenv(SWITCH, "maybe")
    with pytest.raises(ValueError, match=SWITCH):
        install_wire_recorder(tmp_path, "http://127.0.0.1:9/v1/chat/completions")


@pytest.mark.parametrize("bad", [0, -1, True, "1", 1.0])
def test_transport_refuses_malformed_send_index_before_submission(bad):
    with pytest.raises(ValueError, match="send index"):
        kernel_bounded_http.request_bytes(
            b"{}", "http://127.0.0.1:9/v1/chat/completions", "",
            deadline=time.monotonic() + 5, socket_timeout=1, send_index=bad)


def test_switch_is_registered_and_reachable_from_the_shell():
    registry = json.loads(
        (Path(kernel_bounded_http.__file__).parent / "switch-registry.json").read_text())
    assert SWITCH in registry["switches"]
    assert runner.OPT_IN_SWITCHES[SWITCH] == ("off", "on")
    assert kernel_bounded_http.SEND_INDEX_ENV == SWITCH
    assert (kernel_bounded_http.SEND_INDEX_HEADER, kernel_bounded_http.SEND_ATTEMPT_HEADER) == (
        INDEX_HEADER, ATTEMPT_HEADER)
