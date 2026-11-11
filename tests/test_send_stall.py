"""A reachable provider must not hide a blocked request-body upload."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import hashlib
import itertools
import json
import socket
import subprocess
import threading
import time

import pytest

from _timeouts import scaled
from stateframe import runner
from stateframe.engine import kernel_bounded_http


BODY = b'x' * (8 * 1024 * 1024)
REPLY = b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
# A stalled upload reads only this much. The rest of BODY must exceed what loopback buffers
# can absorb (Linux autotunes a send buffer up to 4 MiB), or the client finishes the upload
# and never stalls; half of BODY was not enough on loaded Linux hosts.
STALL_PREFIX = 256 * 1024


@contextmanager
def upload_server(*, stall_first: bool = False, stall_second: bool = False,
                  refuse_gzip: bool = False, health_probe_only: bool = False,
                  probe_waits_for_second_post: bool = False,
                  post_waits_for_probe: bool = False, probe_waits_for_body: bool = False,
                  respond_after: list[float] | None = None,
                  chunk_sleep: float = 0.0):
    received: list[bytes] = []
    probes: list[str] = []
    connections: list[int] = []
    stopped = threading.Event()
    probe_started = threading.Event()
    # POST ordinal per connection, fixed on the serve thread in accept order. Handler threads
    # can start in any order under load, so "first upload" must not mean "first do_POST".
    post_index: dict[int, int] = {}
    post_counter = itertools.count()
    second_post = threading.Event()
    body_done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def setup(self):
            # pop, not get: a closed socket's id can be reused by a later connection.
            self.post_index = post_index.pop(id(self.request), None)
            if (stall_first and self.post_index == 0) or (stall_second and self.post_index == 1):
                self.request.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
            super().setup()

        def do_GET(self):  # noqa: N802
            if health_probe_only and self.path == "/slots":
                self.close_connection = True
                return
            probes.append(self.path)
            probe_started.set()
            if probe_waits_for_second_post:
                # Hold the probe open until the resend is on the wire, however slow the host.
                second_post.wait(scaled(10))
            if probe_waits_for_body:
                # Keep the probe in flight until the whole body has arrived.
                body_done.wait(scaled(10))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):  # noqa: N802
            connections.append(self.client_address[1])
            if self.post_index == 1:
                second_post.set()
            size = int(self.headers["Content-Length"])
            if (stall_first and self.post_index == 0) or (stall_second and self.post_index == 1):
                received.append(self.rfile.read(STALL_PREFIX))
                stopped.wait(scaled(20))
                return
            if post_waits_for_probe:
                # Read the body only once a probe is in flight, so it completes during the probe.
                probe_started.wait(scaled(10))
            data = bytearray()
            while len(data) < size:
                chunk = self.rfile.read(min(65536, size - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                if chunk_sleep:
                    time.sleep(chunk_sleep)
            received.append(bytes(data))
            body_done.set()
            if respond_after:
                # Answer only after the request deadline, so the reply needs queue wait.
                while time.monotonic() <= respond_after[0] and not stopped.is_set():
                    stopped.wait(0.01)
            if refuse_gzip and self.post_index == 0:
                if not probe_started.wait(scaled(5)):
                    return
                self.send_response(415)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(REPLY)))
                self.end_headers()
                self.wfile.write(REPLY)
            except (BrokenPipeError, ConnectionResetError):
                pass

    class Server(ThreadingHTTPServer):
        def process_request(self, request, client_address):
            # Peek the method on the serve thread so POST ordinals follow accept order.
            request.settimeout(scaled(10))
            try:
                head = request.recv(4, socket.MSG_PEEK)
            except OSError:
                head = b""
            finally:
                request.settimeout(None)
            if head == b"POST":
                post_index[id(request)] = next(post_counter)
            super().process_request(request, client_address)

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", received, probes, connections
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=scaled(5))


@pytest.mark.host_sandbox
def test_mid_body_stall_retries_identical_body_on_new_connection(monkeypatch):
    import kernel_measurement

    monkeypatch.setenv("STATEFRAME_SEND_STALL_S", "10")
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "45")
    with upload_server(stall_first=True) as (endpoint, received, probes, connections):
        started = time.monotonic()
        with pytest.raises(kernel_bounded_http.SendStallTimeout) as caught:
            kernel_bounded_http.request_bytes(
                BODY, endpoint, "secret-key", deadline=started + 25,
                run_deadline=started + 40, socket_timeout=35)
        assert 0 < caught.value.bytes_sent < caught.value.bytes_total
        assert caught.value.bytes_total == len(BODY)
        assert 9 <= caught.value.seconds_stalled < 25
        assert time.monotonic() - started < 25
        assert kernel_measurement._transport_retryable(
            caught.value, elapsed_ns=int((time.monotonic() - started) * 1e9),
            http_status=None, relay_error=None, error_type=caught.value.error_type,
            transport_class="upstream")
        assert kernel_bounded_http.request_bytes(
            BODY, endpoint, "secret-key", deadline=time.monotonic() + 20,
            socket_timeout=20) == REPLY
        assert received == [BODY[:STALL_PREFIX], BODY]
        assert len(connections) == 2
        assert probes


@pytest.mark.host_sandbox
def test_progressing_upload_can_outlast_stall_interval(monkeypatch):
    monkeypatch.setenv("STATEFRAME_SEND_STALL_S", "10")
    with upload_server(chunk_sleep=0.09) as (endpoint, received, _probes, _connections):
        started = time.monotonic()
        assert kernel_bounded_http.request_bytes(
            BODY, endpoint, "", deadline=started + 30,
            socket_timeout=30) == REPLY
        assert time.monotonic() - started > 10
        assert received == [BODY]


@pytest.mark.host_sandbox
def test_fully_sent_body_keeps_queue_wait_extension(monkeypatch):
    started = time.monotonic()
    deadline = started + 0.5
    child = _ScriptedQueueChild(deadline)
    child.finish_body(deadline - 0.1)
    probes = []
    waits = []
    monkeypatch.setattr(kernel_bounded_http.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(kernel_bounded_http, "_probe_liveness",
                        lambda *_args: probes.append(True) or 200)
    assert kernel_bounded_http.request_bytes(
        b'{}', "http://127.0.0.1:1/v1", "", deadline=deadline,
        run_deadline=started + 2, socket_timeout=0.5, liveness_timeout=0.15,
        record_queue_wait=lambda *values: waits.append(values)) == REPLY
    assert probes and waits and waits[0][0] > 0
    assert time.monotonic() >= deadline


@pytest.mark.host_sandbox
def test_mid_body_stall_cannot_use_healthy_probe_for_queue_wait(monkeypatch):
    monkeypatch.setenv("STATEFRAME_SEND_STALL_S", "10")
    with upload_server(stall_first=True) as (endpoint, _received, probes, _connections):
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="request deadline"):
            kernel_bounded_http.request_bytes(
                BODY, endpoint, "", deadline=started + 1,
                run_deadline=started + 20, socket_timeout=20)
        assert time.monotonic() - started < 3
        assert probes


@pytest.mark.parametrize("value", ["9", "601", "nan", "inf", "bogus"])
def test_send_stall_setting_rejects_out_of_range_values(monkeypatch, value):
    monkeypatch.setenv("STATEFRAME_SEND_STALL_S", value)
    with pytest.raises(ValueError, match="STATEFRAME_SEND_STALL_S"):
        kernel_bounded_http.send_stall_seconds()
    with pytest.raises(runner.RunError, match="STATEFRAME_SEND_STALL_S"):
        runner.resolve_opt_in_switches()


def test_send_stall_uses_ordinary_retry_policy():
    import kernel_measurement

    error = kernel_bounded_http.SendStallTimeout(1024, 8192, 90, "http://localhost")
    assert kernel_measurement._transport_retryable(
        error, elapsed_ns=200_000_000_000, http_status=None, relay_error=None,
        error_type=error.error_type, transport_class="upstream")


def test_body_completion_marker_is_per_attempt():
    markers = (b'{"body_attempt":1,"body_fully_sent":false}\n'
               b'{"body_attempt":1,"body_fully_sent":true,"completed_at":2.0}\n'
               b'{"body_attempt":2,"body_fully_sent":false}\n')
    assert kernel_bounded_http._body_completion_state(markers, 0, None) == (2, None)
    assert kernel_bounded_http._body_completion_state(
        markers + b'{"body_attempt":2,"body_fully_sent":true,"completed_at":3.0}\n',
        0, None) == (2, 3.0)
    assert kernel_bounded_http._body_completion_state(
        markers + b'{"body_attempt":1,"body_fully_sent":true,"completed_at":2.0}\n',
        2, None) == (2, None)


@pytest.mark.parametrize("withhold_start", [False, True])
def test_completion_marker_delivered_after_request_cap_gets_queue_wait(monkeypatch, withhold_start):
    """The child finished in time, but stderr collection trails the deadline."""
    started = time.monotonic()
    deadline = started + 0.2
    marker = (b'{"body_attempt":1,"body_fully_sent":false}\n'
              + json.dumps({"body_attempt": 1, "body_fully_sent": True,
                            "completed_at": started + 0.01}).encode() + b"\n")

    class DelayedMarkerProcess:
        returncode = None
        reads = 0

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def communicate(self, _packet=None, timeout=None):
            self.reads += 1
            if self.reads == 1:
                # Model a descheduled stderr reader that resumes after the cap.
                threading.Event().wait(max(0, deadline + 0.02 - time.monotonic()))
                raise subprocess.TimeoutExpired(
                    "child", timeout,
                    stderr=b"" if withhold_start else marker.splitlines()[0] + b"\n")
            if self.reads == 2:
                threading.Event().wait(timeout)
                raise subprocess.TimeoutExpired(
                    "child", timeout,
                    stderr=b"" if withhold_start else marker.splitlines()[0] + b"\n")
            self.returncode = 0
            return REPLY, marker

    child = DelayedMarkerProcess()
    waits = []
    original = kernel_bounded_http._communicate_with_liveness

    def known_live(*args, probe_state, **kwargs):
        probe_state["status"] = 200
        return original(*args, probe_state=probe_state, **kwargs)

    monkeypatch.setattr(kernel_bounded_http.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(kernel_bounded_http, "_communicate_with_liveness", known_live)
    assert kernel_bounded_http.request_bytes(
        b'{}', "http://127.0.0.1:1/v1", "", deadline=deadline,
        run_deadline=started + 0.8, socket_timeout=0.2,
        record_queue_wait=lambda *values: waits.append(values)) == REPLY
    assert waits and waits[0][0] > 0
    assert time.monotonic() >= deadline


class _ScriptedQueueChild:
    """Exercise parent timing without depending on Python child startup or socket buffers."""

    def __init__(self, deadline: float) -> None:
        self.response_at = deadline + 0.04
        self.marker = b'{"body_attempt":1,"body_fully_sent":false}\n'
        self.returncode: int | None = None

    def finish_body(self, at: float) -> None:
        self.marker += json.dumps({"body_attempt": 1, "body_fully_sent": True,
                                   "completed_at": at}).encode() + b"\n"

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def communicate(self, _packet=None, timeout=None):
        if self.returncode is not None:
            return b"", self.marker
        remaining = self.response_at - time.monotonic()
        if remaining > 0:
            threading.Event().wait(min(timeout, remaining) if timeout is not None else remaining)
        if time.monotonic() < self.response_at:
            raise subprocess.TimeoutExpired("child", timeout, stderr=self.marker)
        self.returncode = 0
        return REPLY, self.marker + json.dumps({"first_response_at": time.monotonic()}).encode()


def test_child_body_deadline_error_is_a_timeout():
    diagnostic = b'{"error_type":"RequestBodyDeadlineTimeout"}'
    error = kernel_bounded_http._child_transport_error(diagnostic, "", "http://127.0.0.1:1")
    assert type(error) is TimeoutError
    assert "request deadline exceeded before body completion" in str(error)


def test_child_rejects_body_attempt_that_starts_after_deadline():
    tracker = kernel_bounded_http._AbortiveConnectionTracker()
    tracker.report_body_sent = True
    tracker.request_deadline = time.monotonic() - 1
    connection = kernel_bounded_http._AbortiveConnectionMixin(tracker=tracker)
    with pytest.raises(kernel_bounded_http.RequestBodyDeadlineTimeout):
        connection._send_output(b"body")


def test_completion_stamp_is_last_socket_progress_not_output_return(capsys):
    tracker = kernel_bounded_http._AbortiveConnectionTracker()
    tracker.report_body_sent = True
    tracker.request_deadline = tracker.hard_deadline = time.monotonic() + 5

    class Socket:
        def send(self, data):
            return len(data)

        def settimeout(self, _seconds):
            pass

    class Wire:
        def __init__(self):
            self.sock = Socket()

        def send(self, _data):
            pass

        def _send_output(self, body, encode_chunked=False):
            self.send(b"headers")
            self.send(body)
            # Hold after the final socket send, as http.client can do before returning.
            tracker.request_deadline = time.monotonic() + 0.01
            threading.Event().wait(0.03)

    class Connection(kernel_bounded_http._AbortiveConnectionMixin, Wire):
        pass

    Connection(tracker=tracker)._send_output(b"body")
    attempt, completed_at = kernel_bounded_http._body_completion_state(
        capsys.readouterr().err.encode(), 0, None)
    assert attempt == 1 and completed_at is not None
    assert completed_at < tracker.request_deadline < time.monotonic()


def test_child_that_never_answers_stays_inside_hard_cap():
    started = time.monotonic()
    deadline = started + 0.04
    hard_deadline = started + 0.2

    class SilentChild:
        returncode = None
        killed = False

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

        def communicate(self, _packet=None, timeout=None):
            if self.killed:
                return b"", b""
            threading.Event().wait(timeout)
            raise subprocess.TimeoutExpired("child", timeout)

    child = SilentChild()
    state = {"status": 200, "body_attempt": 0, "completed_at": None}
    with pytest.raises(TimeoutError, match="HTTP absolute deadline exceeded"):
        kernel_bounded_http._communicate_with_liveness(
            child, b"packet", "http://127.0.0.1:1/v1", "", deadline=deadline,
            hard_deadline=hard_deadline, liveness_timeout=1, probe_state=state)
    assert child.killed
    assert time.monotonic() - hard_deadline < 0.1


@pytest.mark.host_sandbox
def test_incomplete_upload_gets_child_deadline_verdict_promptly(monkeypatch):
    monkeypatch.setenv("STATEFRAME_SEND_STALL_S", "10")
    observed = []
    original = kernel_bounded_http._child_transport_error

    def record_child_error(diagnostic, key, endpoint):
        observed.append(kernel_bounded_http._child_diagnostic(diagnostic).get("error_type"))
        return original(diagnostic, key, endpoint)

    monkeypatch.setattr(kernel_bounded_http, "_child_transport_error", record_child_error)
    with upload_server(stall_first=True) as (endpoint, _received, probes, _connections):
        deadline = time.monotonic() + 1.5
        with pytest.raises(TimeoutError, match="request deadline exceeded before body completion"):
            kernel_bounded_http.request_bytes(
                BODY, endpoint, "", deadline=deadline, run_deadline=deadline + 10,
                socket_timeout=10, liveness_timeout=0.5)
        assert probes
        assert observed == ["RequestBodyDeadlineTimeout"]
        assert time.monotonic() - deadline < 1


class _ScriptedExitChild:
    """A body-sent child that exits with ``error_type`` at ``exit_at`` (None: never exits)."""

    def __init__(self, exit_at: float | None, error_type: str, *, body_done_at: float | None) -> None:
        self.exit_at = exit_at
        self.marker = b'{"body_attempt":1,"body_fully_sent":false}\n'
        if body_done_at is not None:
            self.marker += json.dumps({"body_attempt": 1, "body_fully_sent": True,
                                       "completed_at": body_done_at}).encode() + b"\n"
        self.diagnostic = json.dumps({"error_type": error_type}).encode()
        self.returncode: int | None = None
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def communicate(self, _packet=None, timeout=None):
        if self.returncode is not None:
            return b"", self.marker + (b"" if self.killed else self.diagnostic)
        wait = timeout if self.exit_at is None else max(0.0, self.exit_at - time.monotonic())
        if timeout is not None:
            wait = min(wait, timeout)
        threading.Event().wait(wait)
        if self.exit_at is None or time.monotonic() < self.exit_at:
            raise subprocess.TimeoutExpired("child", timeout, stderr=self.marker)
        self.returncode = 1
        return b"", self.marker + self.diagnostic


@pytest.mark.parametrize("order", ["child_first", "parent_first"])
def test_absolute_deadline_is_a_typed_timeout_in_either_order(monkeypatch, order):
    # The child's own absolute deadline and the parent's hard cap race; whichever fires first,
    # the caller gets a TimeoutError, never a BoundedTransportError naming one.
    started = time.monotonic()
    deadline, hard = started + 0.2, started + 0.6
    child = _ScriptedExitChild(hard - 0.1 if order == "child_first" else None, "TimeoutError",
                               body_done_at=started + 0.05)
    monkeypatch.setattr(kernel_bounded_http.subprocess, "Popen", lambda *_a, **_k: child)
    monkeypatch.setattr(kernel_bounded_http, "_probe_liveness", lambda *_args: 200)
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "0.6")
    with pytest.raises(TimeoutError, match="^HTTP absolute deadline exceeded$") as caught:
        kernel_bounded_http.request_bytes(
            b'{}', "http://127.0.0.1:1/v1", "", deadline=deadline,
            run_deadline=hard, socket_timeout=2, liveness_timeout=0.15)
    assert not isinstance(caught.value, kernel_bounded_http.BoundedTransportError)
    assert child.killed is (order == "parent_first")
    assert time.monotonic() - hard < 0.15


@pytest.mark.parametrize("order", ["child_first", "parent_first"])
def test_body_deadline_verdict_in_either_order(monkeypatch, order):
    # Child first: its RequestBodyDeadlineTimeout verdict arrives after the request deadline and
    # is the answer. Parent first: a child silent until the hard cap is killed there. Same type.
    started = time.monotonic()
    deadline, hard = started + 0.2, started + 0.6
    child = _ScriptedExitChild(deadline + 0.05 if order == "child_first" else None,
                               "RequestBodyDeadlineTimeout", body_done_at=None)
    monkeypatch.setattr(kernel_bounded_http.subprocess, "Popen", lambda *_a, **_k: child)
    monkeypatch.setattr(kernel_bounded_http, "_probe_liveness", lambda *_args: 200)
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "0.6")
    expected = ("^HTTP request deadline exceeded before body completion$" if order == "child_first"
                else "^HTTP absolute deadline exceeded$")
    with pytest.raises(TimeoutError, match=expected) as caught:
        kernel_bounded_http.request_bytes(
            b'{}', "http://127.0.0.1:1/v1", "", deadline=deadline,
            run_deadline=hard, socket_timeout=2, liveness_timeout=0.15)
    assert not isinstance(caught.value, kernel_bounded_http.BoundedTransportError)
    ended = time.monotonic()
    if order == "child_first":
        assert ended < hard  # the verdict is used as soon as it arrives, not held to the cap
    else:
        assert child.killed and ended - hard < 0.15


def test_undelivered_packet_fails_at_request_deadline_not_hard_cap(monkeypatch):
    # A child that never reads stdin stands in for one too starved to take its multi-MiB
    # packet: it cannot have sent the body, so the deadline verdict must not wait for it.
    real_popen = subprocess.Popen
    children = []

    def stalled_child(_args, **kwargs):
        child = real_popen([kernel_bounded_http.sys.executable, "-I", "-c", "import time; time.sleep(60)"],
                           **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(kernel_bounded_http.subprocess, "Popen", stalled_child)
    monkeypatch.setattr(kernel_bounded_http, "_probe_liveness", lambda *_args: 200)
    started = time.monotonic()
    deadline = started + 0.5
    with pytest.raises(TimeoutError, match="^HTTP request deadline exceeded before body completion$"):
        kernel_bounded_http.request_bytes(
            BODY, "http://127.0.0.1:1/v1", "", deadline=deadline,
            run_deadline=started + 10, socket_timeout=2, liveness_timeout=0.3)
    assert time.monotonic() - deadline < 1
    assert children and children[0].poll() is not None


def test_packet_is_delivered_whole_while_parent_is_probing(monkeypatch):
    # The parent spends the wait in probes; the packet must keep flowing meanwhile.
    real_popen = subprocess.Popen
    script = ("import json, sys, hashlib; data = sys.stdin.buffer.read(); "
              "sys.stdout.write(hashlib.sha256(json.loads(data)['wire'].encode()).hexdigest())")

    def echo_child(_args, **kwargs):
        return real_popen([kernel_bounded_http.sys.executable, "-I", "-c", script], **kwargs)

    def slow_probe(*_args):
        time.sleep(0.2)
        return 200

    monkeypatch.setattr(kernel_bounded_http.subprocess, "Popen", echo_child)
    monkeypatch.setattr(kernel_bounded_http, "_probe_liveness", slow_probe)
    started = time.monotonic()
    assert kernel_bounded_http.request_bytes(
        BODY, "http://127.0.0.1:1/v1", "", deadline=started + 5,
        run_deadline=started + 10, socket_timeout=2,
        liveness_timeout=0.3) == hashlib.sha256(BODY).hexdigest().encode()


@pytest.mark.host_sandbox
def test_body_finished_during_slow_probe_gets_queue_wait(monkeypatch):
    started = time.monotonic()
    deadline = started + 0.5
    child = _ScriptedQueueChild(deadline)
    waits = []
    probe_started = threading.Event()

    def slow_probe(*_args):
        probe_started.set()
        child.finish_body(time.monotonic())
        threading.Event().wait(max(0, deadline + 0.02 - time.monotonic()))
        return 200

    monkeypatch.setattr(kernel_bounded_http.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(kernel_bounded_http, "_probe_liveness", slow_probe)
    assert kernel_bounded_http.request_bytes(
        BODY, "http://127.0.0.1:1/v1", "", deadline=deadline,
        run_deadline=started + 2, socket_timeout=2, liveness_timeout=0.15,
        record_queue_wait=lambda *values: waits.append(values)) == REPLY
    assert probe_started.is_set() and waits and waits[0][0] > 0
    attempt, completed_at = kernel_bounded_http._body_completion_state(child.marker, 0, None)
    assert attempt == 1 and completed_at is not None and completed_at < deadline


# Real-child twins of the two scripted queue-wait tests: they keep the Popen pipe path the
# completion-marker race lived in. The server orders events (probe in flight, body done, reply
# held past the request deadline); the deadline leaves room for child startup and the upload.
@pytest.mark.host_sandbox
def test_fully_sent_body_keeps_queue_wait_extension_real_child(monkeypatch):
    monkeypatch.setenv("STATEFRAME_SEND_STALL_S", "10")
    waits = []
    started = time.monotonic()
    deadline = started + 5
    with upload_server(health_probe_only=True, respond_after=[deadline + 0.3]) as (
            endpoint, received, probes, _connections):
        assert kernel_bounded_http.request_bytes(
            b'{}', endpoint, "", deadline=deadline, run_deadline=started + 15,
            socket_timeout=10, liveness_timeout=3,
            record_queue_wait=lambda *values: waits.append(values)) == REPLY
    assert received == [b'{}']
    assert probes and set(probes) == {"/health"}
    assert waits and waits[0][0] > 0


@pytest.mark.host_sandbox
def test_body_finished_during_slow_probe_gets_queue_wait_real_child(monkeypatch):
    monkeypatch.setenv("STATEFRAME_SEND_STALL_S", "10")
    waits = []
    started = time.monotonic()
    deadline = started + 5
    with upload_server(health_probe_only=True, post_waits_for_probe=True, probe_waits_for_body=True,
                       respond_after=[deadline + 0.3]) as (endpoint, received, probes, _connections):
        assert kernel_bounded_http.request_bytes(
            BODY, endpoint, "", deadline=deadline, run_deadline=started + 15,
            socket_timeout=10, liveness_timeout=3,
            record_queue_wait=lambda *values: waits.append(values)) == REPLY
    assert received == [BODY]
    assert probes and set(probes) == {"/health"}
    assert waits and waits[0][0] > 0


@pytest.mark.host_sandbox
def test_gzip_resend_upload_during_probe_does_not_inherit_first_marker(monkeypatch):
    monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "on")
    monkeypatch.setenv("STATEFRAME_SEND_STALL_S", "10")
    with upload_server(refuse_gzip=True, stall_second=True, health_probe_only=True,
                       probe_waits_for_second_post=True) as (endpoint, received, probes, connections):
        # liveness_timeout=3 probes about once a second, so the first /health lands well before
        # the 5 s cap (the default schedule probes just before the cap, and a 415 arriving after
        # it is not resent). That probe triggers the 415 and answers only once the uncompressed
        # resend has started, so the resend is in flight, and stalled, when the cap passes. If
        # the first attempt's completion marker were inherited, probe answers would extend the
        # wait to the 10 s send-stall watchdog (about 11 s in) or the run deadline.
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="request deadline"):
            kernel_bounded_http.request_bytes(
                BODY, endpoint, "", deadline=started + 5, liveness_timeout=3,
                run_deadline=started + 30, socket_timeout=20)
        assert time.monotonic() - started < 9
    assert len(connections) == 2 and probes and set(probes) == {"/health"}


@pytest.mark.host_sandbox
def test_request_capture_retries_send_stall_with_identical_wire(monkeypatch, tmp_path: Path):
    import kernel_measurement
    import kernel_pilot

    monkeypatch.setenv("STATEFRAME_SEND_STALL_S", "10")
    monkeypatch.setattr(kernel_pilot, "POLICY", kernel_pilot.RetryPolicy(
        transport_attempts=2, transport_backoff_seconds=(0.0,)))
    body = {"messages": [{"role": "user", "content": BODY.decode()}], "tools": []}
    wire = json.dumps(body).encode()
    journal = kernel_measurement.AuditJournal(tmp_path, "send-stall", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    with upload_server(stall_first=True) as (endpoint, received, _probes, connections):
        response, _submission, _started = capture._send_transport_with_retries(
            body, wire, endpoint, "secret-key", admission=None, socket_timeout=25,
            deadline=time.monotonic() + 40, wall_timeout=25, backend_identity={},
            request_extensions={}, transport_attempt_limit=2, frozen_retry_only=True)
    assert response["choices"]
    assert received == [wire[:STALL_PREFIX], wire]
    assert len(connections) == 2 and connections[0] != connections[1]
    records = [json.loads(line) for line in journal.path.read_text().splitlines()]
    submissions = [record for record in records if record["kind"] == "request_submitted"]
    stalls = [record for record in records if record["kind"] == "send_stall"]
    assert len(submissions) == 2
    assert submissions[1]["payload_delta"]["identical_wire"] is True
    assert len(stalls) == 1 and stalls[0]["retry_scheduled"] is True
    assert "secret-key" not in json.dumps(stalls)
