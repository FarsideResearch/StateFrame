"""Queue wait is capped by the engine only when the user sets STATEFRAME_QUEUE_WAIT_MAX_S.

A request queued on a healthy, reachable server must not be failed by an engine default.
Dead servers are caught by the provider liveness probe, which these tests do not change.
"""
from __future__ import annotations

import pytest

from stateframe.engine import kernel_bounded_http


@pytest.fixture(autouse=True)
def _clear_queue_wait_env(monkeypatch):
    monkeypatch.delenv("STATEFRAME_QUEUE_WAIT_MAX_S", raising=False)


def test_unset_queue_wait_has_no_engine_cap_and_hard_deadline_is_run_deadline():
    assert kernel_bounded_http.queue_wait_max_seconds() is None
    run_deadline = 5_000.0
    now = 1_000.0
    assert kernel_bounded_http._hard_deadline(run_deadline, now) == run_deadline


def test_set_queue_wait_caps_hard_deadline_at_now_plus_cap(monkeypatch):
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "600")
    assert kernel_bounded_http.queue_wait_max_seconds() == 600.0
    now = 1_000.0
    # Later run deadline: the user cap wins.
    assert kernel_bounded_http._hard_deadline(now + 10_000.0, now) == pytest.approx(now + 600.0)
    # Earlier run deadline: the run deadline wins.
    assert kernel_bounded_http._hard_deadline(now + 100.0, now) == pytest.approx(now + 100.0)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad"])
def test_invalid_queue_wait_still_raises(monkeypatch, value):
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", value)
    with pytest.raises(ValueError, match="STATEFRAME_QUEUE_WAIT_MAX_S"):
        kernel_bounded_http.queue_wait_max_seconds()
    with pytest.raises(ValueError, match="STATEFRAME_QUEUE_WAIT_MAX_S"):
        kernel_bounded_http._hard_deadline(2_000.0, 1_000.0)


def test_unset_queue_wait_leaves_outer_future_uncapped():
    from stateframe.engine.probes import react_repair
    from types import SimpleNamespace

    assert react_repair._transport_future_timeout(SimpleNamespace(), 0.3, False) is None


def test_unset_queue_wait_drops_queue_term_from_budget():
    from stateframe.engine import kernel_measurement

    # Only the non-queue budget remains: max(legacy request budget, 2 x wall).
    assert kernel_measurement.request_queue_budget_seconds(0.3) == pytest.approx(
        max(kernel_measurement.request_timeout_budget_seconds(), 0.6))


class _StopBeforeWait(Exception):
    """Raised by the captured _communicate_with_liveness so request_bytes returns before any child runs."""


class _FakeProcess:
    stdin = None
    returncode = 0

    def poll(self):
        return 0


def _captured_wait_deadlines(monkeypatch, *, run_deadline, deadline):
    """Run request_bytes up to its parent wait and return (deadline, hard_deadline) it would use."""
    import subprocess
    from types import SimpleNamespace

    captured: dict[str, object] = {}
    monkeypatch.setattr(kernel_bounded_http, "subprocess", SimpleNamespace(
        PIPE=subprocess.PIPE, Popen=lambda *args, **kwargs: _FakeProcess()))

    def capture_wait(process, packet, endpoint, key, *, deadline, hard_deadline, liveness_timeout,
                     probe_state):
        captured.update(deadline=deadline, hard_deadline=hard_deadline)
        raise _StopBeforeWait

    monkeypatch.setattr(kernel_bounded_http, "_communicate_with_liveness", capture_wait)
    with pytest.raises(_StopBeforeWait):
        kernel_bounded_http.request_bytes(
            b"{}", "http://127.0.0.1:9/v1/chat/completions", "",
            deadline=deadline, run_deadline=run_deadline, socket_timeout=5)
    return captured["deadline"], captured["hard_deadline"]


def test_no_cap_and_no_run_deadline_leaves_total_deadline_unset(monkeypatch):
    import time

    started = time.monotonic()
    request_deadline, hard_deadline = _captured_wait_deadlines(
        monkeypatch, run_deadline=None, deadline=started + 600)
    assert request_deadline == pytest.approx(started + 600)
    assert hard_deadline is None


def test_user_cap_still_bounds_total_deadline_when_run_deadline_is_unset(monkeypatch):
    import time

    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "600")
    started = time.monotonic()
    _, hard_deadline = _captured_wait_deadlines(monkeypatch, run_deadline=None, deadline=started + 3000)
    assert hard_deadline == pytest.approx(started + 600, abs=1.0)


def test_user_cap_and_run_deadline_keep_the_earlier_cap(monkeypatch):
    import time

    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "600")
    started = time.monotonic()
    _, hard_deadline = _captured_wait_deadlines(
        monkeypatch, run_deadline=started + 100, deadline=started + 3000)
    assert hard_deadline == pytest.approx(started + 100, abs=1.0)


class _SilentProcess:
    """Stands in for a child that produces no output: each wait times out until `silent_waits` are spent."""
    stdin = None
    returncode = 0

    def __init__(self, silent_waits: int) -> None:
        self.silent_waits = silent_waits
        self.waits = 0

    def poll(self):
        return None

    def communicate(self, input=None, timeout=None):
        import subprocess

        if timeout is None:
            return b"", b""
        self.waits += 1
        if self.waits <= self.silent_waits:
            raise subprocess.TimeoutExpired("child", timeout)
        return b"ok", b""

    def kill(self) -> None:
        pass


def test_parent_wait_with_no_total_deadline_survives_past_request_deadline(monkeypatch):
    import time

    monkeypatch.setattr(kernel_bounded_http, "_probe_liveness", lambda endpoint, key, timeout: 200)
    past = time.monotonic() - 1.0
    # Body finished before the per-attempt deadline and a probe answered: the wait is queue time now.
    state: dict[str, int | float | None] = {"status": 200, "body_attempt": 0, "completed_at": past - 0.5}
    output, _ = kernel_bounded_http._communicate_with_liveness(
        _SilentProcess(silent_waits=5), b"{}", "http://127.0.0.1:9/v1/chat/completions", "",
        deadline=past, hard_deadline=None, liveness_timeout=0.3, probe_state=state)
    assert output == b"ok"


def test_parent_wait_with_no_total_deadline_still_ends_dead_provider(monkeypatch):
    import time

    monkeypatch.setattr(kernel_bounded_http, "_probe_liveness", lambda endpoint, key, timeout: None)
    state: dict[str, int | float | None] = {"status": None, "body_attempt": 0, "completed_at": None}
    started = time.monotonic()
    with pytest.raises(kernel_bounded_http.ProviderLivenessTimeout):
        kernel_bounded_http._communicate_with_liveness(
            _SilentProcess(silent_waits=10**9), b"{}", "http://127.0.0.1:9/v1/chat/completions", "",
            deadline=started + 600, hard_deadline=None, liveness_timeout=0.3, probe_state=state)
    assert time.monotonic() - started < 5


def test_child_first_byte_wait_has_no_total_cap_when_hard_deadline_is_none():
    import socket
    import threading
    import time

    tracker = kernel_bounded_http._AbortiveConnectionTracker()
    tracker.request_deadline = time.monotonic() + 0.2  # per-attempt bound already passed by the reply
    tracker.hard_deadline = None
    tracker.socket_timeout = 5.0
    connection = kernel_bounded_http._AbortiveHTTPConnection("127.0.0.1", tracker=tracker)
    client, server = socket.socketpair()
    connection.sock = client
    connection.putrequest("GET", "/")
    connection.endheaders()

    def reply_late() -> None:
        time.sleep(0.5)
        server.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    writer = threading.Thread(target=reply_late, daemon=True)
    writer.start()
    try:
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == b"ok"
    finally:
        writer.join(timeout=2)
        connection.sock = None
        client.close()
        server.close()
