"""Synthetic attacks on reply-byte liveness and its existing time caps."""

from __future__ import annotations

import subprocess

import pytest

from stateframe.engine import kernel_bounded_http as transport


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


class ScriptedProcess:
    stdin = None
    returncode = None

    def __init__(self, clock, events):
        self.clock = clock
        self.events = list(events)
        self.calls = 0

    def communicate(self, input=None, timeout=None):
        assert input in (None, b"packet")
        self.calls += 1
        self.clock.now += float(timeout or 0)
        event = self.events.pop(0)
        if event is None:
            self.returncode = 0
            return b"complete", b""
        output, diagnostic = event
        raise subprocess.TimeoutExpired(
            "synthetic-child", timeout, output=output, stderr=diagnostic)


def run_script(monkeypatch, events, *, liveness=0.12, deadline=None,
               hard_deadline=None, probe_status=None):
    clock = FakeClock()
    process = ScriptedProcess(clock, events)
    probes = []
    terminated = []
    monkeypatch.setattr(transport, "time", clock)

    def probe(*_args):
        probes.append(clock.now)
        return probe_status.pop(0) if isinstance(probe_status, list) else probe_status

    monkeypatch.setattr(transport, "_probe_liveness", probe)
    monkeypatch.setattr(transport, "_terminate_child",
                        lambda child: terminated.append(child))
    try:
        result = transport._communicate_with_liveness(
            process, b"packet", "http://127.0.0.1:8080/v1", "", deadline=deadline,
            hard_deadline=hard_deadline, liveness_timeout=liveness,
            probe_state={"status": None, "body_attempt": 0, "completed_at": None})
    except Exception as error:
        return error, clock, process, probes, terminated
    return result, clock, process, probes, terminated


def growing_output(count):
    return [(b"x" * size, None) for size in range(1, count + 1)]


def test_cumulative_reply_bytes_renew_liveness(monkeypatch):
    result, _clock, process, probes, terminated = run_script(
        monkeypatch, growing_output(3) + [None])
    assert result == (b"complete", b"")
    assert process.calls == 4
    assert len(probes) == 3
    assert terminated == []


def test_one_new_reply_byte_per_timeout_is_progress(monkeypatch):
    result, _clock, process, probes, terminated = run_script(
        monkeypatch, [(b"x", None), (b"xx", None), (b"xxx", None), None])
    assert result == (b"complete", b"")
    assert process.calls == 4
    assert len(probes) == 3
    assert terminated == []


def test_repeated_cumulative_output_does_not_renew_liveness(monkeypatch):
    error, clock, process, _probes, terminated = run_script(
        monkeypatch, [(b"same", None)] * 8)
    assert isinstance(error, transport.ProviderLivenessTimeout)
    assert clock.now == pytest.approx(0.16)
    assert process.calls == 4
    assert terminated == [process]


def test_same_length_replacement_is_not_new_cumulative_output(monkeypatch):
    error, clock, process, _probes, terminated = run_script(
        monkeypatch, [(b"first", None), (b"other", None), (b"third", None),
                      (b"last!", None)])
    assert isinstance(error, transport.ProviderLivenessTimeout)
    assert clock.now == pytest.approx(0.16)
    assert process.calls == 4
    assert terminated == [process]


def test_stderr_activity_does_not_renew_reply_liveness(monkeypatch):
    error, clock, process, _probes, terminated = run_script(
        monkeypatch, [(None, b"diagnostic" * size) for size in range(1, 8)])
    assert isinstance(error, transport.ProviderLivenessTimeout)
    assert clock.now == pytest.approx(0.12)
    assert process.calls == 3
    assert terminated == [process]


def test_empty_stdout_does_not_renew_reply_liveness(monkeypatch):
    error, clock, process, _probes, terminated = run_script(
        monkeypatch, [(None, None)] * 8)
    assert isinstance(error, transport.ProviderLivenessTimeout)
    assert clock.now == pytest.approx(0.12)
    assert process.calls == 3
    assert terminated == [process]


def test_one_growth_then_silence_times_out_from_last_new_byte(monkeypatch):
    error, clock, process, _probes, terminated = run_script(
        monkeypatch, [(b"new", None)] + [(b"new", None)] * 8)
    assert isinstance(error, transport.ProviderLivenessTimeout)
    assert clock.now == pytest.approx(0.16)
    assert process.calls == 4
    assert terminated == [process]


def test_reply_progress_cannot_pass_absolute_deadline(monkeypatch):
    error, clock, process, _probes, terminated = run_script(
        monkeypatch, growing_output(8), liveness=0.12, hard_deadline=0.10)
    assert isinstance(error, TimeoutError)
    assert "absolute" in str(error)
    assert clock.now == pytest.approx(0.10)
    assert process.calls == 3
    assert terminated == [process]


def test_reply_progress_cannot_pass_explicit_request_deadline(monkeypatch):
    error, clock, process, _probes, terminated = run_script(
        monkeypatch, growing_output(8), liveness=1.0, deadline=0.05)
    assert isinstance(error, TimeoutError)
    assert "request deadline" in str(error)
    assert clock.now == pytest.approx(0.05)
    assert process.calls == 2
    assert terminated == [process]


def test_progress_is_independent_of_probe_http_status(monkeypatch):
    result, _clock, process, probes, terminated = run_script(
        monkeypatch, growing_output(5) + [None],
        probe_status=[503, None, None, None, None])
    assert result == (b"complete", b"")
    assert process.calls == 6
    assert len(probes) == 5
    assert terminated == []
