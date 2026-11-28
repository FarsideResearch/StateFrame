"""Startup probe walls adapt to measured host load within a fixed retry budget."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from stateframe import runner, send1_probe_cache, startup_cache_writes
from stateframe.engine import kernel_safe_probe


def _result(*, timed_out=False, exit_code=0, output=b"ok", truncated=False):
    return SimpleNamespace(timed_out=timed_out, exit_code=exit_code, output=output,
                           output_truncated=truncated)


@pytest.fixture
def probe(monkeypatch):
    calls, script = [], []
    monkeypatch.delenv("STATEFRAME_STARTUP_PROBE_SECONDS", raising=False)

    def fake(argv, workspace, *, timeout, **_kwargs):
        calls.append(timeout)
        return script.pop(0)

    monkeypatch.setattr(kernel_safe_probe, "run_probe", fake)
    return calls, script


def _host(monkeypatch, loadavg, cpu_count):
    monkeypatch.setattr(runner.os, "getloadavg", lambda: (loadavg, loadavg, loadavg))
    monkeypatch.setattr(runner.os, "cpu_count", lambda: cpu_count)


def _run():
    return runner._startup_probe(sys.executable, "pass", label="interpreter x")


def test_idle_host_keeps_the_existing_walls(probe, monkeypatch):
    calls, script = probe
    _host(monkeypatch, 0.0, 8)
    script += [_result(timed_out=True), _result()]

    assert _run() == b"ok"
    assert calls == [runner.STARTUP_PROBE_SECONDS, 2 * runner.STARTUP_PROBE_SECONDS]


def test_load_ratio_four_scales_the_first_wall_fourfold(probe, monkeypatch):
    calls, script = probe
    _host(monkeypatch, 16.0, 4)
    script += [_result(timed_out=True), _result()]

    assert _run() == b"ok"
    assert calls[0] == 4 * runner.STARTUP_PROBE_SECONDS
    assert sum(calls) <= 600


def test_load_scaled_retry_budget_is_capped_at_600_seconds(probe, monkeypatch):
    calls, script = probe
    _host(monkeypatch, 1000.0, 1)
    script += [_result(timed_out=True), _result(timed_out=True)]

    with pytest.raises(runner.RunError):
        _run()
    assert calls == [300.0, 300.0]
    assert sum(calls) == 600.0


def test_capped_retry_is_not_cached_as_clean_for_the_next_launch(probe, monkeypatch, tmp_path):
    calls, script = probe
    _host(monkeypatch, 24.0, 4)
    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "on")
    receipts = {}
    monkeypatch.setattr(send1_probe_cache, "reads_available", lambda: True)
    monkeypatch.setattr(send1_probe_cache, "engine_fingerprint", lambda *_args: "stable")
    monkeypatch.setattr(send1_probe_cache, "lookup",
                        lambda _kind, fingerprint: receipts.get(fingerprint))
    monkeypatch.setattr(send1_probe_cache, "check_sandbox", lambda *_args: True)

    def publish(_kind, fingerprint, facts):
        receipts[fingerprint] = facts
        return True

    monkeypatch.setattr(send1_probe_cache, "publish", publish)
    monkeypatch.setattr(startup_cache_writes, "defer", lambda _kind, write: write())
    options = runner.RunOptions(repo=tmp_path, task="test", endpoint=None)
    options.runner_kind = "generic"
    options._send1_wave4 = True
    script += [_result(timed_out=True), _result(output=b"[3, 12, 0]"),
               _result(output=b"[3, 12, 0]")]

    runner._check_engine_python(options)

    assert calls == [300.0, 300.0]
    assert receipts == {}

    runner._check_engine_python(options)

    assert calls == [300.0, 300.0, 300.0]


def test_environment_override_wins_over_host_load(probe, monkeypatch):
    calls, script = probe
    _host(monkeypatch, 1000.0, 1)
    monkeypatch.setenv("STATEFRAME_STARTUP_PROBE_SECONDS", "2.5")
    script += [_result(timed_out=True), _result()]

    assert _run() == b"ok"
    assert calls == [2.5, 5.0]


@pytest.mark.parametrize("result", [_result(exit_code=1, output=b"broken"),
                                    _result(truncated=True)])
def test_real_failure_is_reported_once_even_on_a_loaded_host(probe, monkeypatch, result):
    calls, script = probe
    _host(monkeypatch, 16.0, 4)
    script += [result, _result()]

    with pytest.raises(runner.RunError):
        _run()
    assert calls == [4 * runner.STARTUP_PROBE_SECONDS]


def test_oversized_output_takes_precedence_over_timeout(probe, monkeypatch):
    calls, script = probe
    _host(monkeypatch, 16.0, 4)
    script += [_result(timed_out=True, truncated=True), _result()]

    with pytest.raises(runner.RunError, match="exceeded its 65536 byte output limit"):
        _run()

    assert calls == [4 * runner.STARTUP_PROBE_SECONDS]


def test_timeout_reports_load_cpu_and_the_walls_used(probe, monkeypatch):
    calls, script = probe
    _host(monkeypatch, 16.0, 4)
    script += [_result(timed_out=True), _result(timed_out=True)]

    with pytest.raises(runner.RunError) as error:
        _run()

    message = str(error.value)
    assert "load-scaled wall" in message
    assert "loadavg=16" in message
    assert "cpu_count=4" in message
    assert "wall used=240 s and then 360 s" in message
    assert "600 s total" in message
    assert calls == [240.0, 360.0]
