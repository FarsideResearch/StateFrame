"""No request deadline exists unless the user set one.

The wrapper must not inject per-request knobs the user never set, and the engine must not
derive an elapsed cap (attempt wall or queue budget) from the legacy no-knob defaults.
A user knob, run --timeout or queue cap still bounds the request.
"""
from __future__ import annotations

import json
import math
import sys
from types import SimpleNamespace

import pytest

import kernel_bounded_http
from stateframe import runner
from stateframe.engine import kernel_measurement

REPLY = b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
BODY = {"messages": [{"role": "user", "content": "hello"}], "tools": []}
WIRE = json.dumps(BODY).encode()
REQUEST_KNOBS = ("REACT_REQUEST_TIMEOUT_S", "STATEFRAME_PER_REQUEST_TIMEOUT_S")


@pytest.fixture(autouse=True)
def _clean_deadline_env(monkeypatch):
    for name in (*REQUEST_KNOBS, "STATEFRAME_QUEUE_WAIT_MAX_S", "STATEFRAME_RUN_WALL_SECONDS"):
        monkeypatch.delenv(name, raising=False)


def _drive_spec(tmp_path, monkeypatch, *, remaining, deadline, environment):
    """Run the real runner.drive wrapper against a pre-written spec; return its env and sources."""
    class Engine:
        returncode = 0
        stdout = iter(())

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: Engine())
    monkeypatch.setattr(runner, "_remaining", lambda _o: remaining)
    monkeypatch.setattr(runner, "_child_env", lambda _o: {})
    monkeypatch.setattr(runner, "engine_reads_test_python", lambda: False)
    monkeypatch.setattr(runner, "_stop_group", lambda _p: None)
    (tmp_path / "spec.json").write_text(json.dumps({"environment": dict(environment)}))
    options = SimpleNamespace(timeout_s=None if math.isinf(remaining) else remaining,
                              _deadline=deadline, python=sys.executable,
                              shell_confirm=None, quiet=True)
    try:
        runner.drive(tmp_path, options)
    except Exception:
        pass  # only the written spec matters; the fake engine is minimal
    spec = json.loads((tmp_path / "spec.json").read_text())
    return spec["environment"], spec.get("environment_sources", {})


# (a) real wrapper path, no user knobs and no --timeout
def test_wrapper_injects_no_request_knobs_without_user_knobs(tmp_path, monkeypatch):
    env, _ = _drive_spec(tmp_path, monkeypatch, remaining=math.inf, deadline=None, environment={})
    for name in REQUEST_KNOBS:
        assert name not in env
    assert "STATEFRAME_RUN_WALL_SECONDS" not in env


# (b) user-set per-request knob is kept, and the wrapper adds no REACT default beside it
def test_wrapper_keeps_user_per_request_knob_only(tmp_path, monkeypatch):
    env, _ = _drive_spec(tmp_path, monkeypatch, remaining=math.inf, deadline=None,
                         environment={"STATEFRAME_PER_REQUEST_TIMEOUT_S": "900"})
    assert float(env["STATEFRAME_PER_REQUEST_TIMEOUT_S"]) == 900.0
    assert "REACT_REQUEST_TIMEOUT_S" not in env


# (c) run --timeout sets the run wall only; request knobs stay absent
def test_run_timeout_sets_wall_but_no_request_knobs(tmp_path, monkeypatch):
    remaining = 600.0
    env, sources = _drive_spec(tmp_path, monkeypatch, remaining=remaining,
                               deadline=runner.time.monotonic() + remaining, environment={})
    assert 0 < float(env["STATEFRAME_RUN_WALL_SECONDS"]) <= remaining
    assert sources["STATEFRAME_RUN_WALL_SECONDS"] == "wrapper:run-deadline"
    for name in REQUEST_KNOBS:
        assert name not in env
        assert name not in sources


# user knob above the run wall is still clamped to it (validation/clamp kept)
def test_user_knob_is_clamped_to_run_wall(tmp_path, monkeypatch):
    remaining = 600.0
    env, _ = _drive_spec(tmp_path, monkeypatch, remaining=remaining,
                         deadline=runner.time.monotonic() + remaining,
                         environment={"STATEFRAME_PER_REQUEST_TIMEOUT_S": "99999"})
    assert float(env["STATEFRAME_PER_REQUEST_TIMEOUT_S"]) <= float(env["STATEFRAME_RUN_WALL_SECONDS"])
    assert float(env["STATEFRAME_PER_REQUEST_TIMEOUT_S"]) < 99999.0


def _engine_send(tmp_path, monkeypatch, recorded):
    """Drive RequestCapture._send_once end to end; only the bounded HTTP call is replaced."""
    def fake_request_bytes(wire, endpoint, key, **kwargs):
        recorded.update(kwargs)
        return REPLY

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", fake_request_bytes)
    journal = kernel_measurement.AuditJournal(tmp_path, "deadline-user-only", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    capture._send_once(dict(BODY), "http://127.0.0.1:1/v1/chat/completions", "",
                       _frozen_retry_only=True)
    return recorded


# (d) engine path with the env the wrapper produced for (a): no elapsed cap at all
def test_engine_send_without_user_knobs_has_no_elapsed_deadline(tmp_path, monkeypatch):
    recorded = _engine_send(tmp_path, monkeypatch, {})
    assert recorded["deadline"] is None
    assert recorded["run_deadline"] is None
    assert recorded["socket_timeout"] == pytest.approx(kernel_measurement.request_timeouts()[0])


# (e) a user knob keeps its finite cap
def test_engine_send_with_user_knob_keeps_finite_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_PER_REQUEST_TIMEOUT_S", "5")
    recorded = _engine_send(tmp_path, monkeypatch, {})
    assert isinstance(recorded["deadline"], float) and math.isfinite(recorded["deadline"])
    assert recorded["socket_timeout"] <= 5.0 + 1e-6
