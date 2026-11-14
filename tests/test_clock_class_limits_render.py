"""Unit guards for the clock-class fix: the limits line is clock-independent; resume keeps its time.

The whole-request killer (test_clock_class_wire_identity) proves the wire; these pin each seam:
the renderer, the admission text, the private time-left sidecar and the resume readers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import kernel_read_fixes
from kernel_read_fixes import ContinuationLimits, ResponseBudget, sandbox_time_left
from stateframe import replay_tape, resume


@pytest.mark.parametrize("ledger_type", (ResponseBudget, ContinuationLimits))
def test_limits_line_is_identical_under_two_clocks(monkeypatch, ledger_type):
    """Repro of crew-07's egress diff: the same ledger rendered later must not change a byte."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(kernel_read_fixes.time, "monotonic", lambda: clock["now"])
    limits = ledger_type(limit=10, mode="response", sandbox_deadline_monotonic=1000.0 + 2400.0,
                         sandbox_time_allowance_seconds=2400)
    limits.responses_used = 3
    early = limits.render_continuation_limits()
    clock["now"] += 1234.5
    late = limits.render_continuation_limits()
    assert early == late
    assert "sandbox_seconds_remaining=limited (wall allowance 2400s);" in early
    assert "minutes left" not in early and "turns at your pace" not in early
    # The deadline still governs the run itself.
    assert limits._sandbox_seconds_remaining() == pytest.approx(2400.0 - 1234.5)


def test_rendered_admission_ignores_the_deadline_but_the_ledger_does_not(monkeypatch):
    monkeypatch.setattr(kernel_read_fixes.time, "monotonic", lambda: 5000.0)
    limits = ContinuationLimits(limit=10, mode="response", sandbox_deadline_monotonic=4000.0,
                                sandbox_time_allowance_seconds=60)
    assert "next_action_admission=allowed;" in limits.render_continuation_limits()
    assert limits.qualify_admission(model_decisions=1, tool_attempts=1) is False
    assert limits.qualify_admission(model_decisions=1, tool_attempts=1, include_deadline=False)
    assert limits.can_start_response() is False


def test_allowance_must_be_positive_seconds():
    with pytest.raises(ValueError):
        ResponseBudget(limit=1, sandbox_time_allowance_seconds=0)
    with pytest.raises(ValueError):
        ResponseBudget(limit=1, sandbox_time_allowance_seconds="soon")
    assert ResponseBudget(limit=1, sandbox_time_allowance_seconds="900").sandbox_time_allowance_seconds == 900.0


def test_time_left_sidecar_follows_the_newest_ledger(monkeypatch):
    monkeypatch.setattr(kernel_read_fixes.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(kernel_read_fixes, "_RESUMED_DEADLINE_MONOTONIC", None)
    unbounded = ResponseBudget(limit=1)
    assert sandbox_time_left() is None
    limited = ResponseBudget(limit=1, sandbox_deadline_monotonic=160.0,
                             sandbox_time_allowance_seconds=90)
    assert sandbox_time_left() == {"seconds_left": 60.0, "time_allowance_seconds": 90.0}
    helper = ResponseBudget(limit=1)  # a later unbounded helper ledger never shadows the run's
    assert sandbox_time_left() == {"seconds_left": 60.0, "time_allowance_seconds": 90.0}
    del unbounded, limited, helper
    assert sandbox_time_left() is None  # only a weak reference is kept


def _legacy(seconds: str) -> bytes:
    return json.dumps({"messages": [{"role": "user", "content": (
        "[CONTINUATION LIMITS | stateframe-continuation-limits/1] "
        f"sandbox_seconds_remaining={seconds}; next_action_admission=allowed")}]}).encode()


def test_recorded_time_left_prefers_the_sidecar_then_legacy_text(tmp_path: Path):
    wire = tmp_path / "wire"
    wire.mkdir()
    current = wire / "0001.request.json"
    current.write_bytes(_legacy("limited (wall allowance 900s)"))
    assert replay_tape.recorded_time_left(current) is None
    (wire / "0001.time-left.json").write_text('{"seconds_left":712.5,"time_allowance_seconds":900.0}')
    assert replay_tape.recorded_time_left(current) == 712.5

    legacy = wire / "0002.request.json"
    legacy.write_bytes(_legacy("00420.0 (about 0007 minutes left, ~---- turns at your pace)"))
    assert replay_tape.recorded_time_left(legacy) == 420.0
    (wire / "0002.time-left.json").write_text("not json")
    assert replay_tape.recorded_time_left(legacy) == 420.0
    assert replay_tape.recorded_time_left(wire / "0003.request.json") is None


def test_resumed_time_and_window_read_the_sidecar(tmp_path: Path):
    wire = tmp_path / "wire"
    wire.mkdir()
    rows = []
    for index, left in ((1, 600.0), (2, 420.0)):
        request = wire / f"{index:04d}.request.json"
        request.write_bytes(_legacy("limited (wall allowance 600s)"))
        (wire / f"{index:04d}.time-left.json").write_text(json.dumps(
            {"seconds_left": left, "time_allowance_seconds": 600.0}))
        (wire / f"{index:04d}.response.json").write_bytes(
            b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}')
        rows.append({"request": str(request)})
    pending = wire / "0003.request.json"
    pending.write_bytes(_legacy("limited (wall allowance 600s)"))
    (wire / "0003.time-left.json").write_text('{"seconds_left":400.0,"time_allowance_seconds":600.0}')

    assert resume._resumed_time(rows, {"request": str(pending)}, wire) == {
        "seconds_left": 400.0, "time_allowance_seconds": 600.0}
    assert resume._resumed_time(rows, None, wire) == {
        "seconds_left": 420.0, "time_allowance_seconds": 600.0}
    assert replay_tape.recorded_countdown_window(wire) == (600.0, 420.0)
    assert resume._resumed_time([], None, tmp_path / "missing") == {}
