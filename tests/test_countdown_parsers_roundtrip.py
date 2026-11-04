"""Class guard for renderer/parser drift; add any new parser of this text here."""

from __future__ import annotations

import json
import re

import pytest

import kernel_countdown_slot as countdown
import kernel_wire_stability as wire_stability
from kernel_read_fixes import ContinuationLimits, ResponseBudget
from stateframe.replay_tape import recorded_countdown_seconds, with_recorded_countdown


@pytest.mark.parametrize("ledger_type", (ResponseBudget, ContinuationLimits))
@pytest.mark.parametrize(
    ("seconds", "now", "used", "expected_seconds", "visible_note", "old_parser_misses"),
    (
        pytest.param(0, 100.0, 0, 0.0, "00000.0", True, id="zero"),
        pytest.param(10**1000, 100.0, 0, 99999.9, "99999.9", True, id="saturated"),
        pytest.param(60, 100.0, 0, 60.0, "~---- turns", True, id="no-pace"),
        pytest.param(60, 60.0, 1, 60.0, "~0001 turns", False, id="one-turn"),
        pytest.param(60, 0.01, 1, 60.0, "~999+ turns", True, id="many-turns"),
        pytest.param(
            60_000, 100.0, 0, 60_000.0, "999+ minutes", True, id="many-minutes"
        ),
        pytest.param(None, 100.0, 0, None, "unbounded", False, id="unbounded"),
    ),
)
def test_rendered_limits_round_trip_through_every_countdown_parser(
    monkeypatch: pytest.MonkeyPatch,
    ledger_type: type[ResponseBudget],
    seconds: int | None,
    now: float,
    used: int,
    expected_seconds: float | None,
    visible_note: str,
    old_parser_misses: bool,
) -> None:
    monkeypatch.setattr(countdown.time, "monotonic", lambda: now)
    limits = ledger_type(limit=10, mode="response")
    limits.responses_used = used
    if isinstance(limits, ContinuationLimits):
        limits.continuation_model_decisions_used = used
    limits._time_left_clock_start = 0.0
    limits._time_left_decisions_at_start = 0
    monkeypatch.setattr(limits, "_sandbox_seconds_remaining", lambda now=None: seconds)

    slot = countdown.render_time_slot(limits, seconds)
    line = limits.render_continuation_limits()
    assert visible_note in slot
    assert f"sandbox_seconds_remaining={slot};" in line

    recorded = json.dumps({"messages": [{"role": "user", "content": line}]}).encode()
    assert recorded_countdown_seconds(recorded) == expected_seconds
    replacement = (
        "00012.3 (about 0000 minutes left, ~---- turns at your pace)"
        if seconds is None
        else "unbounded"
    )
    rebuilt = recorded.replace(
        f"sandbox_seconds_remaining={slot};".encode(),
        f"sandbox_seconds_remaining={replacement};".encode(),
    )
    assert rebuilt != recorded
    replaced = with_recorded_countdown(rebuilt, recorded)
    assert replaced == recorded
    assert recorded_countdown_seconds(replaced) == expected_seconds

    canonical = (
        line if seconds is None else line.replace(slot, countdown.CANONICAL_TIME_SLOT)
    )
    assert countdown.canonical_time_text(line) == canonical
    assert countdown.canonical_countdown(line) == canonical

    assert wire_stability._tail_category(line) == "time"
    assert wire_stability._volatile_segments(line) == [("limits", line)]
    stable = "stable prefix\n"
    assert wire_stability._split_volatile_content(stable + line) == (stable, line, True)
    projected = wire_stability.project(
        {"messages": [{"role": "user", "content": stable + line}]},
        environ={wire_stability.WIRE_STABLE_ENV: "on"},
    )
    assert projected.body["messages"] == [
        {"role": "user", "content": stable},
        {
            "role": "user",
            "content": wire_stability.VOLATILE_TAIL_MARKER + "\n\n" + line,
        },
    ]

    # The replay parser before the fix misses both ---- and 999+ producer values.
    old_parser = re.compile(
        rb"(sandbox_seconds_remaining=)(unbounded|[0-9]+\.[0-9] \(about [0-9]+ minutes left"
        rb"(?:, ~[0-9]+ turns at your pace)?\))"
    )
    assert (old_parser.search(recorded) is None) is old_parser_misses
