"""Class guard for renderer/parser drift; add any new parser of this text here.

The rendered limits line no longer carries a live countdown (see
test_clock_class_wire_identity): it states the declared wall allowance or ``unbounded``. The
legacy countdown parsers stay for tapes recorded by earlier releases and must keep reading that
older text, while never mistaking the new text for a countdown.
"""

from __future__ import annotations

import json
import re

import pytest

import kernel_countdown_slot as countdown
import kernel_wire_stability as wire_stability
from kernel_read_fixes import ContinuationLimits, ResponseBudget
from stateframe.replay_tape import recorded_countdown_seconds, with_recorded_countdown


LEGACY_SLOTS = (
    "00000.0 (about 0000 minutes left, ~---- turns at your pace)",
    "00100.0 (about 0002 minutes left, ~0001 turns at your pace)",
    "10737.6 (about 0179 minutes left, ~0012 turns at your pace)",
    "99999.9 (about 999+ minutes left, ~999+ turns at your pace)",
)


@pytest.mark.parametrize("ledger_type", (ResponseBudget, ContinuationLimits))
@pytest.mark.parametrize(
    ("seconds", "allowance", "expected"),
    (
        pytest.param(None, None, "unbounded", id="unbounded"),
        pytest.param(0.0, 900.0, "limited (wall allowance 900s)", id="spent"),
        pytest.param(123.4, 2400.0, "limited (wall allowance 2400s)", id="running"),
        pytest.param(123.4, None, "limited (wall allowance undeclared)", id="undeclared"),
    ),
)
def test_rendered_limits_state_the_allowance_not_the_clock(
    monkeypatch: pytest.MonkeyPatch,
    ledger_type: type[ResponseBudget],
    seconds: float | None,
    allowance: float | None,
    expected: str,
) -> None:
    limits = ledger_type(limit=10, mode="response", sandbox_time_allowance_seconds=allowance)
    monkeypatch.setattr(limits, "_sandbox_seconds_remaining", lambda now=None: seconds)

    line = limits.render_continuation_limits()
    assert f"sandbox_seconds_remaining={expected};" in line

    recorded = json.dumps({"messages": [{"role": "user", "content": line}]}).encode()
    # Legacy parsers never read a countdown out of the new text, and the sizing projection
    # leaves it alone: there is no live slot to project.
    assert recorded_countdown_seconds(recorded) is None
    assert countdown.canonical_time_text(line) == line
    assert countdown.canonical_countdown(line) == line

    # The default-off final-wire projection still classes the whole line as the limits tail.
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
        {"role": "user", "content": wire_stability.VOLATILE_TAIL_MARKER + "\n\n" + line},
    ]


@pytest.mark.parametrize("slot", LEGACY_SLOTS)
def test_legacy_countdown_text_still_round_trips_through_every_parser(slot: str) -> None:
    line = ("[CONTINUATION LIMITS | stateframe-continuation-limits/1] model_decisions_remaining=10; "
            f"sandbox_seconds_remaining={slot}; next_action_admission=allowed")
    recorded = json.dumps({"messages": [{"role": "user", "content": line}]}).encode()
    assert recorded_countdown_seconds(recorded) == float(slot.split(" ", 1)[0])

    rebuilt = recorded.replace(f"sandbox_seconds_remaining={slot};".encode(),
                               b"sandbox_seconds_remaining=unbounded;")
    assert rebuilt != recorded
    replaced = with_recorded_countdown(rebuilt, recorded)
    assert replaced == recorded

    canonical = line.replace(slot, countdown.CANONICAL_TIME_SLOT)
    assert countdown.canonical_time_text(line) == canonical
    assert countdown.canonical_countdown(line) == canonical

    # The replay parser before the ---- / 999+ fix missed those producer values; the current
    # one reads every legacy slot.
    old_parser = re.compile(
        rb"(sandbox_seconds_remaining=)(unbounded|[0-9]+\.[0-9] \(about [0-9]+ minutes left"
        rb"(?:, ~[0-9]+ turns at your pace)?\))"
    )
    assert (old_parser.search(recorded) is None) is ("----" in slot or "999+" in slot)
