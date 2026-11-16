"""ToolCallLoopBreaker.trip_external: one trip, notice delivered through take_notice, protocol tools count as edit/loop tools."""
from __future__ import annotations

import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from kernel_tool_call_loop_breaker import (  # noqa: E402
    LOOP_BREAKER_NOTICE, ToolCallLoopBreaker, _EDIT_TOOLS, _LOOP_TOOLS,
)


def test_protocol_tools_are_edit_and_loop_tools():
    assert {"edit_source", "rewrite_source"} <= _EDIT_TOOLS
    assert {"edit_source", "rewrite_source"} <= _LOOP_TOOLS


def test_trip_external_trips_once_and_notice_flows_through_take_notice():
    breaker = ToolCallLoopBreaker(threshold=5)
    notice = breaker.trip_external("edit_source", "was refused 3 times with the same refusal fingerprint.")
    assert notice is not None and notice.startswith(LOOP_BREAKER_NOTICE)
    assert "`edit_source`" in notice and "refused 3 times" in notice
    assert breaker.tripped and breaker.first_break_action == "edit_source"
    assert breaker.take_notice() == notice
    assert breaker.take_notice() is None
    assert breaker.trip_external("edit_source", "again.") is None


def test_observe_after_external_trip_does_not_emit_second_notice():
    breaker = ToolCallLoopBreaker(threshold=2)
    breaker.trip_external("edit_source", "x.")
    breaker.take_notice()
    for _ in range(3):
        decision = breaker.observe("read", {"path": "a.py"}, "contents")
    assert decision.notice is None and decision.tripped is False
