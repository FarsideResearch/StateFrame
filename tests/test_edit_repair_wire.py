"""Repair-ready refusals through the real worker under STATEFRAME_EDIT_PROTOCOL=handles.

A refused edit_source batch leaves only codes + fingerprints in the tool reply; the repair
request (current bytes + fresh handles) is the next user message, its handles resolve on the
next call, repeated identical refusals escalate to BLOCKED and trip the loop breaker, and the
wire stays append-only. Each run first reads the file, so request 1 delivers its bytes and the
transport mints request 1's map (handles q1h*); the edits address those handles.
"""
from __future__ import annotations

import json

import pytest

from kernel_edit_handles import CURRENT_SOURCE_PREFIX, HANDLE_PATTERN
from kernel_edit_repair_request import BLOCKED_MARKER, REFUSED_MARKER
from kernel_tool_call_loop_breaker import LOOP_BREAKER_NOTICE
from test_edit_protocol_switch_wiring import (
    RULES, RULES_PATH, _drive, _edit_source_step, _str_replace_step, _tool_replies)


READ = {"tool": "read", "required": True, "arguments": {"path": RULES_PATH}}


def _after_tool(messages: list[dict]) -> list[tuple[int, dict]]:
    """User messages that directly follow a tool reply (where repair requests are placed)."""
    return [(index, message) for index, message in enumerate(messages)
            if index and message.get("role") == "user" and messages[index - 1].get("role") == "tool"]


@pytest.mark.parametrize("bound", ["off", "on"])
def test_refusal_becomes_typed_reply_plus_repair_request_and_retry_applies(monkeypatch, tmp_path,
                                                                           bound):
    # BOUND_EDIT governs only the legacy writer; the protocol repair flow is the same either way.
    monkeypatch.setenv("STATEFRAME_BOUND_EDIT", bound)
    sends, _ = _drive(tmp_path, monkeypatch, protocol="handles", stop_at=4,
                      steps=[READ, _edit_source_step("q1h1", "= 99", "= 1"), _edit_source_step("q2h1", "= 25", "= 30")])
    second = sends[2][0]
    reply = _tool_replies(second)[-1]
    assert reply.startswith(REFUSED_MARKER) and "OLD_NOT_FOUND" in reply
    assert HANDLE_PATTERN.search(reply) is None and "minimum_spend" not in reply
    index, repair = _after_tool(second)[-1]
    text = repair["content"]
    assert text.startswith(REFUSED_MARKER + " request q2:")
    assert RULES in text and "q2h1" in text and CURRENT_SOURCE_PREFIX in text
    assert '"old_text": "= 99"' in text  # the rejected proposal, verbatim
    applied = json.loads(_tool_replies(sends[3][0])[-1])
    assert applied["status"] == "APPLIED" and applied["changed_paths"] == [RULES_PATH]


def test_repair_wire_is_append_only_and_maps_live_only_in_repair_messages(monkeypatch, tmp_path):
    sends, _ = _drive(tmp_path, monkeypatch, protocol="handles", stop_at=4,
                      steps=[READ, _edit_source_step("q1h1", "= 99", "= 1"), _edit_source_step("q2h1", "= 98", "= 1")])
    assert len({json.dumps(tools, sort_keys=True) for _, tools in sends}) == 1

    def is_repair(message):
        return str(message.get("content")).startswith((REFUSED_MARKER, BLOCKED_MARKER))

    def history(messages):
        # Trailing per-request host notes are rebuilt every turn; repair requests are retained.
        end = len(messages)
        while end and messages[end - 1].get("role") == "user" and not is_repair(messages[end - 1]):
            end -= 1
        return messages[:end]

    for earlier, later in zip(sends, sends[1:]):
        assert history(later[0])[:len(history(earlier[0]))] == history(earlier[0])
    for messages, _ in sends:
        repairs = {index for index, message in _after_tool(messages) if is_repair(message)}
        for index, message in enumerate(messages):
            if CURRENT_SOURCE_PREFIX in str(message.get("content")):
                assert index in repairs, f"map outside a repair message at {index} ({message.get('role')})"
    # The newest request carries exactly one new map: the repair after its last tool reply.
    newest = sends[-1][0]
    last_tool = max(i for i, m in enumerate(newest) if m.get("role") == "tool")
    assert [i for i, _ in _after_tool(newest)][-1] == last_tool + 1
    assert "q3h1" in newest[last_tool + 1]["content"]


def test_three_identical_refusals_block_and_trip_the_loop_breaker(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_TOOL_CALL_LOOP_BREAKER", "on")
    sends, _ = _drive(tmp_path, monkeypatch, protocol="handles", stop_at=5,
                      steps=[READ] + [_edit_source_step(f"q{n}h1", "= 99", "= 1") for n in range(1, 4)])
    replies = _tool_replies(sends[4][0])[1:]
    assert [reply.split(" ", 2)[:2] for reply in replies] == [
        REFUSED_MARKER.split(" "), REFUSED_MARKER.split(" "), BLOCKED_MARKER.split(" ")]
    blocked = [m["content"] for _, m in _after_tool(sends[4][0])][-1]
    assert blocked.startswith(BLOCKED_MARKER)
    wire = json.dumps(sends[4][0])
    assert LOOP_BREAKER_NOTICE in wire and "refused 3 times with the same refusal fingerprint" in wire
    assert LOOP_BREAKER_NOTICE not in json.dumps(sends[3][0])


def test_legacy_safe_never_builds_a_repair_request(monkeypatch, tmp_path):
    sends, _ = _drive(tmp_path, monkeypatch, protocol=None, stop_at=2,
                      steps=[_str_replace_step("= 99", "= 1")])
    wire = json.dumps(sends[-1][0])
    assert REFUSED_MARKER not in wire and CURRENT_SOURCE_PREFIX not in wire
    assert "old_str not found" in _tool_replies(sends[-1][0])[-1]
