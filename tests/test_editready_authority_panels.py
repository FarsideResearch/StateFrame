"""EDITREADY never takes the slot of a host-authority panel (0.7.3/0.7.4 P0).

live runs (subframes OFF and --parallel 2) ended ENGINE_ERROR with
RequestSealError("owner presentation limits differ from admitted contract"):
apply_editready_fact picked the standalone ``[CONTINUATION LIMITS | ...]``
user message as its "safe panel" and overwrote it, so the sealed limits line
was gone from the owner presentation.
"""
from __future__ import annotations

import json

import pytest

import kernel_editready_fact as editready
from kernel_completion_owner import _digest, _wire_continuation_limits
from kernel_request_seal import OwnerPresentation, RequestSealError

LIMITS = (
    "[CONTINUATION LIMITS | stateframe-continuation-limits/1] model_decisions_remaining=77; "
    "tool_attempts_remaining=307; automatic_checks_remaining=80; next_action_admission=allowed"
)
class _Runtime:
    """Bare runtime: no evidence pin contract is set, so only the limits check runs."""


RANGES = {"pkg/owned.py": ((1, 40),)}


@pytest.fixture(autouse=True)
def _editready_on(monkeypatch):
    monkeypatch.setenv(editready.MODE_ENV, "on")
    monkeypatch.setattr(editready, "editable_source_ranges", lambda messages, runtime: dict(RANGES))


def _request_four_shape(limits_message: dict[str, object]) -> list[dict[str, object]]:
    # Shape of a live run request: the limits panel is the last user message
    # and no read-reuse or read-order panel exists, so it was candidates[-1].
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "Task: fix the bug"},
        {"role": "user", "content": '[StateFrame]\n{"step": 12}'},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "read", "arguments": '{"path": "pkg/owned.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "1: x = 1"},
        limits_message,
    ]


def test_standalone_limits_panel_survives_and_seal_holds():
    messages = _request_four_shape({"role": "user", "content": LIMITS})
    expected = _digest(_wire_continuation_limits(messages))
    out, audit = editready.apply_editready_fact(messages, object())
    assert out[5] == {"role": "user", "content": LIMITS}
    assert audit["status"] == "appended"
    assert out[-1]["content"] == editready.format_editready_fact(RANGES)
    assert _digest(_wire_continuation_limits(out)) == expected
    # The exact seal check that raised in the live runs now passes.
    OwnerPresentation.capture(_Runtime(), {"budget_wire_digest": expected}, out)


def test_old_behavior_was_the_live_failure():
    """Pins the RED: overwriting the limits panel trips the owner-presentation seal."""
    messages = _request_four_shape({"role": "user", "content": LIMITS})
    expected = _digest(_wire_continuation_limits(messages))
    clobbered = [dict(m) for m in messages]
    clobbered[5] = {"role": "user", "content": editready.format_editready_fact(RANGES)}
    with pytest.raises(RequestSealError, match="owner presentation limits differ"):
        OwnerPresentation.capture(_Runtime(), {"budget_wire_digest": expected}, clobbered)


def test_limits_tail_on_a_panel_is_not_replaced():
    tail = {"role": "user", "content": "[note] something\n\n" + LIMITS}
    out, audit = editready.apply_editready_fact(_request_four_shape(tail), object())
    assert out[5] == tail
    assert audit["status"] == "appended"


@pytest.mark.parametrize("marker", editready._HOST_AUTHORITY_MARKERS)
def test_no_host_authority_panel_is_replaced(marker):
    panel = {"role": "user", "content": f"{marker} payload"}
    out, audit = editready.apply_editready_fact(_request_four_shape(panel), object())
    assert out[5] == panel
    assert audit["status"] == "appended"


def test_stateframe_state_panel_is_never_a_safe_panel():
    assert editready._safe_panel({"role": "user", "content": '[StateFrame]\n{"step": 1}'}) is False


def test_ordinary_panel_is_still_replaced():
    panel = {"role": "user", "content": "[read reuse] pkg/owned.py lines 1-40 unchanged"}
    messages = _request_four_shape({"role": "user", "content": LIMITS})
    messages.insert(5, panel)
    out, audit = editready.apply_editready_fact(messages, object())
    assert audit["status"] == "replaced_panel" and audit["message_index"] == 5
    assert out[6] == {"role": "user", "content": LIMITS}


# --- prerun classes (T1/T2/T3/T4/T5/T7) ---------------------------------------

def test_shipped_profile_runs_editready_on():
    """T7: the shipped profile is what puts every default run on this path."""
    from stateframe.profiles import load as load_profile

    switches = load_profile("r2hc3")
    assert editready.enabled({editready.MODE_ENV: switches[editready.MODE_ENV]}) is True


@pytest.mark.parametrize("editready_mode", ["on", "off"])
@pytest.mark.parametrize("has_limits", [True, False])
def test_switch_pairs_never_drop_the_limits_line(monkeypatch, editready_mode, has_limits):
    """T1: EDITREADY on/off x limits panel present/absent."""
    monkeypatch.setenv(editready.MODE_ENV, editready_mode)
    last = {"role": "user", "content": LIMITS if has_limits else "[note] plain panel"}
    messages = _request_four_shape(last)
    out, _ = editready.apply_editready_fact(messages, object())
    assert _wire_continuation_limits(out) == _wire_continuation_limits(messages)
    assert sum(LIMITS in str(m.get("content")) for m in out) == int(has_limits)


def test_off_is_byte_identical(monkeypatch):
    """T2: with EDITREADY off the messages pass through unchanged."""
    monkeypatch.setenv(editready.MODE_ENV, "off")
    messages = _request_four_shape({"role": "user", "content": LIMITS})
    out, audit = editready.apply_editready_fact(messages, object())
    assert audit is None
    assert json.dumps(out, sort_keys=True) == json.dumps(messages, sort_keys=True)


def test_multi_turn_fact_is_reconciled_and_limits_stay_stable():
    """T3: request N+1 carries the fact from request N; no duplicate, limits intact, deterministic."""
    first, _ = editready.apply_editready_fact(_request_four_shape({"role": "user", "content": LIMITS}), object())
    second, audit = editready.apply_editready_fact(first, object())
    again, _ = editready.apply_editready_fact(first, object())
    assert audit["status"] == "already_present"
    assert second == again == first
    fact = editready.format_editready_fact(RANGES)
    assert sum(m.get("content") == fact for m in second) == 1
    assert sum(m.get("content") == LIMITS for m in second) == 1


def test_marker_quoted_inside_a_panel_makes_it_unsafe_not_clobbered():
    """T4: a panel that only quotes a host marker is still never overwritten."""
    quoted = {"role": "user", "content": "[read reuse] file text mentions [CONTINUATION LIMITS | x] in a string"}
    messages = _request_four_shape({"role": "user", "content": LIMITS})
    messages.insert(5, quoted)
    out, audit = editready.apply_editready_fact(messages, object())
    assert out[5] == quoted and out[6] == {"role": "user", "content": LIMITS}
    assert audit["status"] == "appended"


@pytest.mark.parametrize("content", [
    "", "   ", "﻿" + LIMITS, "  \n" + LIMITS, LIMITS.replace("; ", ";\r\n"),
    "[note] résumé — 漢字\n\n" + LIMITS,
])
def test_boundary_panels_with_limits_are_never_replaced(content):
    """T5: empty, whitespace, BOM, leading newline, CRLF and non-ASCII panels."""
    panel = {"role": "user", "content": content}
    out, _ = editready.apply_editready_fact(_request_four_shape(panel), object())
    assert out[5] == panel
