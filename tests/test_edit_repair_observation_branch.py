"""The handles-mode refusal observation is typed (codes + fingerprints) and parks the repair
text for the next request; legacy_safe keeps the fresh-text annotation path untouched."""
from __future__ import annotations

import ast
import pathlib

import pytest

ENGINE = pathlib.Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
REACT = ENGINE / "probes" / "react_repair.py"


def _source() -> str:
    return REACT.read_text(encoding="utf-8")


def test_react_repair_still_parses():
    ast.parse(_source())


def test_pending_edit_repair_is_reset_per_step():
    src = _source()
    anchor = src.index("tool_target_ref = None        # optional STATEFRAME_BOUND_EDIT target")
    window = src[anchor:anchor + 400]
    assert "_pending_edit_repair = None" in window


def test_handles_branch_precedes_legacy_fresh_text_branch():
    src = _source()
    # Adapted to the landed dispatch: the edit protocol branch names its tool _ep_tool.
    handles = src.index("_ep_tool in PROTOCOL_EDIT_TOOL_NAMES and _edit_source_refused is not None")
    legacy = src.index('if (_name == "str_replace" and tool_batch_args is None')
    assert handles < legacy
    branch = src[handles:legacy]
    assert "build_repair(" in branch and "obs = _repair.observation()" in branch
    assert "_pending_edit_repair = _repair.render()" in branch
    assert "request_ordinal=step + 1" in branch
    assert 'measurement.journal.emit("edit_repair_request"' in branch
    assert "trip_loop_breaker_if_blocked(measurement, _repair, _ep_tool)" in branch


def test_legacy_fresh_text_branch_bytes_unchanged():
    src = _source()
    start = src.index('if (_name == "str_replace" and tool_batch_args is None')
    block = src[start:src.index("# Fallthrough dispatch branches deliver once here.", start)]
    assert "from kernel_edit_rejection_fresh_text import annotate_rejection" in block
    assert "handles_enabled" not in block and "PROTOCOL_EDIT_TOOL_NAMES" not in block


def test_repair_text_is_placed_after_the_protocol_tool_reply():
    # Adapted to the landed dispatch: the edit protocol branch delivers its own tool reply and
    # continues, so the repair text is placed right after that reply (the shared post-green tail
    # is never reached from this branch).
    src = _source()
    branch = src.index('elif line in ("edit_source", "rewrite_source") and tool_edit_protocol_args is not None')
    reply = src.index("msgs.append(_tool_reply_for_call(msgs, tool_call_id, obs) if tool_call_id else", branch)
    repair = src.index("place_repair_message(msgs, _edit_repair_text)", reply)
    assert branch < reply < repair < src.index("continue", repair)
    tail = src[repair:repair + 200]
    assert "_pending_edit_repair = None" in tail


def test_repair_text_is_placed_exactly_once_in_react_repair():
    assert _source().count("place_repair_message(") == 1
