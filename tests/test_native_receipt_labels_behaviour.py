"""Behavioural pin: a closed call's model-visible receipt label is its real outcome.

Uses only NativeToolBatch.action_closed + render_model_receipts (present on
v0.5.11), so it fails by assertion there and passes with the fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"))

from kernel_native_tools import NativeToolBatch  # noqa: E402
from kernel_bounded_locate_code import EDIT_TOOLS as _EDITS, READ_TOOLS as _READS  # noqa: E402

CHECK = ("run_tests", "authored_check")
LOOKUPS = tuple(sorted(_READS | {"read_check_receipt"}))
EDITS = tuple(sorted(_EDITS | {"insert_before"}))
TOOLS = CHECK + LOOKUPS + EDITS

def expected_label(tool, result):
    """Independent oracle: label by tool kind and result class."""
    if result in ("UNKNOWN", "NO_EXECUTION_RECEIPT"):
        return "UNKNOWN"
    if result in ("OK", "APPLIED", "PASS", "GREEN", "NO_OP"):
        return "APPLIED"
    if tool in EDITS:
        return "REJECTED"
    if result in ("HOST_OBSERVED", "HITS", "NO_HITS"):
        return "APPLIED"
    if result in ("RED", "BEHAVIOR_FAIL", "PUBLIC_RED", "FAIL", "FAILED") and tool in CHECK:
        return "RED"
    if result in ("SELECTOR_ERROR", "ERROR", "TIMEOUT"):
        return "ERROR"
    return "REJECTED"


RESULTS = ("OK", "APPLIED", "PASS", "GREEN", "NO_OP", "HOST_OBSERVED", "HITS", "NO_HITS",
           "RED", "BEHAVIOR_FAIL", "PUBLIC_RED", "FAIL", "FAILED", "SELECTOR_ERROR",
           "ERROR", "TIMEOUT", "MALFORMED", "ENV_BLOCKED", "REFUSED_PREDISPATCH",
           "REJECTED_CONTRACT", "DENIED_PATH", "REJECTED", "UNKNOWN", "NO_EXECUTION_RECEIPT")


class _Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))
        return {"event_id": len(self.events)}


@pytest.mark.parametrize("tool", TOOLS)
def test_closed_batch_receipt_and_rendering(monkeypatch, tool):
    """Drive the real action_closed + render path (exists on v0.5.11)."""
    for result in RESULTS:
        batch = NativeToolBatch()
        batch.journal = _Journal()
        monkeypatch.setattr(batch, "_integrity_enabled", lambda: True)
        batch.origin, batch.index = "r1", 1
        batch.inflight = {"id": "call_0", "function": {"name": tool}}
        batch.action_closed({"step_record_id": "s1", "result": result,
                             "observation": "pytest exit 1 - 2 failed"})
        (receipt,) = batch.take_model_receipts()
        want = expected_label(tool, result)
        assert receipt["status"] == want, (tool, result)
        assert f"{tool}: {want}" in NativeToolBatch.render_model_receipts([receipt])
