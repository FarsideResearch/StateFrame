"""Tool receipts are labelled by what actually happened (class-wide pin)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"))

from kernel_native_tools import NativeToolBatch, closed_receipt_status  # noqa: E402

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
def test_label_matches_real_outcome_for_every_tool(tool):
    for result in RESULTS:
        for form in (result, "RESULT_" + result):
            assert closed_receipt_status(tool, form) == expected_label(tool, result), (tool, form)


def test_red_check_is_never_rejected_but_failed_edit_is():
    assert closed_receipt_status("run_tests", "RED") == "RED"
    assert closed_receipt_status("str_replace", "FAIL") == "REJECTED"
    assert closed_receipt_status("str_replace", "MALFORMED") == "REJECTED"
    assert closed_receipt_status("grep", "RED") == "REJECTED"
    assert closed_receipt_status("run_tests", "MALFORMED") == "REJECTED"
    assert closed_receipt_status("run_tests", "ENV_BLOCKED") == "REJECTED"
    assert closed_receipt_status("run_tests", "TIMEOUT") == "ERROR"
