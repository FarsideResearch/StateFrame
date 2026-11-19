"""REFUSED_ADMISSION text must say the content was not delivered, and why.

The refusal text used to reuse the pager's "N more lines; page with cursor=" wording with zero
source lines, so the model read it as a page and re-issued the same read. These tests drive the real
admission path and pin the new wording, the cursor, and the typed refusal_cause.
"""
from __future__ import annotations

import json
import re

import pytest

import kernel_read_admission as admission
import kernel_working_set_live as live
import kernel_working_set_shadow as shadow
from kernel_working_set_stop import ReasonCode

from test_read_receipts_admission import read, runtime

OLD_WORDING = re.compile(r"^\s*\d+ more lines; page with cursor=")
CURSOR = re.compile(r"cursor=(read1:[0-9a-f]{64}:\d+:\d+)")


def _events(rt):
    """Admission events from either journal shape: the in-memory test double or the real AuditJournal."""
    journal = rt.journal
    if hasattr(journal, "events"):
        rows = journal.events
    else:
        rows = [json.loads(line) for line in journal.path.read_bytes().splitlines()]
    return [e for e in rows if e["kind"] == admission.EVENT]


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")


def test_budget_refusal_says_not_delivered_with_path_range_and_cursor(tmp_path, on, monkeypatch):
    """Spent per-request budget: plain 'Not delivered' text, path, line range, resume cursor."""
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 1000)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/value.py").write_text("value = 1\n")
    rt = runtime(tmp_path, fixed=1000)
    (status, text, _), metadata = read(rt, tmp_path, "pkg/value.py")

    assert status == admission.REFUSED_ADMISSION
    assert "Not delivered" in text                                   # (1)
    assert "pkg/value.py" in text
    assert "lines 1-1" in text
    assert "did not fit in this request's source budget" in text
    assert "Nothing from this range was shown" in text
    assert not OLD_WORDING.match(text)                               # (2)
    assert metadata["next_cursor"] in text                           # (3)
    assert CURSOR.search(text)
    assert metadata["refusal_cause"] == "budget"                     # (4)
    [event] = [e for e in _events(rt) if e["decision"] == admission.REFUSED_ADMISSION]
    assert event["refusal_cause"] == "budget"
    assert "reason_code" not in event


def test_floor_infeasible_refusal_is_typed_and_journaled(tmp_path, on, monkeypatch):
    """Request floor cannot be priced feasibly: 'size limit' text and FLOOR_INFEASIBLE evidence."""
    from test_read_admission_qualification import seal_fixture as build

    capture, rt, body, plan, journal, tools = build(tmp_path, monkeypatch)
    real_floor = live.read_admission_floor

    def infeasible_floor(snap, pages=(), *, require_feasible=False, measure=None,
                         violations_out=None):
        if require_feasible:
            if violations_out is not None:
                violations_out.append("prompt_tokens")
            return None
        return real_floor(snap, pages, measure=measure, violations_out=violations_out)

    monkeypatch.setattr(live, "read_admission_floor", infeasible_floor)
    (tmp_path / "a.py").write_text("value = 1\n")
    (status, text, _), metadata = read(rt, tmp_path, "a.py")

    state = admission._state(rt, body["messages"])
    assert state.snapshot is not None  # the floor path was really taken
    assert status == admission.REFUSED_ADMISSION
    assert text.startswith("Not delivered:")                         # (1)
    assert "a.py" in text
    assert "the request is already at its size limit" in text
    assert not OLD_WORDING.match(text)                               # (2)
    assert metadata["next_cursor"] in text                           # (3)
    assert metadata["refusal_cause"] == "floor_infeasible"           # (4)
    [event] = [e for e in _events(rt) if e["decision"] == admission.REFUSED_ADMISSION]
    assert event["refusal_cause"] == "floor_infeasible"              # (5)
    assert event["reason_code"] == ReasonCode.FLOOR_INFEASIBLE.value
