"""FU-070 paging: an oversized owned file pages from ordinal 2 instead of pinning whole-file FULL.

Protected (owned/edit-site) paths were FULL-only in every domain, and a whole-file plan
unit longer than one read page has no recovery route, so it could never be DEFERRED. At
due_ordinal 2 the owned file's plan closure plus fixed bytes exceeded the byte cap
(FLOOR_INFEASIBLE). Subframe workers now split plan units into read-page spans with routes
and let non-DUE, non-pinned protected spans page from ordinal 2. DUE debt and recovery pins
stay FULL, and allocator-OFF/ordinal-1 domains are unchanged.
"""
from __future__ import annotations

import json
import sys
import types
from dataclasses import replace

import pytest

from kernel_working_set_allocator import DEFERRED, FULL
from kernel_working_set_prepare import Blocked, PreparedCandidate, prepare_candidate
from kernel_working_set_render import _slot_modes, propose_slots
from kernel_working_set_snapshot import PreparationSnapshot
from kernel_working_set_stop import ReasonCode
from kernel_working_set_windows import _plan_spans
from test_working_set_render import source

CAP = 82_026
LINES = [f"VALUE_{n:04d} = {n:04d}  # pad\n" for n in range(1, 2801)]
# Fixed envelope sized like the production e2e floor (~20.7 KB), so the owned plan
# closure plus fixed bytes is what crosses the byte cap.
FIXED_ENVELOPE = "policy " * 2_900


@pytest.fixture
def unit_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a compiled parallel subframe: kernel_unit_policy.active() is non-None."""
    monkeypatch.setitem(sys.modules, "kernel_unit_policy",
                        types.SimpleNamespace(active=lambda: object()))


def _snapshot(*slots, due_ordinal: int = 2, cap: int = CAP):
    body = {"model": "local", "messages": [
        {"role": "system", "content": FIXED_ENVELOPE},
        {"role": "user", "content": "Task: inspect the owned source"},
        {"role": "user", "content": "opaque newest suffix"}], "max_tokens": 100}
    return PreparationSnapshot(
        json.dumps(body).encode(), tuple(slots), "request", due_ordinal, cap,
        protected_paths=("a.py",), current_revisions=(("a.py", slots[0].identity.revision),))


def _span(first: int, last: int, *, state: str = "CLOSURE", key: str = "plan",
          ordinal: int = 2):
    text = "".join(LINES[first - 1:last])
    return replace(source(key, text, first=first, state=state, ordinal=ordinal), plan_only=True)


def _delivered_window(first: int, last: int, *, key: str = "w1"):
    text = "".join(LINES[first - 1:last])
    return replace(source(key, text, first=first, state="RESIDENT", ordinal=1),
                   previous_mode="F")


def _domain(snap: PreparationSnapshot, key: str):
    slot = next(s for s in snap.sources if s.key == key)
    return _slot_modes(snap, slot)


def test_delivered_protected_window_pages_from_ordinal_two_in_subframe(unit_active) -> None:
    assert _domain(_snapshot(_delivered_window(1, 1000)), "w1") == (FULL, DEFERRED)


def test_page_sized_plan_span_pages_from_ordinal_two_in_subframe(unit_active) -> None:
    assert _domain(_snapshot(_span(2001, 2800, key="plan")), "plan") == (FULL, DEFERRED)


def test_unrecoverable_whole_file_plan_stays_full_in_subframe(unit_active) -> None:
    # A span longer than one read page has no route; DEFERRED would be an unrecoverable omission.
    assert _domain(_snapshot(_span(1, 2800)), "plan") == (FULL,)


def test_due_debt_and_recovery_pin_stay_full_for_protected_paths(unit_active) -> None:
    due = source("due", "".join(LINES[1000:2000]), first=1001, state="DUE", ordinal=2)
    pinned = replace(_delivered_window(1, 1000), recovery_pin=True)
    assert _domain(_snapshot(due), "due") == (FULL,)
    assert _domain(_snapshot(pinned), "w1") == (FULL,)


def test_ordinal_one_and_allocator_off_keep_protected_slots_full(unit_active, monkeypatch) -> None:
    assert _domain(_snapshot(_delivered_window(1, 1000), due_ordinal=1), "w1") == (FULL,)
    assert _domain(_snapshot(_span(2001, 2800), due_ordinal=1), "plan") == (FULL,)
    monkeypatch.setitem(sys.modules, "kernel_unit_policy",
                        types.SimpleNamespace(active=lambda: None))
    assert _domain(_snapshot(_delivered_window(1, 1000)), "w1") == (FULL,)
    assert _domain(_snapshot(_span(2001, 2800)), "plan") == (FULL,)


def test_plan_spans_split_oversized_unit_into_page_sized_routes(unit_active) -> None:
    spans = _plan_spans(LINES, 1, 2800, "a.py", "sha256:" + "1" * 64, ["anchor"])
    assert [(first, last) for first, last, _ in spans] == [(1, 1000), (1001, 2000), (2001, 2800)]
    assert all(anchors == [] for _, _, anchors in spans)


def test_plan_spans_keep_small_unit_and_inactive_unit_unchanged(monkeypatch) -> None:
    revision = "sha256:" + "1" * 64
    monkeypatch.setitem(sys.modules, "kernel_unit_policy",
                        types.SimpleNamespace(active=lambda: None))
    assert _plan_spans(LINES, 1, 2800, "a.py", revision, ["x"]) == ((1, 2800, ["x"]),)
    monkeypatch.setitem(sys.modules, "kernel_unit_policy",
                        types.SimpleNamespace(active=lambda: object()))
    assert _plan_spans(LINES, 5, 40, "a.py", revision, ["x"]) == ((5, 40, ["x"]),)


def test_protected_full_only_outside_subframe_is_unchanged() -> None:
    slots = propose_slots(_snapshot(_delivered_window(1, 1000), _span(2001, 2800)))
    assert all(tuple(mode for mode, _ in s.domain) == (FULL,) for s in slots)


def test_oversized_owned_file_ordinal_two_is_feasible_under_cap(unit_active) -> None:
    # Ordinal 1 delivered 1-1000. Ordinal 2 owes 1001-2000 and the owned file's plan closure
    # is split into page-sized spans. The FULL-only closure plus fixed bytes was infeasible.
    snap = _snapshot(
        _delivered_window(1, 1000),
        source("w2", "".join(LINES[1000:2000]), first=1001, state="DUE", ordinal=2),
        _span(1001, 2000, key="plan_b"),
        _span(2001, 2800, key="plan_c"),
    )
    result = prepare_candidate(snap)
    assert isinstance(result, PreparedCandidate), getattr(result, "reason", result)
    assert len(result.wire) <= CAP
    assert b"VALUE_1001" in result.wire  # the currently owed window is delivered


def test_whole_file_plan_outside_subframe_still_blocks_as_before() -> None:
    # Parity guard: outside a subframe the protected FULL-only domain is unchanged.
    snap = _snapshot(
        _delivered_window(1, 1000),
        source("w2", "".join(LINES[1000:2000]), first=1001, state="DUE", ordinal=2),
        _span(1, 2800),
    )
    result = prepare_candidate(snap)
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.FLOOR_INFEASIBLE
