"""A read of content that is no longer shown is recovery and is never refused.

The read-budget governor refused "read budget exhausted" even when the requested
span had been evicted from the request or was listed "not shown; read this path".
Class pin: for every refusal site in IdleGovernor, a read marked not visible is
allowed under every budget state; a visible duplicate keeps today's refusal.
"""
from __future__ import annotations

import pytest

from kernel_idle_governor import GovernorDecision, IdleGovernor
from kernel_read_exec import read_span_visible
from kernel_request_source_window import _render_ranges, project_source_window

N1, N2 = 3, 5


def _read(path: str, turn: int, **extra: object) -> dict[str, object]:
    return {"kind": "read", "path": path, "turn": turn,
            "content_digest": "d-" + path, **extra}


def _exhaust(gov: IdleGovernor) -> None:
    """Reach the governed state by re-reading one visible path until it refuses."""
    for turn in range(N2 + 2):
        gov.should_allow(_read("a.py", turn))
    assert gov.governed


def _states():
    fresh = IdleGovernor(n1=N1, n2=N2)
    warned = IdleGovernor(n1=N1, n2=N2)
    for turn in range(N1):
        warned.should_allow(_read("a.py", turn))
    at_edge = IdleGovernor(n1=N1, n2=N2)
    for turn in range(N2 - 1):
        at_edge.should_allow(_read("a.py", turn))
    governed = IdleGovernor(n1=N1, n2=N2)
    _exhaust(governed)
    return {"fresh": fresh, "warned": warned, "edge": at_edge, "governed": governed}


@pytest.mark.parametrize("state", ["fresh", "warned", "edge", "governed"])
@pytest.mark.parametrize("repeat", [False, True])
def test_not_visible_read_is_never_refused_in_any_budget_state(state, repeat):
    gov = _states()[state]
    # Already-delivered key (a re-read) and a first read: both must pass when not shown.
    path = "a.py" if repeat else "never_seen.py"
    before_denials = gov.denials
    before_consec = gov.consecutive_non_product
    decision = gov.should_allow(_read(path, 99, visible=False))
    assert decision == GovernorDecision.ALLOW
    assert gov.last_receipt.event == "IDLE_RECOVERY"
    assert gov.denials == before_denials
    assert gov.recovery_reads == 1
    assert gov.consecutive_non_product == before_consec, "recovery must not spend the budget"


def test_visible_duplicate_keeps_todays_refusal():
    gov = _states()["governed"]
    assert gov.should_allow(_read("a.py", 50, visible=True)) == GovernorDecision.DENY
    assert gov.should_allow(_read("a.py", 51)) == GovernorDecision.DENY  # unknown = legacy
    assert gov.recovery_reads == 0


def test_recovery_reads_do_not_trigger_warning_or_governing():
    gov = IdleGovernor(n1=N1, n2=N2)
    for turn in range(N2 * 3):
        assert gov.should_allow(_read("a.py", turn, visible=False)) == GovernorDecision.ALLOW
    assert not gov.governed and gov.warnings == 0 and gov.recovery_reads == N2 * 3


def test_recovery_read_during_governed_does_not_unblock_visible_duplicates():
    gov = _states()["governed"]
    gov.should_allow(_read("a.py", 60, visible=False))
    assert gov.should_allow(_read("a.py", 61, visible=True)) == GovernorDecision.DENY


def _window(path: str, revision: str, total: int, ranges: list[tuple[int, int]]) -> dict:
    lines = [f"source line {number}\n" for number in range(1, total + 1)]
    rendered = project_source_window(_render_ranges(
        lines, [{"start_line": a, "end_line": b} for a, b in ranges],
        path=path, revision=revision, total_lines=total), owner="recent_editable_read")
    assert rendered
    return {"role": "user", "content": rendered}


def test_read_span_visible_distinguishes_shown_from_evicted():
    rev = "a" * 64
    msgs = [_window("src/x.py", rev, 100, [(1, 40), (41, 60)]),
            {"role": "user", "content": "[Source coverage]\nsrc/y.py: not shown; read this path to see current source."}]
    inside = {"effective_range": {"start_line": 10, "end_line": 50}, "revision": rev}
    outside = {"effective_range": {"start_line": 10, "end_line": 80}, "revision": rev}
    other_rev = {"effective_range": {"start_line": 10, "end_line": 50}, "revision": "b" * 64}
    assert read_span_visible(msgs, "src/x.py", inside) is True
    assert read_span_visible(msgs, "src/x.py", outside) is False
    assert read_span_visible(msgs, "src/x.py", other_rev) is False
    assert read_span_visible(msgs, "src/y.py", inside) is False
    assert read_span_visible(None, "src/x.py", inside) is None
    assert read_span_visible(msgs, "src/x.py", None) is None
    assert read_span_visible(msgs, "src/y.py", None, "x" * 5) is None
    text = "tool output that is plainly present in the request"
    assert read_span_visible([{"role": "tool", "content": "ok " + text}], "q.py", None, text) is True
