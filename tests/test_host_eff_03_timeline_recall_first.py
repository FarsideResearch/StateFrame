"""Host-efficiency item 03: timeline recall runs before candidate rows are built.

When no host selector is set and the recall object can select, the relevance
result comes from recall alone. Building the candidate rows first was wasted
work (every hydrated row was discarded). These tests pin both the call shape
(no candidate build) and the output order, which must match the legacy path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kernel_timeline import TimelineIndex


class _RecallStub:
    """Recall double that returns fixed items and records its keyword arguments."""

    def __init__(self, items: list[dict[str, object]] | None = None, *, view: object | None = None) -> None:
        self._items = items or []
        self._view = view
        self.calls: list[dict[str, object]] = []

    def select(self, focus: object, *, budget_tokens: int, max_items: int, ledger: object,
               consume_requested: bool, track_shown: bool) -> object:
        self.calls.append({
            "focus": focus, "budget_tokens": budget_tokens, "max_items": max_items,
            "ledger": ledger, "consume_requested": consume_requested, "track_shown": track_shown,
        })
        if self._view is not None:
            return self._view
        return SimpleNamespace(items=list(self._items))


def _index(*, recall: object | None = None, selector: object | None = None) -> TimelineIndex:
    index = TimelineIndex("sess-03", recall=recall, selector=selector)
    index.append("action", {"tool": "read"}, turn=1, timestamp_ns=10,
                 content="read a.py", entities=("a.py",))
    index.append("receipt", {"ok": True}, turn=2, timestamp_ns=20,
                 content="receipt for b.py", entities=("b.py",))
    index.append("decision", {"choice": "keep"}, turn=3, timestamp_ns=30,
                 content="keep the cache", entities=())
    return index


@pytest.fixture
def candidate_calls(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Spy on _candidate_rows; each entry is the limit it was called with."""
    calls: list[object] = []
    original = TimelineIndex._candidate_rows

    def spy(self: TimelineIndex, focus: object, *, limit: int | None = None) -> list[dict[str, object]]:
        calls.append(limit)
        return original(self, focus, limit=limit)

    monkeypatch.setattr(TimelineIndex, "_candidate_rows", spy)
    return calls


def test_recall_only_path_skips_candidate_build(candidate_calls: list[object]) -> None:
    recall = _RecallStub([{"id": "r1", "turn": 9, "timestamp_ns": 90}])
    index = _index(recall=recall)
    candidate_calls.clear()

    rows = index._select_relevant("a.py", max_items=3, budget_tokens=200)

    assert candidate_calls == [], "candidate rows must not be built when recall alone selects"
    assert [row["id"] for row in rows] == ["r1"]
    assert rows[0]["source"] == "recall"


def test_recall_receives_legacy_arguments(candidate_calls: list[object]) -> None:
    recall = _RecallStub([])
    index = _index(recall=recall)

    index._select_relevant("a.py", max_items=4, budget_tokens=10)
    index._select_relevant("a.py", max_items=4, budget_tokens=300)

    assert recall.calls == [
        {"focus": "a.py", "budget_tokens": 32, "max_items": 4, "ledger": None,
         "consume_requested": False, "track_shown": False},
        {"focus": "a.py", "budget_tokens": 300, "max_items": 4, "ledger": None,
         "consume_requested": False, "track_shown": False},
    ]


def test_recall_rows_keep_recall_order_and_truncate(candidate_calls: list[object]) -> None:
    items = [{"id": f"r{n}", "turn": 9 - n} for n in range(5)]
    index = _index(recall=_RecallStub(items))

    rows = index._select_relevant("a.py", max_items=2, budget_tokens=200)

    assert [row["id"] for row in rows] == ["r0", "r1"]
    assert all(row["source"] == "recall" for row in rows)
    assert items[0] == {"id": "r0", "turn": 9}, "recall items must be copied, not mutated"


def test_recall_view_without_items_yields_empty(candidate_calls: list[object]) -> None:
    index = _index(recall=_RecallStub(view=object()))

    assert index._select_relevant("a.py", max_items=3, budget_tokens=200) == ()


def test_host_selector_still_receives_candidates(candidate_calls: list[object]) -> None:
    seen: list[list[dict[str, object]]] = []

    def selector(candidates: list[dict[str, object]], **_: object) -> list[dict[str, object]]:
        seen.append(list(candidates))
        return [{"id": "picked"}]

    recall = _RecallStub([{"id": "r1"}])
    index = _index(recall=recall, selector=selector)
    candidate_calls.clear()

    rows = index._select_relevant("a.py", max_items=3, budget_tokens=200)

    assert len(candidate_calls) == 1
    assert len(seen) == 1 and seen[0], "selector must be handed the built candidates"
    assert rows == ({"id": "picked"},)
    assert recall.calls == [], "recall is not consulted when the selector answers"


def test_selector_refusal_falls_back_to_recall(candidate_calls: list[object]) -> None:
    recall = _RecallStub([{"id": "r1"}])
    index = _index(recall=recall, selector=lambda candidates, **_: None)

    rows = index._select_relevant("a.py", max_items=3, budget_tokens=200)

    assert [row["id"] for row in rows] == ["r1"]
    assert len(recall.calls) == 1


def test_no_recall_keeps_legacy_recency_suffix(candidate_calls: list[object]) -> None:
    index = _index()

    rows = index._select_relevant("", max_items=2, budget_tokens=200)

    assert [row["turn"] for row in rows] == [2, 3]
    assert all("source" not in row for row in rows)
