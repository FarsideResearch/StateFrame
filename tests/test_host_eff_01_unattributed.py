"""Unattributed host time is reported separately from tool_execution.

Gaps between a response and the next submission that no host span covers were
booked to tool_execution with nothing marking them as unknown. finish_submission
now also reports them as unattributed_host_ns. That field is a subset of
steps_ns["tool_execution"]; every other field keeps its prior value and order.
"""

from __future__ import annotations

import json
import random

import kernel_measurement as measurement

_PRIORITY = {
    "tool_execution": 0, "prepare_subphases": 1, "menu_composition": 2,
    "contracts": 3, "budget_diet": 4, "journal_io": 5,
}
_NAMES = (
    "executor", "prepare", "menu", "contract", "budget", "journal",
    "fsync", "binding", "Readiness", "DIET", "other",
)
_NEW_KEY = "unattributed_host_ns"


def _clip(spans, manual, started, ended):
    clipped = []
    for name, start, end in [*spans, *((name, start, ended) for name, start in manual)]:
        left, right = max(started, int(start)), min(ended, int(end))
        if right > left:
            clipped.append((str(name), left, right))
    return clipped


def _old_attribution(spans, manual, started, ended):
    """Pre-sweep interval scan from the attribution rules, kept as the parity oracle."""
    clipped = _clip(spans, manual, started, ended)
    boundaries = {started, ended}
    for _, start, end in clipped:
        boundaries.update((start, end))
    ordered = sorted(boundaries)
    buckets = dict.fromkeys((
        "prepare_subphases", "menu_composition", "contracts", "budget_diet",
        "journal_io", "tool_execution",
    ), 0)
    breakdown = {}
    for left, right in zip(ordered, ordered[1:]):  # noqa: RUF007 - mirrors old scan
        if right <= left:
            continue
        active = [row for row in clipped if row[1] <= left and row[2] >= right]
        if active:
            name, _, _ = min(active, key=lambda row: (
                -_PRIORITY[measurement._turn_timing_bucket(row[0])],
                row[2] - row[1], row[0],
            ))
            bucket = measurement._turn_timing_bucket(name)
        else:
            name, bucket = "tool_execution", "tool_execution"
        duration = right - left
        buckets[bucket] += duration
        breakdown[name] = breakdown.get(name, 0) + duration
    return buckets, breakdown


def _uncovered_oracle(spans, manual, started, ended):
    """Gap time where no clipped span is active."""
    clipped = _clip(spans, manual, started, ended)
    boundaries = {started, ended}
    for _, start, end in clipped:
        boundaries.update((start, end))
    ordered = sorted(boundaries)
    total = 0
    for left, right in zip(ordered, ordered[1:]):  # noqa: RUF007 - mirrors old scan
        if right > left and not any(row[1] <= left and row[2] >= right for row in clipped):
            total += right - left
    return total


def _finish(monkeypatch, spans, manual, started=100, ended=200):
    monkeypatch.setenv("STATEFRAME_TURN_TIMING", "on")
    monkeypatch.setattr(measurement.time, "perf_counter_ns", lambda: ended)
    recorder = measurement.TurnTimingRecorder()
    recorder._open = {
        "request_id": "before", "step": 7, "provider_ns": 23,
        "started_ns": started, "spans": spans, "manual": manual,
    }
    return recorder.finish_submission({"request_id": "after"})


def test_uncovered_gaps_are_reported_as_unattributed(monkeypatch):
    # Window is 100..200 ns. Literal expectations, not the oracle, so the RED run
    # fails on an assertion about the field, not on a helper.
    actual = _finish(monkeypatch, [], [])
    assert _NEW_KEY in actual, "finish_submission must report unattributed_host_ns"
    assert actual[_NEW_KEY] == 100
    assert actual["steps_ns"]["tool_execution"] == 100

    actual = _finish(monkeypatch, [("executor", 120, 150)], [])
    assert actual[_NEW_KEY] == 70  # 100..120 and 150..200
    assert actual["steps_ns"]["tool_execution"] == 100

    # A span in the tool_execution bucket that is active is attributed, not a gap.
    actual = _finish(monkeypatch, [("executor", 100, 200)], [])
    assert actual[_NEW_KEY] == 0
    assert actual["steps_ns"]["tool_execution"] == 100

    actual = _finish(monkeypatch, [("menu", 100, 180)], [])
    assert actual[_NEW_KEY] == 20  # only 180..200 is uncovered

    # Spans entirely outside the window clip away, leaving the whole gap.
    actual = _finish(monkeypatch, [("journal", 0, 50), ("menu", 260, 300)], [])
    assert actual[_NEW_KEY] == 100

    # Zero-length window: nothing to attribute.
    actual = _finish(monkeypatch, [], [], 100, 100)
    assert actual[_NEW_KEY] == 0


def test_unattributed_is_additive_and_matches_old_attribution(monkeypatch):
    cases = [
        ([], []),
        ([("journal", 0, 300), ("menu", 100, 200)], []),
        ([("prepare", 100, 150), ("budget", 150, 200)], []),
        ([("executor", 120, 120), ("prepare", 200, 210),
          ("journal", 0, 100)], [("menu", 175)]),
        ([("journal", 190, 110), ("menu", -10, 101),
          ("budget", 199, 250)], [("contract", 200), ("prepare", 50)]),
        ([("DIET", 110, 190), ("budget", 110, 190)], [("contract", 150)]),
        ([], [("menu", 150)]),
    ]
    rng = random.Random(1117)
    for _ in range(120):
        spans = []
        for _ in range(rng.randrange(0, 40)):
            start, end = sorted((rng.randrange(0, 301), rng.randrange(0, 301)))
            spans.append((rng.choice(_NAMES), start, end))
        manual = [(rng.choice(_NAMES), rng.randrange(0, 301))
                  for _ in range(rng.randrange(0, 5))]
        cases.append((spans, manual))

    for spans, manual in cases:
        actual = _finish(monkeypatch, spans, manual)
        buckets, breakdown = _old_attribution(spans, manual, 100, 200)
        uncovered = _uncovered_oracle(spans, manual, 100, 200)

        # Additive: removing the new key leaves exactly the prior record, byte for byte.
        assert _NEW_KEY in actual
        prior = {key: value for key, value in actual.items() if key != _NEW_KEY}
        expected_prior = {
            "producer": "kernel_measurement", "request_id": "before",
            "next_request_id": "after", "step": 7, "provider_ns": 23,
            "wall_gap_ns": 100, "steps_ns": buckets,
            "step_breakdown_ns": breakdown, "steps_sum_ns": sum(buckets.values()),
            "reconciliation_error_ns": 100 - sum(buckets.values()),
            "reconciles_within_5_percent": True, "dispatch": False,
        }
        assert prior == expected_prior
        assert json.dumps(prior) == json.dumps(expected_prior)

        # The new field matches the oracle and is a subset of tool_execution.
        assert actual[_NEW_KEY] == uncovered
        assert 0 <= actual[_NEW_KEY] <= buckets["tool_execution"]
