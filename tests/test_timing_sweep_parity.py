"""Golden behavior for host timing attribution and span serialization."""

from __future__ import annotations

import json
import random
import time

import kernel_measurement as measurement

_PRIORITY = {
    "tool_execution": 0, "prepare_subphases": 1, "menu_composition": 2,
    "contracts": 3, "budget_diet": 4, "journal_io": 5,
}
_NAMES = (
    "executor", "prepare", "menu", "contract", "budget", "journal",
    "fsync", "binding", "Readiness", "DIET", "other",
)


def _old_attribution(spans, manual, started, ended):
    """The pre-sweep interval scan, retained here as a test oracle."""
    clipped = []
    for name, start, end in [*spans, *((name, start, ended) for name, start in manual)]:
        left, right = max(started, int(start)), min(ended, int(end))
        if right > left:
            clipped.append((str(name), left, right))
    boundaries = {started, ended}
    for _, start, end in clipped:
        boundaries.add(start)
        boundaries.add(end)
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


def _finish(monkeypatch, spans, manual, started=100, ended=200):
    monkeypatch.setenv("STATEFRAME_TURN_TIMING", "on")
    monkeypatch.setattr(measurement.time, "perf_counter_ns", lambda: ended)
    recorder = measurement.TurnTimingRecorder()
    recorder._open = {
        "request_id": "before", "step": 7, "provider_ns": 23,
        "started_ns": started, "spans": spans, "manual": manual,
    }
    return recorder.finish_submission({"request_id": "after"})


def test_timing_sweep_matches_interval_scan_and_all_fields(monkeypatch):
    cases = [
        ([], []),
        ([("journal", 0, 300), ("menu", 100, 200)], []),
        ([("prepare", 100, 150), ("budget", 150, 200)], []),
        ([("menu", 100, 180), ("journal", 120, 160),
          ("journal", 130, 150), ("fsync", 130, 150)], []),
        ([("executor", 120, 120), ("prepare", 200, 210),
          ("journal", 0, 100)], [("menu", 175)]),
        ([("journal", 190, 110), ("menu", -10, 101),
          ("budget", 199, 250)], [("contract", 200), ("prepare", 50)]),
        ([("DIET", 110, 190), ("budget", 110, 190),
          ("budget", 110, 190)], [("contract", 150)]),
    ]
    rng = random.Random(721)
    for _ in range(80):
        spans = []
        for _ in range(rng.randrange(0, 45)):
            start, end = sorted((rng.randrange(0, 301), rng.randrange(0, 301)))
            spans.append((rng.choice(_NAMES), start, end))
        manual = [(rng.choice(_NAMES), rng.randrange(0, 301))
                  for _ in range(rng.randrange(0, 6))]
        cases.append((spans, manual))
    for spans, manual in cases:
        actual = _finish(monkeypatch, spans, manual)
        buckets, breakdown = _old_attribution(spans, manual, 100, 200)
        expected = {
            "producer": "kernel_measurement", "request_id": "before",
            "next_request_id": "after", "step": 7, "provider_ns": 23,
            "wall_gap_ns": 100, "steps_ns": buckets,
            "step_breakdown_ns": breakdown, "steps_sum_ns": sum(buckets.values()),
            "reconciliation_error_ns": 100 - sum(buckets.values()),
            "reconciles_within_5_percent": True, "dispatch": False,
        }
        assert actual == expected
        assert json.dumps(actual) == json.dumps(expected)


def test_timing_zero_gap_and_disabled(monkeypatch):
    actual = _finish(monkeypatch, [("journal", 100, 100)], [], 100, 100)
    assert actual["wall_gap_ns"] == actual["steps_sum_ns"] == 0
    assert actual["step_breakdown_ns"] == {}
    monkeypatch.setenv("STATEFRAME_TURN_TIMING", "off")
    recorder = measurement.TurnTimingRecorder()
    recorder._open = {"started_ns": 100, "spans": [], "manual": []}
    assert recorder.finish_submission({}) is None
    assert recorder._open is None


def test_timing_5000_overlapping_spans_finishes_under_50_ms(monkeypatch):
    spans = [(_NAMES[index % len(_NAMES)], index, 10_000 - index)
             for index in range(5_000)]
    # Measure CPU work so unrelated workers on a loaded host cannot cause a
    # false regression through scheduler delay.
    started = time.process_time()
    actual = _finish(monkeypatch, spans, [], started=0, ended=10_000)
    elapsed = time.process_time() - started

    expected_buckets, expected_breakdown = _old_attribution(
        spans, [], 0, 10_000,
    )
    assert actual is not None
    assert actual["steps_ns"] == expected_buckets
    assert actual["step_breakdown_ns"] == expected_breakdown
    assert elapsed < 0.050, f"5,000 span attribution used {elapsed:.3f}s CPU"


def test_host_span_serialization_has_exact_fields_and_order():
    span = measurement.HostSpan(
        "sample", 10, 17, "request_preparation", 4, "action",
        (("z", "last"), ("a", "first")), "root", 6, 1, 2, 3, 4, 5, 6,
    )
    assert json.dumps(span.to_dict(), separators=(",", ":")) == (
        '{"name":"sample","started_ns":10,"ended_ns":17,"elapsed_ns":7,'
        '"boundary":"request_preparation","sequence":4,"action_id":"action",'
        '"attributes":{"z":"last","a":"first"},"parent_span":"root",'
        '"cpu_ns":6,"cache_hits":1,"cache_misses":2,"bytes_scanned":3,'
        '"records_folded":4,"files_read":5,"fsyncs":6}'
    )


def test_host_span_serialization_uses_elapsed_property():
    class CustomSpan(measurement.HostSpan):
        @property
        def elapsed_ns(self) -> int:
            return 99

    span = CustomSpan("sample", 10, 17)
    assert span.to_dict()["elapsed_ns"] == 99


def test_host_span_finish_preserves_stack_and_trace(monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOST_SPANS", "on")
    monkeypatch.setattr(measurement.time, "perf_counter_ns", lambda: 100)
    recorder = measurement.HostSpanRecorder()
    other = measurement.HostSpanRecorder()
    token = measurement._ACTIVE_HOST_SPAN_STACK.set(())
    try:
        outer = recorder.start_span("outer", force=True)
        middle = other.start_span("middle", force=True)
        inner = recorder.start_span("inner", force=True)
        assert outer is not None and middle is not None and inner is not None
        recorder.finish_span(outer, ended_ns=120, cache_hits=2)
        assert measurement._ACTIVE_HOST_SPAN_STACK.get() == (
            (other, middle.span_id), (recorder, inner.span_id))
        recorder.finish_span(inner, ended_ns=130, fsyncs=1)
        assert measurement._ACTIVE_HOST_SPAN_STACK.get() == ((other, middle.span_id),)
        other.finish_span(middle, ended_ns=140)
        assert measurement._ACTIVE_HOST_SPAN_STACK.get() == ()
        assert recorder.snapshot()[0]["cache_hits"] == 2
        assert recorder.snapshot()[1]["fsyncs"] == 1
        assert [row["elapsed_ns"] for row in recorder.trace_snapshot()] == [20, 30]
    finally:
        measurement._ACTIVE_HOST_SPAN_STACK.reset(token)


def test_host_span_bounded_buffers_keep_oldest_to_newest_order(monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOST_SPANS", "on")
    monkeypatch.setattr(measurement.time, "perf_counter_ns", lambda: 100)
    recorder = measurement.HostSpanRecorder(limit=2)
    token = measurement._ACTIVE_HOST_SPAN_STACK.set(())
    try:
        for name in ("first", "second", "third"):
            handle = recorder.start_span(name, force=True)
            assert handle is not None
            recorder.finish_span(handle, ended_ns=110)
        assert [row["name"] for row in recorder.snapshot()] == ["second", "third"]
        assert [row["name"] for row in recorder.trace_snapshot()] == ["second", "third"]
        recorder.limit = 1
        handle = recorder.start_span("fourth", force=True)
        assert handle is not None
        recorder.finish_span(handle, ended_ns=110)
        # The old list path evicts one row per append even if limit changes.
        assert [row["name"] for row in recorder.snapshot()] == ["third", "fourth"]
        assert [row["name"] for row in recorder.trace_snapshot()] == ["third", "fourth"]
    finally:
        measurement._ACTIVE_HOST_SPAN_STACK.reset(token)
