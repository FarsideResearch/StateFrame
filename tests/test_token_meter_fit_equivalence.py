"""Memoized token metering must leave every fit decision and evidence number unchanged.

Each realistic finalize/budget test is driven twice: memos disabled (the old path) and
enabled. A recorder captures every measurement, finalize result and journal event.
"""
from __future__ import annotations

import hashlib
import inspect
import json

import pytest

import kernel_frame_allocation_capture as allocation
import kernel_request_diet as diet
from kernel_request_diet import RequestTokenMeter

import test_final_cache_layout as cache_layout
import test_late_fit_page_deferral as late_fit


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _run(case, tmp_path, enabled: bool) -> tuple[list, dict[str, int]]:
    log: list = []
    stats = {"measures": 0, "memo_hits": 0, "memo_computes": 0}
    mp = pytest.MonkeyPatch()
    real_measure = RequestTokenMeter._measure_with_wire_bytes
    real_memo = diet.memoized_value
    real_finalize = allocation.finalize_frame_candidate

    def measure(self, body, wire_bytes):
        if not enabled:
            self.__dict__.pop("_count_key_last", None)
        result = real_measure(self, body, wire_bytes)
        stats["measures"] += 1
        log.append(("measure", _digest(body), wire_bytes, result))
        return result

    def memo(body, name, compute):
        if not enabled:
            stats["memo_computes"] += 1
            return compute()
        ran = []
        value = real_memo(body, name, lambda: ran.append(1) or compute())
        stats["memo_computes" if ran else "memo_hits"] += 1
        return value

    def finalize(runtime, capture, body, **kwargs):
        result = real_finalize(runtime, capture, body, **kwargs)
        events = getattr(getattr(capture, "journal", None), "events", [])
        log.append(("finalize", _digest(result), len(json.dumps(result)),
                    _digest(events)))
        return result

    mp.setattr(RequestTokenMeter, "_measure_with_wire_bytes", measure)
    mp.setattr(diet, "memoized_value", memo)
    mp.setattr(allocation, "finalize_frame_candidate", finalize)
    try:
        names = inspect.signature(case).parameters
        case(**{k: v for k, v in (("monkeypatch", mp), ("tmp_path", tmp_path)) if k in names})
    finally:
        mp.undo()
    return log, stats


CASES = [
    late_fit.test_late_fit_defers_missing_page_and_noop_fit_preserves_body_bytes,
    late_fit.test_late_fit_defers_verified_partial_receipt_with_authorized_read,
    late_fit.test_late_fit_defers_partial_delivery_with_closure_omission,
    cache_layout.test_exact_plain_budget_reserves_layout_overhead_and_trims_once,
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.__name__)
def test_fit_decisions_and_evidence_identical_with_and_without_memo(case, tmp_path):
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    old_log, old_stats = _run(case, tmp_path / "old", enabled=False)
    new_log, new_stats = _run(case, tmp_path / "new", enabled=True)
    assert old_log == new_log
    assert old_stats["measures"] == new_stats["measures"]
    if case is not cache_layout.test_exact_plain_budget_reserves_layout_overhead_and_trims_once:
        assert new_stats["measures"] > 0  # that case budgets through its own estimator
    print(f"{case.__name__}: {new_stats}")
