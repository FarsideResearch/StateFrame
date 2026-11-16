"""Queue wait is capped by the engine only when the user sets STATEFRAME_QUEUE_WAIT_MAX_S.

A request queued on a healthy, reachable server must not be failed by an engine default.
Dead servers are caught by the provider liveness probe, which these tests do not change.
"""
from __future__ import annotations

import pytest

from stateframe.engine import kernel_bounded_http


@pytest.fixture(autouse=True)
def _clear_queue_wait_env(monkeypatch):
    monkeypatch.delenv("STATEFRAME_QUEUE_WAIT_MAX_S", raising=False)


def test_unset_queue_wait_has_no_engine_cap_and_hard_deadline_is_run_deadline():
    assert kernel_bounded_http.queue_wait_max_seconds() is None
    run_deadline = 5_000.0
    now = 1_000.0
    assert kernel_bounded_http._hard_deadline(run_deadline, now) == run_deadline


def test_set_queue_wait_caps_hard_deadline_at_now_plus_cap(monkeypatch):
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "600")
    assert kernel_bounded_http.queue_wait_max_seconds() == 600.0
    now = 1_000.0
    # Later run deadline: the user cap wins.
    assert kernel_bounded_http._hard_deadline(now + 10_000.0, now) == pytest.approx(now + 600.0)
    # Earlier run deadline: the run deadline wins.
    assert kernel_bounded_http._hard_deadline(now + 100.0, now) == pytest.approx(now + 100.0)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad"])
def test_invalid_queue_wait_still_raises(monkeypatch, value):
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", value)
    with pytest.raises(ValueError, match="STATEFRAME_QUEUE_WAIT_MAX_S"):
        kernel_bounded_http.queue_wait_max_seconds()
    with pytest.raises(ValueError, match="STATEFRAME_QUEUE_WAIT_MAX_S"):
        kernel_bounded_http._hard_deadline(2_000.0, 1_000.0)


def test_unset_queue_wait_leaves_outer_future_uncapped():
    from stateframe.engine.probes import react_repair
    from types import SimpleNamespace

    assert react_repair._transport_future_timeout(SimpleNamespace(), 0.3, False) is None


def test_unset_queue_wait_drops_queue_term_from_budget():
    from stateframe.engine import kernel_measurement

    # Only the non-queue budget remains: max(legacy request budget, 2 x wall).
    assert kernel_measurement.request_queue_budget_seconds(0.3) == pytest.approx(
        max(kernel_measurement.request_timeout_budget_seconds(), 0.6))
