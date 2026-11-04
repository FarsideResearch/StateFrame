from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernel_request_diet import (
    FALLBACK_SAFETY_MARGIN,
    INITIAL_BYTES_PER_TOKEN,
    RequestTokenMeter,
)


FIXTURE = Path(__file__).parent / "fixtures" / "estimator_rows.json"
TEMPLATE_OVERHEAD_TOKENS = 1_148


def _replay_recorded_estimates() -> tuple[list[tuple[str, int, str, int]], int]:
    rows = json.loads(FIXTURE.read_text())
    meters: dict[str, RequestTokenMeter] = {}
    margins: list[tuple[str, int, str, int]] = []

    for row in rows:
        real = row["real"]
        if type(real) is not int:
            continue
        run = row["run"]
        meter = meters.get(run)
        if meter is None:
            meter = RequestTokenMeter(
                disabled=True, template_overhead_tokens=TEMPLATE_OVERHEAD_TOKENS
            )
            meters[run] = meter
        samples_before = meter._calibration_samples
        estimate, method = meter._measure_with_wire_bytes({}, row["bytes"])
        assert method == "calibrated_bytes"
        margin = estimate - real
        mode = "cold" if samples_before < 2 else "calibrated"
        assert margin >= 0, (
            f"{run} send {row['n']} estimate under-counted by {-margin} tokens "
            f"after {samples_before} calibration samples"
        )
        margins.append((run, row["n"], mode, margin))
        meter.observe_response(row["bytes"], real, 0)

    return margins, len(rows)


def test_estimate_overcounts_recorded_prompt_usage() -> None:
    margins, fixture_rows = _replay_recorded_estimates()
    assert fixture_rows == 110
    assert len(margins) == 109
    cold_min = min(margin for _run, _send, mode, margin in margins if mode == "cold")
    calibrated_min = min(margin for _run, _send, mode, margin in margins if mode == "calibrated")

    print(
        f"estimator replay: {len(margins)} rows, 0 undercounts; "
        f"minimum cold margin +{cold_min:,} tokens; "
        f"minimum calibrated margin +{calibrated_min:,} tokens"
    )
    assert cold_min == 7_789
    assert calibrated_min == 2_863


def test_estimator_replay_catches_single_sample_uncapped_control(monkeypatch) -> None:
    def old_fallback_ratio(meter: RequestTokenMeter) -> tuple[float, float]:
        if meter._minimum_bytes_per_token is None:
            return INITIAL_BYTES_PER_TOKEN, 0.0
        return (
            meter._minimum_bytes_per_token * (1.0 - FALLBACK_SAFETY_MARGIN),
            FALLBACK_SAFETY_MARGIN,
        )

    monkeypatch.setattr(RequestTokenMeter, "_fallback_ratio", old_fallback_ratio)
    with pytest.raises(AssertionError, match=r"r2-owner send 2.*515 tokens"):
        _replay_recorded_estimates()


def test_fallback_ratio_waits_for_two_calibration_samples() -> None:
    meter = RequestTokenMeter(disabled=True)
    meter.observe_response(5_000, 1_000, 0)

    assert meter._fallback_ratio() == (INITIAL_BYTES_PER_TOKEN, 0.0)


def test_fallback_ratio_caps_after_two_calibration_samples() -> None:
    meter = RequestTokenMeter(disabled=True)
    meter.observe_response(5_000, 1_000, 0)
    meter.observe_response(4_000, 1_000, 0)

    assert meter._fallback_ratio() == (3.0, FALLBACK_SAFETY_MARGIN)
