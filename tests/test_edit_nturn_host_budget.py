"""Edit n-turn run budget counts host time only; time waiting on the provider is excluded.

Clock values are injected (no sleeping, no os.environ mutation). Engine modules are imported
the way tests/test_reason_log_limits.py does; new names are referenced inside test bodies so
the RED run reports per-test failures instead of a collection error.
"""
import math
from pathlib import Path

import pytest

import kernel_step_contract as sc

ENV_NAME = "STATEFRAME_EDIT_NTURN_RUN_BUDGET_SECONDS"
ROOT = Path(__file__).resolve().parents[1]
REACT_REPAIR = ROOT / "src" / "stateframe" / "engine" / "probes" / "react_repair.py"


def test_run_budget_constant_value_is_unchanged():
    assert sc.EDIT_NTURN_RUN_BUDGET_SECONDS == 1800.0


@pytest.mark.parametrize("environ", [{}, {ENV_NAME: ""}, {ENV_NAME: "   "}])
def test_run_budget_defaults_when_unset_or_empty(environ):
    assert sc.edit_nturn_run_budget_seconds(environ=environ) == 1800.0


@pytest.mark.parametrize("raw", ["off", "OFF", " Off "])
def test_run_budget_off_disables_budget(raw):
    assert sc.edit_nturn_run_budget_seconds(environ={ENV_NAME: raw}) is None


def test_run_budget_accepts_positive_seconds():
    assert sc.edit_nturn_run_budget_seconds(environ={ENV_NAME: "900"}) == 900.0


@pytest.mark.parametrize("raw", ["0", "-5", "nan", "inf", "1e400", "abc"])
def test_run_budget_rejects_invalid_values(raw):
    with pytest.raises(ValueError, match="positive number of seconds or off"):
        sc.edit_nturn_run_budget_seconds(environ={ENV_NAME: raw})


def test_slow_host_provider_wait_excluded_edit_turn_still_admitted():
    # 83 min wall clock, 4700 s of it waiting on the model: 300 s host time used.
    remaining = sc.edit_nturn_remaining_seconds(
        run_started=0.0, now=5000.0, provider_wait_seconds=4700.0,
        budget_seconds=1800.0, sandbox_deadline=None)
    assert remaining == pytest.approx(1500.0)
    assert sc.edit_nturn_budget_allows(remaining, 300.0) is True


def test_fast_host_same_wall_clock_without_wait_credit_is_exhausted():
    remaining = sc.edit_nturn_remaining_seconds(
        run_started=0.0, now=5000.0, provider_wait_seconds=0.0,
        budget_seconds=1800.0, sandbox_deadline=None)
    assert remaining == 0.0
    assert sc.edit_nturn_budget_allows(remaining, 300.0) is False


def test_fast_host_equivalence_subtracts_only_host_wait():
    remaining = sc.edit_nturn_remaining_seconds(
        run_started=0.0, now=1000.0, provider_wait_seconds=20.0,
        budget_seconds=1800.0, sandbox_deadline=None)
    assert remaining == pytest.approx(820.0)


def test_negative_provider_wait_is_clamped_to_zero():
    remaining = sc.edit_nturn_remaining_seconds(
        run_started=0.0, now=1000.0, provider_wait_seconds=-50.0,
        budget_seconds=1800.0, sandbox_deadline=None)
    assert remaining == pytest.approx(800.0)


def test_no_budget_means_unbounded_and_is_admitted():
    remaining = sc.edit_nturn_remaining_seconds(
        run_started=0.0, now=10_000.0, provider_wait_seconds=0.0,
        budget_seconds=None, sandbox_deadline=None)
    assert math.isinf(remaining)
    assert sc.edit_nturn_budget_allows(remaining, 300.0) is True


def test_nan_remaining_fails_closed():
    assert sc.edit_nturn_budget_allows(math.nan, 300.0) is False


def test_sandbox_deadline_wins_and_stays_wall_clock():
    remaining = sc.edit_nturn_remaining_seconds(
        run_started=0.0, now=100.0, provider_wait_seconds=90.0,
        budget_seconds=1800.0, sandbox_deadline=150.0)
    assert remaining == pytest.approx(50.0)


def test_sandbox_deadline_alone_without_budget():
    remaining = sc.edit_nturn_remaining_seconds(
        run_started=0.0, now=100.0, provider_wait_seconds=0.0,
        budget_seconds=None, sandbox_deadline=160.0)
    assert remaining == pytest.approx(60.0)


def test_react_repair_wiring_uses_host_time_budget():
    text = REACT_REPAIR.read_text(encoding="utf-8")
    assert "_edit_nturn_run_started + EDIT_NTURN_RUN_BUDGET_SECONDS" not in text
    assert "edit_nturn_remaining_seconds(" in text
    assert "_edit_nturn_provider_wait +=" in text
    # Both the successful and the failed provider call add their wait.
    assert text.count("_edit_nturn_provider_wait += time.monotonic() - _provider_t0") == 2
    assert any("nonlocal" in line and "_edit_nturn_provider_wait" in line
               for line in text.splitlines())


def test_runner_forwards_the_override_to_the_engine():
    import pytest
    from stateframe import runner

    assert runner.EDIT_NTURN_BUDGET_SETTING in runner.SHELL_SETTINGS
    assert runner.resolve_edit_nturn_budget_env({}) == {}
    assert runner.resolve_edit_nturn_budget_env({runner.EDIT_NTURN_BUDGET_SETTING: " 3600 "}) == {
        runner.EDIT_NTURN_BUDGET_SETTING: "3600"}
    assert runner.resolve_edit_nturn_budget_env({runner.EDIT_NTURN_BUDGET_SETTING: "OFF"}) == {
        runner.EDIT_NTURN_BUDGET_SETTING: "off"}
    for bad in ("0", "-1", "nan", "inf", "lots"):
        with pytest.raises(runner.RunError):
            runner.resolve_edit_nturn_budget_env({runner.EDIT_NTURN_BUDGET_SETTING: bad})
