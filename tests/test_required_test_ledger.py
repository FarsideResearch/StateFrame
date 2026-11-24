from __future__ import annotations

import json

import pytest

from kernel_required_tests import RequiredTests, parse_pytest_report
from stateframe import runner
from stateframe.runner import RunError, _required_test_ledger_enabled


def test_reports_regressions_from_baseline_passes_only():
    ledger = RequiredTests()
    ledger.observe("baseline", complete=True, outcomes={
        "tests/test_math.py::test_add": "passed",
        "tests/test_math.py::test_subtract": "passed",
        "tests/test_math.py::test_old_failure": "failed",
    })
    ledger.observe("current", complete=True, outcomes={
        "tests/test_math.py::test_add": "passed",
        "tests/test_math.py::test_subtract": "failed",
        "tests/test_math.py::test_new": "passed",
    })

    assert ledger.report() == {
        "missing": [],
        "failed": ["tests/test_math.py::test_subtract"],
        "passed": ["tests/test_math.py::test_add"],
        "state": "regressed",
    }


def test_renamed_or_deleted_test_ids_are_missing_without_guessing():
    ledger = RequiredTests()
    ledger.observe("baseline", complete=True, outcomes={
        "tests/test_math.py::test_old_name": "passed",
    })
    ledger.observe("current", complete=True, outcomes={
        "tests/test_math.py::test_new_name": "passed",
    })

    assert ledger.report() == {
        "missing": ["tests/test_math.py::test_old_name"],
        "failed": [],
        "passed": [],
        "state": "regressed",
    }


def test_incomplete_observations_do_not_replace_either_complete_run():
    ledger = RequiredTests()
    ledger.observe("baseline", complete=True, outcomes={"case-a": "passed"})
    ledger.observe("baseline", complete=False, outcomes={"case-b": "passed"})
    ledger.observe("current", complete=True, outcomes={"case-a": "passed"})
    ledger.observe("current", complete=False, outcomes={"case-a": "failed"})

    assert ledger.report() == {
        "missing": [], "failed": [], "passed": ["case-a"], "state": "ok",
    }


def test_complete_observation_clears_only_its_prior_reason():
    ledger = RequiredTests()
    reasons = {"baseline": "baseline report was not produced",
               "current": "current report was not produced"}

    ledger, reason = runner._observe_required_test_run(
        ledger, "baseline", {"required_test_report": {
            "complete": True, "outcomes": {"case-a": "passed"},
        }}, reasons=reasons)

    assert reason is None
    assert reasons == {"current": "current report was not produced"}
    assert ledger.report()["state"] == "unknown"
    assert runner._required_test_ledger_summary(ledger, reasons) == {
        "missing": [], "failed": [], "passed": [], "state": "unknown",
        "reason": "current: current report was not produced",
    }

    ledger, reason = runner._observe_required_test_run(
        ledger, "current", {"required_test_report": {
            "complete": True, "outcomes": {"case-a": "passed"},
        }}, reasons=reasons)

    assert reason is None
    assert reasons == {}
    assert runner._required_test_ledger_summary(ledger, reasons) == {
        "missing": [], "failed": [], "passed": ["case-a"], "state": "ok",
    }


def test_rejected_observation_keeps_prior_evidence_and_does_not_poison_known_state():
    ledger = RequiredTests()
    reasons = {}
    for kind in ("baseline", "current"):
        ledger, reason = runner._observe_required_test_run(
            ledger, kind, {"required_test_report": {
                "complete": True, "outcomes": {"case-a": "passed"},
            }}, reasons=reasons)
        assert reason is None

    rejected, reason = runner._observe_required_test_run(
        ledger, "current", {"required_test_report": {
            "complete": True, "outcomes": {"case-a": "unknown"},
        }}, reasons=reasons)

    assert rejected is ledger
    assert "ValueError" in reason
    assert ledger.report()["state"] == "ok"
    assert runner._required_test_ledger_summary(ledger, reasons) == {
        "missing": [], "failed": [], "passed": ["case-a"], "state": "ok",
    }


def test_latest_complete_current_run_replaces_prior_current_run():
    ledger = RequiredTests()
    ledger.observe("baseline", complete=True, outcomes={
        "case-a": "passed", "case-b": "passed",
    })
    ledger.observe("current", complete=True, outcomes={
        "case-a": "passed", "case-b": "passed",
    })
    ledger.observe("current", complete=False, outcomes={"case-a": "failed"})
    ledger.observe("current", complete=True, outcomes={"case-a": "failed"})

    assert ledger.report() == {
        "missing": ["case-b"],
        "failed": ["case-a"],
        "passed": [],
        "state": "regressed",
    }


def test_missing_baseline_or_current_is_unknown():
    assert RequiredTests().report() == {
        "missing": [], "failed": [], "passed": [], "state": "unknown",
    }
    baseline_only = RequiredTests()
    baseline_only.observe("baseline", complete=True, outcomes={"case-a": "passed"})
    assert baseline_only.report()["state"] == "unknown"


def test_known_report_omits_stale_incomplete_reasons():
    ledger = RequiredTests()
    ledger.observe("baseline", complete=True, outcomes={"case-a": "passed"})
    ledger.observe("current", complete=True, outcomes={"case-a": "passed"})

    assert runner._required_test_ledger_summary(
        ledger, {"baseline": "stale baseline reason", "current": "stale current reason"}) == {
            "missing": [], "failed": [], "passed": ["case-a"], "state": "ok",
        }

    regressed = RequiredTests()
    regressed.observe("baseline", complete=True, outcomes={"case-a": "passed"})
    regressed.observe("current", complete=True, outcomes={"case-a": "failed"})
    assert runner._required_test_ledger_summary(
        regressed, {"baseline": "stale baseline reason", "current": "stale current reason"}) == {
            "missing": [], "failed": ["case-a"], "passed": [], "state": "regressed",
        }


def test_incomplete_replacement_does_not_hide_the_side_with_prior_evidence():
    ledger = RequiredTests()
    ledger.observe("baseline", complete=True, outcomes={"case-a": "passed"})
    reasons = {"baseline": "older baseline failure"}

    ledger, reason = runner._observe_required_test_run(
        ledger, "baseline", {"required_test_report": {
            "complete": False, "outcomes": {},
        }, "required_test_report_reason": "pytest report was not produced"},
        reasons=reasons)

    assert reason == "pytest report was not produced"
    assert reasons == {}
    assert runner._required_test_ledger_summary(ledger, reasons) == {
        "missing": [], "failed": [], "passed": [], "state": "unknown",
        "reason": "no complete baseline and current check reports",
    }


def test_skipped_baseline_pass_is_not_reported_as_passed():
    ledger = RequiredTests()
    ledger.observe("baseline", complete=True, outcomes={"case-a": "passed"})
    ledger.observe("current", complete=True, outcomes={"case-a": "skipped"})

    assert ledger.report() == {
        "missing": ["case-a"], "failed": [], "passed": [], "state": "regressed",
    }


def test_json_round_trip_is_stable_and_preserves_complete_observations():
    ledger = RequiredTests()
    ledger.observe("baseline", complete=True, outcomes={
        "case-b": "passed", "case-a": "passed",
    })
    ledger.observe("current", complete=True, outcomes={"case-a": "passed"})

    encoded = ledger.to_json()
    restored = RequiredTests.from_json(encoded)

    assert encoded == restored.to_json()
    assert json.loads(encoded)["baseline"] == ["case-a", "case-b"]
    assert restored.report() == ledger.report()


@pytest.mark.parametrize("kind", ["other", "BASELINE", ""])
def test_rejects_unknown_observation_kind(kind):
    with pytest.raises(ValueError):
        RequiredTests().observe(kind, complete=True, outcomes={})


def test_rejects_unknown_outcome_without_mutating_state():
    ledger = RequiredTests()
    ledger.observe("baseline", complete=True, outcomes={"case-a": "passed"})

    with pytest.raises(ValueError):
        ledger.observe("current", complete=True, outcomes={
            "case-a": "passed", "case-b": "maybe",
        })

    assert ledger.report()["state"] == "unknown"


def test_structured_report_preserves_node_ids_and_statuses():
    outcomes = {
        "tests/test_math.py::TestMath::test_add[small]": "passed",
        "tests/test_math.py::TestMath::test_subtract": "failed",
        "tests/test_math.py::test_optional": "skipped",
    }
    report = json.dumps({"exit_code": 1, "outcomes": outcomes}).encode()

    assert parse_pytest_report(report, exit_code=1) == {"complete": True, "outcomes": outcomes}


@pytest.mark.parametrize(("report", "exit_code"), [
    (b'{"exit_code": 1, "outcomes": {"test.py::test_ok": "passed"}}', 1),
    (b'{"exit_code": 0, "outcomes": {"test.py::test_error": "failed"}}', 0),
    (b'{"exit_code": 0, "outcomes": {}}', 0),
    (b'{"exit_code": 5, "outcomes": {"test.py::test_ok": "passed"}}', 5),
    (b'{"exit_code": 1, "outcomes": {"test.py::test_failure": "failed"}}', None),
    (b'{"exit_code": 0, "outcomes": {"test.py::test_skip": "unknown"}}', 0),
    (b'{"exit_code": 0, "outcomes": {"": "passed"}}', 0),
    (b'{"outcomes": {"test.py::test_ok": "passed"}}', 0),
    (b'{"exit_code": 0,', 0),
])
def test_incomplete_structured_report_proves_no_ids(report, exit_code):
    assert parse_pytest_report(report, exit_code) == {"complete": False, "outcomes": {}}


def test_complete_empty_report_matches_pytest_no_tests_exit():
    report = b'{"exit_code": 5, "outcomes": {}}'

    assert parse_pytest_report(report, exit_code=5) == {"complete": True, "outcomes": {}}


def test_incomplete_report_reason_must_be_a_known_code():
    from kernel_required_tests import parse_pytest_report_with_reason

    incomplete = b'{"incomplete_reason":"collection_only"}'
    rejected, reason = parse_pytest_report_with_reason(incomplete, exit_code=0)
    assert rejected == {"complete": False, "outcomes": {}}
    assert reason == "pytest was run in collection-only mode"

    free_text = b'{"incomplete_reason":"secret detail from pytest"}'
    rejected, reason = parse_pytest_report_with_reason(free_text, exit_code=0)
    assert rejected == {"complete": False, "outcomes": {}}
    assert reason == "pytest report was not produced"


def test_switch_defaults_off_and_accepts_only_exact_values(monkeypatch):
    monkeypatch.delenv("STATEFRAME_REQUIRED_TEST_LEDGER", raising=False)
    assert _required_test_ledger_enabled() is False

    monkeypatch.setenv("STATEFRAME_REQUIRED_TEST_LEDGER", "off")
    assert _required_test_ledger_enabled() is False
    monkeypatch.setenv("STATEFRAME_REQUIRED_TEST_LEDGER", "on")
    assert _required_test_ledger_enabled() is True

    for invalid in ("", "ON", " on ", "true"):
        monkeypatch.setenv("STATEFRAME_REQUIRED_TEST_LEDGER", invalid)
        with pytest.raises(RunError, match="must be exactly 'off' or 'on'"):
            _required_test_ledger_enabled()


def test_switch_is_host_only_and_not_forwarded_to_the_engine():
    name = "STATEFRAME_REQUIRED_TEST_LEDGER"

    assert name in runner.HOST_ONLY_SWITCHES
    assert name not in runner.OPT_IN_SWITCHES
    assert name not in runner.engine_switches()
