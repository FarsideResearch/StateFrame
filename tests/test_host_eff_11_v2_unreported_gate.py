"""HEB item 11: on the opt-in v2 planner, a gate test pytest never reported is UNKNOWN, never passed.

A red -x run prints FAILED lines only, so every other gate test in that run has no verdict. Such a
test must be absent from the results map so unit_gate counts it as failing and the pointer holds.
"""
from __future__ import annotations

import pytest

import chains_recall
import test_unit_planner_v2 as planner
import unit_planner_v2 as v2

RED_OUTPUT = "FAILED tests/test_p.py::test_consume - assert 1"


@pytest.fixture
def recall_on(monkeypatch):
    monkeypatch.setenv(chains_recall.MODE_ENV, "on")


def test_unreported_gate_test_has_no_verdict_on_red_run():
    plan, _ = planner.plan_for()
    # Only the test pytest named is evidence; test_page was never reported so it is absent, not True.
    assert v2.results_from_run(plan, 1, RED_OUTPUT) == {"test_consume": False}


def test_green_run_keeps_every_gate_test_passed():
    plan, _ = planner.plan_for()
    names = {t for u in plan.units for t in u.gate_tests}
    assert v2.results_from_run(plan, 0, "") == {t: True for t in names}


def test_unreported_gate_test_fails_the_unit_gate(recall_on):
    plan, _ = planner.plan_for()
    results = v2.results_from_run(plan, 1, RED_OUTPUT)
    verdict = planner.gate(planner.PKG, unit="unit:fetch_all:g0", results=results)
    assert not verdict.passed
    assert "test_fetch_all" in verdict.failing_tests


def test_red_run_never_advances_pointer_past_unreported_gate(tmp_path, monkeypatch, recall_on):
    monkeypatch.setenv("STATEFRAME_UNIT_COALESCE", "off")
    monkeypatch.setattr(v2, "_index_class", lambda: planner.RefIndex)
    monkeypatch.setenv(v2.MODE_ENV, "on")
    for rel, text in planner.PKG.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    plan = v2.derive_with_index(planner.row(), tmp_path)
    assert plan is not None
    first = plan.units[0]
    # The test named red belongs to a later unit, so the first unit's gate tests were never reported.
    assert "test_consume" not in first.gate_tests and first.gate_tests
    bolt = planner._Bolt(plan, 1)
    bolt.unit_index = 0
    assert not v2.gate_and_advance(bolt, tmp_path, RED_OUTPUT).passed
    assert bolt.unit_index == 0
