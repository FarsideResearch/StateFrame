"""Contract runner tests: every effective HANDLERS entry x scenario, plus runner self-tests.

The synthetic entry is a tiny real guarded op: inner op inside a broad except, then record_fault. Faults are
injected by patching the INNER op only. Product HANDLERS is empty, so product contributes zero cases today.
"""
from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

from _fault_contract import (
    CASES, OBSERVED_AT, SCENARIOS, ContractCase, Outcome, Scenario, assert_outcome, build_params,
    isolated_reason_log, register_case, registered,
)
from kernel_fault_disposition import record_fault
from kernel_fault_registry import HANDLERS, HandlerSpec, lookup_handler, register_for_tests

SYNTH_HANDLER = "synthetic.contract_guarded_op"
NOCASE_HANDLER = "synthetic.contract_nocase_probe"
FALLBACK = -1
INPUT = 5
SYNTH_SPEC = HandlerSpec(
    handler_id=SYNTH_HANDLER, module=__name__, function="guarded_op", op="synthetic_guarded_op",
    taxonomy="D", tier="T3", requiredness="optional", expected=("ValueError",), dispositions=("boundary",),
    fallback_id=None, owner="crew-08", contract_tests=("tests/test_fault_contract.py::test_fault_contract",),
    policy_version=1,
)
NOCASE_SPEC = HandlerSpec(
    **{**SYNTH_SPEC.__dict__, "handler_id": NOCASE_HANDLER, "op": "synthetic_nocase"}
)


def _inner_op(value: int) -> int:
    return value * 2


def guarded_op(value: int, *, journal: object | None = None) -> int:
    try:
        return _inner_op(value)
    except Exception as error:
        record_fault(SYNTH_HANDLER, error, disposition="boundary", observed_at=OBSERVED_AT, journal=journal)
        return FALLBACK


_FAULTS_BY_SCENARIO: dict[Scenario, Callable[[], BaseException] | None] = {
    "no_fault": None,
    "expected_fault": lambda: ValueError("boom"),
    "unexpected_fault": lambda: KeyError("nope"),
    "repeated_fault": lambda: ValueError("again"),
    "empty_valueerror": lambda: ValueError(),
    "resume": lambda: ValueError("resumed"),
}


def _inject(monkeypatch: pytest.MonkeyPatch, scenario: Scenario) -> None:
    factory = _FAULTS_BY_SCENARIO[scenario]
    if factory is None:
        return

    def failing_inner(value: int) -> int:
        raise factory()

    monkeypatch.setattr(sys.modules[__name__], "_inner_op", failing_inner)


def _run(scenario: Scenario) -> tuple[object, ...]:
    journal = False if scenario == "resume" else None
    calls = 2 if scenario == "repeated_fault" else 1
    return tuple(guarded_op(INPUT, journal=journal) for _ in range(calls))


_OUTCOMES: dict[Scenario, Outcome] = {
    "no_fault": Outcome(results=(INPUT * 2,), rows=0, journal_events=0),
    "expected_fault": Outcome(results=(FALLBACK,), rows=1, expectation="expected",
                              cause_code="ValueError", count=1, journal_events=1),
    "unexpected_fault": Outcome(results=(FALLBACK,), rows=1, expectation="unexpected",
                                cause_code="KeyError", count=1, journal_events=1),
    "repeated_fault": Outcome(results=(FALLBACK, FALLBACK), rows=1, expectation="expected",
                              cause_code="ValueError", count=2, journal_events=1),
    "empty_valueerror": Outcome(results=(FALLBACK,), rows=1, expectation="expected",
                                cause_code="ValueError", detail="<empty message>", count=1,
                                journal_events=1),
    "resume": Outcome(results=(FALLBACK,), rows=1, expectation="expected",
                      cause_code="ValueError", count=1, journal_events=0),
}

register_case(ContractCase(handler_id=SYNTH_HANDLER, inject=_inject, run=_run, outcomes=_OUTCOMES))

EFFECTIVE_SPECS: list[HandlerSpec] = [*HANDLERS.values(), SYNTH_SPEC]


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    with isolated_reason_log(monkeypatch, tmp_path) as harness:
        yield harness


@pytest.mark.parametrize(("spec", "scenario"), build_params(EFFECTIVE_SPECS, CASES))
def test_fault_contract(spec, scenario, isolated, monkeypatch):
    case = CASES.get(spec.handler_id)
    if case is None:
        raise AssertionError("contract missing")
    case.inject(monkeypatch, scenario)
    with registered(spec):
        results = case.run(scenario)
    assert_outcome(spec, results, isolated, case.outcomes[scenario])


# ---- runner self-tests ---------------------------------------------------------------------------

def test_product_handlers_yield_zero_parametrized_cases():
    assert HANDLERS == {}
    assert build_params(HANDLERS.values(), CASES) == []


def test_missing_case_is_reported_strict_xfail_with_temporary_registration():
    with register_for_tests(NOCASE_SPEC):
        params = build_params([lookup_handler(NOCASE_HANDLER)], CASES)
    assert len(params) == len(SCENARIOS)
    for param in params:
        xfail = [m for m in param.marks if m.name == "xfail"]
        assert len(xfail) == 1
        assert xfail[0].kwargs == {"strict": True, "reason": "contract missing"}


def test_covered_case_carries_no_xfail_mark():
    params = build_params([SYNTH_SPEC], CASES)
    assert all(not param.marks for param in params)


def test_contract_case_requires_every_scenario():
    partial = {s: _OUTCOMES[s] for s in SCENARIOS if s != "resume"}
    with pytest.raises(ValueError, match="missing"):
        ContractCase(handler_id="synthetic.partial", inject=_inject, run=_run, outcomes=partial)
