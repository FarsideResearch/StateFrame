"""Contract runner tests: every effective HANDLERS entry x scenario, plus runner self-tests.

The synthetic entry is a tiny real guarded op: inner op inside a broad except, then record_fault. Faults are
injected by patching the INNER op only. Parametrization runs in pytest_generate_tests, so it reads HANDLERS as it
stands at collection time. Product HANDLERS is empty today, so product contributes zero cases.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from _fault_contract import (
    CASES, CONTRACT_MISSING, OBSERVED_AT, SCENARIOS, ContractCase, Outcome, RunResult, Scenario, build_params,
    isolated_reason_log, parametrize_contract, register_case, run_contract,
)
from kernel_fault_disposition import record_fault
from kernel_fault_registry import HANDLERS, HandlerSpec, lookup_handler, register_for_tests

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
ENGINE_DIR = ROOT / "src" / "stateframe" / "engine"

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

# Record return values of record_fault calls made by guarded_op during the current run.
_RECORD_RESULTS: list[bool] = []


def _inner_op(value: int) -> int:
    return value * 2


def guarded_op(value: int, *, journal: object | None = None) -> int:
    try:
        return _inner_op(value)
    except Exception as error:
        _RECORD_RESULTS.append(
            record_fault(SYNTH_HANDLER, error,
                         disposition="boundary", observed_at=OBSERVED_AT, journal=journal))
        return FALLBACK


_FAULTS_BY_SCENARIO: dict[Scenario, Callable[[], BaseException] | None] = {
    "no_fault": None,
    "expected_fault": lambda: ValueError("boom"),
    "unexpected_fault": lambda: KeyError("nope"),
    "repeated_fault": lambda: ValueError("again"),
    "empty_valueerror": lambda: ValueError(),
    "journal_suppressed": lambda: ValueError("suppressed"),
}


def _inject(monkeypatch: pytest.MonkeyPatch, scenario: Scenario) -> None:
    factory = _FAULTS_BY_SCENARIO[scenario]
    if factory is None:
        return

    def failing_inner(value: int) -> int:
        raise factory()

    monkeypatch.setattr(sys.modules[__name__], "_inner_op", failing_inner)


def _run(scenario: Scenario) -> RunResult:
    # journal_suppressed passes the request-path sentinel (journal=False): the bound journal must see no event.
    journal = False if scenario == "journal_suppressed" else None
    calls = 2 if scenario == "repeated_fault" else 1
    _RECORD_RESULTS.clear()
    results = tuple(guarded_op(INPUT, journal=journal) for _ in range(calls))
    return RunResult(results=results, records=tuple(_RECORD_RESULTS))


_OUTCOMES: dict[Scenario, Outcome] = {
    "no_fault": Outcome(results=(INPUT * 2,), rows=0, journal_events=0),
    "expected_fault": Outcome(results=(FALLBACK,), rows=1, records=(True,), expectation="expected",
                              cause_code="ValueError", detail="boom", count=1, journal_events=1),
    "unexpected_fault": Outcome(results=(FALLBACK,), rows=1, records=(True,), expectation="unexpected",
                                cause_code="KeyError", detail="'nope'", count=1, journal_events=1),
    "repeated_fault": Outcome(results=(FALLBACK, FALLBACK), rows=1, records=(True, True), expectation="expected",
                              cause_code="ValueError", detail="again", count=2, journal_events=1),
    "empty_valueerror": Outcome(results=(FALLBACK,), rows=1, records=(True,), expectation="expected",
                                cause_code="ValueError", detail="<empty message>", count=1, journal_events=1),
    "journal_suppressed": Outcome(results=(FALLBACK,), rows=1, records=(True,), expectation="expected",
                                  cause_code="ValueError", detail="suppressed", count=1, journal_events=0),
}

register_case(ContractCase(handler_id=SYNTH_HANDLER, inject=_inject, run=_run, outcomes=_OUTCOMES))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "spec" in metafunc.fixturenames:
        # Effective registry at collection time: product HANDLERS plus the synthetic entry.
        parametrize_contract(metafunc, [*HANDLERS.values(), SYNTH_SPEC], CASES)


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    with isolated_reason_log(monkeypatch, tmp_path) as harness:
        yield harness


def test_fault_contract(spec, scenario, isolated, monkeypatch):
    run_contract(spec, scenario, CASES, isolated, monkeypatch)


# ---- runner self-tests ---------------------------------------------------------------------------

def _xfail_marks(param: object) -> list:
    return [m for m in param.marks if m.name == "xfail"]  # type: ignore[attr-defined]


def test_every_product_handler_has_case_or_strict_xfail(monkeypatch):
    # HANDLERS is empty in product today. Add one case-less and one covered entry for this test only, so the
    # check is not vacuous: neither may be dropped from the parametrization.
    monkeypatch.setitem(HANDLERS, NOCASE_HANDLER, NOCASE_SPEC)
    monkeypatch.setitem(HANDLERS, SYNTH_HANDLER, SYNTH_SPEC)
    params = build_params(HANDLERS.values(), CASES)
    assert {param.values[0].handler_id for param in params} == set(HANDLERS)  # type: ignore[attr-defined]
    assert len(params) == len(HANDLERS) * len(SCENARIOS)
    for param in params:
        spec = param.values[0]  # type: ignore[attr-defined]
        xfail = _xfail_marks(param)
        if spec.handler_id in CASES:
            assert not param.marks, f"{spec.handler_id} has a case but is marked"  # type: ignore[attr-defined]
        else:
            assert len(xfail) == 1, f"{spec.handler_id} has no case and no xfail"
            assert xfail[0].kwargs == {"strict": True, "reason": CONTRACT_MISSING}


def test_missing_case_is_reported_strict_xfail_with_temporary_registration():
    with register_for_tests(NOCASE_SPEC):
        params = build_params([lookup_handler(NOCASE_HANDLER)], CASES)
    assert len(params) == len(SCENARIOS)
    for param in params:
        xfail = _xfail_marks(param)
        assert len(xfail) == 1
        assert xfail[0].kwargs == {"strict": True, "reason": CONTRACT_MISSING}


def test_covered_case_carries_no_xfail_mark():
    params = build_params([SYNTH_SPEC], CASES)
    assert all(not param.marks for param in params)


def test_contract_case_requires_every_scenario():
    partial = {s: _OUTCOMES[s] for s in SCENARIOS if s != "journal_suppressed"}
    with pytest.raises(ValueError, match="missing"):
        ContractCase(handler_id="synthetic.partial", inject=_inject, run=_run, outcomes=partial)


def test_contract_case_rejects_row_without_detail_or_true_record():
    with pytest.raises(ValueError, match="detail"):
        ContractCase(handler_id="synthetic.no_detail", inject=_inject, run=_run,
                     outcomes={**_OUTCOMES, "expected_fault": replace(_OUTCOMES["expected_fault"], detail=None)})
    with pytest.raises(ValueError, match="record_fault must return True"):
        ContractCase(handler_id="synthetic.false_record", inject=_inject, run=_run,
                     outcomes={**_OUTCOMES, "expected_fault": replace(_OUTCOMES["expected_fault"], records=(False,))})


# ---- pytester: the runner must honor the registry as collected -----------------------------------

_INNER_SOURCE = '''
import pytest
from _fault_contract import (
    ContractCase, Outcome, RunResult, SCENARIOS, isolated_reason_log, parametrize_contract, run_contract,
)
from kernel_fault_registry import HandlerSpec

SPEC = HandlerSpec(
    handler_id="synthetic.pytester_probe", module=__name__, function="probe", op="pytester_probe",
    taxonomy="D", tier="T3", requiredness="optional", expected=("ValueError",), dispositions=("boundary",),
    fallback_id=None, owner="crew-08", contract_tests=("tests/test_fault_contract.py::test_fault_contract",),
    policy_version=1,
)


def _inject(monkeypatch, scenario):
    return None


def _run(scenario):
    return RunResult(results=(), records=())


EMPTY = {s: Outcome(results=(), rows=0) for s in SCENARIOS}
__CASES__


def pytest_generate_tests(metafunc):
    if "spec" in metafunc.fixturenames:
        parametrize_contract(metafunc, [SPEC], CASES)


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    with isolated_reason_log(monkeypatch, tmp_path) as harness:
        yield harness


def test_probe(spec, scenario, isolated, monkeypatch):
    run_contract(spec, scenario, CASES, isolated, monkeypatch)
'''

_WITH_CASE = (
    "CASES = {SPEC.handler_id: ContractCase(handler_id=SPEC.handler_id, inject=_inject, run=_run, outcomes=EMPTY)}"
)
_WITHOUT_CASE = "CASES = {}"


def _run_inner(pytester, monkeypatch, cases_line: str):
    inherited = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(value for value in (str(TESTS_DIR), str(ENGINE_DIR), inherited) if value)
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    pytester.makepyfile(test_probe_inner=_INNER_SOURCE.replace("__CASES__", cases_line))
    return pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")


def test_pytester_handler_without_case_is_strict_xfail(pytester, monkeypatch):
    result = _run_inner(pytester, monkeypatch, _WITHOUT_CASE)
    result.assert_outcomes(xfailed=len(SCENARIOS))


def test_pytester_handler_with_case_passes(pytester, monkeypatch):
    result = _run_inner(pytester, monkeypatch, _WITH_CASE)
    result.assert_outcomes(passed=len(SCENARIOS))
