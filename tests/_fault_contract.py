"""Contract runner for registered guard handlers (V-FAILOPEN step 2).

Every HandlerSpec needs a ContractCase: an inject hook that monkeypatches the SMALLEST inner operation the
handler guards (never the fallback or the record path), a run hook that drives the real guarded path and
returns a RunResult, and the expected Outcome for each scenario. tests/test_fault_contract.py parametrizes over
every effective handler x scenario inside pytest_generate_tests (collection time). An entry without a case is
reported xfail(strict=True, reason="contract missing").

This module is the single copy of the reason-log isolation helpers (fresh problems file, _Journal, reset).
"""
from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

import kernel_reason_log as rl
from kernel_fault_registry import HANDLERS, HandlerSpec, register_for_tests

Scenario = Literal[
    "no_fault", "expected_fault", "unexpected_fault", "repeated_fault", "empty_valueerror", "journal_suppressed",
]
SCENARIOS: tuple[str, ...] = (
    "no_fault", "expected_fault", "unexpected_fault", "repeated_fault", "empty_valueerror", "journal_suppressed",
)
OBSERVED_AT = "2026-11-15T00:00:00Z"
CONTRACT_MISSING = "contract missing"


@dataclass(frozen=True)
class RunResult:
    """What one scenario run returned. results: guarded op return per call. records: record_fault return per fault."""

    results: tuple[object, ...]
    records: tuple[bool, ...]


@dataclass(frozen=True)
class Outcome:
    """Expected result of one scenario. rows/count/journal_events describe the reason log after run().

    A scenario that writes a row must also assert records (every record_fault return is True) and detail
    (the row's why text). ContractCase enforces both at definition time.
    """

    results: tuple[object, ...]
    rows: int
    records: tuple[bool, ...] = ()
    expectation: str | None = None
    cause_code: str | None = None
    detail: str | None = None
    count: int | None = None
    journal_events: int = 0


@dataclass(frozen=True)
class ContractCase:
    handler_id: str
    inject: Callable[[pytest.MonkeyPatch, Scenario], None]
    run: Callable[[Scenario], RunResult]
    outcomes: Mapping[Scenario, Outcome]

    def __post_init__(self) -> None:
        missing = [s for s in SCENARIOS if s not in self.outcomes]
        extra = [s for s in self.outcomes if s not in SCENARIOS]
        if missing or extra:
            raise ValueError(f"{self.handler_id}: outcomes missing={missing} unknown={extra}")
        for scenario, outcome in self.outcomes.items():
            if outcome.rows == 0:
                continue
            if not outcome.records or not all(outcome.records):
                raise ValueError(f"{self.handler_id}/{scenario}: a row is expected, so record_fault must return True")
            if outcome.detail is None:
                raise ValueError(f"{self.handler_id}/{scenario}: a row is expected, so detail (why) must be asserted")


# Module-level case table. Test modules register their synthetic cases here; product cases land with batch 1.
CASES: dict[str, ContractCase] = {}


def register_case(case: ContractCase) -> ContractCase:
    if case.handler_id in CASES:
        raise ValueError(f"{case.handler_id}: contract case already registered")
    CASES[case.handler_id] = case
    return case


class _Journal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind: str, **fields: object) -> None:
        self.events.append((kind, fields))


@dataclass(frozen=True)
class ReasonHarness:
    journal: _Journal
    problems: Path

    def rows(self) -> list[dict]:
        if not self.problems.exists():
            return []
        return [json.loads(line) for line in self.problems.read_text().splitlines()]

    def journal_events(self) -> int:
        return len(self.journal.events)


@contextmanager
def isolated_reason_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[ReasonHarness]:
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    journal, problems = _Journal(), tmp_path / "problems.jsonl"
    rl.configure(level="WARN", journal=journal, problems_path=problems, stream=io.StringIO())
    try:
        yield ReasonHarness(journal=journal, problems=problems)
    finally:
        rl.reset_for_tests()


@contextmanager
def registered(spec: HandlerSpec) -> Iterator[HandlerSpec]:
    """Product entries are already in HANDLERS; everything else is registered for the test's duration."""
    if spec.handler_id in HANDLERS:
        yield spec
        return
    with register_for_tests(spec):
        yield spec


def build_params(specs: Iterable[HandlerSpec], cases: Mapping[str, ContractCase]) -> list[object]:
    """One pytest.param per spec x scenario. Specs without a case are strict xfail, so the debt stays visible."""
    params: list[object] = []
    for spec in specs:
        marks = () if spec.handler_id in cases else (
            pytest.mark.xfail(strict=True, reason=CONTRACT_MISSING),)
        for scenario in SCENARIOS:
            params.append(pytest.param(spec, scenario, id=f"{spec.handler_id}-{scenario}", marks=marks))
    return params


def parametrize_contract(metafunc: pytest.Metafunc, specs: Iterable[HandlerSpec],
                         cases: Mapping[str, ContractCase]) -> None:
    """Call from pytest_generate_tests so the spec list is read at collection time, not at import time."""
    metafunc.parametrize(("spec", "scenario"), build_params(specs, cases))


def run_contract(spec: HandlerSpec, scenario: Scenario, cases: Mapping[str, ContractCase],
                 harness: ReasonHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    case = cases.get(spec.handler_id)
    if case is None:
        raise AssertionError(CONTRACT_MISSING)
    case.inject(monkeypatch, scenario)
    with registered(spec):
        run = case.run(scenario)
    assert_outcome(spec, run, harness, case.outcomes[scenario])


def assert_outcome(spec: HandlerSpec, run: RunResult, harness: ReasonHarness, outcome: Outcome) -> None:
    assert tuple(run.results) == outcome.results, f"op results {run.results!r} != {outcome.results!r}"
    # Any emit_reason self-error means the record path failed silently (record_fault also returns False).
    self_errors = int(rl.run_summary().get("self_errors", 0))
    assert self_errors == 0, f"reason log recorded {self_errors} self-error(s) during {spec.handler_id}"
    assert run.records == outcome.records, f"record_fault returned {run.records!r}, expected {outcome.records!r}"
    rows = harness.rows()
    assert len(rows) == outcome.rows, rows
    assert harness.journal_events() == outcome.journal_events, \
        f"bound journal received {harness.journal_events()} events, expected {outcome.journal_events}"
    if outcome.rows == 0:
        return
    row = rows[0]
    fields = row["fields"]
    assert fields["handler_id"] == spec.handler_id
    assert fields["op"] == spec.op
    assert fields["policy_version"] == spec.policy_version
    assert fields["disposition"] in spec.dispositions
    assert fields["disposition_mismatch"] is False
    assert fields["observed_at"] == OBSERVED_AT
    assert fields["expectation"] == outcome.expectation
    assert fields["cause_code"] == outcome.cause_code
    assert row["cause_key"] == f"{spec.handler_id}:{fields['disposition']}:{outcome.cause_code}"
    assert row["why"] == outcome.detail, f"why {row['why']!r} != {outcome.detail!r}"
    summary = rl.run_summary()["by_cause"].get(f"{row['category']}:{row['cause_key']}")
    assert summary is not None, "cause missing from run summary"
    assert summary["count"] == outcome.count
