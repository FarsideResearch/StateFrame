"""Compose checks: fake-trace unit tests for the combination verdicts, and one real-engine allocator x stable-wire pair."""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest
from enginecheck import Need, Scenario, Trace
from enginecheck.compose import (
    ALLOCATOR,
    STABLE_WIRE,
    ComposeReport,
    ComposeRow,
    Feature,
    assert_composes,
    combo_cli_args,
    combo_switches,
    compose,
)
from stateframe import profiles

_PAIRS = ("00", "10", "01", "11")
_WA_GOOD = {"00": False, "10": True, "01": False, "11": True}
_WB_GOOD = {"00": False, "10": False, "01": True, "11": True}


def _event(kind: str) -> dict[str, object]:
    return {"kind": kind}


def _alpha(trace: Trace) -> bool:
    return any(event.get("kind") == "alpha_event" for event in trace.events)


def _beta(trace: Trace) -> bool:
    return any(event.get("kind") == "beta_event" for event in trace.events)


FEAT_A = Feature(name="alpha", env={"X_ALPHA": "on"}, cli_args=("--alpha",), witness=_alpha)
FEAT_B = Feature(name="beta", env={"X_BETA": "on"}, witness=_beta)


def _trace(tmp_path: Path, *, wa: bool, wb: bool) -> Trace:
    events = [_event("alpha_event")] if wa else []
    events += [_event("beta_event")] if wb else []
    return Trace(requests=[b"{}"], request_json=[{}], events=events, exit_code=0, switches={},
                 run_dir=tmp_path, repo=tmp_path)


def _report(tmp_path: Path, *, wa: dict[str, bool] | None = None, wb: dict[str, bool] | None = None,
            survive: dict[str, tuple[str, ...]] | None = None, full: bool = False,
            errors: dict[str, str] | None = None) -> ComposeReport:
    wa_values = {**_WA_GOOD, **(wa or {})}
    wb_values = {**_WB_GOOD, **(wb or {})}
    problems = survive or {}
    refused = errors or {}
    rows: dict[str, ComposeRow] = {}
    combos = (*_PAIRS, "full") if full else _PAIRS
    for combo in combos:
        if combo in refused:
            rows[combo] = ComposeRow(combo=combo, trace=None, survive=(), witness_a=False, witness_b=False,
                                     error=refused[combo])
            continue
        if combo == "full":
            wa_value, wb_value = wa_values.get("full", True), wb_values.get("full", True)
        else:
            wa_value, wb_value = wa_values[combo], wb_values[combo]
        trace = _trace(tmp_path, wa=wa_value, wb=wb_value)
        rows[combo] = ComposeRow(combo=combo, trace=trace, survive=problems.get(combo, ()),
                                 witness_a=FEAT_A.witness(trace), witness_b=FEAT_B.witness(trace))
    return ComposeReport(a=FEAT_A, b=FEAT_B, rows=rows)


def test_all_good_report_passes(tmp_path: Path) -> None:
    assert_composes(_report(tmp_path))


def test_full_run_with_both_witnesses_passes(tmp_path: Path) -> None:
    assert_composes(_report(tmp_path, full=True))


def test_full_run_expectation_follows_the_profile_values(tmp_path: Path) -> None:
    # A profile that sets only alpha on: the full run must show alpha acting and beta absent.
    report = replace(_report(tmp_path, full=True, wb={"full": False}), full_expect=(True, False))
    assert_composes(report)
    with pytest.raises(AssertionError, match="full: beta witness is True, expected False"):
        assert_composes(replace(_report(tmp_path, full=True), full_expect=(True, False)))


def test_full_run_missing_a_witness_fails_naming_full(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="full: alpha witness"):
        assert_composes(_report(tmp_path, full=True, wa={"full": False}))


def test_combination_that_loses_b_fails_naming_11(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="11: beta witness is False, expected True"):
        assert_composes(_report(tmp_path, wb={"11": False}))


def test_leaking_witness_in_00_fails_naming_00(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="00: alpha witness is True, expected False"):
        assert_composes(_report(tmp_path, wa={"00": True}))


def test_vacuous_witness_in_10_fails_naming_10(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="10: alpha witness is False, expected True"):
        assert_composes(_report(tmp_path, wa={"10": False}))


def test_survive_problem_only_in_11_fails_naming_11(tmp_path: Path) -> None:
    problem = "engine crashed: exit code 1"
    with pytest.raises(AssertionError, match="11: survive problems: engine crashed"):
        assert_composes(_report(tmp_path, survive={"11": (problem,)}))


def test_engine_error_in_one_pair_is_reported_and_other_pairs_still_judged(tmp_path: Path) -> None:
    refusal = "enginecheck.HarnessError: engine could not seal request 1: some_refusal"
    with pytest.raises(AssertionError) as info:
        assert_composes(_report(tmp_path, errors={"11": refusal}))
    message = str(info.value)
    assert f"11: engine error: {refusal}" in message
    # The refused pair's witnesses are not judged, and the other pairs were judged and passed.
    assert "11: alpha witness" not in message and "11: beta witness" not in message
    assert not any(line.startswith(("00:", "10:", "01:")) for line in message.splitlines()), message


def test_interaction_must_hold_in_11(tmp_path: Path) -> None:
    report = _report(tmp_path)
    assert_composes(report, interaction=lambda trace: True)
    with pytest.raises(AssertionError, match="11: interaction check failed"):
        assert_composes(report, interaction=lambda trace: False)


def test_failure_lists_every_run_row(tmp_path: Path) -> None:
    with pytest.raises(AssertionError) as info:
        assert_composes(_report(tmp_path, wb={"11": False}, survive={"10": ("boom",)}))
    message = str(info.value)
    for combo in _PAIRS:
        assert f"{combo}  " in message, message
    assert "survive=1" in message


def test_combo_switches_set_each_feature_on_or_explicit_off() -> None:
    scenario = Scenario(files={}, task="t", steps=[], switches={"S": "1"})
    base = {"STATEFRAME_WORKING_SET_ALLOCATOR": "on", "STATEFRAME_STABLE_WIRE": "on", "OTHER": "1"}

    # Explicit off overrides the profile's on, so a profile default cannot leak into an off run.
    off_off = combo_switches(scenario, ALLOCATOR, STABLE_WIRE, "00", base)
    assert off_off == {"S": "1", "OTHER": "1", "STATEFRAME_WORKING_SET_ALLOCATOR": "off",
                       "STATEFRAME_STABLE_WIRE": "off"}

    on_off = combo_switches(scenario, ALLOCATOR, STABLE_WIRE, "10", base)
    assert on_off == {"S": "1", "OTHER": "1", "STATEFRAME_WORKING_SET_ALLOCATOR": "on",
                      "STATEFRAME_STABLE_WIRE": "off"}

    off_on = combo_switches(scenario, ALLOCATOR, STABLE_WIRE, "01", None)
    assert off_on == {"S": "1", "STATEFRAME_WORKING_SET_ALLOCATOR": "off", "STATEFRAME_STABLE_WIRE": "on"}


def test_off_switches_are_explicit_off_values() -> None:
    assert ALLOCATOR.off_switches() == {"STATEFRAME_WORKING_SET_ALLOCATOR": "off"}
    assert STABLE_WIRE.off_switches() == {"STATEFRAME_STABLE_WIRE": "off"}


def test_combo_cli_args_follow_the_features_that_are_on() -> None:
    scenario = Scenario(files={}, task="t", steps=[], cli_args=("--base",))

    assert combo_cli_args(scenario, FEAT_A, FEAT_B, "00") == ("--base",)
    assert combo_cli_args(scenario, FEAT_A, FEAT_B, "10") == ("--base", "--alpha")
    assert combo_cli_args(scenario, FEAT_A, FEAT_B, "11") == ("--base", "--alpha")


def test_real_witnesses_read_their_journal_events() -> None:
    trace = Trace(requests=[], request_json=[], events=[_event("working_set_prepared")], exit_code=0,
                  switches={}, run_dir=Path("."), repo=Path("."))
    assert ALLOCATOR.witness(trace) is True
    assert STABLE_WIRE.witness(trace) is False


def _real_scenario() -> Scenario:
    return Scenario(
        files={"target.py": b"a = 1\n", "notes.md": b"# notes\n"},
        task="Read target.py, read notes.md, then finish.",
        steps=[
            {"tool": "read", "required": True, "arguments": {"path": "target.py"}},
            {"tool": "read", "required": True, "arguments": {"path": "notes.md"}},
            {"tool": "done", "required": True, "arguments": {"summary": "Read both files."}},
        ],
        needs=[Need("the engine sent at least one provider request", lambda trace: len(trace.requests) > 0)],
    )


@pytest.fixture(scope="module")
def real_report(tmp_path_factory: pytest.TempPathFactory) -> ComposeReport:
    """One real compose under the shipped profile, shared by the real-engine tests below."""
    return compose(_real_scenario(), ALLOCATOR, STABLE_WIRE, tmp_path_factory.mktemp("compose"),
                   base_env=profiles.load("r2hc3"))


def test_off_pairs_complete_with_absent_witnesses(real_report: ComposeReport) -> None:
    rows = real_report.rows
    for combo in ("00", "10", "01"):
        assert rows[combo].error is None, f"{combo} was refused: {rows[combo].error}"
        assert rows[combo].survive == (), f"{combo} survive problems: {rows[combo].survive}"
    # stable_wire_epoch absent in 00 and 10 (stable wire off); working_set_prepared absent in 00 and 01 (allocator off).
    assert rows["00"].witness_b is False and rows["10"].witness_b is False
    assert rows["00"].witness_a is False and rows["01"].witness_a is False


# Real-engine finding, not a harness defect (f2run ecitem5): with the shipped profile, pair 11 (allocator and stable
# wire on) is refused by the engine before any provider verdict exists. Pairs 00, 10 and 01 still run and pass.
# Strict xfail: only failures whose lines are all pair 11 or full, each carrying this exact refusal, satisfy it.
_ALLOCATOR_STABLE_WIRE_TEXT = ("working_set_allocator_incompatible: STATEFRAME_WORKING_SET_ALLOCATOR=on "
                               "cannot preserve the stable-wire prefix")
_VERDICT_LINE = re.compile(r"^(00|10|01|11|full): ")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "CACHE-ALLOC-CONFLICT: exit 2: engine could not seal request 1: working_set_allocator_incompatible: STATEFRAME_WORKING_SET_ALLOCATOR=on "
    "cannot preserve the stable-wire prefix (HARNESS_CONTRACT)"))
def test_allocator_and_stable_wire_compose_under_shipped_profile(real_report: ComposeReport) -> None:
    try:
        assert_composes(real_report)
    except AssertionError as error:
        # pytest.fail (not assert) for these checks: an AssertionError would satisfy raises=AssertionError above.
        verdicts = [line for line in str(error).splitlines() if _VERDICT_LINE.match(line)]
        if not verdicts:
            pytest.fail(f"no verdict lines in failure: {error}")
        for line in verdicts:
            if not (line.startswith(("11: ", "full: ")) and _ALLOCATOR_STABLE_WIRE_TEXT in line):
                pytest.fail(f"unexpected failure line: {line}")
        raise
