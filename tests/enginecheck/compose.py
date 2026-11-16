"""Compose two engine features: run one scenario under each on/off pair and check that each feature acted.

Each feature is a switch set plus CLI args and a witness that proves the feature acted in that run. The four pairs
(00, 10, 01, 11) run in separate directories; a shipped-profile run (full) is added when base_env is given. The
combination check fails when a run breaks, when a witness is absent where its feature is on (vacuous or killed by the
other feature), or when a witness is present where its feature is off (leak). An engine refusal is recorded as that
pair's error and the remaining pairs still run.

Scope: this composes switch-driven features only. Subframes composition needs the parent/subframe scripted routing
from tests/fixtures/parallel/e2e_support.py; the enginecheck Script does not serve it, so it is not covered here.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from enginecheck import HarnessError, Scenario, Trace, run
from enginecheck.boundary import check_survives

PAIRS = ("00", "10", "01", "11")
FULL = "full"


@dataclass(frozen=True, kw_only=True)
class Feature:
    """A switch set, optional CLI args, and a witness over one run's trace. Off means each switch key set to off."""

    name: str
    env: Mapping[str, str]
    cli_args: tuple[str, ...] = ()
    witness: Callable[[Trace], bool]

    def off_switches(self) -> dict[str, str]:
        """Explicit off values. Removing a key is not enough: the default profile (r2hc3) sets these on, and an unset
        variable falls back to the profile (runner.py:1028 resolve_context, runner.py:1347 profile load)."""
        return {key: "off" for key in self.env}


@dataclass(frozen=True)
class ComposeRow:
    combo: str
    # None when the engine refused or broke this pair; error then carries the HarnessError text.
    trace: Trace | None
    survive: tuple[str, ...]
    witness_a: bool
    witness_b: bool
    error: str | None = None


@dataclass(frozen=True)
class ComposeReport:
    a: Feature
    b: Feature
    rows: dict[str, ComposeRow] = field(default_factory=dict)
    # Whether each feature should act in the shipped-profile (FULL) run, from the profile's own switch values.
    full_expect: tuple[bool, bool] = (True, True)


def _has_event(kind: str) -> Callable[[Trace], bool]:
    return lambda trace: any(event.get("kind") == kind for event in trace.events)


# Witnesses are journal events emitted only on the active path. Both are checked against off runs by assert_composes.
# Allocator: working_set_prepared, emitted by the live allocator path (kernel_working_set_live.py:307); the allocator
# switch is read at kernel_read_receipts.py:18-19, default off.
ALLOCATOR = Feature(name="allocator", env={"STATEFRAME_WORKING_SET_ALLOCATOR": "on"},
                    witness=_has_event("working_set_prepared"))
# Stable wire: stable_wire_epoch, emitted per stable-wire epoch (kernel_stable_wire.py:1265); mode read at
# kernel_stable_wire.py:57, default off. Requires STATEFRAME_FINAL_CACHE_LAYOUT=on (kernel_cache_layout_candidate.py:319).
STABLE_WIRE = Feature(name="stable_wire", env={"STATEFRAME_STABLE_WIRE": "on"},
                      witness=_has_event("stable_wire_epoch"))


def _check_features_distinct(a: Feature, b: Feature) -> None:
    """Two features must have distinct names and disjoint switch keys, or the pair cannot toggle them independently."""
    if a.name == b.name:
        raise ValueError(f"compose needs two distinct features, both named {a.name!r}")
    shared = sorted(set(a.env) & set(b.env))
    if shared:
        raise ValueError(f"features {a.name!r} and {b.name!r} share switch keys: {shared}")
    if not a.env or not b.env:
        raise ValueError("each compose feature needs at least one switch")


def combo_switches(scenario: Scenario, a: Feature, b: Feature, combo: str,
                   base_env: Mapping[str, str] | None) -> dict[str, str]:
    """Switches for one pair: base and scenario switches, with each feature set to its on or explicit off values."""
    env = {**(base_env or {}), **scenario.switches}
    for feature, bit in ((a, combo[0]), (b, combo[1])):
        env.update(feature.env if bit == "1" else feature.off_switches())
    return env


def combo_cli_args(scenario: Scenario, a: Feature, b: Feature, combo: str) -> tuple[str, ...]:
    """Scenario CLI args plus the CLI args of each feature that is on in this pair."""
    args = list(scenario.cli_args)
    if combo[0] == "1":
        args.extend(a.cli_args)
    if combo[1] == "1":
        args.extend(b.cli_args)
    return tuple(args)


def _run_row(scenario: Scenario, a: Feature, b: Feature, combo: str, directory: Path,
             switches: dict[str, str], cli_args: tuple[str, ...]) -> ComposeRow:
    """One pair's row. A HarnessError (engine refusal or crash) becomes the row's error, so other pairs still run."""
    try:
        trace = run(replace(scenario, switches=switches, cli_args=cli_args), directory)
    except HarnessError as error:
        return ComposeRow(combo=combo, trace=None, survive=(), witness_a=False, witness_b=False,
                          error=str(error))
    return ComposeRow(combo=combo, trace=trace, survive=tuple(check_survives(trace, scenario)),
                      witness_a=a.witness(trace), witness_b=b.witness(trace))


def compose(scenario: Scenario, a: Feature, b: Feature, tmp_path: Path, *,
            base_env: Mapping[str, str] | None = None) -> ComposeReport:
    """Run the scenario under each pair (and under base_env alone as FULL when given) and record survival and witnesses.

    Subframes composition needs the parent/subframe scripted routing from tests/fixtures/parallel/e2e_support.py;
    the enginecheck Script does not serve it, so it is not covered here.
    """
    _check_features_distinct(a, b)
    rows: dict[str, ComposeRow] = {}
    for combo in PAIRS:
        rows[combo] = _run_row(scenario, a, b, combo, tmp_path / f"pair-{combo}",
                               combo_switches(scenario, a, b, combo, base_env),
                               combo_cli_args(scenario, a, b, combo))
    full_expect = (False, False)
    if base_env is not None:
        # The shipped profile as given, no switch removed. Its expected feature state is what the profile sets.
        rows[FULL] = _run_row(scenario, a, b, FULL, tmp_path / f"pair-{FULL}",
                              {**base_env, **scenario.switches}, scenario.cli_args + a.cli_args + b.cli_args)
        full_expect = (_sets_on(a, base_env), _sets_on(b, base_env))
    return ComposeReport(a=a, b=b, rows=rows, full_expect=full_expect)


def _sets_on(feature: Feature, env: Mapping[str, str]) -> bool:
    """True when env holds every switch value the feature turns on."""
    return all(env.get(key) == value for key, value in feature.env.items())


# Which feature is expected to act in each run. Pair "10" means a on, b off. FULL comes from ComposeReport.full_expect.
_EXPECT_A = {"00": False, "10": True, "01": False, "11": True}
_EXPECT_B = {"00": False, "10": False, "01": True, "11": True}


def _witness_failures(combo: str, name: str, actual: bool, expected: bool) -> list[str]:
    if actual == expected:
        return []
    meaning = "vacuous: the feature did not act" if expected else "leaking: the feature acted while off"
    return [f"{combo}: {name} witness is {actual}, expected {expected} ({meaning})"]


def _table(report: ComposeReport) -> str:
    lines = [f"{'combo':<6} {'survive':<11} {report.a.name}  {report.b.name}"]
    for combo, row in report.rows.items():
        state = "engine-error" if row.error is not None else f"survive={len(row.survive)}"
        lines.append(f"{combo:<6} {state:<11} {report.a.name}={row.witness_a}  {report.b.name}={row.witness_b}")
    return "\n".join(lines)


def assert_composes(report: ComposeReport, *, interaction: Callable[[Trace], bool] | None = None) -> None:
    """Fail when a run breaks, a witness is vacuous or leaks, or the optional interaction check fails in pair 11.

    A pair that the engine refused is reported as its error and its witnesses are not judged.
    """
    failures: list[str] = []
    for combo, row in report.rows.items():
        if row.error is not None:
            # One line per verdict: the engine's refusal can span lines in the captured output tail.
            failures.append(f"{combo}: engine error: {' '.join(row.error.split())}")
            continue
        if row.survive:
            failures.append(f"{combo}: survive problems: {'; '.join(row.survive)}")
        expect_a, expect_b = (report.full_expect if combo == FULL else (_EXPECT_A[combo], _EXPECT_B[combo]))
        failures.extend(_witness_failures(combo, report.a.name, row.witness_a, expect_a))
        failures.extend(_witness_failures(combo, report.b.name, row.witness_b, expect_b))
    row_11 = report.rows.get("11")
    if interaction is not None and row_11 is not None and row_11.trace is not None \
            and not interaction(row_11.trace):
        failures.append("11: interaction check failed")
    if failures:
        raise AssertionError(_table(report) + "\n" + "\n".join(failures))
