"""Boundary inputs: a real scripted engine session must survive every boundary-value file, and the check must bite."""
from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from enginecheck import HarnessError, Scenario, Trace, _post_run_failure, run
from enginecheck.boundary import (
    BOUNDARY_CONTENTS,
    assert_survives,
    check_survives,
    limit_values,
    read_scenario,
)

TRACEBACK = "Traceback (most recent call last):\n  File \"engine.py\", line 1, in <module>\nValueError: boom\n"


@pytest.mark.parametrize(("limit", "expected"), [
    (0, (0, 1)),
    (1, (0, 1, 2)),
    (4096, (0, 1, 4095, 4096, 4097)),
])
def test_limit_values_are_sorted_unique_and_non_negative(limit: int, expected: tuple[int, ...]) -> None:
    assert limit_values(limit) == expected


# Real-engine findings, not defects in the check. Each run() raises HarnessError before a survival verdict exists, so
# these are strict xfails. Reasons carry the engine exit code and its error line from the captured run (captured run, 2026-11-15).
# src/ is unchanged and the check is not weakened.
# Each entry: (text the HarnessError must contain, xfail reason). A HarnessError without that text fails the test.
_ENGINE_FINDINGS: dict[str, tuple[str, str]] = {
    "binary_nul": (
        "wire string hygiene",
        "exit 2: engine could not seal request 1: serialized request failed wire string hygiene (HARNESS_CONTRACT)",
    ),
    "invalid_utf8": (
        "wire string hygiene",
        "exit 2: engine could not seal request 2: serialized request failed wire string hygiene (HARNESS_CONTRACT)",
    ),
}


def _boundary_params() -> list[object]:
    return [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, raises=HarnessError, reason="ENGINE-BOUNDARY-REDS: " + _ENGINE_FINDINGS[name][1]))
        if name in _ENGINE_FINDINGS else name
        for name in sorted(BOUNDARY_CONTENTS)
    ]


def _check_expected_finding(name: str, error: HarnessError) -> None:
    """Only the recorded engine finding may xfail. Any other HarnessError, or one missing the recorded text, fails."""
    expected_text = _ENGINE_FINDINGS.get(name, ("", ""))[0]
    assert expected_text and expected_text in str(error), f"unexpected HarnessError for {name!r}: {error}"


@pytest.mark.parametrize("name", _boundary_params())
def test_engine_survives_boundary_content(name: str, tmp_path: Path) -> None:
    scenario = read_scenario(name, BOUNDARY_CONTENTS[name])
    try:
        trace = run(scenario, tmp_path)
    except HarnessError as error:
        _check_expected_finding(name, error)
        raise

    assert_survives(trace, scenario)


def _synthetic(tmp_path: Path, scenario: Scenario, *, exit_code: int = 0, output: str = "",
               log: str | None = None, wire_log: str | None = None, requests: int = 1,
               edit: bytes | None = None) -> Trace:
    """Hand-built trace over a materialized repo. No engine runs; the run dir holds only what the case plants."""
    repo = tmp_path / "repo"
    for name, content in scenario.files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(content)
    if edit is not None:
        (repo / "target.py").write_bytes(edit)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if log is not None:
        (run_dir / "engine.log").write_text(log, encoding="utf-8")
    if wire_log is not None:
        (run_dir / "wire").mkdir()
        (run_dir / "wire" / "1.request.json").write_text(wire_log, encoding="utf-8")
    raw = json.dumps({"messages": []}).encode()
    return Trace(
        requests=[raw] * requests,
        request_json=[json.loads(raw)] * requests,
        events=[],
        exit_code=exit_code,
        switches={},
        run_dir=run_dir,
        repo=repo,
        output=output,
    )


def test_clean_synthetic_trace_passes(tmp_path: Path) -> None:
    scenario = read_scenario("control", b"a = 1\n")

    assert check_survives(_synthetic(tmp_path, scenario), scenario) == []


def test_changed_file_bytes_are_reported(tmp_path: Path) -> None:
    scenario = read_scenario("control", b"a = 1\n")

    problems = check_survives(_synthetic(tmp_path, scenario, edit=b"a = 2\n"), scenario)

    assert any("target.py" in problem and "changed" in problem for problem in problems), problems


def test_changed_output_file_bytes_are_reported(tmp_path: Path) -> None:
    scenario = read_scenario("control", b"a = 1\n")
    trace = _synthetic(tmp_path, scenario)
    (trace.repo / "notes.md").write_bytes(b"rewritten\n")

    problems = check_survives(trace, scenario)

    assert any("notes.md" in problem and "changed" in problem for problem in problems), problems


def test_crash_exit_code_with_traceback_is_reported(tmp_path: Path) -> None:
    scenario = read_scenario("control", b"a = 1\n")

    problems = check_survives(_synthetic(tmp_path, scenario, exit_code=139, output=TRACEBACK), scenario)

    assert any("crashed" in problem and "139" in problem for problem in problems), problems
    assert any("traceback" in problem for problem in problems), problems


def test_traceback_in_output_with_exit_one_is_reported(tmp_path: Path) -> None:
    """Negative control: an uncaught exception exits 1, which is a clean-looking code, so the output must catch it."""
    scenario = read_scenario("control", b"a = 1\n")

    problems = check_survives(_synthetic(tmp_path, scenario, exit_code=1, output=TRACEBACK), scenario)

    assert any("traceback" in problem for problem in problems), problems


def test_traceback_only_in_run_artifact_is_reported(tmp_path: Path) -> None:
    scenario = read_scenario("control", b"a = 1\n")

    problems = check_survives(_synthetic(tmp_path, scenario, log=TRACEBACK), scenario)

    assert any("traceback" in problem for problem in problems), problems


def test_traceback_quoted_in_wire_is_not_a_crash(tmp_path: Path) -> None:
    """Tool results sent to the model may quote tracebacks; wire/ is excluded from the artifact scan."""
    scenario = read_scenario("control", b"a = 1\n")

    problems = check_survives(_synthetic(tmp_path, scenario, wire_log=TRACEBACK), scenario)

    assert problems == []


def test_zero_request_harness_error_carries_exit_code_and_output(tmp_path: Path) -> None:
    """An engine that sends no provider request raises with its exit code in the message and as an attribute."""
    scenario = replace(read_scenario("no-task", b"a = 1\n"), task="")

    with pytest.raises(HarnessError) as info:
        run(scenario, tmp_path)

    assert info.value.exit_code is not None
    assert f"engine exit code: {info.value.exit_code}" in str(info.value)
    # The engine rejects the empty task itself and prints this line; it must reach the captured output.
    assert "stateframe: error: --task is empty" in info.value.output, info.value.output[-500:]


def test_zero_requests_are_reported(tmp_path: Path) -> None:
    scenario = read_scenario("control", b"a = 1\n")

    problems = check_survives(_synthetic(tmp_path, scenario, requests=0), scenario)

    assert any("no provider requests" in problem for problem in problems), problems


def _write_summary(run_dir: Path, summary: dict[str, object]) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.mark.parametrize("outcome", ["hook_blocked", "continuation_blocked", "suspended"])
def test_exit_one_with_named_refusal_outcome_is_clean(tmp_path: Path, outcome: str) -> None:
    scenario = read_scenario("control", b"a = 1\n")
    trace = _synthetic(tmp_path, scenario, exit_code=1)
    _write_summary(trace.run_dir, {"runtime_outcome": outcome})

    assert check_survives(trace, scenario) == []


def test_exit_one_with_summary_error_and_no_traceback_is_reported(tmp_path: Path) -> None:
    """Negative control: exit 1 is a crash path (outcomes.py:293) unless the summary names a refusal outcome."""
    scenario = read_scenario("control", b"a = 1\n")
    trace = _synthetic(tmp_path, scenario, exit_code=1)
    _write_summary(trace.run_dir, {"runtime_outcome": "hook_blocked", "error": "RuntimeError"})

    problems = check_survives(trace, scenario)

    assert any("exit code 1" in problem for problem in problems), problems


def test_exit_one_with_unnamed_outcome_is_reported(tmp_path: Path) -> None:
    scenario = read_scenario("control", b"a = 1\n")
    trace = _synthetic(tmp_path, scenario, exit_code=1)
    _write_summary(trace.run_dir, {"runtime_outcome": "completed"})

    problems = check_survives(trace, scenario)

    assert any("exit code 1" in problem for problem in problems), problems


def test_exit_one_without_summary_is_reported(tmp_path: Path) -> None:
    scenario = read_scenario("control", b"a = 1\n")

    problems = check_survives(_synthetic(tmp_path, scenario, exit_code=1), scenario)

    assert any("exit code 1" in problem for problem in problems), problems


def test_unexpected_harness_error_does_not_match_a_recorded_finding() -> None:
    with pytest.raises(AssertionError):
        _check_expected_finding("binary_nul", HarnessError("engine sent no provider request"))
    with pytest.raises(AssertionError):
        _check_expected_finding("long_line", HarnessError("engine could not seal request 1: wire string hygiene"))


def test_recorded_finding_text_is_accepted() -> None:
    _check_expected_finding("binary_nul", HarnessError("...engine could not seal request 1: wire string hygiene..."))


def test_harness_error_summary_error_is_truncated_to_tail_budget(tmp_path: Path) -> None:
    """A huge summary.json error must not blow up the HarnessError message; it is capped at the tail budget."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_summary(run_dir, {"error": "~" * 5000})
    completed = subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="")

    error = _post_run_failure("engine sent no provider request", run_dir, completed)

    assert str(error).count("~") <= 2000, len(str(error))
