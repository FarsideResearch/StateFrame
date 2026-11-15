"""Boundary inputs: a real scripted engine session must survive every boundary-value file, and the check must bite."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from enginecheck import HarnessError, Scenario, Trace, run
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
# these are strict xfails. Reasons carry the engine exit code and its error line from the captured run (f2run ecitem3).
# src/ is unchanged and the check is not weakened.
_ENGINE_FINDINGS: dict[str, str] = {
    "binary_nul": "exit 2: engine could not seal request 1: serialized request failed wire string hygiene (HARNESS_CONTRACT)",
    "invalid_utf8": "exit 2: engine could not seal request 2: serialized request failed wire string hygiene (HARNESS_CONTRACT)",
    "long_line": "exit 2: engine_result ENGINE_ERROR, error RuntimeError (child exit 1)",
}


def _boundary_params() -> list[object]:
    return [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, raises=HarnessError, reason=_ENGINE_FINDINGS[name]))
        if name in _ENGINE_FINDINGS else name
        for name in sorted(BOUNDARY_CONTENTS)
    ]


@pytest.mark.parametrize("name", _boundary_params())
def test_engine_survives_boundary_content(name: str, tmp_path: Path) -> None:
    scenario = read_scenario(name, BOUNDARY_CONTENTS[name])

    trace = run(scenario, tmp_path)

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
    assert isinstance(info.value.output, str)


def test_zero_requests_are_reported(tmp_path: Path) -> None:
    scenario = read_scenario("control", b"a = 1\n")

    problems = check_survives(_synthetic(tmp_path, scenario, requests=0), scenario)

    assert any("no provider requests" in problem for problem in problems), problems
