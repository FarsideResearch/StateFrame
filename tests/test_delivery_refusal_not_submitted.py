"""A sealed candidate that could not be delivered must never end as SUBMITTED.

Covers the candidate apply helper (in-scope edits still land when out-of-scope baseline files
differ) and the post-summary refusal that rewrites engine_result when delivery failed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from stateframe import outcomes, runner, workcopy


@pytest.fixture()
def work(tmp_path: Path) -> workcopy.WorkCopy:
    original = tmp_path / "repo"
    (original / "data").mkdir(parents=True)
    (original / "a.py").write_bytes(b"A = 1\n")
    (original / "b.py").write_bytes(b"B = 1\n")
    (original / "data" / "x.dbf").write_bytes(b"\x03binary\x00")
    return workcopy.create(original, tmp_path / "copy")


def _collect() -> tuple[list[str], object]:
    lines: list[str] = []
    return lines, lines.append


def test_out_of_scope_drop_still_applies_in_scope_edits(work: workcopy.WorkCopy) -> None:
    candidate: dict[str, bytes | None] = {"a.py": b"A = 2\n", "data/x.dbf": None}
    lines, printer = _collect()

    written, dropped, error = runner._apply_sealed_candidate(
        work, candidate, {"a.py"}, printer)

    assert dropped == ["data/x.dbf"]
    assert error == "sealed candidate changed existing files outside edit scope"
    assert "a.py" in written
    assert (work.copy / "a.py").read_bytes() == b"A = 2\n"
    assert (work.copy / "data" / "x.dbf").read_bytes() == b"\x03binary\x00"
    assert any(line.startswith("warning: candidate changes outside edit scope were dropped")
               and "data/x.dbf" in line for line in lines)


def test_in_scope_only_candidate_applies_without_error(work: workcopy.WorkCopy) -> None:
    candidate: dict[str, bytes | None] = {"a.py": b"A = 3\n"}
    lines, printer = _collect()

    written, dropped, error = runner._apply_sealed_candidate(
        work, candidate, {"a.py"}, printer)

    assert dropped == []
    assert error is None
    assert written == ["a.py"]
    assert (work.copy / "a.py").read_bytes() == b"A = 3\n"
    assert lines == []


def _submitted_summary(*, dropped: list[str], error: str | None) -> dict[str, object]:
    return {
        "engine_result": "SUBMITTED",
        "engine_terminal": {"result": "SUBMITTED", "reason_code": "SUBMITTED_OK"},
        "candidate_captured": True,
        "candidate_error": error,
        "candidate_paths_dropped": dropped,
        "candidate_files_written": ["a.py"],
        "child_exit_code": 0,
        "candidate_sha256": None,
        "candidate_artifact_digest": None,
    }


def test_refusal_rewrites_submitted_when_out_of_scope_dropped() -> None:
    summary = _submitted_summary(
        dropped=["data/x.dbf"],
        error="sealed candidate changed existing files outside edit scope")
    terminal_before = dict(summary["engine_terminal"])

    runner._refuse_submitted_delivery(summary)

    assert summary["engine_result"] == runner.DELIVERY_REFUSED == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"] == {
        "engine_result": "SUBMITTED",
        "reason_code": "CANDIDATE_OUT_OF_SCOPE",
        "detail": "sealed candidate changed existing files outside edit scope",
    }
    assert summary["engine_terminal"] == terminal_before


def test_refusal_reason_code_when_not_delivered_without_drops() -> None:
    summary = _submitted_summary(dropped=[], error="disk full while applying")

    runner._refuse_submitted_delivery(summary)

    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_NOT_DELIVERED"
    assert summary["delivery_refusal"]["detail"] == "disk full while applying"


def test_no_refusal_without_candidate_error() -> None:
    summary = _submitted_summary(dropped=[], error=None)

    runner._refuse_submitted_delivery(summary)

    assert summary["engine_result"] == "SUBMITTED"
    assert "delivery_refusal" not in summary


@pytest.mark.parametrize("result", ["HOOK_BLOCKED", "CONTINUATION_BLOCKED", "ENGINE_ERROR"])
def test_no_refusal_for_non_submitted_results(result: str) -> None:
    summary = _submitted_summary(dropped=["data/x.dbf"], error="sealed candidate changed")
    summary["engine_result"] = result

    runner._refuse_submitted_delivery(summary)

    assert summary["engine_result"] == result
    assert "delivery_refusal" not in summary


def test_exit_code_stays_nonzero_after_refusal() -> None:
    summary = _submitted_summary(
        dropped=["data/x.dbf"],
        error="sealed candidate changed existing files outside edit scope")
    summary.update(outcomes.project(summary))
    summary["exit_code"] = outcomes.exit_code(summary)
    assert summary["exit_code"] == 2

    runner._refuse_submitted_delivery(summary)

    assert summary["exit_code"] == 2
    assert summary["engine_result"] == "DELIVERY_REFUSED"
