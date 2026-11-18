"""A sealed candidate that could not be delivered must never end as SUBMITTED.

Covers the candidate apply helper (in-scope edits still land when out-of-scope baseline files
differ) and the post-summary refusal that rewrites engine_result when delivery failed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from stateframe import outcomes, runner, workcopy
from stateframe.endpoint import Endpoint


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


def _prepare_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate: dict[str, bytes | None],
) -> tuple[runner.RunOptions, Path, workcopy.WorkCopy, dict[str, object], Path]:
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "outside.py").write_text("outside = 1\n", encoding="utf-8")
    options = runner.RunOptions(
        repo=repo, task="update sample", in_place=True, run_dir=tmp_path / "run",
        endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "test", None),
        max_turns=1, test_cmd="true", runner_kind="generic", check_source="command",
        target="sample.py", edit_paths=["sample.py"], no_project_hooks=True,
    )
    run, work, spec = runner.prepare(options, printer=lambda *_args: None)
    digest = "sha256:" + "a" * 64
    terminal = {"result": "SUBMITTED", "submission": {
        "stored": True, "seal_id": digest, "artifact_digest": digest,
    }}

    def drive(*_args):
        (run / "engine-result.json").write_text(json.dumps({"result": terminal}))
        return 0

    monkeypatch.setattr(runner, "drive", drive)
    monkeypatch.setattr(runner, "summarize", lambda *_args: {
        "engine_terminal": terminal, "engine_result": "SUBMITTED", "cached_share_display": "cache",
    })
    monkeypatch.setattr(runner, "final_candidate", lambda _run: candidate)
    monkeypatch.setattr(runner, "_artifact_digest", lambda _candidate: digest)
    monkeypatch.setattr(runner, "_tests_python", lambda _options: sys.executable)
    monkeypatch.setattr(runner, "_outer_test_python", lambda _options: sys.executable)
    monkeypatch.setattr(runner, "run_test_cmd", lambda *_args, **_kwargs: {
        "command": "true", "exit_code": 0, "passed": True, "tail": "ok",
    })
    monkeypatch.setattr(runner, "finish_run_storage", lambda *_args, **_kwargs: {
        "checkpoints": None, "env": None,
    })
    return options, run, work, spec, repo


def _execute_prepared(
    options: runner.RunOptions,
    run: Path,
    work: workcopy.WorkCopy,
    spec: dict[str, object],
) -> dict[str, object]:
    return runner._execute(options, 0.0, lambda *_args: None, run, work, spec)


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


def test_apply_failure_preserves_out_of_scope_evidence(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = {"sample.py": b"value = 2\n", "outside.py": b"outside = 2\n"}
    options, run, work, spec, _repo = _prepare_execution(tmp_path, monkeypatch, candidate)

    def fail_apply(*_args, **_kwargs):
        raise OSError("disk full while applying")

    monkeypatch.setattr(workcopy, "apply_files", fail_apply)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["candidate_paths_dropped"] == ["outside.py"]
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_OUT_OF_SCOPE"
    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["check_outcome"] == "not_run"
    assert summary["exit_code"] == 2


def test_stale_host_check_has_distinct_refusal_and_skips_in_place(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = {"sample.py": b"value = 2\n"}
    options, run, work, spec, repo = _prepare_execution(tmp_path, monkeypatch, candidate)
    check_calls = 0

    def mutate_during_check(*args, **_kwargs):
        nonlocal check_calls
        check_calls += 1
        if check_calls == 2:
            (Path(args[1]) / "sample.py").write_bytes(b"value = 3\n")
        return {"command": "true", "exit_code": 0, "passed": True, "tail": "ok"}

    monkeypatch.setattr(runner, "run_test_cmd", mutate_during_check)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"]["reason_code"] == "CHECKS_STALE"
    assert summary["check_outcome"] == "not_run"
    assert summary["exit_code"] == 2
    assert summary["applied_in_place"] == []
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
