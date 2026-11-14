"""A size or resource refusal is visible beyond the parent tool observation."""
from __future__ import annotations

import json
from types import SimpleNamespace

from stateframe import cli, runner
from stateframe.engine.kernel_parallel import _emit_late_parallel_degrade, _emit_parallel_degrade
from stateframe.engine.kernel_parallel_records import (
    AdmissionResult, ParallelLimits, ParallelMeasured, ParallelResult, Refusal,
    UnitOutcome, UnitUsage,
)


def _refused(reason: str, detail: str) -> AdmissionResult:
    return AdmissionResult(
        "REFUSED", Refusal(reason, detail, ("pkg/a.py",), (), ()), (), (), (),
        0, "manifest", ParallelLimits(unit_request_tokens=12_000),
    )


def test_refused_subframes_emit_measured_degrade_event() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    runtime = SimpleNamespace(journal=SimpleNamespace(
        emit=lambda kind, **fields: events.append((kind, fields))))
    admission = _refused("UNIT_CONTEXT_TOO_LARGE", "first_frame_tokens=12500; cap=12000")

    _emit_parallel_degrade(runtime, admission, ("a", "b"))

    assert events == [("parallel_degraded", {
        "reason": "UNIT_CONTEXT_TOO_LARGE",
        "detail": "first_frame_tokens=12500; cap=12000",
        "subframes": ["a", "b"],
    })]


def test_refused_subframes_have_summary_field_and_printed_line(tmp_path, capsys) -> None:
    run = tmp_path / "run"
    call = run / "parallel" / "call"
    call.mkdir(parents=True)
    admission = _refused("UNIT_CONTEXT_TOO_LARGE", "first_frame_tokens=12500; cap=12000")
    result = ParallelResult(
        "call", admission, (), (), "NOT_RUN", (), "NOT_RUN", "refused",
        ParallelMeasured(0.0, 0.0, 0.0, 0, 0.0),
    )
    (call / "result.json").write_text(json.dumps(result.to_json()))

    parallel = runner._parallel_summary(run)
    assert parallel["degraded"] == [{
        "call_id": "call", "reason": "SUBFRAME_CONTEXT_TOO_LARGE",
        "detail": "first_frame_tokens=12500; cap=12000", "launched_subframes": 0,
    }]
    (run / "final.diff").write_text("")
    summary = {
        "run_dir": str(run), "tests_before": {"passed": None},
        "tests_after": {"command": "none"}, "runtime_outcome": "finished",
        "check_outcome": "not_run", "artifact": "none", "parallel": parallel,
    }
    cli._print_summary(summary)
    output = capsys.readouterr().out
    assert "subframe degrade" in output
    assert "SUBFRAME_CONTEXT_TOO_LARGE" in output
    assert "12500" in output and "12000" in output


def test_non_capacity_refusal_does_not_report_degrade() -> None:
    admission = _refused("OWNERSHIP_OVERLAP", "pkg/a.py belongs to two subframes")
    events: list[tuple[str, dict[str, object]]] = []
    runtime = SimpleNamespace(journal=SimpleNamespace(
        emit=lambda kind, **fields: events.append((kind, fields))))
    _emit_parallel_degrade(runtime, admission, ("a", "b"))
    assert events == []


def test_every_late_size_refusal_keeps_each_measured_frame(tmp_path) -> None:
    run = tmp_path / "run"
    call = run / "parallel" / "call"
    call.mkdir(parents=True)
    admission = AdmissionResult("ADMITTED", None, (), (), ("a", "b"),
                                0, "manifest", ParallelLimits())
    outcomes = tuple(UnitOutcome(
        unit_id, "REFUSED", "SUBFRAME_CONTEXT_TOO_LARGE",
        f"first_frame_tokens={tokens}; cap=12000", None, (), None, "NOT_RUN",
        UnitUsage(0, 0, 0, 0, tokens), 0.0, 0.0,
    ) for unit_id, tokens in (("a", 12_501), ("b", 13_002)))
    events: list[tuple[str, dict[str, object]]] = []
    runtime = SimpleNamespace(journal=SimpleNamespace(
        emit=lambda kind, **fields: events.append((kind, fields))))
    _emit_late_parallel_degrade(runtime, outcomes, 2)
    assert events == [("parallel_degraded", {
        "reason": "SUBFRAME_CONTEXT_TOO_LARGE",
        "detail": ("a: first_frame_tokens=12501; cap=12000; "
                   "b: first_frame_tokens=13002; cap=12000"),
        "subframes": ["a", "b"],
    })]
    result = ParallelResult("call", admission, outcomes, (), "NOT_RUN", (),
                            "FAILED", "refused", ParallelMeasured(0.0, 0.0, 0.0, 0, 0.0))
    (call / "result.json").write_text(json.dumps(result.to_json()))

    parallel = runner._parallel_summary(run)
    assert parallel["degraded"] == [{
        "call_id": "call", "reason": "SUBFRAME_CONTEXT_TOO_LARGE",
        "detail": ("a: first_frame_tokens=12501; cap=12000; "
                   "b: first_frame_tokens=13002; cap=12000"),
        "launched_subframes": 2,
    }]
    assert parallel["totals"]["subframes_with_initial_frame_tokens"] == 2
