from __future__ import annotations

import json
from pathlib import Path

import pytest

import kernel_fanout_continuation as continuation


def _unit(
    subframe_id: str,
    *,
    status: str,
    check_result: str,
    checks: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "id": subframe_id,
        "owns": [],
        "check": list(checks),
        "status": status,
        "check_result": check_result,
        "child_sends": 1,
        "changed_paths": [],
        "contract": True,
    }


def _tool_group(call_id: str, name: str, arguments: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"role": "assistant", "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": "complete"},
    ]


def _write_parent_record(run_dir: Path) -> str:
    call_id = "old-parent-run"
    call_root = run_dir / "parallel" / call_id
    (call_root / "checks" / "final").mkdir(parents=True)
    parent = run_dir / "parent"
    (parent / "shop").mkdir(parents=True)
    (parent / "tests").mkdir()
    (parent / "shop" / "shipping.py").write_text("def quote():\n    return 11\n", encoding="utf-8")
    (parent / "tests" / "test_shipping.py").write_text(
        "def test_quote():\n    assert quote() == 9\n", encoding="utf-8",
    )

    intent = {"request": {"units": [
        {"id": "pricing", "owns": ["shop/pricing.py"],
         "check": {"selectors": ["tests/test_pricing.py"]}},
        {"id": "shipping", "owns": ["shop/shipping.py"],
         "check": {"selectors": ["tests/test_shipping.py"]}},
    ]}}
    result = {
        "outcomes": [
            {"unit_id": "pricing", "status": "COMMITTED", "check": "PASS",
             "contract_ok": True, "files": [{"path": "shop/pricing.py"}],
             "usage": {"requests": 2}, "reason": None, "detail": None},
            {"unit_id": "shipping", "status": "ENGINE_ERROR", "check": "NOT_RUN",
             "contract_ok": None, "files": [], "usage": {"requests": 0},
             "reason": "ENGINE_ERROR", "detail": "worker exited"},
        ],
        "final_check": "FAIL",
        "integration": "FAILED",
    }
    (call_root / "intent.json").write_text(json.dumps(intent), encoding="utf-8")
    (call_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (run_dir / "tool-tape.jsonl").write_text(
        json.dumps({"kind": "complete", "call_id": call_id}) + "\n", encoding="utf-8",
    )
    return call_id


def test_old_parent_record_reads_and_resume_reuses_the_recorded_decisions(tmp_path: Path):
    run_dir = tmp_path / "recorded-run"
    call_id = _write_parent_record(run_dir)

    disposition, workspace = continuation.load_recorded_run(run_dir)

    assert disposition["call_id"] == call_id
    assert [(row["id"], row["status"], row["check_result"])
            for row in disposition["units"]] == [
        ("pricing", "COMMITTED", "PASS"),
        ("shipping", "ENGINE_ERROR", "NOT_RUN"),
    ]
    assert disposition["remaining"] == ["shipping"]
    before_resume, _pin, _events = continuation.render(disposition, workspace, 3072)
    assert "remaining=[shipping]" in before_resume
    assert "next_action: repair only shipping" in before_resume

    messages = _tool_group(call_id, "parallel", {})
    messages.extend(_tool_group("parent-check", "pytest", {"target": "tests/test_shipping.py"}))
    lifecycle = continuation._lifecycle_evidence(disposition, messages, {
        "journal_receipt_records": [{
            "kind": "check_receipt", "sequence": 1, "status": "PASS",
            "selector": "tests/test_shipping.py", "execution_complete": True,
        }],
    })
    resumed = {**disposition, "lifecycle": lifecycle}
    resumed_section, _pin, _events = continuation.render(resumed, workspace, 3072)

    assert lifecycle["passed_checks"] == {"tests/test_shipping.py": 2}
    assert "remaining=[]" in resumed_section
    assert "final_check=FAIL; final_integration=FAILED" in resumed_section
    assert "next_action: subframes finished but parent final check is FAIL, not verified" in resumed_section


def test_model_visible_continuation_terms_name_subframes():
    incomplete = {
        "units": [_unit("shipping", status="FAILED", check_result="FAIL")],
        "final_check": "FAIL",
        "integration": "FAILED",
        "remaining": ["shipping"],
    }

    section, _pin, _events = continuation.render(incomplete, None, 3072)

    assert "subframe=shipping" in section
    assert "subframe check failed" in section
    assert "don't reopen completed subframes" in section
    assert "the remaining subframe check" in section
    assert "unit=shipping" not in section
    assert "unit check failed" not in section
    assert "don't reopen completed units" not in section

    malformed = {"units": None}
    with pytest.raises(continuation.FanoutContinuationError, match="invalid subframe rows"):
        continuation.render(malformed, None, 3072)
