"""Frozen pytest receipt and frame rendering from the original check path."""
from __future__ import annotations

import json
from pathlib import Path

from stateframe.test_runners import parse_command
from stateframe.engine.kernel_check_receipts import _summary
from stateframe.engine.kernel_frame_budget import truth_projection
from stateframe.engine.probes.react_repair import TOOLS, _done_tool_surface


FIXTURE = Path(__file__).parent / "fixtures" / "pytest_runner_baseline.json"
RECEIPT = {
    "execution_id": "baseline-check", "status": "PASS", "outcome": "PASS",
    "reached_observation": True, "failure_phase": "NONE",
    "selector": "tests/test_app.py", "selector_scope": ["tests/test_app.py"],
    "execution_complete": True, "termination_reason": "EXITED",
    "output_truncated": False, "capture_complete": True, "capture_status": "COMPLETE",
    "counts": {"collected": 1, "passed": 1}, "revision": "baseline-revision",
    "assertion_context_preview": "1 passed", "runner_tail": "1 passed",
    "stdout_bytes": 8, "stderr_bytes": 0, "assertion_context_omitted_bytes": 0,
    "stdout_omitted_bytes": 0, "stderr_omitted_bytes": 0,
    "wrapper_parse_error": "", "full_output_handle": "check-receipt:baseline-check",
}
FRAME = {"public_check_status": "PASS", "test_receipts": [RECEIPT],
         "current_revision": "baseline-revision"}


def test_pytest_test_cmd_preserves_receipt_and_frame_rendering():
    assert parse_command("python -m pytest -q tests/test_app.py")[0] == "pytest"
    evidence = {"receipt_text": _summary(RECEIPT), "frame": truth_projection(FRAME)}
    assert evidence == json.loads(FIXTURE.read_text())


def test_pytest_all_done_schema_matches_33b18a6_bytes():
    done = next(tool for tool in _done_tool_surface(TOOLS, {
        "runner_kind": "pytest", "protect_tests": "all",
        "public_check_paths": ["tests/test_app.py"]})
        if tool["function"]["name"] == "done")
    frozen = Path(__file__).parent / "fixtures/pytest_all_done_33b18a6.json"
    assert json.dumps(done, separators=(",", ":"), ensure_ascii=False) == frozen.read_text()


def test_default_off_keeps_done_schema_until_protection_is_explicit():
    row = {"runner_kind": "pytest", "protect_tests": "off"}
    assert _done_tool_surface(TOOLS, row) is TOOLS
    offered = _done_tool_surface(TOOLS, {**row, "test_wrong_wire": True})
    done = next(tool for tool in offered if tool["function"]["name"] == "done")
    assert "test_wrong" in done["function"]["parameters"]["properties"]
