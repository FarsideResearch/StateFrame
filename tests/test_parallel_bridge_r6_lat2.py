"""A subframe block must not mask a stronger terminal failure."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_stateframe_runtime import StateFrameRuntime
from kernel_stop_reason_taxonomy import classify_run_end
from kernel_terminal_lattice import join_terminal, stop_reason_rank
from kernel_terminal_policy import authorize_subframe_block


_STRONGER_TERMINALS = (
    ("RUN_FAILED", "RUN_FAILED"),
    ("FAILED", "RUN_FAILED"),
    ("FAIL", "RUN_FAILED"),
    ("FAILURE", "RUN_FAILED"),
    ("ERROR", "RUN_FAILED"),
    ("RED", "RUN_FAILED"),
    ("AUTHORITY_FAILURE", "AUTHORITY_FAILURE"),
    ("HARDEN8_AUTHORITY_FAILURE", "AUTHORITY_FAILURE"),
    ("HARDEN8_CONTRACT_ERROR", "AUTHORITY_FAILURE"),
    ("HARDEN8_EVIDENCE_PIN_ERROR", "AUTHORITY_FAILURE"),
    ("HARDEN8_OWNER_FINALIZATION_ERROR", "AUTHORITY_FAILURE"),
    ("HARDEN8_ROUTE_ERROR", "AUTHORITY_FAILURE"),
    ("FINALIZATION_ERROR", "AUTHORITY_FAILURE"),
    ("HARNESS_ERROR", "HARNESS_ERROR"),
    ("ENVIRONMENT_BROKEN", "ENVIRONMENT_BROKEN"),
    ("STOPPED_INCOMPLETE", "STOPPED_INCOMPLETE"),
    ("ATTEMPT_BUDGET_EXHAUSTED", "ATTEMPT_BUDGET_EXHAUSTED"),
    ("CANCELLED", "CANCELLED"),
    ("HOOK_CONTAINMENT", "HOOK_CONTAINMENT"),
)

_INCOMPLETE_ALIASES = (
    "INCOMPLETE",
    "RUN_INCOMPLETE",
    "NOT_COMPLETED",
    "TIMEOUT",
    "TIMED_OUT",
    "TIMEOUT_EXHAUSTED",
    "STOPPED",
    "NEEDS_WORK",
)


def _runtime(tmp_path: Path) -> StateFrameRuntime:
    runtime = object.__new__(StateFrameRuntime)
    object.__setattr__(runtime, "stateframe_terminal_lattice_mode", "on")
    object.__setattr__(runtime, "stateframe_hard_budget_only_terminals_mode", "on")
    object.__setattr__(runtime, "_unit_policy", SimpleNamespace())
    object.__setattr__(runtime, "current_tool_call_id", "blocked-dispatch")
    object.__setattr__(runtime, "unit_blocked_reason", "OTHER")
    object.__setattr__(runtime, "unit_blocked_detail", "contract cannot be preserved")
    object.__setattr__(runtime, "journal", SimpleNamespace(path=tmp_path / "events.jsonl"))
    return runtime


def _incomplete_result(value: str = "RUN_INCOMPLETE") -> dict[str, str]:
    return {"result": value, "outcome": value}


def _authorize_block(runtime: StateFrameRuntime) -> None:
    authorize_subframe_block(runtime, "OTHER", "contract cannot be preserved")
    runtime.system_outcome = "unit_blocked"


@pytest.mark.parametrize(("result", "expected_reason"), _STRONGER_TERMINALS)
def test_authorized_subframe_block_never_masks_stronger_terminal(
        tmp_path, result: str, expected_reason: str):
    runtime = _runtime(tmp_path)
    join_terminal(runtime, _incomplete_result(), writer="prior-finish")
    _authorize_block(runtime)
    join_terminal(
        runtime,
        {"result": "SUBFRAME_BLOCKED", "system_outcome": "unit_blocked"},
        writer="authorized-subframe-terminal",
    )
    assert stop_reason_rank(result, allow_subframe_block=True) > stop_reason_rank(
        "SUBFRAME_BLOCKED", allow_subframe_block=True,
    )

    classified = classify_run_end(
        {"result": result, "outcome": result},
        system_outcome=runtime.system_outcome,
        runtime=runtime,
    )

    assert classified["stop_reason"] == expected_reason
    assert classified["terminal_status"] == result


@pytest.mark.parametrize("incomplete", _INCOMPLETE_ALIASES)
def test_authorized_subframe_block_outranks_generic_incomplete(
        tmp_path, incomplete: str):
    runtime = _runtime(tmp_path)
    join_terminal(runtime, _incomplete_result(), writer="prior-finish")
    _authorize_block(runtime)

    joined = join_terminal(
        runtime, _incomplete_result(incomplete), writer="StateFrameRuntime.finish",
    )
    assert stop_reason_rank(incomplete) < stop_reason_rank(
        "SUBFRAME_BLOCKED", allow_subframe_block=True,
    )

    classified = classify_run_end(
        joined, system_outcome=runtime.system_outcome, runtime=runtime,
    )

    assert classified["stop_reason"] == "BLOCKED"
    assert classified["stop_cause"] == "BLOCKED"
