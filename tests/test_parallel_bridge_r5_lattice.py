"""Authorized bridge subframe terminals survive the release terminal lattice."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_stateframe_runtime import StateFrameRuntime
from kernel_stop_reason_taxonomy import classify_run_end
from kernel_terminal_lattice import join_terminal
from kernel_terminal_policy import authorize_subframe_block


def _runtime(tmp_path: Path, *, subframe: bool = True,
             lattice_mode: str = "on") -> StateFrameRuntime:
    runtime = object.__new__(StateFrameRuntime)
    object.__setattr__(runtime, "stateframe_terminal_lattice_mode", lattice_mode)
    object.__setattr__(runtime, "stateframe_hard_budget_only_terminals_mode", "on")
    object.__setattr__(runtime, "_unit_policy", SimpleNamespace() if subframe else None)
    object.__setattr__(runtime, "current_tool_call_id", "blocked-dispatch")
    object.__setattr__(runtime, "unit_blocked_reason", "OTHER")
    object.__setattr__(runtime, "unit_blocked_detail", "contract cannot be preserved")
    object.__setattr__(runtime, "journal", SimpleNamespace(path=tmp_path / "events.jsonl"))
    return runtime


def _incomplete_result():
    return {"result": "RUN_INCOMPLETE", "outcome": "RUN_INCOMPLETE"}


def test_authorized_subframe_terminal_beats_prior_incomplete_and_classifies_blocked(
        tmp_path):
    runtime = _runtime(tmp_path)
    seeded = join_terminal(runtime, _incomplete_result(), writer="prior-finish")
    assert seeded["result"] == "RUN_INCOMPLETE"

    authorize_subframe_block(runtime, "OTHER", "contract cannot be preserved")
    runtime.system_outcome = "unit_blocked"
    assert runtime.system_outcome == "unit_blocked"

    finished = join_terminal(runtime, _incomplete_result(), writer="StateFrameRuntime.finish")
    assert finished["result"] == "RUN_INCOMPLETE"
    assert runtime.system_outcome == "unit_blocked"

    # This is the bridge's explicit system_outcome projection after finish().
    projected_result = {
        "unit_blocked": "SUBFRAME_BLOCKED",
    }.get(runtime.system_outcome, runtime.system_outcome.upper())
    assert projected_result == "SUBFRAME_BLOCKED"

    classified = classify_run_end(
        finished, system_outcome=runtime.system_outcome, runtime=runtime,
    )
    assert classified["stop_reason"] == "BLOCKED"
    assert classified["stop_cause"] == "BLOCKED"
    projected_classification = classify_run_end(
        {"result": projected_result}, system_outcome=projected_result, runtime=runtime,
    )
    assert projected_classification["stop_reason"] == "BLOCKED"


@pytest.mark.parametrize("terminal", [
    "unit_blocked", "SUBFRAME_BLOCKED", "UNRECOGNIZED_TERMINAL",
])
def test_unauthorized_or_unknown_terminal_cannot_outrank_incomplete(tmp_path, terminal):
    runtime = _runtime(tmp_path, subframe=False)
    seeded = join_terminal(runtime, _incomplete_result(), writer="prior-finish")
    assert seeded["result"] == "RUN_INCOMPLETE"

    # Bypass the runtime setter to model an untrusted adapter or malformed result.
    object.__setattr__(runtime, "system_outcome", terminal)
    joined = join_terminal(runtime, {"result": terminal}, writer="untrusted-terminal")

    assert joined["result"] == "RUN_INCOMPLETE"
    assert runtime.system_outcome == "RUN_INCOMPLETE"
    classified = classify_run_end(
        {"result": terminal}, system_outcome=terminal, runtime=runtime,
    )
    assert classified["stop_reason"] != "BLOCKED"


def test_lattice_off_returns_candidate_unchanged_for_subframe_labels(tmp_path):
    runtime = _runtime(tmp_path, subframe=False, lattice_mode="off")
    object.__setattr__(runtime, "system_outcome", "unit_blocked")
    candidate = {"result": "SUBFRAME_BLOCKED", "system_outcome": "unit_blocked"}

    joined = join_terminal(runtime, candidate, writer="lattice-off")

    assert joined is candidate
    assert joined == {"result": "SUBFRAME_BLOCKED", "system_outcome": "unit_blocked"}
    assert json.dumps(joined, separators=(",", ":")).encode() == json.dumps(
        candidate, separators=(",", ":"),
    ).encode()
    assert runtime.system_outcome == "unit_blocked"


def test_ordinary_terminal_bytes_stay_unchanged_without_subframe_authority(tmp_path):
    runtime = _runtime(tmp_path, subframe=False)
    candidate = _incomplete_result()
    original_bytes = json.dumps(candidate, separators=(",", ":")).encode()

    joined = join_terminal(runtime, candidate, writer="ordinary-finish")

    assert joined == candidate
    assert json.dumps(joined, separators=(",", ":")).encode() == original_bytes
