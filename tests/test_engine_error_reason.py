"""ERR:* engine outcomes name the exception class and message, and the catch-all records a reason row.

Incident (crew-09 L4, 0.7.3): the run ended ERR:FrameFit5Error. engine-result.json carried only the result
name, summary.json had no engine_error_reason, and problems.jsonl had no row.
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

import kernel_reason_log as rl
from _fault_contract import isolated_reason_log
from stateframe import _child, outcomes, runner
from stateframe.engine.probes import react_repair
from test_intake_orientation_fail_loud import _fixture_row

INCIDENT_MESSAGE = "typed next-site receipt is incomplete: current_bytes"
HANDLER_ID = "react_repair.run_exception"


class FrameFit5Error(Exception):
    """Stands in for the typed frame-fit fault from the L4 incident."""


# ---- outcomes.engine_error_reason ----------------------------------------------------------------

def test_err_outcome_reason_names_class_and_message():
    reason = outcomes.engine_error_reason({"requests": 7, "engine_terminal": {
        "result": "ERR:FrameFit5Error", "error_type": "FrameFit5Error", "error_message": INCIDENT_MESSAGE}})
    assert reason == f"engine error: FrameFit5Error: {INCIDENT_MESSAGE}"


def test_err_outcome_without_message_falls_back_to_result_name():
    reason = outcomes.engine_error_reason({"engine_terminal": {"result": "ERR:ValueError", "error_type": "ValueError"}})
    assert reason == "engine error: ERR:ValueError"


def test_seal_failure_reason_is_unchanged():
    reason = outcomes.engine_error_reason({"requests": 7, "engine_terminal": {
        "result": "HARNESS_CONTRACT", "error_message": "bad seal"}})
    assert reason == "engine could not seal request 8: bad seal (HARNESS_CONTRACT)"


def test_successful_terminal_has_no_engine_error_reason():
    assert outcomes.engine_error_reason({"engine_terminal": {"result": "SUBMITTED"}}) is None


# ---- react_repair catch-all: returned outcome + problems.jsonl row --------------------------------

@pytest.fixture()
def reason_log(monkeypatch, tmp_path):
    with isolated_reason_log(monkeypatch, tmp_path) as harness:
        yield harness


def test_catch_all_returns_class_and_message_and_records_reason_row(tmp_path, monkeypatch, reason_log):
    # Same worker setup as test_intake_orientation_fail_loud._run_to_first_request, but the first model
    # request raises a typed engine fault instead of stopping, so the run reaches the catch-all.
    run_dir = tmp_path / "run"
    journal_parent = run_dir / "worker" / "measurement-runs"
    journal_parent.mkdir(parents=True)
    private_temp = tmp_path / "worker-tmp"
    private_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    runtime_type = importlib.import_module("kernel_stateframe_runtime").StateFrameRuntime
    original_init = runtime_type.__init__

    def scoped_init(self: object, owner: object, **kwargs: object) -> None:
        kwargs["output"] = journal_parent
        original_init(self, owner, **kwargs)

    monkeypatch.setattr(runtime_type, "__init__", scoped_init)

    def raise_engine_fault(*_args: object, **_kwargs: object) -> object:
        raise FrameFit5Error(INCIDENT_MESSAGE)

    monkeypatch.setattr(react_repair, "ask_tools", raise_engine_fault)
    row = _fixture_row(tmp_path / "fixture", "engine-error-reason-1")
    # Spy on the reason-log call the catch-all makes: the fault row must be written with journal=False so no
    # reason_logged event shifts the identity-bearing journal positions.
    seen_journal_args: list[object] = []
    original_emit_reason = rl.emit_reason

    def spy_emit_reason(category, cause_key, *args, **kwargs):
        if str(cause_key).startswith(HANDLER_ID):
            seen_journal_args.append(kwargs.get("journal", "<absent>"))
        return original_emit_reason(category, cause_key, *args, **kwargs)

    monkeypatch.setattr(rl, "emit_reason", spy_emit_reason)

    result = react_repair.work(row)

    assert result["result"] == "ERR:FrameFit5Error"
    assert result["error_type"] == "FrameFit5Error"
    assert result["error_message"] == INCIDENT_MESSAGE
    assert outcomes.engine_error_reason({"engine_terminal": result}) == (
        f"engine error: FrameFit5Error: {INCIDENT_MESSAGE}")

    assert seen_journal_args == [False]
    assert not [kind for kind, _fields in reason_log.journal.events if kind == "reason_logged"]

    rows = [json.loads(line) for line in Path(reason_log.problems).read_text().splitlines() if line.strip()]
    matching = [item for item in rows if isinstance(item.get("fields"), dict)
                and item["fields"].get("handler_id") == HANDLER_ID]
    assert len(matching) == 1, rows
    fields = matching[0]["fields"]
    assert fields["cause_code"] == "FrameFit5Error"
    assert fields["disposition"] == "boundary"
    assert fields["expectation"] == "unexpected"
    assert INCIDENT_MESSAGE in matching[0]["why"]


REPLAY_REASON = "replay diverged at send 4: prompt bytes differ"


# Truth table for the resume hint. Rule: ERR:* keeps the hint; everything else is the pre-2c334afb rule
# (no engine_error_reason on a stopped run). ReplayUnfaithful/CheckpointUnfaithful arrive as ENGINE_ERROR with a
# reason copied from engine.log, so they must stay suppressed.
@pytest.mark.parametrize("summary, expected", [
    ({"runtime_outcome": "engine_error", "engine_error_reason": "engine error: X: y",
      "engine_terminal": {"result": "ERR:X"}}, True),
    ({"runtime_outcome": "engine_error", "engine_terminal": {"result": "ERR:X"}}, True),
    ({"runtime_outcome": "engine_error", "engine_error_reason": REPLAY_REASON,
      "engine_terminal": {"result": "ENGINE_ERROR", "error": "ReplayUnfaithful"}}, False),
    ({"runtime_outcome": "engine_error", "engine_terminal": {"result": "ENGINE_ERROR", "error": "ValueError"}}, True),
    ({"runtime_outcome": "engine_error", "engine_error_reason": "engine could not seal request 2: z",
      "engine_terminal": {"result": "FINALIZATION_ERROR"}}, False),
    ({"runtime_outcome": "engine_error", "engine_error_reason": "engine could not seal request 2: z",
      "engine_terminal": {"result": "HARNESS_CONTRACT"}}, False),
    ({"runtime_outcome": "budget_stop", "engine_terminal": {"result": "BUDGET_EXHAUSTED"}}, True),
    ({"runtime_outcome": "budget_stop", "engine_error_reason": REPLAY_REASON, "engine_terminal": {}}, False),
    ({"runtime_outcome": "missing_result"}, True),
    ({"runtime_outcome": "completed", "engine_terminal": {"result": "ERR:X"}}, False),
    ({"runtime_outcome": "completed"}, False),
])
def test_resume_hint_truth_table(summary, expected):
    assert outcomes.resume_hint_applies(summary) is expected


# ---- fixture envelope -> terminal receipt -> runner summary -> outcome projection ------------------

def test_fixture_envelope_names_class_and_message_through_summary_and_projection(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    envelope = {"result": {"result": "ERR:FrameFit5Error", "error_type": "FrameFit5Error",
                           "error_message": INCIDENT_MESSAGE, "steps": 3, "edits": 0, "tests": 0},
                "seconds": 1.2}
    (run / "engine-result.json").write_text(json.dumps(envelope), encoding="utf-8")

    receipt = outcomes.terminal_receipt(run)
    assert receipt["error_type"] == "FrameFit5Error"
    assert receipt["error_message"] == INCIDENT_MESSAGE

    # runner.summarize copies error_type/error_message into summary engine_terminal (its key tuple).
    summary = runner.summarize(run, 0.0)
    assert summary["engine_terminal"]["error_type"] == "FrameFit5Error"
    assert summary["engine_terminal"]["error_message"] == INCIDENT_MESSAGE

    projected = outcomes.project(summary)
    assert projected["engine_error_reason"] == f"engine error: FrameFit5Error: {INCIDENT_MESSAGE}"


# ---- _child terminal construction -----------------------------------------------------------------

def test_child_engine_error_terminal_carries_message_capped_at_600():
    result = _child._engine_error_terminal("ENGINE_ERROR", ValueError("boom\n  second line"))
    assert result == {"result": "ENGINE_ERROR", "error": "ValueError", "error_type": "ValueError",
                      "error_message": "boom second line"}
    capped = _child._engine_error_terminal("ENGINE_ERROR", ValueError("x" * 1000))
    assert len(capped["error_message"]) == 600


def test_child_engine_error_terminal_without_message_keeps_bare_shape():
    assert _child._engine_error_terminal("ENGINE_ERROR", ValueError("")) == {
        "result": "ENGINE_ERROR", "error": "ValueError"}


class _RefusingMutationRuntime:
    def invoke_hook(self, **_kwargs: object) -> None:
        raise RuntimeError("hook refused\nlater")

    def close(self) -> None:
        return None


def test_mutation_containment_terminal_carries_class_and_message():
    events: list[tuple[str, dict]] = []
    result, status = _child._finish_mutation_runtime(
        _RefusingMutationRuntime(), {"result": "FINISHED"}, 0,
        lambda kind, **fields: events.append((kind, fields)))
    assert status == 1
    assert result == {"result": "MUTATION_CONTAINMENT", "error": "hook refused\nlater",
                      "error_type": "RuntimeError", "error_message": "hook refused later"}
    assert events and events[0][0] == "engine_error"


def test_outcome_reason_for_engine_error_and_mutation_containment_with_message():
    engine = outcomes.engine_error_reason({"engine_terminal": {
        "result": "ENGINE_ERROR", "error": "ValueError", "error_type": "ValueError", "error_message": "boom"}})
    assert engine == "engine error: ValueError: boom"
    containment = outcomes.engine_error_reason({"engine_terminal": {
        "result": "MUTATION_CONTAINMENT", "error_type": "RuntimeError", "error_message": "hook refused"}})
    assert containment == "engine error: RuntimeError: hook refused"


def test_outcome_reason_unchanged_for_engine_error_without_message():
    assert outcomes.engine_error_reason({"engine_terminal": {"result": "ENGINE_ERROR", "error": "X"}}) is None


def test_replay_guidance_from_engine_log_is_kept_over_terminal_message():
    # ReplayUnfaithful carries actionable guidance in summary engine_error_reason (copied from engine.log).
    summary = {"engine_error_reason": REPLAY_REASON, "engine_terminal": {
        "result": "ENGINE_ERROR", "error": "ReplayUnfaithful", "error_type": "ReplayUnfaithful",
        "error_message": "terminal detail"}}
    assert outcomes.engine_error_reason(summary) == REPLAY_REASON
