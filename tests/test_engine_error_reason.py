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

from _fault_contract import isolated_reason_log
from stateframe import outcomes
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

    result = react_repair.work(row)

    assert result["result"] == "ERR:FrameFit5Error"
    assert result["error_type"] == "FrameFit5Error"
    assert result["error_message"] == INCIDENT_MESSAGE
    assert outcomes.engine_error_reason({"engine_terminal": result}) == (
        f"engine error: FrameFit5Error: {INCIDENT_MESSAGE}")

    rows = [json.loads(line) for line in Path(reason_log.problems).read_text().splitlines() if line.strip()]
    matching = [item for item in rows if isinstance(item.get("fields"), dict)
                and item["fields"].get("handler_id") == HANDLER_ID]
    assert len(matching) == 1, rows
    fields = matching[0]["fields"]
    assert fields["cause_code"] == "FrameFit5Error"
    assert fields["disposition"] == "boundary"
    assert fields["expectation"] == "unexpected"
    assert INCIDENT_MESSAGE in matching[0]["why"]


def test_resume_hint_kept_for_err_crash_and_dropped_for_seal_failure():
    from stateframe import outcomes
    crash = {"runtime_outcome": "engine_error", "engine_error_reason": "engine error: X: y",
             "engine_terminal": {"result": "ERR:X"}}
    seal = {"runtime_outcome": "engine_error", "engine_error_reason": "engine could not seal request 2: z",
            "engine_terminal": {"result": "FINALIZATION_ERROR"}}
    assert outcomes.resume_hint_applies(crash)
    assert not outcomes.resume_hint_applies(seal)
    assert outcomes.resume_hint_applies({"runtime_outcome": "budget_stop"})
    assert not outcomes.resume_hint_applies({"runtime_outcome": "completed"})
