"""A schema-rejected parallel call writes a refusal receipt and never crashes the frame fold."""
from __future__ import annotations

import json

import pytest

import kernel_parallel
import stateframe.engine.kernel_parallel as sf_parallel
from stateframe.engine.kernel_stateframe_framefit5 import FrameFit5Error, _opaque_actions
from test_parallel_dispatch import setup_context
from test_parallel_f4_receipt import _case

SCHEMA_ERROR = "subframes[0]: 'goal' is a required property"


class _Journal:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, name, **fields):
        self.events.append((name, fields))


def _outcome_only_args():
    # Valid for the runtime's legacy contract, invalid for the advertised schema (goal is required).
    return {"subframes": [{"id": "a", "outcome": "Change a", "owns": ["a.py"], "creates": []}]}


def test_schema_refusal_writes_typed_refusal_receipt(tmp_path, monkeypatch):
    run, _root, context = setup_context(tmp_path, monkeypatch)
    call_root = run / "parallel" / "parallel_1"

    result = sf_parallel.execute_parallel(
        context, _outcome_only_args(), host=None, store_root=call_root,
        schema_errors=(SCHEMA_ERROR,),
    )

    assert result.admission.status == "REFUSED"
    assert result.admission.refusal is not None
    assert result.admission.refusal.reason == "INVALID_ARGUMENTS"
    assert 'add "goal"' in result.observation
    assert result.txn_order == ()
    assert (call_root / "result.json").is_file()
    # The receipt is readable by the same path the frame fold uses, so it is a real exchange.
    stored, request = sf_parallel.action_receipt_source("parallel_1")
    assert stored == result
    assert request is None


def test_schema_refusal_turn_folds_without_engine_error(tmp_path, monkeypatch):
    run, _root, context = setup_context(tmp_path, monkeypatch)
    args = _outcome_only_args()
    result = sf_parallel.execute_parallel(
        context, args, host=None, store_root=run / "parallel" / "parallel_1",
        schema_errors=(SCHEMA_ERROR,),
    )
    turns = ((
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "parallel_1", "type": "function",
            "function": {"name": "parallel", "arguments": json.dumps(args)},
        }]},
        {"role": "tool", "tool_call_id": "parallel_1", "content": result.observation},
    ),)
    monkeypatch.setattr(kernel_parallel, "_service", (run, {"parallel_mode": "2"}))

    rows, ids = _opaque_actions(turns)

    assert ids == {"parallel_1"}
    assert len(rows) == 1


def test_missing_parallel_receipt_becomes_typed_refusal_and_journal_event(tmp_path, monkeypatch):
    _, turns = _case(tmp_path, (("alpha", "COMMITTED"),))
    (tmp_path / "parallel" / "call_0" / "result.json").unlink()
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))
    journal = _Journal()

    rows, ids = _opaque_actions(turns, journal)

    assert ids == {"call_0"}
    row, = rows
    assert row["call_id"] == "call_0"
    assert row["outcome"] == "REJECTED"
    assert row["tool"] == "parallel"
    names = [name for name, _fields in journal.events]
    assert "parallel_receipt_missing" in names
    fields = dict(journal.events)["parallel_receipt_missing"]
    assert fields["call_id"] == "call_0"
    assert fields["dispatch"] is False


def test_missing_parallel_receipt_without_journal_still_folds(tmp_path, monkeypatch):
    _, turns = _case(tmp_path, (("alpha", "COMMITTED"),))
    (tmp_path / "parallel" / "call_0" / "result.json").unlink()
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))

    rows, ids = _opaque_actions(turns)

    assert ids == {"call_0"}
    assert len(rows) == 1


def test_corrupt_parallel_receipt_still_fails_closed(tmp_path, monkeypatch):
    # Only a missing receipt is a refusal observation; present-but-corrupt bytes stay strict.
    _, turns = _case(tmp_path, (("alpha", "COMMITTED"),))
    (tmp_path / "parallel" / "call_0" / "result.json").write_text("{not json")
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "2"}))

    with pytest.raises(FrameFit5Error, match="subframe result or expanded plan is invalid"):
        _opaque_actions(turns, _Journal())


def test_schema_input_error_names_missing_goal_as_actionable():
    error = sf_parallel.schema_input_error(
        _outcome_only_args(), (SCHEMA_ERROR,))
    assert error.unit_id == "a"
    assert error.field == "goal"
    assert error.hint == 'add "goal": ... and resend the whole parallel call'
