"""Restore optional model notes from closed, journaled state operations."""
from __future__ import annotations

import copy

from kernel_continuation_restart import WorkerResumeError
from kernel_decision_state import DecisionState


def snapshot(state):
    if state is None:
        return None
    if state.items:
        raise WorkerResumeError("working-note restart cannot restore work items")
    return {"schema": "working-notes/1", "notes": copy.deepcopy(state.notes),
            "archive": copy.deepcopy(state.note_archive), "sequence": state.sequence}

def restore(runtime, saved, records, boundary):
    value = saved.get("working_notes")
    state = runtime.decision_state
    prefix = [e for e in records if e["sequence"] <= boundary["sequence"]]
    updates = [e for e in prefix if e["kind"] == "decision_state_updated"
               and e.get("operation") in {"note_update", "note_attached"}]
    if state is None:
        if value is not None or updates:
            raise WorkerResumeError("working notes in disabled profile")
        return
    replay = DecisionState()
    ledger = saved["ledger"]
    closed = [e["record"] for e in prefix if e["kind"] == "action_outcome_closed"]
    if closed != ledger:
        raise WorkerResumeError("working-note ledger differs from journal")
    try:
        for event in updates:
            if event["operation"] == "note_update":
                accepted = [r for r in closed
                            if r["tool"] == "note_update" and r["result"] == "APPLIED"]
                record = next((r for r in accepted
                               if r["step"] == event["current"]["updated_step"]), None)
                if record is None:
                    raise ValueError("note update lacks closed action")
                closures = [e for e in prefix if e["kind"] == "native_batch_action_closed"
                            and e.get("step_record_id") == record["step_record_id"]]
                if len(closures) != 1:
                    raise ValueError("note lacks unique native closure")
                selected = [e for e in prefix if e["kind"] == "native_batch_action_selected"
                            and e["tool_call"]["id"] == closures[0]["tool_call_id"]]
                if len(selected) != 1:
                    raise ValueError("note lacks unique selected proposal")
                import json
                args = json.loads(selected[0]["tool_call"]["function"]["arguments"])
                if not isinstance(args, dict):
                    raise TypeError("note arguments must be an object")
                args.pop("why", None)
                args.pop("reason", None)
                generated = replay.apply("note_update", args,
                    records=[r for r in closed if r["step"] < record["step"]],
                    binding=record["before_binding"], public_verified=False, step=record["step"])
            else:
                current = event.get("current", {})
                action = event.get("action_record")
                if not isinstance(current, dict) or not isinstance(action, dict):
                    raise ValueError("attached note lacks action record")
                record = next((r for r in closed
                               if r.get("step_record_id") == action.get("step_record_id")), None)
                if record is None or record != action:
                    raise ValueError("attached note action differs from ledger")
                args = event.get("arguments")
                snapshot_value = current.get("dependency_snapshot")
                binding = current.get("binding")
                if not isinstance(args, dict) or not isinstance(snapshot_value, dict):
                    raise TypeError("attached note checkpoint is incomplete")
                generated = replay.apply_attached_note(
                    args, action_record=record, binding=binding,
                    step=int(record["step"]), dependency_snapshot=snapshot_value)
            if any(event.get("state_sequence" if key == "sequence" else key) != item for key, item in generated.items()):
                raise ValueError("note differs from its journaled operation")
        expected_snapshot = copy.deepcopy(value)
        expected_snapshot.setdefault("archive", {})
        if snapshot(replay) != expected_snapshot:
            raise ValueError("checkpoint differs from replayed notes")
    except (KeyError, TypeError, ValueError) as error:
        raise WorkerResumeError("invalid working-note restoration") from error
    state.notes = copy.deepcopy(replay.notes)
    state.note_archive = copy.deepcopy(replay.note_archive)
    state.sequence = replay.sequence
