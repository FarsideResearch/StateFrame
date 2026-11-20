"""Count repeated frame projection work and check fresh inputs across renders."""
from __future__ import annotations

import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import kernel_stateframe_framefit5 as frame
from kernel_continuation_journal import ContinuationJournal
from kernel_continuation_runtime import LiveContinuation
from kernel_request_source_window import verify_current_source_message
from kernel_stateframe_action_ledger import build_action_context
from stateframe.replay_tape import Replayer


@pytest.fixture
def pilot(monkeypatch):
    name = "stateframe.send1_wave4"

    def select(enabled):
        module = ModuleType(name)
        module.enabled = lambda: enabled
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    return select


def _state_message(state):
    return {"role": "user", "content": frame._serialize_state_message(
        {"state": {"framefit5": state}}
    )}


def _wire(messages):
    return json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":")).encode()


def test_one_projection_per_assembly_on_pilot(monkeypatch, pilot):
    original = frame._task_and_latest_turn
    calls = 0

    def counted(messages):
        nonlocal calls
        calls += 1
        return original(messages)

    monkeypatch.setattr(frame, "_task_and_latest_turn", counted)
    system = {"role": "system", "content": "Follow the task."}
    task = {"role": "user", "content": "Task: Inspect the value."}
    suffix = {"role": "user", "content": [{"type": "text", "text": "Include the value."}]}
    messages = [system, task, suffix]
    prepared = [system, _state_message({"mode": frame.PRODUCT_MODE})]

    pilot(False)
    eager = frame.assemble_framefit5_request(prepared, messages)
    assert calls == 2
    calls = 0

    pilot(True)
    projected = frame.assemble_framefit5_request(prepared, messages)
    assert calls == 1
    assert _wire(projected) == _wire(eager)
    projected[-2]["content"][0]["text"] = "Changed result."
    assert suffix["content"][0]["text"] == "Include the value."
    assert _wire(frame.assemble_framefit5_request(prepared, messages)) == _wire(eager)
    assert calls == 2


def test_final_render_and_replay_use_changed_source_and_receipts(tmp_path, pilot):
    path = "value.py"
    target = tmp_path / path
    task = {"role": "user", "content": "Task: Inspect the value."}
    system = {"role": "system", "content": "Follow the task."}
    placement = {"role": "user", "content": "Include the current value."}

    def revision():
        return "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()

    journal_root = tmp_path / "journal"
    journal_root.mkdir()
    journal = ContinuationJournal(journal_root, "projection", {"origin": "test"})
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path, task_row={"msg": task["content"]}, current_step=1),
        edit_contract={"edit_authority": {"editable_paths": [path]}},
        journal=journal,
        continuation_source_refresh=False,
        step_limit=8,
        binding=lambda: {"artifact": revision(), "product_files": {path: revision()}},
    )
    continuation = LiveContinuation(runtime)

    def render(source, call_id, status, *, source_first):
        target.write_text(source)
        current = revision()
        assistant = {"role": "assistant", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "read", "arguments": json.dumps({"path": path})},
        }]}
        result = {
            "role": "tool", "tool_call_id": call_id,
            "content": f"[read page] path={path} revision={current} lines=1-1/1 "
                       f"chars={len(source)} clip=none\ncurrent source\n{source}",
        }
        window = {"role": "user", "content": f"[Current source: {path}]\n{source}"}
        suffix = [window, placement] if source_first else [placement, window]
        messages = [system, task, assistant, result, *suffix]
        pilot(True)
        continuation.prepare(messages)
        continuation_block = {"role": "user", "content": continuation.context()}
        messages.append(continuation_block)
        action_context = build_action_context(messages, workspace_root=tmp_path)
        action_context["journal_receipt_records"] = [{
            "kind": "check_receipt", "status": status, "tested_rev": current,
            "receipt_id": f"check-{call_id}", "selector": "verify",
        }]
        state = frame.build_typed_state(messages, action_context, runtime)
        state_message = _state_message(state)
        prepared = [system, state_message]
        projected = frame.assemble_framefit5_request(prepared, messages)
        pilot(False)
        eager = frame.assemble_framefit5_request(prepared, messages)
        assert _wire(projected) == _wire(eager)
        assert projected[:-1] == [
            system, task, state_message, *suffix, continuation_block, assistant, result,
        ]
        assert verify_current_source_message(window["content"], path=path, source=source)
        assert state["last_verdict"]["check_receipt_id"] == f"check-{call_id}"
        assert state["last_verdict"]["status"] == status
        assert state["next_site"]["revision"] == current
        return _wire(projected), json.loads(continuation_block["content"].split("\n", 1)[1])

    try:
        draft, draft_continuation = render("VALUE = 1\n", "read-1", "PASS", source_first=False)
        final, final_continuation = render("VALUE = 2\n", "read-2", "FAIL", source_first=True)
        assert final_continuation["revision"] > draft_continuation["revision"]
        assert final_continuation["active"]["id"] == draft_continuation["active"]["id"]
        assert final_continuation["active"]["stale_dependencies"]
        assert b"VALUE = 1" not in final
    finally:
        journal.close()

    request_path = tmp_path / "request.json"
    request_path.write_bytes(final)
    response = b'{"choices":[]}'
    response_path = tmp_path / "response.json"
    response_path.write_bytes(response)
    tape = [{
        "send": 1, "request": str(request_path), "response": str(response_path),
        "response_sha256": hashlib.sha256(response).hexdigest(),
    }]
    assert Replayer(tape, strict=True).answer(final) == response
    replay = Replayer(tape, strict=True)
    assert replay.answer(draft) is None
    assert replay.diverged is not None
