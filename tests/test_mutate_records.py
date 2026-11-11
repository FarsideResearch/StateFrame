import json
import base64

import pytest

from kernel_mutate_records import Journal, reconcile


def test_pending_observing_trigger_is_interrupted_once(tmp_path):
    journal = Journal(tmp_path)
    journal.append("HOOK_TRIGGER_INTENT", invocation_id="h123", name="observer",
                   event="post_tool", facts={"tool": "read"})
    first = reconcile([], journal, [])
    assert first.refuse is None
    assert [(action.kind, action.call_id) for action in first.actions] == [("interrupt_hook", "h123")]
    journal.append("HOOK_TRIGGER_INTERRUPTED", invocation_id="h123")
    assert reconcile([], journal, []).actions == []


def test_completed_watch_launch_has_tape_authority(tmp_path):
    from kernel_tool_tape import ToolTape

    journal = Journal(tmp_path)
    tape = ToolTape(tmp_path)
    tape.intent(call_id="w123", tool="mutation_watch", args={"watch": "w1"},
                replay_class="observation", capability_digest="snapshot", definition_digest=None,
                env_identity="mutation-runtime/1", pre_manifest_digest="snapshot")
    journal.append("WATCH_TICK_INTENT", invocation_id="w123", parent_turn_id="t1",
                   snapshot_identity="snapshot")
    journal.append("WATCH_TICK_COMPLETE", invocation_id="w123", parent_turn_id="t1",
                   observation="ok")
    tape.complete(call_id="w123", exit_code=0, observation="ok",
                  post_manifest_digest="snapshot", env_identity="mutation-runtime/1")
    rows = [json.loads(line) for line in tape.path.read_text().splitlines()]
    ledger = [{"schema": "stateframe-mutation-launch/1", "launch": "a" * 32,
               "call": "w123", "state": "REAPED"}]
    assert reconcile(rows, journal, ledger).refuse is None
import kernel_mutate_records
from kernel_mutate_wire import request_key


def terminal_fields(call="c"):
    body = json.dumps({"wire": 1, "call": call, "kind": "result", "result": {}}, separators=(",", ":")).encode()
    frame = b"SFMW1 " + str(len(body)).encode() + b"\n" + body
    return {"frame_b64": base64.b64encode(frame).decode(), "exit_code": 0,
            "stdout_overflow": False, "timed_out": False}


def test_journal_torn_tail_and_middle_corruption(tmp_path):
    journal = Journal(tmp_path)
    journal.append("BROKER_INTENT", call="c")
    with journal.path.open("ab") as stream:
        stream.write(b'{"bad":')
    assert len(journal.events()) == 1
    assert journal.torn_tail
    journal.path.write_bytes(b'{"bad":\n' + journal.path.read_bytes())
    with pytest.raises(ValueError):
        journal.events()


def test_blob_verification(tmp_path):
    journal = Journal(tmp_path)
    digest = journal.put_blob(b"body")
    assert journal.get_blob(digest) == b"body"
    (journal.evidence / digest).chmod(0o600)
    (journal.evidence / digest).write_bytes(b"evil")
    with pytest.raises(ValueError):
        journal.get_blob(digest)


@pytest.mark.parametrize("events,completed,expected", [
    ([], False, "interrupt"),
    (["BROKER_INTENT"], False, "refuse"),
    (["BROKER_INTENT", "BROKER_COMPLETE"], False, "interrupt"),
    (["TERMINAL", "CLEANUP_VERIFIED"], False, "complete"),
    (["EFFECT_PREPARED", "EFFECT_COMMITTED", "CLEANUP_VERIFIED"], False, "refuse"),
    (["CLEANUP_FAILED"], False, "cleanup"),
    (["EFFECT_PREPARED", "EFFECT_COMMITTED", "EFFECT_PREPARED"], False, "refuse"),
    (["TERMINAL", "CLEANUP_VERIFIED"], True, "serve"),
])
def test_reconciliation_table(tmp_path, events, completed, expected):
    journal = Journal(tmp_path)
    for kind in events:
        if kind == "BROKER_INTENT":
            journal.append(kind, call="c", key=request_key("https://a.test", ""), url="https://a.test", accept="")
        elif kind == "BROKER_COMPLETE":
            journal.append(kind, call="c", key=request_key("https://a.test", ""), error="timeout")
        elif kind == "TERMINAL":
            journal.append(kind, call="c", **terminal_fields())
        else:
            journal.append(kind, call="c")
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate", "replay_class": "observation"}]
    if completed:
        rows.append({"kind": "complete", "call_id": "c", "observation": "done"})
    result = reconcile(rows, journal, [])
    assert ("refuse" if result.refuse else result.actions[0].kind) == expected


def test_journal_without_intent_refused(tmp_path):
    journal = Journal(tmp_path)
    journal.append("TERMINAL", call="missing")
    assert reconcile([], journal, []).refuse


def test_completed_without_terminal_evidence_refused(tmp_path):
    journal = Journal(tmp_path)
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate", "replay_class": "observation"},
            {"kind": "complete", "call_id": "c", "observation": "done"}]
    assert "EVIDENCE" in reconcile(rows, journal, []).refuse


def test_completed_with_unfinished_broker_refused(tmp_path):
    journal = Journal(tmp_path)
    for kind in ("BROKER_INTENT", "TERMINAL", "CLEANUP_VERIFIED"):
        journal.append(kind, call="c", **({"key": request_key("https://a.test", ""), "url": "https://a.test", "accept": ""} if kind == "BROKER_INTENT" else terminal_fields() if kind == "TERMINAL" else {}))
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate"}, {"kind": "complete", "call_id": "c"}]
    assert "BROKER" in reconcile(rows, journal, []).refuse


def test_archived_attempts_have_distinct_call_ids(tmp_path):
    journal = Journal(tmp_path)
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate", "attempt": "1"},
            {"kind": "intent", "call_id": "c", "tool": "mutate", "attempt": "2"}]
    result = reconcile(rows, journal, [])
    assert not result.refuse
    assert [(action.attempt, action.kind) for action in result.actions] == [("1", "interrupt"), ("2", "interrupt")]


def test_catalog_completion_needs_no_payload_evidence(tmp_path):
    journal = Journal(tmp_path)
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate", "args": {"op": "catalog"}},
            {"kind": "complete", "call_id": "c", "observation": "catalog"}]
    assert reconcile(rows, journal, []).actions[0].kind == "serve"


def test_pending_metadata_operation_reconciles_idempotently(tmp_path):
    journal = Journal(tmp_path)
    journal.append("EPISODE_STARTED", call="c", episode="e")
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate", "args": {"op": "enter"}}]
    assert reconcile(rows, journal, []).actions[0].kind == "complete_metadata"


def test_journal_refuses_duplicate_keys_and_oversize_append(tmp_path):
    journal = Journal(tmp_path)
    journal.path.write_bytes(b'{"kind":"X","kind":"Y"}\n')
    with pytest.raises(ValueError):
        journal.events()
    journal.path.unlink()
    with pytest.raises(ValueError):
        journal.append("AUDIT", source="x" * (2 << 20))


def test_reconcile_rejects_bad_terminal_base64(tmp_path):
    journal = Journal(tmp_path)
    journal.append("TERMINAL", call="c", frame_b64="!!!")
    journal.append("CLEANUP_VERIFIED", call="c")
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate"}]
    assert "EVIDENCE" in reconcile(rows, journal, []).refuse


def test_reconcile_uses_valid_terminal_frame(tmp_path):
    journal = Journal(tmp_path)
    journal.append("TERMINAL", call="c", **terminal_fields())
    journal.append("CLEANUP_VERIFIED", call="c")
    result = reconcile([{"kind": "intent", "call_id": "c", "tool": "mutate"}], journal, [])
    assert result.actions[0].kind == "complete"
    assert result.actions[0].evidence["frame_b64"]


def test_journal_reader_refuses_symlink_and_oversize(tmp_path):
    journal = Journal(tmp_path, max_bytes=1024)
    journal.path.write_bytes(b'{"kind":"X","padding":"' + b"x" * 1025 + b'"}\n')
    with pytest.raises(ValueError):
        journal.events()
    journal.path.unlink()
    target = tmp_path / "target"
    target.write_text('{"kind":"X"}\n')
    journal.path.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        journal.events()


def test_reconciliation_prioritizes_unreaped_launch(tmp_path):
    journal = Journal(tmp_path)
    journal.append("TERMINAL", call="c")
    journal.append("CLEANUP_VERIFIED", call="c")
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate"},
            {"kind": "complete", "call_id": "c", "observation": "done"}]
    ledger = [{"schema": "stateframe-mutation-launch/1", "launch": "l", "call": "c", "state": "REGISTERED"}]
    assert reconcile(rows, journal, ledger).actions[0].kind == "cleanup"


def test_effect_recovery_requires_verified_pre_post_blobs(tmp_path):
    journal = Journal(tmp_path)
    before = journal.put_blob(b"old")
    after = journal.put_blob(b"new")
    files = [{"path": "a.txt", "pre_sha256": before, "post_sha256": after,
              "pre_blob": before, "post_blob": after}]
    journal.append("TERMINAL", call="c", **terminal_fields())
    journal.append("CLEANUP_VERIFIED", call="c")
    journal.append("EFFECT_PREPARED", call="c", files=files)
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate"}]
    assert reconcile(rows, journal, []).actions[0].kind == "recover_effect"
    journal.append("EFFECT_COMMITTED", call="c")
    assert reconcile(rows, journal, []).actions[0].kind == "verify_effect"
    (journal.evidence / after).chmod(0o600)
    (journal.evidence / after).write_bytes(b"tampered")
    assert "EVIDENCE" in reconcile(rows, journal, []).refuse


def test_rolled_back_edit_is_interrupted_without_rollforward(tmp_path):
    journal = Journal(tmp_path)
    before = journal.put_blob(b"old")
    after = journal.put_blob(b"new")
    journal.append("TERMINAL", call="c", **terminal_fields())
    journal.append("CLEANUP_VERIFIED", call="c")
    journal.append("EFFECT_PREPARED", call="c", files=[{
        "path": "a.txt", "pre_sha256": before, "post_sha256": after,
        "pre_blob": before, "post_blob": after}])
    journal.append("EFFECT_REFUSED", call="c")
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate"}]
    result = reconcile(rows, journal, [])
    assert result.refuse is None
    assert result.actions[0].kind == "interrupt"


def test_edit_without_terminal_refuses(tmp_path):
    journal = Journal(tmp_path)
    before = journal.put_blob(b"old")
    journal.append("EFFECT_PREPARED", call="c", files=[{"path": "a.txt", "pre_sha256": before,
        "post_sha256": None, "pre_blob": before, "post_blob": None}])
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate"}]
    assert "EVIDENCE" in reconcile(rows, journal, []).refuse


def test_journal_reader_handles_short_regular_file_reads(tmp_path, monkeypatch):
    journal = Journal(tmp_path)
    journal.append("EPISODE_STARTED", call="c")
    original = kernel_mutate_records.os.read
    with monkeypatch.context() as patch:
        patch.setattr(kernel_mutate_records.os, "read", lambda fd, size: original(fd, min(size, 5)))
        assert journal.events() == [{"kind": "EPISODE_STARTED", "call": "c"}]


def test_unresolved_cleanup_precedes_bad_terminal_frame(tmp_path):
    journal = Journal(tmp_path)
    journal.append("TERMINAL", call="c", frame_b64="!!!")
    journal.append("CLEANUP_FAILED", call="c")
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate"}]
    assert reconcile(rows, journal, []).actions[0].kind == "cleanup"


def test_send_budget_event_is_not_a_tool_intent(tmp_path):
    journal = Journal(tmp_path)
    journal.append("BUDGET_RESERVE", call="physical-send-1", name="sends_total", amount=1)
    assert reconcile([], journal, []).refuse is None


def test_pending_admission_reconciles_once(tmp_path):
    journal = Journal(tmp_path)
    journal.append("ADMISSION", call="c", admission_digest="sha256:" + "a" * 64)
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate", "args": {"op": "submit"}}]
    action = reconcile(rows, journal, []).actions[0]
    assert action.kind == "complete_metadata"
    assert action.evidence["admission_digest"].startswith("sha256:")


def test_replay_external_never_waives_local_edit_ambiguity(tmp_path):
    journal = Journal(tmp_path)
    journal.append("BROKER_INTENT", call="c", key=request_key("https://a.test", ""), url="https://a.test", accept="")
    journal.append("EFFECT_PREPARED", call="c")
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate"}]
    assert "EFFECT" in reconcile(rows, journal, [], replay_external=True).refuse


def test_broker_completion_must_match_request_key(tmp_path):
    journal = Journal(tmp_path)
    journal.append("BROKER_INTENT", call="c", key=request_key("https://a.test", ""), url="https://a.test", accept="")
    journal.append("BROKER_COMPLETE", call="c", key="sha256:" + "b" * 64, error="timeout")
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate"}]
    assert "BROKER" in reconcile(rows, journal, []).refuse


def test_terminal_result_requires_successful_launch_outcome(tmp_path):
    journal = Journal(tmp_path)
    journal.append("TERMINAL", call="c", **{**terminal_fields(), "exit_code": 1})
    journal.append("CLEANUP_VERIFIED", call="c")
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate"}]
    assert "EVIDENCE" in reconcile(rows, journal, []).refuse


def test_error_terminal_cannot_authorize_edit(tmp_path):
    journal = Journal(tmp_path)
    body = json.dumps({"wire": 1, "call": "c", "kind": "error", "error": {
        "type": "HandlerError", "message": "failed", "trace": ""}, "stderr": ""}, separators=(",", ":")).encode()
    frame = b"SFMW1 " + str(len(body)).encode() + b"\n" + body
    journal.append("TERMINAL", call="c", frame_b64=base64.b64encode(frame).decode(),
                   exit_code=3, stdout_overflow=False, timed_out=False)
    journal.append("CLEANUP_VERIFIED", call="c")
    after = journal.put_blob(b"new")
    journal.append("EFFECT_PREPARED", call="c", files=[{"path": "a.txt", "pre_sha256": None,
        "post_sha256": after, "pre_blob": None, "post_blob": after}])
    rows = [{"kind": "intent", "call_id": "c", "tool": "mutate"}]
    assert "EFFECT" in reconcile(rows, journal, []).refuse
