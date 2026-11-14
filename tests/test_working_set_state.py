"""Shadow lifecycle: durable debt, response adoption, and bounded live authority."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import io
import json
import pickle
from types import SimpleNamespace

import pytest

import kernel_read_receipts as receipts
import kernel_working_set_state as ws
import kernel_continuation_restart as restart
import kernel_checkpoint as checkpoint


def capture(number=1, *, revision="a", path="source.py", payload="x\n"):
    identity = receipts.ReceiptIdentity(path, "sha256:" + revision * 64, 1, 1,
        0, len(payload.encode()), hashlib.sha256(payload.encode()).hexdigest(), None)
    return receipts.ReadCapture(f"capture-{number}", f"attempt-{number}", identity,
        "payload-ref", "page-ref", "disk", "response-1", number, b'{"path":"source.py"}',
        1, 2, *receipts.line_digests_with_numbers(payload, 1))


class Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        event = {"kind": kind, **copy.deepcopy(fields)}
        self.events.append(event)
        return {**event, "event_id": f"event-{len(self.events)}"}


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")


def runtime():
    return SimpleNamespace(journal=Journal())


def submit(rt, request="request-1", modes=None):
    state = ws.state_for(rt)
    if modes is None:
        modes = {o.obligation_id: "F" for o in state.active_obligations()}
    ws.record_submission(rt, request_id=request, body_sha256="1" * 64, modes=modes)


def accept(rt, request="request-1", *, disposition="accepted", digest="1" * 64):
    ws.record_response(rt, request_id=request, response_id=f"response-{request}",
                       body_sha256=digest, disposition=disposition)


def test_due_ordinals_batch_and_adoption():
    rt = runtime()
    submit(rt, "origin", {})
    accept(rt, "origin")
    for n in range(1, 4):
        ws.record_capture(rt, capture(n, path=f"file{n}.py"))
    state = ws.state_for(rt)
    assert {o.due_ordinal for o in state.active_obligations()} == {2}
    assert {o.state for o in state.active_obligations()} == {"DUE"}
    submit(rt)
    assert {o.state for o in state.active_obligations()} == {"DUE"}
    accept(rt, disposition="discarded")
    assert {o.state for o in state.active_obligations()} == {"DUE"}
    submit(rt, "retry")
    accept(rt, "retry", disposition="failed")
    assert {o.state for o in state.active_obligations()} == {"DUE"}
    submit(rt, "adopted")
    accept(rt, "adopted", digest="2" * 64)
    assert {o.state for o in state.active_obligations()} == {"DUE"}
    accept(rt, "adopted")
    assert {o.state for o in state.active_obligations()} == {"RESIDENT"}


def test_fresh_due_and_recovery_cannot_degrade_or_retire():
    rt = runtime()
    ws.record_capture(rt, capture(), recovery_pin=True)
    o, = ws.state_for(rt).active_obligations()
    for mode in ("D", "O"):
        with pytest.raises(ValueError, match="F-only"):
            submit(rt, modes={o.obligation_id: mode})
    submit(rt)
    accept(rt)
    o, = ws.state_for(rt).active_obligations()
    assert o.state == "RESIDENT" and not o.recovery_pin
    submit(rt, "degrade", {o.obligation_id: "D"})
    accept(rt, "degrade")
    assert o.state == "DEGRADED"
    with pytest.raises(ValueError, match="reread"):
        submit(rt, "promote", {o.obligation_id: "F"})
    ws.record_capture(rt, capture(2), recovery_pin=True)
    assert o.state == "DUE" and o.recovery_pin


def test_revision_preserves_unexposed_history_then_supersedes():
    rt = runtime()
    ws.record_capture(rt, capture())
    old, = ws.state_for(rt).active_obligations()
    ws.record_capture(rt, capture(2, revision="b"))
    current = ws.state_for(rt).active_obligations()[-1]
    assert old.state == "DUE" and old.superseded_by == current.obligation_id
    submit(rt)
    accept(rt)
    assert old.state == "SUPERSEDED"
    assert ws.state_for(rt).active_obligations() == (current,)
    ws.record_capture(rt, capture(3, revision="c"))
    assert current.state == "SUPERSEDED"


def test_resident_reread_and_retired_lifetime():
    rt = runtime()
    ws.record_capture(rt, capture())
    first, = ws.state_for(rt).active_obligations()
    submit(rt)
    accept(rt)
    ws.record_capture(rt, capture(2))
    assert first.capture_ids == ("capture-1", "capture-2")
    assert first.state == "DUE" and first.admission_ordinal == 1
    submit(rt, "expose")
    # A capture arriving after submission must survive an older accepted wire.
    ws.record_capture(rt, capture(3))
    accept(rt, "expose")
    assert first.state == "DUE"
    submit(rt, "latest")
    accept(rt, "latest")
    submit(rt, "retire", {first.obligation_id: "O"})
    assert first.state == "RESIDENT"
    accept(rt, "retire")
    assert first.state == "DISCHARGED" and not ws.state_for(rt).active_obligations()
    ws.record_capture(rt, capture(4))
    fresh, = ws.state_for(rt).active_obligations()
    assert fresh.obligation_id != first.obligation_id
    assert fresh.admission_ordinal > first.admission_ordinal and fresh.state == "DUE"


def test_limit_is_explicit_and_capture_history_not_truncated():
    rt = runtime()
    for n in range(ws.MAX_ACTIVE_OBLIGATIONS + 3):
        ws.record_capture(rt, capture(n, path=f"file{n}.py"))
    state = ws.state_for(rt)
    assert len(state.active_obligations()) == ws.MAX_ACTIVE_OBLIGATIONS
    assert state.overflow_count == 3 and not state.shadow_complete
    assert len([e for e in rt.journal.events if e["kind"] == "working_set_capture"]) == ws.MAX_ACTIVE_OBLIGATIONS + 3
    assert rt.journal.events[-1]["capture"]["capture_id"] == f"capture-{ws.MAX_ACTIVE_OBLIGATIONS + 2}"


def test_duplicate_replay_and_checkpoint_roundtrip():
    rt = runtime()
    ws.record_capture(rt, capture(), recovery_pin=True)
    submit(rt)
    # Continuation checkpoints use JSON; generic checkpoints use the engine codec.
    saved = json.loads(json.dumps(ws.state_for(rt).checkpoint()))
    restored = ws.WorkingSetState.from_checkpoint(saved)
    assert restored.checkpoint() == saved
    data = pickle.dumps(ws.state_for(rt))
    stream = io.BytesIO()
    writer = checkpoint._StateWriter(stream)
    writer.write(data)
    writer.finish()
    assert pickle.loads(__import__("zlib").decompress(stream.getvalue())).checkpoint() == saved
    for event in rt.journal.events * 2:
        restored.apply(event)
    assert restored.checkpoint() == saved
    accept(rt)
    restored.apply(rt.journal.events[-1])
    assert restored.checkpoint() == ws.state_for(rt).checkpoint()
    assert "working_set_state" in restart.runtime_fields(rt)
    base = {key: None for key in restart.runtime_fields(rt)}
    base["working_set_state"] = restored.checkpoint()
    target = runtime()
    assert restart.checkpoint_runtime_fields_match(target, base)
    restart.restore_runtime_field_values(target, base)
    assert target.working_set_state.checkpoint() == restored.checkpoint()
    bad = copy.deepcopy(base)
    bad["working_set_state"]["active"][0]["state"] = "BOGUS"
    assert not restart.checkpoint_runtime_fields_match(target, bad)


def test_off_is_inert(monkeypatch):
    monkeypatch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR")
    rt = runtime()
    ws.record_capture(rt, capture())
    submit(rt, modes={})
    accept(rt)
    assert not rt.journal.events and not hasattr(rt, "working_set_state")
    assert "working_set_state" not in restart.runtime_fields(rt)


def test_auxiliary_and_preparation_never_advance_ordinal():
    rt = runtime()
    ws.record_capture(rt, capture())
    before = ws.state_for(rt).checkpoint()
    ws.observe_submission(rt, SimpleNamespace(request_id="aux", request_kind="summary"), b"{}")
    assert ws.state_for(rt).checkpoint() == before


def test_events_have_resume_classification():
    for kind in ws.EVENT_KINDS:
        assert restart.resume_event_rejection_reason(kind) is None


class SequencedJournal(Journal):
    """Records carry the journal position, as ContinuationJournal rows do."""
    def emit(self, kind, **fields):
        event = super().emit(kind, **fields)
        event["sequence"] = len(self.events)
        self.events[-1]["sequence"] = len(self.events)
        return event


def _read_rows(journal, attempt, *, result=True):
    journal.emit("read_attempt", attempt_id=attempt, path="source.py", requested_range={},
                 request_index=0, provider_call_id=None, batch_index=None)
    if result:
        journal.emit("read_result", attempt_id=attempt, status="OK", request_index=0)


def test_restart_keeps_post_checkpoint_obligation_and_interrupted_read():
    """The 38-turn bug class: a read after the last checkpoint must still be due
    on the next wire after a continuation restart, and a read cut by the restart
    reduces to INTERRUPTED instead of vanishing."""
    rt = SimpleNamespace(journal=SequencedJournal())
    _read_rows(rt.journal, "read-attempt-1")
    ws.record_capture(rt, capture(1))
    submit(rt)
    accept(rt)
    saved = json.loads(json.dumps(ws.state_for(rt).checkpoint()))
    checkpoint_sequence = len(rt.journal.events)
    # Tail after the checkpoint: a second read, its capture, then an interrupted third read.
    _read_rows(rt.journal, "read-attempt-2")
    ws.record_capture(rt, capture(2, revision="b", path="other.py", payload="y\n"))
    _read_rows(rt.journal, "read-attempt-3", result=False)
    live = ws.state_for(rt)
    assert [o.state for o in live.active_obligations()] == ["RESIDENT", "DUE"]

    target = runtime()
    base = {key: None for key in restart.runtime_fields(rt)}
    base["working_set_state"] = saved
    restart.restore_runtime_field_values(target, base)
    # Checkpoint alone forgets the post-checkpoint read: exactly the lost obligation.
    assert [o.state for o in target.working_set_state.active_obligations()] == ["RESIDENT"]
    ws.restore_tail(target, rt.journal.events, checkpoint_sequence)
    restored = target.working_set_state
    assert [o.state for o in restored.active_obligations()] == ["RESIDENT", "DUE"]
    due = restored.active_obligations()[1]
    assert due.due_ordinal == restored.next_due_ordinal == live.logical_ordinal + 1
    assert restored.interrupted_reads == ("read-attempt-3",)
    assert restored.open_read_attempt is None
    # Captures are rebuilt from the journal too: the ledger is never checkpointed.
    assert target.read_capture_ledger.all() == (capture(1), capture(2, revision="b", path="other.py", payload="y\n"))
    assert target.read_capture_ledger.all()[0].line_digests == capture(1).line_digests
    assert restored.checkpoint()["active"] == live.checkpoint()["active"]
    # Duplicate replay of the whole journal (including rows the checkpoint covers) is a no-op.
    again = copy.deepcopy(restored.checkpoint())
    ws.replay_tail(restored, rt.journal.events * 2, 0)
    assert restored.checkpoint() == again
    # The rebuilt state round-trips through the checkpoint codec with its read watermark.
    assert ws.WorkingSetState.from_checkpoint(json.loads(json.dumps(again))).checkpoint() == again


def test_due_ordinal_ignores_overwritten_provider_counter():
    rt = SimpleNamespace(journal=SequencedJournal(), _frame_budget_request_count=7)
    ws.record_capture(rt, capture(1))
    submit(rt)
    accept(rt)
    rt._frame_budget_request_count = 0  # what continuation restart / bookmark replay do
    ws.record_capture(rt, capture(2, revision="b", path="other.py", payload="y\n"))
    rebuilt = ws.reconstruct(rt.journal.events)
    assert rebuilt.active_obligations()[1].due_ordinal == 2 == rebuilt.next_due_ordinal


def test_trailing_read_attempt_is_interrupted_only_at_journal_end():
    journal = SequencedJournal()
    _read_rows(journal, "read-attempt-1", result=False)
    state = ws.WorkingSetState()
    for row in journal.events:
        state.apply_record(row)
    assert state.open_read_attempt == "read-attempt-1" and state.interrupted_reads == ()
    _read_rows(journal, "read-attempt-2")
    for row in journal.events:
        state.apply_record(row)
    assert state.interrupted_reads == ("read-attempt-1",) and state.open_read_attempt is None
    assert state.close_interrupted_reads() == ("read-attempt-1",)
    with pytest.raises(ValueError, match="sequence"):
        state.apply_record({"kind": "read_attempt", "attempt_id": "x"})


def test_restore_runtime_replays_working_set_tail_after_field_values():
    import inspect
    source = inspect.getsource(restart.restore_runtime)
    fields = source.index("restore_runtime_field_values(runtime, saved[\"runtime\"])")
    tail = source.index("restore_working_set_tail(runtime, records, event[\"sequence\"])")
    counter = source.index("restore_provider_request_count(runtime, saved)")
    assert fields < tail < counter


def test_failed_and_discarded_responses_close_pending_without_exposure():
    """A failed or discarded reply closes its submission live; nothing is paid."""
    for disposition in ("failed", "discarded"):
        rt = runtime()
        ws.record_capture(rt, capture(1))
        submit(rt)
        state = ws.state_for(rt)
        assert set(state.pending) == {"request-1"}
        request = SimpleNamespace(request_id="request-1")
        ws.observe_response(rt, request, response_id="error-1", disposition=disposition)
        assert state.pending == {} and state.accepted_exposure_id is None
        assert [o.state for o in state.active_obligations()] == ["DUE"]
        assert rt.journal.events[-1]["kind"] == "working_set_response"
        assert rt.journal.events[-1]["disposition"] == disposition
        # Unknown request ids and missing response ids never emit.
        before = len(rt.journal.events)
        ws.observe_response(rt, SimpleNamespace(request_id="other"), response_id="x", disposition="failed")
        ws.observe_response(rt, request, response_id=None, disposition="failed")
        assert len(rt.journal.events) == before


def test_measurement_observes_every_response_disposition():
    """Live wiring: accepted, failed (terminal transport error) and discarded (V43G)."""
    import ast
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1]
              / "src/stateframe/engine/kernel_measurement.py").read_text()
    dispositions = set()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "observe_response"):
            for keyword in node.keywords:
                if keyword.arg == "disposition" and isinstance(keyword.value, ast.Constant):
                    dispositions.add(keyword.value.value)
    assert dispositions == {"accepted", "failed", "discarded"}
