"""Operation-count guards for allocator trials and ordinary durable events."""
from collections import Counter
from types import SimpleNamespace

import pytest

import kernel_working_set_prepare as prepare
from kernel_working_set_allocator import FULL
from kernel_working_set_render import propose_slots, render_assignment
from test_working_set_prepare import snapshot, source


@pytest.mark.parametrize("count", [0, 1, 98])
def test_singleton_floor_renders_one_assignment(monkeypatch, count):
    snap = snapshot(*(source(str(i), state="DUE", ordinal=i + 1) for i in range(count)))
    slots = propose_slots(snap)
    expected = render_assignment(snap, {slot.key: FULL for slot in slots})
    calls = []
    original = prepare._best_render

    def render(frozen, modes):
        calls.append(dict(modes))
        return original(frozen, modes)

    monkeypatch.setattr(prepare, "_best_render", render)
    modes, floor = prepare._minimum_completion(snap, slots)
    assert calls == [modes]
    assert floor == expected


def test_ordinary_on_events_do_not_rebuild_or_prepare(tmp_path, monkeypatch):
    """3,000 actual appends cost one observer/index update apiece, no replay.

    Count work rather than wall time on a shared host. The journal still writes
    every record; only fsync latency is excluded from this CPU regression guard.
    """
    import kernel_checkpoint as checkpoint
    import kernel_read_admission as admission
    import kernel_working_set_shadow as shadow
    import kernel_working_set_snapshot as freezing
    import kernel_working_set_state as state
    from kernel_continuation_executive import ContinuationExecutive
    from kernel_continuation_journal import ContinuationJournal
    from kernel_generated_artifacts import GeneratedArtifactIndex
    from kernel_request_diet import RequestTokenMeter
    from test_working_set_state import capture

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("KERNEL_CONTINUATION_RETENTION_GC", "off")
    monkeypatch.setattr("os.fsync", lambda _fd: None)
    counts = Counter()

    def count_call(owner, name, label):
        original = getattr(owner, name)

        def counted(*args, **kwargs):
            counts[label] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)

    for owner, name, label in (
        (ContinuationJournal, "records", "replay"),
        (ContinuationJournal, "_index_receipt_record", "index"),
        (ContinuationExecutive, "_observe", "executive"),
        (GeneratedArtifactIndex, "observe", "generated"),
        (state.WorkingSetState, "__init__", "state_init"),
        (state.WorkingSetState, "apply_record", "reduce"),
        (state.WorkingSetState, "checkpoint", "state_checkpoint"),
        (state, "rebuild_ledger", "ledger_rebuild"),
        (shadow, "freeze_shadow", "shadow"),
        (freezing, "freeze_preparation", "snapshot"),
        (prepare, "prepare_candidate", "prepare"),
        (admission, "_state", "admission"),
        (checkpoint, "collect", "checkpoint"),
    ):
        count_call(owner, name, label)

    journal = ContinuationJournal(tmp_path, "event-cost", {})
    try:
        generated = GeneratedArtifactIndex(journal.run_id)
        for record in journal.records():
            generated.observe(record)
        journal.generated_artifacts = generated
        executive = ContinuationExecutive(journal)
        meter = RequestTokenMeter(tokenizer=lambda _text: 1)
        meter.attach_journal(journal)
        runtime = SimpleNamespace(journal=journal)
        state.record_capture(runtime, capture())
        working_set = state.state_for(runtime)
        before = working_set.checkpoint()
        sequence = journal._sequence
        counts.clear()

        for batch in range(3):
            for index in range(1_000):
                journal.emit("world_knowledge_event", ordinal=batch * 1_000 + index)
                assert state.state_for(runtime) is working_set
            expected = (batch + 1) * 1_000
            assert counts == {"index": expected, "executive": expected, "generated": expected}
            assert journal._sequence == sequence + expected
            assert executive._sequence == journal._sequence == generated.sequence

        assert working_set.checkpoint() == before
        assert journal.publication_stats().get("observer_failures", 0) == 0
    finally:
        journal.close()
