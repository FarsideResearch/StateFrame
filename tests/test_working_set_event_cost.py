"""Operation-count guards for allocator trials and ordinary durable events."""
import ast
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


def test_small_read_persists_one_journal_event_per_world_observation(tmp_path):
    from detreason.world_knowledge import WorldKnowledgeStore
    from kernel_continuation_journal import ContinuationJournal
    from kernel_retained_context import _bind_world_journal_sink

    world = WorldKnowledgeStore(sandbox_id="source")
    with ContinuationJournal(tmp_path, "source", {}) as journal:
        _bind_world_journal_sink(world, journal)
        world.observe_source(
            path="pricing.py", content="def price():\n    return 1\n",
            revision=0, evidence_ref="read@1",
        )
        assert len(world.events) > 1
        assert tuple(row["event"] for row in journal.records()
                     if row["kind"] == "world_knowledge_event") == world.events


def test_retained_archive_capacity_diagnostic_is_emitted_once_after_saturation():
    from types import SimpleNamespace

    from detreason.source_archive import SourceArchive
    from kernel_retained_context import RetainedContext

    retained = RetainedContext.__new__(RetainedContext)
    retained.archive = SourceArchive("saturated", max_revisions=1)
    retained.context_scope = None
    retained._context_scope_error = None
    retained._archive_capacity_exhausted = False
    events = []
    retained.journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))

    assert retained.observe_source("one.py", b"first", "read@1") is not None
    assert retained.observe_source("two.py", b"second", "read@2") is None
    assert retained.observe_source("three.py", b"third", "read@3") is None
    # Bytes the full archive already holds are still observed.
    assert retained.observe_source("one.py", b"first", "read@4") is not None
    capacity_errors = [fields for kind, fields in events
                       if kind == "retained_context_error" and fields.get("capacity_exhausted")]
    assert len(capacity_errors) == 1


def test_large_read_persists_complete_world_observations_in_durable_batches(tmp_path, monkeypatch):
    import os

    from detreason.world_knowledge import WorldKnowledgeStore
    from kernel_continuation_journal import ContinuationJournal
    from kernel_retained_context import _bind_world_journal_sink

    monkeypatch.delenv("STATEFRAME_TURN_ATOMIC_PUBLICATION", raising=False)
    content = "".join(f"def rule_{index}(value: int) -> int:\n    return value + {index}\n"
                      for index in range(330))
    arguments = {"path": "many_rules.py", "content": content, "revision": 0, "evidence_ref": "read@1"}
    reference = WorldKnowledgeStore(sandbox_id="field")
    reference.observe_source(**arguments)
    live = WorldKnowledgeStore(sandbox_id="field")
    journal_syncs = []
    with ContinuationJournal(tmp_path, "field", {}) as journal:
        fsync = os.fsync

        def count_sync(fd):
            if fd == journal._fd:
                journal_syncs.append(fd)
            fsync(fd)

        monkeypatch.setattr(os, "fsync", count_sync)
        _bind_world_journal_sink(live, journal)
        published = []
        journal.subscribe("source-observations", lambda row: published.append(row["event"]),
                          kinds=("world_knowledge_event",))
        live.observe_source(**arguments)
        assert 1 < len(journal_syncs) < len(reference.events)
        assert tuple(published) == reference.events
        live.observe_source(**{**arguments, "evidence_ref": "read@2"})
        written = tuple(row["event"] for row in journal.records() if row["kind"] == "world_knowledge_event")
        assert written[-1]["event"] == "world_knowledge_reobserved"
        assert written[:-1] == reference.events
        assert written == live.events
        path, head = journal.path, journal.head

    replayed = WorldKnowledgeStore(sandbox_id="field")
    with ContinuationJournal.resume(path, task_id="field", manifest={}, expected_head=head) as resumed:
        replayed.replay(resumed.records())
        assert replayed.facts == live.facts
        replayed.replay(resumed.records())
        assert replayed.facts == live.facts


def test_interrupted_source_batch_recovers_without_turn_publication_switch(tmp_path, monkeypatch):
    import os

    from detreason.world_knowledge import WorldKnowledgeError, WorldKnowledgeStore
    from kernel_continuation_journal import ContinuationJournal, JournalError
    from kernel_retained_context import _bind_world_journal_sink

    monkeypatch.delenv("STATEFRAME_TURN_ATOMIC_PUBLICATION", raising=False)
    world = WorldKnowledgeStore(sandbox_id="field")
    content = "".join(f"def rule_{index}():\n    return {index}\n" for index in range(40))
    with ContinuationJournal(tmp_path, "field", {}) as journal:
        path, head = journal.path, journal.head
        before = path.read_bytes()
        fsync = os.fsync

        def interrupt_sync(fd):
            if fd == journal._fd:
                raise OSError("interrupted journal sync")
            fsync(fd)

        _bind_world_journal_sink(world, journal)
        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", interrupt_sync)
            with pytest.raises(WorldKnowledgeError, match="event sink failed"):
                world.observe_source(path="many_rules.py", content=content,
                                     revision=0, evidence_ref="read@1")
        assert len(path.read_bytes()) > len(before)
        with pytest.raises(JournalError, match="poison"):
            journal.emit("unreachable")

    with ContinuationJournal.resume(path, task_id="field", manifest={}, expected_head=head) as resumed:
        records = list(resumed.records())
        assert records[-2]["record_hash"] == head
        assert records[-1]["kind"] == "continuation_writer_resumed"
        assert records[-1]["previous_head"] == head
        assert path.read_bytes().splitlines(keepends=True)[:-1] == before.splitlines(keepends=True)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_python_fact_coordinates_preserve_unicode_and_multiline_expressions(newline):
    from detreason.world_knowledge_extractors import _SourcePositions

    source = newline.join(("def handle(é: str = 'λ', value=(", "    'one\u2028two',", ")):", "    pass"))
    tree = ast.parse(source)
    positions = _SourcePositions(source)
    defaults = tree.body[0].args.defaults
    assert [positions.segment(node) for node in defaults] == [
        ast.get_source_segment(source, node) for node in defaults
    ]
