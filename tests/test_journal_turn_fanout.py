"""S3: live cursor fanout avoids reading back a durable turn, including eviction."""
from __future__ import annotations

import json
import os
from pathlib import Path

import kernel_continuation_journal as journal_module
import pytest
from kernel_continuation_journal import ContinuationJournal, JournalError
from kernel_continuation_runtime import _ContinuationJournalView
from kernel_stateframe_call_coverage import CoverageIndex


def _count_reads(monkeypatch, path):
    reads = []
    original = os.open

    def tracked_open(file, flags, *args, **kwargs):
        if Path(file) == path and flags & os.O_ACCMODE == os.O_RDONLY:
            reads.append(path)
        return original(file, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", tracked_open)
    return reads


def _emit_turn(journal):
    # Exceed the primary 512 KiB cache while staying inside the existing 1 MiB
    # fanout bound. Real request/response captures also evict the primary cache.
    return tuple(journal.emit("turn_fanout_probe", payload={"text": "x" * 8192},
                              step=step) for step in range(90))


@pytest.mark.parametrize("consumer", ["cached", "borrowed", "reducers"])
def test_warm_turn_has_zero_journal_reads(tmp_path, monkeypatch, consumer):
    journal = ContinuationJournal(tmp_path, "turn-fanout", {})
    try:
        view = _ContinuationJournalView(journal)
        coverage = CoverageIndex(journal)
        if consumer == "cached":
            journal.cached_records_since(0)
        elif consumer == "borrowed":
            journal._borrow_records_since(0)
        else:
            view.sync()
            coverage.sync()
            journal.framefit_receipt_records()
        cursor = journal._sequence
        reads = _count_reads(monkeypatch, journal.path)
        emitted = _emit_turn(journal)
        assert journal._record_cache_start > cursor
        if consumer == "cached":
            assert journal.cached_records_since(cursor) == emitted
        elif consumer == "borrowed":
            assert journal._borrow_records_since(cursor) == emitted
        else:
            view.sync()
            coverage.sync()
            journal.framefit_receipt_records()
            assert view.cursor == coverage.sequence == journal._framefit_receipt_cursor
            assert view.cursor == journal._sequence
        assert len(reads) == 0, f"warm turn read journal {len(reads)} times"
    finally:
        journal.close()


def test_resume_cursor_gap_reads_disk_and_preserves_rows(tmp_path, monkeypatch):
    journal = ContinuationJournal(tmp_path, "turn-resume", {})
    cursor = journal._sequence
    journal.cached_records_since(cursor)
    expected = _emit_turn(journal)
    before = journal.path.read_bytes()
    path, head = journal.path, journal.head
    journal.close()
    reads = _count_reads(monkeypatch, path)
    resumed = ContinuationJournal.resume(path, task_id="turn-resume", manifest={},
                                         expected_head=head)
    try:
        assert reads, "resume must validate the durable journal"
        reads.clear()
        actual = resumed.cached_records_since(cursor)
        assert reads, "a lost live cursor must fall back to verified disk records"
        assert actual[:-1] == expected
        assert actual[-1]["kind"] == "continuation_writer_resumed"
        assert resumed.path.read_bytes().startswith(before)
        # Resume becomes warm after its first read, even when the next turn
        # exceeds primary retention again.
        cursor = resumed._sequence
        reads.clear()
        emitted = _emit_turn(resumed)
        assert resumed.cached_records_since(cursor) == emitted
        assert reads == []
    finally:
        resumed.close()


def test_fanout_matches_disk_projections_and_leaves_journal_and_checkpoint_unchanged(
        tmp_path, monkeypatch):
    from kernel_checkpoint import _journal_state

    journal = ContinuationJournal(tmp_path, "fanout-parity", {})
    try:
        live = _ContinuationJournalView(journal)
        disk = _ContinuationJournalView(journal)
        live.sync()
        disk.sync()
        journal.framefit_receipt_records()
        cursor = journal._sequence
        _emit_turn(journal)
        journal.emit("action_outcome_closed", record={
            "continuation_effect_id": "effect-1", "tool": "shell",
            "result": "NO_OP", "step": 1,
        })
        before = journal.path.read_bytes()
        reads = _count_reads(monkeypatch, journal.path)
        live.sync()
        live_receipts = json.dumps(journal.framefit_receipt_context(), sort_keys=True)
        live_rows = json.dumps(journal.cached_records_since(cursor), sort_keys=True)
        assert reads == []
        saved = _journal_state(journal)
        assert "_records_since_cache" not in saved
        journal._clear_records_since_cache()
        assert _journal_state(journal) == saved
        disk.sync()
        journal._reset_framefit_receipt_projection()
        disk_receipts = json.dumps(journal.framefit_receipt_context(), sort_keys=True)
        disk_rows = json.dumps(journal.cached_records_since(cursor), sort_keys=True)
        assert reads
        assert live.closures_by_effect == disk.closures_by_effect
        assert live_receipts == disk_receipts
        assert live_rows == disk_rows
        assert journal.path.read_bytes() == before
    finally:
        journal.close()


def test_fanout_is_detached_from_callers_emit_returns_and_observers(tmp_path):
    journal = ContinuationJournal(tmp_path, "fanout-isolation", {})
    try:
        cursor = journal._sequence
        assert journal.cached_records_since(cursor) == ()
        journal.subscribe("mutating-observer", lambda row: row.update(payload={"n": -1}))
        payload = {"n": 1}
        emitted = journal.emit("isolation_probe", payload=payload)
        payload["n"] = 2
        emitted["payload"]["n"] = 3
        received = journal.cached_records_since(cursor)
        assert received[0]["payload"] == {"n": 1}
        received[0]["payload"]["n"] = 4
        assert journal._borrow_records_since(cursor)[0]["payload"] == {"n": 1}
        assert journal.cached_records_since(cursor)[0]["payload"] == {"n": 1}
    finally:
        journal.close()


def test_fanout_retention_gap_uses_disk_and_remains_bounded(tmp_path, monkeypatch):
    journal = ContinuationJournal(tmp_path, "fanout-overflow", {})
    try:
        cursor = journal._sequence
        journal._borrow_records_since(cursor)
        expected = _emit_turn(journal) + _emit_turn(journal)
        assert journal._records_since_cache_bytes <= journal_module.RECORD_SUFFIX_CACHE_BYTE_LIMIT
        assert len(journal._records_since_cache) <= journal_module.RECORD_SUFFIX_CACHE_LIMIT
        reads = _count_reads(monkeypatch, journal.path)
        assert journal._borrow_records_since(cursor) == expected
        assert reads
    finally:
        journal.close()


@pytest.mark.parametrize("consumer", ["cached_records_since", "_borrow_records_since"])
def test_foreign_writer_cannot_be_hidden_by_warm_fanout(tmp_path, consumer):
    journal = ContinuationJournal(tmp_path, "fanout-foreign", {})
    try:
        cursor = journal._sequence
        getattr(journal, consumer)(cursor)
        journal.emit("probe")
        with journal.path.open("ab") as handle:
            handle.write(b"foreign append\n")
        with pytest.raises(JournalError, match="changed outside its writer"):
            getattr(journal, consumer)(cursor)
    finally:
        journal.close()
