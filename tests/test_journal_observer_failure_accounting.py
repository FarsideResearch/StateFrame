"""Observer failure accounting survives batching and journal replay."""
from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import kernel_continuation_journal as journal_module
import pytest
from kernel_continuation_journal import ContinuationJournal

_EVENT = "observer_probe"
_FAILURE = "journal_observer_failed"


def _reject(_row):
    raise RuntimeError("observer rejected row")


def _failures_on_disk(journal):
    return [row for row in map(json.loads, journal.path.read_bytes().splitlines())
            if row["kind"] == _FAILURE]


def _accounted(failures):
    return sum(row.get("dropped_count", 1) for row in failures)


def _write_probes(journal, *, batch):
    events = [(_EVENT, {"index": index}) for index in range(32)]
    if batch:
        return journal.emit_batch(events, atomic=True)
    return [journal.emit(kind, **fields) for kind, fields in events]


def test_all_32_batched_observer_failures_are_accounted_for(tmp_path):
    with ContinuationJournal(tmp_path, "observer-accounting", {}) as journal:
        journal.subscribe("rejecting", _reject, kinds=(_EVENT,))
        _write_probes(journal, batch=True)

        assert _accounted(_failures_on_disk(journal)) == 32


def test_batch_and_single_append_persist_the_same_failure_count(tmp_path):
    counts = {}
    for batch in (True, False):
        with ContinuationJournal(tmp_path, "observer-accounting", {}) as journal:
            journal.subscribe("rejecting", _reject, kinds=(_EVENT,))
            _write_probes(journal, batch=batch)
            counts[batch] = _accounted(_failures_on_disk(journal))

    assert counts[True] == counts[False] == 32


@pytest.mark.parametrize("source", ["batch", "single", "replay"])
def test_overflow_reports_exact_count_and_sequence_bounds(tmp_path, source):
    with ContinuationJournal(tmp_path, "observer-overflow", {}) as journal:
        if source == "single":
            for index in range(32):
                journal.subscribe(f"rejecting-{index}", _reject, kinds=(_EVENT,))
            row = journal.emit(_EVENT)
            failed_sequences = [row["sequence"]] * 32
        else:
            def reject_even(row):
                if row["index"] % 2 == 0:
                    _reject(row)

            events = [(_EVENT, {"index": index}) for index in range(64)]
            if source == "batch":
                journal.subscribe("rejecting", reject_even, kinds=(_EVENT,))
                rows = journal.emit_batch(events, atomic=True)
            else:
                rows = [journal.emit(kind, **fields) for kind, fields in events]
                journal.subscribe("rejecting", reject_even, kinds=(_EVENT,))
            failed_sequences = [row["sequence"] for row in rows[::2]]

        failures = _failures_on_disk(journal)
        details = [row["event_sequence"] for row in failures if "dropped_count" not in row]
        overflow, = [row for row in failures if "dropped_count" in row]

        assert details == failed_sequences[:20]
        assert overflow["first_event_sequence"] == failed_sequences[20]
        assert overflow["last_event_sequence"] == failed_sequences[-1]
        assert overflow["dropped_count"] == 12

        # Counts describe failures, even for sparse ranges or many failures on one row.
        assert _accounted(failures) == len(failed_sequences)


def test_failed_delivery_of_failure_report_is_accounted_without_recursing(tmp_path):
    with ContinuationJournal(tmp_path, "observer-recursion", {}) as journal:
        called = Counter()

        def reject(row):
            called[row["kind"]] += 1
            _reject(row)

        journal.subscribe("rejecting", reject, kinds=(_EVENT, _FAILURE))
        probe = journal.emit(_EVENT)
        detail, overflow = _failures_on_disk(journal)

        assert detail["event_sequence"] == probe["sequence"]
        assert overflow["dropped_count"] == 1
        assert overflow["first_event_sequence"] == overflow["last_event_sequence"] == detail["sequence"]
        assert called == {_EVENT: 1, _FAILURE: 1}
        assert journal.subscriber_cursor("rejecting") == overflow["sequence"]

        # A later subscription sees the ordinary records and can advance past the summary.
        replayed = []
        before_replay = journal.path.read_bytes()
        journal.subscribe("replay", replayed.append, kinds=(_EVENT, _FAILURE))
        assert [row["sequence"] for row in replayed] == [probe["sequence"], detail["sequence"]]
        assert journal.subscriber_cursor("replay") == overflow["sequence"]
        assert journal.path.read_bytes() == before_replay


def test_no_failure_batch_and_single_append_are_byte_identical_after_replay_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(journal_module, "uuid", SimpleNamespace(
        uuid4=lambda: SimpleNamespace(hex="a" * 32)))
    monkeypatch.setattr(journal_module, "time", SimpleNamespace(time_ns=lambda: 123))
    journals = {}
    for batch in (True, False):
        parent = tmp_path / str(batch)
        parent.mkdir()
        with ContinuationJournal(parent, "observer-parity", {}) as journal:
            observed = []
            journal.subscribe("live", observed.append, kinds=(_EVENT,))
            rows = _write_probes(journal, batch=batch)
            assert observed == rows
            path, head = journal.path, journal.head

        before_resume = path.read_bytes()
        with ContinuationJournal.resume(path, task_id="observer-parity", manifest={},
                                        expected_head=head) as resumed:
            before_replay = path.read_bytes()
            assert before_replay.startswith(before_resume)
            replayed = []
            resumed.subscribe("replay", replayed.append, kinds=(_EVENT,))
            assert replayed == rows
            assert path.read_bytes() == before_replay
        journals[batch] = (before_resume, path.read_bytes())

    assert journals[True] == journals[False]
