"""Legacy retries validate committed journal rows without copying them."""
from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import pytest

import kernel_continuation_journal as journal_module
from kernel_continuation_journal import ContinuationJournal, JournalError


def _rewrite_ordinal_same_size(journal: ContinuationJournal, sequence: int) -> None:
    offset = journal._record_offsets[sequence - 1]
    end = journal._record_offsets[sequence]
    original_size = journal.path.stat().st_size
    with journal.path.open("r+b") as handle:
        handle.seek(offset)
        original = handle.read(end - offset)
        old = b'"ordinal":1'
        new = b'"ordinal":9'
        assert original.count(old) == 1
        rewritten = original.replace(old, new, 1)
        assert rewritten != original
        assert len(rewritten) == len(original)
        handle.seek(offset)
        assert handle.write(rewritten) == len(rewritten)
        handle.flush()
        os.fsync(handle.fileno())
    assert journal.path.stat().st_size == original_size


def test_legacy_retry_checks_same_size_tampering_before_resuming(
        tmp_path, monkeypatch):
    monkeypatch.setenv(journal_module.MEMBOUND_ENV, "legacy")
    journal = ContinuationJournal(
        tmp_path, "retry-prefix-integrity", {"origin": "retry-prefix-integrity"},
    )
    name = "retry-prefix"
    try:
        rows = [journal.emit("retry_probe", ordinal=ordinal)
                for ordinal in range(1, 5)]
        fail_at = 3
        first_attempt: list[int] = []

        def interrupt_on_third_row(record: dict[str, object]) -> None:
            ordinal = int(record["ordinal"])
            first_attempt.append(ordinal)
            if ordinal == fail_at:
                raise KeyboardInterrupt("injected replay interruption")

        deepcopy_calls = 0
        original_deepcopy = copy.deepcopy

        def count_deepcopy(value, memo=None):
            nonlocal deepcopy_calls
            deepcopy_calls += 1
            return (original_deepcopy(value) if memo is None
                    else original_deepcopy(value, memo))

        indexed_sequences: list[int] = []
        original_resolve = journal._resolve_indexed_line

        def count_indexed_line(line: bytes, sequence: int, blob_cache: dict) -> dict:
            indexed_sequences.append(sequence)
            return original_resolve(line, sequence, blob_cache)

        copy_api = SimpleNamespace(copy=copy.copy, deepcopy=count_deepcopy)
        with monkeypatch.context() as patch:
            patch.setattr(journal_module, "copy", copy_api)
            patch.setattr(journal, "_resolve_indexed_line", count_indexed_line)
            with pytest.raises(KeyboardInterrupt, match="injected replay interruption"):
                journal.subscribe(name, interrupt_on_third_row, kinds=("retry_probe",))

            assert first_attempt == [1, 2, 3]
            assert indexed_sequences == list(range(1, int(rows[2]["sequence"]) + 1))
            expected_cursor = int(rows[1]["sequence"])
            assert journal.subscriber_cursor(name) == expected_cursor
            assert name not in journal._subscribers
            # K attempted deliveries, including the interrupted callback, cost K
            # observer copies; filtered and committed-prefix rows cost none.
            assert deepcopy_calls == len(first_attempt)

            _rewrite_ordinal_same_size(journal, int(rows[0]["sequence"]))
            retry_deliveries: list[int] = []
            before_retry = len(indexed_sequences)
            with pytest.raises(JournalError):
                journal.subscribe(
                    name,
                    lambda record: retry_deliveries.append(int(record["ordinal"])),
                    kinds=("retry_probe",),
                )

            assert indexed_sequences[before_retry:] == list(
                range(1, int(rows[0]["sequence"]) + 1),
            )
            assert retry_deliveries == []
            assert deepcopy_calls == len(first_attempt)
            assert journal.subscriber_cursor(name) == expected_cursor
            assert name not in journal._subscribers
    finally:
        journal.close()
