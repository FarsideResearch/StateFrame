"""Guard legacy replay against mutations to its publicly exposed row cache."""
from __future__ import annotations

import kernel_continuation_journal as journal_module
from kernel_continuation_journal import ContinuationJournal


def test_legacy_subscribe_replays_durable_rows_after_cache_mutation(
        tmp_path, monkeypatch):
    monkeypatch.setenv(journal_module.MEMBOUND_ENV, "legacy")
    journal = ContinuationJournal(
        tmp_path, "replay-cache-integrity", {"origin": "replay-cache-integrity"},
    )
    try:
        assert journal._record_cache_limit is None
        selected = journal.emit(
            "durable_selected", payload={"source": "journal", "ordinal": 1},
        )
        filtered = journal.emit(
            "durable_filtered", payload={"source": "journal", "ordinal": 2},
        )

        cached = journal.cached_records()
        selected_cache_row = cached[selected["sequence"] - 1]
        filtered_cache_row = cached[filtered["sequence"] - 1]
        selected_cache_row["kind"] = "durable_filtered"
        selected_cache_row["payload"]["source"] = "mutated cache"
        filtered_cache_row["kind"] = "durable_selected"
        filtered_cache_row["payload"]["source"] = "mutated cache"

        delivered: list[tuple[str, dict[str, object]]] = []
        journal.subscribe(
            "cache-integrity",
            lambda record: delivered.append((record["kind"], record["payload"])),
            kinds=("durable_selected",),
        )

        assert delivered == [
            ("durable_selected", {"source": "journal", "ordinal": 1}),
        ]
    finally:
        journal.close()
