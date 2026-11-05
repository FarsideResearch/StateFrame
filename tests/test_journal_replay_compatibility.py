"""Compatibility checks for retry filters and direct coverage-index construction."""
from __future__ import annotations

import pytest

from kernel_continuation_journal import ContinuationJournal


def test_changed_filter_retries_from_the_start_of_history(tmp_path):
    journal = ContinuationJournal(
        tmp_path, "changed-filter-retry", {"origin": "changed-filter-retry"},
    )
    try:
        first_b = journal.emit("filter_b", ordinal=1)
        journal.emit("filter_a", ordinal=2)
        second_b = journal.emit("filter_b", ordinal=3)
        journal.emit("filter_a", ordinal=4)

        def fail_on_second_a(record: dict[str, object]) -> None:
            if record.get("kind") == "filter_a" and record.get("ordinal") == 4:
                raise KeyboardInterrupt("retry from the next row")

        with pytest.raises(KeyboardInterrupt, match="retry from the next row"):
            journal.subscribe("filter-retry", fail_on_second_a, kinds=("filter_a",))

        delivered_b: list[int] = []

        def observe_b(record: dict[str, object]) -> None:
            if record.get("kind") == "filter_b":
                delivered_b.append(int(record["sequence"]))

        journal.subscribe("filter-retry", observe_b, kinds=("filter_b",))

        assert delivered_b == [first_b["sequence"], second_b["sequence"]]
    finally:
        journal.close()


def test_direct_coverage_index_construction_registers_context_lookup(tmp_path):
    from kernel_stateframe_call_coverage import CONTEXT_KEY, CoverageIndex, context_index

    journal = ContinuationJournal(
        tmp_path, "coverage-index-construction-compat", {"origin": "coverage-index-compat"},
    )
    try:
        index = CoverageIndex(journal)
        context = {CONTEXT_KEY: index.context_handle}

        assert context_index(context) is index
    finally:
        journal.close()
