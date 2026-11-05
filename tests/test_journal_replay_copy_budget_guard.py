"""Guard subscription replay copies against base 30ce267 behavior.

Base 30ce267's legacy ``records()`` path yielded rows without a copy, so K
matching deliveries cost exactly K observer-isolation copies. In bounded mode,
``records()`` made N replay-row copies, one projection-shadow copy, and K
observer-isolation copies: N + K + 1. The delivery path must stay within those
bounds while excluding filtered rows before any delivery copy.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

import kernel_continuation_journal as journal_module
from kernel_continuation_journal import ContinuationJournal


@pytest.mark.parametrize("mode", ("legacy", "bounded"))
def test_subscribe_replay_copy_budget_respects_kind_filter(tmp_path, monkeypatch, mode):
    monkeypatch.setenv(journal_module.MEMBOUND_ENV, mode)
    journal = ContinuationJournal(
        tmp_path, f"replay-copy-budget-{mode}", {"origin": "replay-copy-budget"},
    )
    selected_ordinals = (1, 4, 7)
    try:
        assert (journal._record_cache_limit is None) == (mode == "legacy")
        for ordinal in range(8):
            kind = "selected_row" if ordinal in selected_ordinals else "filtered_row"
            journal.emit(kind, ordinal=ordinal, payload={"nested": [ordinal]})

        replay_rows = journal._sequence
        delivered: list[int] = []
        deepcopy_calls = 0
        original_deepcopy = copy.deepcopy

        def count_deepcopy(value, memo=None):
            nonlocal deepcopy_calls
            deepcopy_calls += 1
            return (original_deepcopy(value) if memo is None
                    else original_deepcopy(value, memo))

        # Proxy the journal module's copy helpers so recursive calls made inside
        # copy.deepcopy itself do not inflate the count.
        copy_api = SimpleNamespace(copy=copy.copy, deepcopy=count_deepcopy)
        with monkeypatch.context() as patch:
            patch.setattr(journal_module, "copy", copy_api)
            journal.subscribe(
                "copy-budget", lambda record: delivered.append(record["ordinal"]),
                kinds=("selected_row",),
            )

        assert delivered == list(selected_ordinals)
        if mode == "legacy":
            assert deepcopy_calls == len(selected_ordinals)
        else:
            assert deepcopy_calls <= replay_rows + len(selected_ordinals) + 1
    finally:
        journal.close()
