"""Scope-lock literal deltas pair lines by alignment, never by index across an insertion."""
from __future__ import annotations

import time

import kernel_aha10_honest_loop as honest
import kernel_scope_lock as scope_lock


def _repetitive(count: int) -> str:
    return "\n".join(f"timeout = {i}" for i in range(1, count + 1)) + "\n"


def test_insertion_into_repetitive_file_yields_no_literal_deltas() -> None:
    before = _repetitive(1500)
    after = "retries = 9\n" + before
    assert honest.typed_literal_changes(before, after) == []
    assert scope_lock.literal_changes(before, after) == []


def test_single_literal_change_yields_exactly_one_delta() -> None:
    before = _repetitive(1500)
    after = before.replace("timeout = 1\n", "timeout = 5\n", 1)
    changes = honest.typed_literal_changes(before, after)
    assert len(changes) == 1
    assert changes[0]["line"] == 1
    assert changes[0]["old"] == 1
    assert changes[0]["new"] == 5


def test_insertion_plus_real_literal_change_yields_exactly_one_delta() -> None:
    before = _repetitive(1500)
    after = "retries = 9\n" + before.replace("timeout = 700\n", "timeout = 800\n", 1)
    changes = honest.typed_literal_changes(before, after)
    assert len(changes) == 1
    assert changes[0]["old"] == 700
    assert changes[0]["new"] == 800
    assert changes[0]["line"] == 701
    scoped = scope_lock.literal_changes(before, after)
    assert [c["old"] for c in scoped] == [700]
    assert [c["new"] for c in scoped] == [800]


def test_over_budget_inputs_finish_fast_and_never_index_pair() -> None:
    # 1100 shared "x = 0" lines on both sides: matching pairs ~1.21M, over the pair budget.
    before = "\n".join(["alpha = 1"] + ["x = 0"] * 1100 + ["omega = 2"]) + "\n"
    after = "\n".join(["beta = 3"] + ["x = 0"] * 1100 + ["gamma = 4"]) + "\n"
    started = time.monotonic()
    typed = honest.typed_literal_changes(before, after)
    scoped = scope_lock.literal_changes(before, after)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0
    assert typed == []
    assert scoped == []


def test_unequal_replace_block_yields_no_literal_pairs() -> None:
    before = "x = 1\ny = 2\n"
    after = "x = 5\nz = 9\nw = 3\n"
    assert honest.typed_literal_changes(before, after) == []
    assert scope_lock.literal_changes(before, after) == []
