"""CoverageIndex's legacy containers and amortized update work stay compatible."""
from __future__ import annotations

from dataclasses import replace
from threading import RLock
from types import SimpleNamespace

import kernel_stateframe_call_coverage as coverage
from kernel_wire_identity import wire_run_id


_ELEMENT_COPIES = 0


def _charge(count: int) -> None:
    global _ELEMENT_COPIES
    _ELEMENT_COPIES += count


class _CountingList(list):
    def __iter__(self):
        _charge(len(self))
        return super().__iter__()

    def __add__(self, other):
        _charge(len(self))
        return type(self)(super().__add__(other))

    def copy(self):
        _charge(len(self))
        return type(self)(super().copy())


class _CountingSet(set):
    def __iter__(self):
        _charge(len(self))
        return super().__iter__()

    def __or__(self, other):
        _charge(len(self))
        return type(self)(super().__or__(other))

    def copy(self):
        _charge(len(self))
        return type(self)(super().copy())


class _CountingTuple(tuple):
    def __iter__(self):
        _charge(len(self))
        return super().__iter__()

    def __add__(self, other):
        _charge(len(self))
        return type(self)(super().__add__(other))


class _CountingFrozenSet(frozenset):
    def __iter__(self):
        _charge(len(self))
        return super().__iter__()

    def union(self, *others):
        _charge(len(self))
        return type(self)(super().union(*others))


class _MemoryJournal:
    """Small journal surface used to deliver a batch of already-formed events."""

    run_id = "coverage-index-compat"
    task_id = "coverage-index-compat"
    path = "memory://coverage-index-compat"

    def __init__(self):
        self._lock = RLock()
        self.records: list[dict[str, object]] = []

    def receipt_cursor(self):
        return SimpleNamespace(sequence=len(self.records))

    def _borrow_records_since(self, sequence: int):
        return tuple(row for row in self.records if row["sequence"] > sequence)

    def cached_records_since(self, sequence: int):
        return self._borrow_records_since(sequence)

    def emit(self, kind: str, **payload):
        record = {"sequence": len(self.records) + 1, "kind": kind, **payload}
        self.records.append(record)
        return record


def _history_digest() -> dict[str, object]:
    return {
        "kind": "action_history_digest",
        "source_text": "not_carried",
        "call_ids": [],
        "row_count": 1,
        "kind_counts": {"tool": 1},
        "tool_counts": {},
        "rows_sha256": "0" * 64,
    }


def _event(journal, sequence: int, previous_count: int, previous_chain: str,
           previous_frontier: list[str], added: list[str]):
    chain = previous_chain
    for call_id in added:
        chain = coverage._sha([chain, call_id])
    frontier = coverage._frontier(previous_frontier, added)
    digest = _history_digest()
    digest.update(
        coverage_contract=coverage.CONTRACT,
        call_ids=sorted(frontier),
        coverage_run_id=wire_run_id(journal),
        call_count=previous_count + len(added),
        call_ids_sha256=chain,
    )
    proof = {
        "previous_count": previous_count,
        "previous_chain": previous_chain,
        "added": added,
        "digest": digest,
    }
    return {"sequence": sequence, "kind": coverage.EVENT, "coverage_proof": proof}, chain, frontier


def _assert_legacy_container_types(index) -> None:
    assert type(index.sequence) is int
    assert type(index.ids) is list
    assert type(index.members) is set
    assert type(index.chains) is list
    assert type(index.frontier) is list
    assert type(index.proofs) is set


def test_legacy_coverage_index_properties_match_base_types_and_values():
    empty_journal = _MemoryJournal()
    empty = coverage.CoverageIndex(empty_journal)
    root_chain = coverage._sha([])

    _assert_legacy_container_types(empty)
    assert empty.sequence == 0
    assert empty.ids == []
    assert empty.members == set()
    assert empty.chains == [root_chain]
    assert empty.frontier == []
    assert empty.proofs == set()
    empty.ids.append("append-remains-supported")
    assert empty.ids == ["append-remains-supported"]

    journal = _MemoryJournal()
    index = coverage.CoverageIndex(journal)
    row, _chain, _frontier = _event(
        journal, 1, 0, root_chain, [], ["call-a", "call-b"])
    journal.records.append(row)
    index.sync()

    chain_a = coverage._sha([root_chain, "call-a"])
    chain_b = coverage._sha([chain_a, "call-b"])
    digest = row["coverage_proof"]["digest"]
    _assert_legacy_container_types(index)
    assert index.sequence == 1
    assert index.ids == ["call-a", "call-b"]
    assert index.members == {"call-a", "call-b"}
    assert index.chains == [root_chain, chain_a, chain_b]
    assert index.frontier == ["call-a", "call-b"]
    assert index.proofs == {coverage._sha(digest)}


def _planted_4288fd6_copy_cost(event_count: int) -> int:
    """Exercise the old full-prefix tuple/set replacement used after each proof."""
    global _ELEMENT_COPIES
    _ELEMENT_COPIES = 0
    ids = _CountingTuple()
    members = _CountingFrozenSet()
    chains = _CountingTuple((coverage._sha([]),))
    proofs = _CountingFrozenSet()
    for ordinal in range(event_count):
        value = str(ordinal)
        ids = ids + (value,)
        members = members.union((value,))
        chains = chains + (value,)
        proofs = proofs.union((value,))
    return _ELEMENT_COPIES


def _instrument_accumulated_state(index) -> None:
    """Count copies of fields whose size grows with journal history."""
    if "_snapshot" in vars(index):
        # 4288fd6 stores each growing prefix in immutable tuple/set objects.
        old = vars(index)["_snapshot"]
        index._snapshot = replace(
            old,
            ids=_CountingTuple(old.ids),
            members=_CountingFrozenSet(old.members),
            chains=_CountingTuple(old.chains),
            proofs=_CountingFrozenSet(old.proofs),
        )
        return
    index.ids = _CountingList(index.ids)
    index.members = _CountingSet(index.members)
    index.chains = _CountingList(index.chains)
    index.proofs = _CountingSet(index.proofs)


def test_coverage_event_updates_copy_only_linear_accumulated_state():
    global _ELEMENT_COPIES
    event_count = 128
    # The wire frontier is intentionally bounded to 64 entries. This budget
    # allows that fixed-size work per event while rejecting history-sized copies.
    copy_budget = event_count * (coverage.FRONTIER_COUNT + 16)
    planted_cost = _planted_4288fd6_copy_cost(event_count)
    assert planted_cost > copy_budget, "the counter must reject the 4288fd6 copy-per-proof control"

    journal = _MemoryJournal()
    index = coverage.CoverageIndex(journal)
    previous_chain = coverage._sha([])
    previous_frontier: list[str] = []
    for sequence in range(1, event_count + 1):
        row, previous_chain, previous_frontier = _event(
            journal, sequence, sequence - 1, previous_chain, previous_frontier,
            [f"call-{sequence:03d}"])
        journal.records.append(row)

    _instrument_accumulated_state(index)
    _ELEMENT_COPIES = 0
    index.sync()

    assert _ELEMENT_COPIES <= copy_budget, (
        f"coverage replay copied {_ELEMENT_COPIES} accumulated elements; "
        f"linear budget is {copy_budget} for {event_count} events"
    )
    assert index.sequence == event_count
    assert len(index.ids) == event_count
    assert len(index.members) == event_count
    assert len(index.chains) == event_count + 1
    assert len(index.proofs) == event_count
