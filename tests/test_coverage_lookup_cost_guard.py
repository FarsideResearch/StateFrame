"""Guard the prefix-only work performed by historical coverage lookups."""
from __future__ import annotations

from types import SimpleNamespace

import kernel_stateframe_call_coverage as coverage


class _EmptyJournal:
    run_id = "coverage-lookup-cost"
    path = "memory://coverage-lookup-cost"

    def receipt_cursor(self):
        return SimpleNamespace(sequence=0)

    def _borrow_records_since(self, sequence: int):
        assert sequence == 0
        return ()


class _SliceRecordingList(list[str]):
    def __init__(self, values: list[str]):
        super().__init__(values)
        self.requested_slice_lengths: list[int] = []

    def __getitem__(self, key: int | slice):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            self.requested_slice_lengths.append(len(range(start, stop, step)))
        return super().__getitem__(key)


def test_covered_ids_copies_only_the_proved_prefix(monkeypatch):
    def reject_snapshot(_index):
        raise AssertionError("covered_ids must not copy the full coverage snapshot")

    # The f519d15 `state = self.snapshot` control fails immediately here.
    monkeypatch.setattr(
        coverage.CoverageIndex, "snapshot", property(reject_snapshot), raising=False
    )

    call_count = 512
    covered_count = 3
    ids = [f"call-{ordinal}" for ordinal in range(call_count)]
    chains = [coverage._sha([])]
    for call_id in ids:
        chains.append(coverage._sha([chains[-1], call_id]))

    digest = {
        "call_count": covered_count,
        "call_ids_sha256": chains[covered_count],
    }
    index = coverage.CoverageIndex(_EmptyJournal())
    tracked_ids = _SliceRecordingList(ids)
    index.ids = tracked_ids
    index.members = set(ids)
    index.chains = chains
    index.proofs = {coverage._sha(digest)}

    covered = index.covered_ids(digest)

    assert covered == set(ids[:covered_count])
    assert tracked_ids.requested_slice_lengths == [covered_count]
