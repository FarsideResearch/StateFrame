"""Exercise immutable coverage snapshots during commit and replay publication."""
from __future__ import annotations

import threading
import time

import pytest

from _timeouts import scaled
from kernel_continuation_journal import ContinuationJournal
import kernel_stateframe_call_coverage as coverage


def _history_digest():
    return {
        "kind": "action_history_digest",
        "source_text": "not_carried",
        "call_ids": [],
        "row_count": 1,
        "kind_counts": {"tool": 1},
        "tool_counts": {},
        "rows_sha256": "0" * 64,
    }


def _commit(index, call_id, expected_proofs, expected_lock):
    session = coverage.CoverageSession(index, None)
    session.fold(_history_digest(), {call_id})
    digest = session.digest
    assert digest is not None
    proof_hash = coverage._sha(digest)
    with expected_lock:
        expected_proofs[proof_hash] = (digest["call_count"], digest["call_ids_sha256"])
    session.commit()
    # records(rebuild=True) republishes the complete index after its replay scan.
    tuple(index.journal.records(rebuild=True))


def _stage(index, state):
    staged = coverage.CoverageIndex(index.journal, _register=False)
    with staged._lock:
        staged.sequence = state.sequence
        staged.ids = list(state.ids)
        staged.members = set(state.members)
        staged.chains = list(state.chains)
        staged.frontier = list(state.frontier)
        staged.proofs = set(state.proofs)
    return staged


def _assert_consistent(snapshot, expected_proofs, expected_lock, baseline_sequence):
    assert len(snapshot.ids) == len(snapshot.chains) - 1
    assert snapshot.members == frozenset(snapshot.ids)
    # Every distinct proof in this fixture needs a journal row after bootstrap.
    # Other journal rows also advance sequence, so equality is not a contract.
    assert snapshot.sequence >= baseline_sequence + len(snapshot.proofs), (
        "snapshot sequence precedes its coverage events"
    )

    chain = coverage._sha([])
    assert snapshot.chains[0] == chain
    frontier = []
    for offset, call_id in enumerate(snapshot.ids, start=1):
        chain = coverage._sha([chain, call_id])
        assert snapshot.chains[offset] == chain
        frontier = coverage._frontier(frontier, [call_id])
    assert snapshot.frontier == tuple(frontier)

    with expected_lock:
        known_proofs = dict(expected_proofs)
    for proof_hash in snapshot.proofs:
        proof = known_proofs.get(proof_hash)
        assert proof is not None, "snapshot exposes an uncommitted proof"
        count, proof_chain = proof
        assert snapshot.chains[count] == proof_chain
    return snapshot.sequence


def test_snapshots_stay_consistent_during_concurrent_commits_and_rebuilds(tmp_path):
    journal = ContinuationJournal(tmp_path, "coverage-writer-storm", {"origin": "snapshot-storm"})
    try:
        context = coverage.bind_index({}, journal)
        index = coverage.context_index(context)
        assert index is not None
        index.sync()
        baseline_sequence = index.snapshot.sequence
        expected_proofs = {}
        expected_lock = threading.Lock()
        writer_lock = threading.Lock()
        start_writers = threading.Event()
        readers_ready = threading.Barrier(4)
        writers_done = threading.Event()
        writer_active = threading.Event()
        reader_saw_writer = threading.Event()
        errors = []
        errors_lock = threading.Lock()
        reader_samples = []
        reader_overlaps = []

        def record_error(error):
            with errors_lock:
                errors.append(f"{threading.current_thread().name}: {type(error).__name__}: {error}")

        def writer(worker):
            try:
                if not start_writers.wait(scaled(5)):
                    raise TimeoutError("writer start was not released")
                for ordinal in range(10):
                    with writer_lock:
                        writer_active.set()
                        try:
                            if worker == 0 and ordinal == 0 and not reader_saw_writer.wait(scaled(5)):
                                raise TimeoutError("reader did not overlap the writer storm")
                            _commit(index, f"call-{worker}-{ordinal}", expected_proofs, expected_lock)
                        finally:
                            writer_active.clear()
                    time.sleep(0)
            except BaseException as error:
                record_error(error)

        def reader():
            local_samples = 0
            local_overlaps = 0
            previous_sequence = baseline_sequence
            try:
                readers_ready.wait(timeout=scaled(5))
                while not start_writers.is_set():
                    time.sleep(0)
                for _ in range(2_000):
                    active = writer_active.is_set()
                    sequence = _assert_consistent(
                        index.snapshot, expected_proofs, expected_lock, baseline_sequence
                    )
                    assert sequence >= previous_sequence
                    previous_sequence = sequence
                    local_samples += 1
                    local_overlaps += int(active)
                    if active:
                        reader_saw_writer.set()
                while not writers_done.is_set():
                    active = writer_active.is_set()
                    sequence = _assert_consistent(
                        index.snapshot, expected_proofs, expected_lock, baseline_sequence
                    )
                    assert sequence >= previous_sequence
                    previous_sequence = sequence
                    local_samples += 1
                    local_overlaps += int(active)
                    if active:
                        reader_saw_writer.set()
            except BaseException as error:
                record_error(error)
            finally:
                reader_samples.append(local_samples)
                reader_overlaps.append(local_overlaps)

        readers = [threading.Thread(target=reader, name=f"coverage-reader-{n}", daemon=True)
                   for n in range(3)]
        for thread in readers:
            thread.start()
        readers_ready.wait(timeout=scaled(5))

        writers = [threading.Thread(target=writer, args=(n,), name=f"coverage-writer-{n}", daemon=True)
                   for n in range(3)]
        for thread in writers:
            thread.start()
        start_writers.set()
        for thread in writers:
            thread.join(timeout=scaled(30))
        writers_done.set()
        for thread in readers:
            thread.join(timeout=scaled(15))

        assert all(not thread.is_alive() for thread in writers + readers)
        assert errors == []
        assert len(reader_samples) == 3
        assert sum(reader_samples) >= 6_000
        assert sum(reader_overlaps) > 0
        final = index.snapshot
        _assert_consistent(final, expected_proofs, expected_lock, baseline_sequence)
        assert len(final.proofs) == 30
    finally:
        journal.close()


def test_unlocked_snapshot_control_exposes_mixed_replay_publication(tmp_path, monkeypatch):
    journal = ContinuationJournal(tmp_path, "coverage-unlocked-control", {"origin": "snapshot-control"})
    try:
        context = coverage.bind_index({}, journal)
        index = coverage.context_index(context)
        assert index is not None
        index.sync()
        baseline_sequence = index.snapshot.sequence
        expected_proofs = {}
        expected_lock = threading.Lock()
        _commit(index, "call-a", expected_proofs, expected_lock)
        old_state = _stage(index, index.snapshot)
        _commit(index, "call-b", expected_proofs, expected_lock)
        new_state = _stage(index, index.snapshot)
        index._replace_state_from(old_state)

        captured = threading.Event()
        release = threading.Event()
        snapshots = []
        errors = []

        def unlocked_snapshot(instance):
            sequence = instance.sequence
            captured.set()
            if not release.wait(scaled(5)):
                raise TimeoutError("unlocked snapshot control was not released")
            return coverage._CoverageSnapshot(
                sequence=sequence,
                ids=tuple(instance.ids),
                members=frozenset(instance.members),
                chains=tuple(instance.chains),
                frontier=tuple(instance.frontier),
                proofs=frozenset(instance.proofs),
            )

        monkeypatch.setattr(coverage.CoverageIndex, "snapshot", property(unlocked_snapshot))

        def read_snapshot():
            try:
                snapshots.append(index.snapshot)
            except BaseException as error:
                errors.append(error)

        reader = threading.Thread(target=read_snapshot, name="unlocked-snapshot-control", daemon=True)
        reader.start()
        assert captured.wait(scaled(3))
        index._replace_state_from(new_state)
        release.set()
        reader.join(timeout=scaled(5))

        assert not reader.is_alive()
        assert errors == []
        assert len(snapshots) == 1
        with pytest.raises(AssertionError, match="snapshot sequence"):
            _assert_consistent(snapshots[0], expected_proofs, expected_lock, baseline_sequence)
    finally:
        journal.close()
