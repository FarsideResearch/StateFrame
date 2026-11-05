"""Frozen contracts for replay publication, retry, coverage snapshots, and handoff.

The fail-first publication tests describe the manager-approved two-phase replay
contract. The semantic-refusal and live-handoff tests pin behavior that must
remain identical to the 15de58e base.
"""
from __future__ import annotations

import copy
import gc
import subprocess
import sys
import threading
import time
import tracemalloc
from collections.abc import Mapping

import pytest

from kernel_continuation_journal import ContinuationJournal, JournalError


_REPLACE_JOURNAL = (
    "import os, pathlib, sys\n"
    "path = pathlib.Path(sys.argv[1])\n"
    "replacement = path.with_name(f'.{path.name}.external-{os.getpid()}')\n"
    "replacement.write_bytes(path.read_bytes())\n"
    "os.replace(replacement, path)\n"
)


def _external_replace(path) -> None:
    """Replace the journal inode from a real child process without its flock."""
    subprocess.run(
        [sys.executable, "-c", _REPLACE_JOURNAL, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _journal(tmp_path, task_id: str) -> ContinuationJournal:
    return ContinuationJournal(tmp_path, task_id, {"origin": "journal-replay-publication"})


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


def _commit_coverage(index, call_id: str) -> dict[str, object]:
    from kernel_stateframe_call_coverage import CoverageSession

    session = CoverageSession(index, None)
    session.fold(_history_digest(), {call_id})
    session.commit()
    assert session.digest is not None
    return session.digest


def _clone_coverage_index(index):
    # copy.copy gives a detached owner object without registering another context.
    # The production replacement method then copies/publishes its immutable state.
    staged = copy.copy(index)
    if getattr(staged, "_snapshot", None) is None and getattr(staged, "_state", None) is None:
        for name in ("ids", "members", "chains", "frontier", "proofs"):
            setattr(staged, name, copy.deepcopy(getattr(index, name)))
    return staged


def _adopt_coverage_state(index, staged) -> None:
    replace = getattr(index, "_replace_state_from", None)
    if callable(replace):
        replace(staged)
        return
    # 15de58e has no state-adoption method. This compatibility path deliberately
    # exercises its public mutable state fields and is the planted old behavior.
    for name in ("sequence", "ids", "members", "chains", "frontier", "proofs"):
        setattr(index, name, copy.deepcopy(getattr(staged, name)))


def _coverage_signature(index_or_state) -> tuple[object, ...]:
    state = getattr(index_or_state, "_snapshot", None)
    if state is None:
        state = getattr(index_or_state, "_state", None)
    if isinstance(state, Mapping):
        def get(name):
            return state[name]
    elif state is not None:
        def get(name):
            return getattr(state, name)
    else:
        def get(name):
            return getattr(index_or_state, name)
    return (
        get("sequence"),
        tuple(get("ids")),
        tuple(get("chains")),
        frozenset(get("proofs")),
        frozenset(get("members")),
    )


def test_reentrant_observer_reads_only_committed_replay_state(tmp_path):
    journal = _journal(tmp_path, "reentrant-replay")
    try:
        expected_rows = tuple(journal.records())
        expected_sequences = tuple(row["sequence"] for row in expected_rows)
        name = "reentrant-projection"
        before_stats = journal.publication_stats()
        before_cursor = journal.subscriber_cursor(name)
        observations: list[tuple[int, int, dict[str, int], bool, object]] = []

        def observe(record):
            delivered_count = len(observations)
            expected_stats = dict(before_stats)
            expected_stats["replayed"] += delivered_count
            expected_stats["deep_copies"] += delivered_count
            cursor = journal.subscriber_cursor(name)
            stats = journal.publication_stats()
            live = name in journal._subscribers
            try:
                nested_sequences = tuple(row["sequence"] for row in journal.records())
            except BaseException as error:
                nested_sequences = error
            observations.append((int(record["sequence"]), cursor, stats, live, nested_sequences))

        journal.subscribe(name, observe)

        assert len(observations) == len(expected_rows)
        for delivered_count, (row, observed) in enumerate(zip(expected_rows, observations)):
            delivered_sequence, cursor, stats, live, nested_sequences = observed
            expected_cursor = (before_cursor if delivered_count == 0
                               else expected_rows[delivered_count - 1]["sequence"])
            expected_stats = dict(before_stats)
            expected_stats["replayed"] += delivered_count
            expected_stats["deep_copies"] += delivered_count
            assert delivered_sequence == row["sequence"]
            assert cursor == expected_cursor, f"partial cursor visible while delivering {row['sequence']}"
            assert stats == expected_stats, f"partial publication counters visible at row {row['sequence']}"
            assert live is False, f"subscription became live before replay completed at row {row['sequence']}"
            assert nested_sequences == expected_sequences
        assert journal.subscriber_cursor(name) == journal._sequence
        assert name in journal._subscribers
    finally:
        journal.close()


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_baseexception_replay_retry_resumes_at_first_uncommitted_row(
        tmp_path, exception_type):
    journal = _journal(tmp_path, f"retry-{exception_type.__name__.lower()}")
    name = "retry-projection"
    try:
        rows = []
        for ordinal in range(1, 6):
            rows.append(journal.emit("retry_probe", ordinal=ordinal))
        fail_at = 3
        successful_first_attempt: list[int] = []
        failed_once = False

        def interrupt_once(record):
            nonlocal failed_once
            ordinal = int(record["ordinal"])
            if ordinal == fail_at and not failed_once:
                failed_once = True
                # Raise before the callback's externally visible effect. The
                # failed delivery must remain the first row on retry.
                raise exception_type("injected replay interruption")
            successful_first_attempt.append(ordinal)

        with pytest.raises(exception_type, match="injected replay interruption"):
            journal.subscribe(name, interrupt_once, kinds=("retry_probe",))

        expected_committed_cursor = rows[fail_at - 2]["sequence"]
        assert journal.subscriber_cursor(name) == expected_committed_cursor
        assert name not in journal._subscribers

        retry_deliveries: list[int] = []
        journal.subscribe(
            name,
            lambda record: retry_deliveries.append(int(record["ordinal"])),
            kinds=("retry_probe",),
        )

        expected_retry = list(range(fail_at, len(rows) + 1))
        assert retry_deliveries == expected_retry
        assert successful_first_attempt + retry_deliveries == list(range(1, len(rows) + 1))
        assert journal.subscriber_cursor(name) == journal._sequence
    finally:
        journal.close()


def test_callback_triggered_external_replace_stops_replay_and_poisons_writer(tmp_path):
    journal = _journal(tmp_path, "callback-owner-drift")
    published: list[int] = []
    try:
        for ordinal in range(1, 4):
            journal.emit("drift_probe", ordinal=ordinal)

        def replace_after_first(record):
            published.append(int(record["ordinal"]))
            if len(published) == 1:
                _external_replace(journal.path)

        with pytest.raises(JournalError, match="journal changed outside its writer"):
            journal.subscribe("drift-projection", replace_after_first, kinds=("drift_probe",))

        # A real process changed the inode after row one; no later staged row may
        # escape, and the failed ownership check must poison this writer.
        assert published == [1]
        assert journal._poisoned is True
        assert "drift-projection" not in journal._subscribers
    finally:
        journal.close()


@pytest.mark.xfail(
    strict=True,
    reason="pre-existing 0.4.22 behavior: CoverageIndex has no atomic multi-field snapshot; tracked as a follow-up, not part of the journal lease fix",
)
def test_coverage_snapshots_are_atomic_during_concurrent_reads(tmp_path):
    import kernel_stateframe_call_coverage as coverage

    journal = _journal(tmp_path, "coverage-snapshot-race")
    try:
        context = coverage.bind_index({}, journal)
        index = coverage.context_index(context)
        assert index is not None

        digest_a = _commit_coverage(index, "call-a")
        staged_a = _clone_coverage_index(index)
        signature_a = _coverage_signature(staged_a)
        digest_b = _commit_coverage(index, "call-b")
        assert index.covered_ids(digest_b) == {"call-a", "call-b"}
        staged_b = _clone_coverage_index(index)
        signature_b = _coverage_signature(staged_b)
        assert signature_a != signature_b

        # Plant a deterministic read/swap interleave. A reader that fetches
        # sequence, ids, chains, proofs, and members separately sees old sequence
        # followed by new fields; an immutable snapshot reference stays coherent.
        captured = threading.Event()
        release = threading.Event()
        snapshot_result: list[tuple[object, ...]] = []
        snapshot_errors: list[BaseException] = []

        def capture_during_replace():
            try:
                visible = coverage.context_index(context)
                assert visible is not None
                snapshot = getattr(visible, "_snapshot", None)
                if snapshot is None:
                    snapshot = getattr(visible, "_state", None)
                if snapshot is not None:
                    captured.set()
                    if not release.wait(5):
                        raise TimeoutError("snapshot reader was not released")
                    snapshot_result.append(_coverage_signature(snapshot))
                    return

                sequence = visible.sequence
                captured.set()
                if not release.wait(5):
                    raise TimeoutError("legacy snapshot reader was not released")
                # The sleeps make the multi-field read model the concurrent consumer
                # window called out in the review; the barrier guarantees a swap here.
                time.sleep(0)
                ids = tuple(visible.ids)
                time.sleep(0)
                chains = tuple(visible.chains)
                time.sleep(0)
                proofs = frozenset(visible.proofs)
                time.sleep(0)
                members = frozenset(visible.members)
                snapshot_result.append((sequence, ids, chains, proofs, members))
            except BaseException as error:
                snapshot_errors.append(error)

        _adopt_coverage_state(index, _clone_coverage_index(staged_a))
        reader = threading.Thread(target=capture_during_replace, name="coverage-snapshot-reader")
        reader.start()
        assert captured.wait(3)
        _adopt_coverage_state(index, _clone_coverage_index(staged_b))
        release.set()
        reader.join(5)
        assert not reader.is_alive()
        assert not snapshot_errors
        assert snapshot_result == [signature_a]

        # Hammer context_index()/covered_ids() while repeatedly swapping the
        # existing index. Digest A is valid in both complete snapshots.
        stop = threading.Event()
        hammer_ready = threading.Event()
        problems: list[BaseException | str] = []

        def hammer_coverage_reads():
            hammer_ready.set()
            while not stop.is_set():
                visible = coverage.context_index(context)
                if visible is None:
                    problems.append("coverage context disappeared")
                    return
                try:
                    snapshot = getattr(visible, "_snapshot", None)
                    if snapshot is None:
                        snapshot = getattr(visible, "_state", None)
                    if snapshot is not None:
                        signature = _coverage_signature(snapshot)
                    else:
                        sequence = visible.sequence
                        time.sleep(0)
                        signature = (
                            sequence,
                            tuple(visible.ids),
                            tuple(visible.chains),
                            frozenset(visible.proofs),
                            frozenset(visible.members),
                        )
                    covered = visible.covered_ids(digest_a)
                except BaseException as error:
                    problems.append(error)
                    return
                if signature not in {signature_a, signature_b}:
                    problems.append(f"mixed coverage snapshot: {signature!r}")
                    return
                if covered != {"call-a"}:
                    problems.append(f"unexpected covered ids: {covered!r}")
                    return
                time.sleep(0)

        hammer = threading.Thread(target=hammer_coverage_reads, name="coverage-read-hammer")
        hammer.start()
        assert hammer_ready.wait(3)
        for iteration in range(100):
            source = staged_a if iteration % 2 else staged_b
            _adopt_coverage_state(index, _clone_coverage_index(source))
            time.sleep(0)
        stop.set()
        hammer.join(5)
        assert not hammer.is_alive()
        assert problems == []
    finally:
        journal.close()


@pytest.mark.xfail(strict=True, reason="pre-existing 0.4.22 behavior: bind_index() registers a new CoverageIndex before "
                   "attaching it, without the journal lock, so a concurrent thread can resolve it mid-scan; a bind_index "
                   "thread race, not the journal lease bug; tracked as a follow-up (FOLLOWUP_coverage_concurrency)")
def test_bind_index_is_not_exposed_during_replay_scan(tmp_path, monkeypatch):
    import kernel_stateframe_call_coverage as coverage

    journal = _journal(tmp_path, "coverage-bind-during-scan")
    scan_paused = threading.Event()
    scan_release = threading.Event()
    bind_started = threading.Event()
    bind_constructed = threading.Event()
    bind_constructor_release = threading.Event()
    bind_finished = threading.Event()
    scan_errors: list[BaseException] = []
    bind_errors: list[BaseException] = []
    bind_handle = coverage._sha([
        "coverage-index", journal.run_id, str(journal.path),
    ])
    context = {coverage.CONTEXT_KEY: bind_handle}
    original_records = journal.records
    original_init = coverage.CoverageIndex.__init__

    def pause_scan(*, rebuild=True):
        for position, row in enumerate(original_records(rebuild=rebuild)):
            yield row
            if position == 0:
                scan_paused.set()
                if not scan_release.wait(5):
                    raise TimeoutError("coverage scan was not released")

    def observe_bind_constructor(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if threading.current_thread().name == "coverage-bind-writer":
            bind_constructed.set()
            if not bind_constructor_release.wait(5):
                raise TimeoutError("coverage bind constructor was not released")

    monkeypatch.setattr(journal, "records", pause_scan)
    monkeypatch.setattr(coverage.CoverageIndex, "__init__", observe_bind_constructor)

    def scan_records():
        try:
            for _row in journal.records():
                pass
        except BaseException as error:
            scan_errors.append(error)

    def bind_during_scan():
        bind_started.set()
        try:
            coverage.bind_index(context, journal)
        except BaseException as error:
            bind_errors.append(error)
        finally:
            bind_finished.set()

    scan_thread = threading.Thread(target=scan_records, name="coverage-scan")
    bind_thread = threading.Thread(target=bind_during_scan, name="coverage-bind-writer")
    try:
        scan_thread.start()
        assert scan_paused.wait(3)
        bind_thread.start()
        assert bind_started.wait(3)
        # The constructor pauses after returning to bind_index. A correct binder
        # holds publication behind the journal scan and keeps the registry empty.
        time.sleep(0.05)
        exposed_during_scan = coverage.context_index(context)
        bind_constructor_release.set()
        time.sleep(0.05)
        exposed_after_constructor = coverage.context_index(context)
    finally:
        bind_constructor_release.set()
        scan_release.set()
        if scan_thread.is_alive():
            scan_thread.join(5)
        if bind_thread.is_alive():
            bind_thread.join(5)
        journal.close()

    assert not scan_thread.is_alive()
    assert not bind_thread.is_alive()
    assert scan_errors == []
    assert bind_errors == []
    assert bind_finished.is_set()
    assert exposed_during_scan is None
    assert exposed_after_constructor is None
    bound = coverage.context_index(context)
    assert bound is not None
    assert getattr(journal, "_framefit_call_coverage_index", bound) is bound


def _deep_size(value: object, seen: set[int] | None = None) -> int:
    """Approximate the Python heap of a JSON record for a derived cache bound."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(_deep_size(key, seen) + _deep_size(item, seen)
                          for key, item in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_deep_size(item, seen) for item in value)
    return size


def test_subscribe_staging_and_tracemalloc_peak_follow_configured_cache_bound(
        tmp_path, monkeypatch):
    import kernel_continuation_journal as journal_module

    monkeypatch.setenv(journal_module.MEMBOUND_ENV, "bounded")
    journal = _journal(tmp_path, "bounded-replay-staging")
    try:
        record_limit = journal._record_cache_limit
        assert record_limit == journal_module.RECORD_CACHE_LIMIT
        assert isinstance(record_limit, int) and record_limit > 0
        for ordinal in range(record_limit * 3):
            journal.emit("memory_probe", ordinal=ordinal)

        largest_record_heap = max(_deep_size(row) for row in journal.records())
        assert journal._sequence > record_limit * 3

        class Tracker:
            active = 0
            peak = 0
            active_traced_bytes = 0
            peak_traced_bytes = 0
            largest_row_traced_bytes = 0

        tracker = Tracker()

        class TrackedRow(dict):
            def __init__(self, row):
                before = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
                super().__init__(row)
                self._tracker = tracker
                after = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else before
                self._traced_bytes = max(0, after - before)
                tracker.active += 1
                tracker.peak = max(tracker.peak, tracker.active)
                tracker.active_traced_bytes += self._traced_bytes
                tracker.peak_traced_bytes = max(
                    tracker.peak_traced_bytes, tracker.active_traced_bytes,
                )
                tracker.largest_row_traced_bytes = max(
                    tracker.largest_row_traced_bytes, self._traced_bytes,
                )

            def __deepcopy__(self, memo):
                return copy.deepcopy(dict(self), memo)

            def __del__(self):
                self._tracker.active -= 1
                self._tracker.active_traced_bytes -= self._traced_bytes

        def track(row):
            return row if isinstance(row, TrackedRow) else TrackedRow(row)

        windows: list[int] = []
        range_depth = threading.local()
        original_records = journal.records
        original_range = journal._record_range
        original_indexed = journal._indexed_records
        original_cached_since = journal.cached_records_since
        original_borrow_since = journal._borrow_records_since

        def tracked_records(*, rebuild=True):
            for row in original_records(rebuild=rebuild):
                windows.append(1)
                yield track(row)

        def tracked_range(start, stop, *args, **kwargs):
            windows.append(max(0, stop - start))
            depth = getattr(range_depth, "value", 0)
            range_depth.value = depth + 1
            try:
                for row in original_range(start, stop, *args, **kwargs):
                    yield track(row)
            finally:
                range_depth.value = depth

        def tracked_indexed(first, last, *args, **kwargs):
            windows.append(max(0, last - first + 1))
            inside_range = getattr(range_depth, "value", 0) > 0
            for row in original_indexed(first, last, *args, **kwargs):
                yield row if inside_range else track(row)

        def tracked_cached_since(sequence, *args, **kwargs):
            rows = original_cached_since(sequence, *args, **kwargs)
            windows.append(len(rows))
            return tuple(track(row) for row in rows)

        def tracked_borrow_since(sequence, *args, **kwargs):
            rows = original_borrow_since(sequence, *args, **kwargs)
            windows.append(len(rows))
            return tuple(track(row) for row in rows)

        monkeypatch.setattr(journal, "records", tracked_records)
        monkeypatch.setattr(journal, "_record_range", tracked_range)
        monkeypatch.setattr(journal, "_indexed_records", tracked_indexed)
        monkeypatch.setattr(journal, "cached_records_since", tracked_cached_since)
        monkeypatch.setattr(journal, "_borrow_records_since", tracked_borrow_since)

        gc.collect()
        tracing_before = tracemalloc.is_tracing()
        if not tracing_before:
            tracemalloc.start()
        try:
            scan_baseline = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            for _row in journal.records():
                pass
            _, scan_peak = tracemalloc.get_traced_memory()
            scan_growth = max(0, scan_peak - scan_baseline)

            subscribe_baseline = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            delivered_count = 0

            def observe(row):
                nonlocal delivered_count
                delivered_count += 1

            journal.subscribe(
                "memory-projection",
                observe,
            )
            _, subscribe_peak = tracemalloc.get_traced_memory()
            subscribe_growth = max(0, subscribe_peak - subscribe_baseline)
        finally:
            if not tracing_before:
                tracemalloc.stop()

        # Bound read count derives from the engine's configured record-cache
        # limit. The byte ceiling is also engine configured; it can only tighten
        # the amount retained by the cache, not widen the count bound.
        assert tracker.peak <= record_limit, (
            f"peak staged records {tracker.peak} exceeded configured cache limit {record_limit}"
        )
        assert windows, "replay delivery did not use bounded journal reads"
        assert max(windows) <= record_limit, (
            f"replay read window {max(windows)} exceeded configured cache limit {record_limit}"
        )
        traced_staging_bound = record_limit * tracker.largest_row_traced_bytes
        assert tracker.peak_traced_bytes <= traced_staging_bound, (
            "tracemalloc-recorded staged rows exceeded the configured cache count: "
            f"peak={tracker.peak_traced_bytes}, bound={traced_staging_bound}, "
            f"rows={tracker.peak}, row_bytes={tracker.largest_row_traced_bytes}"
        )
        bounded_page_bytes = record_limit * largest_record_heap + sys.getsizeof([None] * record_limit)
        assert subscribe_growth <= scan_growth + bounded_page_bytes, (
            "tracemalloc replay peak exceeded one validated scan plus one configured cache page: "
            f"peak={subscribe_growth}, scan={scan_growth}, page_bound={bounded_page_bytes}, "
            f"record_cache_byte_limit={journal_module.RECORD_CACHE_BYTE_LIMIT}"
        )
        assert delivered_count == journal._sequence
    finally:
        journal.close()


def test_parent_meter_intact_lease_keeps_base_semantic_refusal(tmp_path):
    from kernel_pilot import ParentMeter

    allocation = ParentMeter(attempts=1, steps=None, enforce=False)
    manifest = {"parent_budget": {"limits": dict(allocation.limits), "enforced": False}}
    journal = ContinuationJournal(tmp_path, "parent-meter-intact-refusal", manifest)
    try:
        journal.emit("attempt_terminated", reason="semantic refusal")
        with pytest.raises(ValueError) as raised:
            ParentMeter.restore(journal, attempts=1, steps=None, enforce=False)
        assert type(raised.value) is ValueError
        assert str(raised.value) == "terminated attempt cannot recover execution budget"
        assert journal._poisoned is False
    finally:
        journal.close()


@pytest.mark.parametrize(
    ("phase", "expected_text"),
    [
        ("first", "external effect intent schema is invalid"),
        ("second", "external effect outcome has no preceding dispatch"),
    ],
)
def test_external_effect_intact_lease_keeps_base_semantic_refusal_in_both_passes(
        tmp_path, monkeypatch, phase, expected_text):
    from kernel_unknown_outcomes import (
        INTENT_EVENT,
        PRESENT,
        ExternalEffectAdapter,
    )

    journal = _journal(tmp_path, f"external-effect-{phase}-pass-refusal")
    try:
        adapter = ExternalEffectAdapter(journal)
        if phase == "first":
            journal.emit(
                INTENT_EVENT,
                external_outcome_schema="invalid-schema",
                external_effect_id="effect-with-invalid-schema",
                idempotency_key=["invalid-schema"],
                destination="mail",
                payload={},
                payload_hash="sha256:" + "0" * 64,
            )
        else:
            adapter.begin("second-pass-key", {"value": 1}, destination_name="mail")
            _effect_id, intent = adapter._intent("second-pass-key")
            adapter._record_outcome(intent, outcome=PRESENT, source="lookup")

        original_records = journal.records
        scans = 0

        def counted_records(*, rebuild=True):
            nonlocal scans
            scans += 1
            return original_records(rebuild=rebuild)

        monkeypatch.setattr(journal, "records", counted_records)
        with pytest.raises(ValueError) as raised:
            adapter._replay()
        assert type(raised.value) is ValueError
        assert str(raised.value) == expected_text
        assert scans == (1 if phase == "first" else 2)
        assert journal._poisoned is False
    finally:
        journal.close()


def test_successful_replay_hands_off_once_with_kind_filter_cursor_gaps(tmp_path):
    journal = _journal(tmp_path, "live-handoff-filtered-replay")
    delivered: list[tuple[int, int]] = []
    name = "filtered-handoff"
    try:
        first = journal.emit("handoff_keep", ordinal=1)
        skipped = journal.emit("handoff_skip", ordinal=2)
        third = journal.emit("handoff_keep", ordinal=3)
        assert skipped["sequence"] + 1 == third["sequence"]
        before_stats = journal.publication_stats()

        def observe(record):
            delivered.append((int(record["sequence"]), int(record["ordinal"])))

        journal.subscribe(name, observe, kinds=("handoff_keep",))
        assert delivered == [(first["sequence"], 1), (third["sequence"], 3)]
        assert journal.subscriber_cursor(name) == journal._sequence
        replay_stats = journal.publication_stats()
        assert replay_stats["replayed"] - before_stats["replayed"] == 2
        assert replay_stats["deep_copies"] - before_stats["deep_copies"] == 2
        assert replay_stats["filtered"] - before_stats["filtered"] == 0

        ignored = journal.emit("handoff_skip", ordinal=4)
        assert delivered == [(first["sequence"], 1), (third["sequence"], 3)]
        assert journal.subscriber_cursor(name) == ignored["sequence"]
        assert journal.publication_stats()["filtered"] == replay_stats["filtered"] + 1

        selected = journal.emit("handoff_keep", ordinal=5)
        assert delivered == [
            (first["sequence"], 1),
            (third["sequence"], 3),
            (selected["sequence"], 5),
        ]
        assert journal.subscriber_cursor(name) == selected["sequence"]
        final_stats = journal.publication_stats()
        assert final_stats["published"] - replay_stats["published"] == 1
        assert final_stats["deep_copies"] - replay_stats["deep_copies"] == 1
    finally:
        journal.close()


# Planted controls for the parity tests above: each breaks the behavior its test pins and proves the test fails.

def test_parity_control_parent_meter_refusal_type_change_fails(tmp_path, monkeypatch):
    from kernel_pilot import ParentMeter

    def wrong_type(*_args, **_kwargs):
        raise TypeError("terminated attempt cannot recover execution budget")

    monkeypatch.setattr(ParentMeter, "restore", staticmethod(wrong_type))
    with pytest.raises(TypeError):
        test_parent_meter_intact_lease_keeps_base_semantic_refusal(tmp_path)


def test_parity_control_external_effect_refusal_type_change_fails(tmp_path, monkeypatch):
    from kernel_unknown_outcomes import ExternalEffectAdapter

    def wrong_type(_self):
        raise TypeError("external effect intent schema is invalid")

    monkeypatch.setattr(ExternalEffectAdapter, "_replay", wrong_type)
    with pytest.raises(TypeError):
        test_external_effect_intact_lease_keeps_base_semantic_refusal_in_both_passes(
            tmp_path, monkeypatch, "first", "external effect intent schema is invalid")


def test_parity_control_skipped_live_handoff_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ContinuationJournal, "subscribe", lambda *_args, **_kwargs: None)
    with pytest.raises(AssertionError):
        test_successful_replay_hands_off_once_with_kind_filter_cursor_gaps(tmp_path)
