"""Work and identity guards for the generated-artifact journal observer."""
import json
import pickle
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from time import perf_counter
from types import SimpleNamespace

import kernel_generated_artifacts as artifacts
import pytest
from kernel_continuation_journal import ContinuationJournal
from kernel_measurement import JOURNAL_BLOB_THRESHOLD_BYTES, canonical_bytes


def _row(sequence, **fields):
    return {
        "schema": "stateframe-audit/1",
        "run_id": "run",
        "sequence": sequence,
        "event_id": f"run:{sequence}",
        "kind": "run_started" if sequence == 1 else "receipt_telemetry",
        **fields,
    }


def _index():
    index = artifacts.GeneratedArtifactIndex("run")
    index.observe(_row(1))
    return index


def _canonical_digest(event):
    encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(encoded.encode()).hexdigest()


def test_receipt_history_is_encoded_once_across_duplicate_subscribers(tmp_path, monkeypatch):
    # Isolate CPU work from storage latency while exercising real, detached deliveries.
    monkeypatch.setattr("os.fsync", lambda _fd: None)
    journal = ContinuationJournal(tmp_path, "receipt-history", {})
    counts = Counter()
    state = SimpleNamespace(delivering=None, observer_seconds=0.0)
    index = artifacts.GeneratedArtifactIndex(journal.run_id)
    original_digest = artifacts._digest

    def count_digest(value):
        if state.delivering:
            counts[state.delivering, "stack encodes"] += 1
        return original_digest(value)

    def count_dumps(value, *args, **kwargs):
        if state.delivering:
            label = "row encodes" if isinstance(value, dict) else "scalar visits"
            counts[state.delivering, label] += 1
        return json.dumps(value, *args, **kwargs)

    def deliver(event):
        state.delivering = "repeat" if event["sequence"] <= index.sequence else "admit"
        counts[state.delivering, "deliveries"] += 1
        started = perf_counter()
        try:
            index.observe(event)
        finally:
            state.observer_seconds += perf_counter() - started
            state.delivering = None

    monkeypatch.setattr(artifacts, "_digest", count_digest)
    monkeypatch.setattr(artifacts, "json", SimpleNamespace(dumps=count_dumps, loads=json.loads))
    receipts = [
        {"path": f"src/feature_{ordinal:02}.py", "line": ordinal + 1,
         "proof": {"status": "accepted", "outputs": ["Δ", None, True, ordinal]}}
        for ordinal in range(32)
    ]
    try:
        journal.subscribe("primary", deliver)
        journal.subscribe("duplicate", deliver)
        counts.clear()
        state.observer_seconds = 0.0
        started = perf_counter()
        for ordinal in range(200):
            journal.emit("receipt_telemetry", ordinal=ordinal, receipts=receipts)
        journal.subscribe("history", deliver, kinds=("receipt_telemetry",))
        elapsed = perf_counter() - started
        assert index.sequence == journal._sequence
        assert journal.publication_stats().get("observer_failures", 0) == 0
    finally:
        journal.close()

    print(f"receipt history {elapsed:.6f}s, observer {state.observer_seconds:.6f}s, {dict(counts)}")
    assert counts["admit", "deliveries"] == 200
    assert counts["repeat", "deliveries"] == 400
    assert sum(counts[delivery, kind] for delivery in ("admit", "repeat")
               for kind in ("stack encodes", "row encodes")) <= 200
    assert all(counts["repeat", kind] == 0 for kind in (
        "stack encodes", "row encodes", "scalar visits"))
    assert state.observer_seconds < 2


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_live_and_replay_keep_hydrated_digest_at_blob_boundary(tmp_path, delta):
    # JSON quotes cost two bytes. The two distinct widths bracket the byte threshold.
    body = "界" * ((JOURNAL_BLOB_THRESHOLD_BYTES - 2) // 3)
    body += "a" * (JOURNAL_BLOB_THRESHOLD_BYTES + delta - len(canonical_bytes(body)))
    assert len(canonical_bytes(body)) == JOURNAL_BLOB_THRESHOLD_BYTES + delta
    journal = ContinuationJournal(tmp_path, "blob-boundary", {})
    try:
        index = artifacts.GeneratedArtifactIndex(journal.run_id)
        journal.subscribe("generated", index.observe)
        reference = "stateframe://artifacts/rejected/blob"
        event = journal.emit("rejected_candidate_artifact", text=body,
                             reference=reference, artifact_id="blob")
        stored = json.loads(journal.path.read_bytes().splitlines()[-1])
        assert isinstance(stored["text"], dict) == (delta > 0)
        expected = _canonical_digest(event)
        assert index._events[event["sequence"]] == expected
        assert stored["record_hash"] != "sha256:" + expected
        replayed = index.replay(journal.path, run_id=journal.run_id)
        assert replayed._events == index._events
        assert replayed.recall(reference) == index.recall(reference)
    finally:
        journal.close()


@pytest.mark.parametrize("original,changed", [(1, True), (1, 1.0), (-0.0, 0.0), ("一", "二")])
def test_uninterested_duplicate_conflicts_with_unchanged_record_hash(original, changed):
    index = _index()
    event = _row(2, record_hash="sha256:" + "a" * 64, receipt={"value": original})
    index.observe(event)
    index.observe(deepcopy(event))
    event["receipt"]["value"] = changed
    with pytest.raises(ValueError, match="conflicting replay event"):
        index.observe(event)
    assert index.sequence == 2
    assert index._events[2] == _canonical_digest(_row(
        2, record_hash=event["record_hash"], receipt={"value": original}))


def test_duplicate_respects_canonical_equivalence_and_scope():
    index = _index()
    event = _row(2, receipt={"é": (float("nan"), "\u2028", float("inf")), "a": {}})
    index.observe(event)
    duplicate = _row(2, receipt={"a": {}, "é": [float("nan"), "\u2028", float("inf")]})
    index.observe(duplicate)
    assert index._events[2] == _canonical_digest(event) == _canonical_digest(duplicate)
    with pytest.raises(ValueError, match="identity"):
        index.observe(dict(duplicate, event_id="other:2"))
    with pytest.raises(ValueError, match="gap"):
        index.observe(_row(4))


def test_malformed_payloads_never_advance_or_match_a_previous_row():
    index = _index()
    index.observe(_row(2, receipt={"1": "text"}))
    for sequence in (2, 3):
        with pytest.raises(TypeError, match="mapping keys"):
            index.observe(_row(sequence, receipt={1: "text"}))
        with pytest.raises(TypeError, match="not JSON serializable"):
            index.observe(_row(sequence, receipt=b"bytes"))
        with pytest.raises(UnicodeEncodeError):
            index.observe(_row(sequence, receipt="\ud800"))
    with pytest.raises(ValueError, match="malformed rejected candidate"):
        index.observe(_row(3, kind="rejected_candidate_artifact"))
    assert index.sequence == 2
    assert not index.records


def test_digest_limits_apply_to_admission_and_repeated_delivery(monkeypatch):
    index = _index()
    event = _row(2, receipt=["x", "y", "z"])
    monkeypatch.setattr(artifacts, "_MAX_DIGEST_NODES", 10)
    index.observe(event)  # Root, six fields, and three list entries.
    for sequence in (2, 3):
        with pytest.raises(ValueError, match="node limit"):
            index.observe(_row(sequence, receipt=["x", "y", "z", "extra"]))
    monkeypatch.setattr(artifacts, "_MAX_DIGEST_DEPTH", 2)
    with pytest.raises(ValueError, match="depth limit"):
        index.observe(_row(3, receipt=[[["too deep"]]]))
    assert index._events[2] == _canonical_digest(event)
    assert index.sequence == 2


def test_deep_and_circular_payloads_use_bounded_digest():
    index = _index()
    depth = 2500
    leaf = "text"
    for _ in range(depth):
        leaf = [leaf]
    event = _row(2, receipt=leaf)
    index.observe(event)
    index.observe(event)
    expected = canonical_bytes(_row(2, receipt="marker")).replace(
        b'"marker"', b"[" * depth + b'"text"' + b"]" * depth)
    assert index._events[2] == sha256(expected).hexdigest()

    mapping = {}
    mapping["self"] = mapping
    sequence = []
    sequence.append(sequence)
    for value, reason in ((mapping, "circular mapping"), (sequence, "circular sequence")):
        for row in (2, 3):
            event = _row(row, receipt=value)
            with pytest.raises(ValueError, match=reason):
                index.observe(event)
    index.mark_unobserved(event, error_type="ValueError")
    assert index._events[3] == "UNOBSERVED:ValueError"
    assert index.completeness == "INCOMPLETE"


def test_old_and_restored_rows_still_check_the_hydrated_digest():
    index = _index()
    receipt = "transient receipt " * 32
    event = _row(2, receipt=receipt)
    index.observe(event)
    saved = pickle.dumps(index, protocol=5)
    assert receipt.encode() not in saved
    restored = pickle.loads(saved)
    for sequence in range(3, 260):
        index.observe(_row(sequence))

    for projection in (index, restored):
        projection.observe(deepcopy(event))
        with pytest.raises(ValueError, match="conflicting replay event"):
            projection.observe(dict(event, receipt="different receipt"))
        assert projection._events[2] == _canonical_digest(event)


def test_large_valid_row_keeps_its_digest_and_conflict_checks():
    index = _index()
    text = "界" * (3 * 1024 * 1024)
    event = _row(2, receipt=text)
    index.observe(event)
    index.observe(deepcopy(event))
    assert index._events[2] == _canonical_digest(event)
    with pytest.raises(ValueError, match="conflicting replay event"):
        index.observe(dict(event, receipt=text[:-1] + "文"))
