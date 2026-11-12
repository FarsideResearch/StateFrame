"""Group commit: only pure telemetry may wait for an fsync; every resume-relevant row is durable on return."""
from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from kernel_continuation_journal import (
    BATCHABLE_TELEMETRY_KINDS, ContinuationJournal, JournalError,
)
from kernel_measurement import canonical_bytes

ACTION_KINDS = ("request_submitted", "action_receipt", "tool_mutation", "continuation_worker_checkpoint")


def _fsyncs(monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    return calls


def test_telemetry_rows_defer_fsync_and_boundaries_flush(tmp_path, monkeypatch):
    with ContinuationJournal(tmp_path, "gc", {}) as journal:
        calls = _fsyncs(monkeypatch)
        assert not journal._checkpoint_seen
        for step in range(3):
            journal.emit("turn_timing", step=step)
        assert calls == []
        journal.emit("request_submitted", step=3)  # boundary: one fsync covers the batched rows
        assert len(calls) == 1
        journal.emit("action_envelope_metrics", step=4)
        assert len(calls) == 1
        journal.sync()
        assert len(calls) == 2
        journal.sync()  # nothing pending
        assert len(calls) == 2
        journal.emit("turn_timing", step=5)
        before_close = len(calls)
    assert len(calls) > before_close  # close made the last telemetry row durable


@pytest.mark.parametrize("kind", ["action_receipt", "tool_mutation", "continuation_worker_checkpoint",
                                  "request_submitted", "check_receipt", "seal"])
def test_resume_relevant_rows_fsync_immediately(tmp_path, monkeypatch, kind):
    assert kind not in BATCHABLE_TELEMETRY_KINDS
    with ContinuationJournal(tmp_path, "gc", {}) as journal:
        calls = _fsyncs(monkeypatch)
        journal.emit(kind, step=1)
        assert len(calls) == 1


def test_journal_observer_failed_is_never_deferred():
    assert "journal_observer_failed" not in BATCHABLE_TELEMETRY_KINDS


def test_batching_off_switch_fsyncs_telemetry(tmp_path, monkeypatch):
    monkeypatch.setenv("SF_JOURNAL_GROUP_COMMIT", "0")
    with ContinuationJournal(tmp_path, "gc", {}) as journal:
        calls = _fsyncs(monkeypatch)
        journal.emit("turn_timing", step=1)
        assert len(calls) == 1


def test_line_is_single_encode_canonical_and_hash_chain_holds(tmp_path):
    import hashlib
    with ContinuationJournal(tmp_path, "gc", {}) as journal:
        journal.emit("action_receipt", step=1, text="café ☃", nested={"b": [1, 2], "a": None})
        path = journal.path
    previous = None
    for line in path.read_bytes().splitlines(keepends=True):
        row = json.loads(line)
        assert line == canonical_bytes(row) + b"\n"
        body = {k: v for k, v in row.items() if k != "record_hash"}
        assert row["record_hash"] == "sha256:" + hashlib.sha256(canonical_bytes(body)).hexdigest()
        assert row["previous_hash"] == previous
        previous = row["record_hash"]


_CHILD = r"""
import signal, sys, random
from pathlib import Path
from kernel_continuation_journal import ContinuationJournal
rng = random.Random(int(sys.argv[2]))
stop_after = int(sys.argv[3])
journal = ContinuationJournal(Path(sys.argv[1]), "kill", {})
print("READY", journal.path, flush=True)
step = 0
while True:
    step += 1
    kind = rng.choice(["turn_timing", "action_envelope_metrics", "request_submitted",
                       "action_receipt", "tool_mutation", "continuation_worker_checkpoint"])
    record = journal.emit(kind, step=step, pad="x" * rng.randint(0, 4000))
    print(kind, record["sequence"], flush=True)  # ack only after emit() returned
    if step >= stop_after:
        # The child kills itself so no test-side raw kill bypasses the run-kill barrier.
        signal.raise_signal(signal.SIGKILL)
"""


@pytest.mark.parametrize("seed", range(6))
def test_sigkill_keeps_acked_action_rows_and_torn_tail_is_rejected(tmp_path, seed):
    src = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(src), str(src.parent.parent), os.environ.get("PYTHONPATH", "")])}
    stop_after = random.Random(seed).randint(1, 60)
    proc = subprocess.Popen([sys.executable, "-c", _CHILD, str(tmp_path), str(seed), str(stop_after)],
                            stdout=subprocess.PIPE, text=True, env=env)
    acked: dict[int, str] = {}
    try:
        ready = proc.stdout.readline().split(" ", 1)
        assert ready[0] == "READY"
        path = Path(ready[1].strip())
        for line in proc.stdout:
            kind, seq = line.split()
            acked[int(seq)] = kind
    finally:
        proc.wait(timeout=30)
        proc.stdout.close()
    assert proc.returncode == -signal.SIGKILL
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line.endswith(b"}")]
    on_disk = {row["sequence"]: row["kind"] for row in rows}
    for seq, kind in acked.items():
        if kind not in BATCHABLE_TELEMETRY_KINDS:
            assert on_disk.get(seq) == kind, f"acked {kind} #{seq} lost"
    # A torn tail (partial last line) must be rejected on resume, never silently accepted.
    data = path.read_bytes()
    path.write_bytes(data[: data.rstrip(b"\n").rfind(b"\n") + 1] + data.rstrip(b"\n").splitlines()[-1][:-7])
    with pytest.raises(JournalError):
        ContinuationJournal.resume(path, task_id="kill", manifest={}, allow_unanchored_recovery=True)




def test_telemetry_is_fsynced_immediately_once_a_checkpoint_exists(tmp_path, monkeypatch):
    with ContinuationJournal(tmp_path, "gc", {}) as journal:
        journal.emit("continuation_worker_checkpoint", checkpoint={}, checkpoint_hash="x")
        calls = _fsyncs(monkeypatch)
        journal.emit("turn_timing", step=1)
        assert len(calls) == 1


def test_resumed_journal_with_checkpoint_does_not_defer(tmp_path, monkeypatch):
    with ContinuationJournal(tmp_path, "gc", {}) as journal:
        journal.emit("continuation_worker_checkpoint", checkpoint={}, checkpoint_hash="x")
        path, head = journal.path, journal.head
    resumed = ContinuationJournal.resume(path, task_id="gc", manifest={}, expected_head=head)
    try:
        assert resumed._checkpoint_seen
        calls = _fsyncs(monkeypatch)
        resumed.emit("action_envelope_metrics", step=1)
        assert len(calls) == 1
    finally:
        resumed.close()


def test_every_construction_path_has_group_commit_state(tmp_path):
    with ContinuationJournal(tmp_path / "a", "gc", {}) if (tmp_path / "a").mkdir() is None else None as journal:
        assert journal._unsynced is False and journal._checkpoint_seen is False
        path, head = journal.path, journal.head
    resumed = ContinuationJournal.resume(path, task_id="gc", manifest={}, expected_head=head)
    try:
        assert resumed._unsynced is False and isinstance(resumed.group_commit, bool)
    finally:
        resumed.close()
    bare = ContinuationJournal.__new__(ContinuationJournal)  # bypasses __init__/_initialize
    import threading
    bare._lock, bare._fd = threading.RLock(), None
    bare.sync()  # group-commit attrs missing: getattr defaults: no AttributeError


def _old_encoder(row):
    """The pre-change line encoder: canonical_bytes of the finished record."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode() + b"\n"


def test_single_encode_matches_pre_change_encoder_byte_for_byte(tmp_path):
    with ContinuationJournal(tmp_path, "gc", {}) as journal:
        journal.emit("action_receipt", step=1, f=[0.1, 1e-7, 1e22, -0.0, 3.0, 2 ** 70],
                     nested={"z": {"b": 1, "a": [{"y": 1, "x": 2}]}, "a": None},
                     **{"\u00e9clair": "x", "\u4e2d": "\u2603 caf\u00e9 \U0001f600", "B": 1, "a": 2, "_": 3})
        journal.emit("turn_timing", step=2, text="line\nbreak \"q\" \\ \u2028")
        path = journal.path
    lines = path.read_bytes().splitlines(keepends=True)
    assert len(lines) >= 3
    for line in lines:
        assert line == _old_encoder(json.loads(line))


def _record_sync_points(monkeypatch):
    """Record the file size at every fsync: everything past the last one is lost on power failure."""
    synced = {"size": 0}
    real = os.fsync

    def tracked(fd):
        synced["size"] = os.fstat(fd).st_size
        return real(fd)

    monkeypatch.setattr(os, "fsync", tracked)
    return synced


def _decision(path):
    """Resume decision: can the journal be resumed, and does restart refuse the post-checkpoint tail."""
    from kernel_continuation_restart import resume_tail_rejection_reason
    resumed = ContinuationJournal.resume(path, task_id="gc", manifest={}, allow_unanchored_recovery=True)
    try:
        rows = [r for r in resumed.records() if r["kind"] != "continuation_writer_resumed"]
    finally:
        resumed.close()
    last = max((i for i, r in enumerate(rows) if r["kind"] == "continuation_worker_checkpoint"), default=None)
    tail = [] if last is None else rows[last + 1:]
    return last is not None, resume_tail_rejection_reason(tail)


CORPORA = {
    "no_checkpoint": [("turn_timing", {}), ("action_receipt", {}), ("turn_timing", {}),
                      ("action_envelope_metrics", {})],
    "checkpoint_clean_tail": [("turn_timing", {}), ("action_receipt", {}),
                              ("continuation_worker_checkpoint", {"checkpoint": {}, "checkpoint_hash": "h"}),
                              ("action_receipt", {})],
    "checkpoint_then_telemetry_tail": [
        ("turn_timing", {}), ("action_receipt", {}),
        ("continuation_worker_checkpoint", {"checkpoint": {}, "checkpoint_hash": "h"}),
        ("turn_timing", {}), ("action_envelope_metrics", {}), ("journal_observer_failed", {"x": 1})],
    "telemetry_before_checkpoint_only": [
        ("turn_timing", {}), ("action_envelope_metrics", {}),
        ("continuation_worker_checkpoint", {"checkpoint": {}, "checkpoint_hash": "h"})],
}


@pytest.mark.parametrize("name", sorted(CORPORA))
def test_power_loss_never_changes_the_resume_decision(tmp_path, monkeypatch, name):
    """Truncate to the last fsynced offset (simulated power loss); decision must equal the full journal's."""
    synced = _record_sync_points(monkeypatch)
    with ContinuationJournal(tmp_path, "gc", {}) as journal:
        for step, (kind, fields) in enumerate(CORPORA[name]):
            journal.emit(kind, step=step, **fields)
        path = journal.path
        durable = synced["size"]  # before close() performs its final sync
        full = path.read_bytes()
    full_decision = _decision(path)
    path.write_bytes(full[:durable])  # power loss: unsynced suffix gone
    assert _decision(path) == full_decision
    if name == "no_checkpoint":
        assert durable < len(full)  # telemetry really was deferred, so the truncation tested something
    if name == "checkpoint_then_telemetry_tail":
        assert durable == len(full)  # nothing after a checkpoint is ever left unsynced
