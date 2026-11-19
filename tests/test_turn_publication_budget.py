"""Measure publication separately from workspace admission and provider transport."""
from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import kernel_checkpoint as checkpoint
from kernel_continuation_journal import ContinuationJournal


def test_constant_head_work_and_crash_atomicity(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for number in range(5_000):
        (workspace / f"module_{number}.py").write_text("value = 1\n")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for number in range(200):
        folder = checkpoints / f"{number:04d}-1"
        folder.mkdir()
        (folder / checkpoint.META_FILE).write_text(json.dumps({"seq": number}))
    (checkpoints / "LATEST").write_text(f"{folder.name}\n")

    monkeypatch.setenv("STATEFRAME_TURN_ATOMIC_PUBLICATION", "on")
    with ContinuationJournal(tmp_path, "publication-budget", {}) as journal:
        journal.emit("continuation_worker_checkpoint")
        events = [("action_receipt", {"step": 200, "detail": n}) for n in range(24)]
        for _ in range(200):
            journal.emit_batch(events)
        assert checkpoint._next_seq(checkpoints) == 200  # admit legacy checkpoint lineage

        work = {"journal_fsyncs": 0, "all_fsyncs": 0, "checkpoint_scans": 0,
                "history_scans": 0, "old_metadata_reads": 0}
        seen = []
        journal.subscribe("order", lambda row: seen.append(row["detail"]),
                          kinds=("action_receipt",))
        seen.clear()
        fsync, iterdir, read_text = os.fsync, Path.iterdir, Path.read_text
        inode = os.fstat(journal._fd).st_ino

        def count_fsync(fd):
            work["all_fsyncs"] += 1
            if os.fstat(fd).st_ino == inode:
                work["journal_fsyncs"] += 1
            return fsync(fd)

        def count_iterdir(path):
            if path in (workspace, checkpoints):
                work["checkpoint_scans"] += 1
            return iterdir(path)

        def count_metadata_reads(path, *args, **kwargs):
            if path.name == checkpoint.META_FILE and path.parent.parent == checkpoints:
                work["old_metadata_reads"] += 1
            return read_text(path, *args, **kwargs)

        records = journal._records

        def count_records(*args, **kwargs):
            work["history_scans"] += 1
            return records(*args, **kwargs)

        monkeypatch.setattr(os, "fsync", count_fsync)
        monkeypatch.setattr(Path, "iterdir", count_iterdir)
        monkeypatch.setattr(journal, "_records", count_records)
        monkeypatch.setattr(Path, "read_text", count_metadata_reads)
        start = perf_counter()
        rows = journal.emit_batch(events)
        assert checkpoint._next_seq(checkpoints) == 200
        elapsed = perf_counter() - start
        assert seen == list(range(24))
        assert [row["detail"] for row in rows] == seen
        print(f"publication fixture: {elapsed:.6f} s {work}")
        assert work["journal_fsyncs"] <= 1 and work["all_fsyncs"] <= 4 and (
            work["checkpoint_scans"] == work["history_scans"] == work["old_metadata_reads"] == 0), work
        assert elapsed < 0.45

def test_manifest_compare_and_swap_and_suffix(tmp_path, monkeypatch):
    import pytest

    publication = pytest.importorskip("kernel_turn_publication")
    from kernel_turn_contracts import Basis, Commit, Context, Corrupt, Event, Head, Key, Ref, Stale

    monkeypatch.setenv(publication.SWITCH, "on")
    ctx = Context("principal", "run", "first", tmp_path, "profile")
    with_port = publication.open_port(ctx)
    old = publication.GENESIS
    after = Head(1, "1" * 64)
    basis = Basis("parent", "live", 0, after, 1, "authority")

    class Cancel(Exception):
        pass

    with pytest.raises(Cancel), with_port.transaction():
        payload = with_port.stage("turn/event/1", b'{"body":"first"}')
        event = Event(old, after, "action_receipt", (Key("receipt", b"one"),), payload)
        with_port.record(event)
        original = Commit(basis, None, "send-1", (("receipts", payload),))
        raise Cancel
    assert with_port.head() is None
    with pytest.raises(Corrupt):
        with_port.load(payload)

    with with_port.transaction():
        assert with_port.stage(payload.codec, b'{"body":"first"}') == payload
        publication.record_event(with_port, event)
        publication.record_event(with_port, event)
        with pytest.raises(Corrupt):
            publication.record_event(with_port, event._replace(kind="conflict"))
        first = with_port.commit(original)
        assert publication.open_port(ctx).head() == first  # commit acknowledges durability
        with pytest.raises(Corrupt):
            with_port.commit(original._replace(previous=first, boundary_id="second-in-transaction"))

    reopened = publication.open_port(ctx._replace(attempt="restart"))
    assert reopened.head() == first
    assert list(reopened.suffix(old, after)) == [event]
    assert list(reopened.suffix(after, after)) == []
    amendment = Head(2, "2" * 64)
    with reopened.transaction():
        change = reopened.stage("turn/event/1", b'{"body":"amended"}')
        changed = Event(after, amendment, "action_receipt_amended", event.changed, change)
        reopened.record(event)
        reopened.record(changed)
        second = reopened.commit(Commit(basis._replace(journal=amendment), first, "send-2", ()))

    assert with_port.head() == second
    assert list(with_port.suffix(after, amendment)) == [changed]
    assert json.loads(with_port.load(second))["roots"] == [["receipts", list(payload)]]
    assert with_port.commit(original) == first
    with pytest.raises(Corrupt):
        with_port.commit(original._replace(roots=()))
    with pytest.raises(Stale):
        with_port.commit(original._replace(boundary_id="late"))
    with pytest.raises(Corrupt):
        with_port.commit(Commit(basis._replace(journal=amendment), second, "missing",
                                (("bad", Ref("turn/event/1", "f" * 64, 1)),)))
    assert with_port.head() == second
    with pytest.raises(Corrupt):
        list(with_port.suffix(Head(1, "foreign"), amendment))
    other = publication.open_port(ctx._replace(principal="other"))
    assert other.head() is None
    with pytest.raises(Corrupt):
        other.load(payload)


_JOURNAL_KILL = r"""
import os
import signal
import stat
import sys
from pathlib import Path
import kernel_continuation_journal as journals
import kernel_turn_publication as publication

def stop():
    signal.raise_signal(signal.SIGKILL)

checkpoint = sys.argv[2]
journal = journals.ContinuationJournal(Path(sys.argv[1]), "crash-publication", {})
before = journal.emit("continuation_worker_checkpoint")
print(journal.path, before["record_hash"], flush=True)
atomic_file, write, sync = publication.atomic_file, os.write, os.fsync
sync_dir = journals._sync_directory
replace, unlink = os.replace, Path.unlink
def mark(path, data):
    if checkpoint == "before-intent":
        stop()
    atomic_file(path, data)
    if checkpoint == "after-intent":
        stop()
def write_row(fd, data):
    count = write(fd, data[:len(data) // 2] if fd == journal._fd else data)
    if fd == journal._fd and checkpoint == "partial-row":
        stop()
    return count
def sync_rows(fd):
    sync(fd)
    if fd == journal._fd and checkpoint == "after-journal-sync":
        stop()
    if (fd != journal._fd and stat.S_ISREG(os.fstat(fd).st_mode)
            and checkpoint == "after-intent-file-sync"):
        stop()
def rename_intent(src, dst):
    replace(src, dst)
    if Path(dst).name == journals._PUBLICATION_INTENT and checkpoint == "after-intent-rename":
        stop()
def unlink_intent(path, *args, **kwargs):
    unlink(path, *args, **kwargs)
    if path.name == journals._PUBLICATION_INTENT and checkpoint == "after-intent-unlink":
        stop()
def finish(path):
    sync_dir(path)
    if checkpoint == "after-publication":
        stop()

publication.atomic_file = mark
os.write = write_row if checkpoint == "partial-row" else write
os.fsync = sync_rows
os.replace = rename_intent
Path.unlink = unlink_intent
journals._sync_directory = finish
journal.emit_batch([("action_receipt", {"index": i}) for i in range(3)])
"""


def test_journal_batch_recovers_at_each_persistence_barrier(tmp_path, monkeypatch):
    import signal
    import subprocess
    import sys

    import pytest

    pytest.importorskip("kernel_turn_publication")
    from kernel_continuation_journal import recover_publication_mirror

    monkeypatch.setenv("STATEFRAME_TURN_ATOMIC_PUBLICATION", "on")
    barriers = ("before-intent", "after-intent-file-sync", "after-intent-rename", "after-intent",
                "partial-row", "after-journal-sync", "after-intent-unlink", "after-publication")
    for barrier in barriers:
        parent = tmp_path / barrier
        parent.mkdir()
        child = subprocess.run([sys.executable, "-c", _JOURNAL_KILL, str(parent), barrier],
                               capture_output=True, text=True, timeout=30, check=False)
        assert child.returncode == -signal.SIGKILL, child.stderr
        name, before = child.stdout.strip().split()
        path = Path(name)
        recover_publication_mirror(path)
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        indexes = [row["index"] for row in rows if row["kind"] == "action_receipt"]
        if barrier == "after-intent-unlink":
            assert indexes in ([], [0, 1, 2])
        else:
            assert indexes == ([0, 1, 2] if barrier == "after-publication" else [])
        expected_head = rows[-1]["record_hash"] if indexes else before
        with ContinuationJournal.resume(path, task_id="crash-publication", manifest={},
                                        expected_head=expected_head) as resumed:
            assert resumed.cached_records()[-1]["previous_hash"] == expected_head


_CHECKPOINT_KILL = r"""
import json
import os
import signal
import sys
from pathlib import Path
import kernel_checkpoint as checkpoint
import kernel_turn_publication as publication

run_dir, barrier = Path(sys.argv[1]), sys.argv[2]
root = run_dir / "checkpoints"
staged = root / ".0200-1.tmp"
staged.mkdir()
(staged / checkpoint.META_FILE).write_text(json.dumps({"seq": checkpoint._next_seq(root)}))
real_sync, real_write = publication.sync_directory, checkpoint._write_publication_index
replace = os.replace
def stop():
    signal.raise_signal(signal.SIGKILL)
def sync_dir(path):
    real_sync(path)
    if barrier == "after-directory":
        stop()
def write_index(*args):
    real_write(*args)
    if barrier == "after-index":
        stop()
def replace_pointer(source, destination):
    name = Path(destination).name
    if name == checkpoint._PUBLICATION_INDEX and barrier == "before-index-replace":
        stop()
    if name == "LATEST" and barrier == "before-latest-replace":
        stop()
    replace(source, destination)
    if name == checkpoint._PUBLICATION_INDEX and barrier == "after-index-replace":
        stop()
publication.sync_directory = sync_dir
checkpoint._write_publication_index = write_index
os.replace = replace_pointer
checkpoint._publish_checkpoint(root, staged, root / "0200-1", run_dir)
stop()
"""


def test_checkpoint_index_recovers_latest_at_publication_barriers(tmp_path, monkeypatch):
    import signal
    import subprocess
    import sys

    import pytest

    pytest.importorskip("kernel_turn_publication")
    monkeypatch.setenv("STATEFRAME_TURN_ATOMIC_PUBLICATION", "on")
    barriers = ("after-directory", "before-index-replace", "after-index-replace",
                "after-index", "before-latest-replace", "after-mirror")
    for barrier in barriers:
        run = tmp_path / barrier
        root = run / "checkpoints"
        previous = root / "0199-1"
        previous.mkdir(parents=True)
        (previous / checkpoint.META_FILE).write_text('{"seq":199}')
        (root / "LATEST").write_text("0199-1\n")
        child = subprocess.run([sys.executable, "-c", _CHECKPOINT_KILL, str(run), barrier],
                               capture_output=True, text=True, timeout=30, check=False)
        assert child.returncode == -signal.SIGKILL, child.stderr
        checkpoint.recover_publication_latest(root)
        expected = checkpoint.latest(run).name
        if barrier in ("after-directory", "before-index-replace"):
            assert expected == "0199-1"
        elif barrier == "after-index-replace":
            assert expected in ("0199-1", "0200-1")
        else:
            assert expected == "0200-1"
        assert (root / "LATEST").read_text().strip() == expected
        assert checkpoint.publication_candidates(root)[-1].name == expected


_MANIFEST_KILL = r"""
import signal
import sys
from pathlib import Path
from kernel_turn_contracts import Basis, Commit, Context
from kernel_turn_publication import GENESIS, open_port

port = open_port(Context("owner", "run", "child", Path(sys.argv[1]), "profile"))
basis = Basis("parent", "live", 0, GENESIS, 0, "authority")
with port.transaction():
    root = port.stage("turn/scope/1", b"protected bytes")
    if sys.argv[2] == "before":
        port._connection.set_trace_callback(
            lambda sql: signal.raise_signal(signal.SIGKILL) if sql == "COMMIT" else None
        )
    port.commit(Commit(basis, None, "send-1", (("qualified", root),)))
signal.raise_signal(signal.SIGKILL)
"""


def test_manifest_and_object_survive_together_after_writer_death(tmp_path, monkeypatch):
    import signal
    import subprocess
    import sys

    import pytest

    publication = pytest.importorskip("kernel_turn_publication")
    from kernel_turn_contracts import Context, Corrupt, Ref

    monkeypatch.setenv(publication.SWITCH, "on")
    for barrier in ("before", "after"):
        path = tmp_path / barrier
        path.mkdir()
        child = subprocess.run([sys.executable, "-c", _MANIFEST_KILL, str(path), barrier],
                               capture_output=True, text=True, timeout=30, check=False)
        assert child.returncode == -signal.SIGKILL, child.stderr
        port = publication.open_port(Context("owner", "run", "parent", path, "profile"))
        head = port.head()
        if barrier == "before":
            assert head is None
            missing = publication._ref("turn/scope/1", b"protected bytes")
            with pytest.raises(Corrupt):
                port.load(missing)
        else:
            assert head is not None
            _, root = json.loads(port.load(head))["roots"][0]
            assert port.load(Ref(*root)) == b"protected bytes"


def test_batched_observers_see_their_own_prefix_in_order(tmp_path, monkeypatch):
    import pytest

    pytest.importorskip("kernel_turn_publication")
    monkeypatch.setenv("STATEFRAME_TURN_ATOMIC_PUBLICATION", "on")
    observed = []
    with ContinuationJournal(tmp_path, "observer-order", {}) as journal:
        def observe(row):
            prefix = [item["sequence"] for item in journal.records(rebuild=False)]
            observed.append((row["sequence"], prefix[-1], journal.head))

        journal.subscribe("observe", observe, kinds=("action_receipt",))
        rows = journal.emit_batch([("action_receipt", {"number": index}) for index in range(3)])
        assert observed == [(row["sequence"], row["sequence"], row["record_hash"]) for row in rows]
        assert [row["number"] for row in journal.cached_records() if row["kind"] == "action_receipt"] == [0, 1, 2]
