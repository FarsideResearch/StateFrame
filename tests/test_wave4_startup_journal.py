from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from _child_env import child_env

import kernel_continuation_journal as journals
import kernel_measurement as measurement
import kernel_reason_log
from kernel_continuation_journal import ContinuationJournal, JournalError


@pytest.fixture(autouse=True)
def _fresh_reason_log():
    # Closing a journal appends the process-wide problem summary, so causes logged by an
    # earlier test in the same worker would change these journals' bytes and heads.
    kernel_reason_log.reset_for_tests()
    yield
    kernel_reason_log.reset_for_tests()


def _pilot(monkeypatch, enabled):
    name = "stateframe.send1_wave4"
    helper = ModuleType(name)
    helper.enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, name, helper)


def _count(monkeypatch, owner, name):
    calls = []
    original = getattr(owner, name)

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, counted)
    return calls


@pytest.mark.parametrize("missing", ["stateframe.send1_wave4", "broken_dependency"])
def test_pilot_treats_only_an_absent_helper_as_disabled(monkeypatch, missing):
    import_module = builtins.__import__
    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "on")

    def import_with_error(name, *args, **kwargs):
        if name == "stateframe.send1_wave4":
            raise ModuleNotFoundError(name=missing)
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_error)
    if missing == "stateframe.send1_wave4":
        assert not measurement.startup_pilot_enabled()
    else:
        with pytest.raises(ModuleNotFoundError) as error:
            measurement.startup_pilot_enabled()
        assert error.value.name == missing


def test_fresh_replay_skips_chain_decode_and_only_loads_requested_blobs(tmp_path, monkeypatch):
    _pilot(monkeypatch, True)
    with ContinuationJournal(tmp_path, "startup", {}) as journal:
        journal._record_cache_limit = 1
        first = journal.emit("keep", item={"text": "a" * 6000, "list": [1]})
        later = journal.emit("later", item={"text": "b" * 6000, "list": [2]})
        inline = journal.emit("inline", item={"list": [3]})
        decoded = _count(monkeypatch, journal, "_records")
        indexed = _count(monkeypatch, journal, "_resolve_indexed_line")
        blob_reads = _count(monkeypatch, measurement, "_read_journal_blob")

        delivered = []
        journal.subscribe("initial", delivered.append, kinds=("keep",))
        assert delivered == [first]
        assert len(blob_reads) == 2  # the stored object and its nested string
        initial_blobs = {args[1] for args in blob_reads}
        delivered[0]["item"]["list"].append("changed")
        indexed_row = journal.record_by_event_id(first["event_id"])
        assert indexed_row == first
        assert list(indexed_row) == sorted(first)
        assert decoded == indexed == []

        blob_reads.clear()
        delivered.clear()
        journal.subscribe("deferred", delivered.append, kinds=("later",))
        assert delivered == [later]
        assert len(blob_reads) == 2
        assert initial_blobs.isdisjoint(args[1] for args in blob_reads)
        assert decoded == indexed == []
        assert journal.subscriber_cursor("deferred") == journal._sequence

        all_rows = []
        journal.subscribe("all", all_rows.append)
        assert decoded == indexed == []
        assert list(all_rows[first["sequence"] - 1]) == sorted(first)
        assert list(all_rows[-1]) == list(inline)
        assert all_rows == list(journal.records(rebuild=False))


def test_evicted_prefix_is_validated_and_retained_suffix_can_be_indexed(tmp_path, monkeypatch):
    _pilot(monkeypatch, True)
    monkeypatch.setattr(journals, "RECORD_CACHE_LIMIT", 3)
    with ContinuationJournal(tmp_path, "bounded", {}) as journal:
        for ordinal in range(6):
            last = journal.emit("keep", ordinal=ordinal)
        indexed = _count(monkeypatch, journal, "_resolve_indexed_line")
        assert journal._indexed_record(last["sequence"]) == last
        assert indexed == []

        scans = _count(monkeypatch, journal, "_records")
        delivered = []
        journal.subscribe("projection", delivered.append, kinds=("keep",))
        assert [record["ordinal"] for record in delivered] == list(range(6))
        assert len(scans) == 1


def test_event_bytes_and_receipt_references_match_through_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(journals, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="abc")))
    monkeypatch.setattr(journals, "time", SimpleNamespace(time_ns=lambda: 123456))
    results = []
    for enabled in (False, True):
        _pilot(monkeypatch, enabled)
        root = tmp_path / str(enabled)
        root.mkdir()
        with ContinuationJournal(root, "parity", {}) as journal:
            evidence = journal.emit("evidence", item="é" * 6000)
            receipt = journal.emit("action_receipt", receipt_id=evidence["event_id"])
            journal.emit("request_prepare_timing", elapsed=3)
            journal.sync()
            fresh = []
            journal.subscribe("receipts", fresh.append, kinds=("action_receipt",))
            assert fresh == [receipt]
            assert evidence["event_id"] in journal.receipt_refs_at()
            prefix = journal.path.read_bytes()
            path, head = journal.path, journal.head

        with ContinuationJournal.resume(path, task_id="parity", manifest={}, expected_head=head) as journal:
            assert journal not in (journals._STARTUP_WRITER_ROWS or ())
            scans = _count(monkeypatch, journal, "_records")
            replayed = []
            journal.subscribe("receipts", replayed.append, kinds=("action_receipt",))
            assert len(scans) == 1
            assert replayed == fresh
            assert journal._indexed_record(evidence["sequence"]) == evidence
            results.append((prefix, path.read_bytes(), replayed, journal.receipt_refs_at()))
    assert results[0] == results[1]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mutation", ["blob", "excluded_line", "reorder", "truncate", "replace"])
def test_warm_subscriber_refuses_changed_journal_before_delivery(
        tmp_path, monkeypatch, enabled, mutation):
    _pilot(monkeypatch, enabled)
    with ContinuationJournal(tmp_path, "changed", {}) as journal:
        excluded = journal.emit("skip", ordinal=1)
        first = journal.emit("keep", ordinal=2, item="x" * 6000)
        last = journal.emit("keep", ordinal=3)
        journal.subscribe("warm", lambda row: None, kinds=("keep",))
        data = journal.path.read_bytes()
        lines = data.splitlines(keepends=True)
        if mutation == "blob":
            reference = json.loads(lines[first["sequence"] - 1])["item"]
            path = journal.path.parent / "blobs" / (reference["$blob"] + ".zlib")
            path.write_bytes(b"damaged")
            with pytest.raises(JournalError, match=r"invalid (?:indexed )?journal blob"):
                journal._indexed_record(first["sequence"])
        elif mutation == "excluded_line":
            journal.path.write_bytes(data.replace(b'"ordinal":1', b'"ordinal":7'))
            with pytest.raises(JournalError, match="indexed journal hash"):
                list(journal._indexed_records(excluded["sequence"], last["sequence"],
                                              kinds=frozenset({"keep"})))
        elif mutation == "reorder":
            lines[-2], lines[-1] = lines[-1], lines[-2]
            journal.path.write_bytes(b"".join(lines))
        elif mutation == "truncate":
            journal.path.write_bytes(data[:-1])
        else:
            replacement = journal.path.with_name("replacement")
            replacement.write_bytes(data)
            replacement.replace(journal.path)

        delivered = []
        with pytest.raises(JournalError):
            journal.subscribe("changed", delivered.append, kinds=("keep",))
        assert delivered == []
        assert journal.subscriber_cursor("changed") == 0
        path, head = journal.path, journal.head

    if mutation == "replace":
        with ContinuationJournal.resume(path, task_id="changed", manifest={}, expected_head=head):
            pass
    else:
        with pytest.raises(JournalError):
            ContinuationJournal.resume(path, task_id="changed", manifest={}, expected_head=head)


@pytest.mark.parametrize("enabled", [False, True])
def test_subscriber_retry_hands_off_each_committed_row_once(tmp_path, monkeypatch, enabled):
    _pilot(monkeypatch, enabled)
    with ContinuationJournal(tmp_path, "retry", {}) as journal:
        first = journal.emit("keep", ordinal=0)
        last = journal.emit("keep", ordinal=1)
        journal.emit("skip")
        delivered = []

        def interrupt(row):
            if row == last:
                raise KeyboardInterrupt
            delivered.append(row)

        with pytest.raises(KeyboardInterrupt):
            journal.subscribe("retry", interrupt, kinds=("keep",))
        assert journal.subscriber_cursor("retry") == first["sequence"]
        journal.subscribe("retry", delivered.append, kinds=("keep",))
        live = journal.emit("keep", ordinal=2)
        assert delivered == [first, last, live]
        assert journal.subscriber_cursor("retry") == live["sequence"]


@pytest.mark.parametrize("enabled, reads", [(False, 2), (True, 1)])
def test_existing_blob_is_loaded_once_when_its_identity_is_unchanged(
        tmp_path, monkeypatch, enabled, reads):
    _pilot(monkeypatch, enabled)
    with ContinuationJournal(tmp_path, "blob", {}) as journal:
        payload = {"text": "x" * 6000}
        original = measurement.externalize_journal_fields(payload, journal.path)
        blob_reads = _count(monkeypatch, measurement, "_read_journal_blob")
        assert measurement.externalize_journal_fields(payload, journal.path) == original
        assert len(blob_reads) == reads

        read = measurement._read_journal_blob

        def replace_after_read(path, digest):
            result = read(path, digest)
            blob = journal.path.parent / "blobs" / (digest + ".zlib")
            replacement = blob.with_suffix(".replacement")
            replacement.write_bytes(b"replacement")
            replacement.replace(blob)
            return result

        monkeypatch.setattr(measurement, "_read_journal_blob", replace_after_read)
        with pytest.raises(ValueError, match="cannot be decompressed"):
            measurement.externalize_journal_fields(payload, journal.path)


_CRASH_WRITER = """
import os, stat, sys
from pathlib import Path
from types import ModuleType

import kernel_continuation_journal as journals

root, enabled, point = sys.argv[1:]
name = "stateframe.send1_wave4"
helper = ModuleType(name)
helper.enabled = lambda: enabled == "True"
sys.modules[name] = helper
journal = journals.ContinuationJournal(Path(root), "crash", {})
print(journal.path, journal.head, flush=True)
if point == "before_blob_sync":
    fsync = os.fsync

    def interrupt_blob(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode) and fd != journal._fd:
            os._exit(91)
        fsync(fd)

    os.fsync = interrupt_blob
    journal.emit("keep", item="x" * 6000)
elif point == "before_batch_commit":
    unlink = Path.unlink

    def interrupt_commit(path, *args, **kwargs):
        if path.name == journals._PUBLICATION_INTENT:
            os._exit(91)
        return unlink(path, *args, **kwargs)

    Path.unlink = interrupt_commit
else:
    journal._publish_record = lambda *args: os._exit(91)
journal.emit_batch([("keep", {"ordinal": 0}), ("keep", {"ordinal": 1})])
"""


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("point", ["before_blob_sync", "before_batch_commit", "before_fanout"])
def test_crash_recovery_respects_blob_and_batch_publication(
        tmp_path, monkeypatch, enabled, point):
    _pilot(monkeypatch, enabled)
    monkeypatch.setenv(journals._PUBLICATION_SWITCH, "on")
    stopped = subprocess.run(
        [sys.executable, "-c", _CRASH_WRITER, str(tmp_path), str(enabled), point],
        capture_output=True, text=True, timeout=20, check=False, env=child_env(),
    )
    assert stopped.returncode == 91, stopped.stderr
    raw_path, head = stopped.stdout.split()
    path = Path(raw_path)
    journals.recover_publication_mirror(path)
    expected = []
    if point == "before_fanout":
        head = json.loads(path.read_bytes().splitlines()[-1])["record_hash"]
        expected = [0, 1]
    with ContinuationJournal.resume(path, task_id="crash", manifest={}, expected_head=head) as journal:
        delivered = []
        journal.subscribe("verify", lambda row: delivered.append(row["ordinal"]), kinds=("keep",))
        assert delivered == expected
        journal.emit("keep", ordinal=2)
        assert delivered == [*expected, 2]
