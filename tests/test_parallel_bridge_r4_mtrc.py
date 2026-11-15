"""Engine-free accounting for rejected answers recovered through ACK-bound reads."""
import os
import threading

import pytest
from test_parallel_bridge_r3_chan import _writer

import stateframe.parallel_channel as channel
import stateframe.parallel_host as host
from stateframe.parallel_journal import JournalBusy


def _rejected_answer(tmp_path, monkeypatch):
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    writer, journal = _writer(monkeypatch)
    supervisor._channel_writes = writer
    supervisor._channel_thread = threading.current_thread()
    an = supervisor.parallel_root / "call/attempt-1/an"
    rq = an.parent / "rq/001.json"
    rq.parent.mkdir(parents=True)
    rq.write_text("{}\n")
    os.utime(rq, (100.0, 100.0))
    clock = [10.0]
    monkeypatch.setattr(host.time, "time", lambda: 200.0)
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])
    path = an / "001.json"
    value = {"status": "OK", "payload": {"state": "RUNNING"}}
    with pytest.raises(JournalBusy):
        supervisor._write_json(path, value)
    # Directory fsync can fail after publication. Keep real fsyncs for the visible
    # bytes here; control only the helper's response and its consumption boundary.
    host._write_json(path, value)
    for attempt in range(3):
        ticket = journal.tickets[-1]
        assert ticket.kind == "write"
        journal.complete(ticket, {"ok": False, "error_class": "OSError", "error": "fsync failed"})
        assert writer.drain() == ()
        if attempt < 2:
            assert writer.drain() == ()  # submit the frozen retry
    assert path not in writer._pending
    assert len(journal.consumed) == 3
    assert supervisor._answer_latency.snapshot()["count"] == 0
    rq.unlink()  # recovering the original timer must not depend on request metadata
    return supervisor, writer, journal, path, value, clock


def _ack_sync(journal):
    ticket = journal.tickets[-1]
    assert ticket.kind == "sync"
    channel.sync_record(ticket.payload)
    journal.complete(ticket)


@pytest.mark.parametrize("settle", ["read", "drain"])
@pytest.mark.parametrize("sync_failures", [0, 1])
def test_answer_latency_survives_rejected_publication_revalidation(
        tmp_path, monkeypatch, settle, sync_failures):
    supervisor, writer, journal, path, value, clock = _rejected_answer(tmp_path, monkeypatch)
    with pytest.raises(JournalBusy):
        writer.read(path, "answer", wait=False)
    for _ in range(sync_failures):
        journal.complete(journal.tickets[-1], {
            "ok": False, "error_class": "OSError", "error": "sync fsync failed"})
        assert writer.drain() == ()
        assert writer.drain() == ()
    assert supervisor._answer_latency.snapshot()["count"] == 0
    _ack_sync(journal)
    assert supervisor._answer_latency.snapshot()["count"] == 0, "unconsumed ACK recorded latency"
    clock[0] = 45.0
    if settle == "drain":
        assert writer.drain() == (), "read revalidation must not become a publication ACK"
    assert writer.read(path, "answer", wait=False) == value
    measured = supervisor._answer_latency.snapshot()
    assert measured["count"] == 1, "recovered answer lost its original publication timer"
    assert measured["max"] == 135.0, "latency must include arrival age and rejected/recovery time"

    # Evict only ACK evidence, then revalidate again. A transferred timer must be
    # consumed once even if the original write rejection is still cached.
    monkeypatch.setattr(channel, "_MAX_PENDING", 1)
    other = path.with_name("002.json")
    host._write_json(other, {"other": True})
    with pytest.raises(JournalBusy):
        writer.read(other, "other", wait=False)
    _ack_sync(journal)
    assert writer.read(other, "other", wait=False) == {"other": True}
    assert path not in writer._acked
    with pytest.raises(JournalBusy):
        writer.read(path, "answer", wait=False)
    _ack_sync(journal)
    clock[0] = 60.0
    assert writer.drain() == ()
    for _ in range(3):
        assert writer.read(path, "answer", wait=False) == value
    assert supervisor._answer_latency.snapshot() == measured


@pytest.mark.parametrize("mismatch", ["path", "digest"])
def test_revalidation_transfers_timer_only_for_matching_bytes_after_admission(
        tmp_path, monkeypatch, mismatch):
    supervisor, writer, journal, path, value, clock = _rejected_answer(tmp_path, monkeypatch)
    original = path.read_bytes()
    other = path.with_name("002.json") if mismatch == "path" else path
    host._write_json(other, value if mismatch == "path" else {"different": True},
                     replace_existing=mismatch == "digest")
    with pytest.raises(JournalBusy):
        writer.read(other, "other", wait=False)
    _ack_sync(journal)
    assert writer.read(other, "other", wait=False) is not None
    assert supervisor._answer_latency.snapshot()["count"] == 0, "unrelated bytes stole the timer"
    if mismatch == "digest":
        path.write_bytes(original)
    # A full journal must not consume the timer before accepting the sync intent.
    journal.full = True
    with pytest.raises(JournalBusy, match="full"):
        writer.read(path, "answer", wait=False)
    journal.full = False
    with pytest.raises(JournalBusy):
        writer.read(path, "answer", wait=False)
    _ack_sync(journal)
    clock[0] = 45.0
    assert writer.drain() == ()
    assert writer.read(path, "answer", wait=False) == value
    assert supervisor._answer_latency.snapshot()["count"] == 1
    assert supervisor._answer_latency.snapshot()["max"] == 135.0
