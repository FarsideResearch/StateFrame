"""Engine-free durability gates for consumed channel bytes, retries, and typed rejections."""
import threading
from pathlib import Path

import pytest
from test_parallel_bridge import DelayedJournal
from test_parallel_bridge_lane import _exited_sealable_worker
from test_parallel_host import _request, _write_request

import stateframe.parallel_channel as channel
import stateframe.parallel_host as host
from stateframe.parallel_journal import JournalBusy, JournalFault, JournalProtocolError


class ControlledJournal(DelayedJournal):
    fault = None

    def __init__(self):
        super().__init__()
        self.consumed = []

    def take(self, ticket):
        if not ticket.done.is_set():
            return None
        result = super().take(ticket)
        if result is not None:
            self.consumed.append(ticket)
        return result

    def complete(self, ticket, result=None):
        if result is None:
            payload = ticket.payload
            if ticket.kind == "write":
                host._write_json(Path(payload["path"]), payload["value"],
                                 replace_existing=payload["replace"])
            result = {"ok": True}
        ticket.result = result
        ticket.done.set()


def _writer(monkeypatch):
    journal = ControlledJournal()
    monkeypatch.setattr(channel, "Journal", lambda: journal)
    monkeypatch.setattr(channel, "_RETRY_BACKOFF_SECONDS", 0.0, raising=False)
    return channel.ChannelWrites(), journal


def _attach(supervisor, writer):
    supervisor._channel_writes = writer
    supervisor._channel_thread = threading.current_thread()


def _seal(tmp_path, monkeypatch):
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    monkeypatch.setattr(supervisor, "_start_runtime", lambda: None)
    call = _exited_sealable_worker(tmp_path, supervisor)
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    assert supervisor._enqueue_seal(request, call)[0] == "PENDING"
    return supervisor, call, request


def test_recovered_seal_result_requires_acks_before_reuse(tmp_path, monkeypatch):
    """A fresh lane has no durability evidence, even when both recovered files are visible."""
    supervisor, call, request = _seal(tmp_path, monkeypatch)
    result_path = call / "host/seals/a-1.result.json"
    host._write_json(result_path, {"status": "OK", "payload": {"sealed": True}})
    writer, journal = _writer(monkeypatch)
    _attach(supervisor, writer)
    with pytest.raises(JournalBusy):
        supervisor._enqueue_seal(request, call)
    # Acknowledging only the job must not authorize consumption of the result.
    journal.complete(journal.tickets[-1])
    with pytest.raises(JournalBusy):
        supervisor._enqueue_seal(request, call)
    assert journal.tickets[-1].payload["path"] == str(result_path)
    journal.complete(journal.tickets[-1])
    assert supervisor._enqueue_seal(request, call) == ("OK", None, {"sealed": True})
    assert len(journal.consumed) == 2


def test_result_published_in_barrier_read_gap_is_not_consumed(tmp_path, monkeypatch):
    """Inject at the old check/read gap; consuming unacknowledged bytes must never return OK."""
    supervisor, call, request = _seal(tmp_path, monkeypatch)
    writer, journal = _writer(monkeypatch)
    _attach(supervisor, writer)
    job = call / "host/seals/a-1.json"
    with pytest.raises(JournalBusy):
        writer.write(job, host._read_json(job, "job"), replace_existing=False, wait=False)
    journal.complete(journal.tickets[-1])
    assert writer.drain() == (job,)
    result_path = call / "host/seals/a-1.result.json"
    real_exists = Path.exists
    attempted = False

    def exists(path):
        nonlocal attempted
        if path == result_path and not attempted:
            attempted = True
            if writer._lock.acquire(blocking=False):
                writer._lock.release()
                with pytest.raises(JournalBusy):
                    writer.write(path, {"status": "OK", "payload": {"unacked": True}},
                                 replace_existing=False, wait=False)
                host._write_json(path, journal.tickets[-1].payload["value"])
            # Otherwise the ACK-bound read owns admission: publication cannot enter this gap.
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", exists)
    status, _, payload = supervisor._enqueue_seal(request, call)
    assert attempted
    assert status != "OK" or payload != {"unacked": True}, "seal read escaped its ACK fence"


def test_read_verifies_digest_of_consumed_bytes_and_recovery(tmp_path, monkeypatch):
    writer, journal = _writer(monkeypatch)
    path = tmp_path / "record.json"
    host._write_json(path, {"revision": 1})
    with pytest.raises(JournalBusy):
        writer.read(path, "record", wait=False)
    journal.complete(journal.tickets[-1])
    assert writer.read(path, "record", wait=False) == {"revision": 1}
    # An old ACK for the path cannot authorize new bytes at that path.
    host._write_json(path, {"revision": 2}, replace_existing=True)
    with pytest.raises(JournalBusy):
        writer.read(path, "record", wait=False)
    journal.complete(journal.tickets[-1])
    assert writer.read(path, "record", wait=False) == {"revision": 2}
    assert len(journal.tickets) == 2


def test_scheduled_service_retries_visible_failed_answer_without_redispatch(tmp_path, monkeypatch):
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    rq = tmp_path / "run/parallel/call/attempt-1/rq"
    rq.mkdir(parents=True)
    an = rq.parent / "an"
    request = _request(1, "call", 1, "status", {"unit_id": "a"})
    _write_request(rq / "001.json", request)
    supervisor._store_answer(an, 1, supervisor._request_identity(request),
                             supervisor._answer(request, "OK", None, {"state": "RUNNING"}))
    answer_path = supervisor._answer_path(an, 1)
    writer, journal = _writer(monkeypatch)
    _attach(supervisor, writer)
    value = host._read_json(answer_path, "test answer")
    with pytest.raises(JournalBusy):
        writer.write(answer_path, value, replace_existing=False, wait=False)
    journal.complete(journal.tickets[0], {"ok": False, "error_class": "OSError",
                                          "error": "directory fsync failed"})
    monkeypatch.setattr(supervisor, "_dispatch", lambda *args: pytest.fail("durable identity redispatched"))
    monkeypatch.setattr(supervisor, "_discover_seals", lambda: None)
    monkeypatch.setattr(supervisor, "_flush_metrics", lambda: None)
    for _ in range(8):
        supervisor._service_channels()
        for ticket in journal.tickets:
            if not ticket.done.is_set():
                journal.complete(ticket)
    assert journal.tickets[0] in journal.consumed, "failed ACK retained its journal credit"
    assert len(journal.tickets) >= 2, "visible answer suppressed its durability retry"
    assert answer_path not in writer._pending
    assert writer._acked[answer_path] is not None
    assert not supervisor._protocol_errors


def test_drain_retries_an_abandoned_failed_intent_with_frozen_bytes(tmp_path, monkeypatch):
    writer, journal = _writer(monkeypatch)
    path = tmp_path / "abandoned.json"
    value = {"revision": 1}
    with pytest.raises(JournalBusy):
        writer.write(path, value, replace_existing=False, wait=False)
    value["revision"] = 99
    journal.complete(journal.tickets[0], {"ok": False, "error_class": "OSError", "error": "fsync failed"})
    for _ in range(4):
        writer.drain()
    assert len(journal.tickets) == 2, "drain must retry without the original caller"
    assert journal.tickets[1].payload == journal.tickets[0].payload
    journal.complete(journal.tickets[1])
    assert writer.drain() == (path,)
    assert not writer._pending
    assert host._read_json(path, "retried record") == {"revision": 1}


@pytest.mark.parametrize("error_class,exception", [
    ("ParallelHostError", host.ParallelHostError),
    ("ValueError", JournalProtocolError),
])
def test_terminal_rejection_keeps_type_and_never_resubmits(tmp_path, monkeypatch, error_class, exception):
    writer, journal = _writer(monkeypatch)
    path = tmp_path / "rejected.json"
    with pytest.raises(JournalBusy):
        writer.write(path, {"revision": 1}, replace_existing=False, wait=False)
    journal.complete(journal.tickets[0], {"ok": False, "error_class": error_class, "error": "rejected"})
    journal.fault = "an earlier helper exhausted its restart budget"
    for _ in range(4):
        with pytest.raises(exception, match="rejected"):
            writer.write(path, {"revision": 1}, replace_existing=False, wait=False)
        writer.drain()
    assert len(journal.tickets) == 1
    assert len(journal.consumed) == 1


def test_exhausted_journal_fault_never_gets_a_fresh_channel_restart_budget(tmp_path, monkeypatch):
    writer, journal = _writer(monkeypatch)
    path = tmp_path / "lost.json"
    with pytest.raises(JournalBusy):
        writer.write(path, {}, replace_existing=False, wait=False)
    journal.fault = "helper restart budget exhausted"
    journal.complete(journal.tickets[0], {"ok": False, "lost": True, "error": journal.fault})
    for _ in range(8):
        with pytest.raises(JournalFault, match="restart budget exhausted"):
            writer.write(path, {}, replace_existing=False, wait=False)
        writer.drain()
    assert len(journal.tickets) == 1
    assert len(journal.consumed) == 1


def test_repeated_lost_tickets_have_a_bounded_channel_retry_budget(tmp_path, monkeypatch):
    writer, journal = _writer(monkeypatch)
    path = tmp_path / "losses.json"
    for _ in range(20):
        try:
            writer.write(path, {}, replace_existing=False, wait=False)
        except (JournalBusy, JournalFault):
            pass
        for ticket in journal.tickets:
            if not ticket.done.is_set():
                journal.complete(ticket, {"ok": False, "lost": True, "error": "helper lost"})
        writer.drain()
    assert len(journal.tickets) <= 6, "channel resubmission kept resetting the loss budget"
    assert len(journal.consumed) == len(journal.tickets)


def test_conflicting_visible_seal_result_is_verified_before_completing(tmp_path, monkeypatch):
    supervisor, _, _ = _seal(tmp_path, monkeypatch)
    operation = next(iter(supervisor._seal_ops.values()))
    supervisor._complete_seal(operation, {"status": "OK", "payload": {"ours": True}})
    writer, journal = _writer(monkeypatch)
    _attach(supervisor, writer)
    supervisor._publish_seal_results()
    host._write_json(operation.result, {"status": "OK", "payload": {"other": True}})
    journal.complete(journal.tickets[0], {"ok": False, "error_class": "ParallelHostError",
                                          "error": "immutable channel record raced"})
    supervisor._publish_seal_results()
    assert operation.phase == "COMMITTING", "visible conflict was mistaken for durable authority"
    supervisor._publish_seal_results()
    assert len(journal.tickets) == 2
    assert journal.tickets[-1].kind != "write", "immutable conflict was retried instead of verified"
    journal.complete(journal.tickets[-1])
    supervisor._publish_seal_results()
    assert operation.phase == "DONE"
    assert operation.job in supervisor._sealed_jobs


def test_revalidation_fsyncs_exact_existing_bytes_without_rewriting(tmp_path, monkeypatch):
    import hashlib
    import os

    path = tmp_path / "recovered.json"
    raw = b'{ "revision" : 1 }\n'
    path.write_bytes(raw)
    inode = path.stat().st_ino
    synced = []
    real_fsync = os.fsync

    def fsync(descriptor):
        synced.append(os.fstat(descriptor).st_ino)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    channel.sync_record({"path": str(path), "digest": hashlib.sha256(raw).hexdigest()})
    assert synced == [inode, path.parent.stat().st_ino]
    assert path.stat().st_ino == inode
    assert path.read_bytes() == raw
    with pytest.raises(host.ParallelHostError, match="digest"):
        channel.sync_record({"path": str(path), "digest": hashlib.sha256(b"other").hexdigest()})
    assert synced == [inode, path.parent.stat().st_ino], "changed bytes received durability evidence"


def test_channel_identity_record_recovery_requires_its_ack(tmp_path, monkeypatch):
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    an = tmp_path / "an"
    request = _request(1, "call", 1, "status", {"unit_id": "a"})
    supervisor._store_answer(an, 1, supervisor._request_identity(request),
                             supervisor._answer(request, "OK", None, {}))
    writer, journal = _writer(monkeypatch)
    _attach(supervisor, writer)
    with pytest.raises(JournalBusy):
        supervisor._channel_record(an, 1)
    journal.complete(journal.tickets[-1])
    assert supervisor._channel_record(an, 1)["request_sha256"] == supervisor._request_identity(request)


def test_recovered_read_uses_the_real_journal_helper(tmp_path):
    """Exercise the helper protocol too, without starting an engine or substituting fsync."""
    path = tmp_path / "record.json"
    path.write_bytes(b'{ "revision" : 1 }\n')
    writer = channel.ChannelWrites()
    try:
        assert writer.read(path, "record", wait=True) == {"revision": 1}
        assert writer.read(path, "record", wait=False) == {"revision": 1}
        assert not writer._pending
    finally:
        writer.close()
        writer.journal._thread.join(5)
        assert not writer.journal._thread.is_alive()


def test_pre_runtime_recovered_record_is_durably_revalidated(tmp_path, monkeypatch):
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    an = tmp_path / "an"
    request = _request(1, "call", 1, "status", {"unit_id": "a"})
    supervisor._store_answer(an, 1, supervisor._request_identity(request),
                             supervisor._answer(request, "OK", None, {}))
    synced = []
    real_sync = host._fsync_directory

    def sync(path):
        synced.append(path)
        real_sync(path)

    monkeypatch.setattr(host, "_fsync_directory", sync)
    assert supervisor._channel_writes is None
    assert supervisor._channel_record(an, 1)["request_sha256"] == supervisor._request_identity(request)
    assert synced == [an], "pre-runtime recovery bypassed durability validation"


def test_revalidation_ack_is_not_a_publication_completion(tmp_path, monkeypatch):
    """A sync of visible bytes must not release a seal reservation for another write intent."""
    writer, journal = _writer(monkeypatch)
    path = tmp_path / "seal-job.json"
    host._write_json(path, {"operation_id": "other"})
    with pytest.raises(JournalBusy):
        writer.read(path, "seal job", wait=False)
    journal.complete(journal.tickets[-1])
    assert writer.drain() == (), "revalidation ACK masqueraded as publication of a reserved job"
    assert not writer._pending
    assert len(journal.consumed) == 1
    assert writer.read(path, "seal job", wait=False) == {"operation_id": "other"}


def test_exhausted_file_retry_releases_credit_without_poisoning_other_channels(tmp_path, monkeypatch):
    writer, journal = _writer(monkeypatch)
    monkeypatch.setattr(channel, "_MAX_PENDING", 1)
    failed, healthy = tmp_path / "failed.json", tmp_path / "healthy.json"
    with pytest.raises(JournalBusy):
        writer.write(failed, {"revision": 1}, replace_existing=False, wait=False)
    for _ in range(8):
        for ticket in journal.tickets:
            if not ticket.done.is_set():
                journal.complete(ticket, {"ok": False, "error_class": "PermissionError",
                                          "error": "directory fsync denied"})
        writer.drain()
    assert len(journal.tickets) == 3
    assert failed not in writer._pending
    with pytest.raises(JournalFault, match="directory fsync denied"):
        writer.write(failed, {"revision": 1}, replace_existing=False, wait=False)
    with pytest.raises(JournalBusy):
        writer.write(healthy, {"revision": 2}, replace_existing=False, wait=False)
    journal.complete(journal.tickets[-1])
    assert writer.drain() == (healthy,)
    assert host._read_json(healthy, "healthy channel") == {"revision": 2}
