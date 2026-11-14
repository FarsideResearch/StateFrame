"""A stalled durable channel publication must not pin the channel service lane."""
import json
from pathlib import Path
import threading
import time

import pytest

from _timeouts import scaled

import stateframe.parallel_host as host
from stateframe.parallel_journal import Journal
from test_parallel_host import _request, _write_request
from test_parallel_bridge_journal_progress import PipeProcess


class _WriteProcess(PipeProcess):
    """Exercise descriptor-based journal I/O without an OS helper."""

    def __init__(self):
        def answer(message):
            assert message["kind"] == "write"
            payload = message["payload"]
            host._write_json(Path(payload["path"]), payload["value"],
                             replace_existing=payload["replace"], durable=payload["durable"])
            return {"id": message["id"], "ok": True}
        super().__init__(answer)


def _stop_runtime(supervisor):
    supervisor._stop_runtime()
    for thread in supervisor._runtime_threads:
        thread.join(scaled(3))
    writer = getattr(supervisor, "_channel_writes", None)
    journals = [supervisor._journal, *supervisor._seal_helpers]
    if writer is not None:
        journals.append(writer.journal)
    for journal in journals:
        reactor = getattr(journal, "_thread", None)
        if reactor is not None:
            reactor.join(scaled(3))


def test_stalled_channel_fsync_keeps_servicing_passes_and_preserves_record_before_answer(tmp_path, monkeypatch):
    run = tmp_path / "run"
    rq = run / "parallel/call/attempt-1/rq"
    rq.mkdir(parents=True)
    for seq in (1, 2):
        _write_request(rq / f"{seq:03d}.json", _request(1, "call", seq, "status", {"unit_id": "a"}))
    supervisor = host.ParallelSupervisor(run)
    entered, release, progressed = threading.Event(), threading.Event(), threading.Event()
    durable_writes = []
    service_threads = set()
    dispatches = []
    real_write = host._write_json

    def stalled_write(path, value, **kwargs):
        durable = kwargs.get("durable", True)
        if not durable:
            return  # cursors/metrics are unrelated to the durability gate
        if durable:
            durable_writes.append((path.name, threading.get_ident()))
        # Publish the bytes before holding the ACK, as happens during directory fsync.
        result = real_write(path, value, **dict(kwargs, durable=False))
        if durable:
            entered.set()
            release.wait()
        return result

    def spawn(journal):
        journal.epoch += 1
        return _WriteProcess()

    def dispatch(request, call_root):
        dispatches.append(request.seq)
        return "OK", None, {"revision": len(dispatches)}

    real_service = supervisor._service_channels
    stalled_passes = 0

    def service():
        nonlocal stalled_passes
        service_threads.add(threading.get_ident())
        if entered.is_set() and not release.is_set():
            stalled_passes += 1
            if stalled_passes >= 3:
                progressed.set()
        return real_service()

    monkeypatch.setattr(host, "_write_json", stalled_write)
    monkeypatch.setattr(Journal, "_spawn", spawn)
    monkeypatch.setattr(supervisor, "_dispatch", dispatch)
    monkeypatch.setattr(supervisor, "_service_channels", service)
    try:
        supervisor.tick()
        assert entered.wait(scaled(5)), "channel did not reach durable publication"
        assert progressed.wait(scaled(2)), "a channel fsync pinned the channel service thread"
        assert (rq.parent / "an/001.record.json").exists(), "the test did not expose the pre-ACK record"
        assert not (rq.parent / "an/001.json").exists(), "answer overtook its identity record ACK"
        release.set()
        deadline = time.monotonic() + scaled(5)
        while time.monotonic() < deadline and not all(
                (rq.parent / f"an/{seq:03d}.json").is_file() for seq in (1, 2)):
            time.sleep(0.01)
        for seq in (1, 2):
            answer = json.loads((rq.parent / f"an/{seq:03d}.json").read_bytes())
            record = json.loads((rq.parent / f"an/{seq:03d}.record.json").read_bytes())
            assert answer == record["answer"]
            names = [name for name, _ in durable_writes]
            assert names.index(f"{seq:03d}.record.json") < names.index(f"{seq:03d}.json")
        assert sorted(dispatches) == [1, 2], "pending publication re-dispatched an accepted request"
        assert all(thread_id not in service_threads for _, thread_id in durable_writes)
    finally:
        release.set()
        _stop_runtime(supervisor)


def test_durable_channel_writes_execute_off_the_channel_service_thread(tmp_path, monkeypatch):
    """Durable publication belongs to the journal helper, outside channel service."""
    rq = tmp_path / "run/parallel/call/attempt-1/rq"
    rq.mkdir(parents=True)
    _write_request(rq / "001.json", _request(1, "call", 1, "status", {"unit_id": "a"}))
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    durable_threads = []
    service_threads = set()
    real_write = host._write_json
    real_service = supervisor._service_channels

    def service():
        service_threads.add(threading.get_ident())
        return real_service()

    def write(path, value, **kwargs):
        if not kwargs.get("durable", True):
            return
        durable_threads.append(threading.get_ident())
        return real_write(path, value, **dict(kwargs, durable=False))

    def spawn(journal):
        journal.epoch += 1
        return _WriteProcess()

    monkeypatch.setattr(host, "_write_json", write)
    monkeypatch.setattr(Journal, "_spawn", spawn)
    monkeypatch.setattr(supervisor, "_dispatch", lambda *args: ("OK", None, {"state": "RUNNING"}))
    monkeypatch.setattr(supervisor, "_service_channels", service)
    try:
        supervisor.tick()
        answer_path = rq.parent / "an/001.json"
        deadline = time.monotonic() + scaled(5)
        while not answer_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert answer_path.is_file(), list(supervisor._protocol_errors)
        assert durable_threads
        assert service_threads.isdisjoint(durable_threads)
    finally:
        _stop_runtime(supervisor)


def test_failed_ack_keeps_visible_record_fenced_and_retries_frozen_intent(tmp_path, monkeypatch):
    import stateframe.parallel_channel as channel
    from stateframe.parallel_journal import JournalBusy, JournalFault
    from test_parallel_bridge import DelayedJournal

    journal = DelayedJournal()
    monkeypatch.setattr(channel, "Journal", lambda: journal)
    writer = channel.ChannelWrites()
    path = tmp_path / "record.json"
    value = {"revision": 1}
    with pytest.raises(JournalBusy):
        writer.write(path, value, replace_existing=False, wait=False)
    value["revision"] = 99
    first = journal.tickets[0]
    assert first.payload["value"] == {"revision": 1}
    host._write_json(path, first.payload["value"], durable=False)
    with pytest.raises(JournalBusy):
        writer.barrier(path)
    first.result = {"ok": False, "error": "directory fsync failed"}
    first.done.set()
    with pytest.raises(JournalFault, match="directory fsync failed"):
        writer.barrier(path)
    assert writer.drain() == ()
    with pytest.raises(JournalBusy):
        writer.barrier(path)
    retry = journal.tickets[1]
    assert retry.payload == first.payload
    retry.result = {"ok": True}
    retry.done.set()
    assert writer.drain() == (path,)
    writer.barrier(path)
    writer.write(path, {"revision": 1}, replace_existing=False, wait=False)
    assert len(journal.tickets) == 2
    with pytest.raises(host.ParallelHostError, match="immutable channel record already differs"):
        writer.write(path, value, replace_existing=False, wait=False)


def test_failed_channel_intents_remain_bounded_until_their_ack(tmp_path, monkeypatch):
    import stateframe.parallel_channel as channel
    from stateframe.parallel_journal import JournalBusy, JournalFault
    from test_parallel_bridge import DelayedJournal

    journal = DelayedJournal()
    monkeypatch.setattr(channel, "Journal", lambda: journal)
    monkeypatch.setattr(channel, "_MAX_PENDING", 1)
    writer = channel.ChannelWrites()
    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    with pytest.raises(JournalBusy):
        writer.write(first_path, {"value": 1}, replace_existing=False, wait=False)
    first = journal.tickets[0]
    first.result = {"ok": False, "error": "disk unavailable"}
    first.done.set()
    with pytest.raises(JournalFault):
        writer.barrier(first_path)
    with pytest.raises(JournalBusy, match="reservations are full"):
        writer.write(second_path, {"value": 2}, replace_existing=False, wait=False)
    assert len(journal.tickets) == 1
    with pytest.raises(JournalBusy):
        writer.barrier(first_path)
    retry = journal.tickets[1]
    host._write_json(first_path, retry.payload["value"], durable=False)
    retry.result = {"ok": True}
    retry.done.set()
    assert writer.drain() == (first_path,)
    with pytest.raises(JournalBusy):
        writer.write(second_path, {"value": 2}, replace_existing=False, wait=False)
    assert len(journal.tickets) == 3


def test_same_content_retry_repeats_a_failed_directory_fsync(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    attempts = []

    def directory_fsync(directory):
        attempts.append(directory)
        if len(attempts) == 1:
            raise OSError("directory fsync failed")

    monkeypatch.setattr(host.os, "fsync", lambda descriptor: None)
    monkeypatch.setattr(host, "_fsync_directory", directory_fsync)
    with pytest.raises(OSError, match="directory fsync failed"):
        host._write_json(path, {"revision": 1})
    assert path.exists(), "the failed ACK must leave the already-linked record visible"
    host._write_json(path, {"revision": 1})
    assert attempts == [path.parent, path.parent]
    assert json.loads(path.read_bytes()) == {"revision": 1}


@pytest.mark.parametrize("kind", ["malformed", "conflict"])
def test_pending_protocol_publication_does_not_abort_other_sequences(tmp_path, monkeypatch, kind):
    from stateframe.parallel_journal import JournalBusy

    rq = tmp_path / "run/parallel/call/attempt-1/rq"
    rq.mkdir(parents=True)
    if kind == "malformed":
        (rq / "001.json").write_text("{")
    else:
        _write_request(rq / "001.json", _request(1, "call", 1, "status", {"unit_id": "a"}))
        _write_request(rq / "1.json", _request(1, "call", 1, "status", {"unit_id": "b"}))
    _write_request(rq / "002.json", _request(1, "call", 2, "status", {"unit_id": "a"}))
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    busy = True
    dispatches = []

    def write(path, value, **kwargs):
        if path.name == "001.record.json" and busy:
            raise JournalBusy("durable ACK pending")
        host._write_json(path, value, **dict(kwargs, durable=False))

    def dispatch(request, call_root):
        dispatches.append(request.seq)
        return "OK", None, {"state": "RUNNING"}

    monkeypatch.setattr(supervisor, "_write_json", write)
    monkeypatch.setattr(supervisor, "_dispatch", dispatch)
    assert [(answer.seq, answer.status) for answer in supervisor.serve(rq, rq.parent / "an")] == [(2, "OK")]
    assert not (rq.parent / "an/001.record.json").exists()
    busy = False
    answers = supervisor.serve(rq, rq.parent / "an")
    assert sorted((answer.seq, answer.status) for answer in answers) == [(1, "ERROR"), (2, "OK")]
    assert dispatches == [2]


def test_seal_materialization_waits_for_accepted_job_ack(tmp_path, monkeypatch):
    from stateframe.parallel_host import ProcessInspectionDeferred
    from test_parallel_bridge import DelayedJournal
    from test_parallel_bridge_lane import _exited_sealable_worker

    entered, release, acked = threading.Event(), threading.Event(), threading.Event()
    progressed, premature, admitted = threading.Event(), threading.Event(), threading.Event()
    real_write, real_take = host._write_json, Journal.take
    job_path = None

    def write(path, value, **kwargs):
        if not kwargs.get("durable", True):
            return
        result = real_write(path, value, **dict(kwargs, durable=False))
        if path == job_path:
            entered.set()
            release.wait()
        return result

    def take(journal, ticket):
        result = real_take(journal, ticket)
        if result is not None and result.get("ok") is True and ticket.payload.get("path") == str(job_path):
            acked.set()
        return result

    def spawn(journal):
        journal.epoch += 1
        return _WriteProcess()

    monkeypatch.setattr(host, "_write_json", write)
    monkeypatch.setattr(Journal, "take", take)
    monkeypatch.setattr(Journal, "_spawn", spawn)
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    call = _exited_sealable_worker(tmp_path, supervisor)
    job_path = call / "host/seals/a-1.json"
    supervisor._seal_helpers = [DelayedJournal(), DelayedJournal()]
    real_advance = supervisor._advance_seals
    stalled_passes = 0

    def advance():
        nonlocal stalled_passes
        if entered.is_set() and not acked.is_set():
            stalled_passes += 1
            if stalled_passes >= 3:
                progressed.set()
        real_advance()

    def proof(*args):
        if not acked.is_set():
            premature.set()
        else:
            admitted.set()
        raise ProcessInspectionDeferred("no materialization needed in this test")

    monkeypatch.setattr(supervisor, "_advance_seals", advance)
    monkeypatch.setattr(supervisor, "_fresh_seal_proof", proof)
    results, errors = [], []

    def enqueue():
        try:
            results.append(supervisor._enqueue_seal(
                _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call))
        except Exception as error:
            errors.append(error)

    foreground = threading.Thread(target=enqueue)
    try:
        supervisor.tick()
        foreground.start()
        assert entered.wait(scaled(5))
        assert progressed.wait(scaled(2)), "the lifecycle lane did not progress during the job write"
        assert not acked.is_set()
        assert not premature.is_set(), "seal materialization overtook its accepted job ACK"
        release.set()
        foreground.join(scaled(5))
        assert not foreground.is_alive()
        assert errors == []
        assert len(results) == 1 and results[0][0] == "PENDING"
        assert admitted.wait(scaled(5)), "durable job was never admitted for materialization"
    finally:
        release.set()
        _stop_runtime(supervisor)
        if foreground.ident is not None:
            foreground.join(scaled(3))


def test_existing_record_requires_a_journal_ack_before_reuse(tmp_path, monkeypatch):
    import stateframe.parallel_channel as channel
    from stateframe.parallel_journal import JournalBusy
    from test_parallel_bridge import DelayedJournal

    journal = DelayedJournal()
    monkeypatch.setattr(channel, "Journal", lambda: journal)
    writer = channel.ChannelWrites()
    path = tmp_path / "recovered.json"
    host._write_json(path, {"revision": 1}, durable=False)
    with pytest.raises(JournalBusy):
        writer.write(path, {"revision": 1}, replace_existing=False, wait=False)
    assert len(journal.tickets) == 1
    with pytest.raises(JournalBusy):
        writer.barrier(path)
    ticket = journal.tickets[0]
    ticket.result = {"ok": True}
    ticket.done.set()
    writer.write(path, {"revision": 1}, replace_existing=False, wait=False)
    assert len(journal.tickets) == 1
