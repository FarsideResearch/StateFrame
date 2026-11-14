"""Engine-free regressions for historical completion admission and ACK-cache churn."""
from pathlib import Path
import threading

import pytest

import stateframe.parallel_channel as channel
import stateframe.parallel_host as host
from stateframe.parallel_journal import JournalBusy
from test_parallel_bridge_r3_chan import ControlledJournal
from test_parallel_host import _request, _write_request


def _recovered_channel(tmp_path, monkeypatch, count=64, attempts=1):
    run = tmp_path / "run"
    rq = run / "parallel/call/attempt-1/rq"
    rq.mkdir(parents=True)
    an = rq.parent / "an"
    seed = host.ParallelSupervisor(run)
    for attempt in range(1, attempts + 1):
        requests = rq.parent.parent / f"attempt-{attempt}/rq"
        requests.mkdir(parents=True, exist_ok=True)
        for seq in range(1, count + 1):
            request = _request(attempt, "call", seq, "status", {"unit_id": "a"})
            _write_request(requests / f"{seq:03d}.json", request)
            seed._store_answer(requests.parent / "an", seq, seed._request_identity(request),
                               seed._answer(request, "OK", None, {"state": "RUNNING"}))
    # Recovered bytes have no runtime durability authority in this supervisor.
    supervisor = host.ParallelSupervisor(run)
    journal = ControlledJournal()
    monkeypatch.setattr(channel, "Journal", lambda: journal)
    writer = channel.ChannelWrites()
    supervisor._channel_writes = writer
    supervisor._channel_thread = threading.current_thread()
    dispatches = []

    def dispatch(request, call_root):
        dispatches.append(request.seq)
        return "OK", None, {"state": "RUNNING"}

    monkeypatch.setattr(supervisor, "_dispatch", dispatch)
    return supervisor, rq, an, writer, journal, dispatches


def _pass(supervisor, rq, an, writer):
    writer.drain()
    supervisor._tick_deadline = float("inf")
    return supervisor.serve(rq, an)


def _ack(journal):
    for ticket in journal.tickets:
        if not ticket.done.is_set():
            if ticket.kind == "sync":
                channel.sync_record(ticket.payload)
            journal.complete(ticket)


def test_historical_cache_churn_cannot_starve_new_publication(tmp_path, monkeypatch):
    supervisor, rq, an, writer, journal, dispatches = _recovered_channel(tmp_path, monkeypatch)
    _write_request(rq / "065.json", _request(1, "call", 65, "status", {"unit_id": "a"}))
    for _ in range(12):
        _pass(supervisor, rq, an, writer)
        _ack(journal)
    writer.drain()
    answer = supervisor._answer_path(an, 65)
    assert any(ticket.kind == "write" and Path(ticket.payload["path"]) == answer
               for ticket in journal.consumed), "history prevented consumption of the new publication ACK"
    assert len(writer._acked) <= 64
    assert host._read_json(answer, "new answer")["seq"] == 65
    assert dispatches == [65], "publication pressure redispatched an accepted request"


@pytest.mark.parametrize("attempts", [1, 16])
def test_stalled_recovery_reserves_capacity_for_unanswered_requests(tmp_path, monkeypatch, attempts):
    count = 256 // attempts
    supervisor, rq, an, writer, journal, _ = _recovered_channel(
        tmp_path, monkeypatch, count, attempts)
    # Do not ACK recovery, even as the request cursor advances over many passes.
    for _ in range(8):
        for attempt in range(1, attempts + 1):
            requests = rq.parent.parent / f"attempt-{attempt}/rq"
            _pass(supervisor, requests, requests.parent / "an", writer)
    assert len(writer._pending) <= 16, "history exhausted the bounded publication lane"
    seq = count + 1
    _write_request(rq / f"{seq:03d}.json", _request(1, "call", seq, "status", {"unit_id": "a"}))
    for _ in range(8):
        _pass(supervisor, rq, an, writer)
    assert any(ticket.kind == "write" and Path(ticket.payload["path"]) ==
               supervisor._record_path(an, seq) for ticket in journal.tickets), (
                   "unacknowledged history prevented admission of a new identity publication")


@pytest.mark.parametrize("recovered", [True, False], ids=["recovered", "published"])
def test_confirmed_completion_survives_writer_ack_cache_eviction(tmp_path, monkeypatch, recovered):
    supervisor, rq, an, writer, journal, dispatches = _recovered_channel(
        tmp_path, monkeypatch, 1 if recovered else 0)
    if not recovered:
        _write_request(rq / "001.json", _request(1, "call", 1, "status", {"unit_id": "a"}))
    for _ in range(4):
        if recovered:
            _pass(supervisor, rq, an, writer)
        else:
            writer.drain()
            supervisor._tick_deadline = None
            supervisor.serve(rq, an)
        _ack(journal)
    writer.drain()
    assert supervisor._answer_path(an, 1) in writer._acked
    for index in range(64):
        path = tmp_path / f"other-{index}.json"
        host._write_json(path, {"index": index})
        with pytest.raises(JournalBusy):
            writer.read(path, "unrelated record", wait=False)
        _ack(journal)
        writer.drain()
    assert supervisor._answer_path(an, 1) not in writer._acked
    before = len(journal.tickets)
    for _ in range(4):
        _pass(supervisor, rq, an, writer)
        _ack(journal)
    assert len(journal.tickets) == before, "confirmed history restarted after ACK-cache eviction"
    assert dispatches == ([] if recovered else [1])


def test_recovery_rotates_to_all_history_and_then_stays_idle(tmp_path, monkeypatch):
    supervisor, rq, an, writer, journal, dispatches = _recovered_channel(tmp_path, monkeypatch)
    for _ in range(48):
        _pass(supervisor, rq, an, writer)
        _ack(journal)
    writer.drain()
    paths = {Path(ticket.payload["path"]) for ticket in journal.tickets}
    assert paths == {path for seq in range(1, 65) for path in
                     (supervisor._record_path(an, seq), supervisor._answer_path(an, seq))}
    assert len(journal.tickets) == 128, "history did not converge after one ACK per immutable record"
    before = len(journal.tickets)
    for _ in range(4):
        _pass(supervisor, rq, an, writer)
    assert len(journal.tickets) == before
    assert not writer._pending
    assert dispatches == []


def test_completion_filter_stops_at_the_channel_deadline(tmp_path, monkeypatch):
    supervisor, rq, an, _, _, _ = _recovered_channel(tmp_path, monkeypatch)
    clock = [0.0]
    checked = []
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])

    def complete(an_dir, seq):
        checked.append(seq)
        clock[0] += 0.02
        return True

    monkeypatch.setattr(supervisor, "_channel_answer_complete", complete)
    supervisor._tick_deadline = 0.05
    assert supervisor.serve(rq, an) == ()
    assert len(checked) == 3, "completion filtering ran outside the channel work deadline"
    assert clock[0] == pytest.approx(0.06)
