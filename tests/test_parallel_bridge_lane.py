"""Engine-free checks for the channel lane's independence from the commit lane: channel writes
never queue behind worker commits, one deferred request never aborts a pass or becomes a durable
ERROR, and a launch answer polls its durable ACK instead of sleeping on the channel thread."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

import stateframe.parallel_host as host
from stateframe.parallel_host import ParallelSupervisor, ProcessInspectionDeferred
from stateframe.parallel_journal import JournalBusy


class _SaturatedJournal:
    """A commit lane with every credit held by worker commits: any channel use is the failure."""
    epoch = 1

    def __init__(self):
        self.channel_calls = 0

    def submit(self, kind, payload):
        self.channel_calls += 1
        raise JournalBusy("journal queue is full")

    def call(self, kind, payload):
        self.channel_calls += 1
        raise JournalBusy("journal queue is full")

    def take(self, ticket):
        return None

    def close(self):
        pass


def _channel(run: Path):
    rq = run / "parallel" / "call" / "attempt-1" / "rq"
    rq.mkdir(parents=True, exist_ok=True)
    return rq, rq.parent / "an"


def _status_supervisor(tmp_path):
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    supervisor._workers[("call", "a", 1)] = worker
    worker.status_view = supervisor._status_value(worker)
    return supervisor, worker


def _write(rq: Path, request) -> None:
    from test_parallel_host import _write_request
    _write_request(rq / f"{request.seq:03d}.json", request)


# ---------------------------------------------------------------- body 1: lane independence

def test_channel_writes_never_enter_the_commit_journal(tmp_path):
    supervisor = ParallelSupervisor(tmp_path / "run")
    journal = supervisor._journal = _SaturatedJournal()
    durable, cursor = tmp_path / "an" / "001.record.json", tmp_path / "channel-cursor.json"
    supervisor._write_json(durable, {"a": 1})
    supervisor._write_json(cursor, {"last": 1}, replace_existing=True, durable=False)
    supervisor._write_json(cursor, {"last": 2}, replace_existing=True, durable=False)
    assert json.loads(durable.read_text()) == {"a": 1}
    assert json.loads(cursor.read_text()) == {"last": 2}
    assert journal.channel_calls == 0
    with pytest.raises(host.ParallelHostError, match="immutable channel record already differs"):
        supervisor._write_json(durable, {"a": 2})


def test_status_is_answered_while_the_commit_lane_is_saturated(tmp_path):
    from test_parallel_host import _request
    supervisor, worker = _status_supervisor(tmp_path)
    supervisor._journal = _SaturatedJournal()
    rq, an = _channel(tmp_path / "run")
    _write(rq, _request(1, "call", 1, "status", {"unit_id": "a"}))
    started = time.monotonic()
    answers = supervisor.serve(rq, an)
    assert time.monotonic() - started < 0.5
    assert [(a.seq, a.status) for a in answers] == [(1, "OK")]
    published = json.loads((an / "001.json").read_text())
    assert published["payload"]["state"] == "RUNNING"
    assert (an / "001.record.json").is_file()
    assert json.loads((rq.parent / "channel-cursor.json").read_text()) == {"last": 1}


def test_journal_busy_in_one_request_does_not_abort_the_pass(tmp_path, monkeypatch):
    from test_parallel_host import _request
    supervisor, worker = _status_supervisor(tmp_path)
    rq, an = _channel(tmp_path / "run")
    for seq in (1, 2, 3):
        _write(rq, _request(1, "call", seq, "status", {"unit_id": "a"}))
    real_dispatch = supervisor._dispatch
    busy = {2}

    def dispatch(request, call_root):
        if request.seq in busy:
            raise JournalBusy("journal queue is full")
        return real_dispatch(request, call_root)
    monkeypatch.setattr(supervisor, "_dispatch", dispatch)
    answers = supervisor.serve(rq, an)
    assert sorted(a.seq for a in answers) == [1, 3]
    assert (an / "001.record.json").is_file() and (an / "003.record.json").is_file()
    assert not (an / "002.record.json").exists() and not (an / "002.json").exists()
    busy.clear()
    supervisor._tick_deadline = time.monotonic() + 10  # a scheduled pass skips completed answers
    answers = supervisor.serve(rq, an)
    assert [(a.seq, a.status) for a in answers] == [(2, "OK")]
    assert json.loads((an / "002.json").read_text())["payload"]["state"] == "RUNNING"


# ---------------------------------------------------------------- body 2: deferrals stay pending

def _exited_sealable_worker(tmp_path, supervisor):
    from test_parallel_bridge_channel import _exited_worker
    from test_parallel_host import _spec_and_snapshot
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(tmp_path / "run")
    worker = _exited_worker(call, unit, snapshot, spec_path, spec, 1)
    worker.group_verified = True
    supervisor._persist_worker(worker)
    assert worker.state == "EXITED" and worker.exited_seq == 1
    return call


def test_full_seal_admission_leaves_the_request_unanswered_then_answers_pending(tmp_path, monkeypatch):
    from test_parallel_host import _request
    supervisor = ParallelSupervisor(tmp_path / "run")
    monkeypatch.setattr(supervisor, "_start_runtime", lambda: None)
    call = _exited_sealable_worker(tmp_path, supervisor)
    for index in range(32):
        supervisor._seal_ops[Path(f"held-{index}")] = SimpleNamespace(worker=None, phase="BUILDING")
    rq, an = _channel(tmp_path / "run")
    _write(rq, _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}))
    assert supervisor.serve(rq, an) == ()
    assert not (an / "001.record.json").exists() and not (an / "001.json").exists()
    assert not (call / "host" / "seals" / "a-1.json").exists()
    supervisor._seal_ops.pop(Path("held-0"))
    (answer,) = supervisor.serve(rq, an)
    assert answer.status == "PENDING"
    assert answer.result_id == str(call / "host" / "seals" / "a-1.result.json")
    assert json.loads((an / "001.json").read_text())["status"] == "PENDING"
    assert (call / "host" / "seals" / "a-1.json").is_file()


@pytest.mark.parametrize("error", [
    ProcessInspectionDeferred("process table unavailable"),
    host.ParallelHostError("wrapped"),
])
def test_any_deferral_inside_dispatch_is_never_a_durable_error(tmp_path, monkeypatch, error):
    import subprocess
    from test_parallel_host import _request
    if not isinstance(error, ProcessInspectionDeferred):
        error.__cause__ = subprocess.TimeoutExpired("scan", 1.0)
    supervisor, worker = _status_supervisor(tmp_path)
    rq, an = _channel(tmp_path / "run")
    _write(rq, _request(1, "call", 1, "status", {"unit_id": "a"}))
    _write(rq, _request(1, "call", 2, "status", {"unit_id": "a"}))
    real_dispatch = supervisor._dispatch
    deferred = {1}

    def dispatch(request, call_root):
        if request.seq in deferred:
            raise error
        return real_dispatch(request, call_root)
    monkeypatch.setattr(supervisor, "_dispatch", dispatch)
    answers = supervisor.serve(rq, an)
    assert [(a.seq, a.status) for a in answers] == [(2, "OK")]
    assert not (an / "001.record.json").exists() and not (an / "001.json").exists()
    deferred.clear()
    supervisor._tick_deadline = time.monotonic() + 10  # a scheduled pass skips completed answers
    (answer,) = supervisor.serve(rq, an)
    assert answer.seq == 1 and answer.status == "OK"
    assert json.loads((an / "001.json").read_text())["status"] == "OK"


# ---------------------------------------------------------------- body 3: launch ACK is polled

def _pending_launch(tmp_path, monkeypatch):
    from test_parallel_bridge import DelayedJournal
    from test_parallel_host import _request
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    journal = supervisor._journal = DelayedJournal()
    worker.state, worker.handle = "STARTING", None
    supervisor._persist_worker(worker, launch=True)
    assert worker.commit_intent is not None and len(journal.tickets) == 1
    stored = []
    monkeypatch.setattr(supervisor, "_dispatch", lambda *a: ("PENDING", "launch-op", {"state": "STARTING"}))
    monkeypatch.setattr(supervisor, "_channel_record", lambda *a: None)
    monkeypatch.setattr(supervisor, "_store_answer", lambda an, seq, digest, answer: stored.append(answer))
    request = _request(1, "call", 1, "launch", {"unit_id": "a"})

    def serve_once():
        started = time.monotonic()
        try:
            supervisor._serve_request(request, worker.call_root, worker.call_root / "attempt-1/an", 1)
            outcome = "answered"
        except JournalBusy as error:
            outcome = error
        return outcome, time.monotonic() - started
    return supervisor, worker, journal, stored, serve_once


def test_launch_ack_wait_constant_is_gone():
    assert not hasattr(host, "_LAUNCH_ACK_WAIT_SECONDS")


def test_unacked_launch_leaves_the_sequence_unanswered_without_sleeping(tmp_path, monkeypatch):
    supervisor, worker, journal, stored, serve_once = _pending_launch(tmp_path, monkeypatch)
    for _ in range(3):
        outcome, elapsed = serve_once()
        assert isinstance(outcome, JournalBusy) and "retry pending" in str(outcome)
        assert elapsed < 0.1, "the channel thread must never sleep on a launch ACK"
    assert stored == []
    assert supervisor._runtime_stop is False


def test_launch_is_answered_on_the_pass_after_its_ack(tmp_path, monkeypatch):
    supervisor, worker, journal, stored, serve_once = _pending_launch(tmp_path, monkeypatch)
    outcome, elapsed = serve_once()
    assert isinstance(outcome, JournalBusy) and elapsed < 0.1
    ticket = journal.tickets[0]
    ticket.result = {"ok": True, "value": dict(ticket.payload["value"])}
    supervisor._pump_commits()
    assert worker.launch_committed and worker.launch_commit_event.is_set()
    outcome, elapsed = serve_once()
    assert outcome == "answered" and elapsed < 0.1
    assert [(a.status, a.result_id) for a in stored] == [("PENDING", "launch-op")]


def test_terminally_rejected_launch_commit_answers_error(tmp_path, monkeypatch):
    supervisor, worker, journal, stored, serve_once = _pending_launch(tmp_path, monkeypatch)
    passes = 0
    while worker.commit_intent is not None:
        passes += 1
        assert passes <= 4, "a durably rejected launch must be dropped within the rejection limit"
        outcome, elapsed = serve_once()  # the channel pass before this lifecycle pass
        assert isinstance(outcome, JournalBusy) and elapsed < 0.1, \
            "no channel pass may sleep or answer while the intent is still being retried"
        ticket = journal.tickets[-1]
        ticket.result = {"ok": False, "error": "ParallelHostError: terminal commit identity conflict"}
        worker.commit_retry_at = 0.0  # the lifecycle lane's backoff is not under test
        supervisor._pump_commits()
    assert stored == []
    assert not worker.launch_committed and not worker.launch_commit_event.is_set()
    outcome, elapsed = serve_once()
    assert outcome == "answered" and elapsed < 0.1
    (answer,) = stored
    assert answer.status == "ERROR" and answer.result_id is None
    assert "durable worker commit rejected" in answer.payload["error"]
    assert "terminal commit identity conflict" in answer.payload["error"]
