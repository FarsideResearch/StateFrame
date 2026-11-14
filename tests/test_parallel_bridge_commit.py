"""Engine-free regression checks for terminal commit integrity under cancel, rejection and shutdown."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

import stateframe.parallel_host as host
from stateframe.parallel_host import ParallelHostError, ParallelSupervisor

from test_parallel_bridge import DelayedJournal
from test_parallel_supervisor_scale import _CountingHandle, _finalizing_worker


class _DoneHandle(_CountingHandle):
    """A leader that has already exited: the cancel path must not re-signal it."""

    def poll(self):
        self.polls += 1
        return 0


def _exiting_worker(tmp_path, handle):
    supervisor, worker = _finalizing_worker(tmp_path, handle)
    supervisor._workers[("call", "a", 1)] = worker
    worker.status_view = supervisor._status_value(worker)
    journal = supervisor._journal = DelayedJournal()
    worker.state, worker.terminal, worker.group_verified = "EXITED", "SUBMITTED", True
    supervisor._persist_worker(worker)
    assert worker.state == "EXITING" and worker.commit_version == 1
    return supervisor, worker, journal


def _status(supervisor, worker):
    return supervisor._status(SimpleNamespace(call_id="call", payload={"unit_id": "a"}), worker.call_root)


def _cancel(supervisor, worker):
    request = SimpleNamespace(call_id="call", payload={"unit_id": "a", "reason": "parent cancelled"})
    return supervisor._cancel(request, worker.call_root)


# ---- body 3: cancel while EXITING -------------------------------------------------------------

def test_cancel_while_exiting_keeps_the_first_terminal_commit(tmp_path):
    supervisor, worker, journal = _exiting_worker(tmp_path, _DoneHandle())
    worker.pid = worker.pgid = None  # a record-less exit takes the direct fence branch
    generation = worker.generation
    assert _cancel(supervisor, worker) == {"state": "EXITING"}
    # The SUBMITTED intent is still the one and only terminal record for this incarnation.
    assert worker.commit_version == 1 and worker.generation == generation
    assert worker.commit_intent["value"]["terminal"] == "SUBMITTED"
    assert worker.state == "EXITING" and _status(supervisor, worker)["state"] == "RUNNING"
    ticket = journal.tickets[0]
    ticket.result = {"ok": True, "value": dict(ticket.payload["value"], exited_seq=3)}
    supervisor._pump_commits()
    assert worker.state == "EXITED" and worker.terminal == "SUBMITTED" and worker.exited_seq == 3
    assert _status(supervisor, worker)["state"] == "EXITED"
    assert len(journal.tickets) == 1


def test_cancel_of_exited_worker_is_a_no_op_without_a_generation_bump(tmp_path):
    supervisor, worker = _finalizing_worker(tmp_path, _DoneHandle())
    supervisor._workers[("call", "a", 1)] = worker
    worker.state, worker.terminal, worker.exited_seq = "EXITED", "SUBMITTED", 1
    worker.finished_epoch = time.time()
    worker.status_view = supervisor._status_value(worker)
    assert _cancel(supervisor, worker) == {"state": "EXITED"}
    assert worker.generation == 0 and worker.terminal == "SUBMITTED" and worker.commit_intent is None


def test_fence_refuses_a_worker_whose_exit_is_being_committed(tmp_path):
    supervisor, worker, _ = _exiting_worker(tmp_path, _DoneHandle())
    with pytest.raises(ParallelHostError, match="being committed"):
        supervisor._fence_worker_record(worker, "late cancel")
    assert worker.state == "EXITING" and worker.commit_version == 1


# ---- body 4: permanent helper rejection ------------------------------------------------------

def _reject(journal, supervisor, worker, message, **extra):
    ticket = journal.tickets[-1]
    ticket.result = {"ok": False, "error": message, **extra}
    supervisor._pump_commits()


def test_durable_rejection_is_bounded_with_backoff_and_reaches_status(tmp_path):
    supervisor, worker, journal = _exiting_worker(tmp_path, _DoneHandle())
    _reject(journal, supervisor, worker, "ParallelHostError: terminal commit identity conflict")
    # The first rejection gets one immediate retry (a transient fault clears on the next ACK).
    assert worker.commit_intent is not None and worker.commit_ticket is not None
    assert len(journal.tickets) == 2
    _reject(journal, supervisor, worker, "ParallelHostError: terminal commit identity conflict")
    assert worker.commit_ticket is None and len(journal.tickets) == 2
    assert worker.commit_retry_at > time.monotonic(), "a repeated rejection must back off, not hot-retry"
    supervisor._pump_commits()
    assert worker.commit_ticket is None and len(journal.tickets) == 2, "resubmitted inside the backoff"
    for _ in range(host._COMMIT_REJECTION_LIMIT - 2):
        worker.commit_retry_at = 0.0
        supervisor._pump_commits()
        assert worker.commit_ticket is not None
        _reject(journal, supervisor, worker, "ParallelHostError: terminal commit identity conflict")
    # The rejection is classified permanent: no further retries, and the client sees the fault.
    assert worker.commit_intent is None and worker.commit_ticket is None
    worker.commit_retry_at = 0.0
    supervisor._pump_commits()
    assert len(journal.tickets) == host._COMMIT_REJECTION_LIMIT
    with pytest.raises(ParallelHostError, match="rejected.*identity conflict"):
        _status(supervisor, worker)
    # The incarnation is over (EXITED without a durable sequence), not EXITING forever.
    assert worker.exited_seq is None and worker.state == "EXITED"
    assert any("identity conflict" in error for error in supervisor._protocol_errors)


def test_lost_helper_and_superseded_acks_are_not_counted_as_rejections(tmp_path):
    supervisor, worker, journal = _exiting_worker(tmp_path, _DoneHandle())
    # Lost ACKs never count toward the rejection limit; they retain the intent and back off
    # (the host's own loss count ends the run only after _JOURNAL_LOSS_LIMIT consecutive losses).
    for _ in range(host._JOURNAL_LOSS_LIMIT - 1):
        worker.commit_retry_at = 0.0
        supervisor._pump_commits()
        _reject(journal, supervisor, worker, "seal helper lost", lost=True)
        assert worker.commit_intent is not None and "pending" in worker.error
        assert worker.commit_failures == 0
    # A superseded intent's ACK says nothing about the current one and counts nowhere.
    worker.commit_retry_at = 0.0
    supervisor._pump_commits()
    stale = journal.tickets[-1]
    worker.commit_version += 1
    stale.result = {"ok": False, "error": "ParallelHostError: terminal commit identity conflict"}
    supervisor._pump_commits()
    assert worker.commit_failures == 0 and worker.commit_intent is not None
    assert supervisor._journal_lost_acks == 0, "a delivered ACK resets the loss count"
    assert worker.commit_ticket is not None


def test_malformed_acks_follow_the_same_bounded_classification(tmp_path):
    supervisor, worker, journal = _exiting_worker(tmp_path, _DoneHandle())
    for attempt in range(host._COMMIT_REJECTION_LIMIT):
        worker.commit_retry_at = 0.0
        supervisor._pump_commits()
        journal.tickets[-1].result = {"ok": True, "value": {"state": "EXITED", "exited_seq": 9}}
        supervisor._pump_commits()
        assert worker.exited_seq is None
        if attempt + 1 < host._COMMIT_REJECTION_LIMIT:
            assert worker.commit_intent is not None
    assert worker.commit_intent is None
    with pytest.raises(ParallelHostError, match="identity mismatch"):
        _status(supervisor, worker)


# ---- body 9: shutdown and resume ---------------------------------------------------------------

def test_reap_pumps_until_the_journal_is_drained(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "cleanup_seconds", lambda: 0.0)
    supervisor, worker, journal = _exiting_worker(tmp_path, _DoneHandle())
    worker.handle = None  # the leader was reaped; only the durable commit is outstanding
    ticket = journal.tickets[0]

    def ack_late():
        time.sleep(0.2)
        ticket.result = {"ok": True, "value": dict(ticket.payload["value"], exited_seq=1)}

    threading.Thread(target=ack_late, daemon=True).start()
    supervisor.reap()
    assert worker.state == "EXITED" and worker.exited_seq == 1 and worker.commit_intent is None
    assert not any("unacknowledged" in error for error in supervisor._protocol_errors)


def test_reap_stops_on_a_dead_journal_instead_of_spinning(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "cleanup_seconds", lambda: 0.0)
    supervisor, worker, journal = _exiting_worker(tmp_path, _DoneHandle())
    worker.handle = None
    journal._closed = True
    started = time.monotonic()
    supervisor.reap()
    assert time.monotonic() - started < 2.0
    assert any("unacknowledged" in error for error in supervisor._protocol_errors)


def _durable_running_record(tmp_path, *, started_epoch=None, process_record=True):
    """Persist a RUNNING worker the way a host that died mid-run leaves it: no handle, pid known."""
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    worker.started_epoch = started_epoch
    supervisor._persist_worker(worker, launch=True)
    if process_record:
        proc = worker.run_dir / "proc" / "1"
        proc.mkdir(parents=True, exist_ok=True)
        (proc / "process.json").write_text(json.dumps({
            "pid": 4242, "pgid": 4242, "kernel_identity": "darwin:4242:1", "verified_members": {}}))
        # The commit tests begin after handle cleanup, before its EXITED commit.
        (proc / "family-exited.json").write_text(json.dumps({
            "pid": 4242, "pgid": 4242, "kernel_identity": "darwin:4242:1"}))
    return worker


def test_resume_finalizes_a_handle_less_worker_whose_recorded_group_is_gone(tmp_path, monkeypatch):
    worker = _durable_running_record(tmp_path)
    monkeypatch.setattr(host, "_verified_live_members", lambda pgid, expected: [])
    monkeypatch.setattr(host, "_terminal", lambda run_dir: "SUBMITTED")
    resumed = ParallelSupervisor(tmp_path / "run")
    loaded = resumed._workers[("call", "a", 1)]
    assert loaded.handle is None
    assert loaded.state == "EXITED" and loaded.terminal == "SUBMITTED" and loaded.exited_seq == 1
    assert _status(resumed, loaded)["state"] == "EXITED"
    assert json.loads(worker.state_path.read_text())["state"] == "EXITED"


def test_resume_leaves_a_worker_alone_when_the_recorded_group_cannot_be_proven_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "_terminal", lambda run_dir: "SUBMITTED")
    live = _durable_running_record(tmp_path / "live")
    monkeypatch.setattr(host, "_verified_live_members", lambda pgid, expected: [(4242, "darwin:4242:1")])
    resumed = ParallelSupervisor(tmp_path / "live" / "run")
    assert resumed._workers[("call", "a", 1)].state == "RUNNING"
    assert json.loads(live.state_path.read_text())["state"] == "RUNNING"

    unproven = _durable_running_record(tmp_path / "torn", process_record=False)
    resumed = ParallelSupervisor(tmp_path / "torn" / "run")
    loaded = resumed._workers[("call", "a", 1)]
    assert loaded.state == "RUNNING" and _status(resumed, loaded)["state"] == "RUNNING"
    assert json.loads(unproven.state_path.read_text())["state"] == "RUNNING"
    assert any("could not be verified" in error for error in resumed._protocol_errors)


# ---- body 12: minor ----------------------------------------------------------------------------

def test_pump_commits_snapshots_workers_under_the_host_lock(tmp_path):
    supervisor, worker, _ = _exiting_worker(tmp_path, _DoneHandle())
    lock = supervisor._lock

    class LockedIndex(host._WorkerIndex):
        def values(self):
            assert lock._is_owned(), "worker index read outside the host lock"
            return super().values()

    guarded = LockedIndex()
    guarded[("call", "a", 1)] = worker
    supervisor._workers = guarded
    supervisor._pump_commits()


def test_resume_keeps_a_started_worker_whose_slot_marker_is_gone(tmp_path, monkeypatch):
    started = time.time() - 5
    worker = _durable_running_record(tmp_path, started_epoch=started, process_record=False)
    assert not (worker.run_dir / "proc" / "1" / "slot-lease.json").exists()
    resumed = ParallelSupervisor(tmp_path / "run")
    view = _status(resumed, worker)
    assert view["state"] == "RUNNING" and view["running_since"] == started and view["slot"] is None
    assert not any("status recovery refused" in error for error in resumed._protocol_errors)
