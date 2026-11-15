"""Journal ACK-progress watchdog, typed helper errors, seal/journal epoch decoupling and the
terminal state of a durably rejected exit commit. Engine-free except where a real journal
helper process is the point of the test."""
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

import stateframe.parallel_host as host
import stateframe.parallel_journal as parallel_journal
from stateframe.parallel_host import ParallelHostError
from stateframe.parallel_journal import Journal, JournalFault
from stateframe.parallel_seals import SealOperation

# Resolved lazily so the file still collects (and fails red) on a tree without the typed error.
JournalProtocolError = getattr(parallel_journal, "JournalProtocolError", None)

from test_parallel_bridge import DelayedJournal
from test_parallel_bridge_commit import _DoneHandle, _exiting_worker, _reject, _status
from test_parallel_host import _request, _snapshot_digest
from test_parallel_supervisor_scale import _CountingHandle, _finalizing_worker


class PipeProcess:
    """A fake helper over real pipes, so the reactor's select-based reader sees it the way it
    sees a child process. `answer(message)` returns the reply, or None to read and never answer;
    `trickle` writes the reply one byte per slice so it arrives slowly but with progress."""

    def __init__(self, answer=None, *, trickle=0.0):
        in_read, in_write = os.pipe()
        out_read, out_write = os.pipe()
        self.stdin = os.fdopen(in_write, "wb")
        self.stdout = os.fdopen(out_read, "rb")
        self._requests = os.fdopen(in_read, "rb")
        self._responses = os.fdopen(out_write, "wb")
        self._answer = answer or (lambda message: {"id": message["id"], "ok": True})
        self._trickle = trickle
        self.kills = 0
        self.returncode = None
        self.received = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        try:
            for raw in iter(self._requests.readline, b""):
                message = json.loads(raw)
                self.received.append(message)
                reply = self._answer(message)
                if reply is None:
                    continue  # wedged: the request was read but no byte ever comes back
                encoded = json.dumps(reply).encode() + b"\n"
                if self._trickle:
                    for byte in encoded:
                        self._responses.write(bytes([byte]))
                        self._responses.flush()
                        time.sleep(self._trickle)
                else:
                    self._responses.write(encoded)
                    self._responses.flush()
        except (OSError, ValueError):
            return

    def kill(self):
        self.kills += 1
        self.returncode = -9
        try:
            self._responses.close()  # the reader sees EOF, as it would from a dead child
        except OSError:
            pass

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


def _fast_journal(monkeypatch, *, slices=3, restarts=2):
    monkeypatch.setattr(parallel_journal, "_PROGRESS_SLICE_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(parallel_journal, "_NO_PROGRESS_SLICES", slices, raising=False)
    monkeypatch.setattr(parallel_journal, "_RESTART_BACKOFF_SECONDS", 0.001, raising=False)
    monkeypatch.setattr(parallel_journal, "_RESTART_BACKOFF_MAX_SECONDS", 0.002, raising=False)
    monkeypatch.setattr(parallel_journal, "_MAX_RESTARTS_PER_REQUEST", restarts, raising=False)
    monkeypatch.setattr(parallel_journal, "_HELPER_EXIT_WAIT_SECONDS", 0.01, raising=False)


# ---- body 4: ACK-progress watchdog ---------------------------------------------------------------

def test_reactor_kills_a_helper_that_reads_but_never_answers(monkeypatch):
    _fast_journal(monkeypatch)
    spawned = []

    def spawn(self):
        self.epoch += 1
        spawned.append(PipeProcess(lambda message: None))
        return spawned[-1]
    monkeypatch.setattr(Journal, "_spawn", spawn)
    journal = Journal()
    try:
        started = time.monotonic()
        with pytest.raises(JournalFault, match="no progress"):
            journal.call("write", {"path": "x", "value": {}, "replace": False, "durable": False})
        assert time.monotonic() - started < 5.0
        # Every wedged helper was killed through its handle, and the spawn count is the budget.
        assert len(spawned) == 2 and all(p.kills == 1 and len(p.received) == 1 for p in spawned)
        assert journal.fault is not None
    finally:
        journal.close()


def test_reactor_waits_on_a_slow_helper_that_keeps_making_progress(monkeypatch):
    # Three idle slices of 10 ms would cut the helper off; one byte every 15 ms is progress, so a
    # reply that takes far longer than the no-progress window still lands: not a wall clock.
    _fast_journal(monkeypatch, slices=3)
    process = PipeProcess(trickle=0.015)
    monkeypatch.setattr(Journal, "_spawn", lambda self: process)
    journal = Journal()
    try:
        started = time.monotonic()
        result = journal.call("write", {"path": "x", "value": {}, "replace": False, "durable": False})
        assert result["ok"] is True and process.kills == 0
        assert time.monotonic() - started > 3 * 0.01
    finally:
        journal.close()


def test_lost_acks_back_off_and_a_faulted_journal_fails_the_retained_intent(tmp_path):
    supervisor, worker, journal = _exiting_worker(tmp_path, _DoneHandle())
    for loss in range(1, host._JOURNAL_LOSS_LIMIT):
        _reject(journal, supervisor, worker, "journal helper lost", lost=True)
        assert worker.commit_failures == 0 and "pending" in worker.error
        assert worker.commit_intent is not None
        assert worker.commit_ticket is None and worker.commit_retry_at > time.monotonic(), \
            "a lost ACK must not be resubmitted hot"
        assert supervisor._journal_lost_acks == loss and supervisor._journal_fault is None
        worker.commit_retry_at = 0.0
        supervisor._pump_commits()
        assert worker.commit_ticket is not None, "after the backoff the intent is resubmitted"
    _reject(journal, supervisor, worker, "journal helper lost", lost=True)
    assert supervisor._journal_fault is not None and worker.commit_intent is None
    assert worker.state == "EXITED" and worker.exited_seq is None
    with pytest.raises(ParallelHostError, match="unavailable"):
        _status(supervisor, worker)
    assert sum("journal fault" in error for error in supervisor._protocol_errors) == 1
    assert len(journal.tickets) == host._JOURNAL_LOSS_LIMIT
    supervisor._pump_commits()
    assert len(journal.tickets) == host._JOURNAL_LOSS_LIMIT, "a faulted lane is never resubmitted"


def test_journal_fault_is_read_and_fails_other_retained_intents_without_waiting(tmp_path):
    supervisor, worker, journal = _exiting_worker(tmp_path, _DoneHandle())
    journal.fault = "journal helper unavailable after 6 restarts: OSError: no helper"
    assert worker.commit_ticket is not None
    journal.tickets[-1].result = {"ok": False, "lost": True, "error": journal.fault}
    supervisor._pump_commits()
    assert worker.commit_intent is None and worker.state == "EXITED"
    assert supervisor._journal_fault is not None
    with pytest.raises(ParallelHostError, match="unavailable"):
        _status(supervisor, worker)


def test_reap_returns_when_the_helper_cannot_start(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "cleanup_seconds", lambda: 0.0)
    _fast_journal(monkeypatch, restarts=3)
    attempts = []

    def no_popen(*args, **kwargs):
        attempts.append(args)
        raise OSError("no helper")
    monkeypatch.setattr(parallel_journal.subprocess, "Popen", no_popen)
    supervisor, worker, _ = _exiting_worker(tmp_path, _DoneHandle())
    worker.handle = None
    supervisor._journal = journal = Journal()
    worker.commit_ticket = None
    supervisor._submit_commit(worker)
    started = time.monotonic()
    supervisor.reap()
    assert time.monotonic() - started < 10.0
    assert len(attempts) == 3, "spawn attempts are the per-request budget, never more"
    assert supervisor._journal_fault is not None and journal.fault is not None
    assert worker.commit_intent is None and worker.state == "EXITED"
    with pytest.raises(ParallelHostError, match="unavailable"):
        _status(supervisor, worker)


def test_reap_returns_when_the_helper_is_wedged(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "cleanup_seconds", lambda: 0.0)
    _fast_journal(monkeypatch)
    monkeypatch.setattr(Journal, "_spawn", lambda self: PipeProcess(lambda message: None))
    supervisor, worker, _ = _exiting_worker(tmp_path, _DoneHandle())
    worker.handle = None
    supervisor._journal = Journal()
    worker.commit_ticket = None
    supervisor._submit_commit(worker)
    started = time.monotonic()
    supervisor.reap()
    assert time.monotonic() - started < 10.0
    assert supervisor._journal_fault is not None and worker.commit_intent is None
    assert sum("journal fault" in error for error in supervisor._protocol_errors) == 1


# ---- body 7: typed helper errors ---------------------------------------------------------------

def test_journal_call_raises_typed_errors(monkeypatch):
    answers = {
        "conflict": {"ok": False, "error_class": "ParallelHostError",
                     "error": "ParallelHostError: immutable channel record already differs: x"},
        "protocol": {"ok": False, "error_class": "ValueError", "error": "ValueError: unknown journal operation"},
    }
    monkeypatch.setattr(Journal, "_spawn", lambda self: PipeProcess(
        lambda message: dict(answers[message["kind"]], id=message["id"])))
    journal = Journal()
    try:
        with pytest.raises(ParallelHostError, match="already differs"):
            journal.call("conflict", {})
        with pytest.raises(JournalProtocolError, match="unknown journal operation"):
            journal.call("protocol", {})
    finally:
        journal.close()


def test_seal_result_conflict_with_a_live_journal_completes_the_operation(tmp_path):
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    worker.state, worker.exited_seq = "EXITED", 1
    op = SealOperation(worker, SimpleNamespace(), tmp_path / "job.json", tmp_path / "job.result.json",
                       worker.generation)
    op.phase, op.staged = "COMMITTING", {"result": {"status": "OK", "payload": {"sealed": True}}}
    supervisor._seal_ops[op.job] = op
    op.result.write_bytes(b'{"status":"ERROR","payload":{"error":"earlier"}}\n')
    supervisor._journal = Journal()  # a real helper process answers the publication write
    try:
        supervisor._publish_seal_results()
    finally:
        supervisor._journal.close()
    assert op.phase == "DONE" and supervisor._seal_ops == {} and op.job in supervisor._sealed_jobs
    assert sum("seal result conflict" in error for error in supervisor._protocol_errors) == 1
    assert json.loads(op.result.read_bytes())["status"] == "ERROR", "the durable result is authoritative"


def test_fail_seal_job_defers_on_a_journal_fault_instead_of_marking_the_job_sealed(tmp_path):
    supervisor = host.ParallelSupervisor(tmp_path / "run")

    class Faulted:
        def write(self, *args, **kwargs):
            raise JournalFault("journal helper unavailable")
    # Seal errors publish through the channel durability lane, not the exit journal.
    supervisor._channel_writes = Faulted()
    job = tmp_path / "run" / "seal" / "job.json"
    supervisor._fail_seal_job(job, "unsatisfiable")
    assert job not in supervisor._sealed_jobs
    assert any("deferred" in error for error in supervisor._protocol_errors)
    assert not job.with_name("job.result.json").exists()


# ---- body 10: seal ops survive main-journal epoch changes ---------------------------------------

def test_seal_operation_survives_a_main_journal_epoch_change(tmp_path):
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    worker.state, worker.exited_seq, worker.pgid = "EXITED", 1, None
    supervisor._workers[("call", "a", 1)] = worker
    supervisor._journal = DelayedJournal()
    supervisor._seal_helpers = [DelayedJournal()]
    op = SealOperation(worker, SimpleNamespace(to_json=lambda: {}), tmp_path / "job", tmp_path / "result", 0)
    supervisor._seal_ops[op.job] = op
    supervisor._advance_seals()
    assert op.phase == "BUILDING" and op.ticket is not None
    ticket, nonce = op.ticket, op.nonce
    supervisor._journal.epoch += 1  # the main journal helper restarted
    supervisor._advance_seals()
    assert op.phase == "BUILDING" and op.ticket is ticket and op.nonce == nonce
    ticket.result = {"ok": True, "token": supervisor._seal_token(op), "staged": {"result": {"status": "OK"}}}
    supervisor._journal.epoch += 1
    supervisor._advance_seals()
    assert op.phase in {"STAGED", "PUBLISHING"}, "the build is kept and the seal moves on"


# ---- body 12: a terminally rejected exit commit frees the unit ------------------------------------

def test_terminally_rejected_exit_commit_frees_the_unit_for_the_next_incarnation(tmp_path, monkeypatch):
    supervisor, worker, journal = _exiting_worker(tmp_path, _DoneHandle())
    for _ in range(host._COMMIT_REJECTION_LIMIT):
        worker.commit_retry_at = 0.0
        supervisor._pump_commits()
        _reject(journal, supervisor, worker, "ParallelHostError: terminal commit identity conflict")
    assert worker.commit_intent is None and worker.state == "EXITED" and worker.exited_seq is None
    assert worker.terminal == "ENGINE_ERROR" and worker.finished_epoch is not None
    with pytest.raises(ParallelHostError, match="rejected.*identity conflict"):
        _status(supervisor, worker)
    cancel = SimpleNamespace(call_id="call", payload={"unit_id": "a", "reason": "parent cancelled"})
    assert supervisor._cancel(cancel, worker.call_root) == {"state": "EXITED"}

    proc = worker.run_dir / "proc" / "1"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "process.json").write_text(json.dumps({
        "pid": 4242, "pgid": 4242, "kernel_identity": "linux:4242:1", "verified_members": {}}))
    monkeypatch.setattr(host, "_recorded_kernel_identity", lambda record: None)
    monkeypatch.setattr(host, "_recorded_group_live", lambda *args: False)
    worker.snapshot_manifest_digest = _snapshot_digest(worker.snapshot_path)
    payload = {"unit_id": "a", "execution_id": "execution", "spec_path": str(worker.spec_path),
               "spec_digest": worker.spec_digest, "run_dir": str(worker.run_dir), "incarnation": 2,
               "start_snapshot_path": str(worker.snapshot_path),
               "start_manifest_digest": worker.start_manifest_digest,
               "snapshot_manifest_digest": worker.snapshot_manifest_digest}
    status, result_id, answer = supervisor._launch(_request(1, "call", 2, "launch", payload), worker.call_root)
    assert (status, result_id, answer) == ("PENDING", "call:a:2", {"state": "STARTING"})
    assert supervisor._workers[("call", "a", 2)].incarnation == 2
