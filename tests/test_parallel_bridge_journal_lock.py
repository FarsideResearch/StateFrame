"""Engine-free checks: journal helper restarts are capped with backoff, launch answers never
sleep on the channel thread for their durable ACK, and the host lock is never held across
channel I/O."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

import stateframe.parallel_host as host
import stateframe.parallel_journal as parallel_journal
from stateframe.parallel_journal import Journal, JournalBusy


def _FakeProcess():
    # A healthy helper over real pipes: the reactor reads responses through select on the fd.
    from test_parallel_bridge_journal_progress import PipeProcess
    return PipeProcess(lambda message: {"id": message["id"], "ok": True})


def _lock_is_free(supervisor):
    """The host RLock has no owner-thread query; probe from another thread."""
    state = []

    def probe():
        taken = supervisor._lock.acquire(blocking=False)
        state.append(taken)
        if taken:
            supervisor._lock.release()
    thread = threading.Thread(target=probe)
    thread.start()
    thread.join(2)
    return state == [True]


def test_journal_helper_restart_is_capped_with_exponential_backoff(monkeypatch):
    monkeypatch.setattr(parallel_journal, "_RESTART_BACKOFF_SECONDS", 0.005, raising=False)
    monkeypatch.setattr(parallel_journal, "_RESTART_BACKOFF_MAX_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(parallel_journal, "_MAX_RESTARTS_PER_REQUEST", 6, raising=False)
    spawns = []

    def failing_spawn(self):
        spawns.append(time.monotonic())
        raise OSError("cannot fork")
    monkeypatch.setattr(Journal, "_spawn", failing_spawn)
    journal = Journal()
    try:
        ticket = journal.submit("write", {"path": "unused", "value": 1})
        assert ticket.done.wait(3), "helper restarts must be capped, not retried forever"
        result = journal.take(ticket)
        assert result["ok"] is False and result["lost"] is True
        assert "after 6 restarts" in result["error"] and "cannot fork" in result["error"]
        assert len(spawns) == 6
        gaps = [later - earlier for earlier, later in zip(spawns, spawns[1:])]
        assert gaps[-1] > gaps[0], gaps  # exponential, not a fixed 50 ms spin
        assert sum(gaps) >= 0.14
        assert journal.fault is not None and journal.restart_failures == 6

        # A later healthy helper clears the fault surface.
        monkeypatch.setattr(Journal, "_spawn", lambda self: _FakeProcess())
        assert journal.call("write", {"path": "unused", "value": 2})["ok"] is True
        assert journal.fault is None and journal.restart_failures == 0
    finally:
        journal.close()
        journal._thread.join(2)


def test_journal_call_raises_fault_when_restart_budget_is_spent(monkeypatch):
    monkeypatch.setattr(parallel_journal, "_RESTART_BACKOFF_SECONDS", 0.001, raising=False)
    monkeypatch.setattr(parallel_journal, "_RESTART_BACKOFF_MAX_SECONDS", 0.001, raising=False)
    monkeypatch.setattr(Journal, "_spawn", lambda self: (_ for _ in ()).throw(OSError("no helper")))
    from stateframe.parallel_journal import JournalFault
    journal = Journal()
    try:
        with pytest.raises(JournalFault, match="no helper"):
            journal.call("write", {"path": "unused", "value": 1})
    finally:
        journal.close()
        journal._thread.join(2)


def test_backoff_never_delays_close(monkeypatch):
    monkeypatch.setattr(parallel_journal, "_RESTART_BACKOFF_SECONDS", 0.05, raising=False)
    monkeypatch.setattr(parallel_journal, "_RESTART_BACKOFF_MAX_SECONDS", 30.0, raising=False)
    monkeypatch.setattr(parallel_journal, "_MAX_RESTARTS_PER_REQUEST", 50, raising=False)
    monkeypatch.setattr(Journal, "_spawn", lambda self: (_ for _ in ()).throw(OSError("no helper")))
    journal = Journal()
    ticket = journal.submit("write", {"path": "unused", "value": 1})
    time.sleep(0.3)  # into the second-plus backoff step
    started = time.monotonic()
    journal.close()
    journal._thread.join(5)
    assert not journal._thread.is_alive()
    assert time.monotonic() - started < 2.0
    assert ticket.done.is_set() and ticket.result["ok"] is False


def test_host_write_is_independent_of_a_faulted_commit_journal(tmp_path):
    # Channel writes have their own lane: a commit helper that exhausted its restart budget
    # cannot make a channel answer unavailable (that lane never carries channel I/O).
    from stateframe.parallel_journal import JournalFault
    supervisor = host.ParallelSupervisor(tmp_path / "run")

    class Faulted:
        def call(self, kind, payload):
            raise JournalFault("journal helper unavailable after 6 restarts")

        def submit(self, kind, payload):
            raise JournalFault("journal helper unavailable after 6 restarts")
    supervisor._journal = Faulted()
    supervisor._write_json(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}


def test_launch_answer_wait_for_durable_ack_is_bounded(tmp_path, monkeypatch):
    from test_parallel_bridge import DelayedJournal
    from test_parallel_host import _request
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    supervisor._journal = DelayedJournal()
    worker.state, worker.handle = "STARTING", None
    supervisor._persist_worker(worker, launch=True)
    outcome = []
    monkeypatch.setattr(supervisor, "_dispatch", lambda *a: ("PENDING", "launch-op", {"state": "STARTING"}))
    monkeypatch.setattr(supervisor, "_channel_record", lambda *a: None)
    monkeypatch.setattr(supervisor, "_store_answer", lambda *a: outcome.append("published"))

    def serve():
        try:
            supervisor._serve_request(_request(1, "call", 1, "launch", {"unit_id": "a"}),
                                      worker.call_root, worker.call_root / "attempt-1/an", 1)
        except Exception as error:  # noqa: BLE001 - the class is the assertion
            outcome.append(error)
    publisher = threading.Thread(target=serve)
    publisher.start()
    try:
        publisher.join(2)
        assert not publisher.is_alive(), "an un-ACKed launch must not pin the channel thread"
        assert len(outcome) == 1 and isinstance(outcome[0], JournalBusy)
        assert "retry pending" in str(outcome[0])
        assert supervisor._runtime_stop is False
    finally:
        supervisor._stop_runtime()
        publisher.join(2)


def _committing_seal(tmp_path, supervisor, worker):
    from stateframe.parallel_seals import SealOperation
    op = SealOperation(worker, SimpleNamespace(), tmp_path / "job", tmp_path / "result", worker.generation)
    op.phase = "COMMITTING"
    op.staged = {"result": {"status": "OK", "payload": {"sealed": True}}}  # a publication permit
    supervisor._seal_ops[op.job] = op
    return op


def test_seal_result_publication_runs_outside_host_lock(tmp_path, monkeypatch):
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    op = _committing_seal(tmp_path, supervisor, worker)
    seen = []
    monkeypatch.setattr(supervisor, "_write_json",
                        lambda path, value, **kw: seen.append((path, value, _lock_is_free(supervisor))))
    supervisor._publish_seal_results()
    assert seen == [(op.result, {"status": "OK", "payload": {"sealed": True}}, True)]
    assert op.phase == "DONE" and op.job in supervisor._sealed_jobs and op.job not in supervisor._seal_ops


def test_seal_result_generation_is_rechecked_after_publication_write(tmp_path, monkeypatch):
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    op = _committing_seal(tmp_path, supervisor, worker)

    def write_and_cancel(path, value, **kw):
        worker.generation += 1  # a cancel landed while the journal round trip was in flight
    monkeypatch.setattr(supervisor, "_write_json", write_and_cancel)
    supervisor._publish_seal_results()
    assert op.phase == "PENDING" and op.staged is None
    assert op.job not in supervisor._sealed_jobs and supervisor._seal_ops[op.job] is op


def test_cancel_fences_recorded_group_outside_host_lock(tmp_path, monkeypatch):
    from test_parallel_host import _request
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    worker.handle = None  # recovered record: a live group known only by pid/pgid
    supervisor._workers[("call", "a", 1)] = worker
    seen = []

    def fence(target, reason):
        seen.append((target, reason, _lock_is_free(supervisor)))
        target.state, target.terminal = "EXITED", "CANCELLED"
    monkeypatch.setattr(supervisor, "_fence_worker_record", fence)
    answer = supervisor._cancel(_request(1, "call", 1, "cancel", {"unit_id": "a", "reason": "stop"}),
                                worker.call_root)
    assert answer == {"state": "EXITED"}
    assert seen == [(worker, "stop", True)]
    assert worker.generation == 1


def test_cancel_without_recorded_process_still_fences(tmp_path, monkeypatch):
    from test_parallel_host import _request
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    worker.handle, worker.pid, worker.pgid, worker.state = None, None, None, "STARTING"
    supervisor._workers[("call", "a", 1)] = worker
    seen = []

    def fence(target, reason):
        seen.append(reason)
        target.state, target.terminal = "EXITED", "CANCELLED"
    monkeypatch.setattr(supervisor, "_fence_worker_record", fence)
    answer = supervisor._cancel(_request(1, "call", 1, "cancel", {"unit_id": "a", "reason": "stop"}),
                                worker.call_root)
    assert answer == {"state": "EXITED"} and seen == ["stop"]
