"""Finalizer admission must not replay a terminal intent after a stale process probe."""
import threading
from types import SimpleNamespace

import pytest

from _timeouts import scaled
import stateframe.parallel_host as host
from test_parallel_bridge import DelayedJournal


def _worker(tmp_path, monkeypatch):
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    call = supervisor.parallel_root / "call"
    unit = call / "units/a"
    handle = SimpleNamespace(enforce_deadline=lambda: None, escalate=lambda: None, poll=lambda: 0)
    worker = host._Worker("call", "a", "execution", 1, call, unit / "unit-spec.json", "spec",
                          unit, call / "snapshot", "manifest", {}, state="RUNNING", handle=handle)
    supervisor._workers[("call", "a", 1)] = worker
    supervisor._journal = DelayedJournal()
    monkeypatch.setattr(supervisor, "_notice_running", lambda worker: None)
    calls = []
    monkeypatch.setattr(supervisor, "_finalize_worker", lambda *args: calls.append(args))
    return supervisor, worker, handle, calls


def _join_finalizer(worker):
    if worker.thread is not None:
        worker.thread.join(scaled(2))
        assert not worker.thread.is_alive()


def test_probe_cannot_admit_another_finalizer_after_terminal_intent_was_queued(tmp_path, monkeypatch):
    supervisor, worker, handle, calls = _worker(tmp_path, monkeypatch)

    def finish_during_probe():
        # The lifecycle observed RUNNING, then the existing finalizer queued EXITED and
        # released ownership while this probe was in flight. The durable ACK is still pending.
        worker.state = "EXITED"
        worker.finished_epoch = 10.0
        worker.terminal = "ENGINE_ERROR"
        worker.exit_code = 0
        worker.error = "original terminal result"
        worker.group_verified = True
        supervisor._persist_worker(worker)
        worker.finalizing = False
        return 0

    handle.poll = finish_during_probe
    supervisor._advance_worker(worker)
    _join_finalizer(worker)
    assert calls == [], "a stale RUNNING observation admitted a second terminal finalizer"
    assert worker.state == "EXITING" and worker.commit_version == 1
    (ticket,) = supervisor._journal.tickets
    assert ticket.payload["value"]["finished_epoch"] == 10.0
    ticket.result = {"ok": True, "value": dict(ticket.payload["value"], exited_seq=1)}
    ticket.done.set()
    supervisor._pump_commits()
    assert worker.state == "EXITED" and worker.exited_seq == 1
    assert worker.finished_epoch == 10.0 and worker.error == "original terminal result"
    assert worker.commit_intent is None


def test_finalizing_worker_is_not_probed_by_another_lifecycle_pass(tmp_path, monkeypatch):
    supervisor, worker, handle, calls = _worker(tmp_path, monkeypatch)
    worker.finalizing = True

    def forbidden():
        pytest.fail("the finalizer already owns this exited process")

    handle.enforce_deadline = handle.escalate = handle.poll = forbidden
    supervisor._advance_worker(worker)
    assert calls == []


def test_finalizer_handoff_defers_without_waiting_on_the_host_lock(tmp_path, monkeypatch):
    supervisor, worker, handle, calls = _worker(tmp_path, monkeypatch)
    held, release = threading.Event(), threading.Event()

    def holder():
        with supervisor._lock:
            held.set()
            release.wait(scaled(2))

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert held.wait(scaled(2))
        supervisor._advance_worker(worker)
        _join_finalizer(worker)
        assert calls == [], "a contended finalizer handoff must remain pending"
        assert not worker.finalizing
    finally:
        release.set()
        thread.join(scaled(2))
        assert not thread.is_alive()


def test_failed_finalizer_thread_start_releases_its_reservation(tmp_path, monkeypatch):
    supervisor, worker, handle, calls = _worker(tmp_path, monkeypatch)

    class Unstartable:
        def start(self):
            raise RuntimeError("no thread capacity")

    with monkeypatch.context() as patch:
        patch.setattr(host.threading, "Thread", lambda **kwargs: Unstartable())
        with pytest.raises(RuntimeError, match="no thread capacity"):
            supervisor._advance_worker(worker)
    assert not worker.finalizing and worker.thread is None
    supervisor._advance_worker(worker)
    _join_finalizer(worker)
    assert len(calls) == 1
