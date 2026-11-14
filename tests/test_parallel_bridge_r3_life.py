"""Engine-free recovery containment and nonblocking lifecycle regression checks."""
from dataclasses import replace
import json
import os
import signal
import threading
from types import SimpleNamespace

import pytest

import stateframe.parallel_host as host
from test_parallel_bridge import DelayedJournal
from test_parallel_host import _request, _spec_and_snapshot


_BIRTH = "darwin:4242:1"
_MEMBER_BIRTH = "darwin:5555:1"


def _recorded(tmp_path, monkeypatch):
    call, run, snapshot, spec_path, spec = _spec_and_snapshot(tmp_path / "run")
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    worker = host._Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), run,
                          snapshot, spec.start_manifest_digest, {}, state="RUNNING", pid=4242, pgid=4242)
    supervisor._persist_worker(worker, launch=True)
    (run / "spec.json").write_text(json.dumps({"engine_dir": str(tmp_path / "engine")}))
    process_file = run / "proc" / "1" / "process.json"
    process_file.parent.mkdir(parents=True)
    process_file.write_text(json.dumps({"incarnation": 1, "pid": 4242, "pgid": 4242,
                                       "kernel_identity": _BIRTH, "verified_members": {"5555": _MEMBER_BIRTH}}))
    observer = host.ProcessTable().identity(os.getpid())
    rows = {os.getpid(): observer}
    monkeypatch.setattr(supervisor._process_table, "tick_snapshot", lambda: dict(rows))
    monkeypatch.setattr(supervisor._process_table, "snapshot", lambda: dict(rows))
    monkeypatch.setattr(host, "_live_group_members", lambda pgid: [])
    monkeypatch.setattr(host, "_signal_one", lambda *args: False)
    monkeypatch.setattr(host, "_process_identity", lambda pid:
                        (7777, "R", _MEMBER_BIRTH, ["detached"]) if pid == 5555 else None)
    return supervisor, worker, process_file, rows


@pytest.mark.parametrize("completion", ["probe", "resume", "cancel", "seal"])
def test_recorded_detached_birth_blocks_every_completion_path(tmp_path, monkeypatch, completion):
    supervisor, worker, process_file, _ = _recorded(tmp_path, monkeypatch)
    if completion == "probe":
        assert host._recorded_group_live(worker.run_dir, 4242, 4242, _BIRTH, process_file), \
            "a live birth-bound descendant still owns the worker after leaving its group"
    elif completion == "resume":
        supervisor._finalize_recorded_worker(worker)
        assert worker.state == "RUNNING" and worker.exited_seq is None
    elif completion == "seal":
        worker.state, worker.exited_seq = "EXITED", 1
        assert supervisor._group_live(worker), "a committed exit cannot hide a saved detached descendant"
        process_file.unlink()
        with pytest.raises(host.ProcessInspectionDeferred, match="ownership unavailable"):
            supervisor._group_live(worker)
    else:
        worker.cancel_requested = "stop"
        worker.fence_stage, worker.fence_deadline = "TERM", 0.0
        supervisor._advance_workers()
        assert worker.state == "RUNNING" and worker.exited_seq is None
        assert not json.loads(process_file.read_text()).get("fenced", False)


def test_recorded_signal_reaches_detached_birth_without_signalling_reused_pid(tmp_path, monkeypatch):
    supervisor, worker, process_file, _ = _recorded(tmp_path, monkeypatch)
    signals = []
    monkeypatch.setattr(host, "_signal_one", lambda *args: signals.append(args))
    host._signal_recorded_group(worker.run_dir, 4242, 4242, _BIRTH, signal.SIGTERM, process_file)
    assert signals == [(5555, 7777, _MEMBER_BIRTH, signal.SIGTERM)]
    signals.clear()
    monkeypatch.setattr(host, "_process_identity", lambda pid: (7777, "R", "darwin:5555:2", []))
    host._signal_recorded_group(worker.run_dir, 4242, 4242, _BIRTH, signal.SIGKILL, process_file)
    assert signals == [], "a recycled detached PID is never authority for a signal"
    record = json.loads(process_file.read_text())
    for raw_pid in ("05555", "0", "5" * 5000, "٥٥٥٥"):
        record["verified_members"] = {raw_pid: _MEMBER_BIRTH}
        process_file.write_text(json.dumps(record))
        with pytest.raises(host.ParallelHostError, match="member identities are invalid"):
            host._recorded_group_live(worker.run_dir, 4242, 4242, _BIRTH, process_file)
        with pytest.raises(host.ParallelHostError, match="member identities are invalid"):
            host._signal_recorded_group(worker.run_dir, 4242, 4242, _BIRTH, signal.SIGTERM, process_file)


@pytest.mark.parametrize("stage", ["PENDING", "TERM"])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("lost", [False, True])
def test_recorded_durability_never_stalls_later_worker_watchdogs(tmp_path, monkeypatch, stage, full, lost):
    supervisor, worker, process_file, rows = _recorded(tmp_path, monkeypatch)
    worker.cancel_requested, worker.fence_stage, worker.fence_deadline = "stop", stage, 0.0
    journal = supervisor._journal = DelayedJournal(full=full)
    if stage == "PENDING":
        observer = rows[os.getpid()]
        rows[4242] = replace(observer, pid=4242, pgid=4242, zombie=False)
        child = tmp_path / "_child.py"
        monkeypatch.setattr(host, "_process_identity", lambda pid:
                            (4242, "R", _BIRTH, [str(child), str(worker.run_dir)]) if pid == 4242
                            else (4242, "R", _MEMBER_BIRTH, []))
        monkeypatch.setattr(host, "_live_group_members", lambda pgid: [4242, 5555])
    else:
        monkeypatch.setattr(host, "_process_identity", lambda pid: None)
        # This test isolates durable fence ACKs after an already completed family sweep.
        process_file.with_name("family-exited.json").write_text(json.dumps({
            "pid": 4242, "pgid": 4242, "kernel_identity": _BIRTH}))
    later = replace(worker, unit_id="b", pid=None, pgid=None, handle=SimpleNamespace())
    supervisor._workers[("call", "b", 1)] = later
    visited = threading.Event()
    monkeypatch.setattr(supervisor, "_advance_worker", lambda current: visited.set() if current is later else None)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_write = host._write_json
    signals = []
    monkeypatch.setattr(host, "_signal_one", lambda *args: signals.append(args))

    def stalled_write(path, value, **kwargs):
        if path == process_file:
            entered.set()
            assert release.wait(5), "test must release its durability stall"
        return real_write(path, value, **kwargs)

    monkeypatch.setattr(host, "_write_json", stalled_write)

    def advance():
        try:
            supervisor._advance_workers()
        finally:
            finished.set()

    lifecycle = threading.Thread(target=advance)
    lifecycle.start()
    try:
        assert finished.wait(0.5), "a process-record durability wait blocked the lifecycle lane"
        assert visited.is_set(), "later workers must still receive watchdog service"
        assert signals == [], "persisted identities must be acknowledged before the first signal"
        assert worker.state == "RUNNING" and worker.exited_seq is None
        if full:
            assert journal.tickets == [], "saturation retains the intent without waiting for a credit"
            journal.full = False
            supervisor._advance_workers()
        writes = [ticket for ticket in journal.tickets if ticket.kind == "write"]
        assert len(writes) == 1, "one retained lifecycle intent must own the process-record write"
        for _ in range(3):
            supervisor._advance_workers()
        assert len([ticket for ticket in journal.tickets if ticket.kind == "write"]) == 1
        ticket = writes[0]
        if lost:
            ticket.result = {"ok": False, "lost": True, "error": "ACK lost"}
            supervisor._advance_workers()
            assert signals == [] and worker.state == "RUNNING"
            assert worker.fence_ticket is None and worker.fence_write is not None
            worker.fence_write_retry_at = 0.0
            supervisor._advance_workers()
            ticket = journal.tickets[-1]
            assert ticket is not writes[0] and ticket.payload == writes[0].payload
        real_write(process_file, ticket.payload["value"], replace_existing=True)
        ticket.result = {"ok": True}
        supervisor._advance_workers()
        if stage == "PENDING":
            assert signals and worker.fence_stage == "TERM"
        else:
            assert worker.state == "EXITING" and worker.exited_seq is None
            commit = next(ticket for ticket in journal.tickets if ticket.kind == "commit")
            assert commit.payload["value"]["terminal"] == "CANCELLED"
    finally:
        release.set()
        lifecycle.join(5)
    assert not lifecycle.is_alive()


@pytest.mark.parametrize("finalizing", [False, True])
def test_live_cancel_only_queues_then_lifecycle_delivers_outside_host_lock(tmp_path, monkeypatch, finalizing):
    supervisor, worker, _, _ = _recorded(tmp_path, monkeypatch)
    calls = []
    poll_calls = []

    def cancel(reason):
        assert not supervisor._lock._is_owned(), "process inspection/signalling cannot hold the host lock"
        calls.append((threading.current_thread(), reason))

    worker.handle = SimpleNamespace(cancel=cancel, poll=lambda: poll_calls.append(True),
                                    enforce_deadline=lambda: None, escalate=lambda: None,
                                    reap=lambda: 0, group_live=lambda: False,
                                    timed_out=False, cancel_reason=None)
    worker.finalizing = finalizing
    supervisor._journal = DelayedJournal()
    request = _request(1, "call", 1, "cancel", {"unit_id": "a", "reason": "stop"})
    assert supervisor._cancel(request, worker.call_root) == {"state": "CANCEL_REQUESTED"}
    assert calls == [] and poll_calls == [], "the channel may neither inspect nor signal a live handle"
    assert worker.cancel_requested == "stop"
    lifecycle = threading.Thread(target=(lambda: supervisor._finalize_worker(worker, 0))
                                 if finalizing else supervisor._advance_workers)
    lifecycle.start()
    lifecycle.join(5)
    assert not lifecycle.is_alive()
    if finalizing:
        assert calls == [], "an already contained worker needs no cancellation signal"
        assert worker.commit_intent["value"]["terminal"] == "CANCELLED", \
            "an accepted cancellation must survive a race with finalization"
    else:
        assert calls == [(lifecycle, "stop")]
