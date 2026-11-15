"""Engine-free launch durability and terminal-admission race regressions."""
import io
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_parallel_bridge import DelayedJournal
from test_parallel_host import _request, _spec_and_snapshot, _Worker
from test_parallel_supervisor_scale import _finalizing_worker

import stateframe.parallel_host as host
import stateframe.parallel_worker as worker_module


class _Handle:
    timed_out = False
    cancel_reason = None
    pid = pgid = 4242

    def __init__(self, *args, **kwargs):
        self.watchdogs = 0
        self.reaps = 0
        self.done = False
        self.running = []

    def attach_tick_table(self, table):
        self.table = table

    def enforce_deadline(self):
        self.watchdogs += 1

    def escalate(self):
        pass

    def poll(self):
        return 0 if self.done else None

    def reap(self, **kwargs):
        self.reaps += 1
        return 0

    def group_live(self):
        return False

    def note_running(self, epoch):
        self.running.append(epoch)

    def cancel(self, reason):
        self.cancel_reason = reason


@contextmanager
def _ready(*args):
    yield True, None


def _launch_fixture(tmp_path, monkeypatch):
    root = tmp_path / "run"
    call, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(root)
    (root / "spec.json").write_text("{}")
    supervisor = host.ParallelSupervisor(root)
    supervisor._journal = DelayedJournal()
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, preparing=True, launch_committed=True)
    supervisor._workers[("call", "a", 1)] = worker
    monkeypatch.setattr(host, "prepare_worker", lambda *a, **k: None)
    monkeypatch.setattr(host, "reserve_preparation", _ready)
    monkeypatch.setattr(worker_module, "_launch_spec", lambda path: {
        "engine_dir": str(root / "engine"), "parallel_engine_python": "unused-python"})
    monkeypatch.setattr(worker_module, "_child_environment", lambda spec: {})
    spawned = []

    def popen(*args, **kwargs):
        process = SimpleNamespace(pid=4242, stdout=io.StringIO(""))
        spawned.append(process)
        return process

    monkeypatch.setattr(worker_module.subprocess, "Popen", popen)
    monkeypatch.setattr(worker_module, "capture_launch_identity", lambda *a, **k: object())
    monkeypatch.setattr(worker_module, "kernel_start_identity", lambda pid: "birth-bound")
    monkeypatch.setattr(worker_module, "WorkerHandle", _Handle)
    return supervisor, worker, spawned


@pytest.mark.parametrize("stage", ["directory", "launch-intent.json", "process.json"])
def test_launch_durable_writes_do_not_block_commit_pump_or_sibling_watchdog(
        tmp_path, monkeypatch, stage):
    supervisor, worker, spawned = _launch_fixture(tmp_path, monkeypatch)
    entered, release, progressed = threading.Event(), threading.Event(), threading.Event()
    write, fsync = worker_module._write_json, worker_module._fsync_directory

    def pause():
        entered.set()
        assert release.wait(10), "launch durability barrier was not released"

    def blocked_write(path, *args, **kwargs):
        if path.name == stage:
            pause()
        return write(path, *args, **kwargs)

    def blocked_fsync(path):
        if stage == "directory" and not entered.is_set():
            pause()
        return fsync(path)

    monkeypatch.setattr(worker_module, "_write_json", blocked_write)
    monkeypatch.setattr(worker_module, "_fsync_directory", blocked_fsync)
    sibling_handle = _Handle()
    sibling = _Worker("call", "b", "sibling", 1, worker.call_root, worker.spec_path, "spec",
                      worker.run_dir, worker.snapshot_path, "start", {}, handle=sibling_handle,
                      state="RUNNING", started_epoch=1.0)

    def watchdog():
        supervisor._pump_commits()
        supervisor._advance_worker(sibling)
        progressed.set()

    launch = threading.Thread(target=supervisor._prepare_and_launch, args=(worker,))
    tick = threading.Thread(target=watchdog)
    launch.start()
    try:
        assert entered.wait(10), worker.error
        tick.start()
        assert progressed.wait(1.0), "durable launch held the host lock and stopped watchdogs"
        assert sibling_handle.watchdogs == 1
        if stage == "process.json":
            assert spawned and worker.handle is not None, "spawn must install its handle before fsync"
            assert worker.preparing, "launch ownership must cover process-record durability"
            marker = (1, worker.created_epoch + 0.1)
            monkeypatch.setattr(supervisor, "_slot_marker", lambda candidate: marker)
            worker.handle.done = True
            supervisor._advance_worker(worker)
            assert worker.handle.watchdogs == 1
            assert not worker.finalizing and worker.handle.reaps == 0
            assert worker.state == "RUNNING" and worker.started_epoch == marker[1]
    finally:
        release.set()
        launch.join(10)
        if tick.ident is not None:
            tick.join(10)
    assert not launch.is_alive() and not tick.is_alive()
    assert worker.error is None and worker.handle is not None
    assert not worker.preparing
    if stage == "process.json":
        assert worker.state == "RUNNING", "launch completion rewound the observed slot transition"


@pytest.mark.parametrize("stop", ["cancel", "close"])
@pytest.mark.parametrize("stage", ["launch-intent.json", "process.json"])
def test_cancellation_during_launch_durability_fences_spawn_and_terminal(
        tmp_path, monkeypatch, stop, stage):
    supervisor, worker, spawned = _launch_fixture(tmp_path, monkeypatch)
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    write = worker_module._write_json

    def blocked_write(path, *args, **kwargs):
        if path.name == stage:
            entered.set()
            assert release.wait(10)
        return write(path, *args, **kwargs)

    monkeypatch.setattr(worker_module, "_write_json", blocked_write)

    def stop_launch():
        if stage == "process.json":
            # The leader can exit before its durable process record is acknowledged.
            assert worker.handle is not None
            worker.handle.done = True
        if stop == "cancel":
            supervisor._cancel(_request(1, "call", 1, "cancel", {
                "unit_id": "a", "reason": "cancel during fsync"}), worker.call_root)
        else:
            supervisor.cancel_all("close during fsync")
        stopped.set()

    launch = threading.Thread(target=supervisor._prepare_and_launch, args=(worker,))
    cancel = threading.Thread(target=stop_launch)
    launch.start()
    try:
        assert entered.wait(10), worker.error
        cancel.start()
        assert stopped.wait(1.0), "cancel waited on launch durability"
    finally:
        release.set()
        launch.join(10)
        if cancel.ident is not None:
            cancel.join(10)
    assert not launch.is_alive() and not cancel.is_alive()
    assert not worker.preparing
    if stage == "launch-intent.json":
        assert not spawned and worker.handle is None
    else:
        assert spawned and worker.handle is not None
        supervisor._advance_worker(worker)
        finalizer = worker.thread
        assert finalizer is not None
        finalizer.join(10)
        assert not finalizer.is_alive()
    assert worker.state == "EXITING"
    assert worker.commit_intent["value"]["terminal"] == "CANCELLED"


def test_failed_launch_releases_admission_before_containment_cleanup(tmp_path, monkeypatch):
    supervisor, worker, spawned = _launch_fixture(tmp_path, monkeypatch)
    entered, release, progressed = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(worker_module, "kernel_start_identity", lambda pid: None)

    def abort(process, identity, reason):
        entered.set()
        assert release.wait(10)
        raise worker_module.WorkerLaunchError(reason)

    def pump():
        supervisor._pump_commits()
        progressed.set()

    monkeypatch.setattr(worker_module, "_abort_failed_launch", abort)
    launch = threading.Thread(target=supervisor._prepare_and_launch, args=(worker,))
    tick = threading.Thread(target=pump)
    launch.start()
    try:
        assert entered.wait(10)
        tick.start()
        assert progressed.wait(1.0), "failed launch kept admission locked during cleanup"
    finally:
        release.set()
        launch.join(10)
        if tick.ident is not None:
            tick.join(10)
    assert not launch.is_alive() and not tick.is_alive()
    assert spawned and worker.handle is None and not worker.preparing
    assert "kernel identity" in worker.error
    assert worker.commit_intent["value"]["terminal"] == "ENGINE_ERROR"


@pytest.mark.parametrize("phase", ["reserved", "terminal_pending"])
def test_stale_slot_marker_cannot_replace_terminal_intent_or_admit_another_finalizer(
        tmp_path, monkeypatch, phase):
    handle = _Handle()
    handle.done = True
    supervisor, worker = _finalizing_worker(tmp_path, handle)
    supervisor._workers[("call", "a", 1)] = worker
    journal = supervisor._journal = DelayedJournal()
    worker.state = "STARTING"
    reading, release = threading.Event(), threading.Event()
    reaping, finish = threading.Event(), threading.Event()
    marker = (2, worker.created_epoch + 0.1)
    stale = None
    reap = handle.reap

    def paused_reap():
        result = reap()
        if phase == "reserved":
            reaping.set()
            assert finish.wait(10)
        return result

    def slot_marker(candidate):
        if threading.current_thread() is stale:
            reading.set()
            assert release.wait(10)
            return marker
        return None

    monkeypatch.setattr(supervisor, "_slot_marker", slot_marker)
    monkeypatch.setattr(handle, "reap", paused_reap)
    monkeypatch.setattr(host, "_terminal", lambda run: "SUBMITTED")
    stale = threading.Thread(target=supervisor._advance_worker, args=(worker,))
    stale.start()
    try:
        assert reading.wait(10)
        supervisor._advance_worker(worker)
        first_finalizer = worker.thread
        assert first_finalizer is not None
        if phase == "reserved":
            assert reaping.wait(10) and worker.finalizing
        else:
            first_finalizer.join(10)
            assert not first_finalizer.is_alive()
            assert worker.state == "EXITING" and not worker.finalizing
        intent, version = worker.commit_intent, worker.commit_version
        state = worker.state
        release.set()
        stale.join(10)
        assert not stale.is_alive()
        assert worker.commit_intent is intent and worker.commit_version == version
        assert worker.state == state and worker.started_epoch is None
    finally:
        release.set()
        finish.set()
        stale.join(10)
        if worker.thread is not None:
            worker.thread.join(10)
    assert not stale.is_alive()
    assert handle.reaps == 1, "stale marker admitted a second finalizer before the terminal ACK"
    assert worker.state == "EXITING" and worker.started_epoch is None
    assert handle.running == [] and len(journal.tickets) == 1
    ticket = journal.tickets[0]
    ticket.result = {"ok": True, "value": dict(ticket.payload["value"], exited_seq=1)}
    supervisor._pump_commits()
    assert worker.state == "EXITED" and worker.exited_seq == 1
    assert worker.terminal == "SUBMITTED" and worker.commit_intent is None


def test_finalizer_records_a_late_slot_marker_only_in_its_terminal_commit(tmp_path, monkeypatch):
    handle = _Handle()
    handle.done = True
    supervisor, worker = _finalizing_worker(tmp_path, handle)
    supervisor._workers[("call", "a", 1)] = worker
    journal = supervisor._journal = DelayedJournal()
    worker.state = "STARTING"
    marker = (2, worker.created_epoch + 0.1)
    monkeypatch.setattr(supervisor, "_slot_marker", lambda candidate: marker if worker.finalizing else None)
    monkeypatch.setattr(host, "_terminal", lambda run: "SUBMITTED")
    supervisor._advance_worker(worker)
    finalizer = worker.thread
    assert finalizer is not None
    finalizer.join(10)
    assert not finalizer.is_alive() and handle.reaps == 1
    assert worker.state == "EXITING" and worker.started_epoch == marker[1]
    assert worker.slot_marker == marker
    assert worker.commit_version == 1, "terminal owner must not first enqueue RUNNING"
    assert len(journal.tickets) == 1 and journal.tickets[0].payload["value"]["state"] == "EXITED"
    assert worker.commit_intent["value"]["started_epoch"] == marker[1]
