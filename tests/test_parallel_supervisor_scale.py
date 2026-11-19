"""The subframe supervisor tick must not queue behind scans or copies it does not need.

Each test pins one cost that showed up as multi-second tick latency with sixteen workers:
a second serial scan per liveness check, a private full scan under the host lock when an
exit is numbered, a recycled group number read as a live group, per-sibling and per-worker
polls outside a shared scan, a whole-repository copy under the host lock, and run-wall
handling that re-read the spec every tick and never bounded an unbounded worker.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import stateframe.parallel_host as host
import stateframe.parallel_worker as worker_module
from stateframe.engine.kernel_exec_tracker import ContainmentUnverified, LaunchIdentity, ProcessIdentity
from stateframe.parallel_host import ParallelSupervisor
from stateframe.parallel_worker import TickProcessTable, WorkerHandle

from test_parallel_host import _Worker, _spec_and_snapshot

_SLEEPER = "import time\ntime.sleep(120)\n"


@pytest.fixture
def procs():
    started: list[subprocess.Popen] = []
    yield started
    for process in started:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        process.wait(timeout=10)


def _live_handle(tmp_path, procs, table):
    process = subprocess.Popen([sys.executable, "-c", _SLEEPER], start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(process)
    handle = WorkerHandle(process, process.pid, process.pid, 1, tmp_path, tmp_path / "w.log",
                          time.time(), time.monotonic(), threading.Thread(target=lambda: None))
    handle.attach_tick_table(table)
    return handle


class _FakeTable:
    """A table returning fixed rows; `scans` counts snapshots."""

    def __init__(self, rows):
        self.rows = rows
        self.scans = 0

    def snapshot(self):
        self.scans += 1
        return dict(self.rows)

    def identity(self, pid):
        return self.rows.get(pid)


def _row(pid, pgid, start, *, ppid=1, zombie=False):
    return ProcessIdentity(pid, ppid, pgid, pgid, (start, 0), zombie)


def _reaped_handle(tmp_path, table, *, reaped_bound):
    process = SimpleNamespace(returncode=0)
    handle = WorkerHandle(process, 4242, 4242, 1, tmp_path, tmp_path / "w.log", 0, 0,
                          SimpleNamespace(), launch_identity=LaunchIdentity(4242, None, None))
    handle._family = SimpleNamespace(table=table, reaped_bound=reaped_bound, update=lambda rows: [])
    handle._shared_table = table
    return handle


# ---- B13: one scan for the family and the group ---------------------------------------------

def test_live_unit_takes_one_scan_not_two(tmp_path, procs):
    table = TickProcessTable()
    handle = _live_handle(tmp_path, procs, table)
    before = table.scans
    assert handle._live_unit() is True
    assert table.scans - before == 1


def test_live_unit_reports_a_dead_unit_from_the_same_single_scan(tmp_path, procs):
    table = TickProcessTable()
    handle = _live_handle(tmp_path, procs, table)
    os.killpg(handle.pgid, signal.SIGKILL)
    handle.process.wait(timeout=10)
    deadline = time.monotonic() + 10
    while handle._live_unit() and time.monotonic() < deadline:
        time.sleep(0.02)
    before = table.scans
    assert handle._live_unit() is False
    assert table.scans - before == 1


def test_siblings_are_polled_from_one_scan(tmp_path, procs):
    supervisor = ParallelSupervisor(tmp_path / "run")
    workers = []
    for index in range(6):
        handle = _live_handle(tmp_path, procs, supervisor._process_table)
        os.killpg(handle.pgid, signal.SIGKILL)
        handle.process.wait(timeout=10)  # every sibling has exited, so the loop polls them all
        workers.append(_HostStub(f"u{index}", handle))
    supervisor._workers = {("call", w.unit_id, 1): w for w in workers}
    before = supervisor._process_table.scans
    assert supervisor._other_live_workers(_HostStub("asker", None)) is False
    assert supervisor._process_table.scans - before == 1


def test_cancel_all_decides_who_is_live_from_one_scan(tmp_path, procs, monkeypatch):
    supervisor = ParallelSupervisor(tmp_path / "run")
    cancelled = []
    monkeypatch.setattr(WorkerHandle, "cancel", lambda self, reason: cancelled.append(self.pid))
    workers = []
    for index in range(6):
        handle = _live_handle(tmp_path, procs, supervisor._process_table)
        workers.append(_HostStub(f"u{index}", handle))
    supervisor._workers = {("call", w.unit_id, 1): w for w in workers}
    monkeypatch.setattr(supervisor, "_persist_worker", lambda worker: None)
    before = supervisor._process_table.scans
    supervisor.cancel_all("stop")
    assert supervisor._process_table.scans - before == 1
    assert sorted(cancelled) == sorted(w.handle.pid for w in workers)


class _HostStub:
    generation = 0
    """The fields of a supervisor worker that the poll loops read."""

    def __init__(self, unit_id, handle):
        self.unit_id = unit_id
        self.call_id = "call"
        self.handle = handle
        self.preparing = False
        self.thread = None
        self.error = None
        self.cancel_requested = None


# ---- B15: liveness by birth, outside the lock ----------------------------------------------

def test_recycled_group_number_is_not_a_live_group(tmp_path):
    reaped_at = (500, 0)
    stranger = _row(4242, 4242, 900)  # born after the reap: a recycled pid leading its own group
    table = _FakeTable({4242: stranger, 4300: _row(4300, 4242, 901, ppid=4242)})
    handle = _reaped_handle(tmp_path, table, reaped_bound=reaped_at)
    assert handle.group_live() is False


def test_member_born_before_the_reap_is_still_a_live_group(tmp_path):
    table = _FakeTable({4300: _row(4300, 4242, 400)})
    handle = _reaped_handle(tmp_path, table, reaped_bound=(500, 0))
    assert handle.group_live() is True


def test_group_before_the_reap_is_reported_counts_every_member(tmp_path):
    table = _FakeTable({4300: _row(4300, 4242, 900)})
    handle = _reaped_handle(tmp_path, table, reaped_bound=None)
    assert handle.group_live() is True


def test_zombie_members_are_not_live(tmp_path):
    table = _FakeTable({4300: _row(4300, 4242, 400, zombie=True)})
    handle = _reaped_handle(tmp_path, table, reaped_bound=(500, 0))
    assert handle.group_live() is False


def test_group_probe_failure_is_containment_unverified(tmp_path):
    class Broken(_FakeTable):
        def snapshot(self):
            raise OSError("proc unreadable")

    handle = _reaped_handle(tmp_path, Broken({}), reaped_bound=(500, 0))
    with pytest.raises(ContainmentUnverified):
        handle.group_live()


def _finalizing_worker(tmp_path, handle):
    call, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(tmp_path / "run")
    supervisor = ParallelSupervisor(tmp_path / "run")
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(),
                     run_dir, snapshot, spec.start_manifest_digest, {},
                     state="RUNNING", handle=handle, pid=4242, pgid=4242)
    return supervisor, worker


class _ExitedHandle:
    pgid = 4242
    timed_out = False
    cancel_reason = None

    def __init__(self, group_live):
        self._group_live = group_live

    def reap(self):
        return 0

    def group_live(self):
        return self._group_live()


def test_exit_is_numbered_when_the_handle_sees_no_live_group_even_if_the_number_is_reused(
        tmp_path, monkeypatch):
    # The pgid-number probe says "live" (a stranger leads a recycled group); the birth-aware
    # handle says the worker's own group is gone. The exit must be numbered, not retried forever.
    monkeypatch.setattr(host, "_group_has_live_processes", lambda pgid, _table=None: True)
    supervisor, worker = _finalizing_worker(tmp_path, _ExitedHandle(lambda: False))
    supervisor._finalize_worker(worker, 0)
    assert worker.state == "EXITED"
    assert worker.exited_seq == 1


def test_exit_stays_unnumbered_while_the_handle_sees_a_live_group(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "_group_has_live_processes", lambda pgid, _table=None: False)
    supervisor, worker = _finalizing_worker(tmp_path, _ExitedHandle(lambda: True))
    supervisor._finalize_worker(worker, 0)
    assert worker.state == "RUNNING"
    assert worker.exited_seq is None
    assert "process group is live" in worker.error


def test_finalization_probes_the_group_without_holding_the_host_lock(tmp_path):
    seen: list[bool] = []

    def probe():
        acquired = threading.Event()

        def take():
            with supervisor._lock:
                acquired.set()

        thread = threading.Thread(target=take, daemon=True)
        thread.start()
        seen.append(acquired.wait(1.0))
        thread.join(timeout=2)
        return False

    supervisor, worker = _finalizing_worker(tmp_path, _ExitedHandle(probe))
    supervisor._finalize_worker(worker, 0)
    assert seen == [True], "the group probe ran while the host lock was held"
    assert worker.state == "EXITED"


def test_seal_probe_uses_the_handle_not_the_pgid_number(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "_group_has_live_processes", lambda pgid, _table=None: True)
    supervisor, worker = _finalizing_worker(tmp_path, _ExitedHandle(lambda: False))
    assert supervisor._group_live(worker) is False
    worker.handle = _ExitedHandle(lambda: True)
    assert supervisor._group_live(worker) is True


def test_handle_probe_failure_defers_instead_of_failing(tmp_path):
    def broken():
        raise ContainmentUnverified("scan failed")

    supervisor, worker = _finalizing_worker(tmp_path, _ExitedHandle(broken))
    with pytest.raises(host.ProcessInspectionDeferred):
        supervisor._group_live(worker)


def test_recordless_worker_falls_back_to_the_pgid_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "_group_has_live_processes", lambda pgid, _table=None: pgid == 4242)
    supervisor, worker = _finalizing_worker(tmp_path, None)
    assert supervisor._group_live(worker) is True


# ---- B16: run wall ---------------------------------------------------------------------------

def _wall_handle(tmp_path, monkeypatch, environment, clock):
    (tmp_path / "spec.json").write_text(json.dumps({"environment": environment}))
    monkeypatch.setattr(worker_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(worker_module, "load_tolerant_seconds", lambda name: 5.0)
    monkeypatch.setattr(WorkerHandle, "poll", lambda self: None)
    signals = []
    monkeypatch.setattr(WorkerHandle, "_signal_group", lambda self, sig: signals.append(sig))
    handle = WorkerHandle(SimpleNamespace(), 9876, 9876, 1, tmp_path, tmp_path / "log", 1000.0, 0.0,
                          SimpleNamespace(), launch_identity=LaunchIdentity(9876, None, None))
    return handle, signals


def test_worker_without_a_run_wall_is_unbounded(tmp_path, monkeypatch):
    clock = [0.0]
    handle, signals = _wall_handle(tmp_path, monkeypatch, {}, clock)
    handle.enforce_deadline()
    assert not handle.timed_out
    clock[0] = 4 * 3600 + 6.0
    handle.enforce_deadline()
    assert not handle.timed_out and not signals


def test_explicit_nonpositive_wall_means_no_deadline(tmp_path, monkeypatch):
    clock = [0.0]
    handle, signals = _wall_handle(tmp_path, monkeypatch, {"STATEFRAME_RUN_WALL_SECONDS": "0"}, clock)
    clock[0] = 40 * 3600
    handle.enforce_deadline()
    assert not handle.timed_out and not signals


def test_spec_is_read_once_not_every_tick(tmp_path, monkeypatch):
    clock = [0.0]
    handle, signals = _wall_handle(tmp_path, monkeypatch, {"STATEFRAME_RUN_WALL_SECONDS": "60"}, clock)
    handle.enforce_deadline()
    (tmp_path / "spec.json").write_text("{ not json")  # a later torn rewrite must not cancel the worker
    for _ in range(3):
        handle.enforce_deadline()
    assert not handle.timed_out and not signals
    clock[0] = 66.0
    handle.enforce_deadline()
    assert handle.timed_out


def test_wall_clock_starts_at_the_slot_lease_not_at_launch(tmp_path, monkeypatch):
    clock = [0.0]
    handle, signals = _wall_handle(tmp_path, monkeypatch, {"STATEFRAME_RUN_WALL_SECONDS": "60"}, clock)
    handle.note_running(1050.0)  # acquired the slot 50 s after launch (started_at is 1000)
    clock[0] = 100.0  # launch + 100 s: past 60 + 5 from launch, but only 50 s into the run
    handle.enforce_deadline()
    assert not handle.timed_out
    clock[0] = 116.0
    handle.enforce_deadline()
    assert handle.timed_out and signals == [signal.SIGTERM]


def test_corrupt_wall_value_is_still_refused(tmp_path, monkeypatch):
    clock = [0.0]
    handle, _ = _wall_handle(tmp_path, monkeypatch, {"STATEFRAME_RUN_WALL_SECONDS": "soon"}, clock)
    with pytest.raises(worker_module.WorkerLaunchError, match="not numeric"):
        handle.enforce_deadline()


# ---- B12: relaunch copy outside the host lock ------------------------------------------------

def test_relaunch_copy_runs_before_and_outside_the_host_lock(tmp_path, monkeypatch):
    call, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(tmp_path / "run")
    (tmp_path / "run" / "spec.json").write_text("{}")  # the parent spec the launch fence reads
    supervisor = ParallelSupervisor(tmp_path / "run")
    worker = _Worker("call", "a", "execution", 2, call, spec_path, spec.digest(),
                     run_dir, snapshot, spec.start_manifest_digest, {})
    worker.preparing = True
    events: list[str] = []

    def copy(run, incarnation):
        acquired = threading.Event()

        def take():
            with supervisor._lock:
                acquired.set()

        thread = threading.Thread(target=take, daemon=True)
        thread.start()
        events.append("copy:free" if acquired.wait(1.0) else "copy:locked")
        thread.join(timeout=2)

    def launch(run, incarnation, prepared=False, *, admission):
        events.append(f"launch:prepared={prepared}")
        with admission() as install:
            assert callable(install)
        raise host.WorkerLaunchError("stop before the process starts")

    monkeypatch.setattr(host, "prepare_worker", lambda *a, **k: None)
    monkeypatch.setattr(host, "prepare_relaunch", copy)
    monkeypatch.setattr(host, "launch_worker", launch)
    monkeypatch.setattr(host, "read_snapshot_manifest",
                        lambda path, digest: SimpleNamespace(files={}, source_digest=spec.start_manifest_digest,
                                                             digest=digest))
    monkeypatch.setattr(host, "reserve_preparation", _ready_reservation)
    supervisor._prepare_and_launch(worker)
    assert events == ["copy:free", "launch:prepared=True"], (worker.error, supervisor._protocol_errors)


class _ready_reservation:
    def __init__(self, *args):
        pass

    def __enter__(self):
        return True, None

    def __exit__(self, *exc):
        return False


def test_prepared_relaunch_skips_the_copy_inside_launch_worker(tmp_path, monkeypatch):
    run_dir = tmp_path / "unit"
    run_dir.mkdir()

    def forbidden(*args, **kwargs):
        pytest.fail("launch_worker repeated the relaunch preparation")

    monkeypatch.setattr(worker_module, "prepare_relaunch", forbidden)
    monkeypatch.setattr(worker_module, "_launch_spec",
                        lambda run: (_ for _ in ()).throw(worker_module.WorkerLaunchError("stop")))
    with pytest.raises(worker_module.WorkerLaunchError, match="stop"):
        worker_module.launch_worker(run_dir, 2, True)


def test_unprepared_relaunch_still_prepares_inside_launch_worker(tmp_path, monkeypatch):
    run_dir = tmp_path / "unit"
    run_dir.mkdir()
    prepared = []
    monkeypatch.setattr(worker_module, "prepare_relaunch", lambda run, inc: prepared.append(inc))
    monkeypatch.setattr(worker_module, "_launch_spec",
                        lambda run: (_ for _ in ()).throw(worker_module.WorkerLaunchError("stop")))
    with pytest.raises(worker_module.WorkerLaunchError, match="stop"):
        worker_module.launch_worker(run_dir, 2)
    assert prepared == [2]


def test_first_incarnation_needs_no_relaunch_preparation(tmp_path):
    worker_module.prepare_relaunch(tmp_path, 1)  # nothing to rebuild, nothing read
    with pytest.raises(ValueError):
        worker_module.prepare_relaunch(tmp_path, 0)


def test_slot_lease_marker_starts_the_handles_wall_clock(tmp_path):
    seen: list[float] = []

    class Handle:
        def note_running(self, epoch):
            seen.append(epoch)

    supervisor, worker = _finalizing_worker(tmp_path, Handle())
    worker.state = "STARTING"
    marker = worker.run_dir / "proc" / "1"
    marker.mkdir(parents=True)
    acquired = worker.created_epoch + 0.5
    (marker / "slot-lease.json").write_text(json.dumps({
        "schema": "stateframe-subframe-slot-lease/1", "slot": 3, "pid": worker.pid,
        "acquired_epoch": acquired}))
    supervisor._notice_running(worker)
    assert worker.state == "RUNNING" and worker.started_epoch == acquired
    assert seen == [acquired]


# ---- durable writes on the tick -------------------------------------------------------------

def _status_request_run(tmp_path):
    from test_parallel_host import _request, _write_request

    run = tmp_path / "run"
    attempt = run / "parallel" / "call" / "attempt-1"
    (attempt / "rq").mkdir(parents=True)
    request = _request(1, "call", 1, "status", {"unit_id": "a"})
    _write_request(attempt / "rq" / "001.json", request)
    return run, attempt


def test_tick_delegates_durable_answer_and_never_fsyncs(tmp_path, monkeypatch):
    run, attempt = _status_request_run(tmp_path)
    supervisor = ParallelSupervisor(run)
    synced: list[int] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda descriptor: (synced.append(descriptor), real(descriptor))[1])
    supervisor.tick()
    answer = attempt / "an" / "001.json"
    deadline = time.monotonic() + 15
    while not answer.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert answer.is_file()
    supervisor._stop_runtime()
    # The answer and its record are published durably (file + directory each); the channel
    # cursors and the latency metrics are not: a stalled journal commit must not stall the tick.
    assert synced == [], synced
    assert supervisor.tick_count == 1
    assert json.loads((attempt / "channel-cursor.json").read_text())["last"] == 1


@pytest.mark.parametrize("damage", ["", "{ torn", "[]", '{"last": '])
def test_a_damaged_cursor_restarts_service_instead_of_failing_the_tick(tmp_path, damage):
    run, attempt = _status_request_run(tmp_path)
    (attempt / "channel-cursor.json").write_text(damage)
    (run / "parallel" / "channel-cursor.json").write_text(damage)
    supervisor = ParallelSupervisor(run)
    answer = attempt / "an" / "001.json"
    deadline = time.monotonic() + 15
    while not answer.is_file() and time.monotonic() < deadline:
        supervisor._service_channels()
    assert answer.is_file()


def test_non_durable_write_is_still_atomic_and_leaves_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "cursor.json"
    target.write_text('{"last": 1}\n')
    monkeypatch.setattr(os, "fsync", lambda descriptor: pytest.fail("a non-durable write fsynced"))
    host._write_json(target, {"last": 2}, replace_existing=True, durable=False)
    assert json.loads(target.read_text()) == {"last": 2}
    assert [path.name for path in tmp_path.iterdir()] == ["cursor.json"]


def test_a_tick_over_many_live_workers_takes_one_scan(tmp_path, procs):
    call, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(tmp_path / "run")
    (tmp_path / "spec.json").write_text(json.dumps({"environment": {"STATEFRAME_RUN_WALL_SECONDS": "3600"}}))
    supervisor = ParallelSupervisor(tmp_path / "run")
    for index in range(8):
        handle = _live_handle(tmp_path, procs, supervisor._process_table)
        worker = _Worker("call", f"u{index}", "execution", 1, call, spec_path, spec.digest(),
                         run_dir, snapshot, spec.start_manifest_digest, {},
                         state="RUNNING", handle=handle, pid=handle.pid, pgid=handle.pgid)
        supervisor._workers[("call", f"u{index}", 1)] = worker
    supervisor._advance_workers()  # warm the per-handle state
    before = supervisor._process_table.scans
    supervisor._advance_workers()
    assert supervisor._process_table.scans - before == 1


def test_a_sealed_job_is_not_statted_again_on_every_tick(tmp_path, monkeypatch):
    from test_parallel_host import _request

    run = tmp_path / "run"
    seals = run / "parallel" / "call" / "host" / "seals"
    seals.mkdir(parents=True)
    job = seals / "a-1.json"
    result = seals / "a-1.result.json"
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    job.write_text(json.dumps({"schema": "stateframe-parallel-seal-job/2",
                               "operation_id": host.operation_id(request), "request": request.to_json()}))
    result.write_text(json.dumps({"status": "OK", "payload": {"sealed": True}}))
    supervisor = ParallelSupervisor(run)
    supervisor._discover_seals()
    assert supervisor._sealed_jobs == {job}
    real = type(result).exists
    probes: list[str] = []

    def exists(path):
        probes.append(path.name)
        return real(path)

    monkeypatch.setattr(type(result), "exists", exists)
    for _ in range(3):
        supervisor._discover_seals()
    assert "a-1.result.json" not in probes


# ---- one observation per view ------------------------------------------------------------------

def _count_updates(monkeypatch):
    from stateframe.engine.kernel_exec_tracker import Descendants

    calls = []
    real = Descendants.update
    monkeypatch.setattr(Descendants, "update", lambda self, rows: (calls.append(1), real(self, rows))[1])
    return calls


def test_a_handle_folds_each_tick_view_in_once(tmp_path, procs, monkeypatch):
    table = TickProcessTable()
    handle = _live_handle(tmp_path, procs, table)
    calls = _count_updates(monkeypatch)
    with table.tick():
        for _ in range(3):  # enforce_deadline, escalate and the host's poll all observe
            handle.poll()
        assert len(calls) == 1
    with table.tick():
        handle.poll()
        handle.poll()
    assert len(calls) == 2


def test_a_signal_makes_the_next_observation_fold_in_a_fresh_view(tmp_path, procs, monkeypatch):
    table = TickProcessTable()
    handle = _live_handle(tmp_path, procs, table)
    calls = _count_updates(monkeypatch)
    with table.tick():
        handle.poll()
        table.invalidate()
        handle.poll()
        handle.poll()
    assert len(calls) == 2


def test_two_handles_each_fold_the_shared_view_in(tmp_path, procs, monkeypatch):
    table = TickProcessTable()
    first = _live_handle(tmp_path, procs, table)
    second = _live_handle(tmp_path, procs, table)
    calls = _count_updates(monkeypatch)
    before = table.scans
    with table.tick():
        first.poll()
        second.poll()
        first.poll()
        second.poll()
    assert len(calls) == 2 and table.scans - before == 1


def test_a_view_a_signal_overlapped_is_never_reused(tmp_path, procs, monkeypatch):
    table = TickProcessTable()
    handle = _live_handle(tmp_path, procs, table)
    real = table._scan

    def scan_then_signal():
        rows = real()
        table.invalidate()  # a signal lands while the scan is being taken
        return rows

    monkeypatch.setattr(table, "_scan", scan_then_signal)
    calls = _count_updates(monkeypatch)
    with table.tick():
        assert table.view_token() is None
        handle.poll()
        assert table.view_token() is None
        handle.poll()
    assert len(calls) == 2


def test_status_served_from_a_tick_does_not_observe_the_worker_again(tmp_path):
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    supervisor._workers[("call", "a", 1)] = worker
    request = SimpleNamespace(payload={"unit_id": "a"}, call_id="call")
    worker.state = "RUNNING"
    supervisor._tick_deadline = time.monotonic() + 1
    supervisor._status(request, worker.call_root)
    assert worker.handle.polls == 0
    supervisor._tick_deadline = None
    supervisor._status(request, worker.call_root)
    assert worker.handle.polls == 0


class _CountingHandle:
    pgid = 4242
    polls = 0
    timed_out = False
    cancel_reason = None

    def enforce_deadline(self):
        pass

    def escalate(self):
        pass

    def poll(self):
        self.polls += 1
        return None
