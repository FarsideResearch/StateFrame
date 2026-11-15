"""Engine-free checks for workers known only by their durable record (no handle): cancelling one
never blocks the channel thread, and the lifecycle lane supervises it after resume."""
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from test_parallel_host import _request, _spec_and_snapshot, _write_request

import stateframe.parallel_host as host
from stateframe.parallel_host import ParallelSupervisor, _Worker
from stateframe.parallel_worker import kernel_start_identity

_PASS_BUDGET_SECONDS = 0.1  # a channel request must never pay a cleanup window
_LIFECYCLE_PASS_BUDGET_SECONDS = 1.0  # the scale gate bound for a lifecycle pass; a transition pass scans and fsyncs


@pytest.fixture(autouse=True)
def _unit_storage_barriers(monkeypatch):
    """Allow explicit storage diagnosis; the normal timing proof keeps real fsync calls.

    On a congested host, use private tmpfs storage under the required work directory for
    the proof. Skipping barriers is only a diagnostic option, never the default.
    """
    if os.environ.get("SF_TEST_RECORDED_SKIP_FSYNC") == "1":
        monkeypatch.setattr(os, "fsync", lambda descriptor: None)


def _status(supervisor, worker):
    from types import SimpleNamespace
    return supervisor._status(SimpleNamespace(call_id="call", payload={"unit_id": "a"}), worker.call_root)


def _leader(code="import time; time.sleep(60)", *, ignore_term=False):
    """A session leader (pid == pgid) that is not a child of any supervisor."""
    setup = "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else ""
    command = setup + "print('ready', flush=True); " + code
    process = subprocess.Popen([sys.executable, "-c", command], start_new_session=True,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert select.select([process.stdout], [], [], 5)[0], "fixture did not install its signal handler"
    assert process.stdout.readline() == b"ready\n"
    process.stdout.close()
    assert os.getpgid(process.pid) == process.pid
    return process


def _kill_leader(process):
    if process.poll() is None:
        if sys.platform == "linux":
            assert process.args[2].encode() in Path(f"/proc/{process.pid}/cmdline").read_bytes()
        process.kill()
        process.wait(timeout=5)


def _record(tmp_path, *, state, pid, queued=False, cancel_requested=None):
    """Persist a worker the way a host that died mid-run leaves it, then forget the supervisor."""
    call, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(tmp_path / "run")
    supervisor = ParallelSupervisor(tmp_path / "run")
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), run_dir, snapshot,
                     spec.start_manifest_digest, {}, snapshot_manifest_digest="0" * 64,
                     state=state, pid=pid, pgid=pid, queued=queued,
                     cancel_requested=cancel_requested, started_epoch=time.time() if state == "RUNNING" else None)
    supervisor._persist_worker(worker, launch=True)
    (run_dir / "spec.json").write_text(json.dumps({"engine_dir": str(tmp_path / "engine" / "stateframe")}))
    return worker


def _process_record(worker, process):
    proc = worker.run_dir / "proc" / "1"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "process.json").write_text(json.dumps({
        "incarnation": 1, "pid": process.pid, "pgid": process.pid,
        "kernel_identity": kernel_start_identity(process.pid), "verified_members": {}}))


def _family_exit_proof(worker, process):
    """These fixtures launch one sleeping process with no children; its owner proves cleanup."""
    assert process.poll() is not None
    proc = worker.run_dir / "proc" / "1"
    record = json.loads((proc / "process.json").read_text())
    (proc / "family-exited.json").write_text(json.dumps({
        key: record[key] for key in ("pid", "pgid", "kernel_identity")}))


def _passes(supervisor, until, *, limit=60):
    """Run lifecycle passes at the lane's cadence; every pass must stay inside the channel budget."""
    for _ in range(limit):
        started = time.monotonic()
        supervisor._advance_workers()
        assert time.monotonic() - started < _LIFECYCLE_PASS_BUDGET_SECONDS, "a lifecycle pass slept through a window"
        if until():
            return
        time.sleep(0.02)
    raise AssertionError("the lifecycle lane never reached the expected state")


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="recorded-group fencing needs pidfd or ps")
def test_body5_cancel_answers_before_inspection_and_lifecycle_establishes_containment(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "cleanup_seconds", lambda: 0.3)
    process = _leader(ignore_term=True)
    try:
        worker = _record(tmp_path, state="RUNNING", pid=process.pid)
        _process_record(worker, process)
        supervisor = ParallelSupervisor(tmp_path / "run")
        resumed = supervisor._workers[("call", "a", 1)]
        assert resumed.handle is None and resumed.state == "RUNNING"

        channel = resumed.call_root / "attempt-1"
        rq, an = channel / "rq", channel / "an"
        _write_request(rq / "001.json", _request(1, "call", 1, "cancel", {"unit_id": "a", "reason": "stop"}))
        signals = []
        real_signal = host._signal_recorded_group

        def signal_group(*args):
            signals.append(args[4])
            return real_signal(*args)

        monkeypatch.setattr(host, "_signal_recorded_group", signal_group)
        scans = supervisor._process_table.scans
        started = time.monotonic()
        answers = supervisor.serve(rq, an)
        elapsed = time.monotonic() - started
        assert elapsed < _PASS_BUDGET_SECONDS, f"cancel slept through the cleanup window: {elapsed:.3f}s"
        assert answers[0].payload == {"state": "CANCEL_REQUESTED"}
        assert json.loads((an / "001.json").read_text())["payload"] == {"state": "CANCEL_REQUESTED"}
        assert signals == [] and supervisor._process_table.scans == scans
        assert resumed.fence_stage == "PENDING" and resumed.state == "RUNNING"
        assert process.poll() is None
        assert json.loads(resumed.state_path.read_text())["cancel_requested"] == "stop"

        for seq in range(2, 62):
            _write_request(rq / f"{seq:03d}.json", _request(1, "call", seq, "status", {"unit_id": "a"}))
            started = time.monotonic()
            supervisor.serve(rq, an)
            assert time.monotonic() - started < _PASS_BUDGET_SECONDS, "channel pass blocked during fencing"
            assert (an / f"{seq:03d}.json").is_file()
            if process.poll() is not None:
                _family_exit_proof(resumed, process)
            supervisor._advance_workers()
            if resumed.state == "EXITED":
                break
            assert resumed.exited_seq is None and _status(supervisor, resumed)["state"] != "EXITED"
            time.sleep(0.02)
        assert resumed.state == "EXITED"
        assert signals == [signal.SIGTERM, signal.SIGKILL]
        process.wait(timeout=5)
        assert resumed.terminal == "CANCELLED" and resumed.error == "stop" and resumed.exited_seq == 1
        assert json.loads((worker.run_dir / "proc" / "1" / "process.json").read_text())["fenced"] is True
        assert _status(supervisor, resumed)["state"] == "EXITED"
        # A second cancel after containment is a no-op answer, not a new fence.
        assert supervisor._cancel(_request(1, "call", 2, "cancel", {"unit_id": "a", "reason": "again"}),
                                  resumed.call_root) == {"state": "EXITED"}
    finally:
        _kill_leader(process)


def test_cancel_while_the_lifecycle_lane_fences_does_not_restart_the_window(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "cleanup_seconds", lambda: 30.0)
    _record(tmp_path, state="RUNNING", pid=4242)
    supervisor = ParallelSupervisor(tmp_path / "run")
    resumed = supervisor._workers[("call", "a", 1)]
    resumed.fence_stage, resumed.fence_deadline = "TERM", time.monotonic() + 5.0
    resumed.cancel_requested = "stop"
    signals = []
    monkeypatch.setattr(host, "_signal_recorded_group", lambda *args: signals.append(args[4]))
    deadline = resumed.fence_deadline
    answer = supervisor._cancel(_request(1, "call", 1, "cancel", {"unit_id": "a", "reason": "again"}),
                                resumed.call_root)
    assert answer == {"state": "CANCEL_REQUESTED"}
    assert signals == [] and resumed.fence_stage == "TERM" and resumed.fence_deadline == deadline
    assert resumed.cancel_requested == "again" and resumed.generation == 1


def test_fence_failure_is_logged_once_and_retried_only_by_a_new_cancel(tmp_path, monkeypatch):
    worker = _record(tmp_path, state="RUNNING", pid=4242)
    proc = worker.run_dir / "proc" / "1"
    proc.mkdir(parents=True)
    (proc / "process.json").write_text(json.dumps({"pid": 1, "pgid": 1, "kernel_identity": "darwin:1:1",
                                                   "verified_members": {}}))
    monkeypatch.setattr(host, "_verified_live_members", lambda pgid, expected: [(4242, "darwin:1:1")])
    supervisor = ParallelSupervisor(tmp_path / "run")
    resumed = supervisor._workers[("call", "a", 1)]
    resumed.cancel_requested = "stop"
    for _ in range(3):
        supervisor._advance_workers()
    assert resumed.fence_stage == "FAILED" and "differs from supervisor identity" in resumed.error
    assert sum("fence failed" in error for error in supervisor._protocol_errors) == 1
    assert resumed.state == "RUNNING"
    answer = supervisor._cancel(_request(1, "call", 1, "cancel", {"unit_id": "a", "reason": "stop"}), resumed.call_root)
    assert answer == {"state": "CANCEL_REQUESTED"} and resumed.fence_stage == "PENDING"
    supervisor._advance_workers()
    assert resumed.fence_stage == "FAILED" and "differs from supervisor identity" in resumed.error


# ---- body 9: resume holes ----------------------------------------------------------------------

def test_body9_resume_requeues_a_starting_worker_that_never_got_a_process(tmp_path, monkeypatch):
    worker = _record(tmp_path, state="STARTING", pid=None, queued=False)
    supervisor = ParallelSupervisor(tmp_path / "run")
    resumed = supervisor._workers[("call", "a", 1)]
    assert resumed.queued is True and resumed.state == "STARTING" and resumed.pid is None
    assert json.loads(worker.state_path.read_text())["queued"] is True
    launched = []
    monkeypatch.setattr(supervisor, "_prepare_and_launch", lambda target: launched.append(target.incarnation))
    supervisor._advance_workers()
    assert resumed.preparing is True and resumed.queued is False
    resumed.thread.join(2)
    assert launched == [1], "the same incarnation is relaunched; its proc dir is empty"
    # The kernel's idempotent relaunch of the same incarnation keeps answering PENDING meanwhile.
    assert _status(supervisor, resumed)["state"] == "STARTING"


def test_resume_cancels_an_unstarted_worker_whose_cancel_was_recorded(tmp_path):
    _record(tmp_path, state="STARTING", pid=None, queued=False, cancel_requested="parent cancelled")
    supervisor = ParallelSupervisor(tmp_path / "run")
    resumed = supervisor._workers[("call", "a", 1)]
    assert resumed.state == "EXITED" and resumed.terminal == "CANCELLED" and resumed.exited_seq == 1
    assert resumed.queued is False


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="recorded-group probes need /proc or ps")
def test_resume_adopts_a_process_record_the_host_record_never_received(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "_terminal", lambda run_dir: "SUBMITTED")
    process = _leader("import time; time.sleep(60)")
    try:
        worker = _record(tmp_path, state="STARTING", pid=None, queued=False)
        _process_record(worker, process)
        supervisor = ParallelSupervisor(tmp_path / "run")
        resumed = supervisor._workers[("call", "a", 1)]
        assert (resumed.pid, resumed.pgid) == (process.pid, process.pid)
        assert resumed.state == "STARTING" and resumed.queued is False
        assert json.loads(worker.state_path.read_text())["pid"] == process.pid
        supervisor._advance_workers()
        assert resumed.state == "STARTING" and not resumed.preparing  # live: supervised, not relaunched
        _kill_leader(process)
        _family_exit_proof(resumed, process)
        _passes(supervisor, lambda: resumed.state == "EXITED", limit=2)
        assert resumed.terminal == "SUBMITTED" and resumed.exited_seq == 1
    finally:
        _kill_leader(process)


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="recorded-group probes need /proc or ps")
@pytest.mark.parametrize("state", ["STARTING", "RUNNING"])
def test_body9_resumed_worker_is_finalized_when_its_group_exits_later(tmp_path, monkeypatch, state):
    monkeypatch.setattr(host, "_terminal", lambda run_dir: "SUBMITTED")
    process = _leader("import time; time.sleep(60)")
    try:
        worker = _record(tmp_path, state=state, pid=process.pid)
        _process_record(worker, process)
        supervisor = ParallelSupervisor(tmp_path / "run")
        resumed = supervisor._workers[("call", "a", 1)]
        assert resumed.state == state and resumed.handle is None
        supervisor._advance_workers()
        assert resumed.state == state  # live group: nothing to finalize
        _kill_leader(process)
        _family_exit_proof(resumed, process)
        _passes(supervisor, lambda: resumed.state == "EXITED", limit=2)
        assert resumed.terminal == "SUBMITTED" and resumed.error is None and resumed.exited_seq == 1
        assert _status(supervisor, resumed)["state"] == "EXITED"
        assert json.loads(worker.state_path.read_text())["state"] == "EXITED"
    finally:
        _kill_leader(process)


def test_lifecycle_probe_defers_on_a_failed_shared_scan(tmp_path, monkeypatch):
    import stateframe.parallel_worker as worker_mod
    worker = _record(tmp_path, state="RUNNING", pid=4242)
    monkeypatch.setattr(host, "_verified_live_members", lambda pgid, expected: [(4242, "darwin:4242:1")])
    proc = worker.run_dir / "proc" / "1"
    proc.mkdir(parents=True)
    (proc / "process.json").write_text(json.dumps({"pid": 4242, "pgid": 4242, "kernel_identity": "darwin:4242:1",
                                                   "verified_members": {}}))
    supervisor = ParallelSupervisor(tmp_path / "run")
    resumed = supervisor._workers[("call", "a", 1)]
    strict = []
    monkeypatch.setattr(host, "_recorded_group_live", lambda *args: strict.append(args) or True)

    def failing(self):
        raise OSError("proc denied")
    monkeypatch.setattr(worker_mod.ProcessTable, "snapshot", failing)
    supervisor._advance_workers()
    assert resumed.state == "RUNNING" and strict == [], "a failed scan is not evidence of an exit"
    observer = worker_mod.ProcessTable().identity(os.getpid())
    monkeypatch.setattr(worker_mod.ProcessTable, "snapshot", lambda self: {os.getpid(): observer})
    supervisor._advance_workers()
    assert strict and resumed.state == "RUNNING", "the identity-checked probe decides the transition"


def _synthetic_record(tmp_path, monkeypatch, *, cancel_requested=None):
    """Keep a recorded identity pending at hydration without creating a real signal target."""
    worker = _record(tmp_path, state="RUNNING", pid=4242, cancel_requested=cancel_requested)
    proc = worker.run_dir / "proc" / "1"
    proc.mkdir(parents=True)
    (proc / "process.json").write_text(json.dumps({
        "incarnation": 1, "pid": 4242, "pgid": 4242, "kernel_identity": "darwin:4242:1",
        "verified_members": {}}))
    monkeypatch.setattr(host, "_recorded_group_live", lambda *args: True)
    supervisor = ParallelSupervisor(tmp_path / "run")
    return supervisor, supervisor._workers[("call", "a", 1)]


@pytest.mark.parametrize("failure", ["empty", "denied", "observer_missing"])
@pytest.mark.parametrize("fresh", [False, True])
def test_fence_scan_failure_is_unknown_and_never_publishes_cancel_complete(tmp_path, monkeypatch, failure, fresh):
    supervisor, worker = _synthetic_record(tmp_path, monkeypatch)
    worker.cancel_requested = "stop"
    worker.fence_stage, worker.fence_deadline = "TERM", 0.0
    monkeypatch.setattr(host, "_recorded_group_live", lambda *args: False)
    observer = host.ProcessTable().identity(os.getpid())
    good = {os.getpid(): observer}

    def failed_scan():
        if failure == "denied":
            raise PermissionError("proc denied")
        return {} if failure == "empty" else {4243: observer}

    monkeypatch.setattr(supervisor._process_table, "tick_snapshot", lambda: good)
    monkeypatch.setattr(supervisor._process_table, "snapshot", lambda: good)
    monkeypatch.setattr(supervisor._process_table, "snapshot" if fresh else "tick_snapshot", failed_scan)
    for _ in range(3):
        supervisor._advance_workers()
        assert worker.state == "RUNNING" and worker.terminal is None and worker.exited_seq is None
        assert worker.fence_stage == "TERM", "UNKNOWN is retried, not a permanent fence failure"
        assert _status(supervisor, worker)["state"] == "RUNNING"
        assert "fenced" not in json.loads((worker.run_dir / "proc" / "1" / "process.json").read_text())

    monkeypatch.setattr(supervisor._process_table, "tick_snapshot", lambda: good)
    monkeypatch.setattr(supervisor._process_table, "snapshot", lambda: good)
    supervisor._advance_workers()
    assert worker.state == "EXITED" and worker.terminal == "CANCELLED" and worker.exited_seq == 1


def test_fence_does_not_finish_from_a_cached_clear_group_when_fresh_scan_finds_a_member(tmp_path, monkeypatch):
    from dataclasses import replace
    supervisor, worker = _synthetic_record(tmp_path, monkeypatch)
    worker.cancel_requested = "stop"
    worker.fence_stage, worker.fence_deadline = "TERM", 0.0
    monkeypatch.setattr(host, "_recorded_group_live", lambda *args: False)
    observer = host.ProcessTable().identity(os.getpid())
    monkeypatch.setattr(supervisor._process_table, "tick_snapshot", lambda: {os.getpid(): observer})
    monkeypatch.setattr(supervisor._process_table, "snapshot", lambda: {
        os.getpid(): observer, 4242: replace(observer, pid=4242, pgid=4242, zombie=False)})
    supervisor._advance_workers()
    assert worker.state == "RUNNING" and worker.exited_seq is None and worker.fence_stage == "TERM"


@pytest.mark.parametrize("stage", ["PENDING", "TERM"])
def test_signal_retries_deferred_inspection_without_waiting_for_another_cancel(tmp_path, monkeypatch, stage):
    from dataclasses import replace
    supervisor, worker = _synthetic_record(tmp_path, monkeypatch)
    answer = supervisor._cancel(_request(1, "call", 1, "cancel", {"unit_id": "a", "reason": "stop"}), worker.call_root)
    assert answer == {"state": "CANCEL_REQUESTED"} and worker.fence_stage == "PENDING"
    worker.fence_stage, worker.fence_deadline = stage, 0.0
    observer = host.ProcessTable().identity(os.getpid())
    monkeypatch.setattr(supervisor._process_table, "tick_snapshot", lambda: {
        os.getpid(): observer, 4242: replace(observer, pid=4242, pgid=4242, zombie=False)})

    def denied(*args):
        raise host.ProcessInspectionDeferred("scan unavailable")

    monkeypatch.setattr(host, "_signal_recorded_group", denied)
    supervisor._advance_workers()
    assert worker.fence_stage == stage and worker.state == "RUNNING"
    signals = []
    monkeypatch.setattr(host, "_signal_recorded_group", lambda *args: signals.append(args[4]))
    supervisor._advance_workers()
    expected_signal, expected_stage = (signal.SIGTERM, "TERM") if stage == "PENDING" else (signal.SIGKILL, "KILL")
    assert signals == [expected_signal] and worker.fence_stage == expected_stage


@pytest.mark.parametrize("cancel_requested", [None, "stop"])
@pytest.mark.parametrize("artifact", ["empty_dir", "launch_intent", "symlink"])
def test_resume_never_requeues_or_completes_cancel_with_ambiguous_launch_artifacts(
        tmp_path, monkeypatch, cancel_requested, artifact):
    worker = _record(tmp_path, state="STARTING", pid=None, queued=True, cancel_requested=cancel_requested)
    proc = worker.run_dir / "proc" / "1"
    proc.mkdir(parents=True)
    if artifact == "launch_intent":
        (proc / "launch-intent.json").write_text("{}")
    elif artifact == "symlink":
        (proc / "process.json").symlink_to(tmp_path / "missing-process.json")
    supervisor = ParallelSupervisor(tmp_path / "run")
    resumed = supervisor._workers[("call", "a", 1)]
    monkeypatch.setattr(supervisor, "_prepare_and_launch", lambda target: pytest.fail("ambiguous launch was repeated"))
    supervisor._advance_workers()
    assert resumed.state == "STARTING" and resumed.exited_seq is None
    assert resumed.queued is False and resumed.preparing is False
    assert "could not be recovered" in resumed.error
    assert json.loads(resumed.state_path.read_text())["queued"] is False
    expected_error = "contains a symlink" if artifact == "symlink" else "complete process identity"
    with pytest.raises(host.ParallelHostError, match=expected_error):
        supervisor._cancel(_request(1, "call", 1, "cancel", {"unit_id": "a", "reason": "stop"}), resumed.call_root)


def test_resume_preserves_a_recorded_cancel_when_the_group_has_already_exited(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "_terminal", lambda run_dir: "SUBMITTED")
    supervisor, worker = _synthetic_record(tmp_path, monkeypatch, cancel_requested="parent cancelled")
    assert worker.state == "RUNNING" and worker.cancel_requested == "parent cancelled"
    monkeypatch.setattr(host, "_signal_recorded_group", lambda *args: None)
    monkeypatch.setattr(host, "_recorded_group_live", lambda *args: False)
    _passes(supervisor, lambda: worker.state == "EXITED", limit=3)
    assert worker.terminal == "CANCELLED" and worker.error == "parent cancelled" and worker.exited_seq == 1


def test_hydrating_and_supervising_a_live_group_uses_only_the_shared_observation(tmp_path, monkeypatch):
    process = _leader()
    try:
        worker = _record(tmp_path, state="RUNNING", pid=process.pid)
        _process_record(worker, process)
        monkeypatch.setattr(host, "_recorded_group_live", lambda *args: pytest.fail("live group triggered a private scan"))
        supervisor = ParallelSupervisor(tmp_path / "run")
        resumed = supervisor._workers[("call", "a", 1)]
        assert supervisor._process_table.scans == 1
        supervisor._advance_workers()
        assert resumed.state == "RUNNING" and supervisor._process_table.scans == 2
    finally:
        _kill_leader(process)


def test_channel_stays_responsive_while_lifecycle_signal_inspection_is_stalled(tmp_path, monkeypatch):
    supervisor, worker = _synthetic_record(tmp_path, monkeypatch)
    request = _request(1, "call", 1, "cancel", {"unit_id": "a", "reason": "stop"})
    assert supervisor._cancel(request, worker.call_root) == {"state": "CANCEL_REQUESTED"}
    entered, release = threading.Event(), threading.Event()

    def stalled_signal(*args):
        entered.set()
        assert release.wait(5), "test did not release the signal probe"

    monkeypatch.setattr(host, "_signal_recorded_group", stalled_signal)
    lifecycle = threading.Thread(target=supervisor._advance_workers)
    lifecycle.start()
    try:
        assert entered.wait(5), "lifecycle never reached the signal probe"
        started = time.monotonic()
        assert supervisor._cancel(request, worker.call_root) == {"state": "CANCEL_REQUESTED"}
        assert _status(supervisor, worker)["state"] == "RUNNING"
        # An enqueue call that races with a lifecycle pass cannot enter its process probe.
        supervisor._fence_worker_record(worker, "stop")
        assert time.monotonic() - started < _PASS_BUDGET_SECONDS
        assert worker.fence_stage == "SIGNALLING" and worker.exited_seq is None
    finally:
        release.set()
        lifecycle.join(5)
    assert not lifecycle.is_alive() and worker.fence_stage == "TERM"


def test_fenced_cancel_waits_for_durable_exit_ack_before_reporting_complete(tmp_path, monkeypatch):
    from test_parallel_bridge import DelayedJournal
    supervisor, worker = _synthetic_record(tmp_path, monkeypatch)
    worker.cancel_requested = "stop"
    worker.fence_stage, worker.fence_deadline = "TERM", 0.0
    monkeypatch.setattr(host, "_recorded_group_live", lambda *args: False)
    observer = host.ProcessTable().identity(os.getpid())
    monkeypatch.setattr(supervisor._process_table, "tick_snapshot", lambda: {os.getpid(): observer})
    monkeypatch.setattr(supervisor._process_table, "snapshot", lambda: {os.getpid(): observer})
    monkeypatch.setattr(supervisor, "_write_json", host._write_json)
    journal = supervisor._journal = DelayedJournal()
    supervisor._advance_workers()
    assert worker.state == "RUNNING" and worker.exited_seq is None and worker.terminal is None
    assert len(journal.tickets) == 1 and journal.tickets[0].kind == "write"
    fence_write = journal.tickets[0]
    assert fence_write.payload["value"]["fenced"] is True
    host._write_json(Path(fence_write.payload["path"]), fence_write.payload["value"], replace_existing=True)
    fence_write.result = {"ok": True}
    supervisor._advance_workers()
    assert worker.state == "EXITING" and worker.exited_seq is None and worker.terminal is None
    assert _status(supervisor, worker)["state"] == "RUNNING"
    assert worker.commit_intent["value"]["terminal"] == "CANCELLED"
    assert json.loads((worker.run_dir / "proc" / "1" / "process.json").read_text())["fenced"] is True
    request = _request(1, "call", 1, "cancel", {"unit_id": "a", "reason": "again"})
    assert supervisor._cancel(request, worker.call_root) == {"state": "EXITING"}
    assert len(journal.tickets) == 2
    ticket = journal.tickets[1]
    assert ticket.kind == "commit"
    ticket.result = {"ok": True, "value": dict(ticket.payload["value"], exited_seq=1)}
    supervisor._pump_commits()
    assert _status(supervisor, worker)["state"] == "EXITED"
    assert worker.terminal == "CANCELLED" and worker.exited_seq == 1
