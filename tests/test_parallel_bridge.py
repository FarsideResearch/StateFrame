"""Engine-free regression checks for the bounded supervisor controller."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import stateframe.parallel_host as host
from stateframe.parallel_host import ParallelSupervisor


def test_tick_global_budget_survives_sixteen_ready_channels(tmp_path, monkeypatch):
    run = tmp_path / "run"
    for n in range(16):
        (run / "parallel" / f"c{n:02}" / "attempt-1" / "rq").mkdir(parents=True)
    supervisor = ParallelSupervisor(run)
    clock = [0.0]
    served = []
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervisor, "_advance_workers", lambda: None)
    monkeypatch.setattr(supervisor, "_advance_seals", lambda: None)
    def serve(rq, an):
        served.append(rq.parent.parent.name)
        clock[0] += 0.02
        return (object(),)
    monkeypatch.setattr(supervisor, "serve", serve)
    for _ in range(6):
        before = clock[0]
        supervisor._service_channels()
        assert clock[0] - before <= 0.071
    assert len(set(served)) == 16


def test_no_first_request_exemption_after_tick_deadline(tmp_path, monkeypatch):
    run = tmp_path / "run"
    rq = run / "parallel" / "call" / "attempt-1" / "rq"
    rq.mkdir(parents=True)
    (rq / "001.json").write_text("{}")
    supervisor = ParallelSupervisor(run)
    supervisor._tick_deadline = -1.0
    monkeypatch.setattr(supervisor, "_read_request", lambda *a: pytest.fail("expired request dispatched"))
    assert supervisor.serve(rq, rq.parent / "an") == ()


def test_status_is_memory_only_even_during_direct_serve(tmp_path, monkeypatch):
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    worker.state = "RUNNING"
    supervisor._workers[("call", "a", 1)] = worker
    class NoLock:
        def __enter__(self):
            pytest.fail("status waited on the host lock")
        def __exit__(self, *args):
            pass
    supervisor._lock = NoLock()
    fail = lambda *a, **kw: pytest.fail("status called a blocking dependency")
    for name in ("_latest_worker", "_advance_worker", "_slot_marker"):
        monkeypatch.setattr(supervisor, name, fail)
    original_stat = Path.stat
    with monkeypatch.context() as filesystem_spies:
        for name in ("read_bytes", "read_text", "stat", "iterdir", "is_file", "is_dir"):
            filesystem_spies.setattr(Path, name, fail)
        for deadline in (None, 10.0):
            supervisor._tick_deadline = deadline
            status = supervisor._status(SimpleNamespace(call_id="call", payload={"unit_id": "a"}), worker.call_root)
            assert status["state"] == "RUNNING"
    assert Path.stat is original_stat
    assert worker.call_root.is_dir()
    assert worker.handle.polls == 0


def test_tick_does_not_call_blocking_dependencies(tmp_path, monkeypatch):
    import threading
    supervisor = ParallelSupervisor(tmp_path / "run")
    owner = threading.get_ident()
    def forbid(original):
        def guarded(*args, **kwargs):
            assert threading.get_ident() != owner, "blocking dependency on tick thread"
            return original(*args, **kwargs)
        return guarded
    with monkeypatch.context() as spies:
        for target, names in ((Path, ("stat", "iterdir", "read_text", "read_bytes", "is_dir", "exists")),
                              (host.os, ("fsync",)), (host.subprocess, ("Popen",))):
            for name in names:
                spies.setattr(target, name, forbid(getattr(target, name)))
        supervisor.tick()
        supervisor._stop_runtime()


class DelayedJournal:
    epoch = 1
    def __init__(self, full=False):
        self.full = full
        self.tickets = []
    def submit(self, kind, payload):
        from stateframe.parallel_journal import JournalBusy, Ticket
        if self.full:
            raise JournalBusy("full")
        ticket = Ticket(str(len(self.tickets)), kind, payload)
        self.tickets.append(ticket)
        return ticket
    def take(self, ticket):
        result, ticket.result = ticket.result, None
        return result
    def close(self):
        pass


def test_exit_visibility_waits_for_durable_commit(tmp_path):
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    worker.state = "RUNNING"
    supervisor._workers[("call", "a", 1)] = worker
    worker.status_view = supervisor._status_value(worker)
    journal = supervisor._journal = DelayedJournal()
    worker.state, worker.terminal, worker.group_verified = "EXITED", "SUBMITTED", True
    supervisor._persist_worker(worker)
    request = SimpleNamespace(call_id="call", payload={"unit_id": "a"})
    for _ in range(100):
        supervisor.tick()
    assert worker.state == "EXITING"
    assert supervisor._status(request, worker.call_root)["state"] == "RUNNING"
    assert worker.exited_seq is None and worker.terminal is None
    value = dict(journal.tickets[0].payload["value"], exited_seq=7)
    journal.tickets[0].result = {"ok": True, "value": value}
    supervisor._pump_commits()
    assert supervisor._status(request, worker.call_root)["state"] == "EXITED"
    assert worker.exited_seq == 7 and worker.terminal == "SUBMITTED"


def test_tick_is_bounded_with_all_queues_full(tmp_path):
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    supervisor._journal = DelayedJournal(full=True)
    worker.state, worker.group_verified = "EXITED", True
    supervisor._persist_worker(worker)
    intent = worker.commit_intent
    started = host.time.monotonic()
    for _ in range(1000):
        supervisor.tick()
    assert host.time.monotonic() - started < 1.0
    supervisor._pump_commits()
    assert worker.commit_intent is intent and worker.state == "EXITING"
    assert worker.commit_ticket is None and worker.exited_seq is None


def test_journal_bounds_include_unconsumed_completions():
    import queue
    import threading
    from stateframe.parallel_journal import Journal, JournalBusy
    journal = Journal.__new__(Journal)
    journal._closed = False
    journal._state_lock = threading.Lock()
    journal._credits = threading.BoundedSemaphore(64)
    journal._queue = queue.Queue(65)
    tickets = [journal.submit("write", {"path": "unused", "value": n}) for n in range(64)]
    for ticket in tickets:
        ticket.result = {"ok": True}
        ticket.done.set()
        journal._queue.get_nowait()
    with pytest.raises(JournalBusy):
        journal.submit("write", {})
    assert journal.take(tickets[0]) == {"ok": True}
    journal.submit("write", {})
    journal.close()
    assert journal._queue.qsize() == 2  # one ordinary message plus reserved shutdown


def test_worker_watchdog_progresses_under_backpressure(tmp_path, monkeypatch):
    import threading
    import time
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    worker.state = "RUNNING"
    supervisor._workers[("call", "a", 1)] = worker
    journal = DelayedJournal(full=True)
    monkeypatch.setattr(host, "Journal", lambda: journal)
    deadlines = []
    monkeypatch.setattr(worker.handle, "enforce_deadline", lambda: deadlines.append(time.monotonic()))
    blocked = threading.Event()
    monkeypatch.setattr(supervisor, "_service_channels", lambda: blocked.wait(2))
    monkeypatch.setattr(supervisor, "_advance_seals", lambda: None)
    monkeypatch.setattr(supervisor, "_notice_running", lambda worker: None)
    try:
        supervisor.tick()
        deadline = time.monotonic() + 2
        while worker.handle.polls < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert worker.handle.polls >= 3 and len(deadlines) >= 3
    finally:
        supervisor._stop_runtime()
        blocked.set()


@pytest.mark.parametrize("damage", ["empty", "denied", "partial", "missing_self"])
def test_empty_partial_denied_or_malformed_proc_is_not_quiescence(monkeypatch, damage):
    from stateframe.engine.kernel_exec_tracker import ProcessTable
    from stateframe.parallel_worker import TickProcessTable
    table = TickProcessTable()
    monkeypatch.setattr(host.os, "listdir", lambda path: [] if damage == "empty" else [str(host.os.getpid())])
    def identity(pid):
        if damage == "denied":
            raise PermissionError("denied")
        if damage == "partial":
            raise RuntimeError("partial stat")
        return None
    monkeypatch.setattr(ProcessTable, "identity", lambda self, pid: identity(pid))
    with pytest.raises((OSError, RuntimeError)):
        table.snapshot()


def test_group_live_includes_detached_descendants(tmp_path):
    from test_parallel_supervisor_scale import _reaped_handle, _FakeTable, _row
    child = _row(5050, 5050, 20)
    handle = _reaped_handle(tmp_path, _FakeTable({5050: child}), reaped_bound=(30, 0))
    handle._family.update = lambda rows: [child]
    assert handle.group_live(), "detached owned family member must block exit numbering and seals"


@pytest.mark.parametrize("change", ["incarnation", "generation", "nonce", "epoch", "started", "finished"])
def test_scan_fence_rejects_old_inflight_or_invalidated_results(tmp_path, change):
    from dataclasses import replace
    from stateframe.parallel_seals import SealOperation, QuiescenceProof
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    op = SealOperation(worker, SimpleNamespace(), tmp_path / "job", tmp_path / "result", worker.generation)
    proof = QuiescenceProof(worker.incarnation, worker.generation, op.nonce, 1, op.fence_ns, op.fence_ns + 1)
    assert op.accepts(proof, 1)
    delta = {"incarnation": {"incarnation": worker.incarnation + 1},
             "generation": {"generation": worker.generation + 1}, "nonce": {"nonce": "foreign"},
             "epoch": {"helper_epoch": 2}, "started": {"scan_started_ns": op.fence_ns - 1},
             "finished": {"scan_finished_ns": op.fence_ns - 1}}[change]
    assert not op.accepts(replace(proof, **delta), 1)


def test_helper_loss_invalidates_proofs_and_leaves_operation_pending(tmp_path):
    from stateframe.parallel_seals import SealOperation, QuiescenceProof
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    helper = DelayedJournal()
    supervisor._seal_helpers = [helper]
    op = SealOperation(worker, SimpleNamespace(), tmp_path / "job", tmp_path / "result", worker.generation)
    nonce = op.nonce
    op.phase = "PUBLISHING"
    op.proof = QuiescenceProof(1, worker.generation, nonce, 1, op.fence_ns, op.fence_ns)
    op.helper_epoch = 1
    op.ticket = helper.submit("seal_publish", {})
    op.ticket.result = {"ok": False, "lost": True}
    supervisor._seal_ops[op.job] = op
    supervisor._advance_seals()
    assert op.proof is None and op.staged is None and op.nonce != nonce
    assert op.phase == "PENDING" and not op.result.exists()


def test_owner_rejects_scan_started_before_fence(tmp_path, monkeypatch):
    from stateframe.parallel_seals import SealOperation
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    worker.state, worker.exited_seq, worker.pgid = "EXITED", 1, 4242
    supervisor._workers[("call", "a", 1)] = worker
    op = SealOperation(worker, SimpleNamespace(), tmp_path / "job", tmp_path / "result", 0)
    supervisor._process_table._tick_state.fresh_started_ns = op.fence_ns - 1
    monkeypatch.setattr(supervisor, "_group_live", lambda worker: False)
    with pytest.raises(host.ProcessInspectionDeferred):
        supervisor._fresh_seal_proof(op, 1)


@pytest.mark.parametrize("damage", ["spec_digest", "snapshot_digest", "unowned", "exec", "symlink", "oversize"])
def test_seal_helper_preserves_artifact_validation(tmp_path, damage):
    import time
    from test_parallel_host import _spec_and_snapshot, _Worker, _record_submission, _request
    from stateframe.parallel_journal import Journal
    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    (unit / "workspace" / "pkg").mkdir(parents=True)
    (unit / "workspace" / "pkg" / "a.py").write_text("VALUE = 1\n")
    writes = {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"}
    if damage == "unowned":
        writes["other.py"] = "foreign"
    elif damage == "oversize":
        writes["pkg/a.py"] = "x" * ((16 << 20) + 1)
    _record_submission(unit, writes)
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), unit,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._persist_worker(worker)
    if damage == "spec_digest":
        spec_path.write_text("{}")
    elif damage == "snapshot_digest":
        (snapshot / "snapshot-manifest.json").chmod(0o600)
        (snapshot / "snapshot-manifest.json").write_text("{}")
    elif damage == "exec":
        (unit / "workspace" / "pkg" / "a.py").chmod(0o755)
    elif damage == "symlink":
        (unit / "workspace" / "pkg" / "a.py").rename(unit / "original.py")
        (unit / "workspace" / "pkg" / "a.py").symlink_to(unit / "original.py")
    elif damage == "oversize":
        (unit / "workspace" / "foreign").write_bytes(b"x" * ((16 << 20) + 1))
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    helper = Journal(role="seal")
    try:
        ticket = helper.submit("seal_build", {"request": request.to_json(), "call_root": str(call),
                                             "token": [1, 0, "nonce"], "stage": str(call / "host/seals/test.stage.json")})
        deadline = time.monotonic() + 30
        while not ticket.done.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        result = helper.take(ticket)
        assert result is not None and result["ok"] is False and not result.get("lost"), result
        assert not (call / "proposals/a/proposal.json").exists()
    finally:
        helper.close()


def test_async_operation_replay_is_idempotent_across_restart(tmp_path, monkeypatch):
    from test_parallel_host import _spec_and_snapshot, _Worker, _request
    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    supervisor = ParallelSupervisor(run)
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), unit,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor._persist_worker(worker)
    monkeypatch.setattr(supervisor, "_start_runtime", lambda: None)
    first = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    second = _request(2, "call", 77, "seal", first.payload)
    a = supervisor._enqueue_seal(first, call)
    restarted = ParallelSupervisor(run)
    monkeypatch.setattr(restarted, "_start_runtime", lambda: None)
    b = restarted._enqueue_seal(second, call)
    assert a[1] == b[1]
    assert a[2]["operation_id"] == b[2]["operation_id"]
    assert len(list((call / "host/seals").glob("*.json"))) == 1


def test_async_operation_changed_digest_conflicts(tmp_path, monkeypatch):
    from dataclasses import replace
    from test_parallel_host import _spec_and_snapshot, _Worker, _request
    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    supervisor = ParallelSupervisor(run)
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), unit,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor._persist_worker(worker)
    monkeypatch.setattr(supervisor, "_start_runtime", lambda: None)
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    supervisor._enqueue_seal(request, call)
    with pytest.raises(host.ParallelHostError, match="conflict"):
        supervisor._enqueue_seal(replace(request, seq=2, scope="changed"), call)


def test_async_poll_preserves_original_response_deadline(monkeypatch):
    from stateframe.engine import kernel_parallel as parallel
    from stateframe.engine.kernel_parallel_records import HostAnswer
    clock = [0.0]
    monkeypatch.setenv("STATEFRAME_PARALLEL_HOST_RESPONSE_SECONDS", "5")
    monkeypatch.setattr(parallel.os, "getloadavg", lambda: (0, 0, 0))
    monkeypatch.setattr(parallel.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(parallel.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    class Host:
        def submit(self, *args):
            return "answer.json"
        def poll(self, ref):
            assert clock[0] < 6, "PENDING extended the response deadline"
            return HostAnswer(1, "call", 0, "seal", "PENDING", "operation.result.json", {})
    with pytest.raises(parallel.ParallelError, match="host response.*timed out"):
        parallel._host_answer(Host(), "seal", "a", {"unit_id": "a", "incarnation": 1}, None)
    assert clock[0] <= 5.21


@pytest.mark.parametrize("cancel", [False, True])
def test_launch_waits_for_durable_intent_and_honors_cancellation(tmp_path, monkeypatch, cancel):
    import time
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    journal = supervisor._journal = DelayedJournal()
    worker.state, worker.handle = "STARTING", None
    supervisor._persist_worker(worker, launch=True)
    launched = []
    monkeypatch.setattr(supervisor, "_prepare_and_launch", lambda worker: launched.append(worker))
    supervisor._start_preparation(worker)
    assert worker.queued and not worker.preparing and not launched
    if cancel:
        worker.cancel_requested = "cancel before commit ACK"
    ticket = journal.tickets[0]
    ticket.result = {"ok": True, "value": dict(ticket.payload["value"])}
    supervisor._pump_commits()
    supervisor._start_waiting_workers()
    deadline = time.monotonic() + 1
    while not launched and not cancel and time.monotonic() < deadline:
        time.sleep(0.01)
    assert bool(launched) is not cancel


def test_malformed_commit_ack_never_exposes_exit_sequence(tmp_path):
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    journal = supervisor._journal = DelayedJournal()
    worker.state, worker.group_verified = "EXITED", True
    supervisor._persist_worker(worker)
    journal.tickets[0].result = {"ok": True, "value": {"state": "EXITED", "exited_seq": 123}}
    supervisor._pump_commits()
    assert worker.state == "EXITING" and worker.exited_seq is None
    assert worker.commit_intent is not None


def test_seal_dispatch_never_waits_for_channel_publication_lock(tmp_path):
    import threading
    import time
    from stateframe.parallel_seals import SealOperation
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    worker.state, worker.exited_seq, worker.pgid = "EXITED", 1, None
    supervisor._workers[("call", "a", 1)] = worker
    op = SealOperation(worker, SimpleNamespace(to_json=lambda: {}), tmp_path / "job", tmp_path / "result", 0)
    supervisor._seal_ops[op.job] = op
    supervisor._seal_helpers = [DelayedJournal()]
    acquired, release = threading.Event(), threading.Event()
    def publisher():
        with supervisor._lock:
            acquired.set()
            release.wait(3)
    thread = threading.Thread(target=publisher)
    thread.start()
    assert acquired.wait(1)
    try:
        started = time.monotonic()
        supervisor._advance_seals()
        assert time.monotonic() - started < 0.5
        assert op.phase == "PENDING" and op.ticket is None
    finally:
        release.set()
        thread.join(2)


def test_writer_loss_invalidates_outstanding_seal_proofs(tmp_path, monkeypatch):
    from stateframe.parallel_seals import SealOperation, QuiescenceProof
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    supervisor._journal = DelayedJournal()
    helper = DelayedJournal()
    supervisor._seal_helpers = [helper]
    worker.state, worker.exited_seq, worker.pgid = "EXITED", 1, None
    supervisor._workers[("call", "a", 1)] = worker
    op = SealOperation(worker, SimpleNamespace(to_json=lambda: {}), tmp_path / "job", tmp_path / "result", 0)
    supervisor._seal_ops[op.job] = op
    supervisor._advance_seals()
    assert op.phase == "BUILDING" and op.ticket is not None
    old_nonce = op.nonce
    op.proof = QuiescenceProof(1, 0, old_nonce, helper.epoch, op.fence_ns, op.fence_ns)
    # The seal writer (the operation's own helper, not the main journal) is lost mid-build.
    helper.epoch += 1
    op.ticket.result = {"ok": False, "lost": True, "error": "seal helper lost"}
    supervisor._advance_seals()
    assert op.phase == "PENDING" and op.ticket is None
    monkeypatch.setattr(host, "time", SimpleNamespace(monotonic=lambda: op.retry_at,
                                                     monotonic_ns=host.time.monotonic_ns))
    supervisor._advance_seals()
    assert op.nonce != old_nonce and not op.result.exists()
    assert op.phase == "BUILDING"  # new proof/retry, never old permit replay


def test_journal_shutdown_cannot_overtake_an_accepted_submission():
    import queue
    import threading
    from stateframe.parallel_journal import Journal
    entered, release = threading.Event(), threading.Event()
    class Credits:
        def acquire(self, blocking):
            entered.set()
            assert release.wait(2)
            return True
    journal = Journal.__new__(Journal)
    journal._closed = False
    journal._state_lock = threading.Lock()
    journal._credits = Credits()
    journal._queue = queue.Queue(65)
    producer = threading.Thread(target=lambda: journal.submit("write", {}))
    producer.start()
    assert entered.wait(1)
    closer = threading.Thread(target=journal.close)
    closer.start()
    release.set()
    producer.join(2)
    closer.join(2)
    assert journal._queue.get_nowait() is not None
    assert journal._queue.get_nowait() is None


@pytest.mark.parametrize("damage", ["empty", "denied", "partial"])
def test_legacy_host_inspection_never_converts_failed_proc_to_quiescence(monkeypatch, damage):
    from stateframe.engine.kernel_exec_tracker import ProcessTable
    monkeypatch.setattr(host.os, "listdir", lambda path: [])
    if damage != "empty":
        monkeypatch.setattr(host.os, "listdir", lambda path: [str(host.os.getpid())])
        def broken(self, pid):
            if damage == "denied":
                raise PermissionError("denied")
            raise RuntimeError("partial")
        monkeypatch.setattr(ProcessTable, "identity", broken)
    monkeypatch.setattr(host.subprocess, "run", lambda *a, **kw: pytest.fail("failed scan fell back to ps"))
    with pytest.raises(host.ProcessInspectionDeferred):
        host._process_table()


def test_pending_launch_answer_waits_for_durable_intent_ack(tmp_path, monkeypatch):
    from stateframe.parallel_journal import JournalBusy
    from test_parallel_host import _request
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    journal = supervisor._journal = DelayedJournal()
    worker.state, worker.handle = "STARTING", None
    supervisor._persist_worker(worker, launch=True)
    published = []
    monkeypatch.setattr(supervisor, "_dispatch", lambda *a: ("PENDING", "launch-op", {"state": "STARTING"}))
    monkeypatch.setattr(supervisor, "_channel_record", lambda *a: None)
    monkeypatch.setattr(supervisor, "_store_answer", lambda *a: published.append(a[-1]))
    request = _request(1, "call", 1, "launch", {"unit_id": "a"})
    serve = lambda: supervisor._serve_request(request, worker.call_root, worker.call_root / "attempt-1/an", 1)
    # Before the durable ACK the sequence stays unanswered (re-served later); the channel
    # thread never waits for it.
    with pytest.raises(JournalBusy):
        serve()
    assert published == []
    ticket = journal.tickets[0]
    ticket.result = {"ok": True, "value": dict(ticket.payload["value"])}
    supervisor._pump_commits()
    serve()
    assert [answer.status for answer in published] == ["PENDING"]


def test_delayed_helpers_do_not_stall_status_or_tick(tmp_path, monkeypatch):
    import time
    from stateframe.parallel_seals import SealOperation
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    worker.state = "RUNNING"
    worker.status_view = supervisor._status_value(worker)
    supervisor._journal = DelayedJournal()
    worker.state, worker.group_verified = "EXITED", True
    supervisor._persist_worker(worker)
    helper = DelayedJournal()
    supervisor._seal_helpers = [helper]
    op = SealOperation(worker, SimpleNamespace(), tmp_path / "job", tmp_path / "result", 0)
    op.phase, op.helper_epoch = "BUILDING", 1
    op.ticket = helper.submit("seal_build", {"token": supervisor._seal_token(op)})
    supervisor._seal_ops[op.job] = op
    clock = [100.0]
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])
    request = SimpleNamespace(call_id="call", payload={"unit_id": "a"})
    latencies = []
    for _ in range(30):
        clock[0] += 1
        entered = time.perf_counter()
        supervisor.tick()
        supervisor._advance_seals()
        assert supervisor._status(request, worker.call_root)["state"] == "RUNNING"
        latencies.append(time.perf_counter() - entered)
    assert max(latencies) < 1.0
    assert op.phase == "BUILDING" and not op.result.exists()
    assert worker.exited_seq is None and worker.state == "EXITING"


def test_relaunch_cannot_overlap_seal_materialization(tmp_path, monkeypatch):
    from test_parallel_host import _spec_and_snapshot, _Worker, _request
    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    supervisor = ParallelSupervisor(run)
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), unit,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor._persist_worker(worker)
    monkeypatch.setattr(supervisor, "_start_runtime", lambda: None)
    supervisor._enqueue_seal(_request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call)
    op, = supervisor._seal_ops.values()
    op.phase = "BUILDING"
    payload = {"unit_id": "a", "execution_id": "execution", "spec_path": str(spec_path),
               "spec_digest": spec.digest(), "run_dir": str(unit), "incarnation": 2,
               "start_snapshot_path": str(snapshot), "start_manifest_digest": spec.start_manifest_digest,
               "snapshot_manifest_digest": worker.snapshot_manifest_digest}
    monkeypatch.setattr(supervisor, "_start_preparation", lambda *a: pytest.fail("reserved incarnation relaunched"))
    with pytest.raises(host.ParallelHostError, match="reserved for sealing"):
        supervisor._launch(_request(1, "call", 2, "launch", payload), call)
    assert ("call", "a", 2) not in supervisor._workers


def test_idle_channel_passes_coalesce_metrics_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(host.time, "monotonic", lambda: 1000.0)
    supervisor = ParallelSupervisor(tmp_path / "run")
    supervisor.parallel_root.mkdir(parents=True)
    writes = []
    monkeypatch.setattr(supervisor, "_write_metrics", lambda path, value, **kw: writes.append((path, value)))
    for _ in range(5):
        supervisor._service_channels()
    assert len(writes) == 1
    supervisor.max_tick_latency_seconds = 0.25
    for _ in range(5):
        supervisor._service_channels()
    assert len(writes) == 2
    assert writes[-1][1]["max_tick_latency_seconds"] == 0.25


def test_successful_commit_ack_clears_transient_pending_error(tmp_path):
    from test_parallel_supervisor_scale import _finalizing_worker, _CountingHandle
    supervisor, worker = _finalizing_worker(tmp_path, _CountingHandle())
    journal = supervisor._journal = DelayedJournal()
    worker.state, worker.group_verified, worker.error = "EXITED", True, None
    supervisor._persist_worker(worker)
    journal.tickets[0].result = {"ok": False, "error": "delayed disk"}
    supervisor._pump_commits()
    assert "commit pending" in worker.error and worker.state == "EXITING"
    retry = journal.tickets[-1]
    retry.result = {"ok": True, "value": dict(retry.payload["value"], exited_seq=1)}
    supervisor._pump_commits()
    assert worker.state == "EXITED" and worker.error is None
