import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

from _timeouts import scaled

from stateframe import outcomes
from stateframe.engine.kernel_parallel import _snapshot
from stateframe.engine.kernel_parallel_records import (
    CheckReceipt, CheckRecipe, HostRequest, ParallelLimits, Proposal, UnitContract, UnitSpec, canonical_json,
)
from stateframe.engine.kernel_done_is_submit_authority import artifact_digest
from stateframe.engine.kernel_exec_tracker import ContainmentUnverified, child_exited_unreaped
from stateframe.parallel_host import (
    ParallelHostError, ParallelSupervisor, _Worker as _HostWorker, _group_has_live_processes, _process_table,
)
from stateframe.parallel_worker import WorkerHandle, launch_worker, prepare_worker
from stateframe.endpoint import Endpoint
from fixtures.process_signals import record_identity, signal_owned, signal_verified
from stateframe.runner import RunOptions, prepare


@pytest.fixture(autouse=True)
def _unlock_fixture_snapshots(tmp_path):
    yield
    for path in sorted(tmp_path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)


@pytest.fixture
def group_writable_umask():
    previous = os.umask(0o002)
    try:
        yield
    finally:
        os.umask(previous)


def _spec_and_snapshot(root: Path, *, include_test: bool = False, file_mode: int = 0o644):
    source = root / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "a.py").write_bytes(b"VALUE = 1\n")
    (source / "pkg" / "a.py").chmod(file_mode)
    if include_test:
        (source / "tests").mkdir()
        (source / "tests" / "test_a.py").write_text("def test_a(): assert True\n")
    call_root = root / "parallel" / "call"
    frozen = _snapshot(source, call_root, 3)
    snapshot = frozen.root
    digest = frozen.manifest_digest
    contract = UnitContract("a", "update value", ("pkg/a.py",), ("pkg/new.py",), (), (), (), (),
                            CheckRecipe((), (), "none"))
    spec = UnitSpec("call", "a", "execution", "seed", contract, (), (), ("pkg/a.py", "pkg/new.py"),
                    (), "", 3, 3, digest, "policy", ParallelLimits())
    run_dir = call_root / "units" / "a"
    run_dir.mkdir(parents=True)
    spec_path = run_dir / "unit-spec.json"
    spec_path.write_bytes(canonical_json(spec.to_json()) + b"\n")
    return call_root, run_dir, snapshot, spec_path, spec


def _copy_example_tree(source: Path, destination: Path) -> None:
    """Normalize fixture file permissions so shared-worktree ACLs cannot change receipts."""
    shutil.copytree(source, destination)
    for path in destination.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        mode = path.stat().st_mode
        path.chmod(0o755 if mode & 0o111 else 0o644)


def _snapshot_digest(snapshot: Path) -> str:
    manifest = json.loads((snapshot / "snapshot-manifest.json").read_bytes())
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


def _Worker(*args, **kwargs):
    snapshot = Path(args[8])
    if "snapshot_manifest_digest" not in kwargs and (snapshot / "snapshot-manifest.json").is_file():
        kwargs["snapshot_manifest_digest"] = _snapshot_digest(snapshot)
    return _HostWorker(*args, **kwargs)


def _request(attempt: int, call_id: str, seq: int, op: str, payload: dict) -> HostRequest:
    return HostRequest(attempt, call_id, seq, op, "test", payload,
                       hashlib.sha256(canonical_json(payload)).hexdigest())


def _write_request(path: Path, request: HostRequest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(request.to_json()) + b"\n")


def _store_submission_artifact(run_dir: Path, writes: dict[str, str]) -> str:
    from stateframe.engine.kernel_done_is_submit_authority import store_candidate

    stored = store_candidate(run_dir / "worker/results/react_done_submissions", "sealed", writes)
    digest = stored["artifact_digest"]
    assert stored["stored"] is True and isinstance(digest, str)
    assert digest == artifact_digest(writes)
    return digest


def _record_submission(run_dir: Path, writes: dict[str, str]) -> None:
    digest = _store_submission_artifact(run_dir, writes)
    (run_dir / "engine-result.json").write_text(json.dumps({"result": {
        "result": "SUBMITTED", "submission": {"stored": True, "seal_id": digest,
                                                "artifact_digest": digest}}}))


def _record_authenticated_submission(run_dir: Path, writes: dict[str, str], *,
                                     file_digests: dict[str, str] | None = None) -> None:
    from types import SimpleNamespace
    from stateframe.engine.kernel_done_is_submit_authority import (
        seal_submission, terminal_from_seal,
    )

    digest = _store_submission_artifact(run_dir, writes)
    runtime = SimpleNamespace()
    seal_submission(runtime, source="model", artifact_digest_value=digest, label="submitted",
                    labels={"certification": "not_evaluated"}, file_digests=file_digests)
    terminal = terminal_from_seal(runtime)
    assert terminal is not None
    (run_dir / "engine-result.json").write_text(json.dumps({"result": {
        "result": "SUBMITTED", "submission": terminal["done_seal"],
        "labels": terminal["labels"]}}))


def _worker_diagnostics(run_dir: Path) -> str:
    traceback_paths = [run_dir / "child-traceback.txt",
                       *sorted(run_dir.glob("attempts/*/child-traceback.txt"))]
    child_tracebacks = [f"{path}:\n{path.read_text(errors='replace')}"
                        for path in traceback_paths if path.is_file()]
    log_tail = []
    for log_path in sorted((run_dir / "proc").glob("*/engine.log")):
        lines = log_path.read_text(errors="replace").splitlines()
        log_tail.append(f"{log_path}:\n" + "\n".join(lines[-80:]))
    logs = "\n".join(log_tail) if log_tail else "<no engine logs>"
    traces = "\n".join(child_tracebacks) if child_tracebacks else "<not recorded>"
    return (f"child traceback files:\n{traces}\n"
            f"engine log tail:\n{logs}")


def _assert_subframe_f4_request(request: dict, task: str) -> None:
    messages = request.get("messages")
    assert isinstance(messages, list), "recorded request has no message list"
    contents = [message.get("content") for message in messages if isinstance(message, dict)]
    assert any(isinstance(content, str) and task in content for content in contents), (
        "recorded request omitted the subframe task text")
    frame_payloads = []
    prefix = "[StateFrame]\n"
    for content in contents:
        if not isinstance(content, str) or not content.startswith(prefix):
            continue
        payload_text = content[len(prefix):].split("\n[", 1)[0]
        payload = json.loads(payload_text)
        if isinstance(payload, dict) and payload.get("schema") == "stateframe-f4/1":
            frame_payloads.append(payload)
    assert frame_payloads, "recorded request omitted the shipped F4 StateFrame schema"
    assert any(isinstance(payload.get("state"), dict)
               and isinstance(payload["state"].get("framefit5"), dict)
               and payload["state"]["framefit5"].get("mode") == "FRAME_ONLY"
               for payload in frame_payloads), "recorded request omitted the F4 framefit5 state effect"


def test_channel_same_sequence_returns_stored_answer_and_digest_corruption_is_error(tmp_path):
    run = tmp_path / "run"
    parallel = run / "parallel" / "call" / "attempt-1"
    rq, an = parallel / "rq", parallel / "an"
    request = _request(1, "call", 1, "status", {"unit_id": "a"})
    _write_request(rq / "001.json", request)
    _write_request(rq / "1.json", request)

    supervisor = ParallelSupervisor(run)
    first = supervisor.serve(rq, an)
    stored = json.loads((an / "001.json").read_text())
    repeated = supervisor.serve(rq, an)

    assert len(first) == 1
    assert first[0].status == "ERROR"
    assert first[0].payload["error"] == "ParallelHostError: subframe 'a' has no launch record"
    assert repeated[0].to_json() == stored
    assert json.loads((an / "001.json").read_text()) == stored

    corrupted = request.to_json()
    corrupted["seq"] = 2
    corrupted["payload"] = {"unit_id": "different-payload"}
    (rq / "002.json").write_bytes(canonical_json(corrupted) + b"\n")
    answers = supervisor.serve(rq, an)
    error = next(answer for answer in answers if answer.seq == 2)
    assert error.status == "ERROR"
    assert error.op == "status"
    assert "payload digest" in error.payload["error"]
    assert json.loads((an / "002.json").read_text())["payload"]["error"] == error.payload["error"]
    repeated_error = next(answer for answer in supervisor.serve(rq, an) if answer.seq == 2)
    assert repeated_error.to_json() == json.loads((an / "002.json").read_text())


def test_launch_refuses_snapshot_manifest_digest_mismatch_before_preparation(tmp_path):
    run = tmp_path / "run"
    call, worker_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    payload = {"unit_id": "a", "execution_id": "execution", "spec_path": str(spec_path),
               "spec_digest": spec.digest(), "run_dir": str(worker_dir), "incarnation": 1,
               "start_snapshot_path": str(snapshot), "start_manifest_digest": spec.start_manifest_digest,
               "snapshot_manifest_digest": "0" * 64}
    supervisor = ParallelSupervisor(run)

    with pytest.raises(ParallelHostError, match="snapshot manifest digest mismatch"):
        supervisor._launch(_request(1, "call", 1, "launch", payload), call)
    assert not (call / "host" / "units" / "a").exists()


@pytest.mark.parametrize("probe_first,shutdown", [(True, False), (False, False), (False, True)],
                         ids=["probe-before-launch", "launch-before-probe", "shutdown-before-probe"])
@pytest.mark.parametrize("probe_fails", [False, True], ids=["zero-slots", "probe-error"])
def test_auto_zero_capacity_refuses_launch_durably(tmp_path, monkeypatch, probe_first, shutdown, probe_fails):
    import stateframe.parallel_host as host
    import stateframe.runner as runner

    run = tmp_path / "run"
    call, worker_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    (run / "spec.json").write_text(json.dumps({"parallel_mode": "auto"}))
    probes = []
    monkeypatch.setattr(host.threading.Thread, "start", lambda thread: probes.append(thread))

    def no_slots(_spec, _cap):
        if probe_fails:
            raise runner.RunError("slot probe unavailable")
        return 0

    monkeypatch.setattr(runner, "auto_parallel_concurrency", no_slots)
    supervisor = ParallelSupervisor(run)
    assert supervisor._capacity_value is None
    assert len(probes) == 1
    if probe_first:
        probes[0].run()
    payload = {"unit_id": "a", "execution_id": "execution", "spec_path": str(spec_path),
               "spec_digest": spec.digest(), "run_dir": str(worker_dir), "incarnation": 1,
               "start_snapshot_path": str(snapshot), "start_manifest_digest": spec.start_manifest_digest,
               "snapshot_manifest_digest": _snapshot_digest(snapshot)}
    launch = _request(1, "call", 1, "launch", payload)
    supervisor._launch(launch, call)
    worker = supervisor._workers[("call", "a", 1)]
    if not probe_first:
        assert worker.state == "STARTING"
        assert worker.queued and worker.terminal is None
        if shutdown:
            supervisor.cancel_all("stop before probe completes")
        probes[0].run()

    supervisor._advance_workers()
    assert supervisor._capacity_value == 0
    assert worker.state == "EXITED"
    assert worker.terminal == "RESOURCES_UNAVAILABLE"
    assert worker.error == "no idle subframe model slots are available"
    assert not worker.queued and not worker.preparing
    assert worker.handle is None and len(probes) == 1
    assert worker.finished_epoch is not None and worker.exited_seq == 1
    persisted = json.loads(worker.state_path.read_bytes())
    assert persisted["state"] == "EXITED"
    assert persisted["terminal"] == "RESOURCES_UNAVAILABLE"
    assert persisted["queued"] is False and persisted["exited_seq"] == 1

    # A repeated launch and a fresh supervisor must retain the durable refusal.
    restarted = ParallelSupervisor(run)
    assert restarted._launch(launch, call)[2] == {"state": "EXITED"}
    status_request = _request(1, "call", 2, "status", {"unit_id": "a"})
    status = restarted._status(status_request, call)
    assert status["state"] == "EXITED" and status["terminal"] == "RESOURCES_UNAVAILABLE"
    assert status["exited_seq"] == 1
    assert json.loads(worker.state_path.read_bytes()) == persisted


def test_concurrent_preparations_reserve_copy_space_before_allocating(tmp_path, monkeypatch):
    import stateframe.parallel_host as host
    import stateframe.parallel_resources as resources

    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text("{}")
    supervisor = ParallelSupervisor(run)
    workers = []
    for name in ("first", "second"):
        call, worker_dir, snapshot, spec_path, spec = _spec_and_snapshot(run / name)
        worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), worker_dir,
                         snapshot, spec.start_manifest_digest, {}, preparing=True)
        workers.append(worker)
    supervisor._workers = {(name, "a", 1): worker
                           for name, worker in zip(("first", "second"), workers)}
    repo_bytes = (workers[0].snapshot_path / "pkg/a.py").stat().st_size
    free_disk = (1 << 30) + 2 * repo_bytes
    monkeypatch.setattr(resources.shutil, "disk_usage",
                        lambda _path: type("Usage", (), {"free": free_disk})())
    monkeypatch.setattr(resources, "_memory_available", lambda: (50.0, "test-memory"))

    # Both snapshots are authenticated before either thread can reserve disk.
    barrier = threading.Barrier(2)
    read_snapshot = host.read_snapshot_manifest

    def synchronized_snapshot(*args):
        manifest = read_snapshot(*args)
        barrier.wait(timeout=30)
        return manifest

    monkeypatch.setattr(host, "read_snapshot_manifest", synchronized_snapshot)
    release_copies = threading.Event()
    decisions_ready = threading.Event()
    decision_lock = threading.Lock()
    prepared = []
    decisions = []

    def decided(value):
        with decision_lock:
            decisions.append(value)
            if len(decisions) == 2:
                decisions_ready.set()

    def copy_without_reflinks(_spec, _path, worker_dir, _snapshot, *, parent_spec):
        prepared.append(worker_dir)
        decided("copy")
        assert release_copies.wait(timeout=30)

    persist = supervisor._persist_worker

    def persist_decision(worker):
        persist(worker)
        if worker.queued and worker.error and "disk free" in worker.error:
            decided("queued")

    def stop_after_preparation(*args, **kwargs):
        raise host.WorkerLaunchError("test stops before process launch")

    monkeypatch.setattr(host, "prepare_worker", copy_without_reflinks)
    monkeypatch.setattr(host, "launch_worker", stop_after_preparation)
    monkeypatch.setattr(supervisor, "_persist_worker", persist_decision)
    threads = [threading.Thread(target=supervisor._prepare_and_launch, args=(worker,))
               for worker in workers]
    for thread in threads:
        thread.start()
    try:
        assert decisions_ready.wait(timeout=30), [worker.error for worker in workers]
        assert len(prepared) == 1, "concurrent copies overcommit the 1 GiB safety margin"
        assert sorted(decisions) == ["copy", "queued"]
    finally:
        release_copies.set()
        for thread in threads:
            thread.join(timeout=30)
        assert not any(thread.is_alive() for thread in threads)


def test_exited_sequence_follows_sealable_order_and_survives_supervisor_restart(tmp_path, monkeypatch):
    import stateframe.parallel_host as host

    run = tmp_path / "run"
    call, worker_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    supervisor = ParallelSupervisor(run)
    monkeypatch.setattr(host, "_group_has_live_processes", lambda _pgid, _table=None: False)

    class ExitedHandle:
        pgid = 123
        timed_out = False
        cancel_reason = None

        def reap(self):
            return 0

    workers = {}
    for unit_id in ("a", "b"):
        subframe_dir = worker_dir / unit_id
        subframe_dir.mkdir()
        worker = _Worker("call", unit_id, f"execution-{unit_id}", 1, call, spec_path,
                         spec.digest(), subframe_dir, snapshot, spec.start_manifest_digest, {},
                         created_epoch=100.0, pid=os.getpid())
        supervisor._persist_worker(worker, launch=True)
        workers[unit_id] = worker
    for unit_id, (slot, epoch) in {"a": (3, 103.0), "b": (5, 101.5)}.items():
        marker = workers[unit_id].run_dir / "proc" / "1" / "slot-lease.json"
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps({"schema": "stateframe-subframe-slot-lease/1",
                                      "slot": slot, "pid": os.getpid(),
                                      "acquired_epoch": epoch}))
    workers["b"].started_epoch = 101.5
    workers["b"].handle = ExitedHandle()
    supervisor._finalize_worker(workers["b"], 0)
    workers["a"].started_epoch = 103.0
    workers["a"].handle = ExitedHandle()
    supervisor._finalize_worker(workers["a"], 0)

    restarted = ParallelSupervisor(run)
    a = restarted._status(_request(1, "call", 2, "status", {"unit_id": "a"}), call)
    b = restarted._status(_request(1, "call", 3, "status", {"unit_id": "b"}), call)
    assert (b["exited_seq"], a["exited_seq"]) == (1, 2)
    assert b["running_since"] == 101.5
    assert (b["slot"], a["slot"]) == (5, 3)
    assert b["finished_epoch"] >= b["running_since"]
    assert b["slot_wait_seconds"] == 1.5
    assert a["slot_wait_seconds"] == 3.0
    assert json.loads((call / "host" / "exited-sequence.json").read_text()) == {"last": 2}

    worker_c = _Worker("call", "c", "execution-c", 1, call, spec_path, spec.digest(), worker_dir,
                       snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="BLOCKED")
    restarted._persist_worker(worker_c, launch=True)
    assert restarted._status(_request(1, "call", 4, "status", {"unit_id": "c"}), call)["exited_seq"] == 3


def test_running_time_starts_at_authenticated_slot_lease_marker(tmp_path):
    run = tmp_path / "run"
    call, worker_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    supervisor = ParallelSupervisor(run)
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), worker_dir,
                     snapshot, spec.start_manifest_digest, {}, created_epoch=100.0,
                     pid=os.getpid(), handle=object())
    supervisor._workers[("call", "a", 1)] = worker
    supervisor._persist_worker(worker, launch=True)
    supervisor._notice_running(worker)
    assert worker.started_epoch is None
    assert worker.state == "STARTING"

    marker = worker_dir / "proc" / "1" / "slot-lease.json"
    marker.parent.mkdir(parents=True)
    marker_value = {"schema": "stateframe-subframe-slot-lease/1", "slot": 3,
                    "pid": os.getpid() + 1, "acquired_epoch": 102.25}
    marker.write_text(json.dumps(marker_value))
    with pytest.raises(ParallelHostError, match="invalid identity"):
        supervisor._notice_running(worker)
    assert worker.started_epoch is None
    marker_value["pid"] = os.getpid()
    marker.write_text(json.dumps(marker_value))
    supervisor._notice_running(worker)
    worker.handle = None
    status = supervisor._status(_request(1, "call", 1, "status", {"unit_id": "a"}), call)
    assert status["state"] == "RUNNING"
    assert status["running_since"] == 102.25
    assert status["slot"] == 3
    assert status["slot_wait_seconds"] == 2.25
    assert status["exited_seq"] is None


def test_channel_conflicting_duplicate_sequence_never_dispatches(tmp_path):
    run = tmp_path / "run"
    rq = run / "parallel" / "call" / "attempt-1" / "rq"
    an = rq.parent / "an"
    _write_request(rq / "001.json", _request(1, "call", 1, "status", {"unit_id": "a"}))
    _write_request(rq / "1.json", _request(1, "call", 1, "status", {"unit_id": "b"}))

    answer, = ParallelSupervisor(run).serve(rq, an)

    assert answer.status == "ERROR"
    assert answer.payload == {"error": "duplicate sequence has different request identity"}
    assert json.loads((an / "001.json").read_text())["payload"] == answer.payload


def test_channel_later_conflicting_request_preserves_first_answer(tmp_path):
    run = tmp_path / "run"
    rq = run / "parallel" / "call" / "attempt-1" / "rq"
    an = rq.parent / "an"
    supervisor = ParallelSupervisor(run)
    _write_request(rq / "001.json", _request(1, "call", 1, "status", {"unit_id": "a"}))

    first, = supervisor.serve(rq, an)
    assert first.status == "ERROR"
    assert first.payload == {"error": "ParallelHostError: subframe 'a' has no launch record"}

    conflicting = _request(1, "call", 1, "status", {"unit_id": "b"})
    _write_request(rq / "001.json", conflicting)
    answer, = supervisor.serve(rq, an)

    stored = json.loads((an / "001.json").read_text())
    assert answer.status == "ERROR"
    assert answer.op == "status"
    assert answer.payload == {"error": "duplicate sequence has a different request identity"}
    assert stored == first.to_json()
    assert answer.to_json() != stored
    conflicts = list(an.glob("001.conflict-*.json"))
    assert len(conflicts) == 1
    assert json.loads(conflicts[0].read_text()) == answer.to_json()
    repeated, = supervisor.serve(rq, an)
    assert repeated.to_json() == answer.to_json()
    assert json.loads((an / "001.json").read_text()) == stored


def test_channel_fairly_reaches_request_65_and_ninth_attempt(tmp_path):
    run = tmp_path / "run"
    supervisor = ParallelSupervisor(run)
    call = run / "parallel" / "call"
    for seq in range(1, 66):
        _write_request(call / "attempt-1" / "rq" / f"{seq:03d}.json",
                       _request(1, "call", seq, "status", {"unit_id": "a"}))
    for attempt in range(2, 10):
        _write_request(call / f"attempt-{attempt}" / "rq" / "001.json",
                       _request(attempt, "call", 1, "status", {"unit_id": "a"}))

    # A tick serves at most 8 attempts within a 50 ms global work budget, so how many ticks it takes depends on
    # machine speed. Fairness is that the rotating cursor reaches every attempt within a few ticks and the
    # long attempt still finishes; never that one attempt starves the others.
    ninth = call / "attempt-9" / "an" / "001.json"
    last = call / "attempt-1" / "an" / "065.json"
    ninth_tick = None
    for tick in range(1, 201):
        supervisor._service_channels()
        if ninth_tick is None and ninth.is_file():
            ninth_tick = tick
        if ninth.is_file() and last.is_file():
            break
    assert ninth_tick is not None and ninth_tick <= 16, f"ninth attempt waited {ninth_tick} ticks"
    assert last.is_file(), "request 65 was never answered within 200 ticks"

    answer_65 = json.loads(last.read_text())
    answer_ninth = json.loads(ninth.read_text())
    assert answer_65["seq"] == 65 and answer_65["attempt"] == 1
    assert answer_ninth["seq"] == 1 and answer_ninth["attempt"] == 9
    assert answer_65["payload"]["error"] == "ParallelHostError: subframe 'a' has no launch record"
    metrics = json.loads((run / "parallel" / "host-metrics.json").read_text())
    assert metrics["schema"] == "stateframe-parallel-host-metrics/1"
    assert metrics["max_tick_latency_seconds"] >= 0
    assert set(metrics["phase_seconds"]) == {"commits", "workers", "seals", "seal_discovery", "channels"}
    assert all(value >= 0 for value in metrics["phase_seconds"].values())


def test_channel_recovers_answer_after_atomic_record_write(tmp_path, monkeypatch):
    import stateframe.parallel_host as host

    run = tmp_path / "run"
    rq = run / "parallel" / "call" / "attempt-1" / "rq"
    an = rq.parent / "an"
    _write_request(rq / "001.json", _request(1, "call", 1, "status", {"unit_id": "a"}))
    original = host._write_json

    def crash_after_record(path, value, **kwargs):
        if path == an / "001.json":
            raise OSError("injected answer publication crash")
        return original(path, value, **kwargs)

    monkeypatch.setattr(host, "_write_json", crash_after_record)
    with pytest.raises(OSError, match="injected"):
        host.ParallelSupervisor(run).serve(rq, an)
    monkeypatch.setattr(host, "_write_json", original)

    answer, = host.ParallelSupervisor(run).serve(rq, an)
    assert answer.payload == {"error": "ParallelHostError: subframe 'a' has no launch record"}
    assert json.loads((an / "001.json").read_text()) == answer.to_json()


def test_seal_captures_modify_and_create_and_rejects_stale_incarnation(tmp_path):
    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    workspace = run_dir / "workspace"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "pkg" / "a.py").write_bytes(b"VALUE = 1\n")
    _record_submission(run_dir, {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"})
    worker = _Worker("call", "a", "execution", 2, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 2)] = worker

    with pytest.raises(ParallelHostError, match="stale incarnation 1"):
        supervisor._seal(_request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call_root)

    result = supervisor._seal(_request(1, "call", 2, "seal", {"unit_id": "a", "incarnation": 2}), call_root)
    proposal_path = Path(result["proposal_path"])
    proposal = Proposal.from_json(json.loads(proposal_path.read_text()))

    assert result["proposal_digest"] == proposal.digest()
    assert [(entry.path, entry.kind) for entry in proposal.entries] == [
        ("pkg/a.py", "modify"), ("pkg/new.py", "create")]
    assert (proposal_path.parent / "blobs" / proposal.entries[0].after_sha256).read_bytes() == b"VALUE = 2\n"
    assert (proposal_path.parent / "blobs" / proposal.entries[1].after_sha256).read_bytes() == b"NEW = True\n"
    assert json.loads((proposal_path.parent / "digest.json").read_text()) == {
        "proposal_digest": proposal.digest()}


def test_seal_reuses_authenticated_owned_file_digests_after_one_materialization_read(
        tmp_path, monkeypatch):
    import stateframe.parallel_host as host

    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    workspace = run_dir / "workspace"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "pkg/a.py").write_bytes(b"VALUE = 1\n")
    writes = {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"}
    digests = {path: hashlib.sha256(source.encode()).hexdigest()
               for path, source in writes.items()}
    _record_authenticated_submission(run_dir, writes, file_digests=digests)
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker
    identities = []
    read_options = []
    original_identity = host._entry_identity
    original_read = host._regular_bytes_with_identity

    def record_identity(root, relative):
        identities.append((root, relative))
        return original_identity(root, relative)

    def record_read(path, *, maximum, label, hash_bytes=True):
        read_options.append((label, hash_bytes))
        return original_read(path, maximum=maximum, label=label, hash_bytes=hash_bytes)

    monkeypatch.setattr(host, "_entry_identity", record_identity)
    monkeypatch.setattr(host, "_regular_bytes_with_identity", record_read)
    result = supervisor._seal(
        _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call_root)
    proposal = Proposal.from_json(json.loads(Path(result["proposal_path"]).read_text()))

    assert [(entry.path, entry.after_sha256) for entry in proposal.entries] == [
        ("pkg/a.py", digests["pkg/a.py"]), ("pkg/new.py", digests["pkg/new.py"])]
    assert not any(root == workspace and path in digests for root, path in identities)
    assert read_options == [("materialized file pkg/a.py", False),
                            ("materialized file pkg/new.py", False)]


def test_seal_rejects_authenticated_file_digest_that_does_not_match_artifact_bytes(tmp_path):
    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    workspace = run_dir / "workspace"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "pkg/a.py").write_bytes(b"VALUE = 1\n")
    writes = {"pkg/a.py": "VALUE = 2\n"}
    wrong_digest = hashlib.sha256(b"wrong\n").hexdigest()
    _record_authenticated_submission(
        run_dir, writes, file_digests={"pkg/a.py": wrong_digest})
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker

    with pytest.raises(ParallelHostError, match="sealed candidate file digest differs from its bytes"):
        supervisor._seal(
            _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call_root)


def test_seal_rehashes_a_materialized_file_if_its_identity_changes(tmp_path, monkeypatch):
    import stateframe.parallel_host as host

    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    workspace = run_dir / "workspace"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "pkg/a.py").write_bytes(b"VALUE = 1\n")
    writes = {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"}
    digests = {path: hashlib.sha256(source.encode()).hexdigest()
               for path, source in writes.items()}
    _record_authenticated_submission(run_dir, writes, file_digests=digests)
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker
    original_materialize = supervisor._materialize_submission
    original_identity = host._entry_identity
    rehashed = []

    def tampering_materialize(worker_arg, spec_arg, workspace_arg, **kwargs):
        materialized = original_materialize(worker_arg, spec_arg, workspace_arg, **kwargs)
        (workspace_arg / "pkg/a.py").write_bytes(b"VALUE = 99\n")
        return materialized

    def record_identity(root, relative):
        rehashed.append(relative)
        return original_identity(root, relative)

    monkeypatch.setattr(supervisor, "_materialize_submission", tampering_materialize)
    monkeypatch.setattr(host, "_entry_identity", record_identity)
    with pytest.raises(ParallelHostError, match="changed after sealed candidate validation"):
        supervisor._seal(
            _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call_root)
    assert "pkg/a.py" in rehashed


def test_seal_recovers_after_proposal_record_before_digest_sidecar(tmp_path, monkeypatch):
    import stateframe.parallel_host as host

    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    workspace = run_dir / "workspace" / "pkg"
    workspace.mkdir(parents=True)
    (workspace / "a.py").write_bytes(b"VALUE = 1\n")
    _record_submission(run_dir, {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"})
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker
    original = host._write_json

    def crash_before_digest(path, value, **kwargs):
        if path.name == "digest.json":
            raise OSError("injected seal crash")
        return original(path, value, **kwargs)

    monkeypatch.setattr(host, "_write_json", crash_before_digest)
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    with pytest.raises(OSError, match="injected seal crash"):
        supervisor._seal(request, call_root)
    monkeypatch.setattr(host, "_write_json", original)

    result = supervisor._seal(request, call_root)
    proposal = Proposal.from_json(json.loads(Path(result["proposal_path"]).read_text()))
    assert result["proposal_digest"] == proposal.digest()
    assert json.loads((Path(result["proposal_path"]).parent / "digest.json").read_text()) == {
        "proposal_digest": proposal.digest()}


@pytest.mark.parametrize("boundary", ["blob", "proposal"])
def test_seal_retries_after_each_earlier_publication_boundary(tmp_path, monkeypatch, boundary):
    import stateframe.parallel_host as host

    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    workspace = run_dir / "workspace" / "pkg"
    workspace.mkdir(parents=True)
    (workspace / "a.py").write_bytes(b"VALUE = 1\n")
    _record_submission(run_dir, {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"})
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    original_blob, original_json = host._write_blob, host._write_json

    def crash_blob(*args):
        raise OSError("injected blob crash")

    def crash_proposal(path, value, **kwargs):
        if path.name == "proposal.json":
            raise OSError("injected proposal crash")
        return original_json(path, value, **kwargs)

    if boundary == "blob":
        monkeypatch.setattr(host, "_write_blob", crash_blob)
    else:
        monkeypatch.setattr(host, "_write_json", crash_proposal)
    with pytest.raises(OSError, match="injected"):
        supervisor._seal(request, call_root)
    monkeypatch.setattr(host, "_write_blob", original_blob)
    monkeypatch.setattr(host, "_write_json", original_json)

    result = supervisor._seal(request, call_root)
    proposal = Proposal.from_json(json.loads(Path(result["proposal_path"]).read_text()))
    assert [entry.after_sha256 for entry in proposal.entries] == [
        hashlib.sha256(b"VALUE = 2\n").hexdigest(), hashlib.sha256(b"NEW = True\n").hexdigest()]


def test_seal_preserves_snapshot_mode_under_group_writable_umask(tmp_path, group_writable_umask):
    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    workspace = run_dir / "workspace" / "pkg"
    workspace.mkdir(parents=True)
    (workspace / "a.py").write_bytes(b"VALUE = 1\n")
    assert (workspace / "a.py").stat().st_mode & 0o777 == 0o664
    _record_submission(run_dir, {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"})
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker

    result = supervisor._seal(_request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call_root)
    proposal = Proposal.from_json(json.loads(Path(result["proposal_path"]).read_text()))
    assert [(entry.path, entry.mode) for entry in proposal.entries] == [
        ("pkg/a.py", 0o644), ("pkg/new.py", 0o664)]


def test_seal_preserves_start_mode(tmp_path):
    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run, file_mode=0o444)
    digest = spec.start_manifest_digest
    workspace = run_dir / "workspace" / "pkg"
    workspace.mkdir(parents=True)
    (workspace / "a.py").write_bytes(b"VALUE = 1\n")
    (workspace / "a.py").chmod(0o444)
    _record_submission(run_dir, {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"})
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})

    result = supervisor._seal(request, call_root)
    proposal = Proposal.from_json(json.loads(Path(result["proposal_path"]).read_text()))
    assert proposal.entries[0].mode == 0o444


@pytest.mark.parametrize(("snapshot_mode", "workspace_mode"),
                         [(0o644, 0o755), (0o755, 0o644)], ids=["add-exec", "remove-exec"])
def test_seal_rejects_executable_bit_changes(tmp_path, snapshot_mode, workspace_mode):
    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run, file_mode=snapshot_mode)
    workspace = run_dir / "workspace" / "pkg"
    workspace.mkdir(parents=True)
    (workspace / "a.py").write_bytes(b"VALUE = 1\n")
    (workspace / "a.py").chmod(workspace_mode)
    _record_submission(run_dir, {"pkg/new.py": "NEW = True\n"})
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path,
                     spec.digest(), run_dir, snapshot,
                     spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker
    with pytest.raises(ParallelHostError, match="changed file mode"):
        supervisor._seal(_request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call_root)


def test_seal_records_original_executable_mode_from_snapshot_manifest(tmp_path):
    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run, file_mode=0o755)
    workspace = run_dir / "workspace" / "pkg"
    workspace.mkdir(parents=True)
    (workspace / "a.py").write_bytes(b"VALUE = 1\n")
    (workspace / "a.py").chmod(0o755)
    _record_submission(run_dir, {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"})
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker

    result = supervisor._seal(_request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call_root)
    proposal = Proposal.from_json(json.loads(Path(result["proposal_path"]).read_text()))
    assert [(entry.path, entry.mode) for entry in proposal.entries] == [
        ("pkg/a.py", 0o755), ("pkg/new.py", (workspace / "new.py").stat().st_mode & 0o777)]


def test_repeated_seal_refuses_snapshot_manifest_changed_after_proposal(tmp_path):
    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    workspace = run_dir / "workspace" / "pkg"
    workspace.mkdir(parents=True)
    (workspace / "a.py").write_bytes(b"VALUE = 1\n")
    _record_submission(run_dir, {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"})
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    first = supervisor._seal(request, call_root)
    assert Path(first["proposal_path"]).is_file()

    manifest_path = snapshot / "snapshot-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][0]["mode"] = 0o600
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")
    manifest_path.chmod(0o444)
    with pytest.raises(ParallelHostError, match="snapshot manifest digest mismatch"):
        supervisor._seal(request, call_root)


def test_seal_refuses_workspace_without_verified_submission(tmp_path):
    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    workspace = run_dir / "workspace" / "pkg"
    workspace.mkdir(parents=True)
    (workspace / "a.py").write_bytes(b"VALUE = 2\n")
    (workspace / "new.py").write_bytes(b"NEW = True\n")
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker

    with pytest.raises(ParallelHostError, match="verified submitted candidate seal"):
        supervisor._seal(_request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call_root)


@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("cache_noise", [False, True])
@pytest.mark.parametrize("invalid_cleanup", [False, True])
def test_seal_carries_only_candidate_bound_worker_check_receipt(
        tmp_path, stale, cache_noise, invalid_cleanup):
    from stateframe.engine.kernel_effects import scan
    from stateframe.engine.kernel_unit_policy import _check_ignored, _manifest_digest

    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, old_spec = _spec_and_snapshot(run, include_test=True)
    digest = old_spec.start_manifest_digest
    check = CheckRecipe(("tests/test_a.py",), ("tests/test_a.py",), "pytest")
    contract = UnitContract("a", "update value", ("pkg/a.py",), ("pkg/new.py",), (), (), (), (), check)
    spec = UnitSpec("call", "a", "execution", "seed", contract, (), (),
                    ("pkg/a.py", "pkg/new.py", "tests/test_a.py"), (), "", 3, 3,
                    digest, old_spec.policy_digest, ParallelLimits())
    spec_path.write_bytes(canonical_json(spec.to_json()) + b"\n")
    workspace = run_dir / "workspace"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "pkg" / "a.py").write_bytes(b"VALUE = 2\n")
    (workspace / "pkg" / "new.py").write_bytes(b"NEW = True\n")
    (workspace / "tests" / "test_a.py").write_text("def test_a(): assert True\n")
    _record_submission(run_dir, {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"})
    raw = b"pytest exit 0; 1 passed\n"
    evidence = run_dir / "worker" / "measurement-runs" / "check" / "unit-check-evidence" / (
        hashlib.sha256(raw).hexdigest() + ".txt")
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(raw)
    product_digest = "f" * 64 if stale else _manifest_digest(
        scan(workspace, exclude=_check_ignored))
    receipt = CheckReceipt(check.digest(), product_digest, "h" * 64, "e" * 64, spec.policy_digest,
                           "INVALID" if invalid_cleanup else "PASS",
                           "CHECK_CLEANUP_FAILED" if invalid_cleanup else None,
                           (), str(evidence), "1 passed")
    expected = None if stale or invalid_cleanup else receipt
    terminal = json.loads((run_dir / "engine-result.json").read_text())
    terminal["result"]["check_receipt"] = receipt.to_json()
    (run_dir / "engine-result.json").write_text(json.dumps(terminal))
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker

    if cache_noise:
        cache = workspace / ".test-tmp" / "artifact"
        cache.parent.mkdir()
        cache.write_text("ignored check artifact\n")
        assert supervisor._check_receipt(worker, spec, workspace) == expected
        cache.unlink()
        cache.parent.rmdir()

    sealed = supervisor._seal(_request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1}), call_root)
    proposal = Proposal.from_json(json.loads(Path(sealed["proposal_path"]).read_text()))
    assert proposal.check_receipt == expected


def test_required_check_recipe_refuses_missing_terminal_receipt(tmp_path):
    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, old_spec = _spec_and_snapshot(
        run, include_test=True)
    check = CheckRecipe(("tests/test_a.py",), ("tests/test_a.py",), "pytest")
    contract = UnitContract("a", "update value", ("pkg/a.py",), (), (), (), (), (), check)
    spec = UnitSpec("call", "a", "execution", "seed", contract, (), (),
                    ("pkg/a.py", "tests/test_a.py"), (), "", 3, 3,
                    old_spec.start_manifest_digest, old_spec.policy_digest, ParallelLimits())
    spec_path.write_bytes(canonical_json(spec.to_json()) + b"\n")
    _record_submission(run_dir, {"pkg/a.py": "VALUE = 2\n"})
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")

    with pytest.raises(ParallelHostError, match="required worker check receipt"):
        ParallelSupervisor._check_receipt(worker, spec, run_dir / "workspace")


def _checked_receipt_scenario(tmp_path, *, umask=0o022, checked_mode=0o644,
                              source_copy_mode=None):
    from dataclasses import replace

    from stateframe.engine.kernel_effects import scan
    from stateframe.engine.kernel_unit_policy import _check_ignored, _manifest_digest

    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, old_spec = _spec_and_snapshot(
        run, include_test=True, file_mode=0o664)
    check = CheckRecipe(("tests/test_a.py",), ("tests/test_a.py",), "pytest")
    contract = UnitContract("a", "update value", ("pkg/a.py",), (), (), (), (), (), check)
    spec = UnitSpec("call", "a", "execution", "seed", contract, (), (),
                    ("pkg/a.py", "tests/test_a.py"), (), "", 3, 3,
                    old_spec.start_manifest_digest, old_spec.policy_digest, ParallelLimits())
    spec_path.write_bytes(canonical_json(spec.to_json()) + b"\n")

    workspace = run_dir / "workspace"
    package = workspace / "pkg"
    tests = workspace / "tests"
    package.mkdir(parents=True)
    tests.mkdir()
    (package / "a.py").write_bytes(b"VALUE = 2\n")
    (package / "a.py").chmod(0o664)
    (tests / "test_a.py").write_text("def test_a(): assert True\n", encoding="utf-8")
    (tests / "test_a.py").chmod(0o664)

    source_map_root = run_dir / "worker" / "results" / "react_final" / "execution"
    checked_source = source_map_root / "pkg" / "a.py"
    checked_source.parent.mkdir(parents=True)
    prior_umask = os.umask(umask)
    try:
        checked_source.write_text("VALUE = 2\n", encoding="utf-8")
    finally:
        os.umask(prior_umask)
    if source_copy_mode is not None:
        checked_source.chmod(source_copy_mode)
    source_map_path = source_map_root / "source_map.json"
    source_map_path.write_text(json.dumps({"files": {"pkg/a.py": "VALUE = 2\n"}}),
                               encoding="utf-8")

    manifest = dict(scan(workspace, exclude=_check_ignored))
    manifest["pkg/a.py"] = replace(manifest["pkg/a.py"], mode=checked_mode)
    checked_modes = {path: entry.mode for path, entry in manifest.items()}
    evidence_bytes = b"pytest exit 0; 1 passed\n"
    evidence = run_dir / "worker" / "measurement-runs" / "check" / "unit-check-evidence" / (
        hashlib.sha256(evidence_bytes).hexdigest() + ".txt")
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(evidence_bytes)
    receipt = CheckReceipt(
        check.digest(), _manifest_digest(manifest), "h" * 64, "e" * 64,
        spec.policy_digest, "PASS", None, (), str(evidence), "1 passed",
        product_manifest_modes=checked_modes,
    )
    (run_dir / "engine-result.json").write_text(json.dumps({"result": {
        "result": "SUBMITTED", "check_receipt": receipt.to_json(),
        "source_map_path": str(source_map_path), "final_source_paths": ["pkg/a.py"],
    }}), encoding="utf-8")
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    return worker, spec, workspace, {"pkg/a.py": b"VALUE = 2\n"}, receipt, checked_source


@pytest.mark.parametrize(("umask", "source_copy_mode"), [(0o022, 0o644), (0o077, 0o600)],
                         ids=["umask-022", "umask-077"])
def test_checked_mode_verdict_is_independent_of_source_copy_umask(
        tmp_path, umask, source_copy_mode):
    worker, spec, workspace, candidate, receipt, checked_source = _checked_receipt_scenario(
        tmp_path, umask=umask, checked_mode=0o644)

    assert checked_source.stat().st_mode & 0o777 == source_copy_mode
    assert ParallelSupervisor._check_receipt(worker, spec, workspace, candidate) == receipt


def test_checked_mode_verdict_accepts_empty_candidate_when_full_manifest_matches(tmp_path):
    worker, spec, workspace, _candidate, receipt, _ = _checked_receipt_scenario(tmp_path)

    assert ParallelSupervisor._check_receipt(worker, spec, workspace, {}) == receipt


def test_check_receipt_rejects_executable_bit_and_file_set_or_type_changes(tmp_path):
    from stateframe.engine.kernel_effects import scan
    from stateframe.engine.kernel_unit_policy import _check_ignored, _manifest_digest

    worker, spec, workspace, candidate, receipt, _ = _checked_receipt_scenario(tmp_path)
    from dataclasses import replace

    source = workspace / "pkg" / "a.py"
    source.chmod(0o764)
    assert ParallelSupervisor._check_receipt(worker, spec, workspace, candidate) is None
    source.chmod(0o664)

    checked_manifest = dict(scan(workspace, exclude=_check_ignored))
    checked_manifest["pkg/a.py"] = replace(checked_manifest["pkg/a.py"], mode=0o755)
    modes = {path: entry.mode for path, entry in checked_manifest.items()}
    executable_receipt = replace(
        receipt, product_manifest_digest=_manifest_digest(checked_manifest),
        product_manifest_modes=modes)
    terminal = json.loads((worker.run_dir / "engine-result.json").read_text())
    terminal["result"]["check_receipt"] = executable_receipt.to_json()
    (worker.run_dir / "engine-result.json").write_text(json.dumps(terminal), encoding="utf-8")
    assert ParallelSupervisor._check_receipt(worker, spec, workspace, candidate) is None

    terminal["result"]["check_receipt"] = receipt.to_json()
    (worker.run_dir / "engine-result.json").write_text(json.dumps(terminal), encoding="utf-8")
    (workspace / "pkg" / "a.py").write_bytes(b"VALUE = 3\n")
    assert ParallelSupervisor._check_receipt(
        worker, spec, workspace, {"pkg/a.py": b"VALUE = 3\n"}) is None
    (workspace / "pkg" / "a.py").write_bytes(b"VALUE = 2\n")

    (workspace / "pkg" / "extra.py").write_text("EXTRA = True\n", encoding="utf-8")
    assert ParallelSupervisor._check_receipt(worker, spec, workspace, candidate) is None
    (workspace / "pkg" / "extra.py").unlink()

    checked_manifest = dict(scan(workspace, exclude=_check_ignored))
    checked_manifest["pkg/a.py"] = replace(checked_manifest["pkg/a.py"], mode=0o777)
    type_receipt = replace(
        receipt, product_manifest_digest=_manifest_digest(checked_manifest),
        product_manifest_modes={path: entry.mode for path, entry in checked_manifest.items()})
    terminal["result"]["check_receipt"] = type_receipt.to_json()
    (worker.run_dir / "engine-result.json").write_text(json.dumps(terminal), encoding="utf-8")
    source.unlink()
    link_target = tmp_path / "candidate-bytes.py"
    link_target.write_bytes(candidate["pkg/a.py"])
    source.symlink_to(link_target)
    assert ParallelSupervisor._check_receipt(worker, spec, workspace, candidate) is None


def test_channel_seal_returns_durable_async_result(tmp_path):
    run = tmp_path / "run"
    call_root, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    workspace = run_dir / "workspace" / "pkg"
    workspace.mkdir(parents=True)
    (workspace / "a.py").write_bytes(b"VALUE = 1\n")
    _record_submission(run_dir, {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"})
    worker = _Worker("call", "a", "execution", 1, call_root, spec_path, spec.digest(), run_dir,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker
    rq = call_root / "attempt-1" / "rq"
    an = rq.parent / "an"
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    _write_request(rq / "001.json", request)

    supervisor._persist_worker(worker)
    answer, = supervisor.serve(rq, an)
    assert answer.status == "PENDING"
    result_path = Path(answer.result_id)
    deadline = time.monotonic() + 5
    while not result_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert result_path.exists()
    result = json.loads(result_path.read_text())
    assert result["status"] == "OK"
    proposal = Proposal.from_json(json.loads(Path(result["payload"]["proposal_path"]).read_text()))
    assert result["payload"]["proposal_digest"] == proposal.digest()
    assert json.loads((an / "001.json").read_text()) == answer.to_json()


def test_seal_inspection_timeout_is_retried_on_next_tick(tmp_path, monkeypatch):
    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    (unit / "workspace" / "pkg").mkdir(parents=True)
    (unit / "workspace" / "pkg" / "a.py").write_text("VALUE = 1\n")
    _record_submission(unit, {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"})
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), unit,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED",
                     pgid=9876, exited_seq=1)
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker
    calls = []

    def inspect():
        calls.append(True)
        if len(calls) == 1:
            try:
                raise subprocess.TimeoutExpired("ps", 5)
            except subprocess.TimeoutExpired as error:
                raise ParallelHostError("cannot inspect old subframe processes: TimeoutExpired") from error
        return []

    def group_live(worker):
        inspect()
        supervisor._process_table._tick_state.fresh_started_ns = time.monotonic_ns()
        return False

    supervisor._persist_worker(worker)
    monkeypatch.setattr(supervisor, "_group_live", group_live)
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    _, result_id, _ = supervisor._enqueue_seal(request, call)
    result_path = Path(result_id)
    assert not result_path.exists(), "transient inspection failure permanently refused the seal"
    assert not (call / "proposals" / "a" / "proposal.json").exists()
    deadline = time.monotonic() + 15
    while not result_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert result_path.is_file()
    supervisor._stop_runtime()
    result = json.loads(result_path.read_text())
    assert result["status"] == "OK", result
    assert Path(result["payload"]["proposal_path"]).is_file()
    assert len(calls) >= 3


def test_slow_healthy_worker_survives_transient_supervision_probe(tmp_path, monkeypatch):
    import stateframe.parallel_host as host

    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    clock = [100.0]
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])

    class HealthyHandle:
        cancel_reason = None
        probes = 0

        def enforce_deadline(self):
            self.probes += 1
            if self.probes == 1:
                clock[0] += 64.0
                raise ContainmentUnverified("process inspection temporarily unavailable")

        def escalate(self):
            assert self.cancel_reason is None, "healthy worker was escalated"

        def poll(self):
            return None

        def cancel(self, reason):
            self.cancel_reason = reason

    handle = HealthyHandle()
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), unit,
                     snapshot, spec.start_manifest_digest, {}, state="RUNNING", handle=handle)
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker
    supervisor._advance_workers()
    assert handle.cancel_reason is None, "slow inspection killed a live worker"
    assert worker.state == "RUNNING" and worker.exited_seq is None
    supervisor._advance_workers()
    assert handle.probes == 2 and handle.cancel_reason is None


def test_tick_and_status_requests_share_one_process_table_view_and_status_advances_only_its_worker(
        tmp_path):
    run = tmp_path / "run"
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(run)
    supervisor = ParallelSupervisor(run)
    table = supervisor._process_table
    seen: list[tuple[str, bool]] = []

    class Handle:
        cancel_reason = None

        def __init__(self, name):
            self.name = name

        def enforce_deadline(self):
            pass

        def escalate(self):
            pass

        def poll(self):
            # Observation goes through the same entry point WorkerHandle.poll() uses.
            table.tick_snapshot()
            seen.append((self.name, True))
            return None

    names = [f"w{index}" for index in range(5)]
    workers = {}
    for index, name in enumerate(names):
        worker = _Worker("call", name, "execution", 1, call, spec_path, spec.digest(), unit,
                         snapshot, spec.start_manifest_digest, {}, state="RUNNING", handle=Handle(name))
        supervisor._workers[("call", name, 1)] = worker
        workers[name] = worker

    baseline = table.scans
    with table.tick():
        supervisor._advance_workers()  # the tick's own advance
        supervisor._advance_workers()  # a status request re-advancing inside the same tick
    assert table.scans - baseline == 1
    seen.clear()
    with table.tick():
        supervisor._advance_worker(workers["w3"])
    assert seen == [("w3", True)]


def test_tick_defers_pending_request_when_supervision_exceeds_global_budget(tmp_path, monkeypatch):
    import stateframe.parallel_host as host

    run = tmp_path / "run"
    supervisor = ParallelSupervisor(run)
    rq = run / "parallel" / "call" / "attempt-1" / "rq"
    _write_request(rq / "001.json", _request(1, "call", 1, "status", {"unit_id": "a"}))
    clock = [100.0]
    original_iterdir = Path.iterdir

    def slow_scan(path):
        if path == rq:
            clock[0] += 1.0
        return original_iterdir(path)

    def slow_workers():
        clock[0] += 6.48

    monkeypatch.setattr(supervisor, "_advance_workers", slow_workers)
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(Path, "iterdir", slow_scan)
    supervisor._tick_deadline = clock[0] + 0.05
    supervisor.serve(rq, rq.parent / "an")
    assert not (rq.parent / "an" / "001.json").exists()


def test_unanswered_host_poll_has_terminal_timeout_without_parent_deadline(monkeypatch):
    from stateframe.engine import kernel_parallel as parallel
    from stateframe.engine.kernel_parallel_records import HostAnswer

    clock = [0.0]
    monkeypatch.setenv("STATEFRAME_PARALLEL_HOST_RESPONSE_SECONDS", "5")
    monkeypatch.setattr(parallel.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(parallel.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(parallel.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    class SilentHost:
        def submit(self, op, scope, payload):
            return "unanswered"

        def poll(self, ref):
            assert clock[0] < 10, "parent polled forever without a host response deadline"
            return HostAnswer(1, "call", 0, "pending", "PENDING", ref, {})

    with pytest.raises(parallel.ParallelError, match="host response.*timed out"):
        parallel._host_answer(SilentHost(), "status", "a", {"unit_id": "a"}, None)


@pytest.mark.host_sandbox
def test_real_scripted_worker_submission_seals_recorded_bytes_and_receipt(tmp_path):
    _process_table()
    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    _copy_example_tree(root / "examples" / "discount_bug", repo)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    endpoint_process = subprocess.Popen([
        sys.executable, "-I", "-m", "stateframe.scripted_endpoint", "--script",
        str(root / "examples/discount_bug.script.json"), "--port", "0",
    ], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment)
    supervisor = None
    worker_dir = None
    try:
        assert endpoint_process.stdout is not None
        readable, _, _ = select.select([endpoint_process.stdout], [], [], 10)
        assert readable, "scripted endpoint did not publish its URL"
        url = endpoint_process.stdout.readline().strip()
        assert url.endswith("/v1"), f"scripted endpoint published an invalid URL: {url!r}"
        endpoint = Endpoint.from_args(url, "scripted", None)
        parent_options = RunOptions(
            repo, "Fix the pricing code so the tests pass.", endpoint, run_dir=tmp_path / "run",
            max_turns=16, profile="r2hc3", parallel_mode="off", python=sys.executable,
            test_python=sys.executable, extra_env={"HOME": str(tmp_path / "child-home")})
        parent_options._capabilities["python"] = parent_options.python
        parent_run, _, parent_spec = prepare(parent_options)
        call = parent_run / "parallel" / "call"
        frozen = _snapshot(parent_run / "parent", call, 0)
        snapshot, digest = frozen.root, frozen.manifest_digest
        contract = UnitContract("a", "fix the pricing calculation", ("shop/pricing.py",), (), (), (), (), (),
                                CheckRecipe(("tests/test_pricing.py",), ("tests/test_pricing.py",), "pytest"))
        spec = UnitSpec("call", "a", "execution-a", "0123456789abcdef0123456789abcdef", contract, (), (),
                        ("shop/pricing.py", "tests/test_pricing.py"), (), "", 0, 0, digest,
                        "policy-a", ParallelLimits())
        worker_dir = call / "units" / "a"
        worker_dir.mkdir(parents=True)
        spec_path = worker_dir / "unit-spec.json"
        spec_path.write_bytes(canonical_json(spec.to_json()) + b"\n")
        supervisor = ParallelSupervisor(parent_run)
        payload = {"unit_id": "a", "execution_id": "execution-a", "spec_path": str(spec_path),
                   "spec_digest": spec.digest(), "run_dir": str(worker_dir), "incarnation": 1,
                   "start_snapshot_path": str(snapshot), "start_manifest_digest": digest,
                   "snapshot_manifest_digest": frozen.snapshot_manifest_digest}
        try:
            supervisor._launch(_request(1, "call", 1, "launch", payload), call)
            deadline = time.monotonic() + 180
            state = None
            while time.monotonic() < deadline:
                supervisor.tick()
                state = supervisor._latest_worker("call", "a", call)
                if state is not None and state.terminal is not None:
                    break
                time.sleep(0.1)
            assert state is not None and state.terminal == "SUBMITTED", state.error if state else "missing"
            sealed = supervisor._seal(_request(1, "call", 2, "seal", {"unit_id": "a", "incarnation": 1}), call)
            proposal = Proposal.from_json(json.loads(Path(sealed["proposal_path"]).read_text()))
            assert (proposal.call_id, proposal.unit_id, proposal.execution_id, proposal.incarnation) == (
                "call", "a", "execution-a", 1)
            assert proposal.contract_digest == contract.digest()
            assert [entry.path for entry in proposal.entries] == ["shop/pricing.py"]
            assert proposal.check_receipt is not None
            assert proposal.check_receipt.status == "PASS"
            assert proposal.check_receipt.wrote_paths == ()
            assert proposal.check_receipt.product_manifest_modes
            terminal = outcomes.terminal_receipt(worker_dir)
            assert terminal["final_source_paths"] == ["shop/pricing.py"]
            assert Path(terminal["source_map_path"]).is_file()
            journal_terminal = outcomes.journal_submission(worker_dir)
            assert journal_terminal["check_receipt"] == proposal.check_receipt.to_json()
            assert journal_terminal["source_map_path"] == terminal["source_map_path"]
            assert journal_terminal["final_source_paths"] == ["shop/pricing.py"]
            assert (Path(sealed["proposal_path"]).parent / "blobs" /
                    proposal.entries[0].after_sha256).read_bytes() == (
                        worker_dir / "workspace" / "shop/pricing.py").read_bytes()
            assert proposal.entries[0].after_sha256 == hashlib.sha256(
                (worker_dir / "workspace" / "shop/pricing.py").read_bytes()).hexdigest()
            request_path = worker_dir / "wire" / "0001.request.json"
            request_bytes = request_path.read_bytes()
            request = json.loads(request_bytes)
            _assert_subframe_f4_request(request, "Subframe a: fix the pricing calculation")
            tool_names = [tool["function"]["name"] for tool in request["tools"]
                          if isinstance(tool, dict) and isinstance(tool.get("function"), dict)]
            assert "subframe_blocked" in tool_names
            assert "shell" not in tool_names
            with pytest.raises(AssertionError, match="no message list"):
                _assert_subframe_f4_request({"tools": []}, "Subframe a: fix the pricing calculation")
            prepared = json.loads((worker_dir / "spec.json").read_bytes())
            assert prepared["profile"] == "r2hc3"
            assert prepared["parallel_engine_python"] == parent_spec["capabilities"]["python"] == sys.executable
            assert supervisor._workers[("call", "a", 1)].handle.process.args[0] == sys.executable
        except Exception as error:
            pytest.fail(f"{error}\n{_worker_diagnostics(worker_dir)}", pytrace=False)
    finally:
        if supervisor is not None:
            supervisor.cancel_all("test cleanup")
            supervisor.reap()
        endpoint_process.terminate()
        endpoint_process.wait(timeout=5)


@pytest.mark.host_sandbox
def test_interrupted_worker_resume_replays_exact_wire_without_authorizing_replayed_sends(tmp_path):
    _process_table()
    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    _copy_example_tree(root / "examples" / "discount_bug", repo)
    script = json.loads((root / "examples" / "discount_bug.script.json").read_text())
    script["tokenize"] = {"mode": "slow", "delay": 3.0}
    script_path = tmp_path / "resume-script.json"
    script_path.write_text(json.dumps(script))
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    endpoint_process = subprocess.Popen(
        [sys.executable, "-I", "-m", "stateframe.scripted_endpoint", "--script", str(script_path), "--port", "0"],
        cwd=root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    worker_dir = None
    first_handle = None
    resumed_handle = None
    try:
        assert endpoint_process.stdout is not None
        readable, _, _ = select.select([endpoint_process.stdout], [], [], scaled(10))
        assert readable, "scripted endpoint did not publish its URL"
        endpoint_url = endpoint_process.stdout.readline().strip()
        assert endpoint_url.endswith("/v1"), f"scripted endpoint published an invalid URL: {endpoint_url!r}"
        endpoint = Endpoint.from_args(endpoint_url, "scripted", None)
        parent_options = RunOptions(
            repo, "Fix the pricing code so the tests pass.", endpoint, run_dir=tmp_path / "parent-run",
            max_turns=16, profile="r2hc3", parallel_mode="off", python=sys.executable,
            test_python=sys.executable, extra_env={"LLAMA_CPP_TOKENIZE_URL": endpoint_url + "/tokenize",
                                                   "HOME": str(tmp_path / "child-home")})
        parent_options._capabilities["python"] = parent_options.python
        parent_run, _, parent_spec = prepare(parent_options)
        assert parent_spec["profile"] == "r2hc3"
        assert parent_spec["environment"]["STATEFRAME_F4"] == "on"
        assert parent_spec["capabilities"]["python"] == sys.executable

        call_root = parent_run / "parallel" / "call"
        frozen = _snapshot(parent_run / "parent", call_root, 0)
        snapshot, manifest_digest = frozen.root, frozen.manifest_digest
        contract = UnitContract(
            "a", "fix the pricing calculation", ("shop/pricing.py",), (), (), (), (), (),
            CheckRecipe(("tests/test_pricing.py",), ("tests/test_pricing.py",), "pytest"))
        spec = UnitSpec(
            "call", "a", "execution-resume", "fedcba9876543210fedcba9876543210", contract, (), (),
            ("shop/pricing.py", "tests/test_pricing.py"), (), "", 0, 0, manifest_digest,
            "policy-resume", ParallelLimits())
        worker_dir = call_root / "units" / "a"
        worker_dir.mkdir(parents=True)
        spec_path = worker_dir / "unit-spec.json"
        spec_path.write_bytes(canonical_json(spec.to_json()) + b"\n")
        parent_spec["snapshot_manifest_digest"] = frozen.snapshot_manifest_digest
        prepare_worker(spec, spec_path, worker_dir, snapshot, parent_spec=parent_spec)
        prepared = json.loads((worker_dir / "spec.json").read_bytes())
        assert prepared["environment"]["LLAMA_CPP_TOKENIZE_URL"] == endpoint_url + "/tokenize"
        assert prepared["environment"]["STATEFRAME_F4"] == "on"
        assert prepared["parallel_engine_python"] == parent_spec["capabilities"]["python"]
        frozen_spec_digest = spec.digest()

        first_handle = launch_worker(worker_dir, 1)
        assert first_handle.process.args[0] == parent_spec["capabilities"]["python"]
        first_request_path = worker_dir / "wire" / "0001.request.json"
        first_response_path = worker_dir / "wire" / "0001.response.json"
        first_sends_path = worker_dir / "sends.jsonl"
        deadline = time.monotonic() + scaled(90)
        while time.monotonic() < deadline:
            records = ([json.loads(line) for line in first_sends_path.read_text().splitlines() if line.strip()]
                       if first_sends_path.is_file() else [])
            if first_request_path.is_file() and first_response_path.is_file() and records:
                break
            if first_handle.poll() is not None:
                break
            time.sleep(0.02)
        assert first_request_path.is_file() and first_response_path.is_file(), (
            "worker did not record and receive its first model request")
        initial_wire = first_request_path.read_bytes()
        initial_sends = [json.loads(line) for line in first_sends_path.read_text().splitlines() if line.strip()]
        assert len(initial_sends) == 1 and initial_sends[0].get("replayed") is not True
        assert initial_sends[0]["request_sha256"] == hashlib.sha256(initial_wire).hexdigest()
        _assert_subframe_f4_request(json.loads(initial_wire), "Subframe a: fix the pricing calculation")
        time.sleep(0.05)
        assert sorted(path.name for path in (worker_dir / "wire").glob("*.request.json")) == [
            "0001.request.json"]

        initial_spec = json.loads((worker_dir / "spec.json").read_bytes())
        initial_process = json.loads((worker_dir / "proc" / "1" / "process.json").read_bytes())
        from stateframe.parallel_worker import kernel_start_identity
        assert initial_process["kernel_identity"] == kernel_start_identity(first_handle.pid)
        assert initial_process["verified_members"] == {}
        first_handle.cancel("test interruption after recorded request 1")
        first_exit = first_handle.reap(timeout=scaled(5))
        if first_exit is None:
            first_handle.escalate(grace=0)
            first_exit = first_handle.reap(timeout=scaled(5))
        assert first_exit is not None and first_exit != 0, "interrupted worker did not stop before submission"

        resumed_handle = launch_worker(worker_dir, 2)
        resumed_identity = kernel_start_identity(resumed_handle.pid)
        assert resumed_identity is not None, "resumed worker has no live kernel identity"
        assert resumed_handle.process.args[0] == parent_spec["capabilities"]["python"]
        resumed_exit = resumed_handle.reap(timeout=scaled(180))
        assert resumed_exit == 0, f"resumed worker exited with {resumed_exit}"
        assert outcomes.terminal_receipt(worker_dir).get("result") == "SUBMITTED"
        resumed_spec = json.loads((worker_dir / "spec.json").read_bytes())
        resumed_unit_spec = UnitSpec.from_json(resumed_spec["unit_spec"])
        initial_unit_spec = UnitSpec.from_json(initial_spec["unit_spec"])
        assert resumed_unit_spec.execution_id == initial_unit_spec.execution_id == "execution-resume"
        assert resumed_unit_spec.digest() == initial_unit_spec.digest() == frozen_spec_digest
        assert resumed_unit_spec.run_seed == initial_unit_spec.run_seed == "fedcba9876543210fedcba9876543210"
        assert resumed_spec["run_identity_seed"] == initial_spec["run_identity_seed"] == resumed_unit_spec.run_seed
        second_process = json.loads((worker_dir / "proc" / "2" / "process.json").read_bytes())
        assert second_process["incarnation"] == initial_process["incarnation"] + 1 == 2
        assert second_process["kernel_identity"] == resumed_identity
        archived_wire = worker_dir / "attempts" / "1" / "wire" / "0001.request.json"
        replayed_wire = worker_dir / "wire" / "0001.request.json"
        assert archived_wire.read_bytes() == initial_wire == replayed_wire.read_bytes()

        prior_sends_path = worker_dir / "attempts" / "1" / "sends.jsonl"
        prior_sends = [json.loads(line) for line in prior_sends_path.read_text().splitlines() if line.strip()]
        resumed_sends = [json.loads(line) for line in (worker_dir / "sends.jsonl").read_text().splitlines()
                         if line.strip()]
        replayed_sends = [record for record in resumed_sends if record.get("replayed") is True]
        assert len(prior_sends) == 1 and len(replayed_sends) == 1
        assert replayed_sends[0]["request_sha256"] == prior_sends[0]["request_sha256"]
        assert replayed_sends[0]["request_sha256"] == hashlib.sha256(archived_wire.read_bytes()).hexdigest()
        physical_sends = [record for record in resumed_sends if record.get("replayed") is not True]
        assert physical_sends, "resumed run did not make a physical send to exercise ledger authorization"

        def physical_attempt_ids(process, records):
            return {
                f"{process['incarnation']}:{process['pid']}:{record['send']}:{attempt}:"
                f"{record['request_sha256'][:20]}"
                for record in records
                for attempt, _ in enumerate(record["transport_attempts"], 1)
            }

        expected_authorized = physical_attempt_ids(initial_process, prior_sends)
        expected_authorized.update(physical_attempt_ids(second_process, physical_sends))
        ledger_path = worker_dir / "sends-ledger.jsonl"
        ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
        authorized_rows = [record for record in ledger if record["state"] == "AUTHORIZED"]
        settled_ids = {record["send_id"] for record in ledger if record["state"] == "SETTLED"}
        authorized_ids = {record["send_id"] for record in authorized_rows}
        assert len(authorized_rows) == len(authorized_ids)
        assert authorized_ids == expected_authorized
        assert settled_ids == expected_authorized
        replayed_ids = {
            f"{second_process['incarnation']}:{second_process['pid']}:{record['send']}:1:"
            f"{record['request_sha256'][:20]}"
            for record in replayed_sends
        }
        assert authorized_ids.isdisjoint(replayed_ids)
    except Exception as error:
        pytest.fail(f"{error}\n{_worker_diagnostics(worker_dir) if worker_dir else '<worker not prepared>'}",
                    pytrace=False)
    finally:
        for handle in (resumed_handle, first_handle):
            if handle is not None and handle.poll() is None:
                handle.cancel("test cleanup")
                if handle.reap(timeout=2) is None:
                    handle.escalate(grace=0)
                    handle.reap(timeout=scaled(5))
        if endpoint_process.poll() is None:
            endpoint_process.terminate()
            endpoint_process.wait(timeout=5)


def test_fence_inspects_incarnation_newer_than_stale_descriptor(tmp_path, monkeypatch):
    from stateframe.parallel_host import fence_and_reap
    from stateframe.engine.kernel_parallel_records import ExecutionState, RecoveryDescriptor
    import stateframe.parallel_host as host

    run = tmp_path / "run"
    worker = run / "parallel" / "call" / "units" / "a"
    record = worker / "proc" / "2" / "process.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"incarnation": 2, "pid": 9991, "pgid": 9991,
                                  "started_at": "now", "started_epoch": 1.0,
                                  "kernel_identity": "darwin:1:1", "verified_members": {}}))
    signals = []
    monkeypatch.setattr(host, "_signal_recorded_group", lambda *args: signals.append(args))
    monkeypatch.setattr(host, "_recorded_group_live", lambda *args: False)
    descriptor = RecoveryDescriptor("call", "a" * 64, (), (),
                                    (ExecutionState("a", "execution", str(worker), "b" * 64, 1, None),))

    fence_and_reap(run, descriptor)
    assert signals == [(worker, 9991, 9991, "darwin:1:1", signal.SIGTERM, record)]
    assert json.loads(record.read_text())["fenced"] is True

    omitted = RecoveryDescriptor("call", "a" * 64, (), (), ())
    with pytest.raises(ParallelHostError, match="omitted recorded subframes"):
        fence_and_reap(run, omitted)
    assert signals == [(worker, 9991, 9991, "darwin:1:1", signal.SIGTERM, record)] * 2


def test_recovery_refuses_old_process_record_without_kernel_birth(tmp_path):
    from stateframe.parallel_host import fence_and_reap
    from stateframe.engine.kernel_parallel_records import ExecutionState, RecoveryDescriptor

    run = tmp_path / "run"
    worker = run / "parallel" / "call" / "units" / "a"
    record = worker / "proc" / "1" / "process.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"incarnation": 1, "pid": 9991, "pgid": 9991,
                                  "started_at": "now", "started_epoch": 1.0}))
    descriptor = RecoveryDescriptor("call", "a" * 64, (), (),
                                    (ExecutionState("a", "execution", str(worker), "b" * 64, 1, None),))
    with pytest.raises(ParallelHostError, match="lacks a precise kernel birth identity"):
        fence_and_reap(run, descriptor)


@pytest.mark.parametrize("artifact", ["launch-intent.json", "engine.log"])
def test_recovery_refuses_incarnation_without_complete_process_identity(tmp_path, artifact):
    from stateframe.parallel_host import fence_and_reap
    from stateframe.engine.kernel_parallel_records import ExecutionState, RecoveryDescriptor

    run = tmp_path / "run"
    worker = run / "parallel" / "call" / "units" / "a"
    incarnation = worker / "proc" / "1"
    incarnation.mkdir(parents=True)
    (incarnation / artifact).write_text("pending")
    descriptor = RecoveryDescriptor("call", "a" * 64, (), (),
                                    (ExecutionState("a", "execution", str(worker), "b" * 64, 1, None),))
    with pytest.raises(ParallelHostError, match="complete process identity") as error:
        fence_and_reap(run, descriptor)
    assert str(incarnation) in str(error.value)


def test_recorded_member_can_be_escalated_after_leader_exits(tmp_path, monkeypatch):
    import stateframe.parallel_host as host

    worker = tmp_path / "run with spaces"
    worker.mkdir()
    child = tmp_path / "_child.py"
    (worker / "spec.json").write_text(json.dumps({"engine_dir": str(tmp_path / "engine")}))
    record = worker / "proc" / "1" / "process.json"
    record.parent.mkdir(parents=True)
    birth = "darwin:100:123"
    member_birth = "darwin:101:456"
    record.write_text(json.dumps({"pid": 9871, "pgid": 9871, "kernel_identity": birth,
                                  "verified_members": {}}))
    leader = [(9871, "R", birth, [str(child), str(worker)])]
    identities = {9871: leader[0], 9872: (9871, "R", member_birth, ["member"])}
    monkeypatch.setattr(host, "_process_identity", lambda pid: identities.get(pid))
    monkeypatch.setattr(host, "_live_group_members", lambda pgid: [9871, 9872] if leader else [9872])
    delivered = []
    monkeypatch.setattr(host, "_signal_one", lambda pid, pgid, token, sig: delivered.append(
        (pid, token, sig)))

    host._signal_recorded_group(worker, 9871, 9871, birth, signal.SIGTERM, record)
    assert json.loads(record.read_text())["verified_members"] == {"9872": member_birth}
    assert delivered == [(9872, member_birth, signal.SIGTERM), (9871, birth, signal.SIGTERM)]
    leader.clear()
    identities.pop(9871)
    host._signal_recorded_group(worker, 9871, 9871, birth, signal.SIGKILL, record)
    assert delivered[-1] == (9872, member_birth, signal.SIGKILL)

    identities[9872] = (9871, "R", "darwin:999:1", ["unrelated"])
    with pytest.raises(ParallelHostError, match=r"orphaned members \[9872\]"):
        host._recorded_group_live(worker, 9871, 9871, birth, record)


def test_orphan_without_verified_members_requires_manual_recovery(tmp_path, monkeypatch):
    import stateframe.parallel_host as host

    worker = tmp_path / "worker"
    worker.mkdir()
    (worker / "spec.json").write_text(json.dumps({"engine_dir": str(tmp_path / "engine")}))
    record = worker / "process.json"
    record.write_text(json.dumps({"pid": 9871, "pgid": 9871,
                                  "kernel_identity": "darwin:100:123", "verified_members": {}}))
    monkeypatch.setattr(host, "_process_identity", lambda pid:
                        (9871, "R", "darwin:101:456", ["orphan"]) if pid == 9872 else None)
    monkeypatch.setattr(host, "_live_group_members", lambda pgid: [9872])
    with pytest.raises(ParallelHostError, match=r"orphaned members \[9872\].*kill -TERM 9872"):
        host._signal_recorded_group(worker, 9871, 9871, "darwin:100:123", signal.SIGKILL, record)


def test_new_live_descendant_after_recorded_member_exits_requires_manual_recovery(tmp_path, monkeypatch):
    import stateframe.parallel_host as host

    worker = tmp_path / "worker"
    worker.mkdir()
    (worker / "spec.json").write_text(json.dumps({"engine_dir": str(tmp_path / "engine")}))
    record = worker / "process.json"
    birth = "darwin:100:123"
    record.write_text(json.dumps({"pid": 9871, "pgid": 9871, "kernel_identity": birth,
                                  "verified_members": {"9872": "darwin:101:456"}}))
    identities = {9873: (9871, "R", "darwin:102:789", ["descendant"])}
    monkeypatch.setattr(host, "_process_identity", identities.get)
    monkeypatch.setattr(host, "_live_group_members", lambda _pgid: [9873])
    delivered = []
    monkeypatch.setattr(host, "_signal_one", lambda *args: delivered.append(args))
    with pytest.raises(ParallelHostError, match=r"orphaned members \[9873\]"):
        host._signal_recorded_group(worker, 9871, 9871, birth, signal.SIGKILL, record)
    assert delivered == []
    with pytest.raises(ParallelHostError, match=r"orphaned members \[9873\]"):
        host._recorded_group_live(worker, 9871, 9871, birth, record)

    monkeypatch.setattr(host, "_live_group_members", lambda _pgid: [])
    # An empty group is insufficient; the former handle must also have finished cleanup.
    record.with_name("family-exited.json").write_text(json.dumps({
        "pid": 9871, "pgid": 9871, "kernel_identity": birth}))
    assert host._recorded_group_live(worker, 9871, 9871, birth, record) is False
    host._signal_recorded_group(worker, 9871, 9871, birth, signal.SIGKILL, record)


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "darwin", reason="KERN_PROCARGS2 is macOS only")
def test_macos_process_argv_preserves_run_path_spaces(tmp_path):
    from stateframe.parallel_worker import mac_process_argv

    run_dir = tmp_path / "run path with spaces"
    run_dir.mkdir()
    script = run_dir / "sleeper.py"
    script.write_text("import time\ntime.sleep(30)\n")
    process = subprocess.Popen([sys.executable, str(script), str(run_dir)], start_new_session=True)
    try:
        argv = mac_process_argv(process.pid)
        assert argv is not None
        assert str(script) in argv
        assert str(run_dir) in argv
    finally:
        if process.poll() is None:
            signal_owned(process, signal.SIGTERM)
        process.wait(timeout=10)


def test_finalization_probe_failure_keeps_group_unsealable(tmp_path):
    call, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(tmp_path / "run")
    supervisor = ParallelSupervisor(tmp_path / "run")

    class ExitedProcess:
        pgid = 123
        timed_out = False
        cancel_reason = None

        def reap(self):
            raise ParallelHostError("process table denied")

    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(),
                     run_dir, snapshot, spec.start_manifest_digest, {},
                     state="RUNNING", handle=ExitedProcess())
    supervisor._finalize_worker(worker, 0)

    assert worker.state == "RUNNING"
    assert worker.terminal == "ENGINE_ERROR"
    assert worker.error == "subframe finalization failed: process table denied"
    assert worker.finalize_retry_at > time.monotonic()


def test_finalization_persist_failure_does_not_escape_cleanup(tmp_path, monkeypatch):
    import stateframe.parallel_host as host

    call, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(tmp_path / "run")
    supervisor = ParallelSupervisor(tmp_path / "run")

    class ExitedProcess:
        pgid = 123
        timed_out = False
        cancel_reason = None

        def reap(self):
            return 0

    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(),
                     run_dir, snapshot, spec.start_manifest_digest, {},
                     state="RUNNING", handle=ExitedProcess())
    monkeypatch.setattr(host, "_group_has_live_processes", lambda pgid, _table=None: False)
    monkeypatch.setattr(supervisor, "_persist_worker",
                        lambda current: (_ for _ in ()).throw(OSError("disk full")))

    supervisor._finalize_worker(worker, 0)

    assert worker.state == "RUNNING"
    assert worker.terminal == "ENGINE_ERROR"
    assert supervisor._protocol_errors == ["subframe finalization state could not be persisted: disk full"]


def test_launch_failure_sweeps_exited_leader_before_marking_exited(tmp_path):
    call, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(tmp_path / "run")
    supervisor = ParallelSupervisor(tmp_path / "run")
    sequence = []

    class ExitedHandle:
        def poll(self):
            sequence.append("poll")
            return 0

        def reap(self, timeout=0.0):
            sequence.append("sweep")
            assert timeout > 0
            acquired = threading.Event()

            def tick_lock():
                with supervisor._lock:
                    acquired.set()

            thread = threading.Thread(target=tick_lock, daemon=True)
            thread.start()
            assert acquired.wait(0.5), "subframe sweep held the supervisor tick lock"
            thread.join(timeout=1)
            return 0

    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(),
                     run_dir, snapshot, spec.start_manifest_digest, {},
                     state="STARTING", handle=ExitedHandle())
    original = supervisor._persist_worker

    def persist(current):
        sequence.append(f"persist:{current.state}")
        original(current)

    supervisor._persist_worker = persist
    supervisor._launch_failed(worker, OSError("launch state disk failure"))

    assert sequence == ["poll", "sweep", "persist:EXITED"]
    assert worker.state == "EXITED"
    assert worker.exit_code == 0
    assert worker.terminal == "ENGINE_ERROR"
    assert worker.exited_seq == 1


def test_launch_failure_keeps_unverified_containment_retryable(tmp_path):
    call, run_dir, snapshot, spec_path, spec = _spec_and_snapshot(tmp_path / "run")
    supervisor = ParallelSupervisor(tmp_path / "run")

    class UncontainedHandle:
        def poll(self):
            return 0

        def reap(self, timeout=0.0):
            raise ContainmentUnverified("descendant still live")

    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(),
                     run_dir, snapshot, spec.start_manifest_digest, {},
                     state="STARTING", handle=UncontainedHandle())
    supervisor._launch_failed(worker, OSError("launch state disk failure"))

    assert worker.state == "RUNNING"
    assert worker.exited_seq is None
    assert "descendant still live" in worker.error


@pytest.mark.host_sandbox
def test_parent_exit_cancels_and_reaps_a_real_worker_process_tree(tmp_path):
    _process_table()
    run = tmp_path / "run"
    call_root = run / "parallel" / "call"
    worker_dir = call_root / "units" / "a"
    worker_dir.mkdir(parents=True)
    child_pid_path = tmp_path / "grandchild.pid"
    script = tmp_path / "tree.py"
    script.write_text(
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "# Publish the PID atomically: the test polls for the file and must never read it half-written.\n"
        "with open(sys.argv[1] + '.tmp', 'w') as handle:\n"
        "    handle.write(str(child.pid))\n"
        "os.replace(sys.argv[1] + '.tmp', sys.argv[1])\n"
        "time.sleep(60)\n")
    process = subprocess.Popen([sys.executable, str(script), str(child_pid_path)], cwd=tmp_path,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               start_new_session=True)
    assert process.stdout is not None
    reader = threading.Thread(target=process.stdout.read, daemon=True)
    reader.start()
    handle = WorkerHandle(process, process.pid, process.pid, 1, worker_dir, tmp_path / "worker.log",
                          time.time(), time.monotonic(), reader)
    worker = _Worker("call", "a", "execution", 1, call_root, worker_dir / "unit-spec.json", "spec",
                     worker_dir, call_root / "snapshots" / "g0", "manifest", {},
                     state="RUNNING", pid=process.pid, pgid=process.pid, handle=handle)
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker

    try:
        deadline = time.monotonic() + 5
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.is_file(), "worker did not start its child"
        grandchild_pid = int(child_pid_path.read_text())

        supervisor.cancel_all("parent engine child exited")
        supervisor.reap()

        live = [pid for pid, pgid, state, _ in _process_table()
                if pgid == process.pid and not state.startswith("Z")]
        assert process.poll() is not None
        assert not _group_has_live_processes(process.pid)
        assert process.pid not in live
        assert grandchild_pid not in live
    finally:
        if process.poll() is None:
            try:
                signal_owned(process, 9)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        reader.join(timeout=2)


@pytest.mark.host_sandbox
def test_exited_child_keeps_subframe_unsealable_until_process_group_is_gone(tmp_path):
    _process_table()
    run = tmp_path / "run"
    call = run / "parallel" / "call"
    worker_dir = call / "units" / "a"
    worker_dir.mkdir(parents=True)
    script = tmp_path / "exit_with_descendant.py"
    script.write_text("import subprocess, sys\n"
                      "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n")
    launched_after = time.time()
    process = subprocess.Popen([sys.executable, str(script)], cwd=tmp_path,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               start_new_session=True)
    assert process.stdout is not None
    reader = threading.Thread(target=process.stdout.read, daemon=True)
    reader.start()
    handle = WorkerHandle(process, process.pid, process.pid, 1, worker_dir, tmp_path / "worker.log",
                          time.time(), time.monotonic(), reader)
    worker = _Worker("call", "a", "execution", 1, call, worker_dir / "unit-spec.json", "spec",
                     worker_dir, call / "snapshots" / "g0", "manifest", {}, state="RUNNING",
                     pid=process.pid, pgid=process.pid, handle=handle)
    supervisor = ParallelSupervisor(run)
    supervisor._workers[("call", "a", 1)] = worker
    try:
        exited_by = time.monotonic() + 5
        while not child_exited_unreaped(process) and time.monotonic() < exited_by:
            time.sleep(0.01)
        assert child_exited_unreaped(process)
        assert process.returncode is None
        started = time.monotonic()
        supervisor.tick()
        assert time.monotonic() - started < 0.5
        # The tick must not block, and it may finish cleanup at once (macOS now signals detached
        # descendants directly). What must never happen is EXITED while the group is still live.
        assert worker.state in {"RUNNING", "EXITED"}
        if worker.state == "EXITED":
            assert not _group_has_live_processes(process.pid)
        deadline = time.monotonic() + 15
        while worker.state != "EXITED" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert worker.state == "EXITED"
        assert not _group_has_live_processes(process.pid)
    finally:
        for pid, pgid, state, _ in _process_table():
            if pgid != process.pid or state.startswith("Z"):
                continue
            fragment = str(script) if pid == process.pid else "time.sleep(30)"
            try:
                identity = record_identity(pid, launched_after=launched_after,
                                           command_fragment=fragment)
                signal_verified(pid, identity, signal.SIGKILL)
            except ProcessLookupError:
                pass
        reader.join(timeout=2)


class _SupervisedHandle:
    """A live worker handle whose supervision calls are counted."""

    pgid = 4242
    timed_out = False
    cancel_reason = None

    def __init__(self):
        self.deadline_checks = 0
        self.polls = 0

    def enforce_deadline(self):
        self.deadline_checks += 1

    def escalate(self):
        pass

    def poll(self):
        self.polls += 1
        return None


def _starting_worker(tmp_path):
    run = tmp_path / "run"
    call, worker_dir, snapshot, spec_path, spec = _spec_and_snapshot(run)
    supervisor = ParallelSupervisor(run)
    handle = _SupervisedHandle()
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), worker_dir,
                     snapshot, spec.start_manifest_digest, {}, created_epoch=100.0,
                     pid=os.getpid(), handle=handle)
    supervisor._workers[("call", "a", 1)] = worker
    supervisor._persist_worker(worker, launch=True)
    marker = worker_dir / "proc" / "1" / "slot-lease.json"
    marker.parent.mkdir(parents=True)
    return supervisor, worker, handle, marker


@pytest.mark.parametrize("content", [
    b"", b'{"schema": "stateframe-subframe-slot-lease/1", "slot": 3, "pi', b"[]", b"\xff\xfe",
    json.dumps({"schema": "stateframe-subframe-slot-lease/1", "slot": 3, "pid": os.getpid() + 1,
                "acquired_epoch": 102.0}).encode(),
    json.dumps({"schema": "other/9", "slot": 3, "pid": os.getpid(), "acquired_epoch": 102.0}).encode(),
], ids=["empty", "torn", "not-object", "not-utf8", "foreign-pid", "foreign-schema"])
def test_tick_survives_torn_or_foreign_slot_lease_marker(tmp_path, content):
    supervisor, worker, handle, marker = _starting_worker(tmp_path)
    marker.write_bytes(content)
    supervisor._advance_workers()
    supervisor._advance_workers()
    assert worker.state == "STARTING"
    assert worker.started_epoch is None
    assert worker.error is not None and "slot lease marker" in worker.error
    # Supervision continued past the bad marker on every tick.
    assert handle.deadline_checks == 2
    assert handle.polls == 2
    assert len([line for line in supervisor._protocol_errors if "slot lease marker" in line]) == 1

    marker.write_bytes(json.dumps({"schema": "stateframe-subframe-slot-lease/1", "slot": 3,
                                   "pid": os.getpid(), "acquired_epoch": 102.25}).encode())
    supervisor._advance_workers()
    assert worker.state == "RUNNING"
    assert worker.started_epoch == 102.25


def test_tick_survives_a_concurrent_marker_writer(tmp_path):
    supervisor, worker, handle, marker = _starting_worker(tmp_path)
    valid = json.dumps({"schema": "stateframe-subframe-slot-lease/1", "slot": 3,
                        "pid": os.getpid(), "acquired_epoch": 102.25}).encode()
    stop = threading.Event()

    def tear():
        # A non-atomic writer: create, write a prefix, finish, truncate, repeat.
        while not stop.is_set():
            with open(marker, "wb") as output:
                output.write(valid[:17])
                output.flush()
                output.write(valid[17:40])
                output.flush()
            marker.unlink()

    writer = threading.Thread(target=tear, daemon=True)
    writer.start()
    try:
        for _ in range(300):
            supervisor._advance_workers()
            assert worker.state == "STARTING"
    finally:
        stop.set()
        writer.join(5)
    assert not writer.is_alive()
    assert handle.deadline_checks == 300
    marker.write_bytes(valid)
    supervisor._advance_workers()
    assert worker.state == "RUNNING"
    assert worker.started_epoch == 102.25


def test_submission_fixture_matches_4a475493_artifact(tmp_path):
    fixture = Path(__file__).parent / "fixtures/parallel/host_submission_4a475493.json"
    expected = json.loads(fixture.read_bytes())
    _record_submission(tmp_path, expected["writes"])
    submission = json.loads((tmp_path / "engine-result.json").read_bytes())["result"]["submission"]
    assert submission["artifact_digest"] == expected["artifact_digest"]
    artifact = tmp_path / "worker/results/react_done_submissions/sealed" / (
        expected["artifact_digest"].removeprefix("sha256:") + ".json")
    assert artifact.read_bytes() == fixture.read_bytes()
