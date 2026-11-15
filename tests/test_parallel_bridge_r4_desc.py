"""Engine-free family ownership checks across incomplete inspection and handle loss."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

import stateframe.parallel_host as host
import stateframe.parallel_worker as worker_module
from test_parallel_bridge_r3_life import _recorded
from test_parallel_bridge_r3_lnch import _launch_fixture


@pytest.mark.skipif(sys.platform != "linux", reason="Linux stat truncation regression")
@pytest.mark.parametrize("completion", ["probe", "resume", "cancel", "seal"])
def test_incomplete_detached_identity_never_authorizes_completion(tmp_path, monkeypatch, completion):
    identity_reader = host._process_identity
    supervisor, worker, process_file, _ = _recorded(tmp_path, monkeypatch)
    member = os.getpid()
    birth = worker_module.kernel_start_identity(member)
    record = json.loads(process_file.read_text())
    record["verified_members"] = {str(member): birth}
    process_file.write_text(json.dumps(record))
    # Completion proof alone cannot override an uncertain live-member read.
    process_file.with_name("family-exited.json").write_text(json.dumps({
        key: record[key] for key in ("pid", "pgid", "kernel_identity")}))
    # The complete scan has already witnessed this live, detached descendant. Only the
    # subsequent per-PID stat read is incomplete; it must never become evidence of death.
    real_read = os.read

    def truncated_stat(descriptor, count):
        if os.readlink(f"/proc/self/fd/{descriptor}") == f"/proc/{member}/stat":
            return b"1 (detached) R 1"
        return real_read(descriptor, count)

    monkeypatch.setattr(host, "_process_identity", identity_reader)
    observer = worker_module.ProcessTable().identity(member)
    monkeypatch.setattr(supervisor._process_table, "tick_snapshot", lambda: {member: observer})
    monkeypatch.setattr(supervisor._process_table, "snapshot", lambda: {member: observer})
    monkeypatch.setattr(host, "_live_group_members", lambda pgid: [])
    monkeypatch.setattr(os, "read", truncated_stat)
    if completion == "probe":
        with pytest.raises(host.ProcessInspectionDeferred):
            host._recorded_group_live(worker.run_dir, worker.pid, worker.pgid,
                                      record["kernel_identity"], process_file)
    elif completion == "resume":
        supervisor._finalize_recorded_worker(worker)
        assert worker.state == "RUNNING" and worker.exited_seq is None
    elif completion == "cancel":
        worker.cancel_requested = "stop"
        worker.fence_stage, worker.fence_deadline = "TERM", 0.0
        supervisor._advance_workers()
        assert worker.state == "RUNNING" and worker.exited_seq is None
        assert worker.fence_stage == "TERM", "inspection uncertainty must remain retryable"
        assert not json.loads(process_file.read_text()).get("fenced", False)
    else:
        with pytest.raises(host.ProcessInspectionDeferred):
            supervisor._group_live(worker)


@pytest.mark.parametrize("completion", ["probe", "resume", "cancel", "seal"])
def test_lost_handle_family_authority_never_means_empty_family(tmp_path, monkeypatch, completion):
    supervisor, worker, process_file, rows = _recorded(tmp_path, monkeypatch)
    record = json.loads(process_file.read_text())
    record["verified_members"] = {}
    process_file.write_text(json.dumps(record))
    # Detached PID 5555 was owned only by the former handle. It is present in the
    # complete scan, but neither its PGID nor the durable member map now identify it.
    from dataclasses import replace
    rows[5555] = replace(rows[os.getpid()], pid=5555, pgid=7777, zombie=False)
    if completion == "probe":
        with pytest.raises(host.ProcessInspectionDeferred, match="family"):
            host._recorded_group_live(worker.run_dir, worker.pid, worker.pgid,
                                      record["kernel_identity"], process_file)
    elif completion == "resume":
        supervisor._finalize_recorded_worker(worker)
        assert worker.state == "RUNNING" and worker.exited_seq is None
    elif completion == "cancel":
        worker.cancel_requested = "stop"
        worker.fence_stage, worker.fence_deadline = "TERM", 0.0
        supervisor._advance_workers()
        assert worker.state == "RUNNING" and worker.exited_seq is None
        assert worker.fence_stage == "TERM"
        assert not json.loads(process_file.read_text()).get("fenced", False)
    else:
        with pytest.raises(host.ProcessInspectionDeferred, match="family"):
            supervisor._group_live(worker)


def test_launch_record_retains_birth_bound_members_already_known_to_handle(tmp_path, monkeypatch):
    supervisor, worker, _ = _launch_fixture(tmp_path, monkeypatch)
    row = worker_module.ProcessTable().identity(os.getpid())
    monkeypatch.setattr(worker_module.WorkerHandle, "_family", SimpleNamespace(members={5555: row}),
                        raising=False)
    import threading
    monkeypatch.setattr(worker_module.WorkerHandle, "_family_lock", threading.RLock(), raising=False)
    monkeypatch.setattr(worker_module, "kernel_start_identity", lambda pid, **kwargs: "darwin:5555:1")
    supervisor._prepare_and_launch(worker)
    assert worker.error is None
    record = json.loads((worker.run_dir / "proc" / "1" / "process.json").read_text())
    assert record["verified_members"] == {"5555": "darwin:5555:1"}


@pytest.mark.parametrize("durability_failure", [False, True])
def test_reap_persists_family_exit_proof_before_handle_is_lost(tmp_path, monkeypatch, durability_failure):
    import threading
    from stateframe.engine.kernel_exec_tracker import LaunchIdentity
    row = worker_module.ProcessTable().identity(os.getpid())
    run = tmp_path / "run"
    process_dir = run / "proc" / "1"
    process_dir.mkdir(parents=True)
    birth = worker_module.kernel_start_identity(row.pid, row=row)
    (process_dir / "process.json").write_text(json.dumps({
        "pid": row.pid, "pgid": row.pid, "kernel_identity": birth, "verified_members": {}}))
    process = SimpleNamespace(pid=row.pid, returncode=0, wait=lambda **kwargs: 0)
    identity = LaunchIdentity(row.pid, row.start, None)
    handle = worker_module.WorkerHandle(process, row.pid, row.pid, 1, run, process_dir / "engine.log",
                                        0.0, 0.0, threading.Thread(), launch_identity=identity)
    monkeypatch.setattr(worker_module, "child_exited_unreaped", lambda proc: True)
    monkeypatch.setattr(handle, "_sweep_group", lambda: None)
    monkeypatch.setattr(handle.reader, "join", lambda **kwargs: None)
    if durability_failure:
        def fail_durability(*args):
            raise OSError("proof fsync failed")
        monkeypatch.setattr(worker_module, "_write_json", fail_durability)
        with pytest.raises(worker_module.ContainmentUnverified):
            handle.reap()
        assert not (process_dir / "family-exited.json").exists()
        return
    assert handle.reap() == 0
    # Repeated cleanup revalidates the immutable proof without overwriting it.
    proof = process_dir / "family-exited.json"
    before = proof.stat()
    assert handle.reap() == 0
    assert proof.stat().st_ino == before.st_ino
    assert proof.is_file(), "cleanup authority must survive loss of the handle"
    assert json.loads(proof.read_text()) == {"pid": row.pid, "pgid": row.pid, "kernel_identity": birth}


@pytest.mark.parametrize("invalid", ["birth", "pid", "pgid", "extra", "malformed", "symlink"])
def test_recovered_completion_requires_exact_safe_family_exit_proof(tmp_path, monkeypatch, invalid):
    supervisor, worker, process_file, _ = _recorded(tmp_path, monkeypatch)
    monkeypatch.setattr(host, "_process_identity", lambda pid: None)
    record = json.loads(process_file.read_text())
    proof = {key: record[key] for key in ("pid", "pgid", "kernel_identity")}
    proof_file = process_file.with_name("family-exited.json")
    if invalid in proof:
        proof[invalid] = 7777
    elif invalid == "birth":
        proof["kernel_identity"] = "darwin:4242:2"
    elif invalid == "extra":
        proof["unbound"] = True
    if invalid == "malformed":
        proof_file.write_text("{")
    elif invalid == "symlink":
        target = tmp_path / "foreign-proof.json"
        target.write_text(json.dumps(proof))
        proof_file.symlink_to(target)
    else:
        proof_file.write_text(json.dumps(proof))
    with pytest.raises(host.ProcessInspectionDeferred, match="family"):
        supervisor._group_live(worker)


def test_disappearance_between_scan_and_identity_recheck_is_retryable_unknown(monkeypatch):
    monkeypatch.setattr(host, "_live_group_members", lambda pgid: [5555])
    monkeypatch.setattr(host, "_process_identity", lambda pid: None)
    with pytest.raises(host.ProcessInspectionDeferred):
        host._verified_live_members(4242, {"5555": "darwin:5555:1"})


@pytest.mark.parametrize("observation", [None, (7777, "R", "darwin:5555:1", []),
                                           (4242, "Z", "darwin:5555:1", [])],
                         ids=["gone", "detached", "zombie"])
def test_disappearance_during_recorded_family_discovery_is_retryable_unknown(
        tmp_path, monkeypatch, observation):
    import signal
    _, worker, process_file, _ = _recorded(tmp_path, monkeypatch)
    monkeypatch.setattr(host, "_verified_leader", lambda *args: True)
    monkeypatch.setattr(host, "_live_group_members", lambda pgid: [5555])
    monkeypatch.setattr(host, "_process_identity", lambda pid: observation)
    signals = []
    monkeypatch.setattr(host, "_signal_one", lambda *args: signals.append(args))
    before = process_file.read_bytes()
    with pytest.raises(host.ProcessInspectionDeferred):
        host._signal_recorded_group(worker.run_dir, worker.pid, worker.pgid, "darwin:4242:1",
                                    signal.SIGTERM, process_file)
    assert signals == [] and process_file.read_bytes() == before


@pytest.mark.parametrize("unavailable", ["birth", "arguments"])
def test_unavailable_identity_components_never_mean_confirmed_disappearance(monkeypatch, unavailable):
    row = worker_module.ProcessTable().identity(os.getpid())
    monkeypatch.setattr(host, "ProcessTable", lambda: SimpleNamespace(identity=lambda pid: row))
    monkeypatch.setattr(host, "kernel_start_identity", lambda *args, **kwargs:
                        None if unavailable == "birth" else "darwin:5555:1")
    if unavailable == "arguments":
        monkeypatch.setattr(host.sys, "platform", "darwin")
        monkeypatch.setattr(host, "mac_process_argv", lambda pid: None)
    with pytest.raises(host.ProcessInspectionDeferred):
        host._process_identity(5555)
