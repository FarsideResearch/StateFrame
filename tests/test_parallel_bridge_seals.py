"""Seal reservation lifecycle: resumed workers seal, dead reservations leave, stage files go."""
import json
import os
from types import SimpleNamespace
import pytest
import stateframe.parallel_host as host
import stateframe.parallel_worker as worker_mod
from stateframe.parallel_seals import SealOperation
from test_parallel_bridge import DelayedJournal


def _exited_worker(tmp_path, *, handle, pgid):
    from test_parallel_supervisor_scale import _finalizing_worker
    supervisor, worker = _finalizing_worker(tmp_path, handle)
    worker.state, worker.exited_seq, worker.pgid = "EXITED", 1, pgid
    if handle is None and pgid is not None:
        process_file = worker.run_dir / "proc" / "1" / "process.json"
        process_file.parent.mkdir(parents=True)
        process_file.write_text(json.dumps({"pid": worker.pid, "pgid": pgid,
                                           "kernel_identity": "darwin:4242:1", "verified_members": {}}))
        process_file.with_name("family-exited.json").write_text(json.dumps({
            "pid": worker.pid, "pgid": pgid, "kernel_identity": "darwin:4242:1"}))
    supervisor._workers[("call", "a", 1)] = worker
    return supervisor, worker


def test_resumed_handle_less_worker_seals_through_shared_scan(tmp_path, monkeypatch):
    # A worker hydrated from its record has no handle; its pgid probe must stamp the seal fence.
    supervisor, worker = _exited_worker(tmp_path, handle=None, pgid=4242)
    observer = worker_mod.ProcessTable().identity(os.getpid())
    monkeypatch.setattr(worker_mod.ProcessTable, "snapshot", lambda self: {os.getpid(): observer})
    op = SealOperation(worker, SimpleNamespace(), tmp_path / "job", tmp_path / "result", 0)
    proof = supervisor._fresh_seal_proof(op, 1)
    assert proof.scan_started_ns >= op.fence_ns and op.accepts(proof, 1)


def test_handle_less_probe_treats_failed_scan_as_unknown(tmp_path, monkeypatch):
    supervisor, worker = _exited_worker(tmp_path, handle=None, pgid=4242)
    def failing(self):
        raise OSError("proc denied")
    monkeypatch.setattr(worker_mod.ProcessTable, "snapshot", failing)
    op = SealOperation(worker, SimpleNamespace(), tmp_path / "job", tmp_path / "result", 0)
    with pytest.raises(host.ProcessInspectionDeferred):
        supervisor._fresh_seal_proof(op, 1)


def test_handle_less_probe_sees_live_group_member(tmp_path, monkeypatch):
    from stateframe.engine.kernel_exec_tracker import ProcessIdentity
    supervisor, worker = _exited_worker(tmp_path, handle=None, pgid=4242)
    rows = {4243: ProcessIdentity(4243, 1, 4242, 1, (0, 0), False)}
    monkeypatch.setattr(worker_mod.ProcessTable, "snapshot", lambda self: dict(rows))
    op = SealOperation(worker, SimpleNamespace(), tmp_path / "job", tmp_path / "result", 0)
    with pytest.raises(host.ProcessInspectionDeferred, match="still live"):
        supervisor._fresh_seal_proof(op, 1)


def test_cancel_generation_bump_on_exited_worker_does_not_wedge_seal(tmp_path, monkeypatch):
    from test_parallel_supervisor_scale import _CountingHandle
    supervisor, worker = _exited_worker(tmp_path, handle=_CountingHandle(), pgid=None)
    supervisor._seal_helpers = [DelayedJournal()]
    op = SealOperation(worker, SimpleNamespace(to_json=lambda: {}), tmp_path / "job", tmp_path / "result", 0)
    supervisor._seal_ops[op.job] = op
    worker.generation += 1  # a cancel on an already-exited worker: nothing to stop
    for _ in range(3):
        supervisor._advance_seals()
    assert op.phase == "BUILDING" and op.generation == worker.generation
    assert op.ticket is not None and op.ticket.payload["token"] == supervisor._seal_token(op)


def test_stale_reservation_is_evicted_with_error_result(tmp_path):
    from test_parallel_supervisor_scale import _CountingHandle
    supervisor, worker = _exited_worker(tmp_path, handle=_CountingHandle(), pgid=None)
    supervisor._seal_helpers = [DelayedJournal()]
    seals = tmp_path / "run" / "host" / "seals"
    seals.mkdir(parents=True)
    op = SealOperation(worker, SimpleNamespace(to_json=lambda: {}), seals / "a-1.json", seals / "a-1.result.json", 0)
    supervisor._seal_ops[op.job] = op
    supervisor._workers.latest.pop(("call", "a"))  # a newer incarnation owns the unit now
    supervisor._advance_seals()
    assert op.phase == "COMMITTING"
    supervisor._publish_seal_results()
    assert not supervisor._seal_ops and op.job in supervisor._sealed_jobs
    assert host._read_json(op.result, "result")["status"] == "ERROR"


def test_admission_cap_recovers_after_stale_reservations_leave(tmp_path):
    from test_parallel_supervisor_scale import _CountingHandle
    supervisor, worker = _exited_worker(tmp_path, handle=_CountingHandle(), pgid=None)
    supervisor._seal_helpers = [DelayedJournal(), DelayedJournal()]
    seals = tmp_path / "run" / "host" / "seals"
    seals.mkdir(parents=True)
    for index in range(32):
        op = SealOperation(worker, SimpleNamespace(to_json=lambda: {}), seals / f"u{index}-1.json",
                           seals / f"u{index}-1.result.json", 0)
        supervisor._seal_ops[op.job] = op
    supervisor._workers.latest.pop(("call", "a"))
    for _ in range(40):  # the lifecycle lane visits two operations per pass
        supervisor._advance_seals()
        supervisor._publish_seal_results()
    assert len(supervisor._seal_ops) < 32, "dead reservations must not hold the admission cap"
    assert not supervisor._seal_ops


def test_invalidated_nonce_stage_file_is_removed(tmp_path):
    from test_parallel_supervisor_scale import _CountingHandle
    supervisor, worker = _exited_worker(tmp_path, handle=_CountingHandle(), pgid=None)
    supervisor._seal_helpers = [DelayedJournal()]
    seals = tmp_path / "run" / "host" / "seals"
    seals.mkdir(parents=True)
    op = SealOperation(worker, SimpleNamespace(to_json=lambda: {}), seals / "a-1.json", seals / "a-1.result.json", 0)
    supervisor._seal_ops[op.job] = op
    old_stage = op.stage_for(op.nonce)
    old_stage.write_text("{}")
    op.invalidate()
    supervisor._advance_seals()
    assert not old_stage.exists(), "superseded stage file leaked"
    assert op.phase == "BUILDING" and not op.superseded


def test_published_result_removes_every_stage_file(tmp_path):
    from test_parallel_supervisor_scale import _CountingHandle
    supervisor, worker = _exited_worker(tmp_path, handle=_CountingHandle(), pgid=None)
    seals = tmp_path / "run" / "host" / "seals"
    seals.mkdir(parents=True)
    op = SealOperation(worker, SimpleNamespace(to_json=lambda: {}), seals / "a-1.json", seals / "a-1.result.json", 0)
    supervisor._seal_ops[op.job] = op
    (seals / "a-1.stage-dead.json").write_text("{}")
    op.stage_for(op.nonce).write_text("{}")
    (seals / "a-10.stage-other.json").write_text("{}")  # another unit's stage is not ours
    supervisor._complete_seal(op, {"status": "OK", "payload": {}})
    supervisor._publish_seal_results()
    assert op.phase == "DONE" and op.result.exists()
    assert sorted(path.name for path in seals.glob("a-1.stage-*.json")) == []
    assert (seals / "a-10.stage-other.json").exists()


@pytest.mark.parametrize("source", ["caf\u00e9\n", b"\xff\x00\n"])
def test_subframe_seal_hashes_current_raw_candidate_bytes(source):
    import hashlib
    from stateframe.engine.kernel_done_is_submit_authority import _parallel_file_digests

    contract = SimpleNamespace(owns=("pkg/a.py", "pkg/deleted.py"), creates=("pkg/new.py",))
    runtime = SimpleNamespace(_unit_policy=SimpleNamespace(spec=SimpleNamespace(contract=contract)))
    sources = {"pkg/a.py": source, "pkg/new.py": b"NEW\n", "pkg/deleted.py": None,
               "pkg/hidden.py": b"withheld"}
    expected = source.encode("utf-8") if isinstance(source, str) else source
    assert _parallel_file_digests(runtime, sources) == {
        "pkg/a.py": hashlib.sha256(expected).hexdigest(),
        "pkg/new.py": hashlib.sha256(b"NEW\n").hexdigest(),
    }
    assert _parallel_file_digests(SimpleNamespace(), sources) is None


@pytest.mark.parametrize("source", ["caf\u00e9\n", b"\xff\x00\r\n"])
def test_subframe_recovery_metadata_matches_current_byte_seal(tmp_path, source):
    from stateframe.engine.kernel_done_is_submit_authority import (
        parallel_final_source_fields, store_candidate,
    )

    sources = {"pkg/a.py": source, "pkg/deleted.py": None, "pkg/read.py": b"UNCHANGED\n"}
    stored = store_candidate(tmp_path / "candidates", "sealed", sources)
    fields = parallel_final_source_fields(sources, source_map_path=tmp_path / "source_map.json")
    assert fields == {"artifact_digest": stored["artifact_digest"],
                      "source_map_path": str(tmp_path / "source_map.json"),
                      "final_source_paths": ["pkg/a.py", "pkg/read.py"]}
