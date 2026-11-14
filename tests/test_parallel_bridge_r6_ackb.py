"""ACK evidence reservations stay bounded without changing resume authority."""
import hashlib
import json
import os
from pathlib import Path
import threading

from stateframe.engine import kernel_parallel as parallel
from stateframe.engine.kernel_parallel_records import HostRequest, canonical_json
from stateframe.parallel_journal import JournalBusy
import stateframe.parallel_host as host
from stateframe.parallel_host import ParallelSupervisor


_ANSWER_SCHEMA = "stateframe-parallel-an/1"


def _client(tmp_path: Path):
    client = parallel.HostClient(tmp_path / "parallel" / "call", 1)
    client.runtime_id = host._host_runtime_id(os.getpid())
    return client


def _answer(client: parallel.HostClient) -> tuple[Path, dict[str, object]]:
    path = client.channel / "an" / "001.json"
    answer = {
        "attempt": 1,
        "call_id": "call",
        "seq": 1,
        "op": "status",
        "status": "OK",
        "result_id": None,
        "payload": {"state": "EXITED"},
        "schema": _ANSWER_SCHEMA,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(answer) + b"\n")
    return path, answer


def test_ack_evidence_backpressure_bounds_all_states_and_keeps_lifecycle_live(tmp_path):
    class _StalledWrites:
        def __init__(self):
            self.accepted = 0

        def write(self, path, value, *, replace_existing, wait, on_ack):
            self.accepted += 1
            raise JournalBusy("injected pending durability ACK")

    supervisor = ParallelSupervisor(tmp_path)
    writer = _StalledWrites()
    supervisor._channel_writes = writer
    supervisor._channel_thread = threading.current_thread()
    supervisor._answer_timer = lambda *args: None
    blocked = 0

    for index in range(2048):
        call_root = supervisor.parallel_root / "call"
        if index % 2:
            path = call_root / "host" / "seals" / f"seal-{index}-1.result.json"
            value = {"status": "OK", "payload": {"seq": index}}
        else:
            path = call_root / "attempt-1" / "an" / f"{index // 2 + 1:04d}.json"
            value = {"attempt": 1, "call_id": "call", "seq": index // 2 + 1,
                     "op": "status", "status": "OK", "result_id": None,
                     "payload": {"seq": index}, "schema": _ANSWER_SCHEMA}
        try:
            supervisor._write_json(path, value)
        except JournalBusy:
            blocked += 1

    assert writer.accepted == host._MAX_PENDING_ACK_EVIDENCE
    assert blocked == 2048  # admitted writes await ACK; later writes hit admission backpressure
    assert len(supervisor._ack_evidence_pending) == host._MAX_PENDING_ACK_EVIDENCE
    assert supervisor._ack_evidence_queue.qsize() == 0
    assert not supervisor._ack_evidence_scheduled

    # Simulate those admitted source ACKs arriving and moving through the queue into
    # the worker's in-flight state. The aggregate remains constant through each move.
    for path, (digest, _ready) in tuple(supervisor._ack_evidence_pending.items()):
        supervisor._queue_ack_evidence(path, digest)
    supervisor._channel_thread = None
    supervisor._ensure_ack_evidence_worker = lambda: True
    supervisor._publish_pending_ack_evidence()
    assert not supervisor._ack_evidence_pending
    assert supervisor._ack_evidence_queue.qsize() == host._MAX_PENDING_ACK_EVIDENCE

    for _ in range(32):
        assert supervisor._take_ack_evidence(supervisor._ack_evidence_queue) is not None
    assert len(supervisor._ack_evidence_scheduled) == 32
    with supervisor._ack_evidence_lock:
        assert supervisor._ack_evidence_total_locked() == host._MAX_PENDING_ACK_EVIDENCE

    lifecycle = []
    supervisor._pump_commits = lambda: lifecycle.append("commits")
    supervisor._advance_workers = lambda: lifecycle.append("workers")
    supervisor._advance_seals = lambda: lifecycle.append("seals")
    supervisor._service_lifecycle()
    assert lifecycle == ["commits", "workers", "seals"]


def test_ack_evidence_publications_do_not_keep_a_lifetime_set(tmp_path):
    supervisor = ParallelSupervisor(tmp_path)
    call_root = supervisor.parallel_root / "call"
    answer_dir = call_root / "attempt-1" / "an"
    answer_dir.mkdir(parents=True)

    for seq in range(1, 2049):
        path = answer_dir / f"{seq:04d}.json"
        digest = f"{seq:064x}"
        assert supervisor._publish_ack_evidence(path, digest)
        assert supervisor._ack_evidence_is_current(path, digest)

    assert not hasattr(supervisor, "_ack_evidence_published")
    with supervisor._ack_evidence_lock:
        assert supervisor._ack_evidence_total_locked() == 0


def test_resume_reacks_legacy_numbered_answer_without_sidecar(tmp_path):
    client = _client(tmp_path)
    answer_path, answer = _answer(client)
    record_path = answer_path.with_name("001.record.json")
    record = {"schema": host._CHANNEL_RECORD_SCHEMA, "request_sha256": "0" * 64,
              "invalid_request": False, "answer": answer}
    record_path.write_bytes(canonical_json(record) + b"\n")
    supervisor = ParallelSupervisor(tmp_path)
    supervisor._channel_thread = threading.current_thread()
    ack_path = answer_path.with_name("001.ack.json")

    assert not ack_path.exists()
    assert client.poll(str(answer_path)).status == "PENDING"
    assert supervisor._channel_answer_complete(answer_path.parent, 1)
    assert json.loads(ack_path.read_bytes())["sha256"] == hashlib.sha256(
        answer_path.read_bytes()).hexdigest()
    assert client.poll(str(answer_path)).status == "OK"


def test_resume_reacks_legacy_seal_result_without_sidecar(tmp_path):
    client = _client(tmp_path)
    seals = client.call_root / "host" / "seals"
    seals.mkdir(parents=True)
    payload = {"unit_id": "a", "incarnation": 1}
    request = HostRequest(1, "call", 1, "seal", "a", payload,
                          hashlib.sha256(canonical_json(payload)).hexdigest())
    job = seals / "a-1.json"
    job.write_bytes(canonical_json({"schema": "stateframe-parallel-seal-job/1",
                                    "request": request.to_json()}) + b"\n")
    result_path = seals / "a-1.result.json"
    result = {"status": "OK", "payload": {"proposal_digest": "sealed"}}
    result_path.write_bytes(canonical_json(result) + b"\n")
    supervisor = ParallelSupervisor(tmp_path)
    supervisor._channel_thread = threading.current_thread()
    ack_path = seals / "a-1.result.ack.json"

    assert not ack_path.exists()
    assert client.poll(str(result_path)).status == "PENDING"
    supervisor._discover_seals()
    assert supervisor._ack_evidence_is_current(
        result_path, hashlib.sha256(result_path.read_bytes()).hexdigest())
    assert json.loads(ack_path.read_bytes())["sha256"] == hashlib.sha256(
        result_path.read_bytes()).hexdigest()
    assert client.poll(str(result_path)).status == "OK"
