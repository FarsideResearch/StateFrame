"""Visible parallel results are not ready until their runtime ACK is visible."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import pytest

from stateframe.engine import kernel_parallel as parallel
from stateframe.engine.kernel_parallel_records import HostRequest, canonical_json
from stateframe.parallel_journal import JournalBusy
import stateframe.parallel_host as host
from stateframe.parallel_host import ParallelSupervisor
from _timeouts import scaled


_ANSWER_SCHEMA = "stateframe-parallel-an/1"
_ACK_SCHEMA = "stateframe-parallel-ack/1"


def _client(tmp_path: Path):
    return parallel.HostClient(tmp_path / "parallel" / "call", 1)


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
    path.write_bytes(canonical_json(answer) + b"\n")
    return path, answer


def _ack(path: Path, answer: dict[str, object], client: parallel.HostClient,
         **overrides: object) -> None:
    raw = canonical_json(answer) + b"\n"
    evidence: dict[str, object] = {
        "schema": _ACK_SCHEMA,
        "attempt": 1,
        "call_id": "call",
        "runtime": getattr(client, "runtime_id", "runtime-under-test"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    evidence.update(overrides)
    ack_path = path.with_name(f"{path.stem}.ack.json")
    ack_path.write_bytes(canonical_json(evidence) + b"\n")


def test_visible_numbered_answer_without_ack_is_pending(tmp_path):
    client = _client(tmp_path)
    path, _ = _answer(client)

    got = client.poll(str(path))

    assert got.status == "PENDING"
    assert got.op == "pending"
    assert got.result_id == str(path)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("attempt", 2),
        ("call_id", "another-call"),
        ("runtime", "stale-runtime"),
        ("sha256", "0" * 64),
    ],
)
def test_numbered_answer_requires_matching_ack_evidence(tmp_path, field, bad_value):
    client = _client(tmp_path)
    path, answer = _answer(client)
    _ack(path, answer, client, **{field: bad_value})

    got = client.poll(str(path))

    assert got.status == "PENDING"
    assert got.op == "pending"
    assert got.result_id == str(path)


def test_matching_numbered_answer_ack_makes_answer_ready(tmp_path):
    client = _client(tmp_path)
    path, answer = _answer(client)
    _ack(path, answer, client)

    got = client.poll(str(path))

    assert got.status == "OK"
    assert got.seq == 1
    assert got.payload == {"state": "EXITED"}


def test_scheduled_ack_keeps_visible_answer_out_of_replay_queue(tmp_path):
    client = _client(tmp_path)
    supervisor = ParallelSupervisor(tmp_path)
    answer_path, answer = _answer(client)
    record_path = answer_path.with_name("001.record.json")
    record = {"schema": host._CHANNEL_RECORD_SCHEMA, "request_sha256": "0" * 64,
              "invalid_request": False, "answer": answer}
    record_path.write_bytes(canonical_json(record) + b"\n")
    supervisor._tick_deadline = time.monotonic() + scaled(2)

    assert not supervisor._channel_answer_complete(answer_path.parent, 1)
    assert supervisor._ack_evidence_is_scheduled(answer_path)
    assert supervisor._pending_channel_sequences(answer_path.parent, [1]) == []
    assert (answer_path.parent, 1) in supervisor._channel_completed


def test_visible_seal_result_without_ack_is_pending(tmp_path):
    client = _client(tmp_path)
    path = client.call_root / "host" / "seals" / "a-1.result.json"
    path.parent.mkdir(parents=True)
    result = {"status": "OK", "payload": {"proposal_digest": "sealed"}}
    path.write_bytes(canonical_json(result) + b"\n")

    got = client.poll(str(path))

    assert got.status == "PENDING"
    assert got.op == "pending"
    assert got.result_id == str(path)


def test_matching_seal_result_ack_makes_result_ready(tmp_path):
    client = _client(tmp_path)
    path = client.call_root / "host" / "seals" / "a-1.result.json"
    path.parent.mkdir(parents=True)
    result = {"status": "OK", "payload": {"proposal_digest": "sealed"}}
    raw = canonical_json(result) + b"\n"
    path.write_bytes(raw)
    evidence = {
        "schema": _ACK_SCHEMA,
        "attempt": 1,
        "call_id": "call",
        "runtime": getattr(client, "runtime_id", "runtime-under-test"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    path.with_name(f"{path.stem}.ack.json").write_bytes(canonical_json(evidence) + b"\n")

    got = client.poll(str(path))

    assert got.status == "OK"
    assert got.op == "seal"
    assert got.payload == result["payload"]


class _UnconsumedWriter:
    """Expose helper-written bytes before delivering the helper's durability ACK."""

    def __init__(self):
        self.on_ack = None

    def write(self, path, value, *, replace_existing, wait, on_ack):
        host._write_json(path, value, replace_existing=replace_existing, durable=False)
        self.on_ack = on_ack
        raise JournalBusy("injected pending durability ACK")

    def read(self, path, label, *, wait):
        if not path.is_file():
            return None
        return json.loads(path.read_bytes())

    def consume_ack(self):
        assert self.on_ack is not None
        callback, self.on_ack = self.on_ack, None
        callback()


def _supervisor_with_unconsumed_writer(tmp_path: Path):
    run = tmp_path
    client = _client(run)
    supervisor = ParallelSupervisor(run)
    # The production engine is a child of this supervisor process.
    client.runtime_id = host._host_runtime_id(os.getpid())
    writer = _UnconsumedWriter()
    supervisor._channel_writes = writer
    supervisor._channel_thread = threading.current_thread()
    supervisor._answer_timer = lambda *args: None
    return client, supervisor, writer


def _wait_for_ack(path: Path) -> None:
    deadline = time.monotonic() + scaled(5)
    while not path.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.is_file(), "ACK evidence was not published after durability settled"


def test_supervisor_publishes_numbered_ack_only_after_helper_ack_is_consumed(tmp_path):
    client, supervisor, writer = _supervisor_with_unconsumed_writer(tmp_path)
    path, answer = _answer(client)

    with pytest.raises(JournalBusy):
        supervisor._write_json(path, answer)
    ack_path = path.with_name("001.ack.json")
    assert path.is_file(), "the helper has exposed the answer before settling durability"
    assert not ack_path.exists(), "ACK evidence overtook the helper durability ACK"

    writer.consume_ack()
    supervisor._publish_pending_ack_evidence()
    _wait_for_ack(ack_path)

    expected = {
        "schema": _ACK_SCHEMA,
        "attempt": 1,
        "call_id": "call",
        "runtime": client.runtime_id,
        "sha256": hashlib.sha256(canonical_json(answer) + b"\n").hexdigest(),
    }
    assert json.loads(ack_path.read_bytes()) == expected


def test_supervisor_publishes_seal_ack_only_after_helper_ack_is_consumed(tmp_path):
    client, supervisor, writer = _supervisor_with_unconsumed_writer(tmp_path)
    seals = client.call_root / "host" / "seals"
    seals.mkdir(parents=True)
    payload = {"unit_id": "a", "incarnation": 1}
    request = HostRequest(1, "call", 1, "seal", "a", payload,
                          hashlib.sha256(canonical_json(payload)).hexdigest())
    job = seals / "a-1.json"
    job.write_bytes(canonical_json({"schema": "stateframe-parallel-seal-job/1",
                                    "request": request.to_json()}) + b"\n")
    path = seals / "a-1.result.json"
    result = {"status": "OK", "payload": {"proposal_digest": "sealed"}}

    with pytest.raises(JournalBusy):
        supervisor._write_json(path, result)
    ack_path = seals / "a-1.result.ack.json"
    assert path.is_file(), "the helper has exposed the seal result before settling durability"
    assert not ack_path.exists(), "seal ACK evidence overtook the helper durability ACK"

    writer.consume_ack()
    supervisor._publish_pending_ack_evidence()
    _wait_for_ack(ack_path)

    expected = {
        "schema": _ACK_SCHEMA,
        "attempt": 1,
        "call_id": "call",
        "runtime": client.runtime_id,
        "sha256": hashlib.sha256(canonical_json(result) + b"\n").hexdigest(),
    }
    assert json.loads(ack_path.read_bytes()) == expected
