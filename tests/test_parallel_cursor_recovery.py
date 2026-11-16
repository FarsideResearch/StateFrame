"""Channel cursors are written without fsync and must never be trusted beyond the durable journal.

A cursor only decides where a service pass starts; every answer is published through a durable,
atomic identity record. So a cursor a SIGKILL left stale, torn, empty, missing or pointing past
the durable work must only change service order: resuming replays idempotently with no lost and
no duplicated answers. These tests are engine-free and never call ``sys.setswitchinterval``.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import stateframe.parallel_host as host
from stateframe.engine.kernel_parallel_records import HostAnswer, HostRequest, canonical_json
from stateframe.parallel_host import ParallelSupervisor

from _timeouts import scaled

_SERVED_BEFORE_KILL = 6
_ADDED_AFTER_KILL = 2

# The child serves every pending request, then publishes a non-durable cursor exactly as the
# supervisor does on a tick, reports, and waits to be killed before any flush could happen.
_CHILD = """
import sys
from pathlib import Path
import stateframe.parallel_host as host
from stateframe.parallel_host import ParallelSupervisor

run = Path(sys.argv[1])
attempt = run / "parallel" / "call" / "attempt-1"
supervisor = ParallelSupervisor(run)
supervisor.serve(attempt / "rq", attempt / "an")
host._write_json(attempt / "channel-cursor.json", {"last": int(sys.argv[2])},
                 replace_existing=True, durable=False)
sys.stdout.write("cursor-written\\n")
sys.stdout.flush()
import time
time.sleep(120)
"""


def _request(seq: int) -> HostRequest:
    payload = {"unit_id": "a"}
    return HostRequest(1, "call", seq, "status", "test", payload,
                       hashlib.sha256(canonical_json(payload)).hexdigest())


def _write_requests(attempt: Path, sequences: range) -> None:
    (attempt / "rq").mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        (attempt / "rq" / f"{seq:03d}.json").write_bytes(canonical_json(_request(seq).to_json()) + b"\n")


def _channel_files(an_dir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in _visible_entries(an_dir)
            if path.suffix == ".json" and path.stem.isdigit()}


def _visible_entries(an_dir: Path) -> list[Path]:
    """Sorted channel entries, skipping dot-prefixed atomic-writer mkstemp debris.

    The atomic writer names temps ".{name}.XXXX"; a SIGKILL mid-write leaves them behind and
    the final NNN.* files are never partially visible, so temps are not channel artifacts.
    """
    return sorted(path for path in an_dir.iterdir() if not path.name.startswith("."))


def _channel_artifacts(an_dir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in _visible_entries(an_dir)
            if not (path.name.endswith(".ack.json")
                    and path.name.removesuffix(".ack.json").isdigit())}


def _channel_ack_files(an_dir: Path) -> set[str]:
    return {path.name for path in _visible_entries(an_dir)
            if path.name.endswith(".ack.json")
            and path.name.removesuffix(".ack.json").isdigit()}


def _assert_no_stray_files(an_dir: Path, sequences: range) -> None:
    """Fail on any non-temp file that is not an expected answer, record or ACK sidecar."""
    allowed = _expected_names(sequences) | {f"{seq:03d}.ack.json" for seq in sequences}
    stray = sorted(path.name for path in _visible_entries(an_dir) if path.name not in allowed)
    assert not stray, f"unexpected files in channel: {stray}"


def _assert_answer_ack(an_dir: Path, seq: int) -> None:
    answer_path = an_dir / f"{seq:03d}.json"
    ack_path = an_dir / f"{seq:03d}.ack.json"
    deadline = time.monotonic() + scaled(5)
    while not ack_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ack_path.is_file(), f"served answer {answer_path.name} has no ACK sidecar"
    evidence = json.loads(ack_path.read_bytes())
    assert isinstance(evidence, dict)
    assert set(evidence) == {"schema", "attempt", "call_id", "runtime", "sha256"}
    assert evidence["schema"] == "stateframe-parallel-ack/1"
    assert evidence["attempt"] == 1
    assert evidence["call_id"] == "call"
    assert isinstance(evidence["runtime"], str) and evidence["runtime"]
    assert evidence["sha256"] == hashlib.sha256(answer_path.read_bytes()).hexdigest()


def _expected_names(sequences: range) -> set[str]:
    return {f"{seq:03d}.json" for seq in sequences} | {f"{seq:03d}.record.json" for seq in sequences}


def _expected_answer_names(sequences: range) -> set[str]:
    return {f"{seq:03d}.json" for seq in sequences}


def _kill_recorded_child(process: subprocess.Popen) -> None:
    """SIGKILL only the pid we spawned, after confirming /proc still names our child script."""
    cmdline = Path(f"/proc/{process.pid}/cmdline").read_bytes()
    assert b"cursor-written" in cmdline, cmdline
    os.kill(process.pid, signal.SIGKILL)
    assert process.wait(timeout=scaled(30)) == -signal.SIGKILL


def _serve_until_complete(supervisor: ParallelSupervisor, attempt: Path, sequences: range) -> None:
    """Service channels until every seq has its answer, record and ACK sidecar on disk.

    ACK sidecars are published by the asynchronous ACK-evidence worker, so answers alone are not
    enough: waiting on them keeps the test from racing that worker under load.
    """
    an_dir = attempt / "an"
    deadline = time.monotonic() + scaled(30)
    while time.monotonic() < deadline:
        supervisor._service_channels()
        if all((an_dir / f"{seq:03d}.json").is_file()
               and (an_dir / f"{seq:03d}.record.json").is_file()
               and (an_dir / f"{seq:03d}.ack.json").is_file() for seq in sequences):
            return
    missing_answers = [seq for seq in sequences if not (an_dir / f"{seq:03d}.json").is_file()]
    missing_records = [seq for seq in sequences if not (an_dir / f"{seq:03d}.record.json").is_file()]
    missing_acks = [seq for seq in sequences if not (an_dir / f"{seq:03d}.ack.json").is_file()]
    pytest.fail(
        "resumed supervisor did not answer every pending request before the deadline: "
        f"missing answers={missing_answers} records={missing_records} acks={missing_acks}"
    )


def _damage(kind: str, cursor: Path, an_dir: Path) -> None:
    if kind == "stale_behind":
        return  # the child left the cursor pointing at position 2 while positions 1-6 are durably answered
    if kind == "ahead_of_durable_work":
        # The real tick writes the cursor before serving a seq: a crash between the two leaves a
        # cursor past answers that never became durable. Drop the last two durable records.
        for seq in (_SERVED_BEFORE_KILL - 1, _SERVED_BEFORE_KILL):
            (an_dir / f"{seq:03d}.json").unlink()
            (an_dir / f"{seq:03d}.record.json").unlink()
        cursor.write_bytes(canonical_json({"last": _SERVED_BEFORE_KILL}) + b"\n")
        return
    if kind == "torn":
        data = cursor.read_bytes()
        cursor.write_bytes(data[: len(data) // 2])
        return
    if kind == "empty":
        cursor.write_bytes(b"")
        return
    if kind == "missing":
        cursor.unlink()
        return
    if kind == "far_future":
        cursor.write_bytes(canonical_json({"last": 999}) + b"\n")
        return
    if kind == "wrong_type":
        cursor.write_bytes(canonical_json({"last": "6"}) + b"\n")
        return
    raise AssertionError(kind)


@pytest.mark.parametrize("damage", ["stale_behind", "ahead_of_durable_work", "torn", "empty",
                                    "missing", "far_future", "wrong_type"])
def test_kill9_after_non_durable_cursor_resumes_without_loss_or_duplicates(tmp_path, damage):
    run = tmp_path / "run"
    attempt = run / "parallel" / "call" / "attempt-1"
    an_dir = attempt / "an"
    cursor = attempt / "channel-cursor.json"
    _write_requests(attempt, range(1, _SERVED_BEFORE_KILL + 1))

    process = subprocess.Popen([sys.executable, "-c", _CHILD, str(run), "2"], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert process.stdout.readline().strip() == "cursor-written", process.stderr.read()
        _kill_recorded_child(process)
    finally:
        if process.poll() is None:
            _kill_recorded_child(process)

    before = _channel_files(an_dir)
    before_artifacts = _channel_artifacts(an_dir)
    assert set(before) == _expected_answer_names(range(1, _SERVED_BEFORE_KILL + 1))
    assert set(before_artifacts) == _expected_names(range(1, _SERVED_BEFORE_KILL + 1))
    _assert_no_stray_files(an_dir, range(1, _SERVED_BEFORE_KILL + 1))
    assert json.loads(cursor.read_text()) == {"last": 2}
    _damage(damage, cursor, an_dir)
    surviving = {name: data for name, data in before_artifacts.items() if (an_dir / name).exists()}

    # Resume: new requests arrive, a fresh supervisor services the channel from whatever cursor
    # the crash left, then a direct serve replays the whole channel once more.
    all_sequences = range(1, _SERVED_BEFORE_KILL + _ADDED_AFTER_KILL + 1)
    _write_requests(attempt, range(_SERVED_BEFORE_KILL + 1, all_sequences.stop))
    supervisor = ParallelSupervisor(run)
    _serve_until_complete(supervisor, attempt, all_sequences)
    replayed = supervisor.serve(attempt / "rq", an_dir)

    after = _channel_files(an_dir)
    after_artifacts = _channel_artifacts(an_dir)
    after_acks = _channel_ack_files(an_dir)
    _assert_no_stray_files(an_dir, all_sequences)
    assert set(after) == _expected_answer_names(all_sequences), "an answer was lost, duplicated or conflicted"
    assert set(after_artifacts) == _expected_names(all_sequences), (
        "an answer was lost, duplicated or conflicted")
    assert after_acks == {f"{seq:03d}.ack.json" for seq in all_sequences}
    for name, data in surviving.items():
        assert after_artifacts[name] == data, f"durable {name} was rewritten after resume"
    assert sorted(answer.seq for answer in replayed) == list(all_sequences)
    for answer in replayed:
        assert (answer.attempt, answer.call_id, answer.op) == (1, "call", "status")
        assert HostAnswer.from_json(json.loads(after[f"{answer.seq:03d}.json"])) == answer
        _assert_answer_ack(an_dir, answer.seq)
    # The resumed cursor is a plain ordering hint again, pointing inside the served range.
    assert json.loads(cursor.read_text())["last"] in all_sequences


def test_cursor_only_changes_service_order_never_the_served_set(tmp_path, monkeypatch):
    """Whatever a cursor claims, one pass serves every pending seq exactly once."""
    run = tmp_path / "run"
    attempt = run / "parallel" / "call" / "attempt-1"
    _write_requests(attempt, range(1, 5))
    supervisor = ParallelSupervisor(run)
    for claimed in (None, -7, 0, 2, 4, 10**9, True, "3", 2.0):
        monkeypatch.setattr(host, "_read_cursor", lambda path, value=claimed: value)
        answers = supervisor.serve(attempt / "rq", attempt / "an")
        assert sorted(answer.seq for answer in answers) == [1, 2, 3, 4], claimed
        for answer in answers:
            _assert_answer_ack(attempt / "an", answer.seq)
    answer_files = _channel_files(attempt / "an")
    artifacts = _channel_artifacts(attempt / "an")
    assert set(answer_files) == _expected_answer_names(range(1, 5))
    assert set(artifacts) == _expected_names(range(1, 5))
    assert _channel_ack_files(attempt / "an") == {f"{seq:03d}.ack.json" for seq in range(1, 5)}


def test_cursor_write_is_not_durable_and_answers_are(tmp_path, monkeypatch):
    run = tmp_path / "run"
    attempt = run / "parallel" / "call" / "attempt-1"
    _write_requests(attempt, range(1, 3))
    synced: list[str] = []
    real_write = host._write_json

    def recording_write(path, value, *, replace_existing=False, durable=True):
        synced.append(f"{Path(path).name}:{durable}")
        real_write(path, value, replace_existing=replace_existing, durable=durable)

    monkeypatch.setattr(host, "_write_json", recording_write)
    supervisor = ParallelSupervisor(run)
    answers = supervisor.serve(attempt / "rq", attempt / "an")
    for answer in answers:
        _assert_answer_ack(attempt / "an", answer.seq)
    assert synced.count("channel-cursor.json:False") == 2
    assert "channel-cursor.json:True" not in synced
    assert {"001.record.json:True", "002.record.json:True"} <= set(synced)


def test_supervisor_never_touches_the_interpreter_switch_interval(tmp_path, monkeypatch):
    """The GIL switch interval is process-wide; the supervisor must not change it for the host."""
    assert not hasattr(host, "_SWITCH_INTERVAL_SECONDS")
    monkeypatch.setattr(sys, "setswitchinterval",
                        lambda value: pytest.fail(f"supervisor set switch interval to {value}"))
    ParallelSupervisor(tmp_path / "run")


def test_channel_helpers_skip_atomic_temp_debris_but_flag_stray_files(tmp_path):
    an_dir = tmp_path / "an"
    an_dir.mkdir()
    (an_dir / "001.json").write_bytes(b"{}\n")
    (an_dir / "001.record.json").write_bytes(b"{}\n")
    (an_dir / "001.ack.json").write_bytes(b"{}\n")
    (an_dir / ".002.ack.json.64a9y0i_").write_bytes(b"partial")

    assert set(_channel_artifacts(an_dir)) == {"001.json", "001.record.json"}
    assert _channel_ack_files(an_dir) == {"001.ack.json"}
    assert set(_channel_files(an_dir)) == {"001.json"}
    _assert_no_stray_files(an_dir, range(1, 2))

    (an_dir / "999.json").write_bytes(b"{}\n")
    with pytest.raises(AssertionError, match="999.json"):
        _assert_no_stray_files(an_dir, range(1, 2))
