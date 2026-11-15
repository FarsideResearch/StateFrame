import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import pytest

import stateframe.parallel_resources as resources


@pytest.mark.parametrize(("value", "expected"), [
    (None, None),
    ("2,5", (2, 5)),
    (" 2, 5 ", (2, 5)),
    ([5, 2], (5, 2)),
    ((5, 2), (5, 2)),
])
def test_parse_slot_pool_preserves_operator_order(value, expected):
    assert resources.parse_slot_pool(value) == expected


@pytest.mark.parametrize("value", [
    "", "  ", "2,,5", "2,", "-1,2", "1.5", "２", "2,02", [], [2, 2], [True], ["2"],
])
def test_parse_slot_pool_refuses_invalid_ids_and_duplicates(value):
    with pytest.raises(resources.SlotPoolError, match="--parallel-slots"):
        resources.parse_slot_pool(value)


@pytest.mark.parametrize(("pool", "idle", "parent_slot", "fallback", "expected"), [
    ((2, 5), [2, 5, 7], 5, 4, [5, 2]),
    ((2, 5), [5], 7, 4, [5]),
    ((2, 5), None, 5, 4, [5, 2]),
    ((2, 5), None, 7, 4, [2, 5]),
    ((2, 5), [], 2, 4, [2]),
    ((2, 5), [], 7, 4, []),
    (None, [2, 5], 7, 4, [7, 2, 5]),
    (None, None, 5, 3, [5, 0, 1, 2]),
    (None, [], None, 3, []),
])
def test_slot_candidates_respect_pool_idle_parent_and_fallback_order(
        pool, idle, parent_slot, fallback, expected):
    assert resources.slot_candidates(pool, idle, parent_slot, fallback_capacity=fallback) == expected


def test_idle_slots_returns_unique_idle_ids_and_none_when_endpoint_does_not_support_slots(monkeypatch):
    class FakeEndpoint:
        entries = []

        def __init__(self, endpoint, model, *, timeout_s):
            assert endpoint == "http://127.0.0.1:8080/v1"
            assert model == "stateframe-parallel-slots"
            assert timeout_s == 2.0

        def get_slots(self):
            return self.entries

    FakeEndpoint.entries = [
        {"id": 3, "is_processing": False},
        {"id": 1, "is_processing": True},
        {"id_slot": 2, "is_processing": False},
        {"id": 3, "is_processing": True},
        {"id": -1, "is_processing": False},
        {"id": 4, "is_processing": "false"},
    ]
    monkeypatch.setattr(resources, "Endpoint", FakeEndpoint)

    assert resources.idle_slots("http://127.0.0.1:8080/v1") == [2]

    FakeEndpoint.entries = None
    assert resources.idle_slots("http://127.0.0.1:8080/v1") is None


def test_slot_lease_records_post_acquisition_time_for_current_incarnation(tmp_path, monkeypatch):
    run = tmp_path / "worker"
    (run / "proc" / "2").mkdir(parents=True)
    monkeypatch.chdir(run)
    monkeypatch.setenv("SF_SUBFRAME_INCARNATION", "2")
    monkeypatch.setattr(resources, "LEASE_ROOT", tmp_path / "leases")
    before = time.time()
    lease = resources.lease_slot("http://127.0.0.1:8080/v1", [7])
    after = time.time()
    assert lease is not None
    try:
        marker = json.loads((run / "proc" / "2" / "slot-lease.json").read_text())
        assert marker["schema"] == "stateframe-subframe-slot-lease/1"
        assert marker["slot"] == 7
        assert marker["pid"] == os.getpid()
        assert before <= marker["acquired_epoch"] <= after
    finally:
        lease.close()
    with pytest.raises(resources.ResourceError, match="record subframe slot lease"):
        resources.lease_slot("http://127.0.0.1:8080/v1", [7])


def test_slot_lease_resolves_home_when_each_lease_is_acquired(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "LEASE_ROOT", None)
    monkeypatch.delenv("SF_SUBFRAME_INCARNATION", raising=False)
    first_home = tmp_path / "first-home"
    second_home = tmp_path / "second-home"

    for home in (first_home, second_home):
        monkeypatch.setenv("HOME", str(home))
        lease = resources.lease_slot("http://127.0.0.1:8080/v1", [1])
        assert lease is not None
        try:
            assert lease.path.is_relative_to(home / ".stateframe" / "slot-leases")
        finally:
            lease.close()


def test_slot_lease_fails_clearly_without_a_home(monkeypatch):
    monkeypatch.setattr(resources, "LEASE_ROOT", None)
    monkeypatch.setattr(resources, "safe_home", lambda: None)
    with pytest.raises(resources.ResourceError, match="home directory is unavailable"):
        resources.lease_slot("http://127.0.0.1:8080/v1", [0])


def test_routed_subframe_uses_its_acquired_slot_in_request_authoring(tmp_path, monkeypatch):
    run = tmp_path / "worker"
    (run / "proc" / "1").mkdir(parents=True)
    monkeypatch.chdir(run)
    monkeypatch.setenv("SF_SUBFRAME_INCARNATION", "1")
    monkeypatch.setenv("STATEFRAME_ROUTE_SCOPE", "unit")
    monkeypatch.setenv("STATEFRAME_ROUTE_SLOT", "99")
    monkeypatch.setattr(resources, "LEASE_ROOT", tmp_path / "leases")

    lease = resources.lease_slot("http://127.0.0.1:9090/v1/chat/completions", [3])
    assert lease is not None
    try:
        assert os.environ["STATEFRAME_ROUTE_SLOT"] == "3"
        assert lease.slot == 3
    finally:
        lease.close()


def test_slot_lease_uses_candidate_order(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "LEASE_ROOT", tmp_path / "leases")
    lease = resources.lease_slot("http://127.0.0.1:8080/v1", [7, 2])
    assert lease is not None
    try:
        assert lease.slot == 7
    finally:
        lease.close()


@pytest.mark.host_sandbox
def test_slot_lease_is_exclusive_across_processes_and_released_when_holder_is_killed(tmp_path, monkeypatch):
    lease_root = tmp_path / "leases"
    monkeypatch.setattr(resources, "LEASE_ROOT", lease_root)
    source = Path(__file__).resolve().parents[1] / "src"
    script = tmp_path / "hold_lease.py"
    script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "import stateframe.parallel_resources as resources\n"
        "resources.LEASE_ROOT = Path(sys.argv[1])\n"
        "lease = resources.lease_slot('http://127.0.0.1:8080/v1', [0])\n"
        "print(lease.slot if lease else 'none', flush=True)\n"
        "time.sleep(60)\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source)
    process = subprocess.Popen([sys.executable, str(script), str(lease_root)], cwd=tmp_path, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout is not None
        readable, _, _ = select.select([process.stdout], [], [], 5)
        assert readable, "lease holder did not report its acquired slot"
        assert process.stdout.readline().strip() == "0"
        assert resources.lease_slot("http://127.0.0.1:8080/v1", [0]) is None

        process.kill()
        process.wait(timeout=5)

        lease = resources.lease_slot("http://127.0.0.1:8080/v1", [0])
        assert lease is not None and lease.slot == 0
        lease.close()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_machine_ok_enforces_disk_and_memory_thresholds(monkeypatch, tmp_path):
    required_disk = (2 * 10) + (1 << 30)
    probed = []

    def disk_usage(path):
        probed.append(path)
        return type("Usage", (), {"free": required_disk})()

    monkeypatch.setattr(resources.shutil, "disk_usage", disk_usage)
    monkeypatch.setattr(resources, "_memory_available", lambda: (20.0, "test-memory"))
    ok, detail = resources.machine_ok(10, tmp_path)
    assert ok is True
    assert probed == [tmp_path]
    assert "20.0% free" in detail

    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _path: type("Usage", (), {"free": required_disk - 1})())
    ok, detail = resources.machine_ok(10, tmp_path)
    assert ok is False
    assert "below required" in detail

    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _path: type("Usage", (), {"free": required_disk})())
    monkeypatch.setattr(resources, "_memory_available", lambda: (19.9, "test-memory"))
    ok, detail = resources.machine_ok(10, tmp_path)
    assert ok is False
    assert "19.9% free" in detail


@pytest.mark.parametrize("allocated", [0, 10, 20])
@pytest.mark.parametrize("failed", [False, True])
def test_preparation_reservation_releases_and_reprobes_actual_disk(
        monkeypatch, tmp_path, allocated, failed):
    free_disk = (1 << 30) + 20
    monkeypatch.setattr(resources.shutil, "disk_usage",
                        lambda _path: type("Usage", (), {"free": free_disk})())
    monkeypatch.setattr(resources, "_memory_available", lambda: (50.0, "test-memory"))
    other_run = tmp_path / "other-run"
    other_run.mkdir()

    try:
        with resources.reserve_preparation(10, tmp_path) as (ready, _reason):
            assert ready
            with resources.reserve_preparation(10, other_run) as (ready, reason):
                assert not ready
                assert "below required" in reason
            # Reflinks, full copies and failed partial copies consume different amounts.
            free_disk -= allocated
            if failed:
                raise OSError("copy failed")
    except OSError:
        assert failed

    with resources.reserve_preparation(10, other_run) as (ready, _reason):
        assert ready is (allocated == 0)
    with resources.reserve_preparation(0, other_run) as (ready, _reason):
        assert ready, "completed or failed preparation leaked its reservation"


def test_empty_preparations_can_overlap_without_leaking_reservations(monkeypatch, tmp_path):
    monkeypatch.setattr(resources.shutil, "disk_usage",
                        lambda _path: type("Usage", (), {"free": 1 << 30})())
    monkeypatch.setattr(resources, "_memory_available", lambda: (50.0, "test-memory"))
    with resources.reserve_preparation(0, tmp_path) as (first, _reason):
        with resources.reserve_preparation(0, tmp_path) as (second, _reason):
            assert first and second
    with resources.reserve_preparation(0, tmp_path) as (ready, _reason):
        assert ready


def test_machine_ok_uses_current_home_and_handles_missing_home(monkeypatch, tmp_path):
    required_disk = (2 * 10) + (1 << 30)
    probed = []

    def disk_usage(path):
        probed.append(path)
        return type("Usage", (), {"free": required_disk})()

    monkeypatch.setattr(resources.shutil, "disk_usage", disk_usage)
    monkeypatch.setattr(resources, "_memory_available", lambda: (20.0, "test-memory"))
    for home in (tmp_path / "first-home", tmp_path / "second-home"):
        monkeypatch.setenv("HOME", str(home))
        ok, _detail = resources.machine_ok(10)
        assert ok is True
        assert probed[-1] == home

    monkeypatch.setattr(resources, "safe_home", lambda: None)
    assert resources.machine_ok(10) == (
        False, "home directory is unavailable for disk free-space probe")
    assert len(probed) == 2


def test_send_ledger_charges_authorized_send_after_crash_before_settlement(tmp_path):
    path = tmp_path / "run" / "parallel" / "call" / "units" / "a" / "sends-ledger.jsonl"
    path.parent.mkdir(parents=True)
    ledger = resources.SendLedger(path)
    ledger.authorize("send-1")
    ledger.authorize("send-1")

    recovered = resources.SendLedger(path)
    assert recovered.remaining(3) == 2
    usage = {"prompt_tokens": 17, "completion_tokens": 4}
    recovered.settle("send-1", usage)
    recovered.settle("send-1", usage)

    assert recovered.remaining(3) == 2
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(record["state"], record.get("usage")) for record in records] == [
        ("AUTHORIZED", None), ("SETTLED", usage)]
    with pytest.raises(resources.SendLedgerError, match="conflicting settlement"):
        recovered.settle("send-1", {"prompt_tokens": 18, "completion_tokens": 4})


def test_send_ledger_rejects_hardlinked_file_and_symlinked_ancestor(tmp_path):
    path = tmp_path / "run" / "parallel" / "call" / "units" / "a" / "sends-ledger.jsonl"
    path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"")
    os.link(outside, path)

    with pytest.raises(resources.SendLedgerError, match="hardlink|link count"):
        resources.SendLedger(path).authorize("send-1")
    assert outside.read_bytes() == b""

    alternate = tmp_path / "other" / "parallel" / "call" / "units" / "a"
    alternate.mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path / "other", target_is_directory=True)
    with pytest.raises(resources.SendLedgerError, match="symlink|unsafe"):
        resources.SendLedger(linked / "parallel" / "call" / "units" / "a" /
                             "sends-ledger.jsonl").authorize("send-1")


def test_slot_lease_marker_is_published_atomically(tmp_path, monkeypatch):
    run = tmp_path / "worker"
    process_dir = run / "proc" / "3"
    process_dir.mkdir(parents=True)
    monkeypatch.chdir(run)
    monkeypatch.setenv("SF_SUBFRAME_INCARNATION", "3")
    monkeypatch.setattr(resources, "LEASE_ROOT", tmp_path / "leases")
    marker = process_dir / "slot-lease.json"
    seen: list[bytes] = []
    replaced: list[tuple[str, str]] = []
    real_replace = resources.os.replace

    def observing_replace(source, destination):
        # Before publication nothing is readable at the marker path, only the temp file.
        assert not marker.exists()
        assert Path(source).parent == process_dir
        assert Path(source).read_bytes().endswith(b"}\n")
        replaced.append((str(source), str(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(resources.os, "replace", observing_replace)
    lease = resources.lease_slot("http://127.0.0.1:8080/v1", [4])
    assert lease is not None
    try:
        assert replaced == [(str(process_dir / f".slot-lease.json.{os.getpid()}.tmp"), str(marker))]
        seen.append(marker.read_bytes())
        assert json.loads(seen[0]) == {"schema": "stateframe-subframe-slot-lease/1", "slot": 4,
                                       "pid": os.getpid(), "acquired_epoch": pytest.approx(time.time(), abs=5)}
        assert sorted(path.name for path in process_dir.iterdir()) == ["slot-lease.json"]
    finally:
        lease.close()


def test_slot_lease_marker_failure_leaves_no_torn_file(tmp_path, monkeypatch):
    run = tmp_path / "worker"
    process_dir = run / "proc" / "1"
    process_dir.mkdir(parents=True)
    monkeypatch.chdir(run)
    monkeypatch.setenv("SF_SUBFRAME_INCARNATION", "1")
    monkeypatch.setattr(resources, "LEASE_ROOT", tmp_path / "leases")

    def failing_fsync(descriptor):
        raise OSError("disk went away")

    monkeypatch.setattr(resources.os, "fsync", failing_fsync)
    with pytest.raises(resources.ResourceError, match="record subframe slot lease"):
        resources.lease_slot("http://127.0.0.1:8080/v1", [2])
    # Neither a partial marker nor its temp file survives, and the slot lock was released.
    assert list(process_dir.iterdir()) == []
    monkeypatch.setattr(resources.os, "fsync", lambda descriptor: None)
    lease = resources.lease_slot("http://127.0.0.1:8080/v1", [2])
    assert lease is not None
    lease.close()


def test_slot_lease_marker_reader_never_sees_a_partial_record(tmp_path, monkeypatch):
    import threading

    run = tmp_path / "worker"
    process_dir = run / "proc" / "5"
    process_dir.mkdir(parents=True)
    monkeypatch.chdir(run)
    monkeypatch.setenv("SF_SUBFRAME_INCARNATION", "5")
    monkeypatch.setattr(resources, "LEASE_ROOT", tmp_path / "leases")
    marker = process_dir / "slot-lease.json"
    stop = threading.Event()
    observations: list[bytes] = []

    def read_loop():
        while not stop.is_set():
            try:
                observations.append(marker.read_bytes())
            except FileNotFoundError:
                observations.append(b"")

    reader = threading.Thread(target=read_loop, daemon=True)
    reader.start()
    try:
        lease = resources.lease_slot("http://127.0.0.1:8080/v1", [1])
        assert lease is not None
        lease.close()
        time.sleep(0.05)
    finally:
        stop.set()
        reader.join(5)
    final = marker.read_bytes()
    assert final.endswith(b"}\n")
    assert observations and set(observations) <= {b"", final}
