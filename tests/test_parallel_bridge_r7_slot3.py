"""A child exit cannot release a subframe slot before host family-exit proof."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time

import pytest

from stateframe import parallel_resources


def test_durable_slot_claim_waits_for_host_process_identity(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "worker"
    process_dir = run_dir / "proc" / "7"
    process_dir.mkdir(parents=True)
    process_record = process_dir / "process.json"
    lease_root = tmp_path / "slot-leases"
    endpoint = "http://127.0.0.1:8080/v1"
    published = threading.Event()
    monkeypatch.chdir(run_dir)
    monkeypatch.setenv("SF_SUBFRAME_INCARNATION", "7")
    monkeypatch.setenv("SF_SUBFRAME_SLOT_CLAIM", "durable-v1")
    monkeypatch.setattr(parallel_resources, "LEASE_ROOT", lease_root)

    def publish_identity() -> None:
        from stateframe.parallel_worker import kernel_start_identity

        pid = os.getpid()
        identity = {"incarnation": 7, "pid": pid, "pgid": pid,
                    "kernel_identity": kernel_start_identity(pid)}
        temporary = process_record.with_name(".process.json.tmp")
        time.sleep(0.05)
        temporary.write_text(json.dumps(identity), encoding="utf-8")
        os.replace(temporary, process_record)
        published.set()

    writer = threading.Thread(target=publish_identity, daemon=True)
    writer.start()
    lease = parallel_resources.lease_slot(endpoint, [3])
    published_before_return = published.is_set()
    writer.join(timeout=5)

    try:
        assert not writer.is_alive(), "the host identity writer should finish"
        assert published_before_return, "slot acquisition must wait for the host identity record"
        assert lease is not None
        endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16]
        claim = lease_root / endpoint_hash / "3.handoff.json"
        assert claim.is_file(), "the durable claim must be published before the lease is returned"
    finally:
        if lease is not None:
            lease.close()


def test_sigkill_during_permanent_handoff_failure_keeps_slot_until_family_proof(
        tmp_path: Path, monkeypatch, caplog) -> None:
    run_dir = tmp_path / "worker"
    process_dir = run_dir / "proc" / "7"
    process_dir.mkdir(parents=True)
    lease_root = tmp_path / "slot-leases"
    handoff_retrying = tmp_path / "handoff-retrying"
    endpoint = "http://127.0.0.1:8080/v1"
    source_root = Path(parallel_resources.__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), child_env.get("PYTHONPATH", "")) if part)
    child_env["SF_SUBFRAME_INCARNATION"] = "7"
    child_env["SF_SUBFRAME_SLOT_CLAIM"] = "durable-v1"
    child_code = "\n".join((
        "import json, os, signal, sys",
        "from pathlib import Path",
        "from stateframe import parallel_resources as resources",
        "from stateframe.parallel_worker import kernel_start_identity",
        "lease_root, retrying = map(Path, sys.argv[1:3])",
        "run_dir = Path.cwd()",
        "incarnation = int(os.environ['SF_SUBFRAME_INCARNATION'])",
        "pid = os.getpid()",
        "identity = {'pid': pid, 'pgid': pid, 'kernel_identity': kernel_start_identity(pid)}",
        "process_dir = run_dir / 'proc' / str(incarnation)",
        "process_dir.joinpath('process.json').write_text(",
        "    json.dumps({'incarnation': incarnation, **identity}), encoding='utf-8')",
        "resources.LEASE_ROOT = lease_root",
        f"lease = resources.lease_slot({endpoint!r}, [3])",
        "assert lease is not None",
        "def fail_handoff(*args):",
        "    retrying.touch(exist_ok=True)",
        "    raise OSError('injected permanent handoff failure')",
        "resources._publish_slot_handoff = fail_handoff",
        "signal.signal(signal.SIGTERM, lambda *_: None)",
        "lease.handoff_to_host()",
    ))
    monkeypatch.setattr(parallel_resources, "LEASE_ROOT", lease_root)
    monkeypatch.delenv("SF_SUBFRAME_INCARNATION", raising=False)
    monkeypatch.delenv("SF_SUBFRAME_SLOT_CLAIM", raising=False)

    child = subprocess.Popen([sys.executable, "-c", child_code, str(lease_root), str(handoff_retrying)],
                             cwd=run_dir, env=child_env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 15
        while not handoff_retrying.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert handoff_retrying.exists(), "child did not enter the failed handoff retry loop"

        child.send_signal(signal.SIGTERM)
        time.sleep(0.1)
        assert child.poll() is None, "the child should still hold its flock while handoff retries"
        child.kill()
        child.wait(timeout=5)

        blocked = parallel_resources.lease_slot(endpoint, [3])
        if blocked is not None:
            blocked.close()
        assert blocked is None, "SIGKILL must not release a slot before family-exit proof"
        assert "unresolved or invalid host family-exit claim" in caplog.text

        record = json.loads((process_dir / "process.json").read_text(encoding="utf-8"))
        identity = {key: record[key] for key in ("pid", "pgid", "kernel_identity")}
        (process_dir / "family-exited.json").write_text(json.dumps(identity), encoding="utf-8")
        available = parallel_resources.lease_slot(endpoint, [3])
        assert available is not None, "matching birth-bound family-exit proof should release the claim"
        available.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_visible_matching_claim_retries_directory_fsync(tmp_path: Path, monkeypatch) -> None:
    endpoint = "http://127.0.0.1:8080/v1"
    lease_root = tmp_path / "slot-leases"
    endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16]
    lock_path = lease_root / endpoint_hash / "3.lock"
    lock_path.parent.mkdir(parents=True)
    process_dir = tmp_path / "worker" / "proc" / "7"
    process_dir.mkdir(parents=True)
    proof_path = process_dir / "family-exited.json"
    identity = {
        "pid": 1234,
        "pgid": 1234,
        "kernel_identity": "linux:00000000-0000-0000-0000-000000000000:1",
    }
    marker = lock_path.with_name("3.handoff.json")
    real_fsync = os.fsync
    directory_attempts = 0

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal directory_attempts
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_attempts += 1
            if directory_attempts == 1:
                raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(parallel_resources.os, "fsync", fail_first_directory_fsync)
    with pytest.raises(OSError, match="directory fsync failure"):
        parallel_resources._publish_slot_handoff(lock_path, 3, proof_path, identity, 7)

    assert marker.is_file(), "os.replace must have made the first marker visible"
    visible = marker.read_bytes()
    assert json.loads(visible) == {
        "schema": "stateframe-slot-lease-handoff/1",
        "slot": 3,
        "incarnation": 7,
        "proof_path": str(proof_path),
        "identity": identity,
    }

    parallel_resources._publish_slot_handoff(lock_path, 3, proof_path, identity, 7)
    assert directory_attempts == 2, "an exact retry must repeat the directory durability barrier"
    assert marker.read_bytes() == visible


def test_mismatched_existing_claim_remains_fail_closed(tmp_path: Path, monkeypatch, caplog) -> None:
    endpoint = "http://127.0.0.1:8080/v1"
    lease_root = tmp_path / "slot-leases"
    endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16]
    lock_path = lease_root / endpoint_hash / "3.lock"
    lock_path.parent.mkdir(parents=True)
    marker = lock_path.with_name("3.handoff.json")
    mismatched = {
        "schema": "stateframe-slot-lease-handoff/1",
        "slot": 3,
        "incarnation": 8,
        "proof_path": str(tmp_path / "worker" / "proc" / "8" / "family-exited.json"),
        "identity": {
            "pid": 2345,
            "pgid": 2345,
            "kernel_identity": "linux:00000000-0000-0000-0000-000000000000:2",
        },
    }
    marker_bytes = json.dumps(mismatched).encode("utf-8")
    marker.write_bytes(marker_bytes)
    monkeypatch.setattr(parallel_resources, "LEASE_ROOT", lease_root)
    monkeypatch.delenv("SF_SUBFRAME_INCARNATION", raising=False)
    monkeypatch.delenv("SF_SUBFRAME_SLOT_CLAIM", raising=False)

    lease = parallel_resources.lease_slot(endpoint, [3])
    if lease is not None:
        lease.close()
    assert lease is None, "a mismatched existing claim must keep the slot unavailable"
    assert marker.read_bytes() == marker_bytes, "the conflicting claim must not be replaced or removed"
    assert "unresolved or invalid host family-exit claim" in caplog.text

    requested_identity = {
        "pid": 3456,
        "pgid": 3456,
        "kernel_identity": "linux:00000000-0000-0000-0000-000000000000:3",
    }
    with pytest.raises(FileExistsError):
        parallel_resources._publish_slot_handoff(
            lock_path, 3, Path(mismatched["proof_path"]), requested_identity, 8)
    assert marker.read_bytes() == marker_bytes
