"""R7 reproductions: slot leases must hold until host family-exit proof, and handoff publication must be bounded.

These tests run the real slot code in real child subprocesses with real flock and real files.
They are expected to FAIL on the current commit; they pin the R7 BLOCKER and MAJOR findings.
"""
from __future__ import annotations

from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
from collections import Counter

import pytest

from stateframe import parallel_resources, parallel_worker
from stateframe.parallel_worker import kernel_start_identity

ENDPOINT = "http://127.0.0.1:8080/v1"
SLOT = 3
INCARNATION = 7
MAX_PUBLISH_ATTEMPTS = 50
HANDOFF_JOIN_SECONDS = 90  # bounded retry with capped backoff may legitimately take a while

# Child body: take the real slot lease, make marker publication fail permanently, ignore SIGTERM,
# then enter the real handoff retry loop. The parent SIGKILLs this process mid-handoff.
CHILD_HANDOFF_CODE = "\n".join((
    "import os, signal, sys",
    "from pathlib import Path",
    "from stateframe import parallel_resources as resources",
    "lease_root, ready, endpoint = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]",
    "resources.LEASE_ROOT = lease_root",
    "lease = resources.lease_slot(endpoint, [%d])" % SLOT,
    "assert lease is not None, 'child could not lease the slot'",
    "def permanent_marker_failure(*args):",
    "    raise OSError(errno.EIO, 'injected permanent marker write failure')",
    "resources._publish_slot_handoff = permanent_marker_failure",
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
    "ready.touch()",
    "lease.handoff_to_host()",
))


def _child_env() -> dict[str, str]:
    """Child env as launch_worker builds it: durable slot claim opt-in plus the incarnation marker."""
    source_root = Path(parallel_resources.__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), env.get("PYTHONPATH", "")) if part)
    env["SF_SUBFRAME_INCARNATION"] = str(INCARNATION)
    env["SF_SUBFRAME_SLOT_CLAIM"] = "durable-v1"
    return env


def _launch_record(pid: int, started_epoch: float) -> dict[str, object]:
    """Mirror parallel_worker.launch_worker's launch_record fields (family members not yet verified)."""
    kernel_identity = kernel_start_identity(pid)
    assert kernel_identity is not None, "child exited before its kernel identity could be recorded"
    return {
        "incarnation": INCARNATION,
        "pid": pid,
        "pgid": pid,
        "started_at": datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
        "started_epoch": started_epoch,
        "kernel_identity": kernel_identity,
        "verified_members": {},
        "fenced": False,
    }


def _write_process_record(process_dir: Path, pid: int) -> dict[str, object]:
    """Write the worker process record exactly as the host does for this owner pid."""
    identity: dict[str, object] = {"pid": pid, "pgid": pid, "kernel_identity": kernel_start_identity(pid)}
    process_dir.mkdir(parents=True, exist_ok=True)
    (process_dir / "process.json").write_text(
        json.dumps({"incarnation": INCARNATION, **identity}), encoding="utf-8")
    return identity


def _claim_or_none() -> parallel_resources.SlotLease | None:
    claim = parallel_resources.lease_slot(ENDPOINT, [SLOT])
    if claim is not None:
        claim.close()
    return claim


def test_sigkilled_child_mid_handoff_keeps_slot_until_family_exit_proof(
        tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "worker"
    process_dir = run_dir / "proc" / str(INCARNATION)
    process_dir.mkdir(parents=True)
    lease_root = tmp_path / "slot-leases"
    ready = tmp_path / "in-handoff"
    monkeypatch.setattr(parallel_resources, "LEASE_ROOT", lease_root)
    monkeypatch.delenv("SF_SUBFRAME_INCARNATION", raising=False)
    monkeypatch.chdir(run_dir)

    child = subprocess.Popen(
        [sys.executable, "-c", CHILD_HANDOFF_CODE, str(lease_root), str(ready), ENDPOINT],
        cwd=run_dir, env=_child_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True)
    # Host publishes the record right after spawn; the child's durable claim waits for it.
    # Publish atomically (temp + rename) like the host's link-based _write_json, so the child
    # can never observe a partially written record.
    staged = process_dir / ".process.json.tmp"
    staged.write_text(json.dumps(_launch_record(child.pid, time.time())), encoding="utf-8")
    os.replace(staged, process_dir / "process.json")
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), (
            f"child never reached handoff; returncode={child.poll()} stderr={child.stderr.read() if child.poll() is not None else ''}")

        # Host escalation after TERM is ignored: SIGKILL closes the flock, no family proof exists yet.
        try:  # a fixed child may already have exited on its own; the claim must hold either way
            os.kill(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)
        proof_path = process_dir / "family-exited.json"
        assert not proof_path.exists()

        early = parallel_resources.lease_slot(ENDPOINT, [SLOT])
        if early is not None:
            early.close()
        assert early is None, (
            f"second claimant acquired slot {SLOT} after the owning child was SIGKILLed "
            "mid-handoff, before any family-exit proof was written")

        # Host writes the birth-bound proof after reaping; the same identity the child recorded.
        record = json.loads((process_dir / "process.json").read_text(encoding="utf-8"))
        identity = {key: record[key] for key in ("pid", "pgid", "kernel_identity")}
        proof_path.write_text(json.dumps(identity), encoding="utf-8")

        after_proof = parallel_resources.lease_slot(ENDPOINT, [SLOT])
        assert after_proof is not None, "matching family-exit proof must release the slot"
        after_proof.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        for stream in (child.stdout, child.stderr):
            if stream is not None:
                stream.close()


def test_visible_marker_with_dir_fsync_failure_retries_bounded_then_typed_failure(
        tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "worker"
    process_dir = run_dir / "proc" / str(INCARNATION)
    lease_root = tmp_path / "slot-leases"
    run_dir.mkdir(parents=True)
    _write_process_record(process_dir, os.getpid())
    monkeypatch.setattr(parallel_resources, "LEASE_ROOT", lease_root)
    monkeypatch.setenv("SF_SUBFRAME_INCARNATION", str(INCARNATION))
    monkeypatch.chdir(run_dir)

    lease = parallel_resources.lease_slot(ENDPOINT, [SLOT])
    assert lease is not None
    # Only the directory fsync fails; the marker file fsync and os.replace behave normally.
    real_fsync = os.fsync

    def directory_fsync_fails(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(errno.EIO, "injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(parallel_resources.os, "fsync", directory_fsync_fails)

    real_publish = parallel_resources._publish_slot_handoff
    outcomes: list[str] = []
    guard_tripped = threading.Event()

    def counted_publish(*args: object) -> None:
        # Hard guard: a retry loop that never terminates is itself the failure being pinned.
        if len(outcomes) >= MAX_PUBLISH_ATTEMPTS:
            guard_tripped.set()
            raise RuntimeError("publish retry guard tripped")
        try:
            real_publish(*args)
        except BaseException as error:  # noqa: BLE001 - record every attempt's outcome
            outcomes.append(type(error).__name__)
            raise
        outcomes.append("ok")

    monkeypatch.setattr(parallel_resources, "_publish_slot_handoff", counted_publish)

    result: dict[str, BaseException | None] = {}

    def run_handoff() -> None:
        try:
            lease.handoff_to_host()
            result["outcome"] = None
        except BaseException as error:  # noqa: BLE001 - capture the typed outcome for assertions
            result["outcome"] = error

    # Daemon thread: if the current unbounded retry is still looping after the join, it cannot block
    # interpreter exit. The failure below reports the observed attempt outcomes.
    worker = threading.Thread(target=run_handoff, daemon=True)
    worker.start()
    worker.join(HANDOFF_JOIN_SECONDS)
    summary = dict(Counter(outcomes))
    if worker.is_alive() or guard_tripped.is_set():
        pytest.fail(
            "handoff publish never terminated after the marker became visible with a failed "
            f"directory fsync (guard_tripped={guard_tripped.is_set()}, "
            f"attempts={len(outcomes)}, outcomes={summary})", pytrace=False)

    assert len(outcomes) <= MAX_PUBLISH_ATTEMPTS
    outcome = result.get("outcome")
    # Persistent fsync failure means durability is never proven, so completion is not acceptable.
    assert isinstance(outcome, Exception), (
        f"persistent directory fsync failure must end in a typed failure, outcome={outcome!r}")
    assert not isinstance(outcome, FileExistsError), (
        f"handoff failed with FileExistsError instead of a typed failure: {outcome}")


class _SpawnCaptured(RuntimeError):
    """Raised by the fake Popen to stop launch_worker right after it builds the child environment."""


def test_launch_worker_env_opts_children_into_durable_slot_claim(
        tmp_path: Path, monkeypatch) -> None:
    # Guards the production opt-in: if launch_worker stops setting these, test (a) would pass for the wrong reason.
    run_dir = tmp_path / "worker"
    run_dir.mkdir(parents=True)
    engine_dir = tmp_path / "engine" / "core"
    engine_dir.mkdir(parents=True)
    (engine_dir.parent / "_child.py").write_text("", encoding="utf-8")
    spec = {
        "schema": "stateframe-run/1",
        "public_row": {},
        "engine_dir": str(engine_dir),
        "parallel_engine_python": sys.executable,
        "environment": {},
        "unit_spec": {},
    }
    (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    captured: dict[str, object] = {}

    def capture_popen(command, **kwargs):  # noqa: ANN001 - mirrors subprocess.Popen's keyword surface
        captured["env"] = kwargs["env"]
        captured["cwd"] = kwargs["cwd"]
        raise _SpawnCaptured()

    monkeypatch.setattr(subprocess, "Popen", capture_popen)
    with pytest.raises(_SpawnCaptured):
        parallel_worker.launch_worker(run_dir, 1)

    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get("SF_SUBFRAME_SLOT_CLAIM") == "durable-v1", (
        "launch_worker must opt subframe children into the durable slot claim")
    assert env.get("SF_SUBFRAME_INCARNATION") == "1"
    assert captured["cwd"] == run_dir.resolve()
