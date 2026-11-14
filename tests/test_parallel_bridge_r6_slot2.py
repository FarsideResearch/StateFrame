"""Failed slot handoffs must keep the child flock until host ownership is durable."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from stateframe import parallel_resources


@pytest.mark.parametrize("failure", ["proof_lookup", "marker_write"])
def test_failed_handoff_keeps_slot_unavailable_until_claim_or_family_proof(
        tmp_path: Path, monkeypatch, failure: str) -> None:
    run_dir = tmp_path / "worker"
    incarnation = 7
    process_dir = run_dir / "proc" / str(incarnation)
    process_dir.mkdir(parents=True)
    lease_root = tmp_path / "slot-leases"
    coordination_dir = tmp_path / "coordination"
    coordination_dir.mkdir()
    retry_started = coordination_dir / "retry-started"
    release_retry = coordination_dir / "release-retry"
    endpoint = "http://127.0.0.1:8080/v1"
    source_root = Path(parallel_resources.__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), child_env.get("PYTHONPATH", "")) if part)
    child_env["SF_SUBFRAME_INCARNATION"] = str(incarnation)
    child_code = "\n".join((
        "import ast, json, os, signal, sys, time",
        "from pathlib import Path",
        "from types import SimpleNamespace",
        "from stateframe import _child, parallel_resources as resources",
        "from stateframe.parallel_worker import kernel_start_identity",
        "lease_root, retry_started, release_retry, failure = map(Path, sys.argv[1:5])",
        "run_dir = Path.cwd()",
        "incarnation = int(os.environ['SF_SUBFRAME_INCARNATION'])",
        "pid = os.getpid()",
        "identity = {'pid': pid, 'pgid': pid, 'kernel_identity': kernel_start_identity(pid)}",
        "process_dir = run_dir / 'proc' / str(incarnation)",
        "process_dir.joinpath('process.json').write_text(json.dumps({'incarnation': incarnation, **identity}))",
        "original_proof = resources._current_worker_family_proof",
        "original_publish = resources._publish_slot_handoff",
        "calls = 0",
        "def wait_then_succeed():",
        "    global calls",
        "    calls += 1",
        "    if calls == 1: raise OSError('injected proof lookup failure')",
        "    retry_started.touch(exist_ok=True)",
        "    while not release_retry.exists(): time.sleep(0.01)",
        "    return original_proof()",
        "def proof_lookup():",
        "    if failure.name == 'proof_lookup' and not release_retry.exists():",
        "        return wait_then_succeed()",
        "    return original_proof()",
        "def marker_publish(*args):",
        "    global calls",
        "    if failure.name == 'marker_write' and not release_retry.exists():",
        "        calls += 1",
        "        if calls == 1: raise OSError('injected marker write failure')",
        "        retry_started.touch(exist_ok=True)",
        "        while not release_retry.exists(): time.sleep(0.01)",
        "    return original_publish(*args)",
        "resources._current_worker_family_proof = proof_lookup",
        "resources._publish_slot_handoff = marker_publish",
        "resources.LEASE_ROOT = lease_root",
        f"lease = resources.lease_slot({endpoint!r}, [3])",
        "assert lease is not None",
        "_child.emit = lambda *args, **kwargs: None",
        "tree = ast.parse(Path(_child.__file__).read_text(encoding='utf-8'))",
        "main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')",
        "outer = next(node for node in main.body if isinstance(node, ast.Try) and any(",
        "    isinstance(item, ast.Try) for item in node.finalbody))",
        "session = SimpleNamespace(containment_verified=False, _process=object(),",
        "    stop_all=lambda: SimpleNamespace(ok=False, detail='injected cleanup failure'),",
        "    close=lambda: (_ for _ in ()).throw(RuntimeError('containment unknown'))) ",
        "namespace = {'mutation_runtime': None, 'hook_wiring': None, 'session': session,",
        "    'slot_lease': lease, 'result': {'result': 'SUBMITTED'}, 'status': 0,",
        "    'previous_term': signal.getsignal(signal.SIGTERM), 'signal': signal,",
        "    'emit': _child.emit, '_handoff_slot_lease': getattr(_child, '_handoff_slot_lease', None)}",
        "exec(compile(ast.Module(body=outer.finalbody, type_ignores=[]), '<child cleanup>', 'exec'), namespace)",
        "payload = {'identity': identity, 'result': namespace['result'], 'status': namespace['status']}",
        "print(json.dumps(payload), flush=True)",
    ))
    monkeypatch.setattr(parallel_resources, "LEASE_ROOT", lease_root)
    monkeypatch.delenv("SF_SUBFRAME_INCARNATION", raising=False)
    monkeypatch.chdir(run_dir)

    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(lease_root), str(retry_started),
         str(release_retry), failure],
        cwd=run_dir, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while not retry_started.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if child.poll() is not None and not retry_started.exists():
            early_claim = parallel_resources.lease_slot(endpoint, [3])
            if early_claim is not None:
                early_claim.close()
            assert early_claim is None, "a later claimant acquired the slot after handoff failed"
        assert retry_started.exists(), (
            "child exited or failed to retry the slot handoff after the injected error; "
            f"returncode={child.poll()}")
        assert child.poll() is None, "child must retain its flock while retrying handoff"

        competing_lease = parallel_resources.lease_slot(endpoint, [3])
        if competing_lease is not None:
            competing_lease.close()
        assert competing_lease is None, "a later claimant must not acquire a slot with a live owner"

        release_retry.touch()
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, stderr
        identity = json.loads(stdout.strip().splitlines()[-1])["identity"]

        after_exit = parallel_resources.lease_slot(endpoint, [3])
        if after_exit is not None:
            after_exit.close()
        assert after_exit is None, "the durable handoff marker must protect the slot after child exit"

        proof_path = process_dir / "family-exited.json"
        proof_path.write_text(json.dumps(identity), encoding="utf-8")
        deadline = time.monotonic() + 5
        available = None
        while time.monotonic() < deadline:
            available = parallel_resources.lease_slot(endpoint, [3])
            if available is not None:
                break
            time.sleep(0.02)
        assert available is not None, "only matching birth-bound family-exit proof releases the slot"
        available.close()
    finally:
        release_retry.touch(exist_ok=True)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
