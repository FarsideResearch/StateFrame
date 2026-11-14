"""Subframe slot ownership and failed-send replay compatibility regressions."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from stateframe import _child, parallel_resources, replay_tape


def _main_cleanup_finalbody() -> list[ast.stmt]:
    tree = ast.parse(Path(_child.__file__).read_text(encoding="utf-8"))
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    outer = next(node for node in main.body
                 if isinstance(node, ast.Try)
                 and any(isinstance(child, ast.Try) for child in node.finalbody))
    return outer.finalbody


def test_unverified_child_cleanup_hands_slot_ownership_to_host() -> None:
    class Session:
        containment_verified = False

        def stop_all(self):
            return SimpleNamespace(ok=False, detail="owned descendant 5555 remains", survivors=(5555,))

        def close(self):
            raise RuntimeError("containment unknown; survivor 5555")

    released: list[int] = []
    handed_off: list[int] = []
    lease = SimpleNamespace(
        close=lambda: released.append(5555),
        handoff_to_host=lambda: handed_off.append(5555),
    )
    events = []
    namespace = {
        "mutation_runtime": None,
        "hook_wiring": None,
        "session": Session(),
        "slot_lease": lease,
        "result": {"result": "SUBMITTED"},
        "status": 0,
        "previous_term": None,
        "signal": SimpleNamespace(SIGTERM=15, signal=lambda *args: None),
        "emit": lambda *args, **kwargs: events.append((args, kwargs)),
    }

    exec(compile(ast.Module(body=_main_cleanup_finalbody(), type_ignores=[]),
                 "<main cleanup finalbody>", "exec"), namespace)

    assert released == [], "unverified containment must not release the slot lease"
    assert handed_off == [5555], "slot ownership must transfer to the resource host"
    assert namespace["result"] == {"result": "ENGINE_ERROR", "error": "ShellCleanupFailed"}
    assert namespace["status"] == 1


def test_child_exit_keeps_handed_off_slot_unavailable_until_host_proof(
        tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "worker"
    incarnation = 7
    process_dir = run_dir / "proc" / str(incarnation)
    process_dir.mkdir(parents=True)
    lease_root = tmp_path / "slot-leases"
    endpoint = "http://127.0.0.1:8080/v1"
    source_root = Path(parallel_resources.__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), child_env.get("PYTHONPATH", "")) if part)
    child_env["SF_SUBFRAME_INCARNATION"] = str(incarnation)
    child_code = "\n".join((
        "import json, os, sys",
        "from pathlib import Path",
        "import stateframe.parallel_resources as resources",
        "from stateframe.parallel_worker import kernel_start_identity",
        "resources.LEASE_ROOT = Path(sys.argv[1])",
        "run_dir = Path.cwd()",
        "incarnation = int(os.environ['SF_SUBFRAME_INCARNATION'])",
        "pid = os.getpid()",
        "identity = {'pid': pid, 'pgid': pid, 'kernel_identity': kernel_start_identity(pid)}",
        "process_dir = run_dir / 'proc' / str(incarnation)",
        "process_dir.joinpath('process.json').write_text(json.dumps({'incarnation': incarnation, **identity}))",
        f"lease = resources.lease_slot({endpoint!r}, [3])",
        "assert lease is not None",
        "handoff = getattr(lease, 'handoff_to_host', None)",
        "if callable(handoff):",
        "    handoff()",
        "print(json.dumps(identity), flush=True)",
    ))
    monkeypatch.setattr(parallel_resources, "LEASE_ROOT", lease_root)
    monkeypatch.delenv("SF_SUBFRAME_INCARNATION", raising=False)
    monkeypatch.chdir(run_dir)

    child = subprocess.run([sys.executable, "-c", child_code, str(lease_root)], cwd=run_dir,
                           env=child_env, check=False, capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    identity = json.loads(child.stdout.strip())
    proof_path = process_dir / "family-exited.json"

    early_lease = parallel_resources.lease_slot(endpoint, [3])
    if early_lease is not None:
        early_lease.close()
    assert early_lease is None, "child exit must not release its handed-off slot before host proof"

    proof_path.write_text(json.dumps(identity), encoding="utf-8")
    deadline = time.monotonic() + 5
    available = None
    while time.monotonic() < deadline:
        available = parallel_resources.lease_slot(endpoint, [3])
        if available is not None:
            break
        time.sleep(0.02)
    assert available is not None, "a matching host family-exit proof must release the pending slot"
    available.close()


def test_failed_send_rows_include_endpoint_identity() -> None:
    build_record = getattr(_child, "_failed_send_record", None)
    assert callable(build_record), "failed-send records need a single testable construction path"
    endpoint_identity = {"url": "http://127.0.0.1:8080/v1", "model": "worker"}
    row = build_record(1, b'{"model":"worker","messages":[]}', {}, {},
                       [{"status": "error"}], OSError(), endpoint_identity)
    assert row["endpoint_identity"] == endpoint_identity
    assert row["error"] == "OSError"


def test_failed_send_without_historical_endpoint_identity_is_skipped(
        tmp_path: Path, monkeypatch) -> None:
    wire_dir = tmp_path / "wire"
    request = b'{"model":"worker","messages":[]}'
    identity = {"url": "http://127.0.0.1:8080/v1", "model": "worker"}
    files = {
        wire_dir / "0001.request.json": request,
        wire_dir / "0001.endpoint.json": json.dumps(identity).encode(),
        wire_dir / "0001.error.txt": b"OSError\n",
    }
    record = {
        "send": 1,
        "request_sha256": hashlib.sha256(request).hexdigest(),
        "transport_attempts": [{"status": "error"}],
        "error": "OSError",
    }

    with (patch.object(Path, "is_dir", lambda path: path == wire_dir),
          patch.object(Path, "is_file", lambda path: path in files),
          patch.object(Path, "exists", lambda path: path in files),
          patch.object(Path, "is_symlink", lambda path: False),
          patch.object(Path, "read_bytes", lambda path: files[path]),
          patch.object(Path, "glob", lambda path, pattern:
                       [wire_dir / "0001.request.json"] if path == wire_dir and "request" in pattern else []),
          patch.object(replay_tape, "_send_records", lambda path: {1: record})):
        assert replay_tape.build(wire_dir, tmp_path / "sends.jsonl") == []
