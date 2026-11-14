"""Real process/IPC latency gate, without a model server or engine child."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from stateframe.parallel_host import ParallelSupervisor, _Worker
from stateframe.parallel_journal import Journal
from test_parallel_host import _request, _write_request


class FakeWorker:
    def __init__(self):
        self.watchdogs = 0
    def enforce_deadline(self):
        self.watchdogs += 1
    def escalate(self):
        pass
    def poll(self):
        return None


def test_process_integration_sixteen_channels_thirtytwo_workers(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    shim = tmp_path / "busy_artifact_helper.py"
    shim.write_text("import sys, time\n"
        + "sys.path.insert(0, " + repr(str(root / "src")) + ")\n"
        + "from stateframe import parallel_host as host, parallel_seals as seals, parallel_journal as journal\n"
        + "write = host._write_json\n"
        + "def delayed(*args, **kw):\n"
        + "    if kw.get('durable', True): time.sleep(0.025)\n"
        + "    return write(*args, **kw)\n"
        + "host._write_json = delayed\n"
        + "def busy(kind, payload):\n"
        + "    until = time.monotonic() + 0.12\n"
        + "    n = 0\n"
        + "    while time.monotonic() < until: n = (n + 1) % 1000003\n"
        + "    return {'token': payload['token'], 'staged': {'fake_worker': True}, 'payload': {'completed': True}}\n"
        + "seals.artifact_operation = busy\n"
        + "journal.serve()\n")
    processes = []
    def spawn(journal):
        journal.epoch += 1
        process = subprocess.Popen([sys.executable, "-u", str(shim)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        processes.append(process)
        return process
    monkeypatch.setattr(Journal, "_spawn", spawn)
    run = tmp_path / "run"
    supervisor = ParallelSupervisor(run)
    # These workers have no OS descendants. The integration gate exercises the real
    # transport/reactors and scheduling; containment/artifact authority have separate tests.
    handles, answers, workers = [], [], []
    for channel in range(16):
        call_id = f"c{channel:02}"
        call = run / "parallel" / call_id
        for unit_id in ("a", "b"):
            handle = FakeWorker()
            handles.append(handle)
            unit = call / "units" / unit_id
            worker = _Worker(call_id, unit_id, "fake", 1, call, unit / "spec.json", "fake",
                             unit, call / "snapshot", "fake", {}, state="RUNNING", handle=handle)
            worker.status_view = supervisor._status_value(worker)
            supervisor._workers[(call_id, unit_id, 1)] = worker
            workers.append(worker)
        rq = call / "attempt-1/rq/001.json"
        _write_request(rq, _request(1, call_id, 1, "status", {"unit_id": "a"}))
        answers.append(rq.parent.parent / "an/001.json")
    monkeypatch.setattr(supervisor, "_notice_running", lambda worker: None)
    helpers = [Journal(role="seal"), Journal(role="seal")]
    supervisor._seal_helpers = helpers
    tickets = [(helpers[n % 2], helpers[n % 2].submit(kind, {"token": [n, 0, "integration"]}))
               for n in range(32) for kind in ("seal_build", "seal_publish")]
    latencies, heartbeat_gaps = [], []
    initial_load = os.getloadavg()
    initial_pressure = Path("/proc/pressure/io").read_text() if sys.platform == "linux" else "unavailable"
    started = previous = time.monotonic()
    remaining = list(tickets)
    try:
        # Channel publication and worker commits use independent journals; keep both real
        # transports under test while the seal helpers consume CPU.
        supervisor.tick()
        for worker in workers:
            supervisor._persist_worker(worker)
        while time.monotonic() - started < 90:
            entered = time.monotonic()
            heartbeat_gaps.append(entered - previous)
            supervisor.tick()
            returned = time.monotonic()
            latencies.append(returned - entered)
            previous = entered
            pending = []
            for helper, ticket in remaining:
                result = helper.take(ticket)
                if result is None:
                    pending.append((helper, ticket))
                else:
                    assert result.get("ok") is True and result.get("token") == ticket.payload["token"], result
            remaining = pending
            committed = all(worker.commit_intent is None for worker in workers)
            if not remaining and committed and all(path.is_file() for path in answers) and len(latencies) >= 40:
                break
            time.sleep(max(0, 0.1 - (time.monotonic() - entered)))
        assert not remaining, "busy seal helpers failed to complete"
        assert all(worker.commit_intent is None for worker in workers), "delayed worker commits failed to ACK"
        assert all(worker.state_path.is_file() for worker in workers)
        assert all(path.is_file() for path in answers), "delayed channel publication failed to complete"
        assert all(json.loads(path.read_text())["status"] == "OK" for path in answers)
        assert all(handle.watchdogs >= 3 for handle in handles)
        metrics = {"samples": len(latencies), "max_seconds": max(latencies),
                   "p99_seconds": sorted(latencies)[math.ceil(len(latencies) * 0.99) - 1],
                   "heartbeat_max_seconds": max(heartbeat_gaps), "initial_load": initial_load,
                   "final_load": os.getloadavg(), "initial_io_pressure": initial_pressure,
                   "processes": len(processes), "channels": 16, "workers": 32}
        (tmp_path / "tick-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print("BRIDGE_TICK_METRICS=" + json.dumps(metrics, sort_keys=True))
        assert len(processes) == 4  # two journals plus two CPU-busy seal processes
        assert supervisor._journal._process.pid != supervisor._channel_writes.journal._process.pid
        assert metrics["max_seconds"] <= 1.0, metrics
    finally:
        supervisor._stop_runtime()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                command = Path(f"/proc/{process.pid}/cmdline").read_bytes()
                assert str(shim).encode() in command
                process.terminate()
                process.wait(timeout=10)


def test_writer_process_restart_replays_exit_commit_without_duplicate_sequence(tmp_path, monkeypatch):
    from test_parallel_host import _spec_and_snapshot, _Worker
    root = Path(__file__).resolve().parents[1]
    call, unit, snapshot, spec_path, spec = _spec_and_snapshot(tmp_path / "run")
    supervisor = ParallelSupervisor(tmp_path / "run")
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), unit,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    marker = tmp_path / "committed-before-helper-loss"
    shim = tmp_path / "crash_after_commit.py"
    shim.write_text("import os, sys\nfrom pathlib import Path\n"
        + "sys.path.insert(0, " + repr(str(root / "src")) + ")\n"
        + "from stateframe import parallel_journal as journal\n"
        + "commit = journal.commit_worker\n"
        + "marker = Path(" + repr(str(marker)) + ")\n"
        + "def interrupted(payload, counters):\n"
        + "    result = commit(payload, counters)\n"
        + "    if not marker.exists():\n"
        + "        marker.write_text('durable; ACK not sent')\n"
        + "        os._exit(71)\n"
        + "    return result\n"
        + "journal.commit_worker = interrupted\n"
        + "journal.serve()\n")
    processes = []
    def spawn(journal):
        journal.epoch += 1
        process = subprocess.Popen([sys.executable, "-u", str(shim)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        processes.append(process)
        return process
    monkeypatch.setattr(Journal, "_spawn", spawn)
    helper = Journal()
    try:
        ticket = helper.submit("commit", {"path": str(worker.state_path),
                                         "value": supervisor._worker_state_value(worker), "launch": None})
        deadline = time.monotonic() + 30
        while not ticket.done.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        result = helper.take(ticket)
        assert result is not None and result["ok"] is True, result
        assert len(processes) == 2 and helper.epoch >= 3
        assert result["value"]["exited_seq"] == 1
        assert json.loads((call / "host/exited-sequence.json").read_text()) == {"last": 1}
        assert json.loads(worker.state_path.read_text())["exited_seq"] == 1
    finally:
        helper.close()
        for process in processes:
            process.wait(timeout=10)


import pytest


@pytest.mark.parametrize("mutation", ["valid", "recipe", "policy", "product", "evidence"])
def test_seal_process_preserves_receipt_bindings(tmp_path, mutation):
    from dataclasses import replace
    import hashlib
    from test_parallel_host import _spec_and_snapshot, _Worker, _record_submission
    from stateframe.engine.kernel_parallel_records import CheckReceipt, CheckRecipe, UnitContract, UnitSpec, Proposal
    from stateframe.engine.kernel_effects import scan
    from stateframe.engine.kernel_unit_policy import _check_ignored, _manifest_digest
    call, unit, snapshot, spec_path, old = _spec_and_snapshot(tmp_path / "run", include_test=True)
    check = CheckRecipe(("tests/test_a.py",), ("tests/test_a.py",), "pytest")
    contract = UnitContract("a", "update value", ("pkg/a.py",), ("pkg/new.py",), (), (), (), (), check)
    spec = UnitSpec("call", "a", "execution", "seed", contract, (), (),
                    ("pkg/a.py", "pkg/new.py", "tests/test_a.py"), (), "", 3, 3,
                    old.start_manifest_digest, old.policy_digest, old.limits)
    spec_path.write_bytes(__import__("stateframe.engine.kernel_parallel_records", fromlist=["canonical_json"]).canonical_json(spec.to_json()) + b"\n")
    workspace = unit / "workspace"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "pkg/a.py").write_bytes(b"VALUE = 2\n")
    (workspace / "pkg/new.py").write_bytes(b"NEW = True\n")
    (workspace / "tests/test_a.py").write_text("def test_a(): assert True\n")
    _record_submission(unit, {"pkg/a.py": "VALUE = 2\n", "pkg/new.py": "NEW = True\n"})
    raw = b"check exit 0; 1 passed\n"
    evidence = unit / "checks" / (hashlib.sha256(raw).hexdigest() + ".txt")
    evidence.parent.mkdir()
    evidence.write_bytes(raw)
    receipt = CheckReceipt(check.digest(), _manifest_digest(scan(workspace, exclude=_check_ignored)),
                           "a" * 64, "b" * 64, spec.policy_digest, "PASS", None, (), str(evidence), "1 passed")
    if mutation == "recipe":
        receipt = replace(receipt, recipe_digest="f" * 64)
    elif mutation == "policy":
        receipt = replace(receipt, policy_digest="f" * 64)
    elif mutation == "product":
        receipt = replace(receipt, product_manifest_digest="f" * 64)
    elif mutation == "evidence":
        evidence.write_bytes(b"changed evidence")
    terminal_path = unit / "engine-result.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["result"]["check_receipt"] = receipt.to_json()
    terminal_path.write_text(json.dumps(terminal))
    supervisor = ParallelSupervisor(tmp_path / "run")
    worker = _Worker("call", "a", "execution", 1, call, spec_path, spec.digest(), unit,
                     snapshot, spec.start_manifest_digest, {}, state="EXITED", terminal="SUBMITTED")
    supervisor._persist_worker(worker)
    request = _request(1, "call", 1, "seal", {"unit_id": "a", "incarnation": 1})
    helper = Journal(role="seal")
    process = None
    try:
        result = helper.call("seal_build", {"request": request.to_json(), "call_root": str(call),
                                           "stage": str(call / "host/seals/receipt.stage.json"), "token": [1, 0, "receipt"]})
        process = helper._process
        proposal = Proposal.from_json(json.loads(Path(result["staged"]["stage_path"]).read_text()))
        assert proposal.check_receipt == (None if mutation == "product" else receipt)
        assert mutation in {"valid", "product"}, "invalid check binding was accepted"
    except RuntimeError as error:
        assert mutation in {"recipe", "policy", "evidence"}, error
        assert mutation in str(error).lower(), error
        assert not (call / "proposals/a/proposal.json").exists()
        process = helper._process
    finally:
        helper.close()
        if process is not None:
            process.wait(timeout=10)
