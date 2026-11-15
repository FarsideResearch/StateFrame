"""The shipped profile scales a bounded offline subframe batch through wire requests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from barrier_endpoint import BarrierEndpoint
from e2e_support import (
    assert_shipped_wire,
    assert_subframe_surface,
    parallel_call_roots,
    parent_wire_files,
    read_json,
    result_records,
    run_cli,
    subframe_directories,
    subframe_wire_files,
    tool_step,
)
from stateframe.engine.kernel_parallel_txn import TxnStore


pytestmark = pytest.mark.host_sandbox


def _contract(unit_id: str) -> dict[str, object]:
    return {
        "id": unit_id,
        "outcome": f"Set the value for {unit_id}.",
        "owns": [f"pkg/{unit_id}.py"],
        "creates": [],
        "exports": [],
        "consumes": [],
        "reads": [],
        "requirements": [],
        "check": {"tests": []},
    }


def _lease_check(python: Path, home: Path, endpoint: str, slots: list[int], tmp_path: Path) -> None:
    script = tmp_path / "assert_released_slots.py"
    script.write_text(
        "import json\n"
        "import sys\n"
        "from stateframe.parallel_resources import lease_slot\n"
        "endpoint = sys.argv[1]\n"
        "slots = json.loads(sys.argv[2])\n"
        "leases = []\n"
        "for slot in slots:\n"
        "    lease = lease_slot(endpoint, [slot])\n"
        "    if lease is None:\n"
        "        raise SystemExit(f'slot {slot} remains leased')\n"
        "    leases.append(lease)\n"
        "for lease in leases:\n"
        "    lease.close()\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["HOME"] = str(home)
    completed = subprocess.run(
        [str(python), "-I", str(script), endpoint, json.dumps(slots)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_thirty_two_subframes_reach_sixteen_concurrent_workers_and_release_every_lease(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg/__init__.py").write_text("")
    (repo / "tests/test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    subframes: dict[str, list[dict[str, object]]] = {}
    contracts = []
    for index in range(32):
        unit_id = f"part{index:02d}"
        path = f"pkg/{unit_id}.py"
        (repo / path).write_text("VALUE = 0\n")
        contracts.append(_contract(unit_id))
        subframes[unit_id] = [
            tool_step("str_replace", {"path": path, "old_str": "VALUE = 0", "new_str": "VALUE = 1"}),
            tool_step("done", {"summary": f"{unit_id} is complete."}),
        ]
    first_batch = tuple(f"part{index:02d}" for index in range(16))
    parent = [
        tool_step("parallel", {"subframes": contracts}),
        tool_step("done", {"summary": "All 32 independent subframes committed."}),
    ]
    run_dir = tmp_path / "run"

    with BarrierEndpoint(parent, subframes, slots=17,
                         hold_until_all={unit_id: first_batch for unit_id in first_batch},
                         barrier_timeout=120) as endpoint:
        completed = run_cli(
            isolated_engine_python, endpoint.url, repo, run_dir,
            "Set all independent values using 32 subframes and at most 16 at a time.",
            parallel="16", arguments=("--parallel-max", "16", "--parallel-max-subframes", "32",
                                       "--parallel-max-paths-total", "32"),
            timeout=1800,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()
        assert endpoint.peak_active_requests >= 16

    run_spec = read_json(run_dir / "spec.json")
    assert run_spec["profile"] == "r2hc3"
    parent_wires = parent_wire_files(run_dir)
    assert_shipped_wire(parent_wires)
    summary = read_json(run_dir / "summary.json")
    measured = summary["parallel"]["measured"]
    assert measured["peak_active_subframes"] == 16
    latency = measured["max_tick_latency_seconds"]
    assert isinstance(latency, (int, float)) and 0 <= latency <= 1.0
    # tick() is a memory-only facade; the gate bounds what clients actually wait on: request-to-answer
    # latency and the channel / lifecycle passes that produce those answers (max, p99 and p99.9).
    host_metrics = read_json(run_dir / "parallel" / "host-metrics.json")
    for series in ("answer_latency_seconds", "channel_pass_seconds", "lifecycle_pass_seconds"):
        stats = host_metrics[series]
        assert type(stats["count"]) is int and stats["count"] > 0, series
        for key in ("max", "p99", "p99_9"):
            value = stats[key]
            assert type(value) in (int, float) and 0 <= value <= 1.0, (series, key, value)
        assert stats["p99_9"] <= stats["max"] and stats["p99"] <= stats["p99_9"]
    phases = {"commits", "workers", "seals", "seal_discovery", "channels"}
    assert set(host_metrics["phase_seconds"]) == set(host_metrics["max_phase_seconds"]) == phases
    for phase in phases:
        latest, maximum = host_metrics["phase_seconds"][phase], host_metrics["max_phase_seconds"][phase]
        assert type(latest) in (int, float) and type(maximum) in (int, float)
        assert 0 <= latest <= maximum <= 1.0, (phase, latest, maximum)
    assert measured["parallel_wall_seconds"] < measured["estimated_serial_seconds"] * 0.75

    result, = result_records(run_dir).values()
    assert len(result.outcomes) == 32
    assert all(item.status == "COMMITTED_UNVERIFIED" for item in result.outcomes)
    assert result.txn_order == tuple(range(1, 33))
    call_root, = parallel_call_roots(run_dir)
    published = [(record, state) for record, state in TxnStore(call_root / "txn").records()
                 if state == "PUBLISHED"]
    assert [record.txn_id for record, _ in published] == list(range(1, 33))
    assert len({record.unit_id for record, _ in published}) == 32

    children = subframe_directories(run_dir)
    assert set(children) == {f"part{index:02d}" for index in range(32)}
    used_slots = []
    for unit_id, directory in children.items():
        child_spec = read_json(directory / "spec.json")
        assert child_spec["profile"] == "r2hc3"
        assert Path(child_spec["parallel_engine_python"]).resolve() == Path(
            run_spec["capabilities"]["python"]).resolve()
        slot_record = read_json(directory / "parallel-slot.json")
        assert type(slot_record["slot"]) is int
        used_slots.append(slot_record["slot"])
        requests = subframe_wire_files(directory)
        assert_shipped_wire(requests, task_marker=f"Subframe {unit_id}:")
        for request in requests:
            assert_subframe_surface(request)
        assert (run_dir / "workspace" / f"pkg/{unit_id}.py").read_text() == "VALUE = 1\n"
    assert len(used_slots) == 32
    used_slots = sorted(set(used_slots))
    assert len(used_slots) == 16
    _lease_check(isolated_engine_python, tmp_path / "home", run_spec["chat_url"], used_slots, tmp_path)
