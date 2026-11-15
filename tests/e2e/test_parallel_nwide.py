"""Prove N-wide subframe execution from host-recorded real-worker intervals."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from barrier_endpoint import BarrierEndpoint
from e2e_support import copy_example, parallel_call_roots, read_json, run_cli


pytestmark = pytest.mark.host_sandbox

_UNITS = ("discount", "quantity", "rounding")
_ROOT = Path(__file__).resolve().parents[2]


def _scenario() -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    source = json.loads((_ROOT / "examples" / "parallel_fixes.script.json").read_bytes())
    plan = dict(source["parent"][0]["arguments"])
    subframes = {
        item["id"]: item for item in plan["subframes"] if item["id"] in _UNITS
    }
    plan["subframes"] = [subframes[unit_id] for unit_id in _UNITS]
    parent_steps = [
        {"tool": "parallel", "arguments": plan},
        {"tool": "done", "arguments": {"summary": "The three independent pricing fixes are complete."}},
    ]
    scripts = {unit_id: source["subframes"][unit_id] for unit_id in _UNITS}
    return parent_steps, scripts


def _host_records(call_root: Path) -> dict[str, dict[str, object]]:
    paths = sorted((call_root / "host" / "units").glob("*/1.json"))
    assert len(paths) == len(_UNITS), f"expected three host worker records, found {paths}"
    records = {path.parent.name: read_json(path) for path in paths}
    assert set(records) == set(_UNITS)
    for unit_id, record in records.items():
        assert record["state"] == "EXITED", unit_id
        assert record["terminal"] == "SUBMITTED", unit_id
        assert record["exit_code"] == 0, unit_id
        start = record["started_epoch"]
        finish = record["finished_epoch"]
        assert type(start) in (int, float), unit_id
        assert type(finish) in (int, float), unit_id
        assert start < finish, f"worker {unit_id} has an empty or reversed interval: {record}"
    return records


def _peak_overlap(records: dict[str, dict[str, object]]) -> int:
    events: list[tuple[float, int]] = []
    for record in records.values():
        events.append((float(record["started_epoch"]), 1))
        events.append((float(record["finished_epoch"]), -1))
    active = peak = 0
    for _epoch, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


@pytest.mark.parametrize("parallel", [3, 2], ids=["all-three-overlap", "third-waits-for-slot"])
def test_parallel_width_controls_real_worker_overlap(
        tmp_path: Path, parallel: int) -> None:
    parent_steps, subframe_scripts = _scenario()
    repo = copy_example(tmp_path)
    run_dir = tmp_path / "run"
    selected_checks = (
        f"{sys.executable} -m pytest -q "
        "tests/test_discount.py tests/test_quantity.py tests/test_rounding.py"
    )
    first_wave = _UNITS if parallel == 3 else _UNITS[:2]
    overlap_barrier = {
        unit_id: tuple(other for other in first_wave if other != unit_id)
        for unit_id in first_wave
    }

    with BarrierEndpoint(
            parent_steps, subframe_scripts, slots=3,
            hold_until_all=overlap_barrier,
            barrier_timeout=120) as endpoint:
        completed = run_cli(
            Path(sys.executable), endpoint.url, repo, run_dir,
            "Fix the independent discount, quantity, and rounding calculations with subframes.",
            parallel=str(parallel),
            arguments=("--test-cmd", selected_checks, "--parallel-max", str(parallel),
                       "--parallel-max-subframes", "3",
                       "--parallel-check-permits", str(parallel)),
            timeout=900,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    call_roots = parallel_call_roots(run_dir)
    assert len(call_roots) == 1, f"expected one parallel call, found {call_roots}"
    call_root = call_roots[0]
    process_records = sorted(call_root.glob("units/*/proc/1/process.json"))
    assert len(process_records) == len(_UNITS), (
        f"expected three real child launches, found {process_records}")
    assert {path.parents[2].name for path in process_records} == set(_UNITS)

    result = read_json(call_root / "result.json")
    outcomes = result["outcomes"]
    assert isinstance(outcomes, list) and len(outcomes) == len(_UNITS)
    assert {item["unit_id"]: item["status"] for item in outcomes} == {
        unit_id: "COMMITTED" for unit_id in _UNITS
    }

    records = _host_records(call_root)
    assert _peak_overlap(records) == parallel
    if parallel == 3:
        assert max(float(row["started_epoch"]) for row in records.values()) < min(
            float(row["finished_epoch"]) for row in records.values())
    else:
        by_start = sorted(records.items(), key=lambda item: float(item[1]["started_epoch"]))
        first_two = [record for _, record in by_start[:2]]
        third_id, third = by_start[2]
        first_wave_finish = min(float(record["finished_epoch"]) for record in first_two)
        assert max(float(record["started_epoch"]) for record in first_two) < first_wave_finish
        assert float(third["started_epoch"]) >= first_wave_finish, (
            f"queued worker {third_id} started before either first-wave worker finished")
