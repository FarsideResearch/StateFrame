"""Recorded live subframe plans through the shipped profile and wire recorder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from barrier_endpoint import BarrierEndpoint
from e2e_support import (assert_shipped_wire, copy_example, parallel_call_roots,
                         parent_measurement_events, parent_wire_files, read_json, run_cli,
                         subframe_directories, subframe_wire_files, tool_step, wire_message_text)


pytestmark = pytest.mark.host_sandbox
FIXTURES = Path(__file__).parents[1] / "fixtures/parallel"
SCRIPT = Path(__file__).parents[2] / "examples/parallel_fixes.script.json"


def _recorded(number: int) -> dict:
    return json.loads((FIXTURES / f"live2_{number:04d}.json").read_bytes())


def _steps() -> dict[str, list[dict]]:
    script = json.loads(SCRIPT.read_bytes())
    return {"fix_" + name: steps for name, steps in script["subframes"].items()}


def test_live2_d_unique_import_links_and_measures_on_shipped_wire(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    args = json.loads((FIXTURES / "live_plans" / "live2_D_0002.json").read_bytes())
    repo = copy_example(tmp_path)
    run_dir = tmp_path / "run"
    parent = [tool_step("parallel", args),
              tool_step("done", {"summary": "The pricing subframes are integrated."})]
    with BarrierEndpoint(parent, _steps(), slots=5) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Fix the five pricing modules with subframes.", max_turns=40)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    root, = parallel_call_roots(run_dir)
    result = read_json(root / "result.json")
    assert result["admission"]["status"] == "ADMITTED"
    assert [(edge["src_unit"], edge["dst_unit"]) for edge in result["admission"]["edges"]] == [
        ("fix_tax", "fix_checkout")]
    assert {item["unit_id"]: item["status"] for item in result["outcomes"]} == {
        name: "COMMITTED" for name in _steps()}
    assert (result["final_check"], result["integration"]) == ("PASS", "VERIFIED")
    walls = [item["wall_seconds"] for item in result["outcomes"]]
    assert all(wall > 0 for wall in walls)
    assert result["measured"]["estimated_serial_seconds"] == round(sum(walls), 3)
    assert result["measured"]["parallel_wall_seconds"] > 0
    assert result["measured"]["speedup"] == round(
        result["measured"]["estimated_serial_seconds"] /
        result["measured"]["parallel_wall_seconds"], 3)
    events = parent_measurement_events(run_dir)
    assert [(row["subframe"], row["producer"], row["reason_code"])
            for row in events if row.get("kind") == "parallel_plan_completed"
            and row.get("action") == "linked_coupling"] == [
                ("fix_checkout", "fix_tax", "UNIQUE_COUPLING_OWNER")]
    children = subframe_directories(run_dir)
    assert set(children) == set(_steps())
    for name, child in children.items():
        assert read_json(child / "spec.json")["parallel_engine_python"] == str(isolated_engine_python)
        assert_shipped_wire(subframe_wire_files(child), task_marker=f"Subframe {name}:")
    checkout_wire = subframe_wire_files(children["fix_checkout"])
    assert any("Consume pricing/tax.py:" in wire_message_text(path) and
               "calculate_tax" in wire_message_text(path) for path in checkout_wire)
    slots = {name: read_json(child / "proc" / "1" / "slot-lease.json")["slot"]
             for name, child in children.items()}
    started = [row for row in events if row.get("kind") == "parallel_unit_started"]
    assert len(started) == 5
    assert {row["unit"]: row["slot"] for row in started} == slots
    assert all(isinstance(slot, int) and slot >= 0 for slot in slots.values())
    wires = parent_wire_files(run_dir)
    assert_shipped_wire(wires)
    assert all("estimated serial" not in wire_message_text(path) and
               "speedup" not in wire_message_text(path) for path in wires)


@pytest.mark.parametrize("number", (2, 3))
def test_recorded_live2_plan_commits_with_adaptive_cap_on_shipped_wire(
        tmp_path: Path, isolated_engine_python: Path, number: int) -> None:
    args = _recorded(number)
    repo = copy_example(tmp_path)
    run_dir = tmp_path / "run"
    parent = [tool_step("parallel", args),
              tool_step("done", {"summary": "The pricing subframes are integrated."})]
    with BarrierEndpoint(parent, _steps(), slots=5) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Fix the five pricing modules with subframes.", max_turns=40)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    root, = parallel_call_roots(run_dir)
    result = read_json(root / "result.json")
    assert result["admission"]["status"] == "ADMITTED"
    assert result["admission"]["limits"]["unit_max_requests"] == 7
    assert result["txn_order"] == [1, 2, 3, 4, 5]
    assert {item["unit_id"]: item["status"] for item in result["outcomes"]} == {
        name: "COMMITTED" for name in _steps()}
    assert (result["final_check"], result["integration"]) == ("PASS", "VERIFIED")
    assert "subframe_request_cap=7" in result["observation"]
    intent = read_json(root / "intent.json")
    assert intent["reservations"] == {name: 7 for name in _steps()}
    assert read_json(run_dir / "spec.json")["profile"] == "r2hc3"
    assert read_json(run_dir / "spec.json")["parallel_subframe_max_requests_explicit"] is False

    events = parent_measurement_events(run_dir)
    assert [(row["status"], row["subframe_request_cap"]) for row in events
            if row.get("kind") == "parallel_admission"] == [("ADMITTED", 7)]
    assert [(row["subframe"], row["field"], row["action"], row["producer"], row["reason_code"])
            for row in events if row.get("kind") == "parallel_plan_completed"
            and row.get("action") == "linked_consume"] == [
                ("fix_checkout", "consumes[0]", "linked_consume", "fix_tax", "UNIQUE_PRODUCER")]
    children = subframe_directories(run_dir)
    assert set(children) == set(_steps())
    for name, child in children.items():
        assert read_json(child / "spec.json")["parallel_engine_python"] == str(isolated_engine_python)
        assert_shipped_wire(subframe_wire_files(child), task_marker=f"Subframe {name}:")
        assert read_json(root / "units" / name / "unit-spec.json")["limits"]["unit_max_requests"] == 7

    wires = parent_wire_files(run_dir)
    assert_shipped_wire(wires)
    assert any("subframe_request_cap=7" in wire_message_text(path) for path in wires)
    assert any(any(tool.get("function", {}).get("name") == "parallel"
                   for tool in read_json(path).get("tools", [])) for path in wires)


def test_recorded_live2_explicit_cap_refuses_on_shipped_wire(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    args = _recorded(3)
    repo = copy_example(tmp_path)
    run_dir = tmp_path / "run"
    parent = [tool_step("parallel", args),
              tool_step("done", {"summary": "The requested subframe cap could not be reserved."})]
    with BarrierEndpoint(parent, {}) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Try the five pricing subframes with a fixed request cap.",
                            max_turns=40,
                            arguments=("--parallel-subframe-max-requests", "12"))
        assert completed.returncode == 1, completed.stdout + completed.stderr
        endpoint.assert_consumed()
    summary = read_json(run_dir / "summary.json")
    assert (summary["engine_result"], summary["check_outcome"], summary["files_changed"]) == (
        "SUBMITTED", "fail", [])
    root, = parallel_call_roots(run_dir)
    result = read_json(root / "result.json")
    assert (result["admission"]["status"], result["admission"]["refusal"]["reason"]) == (
        "REFUSED", "RESOURCES_UNAVAILABLE")
    assert read_json(run_dir / "spec.json")["parallel_subframe_max_requests_explicit"] is True
    assert result["outcomes"] == []
    assert subframe_directories(run_dir) == {}
    wires = parent_wire_files(run_dir)
    assert_shipped_wire(wires)
    assert any("required=60" in wire_message_text(path) and
               "--parallel-subframe-max-requests" in wire_message_text(path) for path in wires)


def test_recorded_live2_third_call_uses_existing_tax_definition_on_shipped_wire(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    args = _recorded(5)
    assert "fix_tax" not in {item["id"] for item in args["subframes"]}
    repo = copy_example(tmp_path)
    (repo / "pricing/tax.py").write_text(
        "def calculate_tax(subtotal, rate):\n    return round(subtotal * rate / 100, 2)\n")
    run_dir = tmp_path / "run"
    parent = [tool_step("parallel", args),
              tool_step("done", {"summary": "The remaining pricing subframes are integrated."})]
    steps = {name: value for name, value in _steps().items() if name != "fix_tax"}
    with BarrierEndpoint(parent, steps) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Check the recorded pricing subframe contract.", max_turns=40)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()
    root, = parallel_call_roots(run_dir)
    result = read_json(root / "result.json")
    assert result["admission"]["status"] == "ADMITTED"
    assert result["admission"]["edges"] == []
    # Of the 40 turns, the parent already spent the one that issued this call and holds a
    # four-request completion reserve, leaving 35. Four subframes can each reserve eight.
    assert result["admission"]["limits"]["unit_max_requests"] == 8
    assert {item["unit_id"]: item["status"] for item in result["outcomes"]} == {
        name: "COMMITTED" for name in steps}
    assert result["txn_order"] == [1, 2, 3, 4]
    assert (result["final_check"], result["integration"]) == ("PASS", "VERIFIED")
    assert set(subframe_directories(run_dir)) == set(steps)
    events = parent_measurement_events(run_dir)
    assert [(row["subframe"], row["field"], row["action"], row["reason_code"])
            for row in events if row.get("kind") == "parallel_plan_completed"
            and row.get("action") == "resolved_existing"] == [
                ("fix_checkout", "consumes[0]", "resolved_existing", "UNIQUE_EXISTING_DEFINITION")]
    for name, child in subframe_directories(run_dir).items():
        assert read_json(child / "spec.json")["parallel_engine_python"] == str(isolated_engine_python)
        assert_shipped_wire(subframe_wire_files(child), task_marker=f"Subframe {name}:")
    checkout_wire = subframe_wire_files(subframe_directories(run_dir)["fix_checkout"])
    assert any("Consume pricing/tax.py:" in wire_message_text(path) and
               "calculate_tax" in wire_message_text(path) for path in checkout_wire)
    wires = parent_wire_files(run_dir)
    assert_shipped_wire(wires)
    assert any("subframe_request_cap=8" in wire_message_text(path) for path in wires)
