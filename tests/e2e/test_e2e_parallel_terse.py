"""Shipped-profile wire proof for host-expanded subframe plans."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from barrier_endpoint import BarrierEndpoint
from e2e_support import (assert_shipped_wire, copy_example, parent_event_journals,
                         parent_measurement_events,
                         parallel_call_roots, parent_wire_files, read_json, result_records, run_cli,
                         subframe_directories, subframe_wire_files, wire_message_text)


pytestmark = pytest.mark.host_sandbox


def test_terse_plan_expands_on_shipped_profile_wire(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = copy_example(tmp_path)
    run_dir = tmp_path / "run"
    script_path = Path(__file__).resolve().parents[2] / "examples/parallel_fixes.script.json"
    script = json.loads(script_path.read_bytes())
    full = script["parent"][0]["arguments"]["subframes"]
    script["parent"][0]["arguments"]["subframes"] = [
        {"id": unit["id"], "goal": unit["outcome"], "owns": unit["owns"]}
        for unit in full]
    with BarrierEndpoint(script["parent"], script["subframes"]) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Fix pricing calculations with subframes.", parallel="2", timeout=900)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    assert read_json(run_dir / "spec.json")["profile"] == "r2hc3"
    result, = result_records(run_dir).values()
    assert result.admission.status == "ADMITTED"
    assert [(edge.src_unit, edge.dst_unit) for edge in result.admission.edges] == [("tax", "checkout")]
    children = subframe_directories(run_dir)
    assert set(children) == {unit["id"] for unit in full}
    for unit in full:
        request, = subframe_wire_files(children[unit["id"]])
        assert_shipped_wire([request], task_marker=f"Subframe {unit['id']}:")
        assert f"tests/test_{unit['id']}.py" in wire_message_text(request)
        child_result = read_json(children[unit["id"]] / "engine-result.json")["result"]
        assert child_result["result"] == "SUBMITTED"
        assert child_result["edits"] == 1
        assert child_result["submission"]["source"] == "host"
        assert child_result["submission"]["stored"] is True

    parent_wire = parent_wire_files(run_dir)
    assert_shipped_wire(parent_wire)
    first = read_json(parent_wire[0])
    parallel, = [tool["function"] for tool in first["tools"]
                 if tool["function"]["name"] == "parallel"]
    assert "{id, goal, owns}" in parallel["description"]
    assert "goal" in parallel["parameters"]["properties"]["subframes"]["items"]["properties"]
    assert "Plan tax: owns=pricing/tax.py; check=tests/test_tax.py; consumes=-" in result.observation.splitlines()
    parent_payloads = assert_shipped_wire(parent_wire)
    next_ledger = parent_payloads[1]["state"]["framefit5"]["action_ledger"]
    disposition = next_ledger["latest_parallel_disposition"]
    assert disposition["schema"] == "stateframe-parallel-disposition/1"
    assert disposition["kind"] == "parallel_disposition"
    disposition_section = disposition["sections"]["disposition"]
    next_summary = disposition_section["result_summary"]
    assert "Plan tax: owns=pricing/tax.py; check=tests/test_tax.py; consumes=-" in next_summary
    call_root, = parallel_call_roots(run_dir)
    expanded = read_json(call_root / "intent.json")["request"]["units"]
    assert {unit["id"] for unit in expanded} == {unit["id"] for unit in full}
    for unit in expanded:
        consumes = ",".join(item["export_id"] for item in unit["consumes"]) or "-"
        expected = (f"Plan {unit['id']}: owns={','.join(unit['owns'])}; "
                    f"check={','.join(unit['check']['selectors'])}; consumes={consumes}; "
                    "status=COMMITTED; check_status=PASS")
        assert expected in next_summary
    assert "Counts: committed=5; unverified=0; failed=0; cancelled=0" in next_summary
    parallel_dispositions = []
    for payload in parent_payloads:
        state = payload.get("state")
        frame = state.get("framefit5") if isinstance(state, dict) else None
        ledger = frame.get("action_ledger") if isinstance(frame, dict) else None
        latest = ledger.get("latest_parallel_disposition") if isinstance(ledger, dict) else None
        if isinstance(latest, dict):
            assert latest["schema"] == "stateframe-parallel-disposition/1"
            assert latest["kind"] == "parallel_disposition"
            parallel_dispositions.append(latest["sections"]["disposition"])
    assert any(
        isinstance(section.get("result_summary"), str)
        and "Subframe results: all committed" in section["result_summary"]
        and {tuple(unit) for unit in section["units"]}
        == {(unit["id"], "COMMITTED") for unit in full}
        for section in parallel_dispositions
    )
    journals = parent_event_journals(run_dir)
    assert len(journals) == 1
    events = parent_measurement_events(run_dir)
    expanded = {(event["subframe"], event["field"], event["source"], event["reason_code"])
                for event in events if event.get("kind") == "parallel_plan_completed"
                and event.get("action") == "expanded_field"}
    assert ("tax", "check", "tests", "STATIC_TEST_IMPORT") in expanded
    assert ("tax", "outcome", "goal", "GOAL_SENTENCE") in expanded
