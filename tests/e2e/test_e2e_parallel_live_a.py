"""Replay the live model's subframe plan through the shipped F4 profile."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import pytest
from barrier_endpoint import BarrierEndpoint
from e2e_support import (
    assert_shipped_wire,
    copy_example,
    crash_hook_arguments,
    install_crash_hook,
    interrupt_run,
    parallel_call_roots,
    parent_event_journals,
    parent_measurement_events,
    parent_wire_files,
    read_json,
    run_cli,
    start_resume,
    subframe_directories,
    subframe_wire_files,
    wire_message_text,
)

from stateframe.engine.kernel_parallel_txn import TxnStore
from stateframe.engine.kernel_tool_tape import ToolTape

pytestmark = pytest.mark.host_sandbox
RECORDED = Path(__file__).parents[1] / "fixtures/parallel_live1_a_args.json"


def test_recorded_live_a_plan_launches_five_subframes_and_commits(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    args = json.loads(RECORDED.read_text())
    repo = copy_example(tmp_path)
    script = json.loads((Path(__file__).parents[2] / "examples/parallel_fixes.script.json").read_text())
    # The first live response requested these five reads; its next response is the
    # exact recorded plan. Optional subframe steps make an accidental launch visible.
    parent_steps = [
        {"tool": "read", "arguments": {"path": f"tests/test_{name}.py"}}
        for name in ("tax", "discount", "quantity", "rounding", "checkout")
    ]
    parent_steps.extend([
        {"tool": "parallel", "arguments": args},
        {"tool": "done", "arguments": {"summary": "All pricing subframes are integrated."}},
    ])
    subframe_steps = {"fix_" + key: steps for key, steps in script["subframes"].items()}
    with BarrierEndpoint(parent_steps, subframe_steps, slots=5) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, tmp_path / "run",
                            "Fix the five pricing modules using subframes.", parallel="5",
                            arguments=("--parallel-max", "5"))
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()
        assert {request["marker"] for request in endpoint.requests} == {"parent", *subframe_steps}

    run_dir = tmp_path / "run"
    roots = [path for path in parallel_call_roots(run_dir) if (path / "result.json").is_file()]
    assert len(roots) == 1
    result = read_json(roots[0] / "result.json")
    assert result["admission"]["status"] == "ADMITTED"
    assert result["admission"]["order"] == [
        "fix_discount", "fix_quantity", "fix_rounding", "fix_tax", "fix_checkout"]
    assert {item["unit_id"]: item["status"] for item in result["outcomes"]} == {
        name: "COMMITTED" for name in subframe_steps}
    assert result["txn_order"] == [1, 2, 3, 4, 5]
    assert result["final_check"] == "PASS"
    assert result["integration"] == "VERIFIED"
    tax_spec = read_json(roots[0] / "units/fix_tax/unit-spec.json")
    captured = tax_spec["contract"]["exports"][0]["signature"]
    assert [(item["name"], item["kind"]) for item in captured["params"]] == [
        ("subtotal", "pos"), ("rate", "pos")]
    assert set(subframe_directories(run_dir)) == set(subframe_steps)
    for name, directory in subframe_directories(run_dir).items():
        assert_shipped_wire(subframe_wire_files(directory), task_marker=f"Subframe {name}:")
        assert read_json(directory / "spec.json")["parallel_engine_python"] == str(isolated_engine_python)
    published = [(record.txn_id, record.unit_id) for record, state in TxnStore(roots[0] / "txn").records()
                 if state == "PUBLISHED"]
    assert [txn_id for txn_id, _ in published] == [1, 2, 3, 4, 5]
    assert {unit_id for _, unit_id in published} == set(subframe_steps)
    journals = parent_event_journals(run_dir)
    assert len(journals) == 1
    events = parent_measurement_events(run_dir)
    assert [(item["status"], item["units"]) for item in events
            if item.get("kind") == "parallel_admission"] == [
        ("ADMITTED", sorted(item["id"] for item in args["subframes"]))]
    assert {item["unit"] for item in events if item.get("kind") == "parallel_unit_started"} == set(subframe_steps)
    assert sorted(item["txn_id"] for item in events
                  if item.get("kind") == "parallel_txn" and item.get("disposition") == "PUBLISHED") == [1, 2, 3, 4, 5]
    assert [(item["final_check"], item["integration"]) for item in events
            if item.get("kind") == "parallel_final"] == [("PASS", "VERIFIED")]
    observation = result["observation"]
    assert "Subframe fix_tax: COMMITTED" in observation
    requests = parent_wire_files(run_dir)
    payloads = assert_shipped_wire(requests)
    ledgers = [payload["state"]["framefit5"]["action_ledger"] for payload in payloads]
    disposition_path, disposition = next(
        (path, ledger.get("latest_parallel_disposition"))
        for path, ledger in zip(requests, ledgers)
        if isinstance(ledger.get("latest_parallel_disposition"), dict)
        and ledger["latest_parallel_disposition"].get("call_id") == result["call_id"]
    )
    assert disposition["schema"] == "stateframe-parallel-disposition/1"
    assert disposition["kind"] == "parallel_disposition"
    section = disposition["sections"]["disposition"]
    assert section["admission"]["status"] == "ADMITTED"
    assert (section["final_check"], section["integration"]) == ("PASS", "VERIFIED")
    assert {tuple(unit) for unit in section["units"]} == {
        (name, "COMMITTED") for name in subframe_steps}
    assert section["unit_fields"] == ["id", "status"]
    assert section["unit_details_fields"] == [
        "id", "reason", "changed_files", "contract_ok", "check_status", "requests",
    ]
    assert "Counts: committed=5; unverified=0; failed=0; cancelled=0" in section["result_summary"]
    assert not any(observation in wire_message_text(path) for path in requests)

    receipt_tools = [tool["function"]["description"]
                     for tool in read_json(disposition_path).get("tools", [])
                     if tool.get("function", {}).get("name") == "read_check_receipt"]
    assert len(receipt_tools) == 1
    assert "subframe" in receipt_tools[0] and "next_handle" in receipt_tools[0]

    call_root, = roots
    details = {row[0]: row[1:] for row in section["unit_details"]}
    assert set(details) == set(subframe_steps)
    paged_rows = None
    handle_match = re.search(
        r'read_check_receipt\(handle="(subframe:[^"]+)"\)', section["result_summary"])
    if handle_match is not None:
        handles = [handle_match.group(1)]
        assert handles[0].split(":")[:3] == ["subframe", result["call_id"], "1"]
        receipt_pages = sorted((call_root / "receipt").glob("page-[0-9][0-9][0-9][0-9].json"))
        assert receipt_pages
        paged_rows = []
        for page_number, page_path in enumerate(receipt_pages, 1):
            page_bytes = page_path.read_bytes()
            assert handles[page_number - 1].rsplit(":", 1)[1] == hashlib.sha256(page_bytes).hexdigest()
            page = read_json(page_path)
            assert page["schema"] == "stateframe-subframe-receipt-page/1"
            assert (page["call_id"], page["page"]) == (result["call_id"], page_number)
            paged_rows.extend(page["rows"])
            if page["next_sha256"] is not None:
                handles.append(f"subframe:{result['call_id']}:{page_number + 1}:{page['next_sha256']}")
        assert len(handles) == len(receipt_pages)
        assert len(paged_rows) == len(subframe_steps)
        assert {row["id"] for row in paged_rows} == set(subframe_steps)
        assert all(row["outcome"] == "COMMITTED" and row["check_status"] == "PASS"
                   for row in paged_rows)
    else:
        assert disposition["truncation"]["truncated"] is False

    for unit in args["subframes"]:
        reason, changed_files, contract_ok, check_status, request_count = details[unit["id"]]
        assert reason is None
        assert changed_files
        assert all(path in unit["owns"] and kind in {"create", "modify"}
                   for path, kind, _added, _removed in changed_files)
        assert (contract_ok, check_status, request_count) == (True, "PASS", 1)
        consumes = [item.get("export_id") or f"{item['path']}::{item['symbol']}"
                    for item in unit["consumes"]]
        expected_row = (
            f"Plan {unit['id']}: owns={','.join(unit['owns'])}; "
            f"check={','.join(unit['check']['tests'])}; consumes={','.join(consumes) or '-'}; "
            "status=COMMITTED; check_status=PASS"
        )
        assert expected_row in section["result_summary"]
        if paged_rows is not None:
            row, = [item for item in paged_rows if item["id"] == unit["id"]]
            assert row["owns"] == unit["owns"]
            assert row["check"] == unit["check"]["tests"]
            assert row["consumes"] == consumes
            assert row["expanded"] == expected_row
    assert any(any(tool.get("function", {}).get("name") == "parallel"
                   and "Acceptance conditions" in json.dumps(tool)
                   for tool in read_json(path).get("tools", [])) for path in requests)


def test_pending_input_refusal_crash_resumes_with_identical_next_wire_request(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = copy_example(tmp_path)
    run_dir = tmp_path / "run"
    marker = tmp_path / "crash-marker"
    crash_phase = "after_input_refusal_intent"
    install_crash_hook(isolated_engine_python)
    malformed = {"subframes": [{"id": "bad", "outcome": "fix", "owns": ["pricing/tax.py"],
                               "creates": [], "unexpected": "refuse after tape intent"}]}
    parent_steps = [
        {"tool": "parallel", "arguments": malformed},
        {"tool": "done", "arguments": {"summary": "The invalid subframe plan was refused."}},
    ]
    with BarrierEndpoint(parent_steps, {}, hold_parent_request=2, barrier_timeout=120) as endpoint:
        first = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                        "Reject the malformed subframe plan and continue.", parallel="2",
                        arguments=crash_hook_arguments(crash_phase, marker))
        assert first.returncode != 0
        assert marker.is_file(), first.stdout + first.stderr
        assert marker.read_text() == crash_phase
        pending, = ToolTape(run_dir).pending()
        assert pending.tool == "parallel"
        assert pending.args == {
            "schema": "stateframe-parallel-input-refusal/1",
            "arguments": malformed,
        }
        assert not pending.completed
        assert not list((run_dir / "parallel").glob("*/result.json"))
        resumed = start_resume(isolated_engine_python, endpoint.url, run_dir)
        try:
            assert endpoint.wait_for_requests("parent", 2, timeout=90)
            next_wire = run_dir / "wire/0002.request.json"
            deadline = time.monotonic() + 10
            while not next_wire.is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            first_next_bytes = next_wire.read_bytes()
            assert_shipped_wire([next_wire])
            assert "REFUSED INVALID_ARGUMENTS" in wire_message_text(next_wire)
            interrupt_run(resumed, run_dir)
        finally:
            if resumed.poll() is None:
                interrupt_run(resumed, run_dir)
                try:
                    resumed.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    raise AssertionError(
                        f"owned resume CLI did not exit after SIGTERM: pid={resumed.pid}") from None
            endpoint.release_parent_request()
        again = start_resume(isolated_engine_python, endpoint.url, run_dir)
        try:
            assert again.wait(timeout=180) == 1
        finally:
            if again.poll() is None:
                interrupt_run(again, run_dir)
                again.wait(timeout=30)
        endpoint.assert_consumed()
    assert ToolTape(run_dir).pending() == ()
    assert (run_dir / "wire/0002.request.json").read_bytes() == first_next_bytes
    summary = read_json(run_dir / "summary.json")
    assert (summary["engine_result"], summary["check_outcome"], summary["files_changed"]) == (
        "SUBMITTED", "fail", [])
