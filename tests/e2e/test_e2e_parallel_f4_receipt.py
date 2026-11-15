"""The next shipped-profile wire request carries actionable subframe outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import signal

import pytest

from _journal import measurement_events
from barrier_endpoint import BarrierEndpoint
from e2e_support import (assert_shipped_wire, copy_example, create_engine_venv,
                         install_startup_hook,
                         parent_wire_files, read_json, result_records, run_cli, start_cli,
                         start_resume, interrupt_run)
from process_signals import signal_run_owned
from stateframe.engine.kernel_tool_tape import ToolTape


pytestmark = pytest.mark.host_sandbox
SCRIPT = Path(__file__).resolve().parents[2] / "examples/parallel_fixes.script.json"


def _script() -> dict:
    return json.loads(SCRIPT.read_bytes())


def _await_parent(endpoint: BarrierEndpoint, process: subprocess.Popen[str],
                  count: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        if endpoint.wait_for_requests("parent", count, timeout=min(5.0, deadline - time.monotonic())):
            return True
    return False


def _oversized_refused_plan(script: dict, repo: Path) -> None:
    units = script["parent"][0]["arguments"]["subframes"]
    shared = f"pricing/shared_{'x' * 180}.py"
    for index, unit in enumerate(units):
        for number in range(5):
            path = shared if index < 2 and number == 0 else (
                f"pricing/{unit['id']}_{number}_{'x' * 180}.py")
            (repo / path).write_text("VALUE = 1\n")
            unit["owns"].append(path)


def _next_wire(endpoint: BarrierEndpoint, python: Path, repo: Path, run_dir: Path) -> Path:
    with endpoint:
        process = start_cli(python, endpoint.url, repo, run_dir,
                            "Fix pricing calculations with subframes.")
        try:
            assert _await_parent(endpoint, process, 2, 600), (
                f"second parent request did not arrive; exit={process.poll()}; errors={endpoint.errors}")
            next_wire = run_dir / "wire/0002.request.json"
            deadline = time.monotonic() + 10
            while not next_wire.is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            paths = parent_wire_files(run_dir)
            assert len(paths) >= 2
            assert read_json(run_dir / "spec.json")["profile"] == "r2hc3"
            assert_shipped_wire(paths[:2])
            return paths[1]
        finally:
            if process.poll() is None:
                signal_run_owned(run_dir, process, signal.SIGTERM,
                                 deadline=time.monotonic() + 30, what="receipt request cleanup",
                                 require_checkpoint_saved=False)
            endpoint.release_parent_request()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                signal_run_owned(run_dir, process, signal.SIGKILL,
                                 deadline=time.monotonic() + 30, what="receipt request escalation",
                                 require_checkpoint_saved=False)
                process.wait(timeout=10)


def _parallel_disposition(request: Path) -> dict:
    wire = read_json(request)
    tools = wire.get("tools")
    assert isinstance(tools, list)
    descriptions = [tool["function"]["description"] for tool in tools
                    if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
                    and tool["function"].get("name") == "read_check_receipt"]
    assert len(descriptions) == 1
    assert "subframe" in descriptions[0] and "next_handle" in descriptions[0]
    dispositions = []
    for payload in assert_shipped_wire([request]):
        state = payload.get("state")
        frame = state.get("framefit5") if isinstance(state, dict) else None
        ledger = frame.get("action_ledger") if isinstance(frame, dict) else None
        disposition = ledger.get("latest_parallel_disposition") if isinstance(ledger, dict) else None
        if isinstance(disposition, dict):
            dispositions.append(disposition)
    assert len(dispositions) == 1
    disposition, = dispositions
    assert disposition["schema"] == "stateframe-parallel-disposition/1"
    assert disposition["kind"] == "parallel_disposition"
    sections = disposition.get("sections")
    assert isinstance(sections, dict)
    section = sections.get("disposition")
    assert isinstance(section, dict)
    assert section["unit_fields"] == ["id", "status"]
    assert section["unit_details_fields"] == [
        "id", "reason", "changed_files", "contract_ok", "check_status", "requests"]
    assert isinstance(section["units"], list)
    assert isinstance(section["result_summary"], str) and section["result_summary"]
    assert len(section["result_summary"].encode("utf-8")) <= 3072
    return disposition


def _assert_no_change_failed_check(run_dir: Path, returncode: int) -> None:
    summary = read_json(run_dir / "summary.json")
    assert returncode == 1
    expected = {
        "engine_result": "SUBMITTED",
        "child_exit_code": 0,
        "runtime_outcome": "completed",
        "check_outcome": "fail",
        "artifact": "saved",
        "exit_code": 1,
        "succeeded": False,
        "files_changed": [],
        "product_files_changed": [],
        "test_files_changed": [],
    }
    assert {key: summary[key] for key in expected} == expected


def test_failed_and_cancelled_subframes_reach_parent_before_successes(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    script = _script()
    script["subframes"]["tax"] = [{"tool": "subframe_blocked", "arguments": {
        "reason_code": "INTERFACE_WRONG", "detail": "tax contract cannot be preserved"}}]
    script["parent"] = [script["parent"][0], {"tool": "read", "arguments": {
        "path": "pricing/tax.py"}}]
    repo = copy_example(tmp_path)
    run_dir = tmp_path / "run"
    disposition = _parallel_disposition(_next_wire(
        BarrierEndpoint(script["parent"], script["subframes"], hold_parent_request=2),
        isolated_engine_python, repo, run_dir))
    receipt = disposition["sections"]["disposition"]
    result, = result_records(run_dir).values()
    statuses = {item.unit_id: item.status for item in result.outcomes}
    assert statuses["tax"] == "BLOCKED"
    assert statuses["checkout"] == "CANCELLED"
    assert statuses["discount"] == "COMMITTED"
    tax = next(item for item in result.outcomes if item.unit_id == "tax")
    assert (tax.reason, tax.detail) == ("INTERFACE_WRONG", "tax contract cannot be preserved")
    summary = receipt["result_summary"]
    assert "Plan tax: owns=pricing/tax.py; check=tests/test_tax.py; consumes=-; status=BLOCKED" in summary
    assert "Plan checkout: owns=pricing/checkout.py; check=tests/test_checkout.py; consumes=tax.calculate_tax; status=CANCELLED" in summary
    assert "INTERFACE_WRONG: tax contract cannot be preserved" in summary
    assert "PRODUCER_FAILED" in summary
    assert summary.index("Plan tax:") < summary.index("Plan discount:")
    assert summary.index("Plan checkout:") < summary.index("Plan discount:")


def test_six_admitted_long_failures_keep_every_decision_on_shipped_wire(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = copy_example(tmp_path)
    script = _script()
    units = []
    workers = {}
    for number in range(6):
        unit_id = f"bad{number}"
        source = f"pricing/{unit_id}.py"
        check = f"tests/test_{unit_id}.py"
        (repo / source).write_text("VALUE = 1\n")
        (repo / check).write_text(
            f"from pricing.{unit_id} import VALUE\n\ndef test_value():\n    assert VALUE == 1\n")
        units.append({"id": unit_id, "outcome": f"Update {source}", "owns": [source],
                      "creates": [], "exports": [], "consumes": [], "reads": [],
                      "requirements": [], "check": {"tests": [check]}})
        workers[unit_id] = [{"tool": "subframe_blocked", "arguments": {
            "reason_code": "OTHER", "detail": "x" * 300}}]
    script["parent"] = [{"tool": "parallel", "arguments": {"subframes": units}},
                        {"tool": "done", "arguments": {"summary": "Six failures reviewed."}}]
    run_dir = tmp_path / "run"
    with BarrierEndpoint(script["parent"], workers, hold_parent_request=2) as endpoint:
        process = start_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Review six blocked pricing subframes.")
        try:
            assert _await_parent(endpoint, process, 2, 600), (
                f"second parent request did not arrive; exit={process.poll()}; errors={endpoint.errors}")
            request = run_dir / "wire/0002.request.json"
            receipt = _parallel_disposition(request)["sections"]["disposition"]
            result, = result_records(run_dir).values()
            assert result.admission.status == "ADMITTED"
            result_statuses = {item.unit_id: item.status for item in result.outcomes}
            assert result_statuses == {f"bad{number}": "BLOCKED" for number in range(6)}
            assert [len(item.detail or "") for item in result.outcomes] == [300] * 6
            wire_statuses = {row[0]: row[1] for row in receipt["units"]}
            assert wire_statuses == result_statuses
            summary_lines = {
                line.split(":", 1)[0].removeprefix("Plan "): line
                for line in receipt["result_summary"].splitlines()
                if line.startswith("Plan ")
            }
            assert set(summary_lines) == set(result_statuses)
            for number in range(6):
                line = summary_lines[f"bad{number}"]
                assert "status=BLOCKED; check_status=NOT_RUN;" in line
                assert f"reason=OTHER: {'x' * 300}" in line
            endpoint.release_parent_request()
            process.wait(timeout=300)
            _assert_no_change_failed_check(run_dir, process.returncode)
            endpoint.assert_consumed()
        finally:
            if process.poll() is None:
                signal_run_owned(run_dir, process, signal.SIGTERM,
                                 deadline=time.monotonic() + 30, what="receipt pagination cleanup")
            endpoint.release_parent_request()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                raise AssertionError(f"parent CLI did not exit: pid={process.pid}") from None


def test_over_budget_receipt_counts_committed_subframes_on_parent_wire(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    script = _script()
    repo = copy_example(tmp_path)
    for unit in script["parent"][0]["arguments"]["subframes"]:
        for number in range(3):
            extra = f"pricing/{unit['id']}_{number}_{'x' * 180}.py"
            (repo / extra).write_text(f"value_{unit['id']}_{number} = 1\n")
            unit["owns"].append(extra)
    script["parent"] = [script["parent"][0], {"tool": "read", "arguments": {
        "path": "pricing/tax.py"}}]
    run_dir = tmp_path / "run"
    request = _next_wire(
        BarrierEndpoint(script["parent"], script["subframes"], hold_parent_request=2),
        isolated_engine_python, repo, run_dir)
    receipt = _parallel_disposition(request)["sections"]["disposition"]
    result, = result_records(run_dir).values()
    assert all(item.status == "COMMITTED" for item in result.outcomes)
    assert receipt["omitted_committed"] > 0
    assert f"{receipt['omitted_committed']} more committed" in receipt["result_summary"]
    assert "Subframe results: all committed" in receipt["result_summary"]
    wire_bytes = request.read_bytes()
    wire_sha256 = "sha256:" + hashlib.sha256(wire_bytes).hexdigest()
    bound_receipts = [event for event in measurement_events(run_dir)
                      if event.get("kind") == "request_budget_final_wire"
                      and event.get("body_sha256") == wire_sha256
                      and event.get("status") == "READY"]
    assert bound_receipts, f"no ready tokenizer receipt is bound to {request}"
    bound_receipt = bound_receipts[-1]
    measurement = bound_receipt.get("measurement")
    assert isinstance(measurement, dict)
    assert measurement["body_sha256"] == wire_sha256
    assert measurement["request_bytes"] == len(wire_bytes)
    assert measurement["request_id"] == bound_receipt["request_id"]
    assert measurement["method"] == bound_receipt["measure_method"] == "tokenize"
    assert measurement["tokenizer_identity"]
    tokens = measurement["prompt_tokens"]
    prompt_budget = measurement["max_prompt_tokens"]
    assert bound_receipt["measured_prompt_tokens"] == tokens
    assert bound_receipt["max_prompt_tokens"] == prompt_budget
    assert type(tokens) is int and type(prompt_budget) is int
    assert prompt_budget == 33_344
    assert 0 <= tokens <= prompt_budget


def test_refused_terse_plan_resend_fragment_reaches_parent_wire(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    script = _script()
    full = script["parent"][0]["arguments"]["subframes"]
    script["parent"][0]["arguments"]["subframes"] = [
        {"id": unit["id"], "goal": unit["outcome"], "owns": unit["owns"]}
        for unit in full]
    script["parent"] = [script["parent"][0], {"tool": "read", "arguments": {
        "path": "pricing/tax.py"}}]
    repo = copy_example(tmp_path)
    (repo / "tests" / "test_tax_alternate.py").write_text(
        "from pricing.tax import calculate_tax\n\n"
        "def test_tax_alternate():\n    assert calculate_tax(100, 5) == 5\n")
    run_dir = tmp_path / "run"
    receipt = _parallel_disposition(_next_wire(
        BarrierEndpoint(script["parent"], {}, hold_parent_request=2),
        isolated_engine_python, repo, run_dir))["sections"]["disposition"]
    result, = result_records(run_dir).values()
    assert result.admission.status == "REFUSED"
    assert result.admission.refusal.reason == "INVALID_ARGUMENTS"
    assert "fix:" in result.observation
    assert result.observation == receipt["result_summary"]


def test_paged_refusal_is_readable_by_parent_on_shipped_wire(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    script = _script()
    repo = copy_example(tmp_path)
    _oversized_refused_plan(script, repo)
    script["parent"] = [script["parent"][0], {"tool": "read_check_receipt",
                                               "arguments": {"handle": "pending"}},
                        {"tool": "done", "arguments": {"summary": "Receipt reviewed."}}]
    run_dir = tmp_path / "run"
    with BarrierEndpoint(script["parent"], {}, hold_parent_request=2, barrier_timeout=300) as endpoint:
        process = start_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Review the refused subframe plan.")
        try:
            assert _await_parent(endpoint, process, 2, 240), (
                f"parent stopped before request 2: exit={process.poll()}; errors={endpoint.errors}")
            request = run_dir / "wire/0002.request.json"
            receipt = _parallel_disposition(request)["sections"]["disposition"]
            summary = receipt["result_summary"]
            assert "OWNERSHIP_OVERLAP" in summary
            match = re.search(r'read_check_receipt\(handle="(subframe:[^"]+)"\)', summary)
            assert match is not None
            endpoint.parent_steps[1]["arguments"] = {"handle": match.group(1)}
            endpoint.release_parent_request()
            assert _await_parent(endpoint, process, 3, 120), (
                f"parent stopped before request 3: exit={process.poll()}; errors={endpoint.errors}")
            third = read_json(run_dir / "wire/0003.request.json")
            tool_messages = [message["content"] for message in third["messages"]
                             if message.get("role") == "tool"]
            page, = [json.loads(message) for message in tool_messages
                     if message.startswith('{"call_id":') and '"page":1' in message]
            assert page["page"] == 1
            assert page["rows"][0]["outcome"] == "REFUSED"
            assert_shipped_wire([request, run_dir / "wire/0003.request.json"])
        finally:
            if process.poll() is None:
                signal_run_owned(run_dir, process, signal.SIGTERM,
                                 deadline=time.monotonic() + 30, what="receipt replay cleanup")
            endpoint.release_parent_request()
            process.wait(timeout=30)


@pytest.mark.parametrize("case", ["partial", "invalid_arguments", "paged_refusal"])
def test_completed_receipt_and_next_request_replay_byte_identically(
        tmp_path: Path, isolated_engine_python: Path, case: str) -> None:
    script = _script()
    repo = copy_example(tmp_path)
    if case == "partial":
        script["subframes"]["tax"] = [{"tool": "subframe_blocked", "arguments": {
            "reason_code": "INTERFACE_WRONG", "detail": "tax contract cannot be preserved"}}]
    elif case == "invalid_arguments":
        script["parent"][0]["arguments"] = {"subframes": [{
            "id": "bad", "outcome": "fix", "owns": ["pricing/tax.py"],
            "unexpected": "refuse this plan"}]}
        script["subframes"] = {}
    else:
        _oversized_refused_plan(script, repo)
        script["subframes"] = {}
    script["parent"] = [script["parent"][0], {"tool": "done", "arguments": {
        "summary": "Subframe result reviewed."}}]
    run_dir = tmp_path / "run"
    with BarrierEndpoint(script["parent"], script["subframes"],
                         hold_parent_request=2, barrier_timeout=300) as endpoint:
        process = start_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Replay the subframe result.")
        try:
            assert _await_parent(endpoint, process, 2, 600), (
                f"parent stopped before request 2: exit={process.poll()}; errors={endpoint.errors}")
            next_wire = run_dir / "wire/0002.request.json"
            before = next_wire.read_bytes()
            before_sends = [item for item in endpoint.requests if item["marker"] == "parent"]
            assert len(before_sends) == 2
            assert before_sends[1]["raw_request"] == before
            disposition_before = _parallel_disposition(next_wire)
            receipt_before = disposition_before["sections"]["disposition"]
            result, = result_records(run_dir).values()
            if case == "partial":
                result_statuses = {item.unit_id: item.status for item in result.outcomes}
                wire_statuses = {row[0]: row[1] for row in receipt_before["units"]}
                assert wire_statuses == result_statuses
                observation_lines = {
                    line.split(":", 1)[0].removeprefix("Subframe "): line
                    for line in result.observation.splitlines()
                    if line.startswith("Subframe ") and not line.startswith("Subframe results:")
                }
                assert set(observation_lines) == set(result_statuses)
                summary_lines = {
                    line.split(":", 1)[0].removeprefix("Plan "): line
                    for line in receipt_before["result_summary"].splitlines()
                    if line.startswith("Plan ")
                }
                assert set(summary_lines) == set(result_statuses)
                for unit_id, status in result_statuses.items():
                    assert f": {status};" in observation_lines[unit_id]
                    assert f"status={status};" in summary_lines[unit_id]
            else:
                refusal = result.admission.refusal
                assert refusal is not None
                admission = receipt_before["admission"]
                assert admission["status"] == "REFUSED"
                assert admission["reason"] == refusal.reason
                assert admission["detail"] == refusal.detail
                assert admission["hint"] == refusal.hint
                assert {row[0]: row[1] for row in receipt_before["units"]} == {
                    unit_id: "NOT_STARTED" for unit_id, _ in receipt_before["units"]}
                if case != "paged_refusal":
                    assert receipt_before["result_summary"] == result.observation
            if case == "paged_refusal":
                assert "read_check_receipt(handle=" in receipt_before["result_summary"]
            interrupt_run(process, run_dir)
        finally:
            if process.poll() is None:
                signal_run_owned(run_dir, process, signal.SIGTERM,
                                 deadline=time.monotonic() + 30, what="receipt replay cleanup")
                process.wait(timeout=30)
            endpoint.release_parent_request()
        resumed = start_resume(isolated_engine_python, endpoint.url, run_dir,
                               capture_output=True)
        stdout, stderr = resumed.communicate(timeout=900)
        if case == "partial":
            # Tax is deliberately blocked and checkout depends on it. A faithful
            # replay submits the three committed fixes with a failing full check.
            assert resumed.returncode == 1, f"resume failed: {stdout}\n{stderr}"
            summary = read_json(run_dir / "summary.json")
            assert (summary["engine_result"], summary["child_exit_code"],
                    summary["runtime_outcome"], summary["check_outcome"]) == (
                "SUBMITTED", 0, "completed", "fail")
            assert summary["files_changed"] == [
                "pricing/discount.py", "pricing/quantity.py", "pricing/rounding.py"]
            assert summary["tests_after"]["passed"] is False
        else:
            _assert_no_change_failed_check(run_dir, resumed.returncode)
        after_sends = [item for item in endpoint.requests if item["marker"] == "parent"]
        assert len(after_sends) >= 3
        assert after_sends[2]["raw_request"] == before
        after_path = [path for path in parent_wire_files(run_dir)
                      if path.name == "0002.request.json"][-1]
        assert after_path.read_bytes() == before
        assert _parallel_disposition(after_path) == disposition_before


def test_durable_receipt_read_outage_is_typed_and_resume_rebuilds_wire(
        tmp_path: Path) -> None:
    python = create_engine_venv(tmp_path / "engine-venv")
    marker = tmp_path / "durable-read-outage"
    marker.write_text("fail the first attempt")
    install_startup_hook(python, "_stateframe_e2e_receipt_outage",
        "import errno\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import stateframe\n"
        "sys.path.insert(0, str(Path(stateframe.__file__).parent / 'engine'))\n"
        "import kernel_parallel\n"
        f"marker = Path({str(marker)!r})\n"
        "original = kernel_parallel._read_receipt_file\n"
        "def read_durable(descriptor, name, limit):\n"
        "    if name == 'result.json' and marker.is_file():\n"
        "        raise OSError(errno.EIO, 'planted durable source outage')\n"
        "    return original(descriptor, name, limit)\n"
        "kernel_parallel._read_receipt_file = read_durable\n")
    script = _script()
    repo = copy_example(tmp_path)
    run_dir = tmp_path / "run"
    with BarrierEndpoint(script["parent"], script["subframes"]) as endpoint:
        first = run_cli(python, endpoint.url, repo, run_dir,
                        "Fix pricing calculations with subframes.")
        assert first.returncode == 2, first.stdout + first.stderr
        terminal = read_json(run_dir / "engine-result.json")["result"]
        summary = read_json(run_dir / "summary.json")
        assert terminal["result"] == "RECOVERY_REQUIRED"
        assert terminal["reason_code"] == "RECOVERY_REQUIRED"
        assert "planted durable source outage" in terminal["error_message"]
        assert summary["runtime_outcome"] == "recovery_required"
        assert (summary["child_exit_code"], summary["exit_code"]) == (1, 2)
        assert (run_dir / "parallel").is_dir()
        assert len(parent_wire_files(run_dir)) == 1

        marker.unlink()
        resumed = start_resume(python, endpoint.url, run_dir, capture_output=True)
        stdout, stderr = resumed.communicate(timeout=900)
        assert resumed.returncode == 0, stdout + stderr
        wires = parent_wire_files(run_dir)
        assert len(wires) >= 2
        assert_shipped_wire(wires)
        receipt = _parallel_disposition(next(path for path in wires
                                             if path.name == "0002.request.json"))
        result, = result_records(run_dir).values()
        assert receipt["call_id"] == result.call_id
        section = receipt["sections"]["disposition"]
        assert section["admission"]["status"] == result.admission.status == "ADMITTED"
        assert {row[0]: row[1] for row in section["units"]} == {
            item.unit_id: item.status for item in result.outcomes}
        assert (section["final_check"], section["integration"]) == (
            result.final_check, result.integration) == ("PASS", "VERIFIED")
        completed = ToolTape(run_dir).lookup(result.call_id)
        assert completed is not None and completed.completed
        assert completed.observation == result.observation
