"""End-to-end proofs for shipped-profile subframe execution and integration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from _journal import measurement_events, measurement_journal_paths
from barrier_endpoint import BarrierEndpoint
from e2e_support import (
    assert_shipped_wire,
    assert_subframe_surface,
    copy_example,
    parallel_call_roots,
    parent_event_journals,
    parent_measurement_events,
    parent_wire_files,
    read_json,
    result_records,
    run_cli,
    subframe_directories,
    subframe_wire_files,
    summary_for,
    wire_message_text,
)
from stateframe.engine.kernel_parallel_txn import TxnStore
from stateframe.engine.kernel_request_source_window import (parse_model_source_window,
                                                            verify_model_source_window)


pytestmark = pytest.mark.host_sandbox


def _script() -> dict:
    path = Path(__file__).resolve().parents[2] / "examples" / "parallel_fixes.script.json"
    return json.loads(path.resolve().read_bytes())


def _assert_parallel_schema(request_path: Path) -> None:
    request = read_json(request_path)
    tools = request.get("tools")
    assert isinstance(tools, list)
    parallel = [item["function"] for item in tools
                if isinstance(item, dict) and isinstance(item.get("function"), dict)
                and item["function"].get("name") == "parallel"]
    assert len(parallel) == 1
    schema = parallel[0]["parameters"]
    assert schema["type"] == "object"
    assert schema["required"] == ["subframes"]
    assert schema["properties"]["subframes"]["type"] == "array"
    assert "subframe" in parallel[0]["description"].lower()


def _action_receipts(run_dir: Path, kind: str) -> list[dict]:
    journals = parent_event_journals(run_dir)
    assert len(journals) == 1
    events = parent_measurement_events(run_dir)
    return [event["receipt"] for event in events
            if event.get("kind") == "action_receipt"
            and isinstance(event.get("receipt"), dict)
            and event["receipt"].get("kind") == kind]


def _unit_check_statuses_by_candidate_revision(events: list[dict]) -> list[str]:
    statuses: dict[str, str] = {}
    pending_status: str | None = None
    for event in events:
        if event.get("kind") == "unit_check":
            assert pending_status is None
            pending_status = event["status"]
            assert pending_status in {"FAIL", "PASS"}
        elif event.get("kind") == "test_subprocess_completed" and pending_status is not None:
            binding = event.get("binding")
            assert isinstance(binding, dict)
            revision = binding.get("product_revision")
            assert isinstance(revision, str) and revision
            exit_code = event.get("exit_code")
            assert type(exit_code) is int
            assert pending_status == ("PASS" if exit_code == 0 else "FAIL")
            statuses.setdefault(revision, pending_status)
            pending_status = None
    assert pending_status is None
    return list(statuses.values())


def test_failed_first_subframe_check_requires_second_provider_request(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = copy_example(tmp_path)
    run_dir = tmp_path / "run"
    script = _script()
    script["subframes"]["discount"] = [
        {"tool": "str_replace", "arguments": {
            "path": "pricing/discount.py",
            "old_str": "return round(price * percent / 100, 2)",
            "new_str": "return round(price * (100 - percent) / 10, 2)"}},
        {"tool": "str_replace", "arguments": {
            "path": "pricing/discount.py",
            "old_str": "return round(price * (100 - percent) / 10, 2)",
            "new_str": "return round(price * (100 - percent) / 100, 2)"}},
    ]
    with BarrierEndpoint(script["parent"], script["subframes"]) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Fix discount and quantity using subframes.", parallel="2", timeout=600)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()
    children = subframe_directories(run_dir)
    assert set(children) == {"discount", "quantity", "rounding", "tax", "checkout"}
    discount_requests = subframe_wire_files(children["discount"])
    assert len(discount_requests) == 2
    assert_shipped_wire(discount_requests, task_marker="Subframe discount:")
    for path in discount_requests:
        assert_subframe_surface(path)
    for unit_id in ("quantity", "rounding", "tax", "checkout"):
        requests = subframe_wire_files(children[unit_id])
        assert len(requests) == 1
        assert_shipped_wire(requests, task_marker=f"Subframe {unit_id}:")
    journals = measurement_journal_paths(children["discount"])
    assert len(journals) == 1
    events = measurement_events(children["discount"])
    assert _unit_check_statuses_by_candidate_revision(events) == ["FAIL", "PASS"]
    sends = [json.loads(line) for line in (children["discount"] / "sends.jsonl").read_text().splitlines()]
    assert [send["send"] for send in sends] == [1, 2]
    assert [event["check_receipt"]["status"] for event in events
            if event.get("kind") == "subframe_auto_finished"] == ["PASS"]


def test_shipped_profile_parallel_fixes_overlap_and_publish_parent_receipts(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = copy_example(tmp_path)
    run_dir = tmp_path / "run"
    distant_source = repo / "pricing" / "discount.py"
    distant_source.write_text(
        "".join(f"# stable context line {index}\n" for index in range(85))
        + distant_source.read_text())
    source_by_id = {
        unit_id: (repo / "pricing" / f"{unit_id}.py").read_bytes().decode("utf-8")
        for unit_id in ("discount", "quantity", "rounding", "tax", "checkout")
    }
    source_sentinel = "SOURCE_SENTINEL_SUBFRAME_WITHHELD"
    (repo / "pricing" / "secret.py").write_text(
        '"""' + source_sentinel + "_DOCSTRING\n" +
        "# " + source_sentinel + "_MODULE_HEAD\n" +
        source_sentinel + "_BODY = True\n")
    script = _script()

    with BarrierEndpoint(script["parent"], script["subframes"],
                         hold_until={"discount": "quantity", "quantity": "discount"}) as endpoint:
        completed = run_cli(
            isolated_engine_python, endpoint.url, repo, run_dir,
            "Fix the pricing calculations with independent subframes, then verify checkout.",
            parallel="2", timeout=900,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

        parent_spec = read_json(run_dir / "spec.json")
        assert parent_spec["profile"] == "r2hc3"
        parent_python = Path(parent_spec["capabilities"]["python"]).resolve()
        summary = summary_for(run_dir)
        parallel_summary = summary["parallel"]
        assert parallel_summary["totals"]["subframes"] == 5
        assert parallel_summary["totals"]["requests"] == 5
        recorded_frame_tokens = {item["subframe_id"]: item["initial_frame_tokens"]
                                 for item in parallel_summary["subframes"]}
        assert set(recorded_frame_tokens) == set(source_by_id)
        assert all(type(value) is int and 0 < value <= parent_spec["parallel_limits"]["unit_request_tokens"]
                   for value in recorded_frame_tokens.values())
        assert parallel_summary["totals"]["initial_frame_tokens"] == sum(recorded_frame_tokens.values())
        assert parallel_summary["totals"]["subframes_with_initial_frame_tokens"] == 5
        assert parallel_summary["measured"]["peak_active_subframes"] == 2
        assert parallel_summary["measured"]["max_tick_latency_seconds"] is not None
        assert set(summary["files_changed"]) == {
            "pricing/discount.py", "pricing/quantity.py", "pricing/rounding.py",
            "pricing/tax.py", "pricing/checkout.py",
        }

        results = result_records(run_dir)
        assert len(results) == 1
        result = next(iter(results.values()))
        assert result.admission.status == "ADMITTED"
        assert result.admission.order == ("discount", "quantity", "rounding", "tax", "checkout")
        assert {item.unit_id: item.status for item in result.outcomes} == {
            "discount": "COMMITTED", "quantity": "COMMITTED", "rounding": "COMMITTED",
            "tax": "COMMITTED", "checkout": "COMMITTED",
        }
        assert result.final_check == "PASS"
        assert result.integration == "VERIFIED"
        assert result.txn_order == (1, 2, 3, 4, 5)

        call_root, = parallel_call_roots(run_dir)
        published = [(record, state) for record, state in TxnStore(call_root / "txn").records()
                     if state == "PUBLISHED"]
        assert [record.txn_id for record, _ in published] == [1, 2, 3, 4, 5]
        transactions = {record.unit_id: record for record, _ in published}
        checkout_spec = read_json(call_root / "units" / "checkout" / "unit-spec.json")
        assert checkout_spec["start_generation"] >= transactions["tax"].post_generation
        tax_projection = checkout_spec["interfaces"][0]
        assert tax_projection["source_digest"] == hashlib.sha256(
            (run_dir / "workspace" / "pricing" / "tax.py").read_bytes()).hexdigest()

        children = subframe_directories(run_dir)
        assert set(children) == set(source_by_id)
        request_heads = []
        for unit_id, directory in children.items():
            child_spec = read_json(directory / "spec.json")
            assert child_spec["profile"] == "r2hc3"
            assert Path(child_spec["parallel_engine_python"]).resolve() == parent_python
            child_result = read_json(directory / "engine-result.json")["result"]
            assert child_result["result"] == "SUBMITTED"
            assert child_result["edits"] == 1
            assert child_result["submission"]["schema"] == "stateframe-done-seal/2"
            assert child_result["submission"]["source"] == "host"
            assert child_result["submission"]["stored"] is True
            requests = subframe_wire_files(directory)
            assert len(requests) == 1, f"{unit_id} needed another model request after its first edit"
            assert_shipped_wire(requests, task_marker=f"Subframe {unit_id}:")
            for request_path in requests:
                assert_subframe_surface(request_path)
            first_wire_text = wire_message_text(requests[0])
            owned = source_by_id[unit_id]
            path = f"pricing/{unit_id}.py"
            marker = f"[Current source: {path}]\n"
            assert marker in first_wire_text
            source_block = first_wire_text.split(marker, 1)[1]
            metadata = parse_model_source_window(source_block)
            assert metadata is not None and metadata["status"] == "COMPLETE"
            assert verify_model_source_window(source_block, path=path, source=owned)
            test_path = f"tests/test_{unit_id}.py"
            test_source = (repo / test_path).read_text()
            test_marker = f"[Current source: {test_path}]\n"
            assert test_marker in first_wire_text
            assert verify_model_source_window(
                first_wire_text.split(test_marker, 1)[1],
                path=test_path, source=test_source)
            request_body = read_json(requests[0])
            request_heads.append(json.dumps(
                (request_body["messages"][0], request_body["tools"]),
                separators=(",", ":"), ensure_ascii=False).encode())
            send, = [json.loads(line) for line in (directory / "sends.jsonl").read_text().splitlines()]
            cached_tokens = send["cached_tokens"]
            assert cached_tokens is None or (type(cached_tokens) is int and cached_tokens >= 0)
            journals = measurement_journal_paths(directory)
            assert len(journals) == 1
            auto_finishes = [event for event in measurement_events(directory)
                             if event.get("kind") == "subframe_auto_finished"]
            assert [(event["reason_code"], event["check_receipt"]["status"])
                    for event in auto_finishes] == [("auto_finished_on_green_check", "PASS")]
            if unit_id == "discount":
                assert owned.splitlines().index("# stable context line 84") == 84
                assert any("def " in line for line in owned.splitlines()[85:])
            assert not any(message.get("role") == "tool"
                           for message in read_json(requests[0])["messages"]
                           if isinstance(message, dict)), "the first frame invented a read receipt"
            assert source_sentinel not in "\n".join(wire_message_text(path) for path in requests)
        assert len(set(request_heads)) == 1

        parent_requests = parent_wire_files(run_dir)
        assert_shipped_wire(parent_requests)
        _assert_parallel_schema(parent_requests[0])
        assert any("return round(subtotal * rate / 100, 2)" in wire_message_text(path)
                   for path in parent_requests[1:]), "next parent request did not see the published tax bytes"
        assert (repo / "pricing" / "tax.py").read_text() == source_by_id["tax"]
        candidate = run_dir / "workspace" / "pricing" / "tax.py"
        assert "return round(subtotal * rate / 100, 2)" in candidate.read_text()

        receipts = _action_receipts(run_dir, "shell_effect")
        changed = {item["path"] for receipt in receipts for item in receipt.get("changed_files", [])}
        assert changed == set(summary["files_changed"])
        assert all(receipt["result_class"] == "APPLIED" for receipt in receipts)

        discount_intervals = endpoint.intervals("discount")
        quantity_intervals = endpoint.intervals("quantity")
        assert discount_intervals and quantity_intervals
        discount_start, discount_end = discount_intervals[0]
        quantity_start, quantity_end = quantity_intervals[0]
        assert min(discount_end, quantity_end) - max(discount_start, quantity_start) > 0
        assert endpoint.peak_active_requests >= 2
