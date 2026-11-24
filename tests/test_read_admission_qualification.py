"""Read scenarios through real preparation, closure/product proof and sealing."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from kernel_completion_owner import establish_lean_request_action_contract
from kernel_final_wire_allocation import build_qualified_request
from kernel_measurement import AuditJournal, RequestCapture
from kernel_request_diet import RequestTokenMeter
from kernel_stateframe_action_ledger import build_action_context
from kernel_stateframe_framefit5 import _serialize_state_message, build_typed_state
from kernel_working_set_proof import verify_certificate


def seal_fixture(tmp_path, monkeypatch, *, identity_content=None, wire_truth=False):
    """Real runtime, capture, contract and sealed body up to the read batch."""
    import kernel_read_admission as admission
    import kernel_wire_truth_gate as truth
    from kernel_provider_dialect import ProviderCapabilities

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
    monkeypatch.setattr("kernel_request_budget.HARD_MAX_REQUEST_BYTES", 12_000)
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 12_000)
    if not wire_truth:
        monkeypatch.setattr(truth, "capture_context", lambda *a, **k: {})
        monkeypatch.setattr("kernel_wire_truth_facts.require_fact_context", lambda value: value)
        monkeypatch.setattr(truth, "enabled", lambda: False)
    monkeypatch.setattr("kernel_provider_dialect.capabilities_for",
                        lambda *a, **k: ProviderCapabilities(dialect="llama.cpp"))
    task = {"role": "user", "content": "Task: inspect source"}
    journal = AuditJournal(tmp_path, "read-admission", {})
    context = build_action_context([task], workspace_root=tmp_path)
    context["journal_receipt_records"] = []
    plan = {"paths": [], "units": [], "source_contents": {}, "revisions": {},
            "workspace_authorization": {"scope": "test", "generation": "test"}}
    rt = SimpleNamespace(owner=SimpleNamespace(root=tmp_path, task_row={}), journal=journal,
        _framefit_action_context=context, _request_source_closure=plan,
        _empirical_frame_sources={}, frame_ceiling=33_344, retained=None, step_ledger=None,
        _frame_budget_request_count=1, binding=lambda: {"product_revision": "fixture"},
        _request_token_meter=RequestTokenMeter(disabled=True))
    capture = RequestCapture(journal, 1, "snapshot", identity_content=identity_content)
    capture._framefit5_task_message = task
    capture._request_source_closure_override = plan
    rt.prepared_binding = rt.binding()
    rt.current_request_id = capture.request_id
    rt.owner.current_request_id = capture.request_id
    rt._active_capture = capture
    tools = [{"type": "function", "function": {"name": "read", "description": "Read source",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"], "additionalProperties": False}}}]
    frame = build_typed_state([task], context, rt)
    body = {"model": "local", "messages": [{"role": "system", "content": "rules"}, task,
        {"role": "user", "content": _serialize_state_message({"state": {"framefit5": frame}})},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"}], "tools": tools, "max_tokens": 100}
    capture.request_action_contract = establish_lean_request_action_contract(
        rt, body["messages"], tools, request_id=capture.request_id)
    from kernel_request_seal import OwnerPresentation
    capture.owner_presentation = OwnerPresentation.capture(rt, capture.request_action_contract, body["messages"])
    admission.note_sealed_request(rt, body["messages"], tools)
    return capture, rt, body, plan, journal, tools


def seal_case(tmp_path, monkeypatch, scenario):
    capture, rt, body, plan, journal, tools = seal_fixture(tmp_path, monkeypatch)
    contents = {"giant.py": ("# " + "λ" * 100 + "\n") * 100}
    if scenario == "K2":
        contents = {"first.py": ("# " + "x" * 100 + "\n") * 40,
                    "large.py": "# " + "λ" * 4000 + "\n", "tiny.py": "a\n"}
    calls = [{"id": f"read-{i}", "type": "function", "function": {
        "name": "read", "arguments": json.dumps({"path": path, "max_lines": 1000})}}
        for i, path in enumerate(contents)]
    rt.native_batch = SimpleNamespace(pending=calls, origin="read-batch")
    results, replies = [], []
    for call, (path, content) in zip(calls, contents.items()):
        from kernel_read_exec import execute_read
        from kernel_read_receipts import begin_read_attempt
        (tmp_path / path).write_text(content)
        args = json.loads(call["function"]["arguments"])
        metadata = {}
        scope = begin_read_attempt(rt, args, call_id=call["id"])
        result = execute_read(tmp_path, tmp_path / path, path, path, 0, args=args,
                              measurement=rt, step=1, delivery=metadata)
        scope.finish(result[0])
        results.append((result, metadata))
        replies.append({"role": "tool", "tool_call_id": call["id"], "content": result[1]})
    assert results[0][0][0] == "OK"
    if scenario == "K1":
        assert results[0][1]["next_cursor"]
    else:
        assert [result[0] for result, _ in results] == ["OK", "REFUSED_ADMISSION", "REFUSED_ADMISSION"]
        assert all(metadata["next_cursor"] for _, metadata in results[1:])
    # Bind a real closure to exactly the admitted source ranges. Refused reads
    # have recovery cursors and create no fresh FULL obligation.
    from kernel_request_evidence_closure import SpanNeed
    for captured in rt.read_capture_ledger.all():
        identity = captured.identity
        path = identity.path
        plan["paths"].append(path)
        unit = SpanNeed(path, identity.first_line, identity.last_line,
                        "read", "admitted range", 0).as_dict()
        unit["unit_handle"] = f"admitted:{path}:{identity.first_line}-{identity.last_line}"
        plan["units"].append(unit)
        plan["revisions"][path] = identity.revision
        plan["source_contents"][path] = contents[path]
    body["messages"][-1:-1] = [{"role": "assistant", "content": None, "tool_calls": calls}, *replies]
    record = build_qualified_request(capture, rt, body, endpoint="http://local/v1", key="")
    assert capture.qualified_request is record
    assert record.wire == json.dumps(record.body).encode()
    assert len(record.wire) <= 12_000
    assert verify_certificate(record.wire, capture._working_set_certificate)
    record.validate(capture, rt)
    rows = [json.loads(line) for line in journal.path.read_bytes().splitlines()]
    assert any(row["kind"] == "qualified_request_sealed" and
               row["body_sha256"] == "sha256:" + record.body_digest for row in rows)
    assert capture.framefit5_product_validation["checked_before_provider"]
    assert capture.request_source_final_delivery["proof_status"] == "VERIFIED"

    return record, capture, rt


@pytest.mark.parametrize("scenario", ["K1", "K2"])
def test_reads_reach_seal(tmp_path, monkeypatch, scenario):
    seal_case(tmp_path, monkeypatch, scenario)
