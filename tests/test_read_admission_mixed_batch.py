"""A2: a mixed read+shell batch must still seal after the shell result lands.

Admission prices the batch's remaining non-read calls as 192-byte placeholders
on c1c6ad495, so an admitted read can leave the next frame no room for a
bounded shell result.  These scenarios go through real next-frame construction
(build_qualified_request) with the shell placeholder replaced by the largest
result the host's bound permits.
"""
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

CAP = 100_000


def _runtime(tmp_path, monkeypatch, *, lean):
    import kernel_read_admission as admission
    import kernel_wire_truth_gate as truth
    from kernel_provider_dialect import ProviderCapabilities

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
    monkeypatch.setenv("STATEFRAME_LEAN", "on" if lean else "off")
    monkeypatch.setattr("kernel_request_budget.HARD_MAX_REQUEST_BYTES", CAP)
    monkeypatch.setattr(admission, "request_byte_cap", lambda: CAP)
    monkeypatch.setattr(truth, "capture_context", lambda *a, **k: {})
    monkeypatch.setattr("kernel_wire_truth_facts.require_fact_context", lambda value: value)
    monkeypatch.setattr(truth, "enabled", lambda: False)
    monkeypatch.setattr("kernel_provider_dialect.capabilities_for",
                        lambda *a, **k: ProviderCapabilities(dialect="llama.cpp"))
    task = {"role": "user", "content": "Task: inspect source then run the checks"}
    journal = AuditJournal(tmp_path, "read-admission-mixed", {})
    context = build_action_context([task], workspace_root=tmp_path)
    context["journal_receipt_records"] = []
    plan = {"paths": [], "units": [], "source_contents": {}, "revisions": {},
            "workspace_authorization": {"scope": "test", "generation": "test"}}
    rt = SimpleNamespace(owner=SimpleNamespace(root=tmp_path, task_row={}), journal=journal,
        _framefit_action_context=context, _request_source_closure=plan,
        _empirical_frame_sources={}, frame_ceiling=33_344, retained=None, step_ledger=None,
        _frame_budget_request_count=1, binding=lambda: {"product_revision": "fixture"},
        _request_token_meter=RequestTokenMeter(disabled=True))
    capture = RequestCapture(journal, 1, "snapshot")
    capture._framefit5_task_message = task
    capture._request_source_closure_override = plan
    rt.prepared_binding = rt.binding()
    rt.current_request_id = capture.request_id
    rt.owner.current_request_id = capture.request_id
    rt._active_capture = capture
    tools = [{"type": "function", "function": {"name": name, "description": name,
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": [], "additionalProperties": False}}}
        for name in ("read", "shell")]
    frame = build_typed_state([task], context, rt)
    body = {"model": "local", "messages": [{"role": "system", "content": "rules"}, task,
        {"role": "user", "content": _serialize_state_message({"state": {"framefit5": frame}})},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"}], "tools": tools, "max_tokens": 100}
    capture.request_action_contract = establish_lean_request_action_contract(
        rt, body["messages"], tools, request_id=capture.request_id)
    from kernel_request_seal import OwnerPresentation
    capture.owner_presentation = OwnerPresentation.capture(rt, capture.request_action_contract, body["messages"])
    admission.note_sealed_request(rt, body["messages"], tools)
    return rt, capture, body, plan


def _read_then_shell(tmp_path, rt, body, plan, *, shell_result):
    """Admit one read with a shell still pending, then seal the next frame."""
    from kernel_read_exec import execute_read
    from kernel_read_receipts import begin_read_attempt
    from kernel_request_evidence_closure import SpanNeed

    content = ("# " + "x" * 98 + "\n") * 400          # 40 KB: far above any allowance
    (tmp_path / "big.py").write_text(content)
    calls = [
        {"id": "read-0", "type": "function", "function": {
            "name": "read", "arguments": json.dumps({"path": "big.py", "max_lines": 1000})}},
        {"id": "shell-1", "type": "function", "function": {
            "name": "shell", "arguments": json.dumps({"cmd": "pytest -q"})}},
    ]
    rt.native_batch = SimpleNamespace(pending=calls, origin="mixed-batch")
    args = json.loads(calls[0]["function"]["arguments"])
    metadata = {}
    scope = begin_read_attempt(rt, args, call_id="read-0")
    result = execute_read(tmp_path, tmp_path / "big.py", "big.py", "big.py", 0, args=args,
                          measurement=rt, step=1, delivery=metadata)
    scope.finish(result[0])
    ledger = getattr(rt, "read_capture_ledger", None)
    for captured in (ledger.all() if ledger is not None else ()):
        identity = captured.identity
        plan["paths"].append(identity.path)
        unit = SpanNeed(identity.path, identity.first_line, identity.last_line,
                        "read", "admitted range", 0).as_dict()
        unit["unit_handle"] = f"admitted:{identity.path}:{identity.first_line}-{identity.last_line}"
        plan["units"].append(unit)
        plan["revisions"][identity.path] = identity.revision
        plan["source_contents"][identity.path] = content
    replies = [{"role": "tool", "tool_call_id": "read-0", "content": result[1]},
               {"role": "tool", "tool_call_id": "shell-1", "content": shell_result}]
    body["messages"][-1:-1] = [{"role": "assistant", "content": None, "tool_calls": calls}, *replies]
    return result, metadata


def test_admitted_read_leaves_room_for_a_bounded_shell_result(tmp_path, monkeypatch):
    from kernel_lean import MAX_TOOL_RESULT_CHARS, bound_tool_result

    rt, capture, body, plan = _runtime(tmp_path, monkeypatch, lean=True)
    # The largest reply the host bound lets through: escaped on the wire at 6 bytes/char.
    shell_result = bound_tool_result("é" * MAX_TOOL_RESULT_CHARS)
    result, metadata = _read_then_shell(tmp_path, rt, body, plan, shell_result=shell_result)
    # The batch allowance already reserves the shell's bounded reply (red on c1c6ad495).
    import kernel_read_admission as admission
    state = admission._state(rt, None)
    assert state.allowance <= CAP - len(json.dumps(shell_result).encode())
    assert result[0] in {"OK", "REFUSED_ADMISSION"}
    if result[0] == "REFUSED_ADMISSION":
        assert metadata["next_cursor"]
    record = build_qualified_request(capture, rt, body, endpoint="http://local/v1", key="")
    assert len(record.wire) <= CAP
    assert verify_certificate(record.wire, capture._working_set_certificate)
    record.validate(capture, rt)
    assert capture.request_source_final_delivery["proof_status"] == "VERIFIED"
    # The admitted page must reach the wire as promised: no degrade, no deferral.
    prepared = next(json.loads(line) for line in capture.journal.path.read_bytes().splitlines()
                    if json.loads(line).get("kind") == "working_set_prepared")
    assert prepared["fits_without_degrade"] is True
    assert capture.request_source_final_delivery.get("deferred_paths", []) == []


def test_unbounded_producer_closes_admission_with_a_cursor(tmp_path, monkeypatch):
    """Without a host bound on the shell reply, admission cannot promise any read."""
    rt, capture, body, plan = _runtime(tmp_path, monkeypatch, lean=False)
    result, metadata = _read_then_shell(tmp_path, rt, body, plan, shell_result="ok\n")
    assert result[0] == "REFUSED_ADMISSION"
    assert metadata["next_cursor"]
    assert metadata["clip_reason"] == "admission"
    import kernel_read_admission as admission
    state = admission._state(rt, None)
    assert state.closed and state.allowance == 0
    assert state.degraded_reason is None  # Policy refusal must not take G3's fallback.


def test_shell_result_alone_fits_the_cap(tmp_path, monkeypatch):
    """Guard the scenario: only the admitted read can tip the frame over the cap."""
    from kernel_lean import MAX_TOOL_RESULT_CHARS, bound_tool_result
    from kernel_working_set_live import pending_result_reserve

    monkeypatch.setenv("STATEFRAME_LEAN", "on")
    reserve = pending_result_reserve({"function": {"name": "shell"}})
    assert reserve == bound_tool_result("é" * MAX_TOOL_RESULT_CHARS)
    assert len(json.dumps(reserve).encode()) < CAP - 20_000


def test_admission_floor_uses_the_live_token_authority(tmp_path, monkeypatch):
    """Both admission verdicts follow the live meter against the offline estimate."""
    from dataclasses import replace
    import kernel_read_admission as admission
    import kernel_working_set_live as live
    from kernel_read_exec import execute_read
    from kernel_read_receipts import begin_read_attempt
    from kernel_source_delivery import deliver_source, normalize_read_request

    for live_accepts in (True, False):
        root = tmp_path / ("accept" if live_accepts else "refuse")
        root.mkdir()
        rt, _, body, _ = _runtime(root, monkeypatch, lean=True)
        calls = []
        limit = 500 if live_accepts else 33_344

        def meter(wire):
            calls.append(wire)
            return (1, 1, "live-test") if live_accepts else (limit + 1, limit + 1, "live-test")

        monkeypatch.setattr(live, "read_admission_measure", lambda runtime: meter)
        rt.native_batch = SimpleNamespace(pending=[], origin="live-token-probe")
        state = admission._state(rt, body["messages"])
        assert state.snapshot is not None
        state.snapshot = replace(state.snapshot, prompt_limit=limit)
        raw = ("live-token-source " + "x" * 4000 + "\n").encode()
        (root / "live.py").write_bytes(raw)
        args = {"path": "live.py", "max_chars": 10_000}
        request = normalize_read_request(args)
        page = deliver_source(raw, "live.py", 0, args=request)
        slot = live.read_admission_slot(page, path="live.py", arguments=request.arguments, ordinal=1)
        offline = live.read_admission_floor(state.snapshot, ((slot, "live-read"),),
                                            require_feasible=True)
        assert (offline is None) == live_accepts
        metadata = {}
        scope = begin_read_attempt(rt, args, call_id="live-read")
        result = execute_read(root, root / "live.py", "live.py", "live.py", 0,
            args=args, measurement=rt, step=1, delivery=metadata)
        scope.finish(result[0])
        assert calls and any(b"live-token-source" in wire for wire in calls)
        assert result[0] == ("OK" if live_accepts else "REFUSED_ADMISSION")
        if live_accepts:
            assert raw.decode() in result[1]
            assert state.admitted_count == 1 and len(state.pages) == 1
        else:
            assert metadata["next_cursor"] and not state.pages
            assert not hasattr(rt, "read_capture_ledger")


def test_pending_bytes_do_not_survive_a_resume(tmp_path, monkeypatch):
    """S11: a sealed request (and a restored request clock) ends the admission batch."""
    import kernel_read_admission as admission
    from kernel_continuation_restart import restore_provider_request_count

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 1_000)
    rt = SimpleNamespace(_frame_budget_request_count=4, native_batch=None, journal=None,
                         working_set_fixed_bytes=0)
    admission.spend(rt, None, "x" * 600, path="a.txt")
    assert admission.remaining_allowance(rt, None) == 400
    # Same runtime object, same restored clock: the next sealed request must start clean.
    restore_provider_request_count(rt, {"provider_request_count": 4})
    admission.note_sealed_request(rt, [{"role": "user", "content": "task"}], [])
    assert admission.remaining_allowance(rt, None) == 1_000
    # A fresh runtime after resume carries no admission state at all.
    fresh = SimpleNamespace(native_batch=None, journal=None, working_set_fixed_bytes=0)
    restore_provider_request_count(fresh, {"provider_request_count": 4})
    assert admission.remaining_allowance(fresh, None) == 1_000


def test_unbounded_policy_refusal_is_distinct_from_snapshot_failure(tmp_path, monkeypatch):
    import kernel_working_set_live as live

    rt, capture, body, plan = _runtime(tmp_path, monkeypatch, lean=False)
    rt.native_batch = SimpleNamespace(pending=[{"id": "shell", "type": "function",
        "function": {"name": "shell", "arguments": "{}"}}], origin="unbounded")
    with pytest.raises(live.ReadAdmissionUnbounded):
        live.read_admission_snapshot(rt, body["messages"], byte_cap=CAP)


def test_live_token_regression_detects_disconnected_meter(tmp_path, monkeypatch):
    """The authority regression must fail if admission drops its live meter."""
    import kernel_working_set_live as live

    real_floor = live.read_admission_floor

    def disconnected(snapshot, pages=(), **kwargs):
        kwargs.pop("measure", None)
        return real_floor(snapshot, pages, **kwargs)

    monkeypatch.setattr(live, "read_admission_floor", disconnected)
    with pytest.raises(AssertionError):
        test_admission_floor_uses_the_live_token_authority(tmp_path, monkeypatch)
