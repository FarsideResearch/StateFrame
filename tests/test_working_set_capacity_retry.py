"""Physical capacity retries must retain the assignment of the answering wire."""
import ast
import copy
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import time
from types import SimpleNamespace

import pytest

import kernel_measurement as measurement
import kernel_working_set_state as ws
from kernel_bounded_http import BoundedTransportError
from kernel_working_set_allocator import DEFERRED, FULL
from kernel_working_set_render import render_assignment
from test_working_set_render import snapshot, source
from test_working_set_state import capture as read_capture


def retry_case(tmp_path, monkeypatch, *, refit=None, journal_type=measurement.AuditJournal,
               runtime_factory=lambda journal: SimpleNamespace(journal=journal)):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_TRANSPORT_BACKOFF_SECONDS", "0")
    monkeypatch.setattr("kernel_wire_truth_gate.enabled", lambda: False)
    monkeypatch.setattr("kernel_wire_truth_gate.capture_context", lambda *a, **k: {})
    monkeypatch.setattr("kernel_wire_truth_facts.require_fact_context", lambda value: value)
    journal = journal_type(tmp_path, "capacity-working-set", {})
    rt = runtime_factory(journal)
    captured = read_capture(payload="x" * 4000 + "\n")
    ws.record_capture(rt, captured)
    obligation, = ws.state_for(rt).active_obligations()
    ws.record_submission(rt, request_id="expose", body_sha256="1" * 64,
                         modes={obligation.obligation_id: "F"})
    ws.record_response(rt, request_id="expose", response_id="exposed",
                      body_sha256="1" * 64, disposition="accepted")
    slot = replace(source(text="x" * 4000 + "\n", state="RESIDENT"),
                   identity=captured.identity, obligation_ids=(obligation.obligation_id,),
                   capture_ids=(captured.capture_id,),
                   recovery_arguments=b'{"path":"source.py","start_line":1,"end_line":1}')
    request = measurement.RequestCapture(journal, 1, "snapshot")
    snap = replace(snapshot(slot), request_id=request.request_id)
    full = render_assignment(snap, {slot.key: FULL})
    smaller = render_assignment(snap, {slot.key: DEFERRED})
    assert len(smaller.wire) < len(full.wire)
    request.completion_owner_runtime = rt
    request._working_set_certificate = full.certificate
    request._capacity_retry_original_body = json.loads(full.wire)
    request._request_measurement = SimpleNamespace(prompt_tokens=2000, output_tokens=100)
    owner_calls = []
    monkeypatch.setattr("kernel_completion_owner.note_request_submitted",
                        lambda *a, **k: owner_calls.append(k))
    monkeypatch.setattr("kernel_request_kind.guard_physical_send", lambda *a, **k: None)
    monkeypatch.setattr("kernel_pilot.ACTIVE_RUNTIME", SimpleNamespace(get=lambda: rt))
    wires = []

    def send(_capture, wire, *a, **k):
        wires.append(wire)
        if len(wires) == 1:
            raise BoundedTransportError(error_type="HTTPError", http_status=400,
                relay_error=None, server_error='{"error":{"code":"context_length_exceeded",'
                '"message":"maximum context length is 32768 tokens"}}')
        return b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'

    def reseal(*args):
        request._working_set_certificate = smaller.certificate
        request._request_measurement = SimpleNamespace(prompt_tokens=1000, output_tokens=100)
        return json.loads(smaller.wire), smaller.wire, args[-1]

    monkeypatch.setattr(measurement, "_request_bytes_for_capture", send)
    monkeypatch.setattr("kernel_capacity_retry.reseal_capacity_retry", refit or reseal)
    return rt, request, full, smaller, wires, owner_calls


def transport(request, full):
    return request._send_transport_with_retries(json.loads(full.wire), full.wire,
        "http://127.0.0.1:1/v1/chat/completions", "", admission=None,
        socket_timeout=2, deadline=time.monotonic() + 20, wall_timeout=2,
        backend_identity={}, request_extensions={}, transport_attempt_limit=2)


def test_capacity_retry_adopts_answering_wire_and_replays(tmp_path, monkeypatch):
    rt, request, full, smaller, wires, owner_calls = retry_case(tmp_path, monkeypatch)
    _, sent, _ = transport(request, full)
    pending = json.loads(json.dumps(ws.state_for(rt).checkpoint()))
    assert ws.WorkingSetState.from_checkpoint(pending).checkpoint() == pending
    ws.observe_response(rt, request, response_id="stale", disposition="accepted",
                        transport_attempt_id="rejected-attempt")
    assert ws.state_for(rt).checkpoint() == pending
    ws.observe_response(rt, request, response_id="answer", disposition="accepted",
                        transport_attempt_id=sent["event_id"])
    assert wires == [full.wire, smaller.wire]
    rows = [json.loads(line) for line in rt.journal.path.read_bytes().splitlines()]
    submissions = [row for row in rows if row["kind"] == "working_set_submission"
                   and row["request_id"] == request.request_id]
    assert [row["body_sha256"] for row in submissions] == [
        hashlib.sha256(wire).hexdigest() for wire in wires]
    assert submissions[-1]["transport_attempt_id"] == sent["event_id"]
    assert not ws.state_for(rt).active_obligations()
    assert ws.state_for(rt).logical_ordinal == 2
    assert len(owner_calls) == 1
    assert ws.reconstruct(rows).checkpoint() == ws.state_for(rt).checkpoint()
    saved = json.loads(json.dumps(ws.state_for(rt).checkpoint()))
    assert ws.WorkingSetState.from_checkpoint(saved).checkpoint() == saved


def test_capacity_retry_suspension_reaches_worker_terminal(tmp_path, monkeypatch):
    import kernel_working_set_stop as stop
    from kernel_continuation_journal import ContinuationJournal
    from kernel_completion_owner import terminal_result
    from test_working_set_stop import _runtime, suspension, _assert_summary_outcome_and_resume_contract

    rt, request, full, _, wires, _ = retry_case(tmp_path, monkeypatch,
        journal_type=ContinuationJournal, runtime_factory=_runtime)

    def refuse(*args):
        stop.suspend(rt, replace(suspension(), logical_request_id=request.request_id))

    monkeypatch.setattr("kernel_capacity_retry.reseal_capacity_retry", refuse)
    try:
        with pytest.raises(stop.WorkingSetSuspended) as caught:
            transport(request, full)
        assert wires == [full.wire]  # A suspended refit never reaches transport.
        root = Path(__file__).resolve().parents[1]
        worker = ast.parse((root / "src/stateframe/engine/probes/react_repair.py").read_text())
        handler = next(n for n in ast.walk(worker) if isinstance(n, ast.ExceptHandler)
            and isinstance(n.type, ast.Name) and n.type.id == "WorkingSetSuspended"
            and any(isinstance(s, ast.Return) for s in n.body))
        fn = ast.FunctionDef(name="handle", args=ast.arguments(posonlyargs=[],
            args=[ast.arg(arg="exc")], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=copy.deepcopy(handler.body), decorator_list=[])
        env = dict(measurement=rt, msgs=[], fn="capacity", step=1,
            finish_native_batch_transcript=lambda _: None,
            _attach_env_receipt=lambda value, _: value)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                     "worker_suspension_handler", "exec"), env)
        result = env["handle"](caught.value)
        assert result["result"] == rt._last_finish_result["result"] == "BLOCKED"
        assert terminal_result(rt, result)["result"] == "BLOCKED"
        child = ast.parse((root / "src/stateframe/_child.py").read_text())
        emit = next(n for n in ast.walk(child) if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "emit" and n.value.args
            and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value == "engine_end")
        ends = []
        exec(compile(ast.Module(body=[emit], type_ignores=[]), "engine_end", "exec"),
             dict(result=result, elapsed=0, emit=lambda kind, **f: ends.append((kind, f))))
        assert ends[0][1]["result"] == "BLOCKED"
        rows = [json.loads(line) for line in rt.journal.path.read_bytes().splitlines()]
        assert any(row["kind"] == "working_set_blocked" for row in rows)
        assert not any(row["kind"] == "capacity_retry_refit_failed" for row in rows)
        _assert_summary_outcome_and_resume_contract(tmp_path, result)
    finally:
        rt.journal.close()
