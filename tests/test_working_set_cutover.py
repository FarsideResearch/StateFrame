"""Live allocator cutover: the certified wire alone owns source membership."""
from __future__ import annotations

import json
import lzma
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kernel_working_set_stop import MANDATORY_ENVELOPE_REASONS, ReasonCode


def test_previous_mode_survives_acceptance_checkpoint_and_replay(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    from test_working_set_state import runtime, capture, submit, accept
    import kernel_working_set_state as ws
    rt = runtime()
    ws.record_capture(rt, capture())
    submit(rt)
    accept(rt)
    entry, = ws.state_for(rt).active_obligations()
    assert entry.previous_mode == "F"
    submit(rt, "degraded", {entry.obligation_id: "D"})
    accept(rt, "degraded")
    saved = ws.state_for(rt).checkpoint()
    assert saved["active"][0]["previous_mode"] == "D"
    assert ws.WorkingSetState.from_checkpoint(saved).checkpoint() == saved
    assert ws.reconstruct(rt.journal.events).checkpoint() == saved
    with pytest.raises(ValueError, match="promotion|reread"):
        submit(rt, "promotion", {entry.obligation_id: "F"})


def test_certified_failed_send_and_checkpoint_replay_keep_exposure_debt(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    from test_read_receipts import runtime
    from test_working_set_shadow import page_slot
    from kernel_read_receipts import capture_read
    from kernel_working_set_snapshot import freeze_preparation
    from kernel_working_set_prepare import prepare_candidate, PreparedCandidate
    import kernel_working_set_state as ws
    rt, slot = runtime(tmp_path), page_slot()
    capture_read(rt, path=slot.identity.path, metadata={"result_class": "OK",
        "revision": slot.identity.revision, "effective_range": {"start_line": 1, "end_line": 1},
        "byte_start": 0, "byte_end": len(slot.payload.encode())},
        payload=slot.payload, original_page=slot.page, source="disk", arguments={"path": "a.py"})
    snapshot = freeze_preparation({"messages": [{"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "suffix"}]}, rt, request_id="send", byte_cap=82026)
    prepared = prepare_candidate(snapshot)
    assert isinstance(prepared, PreparedCandidate)
    capture = SimpleNamespace(request_id="send", _working_set_certificate=prepared.rendered.certificate)
    ws.observe_submission(rt, capture, prepared.wire)
    ws.observe_response(rt, capture, disposition="failed", response_id="failed")
    entry, = ws.state_for(rt).active_obligations()
    assert entry.state == "DUE" and entry.previous_mode is None
    saved = ws.state_for(rt).checkpoint()
    assert ws.WorkingSetState.from_checkpoint(saved).checkpoint() == saved
    assert ws.reconstruct(rt.journal.events).checkpoint() == saved
    rt.working_set_state = ws.WorkingSetState.from_checkpoint(saved)
    retry = prepare_candidate(freeze_preparation(json.loads(snapshot.fixed_body), rt,
        request_id="retry", byte_cap=82026))
    assert retry.wire == prepared.wire
    capture.request_id = "retry"
    with pytest.raises(ValueError, match="request identity"):
        ws.observe_submission(rt, capture, prepared.wire)
    capture._working_set_certificate = retry.rendered.certificate
    ws.observe_submission(rt, capture, prepared.wire)
    ws.observe_response(rt, capture, disposition="accepted", response_id="accepted")
    entry, = ws.state_for(rt).active_obligations()
    assert entry.state == "RESIDENT" and entry.previous_mode == "F"


def test_carried_slots_never_promote_on_more_capacity():
    from dataclasses import replace
    from test_working_set_render import snapshot, source
    from kernel_working_set_prepare import prepare_candidate
    from kernel_working_set_render import propose_slots
    from kernel_working_set_allocator import PriorityTier, DEFERRED
    frozen = snapshot(source())
    carried = replace(frozen.sources[0], state="RESIDENT", previous_mode="O")
    frozen = replace(frozen, sources=(carried,), byte_cap=100_000)
    slot, = propose_slots(frozen)
    assert slot.priority_tier is PriorityTier.CARRIED
    assert {mode for mode, _ in slot.allowed} == {DEFERRED}
    result = prepare_candidate(frozen)
    assert dict(result.rendered.certificate.assignment)[carried.key] == "DEFERRED"


def test_embedded_head_is_an_allocator_proposal(tmp_path):
    from kernel_stateframe_headcarry import _module_head, _panel
    from kernel_working_set_snapshot import freeze_preparation
    from kernel_working_set_render import render_assignment
    from kernel_working_set_proof import verify_certificate
    from kernel_working_set_allocator import DEFERRED
    raw = b"def important():\n    return 42\n"
    (tmp_path / "a.py").write_bytes(raw)
    head = _module_head(tmp_path, "a.py", raw)
    frame = {"headcarry": _panel([head], omitted_count=0, deduplicated_count=0, byte_budget=6000),
             "owner_protocol": {"admitted_actions": ["read"]}}
    body = {"messages": [{"role": "system", "content": "rules"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps({"state": {"framefit5": frame}})},
        {"role": "user", "content": "suffix"}], "tools": [{"type": "function", "function": {
            "name": "read", "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]}}}]}
    frozen = freeze_preparation(body, SimpleNamespace(owner=SimpleNamespace(root=tmp_path)),
                                request_id="head", byte_cap=82026)
    slot, = frozen.sources
    assert slot.payload == raw.decode()
    deferred = render_assignment(frozen, {slot.key: DEFERRED})
    assert verify_certificate(deferred.wire, deferred.certificate)
    typed = next(json.loads(row["content"].split("\n", 1)[1]) for row in
                 json.loads(deferred.wire)["messages"] if row["content"].startswith("[StateFrame]"))
    assert typed["state"]["framefit5"]["headcarry"]["heads"] == []


def test_embedded_source_cannot_escape_workspace_through_symlink(tmp_path):
    from kernel_stateframe_headcarry import _module_head
    from kernel_working_set_snapshot import freeze_preparation, SnapshotConstructionError
    root = tmp_path / "repo"
    root.mkdir()
    raw = b"KNOWN = 1\n"
    outside = tmp_path / "outside.py"
    outside.write_bytes(raw)
    (root / "alias.py").symlink_to(outside)
    head = _module_head(root, "alias.py", raw)
    body = {"messages": [{"role": "user", "content": "[StateFrame]\n" + json.dumps({
        "state": {"framefit5": {"headcarry": {"heads": [head]}}}})}]}
    with pytest.raises(SnapshotConstructionError, match="SOURCE_SNAPSHOT_UNSTABLE"):
        freeze_preparation(body, SimpleNamespace(owner=SimpleNamespace(root=root)),
                           request_id="confined", byte_cap=82026)


def test_snapshot_preserves_opaque_suffix_and_stateframe_position(monkeypatch):
    from kernel_working_set_snapshot import freeze_preparation
    from kernel_working_set_prepare import prepare_candidate
    suffix = {"role": "user", "content": "[Host fact]\nuser-owned opaque instruction"}
    state = {"role": "user", "content": '[StateFrame]\n{"state":{"framefit5":{}}}'}
    body = {"messages": [{"role": "system", "content": "rules"},
        {"role": "user", "content": "Task: fix"}, state, suffix,
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"}]}
    runtime = SimpleNamespace(_active_capture=SimpleNamespace(_framefit5_newest_user_suffix=(suffix,)))
    prepared = prepare_candidate(freeze_preparation(body, runtime, request_id="suffix", byte_cap=82026))
    messages = json.loads(prepared.wire)["messages"]
    assert messages[2]["content"].startswith("[StateFrame]\n")
    assert messages[-2] == suffix


def test_full_repeat_batch_accounting_never_claims_unchanged_delivery(monkeypatch):
    from kernel_tool_call_loop_breaker import UnboundedProgressWatch
    row = {"kind": "read", "result_class": "OK", "delivery": {
        "working_set_repeat_capture": True, "unchanged_since_request": False,
        "requested_bytes_delivered": 100, "effective_range": {"start_line": 1, "end_line": 3}}}
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    assert UnboundedProgressWatch._unchanged_read_batch([row])
    row["delivery"]["working_set_repeat_capture"] = False
    assert not UnboundedProgressWatch._unchanged_read_batch([row])
    row["delivery"]["working_set_repeat_capture"] = True
    monkeypatch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR", raising=False)
    assert not UnboundedProgressWatch._unchanged_read_batch([row])


def test_on_real_qualified_hook_sends_certified_wire(tmp_path, monkeypatch):
    from test_request_determinism import _run_scripted_example, EXAMPLES
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    _hashes, requests, _ = _run_scripted_example(
        monkeypatch, "discount_bug", EXAMPLES["discount_bug"], tmp_path)
    events = []
    for path in (tmp_path / "run").rglob("events.jsonl*"):
        raw = lzma.decompress(path.read_bytes()) if path.suffix == ".xz" else path.read_bytes()
        events.extend(json.loads(line) for line in raw.splitlines() if line)
    plans = [row for row in events if row.get("kind") == "working_set_prepared"]
    assert plans, "build_qualified_request never published a certified candidate"
    assert len(plans) == len(requests)
    sealed = {row["body_sha256"].removeprefix("sha256:") for row in events
              if row.get("kind") == "qualified_request_sealed"}
    assert all(row["body_sha256"] in sealed and row["proof_passed"] for row in plans)
    assert not any(row.get("kind") == "working_set_blocked" for row in events)


def test_off_complete_request_bytes_match_1de34aa4(tmp_path, tmp_path_factory):
    from tools.pinned_tree import pinned_tree
    root = Path(__file__).resolve().parents[1]
    base = pinned_tree("56af0dc6d6", tmp_path_factory)
    driver = '''
import json, sys
from pathlib import Path
import pytest
from test_request_determinism import _run_scripted_example, EXAMPLES
with pytest.MonkeyPatch.context() as patch:
    patch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR", raising=False)
    patch.setenv("SF_LAB_DETFIX3", "on")
    hashes, _, _ = _run_scripted_example(patch, "discount_bug", EXAMPLES["discount_bug"], Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(hashes))
'''
    hashes = []
    for label, tree in (("base", base), ("current", root)):
        destination = tmp_path / f"{label}.json"
        environment = {k: v for k, v in os.environ.items() if not k.startswith("STATEFRAME_")}
        environment["PYTHONPATH"] = os.pathsep.join(str(tree / path) for path in
            ("src", "src/stateframe/engine", "tests"))
        result = subprocess.run([sys.executable, "-c", driver, str(tmp_path / label),
                                 str(destination)], cwd=tree, env=environment,
                                capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        hashes.append(json.loads(destination.read_text()))
    assert hashes[0] and hashes[0] == hashes[1]


@pytest.fixture
def hook(tmp_path, monkeypatch):
    """Real builder/oracle; isolate unrelated owner/transport services."""
    import kernel_request_candidate as candidates
    import kernel_request_budget as budget
    import kernel_wire_truth_gate as truth
    from kernel_request_diet import RequestTokenMeter
    from kernel_provider_dialect import ProviderCapabilities
    from kernel_final_wire_allocation import build_qualified_request
    from kernel_measurement import AuditJournal

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
    journal = AuditJournal(tmp_path, "cutover", {})
    plan = {"paths": [], "units": [], "source_contents": {}, "revisions": {},
            "workspace_authorization": {"scope": "test", "generation": "test"}}
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path), journal=journal,
        _request_source_closure=plan, _empirical_frame_sources={}, frame_ceiling=None)
    capture = SimpleNamespace(request_id="hook", step=1, journal=journal,
        request_action_contract={}, _request_source_closure_override=plan,
        _ensure_final_wire_ticket=lambda body: body,
        _framefit5_task_message={"role": "user", "content": "Task: fix"})
    meter = RequestTokenMeter(disabled=True)
    meter.bind_request("hook", "hook")
    monkeypatch.setattr(budget, "_request_token_meter", lambda *a, **k: meter)
    monkeypatch.setattr("kernel_request_kind.classify", lambda *a: True)
    monkeypatch.setattr(truth, "capture_context", lambda *a, **k: {})
    monkeypatch.setattr("kernel_wire_truth_facts.require_fact_context", lambda value: value)
    monkeypatch.setattr(truth, "enabled", lambda: False)
    monkeypatch.setattr("kernel_read_once.append_final_message", lambda rt, rows: rows)
    monkeypatch.setattr(candidates, "render_candidate", lambda cap, rt, body: body)
    monkeypatch.setattr("kernel_provider_dialect.capabilities_for",
                        lambda *a, **k: ProviderCapabilities(dialect="llama.cpp"))
    monkeypatch.setattr("kernel_request_tool_surface.author_final_tool_surface", lambda *a, **k: None)
    monkeypatch.setattr("kernel_cache_layout_candidate.compose_stable_guidance", lambda cap, body: body)
    monkeypatch.setattr("kernel_greenfact.maybe_append_greenfact", lambda *a, **k: None)
    monkeypatch.setattr("kernel_stateframe_framefit5.validate_final_product_boundary", lambda *a, **k: {})
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *a, **k: {"status": "COMPLETE", "proof_status": "VERIFIED"})
    body = {"model": "local", "messages": [
        {"role": "system", "content": "rules"}, capture._framefit5_task_message,
        {"role": "user", "content": '[StateFrame]\n{"state":{"framefit5":{"mode":"FRAME_ONLY"}}}'},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"}], "tools": [], "max_tokens": 100}
    def run(extension):
        capture._request_capture_pre_wire_extensions = extension
        return build_qualified_request(capture, runtime, body, endpoint="http://local/v1", key="")
    return run, capture, runtime, body


def test_validator_mutation_is_rejected_before_seal(hook):
    from kernel_request_seal import RequestSealError
    run, capture, _, _ = hook
    def mutate(body, *args, **kwargs):
        body["messages"][0]["content"] = "wrong"
        return {}
    with pytest.raises(RequestSealError, match="changed certified"):
        run(mutate)
    assert getattr(capture, "qualified_request", None) is None


def test_mandatory_overflow_is_durable_blocked_not_err(hook, monkeypatch):
    from kernel_working_set_stop import WorkingSetSuspended
    from kernel_working_set_state import WorkingSetState
    run, capture, runtime, _ = hook
    monkeypatch.setattr("kernel_request_budget.HARD_MAX_REQUEST_BYTES", 1)
    with pytest.raises(WorkingSetSuspended) as stopped:
        run(lambda *a, **k: {})
    assert stopped.value.outcome()["result"] == "BLOCKED"
    assert runtime.working_set_state.suspension is not None
    saved = runtime.working_set_state.checkpoint()
    assert WorkingSetState.from_checkpoint(saved).checkpoint() == saved
    rows = [json.loads(line) for line in capture.journal.path.read_bytes().splitlines()]
    assert any(row["kind"] == "working_set_blocked" for row in rows)
    assert not any(row["kind"] == "qualified_request_sealed" for row in rows)


@pytest.mark.parametrize("reason", [reason for reason in ReasonCode
                                   if reason not in MANDATORY_ENVELOPE_REASONS])
def test_non_floor_refusal_crosses_qualification_as_durable_suspension(hook, monkeypatch, reason):
    from kernel_working_set_prepare import Blocked
    from kernel_working_set_state import WorkingSetState
    from kernel_working_set_stop import WorkingSetSuspended

    run, capture, runtime, _ = hook
    monkeypatch.setattr("kernel_working_set_prepare.prepare_candidate",
                        lambda snapshot, **kwargs: Blocked(reason, 0, 0, snapshot.byte_cap, ()))
    with pytest.raises(WorkingSetSuspended) as stopped:
        run(lambda *a, **k: {})
    outcome = stopped.value.outcome()
    assert outcome["result"] == "BLOCKED" and outcome["resumable"]
    assert outcome["suspension"]["reason_code"] == reason.value
    assert outcome["suspension"]["proof"] is None
    saved = runtime.working_set_state.checkpoint()
    assert WorkingSetState.from_checkpoint(saved).checkpoint() == saved
    rows = [json.loads(line) for line in capture.journal.path.read_bytes().splitlines()]
    blocked, = [row for row in rows if row["kind"] == "working_set_blocked"]
    assert blocked["suspension"] == outcome["suspension"]
    assert "mandatory_envelope_blob_ref" not in blocked
    assert not any(row["kind"] == "qualified_request_sealed" for row in rows)
    assert getattr(capture, "qualified_request", None) is None


def test_on_never_calls_legacy_fitters_or_cache_callbacks(hook, monkeypatch):
    run, _, _, _ = hook
    def forbidden(*args, **kwargs):
        pytest.fail("legacy source selector ran on ON")
    for name in (
        "kernel_frame_allocation_capture.finalize_frame_candidate",
        "kernel_cache_layout_candidate.render_cache_candidate",
        "kernel_cache_layout_candidate.render_final_cache_candidate",
        "kernel_final_wire_allocation._recover_post_fit_growth",
        "kernel_stable_wire.project_stable_wire",
        "kernel_read_reuse_visibility.prepare_final_visibility",
    ):
        monkeypatch.setattr(name, forbidden)
    class ReachedValidators(Exception):
        pass
    def observed(body, *args, **kwargs):
        raise ReachedValidators
    with pytest.raises(ReachedValidators):
        run(observed)


@pytest.mark.parametrize("count", [1, 16], ids=["K1-giant", "K2-batch"])
def test_admitted_reads_reach_real_qualified_hook(hook, tmp_path, monkeypatch, count):
    import kernel_read_admission as admission
    import kernel_working_set_live as live
    from test_read_receipts_admission import big_file, read
    from kernel_working_set_proof import verify_certificate
    run, _, runtime, body = hook
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 12_000)
    monkeypatch.setattr("kernel_request_budget.HARD_MAX_REQUEST_BYTES", 12_000)
    runtime.working_set_fixed_bytes = 5_000
    runtime.retained = runtime.step_ledger = None
    results = []
    for index in range(count):
        path = f"page{index}.py"
        big_file(tmp_path, path, lines=1000)
        result, delivery = read(runtime, tmp_path, path,
                                {"path": path, "max_lines": 1000, "max_chars": 64_000})
        results.append((path, result, delivery))
    assert results[0][1][0] == "OK"
    assert results[0][2]["truncated"] and results[0][2]["next_cursor"]
    if count > 1:
        statuses = [result[0] for _, result, _ in results]
        first = statuses.index(admission.REFUSED_ADMISSION)
        assert all(status == admission.REFUSED_ADMISSION for status in statuses[first:])
        assert all(delivery["next_cursor"] for _, _, delivery in results[first:])
    calls, replies = [], []
    for index, (path, result, _) in enumerate(results):
        call_id = f"read-{index}"
        calls.append({"id": call_id, "type": "function", "function": {
            "name": "read", "arguments": json.dumps({"path": path})}})
        replies.append({"role": "tool", "tool_call_id": call_id, "content": result[1]})
    body["messages"][-1:-1] = [{"role": "assistant", "content": None,
                                "tool_calls": calls}, *replies]
    captured = []
    original = live.prepare_live
    def record(*args, **kwargs):
        value = original(*args, **kwargs)
        captured.append(value[1])
        return value
    monkeypatch.setattr(live, "prepare_live", record)
    class ReachedValidators(Exception):
        pass
    def observed(*args, **kwargs):
        raise ReachedValidators
    with pytest.raises(ReachedValidators):
        run(observed)
    prepared, = captured
    assert verify_certificate(prepared.wire, prepared.rendered.certificate)
    assert len(prepared.wire) <= 12_000
    certificate = prepared.rendered.certificate
    due = [slot for slot in certificate.snapshot.sources if slot.state == "DUE"]
    assert len(due) == sum(result[0] == "OK" for _, result, _ in results)
    assert all(dict(certificate.assignment)[slot.key] == "FULL" for slot in due)
    wire_body = json.loads(prepared.wire)
    for slot in due:
        assert any(slot.payload in str(row.get("content", "")) for row in wire_body["messages"])
    _assert_context_health(prepared, runtime, monkeypatch)


def _assert_context_health(prepared, runtime, monkeypatch):
    """Use crew-14's independent oracle when running the coordinated gate."""
    import importlib.util
    reference = os.environ.get("SF_TEST_CONTEXT_ALARM_SOURCE")
    if reference:
        spec = importlib.util.spec_from_file_location("wsa_context_alarm_reference", reference)
        assert spec is not None and spec.loader is not None
        alarm = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, alarm)
        spec.loader.exec_module(alarm)
    else:
        try:
            import kernel_context_alarm as alarm
        except ImportError:
            return  # crew-14 is independently shipped; source/cursor assertions above always run
    captures = runtime.read_capture_ledger.all()
    receipts = [capture.receipt() for capture in captures]
    index = alarm.index_request(prepared.wire)
    assert index.valid
    health = alarm.assess(index, receipts,
        {capture.identity.path: capture.identity.revision for capture in captures},
        alarm.AlarmState(), max(receipt.due_request for receipt in receipts),
        fits_without_degrade=prepared.fits_without_degrade)
    for invariant in ("I2", "I4", "I6"):
        assert health.invariants[invariant] == {"violations": 0, "unverifiable": 0}
