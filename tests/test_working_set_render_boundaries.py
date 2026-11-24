"""Allocator carriers must survive real closure and frozen-owner boundaries."""
import copy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from kernel_read_receipts import capture_read
from kernel_request_source_window import (
    _render_ranges, project_source_window, source_delivery_satisfied,
    source_revision, verify_final_source_delivery,
)
from kernel_source_delivery import deliver_source
from kernel_working_set_allocator import FULL
from kernel_working_set_prepare import PreparedCandidate, prepare_candidate
from kernel_working_set_proof import verify_certificate
from kernel_working_set_snapshot import freeze_preparation
from kernel_working_set_state import record_response, record_submission
from test_read_receipts import runtime


def envelope(suffix=(), *, tool_content="newest result"):
    return {"model": "local", "messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": '[StateFrame]\n{"state":{"framefit5":{}}}'},
        *copy.deepcopy(suffix),
        {"role": "assistant", "tool_calls": [{"id": "latest", "type": "function",
            "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "latest", "content": tool_content},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"},
    ], "tools": [], "max_tokens": 100}


def _retain_page(rt, path, text, first, last):
    args = {"path": path, "start_line": first, "end_line": last}
    page = deliver_source(text.encode(), path, 0, args=args)
    captured = capture_read(rt, path=path, metadata=page.metadata,
        payload=page.content, original_page=page.text, source="disk",
        arguments=args)
    request_id = f"exposed-{path}-{first}-{last}"
    record_submission(rt, request_id=request_id, body_sha256="1" * 64,
        modes={obligation.obligation_id: "F" for obligation in rt.working_set_state.active_obligations()})
    record_response(rt, request_id=request_id, response_id=f"accepted-{request_id}",
        body_sha256="1" * 64, disposition="accepted")
    return captured, page


def retained(tmp_path, monkeypatch, text, first, last):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    rt = runtime(tmp_path)
    captured, page = _retain_page(rt, "a.py", text, first, last)
    plan = {"paths": ["a.py"], "units": [{"path": "a.py", "start_line": first,
        "end_line": last, "role": "target", "unit_handle": "target"}],
        "revisions": {"a.py": source_revision(text)},
        "source_contents": {"a.py": text},
        "workspace_authorization": {"scope": str(tmp_path), "generation": "0"}}
    rt._request_source_closure = plan
    return rt, captured, page, plan


@pytest.mark.parametrize("rendered_closure", [True, False])
@pytest.mark.parametrize("text,first,last", [("one\n\ntwo λ\n", 1, 3),
                                             ("one\nsecond\nlast", 2, 3)])
def test_retained_full_capture_passes_real_closure_proof(
        tmp_path, monkeypatch, rendered_closure, text, first, last):
    rt, captured, page, plan = retained(tmp_path, monkeypatch, text, first, last)
    body = envelope()
    if rendered_closure:
        window = project_source_window(_render_ranges(text.splitlines(keepends=True),
            [{"start_line": first, "end_line": last, "anchors": []}],
            path="a.py", revision=source_revision(text), total_lines=len(text.splitlines())),
            owner="source_closure")
        body["messages"].insert(3, {"role": "user", "content": window})
    snapshot = freeze_preparation(body, rt, request_id="retained", byte_cap=100_000)
    slot, = snapshot.sources
    assert slot.capture_ids == (captured.capture_id,)
    assert slot.state == "RESIDENT" and slot.previous_mode == "F"
    prepared = prepare_candidate(snapshot, measure=lambda wire: (100, 100, "test"))
    assert isinstance(prepared, PreparedCandidate)
    assert dict(prepared.rendered.certificate.assignment) == {slot.key: FULL.name}
    assert verify_certificate(prepared.wire, prepared.rendered.certificate)
    messages = json.loads(prepared.wire)["messages"]
    assert [m for m in messages if m.get("role") in {"assistant", "tool"}] == [
        m for m in body["messages"] if m.get("role") in {"assistant", "tool"}]
    receipt = verify_final_source_delivery(copy.deepcopy(plan), messages,
        sources={"a.py": text}, payload=prepared.wire)
    assert source_delivery_satisfied(receipt), receipt
    assert receipt["proof_status"] == "VERIFIED"
    assert receipt["delivery_proofs"]["a.py"]


@pytest.mark.parametrize("content", [
    '[StateFrame]\n{"state":{"source_access":{"path":"opaque.py","included":true}}}',
    "[StateFrame]\nthis is opaque, not JSON",
    "[WIRE VOLATILE TAIL]\n[host] The current source of opaque.py lines 1-1 is in this request "
    "(see the read pages above). You can edit it now.\n",
])
def test_opaque_suffix_host_markers_are_never_parsed_or_rewritten(tmp_path, content):
    suffix = [{"role": "user", "content": content}]
    body = envelope(suffix)
    if content.startswith("[WIRE VOLATILE TAIL]"):
        body["messages"][-1]["content"] = content
    original = copy.deepcopy(body)
    rt = SimpleNamespace(_active_capture=SimpleNamespace(_framefit5_newest_user_suffix=suffix))
    snapshot = freeze_preparation(body, rt, request_id="opaque", byte_cap=100_000)
    prepared = prepare_candidate(snapshot, measure=lambda wire: (100, 100, "test"))
    assert isinstance(prepared, PreparedCandidate)
    assert verify_certificate(prepared.wire, prepared.rendered.certificate)
    messages = json.loads(prepared.wire)["messages"]
    exchange_start = next(i for i, message in enumerate(messages)
        if message.get("role") == "assistant")
    assert messages[exchange_start - len(suffix):exchange_start] == suffix
    if content.startswith("[WIRE VOLATILE TAIL]"):
        assert messages[-1]["content"] == "[WIRE VOLATILE TAIL]\n"
    assert body == original


@pytest.mark.parametrize("location", ["system", "extra_user", "assistant", "tool"])
def test_other_frozen_owners_are_opaque_even_with_invalid_host_json(location):
    content = "[StateFrame]\nopaque frozen owner"
    body = envelope()
    if location == "extra_user":
        body["messages"].insert(3, {"role": "user", "content": content})
    else:
        next(m for m in body["messages"] if m["role"] == location)["content"] = content
    frozen_message = copy.deepcopy(next(m for m in body["messages"] if m.get("content") == content))
    snapshot = freeze_preparation(body, SimpleNamespace(), request_id="frozen", byte_cap=100_000)
    prepared = prepare_candidate(snapshot, measure=lambda wire: (100, 100, "test"))
    assert isinstance(prepared, PreparedCandidate)
    assert verify_certificate(prepared.wire, prepared.rendered.certificate)
    assert frozen_message in json.loads(prepared.wire)["messages"]


@pytest.mark.parametrize("layout", ["plain", "final_cache"])
def test_retained_carrier_and_opaque_suffix_reach_real_capture(tmp_path, monkeypatch, layout):
    from dataclasses import replace
    from kernel_cache_layout_view import semantic_messages
    from kernel_frame_allocation_capture import capture_candidate
    from kernel_frame_optional_allocation import snapshot_optional_inputs
    from test_final_cache_layout import _body

    text = "first\nsecond\n"
    rt, _, _, plan = retained(tmp_path, monkeypatch, text, 1, 2)
    suffix = [{"role": "user", "content": "opaque latest instruction"}]
    if layout == "plain":
        suffix[0]["content"] = '[StateFrame]\n{"state":{"source_access":{"included":true}}}'
    body = envelope(suffix)
    body["messages"][2] = _body()["messages"][2]
    state_payload = json.loads(body["messages"][2]["content"].split("\n", 1)[1])
    state_payload["state"]["framefit5"]["mode"] = "FRAME_ONLY"
    body["messages"][2]["content"] = "[StateFrame]\n" + json.dumps(state_payload)
    cap = SimpleNamespace(request_id="capture", _framefit5_newest_user_suffix=suffix)
    rt._active_capture = cap
    rt.binding = lambda: {"artifact": "test", "product_revision": "test"}
    rt.frame_ceiling = 33_344
    snap = freeze_preparation(body, rt, request_id="capture", byte_cap=100_000)
    snap = replace(snap, layouts=("plain", "final_cache"))
    from kernel_working_set_render import render_assignment
    rendered = render_assignment(snap, {slot.key: FULL for slot in snap.sources}, layout)
    final_body = json.loads(rendered.wire)
    receipt = verify_final_source_delivery(copy.deepcopy(plan), final_body["messages"],
        sources={"a.py": text}, payload=rendered.wire)
    assert source_delivery_satisfied(receipt), receipt
    state = json.loads(semantic_messages(final_body["messages"])[2]["content"].split("\n", 1)[1])
    allocation = capture_candidate(runtime=rt, capture=cap, body=final_body,
        source_plan=plan, receipt=receipt, sources={"a.py": text},
        optional_inputs=snapshot_optional_inputs(state["state"]["framefit5"], tmp_path),
        byte_ceiling=100_000, token_ceiling=33_344, measurement_identity="test")
    assert allocation.source_layout["newest_user_suffix"] == suffix
    assert verify_certificate(rendered.wire, rendered.certificate)


def test_identical_closure_keeps_annotated_native_carrier(tmp_path, monkeypatch):
    text = "source bytes\n"
    rt, _, page, plan = retained(tmp_path, monkeypatch, text, 1, 1)
    annotated = page.text + "\n[read] current requested content is in this result."
    body = envelope(tool_content=annotated)
    snap = freeze_preparation(body, rt, request_id="native", byte_cap=100_000)
    prepared = prepare_candidate(snap, measure=lambda wire: (100, 100, "test"))
    assert isinstance(prepared, PreparedCandidate)
    final = json.loads(prepared.wire)
    witness, = prepared.rendered.certificate.witnesses
    assert final["messages"][witness.message_index]["role"] == "tool"
    assert final["messages"][witness.message_index]["content"] == annotated
    assert sum(text in m.get("content", "") for m in final["messages"]) == 1
    assert source_delivery_satisfied(verify_final_source_delivery(copy.deepcopy(plan),
        final["messages"], sources={"a.py": text}, payload=prepared.wire))


def test_explicitly_frozen_stateframe_owner_is_not_rewritten():
    from dataclasses import replace
    from test_working_set_render import snapshot
    from kernel_working_set_facts import rebuild_delivery_truth
    from kernel_working_set_render import render_assignment

    body = envelope()
    body["messages"][2]["content"] = '[StateFrame]\n{"state":{"source_access":{"included":true}}}'
    snap = replace(snapshot(body=body), truth_message_indexes=())
    detached = json.loads(snap.fixed_body)
    rebuild_delivery_truth(detached, snap, [])
    assert detached == body
    rendered = render_assignment(snap, {})
    assert body["messages"][2] in json.loads(rendered.wire)["messages"]


def test_user_read_page_does_not_gain_uncertified_closure_authority(tmp_path, monkeypatch):
    text = "exact source\n"
    _, _, page, plan = retained(tmp_path, monkeypatch, text, 1, 1)
    for role, expected in (("user", False), ("tool", True)):
        body = {"messages": [{"role": role, "content": page.text}]}
        receipt = verify_final_source_delivery(copy.deepcopy(plan), body["messages"],
            sources={"a.py": text}, payload=json.dumps(body).encode())
        assert source_delivery_satisfied(receipt) is expected


def test_certified_user_read_page_requires_matching_wire_position_and_identity(tmp_path, monkeypatch):
    text = "before\nexact λ source\nlast line"
    rt, _, page, plan = retained(tmp_path, monkeypatch, text, 2, 3)
    # Freeze the capture alone so no standalone source window can prove the plan.
    rt._request_source_closure = {}
    snap = freeze_preparation(envelope(), rt, request_id="native-user", byte_cap=100_000)
    prepared = prepare_candidate(snap, measure=lambda wire: (100, 100, "test"))
    assert isinstance(prepared, PreparedCandidate)
    certificate = prepared.rendered.certificate
    assert verify_certificate(prepared.wire, certificate)
    witness, = certificate.witnesses
    body = json.loads(prepared.wire)
    messages = body["messages"]
    index = witness.message_index
    assert messages[index] == {"role": "user", "content": page.text}

    def receipt(carrier, wire=prepared.wire):
        return verify_final_source_delivery(copy.deepcopy(plan), json.loads(wire)["messages"],
            payload=wire, verified_capture_witnesses=(carrier,))

    accepted = receipt(witness)
    assert source_delivery_satisfied(accepted), accepted
    proof, = accepted["delivery_proofs"]["a.py"]
    assert proof["final_message_location"] == f"$.messages[{index}].content"

    for invalid in (
        replace(witness, message_index=index + 1),
        replace(witness, mode="DEFERRED"),
        replace(witness, receipt_identity=replace(witness.receipt_identity, revision="sha256:" + "0" * 64)),
        replace(witness, receipt_identity=replace(witness.receipt_identity, first_line=1)),
        replace(witness, payload_start=witness.payload_start + 1),
    ):
        assert not source_delivery_satisfied(receipt(invalid))

    # An index with an exact page copy cannot stand in for the certified row.
    messages.insert(index, {"role": "user", "content": "different row"})
    assert not source_delivery_satisfied(receipt(witness, json.dumps(body).encode()))
    messages.pop(index)
    altered = page.text.replace("exact λ source", "other λ source")
    messages[index]["content"] = altered
    altered_wire = json.dumps(body).encode()
    assert not source_delivery_satisfied(receipt(witness, altered_wire))
    # The read-page parser must still reject altered source under the same revision.
    altered_witness = replace(witness, carrier_sha256=hashlib.sha256(altered.encode()).hexdigest())
    assert not source_delivery_satisfied(receipt(altered_witness, altered_wire))


def test_historical_capture_keeps_read_grammar_and_cannot_prove_current_source(tmp_path, monkeypatch):
    rt, captured, _, plan = retained(tmp_path, monkeypatch, "old source\n", 1, 1)
    text = "new source\n"
    plan["source_contents"]["a.py"] = text
    plan["revisions"]["a.py"] = source_revision(text)
    snap = freeze_preparation(envelope(), rt, request_id="historical", byte_cap=100_000)
    old, current = snap.sources
    assert old.capture_ids == (captured.capture_id,) and old.standalone_page is None
    assert old.page.startswith("[read page]")
    assert current.identity.revision == source_revision(text)
    prepared = prepare_candidate(snap, measure=lambda wire: (100, 100, "test"))
    assert isinstance(prepared, PreparedCandidate)
    receipt = verify_final_source_delivery(copy.deepcopy(plan), json.loads(prepared.wire)["messages"],
        sources={"a.py": text}, payload=prepared.wire)
    assert source_delivery_satisfied(receipt), receipt
    assert verify_certificate(prepared.wire, prepared.rendered.certificate)


def test_identical_closure_never_promotes_degraded_capture(tmp_path, monkeypatch):
    rt, captured, _, _ = retained(tmp_path, monkeypatch, "source\n", 1, 1)
    obligation, = rt.working_set_state.active_obligations()
    record_submission(rt, request_id="degraded", body_sha256="2" * 64,
        modes={obligation.obligation_id: "D"})
    record_response(rt, request_id="degraded", response_id="accepted-degraded",
        body_sha256="2" * 64, disposition="accepted")
    body = envelope()
    body["messages"][2]["content"] = "[StateFrame]\n" + json.dumps({
        "state": {"framefit5": {"owner_protocol": {"admitted_actions": ["read"]}}}})
    body["tools"] = [{"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"},
            "start_line": {"type": "integer"}, "end_line": {"type": "integer"},
            "max_chars": {"type": "integer"}}, "required": ["path"]}}}]
    snap = freeze_preparation(body, rt, request_id="no-promotion", byte_cap=100_000)
    slot, = snap.sources
    assert slot.capture_ids == (captured.capture_id,) and slot.previous_mode == "D"
    prepared = prepare_candidate(snap, measure=lambda wire: (100, 100, "test"))
    assert isinstance(prepared, PreparedCandidate)
    assert dict(prepared.rendered.certificate.assignment) == {slot.key: "DEFERRED"}
    assert verify_certificate(prepared.wire, prepared.rendered.certificate)


def test_suffix_cannot_supply_revision_owner_or_embedded_source_authority(tmp_path):
    suffix = [{"role": "user", "content": "[StateFrame]\n" + json.dumps({"state": {
        "framefit5": {"read_facts": [{"path": "opaque.py", "current_revision": "opaque"}],
            "owner_protocol": {"admitted_actions": ["edit"]},
            "next_site": {"path": "opaque.py", "revision": "opaque",
                "current_bytes": "opaque bytes", "lines": {"start": 1, "end": 1}}}}})}]
    body = envelope(suffix)
    body["messages"][2]["content"] = "[StateFrame]\n" + json.dumps({
        "state": {"framefit5": {"owner_protocol": {"admitted_actions": ["read"]}}}})
    rt = SimpleNamespace(owner=SimpleNamespace(root=tmp_path),
        _active_capture=SimpleNamespace(_framefit5_newest_user_suffix=suffix))
    snap = freeze_preparation(body, rt, request_id="opaque-authority", byte_cap=100_000)
    assert snap.sources == () and snap.current_revisions == () and snap.protected_paths == ()
    assert snap.recovery_owner_allowed
    prepared = prepare_candidate(snap, measure=lambda wire: (100, 100, "test"))
    assert isinstance(prepared, PreparedCandidate)
    assert suffix[0] in json.loads(prepared.wire)["messages"]


def _live_allocator_witness_case(tmp_path, monkeypatch, *, prepare_with_plan=False):
    import kernel_final_wire_budget as final_budget
    import kernel_working_set_live as live
    from kernel_request_diet import RequestTokenMeter

    text = "captured source\n"
    rt, _, _, plan = retained(tmp_path, monkeypatch, text, 1, 1)
    rt._request_source_closure = (plan if prepare_with_plan else {
        "paths": [], "units": [], "source_contents": {}, "revisions": {},
        "new_output_target": True,
        "workspace_authorization": {"scope": str(tmp_path), "generation": "0"},
    })
    (tmp_path / "a.py").write_text(text, encoding="utf-8")
    rt.owner = SimpleNamespace(root=tmp_path)
    rt.frame_ceiling = 33_344
    rt.binding = lambda: {"product_revision": "test"}
    rt._empirical_frame_sources = {"a.py": {"content": text}}
    meter = RequestTokenMeter(disabled=True)
    meter.bind_request("final-witness", "final-witness")
    monkeypatch.setattr("kernel_request_budget._request_token_meter",
                        lambda *_args, **_kwargs: meter)
    capture = SimpleNamespace(
        request_id="final-witness", step=1, journal=rt.journal,
        request_action_contract={"admitted_tool_schemas": []},
        request_budget_final_wire={}, final_tool_surface=None,
    )
    body = {"model": "local", "max_tokens": 100, "tools": [], "messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps({"state": {
            "framefit5": {"mode": "FRAME_ONLY", "owner_protocol": {
                "admitted_actions": ["read"]}}}})},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"},
    ]}

    body, prepared = live.prepare_live(
        capture, rt, body, dialect="llama.cpp", reasoning_mode="on",
    )
    assert verify_certificate(prepared.wire, prepared.rendered.certificate)
    final_body = json.loads(prepared.wire)
    assignment = dict(prepared.rendered.certificate.assignment)
    capture_slots = {slot.key for slot in prepared.rendered.certificate.snapshot.sources
                     if slot.capture_ids and assignment[slot.key] == FULL.name}
    verified_capture_witnesses = tuple(
        witness for witness in prepared.rendered.certificate.witnesses
        if witness.slot_id in capture_slots
    )
    assert verified_capture_witnesses
    assert "source_delivery_witness_binding" in capture._frame_candidate_inputs
    if prepare_with_plan:
        initial_plan = copy.deepcopy(capture._request_source_closure_override)
        initial_receipt = capture.request_source_final_delivery
    else:
        initial_plan = copy.deepcopy(plan)
        initial_receipt = verify_final_source_delivery(
            initial_plan, final_body["messages"], sources={"a.py": text},
            payload=prepared.wire, verified_capture_witnesses=verified_capture_witnesses,
        )
        assert source_delivery_satisfied(initial_receipt), initial_receipt
        capture.request_source_final_delivery = initial_receipt
        capture._request_source_closure_override = initial_plan
    binding = final_budget.capture_source_witness_binding(
        prepared.rendered.certificate, initial_plan, {"a.py": text},
        verified_capture_witnesses,
    )
    assert binding is not None
    if not prepare_with_plan:
        capture._frame_candidate_inputs["source_delivery_witness_binding"] = binding
    return final_budget, capture, rt, final_body, initial_plan, initial_receipt, verified_capture_witnesses


def test_live_allocator_witness_reaches_real_final_revalidation(tmp_path, monkeypatch):
    final_budget, capture, rt, final_body, _plan, initial_receipt, witnesses = (
        _live_allocator_witness_case(tmp_path, monkeypatch)
    )
    frozen = final_budget.snapshot_revalidation_inputs(capture, rt, final_body)
    receipt = final_budget.reconcile_frozen_budget(
        capture, rt, final_body, frozen_inputs=frozen,
    )
    capture_locations = {
        f"$.messages[{witness.message_index}].content"
        for witness in witnesses
    }
    initial_locations = {proof["final_message_location"]
                         for proof in initial_receipt["delivery_proofs"]["a.py"]}
    assert initial_locations == capture_locations, initial_receipt
    assert source_delivery_satisfied(receipt["source_delivery"]), receipt
    assert "source_delivery_unqualified" not in receipt["final_validation_failures"]
    final_locations = {proof["final_message_location"]
                       for proof in receipt["source_delivery"].get("delivery_proofs", {}).get("a.py", ())}
    assert final_locations == initial_locations, receipt


def test_live_allocator_binds_its_real_closure_proof(tmp_path, monkeypatch):
    final_budget, capture, rt, body, _plan, initial_receipt, _witnesses = (
        _live_allocator_witness_case(tmp_path, monkeypatch, prepare_with_plan=True)
    )
    frozen = final_budget.snapshot_revalidation_inputs(capture, rt, body)
    receipt = final_budget.reconcile_frozen_budget(capture, rt, body, frozen_inputs=frozen)
    assert source_delivery_satisfied(initial_receipt), initial_receipt
    assert source_delivery_satisfied(receipt["source_delivery"]), receipt

    # The same inputs are splatted into the allocation capture: a key it does not take ends the run.
    import inspect

    from kernel_final_wire_allocation import allocation_keywords
    from kernel_frame_allocation_capture import capture_candidate

    assert "source_delivery_witness_binding" in capture._frame_candidate_inputs
    assert set(allocation_keywords(capture)) <= set(inspect.signature(capture_candidate).parameters)


@pytest.fixture
def gate_case(tmp_path, monkeypatch):
    import kernel_wire_truth_gate as truth
    from kernel_completion_owner import establish_lean_request_action_contract
    from test_read_admission_qualification import seal_fixture

    capture, rt, draft, plan, journal, _ = seal_fixture(
        tmp_path, monkeypatch, identity_content={"fixture": "source witness"}, wire_truth=True,
    )
    monkeypatch.setenv(truth.MODE_ENV, "on")
    monkeypatch.setattr("kernel_request_budget.HARD_MAX_REQUEST_BYTES", 10_500)

    # Give both entry points the same prepared frame, with source placement left to the allocator.
    def prepared_messages(messages, *args, **kwargs):
        messages = copy.deepcopy(messages)
        state = json.loads(messages[2]["content"].split("\n", 1)[1])
        state["state"]["framefit5"].setdefault("owner_protocol", {})["admitted_actions"] = ["read"]
        messages[2]["content"] = "[StateFrame]\n" + json.dumps(state)
        return messages

    monkeypatch.setattr("kernel_stateframe_framefit5.assemble_framefit5_request", prepared_messages)
    draft["tools"][0]["function"]["parameters"]["properties"]["start_line"] = {"type": "integer"}
    capture.request_action_contract = establish_lean_request_action_contract(
        rt, draft["messages"], draft["tools"], request_id=capture.request_id,
    )
    capture_context = truth.capture_context

    def source_context(candidate, runtime, body, **kwargs):
        candidate = copy.copy(candidate)
        candidate.wire_truth_context = {}
        return capture_context(candidate, runtime, body, **kwargs)

    monkeypatch.setattr(truth, "capture_context", source_context)
    for path, source in {"a.py": "# " + "a" * 2500 + "\n# " + "b" * 2500 + "\n",
                         "b.py": "dependency = True\n"}.items():
        (tmp_path / path).write_text(source, encoding="utf-8")
        _retain_page(rt, path, source, 1, len(source.splitlines()))
        plan["paths"].append(path)
        plan["units"].append({"path": path, "start_line": 1, "end_line": 1,
                              "role": "target", "unit_handle": path})
        plan["source_contents"][path] = source
        plan["revisions"][path] = source_revision(source)
    # The read covers a.py's required line. The cap excludes the redundant
    # standing window. b.py discharges a deferral declared by the plan.
    plan["deferred_paths"] = ["b.py"]
    return capture, rt, draft, journal


@pytest.mark.parametrize("replacement", [False, True], ids=["first", "replacement"])
def test_wire_truth_uses_certified_read_pages_after_deferral_discharge(gate_case, replacement):
    from kernel_final_wire_allocation import build_qualified_request
    from kernel_request_reprepare import PreparationInput, prepare_replacement

    capture, rt, draft, _ = gate_case
    request_id = capture.request_id
    endpoint = {"endpoint": "http://127.0.0.1:1/v1", "key": ""}
    if replacement:
        prepare_replacement(rt, capture, PreparationInput.capture(capture, rt, draft), draft,
                            deadline=None, **endpoint)
    else:
        build_qualified_request(capture, rt, draft, **endpoint)
    assert (capture.request_id != request_id) is replacement
    wire = capture.qualified_request.wire
    body = json.loads(wire)
    plan = capture._request_source_closure_override
    receipt = capture.request_budget_final_wire["source_delivery"]
    witnesses = capture._frame_candidate_inputs["source_delivery_witness_binding"]["witnesses"]
    assert plan["deferred_paths"] == ["b.py"]
    assert not receipt["deferred_paths"]
    assert {row["status"] for row in receipt["paths"].values()} == {"COMPLETE"}
    assert not source_delivery_satisfied(verify_final_source_delivery(
        copy.deepcopy(plan), body["messages"], payload=wire,
    ))
    assert {proof["final_message_location"] for proofs in receipt["delivery_proofs"].values()
            for proof in proofs} == {
        f"$.messages[{witness.message_index}].content" for witness in witnesses
    }
    assert capture._wire_truth_check.verdict == "PASS"
    assert capture._wire_truth_check.wire == wire


@pytest.mark.parametrize("drift", [
    "missing_window", "request", "stale_frozen", "wire", "plan", "source", "forged_witness", "duplicate_witness",
])
def test_wire_truth_refuses_unbound_capture_witnesses(gate_case, drift):
    from kernel_final_wire_allocation import build_qualified_request
    from kernel_final_wire_budget import snapshot_revalidation_inputs
    from kernel_wire_truth_gate import WireTruthBlocked, check_wire

    capture, rt, draft, _ = gate_case
    record = build_qualified_request(capture, rt, draft, endpoint="http://127.0.0.1:1/v1", key="")
    body, wire = record.body, record.wire
    context = capture._wire_truth_context
    binding = capture._frame_candidate_inputs["source_delivery_witness_binding"]
    target = next(w for w in binding["witnesses"] if w.receipt_identity.path == "a.py")
    if drift == "missing_window":
        body["messages"][target.message_index]["content"] = "required read page removed"
        wire = json.dumps(body).encode()
    elif drift == "request":
        capture.request_id = context["request_id"] = "another-request"
    elif drift == "wire":
        wire += b" "  # Same parsed request, different transport bytes.
    elif drift == "plan":
        context["source_plan"]["units"][0]["unit_handle"] = "different-unit"
    elif drift == "source":
        context["source_contents"]["a.py"] = "changed source\n"
    elif drift == "forged_witness":
        binding["witnesses"] = tuple(
            replace(witness, message_index=witness.message_index + 1) if witness is target else witness
            for witness in binding["witnesses"]
        )
    elif drift == "duplicate_witness":
        binding["witnesses"] += (target,)
    frozen = snapshot_revalidation_inputs(capture, rt, body)
    if drift == "stale_frozen":
        capture.request_id = "a-later-request"  # The inputs and context are the earlier request's.
    with pytest.raises(WireTruthBlocked) as blocked:
        check_wire(wire, context, capture=capture, frozen_inputs=frozen)
    assert blocked.value.failed == ("source_delivery_incomplete",)


def test_replay_rebuilds_in_flight_request_that_needed_capture_witnesses(gate_case, tmp_path):
    from kernel_final_wire_allocation import build_qualified_request
    from kernel_wire_truth_gate import WireTruthBlocked, check_wire
    from kernel_working_set_state import WorkingSetState, observe_submission, restore_tail
    from stateframe.replay_tape import Replayer

    capture, rt, draft, journal = gate_case
    resumed = copy.copy(capture)
    restored = copy.copy(rt)
    restored.owner = copy.copy(rt.owner)
    restored._active_capture = resumed
    restored.working_set_state = WorkingSetState()
    endpoint = {"endpoint": "http://127.0.0.1:1/v1", "key": ""}
    record = build_qualified_request(capture, rt, draft, **endpoint)
    with pytest.raises(WireTruthBlocked) as blocked:
        check_wire(record.wire, capture._wire_truth_context)
    assert blocked.value.failed == ("source_delivery_incomplete",)
    observe_submission(rt, capture, record.wire)
    path = tmp_path / "pending.request.json"
    path.write_bytes(record.wire)
    tape = Replayer([], strict=True, pending={"send": 3, "request": str(path)})
    restore_tail(restored, [json.loads(row) for row in journal.path.read_bytes().splitlines()], 0)
    rebuilt = build_qualified_request(resumed, restored, draft, **endpoint)
    assert rebuilt.wire == record.wire
    assert resumed._wire_truth_check.verdict == "PASS"
    assert tape.answer(rebuilt.wire) is None
    assert tape.pending_verified["request_sha256"] == record.body_digest


def test_no_call_site_splats_the_raw_candidate_inputs():
    from pathlib import Path

    import kernel_final_wire_allocation

    source = Path(kernel_final_wire_allocation.__file__).read_text(encoding="utf-8")
    assert "**candidate._frame_candidate_inputs" not in source


@pytest.mark.parametrize("drift", [
    "source_snapshot", "current_source", "plan", "carrier", "forged_witness",
    "stale_witness", "missing_witness",
])
def test_final_revalidation_refuses_capture_witness_drift(tmp_path, monkeypatch, drift):
    final_budget, capture, rt, body, _plan, _initial_receipt, witnesses = (
        _live_allocator_witness_case(tmp_path, monkeypatch)
    )
    if drift == "carrier":
        index = witnesses[0].message_index
        body["messages"][index]["content"] += "\nchanged carrier"
    frozen = final_budget.snapshot_revalidation_inputs(capture, rt, body)
    if drift == "source_snapshot":
        frozen["plan"]["source_contents"]["a.py"] = "changed source\n"
    elif drift == "current_source":
        frozen["sources"]["a.py"] = "changed source\n"
    elif drift == "plan":
        frozen["plan"]["units"][0]["end_line"] = 2
    elif drift == "forged_witness":
        binding = frozen["source_delivery_witness_binding"]
        binding["witnesses"] = (replace(witnesses[0], message_index=witnesses[0].message_index + 1),)
    elif drift == "stale_witness":
        binding = capture._frame_candidate_inputs["source_delivery_witness_binding"]
        binding["snapshot_digest"] = "sha256:" + "0" * 64
        frozen = final_budget.snapshot_revalidation_inputs(capture, rt, body)
    elif drift == "missing_witness":
        del capture._frame_candidate_inputs["source_delivery_witness_binding"]
        frozen = final_budget.snapshot_revalidation_inputs(capture, rt, body)

    receipt = final_budget.reconcile_frozen_budget(capture, rt, body, frozen_inputs=frozen)
    assert "source_delivery_unqualified" in receipt["final_validation_failures"], receipt
    assert not source_delivery_satisfied(receipt["source_delivery"]), receipt


def test_allocator_off_keeps_final_source_verification_and_wire_unchanged(tmp_path, monkeypatch):
    import kernel_body_serialization_memo as serialization
    import kernel_read_receipts

    final_budget, capture, rt, body, _plan, _initial_receipt, _witnesses = (
        _live_allocator_witness_case(tmp_path, monkeypatch)
    )
    frozen = final_budget.snapshot_revalidation_inputs(capture, rt, body)
    before = serialization.plain_wire(body)
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    assert not kernel_read_receipts.enabled()
    baseline = verify_final_source_delivery(
        copy.deepcopy(frozen["plan"]), body["messages"],
        sources=frozen["sources"], payload=before,
    )
    receipt = final_budget.reconcile_frozen_budget(capture, rt, body, frozen_inputs=frozen)
    assert serialization.plain_wire(body) == before
    assert receipt["source_delivery"] == baseline
