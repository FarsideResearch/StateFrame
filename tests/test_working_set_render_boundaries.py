"""Allocator carriers must survive real closure and frozen-owner boundaries."""
import copy
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


def retained(tmp_path, monkeypatch, text, first, last):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    rt = runtime(tmp_path)
    page = deliver_source(text.encode(), "a.py", 0,
        args={"path": "a.py", "start_line": first, "end_line": last})
    captured = capture_read(rt, path="a.py", metadata=page.metadata,
        payload=page.content, original_page=page.text, source="disk",
        arguments={"path": "a.py", "start_line": first, "end_line": last})
    obligation, = rt.working_set_state.active_obligations()
    record_submission(rt, request_id="exposed", body_sha256="1" * 64,
        modes={obligation.obligation_id: "F"})
    record_response(rt, request_id="exposed", response_id="accepted",
        body_sha256="1" * 64, disposition="accepted")
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
