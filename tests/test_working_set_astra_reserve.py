"""Astra 11: replenish the native-batch reserve with accepted O carriers."""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import kernel_read_receipts as receipts
import kernel_working_set_state as ws
import pytest
from kernel_source_delivery import deliver_source
from kernel_working_set_allocator import DEFERRED, FULL
from kernel_working_set_render import domain, render_assignment
from kernel_working_set_snapshot import freeze_preparation
from test_read_receipts import runtime


@pytest.fixture(autouse=True)
def allocator_on(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")


def add_capture(rt, number, *, recovery_pin=False):
    name = f"source{number}.py"
    page = deliver_source(b"x\n", name, 0, args={"path": name})
    capture = receipts.capture_read(rt, path=name, metadata=page.metadata,
        payload=page.content, original_page=page.text, source="disk", arguments={"path": name})
    if recovery_pin:
        # Real recovery capture event, rather than mutating an index entry.
        recovered = replace(capture, capture_id=capture.capture_id + "-recovery")
        rt.read_capture_ledger.add(recovered)
        ws.record_capture(rt, recovered, recovery_pin=True)


def seed_residents(rt):
    for number in range(ws.MAX_RETAINED_RESIDENTS):
        add_capture(rt, number)
    state = ws.state_for(rt)
    ws.record_submission(rt, request_id="seed", body_sha256="1" * 64,
        modes={entry.obligation_id: "F" for entry in state.active_obligations()})
    ws.record_response(rt, request_id="seed", response_id="accepted-seed",
                       body_sha256="1" * 64, disposition="accepted")
    assert len(state.active_obligations()) == ws.MAX_RETAINED_RESIDENTS


def candidate(rt, request):
    body = {"messages": [{"role": "system", "content": "rules"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps({"state": {"framefit5": {
            "owner_protocol": {"admitted_actions": ["read"]}}}})},
        {"role": "user", "content": "suffix"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"},
            "start_line": {"type": "integer"}, "end_line": {"type": "integer"},
            "max_chars": {"type": "integer"}}, "required": ["path"]}}}]}
    frozen = freeze_preparation(body, rt, request_id=request, byte_cap=1_000_000)
    modes = {slot.key: DEFERRED if slot.state == "RETIRING" else FULL for slot in frozen.sources}
    for slot in frozen.sources:
        if slot.state == "RETIRING":
            assert domain(frozen, slot) == (DEFERRED,)
    rendered = render_assignment(frozen, modes)
    capture = SimpleNamespace(request_id=request, _working_set_certificate=rendered.certificate)
    return capture, rendered.wire


def test_reserve_retirement_is_lru_certified_and_accepted_only(tmp_path):
    rt = runtime(tmp_path)
    seed_residents(rt)
    state = ws.state_for(rt)
    for batch in range(3):
        before = state.active_obligations()
        oldest = sorted(before, key=lambda o: (o.last_use_ordinal, o.admission_ordinal, o.obligation_id))
        expected = {entry.obligation_id for entry in oldest[:ws.NATIVE_BATCH_RESERVE]}
        for offset in range(ws.NATIVE_BATCH_RESERVE):
            add_capture(rt, ws.MAX_RETAINED_RESIDENTS + batch * ws.NATIVE_BATCH_RESERVE + offset)
        active = state.active_obligations()
        assert {o.obligation_id for o in active if o.state == "RETIRING"} == expected
        assert len(active) == ws.MAX_ACTIVE_OBLIGATIONS and state.overflow_count == 0
        assert sum(o.state == "DUE" for o in active) == ws.NATIVE_BATCH_RESERVE
        saved = state.checkpoint()
        capture, wire = candidate(rt, f"failed-{batch}")
        assert state.checkpoint() == saved  # preparation never retires or touches LRU
        ws.observe_submission(rt, capture, wire)
        ws.observe_response(rt, capture, disposition="failed", response_id="failed")
        assert {o.obligation_id for o in state.active_obligations()} == {o.obligation_id for o in active}
        capture, wire = candidate(rt, f"accepted-{batch}")
        ws.observe_submission(rt, capture, wire)
        assert len(state.active_obligations()) == ws.MAX_ACTIVE_OBLIGATIONS
        ws.observe_response(rt, capture, disposition="accepted", response_id="accepted")
        assert len(state.active_obligations()) == ws.MAX_RETAINED_RESIDENTS
        assert not expected & {o.obligation_id for o in state.active_obligations()}
        assert state.overflow_count == 0 and state.shadow_complete
        assert len(rt.read_capture_ledger) == ws.MAX_RETAINED_RESIDENTS + (batch + 1) * ws.NATIVE_BATCH_RESERVE
        assert ws.WorkingSetState.from_checkpoint(state.checkpoint()).checkpoint() == state.checkpoint()
        assert ws.reconstruct(rt.journal.events).checkpoint() == state.checkpoint()


def test_reserve_reread_and_recovery_pin_stay_full(tmp_path):
    rt = runtime(tmp_path)
    seed_residents(rt)
    state = ws.state_for(rt)
    add_capture(rt, ws.MAX_RETAINED_RESIDENTS)
    retiring, = [o for o in state.active_obligations() if o.state == "RETIRING"]
    assert retiring.identity.path == "source0.py"
    add_capture(rt, 0, recovery_pin=True)
    active = state.active_obligations()
    reread, = [o for o in active if o.identity.path == "source0.py"]
    assert reread.state == "DUE" and reread.recovery_pin
    assert {o.identity.path for o in active if o.state == "RETIRING"} == {"source1.py"}
    capture, wire = candidate(rt, "recovery")
    selected = dict(capture._working_set_certificate.assignment)
    assert selected[reread.obligation_id] == "FULL"
    ws.observe_submission(rt, capture, wire)
    ws.observe_response(rt, capture, disposition="accepted", response_id="accepted")
    assert reread.state == "RESIDENT" and not reread.recovery_pin
    assert len(state.active_obligations()) == ws.MAX_RETAINED_RESIDENTS


def test_reserve_uses_last_use_before_admission_order(tmp_path):
    rt = runtime(tmp_path)
    seed_residents(rt)
    state = ws.state_for(rt)
    first = state.active_obligations()[0]
    # A matching adopted assignment touches an existing resident's recency.
    ws.record_submission(rt, request_id="touch", body_sha256="2" * 64,
                         modes={first.obligation_id: "F"})
    ws.record_response(rt, request_id="touch", response_id="accepted-touch",
                       body_sha256="2" * 64, disposition="accepted")
    add_capture(rt, ws.MAX_RETAINED_RESIDENTS)
    retiring, = [o for o in state.active_obligations() if o.state == "RETIRING"]
    assert retiring.identity.path == "source1.py"
    assert first.state == "RESIDENT"
    with pytest.raises(ValueError, match="O assignment"):
        ws.record_submission(rt, request_id="bad-retirement", body_sha256="3" * 64,
                             modes={retiring.obligation_id: "F"})
