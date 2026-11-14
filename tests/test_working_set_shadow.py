"""Shadow admission is observational and uses detached capture authority."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_read_receipts import ReceiptIdentity, line_digests_with_numbers
from kernel_working_set_snapshot import PreparationSnapshot, SourceSlot, freeze_preparation
from kernel_working_set_shadow import freeze_shadow, journal_shadow


class Journal:
    def __init__(self):
        self.events = []
    def emit(self, kind, **fields):
        self.events.append({"kind": kind, **fields})


def page_slot():
    text = "fresh bytes\n"
    identity = ReceiptIdentity("a.py", "sha256:" + "1" * 64, 1, 1, 0,
                               len(text.encode()), hashlib.sha256(text.encode()).hexdigest(), None)
    ds, ns = line_digests_with_numbers(text, 1)
    page = f"[read page] path=a.py revision={identity.revision} lines=1-1/1\n0 more; complete=true\n{text}"
    return SourceSlot("fresh", ("fresh",), identity, text, page, ds, ns,
                      b'{"path":"a.py","start_line":1,"end_line":1}', 1, 1, "DUE")


def test_shadow_on_is_observational_and_reports_due_coverage(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    body = {"model": "local", "messages": [{"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "opaque suffix"}], "max_tokens": 100}
    wire = json.dumps(body).encode()
    snapshot = PreparationSnapshot(wire, (page_slot(),), "request", 1, 82026)
    journal = Journal()
    result = journal_shadow(snapshot, wire, journal=journal)
    assert json.dumps(body).encode() == wire
    event, = journal.events
    assert event["kind"] == "working_set_shadow_plan"
    assert event["blocked_reason"] is None
    assert event["due_reads"]["full"] == 1
    assert event["due_reads"]["legacy_full"] == 0
    assert event["floor"] <= event["total"] <= event["cap"]
    assert result.wire != wire


def test_shadow_off_has_no_snapshot_calls_or_events(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    journal = Journal()
    assert freeze_shadow({}, object(), request_id="request") is None
    assert journal_shadow(object(), b"unchanged", journal=journal) is None
    assert journal.events == []


def test_shadow_snapshot_errors_are_diagnostic_and_never_break_legacy(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    journal = Journal()
    snapshot = freeze_shadow({}, object(), request_id="request")
    assert journal_shadow(snapshot, b"unchanged", journal=journal) is None
    event, = journal.events
    assert event["blocked_reason"] == "PREPARATION_CONTRACT_FAILURE"
    assert event["error_type"]


def test_shadow_event_is_safe_tail():
    from kernel_continuation_restart import SAFE_TAIL
    assert "working_set_shadow_plan" in SAFE_TAIL


def test_shadow_freezes_each_endpoint_reserve_once(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    from kernel_request_budget import (ENDPOINT_THINKING_BUDGET_TOKENS,
                                       ENDPOINT_SAFETY_HEADROOM_TOKENS)
    body = {"messages": [{"role": "system", "content": "system"},
                         {"role": "user", "content": "Task: fix"}]}
    snapshot = freeze_shadow(body, SimpleNamespace(), request_id="request")
    assert isinstance(snapshot, PreparationSnapshot)
    assert snapshot.output_reserve == ENDPOINT_THINKING_BUDGET_TOKENS
    assert snapshot.safety_reserve == ENDPOINT_SAFETY_HEADROOM_TOKENS


def test_incomplete_active_index_is_a_typed_shadow_diagnostic(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    body = {"messages": [{"role": "system", "content": "system"}]}
    snapshot = freeze_shadow(body, SimpleNamespace(working_set_state=SimpleNamespace(
        overflow_count=1)), request_id="request")
    journal = Journal()
    journal_shadow(snapshot, json.dumps(body).encode(), journal=journal)
    event, = journal.events
    assert event["blocked_reason"] == "ACTIVE_OBLIGATION_LIMIT"


def test_recorded_wire_0020_fresh_page_full_and_within_cap(tmp_path, monkeypatch):
    """Hydrate the recorded complete envelope, not an underestimated toy shell."""
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    path = Path.home() / "crew-lanes/tmp/crew-01/ret/r38g/run/wire/0020.request.json"
    if not path.exists():
        pytest.skip("recorded r38g wire absent")
    from kernel_read_receipts import capture_read
    from test_read_receipts import runtime
    import re
    body = json.loads(path.read_bytes())
    legacy_wire = json.dumps(body).encode()
    # This recording predates removal of source_refs overlay indirection.
    # Hydrate each declared physical span exactly, then feed the resulting
    # current overlay through the normal strict semantic reconstruction.
    from kernel_stateframe_final_message import AUTHORITY_OVERLAY_HEADER
    marker = AUTHORITY_OVERLAY_HEADER + "\n"
    content = body["messages"][-1]["content"]
    prefix, encoded = content.split(marker, 1)
    overlay, consumed = json.JSONDecoder().raw_decode(encoded)
    for field in overlay["fields"]:
        references = field.pop("source_refs", ())
        for reference in references:
            value = field["value"]
            assert isinstance(value, dict) and value[reference["key"]] is None
            source = body["messages"][reference["message"]]["content"]
            first_char, last_char = reference["span"]
            assert 0 <= first_char < last_char <= len(source)
            value[reference["key"]] = source[first_char:last_char]
    body["messages"][-1]["content"] = prefix + marker + json.dumps(
        overlay, separators=(",", ":"), sort_keys=True) + encoded[consumed:]
    page = next(m["content"] for m in body["messages"] if m.get("role") == "tool"
                and str(m.get("content", "")).startswith("[read page]"))
    header, _more, payload = page.split("\n", 2)
    fields = dict(re.findall(r"([a-z_]+)=([^ ]+)", header))
    first, last, total = map(int, re.fullmatch(r"(\d+)-(\d+)/(\d+)", fields["lines"]).groups())
    assert first == 1 and last == total
    assert "sha256:" + hashlib.sha256(payload.encode()).hexdigest() == fields["revision"]
    rt = runtime(tmp_path)
    metadata = {"result_class": "OK", "revision": fields["revision"],
                "effective_range": {"start_line": first, "end_line": last},
                "byte_start": 0, "byte_end": len(payload.encode())}
    capture_read(rt, path=fields["path"], metadata=metadata, payload=payload,
        original_page=page, source="disk", arguments={"path": fields["path"],
        "start_line": first, "end_line": last})
    snapshot = freeze_preparation(body, rt, request_id="0fe10ea2", byte_cap=82026)
    # The frozen fixed body contains the entire tool schema and typed state.
    assert len(snapshot.fixed_body) > 50_000
    result = journal_shadow(snapshot, legacy_wire, journal=rt.journal)
    from kernel_working_set_prepare import PreparedCandidate
    assert isinstance(result, PreparedCandidate)
    event = next(e for e in rt.journal.events if e["kind"] == "working_set_shadow_plan")
    assert event["due_reads"]["fresh_full"] is True
    assert event["due_reads"]["full"] == 1
    assert event["floor"] <= event["total"] <= event["cap"] == 82026
    print("recorded shadow", event["floor"], event["total"], event["cap"], "fresh FULL yes")


def test_snapshot_copies_mutable_sequence_inputs():
    slot = page_slot()
    obligations = ["fresh"]
    frozen_slot = replace(slot, obligation_ids=obligations)
    slots = [frozen_slot]
    protected = ["a.py"]
    revisions = [["a.py", slot.identity.revision]]
    snapshot = PreparationSnapshot(b'{"messages":[]}', slots, "request", 1, 82026,
        protected_paths=protected, current_revisions=revisions, layouts=["plain"])
    before = snapshot.snapshot_digest
    obligations.append("forged")
    slots.clear()
    protected.clear()
    revisions[0][1] = "forged"
    assert snapshot.snapshot_digest == before
    assert snapshot.sources == (frozen_slot,)
    assert frozen_slot.obligation_ids == ("fresh",)


def test_snapshot_protects_pending_edit_surface_and_latest_failure_only():
    frame = {"state": {"framefit5": {"open_sites": [{"path": "active.py"}],
             "owner_protocol": {"admitted_actions": ["read", "edit"]}}}}
    body = {"messages": [{"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps(frame)}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]}
    rt = SimpleNamespace(edit_contract={"edit_authority": {"editable_paths": [
        "active.py", "unused.py", "queued.py"]}},
        native_batch=SimpleNamespace(pending=[{"function": {"name": "str_replace",
            "arguments": '{"path":"queued.py","old":"a","new":"b"}'}}]),
        _check_output_receipts={"old": {"exit_code": 1, "stderr": 'File "old.py", line 2'},
                                "new": {"exit_code": 1, "stderr": 'File "failed.py", line 3'}})
    snapshot = freeze_preparation(body, rt, request_id="request", byte_cap=82026)
    assert set(snapshot.protected_paths) == {"active.py", "queued.py", "failed.py"}
    assert snapshot.recovery_owner_allowed
    assert json.loads(snapshot.recovery_schema)["required"] == ["path"]


def test_closure_coordinates_are_decoded_and_do_not_include_end_marker():
    from kernel_request_source_window import _render_ranges, project_source_window, source_revision
    text = "first\nlast without newline"
    revision = source_revision(text)
    page = project_source_window(_render_ranges(text.splitlines(keepends=True),
        [{"start_line": 2, "end_line": 2, "anchors": []}],
        path="a.py", revision=revision, total_lines=2), owner="test")
    body = {"messages": [{"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix"}, {"role": "user", "content": page},
        {"role": "user", "content": "opaque suffix"}]}
    snapshot = freeze_preparation(body, SimpleNamespace(), request_id="request", byte_cap=82026)
    slot, = snapshot.sources
    assert slot.payload == "last without newline"
    assert slot.identity.byte_start is None and slot.identity.byte_end is None
    assert slot.identity.lossy_span.presentation_version == "model-source-window/1"
    assert slot.page.encode()[slot.payload_start:slot.payload_start + len(slot.payload.encode())] == slot.payload.encode()


def test_unrendered_plan_unit_uses_frozen_sources_and_is_an_allocator_slot():
    from kernel_request_source_window import source_revision
    from kernel_working_set_allocator import FULL, DEFERRED
    from kernel_working_set_render import propose_slots
    source = "first\nplanned edit\nlast\n"
    revision = source_revision(source)
    frame = {"state": {"framefit5": {"owner_protocol": {"admitted_actions": ["read"]}}}}
    body = {"messages": [{"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps(frame)},
        {"role": "user", "content": "opaque suffix"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]}
    plan = {"units": [{"path": "a.py", "start_line": 2, "end_line": 2, "role": "plan"}],
            "source_contents": {}, "revisions": {"a.py": revision}}
    cache = {"a.py": {"content": source}}
    rt = SimpleNamespace(_request_source_closure=plan, _empirical_frame_sources=cache)
    snapshot = freeze_preparation(body, rt, request_id="request", byte_cap=82026)
    slot, = snapshot.sources
    assert slot.state == "PLAN" and slot.payload == "planned edit\n"
    assert (slot.identity.first_line, slot.identity.last_line) == (2, 2)
    offered, = propose_slots(snapshot)
    assert {mode for mode, _cost in offered.domain} == {FULL, DEFERRED}
    before = snapshot.snapshot_digest
    cache["a.py"]["content"] = "changed\n"
    plan["revisions"]["a.py"] = "sha256:" + "0" * 64
    plan["units"].clear()
    assert snapshot.snapshot_digest == before and slot.payload == "planned edit\n"


def test_incomplete_active_index_is_typed_instead_of_silently_omitted():
    from kernel_working_set_snapshot import SnapshotConstructionError
    from kernel_working_set_state import WorkingSetState
    state = WorkingSetState()
    state.overflow_count = 1
    with pytest.raises(SnapshotConstructionError) as failure:
        freeze_preparation({"messages": []}, SimpleNamespace(working_set_state=state),
                           request_id="request", byte_cap=82026)
    assert failure.value.reason == "ACTIVE_OBLIGATION_LIMIT"


def test_snapshot_canonicalizes_admitted_path_alias_without_changing_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    from kernel_read_receipts import capture_read
    from test_read_receipts import runtime
    rt, slot = runtime(tmp_path), page_slot()
    original_arguments = {"path": str(tmp_path / "a.py"), "start_line": 1, "end_line": 1}
    capture = capture_read(rt, path="a.py", metadata={"result_class": "OK",
        "revision": slot.identity.revision, "effective_range": {"start_line": 1, "end_line": 1},
        "byte_start": 0, "byte_end": len(slot.payload.encode())}, payload=slot.payload,
        original_page=slot.page, source="disk", arguments=original_arguments)
    body = {"messages": [{"role": "user", "content": "Task: fix"},
                         {"role": "user", "content": "opaque suffix"}]}
    snapshot = freeze_preparation(body, rt, request_id="request", byte_cap=82026)
    assert json.loads(snapshot.sources[0].recovery_arguments) == {
        "path": "a.py", "start_line": 1, "end_line": 1}
    assert json.loads(capture.recovery_arguments) == original_arguments


def test_shadow_prepare_crash_is_journaled_and_never_raises_into_legacy(monkeypatch):
    """The legacy wire owner must survive any shadow failure, not only typed ones."""
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    import kernel_working_set_prepare
    def explode(_snapshot, **_kwargs):
        raise RuntimeError("shadow bug")
    monkeypatch.setattr(kernel_working_set_prepare, "prepare_candidate", explode)
    body = {"messages": [{"role": "system", "content": "system"},
                         {"role": "user", "content": "Task: fix"}]}
    wire = json.dumps(body).encode()
    snapshot = PreparationSnapshot(wire, (page_slot(),), "request", 1, 82026)
    journal = Journal()
    assert journal_shadow(snapshot, wire, journal=journal, byte_cap=82026, prompt_limit=33_344) is None
    event, = journal.events
    assert event["blocked_reason"] == "PREPARATION_CONTRACT_FAILURE"
    assert event["error_type"] == "RuntimeError"
    assert "shadow bug" not in json.dumps(event)


def test_freeze_shadow_copies_hook_inputs_and_tolerates_odd_runtime_attributes(monkeypatch):
    """The hook hands freeze_shadow the live body; nothing in it may change or raise."""
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    body = {"messages": [{"role": "system", "content": "system"},
                         {"role": "user", "content": "Task: fix"},
                         {"role": "user", "content": "opaque suffix"}], "max_tokens": 50}
    before = json.dumps(body, sort_keys=True)
    rt = SimpleNamespace(frame_ceiling=None, working_set_state=None, read_capture_ledger=None)
    snapshot = freeze_shadow(body, rt, request_id="request")
    assert json.dumps(body, sort_keys=True) == before
    # A runtime without a usable frame ceiling still freezes; it just has no frame limit.
    assert isinstance(snapshot, PreparationSnapshot) and snapshot.frame_limit is None
    journal = Journal()
    journal_shadow(snapshot, json.dumps(body).encode(), journal=journal)
    assert json.dumps(body, sort_keys=True) == before
    assert len(journal.events) == 1 and journal.events[0]["kind"] == "working_set_shadow_plan"
