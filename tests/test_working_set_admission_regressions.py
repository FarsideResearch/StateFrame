"""Read admission must price the allocator's complete wire and close refused batches."""
from __future__ import annotations

from types import SimpleNamespace
import json

import kernel_read_admission as admission
import pytest
from kernel_working_set_prepare import PreparedCandidate, prepare_candidate
from kernel_working_set_snapshot import freeze_preparation
from test_read_receipts_admission import read, runtime


def envelope(padding=4000):
    return {"model": "local", "messages": [
        {"role": "system", "content": "rules" + "x" * padding},
        {"role": "user", "content": "Task: inspect"},
        {"role": "user", "content": '[StateFrame]\n{"state":{"framefit5":{}}}'},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"}], "tools": [], "max_tokens": 100}


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")


@pytest.mark.parametrize("line", ["λ" * 100, '"\\\t' * 100, "😀" * 100],
                         ids=["unicode", "json-escaping", "utf8-expansion"])
def test_admitted_page_has_feasible_real_serialized_candidate(tmp_path, monkeypatch, line):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 12_000)
    monkeypatch.setattr("kernel_request_budget.HARD_MAX_REQUEST_BYTES", 12_000)
    rt = runtime(tmp_path)
    del rt.working_set_fixed_bytes
    rt.owner = SimpleNamespace(root=tmp_path)
    body = envelope()
    admission.note_sealed_request(rt, body["messages"], body["tools"])
    (tmp_path / "unicode.py").write_text((line + "\n") * 25)
    result, metadata = read(rt, tmp_path, "unicode.py", {
        "path": "unicode.py", "max_lines": 1000, "max_chars": 64000})
    assert result[0] == "OK"
    assert metadata["truncated"] and metadata["next_cursor"]
    snapshot = freeze_preparation(body, rt, request_id="next", byte_cap=12_000)
    prepared = prepare_candidate(snapshot)
    assert isinstance(prepared, PreparedCandidate), prepared
    assert prepared.total_bytes <= 12_000
    from kernel_working_set_live import read_admission_floor
    state = rt._read_admission_state
    pending_floor = read_admission_floor(state.snapshot, state.pages)
    assert state.pending_bytes == len(pending_floor.wire) - state.floor_bytes
    assert admission.remaining_allowance(rt, None) == 12_000 - len(pending_floor.wire)


def test_heterogeneous_batch_refuses_small_read_after_large_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 1000)
    rt = runtime(tmp_path, fixed=700)
    (tmp_path / "large.py").write_text("x" * 2000 + "\n")
    (tmp_path / "small.py").write_text("a\n")
    results = [read(rt, tmp_path, name)[0][0] for name in ("large.py", "small.py")]
    assert results == [admission.REFUSED_ADMISSION] * 2
    events = [e for e in rt.journal.events if e["kind"] == "read_result"]
    assert [e["status"] for e in events] == results
    rt._frame_budget_request_count += 1
    result, metadata = read(rt, tmp_path, "small.py")
    assert result[0] == "OK" and metadata["effective_range"]


def test_protected_minimum_reduces_fresh_allowance(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 12_000)
    monkeypatch.setattr("kernel_request_budget.HARD_MAX_REQUEST_BYTES", 12_000)
    rt = runtime(tmp_path)
    rt.owner = SimpleNamespace(root=tmp_path)
    (tmp_path / "protected.py").write_text("# protected " + "p" * 5000 + "\n")
    assert read(rt, tmp_path, "protected.py")[0][0] == "OK"
    obligation, = rt.working_set_state.active_obligations()
    obligation.recovery_pin = True
    rt._frame_budget_request_count += 1
    del rt.working_set_fixed_bytes
    body = envelope(1000)
    admission.note_sealed_request(rt, body["messages"], body["tools"])
    (tmp_path / "fresh.py").write_text(("# " + "x" * 100 + "\n") * 60)
    result, metadata = read(rt, tmp_path, "fresh.py", {"path": "fresh.py", "max_lines": 1000})
    assert result[0] == "OK" and metadata["truncated"]
    prepared = prepare_candidate(freeze_preparation(body, rt, request_id="next", byte_cap=12_000))
    assert isinstance(prepared, PreparedCandidate), prepared
    assert prepared.total_bytes <= 12_000


def test_refusal_resets_on_new_origin_of_reused_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 1000)
    rt = runtime(tmp_path, fixed=700)
    rt.native_batch = SimpleNamespace(origin="first", pending=[])
    (tmp_path / "large.py").write_text("x" * 2000 + "\n")
    (tmp_path / "tiny.py").write_text("a\n")
    assert read(rt, tmp_path, "large.py")[0][0] == admission.REFUSED_ADMISSION
    # Frame counters can be rewritten while draining a batch. That cannot open
    # admission again; the native origin is the batch authority.
    rt._frame_budget_request_count += 1
    assert read(rt, tmp_path, "tiny.py")[0][0] == admission.REFUSED_ADMISSION
    rt.native_batch.origin = "second"
    assert read(rt, tmp_path, "tiny.py")[0][0] == "OK"


def test_dormant_native_batch_still_resets_at_request_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 1000)
    rt = runtime(tmp_path)
    rt.native_batch = SimpleNamespace(origin=None, pending=[])
    (tmp_path / "tiny.py").write_text("a\n")
    assert read(rt, tmp_path, "tiny.py")[0][0] == "OK"
    rt._frame_budget_request_count += 1
    assert admission.remaining_allowance(rt, None) == 1000


def _qualify_resident(root, rt, capture, body, plan):
    """A real sealed request with source debt accepted before the two fresh reads."""
    from kernel_final_wire_allocation import build_qualified_request
    from kernel_read_receipts import capture_read
    from kernel_request_evidence_closure import SpanNeed
    from kernel_source_delivery import deliver_source
    from kernel_completion_owner import establish_lean_request_action_contract
    from kernel_request_seal import OwnerPresentation
    import kernel_working_set_state as ws

    properties = body["tools"][0]["function"]["parameters"]["properties"]
    properties.update({name: {"type": "integer"} for name in
                       ("start_line", "end_line", "max_lines", "max_chars")})
    properties.update({name: {"type": "string"} for name in ("cursor", "expected_revision")})
    capture.request_action_contract = establish_lean_request_action_contract(
        rt, body["messages"], body["tools"], request_id=capture.request_id)
    capture.owner_presentation = OwnerPresentation.capture(
        rt, capture.request_action_contract, body["messages"])
    text = ("# " + "a" * 97 + "\n") * 240
    (root / "first.py").write_text(text)
    args = {"path": "first.py", "max_lines": 240, "max_chars": 64000}
    page = deliver_source(text.encode(), "first.py", 0, args=args)
    captured = capture_read(rt, path="first.py", metadata=page.metadata,
        payload=page.content, original_page=page.text, source="disk", arguments=args)
    plan["paths"].append("first.py")
    unit = SpanNeed("first.py", 1, 240, "read", "resident range", 0).as_dict()
    unit["unit_handle"] = "resident:1-240"
    plan["units"].append(unit)
    plan["revisions"]["first.py"] = captured.identity.revision
    plan["source_contents"]["first.py"] = text
    record = build_qualified_request(capture, rt, body, endpoint="http://local/v1", key="")
    ws.observe_submission(rt, capture, record.wire)
    ws.observe_response(rt, capture, disposition="accepted", response_id="resident-seed")
    resident, = rt.working_set_state.active_obligations()
    assert resident.state == "RESIDENT"
    return record


@pytest.mark.parametrize("freeze_fails", [False, True], ids=["exact-floor", "failed-floor"])
def test_two_reads_reserve_complete_fixed_layout(tmp_path, monkeypatch, freeze_fails):
    """Request 6: an estimated envelope fits both pages, the actual floor does not."""
    from kernel_read_receipts import begin_read_attempt
    from kernel_read_exec import execute_read
    from kernel_source_delivery import deliver_source
    from kernel_working_set_live import read_admission_floor, read_admission_slot
    from test_read_admission_qualification import seal_fixture
    import kernel_working_set_live as live

    capture, rt, body, plan, journal, tools = seal_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    cap = 82_026
    monkeypatch.setattr(admission, "request_byte_cap", lambda: cap)
    monkeypatch.setattr("kernel_request_budget.HARD_MAX_REQUEST_BYTES", cap)
    # Keep the real typed frame/owner protocol, schemas and native exchange.
    # The sealed estimate excludes mandatory source added by the next layout.
    body["messages"][0]["content"] += "x" * 45_000
    _qualify_resident(tmp_path, rt, capture, body, plan)
    admission.note_sealed_request(rt, body["messages"], tools)
    contents = {"first.py": ("# " + "a" * 97 + "\n") * 240,
                "second.py": ("# " + "b" * 97 + "\n") * 60}
    calls = [{"id": name, "type": "function", "function": {"name": "read",
              "arguments": json.dumps({"path": name, "max_lines": 60})}} for name in contents]
    rt.native_batch = SimpleNamespace(pending=calls, origin="two-due-reads")
    with monkeypatch.context() as guard:
        def already_qualified(*args, **kwargs):
            pytest.fail("admission must preserve the complete qualified envelope")
        guard.setattr("kernel_request_candidate.render_candidate", already_qualified)
        baseline = live.read_admission_snapshot(rt,
            [{"role": "user", "content": "[StateFrame]\n{}"}], byte_cap=cap)
    assert "x" * 45_000 in json.loads(baseline.fixed_body)["messages"][0]["content"]
    # Reading first.py can promote it to the next edit site, making its older
    # resident range FULL-only too. Price that dependent protection before capture.
    from dataclasses import replace
    baseline = replace(baseline, protected_paths=(), deferred_paths=("docs/context.md",))
    from kernel_working_set_allocator import DEFERRED
    from kernel_working_set_render import domain
    assert any(slot.state == "RESIDENT" and DEFERRED in domain(baseline, slot)
               for slot in baseline.sources)
    proposals = [(read_admission_slot(deliver_source(text.encode(), name, 0,
                    args={"path": name, "max_lines": 60}), path=name,
                    arguments={"path": name, "max_lines": 60}, ordinal=i + 2), name)
                 for i, (name, text) in enumerate(contents.items())]
    assert len(read_admission_floor(baseline).wire) < cap
    over_cap = read_admission_floor(baseline, proposals)
    assert len(over_cap.wire) > cap
    assert sum(slot.state == "DUE" for slot in over_cap.certificate.snapshot.sources) == 2
    assert set(contents) <= set(over_cap.certificate.snapshot.protected_paths)
    revisions = dict(over_cap.certificate.snapshot.current_revisions)
    assert all(revisions.get(slot.identity.path) == slot.identity.revision for slot, _ in proposals)
    facts = "\n".join(row["content"] for row in json.loads(over_cap.wire)["messages"]
                      if str(row.get("content", "")).startswith("[Host fact]\n"))
    assert all(f"{slot.identity.path} lines {slot.identity.first_line}-{slot.identity.last_line} "
               f"revision {slot.identity.revision}" in facts for slot, _ in proposals)
    assert dict(over_cap.section_breakdown)["host_facts"] > 0
    assert dict(over_cap.section_breakdown)["deferred_notices"] > 0
    assert sum(admission._wire_bytes(deliver_source(text.encode(), name, 0,
        args={"path": name, "max_lines": 60}).text) for name, text in contents.items()) < (
            cap - admission.sealed_fixed_bytes(rt))

    def snapshot(*args, **kwargs):
        if freeze_fails:
            raise live.ReadAdmissionUnavailable("PREPARATION_CONTRACT_FAILURE")
        return baseline

    monkeypatch.setattr(live, "read_admission_snapshot", snapshot)
    replies = []
    for name, text in contents.items():
        (tmp_path / name).write_text(text)
        args = {"path": name, "max_lines": 60}
        scope = begin_read_attempt(rt, args, call_id=name)
        metadata = {}
        result = execute_read(tmp_path, tmp_path / name, name, name, 0, args=args,
                              measurement=rt, step=1, delivery=metadata)
        scope.finish(result[0])
        replies.append((result, metadata))
    state = admission._state(rt, None)
    if freeze_fails:
        assert [r[0] for r, _ in replies] == [admission.REFUSED_ADMISSION] * 2
        assert state.pending_bytes == 0 and state.pages == []
    else:
        assert replies[0][0][0] == "OK"
        assert state.pages
        assert read_admission_floor(baseline, state.pages, require_feasible=True) is not None
    for result, metadata in replies:
        if result[0] == admission.REFUSED_ADMISSION or metadata.get("truncated"):
            assert metadata["next_cursor"] and metadata["next_cursor"] in result[1]
