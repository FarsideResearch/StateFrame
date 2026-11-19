"""bf06 request 14. A resident native page is the only carrier of a selected unit."""
import json
from types import SimpleNamespace

from kernel_read_receipts import capture_read
from kernel_request_diet import RequestTokenMeter
from kernel_request_source_window import source_delivery_satisfied, source_revision
from kernel_source_delivery import deliver_source
from kernel_working_set_allocator import DEFERRED, FULL
from kernel_working_set_live import prepare_live
from kernel_working_set_proof import verify_certificate
from kernel_working_set_render import render_assignment
from kernel_working_set_snapshot import freeze_preparation
from test_read_receipts import runtime
from test_working_set_recovery import schema
from test_working_set_state import accept, submit

SOURCE_PATH = "sample.py"


def test_resident_read_page_proves_overlapping_closure_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
    rt = runtime(tmp_path)
    rt.owner = SimpleNamespace(root=tmp_path)
    rt._request_token_meter = RequestTokenMeter(disabled=True)
    source = "".join(f"value_{i} = {i}\n" for i in range(1, 435))
    file = tmp_path / SOURCE_PATH
    file.write_text(source, encoding="utf-8")
    args = {"path": SOURCE_PATH}
    page = deliver_source(source.encode(), SOURCE_PATH, 0, args=args)
    assert page.metadata["effective_range"]["end_line"] == 120
    capture_read(rt, path=SOURCE_PATH, metadata=page.metadata, payload=page.content,
                 original_page=page.text, source="disk", arguments=args)
    submit(rt)
    accept(rt)
    rt._request_source_closure = {
        "paths": [SOURCE_PATH], "revisions": {SOURCE_PATH: source_revision(source)},
        "source_contents": {SOURCE_PATH: source},
        "units": [{"path": SOURCE_PATH, "start_line": 1, "end_line": 5, "handle": "bf06-first-unit"},
                  {"path": SOURCE_PATH, "start_line": 7, "end_line": 7, "handle": "bf06-import"}],
        "workspace_authorization": {"scope": "workspace", "generation": "test-generation"},
    }
    body = {"model": "local", "messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "Task: repair"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps({"state": {"framefit5": {
            "mode": "FRAME_ONLY", "owner_protocol": {"admitted_actions": ["read"]}}}})},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": json.loads(
            schema("path", "start_line", "end_line", "max_lines", "max_chars", "expected_revision"))}}]}
    # Preserve the field assignment without depending on unrelated frame padding.
    # The exact 1..120 page fits; a second copy of 1..5 costs extra.
    frozen = freeze_preparation(body, rt, request_id="bf06", byte_cap=82_026)
    resident, unit, short_unit = frozen.sources
    assert (resident.state, unit.identity.first_line, unit.identity.last_line) == ("RESIDENT", 1, 5)
    budget_candidate = render_assignment(frozen, {resident.key: FULL, unit.key: DEFERRED, short_unit.key: FULL})
    cap = SimpleNamespace(request_id="bf06", journal=rt.journal, _capacity_retry_limits=SimpleNamespace(
        max_request_bytes=len(budget_candidate.wire), max_prompt_tokens=33344))
    output, prepared = prepare_live(cap, rt, body, dialect="llama.cpp", reasoning_mode="on")
    assert output == json.loads(prepared.wire)
    certificate = prepared.rendered.certificate
    assert verify_certificate(prepared.wire, certificate)
    assert dict(certificate.assignment)[unit.key] == DEFERRED.name
    assert source_delivery_satisfied(cap.request_source_final_delivery)
    assert len(prepared.wire) <= len(budget_candidate.wire)
    row = cap.request_source_final_delivery["paths"][SOURCE_PATH]
    assert row["status"] == "COMPLETE" and set(row["delivered_unit_handles"]) == {"bf06-first-unit", "bf06-import"}
