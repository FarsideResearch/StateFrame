"""bf03's complete 38-hit search has a cheaper, recoverable delivery."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from kernel_completion_owner import validate_json_schema
from kernel_measurement import AuditJournal
from kernel_navigation_pages import NavigationPage, search_repository
from kernel_request_diet import RequestTokenMeter
from kernel_request_source_window import source_revision
from kernel_working_set_live import prepare_live
from kernel_working_set_prepare import Blocked, PreparedCandidate, prepare_candidate
from kernel_working_set_proof import verify_certificate
from kernel_working_set_snapshot import freeze_preparation
from kernel_working_set_stop import WorkingSetSuspended

GREP_CALL_ID = "grep-call"
BYTE_CAP = 8_192


def search_body(text: str, arguments: dict[str, object]) -> dict[str, object]:
    frame = {"mode": "FRAME_ONLY", "owner_protocol": {"admitted_actions": ["grep"]}}
    schema = {"type": "object", "properties": {"pattern": {"type": "string"},
              "page_size": {"type": "integer"}, "cursor": {"type": "string"}},
              "required": ["pattern"], "additionalProperties": False}
    return {"model": "scripted", "max_tokens": 100, "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: find matching rows"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps({"state": {"framefit5": frame}})},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": GREP_CALL_ID, "type": "function", "function": {
                "name": "grep", "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "name": "grep", "tool_call_id": GREP_CALL_ID, "content": text},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"},
    ], "tools": [{"type": "function", "function": {"name": "grep", "parameters": schema}}]}


def search_receipt(page: NavigationPage) -> dict[str, object]:
    return {"kind": "grep", "intent": page.metadata["pattern"],
            "result_set_digest": page.metadata["result_set_digest"], "grep_page": {
                "provider_call_id": GREP_CALL_ID, "chars": len(page.text),
                "sha256": hashlib.sha256(page.text.encode()).hexdigest()}}


def tool_rows(wire: bytes) -> list[dict[str, object]]:
    return [row for row in json.loads(wire)["messages"] if row["role"] == "tool"]


def _prepare_search(body, receipt, plan, workspace, scratch):
    journal = AuditJournal(scratch, "bf03", {})
    capture = SimpleNamespace(request_id="bf03-request-22", journal=journal,
                              _capacity_retry_limits=SimpleNamespace(
                                  max_request_bytes=BYTE_CAP, max_prompt_tokens=33_344))
    runtime = SimpleNamespace(owner=SimpleNamespace(root=workspace, _receipts=[receipt]),
                              journal=journal, _request_source_closure=plan,
                              _request_token_meter=RequestTokenMeter(disabled=True))
    try:
        _, prepared = prepare_live(capture, runtime, body, dialect="llama.cpp", reasoning_mode="on")
    except WorkingSetSuspended as exc:
        suspension = exc.outcome()["suspension"]
        return {"status": suspension["reason_code"], "floor": suspension["floor_bytes"],
                "cap": suspension["cap"]}
    assert verify_certificate(prepared.wire, prepared.rendered.certificate)
    (reply,) = tool_rows(prepared.wire)
    lines = reply["content"].splitlines()
    return {"status": "PREPARED", "closure": capture.request_source_final_delivery["status"],
            "wire_bytes": len(prepared.wire), "cap": prepared.cap, "shown": len(lines) - 2,
            "page": lines}


def test_request_22_delivers_a_recoverable_prefix_of_complete_grep(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    workspace = tmp_path / "repo"
    workspace.mkdir()
    rows = [f"match_{n} " + "x" * 256 for n in range(38)]
    (workspace / "rows.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    path, source = "target.py", "value = 1\n"
    (workspace / path).write_text(source, encoding="utf-8")
    plan = {"paths": [path], "target_path": path, "source_contents": {path: source},
            "revisions": {path: source_revision(source)},
            "units": [{"path": path, "start_line": 1, "end_line": 1, "handle": "unit:target"}],
            "workspace_authorization": {"scope": str(workspace), "generation": "0"}}

    args = {"pattern": "match", "page_size": 38}
    page = search_repository(workspace, "match", requested_page_size=38)
    body = search_body(page.text, args)
    receipt = search_receipt(page)
    header, footer, *original = page.text.splitlines()
    assert "shown=38 total=38 offset=0 complete=true" in header and footer == "0 more; complete=true"
    assert len(original) == 38 and len(page.text.encode()) > BYTE_CAP
    one_hit = search_repository(workspace, "match", requested_page_size=1)
    assert one_hit.metadata["result_set_digest"] == receipt["result_set_digest"]
    control = search_body(one_hit.text, {**args, "page_size": 1})
    witness = _prepare_search(control, search_receipt(one_hit), plan, workspace, tmp_path / "control")
    assert witness["status"] == "PREPARED" and witness["closure"] == "COMPLETE", witness
    assert witness["shown"] == 1 and witness["page"] == one_hit.text.splitlines()

    report = {"actual": _prepare_search(body, receipt, plan, workspace, tmp_path / "actual"),
              "one_hit_witness": witness}
    result = report["actual"]
    assert result["status"] == "PREPARED", report
    assert result["closure"] == "COMPLETE" and result["wire_bytes"] <= result["cap"]
    assert 0 < result["shown"] <= 38
    shown = result["shown"]
    result_header, result_footer, *hits = result["page"]
    assert shown < 38 and hits == original[:shown]
    assert result_header == header.replace("shown=38 ", f"shown={shown} ", 1).replace(
        " complete=true", " complete=false", 1)
    cursor = result_footer.split("cursor=", 1)[1]
    assert cursor == f"nav1:{receipt['result_set_digest']}:{shown}"
    schema = body["tools"][0]["function"]["parameters"]
    assert not validate_json_schema(schema, {**args, "cursor": cursor})
    following = search_repository(workspace, "match", cursor=cursor, requested_page_size=1)
    assert following.text.splitlines()[2:] == original[shown:shown + 1]


def test_complete_grep_page_resumes_from_absolute_result_offset(tmp_path):
    # Carry an actual cursor in the request so the portable case has a trusted
    # result-set identity even without the field receipt used by the offset-zero replay.
    hits = [f"match_{n} " + "x" * 2120 for n in range(39)]
    (tmp_path / "rows.txt").write_text("\n".join(hits) + "\n", encoding="utf-8")
    first = search_repository(tmp_path, "match", requested_page_size=1)
    cursor = first.metadata["continuation"]
    last = search_repository(tmp_path, "match", cursor=cursor, requested_page_size=40)
    assert last.metadata["complete"] and last.metadata["shown"] == 38
    body = search_body(last.text, {"pattern": "match", "page_size": 40, "cursor": cursor})
    snap = freeze_preparation(body, SimpleNamespace(), request_id="final-search-page", byte_cap=16_384)
    result = prepare_candidate(snap)
    assert isinstance(result, PreparedCandidate), (result.reason, result.floor_bytes, result.cap)
    assert verify_certificate(result.wire, result.rendered.certificate) and len(result.wire) <= snap.byte_cap
    (reply,) = tool_rows(result.wire)
    header, footer, *shown = reply["content"].splitlines()
    assert 0 < len(shown) < 38 and "offset=1 complete=false" in header
    assert shown == last.text.splitlines()[2:2 + len(shown)]
    next_page = search_repository(tmp_path, "match", cursor=footer.split("cursor=")[1], requested_page_size=1)
    assert next_page.metadata["offset"] == 1 + len(shown)
    assert next_page.text.splitlines()[2] == last.text.splitlines()[2 + len(shown)]


def test_incomplete_grep_page_reuses_its_continuation_digest(tmp_path):
    (tmp_path / "rows.txt").write_text(("match " + "x" * 650 + "\n") * 10, encoding="utf-8")
    arguments = {"pattern": "match", "page_size": 8}
    page = search_repository(tmp_path, "match", requested_page_size=8)
    assert not page.metadata["complete"]
    body = search_body(page.text, arguments)
    snapshot = freeze_preparation(body, SimpleNamespace(), request_id="incomplete-page", byte_cap=2_000)
    result = prepare_candidate(snapshot)
    assert isinstance(result, PreparedCandidate)
    assert verify_certificate(result.wire, result.rendered.certificate)
    (reply,) = tool_rows(result.wire)
    _, footer, *hits = reply["content"].splitlines()
    assert 0 < len(hits) < 8
    assert hits == page.text.splitlines()[2:2 + len(hits)]
    cursor = footer.split("cursor=", 1)[1]
    assert cursor == f"nav1:{page.metadata['result_set_digest']}:{len(hits)}"
    following = search_repository(tmp_path, "match", cursor=cursor, requested_page_size=1)
    assert following.text.splitlines()[2] == page.text.splitlines()[2 + len(hits)]


def test_complete_grep_retains_scan_warning_and_trailing_annotation(tmp_path, monkeypatch):
    from test_grep_retained_rows import _open_trace

    (tmp_path / "rows.txt").write_text(("match " + "x" * 650 + "\n") * 8, encoding="utf-8")
    (tmp_path / "unreadable.txt").touch()
    _open_trace(monkeypatch, tmp_path, fail_names=frozenset({"unreadable.txt"}))
    page = search_repository(tmp_path, "match")
    assert page.metadata["complete"] and not page.metadata["scan_complete"]
    original_header, original_footer, *original = page.text.splitlines()
    annotation = "[host] preserve this note"
    body = search_body(page.text + "\n" + annotation, {"pattern": "match"})
    owner = SimpleNamespace(_receipts=[search_receipt(page)])
    snap = freeze_preparation(body, SimpleNamespace(owner=owner), request_id="warned-page", byte_cap=2_000)
    owner._receipts.clear()
    result = prepare_candidate(snap)
    assert isinstance(result, PreparedCandidate)
    assert verify_certificate(result.wire, result.rendered.certificate)
    (reply,) = tool_rows(result.wire)
    header, footer, *shown = reply["content"].splitlines()
    assert shown.pop() == annotation
    assert 0 < len(shown) < 8 and shown == original[:len(shown)]
    assert header.split(" scan_complete=")[1] == original_header.split(" scan_complete=")[1]
    cursor, warning = footer.split("cursor=", 1)[1].split(";", 1)
    assert warning == original_footer.split(";", 2)[2]
    following = search_repository(tmp_path, "match", cursor=cursor, requested_page_size=1)
    assert following.text.splitlines()[2] == original[len(shown)]


@pytest.mark.parametrize("missing_authority", [
    "receipt", "call", "page", "owner", "owner-actions", "cursor-property", "cursor-properties", "cursor-schema",
])
def test_complete_grep_requires_a_bound_receipt_and_callable_cursor(tmp_path, missing_authority):
    (tmp_path / "rows.txt").write_text(("match " + "x" * 650 + "\n") * 8, encoding="utf-8")
    page = search_repository(tmp_path, "match")
    body = search_body(page.text, {"pattern": "match"})
    receipt = search_receipt(page)
    owner = SimpleNamespace(_receipts=[receipt])
    if missing_authority == "receipt":
        owner._receipts.clear()
    elif missing_authority == "call":
        receipt["grep_page"]["provider_call_id"] = "different-call"
    elif missing_authority == "page":
        receipt["grep_page"]["sha256"] = "0" * 64
    elif missing_authority == "owner":
        body["messages"][2]["content"] = '[StateFrame]\n{"state":{"framefit5":{"mode":"FRAME_ONLY"}}}'
    elif missing_authority == "owner-actions":
        state = body["messages"][2]
        frame = json.loads(state["content"].split("\n", 1)[1])
        frame["state"]["framefit5"]["owner_protocol"]["admitted_actions"] = None
        state["content"] = "[StateFrame]\n" + json.dumps(frame)
    elif missing_authority == "cursor-properties":
        body["tools"][0]["function"]["parameters"]["properties"] = None
    else:
        fn = body["tools"][0]["function"]
        schema = fn["parameters"]
        properties = dict(schema["properties"])
        if missing_authority == "cursor-property":
            del properties["cursor"]
        else:
            properties["cursor"] = {"type": "string", "pattern": "^different-format:"}
        fn["parameters"] = {**schema, "properties": properties}

    snap = freeze_preparation(body, SimpleNamespace(owner=owner), request_id="closed-page", byte_cap=2_000)
    assert isinstance(prepare_candidate(snap), Blocked)
    assert json.loads(snap.fixed_body)["messages"][4]["content"] == page.text
