"""Exact shell source receipts must obey the same admission floor as read()."""
import json
from types import SimpleNamespace

import pytest

from kernel_final_wire_allocation import build_qualified_request
from kernel_lean import bound_tool_result
from kernel_pilot import ACTIVE_RUNTIME
from kernel_working_set_proof import verify_certificate
from kernel_working_set_state import state_for
from test_read_admission_mixed_batch import CAP, _runtime
from kernel_shell_read_receipts import execute_with_read_audit


@pytest.mark.parametrize("raw, total_lines", [
    (b"one\ntwo\n", 2),
    (b"one\ntwo", 2),
    (b"one\r\ntwo\r\n", 2),
    (b"one\rtwo\n", 1),
])
def test_shell_admission_metadata_counts_newline_delimited_lines(
        tmp_path, monkeypatch, raw, total_lines):
    """Count shell source lines without splitting the whole snapshot into objects."""
    import kernel_read_admission as admission

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "app.py").write_bytes(raw)
    rt, _, _, _ = _runtime(root, monkeypatch, lean=True)
    reply = SimpleNamespace(observation="$ cat app.py\nexit 0\n" + raw.decode(),
                            exit_code=0, fatal=False)
    tool = SimpleNamespace(root=root, run_dir=tmp_path, shell_state=None,
                           _execute=lambda *a: reply)
    pages = []

    def admit(runtime, **kwargs):
        pages.append(kwargs["source"])
        return kwargs["source"]

    monkeypatch.setattr(admission, "admit", admit)
    token = ACTIVE_RUNTIME.set(rt)
    try:
        assert execute_with_read_audit(tool, {"command": "cat app.py"}, "lines") is reply
    finally:
        ACTIVE_RUNTIME.reset(token)
    page, = pages
    assert page.metadata["total_lines"] == total_lines
    assert page.metadata["effective_range"] == {"start_line": 1, "end_line": total_lines}
    assert page.content == raw.decode()


@pytest.mark.parametrize("size", [100, 100_000], ids=["admitted", "opaque-over-cap"])
def test_shell_capture_is_admitted_or_remains_opaque(tmp_path, monkeypatch, size):
    root = tmp_path / "workspace"
    root.mkdir()
    raw = ("x" * 99 + "\n") * (size // 100)
    (root / "app.py").write_text(raw)
    rt, capture, body, plan = _runtime(root, monkeypatch, lean=True)
    call = {"id": "shell-read", "type": "function", "function": {
        "name": "shell", "arguments": json.dumps({"command": "cat app.py"})}}
    rt.native_batch = SimpleNamespace(pending=[], inflight=call, origin="shell-admission")
    expected = SimpleNamespace(observation="$ cat app.py\nexit 0\n" + raw, exit_code=0, fatal=False)
    # Exercise the audit boundary with an exact, unbounded producer reply.
    tool = SimpleNamespace(root=root, run_dir=tmp_path, shell_state=None,
                           _execute=lambda *a: expected)
    token = ACTIVE_RUNTIME.set(rt)
    try:
        reply = execute_with_read_audit(tool, {"command": "cat app.py"}, call["id"])
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert reply is expected and raw in reply.observation
    captured = getattr(rt, "read_capture_ledger", None)
    pages = captured.all() if captured is not None else ()
    if size > CAP - 1000:
        assert not pages
        assert not state_for(rt).active_obligations()
        rows = [json.loads(line) for line in rt.journal.path.read_bytes().splitlines()]
        assert any(row["kind"] == "read_capture_skipped" and row["reason"] == "admission"
                   for row in rows)
        assert not any(row["kind"] == "read_capture_failed" for row in rows)
    else:
        assert len(pages) == 1
        import kernel_read_admission as admission
        assert admission._state(rt, None).admitted_count == 1
        return
    # Drive the ordinary bounded observation through the real next-frame seal.
    from kernel_request_evidence_closure import SpanNeed
    from kernel_request_source_window import source_revision
    # The qualification fixture requires a concrete closure to verify.
    (root / "helper.py").write_text("helper = 1\n")
    plan["paths"].append("helper.py")
    plan["source_contents"]["helper.py"] = "helper = 1\n"
    plan["revisions"]["helper.py"] = source_revision("helper = 1\n")
    unit = SpanNeed("helper.py", 1, 1, "read", "fixture closure", 0).as_dict()
    unit["unit_handle"] = "helper"
    plan["units"].append(unit)
    rt.native_batch = None
    body["messages"][-1:-1] = [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": bound_tool_result(reply.observation)}]
    record = build_qualified_request(capture, rt, body, endpoint="http://local/v1", key="")
    assert len(record.wire) <= CAP
    assert verify_certificate(record.wire, capture._working_set_certificate)
    record.validate(capture, rt)
