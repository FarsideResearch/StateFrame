"""Exercise recovery using physical capture, live reads and source rendering."""
from __future__ import annotations

import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from kernel_idle_governor import GovernorDecision, IdleGovernor
from kernel_measurement import ACTIVE_REQUEST, AuditJournal, RequestCapture
from kernel_pilot import ACTIVE_RUNTIME
from kernel_read_exec import dispatch_live_read, read_span_visible
from kernel_request_source_window import verify_model_source_window
from probes.react_repair import _read_span_currently_visible


def _runtime(root):
    runtime = SimpleNamespace(retained=None, step_ledger=None)
    runtime.owner = SimpleNamespace(root=root, workspace_revision=1, _receipts=[])
    runtime.owner.receipt = lambda kind, intent, fingerprint, result, **fields: (
        runtime.owner._receipts.append(dict(kind=kind, result_class=result, **fields)))
    runtime.journal = AuditJournal(root, "read-wire", {})
    return runtime


def _read(runtime, **args):
    path = runtime.owner.root / "module.py"
    return dispatch_live_read(
        runtime.owner.root, path, "module.py", "module.py", 0,
        args={"path": "module.py", **args}, measurement=runtime, step=1,
        bolt=runtime.owner, policy=None, resolved_path="module.py")


def _capture(runtime, messages, monkeypatch):
    import kernel_bounded_http

    wires = []

    def transport(wire, *_args, **_kwargs):
        wires.append(wire)
        return b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", transport)
    capture = RequestCapture(runtime.journal, 1, "snapshot")
    body = {"model": "scripted", "messages": messages, "tools": []}
    wire = json.dumps(body).encode()
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        capture._send_transport_with_retries(
            body, wire, "http://127.0.0.1:1/v1/chat/completions", "",
            admission=None, socket_timeout=5, deadline=time.monotonic() + 10,
            wall_timeout=5, backend_identity={}, request_extensions={})
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert wires == [wire]
    assert runtime._last_wire_messages_for_reads == json.loads(wire)["messages"]
    return capture


@pytest.mark.parametrize("subrange", [False, True])
def test_captured_read_page_covers_subranges_despite_new_annotations(tmp_path, monkeypatch, subrange):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    (tmp_path / "module.py").write_text("first = 1\nsecond = 2\nthird = 3\n")
    runtime = _runtime(tmp_path)
    status, text, _, _ = _read(runtime, start_line=1, end_line=3)
    assert status == "OK"
    _capture(runtime, [{"role": "tool", "content": text + "\nold annotation"}], monkeypatch)
    status, obs, _, delivery = _read(runtime, start_line=2 if subrange else 1, end_line=3)
    assert status == "OK"
    assert _read_span_currently_visible(
        runtime, tmp_path, tmp_path / "module.py", delivery, obs + "\nnew annotation") is True
    governor = IdleGovernor(n1=1, n2=2)
    for turn in range(3):
        governor.should_allow({"kind": "read", "path": "module.py", "turn": turn,
                               "content_digest": delivery["requested_bytes_sha256"]})
    visible = _read_span_currently_visible(
        runtime, tmp_path, tmp_path / "module.py", delivery, obs)
    assert governor.should_allow({"kind": "read", "path": "module.py", "turn": 4,
                                  "content_digest": delivery["requested_bytes_sha256"],
                                  "visible": visible}) == GovernorDecision.DENY
    assert governor.recovery_reads == 0


def test_real_partial_window_covers_only_its_columns(tmp_path, monkeypatch):
    from kernel_frame_source_snapshot import _current_window, _partial_read_window
    from kernel_source_delivery import deliver_source

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    source = "abcdefghij" * 200 + "\n"
    (tmp_path / "module.py").write_text(source)
    runtime = _runtime(tmp_path)
    args = {"start_line": 1, "column": 7, "max_chars": 800}
    status, obs, _, delivery = _read(runtime, **args)
    excerpt = deliver_source(source.encode(), "module.py", 0, args=args).content
    assert status == "OK" and excerpt
    assert "sha256:" + hashlib.sha256(excerpt.encode()).hexdigest() == delivery["requested_bytes_sha256"]
    full = _current_window(tmp_path, "module.py", 1, 1, delivery["revision"])
    assert full is not None
    partial = _partial_read_window(
        tmp_path, "module.py", 1, 1, delivery["revision"],
        {"content": excerpt, "content_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
         **delivery["effective_range"]}, full)
    assert partial is not None
    assert verify_model_source_window(partial[0], path="module.py", source=source)
    _capture(runtime, [{"role": "user", "content": partial[0]}], monkeypatch)
    assert _read_span_currently_visible(
        runtime, tmp_path, tmp_path / "module.py", delivery, obs + "\nchanged annotation") is True
    wider = {**delivery, "effective_range": {**delivery["effective_range"], "start_column": 0}}
    assert read_span_visible(runtime._last_wire_messages_for_reads, "module.py", wider) is False
    stale = {**delivery, "revision": "sha256:" + "0" * 64}
    assert read_span_visible(runtime._last_wire_messages_for_reads, "module.py", stale) is False


def test_adjacent_host_pages_cover_union_but_never_a_missing_line(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    (tmp_path / "module.py").write_text("first = 1\nsecond = 2\nthird = 3\n")
    runtime = _runtime(tmp_path)
    messages = []
    for line in (1, 2):
        status, text, _, _ = _read(runtime, start_line=line, end_line=line)
        assert status == "OK"
        messages.append({"role": "tool", "content": text})
    _capture(runtime, messages, monkeypatch)
    for end, expected in ((2, True), (3, False)):
        status, obs, _, delivery = _read(runtime, start_line=1, end_line=end)
        assert status == "OK"
        assert _read_span_currently_visible(
            runtime, tmp_path, tmp_path / "module.py", delivery, obs) is expected
    assert read_span_visible(messages, ".module.py", delivery) is False


def test_recovered_key_cannot_escape_as_novel_on_next_visible_read():
    governor = IdleGovernor(n1=1, n2=2)
    for turn in range(3):
        governor.should_allow({"kind": "grep", "turn": turn})
    assert governor.governed
    action = {"kind": "read", "path": "new.py", "content_digest": "delivered"}
    assert governor.should_allow({**action, "visible": False, "turn": 4}) == GovernorDecision.ALLOW
    assert governor.should_allow({**action, "visible": True, "turn": 5}) == GovernorDecision.DENY


def test_empty_live_delivery_cannot_be_classified_as_recovery(tmp_path, monkeypatch):
    (tmp_path / "module.py").write_text("a\n")
    runtime = _runtime(tmp_path)
    _capture(runtime, [{"role": "user", "content": "source was evicted"}], monkeypatch)
    status, obs, _, delivery = _read(runtime, start_line=1, column=2)
    assert status == "OK" and delivery["requested_bytes_delivered"] == 0
    assert _read_span_currently_visible(
        runtime, tmp_path, tmp_path / "module.py", delivery, obs) is None


def test_native_queue_restore_retains_originating_physical_wire(tmp_path, monkeypatch):
    from kernel_native_restart import empty, prepare_restore, restore_batch
    from kernel_native_tools import NativeToolBatch

    (tmp_path / "module.py").write_text("value = 1\n")
    runtime = _runtime(tmp_path)
    capture = _capture(runtime, [{"role": "user", "content": "source was evicted"}], monkeypatch)
    records = [json.loads(line) for line in runtime.journal.path.read_text().splitlines()]
    submitted = next(row for row in records if row["kind"] == "request_submitted")
    calls = [{"id": "read-1", "type": "function", "function": {
        "name": "read", "arguments": '{"path":"module.py"}'}}]
    response = runtime.journal.emit(
        "response_received", request_id=capture.request_id,
        transport_attempt_id=submitted["event_id"],
        response={"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": calls}}]})
    runtime.journal.emit("native_batch_received", request_id=capture.request_id,
                         origin_snapshot_id="snapshot", origin_response_event_id=response["event_id"],
                         origin_binding={}, calls=calls, batch_size=1)
    records = [json.loads(line) for line in runtime.journal.path.read_text().splitlines()]
    resumed = SimpleNamespace(journal=runtime.journal, native_batch=NativeToolBatch(), binding=lambda: {})
    resumed.journal.manifest["native_tool_batches"] = True
    assert prepare_restore(resumed, {"native": empty(), "binding": {}}, records, {"sequence": 0})
    restore_batch(resumed)
    # Select the retained proposal through the native queue without resampling.
    resumed.owner, resumed.retained, resumed.step_ledger = runtime.owner, None, None
    active = ACTIVE_REQUEST.set(RequestCapture(resumed.journal, 2, "resumed-snapshot"))
    token = ACTIVE_RUNTIME.set(resumed)
    try:
        _, selected = resumed.native_batch.next(
            lambda *_args, **_kwargs: pytest.fail("queued resume asked the provider again"), [])
        assert len(selected) == 1
        status, obs, _, delivery = _read(resumed, **json.loads(selected[0]["function"]["arguments"]))
    finally:
        ACTIVE_RUNTIME.reset(token)
        ACTIVE_REQUEST.reset(active)
    assert status == "OK"
    assert _read_span_currently_visible(
        resumed, tmp_path, tmp_path / "module.py", delivery, obs) is False
    assert resumed._last_wire_messages_for_reads == runtime._last_wire_messages_for_reads
    assert len([row for row in records if row["kind"] == "request_submitted"]) == 1


@pytest.mark.host_sandbox
def test_work_loop_empty_reads_never_receive_recovery_admission(tmp_path):
    from _journal import measurement_events
    from test_shell_e2e import finish, run_script, step

    def prepare(repo):
        lines = ["# filler " + "x" * 100 + "\n"] * 3000
        lines[2499] = "a\n"
        (repo / "pkg/empty_span.py").write_text("".join(lines))

    run, _ = run_script(tmp_path, [
        step("read", path="pkg/empty_span.py", start_line=2500, column=2, end_line=2500),
        step("read", path="pkg/empty_span.py", start_line=2500, column=2, end_line=2500),
        *finish()], prepare=prepare)
    events = measurement_events(run)
    reads = [row["receipt"] for row in events if row["kind"] == "action_receipt"
             and row.get("receipt", {}).get("kind") == "read"]
    assert len(reads) == 2
    assert all(row["result_class"] == "OK" for row in reads)
    assert all(row["effective_range"]["byte_start"] == row["effective_range"]["byte_end"]
               for row in reads)
    recoveries = [row for row in events
                  if row["kind"] == "read_recovery_admitted"]
    assert recoveries == []


@pytest.mark.host_sandbox
def test_work_loop_recovers_unshown_source_once_then_sees_repeated_bytes(tmp_path):
    from _journal import measurement_events
    from test_shell_e2e import finish, run_script, step

    def prepare(repo):
        (repo / "pkg/other.py").write_text("".join(
            f"# line {line}: " + "x" * 100 + "\n" for line in range(3000)))

    run, _ = run_script(tmp_path, [
        step("read", path="pkg/other.py", start_line=2500, end_line=2502),
        step("read", path="pkg/other.py", start_line=2500, end_line=2502),
        step("read", path="pkg/other.py", start_line=2501, end_line=2502),
        *finish()], prepare=prepare)
    recoveries = [row for row in measurement_events(run)
                  if row["kind"] == "read_recovery_admitted"]
    assert len(recoveries) == 1
    assert recoveries[0]["path"] == "pkg/other.py"
    assert recoveries[0]["recovery_reads"] == 1
