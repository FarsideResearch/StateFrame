"""Read capture identity, dispatch completeness, and observational-only parity."""
from __future__ import annotations

import ast
import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from collections.abc import Mapping

import pytest

import kernel_read_exec as reads
import kernel_read_receipts as receipts
from kernel_measurement import canonical_bytes, resolve_journal_blob_refs
from kernel_native_tools import NativeToolBatch
from stateframe import runner

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src/stateframe/engine"


class Journal:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.events = []

    def emit(self, kind, **fields):
        row = dict(kind=kind, **fields)
        self.events.append(row)
        return {**row, "event_id": f"event-{len(self.events)}"}


def runtime(tmp_path):
    return SimpleNamespace(journal=Journal(tmp_path / "audit/events.jsonl"),
                           retained=None, step_ledger=None, _frame_budget_request_count=3)


def read(rt, root, args=None, module=reads, messages=None):
    return module.execute_read(root, root / "source.txt", "source.txt", "source.txt", 0,
                               args=args or {"path": "source.txt"}, measurement=rt,
                               step=1, delivery={}, visible_messages=messages)


@pytest.fixture(autouse=True)
def switch(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")


def test_switch_exact_value(monkeypatch):
    for value in ("off", "ON", " on ", "1", ""):
        monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", value)
        assert not receipts.enabled()
    monkeypatch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR")
    assert not receipts.enabled()


def test_cache_live_identity_and_blobs(tmp_path):
    (tmp_path / "source.txt").write_text("alpha\n\nbeta\n")
    rt = runtime(tmp_path)
    first = read(rt, tmp_path)
    second = read(rt, tmp_path)
    entry = next(iter(rt._read_contract_source_cache.values()))
    entry.content = None  # Exercise the page-only cache, not the file-byte cache.
    third = read(rt, tmp_path)
    captures = rt.read_capture_ledger.all()
    assert len(captures) == 3
    assert [c.source for c in captures] == ["disk", "file_cache", "page_cache"]
    assert captures[0].identity == captures[1].identity == captures[2].identity
    assert first[:2] == second[:2] == third[:2]
    assert captures[0].capture_index == 3 and captures[0].due_request == 4
    assert rt.read_capture_ledger.by_identity(captures[0].identity) == captures
    c = captures[0]
    text = "alpha\n\nbeta\n"
    logical_size = len(canonical_bytes(text))
    assert resolve_journal_blob_refs({"$blob": c.payload_blob_ref, "bytes": logical_size},
                                     rt.journal.path) == text
    assert c.identity.payload_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert c.line_digests == receipts.line_digests_for(text)
    assert receipts.line_digest("alpha") == hashlib.blake2b(b"alpha", digest_size=8).hexdigest()
    with pytest.raises(FrozenInstanceError):
        c.capture_index = 99


@pytest.mark.parametrize("lossy", [False, True])
def test_different_columns_do_not_collide(tmp_path, lossy):
    from kernel_source_delivery import deliver_source
    rt = runtime(tmp_path)
    captures = []
    for col in (0, 2):
        # The source delivery contract has a 512-character minimum; use a
        # physically valid clipped page to isolate the identical substring.
        page = deliver_source(b"abab", "source.txt", 0,
                              args={"path": "source.txt", "column": col})
        meta = copy.deepcopy(page.metadata)
        # Legacy lossy receipts remain readable, but live reads now refuse invalid UTF-8.
        meta["decoded_with_replacement"] = lossy
        meta["effective_range"]["end_column"] = col + 2
        if not meta["decoded_with_replacement"]:
            meta["byte_end"] = meta["byte_start"] + 2
        captures.append(receipts.capture_read(rt, path="source.txt", metadata=meta,
            payload="ab", original_page="ab", source="disk", arguments={"column": col}))
    assert captures[0].identity != captures[1].identity


def test_lossy_offsets_are_never_decoded_bytes(tmp_path):
    from kernel_source_delivery import deliver_source
    rt = runtime(tmp_path)
    page = deliver_source(b"x\n", "source.txt", 0, args={"path": "source.txt"})
    metadata = {**page.metadata, "decoded_with_replacement": True}
    capture = receipts.capture_read(rt, path="source.txt", metadata=metadata,
        payload=page.content, original_page=page.text, source="disk", arguments={})
    identity = capture.identity
    assert identity.byte_start is identity.byte_end is None
    assert identity.lossy_span == receipts.DecodedSpan(0, 2, "read-page/1")
    assert capture.receipt().lossy_span == identity.lossy_span


@pytest.mark.parametrize("setting", ["off", "on"])
def test_lossy_read_keeps_base_delivery_and_decoded_capture(tmp_path, monkeypatch, setting):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", setting)
    (tmp_path / "source.txt").write_bytes(b"\xff\nx\n")
    rt = runtime(tmp_path)
    scope = receipts.begin_read_attempt(rt, {"path": "source.txt"})
    status, observation, _ = read(rt, tmp_path)
    receipts.finish_pending(rt, status=status)
    # 0.6.23 delivers invalid UTF-8 with replacement. Keep that wire contract;
    # the audit must describe decoded columns without inventing raw offsets.
    assert status == "OK"
    assert "\ufffd" in observation and "decoded_with_replacement=true" in observation
    if setting == "on":
        capture, = rt.read_capture_ledger.all()
        assert capture.identity.byte_start is capture.identity.byte_end is None
        assert capture.identity.lossy_span is not None
    else:
        assert not hasattr(rt, "read_capture_ledger")
    outcomes = [row for row in rt.journal.events if row["kind"] == "read_result"]
    assert [row["status"] for row in outcomes] == (["OK"] if setting == "on" else [])
    if setting == "off":
        assert scope is None
    else:
        assert scope.finished


@pytest.mark.parametrize("change", [
    {"first_line": 0}, {"last_line": 0}, {"byte_start": None},
    {"byte_end": -1}, {"lossy_span": receipts.DecodedSpan(0, 1, "read-page/1")},
    {"path": "../escape"}, {"revision": "f" * 64},
])
def test_constructor_rejects_invalid_identity(change):
    args = dict(path="source.txt", revision="sha256:" + "a" * 64,
                first_line=1, last_line=1, byte_start=0, byte_end=2,
                payload_sha256="b" * 64, lossy_span=None)
    with pytest.raises(ValueError):
        receipts.ReceiptIdentity(**{**args, **change})


@pytest.mark.parametrize("error,status", [(KeyboardInterrupt(), "INTERRUPTED"),
    (TimeoutError(), "DEADLINE"), (PermissionError(), "PERMISSION"),
    (FileNotFoundError(), "MISSING"), (RuntimeError(), "ERROR")])
def test_interruption_finishes_once_without_capture(tmp_path, error, status):
    rt = runtime(tmp_path)
    with pytest.raises(type(error)):
        with receipts.begin_read_attempt(rt, {"path": "source.txt"}) as scope:
            raise error
    receipts.finish_pending(rt)
    scope.finish()
    assert [row["kind"] for row in rt.journal.events] == ["read_attempt", "read_result"]
    assert rt.journal.events[-1]["status"] == status
    assert "identity" not in rt.journal.events[-1]
    assert "payload_blob_ref" not in rt.journal.events[-1]


def _production_read_route():
    tree = ast.parse((ENGINE / "probes/react_repair.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and ast.unparse(node.test) == "line.startswith('read ')"):
            assert any(isinstance(n, ast.Try) for n in node.body)
            loop = ast.For(target=ast.Name(id="_", ctx=ast.Store()),
                iter=ast.Tuple(elts=[ast.Constant(1)], ctx=ast.Load()),
                body=node.body, orelse=[])
            return compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])),
                           "production_read_route", "exec")
    raise AssertionError("production read boundary missing")


@pytest.mark.parametrize("branch", ["ambiguous", "policy", "hook", "missing", "ok", "interrupt"])
def test_real_text_read_dispatch_guards(tmp_path, branch):
    rt = runtime(tmp_path)
    (tmp_path / "source.txt").write_text("alpha\n")
    if branch == "missing":
        args = {"path": "missing.txt"}
    else:
        args = {"path": "source.txt"}
    scope = receipts.begin_read_attempt(rt, args)
    class AmbiguousToolPath(ValueError):
        pass
    def resolve(root, raw):
        if branch == "ambiguous":
            raise AmbiguousToolPath("ambiguous path")
        if branch == "interrupt":
            raise KeyboardInterrupt()
        return root / raw
    bolt = SimpleNamespace(receipt=lambda *a, **k: None, world=None)
    rt.observe = lambda *a: None
    env = dict(_name="", _args={}, line="read " + args["path"], multifile=False,
        root=tmp_path, pathlib=__import__("pathlib"), Mapping=Mapping,
        resolve_tool_path=resolve, AmbiguousToolPath=AmbiguousToolPath,
        information=SimpleNamespace(check_path=lambda *a: branch != "policy"),
        _hook_gate_route=lambda *a: branch == "hook", bolt=bolt, measurement=rt,
        msgs=[], tool_call_id=None, _tool_reply_for_call=lambda *a: {}, step=1,
        _lean_read_page_line_limit=None, _lean_read_page_byte_limit=None,
        _finish_read_attempt=receipts.finish_pending, sys=sys, MODEL="scripted",
        interventions=False, _read_path_counts={})
    if branch == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            exec(_production_read_route(), env)
    else:
        exec(_production_read_route(), env)
    scope.finish()
    results = [r for r in rt.journal.events if r["kind"] == "read_result"]
    assert len(results) == 1
    expected = {"ok": "OK", "missing": "MISSING", "interrupt": "INTERRUPTED", "policy": "DENIED"}.get(branch, "REFUSED")
    assert results[0]["status"] == expected
    assert ("payload_blob_ref" in results[0]) == (branch == "ok")
    assert env["_read_path_counts"] == ({"source.txt": 1} if branch == "ok" else {})


def _native_guard_source(branch):
    tree = ast.parse((ENGINE / "probes/react_repair.py").read_text())
    starts = {"policy": "_args.get('path') and (not information.check_path",
              "series": "_series_feedback is not None",
              "stub_sweep": "_stub_sweep_dispatch_feedback is not None",
              "green_lock": "_green_refusal"}
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and ast.unparse(n.test).startswith(starts[branch])]
    assert guards
    # Execute the production refusal body and the production native finally
    # settlement, with guard conditions supplied by the fixture.
    final = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'not _read_forwarded')
    body = [ast.Try(body=guards[0].body, handlers=[], orelse=[], finalbody=[final])]
    loop = ast.For(target=ast.Name(id="_", ctx=ast.Store()),
        iter=ast.Tuple(elts=[ast.Constant(1)], ctx=ast.Load()), body=body, orelse=[])
    return compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])),
                   "production_native_guard", "exec")


@pytest.mark.parametrize("branch", ["policy", "series", "stub_sweep", "green_lock"])
def test_real_native_refusals_finish_once(tmp_path, monkeypatch, branch):
    import kernel_navigation_receipt as navigation
    monkeypatch.setattr(navigation, "answer_despite_gate", lambda *a: None)
    rt = runtime(tmp_path)
    rt.observe = lambda *a: None
    receipts.begin_read_attempt(rt, {"path": "source.txt"})
    env = dict(_args={"path": "source.txt"}, _name="read", measurement=rt,
        msgs=[], root=tmp_path, tool_call_id="call", _read_forwarded=False,
        _series_feedback="refused", _stub_sweep_dispatch_feedback="refused",
        _green_refusal="refused", _hook_gate=lambda *a: False,
        _tool_reply_for_call=lambda *a: {}, bolt=SimpleNamespace(receipt=lambda *a, **k: None),
        _finish_read_attempt=receipts.finish_pending, sys=sys)
    exec(_native_guard_source(branch), env)
    receipts.finish_pending(rt)
    assert [e["kind"] for e in rt.journal.events] == ["read_attempt", "read_result"]
    assert rt.journal.events[-1]["status"] == ("DENIED" if branch == "policy" else "REFUSED")
    assert "capture_id" not in rt.journal.events[-1]


def test_native_batch_rejection_and_discard(tmp_path):
    rt = runtime(tmp_path)
    batch = NativeToolBatch()
    batch.journal = rt.journal
    batch._read_audit_runtime = rt
    calls = [{"id": str(i), "function": {"name": "read", "arguments": '{"path":"x"}'}}
             for i in range(3)]
    batch.pending = calls
    batch.discard_pending("interrupted batch")
    rows = [e for e in rt.journal.events if e["kind"] == "read_result"]
    assert len(rows) == 3 and len({e["attempt_id"] for e in rows}) == 3
    assert all(e["status"] == "NOT_DISPATCHED" and "capture_id" not in e for e in rows)
    receipts.refused_calls(rt, [{"function": {"name": "read", "arguments": "bad json"}},
                                {"function": {"name": "done"}}, None])
    assert len([e for e in rt.journal.events if e["kind"] == "read_result"]) == 4


def test_resume_switch_is_immutable(monkeypatch):
    for saved, current in (("on", "off"), ("off", "on"), (None, "on")):
        monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", current)
        with pytest.raises(runner.resume_module.ResumeRefused):
            runner._validate_read_capture_resume({} if saved is None else
                                                {"STATEFRAME_WORKING_SET_ALLOCATOR": saved})
    monkeypatch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR")
    runner._validate_read_capture_resume({"STATEFRAME_WORKING_SET_ALLOCATOR": "on"})


def test_off_baseline_observations_and_journal_bytes(tmp_path, monkeypatch):
    source = subprocess.check_output(["git", "show", "56af0dc6d6:src/stateframe/engine/kernel_read_exec.py"],
                                     cwd=ROOT)
    path = tmp_path / "baseline_read_exec.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("receipt_baseline", path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    (tmp_path / "source.txt").write_text("alpha\nbeta\n")
    def run(module, setting):
        monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", setting)
        rt = runtime(tmp_path)
        results = [read(rt, tmp_path, module=module) for _ in range(3)]
        return results, canonical_bytes(rt.journal.events)
    assert run(baseline, "off") == run(reads, "off")
    assert run(reads, "off")[0] == run(reads, "on")[0]


def test_receipt_fields_do_not_enter_wire_identity(tmp_path, monkeypatch):
    from kernel_wire_identity import request_wire_id
    rt = runtime(tmp_path)
    rt.journal.run_id = "fixed-seed"
    rt.journal.task_id = "task"
    rt.journal.records = lambda: rt.journal.events
    body = {"messages": [{"role": "user", "content": "read source.txt"}]}
    before = request_wire_id(rt.journal, 1, body)
    receipts.begin_read_attempt(rt, {"path": "source.txt"})
    receipts.finish_pending(rt)
    assert request_wire_id(rt.journal, 1, body) == before
    assert canonical_bytes(body) == b'{"messages":[{"content":"read source.txt","role":"user"}]}'


@pytest.mark.parametrize("case", ["oversized", "malformed", "done_order", "routing_discard"])
def test_native_next_rejected_reads_have_one_result(tmp_path, monkeypatch, case):
    import kernel_native_tools as native
    from kernel_measurement import ACTIVE_REQUEST
    from kernel_pilot import ACTIVE_RUNTIME
    rt = runtime(tmp_path)
    batch = native.NativeToolBatch()
    monkeypatch.setattr(batch, "_integrity_enabled", lambda: False)
    monkeypatch.setattr(native._frame_routes, "ROUTES_ACTIVE", case == "routing_discard")
    calls = [{"id": "r", "function": {"name": "read", "arguments": '{"path":"x"}'}}]
    if case == "oversized":
        calls *= 17
    elif case == "malformed":
        calls[0]["function"]["arguments"] = {}
    elif case == "done_order":
        calls.insert(0, {"id": "d", "function": {"name": "done", "arguments": "{}"}})
    token = ACTIVE_REQUEST.set(SimpleNamespace(journal=rt.journal, request_id="request", snapshot_id="snap"))
    runtime_token = ACTIVE_RUNTIME.set(rt)
    def discard(_calls):
        raise RuntimeError("routing discarded this response")
    try:
        if case == "routing_discard":
            with pytest.raises(RuntimeError, match="routing discarded"):
                batch.next(lambda *a, **k: ("", calls), [], before_commit=discard)
        else:
            batch.next(lambda *a, **k: ("", calls), [])
    finally:
        ACTIVE_RUNTIME.reset(runtime_token)
        ACTIVE_REQUEST.reset(token)
    attempts = [e for e in rt.journal.events if e["kind"] == "read_attempt"]
    results = [e for e in rt.journal.events if e["kind"] == "read_result"]
    assert len(attempts) == len(results) == (17 if case == "oversized" else 1)
    assert {e["attempt_id"] for e in attempts} == {e["attempt_id"] for e in results}
    assert all(e["status"] == "NOT_DISPATCHED" and "capture_id" not in e for e in results)


@pytest.mark.parametrize("error,status", [(PermissionError(), "PERMISSION"),
    (TimeoutError(), "DEADLINE"), (OSError(), "ERROR")])
def test_executor_typed_host_failures(tmp_path, monkeypatch, error, status):
    rt = runtime(tmp_path)
    (tmp_path / "source.txt").write_text("alpha\n")
    args = {"path": "source.txt"}
    receipts.begin_read_attempt(rt, args)
    def fail(_self):
        raise error
    monkeypatch.setattr(Path, "read_bytes", fail)
    outcome = reads.dispatch_live_read(tmp_path, tmp_path / "source.txt", "source.txt",
        "source.txt", 0, args=args, measurement=rt, step=1,
        bolt=SimpleNamespace(receipt=lambda *a, **k: None), policy=None, resolved_path="source.txt")
    assert outcome[0] == "ERROR"  # Existing behavior stays exactly the same.
    receipts.finish_pending(rt)
    result = rt.journal.events[-1]
    assert result["kind"] == "read_result" and result["status"] == status
    assert "identity" not in result and "payload_blob_ref" not in result


def test_allocator_reread_keeps_full_bytes_and_original_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_DELIVER_REQUESTED_BYTES", "on")
    monkeypatch.setenv("STATEFRAME_REREAD_RESULT_TRUTH", "off")
    rt = runtime(tmp_path)
    rt._code_bookmarks_active = True
    (tmp_path / "source.txt").write_text("alpha\nbeta\n")
    args = {"path": "source.txt"}
    bolt = SimpleNamespace(receipt=lambda *a, **k: None)
    def dispatch():
        receipts.begin_read_attempt(rt, args)
        return reads.dispatch_live_read(tmp_path, tmp_path / "source.txt", "source.txt",
            "source.txt", 0, args=args, measurement=rt, step=1,
            bolt=bolt, policy=None, resolved_path="source.txt")
    first = dispatch()
    reads.record_serialized_read_pages(rt, [{"role": "tool", "content": first[1]}], 3)
    rt._frame_budget_request_count = 4
    second = dispatch()
    receipts.finish_pending(rt)
    results = [e for e in rt.journal.events if e["kind"] == "read_result"]
    assert [e["status"] for e in results] == ["OK", "OK"]
    assert "alpha" in second[1]
    assert "unchanged since request" not in second[1]
    assert results[0]["payload_blob_ref"] == results[1]["payload_blob_ref"]
    assert results[1]["source"] == "read"
    assert rt.read_capture_ledger.all()[0].identity == rt.read_capture_ledger.all()[1].identity
    assert all("alpha" not in canonical_bytes(e).decode() for e in results)


def test_saved_golden_request_bytes_and_selection_stay_unchanged(tmp_path, monkeypatch):
    from kernel_measurement import _serialize_final_request_body
    from kernel_wire_identity import request_wire_id
    golden = ROOT / "tests/fixtures/pin_bm10_b/0020.source-priority-final.request.json"
    body = json.loads(golden.read_text())
    rt = runtime(tmp_path)
    rt.journal.task_id = "fixed-task"
    rt.journal.manifest = {"run_identity_seed": "fixed-seed"}
    before = _serialize_final_request_body(body, rt, "request")
    before_id = request_wire_id(rt.journal, 20, body)
    (tmp_path / "source.txt").write_text("alpha\nbeta\n")
    receipts.begin_read_attempt(rt, {"path": "source.txt"})
    read(rt, tmp_path)
    rt._read_attempt_scope.finish("OK")
    after = _serialize_final_request_body(body, rt, "request")
    assert after == before
    assert request_wire_id(rt.journal, 20, body) == before_id
    # Also exercise the real read-derived menu selector with captures enabled.
    from test_code_bookmarks_watch import _trigger, _live_read, _menu
    work = tmp_path / "selection"
    a, _ = _trigger(work)
    a.journal.path = rt.journal.path
    a._frame_budget_request_count = 3
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    first_result = _live_read(a, "req-one", "pkg/service.py", {"path": "pkg/service.py"})
    first_menu = _menu(a, "req-one")
    b, _ = _trigger(work)
    b.journal.path = rt.journal.path
    b._frame_budget_request_count = 3
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    second_result = _live_read(b, "req-one", "pkg/service.py", {"path": "pkg/service.py"})
    second_menu = _menu(b, "req-one")
    assert first_result[:3] == second_result[:3]
    assert second_result[3]["working_set_repeat_capture"] is False
    assert first_menu == second_menu


def test_package_import_does_not_require_engine_path(tmp_path):
    result = subprocess.run([sys.executable, "-c",
        "from stateframe.engine.kernel_read_receipts import line_digests_for; "
        "assert len(line_digests_for('alpha\\n\\n')) == 1"],
        env={"PATH": __import__("os").environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "src")},
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_real_journal_stores_tiny_payload_as_blob(tmp_path):
    from kernel_continuation_journal import ContinuationJournal
    from _journal import measurement_events
    journal_parent = tmp_path / "worker/measurement-runs"
    journal_parent.mkdir(parents=True)
    journal = ContinuationJournal(journal_parent, "receipt-task", {})
    rt = SimpleNamespace(journal=journal, retained=None, step_ledger=None,
                         _frame_budget_request_count=1)
    (tmp_path / "source.txt").write_text("x\n")
    try:
        receipts.begin_read_attempt(rt, {"path": "source.txt"})
        read(rt, tmp_path)
        rt._read_attempt_scope.finish("OK")
        receipts.finish_pending(rt)
        events = list(measurement_events(tmp_path))
        result = next(e for e in events if e["kind"] == "read_result")
        assert result["status"] == "OK"
        capture = rt.read_capture_ledger.all()[0]
        assert result["payload_blob_ref"] == capture.payload_blob_ref
        assert resolve_journal_blob_refs({"$blob": capture.payload_blob_ref,
            "bytes": len(canonical_bytes("x\n"))}, journal.path) == "x\n"
    finally:
        journal.close()


def test_native_selected_force_close_does_not_duplicate_attempt(tmp_path):
    rt = runtime(tmp_path)
    batch = NativeToolBatch()
    batch.journal = rt.journal
    batch._read_audit_runtime = rt
    batch.inflight = {"id": "read-call", "function": {"name": "read", "arguments": "{}"}}
    receipts.begin_read_attempt(rt, {"path": "source.txt"}, call_id="read-call")
    batch.skip_inflight("budget exhausted")
    receipts.finish_pending(rt)
    assert len([e for e in rt.journal.events if e["kind"] == "read_attempt"]) == 1
    assert len([e for e in rt.journal.events if e["kind"] == "read_result"]) == 1


def test_transcript_cleanup_reuses_settled_selected_attempt(tmp_path):
    rt = runtime(tmp_path)
    batch = NativeToolBatch()
    batch.journal = rt.journal
    batch._read_audit_runtime = rt
    batch.inflight = {"id": "read-call", "function": {"name": "read", "arguments": "{}"}}
    receipts.begin_read_attempt(rt, {"path": "source.txt"}, call_id="read-call")
    receipts.finish_pending(rt, error=KeyboardInterrupt())
    batch.skip_inflight("interrupted")
    assert len([e for e in rt.journal.events if e["kind"] == "read_attempt"]) == 1
    rows = [e for e in rt.journal.events if e["kind"] == "read_result"]
    assert len(rows) == 1 and rows[0]["status"] == "INTERRUPTED"


def test_recorded_native_read_replay_off_selection_and_on_captures(tmp_path, monkeypatch):
    """OFF keeps recorded reuse; ON delivers bytes and retains every capture.

    Complete OFF wire parity lives in test_working_set_cutover. The former
    shadow-only ON/OFF byte equality is intentionally no longer the contract.
    """
    import test_pin_bm6diag_replay as replay
    original_runtime = replay._runtime
    original_menu = replay._menu
    original_dispatch = replay._dispatch
    wires = []
    def with_blob_store(root, recorded):
        rt, events = original_runtime(root, recorded)
        rt.journal.path = tmp_path / "audit/events.jsonl"
        rt.journal.path.parent.mkdir(exist_ok=True)
        return rt, events
    def record_menu(rt, case):
        names, wire = original_menu(rt, case)
        wires.append(wire)
        return names, wire
    def dispatch(rt, case, index):
        receipts.begin_read_attempt(rt, case["calls"][index], call_id=f"call-{index}")
        try:
            if receipts.enabled():
                row = case["receipts"][index] if index < len(case["receipts"]) else None
                path = case["calls"][index]["path"]
                rt.current_request_id = (row["request_id"] if row else
                    f"request-{case['request']}-action-{index + 1}")
                status, observation, _, delivery = reads.dispatch_live_read(
                    rt.owner.root, rt.owner.root / path, path, path, 0,
                    args=case["calls"][index], measurement=rt,
                    step=row["turn"] if row else 1000 + index,
                    bolt=rt.owner, policy=None, resolved_path=path)
                assert status == "OK" and observation.startswith("[read page]")
                assert delivery["unchanged_since_request"] is False
                assert delivery["requested_bytes_delivered"] > 0
                return rt.owner._receipts[-1]
            return original_dispatch(rt, case, index)
        finally:
            receipts.finish_pending(rt, error=sys.exc_info()[1])
    monkeypatch.setattr(replay, "_runtime", with_blob_store)
    monkeypatch.setattr(replay, "_menu", record_menu)
    monkeypatch.setattr(replay, "_dispatch", dispatch)
    results = []
    for setting in ("off", "on"):
        monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", setting)
        wires.clear()
        rt, watch, events, cases, counters = replay._replay(tmp_path / "recorded", monkeypatch)
        # The cutover adds capture authority and no longer emits counter-based reuse stubs.
        from kernel_working_set_state import EVENT_KINDS as shadow_kinds
        results.append((list(wires), counters, watch.streak,
                        [e for e in events if e["kind"] not in {"read_attempt", "read_result", "read_admission", *shadow_kinds}]))
        if setting == "on":
            assert any(e["kind"] == "working_set_capture" for e in events)
        attempts = [e for e in events if e["kind"] == "read_attempt"]
        outcomes = [e for e in events if e["kind"] == "read_result"]
        assert len(attempts) == len(outcomes) == (30 if setting == "on" else 0)
        if setting == "on":
            assert all(e["status"] == "OK" for e in outcomes)
            assert len(rt.read_capture_ledger.all()) >= 30
    assert len(results[0][0]) == len(results[1][0]) == 4


def test_attempt_ids_survive_a_fresh_runtime_on_existing_journal(tmp_path):
    from kernel_continuation_journal import ContinuationJournal
    journal = ContinuationJournal(tmp_path, "receipt-resume-task", {})
    try:
        first = SimpleNamespace(journal=journal)
        old = receipts.begin_read_attempt(first, {"path": "source.txt"})
        receipts.finish_pending(first)
        restored = SimpleNamespace(journal=journal)
        new = receipts.begin_read_attempt(restored, {"path": "source.txt"})
        receipts.finish_pending(restored)
        assert old.attempt_id != new.attempt_id
    finally:
        journal.close()


def test_line_digest_matches_the_shared_alarm_definition_byte_for_byte():
    # The context alarm uses this exact definition; trailing whitespace is ignored.
    def alarm_digest(line):
        return hashlib.blake2b(line.rstrip().encode("utf-8"), digest_size=8).hexdigest()
    for line in ("x = 1", "x = 1   ", "x = 1\t", "  indented\r"):
        assert receipts.line_digest(line) == alarm_digest(line)
    assert receipts.line_digest("x = 1  ") == receipts.line_digest("x = 1")


def test_line_numbers_align_with_digests_and_skip_blank_lines():
    digests, numbers = receipts.line_digests_with_numbers("a\n\n  \nb\nc\n", 3)
    assert numbers == (3, 6, 7)
    assert digests == receipts.line_digests_for("a\n\n  \nb\nc\n")
    assert len(digests) == len(numbers)


def test_relative_path_outside_the_workspace_is_none_not_an_error(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("x\n")
    assert receipts.relative_path(root, root / "a.py") == "a.py"
    assert receipts.relative_path(root, tmp_path / "elsewhere.py") is None


def test_bare_read_command_never_indexes_an_empty_argument_list():
    # Both text-mode read entry points build their attempt argument before the switch check,
    # so a bare "read " line must not raise even with the switch off.
    source = Path(receipts.__file__).with_name("probes").joinpath("react_repair.py").read_text()
    assert 'line[5:].split()[0]' not in source.replace('(line[5:].split() or [""])[0]', "")
