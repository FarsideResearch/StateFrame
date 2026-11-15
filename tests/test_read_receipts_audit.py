"""Read audit coverage at the real shell, grep and check-output boundaries."""
from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

import kernel_read_exec as reads
import kernel_read_receipts as receipts
from kernel_check_receipts import observe_check_output, retrieve_check_output
from kernel_execution_reply_budget import record_delivery
from kernel_pilot import ACTIVE_RUNTIME
from test_read_receipts import ENGINE, ROOT, runtime
import test_shell_state


@pytest.fixture
def shell_setup(tmp_path, monkeypatch):
    # Reuse the real foreground process boundary without starting an engine.
    return test_shell_state.setup.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")


def _results(rt):
    attempts = [e for e in rt.journal.events if e["kind"] == "read_attempt"]
    results = [e for e in rt.journal.events if e["kind"] == "read_result"]
    assert len(attempts) == len(results)
    assert [e["attempt_id"] for e in attempts] == [e["attempt_id"] for e in results]
    return results


def _dispatch(rt, root, path, args):
    receipts.begin_read_attempt(rt, args)
    try:
        return reads.dispatch_live_read(root, path, str(path), str(path), 0,
            args=args, measurement=rt, step=1, policy=None, resolved_path=str(path),
            bolt=SimpleNamespace(receipt=lambda *a, **k: None))
    finally:
        receipts.finish_pending(rt)


@pytest.mark.parametrize("symlink", [False, True])
def test_successful_external_read_is_uncaptured_not_refused(tmp_path, symlink):
    from kernel_information_policy import InformationPolicy
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("visible source\n")
    target = root / "link.txt" if symlink else outside
    if symlink:
        target.symlink_to(outside)
    policy = object.__new__(InformationPolicy)
    policy.strict = False
    assert policy.check_path(root, str(target))
    rt = runtime(tmp_path)
    output = _dispatch(rt, root, target, {"path": str(target)})
    assert output[0] == "OK" and "visible source" in output[1]
    result, = _results(rt)
    assert result["status"] == "OK_UNCAPTURED"
    assert result["reason"] == ("symlink_outside" if symlink else "external_path")
    skipped, = [e for e in rt.journal.events if e["kind"] == "read_capture_skipped"]
    assert skipped["reason"] == result["reason"]
    assert skipped["status"] == "OK_UNCAPTURED"
    assert "capture_id" not in result


@pytest.mark.parametrize("args,status", [
    ({"expected_revision": "sha256:" + "0" * 64}, "STALE_REVISION"),
    ({"start_line": 20}, "OUT_OF_RANGE"),
    ({"symbol": "does_not_exist"}, "SYMBOL_NOT_FOUND"),
    ({"cursor": "invalid"}, "STALE_CURSOR"),
])
def test_source_statuses_survive_the_audit(tmp_path, args, status):
    path = tmp_path / "source.py"
    path.write_text("def present():\n    return 1\n")
    rt = runtime(tmp_path)
    output = _dispatch(rt, tmp_path, path, {"path": "source.py", **args})
    result, = _results(rt)
    assert result["status"] == status
    assert "capture_id" not in result
    assert output[0] in {status, "READ_CURSOR_INVALID", "READ_CURSOR_STALE"}


@pytest.mark.parametrize("status", sorted(receipts.READ_STATUSES - {"OK", "PAGE"}))
def test_typed_statuses_do_not_collapse_to_refused(tmp_path, status):
    rt = runtime(tmp_path)
    scope = receipts.begin_read_attempt(rt, {})
    scope.finish(status)
    assert _results(rt)[0]["status"] == status


def test_trailing_attempt_reduces_to_interrupted_without_rewriting_journal(tmp_path):
    rt = runtime(tmp_path)
    receipts.begin_read_attempt(rt, {"path": "a"}).finish("MISSING")
    pending = receipts.begin_read_attempt(rt, {"path": "b"})
    before = json.dumps(rt.journal.events, sort_keys=True)
    reduced = receipts.reduce_read_results(iter(rt.journal.events))
    assert [e["status"] for e in reduced] == ["MISSING", "INTERRUPTED"]
    assert reduced[-1]["attempt_id"] == pending.attempt_id
    assert json.dumps(rt.journal.events, sort_keys=True) == before


def test_capture_skip_is_typed_and_switch_gated(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    args = dict(path="x.py", metadata={"result_class": "STALE_REVISION"},
                payload="secret payload", original_page="secret page", source="disk", arguments={})
    assert receipts.capture_read(rt, **args) is None
    assert rt.journal.events == [{"kind": "read_capture_skipped", "attempt_id": None,
                                  "status": "STALE_REVISION", "reason": "stale"}]
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    before = list(rt.journal.events)
    assert receipts.capture_read(rt, **args) is None
    receipts.capture_failed(rt, ValueError("test"))
    assert receipts.reduce_read_results(before) == ()
    assert rt.journal.events == before


@pytest.mark.parametrize("cache", ["disk", "file_cache", "page_cache"])
@pytest.mark.parametrize("switch", ["on", "off"])
def test_capture_metadata_failure_withholds_delivery_only_when_on(tmp_path, monkeypatch, cache, switch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", switch)
    path = tmp_path / "source.py"
    path.write_text("def present():\n    return 1\n")
    args = {"path": "source.py"}
    rt = runtime(tmp_path)
    expected = _dispatch(rt, tmp_path, path, args)
    rt.journal.events.clear()
    if cache == "disk":
        rt._read_contract_source_cache.clear()
    if cache == "page_cache":
        entry = next(iter(rt._read_contract_source_cache.values()))
        entry.content = None
        expected = _dispatch(rt, tmp_path, path, args)
        rt.journal.events.clear()
        for page in entry.pages.values():
            page["metadata"] = {**page["metadata"], "byte_start": None}
    else:
        original = reads.deliver_source
        def malformed(*a, **k):
            page = original(*a, **k)
            return replace(page, metadata={**page.metadata, "byte_start": None})
        monkeypatch.setattr(reads, "deliver_source", malformed)
    output = _dispatch(rt, tmp_path, path, args)
    if switch == "off":
        assert output[:3] == expected[:3]
        assert [e["kind"] for e in rt.journal.events] == ["read_repeat_assessed"]
        assert not hasattr(rt, "read_capture_ledger")
        return
    assert output[0] == "ERROR" and output[2] is None
    assert "Retry" in output[1] and "def present" not in output[1]
    delivery = output[3]
    assert delivery["result_class"] == "ERROR"
    assert delivery["failure_class"] == "capture_failed"
    assert delivery["delivery_status"] == "WITHHELD"
    assert delivery["effective_range"] is None
    assert delivery["requested_bytes_delivered"] == 0
    assert not [e for e in rt.journal.events if e["kind"] == "working_set_capture"]
    result, = _results(rt)
    assert result["status"] == "ERROR" and result["reason"] == "capture_failed"
    failed, = [e for e in rt.journal.events if e["kind"] == "read_capture_failed"]
    assert failed["error_type"] == "ValueError" and len(failed["message"]) <= 160
    assert "def present" not in json.dumps(failed)
    assert "payload_blob_ref" not in result
    assert "identity" not in result


@pytest.mark.parametrize("command,first,last,payload", [
    ("cat app.py", 1, 4, "one\ntwo\nthree\nfour\n"),
    ("sed -n '2,3p' app.py", 2, 3, "two\nthree\n"),
    ("sed -n '3p' app.py", 3, 3, "three\n"),
    ("head -n 2 app.py", 1, 2, "one\ntwo\n"),
    ("head -2 app.py", 1, 2, "one\ntwo\n"),
    ("head -n2 app.py", 1, 2, "one\ntwo\n"),
    ("tail -n 2 app.py", 3, 4, "three\nfour\n"),
    ("tail -n +3 app.py", 3, 4, "three\nfour\n"),
    ("head app.py", 1, 4, "one\ntwo\nthree\nfour\n"),
    ("tail app.py", 1, 4, "one\ntwo\nthree\nfour\n"),
])
def test_shell_exact_read_captures_bytes_and_range(shell_setup, tmp_path, command, first, last, payload):
    make, root, _ = shell_setup
    raw = b"one\ntwo\nthree\nfour\n"
    (root / "app.py").write_bytes(raw)
    tool, _ = make("off")
    rt = runtime(tmp_path)
    token = ACTIVE_RUNTIME.set(rt)
    try:
        reply = tool.execute({"command": command}, "shell-read")
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert payload in reply.observation
    result, = _results(rt)
    assert result["status"] == "OK"
    capture, = rt.read_capture_ledger.all()
    identity = capture.identity
    assert (identity.path, identity.first_line, identity.last_line) == ("app.py", first, last)
    assert identity.revision == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert raw[identity.byte_start:identity.byte_end] == payload.encode()
    assert identity.payload_sha256 == hashlib.sha256(payload.encode()).hexdigest()


@pytest.mark.parametrize("command", ["cat app.py | head -n 1", "grep VALUE app.py", "cat missing.py"])
def test_shell_opaque_reads_still_finish_once(shell_setup, tmp_path, command):
    make, _, _ = shell_setup
    tool, _ = make("off")
    rt = runtime(tmp_path)
    token = ACTIVE_RUNTIME.set(rt)
    try:
        tool.execute({"command": command}, "opaque")
    finally:
        ACTIVE_RUNTIME.reset(token)
    result, = _results(rt)
    assert result["status"] == "SHELL_READ" and "capture_id" not in result


def test_shell_capture_uses_persisted_cwd_and_exact_utf8_bytes(shell_setup, tmp_path):
    make, root, _ = shell_setup
    (root / "sub").mkdir()
    raw = "first\nlast é".encode()
    (root / "sub/app.py").write_bytes(raw)
    tool, _ = make("on")
    rt = runtime(tmp_path)
    token = ACTIVE_RUNTIME.set(rt)
    try:
        tool.execute({"command": "cd sub"}, "cd")
        reply = tool.execute({"command": "tail -n1 app.py"}, "read")
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert "last é" in reply.observation
    assert [r["status"] for r in _results(rt)] == ["SHELL_READ", "OK"]
    capture, = rt.read_capture_ledger.all()
    assert capture.identity.path == "sub/app.py"
    assert capture.identity.first_line == capture.identity.last_line == 2
    assert raw[capture.identity.byte_start:capture.identity.byte_end] == "last é".encode()


def test_shell_snapshot_rejects_special_files_and_bounds_reads(tmp_path, monkeypatch):
    import kernel_shell_read_receipts as shell_audit
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    assert shell_audit._bounded_bytes(fifo) is None
    path = tmp_path / "large"
    path.write_bytes(b"1234")
    monkeypatch.setattr(shell_audit, "MAX_SNAPSHOT_BYTES", 3)
    assert shell_audit._bounded_bytes(path) is None


@pytest.mark.parametrize("kind", ["stale", "truncated", "capture_failure"])
def test_shell_capture_failure_or_unproven_bytes_never_break_reply(shell_setup, tmp_path, monkeypatch, kind):
    make, root, run = shell_setup
    tool, _ = make("off")
    rt = runtime(tmp_path)
    def execute(args, call_id):
        if kind == "stale":
            (root / "app.py").write_text("VALUE = 2\n")
        output = "VALUE = 1\n" if kind != "truncated" else "VALUE [... omitted ...]"
        observation = record_delivery(run, "$ cat app.py\nexit 0 | 0.01s\n", output, "",
                                      omitted_bytes=10 if kind == "truncated" else 0, call_id=call_id)
        return SimpleNamespace(exit_code=0, fatal=False, observation=observation)
    monkeypatch.setattr(tool, "_execute", execute)
    if kind == "capture_failure":
        import kernel_shell_read_receipts as shell_audit
        def fail(*a, **k):
            raise ValueError("invalid metadata")
        monkeypatch.setattr(shell_audit, "capture_read", fail)
    token = ACTIVE_RUNTIME.set(rt)
    try:
        reply = tool.execute({"command": "cat app.py"}, "shell")
    finally:
        ACTIVE_RUNTIME.reset(token)
    result, = _results(rt)
    assert result["status"] == "SHELL_READ" and "capture_id" not in result
    assert "VALUE" in reply.observation
    assert result["reason"] == {"stale": "stale", "truncated": "transformed_or_truncated",
                                "capture_failure": "capture_failed"}[kind]


def _grep_route(source=None):
    tree = ast.parse(source if source is not None else (ENGINE / "probes/react_repair.py").read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "line.startswith('grep ')")
    loop = ast.For(target=ast.Name(id="_", ctx=ast.Store()),
        iter=ast.Tuple(elts=[ast.Constant(1)], ctx=ast.Load()), body=node.body, orelse=[])
    return compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), "grep_route", "exec")


def _grep(rt, root, *, native=False, source=None):
    env = dict(line="grep VALUE app.py", _name="grep" if native else "",
        _args={"pattern": "VALUE", "path": "app.py"} if native else {},
        begin_read_attempt=receipts.begin_read_attempt, measurement=rt, root=root,
        tool_call_id="grep-call" if native else None, Mapping=Mapping,
        resolve_tool_path=lambda root, path: root / path, AmbiguousToolPath=ValueError,
        information=SimpleNamespace(check_path=lambda *a: True),
        _hook_gate_route=lambda *a: False, MODEL="scripted", multifile=False,
        bolt=SimpleNamespace(receipt=lambda *a, **k: None, sure_observe=lambda *a: None),
        interventions=False, re=re, sys=sys, json=json, msgs=[],
        _workspace_truth=SimpleNamespace(enabled=lambda: False,
            render_legacy_symbol_answer=lambda *args: ""))
    exec(_grep_route(source), env)
    return env["obs"]


@pytest.mark.parametrize("native", [False, True])
def test_grep_source_reply_has_exactly_one_result(tmp_path, native):
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    rt = runtime(tmp_path)
    assert "VALUE = 1" in _grep(rt, tmp_path, native=native)
    result, = _results(rt)
    assert result["status"] == "GREP" and "capture_id" not in result


def test_grep_on_off_model_visible_reply_is_identical(tmp_path, monkeypatch):
    from kernel_measurement import canonical_bytes
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    rt = runtime(tmp_path)
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    original = canonical_bytes({"messages": [{"role": "tool", "content": _grep(rt, tmp_path)}]})
    assert rt.journal.events == []
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    audited = canonical_bytes({"messages": [{"role": "tool", "content": _grep(rt, tmp_path)}]})
    assert audited == original
    assert len(_results(rt)) == 1


@pytest.mark.parametrize("observe", [False, True])
@pytest.mark.parametrize("memo", ["off", "on"])
def test_check_output_each_page_has_one_result_even_with_observe_and_memo(tmp_path, observe, memo):
    rt = runtime(tmp_path)
    rt.stateframe_check_receipt_memo_mode = memo
    rt.observe = lambda *a: None
    rt._check_output_receipts = {"check-receipt:x": {
        "revision": "UNKNOWN", "selector": "tests", "stdout": "source\n", "stderr": "details\n"}}
    for _ in range(2):
        if observe:
            page = json.loads(observe_check_output(rt, SimpleNamespace(receipt=lambda *a: None),
                                                 "check-receipt:x"))
        else:
            page = retrieve_check_output(rt, "check-receipt:x")
        assert page["content"] == "source\ndetails\n"
    assert [r["status"] for r in _results(rt)] == ["CHECK_OUTPUT", "CHECK_OUTPUT"]


@pytest.mark.parametrize("options,status", [
    ({"stream": "unknown"}, "INVALID_ARGUMENT"),
    ({"offset": -1}, "INVALID_ARGUMENT"),
    ({"limit": 0}, "INVALID_ARGUMENT"),
    ({"handle": "missing"}, "MISSING"),
])
def test_check_lookup_errors_remain_typed(tmp_path, options, status):
    rt = runtime(tmp_path)
    rt._check_output_receipts = {"check-receipt:x": {"revision": "UNKNOWN", "stdout": "source\n"}}
    retrieve_check_output(rt, **{"handle": "check-receipt:x", **options})
    result, = _results(rt)
    assert result["status"] == status and "capture_id" not in result


def _baseline(tmp_path, filename):
    source = subprocess.check_output(["git", "show", f"56af0dc6d6:src/stateframe/engine/{filename}.py"], cwd=ROOT)
    path = tmp_path / f"baseline_{filename}.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_switch_off_grep_check_delivery_and_journal_match_base(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    rt = runtime(tmp_path)
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    baseline_grep = subprocess.check_output(
        ["git", "show", "56af0dc6d6:src/stateframe/engine/probes/react_repair.py"], cwd=ROOT)
    assert _grep(rt, tmp_path) == _grep(rt, tmp_path, source=baseline_grep)
    assert rt.journal.events == []

    rt._check_output_receipts = {"check-receipt:x": {"revision": "UNKNOWN", "stdout": "source\n"}}
    baseline = _baseline(tmp_path, "kernel_check_receipts")
    assert retrieve_check_output(rt, "check-receipt:x") == baseline.retrieve_check_output(rt, "check-receipt:x")
    assert rt.journal.events == []


def test_shell_on_off_delivery_request_bytes_and_non_audit_journal_match_base(shell_setup, tmp_path, monkeypatch):
    from kernel_measurement import canonical_bytes
    make, _, _ = shell_setup
    tool, _ = make("off")
    baseline = _baseline(tmp_path, "kernel_shell_tool")
    rt = runtime(tmp_path)
    token = ACTIVE_RUNTIME.set(rt)
    try:
        monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
        original = baseline.ShellTool.execute(tool, {"command": "cat app.py"}, "base")
        original_journal = canonical_bytes(rt.journal.events)
        current = tool.execute({"command": "cat app.py"}, "off")
        assert canonical_bytes(rt.journal.events) == original_journal
        monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
        audited = tool.execute({"command": "cat app.py"}, "on")
    finally:
        ACTIVE_RUNTIME.reset(token)
    bodies = [canonical_bytes({"messages": [{"role": "tool", "content": r.observation}]})
              for r in (original, current, audited)]
    assert bodies[0] == bodies[1] == bodies[2]
    from kernel_working_set_state import EVENT_KINDS as shadow_kinds
    audit_kinds = {"read_attempt", "read_result", "read_capture_skipped", "read_capture_failed",
                   "read_admission", *shadow_kinds}
    assert len([e for e in rt.journal.events if e["kind"] == "working_set_capture"]) == 1
    assert canonical_bytes([e for e in rt.journal.events if e["kind"] not in audit_kinds]) == original_journal
    assert len(_results(rt)) == 1


def _paired(rt):
    """Nested audits finish inner-first, so compare sets rather than order."""
    attempts = [e for e in rt.journal.events if e["kind"] == "read_attempt"]
    results = [e for e in rt.journal.events if e["kind"] == "read_result"]
    assert len(attempts) == len(results)
    assert {e["attempt_id"] for e in attempts} == {e["attempt_id"] for e in results}
    return {e["attempt_id"]: e for e in results}


def test_check_output_lookup_restores_an_enclosing_read_attempt(tmp_path):
    rt = runtime(tmp_path)
    rt._check_output_receipts = {"check-receipt:x": {
        "revision": "UNKNOWN", "selector": "tests", "stdout": "source\n", "stderr": ""}}
    outer = receipts.begin_read_attempt(rt, {"path": "app.py"}, call_id="outer")
    retrieve_check_output(rt, "check-receipt:x")
    assert rt._read_attempt_scope is outer and not outer.finished
    outer.finish("DENIED")
    results = _paired(rt)
    assert len(results) == 2
    assert results[outer.attempt_id]["status"] == "DENIED"
    inner, = [r for r in results.values() if r["attempt_id"] != outer.attempt_id]
    assert inner["status"] == "CHECK_OUTPUT"


def test_shell_audit_restores_an_enclosing_read_attempt(shell_setup, tmp_path):
    make, root, _ = shell_setup
    (root / "app.py").write_bytes(b"one\n")
    tool, _ = make("off")
    rt = runtime(tmp_path)
    outer = receipts.begin_read_attempt(rt, {"path": "app.py"}, call_id="outer")
    token = ACTIVE_RUNTIME.set(rt)
    try:
        tool.execute({"command": "cat app.py"}, "shell-nested")
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert rt._read_attempt_scope is outer and not outer.finished
    outer.finish("DENIED")
    results = _paired(rt)
    assert len(results) == 2 and results[outer.attempt_id]["status"] == "DENIED"
    inner, = [r for r in results.values() if r["attempt_id"] != outer.attempt_id]
    assert inner["status"] == "OK"


def test_shell_read_of_missing_file_is_opaque_not_a_capture_failure(shell_setup, tmp_path):
    make, _, _ = shell_setup
    tool, _ = make("off")
    rt = runtime(tmp_path)
    token = ACTIVE_RUNTIME.set(rt)
    try:
        tool.execute({"command": "cat missing.py"}, "missing")
    finally:
        ACTIVE_RUNTIME.reset(token)
    result, = _results(rt)
    assert result["status"] == "SHELL_READ"
    assert [e for e in rt.journal.events if e["kind"] == "read_capture_failed"] == []


def test_grep_audit_restores_an_enclosing_read_attempt(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    rt = runtime(tmp_path)
    outer = receipts.begin_read_attempt(rt, {"path": "app.py"}, call_id="outer")
    assert "VALUE = 1" in _grep(rt, tmp_path, native=True)
    assert rt._read_attempt_scope is outer and not outer.finished
    outer.finish("DENIED")
    results = _paired(rt)
    assert len(results) == 2 and results[outer.attempt_id]["status"] == "DENIED"
    inner, = [r for r in results.values() if r["attempt_id"] != outer.attempt_id]
    assert inner["status"] == "GREP"
