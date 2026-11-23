"""Code bookmark replay, bookmarks-off bytes, and progress-watch batch accounting."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from _timeouts import scaled
from kernel_read_exec import dispatch_live_read, execute_read, record_serialized_read_pages
from kernel_request_tool_surface import _apply_request_tool_floor, _request_tool_floor_schemas
from kernel_agent_schemas import BUILTIN_SCHEMAS
from kernel_reread_result_truth import checkpoint_replay, restore_replay
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_tool_call_loop_breaker import UnboundedProgressWatch


TOOLS = [
    {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
    # Ordered as the recorded T2 probe's request 14 menu.
    for name in ("edit", "str_replace", "read", "read_check_receipt", "run_tests",
                 "grep", "file_delete", "file_move", "shell", "authored_check",
                 "done", "file_create")
]


def _names(tools):
    return [tool["function"]["name"] for tool in tools]


def _runtime(root: Path):
    events = []
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(root=root, _receipts=[])
    runtime.journal = SimpleNamespace(emit=lambda kind, **fields: events.append({"kind": kind, **fields}))
    runtime._base_tool_menu = lambda tools, _messages: list(tools)
    runtime._code_bookmarks_active = True
    runtime.stateframe_framefit_mode = "on"
    runtime._code_bookmark_carry = [
        {"path": path, "symbol": "value", "read_only": False}
        for path in ("pkg/batch/contracts.py", "pkg/service.py")]
    runtime.edit_contract = {"edit_authority": {"editable_paths": [
        "pkg/batch/contracts.py", "pkg/service.py"]}}
    runtime._framefit_action_context = {"framefit5": {"read_facts": [
        {"path": path, "body_delivery": "present"}
        for path in ("pkg/batch/contracts.py", "pkg/service.py")]}}
    runtime.current_request_id = None
    runtime.binding = lambda: {"product_revision": "sha256:probe-revision"}
    failure = {
        "receipt_id": "public-check-13", "revision": "sha256:probe-revision",
        "status": "FAIL", "outcome": "FAIL", "failure_phase": "pytest",
        "runner_tail_digest": "a" * 64,
        "assertion_context_preview": "assert 'BatchResult' in batch.__all__",
    }
    runtime._latest_check_receipt = failure
    runtime.host_check_receipts = [failure]
    return runtime, events


def _read(runtime, request_id: str, path: str, *, unchanged: bool = True):
    source = (runtime.owner.root / path).read_bytes()
    runtime.owner._receipts.append({
        "request_id": request_id, "kind": "read", "result_class": "OK",
        "target_path": path,
        "delivery": {"unchanged_since_request": unchanged,
                     "requested_bytes_delivered": 0 if unchanged else len(source),
                     "effective_range": {"byte_start": 0, "byte_end": len(source)},
                     "revision": "sha256:" + hashlib.sha256(source).hexdigest()},
    })


def _menu(runtime, previous: str):
    runtime.current_request_id = previous
    return StateFrameRuntime.tool_menu(runtime, TOOLS, [])


def _trigger(tmp_path):
    # The source paths and failed public assertion mirror the recorded T2 probe.
    for path in ("pkg/batch/contracts.py", "pkg/service.py"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("value = 1\n")
    runtime, events = _runtime(tmp_path)
    assert "read" in _names(_menu(runtime, "probe-request-14"))
    _read(runtime, "probe-request-14", "pkg/batch/contracts.py")
    _read(runtime, "probe-request-15", "pkg/service.py")
    return runtime, events


def _live_read(runtime, request_id, path, arguments):
    runtime.current_request_id = request_id
    runtime.retained = None
    runtime.step_ledger = None
    runtime.owner.workspace_revision = 1

    def receipt(kind, intent, fingerprint, result, **fields):
        runtime.owner._receipts.append({"request_id": request_id, "kind": kind,
                                        "intent": intent, "fingerprint": fingerprint,
                                        "result_class": result, **fields})

    runtime.owner.receipt = receipt
    return dispatch_live_read(
        runtime.owner.root, runtime.owner.root / path, path, path, 0,
        args=arguments, measurement=runtime, step=len(runtime.owner._receipts) + 1,
        bolt=runtime.owner, policy=None, resolved_path=path)


def test_bookmark_kill_switch_keeps_menu_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    runtime, events = _trigger(tmp_path)
    expected = json.dumps(TOOLS, sort_keys=True)
    assert json.dumps(_menu(runtime, "probe-request-15"), sort_keys=True) == expected
    assert not events


def test_default_leaves_menu_and_events_untouched(tmp_path):
    runtime, events = _trigger(tmp_path)
    assert _names(_menu(runtime, "probe-request-15")) == _names(TOOLS)
    assert not events


def test_bookmarks_off_repeated_read_matches_release_bytes(tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    monkeypatch.setenv("STATEFRAME_DELIVER_REQUESTED_BYTES", "on")
    baseline_source = subprocess.check_output([
        "git", "show",
        "97cd52d6c2225a823fdfeb632aad7bfd738f4d91:src/stateframe/engine/kernel_read_exec.py",
    ], cwd=Path(__file__).resolve().parents[1])
    baseline_path = tmp_path / "release_read_exec.py"
    baseline_path.write_bytes(baseline_source)
    spec = importlib.util.spec_from_file_location("release_read_exec", baseline_path)
    assert spec is not None and spec.loader is not None
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)

    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")

    def run(module):
        runtime = SimpleNamespace(owner=SimpleNamespace(workspace_revision=1),
                                  retained=None, step_ledger=None,
                                  _frame_budget_request_count=1)
        replies = []
        for number in (1, 2):
            runtime._frame_budget_request_count = number
            delivery = {}
            result, observation, content = module.execute_read(
                tmp_path, source, "module.py", "module.py", 0,
                args={"path": "module.py"}, measurement=runtime, step=number,
                delivery=delivery)
            replies.append((result, observation, content, delivery))
            if number == 1 and module is not baseline:
                record_serialized_read_pages(
                    runtime, [{"role": "tool", "content": observation}], number)
        return replies

    assert run(sys.modules[execute_read.__module__]) == run(baseline)


def test_resumed_dispatch_preserves_failure_identity_and_request_bytes(tmp_path, monkeypatch):
    from kernel_continuation_journal import ContinuationJournal
    from kernel_reread_result_truth import _current_failure

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_DELIVER_REQUESTED_BYTES", "on")
    path = tmp_path / "pkg/service.py"
    path.parent.mkdir(parents=True)
    path.write_text("value = 1\n")
    live, _ = _runtime(tmp_path)
    failure = live._latest_check_receipt
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    journal = ContinuationJournal(journal_dir, "pin-resume", {"origin": "fixture"})
    try:
        journal.emit("check_receipt", **failure)
        projected = journal.framefit_receipt_records()
    finally:
        journal.close()
    assert projected[-1]["assertion_context_preview"] == failure["assertion_context_preview"]
    live._framefit_action_context.update({
        "framefit5": {"last_verdict": {"status": "FAIL",
                                     "tested_revision": "sha256:probe-revision",
                                     "current_revision": "sha256:probe-revision",
                                     "check_receipt_id": failure["receipt_id"]},
                      "read_facts": live._framefit_action_context["framefit5"]["read_facts"]},
        "journal_receipt_records": projected,
    })
    _menu(live, "request-one")
    first = _live_read(live, "request-one", "pkg/service.py",
                       {"path": "pkg/service.py", "start_line": 1, "end_line": 1})
    assert first[3]["requested_bytes_delivered"] > 0
    snapshot = json.loads(json.dumps(checkpoint_replay(live)))
    resumed, _ = _runtime(tmp_path)
    resumed._framefit_action_context = json.loads(json.dumps(live._framefit_action_context))
    resumed._latest_check_receipt = None
    resumed.host_check_receipts = []
    restore_replay(resumed, snapshot)
    assert _current_failure(live) == _current_failure(resumed)

    def wire(runtime):
        menu = _menu(runtime, "request-two")
        messages = [{"role": "user", "content": "[StateFrame]\ncurrent state"}]
        result, observation, _source, delivery = _live_read(
            runtime, "request-two", "pkg/service.py",
            {"path": "pkg/service.py", "start_line": 1, "end_line": 1})
        assert result == "OK" and delivery["unchanged_since_request"] is False
        messages.append({"role": "tool", "content": observation})
        return json.dumps({"messages": messages, "tools": menu},
                          sort_keys=True, separators=(",", ":")).encode()

    assert wire(live) == wire(resumed)


def test_abrupt_restart_replays_delivered_default_page_bytes(tmp_path, monkeypatch):
    from kernel_continuation_journal import ContinuationJournal

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_DELIVER_REQUESTED_BYTES", "on")
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    journal_parent = tmp_path / "journal"
    journal_parent.mkdir()
    marker = tmp_path / "checkpoint.json"
    child = """
import json, os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from kernel_continuation_journal import ContinuationJournal
from kernel_read_exec import execute_read, record_serialized_read_pages
from kernel_reread_result_truth import checkpoint_replay
root, parent, marker = map(Path, sys.argv[2:])
measurement = SimpleNamespace(owner=SimpleNamespace(workspace_revision=1), retained=None,
    step_ledger=None, _frame_budget_request_count=1)
status, first, _ = execute_read(root, root / 'module.py', 'module.py', 'module.py', 0,
    args={'path': 'module.py'}, measurement=measurement, step=1, delivery={})
assert status == 'OK'
record_serialized_read_pages(measurement, [{'role': 'tool', 'content': first}], 1)
journal = ContinuationJournal(parent, 'page-restart', {'origin': 'fixture'})
journal.emit('continuation_worker_checkpoint', checkpoint={
    'code_bookmark_replay': checkpoint_replay(measurement)})
marker.write_text(json.dumps({'path': str(journal.path), 'head': journal.head}))
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", child,
                    str(Path(__file__).resolve().parents[1] / "src/stateframe/engine"),
                    str(tmp_path), str(journal_parent), str(marker)],
                   check=True, timeout=scaled(20), env=os.environ.copy())
    saved = json.loads(marker.read_text())
    journal = ContinuationJournal.resume(
        Path(saved["path"]), task_id="page-restart", manifest={"origin": "fixture"},
        expected_head=saved["head"])
    try:
        checkpoint = next(row["checkpoint"] for row in journal.records()
                          if row["kind"] == "continuation_worker_checkpoint")
    finally:
        journal.close()
    assert len(checkpoint["code_bookmark_replay"]["pages"]) == 1

    corrupted = json.loads(json.dumps(checkpoint["code_bookmark_replay"]))
    corrupted["pages"][0][1] = corrupted["request_number"] + 1
    with pytest.raises(ValueError, match="invalid bookmark read replay"):
        restore_replay(SimpleNamespace(), corrupted)

    def measurement():
        return SimpleNamespace(owner=SimpleNamespace(workspace_revision=1), retained=None,
                               step_ledger=None, _frame_budget_request_count=1)

    uninterrupted = measurement()
    status, first, _ = execute_read(tmp_path, source, "module.py", "module.py", 0,
                                    args={"path": "module.py"}, measurement=uninterrupted,
                                    step=1, delivery={})
    assert status == "OK"
    record_serialized_read_pages(uninterrupted, [{"role": "tool", "content": first}], 1)
    resumed = measurement()
    restore_replay(resumed, checkpoint["code_bookmark_replay"])
    assert resumed._frame_budget_request_count == uninterrupted._frame_budget_request_count == 1
    uninterrupted._frame_budget_request_count += 1
    resumed._frame_budget_request_count += 1

    def reply(runtime, args):
        delivery = {}
        status, text, _ = execute_read(tmp_path, source, "module.py", "module.py", 0,
                                       args=args, measurement=runtime, step=2,
                                       delivery=delivery)
        assert status == "OK"
        return text.encode(), delivery["requested_bytes_delivered"]

    live_bytes, live_count = reply(uninterrupted, {"path": "module.py"})
    resumed_bytes, resumed_count = reply(resumed, {"path": "module.py"})
    assert live_bytes == resumed_bytes
    assert b"unchanged since request 1" in resumed_bytes
    assert live_count == resumed_count == 0
    legacy = dict(checkpoint["code_bookmark_replay"])
    legacy["schema"] = "stateframe-code-bookmark-read-replay/2"
    legacy.pop("pages")
    legacy.pop("request_number")
    legacy_runtime = measurement()
    restore_replay(legacy_runtime, legacy)
    legacy_runtime._frame_budget_request_count = 2
    legacy_bytes, legacy_count = reply(legacy_runtime, {"path": "module.py"})
    assert b"VALUE = 1" in legacy_bytes and legacy_count > 0
    ranged_bytes, ranged_count = reply(resumed, {"path": "module.py", "start_line": 1})
    assert b"VALUE = 1" in ranged_bytes and ranged_count > 0
    source.write_text("VALUE = 2\n")
    changed_bytes, changed_count = reply(resumed, {"path": "module.py"})
    assert b"VALUE = 2" in changed_bytes and changed_count > 0


def test_capability_floor_with_selected_skill_and_mcp_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    runtime, _events = _runtime(tmp_path)
    skill_schema = next(tool for tool in BUILTIN_SCHEMAS
                        if tool["function"]["name"] == "skill_load")
    mcp_schema = {"type": "function", "function": {"name": "mcp_repo__lookup",
                  "parameters": {"type": "object"}}}
    runtime._completion_owner_full_tools = [*TOOLS, skill_schema, mcp_schema]
    incoming = [*(tool for tool in TOOLS if tool["function"]["name"] != "read"),
                mcp_schema]
    names, schemas, _digest = _request_tool_floor_schemas(runtime, incoming)
    final = _apply_request_tool_floor(incoming, names, schemas)
    assert "read" in _names(final)
    assert "skill_load" in _names(final)
    assert "mcp_repo__lookup" in _names(final)
    assert schemas["skill_load"] == skill_schema


def test_inactive_replay_keeps_bookmark_state_and_menu(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, _ = _runtime(tmp_path)
    runtime._code_bookmarks_active = False
    (tmp_path / "module.py").write_text("value = 1\n")
    _menu(runtime, "request-14")
    _read(runtime, "request-14", "module.py")
    snapshot = json.loads(json.dumps(checkpoint_replay(runtime)))
    assert snapshot["active"] is False
    resumed, _ = _runtime(tmp_path)
    resumed.owner._receipts = list(runtime.owner._receipts)
    restore_replay(resumed, snapshot)
    assert resumed._code_bookmarks_active is False
    assert _names(_menu(runtime, "request-15")) == _names(_menu(resumed, "request-15"))


def test_watch_charges_one_unchanged_native_batch_and_restores_tail(monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    watch = UnboundedProgressWatch(threshold=10)
    revision = "sha256:" + "a" * 64
    for path in ("a.py", "b.py", "c.py"):
        watch._new(("read", path, revision))
    watch.streak = 5
    receipts = []
    for index, path in enumerate(("a.py", "b.py", "c.py")):
        receipts.append({"request_id": "origin" if index == 0 else f"action-{index}",
                         "kind": "read", "result_class": "OK", "target_path": path,
                         "delivery": {"revision": revision,
                                      "unchanged_since_request": True,
                                      "requested_bytes_delivered": 0,
                                      "effective_range": {"start_line": 1, "end_line": 1}}})
        watch.observe({}, receipts, boundary=index + 1, batch_origin="origin",
                      batch_complete=index == 2)
    assert watch.streak == 6
    resumed = UnboundedProgressWatch.from_snapshot(json.loads(json.dumps(watch.snapshot())))
    resumed.observe({}, receipts, boundary=4, batch_origin="origin", batch_complete=True)
    assert resumed.streak == 6
    resumed.observe({}, receipts, boundary=5, batch_origin="origin", batch_complete=True)
    assert resumed.streak == 6
    legacy = watch.snapshot()
    legacy["schema"] = "unbounded-progress-watch/1"
    del legacy["batch_tail_pending"]
    del legacy["seen_filter"]
    assert UnboundedProgressWatch.from_snapshot(legacy).streak == 6


def test_watch_does_not_collapse_mixed_or_unproved_batch():
    watch = UnboundedProgressWatch(threshold=10)
    revision = "sha256:" + "a" * 64
    watch._new(("read", "a.py", revision))
    watch.streak = 5
    read = {"request_id": "origin", "kind": "read", "result_class": "OK",
            "target_path": "a.py", "delivery": {"revision": revision,
            "unchanged_since_request": True, "requested_bytes_delivered": 0,
            "effective_range": {"start_line": 1, "end_line": 1}}}
    rows = [read]
    watch.observe({}, rows, boundary=1, batch_origin="origin")
    rows.append({"request_id": "action-2", "kind": "edit", "result_class": "REJECTED"})
    watch.observe({}, rows, boundary=2, batch_origin="origin")
    rows.append({**read, "request_id": "action-3"})
    watch.observe({}, rows, boundary=3, batch_origin="origin", batch_complete=True)
    assert watch.streak == 8
    watch.observe({}, rows, boundary=4, batch_origin="origin", batch_complete=True)
    assert watch.streak == 8


def test_watch_charges_early_unchanged_reads_in_completed_mixed_batch():
    watch = UnboundedProgressWatch(threshold=10)
    rows = []
    for index in range(3):
        rows.append({"request_id": "origin" if index == 0 else f"action-{index}",
                     "kind": "read", "result_class": "OK", "target_path": "same.py",
                     "delivery": {"revision": "sha256:same", "unchanged_since_request": True,
                                  "requested_bytes_delivered": 0,
                                  "effective_range": {"start_line": 1, "end_line": 1}}})
        assert not watch.observe({}, rows, boundary=index + 1, batch_origin="origin")
        assert watch.streak == 0
    rows.append({"request_id": "action-3", "kind": "edit", "result_class": "REJECTED"})
    assert not watch.observe({}, rows, boundary=4, batch_origin="origin", batch_complete=True)
    assert watch.streak == 4


def test_bookmarks_off_watch_snapshot_uses_release_schema(monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    watch = UnboundedProgressWatch()
    snapshot = watch.snapshot()
    assert snapshot["schema"] == "unbounded-progress-watch/1"
    assert snapshot["seen"] == []
    assert "batch_tail_pending" not in snapshot
    assert "seen_filter" not in snapshot
    assert UnboundedProgressWatch.from_snapshot(snapshot).snapshot() == snapshot
