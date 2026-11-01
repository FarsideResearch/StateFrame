"""The live request menu bounds repeated reads that returned no source bytes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_read_lockout import append_frame_line
from kernel_read_exec import dispatch_live_read, execute_read, record_serialized_read_pages
from kernel_request_tool_surface import _request_tool_floor_schemas
from kernel_stateframe_framefit5 import _is_framefit5_host_context_panel
from kernel_reread_result_truth import checkpoint_replay, restore_replay
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_tool_call_loop_breaker import UnboundedProgressWatch, stop_unbounded_no_progress


TOOLS = [
    {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
    # Ordered as the recorded T2 probe's request 14 menu.
    for name in ("edit", "str_replace", "read", "read_check_receipt", "run_tests",
                 "grep", "file_delete", "file_move", "shell", "authored_check",
                 "done", "file_create")
]


@pytest.fixture(autouse=True)
def _opt_in_lockout(monkeypatch):
    monkeypatch.setenv("STATEFRAME_READ_LOCKOUT", "on")


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


def test_probe_state_next_request_has_no_read_and_factual_frame_line(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, events = _trigger(tmp_path)
    menu = _menu(runtime, "probe-request-15")
    assert "read" not in _names(menu)
    assert {"edit", "str_replace", "file_create", "run_tests", "authored_check", "done"} <= set(_names(menu))
    frame = [{"role": "user", "content": "[StateFrame]\ncurrent state"}]
    append_frame_line(runtime, frame)
    assert frame[0]["content"] == "[StateFrame]\ncurrent state"
    assert "pkg/service.py were re-read unchanged" in frame[1]["content"]
    assert "assert 'BatchResult' in batch.__all__" in frame[1]["content"]
    assert "read is unavailable for this request" in frame[1]["content"]
    assert _is_framefit5_host_context_panel(frame[1])
    assert not _is_framefit5_host_context_panel({"role": "user", "content": frame[1]["content"]})
    assert [row["reason_code"] for row in events if row["kind"] == "read_lockout_decision"] == [
        "UNCHANGED_READ_REQUEST"]
    assert "read" not in _names(_menu(runtime, "probe-request-15"))
    assert len(events) == 1


@pytest.mark.parametrize("kind", ("edit", "test"))
def test_edit_or_test_releases_read_immediately(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, events = _trigger(tmp_path)
    _menu(runtime, "probe-request-15")
    runtime.owner._receipts.append({"request_id": "probe-request-15", "kind": kind,
                                    "result_class": "APPLIED" if kind == "edit" else "FAIL"})
    assert "read" in _names(_menu(runtime, "probe-request-16"))
    assert events[-1]["reason_code"] == "NEW_EVIDENCE"


def test_lockout_caps_at_two_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, events = _trigger(tmp_path)
    assert "read" not in _names(_menu(runtime, "probe-request-15"))
    assert "read" not in _names(_menu(runtime, "probe-request-16"))
    assert "read" in _names(_menu(runtime, "probe-request-17"))
    assert events[-1]["reason_code"] == "REQUEST_CAP"


def test_unseen_or_changed_source_never_triggers(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, _events = _trigger(tmp_path)
    (tmp_path / "pkg/unseen.py").write_text("NEW = 1\n")
    _read(runtime, "probe-request-15", "pkg/unseen.py", unchanged=False)
    assert "read" in _names(_menu(runtime, "probe-request-15"))

    runtime, events = _trigger(tmp_path)
    (tmp_path / "pkg/service.py").write_text("value = 2\n")
    assert "read" in _names(_menu(runtime, "probe-request-15"))
    assert not [row for row in events if row["kind"] == "read_lockout_decision"]


def test_checkpoint_restores_same_menu_decision(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, _events = _trigger(tmp_path)
    original = _names(_menu(runtime, "probe-request-15"))
    snapshot = json.loads(json.dumps(checkpoint_replay(runtime)))
    resumed, _ = _runtime(tmp_path)
    resumed.owner._receipts = list(runtime.owner._receipts)
    resumed.current_request_id = runtime.current_request_id
    restore_replay(resumed, snapshot)
    assert _names(_menu(resumed, "probe-request-15")) == original
    assert resumed._read_lockout_notice == runtime._read_lockout_notice


def test_legacy_lockout_checkpoint_resumes_unlocked(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, _ = _trigger(tmp_path)
    _menu(runtime, "probe-request-15")
    old = dict(checkpoint_replay(runtime)["read_lockout"])
    old["schema"] = "stateframe-read-lockout/1"
    del old["counted"]
    resumed, _ = _runtime(tmp_path)
    from kernel_read_lockout import restore
    restore(resumed, old)
    assert resumed._read_lockout_state["remaining"] == 0
    assert resumed._read_lockout_state["streak"] == 0
    assert "read" in _names(_menu(resumed, "probe-request-16"))


def test_production_base_menu_projects_lockout(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_LEAN", "on")
    runtime, _events = _trigger(tmp_path)
    del runtime._base_tool_menu
    menu = _menu(runtime, "probe-request-15")
    assert "read" not in _names(menu)
    assert {"edit", "str_replace", "run_tests", "done"} <= set(_names(menu))


def test_changed_after_trigger_or_deferred_source_releases(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, events = _trigger(tmp_path)
    _menu(runtime, "probe-request-15")
    (tmp_path / "pkg/service.py").write_text("value = 3\n")
    assert "read" in _names(_menu(runtime, "probe-request-15"))
    assert events[-1]["reason_code"] == "FRESH_READ_REQUIRED"

    runtime, events = _trigger(tmp_path)
    runtime._request_deferred_source_paths = ("pkg/unseen.py",)
    assert "read" in _names(_menu(runtime, "probe-request-15"))
    assert not events


def test_bookmark_kill_switch_keeps_menu_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    runtime, events = _trigger(tmp_path)
    expected = json.dumps(TOOLS, sort_keys=True)
    assert json.dumps(_menu(runtime, "probe-request-15"), sort_keys=True) == expected
    assert not events
    assert not hasattr(runtime, "_read_lockout_state")


def test_default_lockout_off_leaves_menu_journal_and_checkpoint_untouched(tmp_path, monkeypatch):
    monkeypatch.delenv("STATEFRAME_READ_LOCKOUT", raising=False)
    runtime, events = _trigger(tmp_path)
    assert _names(_menu(runtime, "probe-request-15")) == _names(TOOLS)
    assert not events
    assert not hasattr(runtime, "_read_lockout_state")
    assert "read_lockout" not in checkpoint_replay(runtime)

def test_bookmarks_off_repeated_read_matches_release_bytes(tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    monkeypatch.setenv("STATEFRAME_READ_LOCKOUT", "off")
    monkeypatch.setenv("STATEFRAME_DELIVER_REQUESTED_BYTES", "on")
    baseline_source = subprocess.check_output([
        "git", "show", "v0.2.1:src/stateframe/engine/kernel_read_exec.py",
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




def test_live_read_receipt_marks_only_the_exact_unchanged_notice(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_DELIVER_REQUESTED_BYTES", "on")
    path = tmp_path / "module.py"
    path.write_text("VALUE = 1\n")
    measurement = SimpleNamespace(
        owner=SimpleNamespace(workspace_revision=1), retained=None,
        step_ledger=None, _code_bookmarks_active=True,
        _frame_budget_request_count=13,
        journal=SimpleNamespace(emit=lambda *_args, **_kwargs: None),
    )
    status, first, _ = execute_read(
        tmp_path, path, path.name, path.name, 0,
        args={"path": path.name}, measurement=measurement, step=1, delivery={})
    assert status == "OK"
    record_serialized_read_pages(measurement, [{"role": "tool", "content": first}], 13)
    measurement._frame_budget_request_count = 14

    class Owner:
        def __init__(self):
            self._receipts = []

        def receipt(self, kind, intent, fingerprint, result, **fields):
            self._receipts.append({"kind": kind, "intent": intent,
                                   "fingerprint": fingerprint, "result_class": result,
                                   **fields})

    owner = Owner()
    result, observation, _source, delivery = dispatch_live_read(
        tmp_path, path, path.name, path.name, 0,
        args={"path": path.name}, measurement=measurement, step=2,
        bolt=owner, policy=None, resolved_path=path.name)
    assert result == "OK"
    assert "unchanged since request 13" in observation
    assert delivery["unchanged_since_request"] is True
    assert owner._receipts[-1]["delivery"]["unchanged_since_request"] is True
    assert delivery["requested_bytes_delivered"] == 0
    path.write_text("VALUE = 2\n")
    _result, observation, _source, delivery = dispatch_live_read(
        tmp_path, path, path.name, path.name, 0,
        args={"path": path.name}, measurement=measurement, step=3,
        bolt=owner, policy=None, resolved_path=path.name)
    assert "VALUE = 2" in observation
    assert delivery["unchanged_since_request"] is False
    assert delivery["requested_bytes_delivered"] > 0


def test_dispatched_fresh_ranges_and_new_file_cannot_start_lockout(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_DELIVER_REQUESTED_BYTES", "on")
    runtime, events = _runtime(tmp_path)
    for path in ("pkg/batch/contracts.py", "pkg/service.py", "pkg/new.py"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ONE = 1\nTWO = 2\nTHREE = 3\n")
    for number, path, arguments in (
            (1, "pkg/batch/contracts.py", {"path": "pkg/batch/contracts.py", "start_line": 1, "end_line": 1}),
            (2, "pkg/batch/contracts.py", {"path": "pkg/batch/contracts.py", "start_line": 2, "end_line": 2}),
            (3, "pkg/new.py", {"path": "pkg/new.py"}),
            (4, "pkg/service.py", {"path": "pkg/service.py"})):
        request = f"request-{number}"
        _menu(runtime, request)
        result, observation, _source, delivery = _live_read(runtime, request, path, arguments)
        assert result == "OK" and "=" in observation
        assert delivery["unchanged_since_request"] is False
        assert delivery["requested_bytes_delivered"] > 0
        assert "read" in _names(_menu(runtime, f"request-{number + 1}"))
    assert not [row for row in events if row["kind"] == "read_lockout_decision"]


@pytest.mark.parametrize("result", ["REJECTED", "FAILED", "NO_OP"])
def test_dispatched_rejected_edit_does_not_release_lock(tmp_path, monkeypatch, result):
    from kernel_bolt import KernelBolt

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, events = _trigger(tmp_path)
    assert "read" not in _names(_menu(runtime, "probe-request-15"))
    bolt = KernelBolt.__new__(KernelBolt)
    bolt.root = tmp_path
    bolt._receipts = runtime.owner._receipts
    bolt.run_id = "pin-rejected"
    bolt.current_step = 15
    bolt.workspace_revision = 1
    bolt.current_request_id = "probe-request-15"
    bolt.audit_journal = runtime.journal
    bolt.measurement = None
    runtime.owner = bolt
    bolt.receipt("edit", "replace_fn pkg/service.py", "pkg/service.py",
                 result, detail="edit was not applied", target_path="pkg/service.py")
    assert bolt._receipts[-1]["request_id"] == "probe-request-15"
    assert bolt.workspace_revision == 1
    assert "read" not in _names(_menu(runtime, "probe-request-16"))
    assert [row["reason_code"] for row in events if row["kind"] == "read_lockout_decision"] == [
        "UNCHANGED_READ_REQUEST"]
    assert "read" in _names(_menu(runtime, "probe-request-17"))


@pytest.mark.parametrize("menu", [
    ("read", "done"), ("read", "edit", "done"),
    ("read", "run_tests", "done"), ("read", "edit", "run_tests", "done"),
])
def test_lockout_needs_visible_editable_source_and_repair_route(tmp_path, monkeypatch, menu):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, _events = _trigger(tmp_path.resolve())
    selected = [tool for tool in TOOLS if tool["function"]["name"] in menu]
    runtime.current_request_id = "probe-request-15"
    names = _names(StateFrameRuntime.tool_menu(runtime, selected, []))
    assert ("read" not in names) == ("edit" in menu and "run_tests" in menu)
    runtime._code_bookmark_carry[0]["read_only"] = True
    runtime._code_bookmark_carry[1]["read_only"] = True
    assert "read" in _names(StateFrameRuntime.tool_menu(runtime, selected, []))


def test_other_mode_or_hidden_source_never_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, _ = _trigger(tmp_path)
    runtime.stateframe_framefit_mode = "off"
    assert "read" in _names(_menu(runtime, "probe-request-15"))
    runtime, _ = _trigger(tmp_path)
    runtime._code_bookmark_carry = []
    assert "read" in _names(_menu(runtime, "probe-request-15"))


def test_progress_watch_pauses_idle_charge_during_lockout_without_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, _events = _trigger(tmp_path)
    _menu(runtime, "probe-request-15")
    watch = UnboundedProgressWatch(threshold=2)
    watch.observe(runtime.binding(), runtime.owner._receipts, boundary=0)
    watch.streak = 1
    assert not stop_unbounded_no_progress(runtime, watch, boundary=10)
    assert watch.streak == 1
    assert not stop_unbounded_no_progress(runtime, watch, boundary=11)
    assert watch.streak == 1
    _menu(runtime, "probe-request-16")
    _menu(runtime, "probe-request-17")
    assert stop_unbounded_no_progress(runtime, watch, boundary=12)
    assert watch.streak == 2


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
        append_frame_line(runtime, messages)
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
                   check=True, timeout=20, env=os.environ.copy())
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


def test_final_capability_floor_cannot_reintroduce_locked_read(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, _events = _trigger(tmp_path)
    _menu(runtime, "probe-request-15")
    runtime.available_tools = TOOLS
    names, schemas, _digest = _request_tool_floor_schemas(runtime, TOOLS)
    assert "read" not in names
    assert "read" not in schemas
    assert {"edit", "str_replace", "run_tests", "done"} <= set(names)

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    names, schemas, _digest = _request_tool_floor_schemas(runtime, TOOLS)
    assert "read" in names and "read" in schemas


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


def test_mixed_provider_batch_cannot_count_as_unchanged_reads(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, events = _runtime(tmp_path)
    (tmp_path / "module.py").write_text("value = 1\n")
    _menu(runtime, "request-14")
    _read(runtime, "request-14", "module.py")
    runtime.owner._receipts.append({"request_id": "request-14",
                                    "kind": "edit", "result_class": "APPLIED"})
    assert "read" in _names(_menu(runtime, "request-15"))
    _read(runtime, "request-15", "module.py")
    assert "read" not in _names(_menu(runtime, "request-16"))
    assert [row["reason_code"] for row in events] == ["UNCHANGED_READ_REQUEST"]


def test_locked_menu_is_enforced_at_dispatch(tmp_path, monkeypatch):
    from kernel_completion_owner import request_menu_action_admissible

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, _ = _trigger(tmp_path)
    names = _names(_menu(runtime, "probe-request-15"))
    runtime._completion_owner_request_contract = {
        "request_id": "probe-request-15", "advertised_actions": names,
    }
    for tool in ("read", "grep", "shell", "list_files", "where_defined",
                 "who_calls", "references"):
        assert tool not in names
        assert not request_menu_action_admissible(runtime, tool, "probe-request-15")
    assert request_menu_action_admissible(runtime, "edit", "probe-request-15")


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


def test_same_path_with_two_revisions_is_not_one_unchanged_batch(tmp_path):
    from kernel_read_lockout import _request_reads

    runtime, _events = _runtime(tmp_path)
    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    current = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    rows = [{"kind": "read", "result_class": "OK", "target_path": "module.py",
             "delivery": {"unchanged_since_request": True,
                          "requested_bytes_delivered": 0,
                          "effective_range": {"start_line": 1, "end_line": 1},
                          "revision": revision}}
            for revision in ("sha256:" + "a" * 64, current)]
    assert _request_reads(runtime, rows) is None
