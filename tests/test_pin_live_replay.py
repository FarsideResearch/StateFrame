"""Offline replay of bm5_b's host receipts and final request frames.

The fixture contains recorded host fields and source bytes only. Check receipt
projection and repeated-read notes are rebuilt through production functions.
"""

from __future__ import annotations

import copy
import hashlib
import json
from bisect import bisect_right
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_code_bookmarks
from kernel_check_receipts import _emit as emit_check_receipt
from kernel_code_bookmarks import windows
from kernel_continuation_journal import ContinuationJournal
from kernel_read_exec import _source_identity
from kernel_read_lockout import append_frame_line
from kernel_reread_result_truth import result_note
from kernel_stateframe_framefit5 import _serialize_state_message
from kernel_stateframe_runtime import StateFrameRuntime


FIXTURE = Path(__file__).parent / "fixtures" / "pin_bm5_b"


def _recorded():
    return json.loads((FIXTURE / "recorded.json").read_text())


def _source_root(tmp_path, recorded):
    # The production source reader refuses symlinked parent components. macOS
    # exposes its temporary directory through /var -> /private/var.
    root = (tmp_path / "source").resolve()
    for row in recorded["cases"][0]["frame"]["read_facts"]:
        path = row["path"]
        target = root / path
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((FIXTURE / "source" / (path + ".txt")).read_bytes())
        assert "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest() == row[
            "current_revision"]
    return root


def _project_checks(tmp_path, recorded):
    tmp_path.mkdir(parents=True, exist_ok=True)
    journal = ContinuationJournal(tmp_path, "bm5_b", {"origin": recorded["origin"]})
    try:
        for check in recorded["checks"]:
            journal.emit("check_receipt", **{key: value for key, value in check.items()
                                             if key not in {"kind", "sequence"}})
        return journal.framefit_receipt_records()
    finally:
        journal.close()


def _runtime(case, checks, root, actions=()):
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(
        root=root, workspace_revision=8,
        _receipts=list(actions),
    )
    runtime.edit_contract = {"edit_authority": case["edit_authority"]}
    runtime._framefit_action_context = {
        "journal_receipt_records": checks,
        "framefit5": case["frame"],
    }
    runtime._latest_check_receipt = None
    runtime.host_check_receipts = []
    runtime.binding = lambda: {}  # The recorded non-rich binding has no product_revision.
    runtime._base_tool_menu = lambda tools, _messages: list(tools)
    runtime._replay_events = []
    runtime.journal = SimpleNamespace(emit=lambda kind, **fields: runtime._replay_events.append(
        {"kind": kind, **fields}))
    runtime._code_bookmarks_active = False
    return runtime


def _assert_activated(runtime, frame):
    decision = {}
    bookmarks = kernel_code_bookmarks.collect(runtime, frame, decision)
    assert bookmarks, "recorded failure must not silently lose every bookmark"
    assert decision["reason"] == "activated", decision
    assert any(row["path"] == "tests/test_registration_serialization.py"
               and row["symbol"] == "test_batch_capability_is_registered_to_real_runner"
               and row["rank"] == 1 and row["read_only"] is True
               for row in bookmarks)
    assert any(row["path"] == "pkg/registry.py" and row["symbol"] == "lookup"
               and row["rank"] == 1 for row in bookmarks)
    assert any(row["path"] == "pkg/batch/__init__.py" and row["symbol"] == "run_batch"
               and row["rank"] == 1 for row in bookmarks)
    return bookmarks


def _request_number(turn, starts):
    return bisect_right(starts, turn)


def _replayed_actions(recorded, checks, root):
    """Recompute notes from exact recorded read paths, revisions and turn order."""
    case = recorded["cases"][0]
    runtime = _runtime(case, checks, root)
    starts = [row["turn"] for row in recorded["request_turns"]]
    rows = copy.deepcopy(recorded["actions"])
    for row in rows:
        if row["kind"] != "read" or row.get("result_class") != "OK":
            continue
        path = row["target_path"]
        runtime.owner.workspace_revision = row["workspace_revision"]
        runtime.current_request_id = row["request_id"]
        runtime._frame_budget_request_count = _request_number(row["turn"], starts)
        note = result_note(runtime, _source_identity(runtime.owner.root,
                           runtime.owner.root / path), str(row["workspace_revision"]),
                           row["delivery"]["revision"], row["turn"])
        row["replayed_result_note"] = note
        row["delivery"]["unchanged_since_request"] = bool(
            note and note.startswith("[read] requested bytes are unchanged since request "))
    return rows


def _menu_replay(recorded, checks, root, actions, *, stop_on_lockout=False):
    runtime = _runtime(recorded["cases"][0], checks, root)
    decisions = []
    for case in recorded["cases"]:
        runtime._framefit_action_context["framefit5"] = case["frame"]
        runtime.owner._receipts = [row for row in actions if row["turn"] < case["turn"]]
        runtime.owner.workspace_revision = 8
        runtime.current_request_id = case["request_id"]
        tools = [{"type": "function", "function": {"name": name,
                  "parameters": {"type": "object"}}} for name in case["menu"]]
        menu = StateFrameRuntime.tool_menu(runtime, tools, [])
        names = [tool["function"]["name"] for tool in menu]
        messages = [{"role": "user", "content": "[StateFrame]\nrecorded request state"}]
        append_frame_line(runtime, messages)
        decisions.append((case["request"], names, json.dumps(messages).encode(),
                          tuple(runtime._replay_events)))
        if stop_on_lockout and "read" not in names:
            break  # Later recorded read actions require a menu the fixed run excludes.
    return decisions


def test_recorded_bm5_b_bookmarks_and_rebuilt_request_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    recorded = _recorded()
    root = _source_root(tmp_path, recorded)
    checks = _project_checks(tmp_path, recorded)
    assert any("tests/test_registration_serialization.py:10: AssertionError"
               in row.get("assertion_context_preview", "") for row in checks)
    for case in recorded["cases"]:
        runtime = _runtime(case, checks, root)
        bookmarks = _assert_activated(runtime, case["frame"])
        assert case["decision"] == {"reason": "no_symbols", "collected_count": 0}
        if case["request"] == 19:
            payload = copy.deepcopy(recorded["request19_state_payload"])
            payload["state"]["framefit5"]["code_bookmarks"] = bookmarks
            messages = [{"role": "user", "content": _serialize_state_message(payload)}]
            messages.extend({"role": "user", "content": row["content"]}
                            for row in windows(runtime.owner.root, bookmarks))
            request_bytes = json.dumps({"messages": messages},
                                       separators=(",", ":")).encode()
            assert b'test_batch_capability_is_registered_to_real_runner' in request_bytes
            assert b'pkg/registry.py::lookup' in request_bytes
            assert b'pkg/batch/__init__.py::run_batch' in request_bytes


def test_planted_empty_collector_fails_replay_control(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    recorded = _recorded()
    root = _source_root(tmp_path, recorded)
    runtime = _runtime(recorded["cases"][-1], _project_checks(tmp_path, recorded), root)
    monkeypatch.setattr(kernel_code_bookmarks, "collect", lambda *_args, **_kw: [])
    with pytest.raises(AssertionError):
        _assert_activated(runtime, recorded["cases"][-1]["frame"])


def test_legacy_check_summary_keeps_quoted_marker_in_assertion():
    assertion = "tests/test_registration_serialization.py:10: AssertionError: runner_tail=literal"
    record = {"summary": f"status=FAIL assertion_context={assertion!r} runner_tail='end'"}
    assert ContinuationJournal._framefit_failure_context(record) == assertion


def test_check_event_bookmark_switch_preserves_off_bytes(monkeypatch):
    events = []
    runtime = SimpleNamespace(journal=SimpleNamespace(emit=lambda *args, **fields:
                              events.append((args, fields))))
    receipt = {
        "receipt_id": "check-1", "execution_id": "execution-1",
        "host_execution_id": "host-1", "full_output_handle": "check-1",
        "selector": "tests/test_registration_serialization.py", "status": "FAIL",
        "outcome": "FAIL", "reached_observation": True, "failure_phase": "pytest",
        "revision": "sha256:current", "summary": "failed", "stdout_bytes": 0,
        "stderr_bytes": 0, "stdout_omitted_bytes": 0, "stderr_omitted_bytes": 0,
        "assertion_context_omitted_bytes": 0,
        "assertion_context_preview": "tests/test_registration_serialization.py:10: AssertionError",
    }
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    emit_check_receipt(runtime, receipt)
    off_bytes = json.dumps(events[-1], sort_keys=True)
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    emit_check_receipt(runtime, receipt)
    _args, on_fields = events[-1]
    assert on_fields["assertion_context_preview"] == receipt["assertion_context_preview"]
    assert json.dumps((events[-1][0], {key: value for key, value in on_fields.items()
                                      if key != "assertion_context_preview"}),
                      sort_keys=True) == off_bytes


def test_check_projection_bookmark_switch_preserves_off_shape(tmp_path, monkeypatch):
    recorded = _recorded()
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    off = _project_checks(tmp_path / "off", recorded)
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    on = _project_checks(tmp_path / "on", recorded)
    off_checks = [row for row in off if row.get("kind") == "check_receipt"]
    on_checks = [row for row in on if row.get("kind") == "check_receipt"]
    assert len(off_checks) == len(on_checks) == 4
    assert all("assertion_context_preview" not in row for row in off_checks)
    assert any("assertion_context_preview" in row for row in on_checks)


def test_recorded_bm5_b_notes_alone_cannot_lock_without_live_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    recorded = _recorded()
    root = _source_root(tmp_path, recorded)
    checks = _project_checks(tmp_path, recorded)
    original = _menu_replay(recorded, checks, root, recorded["actions"])
    assert all("read" in names for _, names, _, _ in original)
    actions = _replayed_actions(recorded, checks, root)
    assert b"unchanged since request 14" in json.dumps([
        {"role": "tool", "content": row["replayed_result_note"]}
        for row in actions if row.get("replayed_result_note")]).encode()
    replayed = _menu_replay(recorded, checks, root, actions)
    assert all("read" in names and b"[Read lockout]" not in wire and not events
               for _, names, wire, events in replayed)


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    data = _recorded()
    with TemporaryDirectory() as directory:
        temporary = Path(directory)
        root = _source_root(temporary, data)
        proof = _project_checks(temporary, data)
        for item in data["cases"]:
            target = _runtime(item, proof, root)
            decision = {}
            marks = kernel_code_bookmarks.collect(target, item["frame"], decision)
            print(f"request {item['request']}: recorded={item['decision']['reason']} "
                  f"replay={decision['reason']} bookmarks={len(marks)}")
        for number, names, _wire, _events in _menu_replay(
                data, proof, root, _replayed_actions(data, proof, root),
                stop_on_lockout=True):
            print(f"request {number}: read={'read' in names} grep={'grep' in names} "
                  f"shell={'shell' in names}")
