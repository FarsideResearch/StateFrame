"""The live request menu bounds repeated reads that returned no source bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_read_lockout import append_frame_line
from kernel_read_exec import dispatch_live_read, execute_read, record_serialized_read_pages
from kernel_request_tool_surface import _request_tool_floor_schemas
from kernel_stateframe_framefit5 import _is_framefit5_host_context_panel
from kernel_reread_result_truth import checkpoint_replay, restore_replay
from kernel_stateframe_runtime import StateFrameRuntime


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
    _read(runtime, "probe-request-14", "pkg/batch/contracts.py")
    assert "read" in _names(_menu(runtime, "probe-request-14"))
    _read(runtime, "probe-request-15", "pkg/service.py")
    return runtime, events


def test_probe_state_next_request_has_no_read_and_factual_frame_line(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, events = _trigger(tmp_path)
    menu = _menu(runtime, "probe-request-15")
    assert "read" not in _names(menu)
    assert {"edit", "str_replace", "file_create", "run_tests", "authored_check", "done"} <= set(_names(menu))
    frame = [{"role": "user", "content": "[StateFrame]\ncurrent state"}]
    append_frame_line(runtime, frame)
    assert frame[0]["content"] == "[StateFrame]\ncurrent state"
    assert "pkg/batch/contracts.py, pkg/service.py were re-read unchanged" in frame[1]["content"]
    assert "assert 'BatchResult' in batch.__all__" in frame[1]["content"]
    assert "read is unavailable for this request" in frame[1]["content"]
    assert _is_framefit5_host_context_panel(frame[1])
    assert not _is_framefit5_host_context_panel({"role": "user", "content": frame[1]["content"]})
    assert [row["reason_code"] for row in events] == ["TWO_UNCHANGED_READ_REQUESTS"]
    assert "read" not in _names(_menu(runtime, "probe-request-15"))
    assert len(events) == 1


@pytest.mark.parametrize("kind", ("edit", "test"))
def test_edit_or_test_releases_read_immediately(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    runtime, events = _trigger(tmp_path)
    _menu(runtime, "probe-request-15")
    runtime.owner._receipts.append({"request_id": "probe-request-16", "kind": kind})
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
    assert not events


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
    runtime.owner._receipts.append({"request_id": "another-action-in-14",
                                    "kind": "edit", "result_class": "APPLIED"})
    assert "read" in _names(_menu(runtime, "request-15"))
    _read(runtime, "request-15", "module.py")
    assert "read" in _names(_menu(runtime, "request-16"))
    assert not events


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
