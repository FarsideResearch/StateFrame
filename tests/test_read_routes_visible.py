"""Read routes are never hidden by the engine on its own.

An unchanged re-read used to trigger a self-inflicted stop: the engine dropped
read, grep and shell from the menu for the next request. That path is gone, so
the menu must keep offering read routes whatever the bookmark and lockout
switches say. STATEFRAME_READ_LOCKOUT is retired and nothing reads it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

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


def test_read_routes_stay_offered_after_an_unchanged_reread(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_READ_LOCKOUT", "on")
    runtime, events = _trigger(tmp_path)
    names = _names(_menu(runtime, "probe-request-15"))
    assert {"read", "grep", "shell"} <= set(names)
    assert not [event for event in events if event["kind"] == "read_lockout_decision"]


def test_read_routes_stay_offered_with_lockout_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.delenv("STATEFRAME_READ_LOCKOUT", raising=False)
    runtime, events = _trigger(tmp_path)
    names = _names(_menu(runtime, "probe-request-15"))
    assert {"read", "grep", "shell"} <= set(names)
    assert not [event for event in events if event["kind"] == "read_lockout_decision"]
