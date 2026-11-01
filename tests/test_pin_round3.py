"""Regression checks for the shipped read clock and bounded stall memory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateframe import profiles
from kernel_framefit_receipt_history import ReceiptHistory
from kernel_read_exec import dispatch_live_read, execute_read
from kernel_stateframe_runtime import StateFrameRuntime, note_provider_request
from kernel_tool_call_loop_breaker import UnboundedProgressWatch


def test_shipped_profile_numbers_repeated_read_and_charges_batch_once(tmp_path, monkeypatch):
    switches = profiles.load(profiles.DEFAULT)
    assert switches["STATEFRAME_F4"] == "on"
    assert switches["STATEFRAME_FRAMEFIT"] == "on"
    assert switches["STATEFRAME_CODE_BOOKMARKS"] == "on"
    assert switches["STATEFRAME_READ_LOCKOUT"] == "off"
    for name in ("STATEFRAME_CODE_BOOKMARKS", "STATEFRAME_DELIVER_REQUESTED_BYTES",
                 "STATEFRAME_READ_LOCKOUT"):
        monkeypatch.setenv(name, switches[name])

    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    receipts = []
    owner = SimpleNamespace(root=tmp_path, workspace_revision=1, _receipts=receipts)
    owner.receipt = lambda kind, intent, fingerprint, result, **fields: receipts.append(
        {"kind": kind, "intent": intent, "fingerprint": fingerprint,
         "result_class": result, **fields})
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.owner = owner
    runtime.retained = None
    runtime.step_ledger = None
    runtime.journal = SimpleNamespace(emit=lambda *_args, **_kwargs: None)

    first_delivery = {}
    status, first, _ = execute_read(
        tmp_path, source, "module.py", "module.py", 0,
        args={"path": "module.py"}, measurement=runtime, step=1,
        delivery=first_delivery)
    assert status == "OK"
    note_provider_request(runtime, [{"role": "tool", "content": first}])
    assert runtime._frame_budget_request_count == 1
    note_provider_request(runtime, [])
    assert runtime._frame_budget_request_count == 2

    revision = first_delivery["revision"]
    watch = UnboundedProgressWatch(threshold=10)
    prior = {"kind": "read", "result_class": "OK", "target_path": "module.py",
             "delivery": {"revision": revision}}
    watch.observe({}, [prior], boundary=1)
    for boundary in range(2, 7):
        watch.observe({}, [prior], boundary=boundary)
    assert watch.streak == 5
    batch = []
    for index in range(3):
        result, observation, _content, delivery = dispatch_live_read(
            tmp_path, source, "module.py", "module.py", 0,
            args={"path": "module.py"}, measurement=runtime, step=index + 2,
            bolt=owner, policy=None, resolved_path="module.py")
        assert result == "OK"
        assert "unchanged since request 1" in observation
        assert delivery["unchanged_since_request"] is True
        assert delivery["requested_bytes_delivered"] == 0
        row = dict(receipts[-1], request_id="batch" if index == 0 else f"action-{index}")
        batch.append(row)
        watch.observe({}, batch, boundary=index + 7, batch_origin="batch",
                      batch_complete=index == 2)
    assert watch.streak == 6


def test_bookmarks_off_multi_selector_context_and_request_match_release(monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    events = [
        {"kind": "check_receipt", "selector": "tests/a.py", "receipt_id": "a", "status": "FAIL"},
        {"kind": "check_receipt", "selector": "tests/b.py", "receipt_id": "b", "status": "PASS"},
    ]
    history = ReceiptHistory()
    for event in events:
        history.observe(event)

    # Captured from the v0.2.1 reducer, with the same two selectors and wire
    # serialization. Compare the complete context and complete request bytes.
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "pin_receipt_021.json").read_text())
    assert fixture["tag"] == "v0.2.1"
    assert history.context() == fixture["context"]
    request = json.dumps({"messages": [{"role": "user", "content": history.context()}],
                          "tools": []}, sort_keys=True, separators=(",", ":")).encode()
    assert request == fixture["request"].encode()


@pytest.mark.parametrize("bookmarks", ["on", "off"])
def test_watch_remembers_more_than_4096_read_identities_after_checkpoint(monkeypatch, bookmarks):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", bookmarks)
    watch = UnboundedProgressWatch(threshold=32)
    revision = "sha256:" + hashlib.sha256(b"same source").hexdigest()
    for index in range(4097):
        assert watch._new(("read", f"path-{index}.py", revision))
    restored = UnboundedProgressWatch.from_snapshot(json.loads(json.dumps(watch.snapshot())))
    assert len(restored._seen) == 4096
    damaged = watch.snapshot()
    damaged["seen_filter"] = "A" * len(damaged["seen_filter"])
    with pytest.raises(ValueError, match="invalid progress watch identities"):
        UnboundedProgressWatch.from_snapshot(damaged)
    receipts = []
    for index in range(32):
        row = {"kind": "read", "result_class": "OK",
               "target_path": f"path-{index}.py", "delivery": {"revision": revision}}
        receipts.append(row)
        assert restored.observe({}, receipts, boundary=index + 1) is (index == 31)
    assert restored.streak == 32
