"""Offline provider-batch replay of the recorded bm6diag_a stall."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from kernel_read_exec import dispatch_live_read, execute_read
from kernel_final_wire_allocation import publish_code_bookmark_transition
from kernel_read_lockout import append_frame_line
from kernel_request_tool_surface import _request_tool_floor_schemas
from kernel_stateframe_runtime import StateFrameRuntime, note_provider_request
from kernel_tool_call_loop_breaker import UnboundedProgressWatch, stop_unbounded_no_progress


FIXTURE = Path(__file__).parent / "fixtures" / "pin_bm6diag_a"


def _recorded():
    return json.loads((FIXTURE / "recorded.json").read_text())


def _source_root(tmp_path, cases):
    root = (tmp_path / "source").resolve()
    for case in cases:
        for call in case["calls"]:
            path = call["path"]
            target = root / path
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((FIXTURE / "source" / (path + ".txt")).read_bytes())
        for receipt in case["receipts"]:
            path = receipt["target_path"]
            target = root / path
            assert "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest() == receipt["revision"]
    return root


def _runtime(root, recorded):
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    events = []
    runtime.journal = SimpleNamespace(emit=lambda kind, **fields: events.append({"kind": kind, **fields}))
    runtime.owner = SimpleNamespace(root=root, _receipts=[], workspace_revision=8)
    runtime.owner.receipt = lambda kind, intent, fingerprint, result, **fields: (
        runtime.owner._receipts.append({"request_id": runtime.current_request_id,
                                        "kind": kind, "intent": intent,
                                        "fingerprint": fingerprint,
                                        "result_class": result, **fields}))
    runtime.binding = lambda: {"product_revision": recorded["check"]["revision"]}
    runtime._latest_check_receipt = recorded["check"]
    runtime.host_check_receipts = [recorded["check"]]
    runtime._base_tool_menu = lambda tools, _messages: list(tools)
    runtime.stateframe_framefit_mode = "on"
    runtime._code_bookmarks_active = True
    runtime.retained = None
    runtime.step_ledger = None
    runtime.native_batch = None
    runtime.system_outcome = None
    runtime.stateframe_hard_budget_only_terminals_mode = "off"
    runtime.step_limit = None
    return runtime, events


def _case(runtime, case, carry=None):
    frame = case["frame"]["state"]["framefit5"]
    runtime._framefit_action_context = {"framefit5": frame}
    runtime.edit_contract = {"edit_authority": case["frame"]["state"]["edit_authority"]}
    publish_code_bookmark_transition(
        runtime, (True, list(carry if carry is not None else frame["code_bookmarks"])))
    runtime.current_request_id = case["receipts"][0]["request_id"]


def _tools(case):
    return [{"type": "function", "function": {"name": name,
             "parameters": {"type": "object"}}} for name in case["menu"]]


def _menu(runtime, case):
    offered = _tools(case)
    tools = StateFrameRuntime.tool_menu(runtime, offered, [])
    messages = [{"role": "user", "content": "[StateFrame] " + json.dumps(case["frame"],
                 sort_keys=True, separators=(",", ":"))}]
    messages.extend({"role": "user", "content": text} for text in case["source_messages"])
    append_frame_line(runtime, messages)
    names = [tool["function"]["name"] for tool in tools]
    runtime.available_tools = offered
    final_names, _schemas, _digest = _request_tool_floor_schemas(runtime, offered)
    assert ("read" in final_names) == ("read" in names)
    if runtime.native_batch is None:
        note_provider_request(runtime, messages)
    return names, json.dumps({"messages": messages, "tools": tools},
                             separators=(",", ":")).encode()


def _prime_pages(runtime, recorded):
    for _ in range(11):
        note_provider_request(runtime, [])
    replies = []
    for prior in recorded["prior_reads"]:
        path = prior["path"]
        target = runtime.owner.root / path
        delivery = {}
        status, text, _ = execute_read(
            runtime.owner.root, target, path, path, 0,
            args={"path": path}, measurement=runtime, step=prior["turn"],
            delivery=delivery)
        assert status == "OK"
        assert delivery["revision"] == prior["revision"]
        assert delivery["effective_range"] == prior["effective_range"]
        assert delivery["requested_range"] == prior["requested_range"]
        replies.append({"role": "tool", "content": text})
    note_provider_request(runtime, replies)


def _dispatch(runtime, case, index):
    row = case["receipts"][index] if index < len(case["receipts"]) else None
    path = case["calls"][index]["path"]
    runtime.current_request_id = (row["request_id"] if row is not None else
                                  f"request-{case['request']}-action-{index + 1}")
    result, observation, _source, delivery = dispatch_live_read(
        runtime.owner.root, runtime.owner.root / path, path, path, 0,
        args=case["calls"][index], measurement=runtime,
        step=row["turn"] if row is not None else 1000 + index,
        bolt=runtime.owner, policy=None, resolved_path=path)
    assert result == "OK"
    if row is not None:
        assert delivery["revision"] == row["revision"]
    assert delivery["unchanged_since_request"] is True
    assert delivery["requested_bytes_delivered"] == 0
    assert observation.startswith("[read] ")
    return runtime.owner._receipts[-1]


def _replay(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.delenv("STATEFRAME_READ_LOCKOUT", raising=False)
    monkeypatch.setenv("STATEFRAME_DELIVER_REQUESTED_BYTES", "on")
    recorded = _recorded()
    cases = recorded["cases"]
    runtime, events = _runtime(_source_root(tmp_path, cases), recorded)
    _prime_pages(runtime, recorded)
    watch = UnboundedProgressWatch()
    prior = [{"kind": "read", "result_class": "OK", "target_path": row["path"],
              "delivery": {"revision": row["revision"]}}
             for row in recorded["prior_reads"]]
    watch.observe(runtime.binding(), prior, boundary=0)
    watch.streak = 22  # Near the recorded stall boundary; no lockout may reset it.
    counters = []
    boundary = 0
    for index_case, case in enumerate(cases):
        previous = cases[index_case - 1] if index_case else None
        carry = (previous["frame"]["state"]["framefit5"]["code_bookmarks"]
                 if previous is not None else None)
        _case(runtime, case, carry=carry)
        names, wire = _menu(runtime, case)
        assert runtime._frame_budget_request_count == case["request"]
        assert "read" in names and b"[Read lockout]" not in wire
        assert runtime._code_bookmarks_active is True
        assert case["frame"]["state"]["framefit5"]["code_bookmarks"]
        runtime.native_batch = SimpleNamespace(
            origin=case["receipts"][0]["request_id"], index=0,
            size=len(case["calls"]), pending=list(case["calls"]), inflight=None)
        for index in range(len(case["calls"])):
            _dispatch(runtime, case, index)
            runtime.native_batch.index = index + 1
            runtime.native_batch.pending.pop(0)
            boundary += 1
            assert not stop_unbounded_no_progress(runtime, watch, boundary=boundary)
            if runtime.native_batch.pending:
                runtime.current_request_id = (case["receipts"][index + 1]["request_id"]
                                              if index + 1 < len(case["receipts"]) else
                                              f"request-{case['request']}-action-{index + 2}")
                assert "read" in [tool["function"]["name"] for tool in
                                  StateFrameRuntime.tool_menu(runtime, _tools(case), [])]
        counters.append((case["request"], watch.streak, True))
        runtime.native_batch = None
    return runtime, watch, events, cases, counters


def test_recorded_bm6diag_default_read_survives_request_16(tmp_path, monkeypatch):
    runtime, watch, events, cases, counters = _replay(tmp_path, monkeypatch)
    assert [row[2] for row in counters] == [True, True, True, True]
    assert [row[1] for row in counters] == [23, 24, 25, 26]
    assert watch.streak < watch.threshold
    assert runtime.system_outcome is None
    assert not [row for row in events if row["kind"] == "read_lockout_decision"]
    assert [len(case["calls"]) for case in cases] == [10, 1, 9, 10]
    assert [len(case["receipts"]) for case in cases] == [10, 1, 9, 9]


def test_default_rereads_still_stall_in_bounded_requests(tmp_path, monkeypatch):
    runtime, watch, _events, cases, _counters = _replay(tmp_path, monkeypatch)
    case = cases[-1]
    boundary = 30
    requests = 0
    while requests < 30 and runtime.system_outcome != "stalled":
        requests += 1
        request_id = f"post-lock-{requests}"
        runtime.current_request_id = request_id
        names, _ = _menu(runtime, case)
        if "read" in names:
            path = case["calls"][0]["path"]
            result, _text, _source, delivery = dispatch_live_read(
                runtime.owner.root, runtime.owner.root / path, path, path, 0,
                args=case["calls"][0], measurement=runtime, step=100 + requests,
                bolt=runtime.owner, policy=None, resolved_path=path)
            assert result == "OK" and delivery["requested_bytes_delivered"] == 0
        boundary += 1
        stop_unbounded_no_progress(runtime, watch, boundary=boundary)
    assert runtime.system_outcome == "stalled"
    assert watch.streak == watch.threshold == 32
    assert requests == 6


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from pytest import MonkeyPatch

    with TemporaryDirectory() as directory, MonkeyPatch.context() as patch:
        _runtime, _watch, _events, _cases, _counters = _replay(Path(directory), patch)
        for request, streak, read in _counters:
            print(f"request {request}: read={read} watch={streak}/32")
