"""A queued native sibling is an action boundary, not a model request."""

from __future__ import annotations

import json
import sys
import tempfile
from types import SimpleNamespace

import pytest

from kernel_measurement import ACTIVE_REQUEST, RequestCapture
from kernel_native_tools import NativeToolBatch
from kernel_pilot import ACTIVE_RUNTIME
from kernel_stateframe_runtime import PreparedRequest, StateFrameRuntime


class Journal:
    task_id = "three-call-native-batch"

    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, kind: str, **fields: object) -> dict:
        event = {"kind": kind, "event_id": f"event-{len(self.events) + 1}", **fields}
        self.events.append(event)
        return event


def test_three_call_batch_prepares_only_the_sent_request():
    journal = Journal()
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.journal = journal
    runtime.owner = SimpleNamespace(current_step=1, current_request_id=None)
    runtime.snapshot_id = "snapshot"
    runtime.endpoint_profile = None
    runtime.prepared_binding = {"product_revision": "revision"}
    runtime.binding = lambda: {"product_revision": "revision"}
    runtime._active_capture = None
    runtime.stateframe_f4_mode = "off"
    runtime.stateframe_hybrid_mode = "off"
    runtime.self_plan_state = None
    runtime.unified_lifecycle = None
    runtime.projector_kind = "legacy"
    runtime.continuation = None
    runtime.latest_exchange_mode = "off"
    runtime.step_ledger = None
    runtime.memory_isolation = None
    runtime._last_capture = None
    runtime._deliver_auto_verify = lambda *_args: None
    runtime._record_passive_retention_capture = lambda *_args: None
    runtime._prepare_continuation_boundary = lambda *_args: None
    runtime._frame = lambda: ({}, "", "")
    runtime._sync_work_carry = lambda *_args: None
    runtime._configure_probe_gate = lambda *_args: None
    runtime._bind_long_horizon_capture = lambda *_args, **_kwargs: None

    sent = PreparedRequest(
        ({"role": "user", "content": "task"},),
        RequestCapture(journal, 1, "snapshot"),
        tools=({"type": "function", "function": {"name": "read"}},),
    )
    wire = json.dumps({"messages": list(sent.messages), "tools": list(sent.tools)}).encode()
    assert wire == (b'{"messages": [{"role": "user", "content": "task"}], '
                    b'"tools": [{"type": "function", "function": {"name": "read"}}]}')
    journal.emit("request_prepared", request_id=sent.capture.request_id)
    batch = NativeToolBatch()
    runtime.native_batch = batch
    runtime._active_capture = sent.capture
    sent.capture.response_event_id = "response-1"
    calls = [{"id": f"call-{index}", "type": "function",
              "function": {"name": "read", "arguments": "{}"}}
             for index in range(3)]
    runtime_token = ACTIVE_RUNTIME.set(runtime)
    try:
        for index in range(3):
            prepared = sent if index == 0 else runtime.prepare_queued_native_action(
                [], uncapped=[], tools=sent.tools)
            capture_token = ACTIVE_REQUEST.set(prepared.capture)
            try:
                content, selected = batch.next(
                    lambda *_args, **_kwargs: ("", calls), list(prepared.messages),
                    tools=list(sent.tools))
            finally:
                ACTIVE_REQUEST.reset(capture_token)
            assert content == ""
            assert len(selected) == 1
            assert selected[0]["id"] == f"native_{sent.capture.request_id}_{index}"
            batch.action_closed({"step_record_id": f"step-{index}", "result": "OK",
                                 "after_binding": runtime.binding()})
            runtime.owner.current_step += 1
    finally:
        ACTIVE_RUNTIME.reset(runtime_token)
    assert len([event for event in journal.events
                if event["kind"] == "request_prepared"]) == 1
    assert len([event for event in journal.events
                if event["kind"] == "native_batch_action_selected"]) == 3


class StopAfterSecondSend(BaseException):
    """Leave the real worker after observing the next model request."""


def test_real_worker_three_call_batch_has_one_prepare_per_send(tmp_path, monkeypatch):
    from stateframe.engine.probes import react_repair
    from stateframe.scripted_endpoint import Script, serve
    from test_intake_orientation_fail_loud import _fixture_row
    from _journal import measurement_events

    journal_parent = tmp_path / "run" / "worker" / "measurement-runs"
    journal_parent.mkdir(parents=True)
    private_temp = tmp_path / "worker-tmp"
    private_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setenv("REACT_NATIVE_TOOL_BATCHES", "1")
    monkeypatch.setenv("STATEFRAME_VIEW", "operational")
    monkeypatch.setenv("STATEFRAME_PROJECTOR", "rich")
    monkeypatch.setenv("STATEFRAME_IDEALPATH", "off")
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    original_init = StateFrameRuntime.__init__

    def scoped_init(self, owner, **kwargs):
        kwargs["output"] = journal_parent
        original_init(self, owner, **kwargs)

    monkeypatch.setattr(StateFrameRuntime, "__init__", scoped_init)
    monkeypatch.setattr(react_repair, "MAX_STEPS", 5)
    script = Script({"model": "scripted", "steps": [], "tokenize": {"mode": "ok"}})
    def next_call(offered):
        for name, arguments in (
            ("list_files", {}), ("read", {"path": "app/answer.py"}),
            ("grep", {"pattern": "answer"}), ("shell", {"command": "pwd"}),
        ):
            if name in offered:
                return {"tool": name, "arguments": arguments}
        assert False, f"no safe read tool offered: {offered}"
    script.next_call = next_call
    server = serve(script)
    monkeypatch.setattr(react_repair, "API", f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
    monkeypatch.setattr(react_repair, "MODEL", "scripted")
    monkeypatch.setattr(react_repair, "KEY", "")
    original_ask = react_repair.ask_tools
    sends = 0

    def ask_with_batch(*args, **kwargs):
        nonlocal sends
        content, calls = original_ask(*args, **kwargs)
        sends += 1
        if sends == 2:
            raise StopAfterSecondSend
        assert len(calls) == 1
        assert calls[0]["function"]["name"] != "done", script.requests[0].get("tools")
        return content, [{**calls[0], "id": f"batch-{index}"} for index in range(3)]

    monkeypatch.setattr(react_repair, "ask_tools", ask_with_batch)
    from kernel_verdict_identity import _ACTIVE_PROVIDER
    from kernel_pilot import ACTIVE_PARENT
    for context in (ACTIVE_PARENT, ACTIVE_RUNTIME, ACTIVE_REQUEST, _ACTIVE_PROVIDER):
        context.set(None)
    try:
        with pytest.raises(StopAfterSecondSend):
            react_repair.work(_fixture_row(tmp_path, "queued-native-s1"))
    finally:
        server.shutdown()
        server.server_close()
        for context in (ACTIVE_PARENT, ACTIVE_RUNTIME, ACTIVE_REQUEST, _ACTIVE_PROVIDER):
            context.set(None)
    assert sends == 2
    assert len(script.requests) == 2
    import os
    if os.environ.get("S1_DUMP_WIRE"):
        with open(os.environ["S1_DUMP_WIRE"], "w") as handle:
            json.dump(script.requests, handle, sort_keys=True)
    events = measurement_events(tmp_path / "run")
    assert len([event for event in events if event["kind"] == "request_prepared"]) == 2
    assert len([event for event in events if event["kind"] == "native_batch_action_selected"]) == 3


def test_queued_action_consumes_pending_advisories_like_a_full_prepare():
    """A card a discarded body used to swallow must not reappear on the next send."""
    from kernel_terminal_policy import _state

    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.native_batch = NativeToolBatch()
    runtime.native_batch.pending = [{"id": "c"}]
    runtime.memory_isolation = None
    state = _state(runtime)
    state["pending"] = [{"message": "advisory"}]
    runtime._active_capture = None
    runtime._record_passive_retention_capture = lambda *_a: None
    runtime._deliver_auto_verify = lambda *_a: None
    runtime.close_step = lambda *_a: None
    runtime.projector_kind = "legacy"
    runtime.continuation = None
    runtime._prepare_continuation_boundary = lambda *_a: "stop"
    runtime.self_plan_state = None
    runtime.unified_lifecycle = None
    assert runtime.prepare_queued_native_action([], uncapped=[], tools=()) == "stop"
    assert state["pending"] == []
