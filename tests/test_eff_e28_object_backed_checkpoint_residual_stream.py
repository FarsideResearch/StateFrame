import builtins
import contextvars
import hashlib
import io
import json
import os
import random
import shutil
import sys

import kernel_workspace_objects as objects
import kernel_measurement as measurement
from probes import react_repair
from test_workspace_record_migration import _bind, checkpoint, run  # noqa: F401


def test_object_checkpoint_reuses_leaf_after_one_file_edit(run, monkeypatch):
    work = run / "workspace"
    work.mkdir()
    leaf = random.Random(42).randbytes(256 * 1024)
    (work / "stable.bin").write_bytes(leaf)
    target = work / "changed.txt"
    target.write_bytes(b"before")
    consumers = _bind(run, work, monkeypatch)
    pending_wire = b'{"model":"offline","messages":[]}'
    shared = {"request": pending_wire, "history": leaf}
    frame = {"pending": shared, "alias": shared}
    state = {"frame": frame, "modules": {}, "classes": {}, "patched": {}, "participants": {},
             "random": random.getstate(), "context": []}
    context = {key: {} for key in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    monkeypatch.setattr(checkpoint, "collect", lambda unused: (state, context))
    staged = {}
    stage = objects._stage_object

    def count(directory, ref, source):
        staged[ref.digest] = ref.size
        return stage(directory, ref, source)

    monkeypatch.setattr(objects, "_stage_object", count)
    checkpoint.save(sys._getframe(), 1, run)
    cold = sum(staged.values())
    digest = hashlib.sha256(leaf).hexdigest()
    assert staged[digest] == len(leaf)
    staged.clear()
    with consumers.state.begin_write(("changed.txt",)):
        target.write_bytes(b"after")
    saved = checkpoint.save(sys._getframe(), 2, run)
    assert digest not in staged
    assert 0 < sum(staged.values()) < 0.7 * cold

    pending = alias = None
    monkeypatch.setattr(checkpoint, "install_continuous_clock", lambda _: None)
    target.write_bytes(b"later")
    contextvars.Context().run(checkpoint.restore, sys._getframe(), saved, run)
    assert target.read_bytes() == b"after" and (work / "stable.bin").read_bytes() == leaf
    assert pending is alias
    assert pending["request"] == pending_wire


class _WriteCounter:
    def __init__(self, stream, total):
        self._stream = stream
        self._total = total

    def write(self, value):
        written = self._stream.write(value)
        if isinstance(value, str):
            encoding = getattr(self._stream, "encoding", None) or "utf-8"
            errors = getattr(self._stream, "errors", None) or "strict"
            size = len(value[:written].encode(encoding, errors))
        else:
            size = len(value[:written])
        self._total[0] += size
        return written

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args):
        return self._stream.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install_write_counter(patch):
    total = [0]

    def wrap_open(factory):
        def counted(*args, **kwargs):
            mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
            stream = factory(*args, **kwargs)
            if any(flag in mode for flag in "wax+"):
                return _WriteCounter(stream, total)
            return stream
        return counted

    real_write = os.write

    def counted_write(fd, data):
        written = real_write(fd, data)
        total[0] += written
        return written

    real_writev = getattr(os, "writev", None)
    if real_writev is not None:
        def counted_writev(fd, buffers):
            written = real_writev(fd, buffers)
            total[0] += written
            return written
        patch.setattr(os, "writev", counted_writev)
    patch.setattr(os, "write", counted_write)
    patch.setattr(os, "fdopen", wrap_open(os.fdopen))
    patch.setattr(io, "open", wrap_open(io.open))
    patch.setattr(builtins, "open", wrap_open(builtins.open))
    patch.setattr(shutil, "_USE_CP_SENDFILE", False)
    return total


def _next_wire(frame, workspace, output, monkeypatch):
    import kernel_bounded_http
    from detreason.world_knowledge import WorldKnowledgeStore
    from kernel_bolt import KernelBolt
    from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
    from kernel_read_receipts import enabled as working_set_allocator_enabled
    from kernel_stateframe_runtime import StateFrameRuntime

    assert working_set_allocator_enabled(), "request comparison must use the shipped working-set allocator"
    parent_token = ACTIVE_PARENT.set(None)
    inactive_runtime_token = ACTIVE_RUNTIME.set(None)
    try:
        effect = frame["pending_effect"]
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Task: Continue from the restored request state."},
            {"role": "user", "content": (
                effect["result"] + ":" + hashlib.sha256(effect["history"]).hexdigest())},
        ]
        owner = KernelBolt(workspace, "", "checkpoint-wire")
        owner.world = WorldKnowledgeStore(sandbox_id="checkpoint-wire")
        owner.task_row = {"msg": "Continue from the restored request state.",
                          "files": {"app.py": "VALUE = 1\n"},
                          "entry_path": "app.py", "tests": []}
        owner.current_step = 3
        runtime = StateFrameRuntime(
            owner, output=output, target=workspace / "app.py", model="offline",
            tools=frame["tools"], temperature=0,
            source_resolver=lambda _path, _symbol: None,
        )
        prepared = runtime.prepare(messages, uncapped=messages, tools=frame["tools"])
        assert prepared.overflow is None
        request_tools = react_repair._prepared_tool_contract(prepared, frame["tools"])
        assert request_tools == frame["tools"]
        request_messages = list(prepared.messages)
        from kernel_request_budget import DEFAULT_MAX_REQUEST_BYTES
        from kernel_request_source_window import bound_runtime_sources
        bound_runtime_sources(
            request_messages, runtime,
            max_request_bytes=DEFAULT_MAX_REQUEST_BYTES, tools=request_tools,
        )
        assert isinstance(getattr(runtime, "_request_source_closure", None), dict)
        from kernel_completion_owner import establish_request_action_contract
        prepared.capture.request_action_contract = establish_request_action_contract(
            runtime, request_messages, request_tools,
            request_id=prepared.capture.request_id,
        )
        from kernel_request_kind import require_adapter

        def gate(record, capture, active_runtime):
            record.validate(capture, active_runtime)

        def host_preparer(request_messages, **_kwargs):
            return request_messages, {"test_identity_composer": True}

        assert require_adapter(
            prepared.capture, runtime, request_messages, gate=gate,
            host_preparer=host_preparer, adapter="checkpoint-test",
        )

        posts = []

        def capture_transport(wire, *_args, **_kwargs):
            posts.append(wire)
            return b'{"choices":[{"message":{"content":"ack"},"finish_reason":"stop"}]}'

        monkeypatch.setattr(kernel_bounded_http, "request_bytes", capture_transport)
        runtime_token = ACTIVE_RUNTIME.set(runtime)
        request_token = runtime.activate(prepared)
        try:
            assert react_repair.ask_tools(
                request_messages, temp=1.0, tools=request_tools) == ("ack", [])
        finally:
            measurement.ACTIVE_REQUEST.reset(request_token)
            ACTIVE_RUNTIME.reset(runtime_token)
    finally:
        ACTIVE_RUNTIME.reset(inactive_runtime_token)
        ACTIVE_PARENT.reset(parent_token)
    assert len(posts) == 1
    capture = runtime._active_capture
    qualified = getattr(capture, "qualified_request", None)
    assert qualified is not None, "FrameFit5 transport bypassed qualified-request sealing"
    assert qualified.wire == posts[0]
    certificate = getattr(capture, "_working_set_certificate", None)
    assert certificate is not None, "shipped allocator request must carry a working-set certificate"
    from kernel_working_set_proof import verify_certificate
    assert verify_certificate(posts[0], certificate)
    body = json.loads(posts[0])
    from kernel_stateframe_framefit5 import has_framefit5_state
    assert has_framefit5_state(body["messages"])
    assert body["tools"] == request_tools
    assert body["max_tokens"] == 4000
    return posts[0]


def _checkpoint_payload(leaf):
    shared = {"request": b'{"model":"offline","messages":[]}', "history": leaf}
    effect = {"result": "edit settled", "history": leaf}
    restored_tool = {
        "type": "function",
        "function": {
            "name": "restored_probe",
            "description": "A named tool restored from the checkpoint.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    frame = {
        "pending": shared,
        "pending_alias": shared,
        "pending_effect": effect,
        "effect_alias": effect,
        "tools": [restored_tool],
    }
    state = {"frame": frame, "modules": {}, "classes": {}, "patched": {},
             "participants": {}, "random": random.getstate(), "context": []}
    context = {key: {} for key in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    return state, context


def _restore_and_measure_next_wire(saved, root, workspace, output, monkeypatch):
    pending = pending_alias = pending_effect = effect_alias = tools = None
    monkeypatch.setattr(checkpoint, "install_continuous_clock", lambda _saved: None)
    # This synthetic payload has no saved ContextVars; exclude pytest's active ones.
    contextvars.Context().run(checkpoint.restore, sys._getframe(), saved, root)
    assert pending is pending_alias
    assert pending_effect is effect_alias
    assert pending["request"] == b'{"model":"offline","messages":[]}'
    return _next_wire({"pending_effect": pending_effect, "tools": tools},
                      workspace, output, monkeypatch)


def test_object_checkpoint_total_writes_and_restored_next_wire_match_legacy(run, tmp_path, monkeypatch):
    from kernel_read_receipts import enabled as working_set_allocator_enabled

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_WORKING_SET_STABLE_FIRST", "on")
    assert working_set_allocator_enabled(), "comparison must exercise the shipped working-set allocator"
    # Match the shipped profile's allocator-certified F4/FrameFit5 request path.
    monkeypatch.setenv("STATEFRAME_F4", "on")
    monkeypatch.setenv("STATEFRAME_FRAMEFIT", "on")
    monkeypatch.setenv("STATEFRAME_F4_FIT", "off")
    legacy = run.parent / "legacy-run"
    legacy.mkdir(mode=0o700)
    wire_workspace = tmp_path / "wire-workspace"
    wire_workspace.mkdir()
    (wire_workspace / "app.py").write_text("VALUE = 1\n")
    leaf = random.Random(73).randbytes(512 * 1024)
    expected_state, _ = _checkpoint_payload(leaf)
    expected_wire = _next_wire(
        expected_state["frame"], wire_workspace, tmp_path / "expected-wire", monkeypatch)
    expected_body = json.loads(expected_wire)
    assert expected_body["tools"] == expected_state["frame"]["tools"]
    assert expected_body["max_tokens"] == 4000

    # The control and object path receive identical state and workspace contents.
    legacy_work = legacy / "workspace"
    legacy_work.mkdir()
    (legacy_work / "stable.bin").write_bytes(leaf)
    object_work = run / "workspace"
    object_work.mkdir()
    (object_work / "stable.bin").write_bytes(leaf)
    consumers = _bind(run, object_work, monkeypatch)
    assert consumers is not None

    totals = {}
    saved_points = {}
    for label, run_dir, mode in (("legacy", legacy, "off"), ("objects", run, "on")):
        monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", mode)
        state, ctx = _checkpoint_payload(leaf)
        monkeypatch.setattr(checkpoint, "collect", lambda _frame, state=state, ctx=ctx: (state, ctx))
        with monkeypatch.context() as patch:
            counter = _install_write_counter(patch)
            checkpoint.save(sys._getframe(), 1, run_dir)
            saved = checkpoint.save(sys._getframe(), 2, run_dir)
        totals[label] = counter[0]
        saved_points[label] = (run_dir, saved)

    assert totals["objects"] <= 0.7 * totals["legacy"]

    wires = {}
    for label, (root, saved) in saved_points.items():
        monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on" if label == "objects" else "off")
        wires[label] = _restore_and_measure_next_wire(
            saved, root, wire_workspace, tmp_path / f"{label}-wire", monkeypatch)
    assert wires["legacy"] == expected_wire
    assert wires["objects"] == expected_wire
