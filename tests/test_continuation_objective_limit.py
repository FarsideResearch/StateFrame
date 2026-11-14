"""Long task objectives stay intact while continuation state remains bounded."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
import tempfile

import pytest

import kernel_continuation_state as continuation_state
from kernel_continuation_evidence import binding_versions
from kernel_measurement import canonical_bytes
from kernel_continuation_state import ContinuationState, transition
from test_intake_orientation_fail_loud import _fixture_row

MAX_CONTINUATION_TEXT_CHARS = 4096


def _bounded_objective(task: str) -> str:
    helper = getattr(continuation_state, "bounded_objective", None)
    assert callable(helper), "continuation state needs bounded_objective(task)"
    return helper(task)


@pytest.mark.parametrize("task", [
    "Fix the pricing calculation and verify its tests.",
    "x" * MAX_CONTINUATION_TEXT_CHARS,
])
def test_bounded_objective_preserves_tasks_at_or_under_limit(task: str) -> None:
    objective = _bounded_objective(task)
    assert objective == task

    baseline_data = {
        "new_id": "root", "objective": task,
        "pending_decision": "Complete the original objective and verify its public requirements.",
        "dependencies": {}, "turns": 8, "acceptance": {}, "now_us": 0,
    }
    derived_data = {**baseline_data, "objective": objective}
    baseline_state = transition(ContinuationState(), "create", baseline_data)
    derived_state = transition(ContinuationState(), "create", derived_data)
    assert canonical_bytes(derived_data) == canonical_bytes(baseline_data)
    assert canonical_bytes(derived_state.document()) == canonical_bytes(baseline_state.document())
    assert canonical_bytes({"objective": derived_state.intentions["root"].objective}) == (
        canonical_bytes({"objective": task})
    )


def test_long_objective_is_deterministic_bounded_and_cut_at_a_safe_boundary() -> None:
    task = "Complete every required feature and verify the result. " * 220
    objective = _bounded_objective(task)

    assert objective == _bounded_objective(task)
    assert len(objective) <= MAX_CONTINUATION_TEXT_CHARS
    marker = re.search(
        r"\n\n\[Objective truncated; full_length=(\d+); sha256=([0-9a-f]{64})\]$",
        objective,
    )
    assert marker is not None
    assert int(marker.group(1)) == len(task)
    assert marker.group(2) == hashlib.sha256(task.encode("utf-8")).hexdigest()
    prefix = objective[:marker.start()]
    assert task.startswith(prefix)
    assert not prefix or task[len(prefix)].isspace()


def test_twelve_thousand_character_task_reaches_first_request_and_full_context(
        tmp_path, monkeypatch) -> None:
    from stateframe.engine.probes import react_repair
    from stateframe.scripted_endpoint import Script, serve

    tail = " FULL_TASK_TAIL"
    repeated = "Preserve every stated feature and validation requirement. "
    task = (repeated * (12000 // len(repeated) + 1))[:12000 - len(tail)] + tail
    assert len(task) == 12000
    monkeypatch.setattr(react_repair, "POC", tmp_path / "engine")
    run_dir = tmp_path / "run"
    journal_parent = run_dir / "worker" / "measurement-runs"
    journal_parent.mkdir(parents=True)
    private_temp = tmp_path / "worker-tmp"
    private_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    monkeypatch.setenv("KERNEL_CONTINUATION_EXECUTIVE", "1")
    monkeypatch.setenv("KERNEL_MODE", "render_selection:ENFORCE")
    monkeypatch.setenv("REACT_STEP_CONTRACT", "1")
    monkeypatch.setenv("STATEFRAME_PROJECTOR", "rich")
    monkeypatch.setenv("STATEFRAME_VIEW", "operational")

    runtime_module = importlib.import_module("kernel_stateframe_runtime")
    runtime_type = runtime_module.StateFrameRuntime
    original_init = runtime_type.__init__

    def scoped_init(self: object, owner: object, **kwargs: object) -> None:
        kwargs["output"] = journal_parent
        original_init(self, owner, **kwargs)

    monkeypatch.setattr(runtime_type, "__init__", scoped_init)
    monkeypatch.setattr(react_repair, "MAX_STEPS", 1)
    script = Script({"model": "scripted", "steps": [], "tokenize": {"mode": "ok"}})

    def first_call(offered: set[str]) -> dict[str, object]:
        assert "read" in offered
        return {"tool": "read", "arguments": {"path": "app/answer.py"}}

    script.next_call = first_call
    server = serve(script)
    monkeypatch.setattr(
        react_repair, "API",
        f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
    )
    monkeypatch.setattr(react_repair, "MODEL", "scripted")
    monkeypatch.setattr(react_repair, "KEY", "")
    row = _fixture_row(tmp_path, "continuation-long-objective")
    row["msg"] = task
    pilot = importlib.import_module("kernel_pilot")
    measurement = importlib.import_module("kernel_measurement")
    identity = importlib.import_module("kernel_verdict_identity")
    context_vars = (pilot.ACTIVE_PARENT, pilot.ACTIVE_RUNTIME, measurement.ACTIVE_REQUEST,
                    identity._ACTIVE_PROVIDER)
    for context_var in context_vars:
        context_var.set(None)
    try:
        react_repair.work(row)
    finally:
        server.shutdown()
        server.server_close()
        for context_var in context_vars:
            context_var.set(None)

    from _journal import measurement_events
    events = measurement_events(run_dir)
    root_create = next(
        event for event in events
        if event.get("kind") == "continuation_transition" and event.get("operation") == "create"
    )
    from _journal import measurement_journal_paths
    from kernel_measurement import resolve_journal_blob_refs
    journal_path = measurement_journal_paths(run_dir)[0]
    root_create = resolve_journal_blob_refs(root_create, journal_path)
    stored_objective = root_create["data"]["objective"]
    assert stored_objective == _bounded_objective(task)
    assert len(stored_objective) <= MAX_CONTINUATION_TEXT_CHARS

    assert len(script.requests) == 1
    body = script.requests[0]
    user_messages = [message for message in body["messages"] if message.get("role") == "user"]
    assert any(task in str(message.get("content", "")) for message in user_messages)
    continuation = next(
        message["content"] for message in user_messages
        if str(message.get("content", "")).startswith("[Continuation]\n")
    )
    state = json.loads(continuation.split("\n", 1)[1])
    assert state["active"]["objective"] == "Use the complete task in the original user request."
    assert state["active"]["objective_is_reference"] is True
    assert "complete task in the original user request" in state["instructions"]


def test_long_dependency_path_uses_bounded_digest_key() -> None:
    path = "nested/" + ("segment/" * 140) + "module.py"
    path = path[:1015]
    binding = {"product_files": {path: "sha256:" + "a" * 64}, "harness_files": {}}

    dependencies = binding_versions(binding)

    assert "product_files:*" in dependencies
    assert all(len(key) <= 1024 for key in dependencies)
    changed = {"product_files": {path: "sha256:" + "b" * 64}, "harness_files": {}}
    assert binding_versions(changed)["product_files:*"] != dependencies["product_files:*"]
