"""Parity and freshness at the terminal projection and request boundaries."""

import copy
import json
from types import SimpleNamespace

import kernel_completion_owner as owner
import kernel_done_readiness_authority as done_authority
import kernel_green_coherence as coherence
import kernel_greenlock_menu_authority as green_menu
import kernel_readiness_authority as readiness
import kernel_wire_truth_gate as wire_gate
import pytest
from kernel_stateframe_runtime import StateFrameRuntime


def _tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Original description",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _receipt(binding, exit_code):
    return {
        "receipt_id": "check-1",
        "producer": "test_subprocess",
        "selectors": ["check"],
        "exit_code": exit_code,
        "binding": dict(binding),
        "after_binding": dict(binding),
        "binding_stable": True,
    }


def _runtime():
    events = []
    runtime = SimpleNamespace(
        journal=SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields))),
        stateframe_readiness_authority_mode="on",
        stateframe_readiness_single_authority_mode="off",
        current_request_id="request-1",
        current_binding={
            "product_revision": "revision-1",
            "harness_revision": "harness-1",
            "environment_revision": "environment-1",
            "check_recipe_digest": "recipe-1",
        },
        test_receipts=[],
    )
    runtime.binding = lambda: runtime.current_binding
    runtime.required_public_selectors = lambda: ["check"]
    runtime.readiness_decision = StateFrameRuntime.readiness_decision.__get__(runtime)
    return runtime, events


@pytest.mark.parametrize("exit_code, check_status", [(0, "PASS"), (1, "FAIL"), (None, "BLOCKED")])
def test_readiness_transition_keeps_frame_menu_and_journal_parity(
        monkeypatch, exit_code, check_status):
    full_tools = [_tool(name) for name in ("done", "read", "edit", "run_tests")]

    def run(reuse):
        monkeypatch.setattr(wire_gate, "assembly_reuse_enabled", lambda: reuse)
        runtime, events = _runtime()
        runtime.required_public_selectors = lambda: []
        frame = {"verification": {}, "resume": {"recommended_next_action": "read"}}

        def observe():
            public = coherence.project_frame(frame, runtime=runtime)
            decision = runtime._readiness_decision
            menu, changes = readiness.project_menu(
                decision, full_tools, full_tools, capability_floor_enabled=False)
            return public, decision.to_dict(), decision.refusal_text(), menu, changes

        initial = observe()
        assert initial[1]["done_allowed"]
        assert initial[1]["public_check_status"] == "NO_HOST_CHECKS"
        runtime.required_public_selectors = lambda: ["check"]
        missing = observe()
        assert missing[1]["public_check_status"] == "BLOCKED"
        assert not missing[1]["done_allowed"]
        assert missing[0]["terminal_status"]["receipt_revision"] is None
        runtime.test_receipts[:] = [_receipt(runtime.binding(), exit_code)]
        current = observe()
        assert current[1]["state_version"] == (2 if exit_code is None else 3)
        assert current[1]["public_check_status"] == check_status
        assert current[1]["done_allowed"] == (exit_code == 0)
        assert current[1]["dispositions"]["edit"] == ("deny" if exit_code == 0 else "allow")
        assert current[0]["terminal_status"]["receipt_revision"] == "revision-1"
        assert observe() == current
        runtime.current_binding["check_recipe_digest"] = "recipe-2"
        stale = observe()
        assert stale[1]["public_check_status"] == "BLOCKED"
        assert not stale[1]["done_allowed"]
        assert [kind for kind, _ in events] == [readiness.DECISION_EVENT] * (2 if exit_code is None else 4)
        return initial, missing, current, stale, events

    assert run(True) == run(False)


def test_frame_only_observation_is_reused_and_dropped_at_return(monkeypatch):
    binding = {"product_revision": "revision-1"}
    frame = {
        "binding": binding,
        "required_public_checks": ["check"],
        "test_receipts": [_receipt(binding, 0)],
    }
    lock = object()
    original = coherence.terminal_status
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(coherence, "terminal_status", count)
    monkeypatch.setattr(wire_gate, "assembly_reuse_enabled", lambda: False)
    eager = coherence.project_frame(frame, lock=lock)
    assert len(calls) == 2
    monkeypatch.setattr(wire_gate, "assembly_reuse_enabled", lambda: True)
    calls.clear()
    assert coherence.project_frame(frame, lock=lock) == eager
    assert len(calls) == 1
    binding["product_revision"] = "revision-2"
    changed = coherence.project_frame(frame, lock=lock)
    assert changed["public_check_status"] == "BLOCKED"
    assert changed["terminal_status"]["current_revision"] == "revision-2"
    changed["binding"]["product_revision"] = "caller-owned"
    assert frame["binding"]["product_revision"] == "revision-2"


def test_frame_projection_detaches_before_calling_live_authority(monkeypatch):
    monkeypatch.setattr(wire_gate, "assembly_reuse_enabled", lambda: True)
    runtime, _ = _runtime()
    frame = {"metadata": {"value": "before live read"}}
    revisions = []

    def binding():
        revisions.append(runtime.current_binding["product_revision"])
        frame["metadata"]["value"] = "after live read"
        return runtime.current_binding

    runtime.binding = binding
    assert coherence.project_frame(frame, runtime=runtime)["metadata"] == {
        "value": "before live read",
    }
    assert len(revisions) > 1
    assert frame["metadata"]["value"] == "after live read"


def test_wire_schema_hashes_the_parsed_snapshot_once_per_verdict(monkeypatch):
    tools = [_tool("read"), _tool("edit")]
    wire = json.dumps({"messages": [], "tools": tools}).encode()
    authority_schemas = copy.deepcopy(tools)
    context = {
        "schema_authorities": [
            {"name": "typed", "tools": authority_schemas},
            {"name": "owner", "schema_digest": wire_gate.schema_digest(tools)},
        ],
    }
    digest = wire_gate.schema_digest
    calls = []

    def count(value):
        calls.append(value)
        return digest(value)

    monkeypatch.setattr(wire_gate, "schema_digest", count)
    monkeypatch.setattr(wire_gate, "assembly_reuse_enabled", lambda: False)
    eager = wire_gate.evaluate_wire(wire, context)
    assert eager.verdict == "PASS"
    assert len(calls) == 4

    monkeypatch.setattr(wire_gate, "assembly_reuse_enabled", lambda: True)
    calls.clear()
    assert wire_gate.evaluate_wire(wire, context) == eager
    assert len(calls) == 2
    authority_schemas[0]["function"]["description"] = "Modified description"
    assert wire_gate.evaluate_wire(wire, context).failed == ("schema_authority_mismatch:typed",)
    reordered = json.dumps({"messages": [], "tools": tools[::-1]}).encode()
    assert wire_gate.evaluate_wire(reordered, context).failed == (
        "schema_authority_mismatch:typed", "schema_authority_mismatch:owner")


def test_contract_subproof_preserves_snapshots_live_bindings_and_error_order(monkeypatch):
    original_digest = owner._digest

    def run(reuse):
        monkeypatch.setattr(wire_gate, "assembly_reuse_enabled", lambda: reuse)
        runtime, events = _runtime()
        messages = [{"role": "user", "content": "Read the file."}]
        tools = [_tool("read"), _tool("edit")]
        schema_hashes = []

        def count(value):
            if value == tools:
                schema_hashes.append(1)
            return original_digest(value)

        monkeypatch.setattr(owner, "_digest", count)
        contract = owner.establish_request_action_contract(
            runtime, messages, tools, request_id="request-1")
        sealed = owner.assert_request_action_contract(
            runtime, messages, tools, request_id="request-1", boundary="prepared",
            frozen_contract=owner._freeze_contract(contract), read_only=True)
        assert contract == sealed
        assert len(schema_hashes) == (2 if reuse else 4)
        assert owner.request_menu_action_admissible(runtime, "read", "request-1")
        assert owner.request_menu_action_admissible(runtime, "edit", "request-1")
        assert not owner.request_menu_action_admissible(runtime, "done", "request-1")
        runtime._completion_owner_full_tools = [*tools, _tool("done")]
        owner_notice = [{
            "role": "user",
            "content": "Use done when ready.",
            owner.OWNER_MESSAGE_BLOCK_KEY: "next-action:notice",
        }]
        with pytest.raises(owner.RequestActionContractError, match="off-menu rewrite"):
            owner.assert_request_action_contract(
                runtime, owner_notice, tools, request_id="request-1", read_only=True)
        assert owner_notice[0]["content"] == "Use done when ready."
        owner.assert_request_action_contract(
            runtime, owner_notice, tools, request_id="request-1", boundary="rewrite")
        assert owner_notice[0]["content"] == "Use when ready."
        sealed["admitted_tool_schemas"][0]["function"]["description"] = "Changed by caller"
        changed_tools = copy.deepcopy(tools)
        changed_tools[0]["function"]["description"] = "Modified description"
        with pytest.raises(owner.RequestActionContractError, match="differ from serialized"):
            owner.assert_request_action_contract(
                runtime, messages, changed_tools, request_id="request-1", boundary="final")
        runtime.current_binding["product_revision"] = "revision-2"
        with pytest.raises(owner.RequestActionContractError, match="revision is stale"):
            owner.assert_request_action_contract(
                runtime, messages, tools, request_id="request-1", boundary="final")
        replacement = owner.establish_request_action_contract(
            runtime, messages, changed_tools, request_id="request-2")
        assert replacement["revision"] == "revision-2"
        assert replacement["tool_schema_digest"] != contract["tool_schema_digest"]
        assert messages == [{"role": "user", "content": "Read the file."}]
        return contract, replacement, events

    assert run(True) == run(False)


def test_pass_gate_change_after_contract_assertion_is_seen_by_done(monkeypatch):
    monkeypatch.setenv(done_authority.UNCERTIFIED_HOST_DEBT_ENV, "off")

    def run(reuse):
        monkeypatch.setattr(wire_gate, "assembly_reuse_enabled", lambda: reuse)
        runtime, events = _runtime()
        runtime.stateframe_readiness_single_authority_mode = "on"
        tools = [_tool(name) for name in ("done", "edit", "run_tests")]
        runtime._completion_owner_full_tools = tools
        runtime.test_receipts[:] = [_receipt(runtime.binding(), 0)]
        gate = SimpleNamespace(
            artifact_identity="revision-1", evidence_identity="passing", blockers=())
        runtime.pass_gates_vector = lambda *_args: gate
        initial = runtime.readiness_decision()
        assert initial.done_authority.decision == "ACCEPTED_CERTIFIED"
        messages = [{"role": "user", "content": "Task."}]
        contract = owner.establish_request_action_contract(
            runtime, messages, tools, request_id="request-1")
        owner.assert_request_action_contract(
            runtime, messages, tools, request_id="request-1",
            frozen_contract=owner._freeze_contract(contract), read_only=True)
        runtime._greenlock_frozen_menu = {
            "tools": tools, "allow_set": contract["advertised_actions"],
        }
        runtime.stateframe_done_is_submit_mode = "off"
        lock = SimpleNamespace(
            runtime=runtime, mode="terminal", terminal_engaged=False,
            sync=lambda: None, _coherence_enabled=lambda: False,
            _repair_blocks_done_refusal=lambda: None,
            _repair_enabled=lambda: False, _green_revision="revision-1",
            _readiness_snapshot=lambda: initial, _workspace_path=str,
            _emit=lambda kind, **fields: events.append((kind, fields)),
        )
        assert green_menu.dispatch_action(lock, "done", attempt_id="done-before") is None

        gate.evidence_identity = "blocked"
        gate.blockers = (SimpleNamespace(
            blocker_id="live-pass-gate",
            resolver={"tool": "edit", "arguments": {
                "path": "src/example.py", "function_source": "def example():\n    return 1\n",
            }},
        ),)
        refusal = green_menu.dispatch_action(lock, "done", attempt_id="done-1")
        current = done_authority.cached_decision(runtime)
        assert current.decision == "REFUSED"
        assert current.blocker_ids == ("live-pass-gate",)
        assert current.recovery.tool == "edit"
        assert refusal == done_authority.render_done_reason(current)
        assert green_menu.dispatch_action(lock, "edit", {"path": "src/example.py"},
                                          attempt_id="edit-1") is None
        return current.to_dict(), refusal, events

    assert run(True) == run(False)
