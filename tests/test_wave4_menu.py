"""Ordered schema reuse preserves live menu and dispatch authority."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import kernel_agent_tools
import kernel_done_readiness_authority as done_authority
import kernel_green_lock_repair
import kernel_greenlock_menu_authority as authority
import kernel_menu_composer as composer
import kernel_post_green_menu
import kernel_tool_surface


def _tool(name, description="original"):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }}


@pytest.fixture
def permit(monkeypatch):
    settings = SimpleNamespace(reuse=False, names=set(), blocked=set())
    monkeypatch.setattr(composer, "_reuse_switch", lambda: lambda: settings.reuse)
    monkeypatch.setattr(authority, "_history_arm", lambda _runtime: False)
    monkeypatch.setattr(authority, "_relaxed", lambda: False)
    monkeypatch.setattr(kernel_post_green_menu, "capability_floor_needed", lambda _runtime: False)
    monkeypatch.setattr(done_authority, "uncertified_host_debt_enabled", lambda: False)
    host = SimpleNamespace(
        names=settings.names, custom=settings.names,
        offered=lambda name: name not in settings.blocked,
    )
    monkeypatch.setattr(kernel_agent_tools, "_active", host)
    composer._menu_digest_for_bytes.cache_clear()
    yield settings
    composer._menu_digest_for_bytes.cache_clear()


def _runtime():
    events = []
    runtime = SimpleNamespace(
        stateframe_greenlock_menu_authority_mode="on",
        stateframe_done_is_submit_mode="off",
        journal=SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields))),
    )
    lock = SimpleNamespace(
        runtime=runtime, mode="terminal", terminal_engaged=False, evidence=(),
        _coherence_enabled=lambda: False, sync=lambda: None, _green_revision="revision-1",
        _repair_blocks_done_refusal=lambda: None,
        _record_post_green_refusal=lambda _action: False,
        _workspace_path=lambda path: path,
        _emit=runtime.journal.emit,
    )
    lock._repair_enabled = lambda: bool(lock.evidence)
    lock._repair_evidence = lambda: lock.evidence
    lock._readiness_snapshot = lambda: SimpleNamespace(blockers=lock.evidence)
    runtime.green_lock = lock
    return runtime, events


def test_composer_runs_live_stages_in_order_on_every_call():
    calls = []
    initial = [_tool("read")]

    def seed(tools):
        calls.append("seed")
        return [*tools, _tool("work")]

    def project(tools):
        calls.append("project")
        return tools if calls.count("project") == 1 else tools[:1]

    stages = {"terminal_green_lock": project, "work_record_seed": seed}
    assert composer.compose_menu(initial, stages) == [*initial, _tool("work")]
    assert composer.compose_menu(initial, stages) == initial
    assert calls == ["seed", "project", "seed", "project"]


def test_schema_memo_hits_detach_values_and_require_exact_ordered_bytes(permit, monkeypatch):
    permit.reuse = True
    runtime, _events = _runtime()
    canonicalizations = []
    canonical = composer._canonical_menu_digest

    def count(payload):
        canonicalizations.append(payload)
        return canonical(payload)

    monkeypatch.setattr(composer, "_canonical_menu_digest", count)
    tools = [_tool("edit"), _tool("done")]
    assert authority.finalize_wire_menu_authority(runtime, tools) == ("edit", "done")
    first = runtime._greenlock_frozen_menu
    first["tools"][0]["function"]["parameters"]["properties"]["path"]["type"] = "number"
    assert authority.finalize_wire_menu_authority(runtime, tools) == ("edit", "done")
    second = runtime._greenlock_frozen_menu
    assert len(canonicalizations) == 1
    assert composer._menu_digest_for_bytes.cache_info().hits == 1
    assert second["tools"] == tools
    assert second["digest"] == first["digest"]

    tools[0] = _tool("edit", "modified")
    authority.finalize_wire_menu_authority(runtime, tools)
    third = runtime._greenlock_frozen_menu
    assert third["digest"] != second["digest"]
    assert second["tools"][0]["function"]["description"] == "original"

    tools.reverse()
    assert authority.finalize_wire_menu_authority(runtime, tools) == ("done", "edit")
    fourth = runtime._greenlock_frozen_menu
    assert fourth["digest"] != third["digest"]

    tools[0] = dict(reversed(tuple(tools[0].items())))
    authority.finalize_wire_menu_authority(runtime, tools)
    assert runtime._greenlock_frozen_menu["digest"] == fourth["digest"]
    assert list(runtime._greenlock_frozen_menu["tools"][0]) == list(tools[0])
    assert len(canonicalizations) == 4


def test_schema_values_that_do_not_round_trip_remain_detached(permit):
    permit.reuse = True
    runtime, _events = _runtime()
    tool = _tool("edit")
    tool["function"]["parameters"]["examples"] = ({"path": "product.py"},)
    eager = composer.freeze_menu_payload({"tools": [tool]}, reuse=False)
    assert composer.freeze_menu_payload({"tools": [tool]}, reuse=True) == eager
    authority.finalize_wire_menu_authority(runtime, [tool])
    tool["function"]["parameters"]["examples"][0]["path"] = "other.py"
    frozen = runtime._greenlock_frozen_menu["tools"][0]
    assert frozen["function"]["parameters"]["examples"] == ({"path": "product.py"},)
    assert composer._menu_digest_for_bytes.cache_info().currsize == 0


def test_index_skips_name_scans_but_rechecks_all_permissions(permit, monkeypatch):
    tools = [_tool(f"custom_{index}") for index in range(24)]
    permit.names.update(tool["function"]["name"] for tool in tools)
    runtime, events = _runtime()
    runtime.green_lock.terminal_engaged = True
    reads, checks = [], []
    name, admissible = authority._name, kernel_tool_surface.dispatch_admissible

    def count_name(tool):
        reads.append(tool)
        return name(tool)

    def count_check(action, **kwargs):
        checks.append(action)
        return admissible(action, **kwargs)

    monkeypatch.setattr(authority, "_name", count_name)
    monkeypatch.setattr(kernel_tool_surface, "dispatch_admissible", count_check)
    selected = [_tool("custom_8", "selected"), *tools[4:12]]
    off = authority.project_request_menu(runtime, selected, tools)
    eager_reads, eager_checks = len(reads), checks[:]
    reads.clear()
    checks.clear()

    permit.reuse = True
    assert authority.project_request_menu(runtime, selected, tools) == off
    assert events[0] == events[1]
    assert len(reads) < eager_reads // 2
    assert checks == eager_checks
    assert off[0]["function"]["description"] == "selected"


def test_replacement_rechecks_repair_and_custom_permissions(permit, monkeypatch):
    evidence = ({"obligation_id": "check-1", "targets": {"paths": ["product.py"]}},)
    monkeypatch.setattr(
        kernel_green_lock_repair, "readiness_repair_counterexamples",
        lambda runtime, _frame=None: runtime.green_lock.evidence,
    )

    def exercise(reuse):
        permit.reuse = reuse
        permit.names.clear()
        permit.names.add("custom_one")
        permit.blocked.clear()
        runtime, events = _runtime()
        lock = runtime.green_lock
        initial = [_tool("done"), _tool("custom_one"), _tool("edit")]
        tools = authority.project_request_menu(runtime, initial, initial)
        assert authority.finalize_wire_menu_authority(runtime, tools) == (
            "done", "custom_one", "edit",
        )
        assert authority.dispatch_action(lock, "custom_one", attempt_id="first") is None
        first = json.dumps(tools)

        # Same frozen schemas, but a new permission observation at dispatch.
        permit.blocked.add("custom_one")
        refused = authority.dispatch_action(lock, "custom_one", attempt_id="revoked")
        assert refused == authority.UPSTREAM_REFUSAL.format(action="custom_one")
        assert events[-1][1]["reason"] == "runner_capability_unavailable"

        lock.terminal_engaged = True
        lock.evidence = evidence
        lock._green_revision = "revision-2"
        permit.names.add("custom_two")
        replacement = [_tool("custom_two"), _tool("edit", "modified"), *initial[:2]]
        tools = authority.project_request_menu(runtime, replacement, replacement, boundary=True)
        assert authority.finalize_wire_menu_authority(
            runtime, tools, preserve_input_surface=True,
        ) == ("custom_two", "edit")
        assert runtime.completion_menu == runtime.offered_tools == ("custom_two", "edit")
        assert not runtime._question_done_available
        refused_done = authority.dispatch_action(lock, "done", attempt_id="replacement")
        assert refused_done == authority.UPSTREAM_REFUSAL.format(action="done")
        assert authority.dispatch_action(lock, "edit", {"path": "product.py"}) is None
        return first, json.dumps(tools), copy.deepcopy(runtime._greenlock_frozen_menu), events

    assert exercise(False) == exercise(True)


def test_final_assertion_stays_live_on_schema_memo_hit(permit, monkeypatch):
    permit.reuse = True
    runtime, _events = _runtime()
    decision = done_authority.DoneReadinessDecision(
        "revision-1", "evidence-1", "scope-1", "REFUSED", "TESTS_RED",
        ("check-1",), ("check-1",), (),
        done_authority.RecoveryAction(
            "edit", '{"function_source":"def fix(): return 1","path":"product.py"}', "check-1",
        ),
    )
    assertions = []
    assert_recovery = done_authority.assert_refusal_recovery

    def record_assertion(decision, menu, full_tools):
        assert_recovery(decision, menu, full_tools)
        assertions.append(decision.key)

    monkeypatch.setattr(done_authority, "enabled", lambda _runtime: True)
    monkeypatch.setattr(done_authority, "decision_for_runtime", lambda *_args, **_kwargs: decision)
    monkeypatch.setattr(done_authority, "assert_refusal_recovery", record_assertion)
    monkeypatch.setattr(done_authority, "_menu_projection_disabled", lambda: False)
    original = [_tool("done"), _tool("edit")]
    for _ in range(2):
        assert authority.finalize_wire_menu_authority(runtime, list(original)) == ("edit",)
    first = runtime._greenlock_frozen_menu
    assert composer._menu_digest_for_bytes.cache_info().hits == 1
    assert assertions == [decision.key, decision.key]

    decision = replace(decision, evidence_identity="evidence-2")
    assert authority.finalize_wire_menu_authority(runtime, list(original)) == ("edit",)
    updated = runtime._greenlock_frozen_menu
    assert updated["tools"] == first["tools"]
    assert updated["done_decision_key"] == list(decision.key)
    assert updated["digest"] != first["digest"]
    assert assertions[-1] == decision.key
