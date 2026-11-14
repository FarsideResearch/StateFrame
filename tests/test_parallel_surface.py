"""The parent subframe action survives shipped-profile request authoring."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _prepared_parent(tmp_path: Path, monkeypatch, *, mode: str):
    from detreason.world_knowledge import WorldKnowledgeStore
    from kernel_bolt import KernelBolt
    import kernel_parallel
    from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
    from kernel_stateframe_runtime import StateFrameRuntime
    from probes.react_repair import _tools_for_probe_modes

    source = tmp_path / "shop/pricing.py"
    source.parent.mkdir()
    source.write_text("def price(value):\n    return value\n")
    run = tmp_path / "run"
    (run / "parent").mkdir(parents=True)
    monkeypatch.setattr(kernel_parallel, "_service", None)
    if mode != "off":
        kernel_parallel.install(run, {
            "parallel_mode": mode, "parent_snapshot": str(run / "parent"),
        })

    owner = KernelBolt(tmp_path, "", "parent-surface")
    owner.world = WorldKnowledgeStore(sandbox_id="parent-surface")
    owner.task_row = {"msg": "Fix pricing.", "files": {"shop/pricing.py": source.read_text()},
                      "entry_path": "shop/pricing.py", "tests": []}
    owner.current_step = 1
    tools = _tools_for_probe_modes(probe_gate_on=False, observation_probe_on=False)
    parent_token = ACTIVE_PARENT.set(None)
    runtime_token = ACTIVE_RUNTIME.set(None)
    try:
        runtime = StateFrameRuntime(
            owner, output=tmp_path / "measurement", target=source,
            model="scripted", tools=tools, temperature=0,
            source_resolver=lambda _path, _symbol: None,
            multi_file=True,
            edit_contract={"edit_authority": {"editable_paths": ["shop/pricing.py"]}},
        )
        assert runtime.continuation is not None
        messages = [{"role": "system", "content": "rules"},
                    {"role": "user", "content": "Task: Fix pricing."}]
        prepared = []
        for step in (1, 2):
            owner.current_step = step
            request = runtime.prepare(messages, uncapped=messages, tools=tools)
            assert request.overflow is None
            prepared.append(json.loads(json.dumps({"messages": request.messages,
                                                   "tools": request.tools})))
    finally:
        ACTIVE_RUNTIME.reset(runtime_token)
        ACTIVE_PARENT.reset(parent_token)
    return prepared, runtime.continuation


@pytest.mark.parametrize("mode,expected", [("2", True), ("off", False)])
def test_shipped_f4_parent_wire_offers_parallel_on_every_editable_request(
        tmp_path, monkeypatch, shipped_runtime_profile, mode, expected):
    requests, _continuation = _prepared_parent(tmp_path, monkeypatch, mode=mode)
    assert len(requests) == 2
    for request in requests:
        names = {tool["function"]["name"] for tool in request["tools"]}
        assert ("parallel" in names) is expected
        assert any(message.get("content", "").startswith("[StateFrame]\n")
                   for message in request["messages"])


def test_parallel_receipt_contract_and_runner_capability_are_conditional(
        tmp_path, monkeypatch):
    from kernel_continuation_evidence import (
        has_execution_receipt_contract, result_class_disposition,
    )
    from kernel_tool_surface import dispatch_admissible
    import kernel_parallel

    assert has_execution_receipt_contract("parallel")
    assert result_class_disposition("parallel", "APPLIED") == "settled_applied"
    assert result_class_disposition("parallel", "REFUSED") == "settled_not_applied"
    assert result_class_disposition("parallel", "planted_unknown") == "unknown"
    monkeypatch.setattr(kernel_parallel, "_service", None)
    assert not dispatch_admissible("parallel")
    run = tmp_path / "run"
    (run / "parent").mkdir(parents=True)
    kernel_parallel.install(run, {"parallel_mode": "2",
                                  "parent_snapshot": str(run / "parent")})
    assert dispatch_admissible("parallel")


def test_parent_parallel_call_is_admitted_by_live_continuation(
        tmp_path, monkeypatch, shipped_runtime_profile):
    _requests, continuation = _prepared_parent(tmp_path, monkeypatch, mode="2")
    assert isinstance(continuation.engine.state.active_id, str)
    assert continuation.engine.state.active_id.startswith("intention:")
    assert "parallel" in continuation.per_step_allow_set()
    continuation.runtime.step_ledger.pending = {}
    assert continuation.authorize("parallel", {"subframes": []}) is None
    assert isinstance(continuation.effect_id, str)


def test_scripted_parallel_call_reaches_parent_executor(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import kernel_parallel
    from kernel_completion_owner import dispatch_tool_call
    from kernel_parallel_admission import DirectorySnapshot, admit
    from kernel_parallel_records import ParallelLimits
    from probes.react_repair import _execute_parent_parallel, _tools_for_probe_modes

    run = tmp_path / "run"
    (run / "parent").mkdir(parents=True)
    for name in ("a.py", "b.py", "tests/test_a.py"):
        source = run / "parent" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("VALUE = 1\n")
    monkeypatch.setattr(kernel_parallel, "_service", None)
    kernel_parallel.install(run, {
        "parallel_mode": "2", "parallel_limits": ParallelLimits().to_json(),
        "parent_snapshot": str(run / "parent"),
        "run_identity_seed": "parent-seed",
    })
    tools = _tools_for_probe_modes(probe_gate_on=False, observation_probe_on=False)
    args = {"subframes": [{"id": name, "outcome": "Update source", "owns": [f"{name}.py"],
                           "creates": [], "exports": [], "consumes": [], "reads": [],
                           "requirements": [], "check": {"tests": []}}
                          for name in ("a", "b")]}
    assert dispatch_tool_call(tools, "parallel", args).name == "parallel"
    malformed = {"subframes": [{**args["subframes"][0], "unexpected": "refuse after tape intent"}]}
    assert dispatch_tool_call(tools, "parallel", malformed).name == "parallel"
    with pytest.raises(kernel_parallel.ParallelInputError, match="unexpected"):
        kernel_parallel._request(malformed, "call-1", "2")
    observed = []
    protected_test = run / "parent/tests/test_a.py"

    def executor(context, arguments, *, host, store_root):
        observed.append((context.mode, context.parent_seed, arguments, host, store_root))
        assert context.protected("tests/test_a.py") is True
        assert context.protected("a.py") is False
        snapshot = DirectorySnapshot(run / "parent", 0)
        request = kernel_parallel._request(arguments, "call-1", context.mode)
        assert admit(request, snapshot, (), context.limits,
                     protected=context.protected, case_insensitive=False).status == "ADMITTED"
        protected_args = {"subframes": [
            {**arguments["subframes"][0], "owns": ["tests/test_a.py"]},
            arguments["subframes"][1],
        ]}
        protected_request = kernel_parallel._request(protected_args, "call-1", context.mode)
        refusal = admit(protected_request, snapshot, (), context.limits,
                        protected=context.protected, case_insensitive=False)
        assert refusal.status == "REFUSED"
        assert refusal.refusal.reason == "UNSAFE_PATH"
        assert refusal.refusal.paths == ("tests/test_a.py",)
        return SimpleNamespace(observation="REFUSED OWNERSHIP_OVERLAP", txn_order=())

    monkeypatch.setattr(kernel_parallel, "execute_parallel", executor)
    runtime = SimpleNamespace(current_tool_call_id="call-1")
    call_name, result = _execute_parent_parallel(
        runtime, object(), run / "parent", set(), {str(protected_test)}, None, "call-1", args, run)
    assert call_name == "call-1"
    assert result.observation == "REFUSED OWNERSHIP_OVERLAP"
    assert observed == [("2", "parent-seed", args, None, run / "parallel/call-1")]
