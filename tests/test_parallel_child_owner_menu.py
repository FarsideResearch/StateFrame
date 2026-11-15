"""Child request authority and optional owner menus agree with the unit policy."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import kernel_unit_policy
from detreason.world_knowledge import WorldKnowledgeStore
from kernel_bolt import KernelBolt
from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract, UnitSpec
from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
from kernel_stateframe_envelope import serialize_stateframe_payload
from kernel_stateframe_framefit5 import attach_owner_protocol
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_unit_policy import UnitPolicy
from kernel_unit_task import build_public_row
from probes.react_repair import _tools_for_probe_modes


@pytest.fixture
def assert_subframe_surface(monkeypatch):
    # e2e_support imports process_signals from tests/fixtures, so that dir must be importable.
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "fixtures"))
    from fixtures.parallel.e2e_support import assert_subframe_surface as check
    return check


@pytest.fixture(params=[False, True], ids=["no-check", "frozen-check"])
def child_request(request, tmp_path, monkeypatch, shipped_runtime_profile):
    source = b"VALUE = 0\n"
    check = b"def test_value():\n    from value import VALUE\n    assert VALUE == 1\n"
    (tmp_path / "value.py").write_bytes(source)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_value.py").write_bytes(check)
    recipe = (CheckRecipe(("tests/test_value.py",), ("tests/test_value.py",), "pytest")
              if request.param else CheckRecipe((), (), "none"))
    contract = UnitContract("value", "Set VALUE to one.", ("value.py",),
                            (), (), (), (), (), recipe)
    spec = UnitSpec("call", "value", "execution", "seed", contract, (), (),
                    ("value.py", "tests/test_value.py"), (), "", 0, 0,
                    "manifest", "policy", ParallelLimits())
    policy = UnitPolicy(spec)
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: policy)
    row = build_public_row(spec, {"value.py": source, "tests/test_value.py": check})
    owner = KernelBolt(tmp_path, "", "execution")
    owner.world = WorldKnowledgeStore(sandbox_id="execution")
    owner.task_row = row
    owner.f2p_ids = recipe.selectors
    owner.f2p_names = recipe.selectors
    owner.current_step = 1
    tools = _tools_for_probe_modes(probe_gate_on=False, observation_probe_on=False)
    parent_token = ACTIVE_PARENT.set(None)
    runtime_token = ACTIVE_RUNTIME.set(None)
    try:
        runtime = StateFrameRuntime(
            owner, output=tmp_path / "measurement", target=tmp_path / "value.py",
            model="scripted", tools=tools, temperature=0,
            source_resolver=lambda _path, _symbol: None,
            multi_file=True, edit_contract={"edit_authority": {"editable_paths": ["value.py"]}},
        )
        messages = [{"role": "system", "content": "rules"},
                    {"role": "user", "content": "Task: " + row["msg"]}]
        prepared = runtime.prepare(messages, uncapped=messages, tools=tools)
        assert prepared.overflow is None
        yield policy, runtime, prepared, tmp_path / "child.request.json"
    finally:
        ACTIVE_RUNTIME.reset(runtime_token)
        ACTIVE_PARENT.reset(parent_token)


def _write_request(path: Path, messages: list, tools: list) -> None:
    path.write_text(json.dumps({"messages": messages, "tools": tools}))


def test_child_admitted_actions_stay_within_allowed_tools(child_request, assert_subframe_surface):
    policy, runtime, prepared, path = child_request
    contract = prepared.capture.request_action_contract
    admitted = {tool["function"]["name"] for tool in contract["admitted_tool_schemas"]}
    assert admitted
    assert admitted <= policy.allowed_tools
    assert set(contract["advertised_actions"]) == admitted
    assert {tool["function"]["name"] for tool in prepared.tools} == admitted
    attached = attach_owner_protocol(prepared.messages, prepared.messages, runtime, contract)
    for boundary, messages in (("prepared", prepared.messages), ("attached", attached)):
        request_path = path.with_name(f"{boundary}.request.json")
        _write_request(request_path, messages, prepared.tools)
        frame = assert_subframe_surface(request_path)["state"]["framefit5"]
        if boundary == "attached":
            assert set(frame["owner_protocol"]["admitted_actions"]) == admitted
        assert frame["subframe_scope"]["unit_id"] == policy.spec.unit_id


def test_child_surface_accepts_request_without_owner_protocol(tmp_path, assert_subframe_surface):
    payload = {"schema": "stateframe-f4/1", "state": {"framefit5": {
        "mode": "FRAME_ONLY",
    }}}
    messages = [{"role": "user", "content": serialize_stateframe_payload(payload)}]
    path = tmp_path / "child.request.json"
    _write_request(path, messages, [kernel_unit_policy.UNIT_BLOCKED_TOOL])
    result = assert_subframe_surface(path)
    assert "owner_protocol" not in result["state"]["framefit5"]


def test_child_surface_accepts_schema_only_owner_protocol(tmp_path, assert_subframe_surface):
    payload = {"schema": "stateframe-f4/1", "state": {"framefit5": {
        "mode": "FRAME_ONLY", "owner_protocol": {"schema": "stateframe-framefit5-owner/1"},
    }}}
    messages = [{"role": "user", "content": serialize_stateframe_payload(payload)}]
    path = tmp_path / "child.request.json"
    _write_request(path, messages, [kernel_unit_policy.UNIT_BLOCKED_TOOL])
    assert_subframe_surface(path)


@pytest.mark.parametrize("owner", [None, [], {}, {"schema": "other/1"},
                                  {"schema": "stateframe-framefit5-owner/1", "route": "ACTION"},
                                  {"schema": "stateframe-framefit5-owner/1", "menu_source": "x"},
                                  {"schema": "stateframe-framefit5-owner/1", "admitted_actions": "read"},
                                  {"schema": "stateframe-framefit5-owner/1", "admitted_actions": [1]},
                                  {"schema": "stateframe-framefit5-owner/1", "admitted_actions": ["shell"]}])
def test_child_surface_rejects_invalid_attached_owner_protocol(tmp_path, assert_subframe_surface, owner):
    payload = {"schema": "stateframe-f4/1", "state": {"framefit5": {
        "mode": "FRAME_ONLY", "owner_protocol": owner,
    }}}
    messages = [{"role": "user", "content": serialize_stateframe_payload(payload)}]
    path = tmp_path / "child.request.json"
    _write_request(path, messages, [kernel_unit_policy.UNIT_BLOCKED_TOOL])
    with pytest.raises(AssertionError):
        assert_subframe_surface(path)
