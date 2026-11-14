"""A green subframe check may submit after its edit action is closed."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

from detreason.world_knowledge import WorldKnowledgeStore
from kernel_bolt import KernelBolt
import kernel_check_permits
import kernel_unit_policy
from kernel_done_is_submit_authority import (
    _pending_mutations, final_result, submission_pending, submit_done,
    terminal_from_seal,
)
from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract, UnitSpec
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_unit_policy import UnitPolicy, run_frozen_check
from probes import react_repair


def _permit(tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_check_permits, "_installed", (tmp_path / "permits", 1))


def test_shipped_profile_auto_finish_closes_applied_edit_without_wait(
        tmp_path, monkeypatch, shipped_runtime_profile):
    _permit(tmp_path, monkeypatch)

    root = tmp_path / "candidate"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    source = root / "pkg/a.py"
    source.write_text("value = 1\n")
    (root / "tests/test_a.py").write_text(
        "from pkg.a import value\n\ndef test_value(): assert value == 2\n")
    recipe = CheckRecipe(("tests/test_a.py",), ("tests/test_a.py::test_value",), "pytest")
    contract = UnitContract("a", "fix a", ("pkg/a.py",), (), (), (), (), (), recipe)
    spec = UnitSpec("call", "a", "execution", "seed", contract, (), (),
                    ("pkg/a.py", "tests/test_a.py"), (), "", 0, 0, "manifest", "policy",
                    ParallelLimits())
    selected = UnitPolicy(spec)
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: selected)
    owner = KernelBolt(root, "", "subframe-auto-finish")
    owner.world = WorldKnowledgeStore(sandbox_id="subframe-auto-finish")
    owner.task_row = {"msg": "Fix a.", "files": {"pkg/a.py": "value = 1\n"},
                      "entry_path": "pkg/a.py", "tests": ["tests/test_a.py"]}
    owner.current_step = 0
    tools = react_repair._tools_for_probe_modes(probe_gate_on=False,
                                                observation_probe_on=False)
    runtime = StateFrameRuntime(
        owner, output=tmp_path / "measurement", target=source,
        model="scripted", tools=tools, temperature=0,
        source_resolver=lambda _path, _symbol: None, multi_file=True,
        edit_contract={"edit_authority": {"editable_paths": ["pkg/a.py"]}},
    )
    assert runtime._unit_policy is selected
    assert runtime.continuation is not None
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "Task: Fix a."}]
    prepared = runtime.prepare(messages, uncapped=messages, tools=tools)
    assert prepared.overflow is None
    # The root intention's id is content-derived ("intention:<digest>", not the literal "root"
    # role name); the active intention must still be the parentless root for this task.
    state = runtime.continuation.engine.state
    assert state.active_id.startswith("intention:")
    assert state.intentions[state.active_id].parent_id is None
    assert state.intentions[state.active_id].objective == "Fix a."
    runtime.prepared_binding = runtime.binding()
    arguments = {"path": "pkg/a.py", "old_str": "value = 1", "new_str": "value = 2"}
    runtime.step_ledger.start("str_replace", arguments, 0, runtime.binding())
    assert runtime.continuation.authorize("str_replace", arguments) is None
    effect_id = runtime.continuation.effect_id
    assert isinstance(effect_id, str)
    source.write_text("value = 2\n")
    owner.receipt("str_replace", "replace", "pkg/a.py", "APPLIED",
                  target_path="pkg/a.py")
    def execute(candidate, _selector):
        if (candidate / "pkg/a.py").read_text() == "value = 2\n":
            return "pytest exit 0\n1 passed in 0.01s"
        return "pytest exit 1\n1 failed in 0.01s"

    control = tmp_path / "control"
    (control / "pkg").mkdir(parents=True)
    (control / "tests").mkdir()
    (control / "pkg/a.py").write_text("value = 1\n")
    (control / "tests/test_a.py").write_text((root / "tests/test_a.py").read_text())
    control_receipt = run_frozen_check(selected, control, execute=execute,
                                       evidence_dir=tmp_path / "control-evidence")
    assert control_receipt.status == "FAIL"
    receipt = run_frozen_check(selected, root, execute=execute,
                               evidence_dir=tmp_path / "evidence")
    assert receipt.status == "PASS"
    runtime._unit_check_receipt = receipt
    runtime._unit_auto_finish_receipt = receipt
    assert runtime.continuation.engine.state.effects[effect_id].status == "pending"
    assert _pending_mutations(runtime) == ("continuation:unsettled_effects",)

    started = time.monotonic()
    finished = react_repair._auto_finish_checked_subframe(
        runtime, selected, root, {"pkg/a.py": "value = 1\n"},
        [{"role": "user", "content": "edit applied"}], None)
    assert time.monotonic() - started < 1.0
    assert finished is True
    assert _pending_mutations(runtime) == ()
    assert runtime.continuation.engine.state.effects[effect_id].status == "applied"
    assert submission_pending(runtime)
    events = runtime.journal.records()
    assert [event["kind"] for event in events if event["kind"] == "subframe_auto_finished"] == [
        "subframe_auto_finished"]
    assert runtime.disposition_v2
    react_repair._classify_unsubmitted_loop(runtime, None)
    assert runtime.system_outcome is None
    assert not [event for event in runtime.journal.records()
                if event["kind"] == "v2_terminal"]

    submit_done(runtime, source="host", destination=tmp_path / "submissions",
                task_id="subframe-a", sources={"pkg/a.py": source.read_text()},
                parent_sources={"pkg/a.py": "value = 1\n"})
    terminal = terminal_from_seal(runtime)
    assert terminal is not None
    assert terminal["done_seal"]["source"] == "host"
    assert terminal["done_seal"]["schema"] == "stateframe-done-seal/2"
    assert terminal["done_seal"]["file_digests"] == {
        "pkg/a.py": hashlib.sha256(b"value = 2\n").hexdigest(),
    }
    assert terminal["done_seal"]["stored"] is True
    assert final_result(runtime, "fail") == "SUBMITTED"

    model = SimpleNamespace()
    submit_done(model, source="model", destination=tmp_path / "model-submissions",
                task_id="model-a", sources={"pkg/a.py": source.read_text()},
                parent_sources={"pkg/a.py": "value = 1\n"})
    model_terminal = terminal_from_seal(model)
    assert model_terminal is not None
    for field in ("result", "outcome", "result_class", "accepted", "artifact",
                  "delivery", "stop_reason", "behaviour"):
        assert terminal[field] == model_terminal[field]

    runtime.finish({"result": final_result(runtime, "fail"), "steps": 1,
                    "edits": 1, "tests": 1, "behaviour": "not_evaluated"},
                   messages=messages)
    finish = runtime._last_finish_result
    assert finish["result"] == "SUBMITTED"
    assert finish["accepted"] is True
    assert finish["terminal_disposition"]["artifact"] == "STORED"
    finished_events = [json.loads(line) for line in Path(runtime.journal.path).read_text().splitlines()]
    stop_events = [event for event in finished_events
                   if event["kind"] == "stop_reason_classified"]
    assert len(stop_events) == 1
    assert stop_events[0]["stop_cause"] == "DONE_ACCEPTED"
    terminations = [event for event in finished_events
                    if event["kind"] == "attempt_terminated"]
    assert len(terminations) == 1
    assert terminations[0]["stop_cause"] == "DONE_ACCEPTED"
    assert terminations[0]["system_outcome"] == "SUBMITTED"


def test_auto_finish_keeps_genuinely_pending_edit_blocked(tmp_path, monkeypatch, shipped_runtime_profile):
    # A pending effect with no closed host action must remain pending.
    _permit(tmp_path, monkeypatch)
    root = tmp_path / "candidate"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg/a.py").write_text("value = 2\n")
    (root / "tests/test_a.py").write_text("def test_value(): assert True\n")
    recipe = CheckRecipe(("tests/test_a.py",), ("tests/test_a.py::test_value",), "pytest")
    contract = UnitContract("a", "fix a", ("pkg/a.py",), (), (), (), (), (), recipe)
    selected = UnitPolicy(UnitSpec("call", "a", "execution", "seed", contract, (), (),
                                   ("pkg/a.py", "tests/test_a.py"), (), "", 0, 0,
                                   "manifest", "policy", ParallelLimits()))
    receipt = run_frozen_check(selected, root, execute=lambda _root, _selector:
                               "pytest exit 0\n1 passed in 0.01s",
                               evidence_dir=tmp_path / "evidence")
    assert receipt.status == "PASS"
    class PendingRuntime:
        _unit_check_receipt = receipt
        _unit_auto_finish_receipt = receipt
        _pending_mutation_ids = ("still-running",)
        system_outcome = None
        stateframe_done_is_submit_mode = "on"

        def __init__(self):
            self.events = []
            self.journal = type("Journal", (), {"emit": lambda _self, kind, **fields:
                                                 self.events.append((kind, fields))})()

        def close_step(self, _messages):
            pass

        def _deliver_auto_verify(self, _messages):
            pass

    runtime = PendingRuntime()
    assert not react_repair._auto_finish_checked_subframe(
        runtime, selected, root, {"pkg/a.py": "value = 1\n"}, [], None)
    assert runtime.events[-1] == ("subframe_auto_finish_declined",
                                   {"reason": "pending_mutations", "dispatch": False})
    assert not submission_pending(runtime)
    runtime._pending_mutation_ids = ()
    runtime.events.clear()
    assert not react_repair._auto_finish_checked_subframe(
        runtime, selected, root, {"pkg/a.py": "value = 1\n"}, [],
        SimpleNamespace(pending=[{"function": {"name": "str_replace"}}]))
    assert runtime.events[-1] == ("subframe_auto_finish_declined",
                                   {"reason": "pending_calls", "dispatch": False})
    assert not submission_pending(runtime)


def test_unsubmitted_loop_still_classifies_incomplete_on_shipped_profile(monkeypatch, shipped_runtime_profile):
    events = []
    runtime = SimpleNamespace(
        disposition_v2=True, system_outcome=None,
        journal=SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields))),
    )
    react_repair._classify_unsubmitted_loop(runtime, None)
    assert runtime.system_outcome == "run_incomplete"
    assert events == [("v2_terminal", {
        "terminal": "INCOMPLETE", "reason": "action loop ended without accepted done",
    })]
