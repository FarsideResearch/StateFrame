"""Terminal continuation fences retain a truthful, replayable outcome."""
import json
import inspect
import ast
import shutil
import textwrap
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import kernel_continuation_runtime as continuation_runtime
import kernel_continuation_evidence as continuation_evidence
import kernel_frame_routes
from kernel_continuation_runtime import LiveContinuation, tool_surface
from kernel_continuation_executive import ContinuationExecutive
from kernel_continuation_evidence import (APPLIED_RECEIPT_CLOSURES, READ_ONLY,
                                          binding_versions)
from kernel_continuation_evidence import (READ_OUTCOMES, READ_RECEIPT_KINDS,
                                          READ_RESULT_CLASSES, RUN_TEST_RESULT_CLASSES,
                                          PROBE_RESULT_CLASSES, PROBE_REFUSAL_RESULTS,
                                          HOST_OBLIGATION_RESULT_CLASSES)
from kernel_continuation_journal import ContinuationJournal
from kernel_continuation_menu_authority import EDIT_TOOLS
from kernel_continuation_state import ContinuationState, Effect, Intention
from kernel_measurement import digest
from kernel_pilot import ACTIVE_RUNTIME
from kernel_stateframe_runtime import StateFrameRuntime


def _live_effect(tool="shell", *, repair=None, turns_left=0):
    intention = Intention("root", None, "fix", "continue", {}, 0, 0, turns_left,
                          acceptance={"selectors": []}, phase="active")
    effect = Effect("effect-1", "root", "hash", {}, tool in READ_ONLY, tool=tool,
                    status="unknown", receipt_ref="closed-1")
    state = ContinuationState(active_id="root", intentions={"root": intention},
                              effects={"effect-1": effect})
    transitions = []
    engine = SimpleNamespace(state=state, retention_checkpoint=lambda: None,
                             choose_next=lambda _binding: None,
                             reobserve_host_effect=lambda **kwargs: transitions.append(kwargs) or "unknown")
    runtime = SimpleNamespace(binding=lambda: {}, completion_menu=(),
                              _completion_owner_full_tools=())
    live = LiveContinuation.__new__(LiveContinuation)
    live.runtime = runtime
    live.engine = engine
    live.journal = SimpleNamespace(emit=lambda *args, **kwargs: None, records=lambda: ())
    live.guarded_effects = None
    live.completed = False
    live.certified = False
    live.terminal_blocked = False
    live._stranded_blocker = None
    live._stranded_revision = None
    live.reconcile_external_effects = lambda: set()
    live.completion_blocker = lambda: "blocked"
    live._unresolved_effect_details = lambda _identity: [{
        "effect_id": "effect-1", "tool": tool, "status": "unknown",
        "read_only": tool in READ_ONLY,
        "repair_action": repair,
    }]
    return live, transitions


def test_candidate_done_only_menu_cannot_stop_available_read_repair():
    live, _ = _live_effect("read", repair={"tool": "read", "arguments": {"path": "a.py"}})
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.binding = lambda: {}
    runtime.completion_menu = ()
    runtime._completion_owner_full_tools = ({"function": {"name": "read"}},)
    runtime._continuation_request_tools = ({"function": {"name": "done"}},)
    runtime.stateframe_hard_budget_only_terminals_mode = "off"
    runtime.continuation = live
    live.runtime = runtime

    assert runtime._prepare_continuation_boundary([{"content": "task"}]) is None
    assert live.terminal_blocked is False
    assert runtime._finalize_continuation_menu(
        SimpleNamespace(request_id="request"), ({"function": {"name": "read"}},)) is None
    assert live.terminal_blocked is False
    live.completion_finding = lambda: {"code": "CONTINUATION_BLOCKED"}
    empty_menu_stop = runtime._finalize_continuation_menu(
        SimpleNamespace(request_id="empty-request"), ())
    assert empty_menu_stop is not None
    assert empty_menu_stop.overflow["kind"] == "CONTINUATION_BLOCKED"


def test_final_menu_block_terminates_under_both_budget_policies():
    for mode in ("off", "on"):
        live, _ = _live_effect()
        runtime = StateFrameRuntime.__new__(StateFrameRuntime)
        runtime.continuation = live
        runtime.stateframe_hard_budget_only_terminals_mode = mode
        runtime.system_outcome = None
        runtime.journal = SimpleNamespace(emit=lambda *args, **kwargs: None)
        live.runtime = runtime
        live.completion_finding = lambda: {"code": "CONTINUATION_BLOCKED"}

        result = runtime._finalize_continuation_menu(
            SimpleNamespace(request_id="request"), ({"function": {"name": "done"}},))
        assert result is not None
        assert result.overflow["kind"] == "CONTINUATION_BLOCKED"
        assert live.terminal_blocked is True
        assert runtime.system_outcome == "continuation_blocked"


def test_default_policy_preparation_stops_before_an_unrepairable_done_only_request():
    from kernel_terminal_policy import terminal_outcome_permitted

    live, _ = _live_effect("read")
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.continuation = live
    runtime.binding = lambda: {}
    runtime.completion_menu = ("done",)
    runtime._completion_owner_full_tools = ({"function": {"name": "done"}},)
    runtime.stateframe_hard_budget_only_terminals_mode = "on"
    runtime.system_outcome = None
    runtime.journal = SimpleNamespace(emit=lambda *args, **kwargs: None)
    runtime.catalog_completion_blocker = lambda: None
    live.runtime = runtime
    live.completion_finding = lambda: {"code": "CONTINUATION_BLOCKED"}

    assert runtime._prepare_continuation_boundary([{"role": "user", "content": "task"}]) is None
    prepared = runtime._finalize_continuation_menu(
        SimpleNamespace(request_id="request"), ({"function": {"name": "done"}},))
    assert prepared is not None and prepared.messages == ()
    assert prepared.overflow["kind"] == "CONTINUATION_BLOCKED"
    assert terminal_outcome_permitted(runtime.system_outcome, runtime)


def test_pre_request_continuation_preparation_has_an_iteration_bound():
    attempts = []
    continuation = SimpleNamespace(
        engine=SimpleNamespace(state=SimpleNamespace(active_id="root")),
        prepare=lambda _messages: attempts.append(True),
        _unresolved_effect_details=lambda _identity: [], completed=False)
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.continuation = continuation
    runtime.owner = SimpleNamespace(current_step=1)
    runtime.current_request_id = None
    runtime._completion_owner_full_tools = ()
    runtime._continuation_request_tools = ()
    for index in range(8):
        runtime.current_request_id = f"unsent-{index}"
        assert runtime._prepare_continuation_boundary([]) is None
    runtime.current_request_id = "unsent-ninth"
    with pytest.raises(RuntimeError, match="exceeded eight attempts"):
        runtime._prepare_continuation_boundary([])
    assert len(attempts) == 8
    runtime.owner.current_step = 2
    assert runtime._prepare_continuation_boundary([]) is None
    assert len(attempts) == 9


# Request-byte parity is checked against a release by
# test_hook_wiring_parity.py::test_no_hooks_raw_requests_and_journal_match_release
# and test_stable_tool_head_e2e.py::test_switch_off_requests_match_the_released_baseline.


def test_lean_checks_frozen_menu_before_prepared_request(monkeypatch):
    source = inspect.getsource(StateFrameRuntime._prepare_lean_plain_loop)
    assert (source.index("establish_lean_request_action_contract")
            < source.index("_finalize_continuation_menu(")
            < source.index('"request_prepared"'))
    monkeypatch.setenv("STATEFRAME_LEAN", "on")
    live, _ = _live_effect()
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.continuation = live
    runtime.stateframe_hard_budget_only_terminals_mode = "off"
    live.runtime = runtime
    live.completion_finding = lambda: {"code": "CONTINUATION_BLOCKED"}
    result = runtime._finalize_continuation_menu(
        SimpleNamespace(request_id="lean-request"), ({"function": {"name": "done"}},))
    assert result is not None and result.overflow["kind"] == "CONTINUATION_BLOCKED"


def test_unknown_write_without_durable_closure_stays_unsettled():
    live, transitions = _live_effect()
    live.prepare([{"content": "task"}])
    assert live.terminal_blocked is True
    assert transitions == []


def test_ordinary_refusal_does_not_terminalize_even_when_scheduler_stalls():
    """Pure regression guard: ordinary refusal never becomes a terminal effect fence."""
    live, _ = _live_effect()
    live.engine.state.effects.clear()
    live._unresolved_effect_details = lambda _identity: []
    live.prepare([{"content": "task"}])
    assert live.terminal_blocked is False
    assert live.completed is False


def _journaled_unknown(tmp_path, tool, *, interrupt=False, read_only=None):
    journal = ContinuationJournal(tmp_path, "no-dead-fence", {"origin": "test"})
    engine = ContinuationExecutive(journal)
    current = {"product_files": {"pkg/value.py": "sha256:source"}}
    engine.apply("create", new_id="root", objective="inspect", pending_decision="inspect",
                 dependencies=binding_versions(current), turns=2,
                 acceptance={"kind": "no_public_checks", "selectors": []})
    assert engine.choose_next(current) == "root"
    args = ({"path": "pkg/value.py"} if tool == "read" else
            {"target": "tests/test_value.py"} if tool == "run_tests" else
            {"command": "printf 'changed\\n' > pkg/value.py"}
            if tool == "shell" and interrupt else {"command": "true"})
    arguments_hash = digest(args)
    fence = journal.emit("step_dispatch_binding", producer="stateframe_runtime",
                         snapshot_id="snapshot", accepted=True, expected=current, current=current,
                         continuation_id="root", continuation_effect_id="effect-1",
                         request_id="request-1", step=0, tool=tool, arguments_hash=arguments_hash)
    engine.apply("begin_effect", intention_id="root", effect_id="effect-1",
                 action_hash=digest({"tool": tool, "arguments_hash": arguments_hash}),
                  arguments_hash=arguments_hash, tool=tool,
                  read_only=(tool == "read" if read_only is None else read_only),
                 step=0, request_id="request-1", snapshot_id="snapshot",
                 dispatch_ref=fence["event_id"], current_binding=current,
                 current_dependencies=binding_versions(current))
    if interrupt:
        assert engine.state.effects["effect-1"].status == "pending"
        return journal, engine, current
    record = {"tool": tool, "arguments_hash": arguments_hash, "step": 0,
              "continuation_effect_id": "effect-1", "continuation_dispatch_ref": fence["event_id"],
              "continuation_id": "root", "before_binding": current, "after_binding": current,
              "action_parameters": args, "result": "UNKNOWN", "receipt_ids": []}
    record["step_record_id"] = digest(record)
    closure = journal.emit("action_outcome_closed", record=record)
    engine.apply("settle_effect", effect_id="effect-1", receipt_ref=closure["event_id"],
                 current_binding=current, status="unknown")
    return journal, engine, current


def test_interrupted_pending_shell_write_blocks_without_a_durable_closure(tmp_path):
    journal, engine, current = _journaled_unknown(tmp_path, "shell", interrupt=True)
    live = _journal_live(journal, engine, current, ({"function": {"name": "done"}},))
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.continuation = live
    runtime.stateframe_hard_budget_only_terminals_mode = "off"
    runtime.system_outcome = None
    runtime.journal = SimpleNamespace(emit=lambda *args, **kwargs: None)
    runtime.catalog_completion_blocker = lambda: None
    live.runtime = runtime
    runtime.binding = lambda: current
    tools = ({"function": {"name": "done"}},)
    runtime._completion_owner_full_tools = tools
    live._request_tools = tools
    try:
        live.prepare([{"content": "inspect"}])
        assert "continuation_reobserve" not in live.per_step_allow_set()
        assert "continuation_reobserve" not in {tool["function"]["name"] for tool in tool_surface(
            [{"function": {"name": "done"}}], runtime=runtime)}
        detail = live._unresolved_effect_details("root")[0]
        assert detail["repair_action"] is None
        assert not live._live_repair(detail)
        assert engine.state.effects["effect-1"].status == "pending"
        assert runtime._finalize_continuation_menu(
            SimpleNamespace(request_id="pending-shell"), tools).overflow["kind"] == (
                "CONTINUATION_BLOCKED")
    finally:
        journal.close()


def test_pending_write_without_final_repair_blocks(tmp_path):
    journal, engine, current = _journaled_unknown(tmp_path, "shell", interrupt=True)
    live = _journal_live(journal, engine, current, ({"function": {"name": "done"}},))
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.continuation = live
    runtime.system_outcome = None
    runtime.journal = SimpleNamespace(emit=lambda *args, **kwargs: None)
    live.runtime = runtime
    runtime.binding = lambda: current
    runtime._completion_owner_full_tools = ()
    runtime.catalog_completion_blocker = lambda: None
    try:
        live.prepare([{"content": "inspect"}])
        result = runtime._finalize_continuation_menu(
            SimpleNamespace(request_id="pending-no-repair"),
            ({"function": {"name": "done"}},))
        assert result is not None
        assert result.overflow["kind"] == "CONTINUATION_BLOCKED"
    finally:
        journal.close()


def test_multiple_unresolved_effects_do_not_advertise_a_single_effect_repair():
    live, _ = _live_effect("read", repair={
        "tool": "read", "arguments": {"path": "pkg/value.py"}})
    detail = live._unresolved_effect_details("root")[0]
    live._unresolved_effect_details = lambda _identity: [
        detail, {**detail, "effect_id": "effect-2"}]
    live._request_tools = ({"function": {"name": "read"}},
                           {"function": {"name": "done"}})
    assert "read" not in live.per_step_allow_set()
    assert not live._live_repair(detail)
    assert live.completion_finding()["code"] == "CONTINUATION_BLOCKED"


def test_pending_shell_reobservation_settles_matching_durable_host_closure(tmp_path):
    journal, engine, current = _journaled_unknown(tmp_path, "shell", interrupt=True)
    effect = engine.state.effects["effect-1"]
    receipt_id = "shell-no-op"
    journal.emit("action_receipt", receipt={
        "receipt_id": receipt_id, "run_id": journal.run_id,
        "request_id": effect.request_id, "turn": effect.step,
        "kind": "shell", "result_class": "NO_OP",
        "continuation_id": effect.intention_id,
        "continuation_effect_id": effect.effect_id,
        "continuation_dispatch_ref": effect.dispatch_ref,
    })
    record = {"tool": "shell", "arguments_hash": effect.arguments_hash,
              "step": effect.step, "continuation_effect_id": effect.effect_id,
              "continuation_dispatch_ref": effect.dispatch_ref,
              "continuation_id": effect.intention_id, "before_binding": current,
              "after_binding": current, "result": "NO_OP", "changes": [],
              "receipt_ids": [receipt_id]}
    record["step_record_id"] = digest(record)
    journal.emit("action_outcome_closed", record=record)
    live = _journal_live(journal, engine, current, ())
    live.dispatch = {"tool": "continuation_reobserve", "event_id": "repair-dispatch",
                     "accepted": True, "continuation_id": "root"}
    live._authorized_arguments = {"effect_id": effect.effect_id}
    try:
        feedback = json.loads(live.operation("continuation_reobserve", {"effect_id": effect.effect_id}))
        assert feedback == {"status": "SETTLED", "effect_id": effect.effect_id}
        assert engine.state.effects[effect.effect_id].status == "not_applied"
    finally:
        journal.close()


def test_pending_shell_reobservation_keeps_unknown_host_outcome_unsettled(tmp_path):
    journal, engine, current = _journaled_unknown(tmp_path, "shell", interrupt=True)
    effect = engine.state.effects["effect-1"]
    record = {"tool": "shell", "arguments_hash": effect.arguments_hash,
              "step": effect.step, "continuation_effect_id": effect.effect_id,
              "continuation_dispatch_ref": effect.dispatch_ref,
              "continuation_id": effect.intention_id, "before_binding": current,
              "after_binding": current, "result": "UNKNOWN", "changes": [],
              "receipt_ids": []}
    record["step_record_id"] = digest(record)
    journal.emit("action_outcome_closed", record=record)
    live = _journal_live(journal, engine, current, ())
    live.dispatch = {"tool": "continuation_reobserve", "event_id": "repair-dispatch",
                     "accepted": True, "continuation_id": "root"}
    live._authorized_arguments = {"effect_id": effect.effect_id}
    try:
        before = engine.state.revision
        feedback = json.loads(live.operation("continuation_reobserve", {"effect_id": effect.effect_id}))
        assert feedback["status"] == "CONTINUATION_BLOCKED"
        assert engine.state.effects[effect.effect_id].status == "pending"
        assert engine.state.revision == before
    finally:
        journal.close()


def test_unknown_write_stops_before_optional_request_work(tmp_path):
    journal, engine, current = _journaled_unknown(tmp_path, "file_create")
    journal.emit("continuation_reobserve_attempt", effect_id="effect-1", dispatch=False)
    live = _journal_live(journal, engine, current, ({"function": {"name": "done"}},))
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.continuation = live
    runtime.binding = lambda: current
    runtime.owner = SimpleNamespace(current_step=2, task_row={})
    runtime.snapshot_id = "snapshot"
    runtime.endpoint_profile = object()
    runtime.journal = journal
    runtime.system_outcome = None
    runtime._completion_owner_full_tools = ({"function": {"name": "done"}},)
    runtime._continuation_request_tools = runtime._completion_owner_full_tools
    runtime.catalog_completion_blocker = lambda: None
    live.runtime = runtime
    try:
        stopped = runtime._prepare_continuation_boundary([{"content": "inspect"}])
        assert stopped is not None and stopped.messages == ()
        assert stopped.overflow["kind"] == "CONTINUATION_BLOCKED"
        assert runtime.system_outcome == "continuation_blocked"
    finally:
        journal.close()


def _journal_live(journal, engine, current, tools):
    runtime = SimpleNamespace(binding=lambda: current, completion_menu=(),
                              _completion_owner_full_tools=(), owner=SimpleNamespace(task_row={}))
    live = LiveContinuation.__new__(LiveContinuation)
    live.runtime, live.engine, live.journal = runtime, engine, journal
    live.guarded_effects = None
    live.completed = live.certified = live.terminal_blocked = False
    live._stranded_blocker = live._stranded_revision = None
    live._request_tools = tools
    live.reconcile_external_effects = lambda: set()
    live.completion_blocker = lambda: "unresolved effect"
    runtime.continuation = live
    return live


def _closed_noop_write(journal, effect, current):
    receipt_id = "write-no-op"
    journal.emit("action_receipt", receipt={
        "receipt_id": receipt_id, "run_id": journal.run_id,
        "request_id": effect.request_id, "turn": effect.step,
        "kind": effect.tool, "result_class": "NO_OP",
        "continuation_id": effect.intention_id,
        "continuation_effect_id": effect.effect_id,
        "continuation_dispatch_ref": effect.dispatch_ref,
    })
    record = {"tool": effect.tool, "arguments_hash": effect.arguments_hash,
              "step": effect.step, "continuation_effect_id": effect.effect_id,
              "continuation_dispatch_ref": effect.dispatch_ref,
              "continuation_id": effect.intention_id, "before_binding": current,
              "after_binding": current, "result": "NO_OP", "changes": [],
              "receipt_ids": [receipt_id]}
    record["step_record_id"] = digest(record)
    return journal.emit("action_outcome_closed", record=record)


def _closed_case(tmp_path, kind, *, read_result=None):
    """Journal a closure and stop at the exact pre-closed crash boundary."""
    journal = ContinuationJournal(tmp_path, "no-dead-fence", {"origin": "test"})
    journal_now = time.time_ns() // 1000
    engine = ContinuationExecutive(journal, clock_us=lambda: journal_now)
    current = {"product_files": {"pkg/value.py": "sha256:source"}}
    source_child = kind in {"source_read_child", "source_read_child_missing"}
    tool = ("read" if kind in {"applied_read", "read_no_source", "not_applied_read",
                                "unknown_read", "source_read_child", "source_read_child_missing"}
            else "run_tests" if kind in {"run_tests_green", "run_tests_red",
                                         "run_tests_no_tests", "run_tests_no_verdict",
                                         "reused_verification"}
            else "edit")
    acceptance = {"kind": "public_checks", "selectors": ["tests/test_value.py"]} if tool == "run_tests" else {
        "kind": "no_public_checks", "selectors": []}
    engine.apply("create", new_id="root", objective="inspect", pending_decision="inspect",
                 dependencies=binding_versions(current), turns=2, acceptance=acceptance)
    assert engine.choose_next(current) == "root"
    identity = "root"
    if source_child:
        engine.apply("detour", intention_id="root", reason="inspect source", wake_token="wake",
                     deadline_us=journal_now + 10_000_000, child={"new_id": "child", "objective": "read source",
                     "pending_decision": "inspect", "dependencies": binding_versions(current),
                     "turns": 1, "acceptance": {"kind": "source_read", "path": "pkg/value.py",
                                                "start_line": 1, "end_line": 1}})
        assert engine.choose_next(current) == "child"
        assert engine.state.intentions["child"].turns_left == 0
        identity = "child"

    def begin(effect_id, step):
        args = ({"path": "pkg/value.py"} if tool == "read" else
                {"target": "tests/test_value.py"} if tool == "run_tests" else
                {"path": "pkg/value.py"})
        arguments_hash = digest(args)
        fence = journal.emit("step_dispatch_binding", producer="stateframe_runtime",
            snapshot_id="snapshot", accepted=True, expected=current, current=current,
            continuation_id=identity, continuation_effect_id=effect_id,
            request_id="request-1", step=step, tool=tool, arguments_hash=arguments_hash)
        engine.apply("begin_effect", intention_id=identity, effect_id=effect_id,
            action_hash=digest({"tool": tool, "arguments_hash": arguments_hash}),
            arguments_hash=arguments_hash, tool=tool, read_only=tool == "read",
            step=step, request_id="request-1", snapshot_id="snapshot",
            dispatch_ref=fence["event_id"], current_binding=current,
            current_dependencies=binding_versions(current))
        return engine.state.effects[effect_id]

    def close(effect, result):
        receipt_id = "receipt-" + effect.effect_id
        journal.emit("action_receipt", receipt={
            "receipt_id": receipt_id, "run_id": journal.run_id,
            "request_id": effect.request_id, "turn": effect.step,
            "kind": READ_RECEIPT_KINDS[tool] if tool == "read" else
                    "test" if tool == "run_tests" else "edit",
            "result_class": result, "continuation_id": identity,
            "continuation_effect_id": effect.effect_id,
            "continuation_dispatch_ref": effect.dispatch_ref})
        if result in {"GREEN", "RED"}:
            journal.emit("test_subprocess_completed", producer="test_subprocess",
                selectors=["tests/test_value.py"], exit_code=0 if result == "GREEN" else 1,
                binding=current, after_binding=current, binding_stable=True,
                request_id=effect.request_id, continuation_id=identity,
                continuation_effect_id=effect.effect_id,
                continuation_dispatch_ref=effect.dispatch_ref)
        record = {"tool": tool, "arguments_hash": effect.arguments_hash,
            "step": effect.step, "continuation_effect_id": effect.effect_id,
            "continuation_dispatch_ref": effect.dispatch_ref,
            "continuation_id": identity, "before_binding": current,
            "after_binding": current, "result": result,
            "changes": ["pkg/value.py"] if result == "APPLIED" else [],
            "receipt_ids": [receipt_id]}
        if tool == "read" and result == "OK":
            record["read_evidence"] = {"authority": "HOST_OBSERVED_READ",
                "producer": "workspace_read", "path": "pkg/value.py",
                "file_revision": "sha256:source", "content": "value = 1",
                "content_hash": digest("value = 1"), "start_line": 1,
                "end_line": 1, "total_lines": 1, "omitted_chars": 0}
        record["step_record_id"] = digest(record)
        return journal.emit("action_outcome_closed", record=record)

    if kind == "reused_verification":
        first = begin("prior-test", 0)
        prior = close(first, "GREEN")
        live = _journal_live(journal, engine, current, ())
        live.closed(prior)
        effect = begin("effect-1", 1)
        verdict = continuation_runtime.evaluate_required_tests(
            current, ["tests/test_value.py"], engine._evidence.scan_rows())
        assert verdict.state == continuation_runtime.SATISFIED
        staged = (effect.effect_id, verdict, verdict.trusted_receipts[0])
        closure = close(effect, "REUSED_VERIFICATION")
    else:
        effect = begin("effect-1", 0)
        result = {"applied_write": "APPLIED", "not_applied_write": "NO_OP",
                  "unknown_write": "UNKNOWN",
                  "applied_read": "OK", "read_no_source": "MISSING",
                  "not_applied_read": "ERROR", "unknown_read": "UNKNOWN",
                   "source_read_child": "OK", "source_read_child_missing": "MISSING",
                   "run_tests_green": "GREEN",
                  "run_tests_red": "RED", "run_tests_no_tests": "NO_TESTS",
                  "run_tests_no_verdict": "NO_VERDICT"}[kind]
        closure = close(effect, read_result if read_result is not None else result)
        staged = None
    return journal, engine, current, closure, staged


def _discovered_closure_kinds():
    kinds = set()
    result_registries = {
        **READ_RESULT_CLASSES,
        "run_tests": RUN_TEST_RESULT_CLASSES,
        "probe": frozenset(PROBE_RESULT_CLASSES) | PROBE_REFUSAL_RESULTS,
        "host_obligation_check": HOST_OBLIGATION_RESULT_CLASSES,
    }
    kinds.update(f"result:{tool}:{result}" for tool, results in result_registries.items()
                 for result in results)
    kinds.update(f"receipt_closure:{tool}:{kind}"
                 for tool, kind in APPLIED_RECEIPT_CLOSURES.items())
    if continuation_runtime.registered_write_effect_tools() - {"run_tests"}:
        kinds.update({"applied_write", "not_applied_write", "unknown_write"})
    if continuation_runtime.READ_ONLY & READ_RECEIPT_KINDS.keys():
        kinds.update({"applied_read", "not_applied_read", "unknown_read"})
    if READ_OUTCOMES.get("read", frozenset()) - {"OK"}:
        kinds.add("read_no_source")
    if "read" in continuation_runtime.READ_ONLY:
        kinds.add("source_read_child")
    if "run_tests" in continuation_runtime.registered_write_effect_tools():
        kinds.update({"run_tests_green", "run_tests_red", "reused_verification"})
        if any(disposition == "settled_not_applied" and result != "REUSED_VERIFICATION"
               for result, disposition in RUN_TEST_RESULT_CLASSES.items()):
            kinds.add("run_tests_no_tests")
        if any(disposition == "retryable_unknown"
               for disposition in RUN_TEST_RESULT_CLASSES.values()):
            kinds.add("run_tests_no_verdict")
    special_tools = {"read": "source_read_child", "run_tests": "run_tests_green"}
    special_results = {"OK": "applied_read", "REUSED_VERIFICATION": "reused_verification"}
    for field, mapping in (("tool", special_tools), ("result", special_results)):
        for value in _closure_branch_values(field):
            kinds.add(mapping.get(value, f"uncovered_{field}:{value}"))
    for value in _closure_acceptance_kinds():
        kinds.add({"source_read": "source_read_child"}.get(
            value, f"uncovered_acceptance:{value}"))
    return kinds


def _closure_branch_values(field):
    """Catch a newly special-cased closure branch even if its registry class is old."""
    values = set()
    for method in (LiveContinuation.closed, LiveContinuation._closed_applied_followups):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            operands = (node.left, *node.comparators)
            reads_field = any(
                isinstance(item, ast.Subscript) and isinstance(item.value, ast.Name)
                and item.value.id == "record" and isinstance(item.slice, ast.Constant)
                and item.slice.value == field
                or isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                and isinstance(item.func.value, ast.Name) and item.func.value.id == "record"
                and item.func.attr == "get" and item.args
                and isinstance(item.args[0], ast.Constant) and item.args[0].value == field
                for item in operands)
            if reads_field:
                values.update(item.value for item in operands
                              if isinstance(item, ast.Constant) and isinstance(item.value, str))
    return values


def _closure_acceptance_kinds():
    tree = ast.parse(textwrap.dedent(inspect.getsource(
        LiveContinuation._closed_applied_followups)))
    values = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = (node.left, *node.comparators)
        reads_acceptance = any(
            isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
            and item.func.attr == "get" and item.args
            and isinstance(item.args[0], ast.Constant) and item.args[0].value == "kind"
            and isinstance(item.func.value, ast.Attribute)
            and item.func.value.attr == "acceptance"
            for item in operands)
        if reads_acceptance:
            values.update(item.value for item in operands
                          if isinstance(item, ast.Constant) and isinstance(item.value, str))
    return values


_CLOSURE_KINDS = frozenset({"applied_write", "not_applied_write", "unknown_write",
    "applied_read", "read_no_source", "not_applied_read", "unknown_read",
    "source_read_child", "run_tests_green", "run_tests_red", "run_tests_no_tests",
    "run_tests_no_verdict", "reused_verification"})

_EXPECTED_RESULT_CLASSES = {
    **{tool: {"NO_OP", "REFUSED"} for tool in
       ("job_wait", "job_output", "git", "web_fetch", "skill_load")},
    **{tool: {"OK", "MISSING", "DIRECTORY", "ALREADY_PRESENT", "ERROR", "OUT_OF_RANGE"}
       for tool in ("read", "text")},
    "grep": {"HITS", "NO_HITS", "ERROR", "OUT_OF_RANGE"},
    "ls": {"OK"},
    "list_files": {"HOST_OBSERVED", "REJECTED"},
    **{tool: {"HOST_OBSERVED", "ERROR", "OUT_OF_RANGE"}
       for tool in ("where_defined", "who_calls", "references")},
    **{tool: {"RECALLED", "RECALL_REJECTED"}
       for tool in ("recall", "recall_evidence", "recall_generated_artifact")},
    "read_check_receipt": {"HOST_OBSERVED", "NOT_FOUND", "REJECTED", "ERROR", "OUT_OF_RANGE"},
    "run_tests": {"GREEN", "PARTIAL_PASS", "RED", "FAIL_WITH_WITNESS", "HARNESS_ERROR",
                  "REFUSED_PREDISPATCH",
                  "SELECTOR_ERROR", "NO_TESTS", "NO_VERDICT", "REUSED_VERIFICATION",
                  "UNSUPPORTED_ACTION"},
    "probe": {"COMPLETED", "RAISED", "TIMED_OUT", "RESOURCE_LIMIT", "POLICY_DENIED",
              "RUNNER_ERROR", "NOT_RUN", "STALE", "HOST_OBSERVED", "REJECTED",
              "REFUSED", "REFUSED_PREDISPATCH"},
    "host_obligation_check": {"PASS", "COUNTEREXAMPLE", "NOT_EXECUTED",
                              "HARNESS_ERROR", "REUSED"},
}
_EXPECTED_RESULT_ROWS = frozenset(
    f"result:{tool}:{result}" for tool, results in _EXPECTED_RESULT_CLASSES.items()
    for result in results)
_EXPECTED_RECEIPT_CLOSURE_ROWS = frozenset({"receipt_closure:shell:shell_effect"})


def _assert_closure_inventory_complete():
    assert _discovered_closure_kinds() == (
        _CLOSURE_KINDS | _EXPECTED_RESULT_ROWS | _EXPECTED_RECEIPT_CLOSURE_ROWS)


@pytest.mark.parametrize("row", sorted(_EXPECTED_RESULT_ROWS | _EXPECTED_RECEIPT_CLOSURE_ROWS))
def test_closure_inventory_has_one_row_per_class(row):
    assert row in _discovered_closure_kinds()


def test_closure_inventory_catches_planted_read_result(monkeypatch):
    _assert_closure_inventory_complete()
    monkeypatch.setitem(READ_RESULT_CLASSES, "read",
                        READ_RESULT_CLASSES["read"] | {"PLANTED_RESULT"})
    with pytest.raises(AssertionError):
        _assert_closure_inventory_complete()
    assert "result:read:PLANTED_RESULT" in _discovered_closure_kinds()


@pytest.mark.parametrize("kind", sorted(_CLOSURE_KINDS))
def test_journaled_closure_resume_matches_uncrashed_followups(tmp_path, kind):
    _assert_closure_inventory_complete()
    start = tmp_path / "start"
    start.mkdir()
    journal, engine, current, closure, staged = _closed_case(start, kind)
    journal_now = closure["timestamp_ns"] // 1000
    head, path = journal.head, journal.path.relative_to(start)
    journal.close()
    baseline_dir = tmp_path / "baseline"
    crashed_dir = tmp_path / "crashed"
    shutil.copytree(start, baseline_dir)
    shutil.copytree(start, crashed_dir)

    def resume(directory):
        resumed = ContinuationJournal.resume(directory / path,
            task_id="no-dead-fence", manifest={"origin": "test"}, expected_head=head)
        executive = ContinuationExecutive(resumed, clock_us=lambda: journal_now)
        live = _journal_live(resumed, executive, current, ({"function": {"name": "done"}},))
        return resumed, executive, live

    baseline, baseline_engine, baseline_live = resume(baseline_dir)
    crashed, crashed_engine, crashed_live = resume(crashed_dir)
    try:
        if staged is not None:
            baseline_live._pending_reused_test = staged
        baseline_live.closed(closure)
        first_close_state = baseline_engine.state.document()
        first_close_rows = tuple(baseline.records())
        baseline_live.closed(closure)
        assert baseline_engine.state.document() == first_close_state
        assert tuple(baseline.records()) == first_close_rows
        baseline_live.prepare([{"content": "inspect"}])
        crashed_live.prepare([{"content": "inspect"}])
        assert crashed_engine.state.document() == baseline_engine.state.document()
        if kind == "source_read_child":
            child = crashed_engine.state.intentions["child"]
            parent = crashed_engine.state.intentions["root"]
            assert child.phase == "completed"
            assert parent.phase != "suspended"
            assert parent.phase in {"active", "eligible", "completed"} or (
                crashed_live.completion_finding()["code"] == "CONTINUATION_BLOCKED"
                and crashed_live.completion_finding()["detail"])
        before = tuple(crashed.records())
        crashed_live.closed(closure)
        assert crashed_engine.state.document() == baseline_engine.state.document()
        assert tuple(crashed.records()) == before
    finally:
        crashed.close()
        baseline.close()


def _exercise_source_child_recovery(tmp_path, kind, settled_first):
    start = tmp_path / "start"
    start.mkdir()
    journal, engine, current, closure, _ = _closed_case(start, kind)
    deadline = engine.state.intentions["root"].wake.deadline_us
    closure_time = closure["timestamp_ns"] // 1000
    assert closure_time < deadline
    if settled_first:
        engine.apply("settle_effect", effect_id="effect-1", receipt_ref=closure["event_id"],
                     current_binding=current, status="applied")
    head, path = journal.head, journal.path.relative_to(start)
    journal.close()
    baseline_dir, recovered_dir = tmp_path / "baseline", tmp_path / "recovered"
    shutil.copytree(start, baseline_dir)
    shutil.copytree(start, recovered_dir)

    def open_case(directory, now):
        resumed = ContinuationJournal.resume(directory / path, task_id="no-dead-fence",
            manifest={"origin": "test"}, expected_head=head)
        executive = ContinuationExecutive(resumed, clock_us=lambda: now)
        return resumed, executive, _journal_live(resumed, executive, current, ())

    baseline, baseline_engine, baseline_live = open_case(baseline_dir, closure_time)
    recovered, recovered_engine, recovered_live = open_case(recovered_dir, deadline + 1)
    try:
        baseline_live.closed(closure)
        baseline_live.prepare([{"content": "inspect"}])
        recovered_live.prepare([{"content": "inspect"}])
        assert recovered_engine.state.document() == baseline_engine.state.document()
        assert recovered_engine.state.intentions["root"].phase != "blocked"
    finally:
        recovered.close()
        baseline.close()


@pytest.mark.parametrize("kind,settled_first", [
    ("source_read_child", False), ("source_read_child", True),
    ("source_read_child_missing", False), ("source_read_child_missing", True),
])
def test_source_child_recovery_uses_closure_time_after_parent_deadline(
        tmp_path, kind, settled_first):
    _exercise_source_child_recovery(tmp_path, kind, settled_first)


def test_live_non_ok_child_still_expires_parent_after_deadline(tmp_path):
    journal, engine, current, closure, _ = _closed_case(tmp_path, "source_read_child_missing")
    live = _journal_live(journal, engine, current, ())
    try:
        live.closed(closure)
        deadline = engine.state.intentions["root"].wake.deadline_us
        engine.clock_us = lambda: deadline + 1
        live.prepare([{"content": "inspect"}])
        assert engine.state.intentions["root"].phase == "blocked"
    finally:
        journal.close()


@pytest.mark.parametrize("result", sorted(READ_RESULT_CLASSES["read"] - {"OK"}))
def test_exhausted_source_read_names_failed_path_and_result_in_terminal(tmp_path, result):
    journal, engine, current, closure, _ = _closed_case(
        tmp_path, "source_read_child_missing", read_result=result)
    live = _journal_live(journal, engine, current, ())
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.continuation = live
    runtime.journal = journal
    runtime.binding = lambda: current
    runtime.owner = SimpleNamespace(task_row={}, current_step=1)
    runtime.system_outcome = None
    live.runtime = runtime
    try:
        live.closed(closure)
        live.prepare([{"content": "inspect"}])
        terminal = runtime._finalize_continuation_menu(
            SimpleNamespace(request_id="failed-source-read"),
            ({"function": {"name": "done"}},))
        assert terminal is not None
        assert terminal.overflow["kind"] == "CONTINUATION_BLOCKED"
        assert runtime.system_outcome == "continuation_blocked"
        reason = terminal.overflow["finding"]["detail"]
        assert "pkg/value.py" in reason
        assert result in reason
        runtime._finished = False
        runtime._active_capture = None
        runtime._record_passive_retention_capture = lambda _capture: None
        runtime._record_passive_retention_delivery_metric = lambda: None
        runtime.stateframe_completion_owner_mode = "off"
        runtime.stateframe_host_done_facts_mode = "off"
        runtime.stateframe_cache_prefill_economics_mode = "off"
        runtime.delivery_contract = None
        runtime.information = None
        runtime.lifecycle = lambda _event: None
        runtime.actions = SimpleNamespace(actions=[], diagnose=lambda: {})
        runtime.step_limit = 10
        runtime.parent = SimpleNamespace(snapshot=lambda: {}, completed_attempts=[])
        runtime._emit_tool_call_cap_audit = lambda: None
        runtime._runtime_token = ACTIVE_RUNTIME.set(runtime)
        runtime._parent_token = None
        runtime.finish({"result": "CONTINUATION_BLOCKED", "steps": 1})
        terminal_reason = runtime._last_finish_result["error_message"]
        assert "pkg/value.py" in terminal_reason
        assert result in terminal_reason
    finally:
        journal.close()


def test_completed_source_child_replays_missing_return_ready(tmp_path):
    journal, engine, current, closure, _ = _closed_case(tmp_path, "source_read_child")
    live = _journal_live(journal, engine, current, ())
    try:
        live.closed(closure)
        assert engine.state.intentions["child"].phase == "completed"
        published = []
        engine._return_ledger = object()
        engine.mark_continuation_return = lambda *args, **kwargs: published.append((args, kwargs))
        live.closed(closure)
        assert len(published) == 1
        assert published[0][1]["receipt"] == closure["event_id"]
    finally:
        journal.close()


def test_closure_settlement_is_its_processed_record(tmp_path):
    journal, engine, current, closure, _ = _closed_case(tmp_path, "not_applied_write")
    live = _journal_live(journal, engine, current, ())
    try:
        before = len(tuple(journal.records()))
        live.closed(closure)
        appended = tuple(row for row in journal.records() if row["sequence"] > before)
        assert len(appended) == 1
        assert appended[0]["kind"] == "continuation_transition"
        assert appended[0]["operation"] == "settle_effect"
        assert appended[0]["data"]["receipt_ref"] == closure["event_id"]
    finally:
        journal.close()


def test_old_partial_unknown_settlement_replay_does_not_republish(tmp_path):
    journal, engine, current, closure, _ = _closed_case(tmp_path, "unknown_write")
    engine.apply("settle_effect", effect_id="effect-1", receipt_ref=closure["event_id"],
                 current_binding=current, status="unknown")
    revision = engine.state.revision
    head, path = journal.head, journal.path
    journal.close()
    resumed = ContinuationJournal.resume(path, task_id="no-dead-fence",
        manifest={"origin": "test"}, expected_head=head)
    executive = ContinuationExecutive(resumed)
    live = _journal_live(resumed, executive, current, ())
    try:
        live._reconcile_unresolved_effects()
        assert executive.state.revision == revision
        assert len([row for row in resumed.records() if row.get("kind") == "continuation_transition"
                    and row.get("operation") == "settle_effect"
                    and row["data"].get("receipt_ref") == closure["event_id"]]) == 1
    finally:
        resumed.close()


@pytest.mark.parametrize("status", ["applied", "not_applied"])
def test_settled_closures_use_the_same_matrix_without_status_filter(tmp_path, status):
    kind = "applied_write" if status == "applied" else "not_applied_write"
    journal, engine, current, closure, _ = _closed_case(tmp_path, kind)
    engine.apply("settle_effect", effect_id="effect-1", receipt_ref=closure["event_id"],
                 current_binding=current, status=status)
    later = journal.emit("action_outcome_closed", record=closure["record"])
    live = _journal_live(journal, engine, current, ())
    seen = []
    live._settle_durable_effect = lambda effect_id, binding, matches, **kwargs: (
        seen.append(tuple(row["event_id"] for row in matches)))
    try:
        assert continuation_runtime.reconciliation_action(
            "shell", status, closure=True) == continuation_runtime.SETTLE_FROM_DURABLE_CLOSURE
        live._reconcile_unresolved_effects()
        assert seen == [(closure["event_id"], later["event_id"])]
    finally:
        journal.close()


def test_every_registered_effect_resolves_to_a_class_row(monkeypatch):
    monkeypatch.setattr(kernel_frame_routes, "WRITE", kernel_frame_routes.WRITE | {"planted_write"})
    monkeypatch.setattr(continuation_runtime, "READ_ONLY",
                        continuation_runtime.READ_ONLY | {"planted_observation"})
    monkeypatch.setattr(continuation_evidence, "READ_ONLY",
                        continuation_evidence.READ_ONLY | {"planted_observation"})
    for tool in (continuation_runtime.registered_write_effect_tools()
                 | continuation_runtime.READ_ONLY):
        effect_class = continuation_runtime.effect_class(tool)
        assert effect_class in {"write", "read-only-effect"}
        for status in ("pending", "unknown"):
            for closure in (False, True):
                assert (effect_class, status, closure) in continuation_runtime.EFFECT_RECONCILIATION
                expected = (continuation_runtime.TYPED_REOBSERVE
                            if tool == "read" and not closure else
                            continuation_runtime.EFFECT_RECONCILIATION[
                                effect_class, status, closure])
                assert continuation_runtime.reconciliation_action(
                    tool, status, closure=closure) == expected


@pytest.mark.parametrize("status", ["pending", "unknown"])
def test_registry_writers_block_without_durable_closure(status):
    for tool in continuation_runtime.registered_write_effect_tools():
        live, _ = _live_effect(tool)
        del live._unresolved_effect_details
        live.engine.state.effects["effect-1"].status = status
        live._request_tools = ({"function": {"name": "done"}},)
        live.runtime.continuation = live
        detail = live._unresolved_effect_details("root")[0]
        assert detail["repair_action"] is None
        assert continuation_runtime.reconciliation_action(
            tool, status) == continuation_runtime.CONTINUATION_BLOCKED


@pytest.mark.parametrize("tool,read_only", [
    ("planted_write", False), ("planted_observation", True),
])
@pytest.mark.parametrize("status", ["pending", "unknown"])
@pytest.mark.parametrize("closure_present", [False, True])
def test_planted_effect_reconciliation_covers_crash_window(
        tmp_path, monkeypatch, tool, read_only, status, closure_present):
    monkeypatch.setattr(kernel_frame_routes, "WRITE", kernel_frame_routes.WRITE | {"planted_write"})
    monkeypatch.setattr(continuation_runtime, "READ_ONLY",
                        continuation_runtime.READ_ONLY | {"planted_observation"})
    monkeypatch.setattr(continuation_evidence, "READ_ONLY",
                        continuation_evidence.READ_ONLY | {"planted_observation"})
    journal, engine, current = _journaled_unknown(
        tmp_path, tool, interrupt=status == "pending", read_only=read_only)
    effect = engine.state.effects["effect-1"]
    if closure_present:
        _closed_noop_write(journal, effect, current)
    saved_head, path = journal.head, journal.path
    journal.close()
    journal = ContinuationJournal.resume(path, task_id="no-dead-fence",
                                         manifest={"origin": "test"}, expected_head=saved_head)
    engine = ContinuationExecutive(journal)
    live = _journal_live(journal, engine, current, ({"function": {"name": "done"}},))
    try:
        live.prepare([{"content": "inspect"}])
        if closure_present:
            assert engine.state.effects["effect-1"].status == "not_applied"
            assert not live._unresolved_effect_details("root")
        else:
            assert engine.state.effects["effect-1"].status == status
            finding = live.completion_finding()
            assert finding["code"] == "CONTINUATION_BLOCKED"
            assert "No durable closure" in finding["detail"]
    finally:
        journal.close()


@pytest.mark.parametrize("tool", sorted(EDIT_TOOLS))
@pytest.mark.parametrize("status", ["pending", "unknown"])
@pytest.mark.parametrize("closure_present", [False, True])
def test_write_status_reconciles_or_blocks_without_closure(
        tmp_path, tool, status, closure_present):
    """A write is never repeated; only correlated durable evidence may settle it."""
    journal, engine, current = _journaled_unknown(
        tmp_path, tool, interrupt=status == "pending")
    effect = engine.state.effects["effect-1"]
    if closure_present:
        _closed_noop_write(journal, effect, current)
    saved_head, path = journal.head, journal.path
    journal.close()
    journal = ContinuationJournal.resume(path, task_id="no-dead-fence",
                                         manifest={"origin": "test"}, expected_head=saved_head)
    engine = ContinuationExecutive(journal)
    tools = ({"function": {"name": "continuation_reobserve"}},
             {"function": {"name": "done"}})
    live = _journal_live(journal, engine, current, tools)
    try:
        live.prepare([{"content": "inspect"}])
        if closure_present:
            assert engine.state.effects["effect-1"].status == "not_applied"
            assert not live._unresolved_effect_details("root")
        else:
            assert engine.state.effects["effect-1"].status == status
            detail = live._unresolved_effect_details("root")[0]
            assert detail["repair_action"] is None
            assert live.completion_finding()["code"] == "CONTINUATION_BLOCKED"
            assert engine.state.effects["effect-1"].status == status
    finally:
        journal.close()


def test_reobservation_uses_one_binding_and_reports_committed_outcome(tmp_path):
    journal, engine, current = _journaled_unknown(tmp_path, "file_create", interrupt=True)
    effect = engine.state.effects["effect-1"]
    _closed_noop_write(journal, effect, current)
    live = _journal_live(journal, engine, current, ())
    changed = {"product_files": {"pkg/value.py": "sha256:changed"}}
    bindings = iter((current, changed))
    calls = []
    live.runtime.binding = lambda: calls.append(True) or next(bindings)
    live.dispatch = {"tool": "continuation_reobserve", "event_id": "repair-dispatch",
                     "accepted": True, "continuation_id": "root"}
    live._authorized_arguments = {"effect_id": "effect-1"}
    try:
        feedback = json.loads(live.operation("continuation_reobserve", {"effect_id": "effect-1"}))
        assert feedback["status"] == "SETTLED"
        assert engine.state.effects["effect-1"].status == "not_applied"
        assert len(calls) == 1
    finally:
        journal.close()


def test_reobservation_does_not_claim_settled_without_committed_status(tmp_path, monkeypatch):
    journal, engine, current = _journaled_unknown(tmp_path, "edit", interrupt=True)
    effect = engine.state.effects["effect-1"]
    _closed_noop_write(journal, effect, current)
    live = _journal_live(journal, engine, current, ())
    live.dispatch = {"tool": "continuation_reobserve", "event_id": "repair-dispatch",
                     "accepted": True, "continuation_id": "root"}
    live._authorized_arguments = {"effect_id": "effect-1"}
    monkeypatch.setattr(engine, "settle_host_effect", lambda **_kwargs: {"data": {"status": "unknown"}})
    try:
        feedback = json.loads(live.operation("continuation_reobserve", {"effect_id": "effect-1"}))
        assert feedback["status"] == "CONTINUATION_BLOCKED"
        assert engine.state.effects["effect-1"].status == "pending"
    finally:
        journal.close()


def _wire_request(live, tools):
    allowed = live.per_step_allow_set()
    return json.dumps({"messages": [{"role": "user", "content": "inspect"}],
                       "tools": [tool for tool in tools if tool["function"]["name"] in allowed]},
                      sort_keys=True, separators=(",", ":")).encode()


def test_restarted_unknown_read_rebuilds_same_wire_request_from_journal(tmp_path):
    journal, engine, current = _journaled_unknown(tmp_path, "read")
    tools = ({"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}},
        "required": ["path"], "additionalProperties": False}}},
             {"type": "function", "function": {"name": "done"}})
    first = _journal_live(journal, engine, current, tools)
    first.prepare([{"content": "inspect"}])
    assert first.terminal_blocked is False
    assert "read" in first.per_step_allow_set()
    first_request = _wire_request(first, tools)
    first_state = engine.state.document()
    saved_head, path = journal.head, journal.path
    journal.close()
    resumed_journal = ContinuationJournal.resume(path, task_id="no-dead-fence",
        manifest={"origin": "test"}, expected_head=saved_head)
    try:
        resumed_engine = ContinuationExecutive(resumed_journal)
        resumed = _journal_live(resumed_journal, resumed_engine, current, tools)
        assert resumed_engine.state.document() == first_state
        resumed.prepare([{"content": "inspect"}])
        assert resumed.terminal_blocked is False
        assert _wire_request(resumed, tools) == first_request
    finally:
        resumed_journal.close()


def test_restarted_unknown_write_has_same_terminal_from_journal(tmp_path):
    journal, engine, current = _journaled_unknown(tmp_path, "shell")
    tools = ({"type": "function", "function": {"name": "done"}},)
    first = _journal_live(journal, engine, current, tools)
    first.prepare([{"content": "inspect"}])
    assert first.terminal_blocked is True
    first_finding = json.dumps(first.completion_finding(), sort_keys=True)
    first_state = engine.state.document()
    saved_head, path = journal.head, journal.path
    journal.close()
    resumed_journal = ContinuationJournal.resume(path, task_id="no-dead-fence",
        manifest={"origin": "test"}, expected_head=saved_head)
    try:
        resumed_engine = ContinuationExecutive(resumed_journal)
        resumed = _journal_live(resumed_journal, resumed_engine, current, tools)
        assert resumed_engine.state.document() == first_state
        resumed.prepare([{"content": "inspect"}])
        assert resumed.terminal_blocked is True
        assert json.dumps(resumed.completion_finding(), sort_keys=True) == first_finding
        assert resumed_engine.state.document() == first_state
    finally:
        resumed_journal.close()


def test_final_menu_removal_of_test_retry_terminalizes_unknown_effect():
    live, _ = _live_effect("run_tests", repair={
        "tool": "run_tests", "arguments": {"target": "tests/test_value.py"}})
    live.runtime._completion_owner_full_tools = ({"function": {"name": "run_tests"}},)
    live._request_tools = ({"function": {"name": "done"}},)
    live.prepare([{"content": "task"}])
    assert live.terminal_blocked is True
    assert live.completion_finding()["code"] == "CONTINUATION_BLOCKED"


def test_read_retry_requiring_missing_argument_terminalizes_unknown_effect():
    live, _ = _live_effect("read")
    del live._unresolved_effect_details
    live._effect_path = lambda _effect: "pkg/value.py"
    live._request_tools = ({"function": {"name": "read", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"},
                                       "cursor": {"type": "string"}},
        "required": ["path", "cursor"], "additionalProperties": False}}},)
    live.prepare([{"content": "task"}])
    assert live.terminal_blocked is True


def test_test_retry_uses_final_schema_instead_of_cached_owner_schema():
    live, _ = _live_effect("run_tests")
    del live._unresolved_effect_details
    live.journal.records = lambda: [{"record": {
        "continuation_effect_id": "effect-1",
        "action_parameters": {"target": "tests/test_value.py"},
    }}]
    live.runtime._completion_owner_full_tools = ({"function": {
        "name": "run_tests", "parameters": {"type": "object", "properties": {
            "target": {"type": "string"}}, "required": ["target"]}}},)
    live._request_tools = ({"function": {
        "name": "run_tests", "parameters": {"type": "object", "properties": {
            "target": {"type": "string"}, "mode": {"type": "string"}},
            "required": ["target", "mode"]}}},)
    live.prepare([{"content": "task"}])
    assert live.terminal_blocked is True


def test_unknown_reobservation_does_not_publish_another_settlement():
    live, _ = _live_effect()
    executive = ContinuationExecutive.__new__(ContinuationExecutive)
    executive._state = live.engine.state
    executive.journal = SimpleNamespace(ownership=nullcontext)
    executive._evidence = SimpleNamespace(effect_outcome=lambda *_args: "unknown")
    transitions = []
    executive.apply = lambda *args, **kwargs: transitions.append((args, kwargs))
    assert executive.reobserve_host_effect(effect_id="effect-1", current_binding={}) == "unknown"
    assert executive.reobserve_host_effect(effect_id="effect-1", current_binding={}) == "unknown"
    assert transitions == []


def test_reobservation_can_settle_newly_proven_effect():
    live, _ = _live_effect()
    executive = ContinuationExecutive.__new__(ContinuationExecutive)
    executive._state = live.engine.state
    executive.journal = SimpleNamespace(ownership=nullcontext)
    executive._evidence = SimpleNamespace(effect_outcome=lambda *_args: "applied")
    transitions = []
    executive.apply = lambda operation, **kwargs: (
        transitions.append((operation, kwargs))
        or {"operation": operation, "data": kwargs})
    assert executive.reobserve_host_effect(effect_id="effect-1", current_binding={}) == "applied"
    assert transitions == [("settle_effect", {"effect_id": "effect-1",
                                        "receipt_ref": "closed-1", "current_binding": {},
                                        "status": "applied"})]


def test_resumed_read_repair_is_projected_before_completion_menu_exists():
    live, _ = _live_effect("read")
    del live._unresolved_effect_details
    live._effect_path = lambda _effect: "pkg/value.py"
    live.runtime._completion_owner_full_tools = ({"function": {"name": "read", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}},
        "required": ["path"], "additionalProperties": False}}},
                                                 {"function": {"name": "done"}})
    assert "read" in live.per_step_allow_set()


def test_unknown_test_effect_blocks_even_with_recorded_target():
    live, _ = _live_effect("run_tests")
    del live._unresolved_effect_details
    live.journal.records = lambda: [{"record": {
        "continuation_effect_id": "effect-1",
        "action_parameters": {"target": "tests/test_value.py"},
    }}]
    live.runtime._completion_owner_full_tools = ({"function": {
        "name": "run_tests", "parameters": {"type": "object", "properties": {
            "target": {"type": "string"}}, "required": ["target"],
            "additionalProperties": False}}},)
    assert "run_tests" not in live.per_step_allow_set()
    assert live.completion_finding()["code"] == "CONTINUATION_BLOCKED"


def test_repair_finding_requires_action_on_projected_menu():
    live, _ = _live_effect("shell", repair={"tool": "read", "arguments": {"path": "a.py"}})
    live.per_step_allow_set = lambda: frozenset({"done"})
    finding = live.completion_finding()
    assert finding["code"] == "CONTINUATION_BLOCKED"


def test_unknown_read_cross_tool_route_is_not_a_dispatchable_repair():
    for tool, arguments in (("probe", {"clause_id": "clause"}),
                            ("run_tests", {"target": "tests/test_value.py"})):
        live, _ = _live_effect("read", repair={"tool": tool, "arguments": arguments})
        live.runtime.completion_menu = (tool, "done")
        live._request_tools = ({"function": {"name": tool}},
                               {"function": {"name": "done"}})
        assert tool not in live.per_step_allow_set()
        assert live.completion_finding()["code"] == "CONTINUATION_BLOCKED"


@pytest.mark.parametrize("tool,arguments", [
    ("probe", {"clause_id": "clause"}),
    ("run_tests", {"target": "tests/test_value.py"}),
])
def test_unknown_read_with_only_owner_cross_tool_route_blocks(monkeypatch, tool, arguments):
    import kernel_completion_owner

    live, _ = _live_effect("read")
    del live._unresolved_effect_details
    live._effect_path = lambda _effect: None
    name = next(iter(arguments))
    schema = {"function": {"name": tool, "parameters": {
        "type": "object", "properties": {name: {"type": "string"}},
        "required": [name], "additionalProperties": False}}}
    live.runtime._completion_owner_full_tools = (schema,)
    live._request_tools = (schema,)
    monkeypatch.setattr(kernel_completion_owner, "legal_next_actions",
                        lambda *_args, **_kwargs: ({"tool": tool, "arguments": arguments},))

    assert live._unresolved_effect_details("root")[0]["repair_action"] is None
    assert tool not in live.per_step_allow_set()
    assert live.completion_finding()["code"] == "CONTINUATION_BLOCKED"


def test_matching_read_repair_predicate_dispatches_with_host_evidence(tmp_path):
    for tool, arguments in (("read", {"path": "pkg/value.py"}),):
        journal_parent = tmp_path / tool
        journal_parent.mkdir()
        journal, engine, current = _journaled_unknown(journal_parent, tool)
        argument_name = next(iter(arguments))
        tools = ({"function": {"name": tool, "parameters": {
            "type": "object", "properties": {argument_name: {"type": "string"}},
            "required": [argument_name], "additionalProperties": False}}},)
        live = _journal_live(journal, engine, current, tools)
        try:
            detail = live._unresolved_effect_details("root")[0]
            assert live._live_repair(detail)
            assert live._repair_unresolved_effect(tool, arguments, "root")
            assert engine.state.effects["effect-1"].status == "not_applied"
        finally:
            journal.close()
