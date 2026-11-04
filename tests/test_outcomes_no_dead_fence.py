"""A continuation fence is a named terminal throughout the product outcome."""
from types import SimpleNamespace

from kernel_done_is_submit_authority import final_result
from kernel_completion_owner import terminal_result
from kernel_continuation_journal import ContinuationJournal
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_pilot import ACTIVE_RUNTIME
from kernel_stop_reason_taxonomy import _event_reason
from stateframe import outcomes, runner


def test_continuation_blocked_projects_to_summary_and_cli_exit():
    terminal = {"result": "CONTINUATION_BLOCKED"}
    projected = outcomes.project({"engine_terminal": terminal, "engine_result": terminal["result"],
                                  "child_exit_code": 0})
    assert projected["runtime_outcome"] == "continuation_blocked"
    assert outcomes.exit_code(projected) == 1
    assert runner._session_end_is_final({"engine_terminal": terminal})
    runtime = SimpleNamespace(system_outcome="blocked",
                              continuation=SimpleNamespace(terminal_blocked=True))
    assert final_result(runtime, "fail") == "CONTINUATION_BLOCKED"


def test_v2_terminal_event_reports_continuation_blocked_reason(tmp_path):
    journal = ContinuationJournal(tmp_path, "blocked-reason", {"origin": "test"})
    try:
        event = journal.emit("v2_terminal", terminal="CONTINUATION_BLOCKED")
        assert _event_reason(event) == "CONTINUATION_BLOCKED"
    finally:
        journal.close()


def test_blocked_boundary_survives_completion_owner_finalization(tmp_path):
    journal = ContinuationJournal(tmp_path, "blocked-result", {"origin": "test"})
    try:
        runtime = StateFrameRuntime.__new__(StateFrameRuntime)
        runtime.journal = journal
        runtime.owner = SimpleNamespace(current_step=1)
        runtime.snapshot_id = "snapshot"
        runtime.endpoint_profile = None
        runtime.system_outcome = None
        runtime.stateframe_hard_budget_only_terminals_mode = "off"
        runtime.continuation = SimpleNamespace(
            prepare=lambda _messages: None, completed=False, terminal_blocked=True,
            engine=SimpleNamespace(state=SimpleNamespace(active_id="root")),
            _unresolved_effect_details=lambda _identity: [{"status": "unknown"}],
            _live_repair=lambda _detail: False,
            completion_finding=lambda: {"code": "CONTINUATION_BLOCKED"},
            finish=lambda: None)

        assert runtime._prepare_continuation_boundary([]) is None
        prepared = runtime._finalize_continuation_menu(
            SimpleNamespace(request_id="request"), ({"function": {"name": "done"}},))
        assert prepared.overflow["kind"] == "CONTINUATION_BLOCKED"
        assert runtime.system_outcome == "continuation_blocked"
        finalized = terminal_result(runtime, {"result": runtime.system_outcome.upper()})
        assert finalized["result"] == "CONTINUATION_BLOCKED"
        assert finalized["accepted"] is False

        runtime._finished = False
        runtime._active_capture = None
        runtime._record_passive_retention_capture = lambda _capture: None
        runtime._record_passive_retention_delivery_metric = lambda: None
        runtime.stateframe_completion_owner_mode = "on"
        runtime.stateframe_host_done_facts_mode = "off"
        runtime.stateframe_cache_prefill_economics_mode = "off"
        runtime.delivery_contract = None
        runtime.information = None
        runtime.lifecycle = lambda _event: None
        runtime.step_limit = 10
        runtime.parent = SimpleNamespace(snapshot=lambda: {}, completed_attempts=[])
        runtime.actions = SimpleNamespace(actions=[], diagnose=lambda: {})
        runtime.binding = lambda: {"product_revision": "sha256:fixture"}
        runtime._emit_tool_call_cap_audit = lambda: None
        runtime._runtime_token = ACTIVE_RUNTIME.set(runtime)
        runtime._parent_token = None
        runtime.finish({"result": runtime.system_outcome.upper(), "steps": 1})
        assert runtime._last_finish_result["result"] == "CONTINUATION_BLOCKED"
        assert runtime._last_finish_result["accepted"] is False
    finally:
        journal.close()
