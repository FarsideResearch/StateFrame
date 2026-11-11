"""Platform qualification refusals must leave mutation hooks and run outcomes safe."""

import json

import pytest

import kernel_mutate_exec as execute
from kernel_mutate_records import Journal, reconcile
from kernel_mutate_policy import Budget
from kernel_mutate_projection import authoring_view, panel, reserve_send
from kernel_mutate_runtime import MutationRuntime, MutationUnavailable
from stateframe import _child


def _observer():
    return {
        "schema": "stateframe-mutation-unit/1", "name": "observer", "revision": 1,
        "parent": None, "entry": "main.py", "summary": "Observe tool metadata",
        "kind": "hook", "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "requires": {"network": False, "writes": False},
        "limits": {"wall_s": 1, "cpu_s": 1, "mem_mb": 128},
        "source": "def handle(input, ctx):\n    return {}\n",
    }


def _rows(journal, kind):
    return [row for row in journal.events() if row["kind"] == kind]


def _rehydration_journal(run_dir, *, refused, runtime_state=False):
    journal = Journal(run_dir)
    if runtime_state:
        journal.append("episode_enter", episode="e1", goal="inspect")
        journal.append("watch_start", watch="w1", call="watch-call")
        journal.append("BUDGET_RESERVE", name="invocations", amount=2, id="charge-1")
    journal.append("HOOK_ATTACHED", name="observer", event="post_tool")
    journal.append("HOOK_TRIGGER_INTENT", invocation_id="hook-1", name="observer",
                   event="post_tool", parent_turn_id="read-call")
    if refused:
        journal.append("HOOK_REFUSED", invocation_id="hook-1", name="observer",
                       event="post_tool", code="MUTATION_PLATFORM_UNQUALIFIED",
                       reason="ldd is unavailable")
    journal.append("HOOK_DISABLED", name="observer", reason="platform_unqualified")
    return journal


def test_hook_on_unqualified_host_refuses_once_and_disables_as_platform_unqualified(tmp_path):
    calls = []

    def factory():
        calls.append(1)
        raise execute.MutationPlatformError(
            "cannot qualify Python runtime libraries: ldd is unavailable (FileNotFoundError)")

    journal = Journal(tmp_path)
    runtime = MutationRuntime(level="authored", journal=journal, executor_factory=factory)
    runtime.attach_hook(_observer(), event="post_tool")
    for index in range(4):
        result = runtime.invoke_hook(event="post_tool", facts={
            "tool": "read", "call_id": f"call-{index}"})
        assert not result.enabled

    refused = _rows(journal, "HOOK_REFUSED")
    assert len(refused) == 1
    assert refused[0]["code"] == "MUTATION_PLATFORM_UNQUALIFIED"
    assert "ldd is unavailable" in refused[0]["reason"]
    assert refused[0]["name"] == "observer"
    assert refused[0]["event"] == "post_tool"
    assert refused[0]["invocation_id"]
    assert calls == [1]
    assert _rows(journal, "HOOK_FAILURE") == []
    disabled = _rows(journal, "HOOK_DISABLED")
    assert len(disabled) == 1
    assert disabled[0]["reason"] == "platform_unqualified"
    assert runtime.hook_lines == []
    assert result.text == ""

    # Qualification is known now: attaching another observer is a recorded refusal.
    runtime.attach_hook(_observer(), event="post_tool")
    assert len(_rows(journal, "HOOK_REFUSED")) == 1
    assert not runtime._hook_manager().attached

    # A hook attached before another tool discovers the platform error is refused
    # before it can write a trigger intent or start a tape operation.
    known_journal = Journal(tmp_path / "preknown")
    known_runtime = MutationRuntime(level="authored", journal=known_journal)
    known_runtime.attach_hook(_observer(), event="post_tool")
    known_runtime._platform_error = "ldd is unavailable"
    result = known_runtime.invoke_hook(event="post_tool", facts={
        "tool": "read", "call_id": "preknown-call"})
    assert not result.enabled
    assert len(_rows(known_journal, "HOOK_REFUSED")) == 1
    assert _rows(known_journal, "HOOK_TRIGGER_INTENT") == []
    assert _rows(known_journal, "HOOK_DISABLED")[0]["reason"] == "platform_unqualified"
    assert known_runtime.hook_lines == []
    assert result.text == ""


def test_hook_refused_rows_do_not_change_runtime_rehydration(tmp_path):
    with_refusal = _rehydration_journal(tmp_path / "with", refused=True, runtime_state=True)
    without_refusal = _rehydration_journal(tmp_path / "without", refused=False, runtime_state=True)
    runtimes = [MutationRuntime(level="authored", journal=journal, run_dir=run_dir)
                for journal, run_dir in ((with_refusal, tmp_path / "with"),
                                         (without_refusal, tmp_path / "without"))]

    def rehydrated_state(runtime):
        return (runtime.episodes_started, runtime.episode, runtime.ready, runtime.watches,
                runtime.last, runtime.hook_lines, runtime._platform_error)

    assert rehydrated_state(runtimes[0]) == rehydrated_state(runtimes[1])
    assert panel(runtimes[0]) == panel(runtimes[1])
    assert runtimes[0].episodes_started == 1
    assert runtimes[0].episode is None
    assert runtimes[0].watches == {}
    assert _rows(with_refusal, "watch_stop") == _rows(without_refusal, "watch_stop")
    with_budget = Budget().fold(with_refusal.events())
    without_budget = Budget().fold(without_refusal.events())
    assert (with_budget.disabled, with_budget.remaining()) == (
        without_budget.disabled, without_budget.remaining())
    expected_rows = without_refusal.events()
    assert [row for row in with_refusal.events() if row["kind"] != "HOOK_REFUSED"] == expected_rows
    for runtime in runtimes:
        runtime.episode = {"id": "e1", "goal": "inspect", "sends_left": 4,
                           "draft": {}, "diagnostic": "none"}
    assert [reserve_send(runtime, "request-1", 1, 0) for runtime in runtimes] == [True, True]
    assert runtimes[0].episode == runtimes[1].episode
    assert panel(runtimes[0]) == panel(runtimes[1])
    assert authoring_view(runtimes[0]) == authoring_view(runtimes[1])


def test_hook_refused_rows_only_complete_hook_intents_for_reconciliation(tmp_path):
    with_refusal = _rehydration_journal(tmp_path / "with", refused=True)
    without_refusal = _rehydration_journal(tmp_path / "without", refused=False)
    completed = reconcile([], with_refusal, [])
    pending = reconcile([], without_refusal, [])
    assert completed.refuse is None
    assert completed.actions == []
    assert pending.refuse is None
    assert [(action.kind, action.call_id) for action in pending.actions] == [
        ("interrupt_hook", "hook-1")]


def test_refusal_reader_never_sees_hook_refused_rows(tmp_path):
    from kernel_tool_tape import ToolTape

    call_id = "mutate-refused"
    tape = ToolTape(tmp_path)
    tape.intent(call_id=call_id, tool="mutate", args={"op": "use"},
                replay_class="observation", capability_digest="cap", definition_digest=None,
                env_identity="env", pre_manifest_digest="before")
    observation = "REFUSED MUTATION_PLATFORM_UNQUALIFIED ldd is unavailable"
    tape.complete(call_id=call_id, exit_code=None, observation=observation,
                  post_manifest_digest="before", env_identity="env")
    tape_rows = [json.loads(line) for line in tape.path.read_text().splitlines()]
    journal = Journal(tmp_path)
    journal.append("REFUSAL", call=call_id, observation=observation)
    baseline = reconcile(tape_rows, journal, [])
    assert baseline.refuse is None
    assert [(action.kind, action.call_id) for action in baseline.actions] == [("serve", call_id)]

    journal.append("HOOK_TRIGGER_INTENT", invocation_id="hook-planted",
                   name="observer", event="post_tool")
    journal.append("HOOK_REFUSED", invocation_id=call_id, name="observer",
                   event="post_tool", code="MUTATION_PLATFORM_UNQUALIFIED")
    journal.append("HOOK_REFUSED", invocation_id="hook-planted", call=call_id,
                   name="observer", event="post_tool", code="MUTATION_PLATFORM_UNQUALIFIED")
    result = reconcile(tape_rows, journal, [])
    assert result.refuse is None
    assert result.actions == baseline.actions


def test_refused_hook_trigger_is_complete_for_resume_reconciliation(tmp_path):
    def factory():
        raise execute.MutationPlatformError("ldd is unavailable")

    journal = Journal(tmp_path)
    runtime = MutationRuntime(level="authored", journal=journal, executor_factory=factory)
    runtime.attach_hook(_observer(), event="post_tool")
    for index in range(3):
        result = runtime.invoke_hook(event="post_tool", facts={
            "tool": "read", "call_id": f"call-{index}"})
        assert not result.enabled

    intents = _rows(journal, "HOOK_TRIGGER_INTENT")
    refused = _rows(journal, "HOOK_REFUSED")
    assert len(intents) == len(refused) == 1
    assert intents[0]["invocation_id"] == refused[0]["invocation_id"]
    report = reconcile([], journal, [])
    assert report.refuse is None
    assert report.actions == []


def test_refused_hook_trigger_after_fresh_runtime_has_own_completion(tmp_path):
    def factory():
        raise execute.MutationPlatformError("ldd is unavailable")

    journal = Journal(tmp_path)
    for call_id in ("first-call", "resumed-call"):
        runtime = MutationRuntime(level="authored", journal=Journal(tmp_path),
                                  executor_factory=factory)
        assert runtime._platform_error is None
        runtime.attach_hook(_observer(), event="post_tool")
        assert not runtime.invoke_hook(event="post_tool", facts={
            "tool": "read", "call_id": call_id}).enabled

    intents = _rows(journal, "HOOK_TRIGGER_INTENT")
    refused = _rows(journal, "HOOK_REFUSED")
    # HOOK_DISABLED persists, so a fresh runtime cannot reattach this hook.
    assert len(intents) == len(refused) == 1
    assert {row["invocation_id"] for row in intents} == {
        row["invocation_id"] for row in refused}
    runtime._hook_manager().refuse_platform(
        "observer", "post_tool", refused[-1]["invocation_id"], "ldd is unavailable")
    assert len(_rows(journal, "HOOK_REFUSED")) == 1
    report = reconcile([], journal, [])
    assert report.refuse is None
    assert report.actions == []


def test_runtime_root_executor_platform_error_is_cached_and_refused(tmp_path, monkeypatch):
    calls = []

    def reject(_executor, *_args, **_kwargs):
        calls.append(1)
        raise execute.MutationPlatformError("ldd is unavailable")

    monkeypatch.setattr(execute.MutationExecutor, "__init__", reject)
    runtime = MutationRuntime(level="prebuilt", journal=Journal(tmp_path),
                              runtime_root=tmp_path)
    for _ in range(2):
        with pytest.raises(MutationUnavailable, match="ldd is unavailable"):
            runtime._ensure_executor()
    assert calls == [1]
    runtime._dispatch = lambda *_args: runtime._ensure_executor()
    reply = runtime.dispatch({"op": "use", "id": "file_facts", "args": "{}"}, "call-1")
    assert reply.startswith("REFUSED MUTATION_PLATFORM_UNQUALIFIED ")
    assert "ldd is unavailable" in reply
    assert calls == [1]


def test_session_end_platform_refusal_keeps_run_outcome():
    class Runtime:
        def __init__(self, hook_error, close_error=None):
            self.hook_error = hook_error
            self.close_error = close_error
            self.close_calls = 0

        def invoke_hook(self, *, event, facts):
            assert event == "session_end"
            assert facts == {"result": "SUBMITTED"}
            raise self.hook_error

        def close(self):
            self.close_calls += 1
            if self.close_error is not None:
                raise self.close_error

    original = {"result": "SUBMITTED"}
    events = []
    runtime = Runtime(MutationUnavailable("ldd is unavailable"))
    result, status = _child._finish_mutation_runtime(
        runtime, original, 0, lambda kind, **fields: events.append((kind, fields)))
    assert result is original
    assert status == 0
    assert runtime.close_calls == 1
    assert events == [("mutation_refused", {
        "code": "MUTATION_PLATFORM_UNQUALIFIED", "detail": "ldd is unavailable"})]

    runtime = Runtime(execute.MutationContainmentError("hook containment"))
    result, status = _child._finish_mutation_runtime(runtime, original, 0, lambda *_a, **_k: None)
    assert result["result"] == "MUTATION_CONTAINMENT"
    assert status == 1
    assert runtime.close_calls == 1

    runtime = Runtime(MutationUnavailable("ldd is unavailable"),
                      execute.MutationContainmentError("close containment"))
    result, status = _child._finish_mutation_runtime(runtime, original, 0, lambda *_a, **_k: None)
    assert result["result"] == "MUTATION_CONTAINMENT"
    assert result["error"] == "close containment"
    assert status == 1
    assert runtime.close_calls == 1
