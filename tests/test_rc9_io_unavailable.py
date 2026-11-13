"""RC9 live I/O gaps stay typed and journaled; request contracts still escape."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_green_coherence as coherence
import kernel_green_lock as green_lock
import kernel_intent_carry as intent_carry
import kernel_owner_first as owner_first
import kernel_request_budget as request_budget
import kernel_request_diet as request_diet
import kernel_request_source_window as source_window
import kernel_stateframe_hstate_plus as hstate_plus
import kernel_stateframe_get_sure as get_sure
import kernel_stateframe_runtime as stateframe_runtime
import kernel_step_contract as step_contract
from kernel_completion_owner import RequestActionContractError
from kernel_obligation_assess import UnavailableEvidence


def _journal():
    events = []
    return SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields))), events


def _gap(value, events, code, producer, *, sub_reason=None):
    assert isinstance(value, UnavailableEvidence)
    assert value.reason_code == code
    fields = {"producer": producer, "dispatch": False, "reason_code": code}
    if sub_reason is not None:
        fields["sub_reason"] = sub_reason
    assert events == [("host_obligation_evidence_failure", fields)]


def _raises(error):
    def raise_error(*_args, **_kwargs):
        raise error
    return raise_error


def test_planted_live_owner_first_target_gap(monkeypatch):
    journal, events = _journal()
    monkeypatch.setattr(source_window, "_path_from_runtime", _raises(OSError("target read")))

    result = owner_first.build_priority_plan({}, SimpleNamespace(journal=journal))

    _gap(result, events, "OWNER_FIRST_TARGET_UNAVAILABLE", "kernel_owner_first")


def test_transport_budget_closes_on_owner_first_gap(monkeypatch):
    journal, events = _journal()
    runtime = SimpleNamespace(journal=journal)
    capture = SimpleNamespace(
        journal=journal, presentation_runtime=runtime, request_budget_prepared=True,
    )
    body = {"messages": [], "tools": []}
    monkeypatch.setenv("STATEFRAME_OWNER_FIRST_ALLOC", "on")
    monkeypatch.setattr(source_window, "_path_from_runtime", _raises(OSError("target read")))

    result = request_budget.apply_capture_budget(capture, body)

    assert result is body
    gap = capture.request_budget_unavailable
    assert isinstance(gap, UnavailableEvidence)
    assert gap.reason_code == "OWNER_FIRST_TARGET_UNAVAILABLE"
    assert capture.request_budget_final_wire["status"] == request_budget.HOST_CONTRACT_LIMIT
    assert events == [("host_obligation_evidence_failure", {
        "producer": "kernel_owner_first", "dispatch": False,
        "reason_code": "OWNER_FIRST_TARGET_UNAVAILABLE",
    })]


def test_request_diet_propagates_owner_first_gap(monkeypatch):
    journal, events = _journal()
    runtime = SimpleNamespace(journal=journal)
    import kernel_pilot
    monkeypatch.setenv("STATEFRAME_OWNER_FIRST_ALLOC", "on")
    monkeypatch.setattr(kernel_pilot, "ACTIVE_RUNTIME", SimpleNamespace(get=lambda: runtime))
    monkeypatch.setattr(source_window, "_path_from_runtime", _raises(OSError("target read")))
    body = {"messages": [{"role": "user", "content": "brief"}], "tools": []}

    result = request_diet.apply_diet(
        body, stage="FREE", max_request_bytes=100_000,
        token_budget=100_000,
        token_meter=request_diet.RequestTokenMeter(tokenizer=lambda _text: 1),
    )

    _gap(result, events, "OWNER_FIRST_TARGET_UNAVAILABLE", "kernel_owner_first")
    assert body == {"messages": [{"role": "user", "content": "brief"}], "tools": []}


@pytest.mark.parametrize("failure", [OSError("secondary read"), KeyError("source lookup")])
def test_planted_live_secondary_source_gap(monkeypatch, tmp_path, failure):
    journal, events = _journal()
    target = tmp_path / "requested.py"
    target.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(Path, "read_text", _raises(failure))
    read_requested_source = getattr(stateframe_runtime, "_read_requested_source", None)
    assert callable(read_requested_source), "_frame must route secondary reads through its unavailable boundary"

    result = read_requested_source(target, tmp_path.resolve(), {}, journal)

    _gap(result, events, "STATEFRAME_SECONDARY_SOURCE_UNAVAILABLE", "kernel_stateframe_runtime")


@pytest.mark.parametrize("failure", [OSError("revision read"), KeyError("revision lookup")])
def test_planted_live_get_sure_revision_gap(monkeypatch, tmp_path, failure):
    journal, events = _journal()
    target = tmp_path / "target.py"
    target.write_text("value = 1\n", encoding="utf-8")
    original = Path.read_bytes

    def broken_read(path):
        if path == target:
            raise failure
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", broken_read)
    result = get_sure.runtime_table(SimpleNamespace(target=target, journal=journal))

    _gap(result, events, "GET_SURE_RUNTIME_REVISION_UNAVAILABLE", "kernel_stateframe_get_sure")


def test_planted_live_intent_path_gap(monkeypatch, tmp_path):
    journal, events = _journal()
    target = tmp_path / "target.py"
    original = Path.resolve

    def broken_resolve(path, *args, **kwargs):
        if path == target:
            raise OSError("intent path changed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", broken_resolve)
    result = intent_carry.normalize_path("target.py", tmp_path, journal=journal)

    _gap(result, events, "INTENT_PATH_UNAVAILABLE", "kernel_intent_carry")


def test_planted_live_intent_redirect_source_gap(monkeypatch, tmp_path):
    journal, events = _journal()
    target = tmp_path / "target.py"
    target.write_text("def change():\n    return 2\n", encoding="utf-8")
    state = intent_carry.IntentCarryState(journal, root=tmp_path)
    pending = intent_carry.PendingIntent("target.py", "change", "change the function", 1)
    state._pending[pending.intent_id] = pending
    monkeypatch.setattr(Path, "read_text", _raises(OSError("current source changed")))

    result = state.redirect_already_present({"message": "already present"}, "target.py", 0, step=2)

    _gap(result, events, "INTENT_CURRENT_SOURCE_UNAVAILABLE", "kernel_intent_carry")


def test_planted_live_public_plan_root_gap(monkeypatch, tmp_path):
    journal, events = _journal()
    original = Path.resolve

    def broken_resolve(path, *args, **kwargs):
        if path == tmp_path:
            raise OSError("repository root changed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", broken_resolve)
    result = step_contract._resolve_public_plan_paths(
        (step_contract.EditNTurnPlanSite("owner", "target.py"),), tmp_path,
        journal=journal,
    )

    _gap(result, events, "EDIT_NTURN_PLAN_ROOT_UNAVAILABLE", "kernel_step_contract")

    episode_journal, episode_events = _journal()
    original = Path.resolve

    def broken_episode_root(path, *args, **kwargs):
        if path == tmp_path:
            raise OSError("episode root changed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", broken_episode_root)
    episode = step_contract.start_edit_nturn_episode(
        True, episode_journal, repository_root=tmp_path,
    )
    _gap(episode, episode_events, "EDIT_NTURN_PLAN_ROOT_UNAVAILABLE", "kernel_step_contract")


def test_planted_live_public_plan_site_gap(monkeypatch, tmp_path):
    journal, events = _journal()
    import kernel_tool_paths

    monkeypatch.setattr(kernel_tool_paths, "resolve_tool_path", _raises(OSError("site changed")))
    result = step_contract._resolve_public_plan_paths(
        (step_contract.EditNTurnPlanSite("owner", "target.py"),), tmp_path,
        journal=journal,
    )

    _gap(result, events, "EDIT_NTURN_PLAN_PATH_UNAVAILABLE", "kernel_step_contract")


def test_planted_live_green_binding_gap():
    journal, events = _journal()
    runtime = SimpleNamespace(binding=_raises(OSError("binding changed")), journal=journal)

    result = coherence._binding(runtime, {})

    _gap(result, events, "GREEN_COHERENCE_BINDING_UNAVAILABLE", "kernel_green_coherence")

    status_journal, status_events = _journal()
    status_runtime = SimpleNamespace(
        binding=_raises(OSError("binding changed")), journal=status_journal,
        green_lock=None,
    )
    status_runtime.green_lock = SimpleNamespace(
        _green_now=lambda frame: (
            coherence._binding(status_runtime, frame), False,
        )[1],
    )
    status = coherence.terminal_status(status_runtime, {})
    assert status.green is False
    assert status.public_check_status == "UNAVAILABLE"
    assert status.submission_state == "BLOCKED"
    assert len(status_events) == 1
    assert status_events[0][1]["reason_code"] == "GREEN_COHERENCE_BINDING_UNAVAILABLE"


def test_planted_live_green_lock_evaluation_gap():
    journal, events = _journal()
    lock = SimpleNamespace(_green_now=_raises(OSError("live green check changed")))
    runtime = SimpleNamespace(green_lock=lock, journal=journal)

    result = coherence._lock_green(runtime, {})

    _gap(result, events, "GREEN_LOCK_LIVE_STATUS_UNAVAILABLE", "kernel_green_coherence")


@pytest.mark.parametrize("failure", [KeyError("product_revision"), LookupError("binding lookup")])
def test_planted_live_green_lock_binding_lookup_gap(failure):
    journal, events = _journal()
    runtime = SimpleNamespace(binding=_raises(failure), journal=journal)

    result = green_lock._current_binding(runtime)

    _gap(result, events, "GREEN_LOCK_BINDING_UNAVAILABLE", "kernel_green_lock",
         sub_reason=type(failure).__name__)


@pytest.mark.parametrize("failure", [OSError("private path"), ValueError("private content")])
def test_green_lock_binding_error_records_only_bounded_class(failure):
    journal, events = _journal()
    runtime = SimpleNamespace(binding=_raises(failure), journal=journal)

    result = green_lock._current_binding(runtime)

    _gap(result, events, "GREEN_LOCK_BINDING_UNAVAILABLE", "kernel_green_lock",
         sub_reason=type(failure).__name__)
    assert "private" not in str(events)


def test_green_lock_binding_error_identifies_uninitialized_runtime():
    journal, events = _journal()
    failure = AttributeError("private path", name="_binding_cache")
    runtime = SimpleNamespace(binding=_raises(failure), journal=journal)

    result = green_lock._current_binding(runtime)

    _gap(result, events, "GREEN_LOCK_BINDING_UNAVAILABLE", "kernel_green_lock",
         sub_reason="runtime_not_initialized")
    assert "private" not in str(events)


def test_green_lock_binding_error_bounds_custom_class_name():
    journal, events = _journal()
    error_type = type("B" * 100, (OSError,), {})
    runtime = SimpleNamespace(binding=_raises(error_type("private path")), journal=journal)

    result = green_lock._current_binding(runtime)

    _gap(result, events, "GREEN_LOCK_BINDING_UNAVAILABLE", "kernel_green_lock",
         sub_reason="B" * 64)
    assert "private" not in str(events)


def test_green_lock_fact_never_claims_green_when_binding_is_unavailable():
    journal, events = _journal()
    runtime = SimpleNamespace(binding=_raises(OSError("binding changed")), journal=journal)

    result = green_lock.green_lock_fact(runtime)

    assert "current binding is unavailable" in result
    assert "completion cannot be confirmed" in result
    assert "TERMINAL_GREEN_LOCK" not in result
    assert events[-1][1]["reason_code"] == "GREEN_LOCK_BINDING_UNAVAILABLE"


def test_live_parent_comparison_does_not_fallback_to_cached_revision():
    journal, events = _journal()
    runtime = SimpleNamespace(
        binding=_raises(OSError("binding changed")), journal=journal,
        product_revision="new-revision", parent_revision="old-revision",
        owner=None, green_lock=None,
    )

    result = green_lock.product_revision_change_from_parent(runtime)

    assert result == (
        False, "UNKNOWN",
        "current_product_binding_unavailable:GREEN_LOCK_BINDING_UNAVAILABLE",
    )
    assert events[-1][1]["reason_code"] == "GREEN_LOCK_BINDING_UNAVAILABLE"


def test_runtime_read_already_present_propagates_intent_gap(monkeypatch):
    gap = UnavailableEvidence("INTENT_CURRENT_SOURCE_UNAVAILABLE")
    runtime = object.__new__(stateframe_runtime.StateFrameRuntime)
    runtime.empirical_fixes_mode = "off"
    runtime.stateframe_tokenlab_stall_breaker_mode = "off"
    runtime.intent_carry = object()
    runtime.owner = SimpleNamespace(current_step=1)
    monkeypatch.setattr(intent_carry, "delivered_source_observation", lambda *_args: gap)
    import kernel_read_fixes
    import kernel_host_autopilot

    monkeypatch.setattr(kernel_read_fixes, "read_receipt_enabled", lambda: False)
    monkeypatch.setattr(kernel_host_autopilot, "enabled", lambda: False)

    assert runtime.read_already_present("target.py", 0) is gap


def test_runtime_model_decision_intent_gap_blocks_next_frame(monkeypatch):
    gap = UnavailableEvidence("INTENT_PATH_UNAVAILABLE")

    class Timeline:
        def __len__(self):
            return 0

        def record_decision(self, *_args, **_kwargs):
            return None

    runtime = object.__new__(stateframe_runtime.StateFrameRuntime)
    runtime.intent_carry = SimpleNamespace(observe_model=lambda **_kwargs: gap)
    runtime.owner = SimpleNamespace(current_step=1, fn="change")
    runtime.timeline_index = Timeline()
    runtime._target_relative = lambda: "target.py"
    import kernel_qgates
    monkeypatch.setattr(kernel_qgates, "record_statement", lambda *_args, **_kwargs: None)

    runtime.record_model_decision(content="change target", step=1)

    assert runtime._intent_carry_unavailable is gap
    assert runtime._frame() is gap


def test_unavailable_frame_prepares_terminal_capture_with_replan_identity():
    class Journal:
        task_id = "frame-unavailable"

        def request_records(self, _request_id):
            return []

    journal = Journal()
    runtime = SimpleNamespace(
        journal=journal, owner=SimpleNamespace(current_step=1),
        snapshot_id="snapshot", endpoint_profile=None, system_outcome=None,
    )
    ordinary = stateframe_runtime._prepare_request_capture(runtime)
    runtime._source_replan_origin_request_id = ordinary.request_id

    stopped = stateframe_runtime._unavailable_frame_request(
        runtime, "INTENT_PATH_UNAVAILABLE")

    assert stopped.messages == ()
    assert stopped.overflow == {
        "kind": "REQUEST_BUILDER_UNAVAILABLE",
        "reason_code": "INTENT_PATH_UNAVAILABLE",
    }
    assert stopped.capture.request_id != ordinary.request_id
    assert runtime.system_outcome == "request_builder_unavailable"


def test_get_sure_prediction_entrypoints_render_typed_unavailability(monkeypatch):
    gap = UnavailableEvidence("GET_SURE_RUNTIME_REVISION_UNAVAILABLE")
    monkeypatch.setattr(get_sure, "predict_runtime", lambda *_args: gap)

    assert hstate_plus.prediction(SimpleNamespace(), {}) == {
        "prediction_status": "UNAVAILABLE",
        "reason_code": "GET_SURE_RUNTIME_REVISION_UNAVAILABLE",
    }

    runtime = object.__new__(stateframe_runtime.StateFrameRuntime)
    runtime.stateframe_get_sure_mode = "on"
    assert runtime.get_sure_operation("predict", {}) == {
        "prediction_status": "UNAVAILABLE",
        "reason_code": "GET_SURE_RUNTIME_REVISION_UNAVAILABLE",
    }


def test_request_action_contract_errors_escape_shared_unavailable_boundaries(
        monkeypatch, tmp_path):
    journal, _events = _journal()
    contract_error = RequestActionContractError("contract must remain visible")

    with monkeypatch.context() as patch:
        patch.setattr(source_window, "_path_from_runtime", _raises(contract_error))
        with pytest.raises(RequestActionContractError, match="contract must remain visible"):
            owner_first.build_priority_plan({}, SimpleNamespace(journal=journal))

    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", _raises(contract_error))
        with pytest.raises(RequestActionContractError, match="contract must remain visible"):
            intent_carry.normalize_path("target.py", tmp_path, journal=journal)
        with pytest.raises(RequestActionContractError, match="contract must remain visible"):
            stateframe_runtime._read_requested_source(
                tmp_path / "target.py", tmp_path, {}, journal)
        with pytest.raises(RequestActionContractError, match="contract must remain visible"):
            step_contract._resolve_public_plan_paths(
                (step_contract.EditNTurnPlanSite("owner", "target.py"),), tmp_path,
                journal=journal,
            )

    with pytest.raises(RequestActionContractError, match="contract must remain visible"):
        coherence._binding(SimpleNamespace(binding=_raises(contract_error), journal=journal), {})

    with pytest.raises(RequestActionContractError, match="contract must remain visible"):
        coherence._lock_green(
            SimpleNamespace(green_lock=SimpleNamespace(_green_now=_raises(contract_error)),
                            journal=journal), {})

    with pytest.raises(RequestActionContractError, match="contract must remain visible"):
        green_lock._current_binding(
            SimpleNamespace(binding=_raises(contract_error), journal=journal))

    import kernel_tool_paths
    state = intent_carry.IntentCarryState(journal, root=tmp_path)
    pending = intent_carry.PendingIntent("target.py", "change", "change the function", 1)
    state._pending[pending.intent_id] = pending
    (tmp_path / "target.py").write_text("def change(): return 1\n", encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(kernel_tool_paths, "resolve_tool_path", _raises(contract_error))
        with pytest.raises(RequestActionContractError, match="contract must remain visible"):
            step_contract._resolve_public_plan_paths(
                (step_contract.EditNTurnPlanSite("owner", "target.py"),), tmp_path,
                journal=journal,
            )
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", _raises(contract_error))
        with pytest.raises(RequestActionContractError, match="contract must remain visible"):
            get_sure._runtime_revision(
                SimpleNamespace(target=tmp_path / "target.py", journal=journal))
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", _raises(contract_error))
        with pytest.raises(RequestActionContractError, match="contract must remain visible"):
            state.redirect_already_present(
                {"message": "already present"}, "target.py", 0, step=2)
