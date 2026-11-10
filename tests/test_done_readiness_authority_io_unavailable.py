"""Live done decisions leave a journaled gap when required evidence cannot be read."""
from __future__ import annotations

from types import SimpleNamespace

import kernel_completion_owner
import kernel_done_readiness_authority as authority
import pytest
from kernel_obligation_assess import UnavailableEvidence


DONE_TOOLS = ({"name": "done"},)


def _runtime(binding=None, scope=None):
    events = []
    journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    runtime = SimpleNamespace(
        stateframe_readiness_single_authority_mode="on",
        readiness_decision=lambda *_args: None,
        pass_gates_vector=lambda *_args: None,
        binding=binding or (lambda: {"product_revision": "revision-1"}),
        _catalog_task_scope=scope or (lambda: ("task-1", "task")),
        journal=journal,
        current_step=1,
    )
    readiness = SimpleNamespace(candidate_digest="check-1", public_check_status="PASS",
                                green=True, revision="revision-1")
    vector = SimpleNamespace(artifact_identity="revision-1", evidence_identity="gate-1")
    return runtime, readiness, vector, events


def _assert_gap(result, runtime, events, code):
    assert result is None
    assert authority.cached_decision(runtime) is None
    assert events == [("host_obligation_evidence_failure", {
        "producer": "kernel_done_readiness_authority",
        "dispatch": False,
        "reason_code": code,
    })]


def _stable_predecision(monkeypatch):
    monkeypatch.setattr(authority, "_blocker_rows", lambda *_args: [])
    monkeypatch.setattr(authority, "_assessment_rows", lambda *_args: ())


def test_planted_live_scope_io_emits_event(monkeypatch):
    def broken_scope():
        raise OSError("planted scope read failure")

    runtime, readiness, vector, events = _runtime(scope=broken_scope)
    _stable_predecision(monkeypatch)
    monkeypatch.setattr(authority, "_open_ids", lambda *_args: ())
    _assert_gap(authority.decision_for_runtime(runtime, readiness, vector=vector,
                                               full_tools=DONE_TOOLS), runtime, events,
                "TASK_SCOPE_UNAVAILABLE")
    # Every consumer of this step sees the same absent shared decision.
    assert authority.decision_for_runtime(runtime, readiness, vector=vector,
                                          full_tools=DONE_TOOLS) is None
    assert len(events) == 1


def test_planted_live_open_ids_io_emits_event(monkeypatch):
    def broken_open_ids(_runtime):
        raise OSError("planted obligation read failure")

    runtime, readiness, vector, events = _runtime()
    _stable_predecision(monkeypatch)
    monkeypatch.setattr(kernel_completion_owner, "open_ids", broken_open_ids)
    _assert_gap(authority.decision_for_runtime(runtime, readiness, vector=vector,
                                               full_tools=DONE_TOOLS), runtime, events,
                "OPEN_IDS_UNAVAILABLE")


def test_open_ids_io_returns_the_shared_frozen_gap(monkeypatch):
    def broken_open_ids(_runtime):
        raise OSError("planted obligation read failure")

    runtime, _readiness, _vector, events = _runtime()
    monkeypatch.setattr(kernel_completion_owner, "open_ids", broken_open_ids)
    gap = authority._open_ids(runtime)
    assert isinstance(gap, UnavailableEvidence)
    assert gap.reason_code == "OPEN_IDS_UNAVAILABLE"
    assert events[0][1]["reason_code"] == gap.reason_code


def test_planted_d2_off_host_debt_open_ids_io_emits_event(monkeypatch):
    def broken_open_ids(_runtime):
        raise OSError("planted obligation read failure")

    runtime, readiness, vector, events = _runtime()
    runtime.stateframe_readiness_single_authority_mode = "off"
    _stable_predecision(monkeypatch)
    monkeypatch.setattr(authority, "_candidate_changed_from_parent", lambda *_args: True)
    monkeypatch.setattr(authority, "_public_check_current_for_product", lambda *_args: True)
    monkeypatch.setattr(authority, "_hard_host_blockers", lambda *_args: ())
    monkeypatch.setattr(kernel_completion_owner, "open_ids", broken_open_ids)
    _assert_gap(authority.decision_for_runtime(runtime, readiness, vector=vector,
                                               full_tools=DONE_TOOLS), runtime, events,
                "OPEN_IDS_UNAVAILABLE")


def test_planted_live_decision_binding_io_emits_event(monkeypatch):
    def broken_binding():
        raise OSError("planted decision binding failure")

    runtime, readiness, vector, events = _runtime(binding=broken_binding)
    _stable_predecision(monkeypatch)
    monkeypatch.setattr(authority, "_open_ids", lambda *_args: ())
    monkeypatch.setattr(authority, "_hard_host_blockers", lambda *_args: ())
    _assert_gap(authority.decision_for_runtime(runtime, readiness, vector=vector,
                                               full_tools=DONE_TOOLS), runtime, events,
                "DECISION_BINDING_UNAVAILABLE")


def test_planted_live_artifact_binding_io_emits_event(monkeypatch):
    def broken_binding():
        raise OSError("planted artifact binding failure")

    runtime, readiness, vector, events = _runtime(binding=broken_binding)
    vector.artifact_identity = ""
    _stable_predecision(monkeypatch)
    _assert_gap(authority.decision_for_runtime(runtime, readiness, vector=vector,
                                               full_tools=DONE_TOOLS), runtime, events,
                "ARTIFACT_BINDING_UNAVAILABLE")


@pytest.mark.parametrize("source", ["scope", "open_ids", "decision_binding", "artifact_binding"])
def test_unexpected_live_error_propagates(monkeypatch, source):
    def broken(*_args):
        raise LookupError("unexpected live failure")

    runtime, readiness, vector, events = _runtime()
    _stable_predecision(monkeypatch)
    monkeypatch.setattr(authority, "_hard_host_blockers", lambda *_args: ())
    if source == "scope":
        runtime._catalog_task_scope = broken
        monkeypatch.setattr(authority, "_open_ids", lambda *_args: ())
    elif source == "open_ids":
        monkeypatch.setattr(kernel_completion_owner, "open_ids", broken)
    else:
        runtime.binding = broken
        monkeypatch.setattr(authority, "_open_ids", lambda *_args: ())
        if source == "artifact_binding":
            vector.artifact_identity = ""
    with pytest.raises(LookupError, match="unexpected live failure"):
        authority.decision_for_runtime(runtime, readiness, vector=vector,
                                       full_tools=DONE_TOOLS)
    assert events == []
