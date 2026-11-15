"""Snapshot failures refuse fresh debt; preparation suspends with typed evidence.

Refusal lasts only for the current batch. S4/A6: a SnapshotFailure in prepare_live/freeze_shadow is a durable,
typed non-floor suspension, never RequestSealError, injected through the real
qualification boundary (build_qualified_request).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import kernel_read_admission as admission
import kernel_working_set_live as live
import kernel_working_set_shadow as shadow
import kernel_working_set_state as ws
import kernel_working_set_stop as stop
from kernel_working_set_snapshot import SnapshotConstructionError

from test_read_admission_qualification import seal_fixture

SNAPSHOT_REASONS = ("ACTIVE_OBLIGATION_LIMIT", "SOURCE_SNAPSHOT_UNSTABLE",
                    "CAPTURE_BLOB_UNAVAILABLE")


@pytest.fixture(autouse=True)
def on(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")


def _failing_freeze(reason):
    def freeze(body, runtime, **kwargs):
        raise SnapshotConstructionError(reason)
    return freeze


@pytest.mark.parametrize("reason", SNAPSHOT_REASONS)
def test_snapshot_failure_at_read_boundary_refuses_unpriced_debt(monkeypatch, reason):
    """An estimate cannot authorize FULL-only pages after a failed freeze."""
    freeze = shadow.freeze_preparation
    monkeypatch.setattr(shadow, "freeze_preparation", _failing_freeze(reason))
    events = []
    journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    rt = SimpleNamespace(_frame_budget_request_count=7, native_batch=None, journal=journal)
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "Task: read"},
                {"role": "user", "content": '[StateFrame]\n{"state":{"framefit5":{}}}'},
                {"role": "user", "content": "[WIRE VOLATILE TAIL]"}]
    state = admission._state(rt, messages)
    assert state.snapshot is None
    assert state.closed is True
    assert state.allowance == 0
    assert state.degraded_reason == reason
    assert admission.remaining_allowance(rt, messages) == state.allowance
    degraded = [f for k, f in events if k == admission.EVENT and f["decision"] == "DEGRADED"]
    assert degraded and degraded[0]["reason"] == reason
    # The next request index starts clean: the failure does not stick.
    monkeypatch.setattr(shadow, "freeze_preparation", freeze)
    rt._frame_budget_request_count = 8
    fresh = admission._state(rt, messages)
    assert fresh is not state and fresh.closed is False and fresh.allowance > 0


def test_read_admission_snapshot_failure_is_typed(monkeypatch, tmp_path):
    """The boundary raises a typed, reasoned error so callers never guess."""
    monkeypatch.setattr(shadow, "freeze_preparation", _failing_freeze("SOURCE_SNAPSHOT_UNSTABLE"))
    rt = SimpleNamespace(owner=SimpleNamespace(root=tmp_path), journal=None, native_batch=None)
    messages = [{"role": "user", "content": "[StateFrame]\n{}"}]
    with pytest.raises(live.RequestSealError) as caught:
        live.read_admission_snapshot(rt, messages, byte_cap=10_000)
    assert caught.value.reason == "SOURCE_SNAPSHOT_UNSTABLE"
    assert isinstance(caught.value, live.RequestSealError)


@pytest.mark.parametrize("reason", SNAPSHOT_REASONS)
def test_snapshot_failure_returns_typed_refusals_and_cursors(tmp_path, monkeypatch, reason):
    from test_read_receipts_admission import read

    capture, rt, body, plan, journal, tools = seal_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(shadow, "freeze_preparation", _failing_freeze(reason))
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("value = 1\n")
        (status, text, _), metadata = read(rt, tmp_path, name)
        assert status == admission.REFUSED_ADMISSION
        assert metadata["effective_range"] is None
        assert metadata["next_cursor"] in text
    state = admission._state(rt, body["messages"])
    assert state.allowance == 0
    assert state.admitted_count == 0 and state.pending_bytes == 0
    assert state.refused_count == 2
    assert admission.remaining_allowance(rt, None) == 0
    assert state.degraded_reason == reason and state.closed


def test_snapshot_failure_respects_real_capacity_exhaustion(monkeypatch):
    monkeypatch.setattr(shadow, "freeze_preparation", _failing_freeze("ACTIVE_OBLIGATION_LIMIT"))
    rt = SimpleNamespace(working_set_fixed_bytes=admission.request_byte_cap())
    state = admission._state(rt, [{"role": "user", "content": "[StateFrame]\n{}"}])
    assert state.degraded_reason == "ACTIVE_OBLIGATION_LIMIT"
    assert state.allowance == 0 and state.closed


def test_in_batch_edit_with_stale_visible_source_refuses_until_refrozen(tmp_path):
    """Scout r3: freeze the real stale open-site authority without fault injection."""
    from kernel_request_source_window import source_revision
    from test_read_receipts_admission import read, runtime

    path = tmp_path / "a.py"
    path.write_text("old\n")
    messages = [{"role": "user", "content": "[StateFrame]\n" + json.dumps({
        "state": {"framefit5": {"open_sites": [{"path": "a.py",
            "revision": source_revision("old\n"), "lines": {"start": 1, "end": 1},
            "current_bytes": "old\n"}]}}})}]
    rt = runtime(tmp_path)
    rt.owner = SimpleNamespace(root=tmp_path)
    before = live.read_admission_snapshot(rt, messages, byte_cap=admission.request_byte_cap())
    assert before is not None
    path.write_text("new\n")
    state = admission._state(rt, messages)
    assert state.closed and state.degraded_reason == "SOURCE_SNAPSHOT_UNSTABLE"
    with pytest.raises(live.RequestSealError) as caught:
        live.read_admission_snapshot(rt, messages, byte_cap=admission.request_byte_cap())
    assert caught.value.reason == "SOURCE_SNAPSHOT_UNSTABLE"
    (status, text, _), metadata = read(rt, tmp_path, "a.py")
    assert status == admission.REFUSED_ADMISSION and metadata["next_cursor"] in text
    assert metadata["revision"] == source_revision("new\n")
    assert state.pending_bytes == 0


def test_untyped_admission_error_has_a_contract_reason(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("private source text must not appear in diagnostics")
    monkeypatch.setattr(live, "read_admission_snapshot", fail)
    state = admission._state(SimpleNamespace(), None)
    assert state.degraded_reason == "PREPARATION_CONTRACT_FAILURE"
    assert state.allowance == 0 and state.closed


@pytest.mark.parametrize("reason", SNAPSHOT_REASONS)
def test_prepare_snapshot_failure_is_a_typed_suspension(tmp_path, monkeypatch, reason):
    """S4/A6: freeze_shadow failure through build_qualified_request -> durable BLOCKED."""
    from kernel_final_wire_allocation import build_qualified_request
    from kernel_request_seal import RequestSealError

    capture, rt, body, plan, journal, tools = seal_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(shadow, "freeze_preparation", _failing_freeze(reason))
    try:
        with pytest.raises(stop.WorkingSetSuspended) as caught:
            build_qualified_request(capture, rt, body, endpoint="http://local/v1", key="")
    except RequestSealError as error:  # pragma: no cover - the red path on c1c6ad495
        pytest.fail(f"snapshot failure escaped as RequestSealError: {error}")
    suspension = caught.value.suspension
    assert suspension.reason_code == stop.ReasonCode(reason)
    assert suspension.proof is None and suspension.minimum_policy_candidate_ref is None
    assert suspension.violated_limits == ()
    assert suspension.logical_request_id == capture.request_id
    assert "SnapshotConstructionError" in suspension.detail
    outcome = caught.value.outcome()
    assert outcome["result"] == "BLOCKED" and outcome["resumable"] is True
    assert stop.is_suspended_result(outcome)
    # Durable: the blocked event is journaled and the checkpoint carries it.
    rows = [json.loads(line) for line in journal.path.read_bytes().splitlines()]
    blocked = [row for row in rows if row["kind"] == "working_set_blocked"]
    assert blocked and blocked[-1]["suspension"]["reason_code"] == reason
    assert rt.working_set_state.suspension.document() == suspension.document()
    assert ws.reconstruct(rows).checkpoint() == rt.working_set_state.checkpoint()
    restored = ws.WorkingSetState.from_checkpoint(rt.working_set_state.checkpoint())
    assert restored.suspension.document() == suspension.document()
    ws.retry_unsent_boundary(restored)
    assert restored.suspension is None
    assert restored.active_obligations() == rt.working_set_state.active_obligations()
    assert getattr(capture, "qualified_request", None) is None


def test_prepare_contract_failure_is_a_typed_suspension(tmp_path, monkeypatch):
    """A plain ValueError from freeze_preparation is PREPARATION_CONTRACT_FAILURE, not ERR."""
    from kernel_final_wire_allocation import build_qualified_request

    capture, rt, body, plan, journal, tools = seal_fixture(tmp_path, monkeypatch)
    def freeze(body, runtime, **kwargs):
        raise ValueError("source window differs from current revision observation")
    monkeypatch.setattr(shadow, "freeze_preparation", freeze)
    with pytest.raises(stop.WorkingSetSuspended) as caught:
        build_qualified_request(capture, rt, body, endpoint="http://local/v1", key="")
    assert caught.value.suspension.reason_code is stop.ReasonCode.PREPARATION_CONTRACT_FAILURE
    assert caught.value.suspension.proof is None


@pytest.mark.parametrize("failure", [
    None,
    shadow.SnapshotFailure("unused", "ValueError", "UNKNOWN"),
    shadow.SnapshotFailure("unused", "ValueError", "FLOOR_INFEASIBLE"),
])
def test_invalid_snapshot_failure_cannot_claim_a_floor(tmp_path, monkeypatch, failure):
    from kernel_final_wire_allocation import build_qualified_request

    capture, rt, body, plan, journal, tools = seal_fixture(tmp_path, monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(shadow, "freeze_shadow", lambda *a, **k: failure)
        with pytest.raises(stop.WorkingSetSuspended) as caught:
            build_qualified_request(capture, rt, body, endpoint="http://local/v1", key="")
        with pytest.raises(live.RequestSealError) as unavailable:
            live.read_admission_snapshot(rt, body["messages"], byte_cap=12_000)
    assert caught.value.suspension.reason_code is stop.ReasonCode.PREPARATION_CONTRACT_FAILURE
    assert caught.value.suspension.proof is None
    assert unavailable.value.reason == "PREPARATION_CONTRACT_FAILURE"
    # Repairing the failure restores exact admission at a fresh read boundary.
    ws.retry_unsent_boundary(rt.working_set_state)
    rt._frame_budget_request_count += 1
    fresh = admission._state(rt, body["messages"])
    assert fresh.snapshot is not None and fresh.degraded_reason is None
    assert fresh.allowance > 0 and not fresh.closed


def test_off_request_bytes_match_carried_3b(tmp_path, tmp_path_factory):
    from test_working_set_off_parity_base import test_off_provider_bytes_match_cutover_base

    test_off_provider_bytes_match_cutover_base(tmp_path, tmp_path_factory, "1f6b7e9bc1")
