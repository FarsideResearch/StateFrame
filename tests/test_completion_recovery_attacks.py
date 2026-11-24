"""Adversarial attacks on semantic completion recovery."""
from __future__ import annotations

import pytest

import kernel_completion_authority as authority_module


def _authority(
    reason: str = "PRODUCT_VIOLATION", budget=None,
) -> authority_module.CompletionAuthority:
    authority = authority_module.CompletionAuthority(
        candidate={"revision": "state-0"},
        available_actions=("edit", "run_tests"),
        budget=budget or authority_module.RootFuelBudget(unbounded=True),
    )
    authority.refuse(reason, detail="synthetic unresolved condition", obligation_id="item-a")
    return authority


def _dispatched(authority, tool: str):
    decision = authority.reduce()
    assert decision.action is not None
    assert decision.action["tool"] == tool
    return decision


@pytest.mark.parametrize("cycle_length", [2, 3, 5])
def test_candidate_cycle_stops_when_any_prior_state_repeats(cycle_length: int) -> None:
    authority = _authority()
    _dispatched(authority, "edit")
    decision = None

    for index in range(1, cycle_length * 3 + 1):
        if authority.terminal:
            break
        authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})
        authority.set_candidate({"revision": f"state-{index % cycle_length}"})
        decision = authority.reduce()

    assert (decision is not None and decision.terminal) or authority.terminal


def test_result_receipt_ids_timestamps_and_durations_are_not_progress() -> None:
    authority = _authority()
    _dispatched(authority, "edit")
    authority.record_action_result({
        "action": "edit", "status": "PARTIAL", "receipt_id": "r-1",
        "timestamp": 1, "duration_ms": 4,
    })
    _dispatched(authority, "edit")

    authority.record_action_result({
        "action": "edit", "status": "PARTIAL", "receipt_id": "r-2",
        "timestamp": 2, "duration_ms": 99,
    })
    stopped = authority.reduce()

    assert stopped.terminal
    assert stopped.reason_codes == ("NO_FEASIBLE_RECOVERY",)


def test_new_candidate_revision_allows_recovery_retry() -> None:
    authority = _authority()
    _dispatched(authority, "edit")
    authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})
    authority.set_candidate({"revision": "state-1"})

    _dispatched(authority, "edit")


def test_shrinking_failed_test_set_allows_recovery_retry() -> None:
    authority = _authority("REQUIRED_EVIDENCE_GAP")
    _dispatched(authority, "run_tests")
    authority.record_action_result({
        "action": "run_tests", "status": "FAIL", "receipt": {
            "receipt_id": "check-1", "obligation_id": "item-a", "status": "FAIL",
            "candidate_digest": authority.candidate_digest, "evidence": ["case-a", "case-b"],
        },
    })
    _dispatched(authority, "run_tests")
    authority.record_action_result({
        "action": "run_tests", "status": "FAIL", "receipt": {
            "receipt_id": "check-2", "obligation_id": "item-a", "status": "FAIL",
            "candidate_digest": authority.candidate_digest, "evidence": ["case-b"],
        },
    })

    _dispatched(authority, "run_tests")


def test_growing_failed_test_diagnostics_do_not_count_as_progress() -> None:
    authority = _authority("REQUIRED_EVIDENCE_GAP")
    _dispatched(authority, "run_tests")
    authority.record_action_result({
        "action": "run_tests", "status": "FAIL", "receipt": {
            "receipt_id": "check-1", "obligation_id": "item-a", "status": "FAIL",
            "candidate_digest": authority.candidate_digest,
            "evidence": ["case-a", "case-b"],
        },
    })
    _dispatched(authority, "run_tests")
    authority.record_action_result({
        "action": "run_tests", "status": "FAIL", "receipt": {
            "receipt_id": "check-2", "obligation_id": "item-a", "status": "FAIL",
            "candidate_digest": authority.candidate_digest,
            "evidence": ["case-a", "case-b", "volatile diagnostic"],
        },
    })

    stopped = authority.reduce()

    assert stopped.terminal
    assert stopped.reason_codes == ("NO_FEASIBLE_RECOVERY",)


def test_edit_can_resolve_one_obligation_while_another_remains() -> None:
    authority = _authority()
    authority.refuse("REQUIRED_EVIDENCE_GAP", detail="synthetic check", obligation_id="item-b")
    _dispatched(authority, "edit")
    authority.record_action_result({"action": "edit", "status": "APPLIED"})

    _dispatched(authority, "run_tests")


def test_progress_from_an_interleaved_reason_does_not_reset_this_reason() -> None:
    authority = _authority()
    authority.refuse("REQUIRED_EVIDENCE_GAP", detail="synthetic check", obligation_id="item-b")
    assert authority.dispatch_action("edit", reason_code="PRODUCT_VIOLATION").action is not None
    authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})
    assert authority.dispatch_action("run_tests", reason_code="REQUIRED_EVIDENCE_GAP").action is not None
    authority.record_action_result({"action": "run_tests", "status": "PARTIAL"})

    stopped = authority.dispatch_action("edit", reason_code="PRODUCT_VIOLATION")

    assert stopped.terminal
    assert stopped.reason_codes == ("NO_FEASIBLE_RECOVERY",)


def test_unresolved_obligation_list_growth_is_not_progress() -> None:
    authority = _authority()
    _dispatched(authority, "edit")

    decision = None
    for size in range(1, 12):
        authority.record_action_result({
            "action": "edit", "status": "NO_CHANGE",
            "obligations": ["item-a"] * size,
        })
        decision = authority.reduce()
        if decision.terminal:
            break

    assert decision is not None and decision.terminal


def test_newly_met_obligation_in_result_allows_recovery_retry() -> None:
    authority = _authority()
    _dispatched(authority, "edit")
    authority.record_action_result({
        "action": "edit", "status": "NO_CHANGE",
        "obligations": [{"obligation_id": "item-b", "status": "MET"}],
    })

    _dispatched(authority, "edit")


def test_cycle_detection_survives_checkpoint_and_process_restore(tmp_path) -> None:
    authority = _authority()
    _dispatched(authority, "edit")
    resumed = authority_module.CompletionAuthority.from_snapshot(
        authority.snapshot(), available_actions=("edit", "run_tests"),
    )
    decision = None
    for index in range(1, 7):
        if resumed.terminal:
            break
        resumed.record_action_result({"action": "edit", "status": "NO_CHANGE"})
        resumed.set_candidate({"revision": f"state-{index % 2}"})
        decision = resumed.reduce()

    assert (decision is not None and decision.terminal) or resumed.terminal


def test_release_snapshot_without_new_progress_fields_preserves_recovery_frame(tmp_path) -> None:
    authority = _authority()
    _dispatched(authority, "edit")
    legacy_snapshot = authority.snapshot()
    for key in (
        "recovery_progress", "recovery_progress_seen", "action_result_progress",
        "action_result_progress_by_reason", "blocker_state", "history_dropped",
    ):
        legacy_snapshot.pop(key, None)

    resumed = authority_module.CompletionAuthority.from_snapshot(
        legacy_snapshot, available_actions=("edit", "run_tests"),
    )

    assert resumed.render() == authority.render()
    assert resumed.reduce().action is None


def test_legacy_dispatch_proof_in_history_still_stops_after_resume() -> None:
    authority = _authority()
    _dispatched(authority, "edit")
    legacy_digest = authority_module._digest({
        "candidate_digest": authority.candidate_digest,
        "current_revision": authority.current_revision,
        "policy_digest": authority.policy_digest,
        "required_evidence": sorted(authority._required_evidence),
        "recheck_required": sorted(authority._recheck_required),
        "blockers": [{
            "kind": type(cause).__name__, "reason_code": cause.reason_code,
            "obligation_id": cause.obligation_id, "detail": cause.detail,
            "candidate_digest": cause.candidate_digest,
        } for _, cause in sorted(authority._blockers.items())
            if cause.reason_code == "PRODUCT_VIOLATION"],
        "receipts": sorted({authority_module._digest({
            "obligation_id": item.obligation_id, "status": item.status,
            "candidate_digest": item.candidate_digest, "revision": item.revision,
            "stale": item.stale, "evidence": item.evidence, "identity": item.identity,
        }) for item in authority._receipts}),
        "current_evidence": {},
        "last_action_result_progress": None,
    })
    saved = authority.snapshot()
    for key in ("recovery_progress", "recovery_progress_seen", "recovery_progress_delta"):
        saved.pop(key, None)
    saved["history"] = [{
        "kind": "action_dispatched", "reason_code": "PRODUCT_VIOLATION",
        "progress_digest": legacy_digest,
    }]
    resumed = authority_module.CompletionAuthority.from_snapshot(
        saved, available_actions=("edit", "run_tests"),
    )
    resumed.record_action_result({"action": "edit", "status": "NO_CHANGE"})

    stopped = resumed.reduce()

    assert stopped.terminal
    assert stopped.reason_codes == ("NO_FEASIBLE_RECOVERY",)


def test_fork_retains_progress_proof_without_spending_parent_budget() -> None:
    authority = _authority(budget=authority_module.RootFuelBudget.from_total(4))
    _dispatched(authority, "edit")
    parent_remaining = authority.fuel.total_remaining

    child = authority.fork()
    child.record_action_result({"action": "edit", "status": "NO_CHANGE"})
    decision = child.reduce()

    assert decision.terminal
    assert authority.fuel.total_remaining == parent_remaining
    assert not authority.terminal


def test_recovery_and_audit_memory_stay_bounded_for_ten_thousand_attempts(tmp_path) -> None:
    authority = _authority()
    _dispatched(authority, "edit")

    for index in range(10_000):
        if authority.terminal:
            break
        authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})
        authority.set_candidate({"revision": f"state-{index + 1}"})
        decision = authority.reduce()
        if decision.terminal:
            break

    assert len(authority.history) <= 4096
    assert len(authority._seen_recovery_progress["PRODUCT_VIOLATION"]) == 10_001
