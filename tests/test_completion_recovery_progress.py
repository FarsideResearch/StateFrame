"""Completion recovery retries depend on observed progress, not attempt counts."""
from __future__ import annotations

import pytest

import kernel_completion_authority as authority_module


def _authority() -> authority_module.CompletionAuthority:
    return authority_module.CompletionAuthority(
        candidate={"revision": "synthetic-1"},
        available_actions=("edit", "run_tests"),
        budget=authority_module.RootFuelBudget.from_total(8, terminalization_reserve=1),
    )


def _start_recovery() -> authority_module.CompletionAuthority:
    authority = _authority()
    authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic partial defect",
        obligation_id="synthetic-item",
    )
    first = authority.reduce()
    assert first.action is not None and first.action["tool"] == "edit"
    return authority


@pytest.mark.parametrize("progress", ["source_revision", "check_result", "obligation"])
def test_recovery_retries_after_each_kind_of_observed_progress(progress: str) -> None:
    authority = _start_recovery()

    authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})
    if progress == "source_revision":
        authority.set_candidate({"revision": "synthetic-2"}, revision="synthetic-2")
    elif progress == "check_result":
        authority.record_receipt({
            "receipt_id": "synthetic-check", "obligation_id": "synthetic-check",
            "status": "FAIL", "candidate_digest": authority.candidate_digest,
        })
    elif progress == "obligation":
        authority.refuse(
            "PRODUCT_VIOLATION", detail="another synthetic obligation",
            obligation_id="synthetic-item-2",
        )

    second = authority.reduce()

    assert not second.terminal
    assert second.action is not None and second.action["tool"] == "edit"
    dispatches = [row for row in authority.history if row["kind"] == "action_dispatched"]
    assert len(dispatches) == 2
    assert all(row.get("progress_digest") for row in dispatches)
    assert not hasattr(authority.registry.require("PRODUCT_VIOLATION"), "attempt_limit")


def test_completion_authority_allows_repeated_partial_recovery() -> None:
    authority = _start_recovery()
    authority.record_action_result({"action": "edit", "status": "PARTIAL"})

    second = authority.reduce()

    assert not second.terminal
    assert second.action is not None and second.action["tool"] == "edit"
    dispatches = [row for row in authority.history if row["kind"] == "action_dispatched"]
    assert [row["attempt"] for row in dispatches] == [1, 2]
    assert all(row.get("progress_digest") for row in dispatches)


def test_partial_recovery_progress_survives_resume_without_changing_render() -> None:
    authority = _start_recovery()
    authority.record_action_result({"action": "edit", "status": "PARTIAL"})
    saved = authority.snapshot()

    resumed = authority_module.CompletionAuthority.from_snapshot(
        saved, available_actions=("edit", "run_tests"),
    )
    next_decision = resumed.reduce()

    assert not next_decision.terminal
    assert next_decision.action is not None and next_decision.action["tool"] == "edit"
    assert "recovery_progress" not in resumed.render()
    assert len([row for row in resumed.history if row["kind"] == "action_dispatched"]) == 2


def test_200_identical_no_change_observations_prove_no_progress_once() -> None:
    authority = _start_recovery()
    authority.record_action_result({"action": "edit", "status": "PARTIAL"})
    second = authority.reduce()
    assert second.action is not None and second.action["tool"] == "edit"
    result = {"action": "edit", "status": "NO_CHANGE"}

    decisions = []
    for _ in range(200):
        authority.record_action_result(result)
        decisions.append(authority.reduce())

    assert all(decision.terminal for decision in decisions)
    assert all(decision == decisions[0] for decision in decisions)
    assert decisions[0].reason_codes == ("NO_FEASIBLE_RECOVERY",)
    assert "no progress" in decisions[0].detail.lower()
    proofs = [row for row in authority.history if row["kind"] == "recovery_no_progress"]
    assert len(proofs) == 1
    assert proofs[0]["previous_digest"] == proofs[0]["observed_digest"]
    assert proofs[0]["unchanged"] is True
    assert len([row for row in authority.history if row["kind"] == "action_dispatched"]) == 2
    assert len([row for row in authority.history if row["kind"] == "terminalized"]) == 1


def test_accepted_edit_is_progress_when_another_obligation_remains() -> None:
    authority = _start_recovery()
    authority.refuse(
        "REQUIRED_EVIDENCE_GAP", detail="synthetic check required",
        obligation_id="synthetic-check",
    )
    authority.record_action_result({"action": "edit", "status": "APPLIED"})

    decision = authority.reduce()

    assert not decision.terminal
    assert decision.action is not None and decision.action["tool"] == "run_tests"
