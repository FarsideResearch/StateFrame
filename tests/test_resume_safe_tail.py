"""Regression coverage for the real StateFrame census-after-checkpoint tail."""

from __future__ import annotations

import kernel_continuation_restart as restart


def test_census_events_from_killed_worker_are_admitted_for_resume() -> None:
    rejection_reason = getattr(restart, "resume_event_rejection_reason", None)
    assert callable(rejection_reason), "resume must expose its event-kind decision"

    # These are the event kinds recorded after step 13 in the real SIGKILL repro.
    for kind in (
        "callsite_census_outcome",
        "callsite_census_prepared",
        "callsite_census_wire_outcome",
        "callsite_census_fired",
        "callsite_census_shadow",
        "callsite_census_incremental_shadow",
    ):
        assert rejection_reason(kind) is None, kind


def test_real_repro_neighbor_has_a_named_reconciliation_reason() -> None:
    rejection_reason = getattr(restart, "resume_event_rejection_reason", None)
    assert callable(rejection_reason), "resume must expose its event-kind decision"

    reason = rejection_reason("host_check_bundle_source")
    assert isinstance(reason, str)
    assert "host_check_bundle_source" in reason
    assert "checkpoint" in reason.lower()


def test_capacity_event_tails_are_accepted_or_refused_by_resume_policy() -> None:
    for kind, (classification, _reason) in restart.CAPACITY_EVENT_CLASSIFICATION.items():
        rejection = restart.resume_tail_rejection_reason([{"kind": kind}])
        if classification == "SAFE_TAIL":
            assert rejection is None, kind
        else:
            assert isinstance(rejection, str), kind
            assert kind in rejection, kind


def test_evidence_failure_tails_follow_resume_policy() -> None:
    classifications = {
        "completion_owner_unavailable": "REJECT",
        "host_obligation_evidence_failure": "REJECT",
        "unified_stop_evidence_failure": "REJECT",
        "wire_truth_fact_evidence_failure": "REJECT",
    }
    for kind, classification in classifications.items():
        if kind in restart.SAFE_TAIL:
            actual = "SAFE_TAIL"
        elif kind in restart.DOCUMENTED_REJECT:
            actual = "REJECT"
        else:
            actual = "UNCLASSIFIED"
        assert actual == classification, kind
        rejection = restart.resume_tail_rejection_reason([{"kind": kind}])
        if actual == "SAFE_TAIL":
            assert rejection is None, kind
        else:
            assert actual == "REJECT", kind
            assert isinstance(rejection, str), kind
            assert kind in rejection, kind
            assert restart.DOCUMENTED_REJECT[kind] in rejection, kind
