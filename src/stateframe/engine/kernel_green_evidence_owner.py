"""Own evidence of passing checks for the current run.

Evidence is bound to the revision and selectors it verified. A later edit must
not inherit a passing result from an incompatible source state.
"""
from __future__ import annotations

import os

STALE_CERTIFICATION_REASON = "required test is missing, stale, red or newer than certification"
EXHAUSTED_REASON = (
    "green evidence owner: refusal bound exhausted for '" + STALE_CERTIFICATION_REASON
    + "' -- run_tests must settle fresh evidence before done is resubmitted")

# Cap on consecutive identical stale-certification refusals before we stop
# repeating the same rejection and latch a terminal, differently worded one.
_REFUSAL_BOUND = 3

_refusal_count = 0
_exhausted = False


def _active_emit(kind: str, **fields: object) -> None:
    """Journal an own event on the active runtime (fired proof).  Never raises."""
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
        journal = getattr(runtime, "journal", None)
    except (ImportError, LookupError):
        return
    emit = getattr(journal, "emit", None)
    if callable(emit):
        try:
            emit(kind, dispatch=False, **fields)
        except Exception as error:  # noqa: BLE001 - telemetry failure is explicitly advisory
            from kernel_harden8 import record_failure
            record_failure(
                runtime,
                "green_evidence_owner.telemetry",
                error,
                authority=False,
                disposition="degrade",
            )


def enabled() -> bool:
    return os.environ.get("STATEFRAME_GREEN_EVIDENCE_OWNER", "off").strip().lower() == "on"


def record_refusal(reason: str) -> int:
    """Record one continuation refusal of ``done``.  Returns the new
    consecutive count when ``reason`` is the stale-certification reason and
    the flag is on; returns 0 (no-op) otherwise, including when the flag is
    off, so a caller cannot accidentally build up state it will never read.
    """
    global _refusal_count, _exhausted
    if not enabled() or reason != STALE_CERTIFICATION_REASON:
        return 0
    _refusal_count += 1
    if _refusal_count >= _REFUSAL_BOUND:
        _exhausted = True
    _active_emit("green_evidence_owner_acted", producer="kernel_green_evidence_owner", stage="acted",
                 refusal_count=_refusal_count, exhausted=_exhausted)
    return _refusal_count


def record_fresh_test() -> None:
    """A new, current ``run_tests`` effect settled as applied: the evidence
    the stale-certification refusals were about is no longer stale, so clear
    the count and any latched exhaustion.  Safe to call with the flag off.
    """
    global _refusal_count, _exhausted
    _refusal_count = 0
    _exhausted = False


def is_exhausted() -> bool:
    return enabled() and _exhausted


def should_allow_run_tests() -> bool:
    """True once at least one stale-certification refusal has been recorded
    and the bound has not yet tripped.  Readiness consults this so it does
    not deny the one repair action that can resolve the disagreement.
    """
    return enabled() and 0 < _refusal_count < _REFUSAL_BOUND


def refusal_count() -> int:
    return _refusal_count


def reset() -> None:
    """Test hook: clear owner state between runs/cases."""
    global _refusal_count, _exhausted
    _refusal_count = 0
    _exhausted = False



# ---------------------------------------------------------------------------
# R4' (r46b): one completion-evidence predicate shared by the continuation
# commit gate (``HostEvidence._public_completion``) and the preflight
# (``ContinuationRuntime.completion_finding``).  Only consulted when the flag
# is on; flag-off callers keep their original inline check byte-for-byte.
#
# J02 (r44d 370 ldev_02, journal seqs 2939/2950/2957/2976/3202/3221): a
# post-green ``str_replace`` was refused by the green lock and settled
# ``not_applied`` with ``binding_changed: false``; the host's auto-verification
# then ran the required test under that same effect id and it came back green
# at the unchanged binding.  The historical predicate demanded
# ``effect.status == "applied"`` and so refused ``done`` 36x although nothing
# about the evidence was stale.  Here a settled ``not_applied`` WRITE effect is
# accepted as the carrier of a green test only when the host's own closed
# record proves the workspace did not move (before == after == current, no
# changes) and the test ran after the effect's intent.  Every other fence
# (ancestry, dispatch ref, producer, exit code, stable binding, order vs the
# certification) is unchanged; pending/unknown effects are never accepted.
# ---------------------------------------------------------------------------

EVIDENCE_MISSING = "EVIDENCE_MISSING"
FOREIGN_INTENTION = "FOREIGN_INTENTION"
EFFECT_MISSING = "EFFECT_MISSING"
EFFECT_UNRESOLVED = "EFFECT_UNRESOLVED"
EFFECT_NOT_APPLIED = "EFFECT_NOT_APPLIED"
EFFECT_LINEAGE_MISMATCH = "EFFECT_LINEAGE_MISMATCH"
NOT_TEST_SUBPROCESS = "NOT_TEST_SUBPROCESS"
TEST_RED = "TEST_RED"
BINDING_UNSTABLE = "BINDING_UNSTABLE"
BINDING_STALE = "BINDING_STALE"
NEWER_THAN_CERTIFICATION = "NEWER_THAN_CERTIFICATION"


def _resolve_test(evidence, reference):
    test = evidence.event_for_receipt(reference)
    return test or {}


def _unchanged_not_applied_write(evidence, effect, test, current) -> bool:
    """A settled not_applied write whose host closure proves no workspace change."""
    if effect.status != "not_applied" or effect.read_only or effect.tool == "run_tests":
        return False
    try:
        _, record = evidence._closed(effect.receipt_ref)
    except Exception:  # noqa: BLE001 - missing/invalid closure proves nothing
        return False
    return (record.get("continuation_effect_id") == effect.effect_id
            and record.get("before_binding") == current
            and record.get("after_binding") == current
            and not record.get("changes")
            and type(test.get("sequence")) is int
            and test["sequence"] > effect.intent_sequence)


def completion_evidence_failure(evidence, state, reference, intention_id, current,
                                certification_sequence=None):
    """Return ``None`` when ``reference`` is acceptable completion evidence,
    else a typed failed-predicate code.  ``certification_sequence=None`` is the
    preflight form (no certification exists yet); the commit passes the
    certification event's sequence.
    """
    test = _resolve_test(evidence, reference)
    if not test:
        return EVIDENCE_MISSING
    ancestor = test.get("continuation_id")
    while ancestor in state.intentions and ancestor != intention_id:
        ancestor = state.intentions[ancestor].parent_id
    if ancestor != intention_id:
        return FOREIGN_INTENTION
    effect = state.effects.get(test.get("continuation_effect_id"))
    if effect is None:
        return EFFECT_MISSING
    if effect.status in {"pending", "unknown"}:
        return EFFECT_UNRESOLVED
    if effect.status != "applied" and not _unchanged_not_applied_write(evidence, effect, test, current):
        return EFFECT_NOT_APPLIED
    if (effect.intention_id != test.get("continuation_id")
            or effect.dispatch_ref != test.get("continuation_dispatch_ref")):
        return EFFECT_LINEAGE_MISMATCH
    if test.get("producer") != "test_subprocess":
        return NOT_TEST_SUBPROCESS
    if type(test.get("exit_code")) is not int or test.get("exit_code") != 0:
        return TEST_RED
    if test.get("binding_stable") is not True:
        return BINDING_UNSTABLE
    if test.get("binding") != current or test.get("after_binding") != current:
        return BINDING_STALE
    if certification_sequence is not None:
        if test.get("sequence", certification_sequence) >= certification_sequence:
            return NEWER_THAN_CERTIFICATION
    return None


def typed_refusal(code: str) -> str:
    return STALE_CERTIFICATION_REASON + " [failed_predicate=" + code + "]"


__all__ = [
    "EXHAUSTED_REASON", "STALE_CERTIFICATION_REASON", "completion_evidence_failure", "typed_refusal",
    "enabled", "is_exhausted", "record_fresh_test", "record_refusal",
    "refusal_count", "reset", "should_allow_run_tests",
]
