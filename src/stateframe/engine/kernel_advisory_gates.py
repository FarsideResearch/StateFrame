"""Typed completion classification: three hard truths gate; evidence gaps advise.

This is a completion input, not a second terminal owner.  The completion owner
consults it before candidate admission; GOVFIX still owns any concrete
requirement refusal and its once-per-episode release.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_ADVISORY_GATES"
SCHEMA = "stateframe-advisory-gates/1"
_ADVISORY_CODES = frozenset({
    "ADVISORY_ONLY", "CLAUSE_COVERAGE", "COVERAGE_OPEN", "GET_SURE_OPEN_CLAUSE",
    "HOST_OBLIGATION", "INCONCLUSIVE", "NAMED_LAYER_COVERAGE_UNVERIFIED",
    "OBLIGATION_EVIDENCE_OPEN", "PROVABILITY_OPEN", "REQUIRED_EVIDENCE_GAP",
    "REQUIREMENT_ALIGNMENT_OPEN", "UNBOUND_OBSERVABLE", "UNRESOLVED",
    "UNVERIFIED_CHECKS", "MISSING_CHECKS", "NO_VERIFIER", "METADATA_OPEN",
    "V20_TICKET_CLAUSE_UNPROVEN", "RECHECK_REQUIRED", "STALE_EVIDENCE", "STALE",
    "MISSING_SELECTOR", "NOT_EXECUTED", "AMBIGUOUS_CONTRACT", "UNBOUND_ENTRYPOINT",
    "UNBOUND_INVOCATION", "SKIPPED", "CONFLICTING_RECEIPTS", "CONFLICTING_RESULTS",
    "SELECTOR_ERROR", "VERIFY", "TESTS_NOT_RUN", "SOURCE_EVIDENCE", "READ_EVIDENCE",
    "EVIDENCE_GAP", "ORACLE_UNAVAILABLE", "LATE_VETO", "UNKNOWN",
})
_VIOLATION_STATUSES = frozenset({
    "COUNTEREXAMPLE", "ERROR", "FAIL", "FAILED", "PRODUCT_RED", "RED",
    "REGRESSION", "UNSATISFIED", "VIOLATED",
})
_ADVISORY_REASONS = _ADVISORY_CODES
_SATISFIED = frozenset({"SATISFIED", "PASS", "PASSED", "GREEN", "CLOSED", "COMPLETE"})


def mode(environ: Mapping[str, str] | None = None) -> str:
    value = str((os.environ if environ is None else environ).get(MODE_ENV, "off") or "off").strip().lower()
    if value in {"", "0", "off", "false", "no"}:
        return "off"
    if value in {"1", "on", "true", "yes"}:
        return "on"
    raise ValueError(f"{MODE_ENV} must be on or off")


def enabled(runtime: object | None = None) -> bool:
    value = getattr(runtime, "stateframe_advisory_gates_mode", None)
    return str(value).strip().lower() == "on" if value is not None else mode() == "on"


def _field(value: object, name: str, default: object = "") -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _public_test_blocker(runtime: object) -> str | None:
    """Return only a witnessed public-red blocker; missing evidence is advisory."""
    verdict_fn = getattr(runtime, "public_test_verdict", None)
    binding_fn = getattr(runtime, "binding", None)
    if not callable(verdict_fn) or not callable(binding_fn):
        return None
    try:
        verdict = verdict_fn()
        current = binding_fn()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    state = str(_field(verdict, "state", "")).upper()
    if state in _VIOLATION_STATUSES:
        return "public_red"
    if state not in _SATISFIED:
        return None
    if _field(verdict, "current_binding", None) != current:
        return None
    receipts = _sequence(_field(verdict, "trusted_receipts", ()))
    for receipt in receipts:
        exit_code = _field(receipt, "exit_code", None)
        status = str(_field(receipt, "result", _field(receipt, "status", ""))).upper()
        if status in _VIOLATION_STATUSES or (isinstance(exit_code, int) and exit_code != 0):
            return "public_red"
    return None


def _counterexample_blocker(runtime: object) -> str | None:
    for name in ("open_counterexamples", "counterexamples", "live_counterexamples"):
        if getattr(runtime, name, ()):
            return name
    for record in _sequence(getattr(runtime, "behavior_witness_records", ())):
        if str(_field(record, "result", "")).upper() in {"FAIL", "COUNTEREXAMPLE"}:
            return "behavior_witness_counterexample"
    for assessment in _sequence(getattr(runtime, "host_closure_assessments", ())):
        status = str(_field(assessment, "verdict", _field(assessment, "status", ""))).upper()
        reason = str(_field(assessment, "reason", "")).upper()
        if status in _VIOLATION_STATUSES or reason in _VIOLATION_STATUSES:
            return "observed_counterexample"
    return None


def _unsafe_or_unsettled_blocker(runtime: object) -> str | None:
    for name in ("unsafe_actions", "unsafe_effects", "unsettled_effects", "pending_effects"):
        if getattr(runtime, name, ()):
            return name
    for name in ("destructive_actions", "effect_audit_log"):
        for action in _sequence(getattr(runtime, name, ())):
            status = str(_field(action, "status", "")).upper()
            if bool(_field(action, "unsafe", False)) or bool(_field(action, "destructive", False)):
                if status not in {"APPLIED", "COMMITTED", "COMPLETE", "COMPLETED", "SETTLED"}:
                    return f"{name}_unsafe"
    continuation = getattr(runtime, "continuation", None)
    engine = getattr(continuation, "engine", None)
    unsettled_summary = getattr(engine, "has_unsettled_effects", None)
    if callable(unsettled_summary):
        if unsettled_summary():
            return "unsettled_effects"
        return None
    state = getattr(engine, "state", None)
    effects = _field(state, "effects", {})
    if isinstance(effects, Mapping) and any(
            str(_field(effect, "status", "")).lower() in {"pending", "unknown", "unsettled"}
            for effect in effects.values()):
        return "unsettled_effects"
    return None


def _assessment_id(assessment: object) -> str:
    return str(_field(assessment, "obligation_id", _field(assessment, "id", ""))).strip()


def advisory_ids(runtime: object) -> tuple[str, ...]:
    values: list[str] = []
    for value in _sequence(getattr(runtime, "_v39_uncertified_submit_ids", ())):
        if str(value).strip():
            values.append(str(value))
    for assessment in _sequence(getattr(runtime, "host_closure_assessments", ())):
        reason = str(_field(assessment, "reason", _field(assessment, "reason_code", ""))).upper()
        status = str(_field(assessment, "verdict", _field(assessment, "status", ""))).upper()
        identifier = _assessment_id(assessment)
        if identifier and reason in _ADVISORY_REASONS and status not in _VIOLATION_STATUSES:
            values.append(identifier)
    completion = getattr(runtime, "_last_completion_decision", None)
    for value in _sequence(_field(completion, "needs_work_ids", ())):
        if str(value).strip():
            values.append(str(value))
    report = getattr(runtime, "_provability_last_report", None)
    if isinstance(report, Mapping):
        for key in ("open_ids", "advisory_ids", "uncertified_ids"):
            for value in _sequence(report.get(key, ())):
                if str(value).strip():
                    values.append(str(value))
    return tuple(dict.fromkeys(values))


def evaluate(runtime: object) -> dict[str, object]:
    """Return hard blockers separately; evidence-gap rows never become blockers."""
    blockers = [item for item in (
        _public_test_blocker(runtime),
        _unsafe_or_unsettled_blocker(runtime),
        _counterexample_blocker(runtime),
    ) if item]
    advisories = advisory_ids(runtime)
    return {
        "schema": SCHEMA,
        "hard_blockers": list(dict.fromkeys(blockers)),
        "advisories": list(advisories),
        "allow_done": not blockers,
        "uncertified": bool(advisories),
    }


def only_advisory_codes(codes: Sequence[object]) -> bool:
    normalized = {str(code).strip().upper() for code in codes if str(code).strip()}
    if not normalized:
        return False
    return all(
        code in _ADVISORY_CODES
        or code.startswith(("COVERAGE_", "PROVABILITY_", "OBLIGATION_EVIDENCE_"))
        for code in normalized
    )


__all__ = [
    "MODE_ENV", "SCHEMA", "advisory_ids", "enabled", "evaluate", "mode",
    "only_advisory_codes",
]
