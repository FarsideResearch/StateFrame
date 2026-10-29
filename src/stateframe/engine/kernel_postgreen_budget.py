"""Typed post-green response budget feeding the shared completion owner."""
from __future__ import annotations

import os
from collections.abc import Mapping
from kernel_done_acceptance import done_accepted


MODE_ENV = "STATEFRAME_POSTGREEN_BUDGET"
K_ENV = "STATEFRAME_POSTGREEN_BUDGET_K"
DEFAULT_K = 6
MAX_K = 1000
MANIFEST_KEY = "stateframe_postgreen_budget"
PRODUCER = "kernel_postgreen_budget"
SCHEMA = "stateframe-postgreen-budget/1"
EVALUATED_EVENT = "postgreen_budget_evaluated"
SUBMITTED_EVENT = "postgreen_budget_submitted"
BOUNDARY = "postgreen_budget"

_COUNT = "_postgreen_budget_responses"
_BASELINE = "_postgreen_budget_baseline_ids"
_OPEN = "_postgreen_budget_window_open"
_DONE = "_postgreen_budget_submitted"


def mode(environ: Mapping[str, str] | None = None) -> str:
    value = str((os.environ if environ is None else environ).get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def budget_k(environ: Mapping[str, str] | None = None) -> int:
    source = os.environ if environ is None else environ
    raw = source.get(K_ENV)
    if raw is None or not str(raw).strip():
        return DEFAULT_K
    text = str(raw).strip()
    if not text.isdigit() or not text.isascii():
        raise ValueError(f"{K_ENV} must be a positive integer, got {raw!r}")
    value = int(text)
    if not 1 <= value <= MAX_K:
        raise ValueError(f"{K_ENV} must be in [1, {MAX_K}], got {value}")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_postgreen_budget_mode", None) if runtime is not None else None
    return str(configured).strip().lower() == "on" if configured is not None else mode() == "on"


def post_green_responses(runtime: object) -> int:
    return int(getattr(runtime, _COUNT, 0) or 0)


def _emit(runtime: object, kind: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if not callable(emit):
        return
    try:
        emit(kind, producer=PRODUCER, payload_schema=SCHEMA, dispatch=False, **fields)
    except Exception:  # noqa: BLE001 - telemetry never changes the boundary
        return


def _verdict(runtime: object) -> str:
    from kernel_greenlock_terminal_fixes import public_verdict
    return public_verdict(runtime)


def _decision(runtime: object):
    method = getattr(runtime, "readiness_decision", None)
    if not callable(method):
        return None
    try:
        return method()
    except (AttributeError, TypeError, ValueError):
        return None


def _row_id(row: object) -> str:
    if not isinstance(row, Mapping):
        return ""
    for key in ("check_id", "id", "obligation_id", "counterexample_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _advisory(decision: object) -> tuple[Mapping[str, object], ...]:
    rows = getattr(decision, "advisory", ()) if decision is not None else ()
    return tuple(row for row in (rows or ()) if isinstance(row, Mapping))


def fresh_counterexamples(runtime: object, decision: object) -> tuple[str, ...]:
    baseline = set(getattr(runtime, _BASELINE, ()) or ())
    fresh: list[str] = []
    for index, row in enumerate(_advisory(decision)):
        row_id = _row_id(row) or f"#{index}"
        if bool(row.get("repairable")) or row_id not in baseline:
            if row_id not in fresh:
                fresh.append(row_id)
    return tuple(fresh)


def at_request_boundary(runtime: object) -> bool:
    """Count green requests and submit through the existing owner after K."""
    if not enabled(runtime) or done_accepted(runtime):
        return False
    if getattr(runtime, _DONE, False):
        return False
    verdict = _verdict(runtime)
    if verdict == "RED":
        setattr(runtime, _OPEN, False)
        setattr(runtime, _COUNT, 0)
        setattr(runtime, _BASELINE, ())
        setattr(runtime, "_postgreen_budget_current_verification_reserved", False)
        setattr(runtime, "_postgreen_budget_terminal_offer_reserved", False)
        return False
    if verdict == "GREEN":
        from kernel_green_lock import (
            note_unchanged_parent_green_suppressed,
            product_revision_change_from_parent,
        )
        changed, _revision, _comparison = product_revision_change_from_parent(runtime)
        from kernel_green_lock import PROVEN_PARENT_UNCHANGED as _PROVEN_UNCHANGED
        changed = changed or _comparison not in _PROVEN_UNCHANGED  # unknown parent identity is not "unchanged"
        from kernel_done_is_submit_authority import enabled as done_submit_enabled
        if not changed and not done_submit_enabled(runtime):
            setattr(runtime, _OPEN, False)
            setattr(runtime, _COUNT, 0)
            setattr(runtime, _BASELINE, ())
            setattr(runtime, "_postgreen_budget_current_verification_reserved", False)
            setattr(runtime, "_postgreen_budget_terminal_offer_reserved", False)
            note_unchanged_parent_green_suppressed(
                runtime, reason="postgreen_budget")
            return False
    if verdict == "GREEN" and not getattr(runtime, _OPEN, False):
        decision = _decision(runtime)
        setattr(runtime, _OPEN, True)
        setattr(runtime, _COUNT, 0)
        setattr(runtime, _BASELINE, tuple(
            _row_id(row) or f"#{index}" for index, row in enumerate(_advisory(decision))))
    if not getattr(runtime, _OPEN, False):
        return False
    k = budget_k()
    spent = post_green_responses(runtime)
    verification_reserved = verdict == "GREEN"
    terminal_offer_reserved = verification_reserved and spent >= k
    setattr(runtime, "_postgreen_budget_current_verification_reserved", verification_reserved)
    setattr(runtime, "_postgreen_budget_terminal_offer_reserved", terminal_offer_reserved)
    fields: dict[str, object] = {
        "post_green_responses": spent,
        "k": k,
        "public_verdict": verdict,
        "current_verification_reserved": verification_reserved,
        "terminal_offer_reserved": terminal_offer_reserved,
    }
    if spent < k:
        setattr(runtime, _COUNT, spent + 1)
        _emit(runtime, EVALUATED_EVENT, stage="counted", **fields)
        return False
    if verdict != "GREEN":
        _emit(runtime, EVALUATED_EVENT, stage="held", reason="public_verdict_not_green", **fields)
        setattr(runtime, _COUNT, spent + 1)
        return False
    decision = _decision(runtime)
    fresh = fresh_counterexamples(runtime, decision)
    from kernel_done_is_submit_authority import enabled as done_submit_enabled
    if fresh and not done_submit_enabled(runtime):
        _emit(runtime, EVALUATED_EVENT, stage="held", reason="fresh_counterexample_open",
              fresh_counterexamples=list(fresh)[:16], **fields)
        setattr(runtime, _COUNT, spent + 1)
        return False
    from kernel_green_then_exhaust import submit_if_ready
    completion = submit_if_ready(runtime, boundary=BOUNDARY)
    accepted = bool(getattr(completion, "accepted", False))
    if not accepted:
        _emit(runtime, EVALUATED_EVENT, stage="held",
              reason="not_submittable" if completion is None else "completion_refused", **fields)
        setattr(runtime, _COUNT, spent + 1)
        return False
    setattr(runtime, _DONE, True)
    _emit(runtime, EVALUATED_EVENT, stage="submitted", **fields)
    _emit(runtime, SUBMITTED_EVENT, rendered=True, submission_initiator="host", **fields)
    return True


def diagnosis_fields(runtime: object) -> dict[str, object]:
    if not enabled(runtime):
        return {}
    return {"post_green_responses": post_green_responses(runtime)}


__all__ = [
    "BOUNDARY", "DEFAULT_K", "EVALUATED_EVENT", "K_ENV", "MANIFEST_KEY", "MODE_ENV", "SUBMITTED_EVENT",
    "at_request_boundary", "budget_k", "diagnosis_fields", "enabled", "fresh_counterexamples", "mode",
    "post_green_responses",
]
