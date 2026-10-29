"""Bound repeated post-check actions that make no progress.

Progress accounting remains run-wide and distinguishes refusals from completed
actions, so repeated requests cannot replenish an exhausted budget.
"""
from __future__ import annotations

import os
from collections.abc import Mapping


MODE_ENV = "STATEFRAME_POSTGREEN_NO_PROGRESS"
THRESHOLD_ENV = "STATEFRAME_POSTGREEN_NO_PROGRESS_N"
DEFAULT_THRESHOLD = 3
SCHEMA = "stateframe-postgreen-no-progress/1"
PRODUCER = "kernel_postgreen_no_progress"
CAP_REASON = "POSTGREEN_NO_PROGRESS_CAP"

# The two no-progress kinds this module recognizes.  Kept as an explicit
# tuple (not an open string) so a caller passing a typo fails loudly instead
# of silently never capping.
REUSED_VERIFICATION = "reused_verification"
REFUSED_DONE = "refused_done"
KINDS = (REUSED_VERIFICATION, REFUSED_DONE)

_COUNT_ATTR = "_postgreen_no_progress_count"
_ATTEMPTED_ATTR = "_postgreen_no_progress_attempted"


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    from kernel_capability_floor import enabled as capability_floor_enabled

    if capability_floor_enabled(runtime):
        # Post-green stagnation may remain observable, but it cannot replace
        # ordinary recovery or the explicit hard-budget authorities.
        return False
    if runtime is not None:
        configured = getattr(runtime, "stateframe_postgreen_no_progress_mode", None)
        if configured is not None:
            return str(configured).strip().lower() == "on"
    return mode() == "on"


def threshold(runtime: object | None = None) -> int:
    configured = getattr(runtime, "stateframe_postgreen_no_progress_n", None) if runtime is not None else None
    raw = configured if configured is not None else os.environ.get(THRESHOLD_ENV)
    if raw is None:
        return DEFAULT_THRESHOLD
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{THRESHOLD_ENV} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{THRESHOLD_ENV} must be a positive integer")
    return value


def _green_engaged(runtime: object) -> bool:
    lock = getattr(runtime, "green_lock", None)
    if lock is None:
        # No lock attached (green-lock mode off): the no-progress cap is
        # scoped to post-green behaviour only and must fail closed.
        return False
    return bool(getattr(lock, "_green", False))


def _record(runtime: object, event_kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    payload = {"producer": PRODUCER, "payload_schema": SCHEMA, "dispatch": False, **fields}
    try:
        emit(event_kind, **payload)
    except Exception:  # noqa: BLE001 - telemetry cannot change no-progress behavior
        return


def note_no_progress(runtime: object, *, kind: str) -> int:
    """Record one post-green response that made no progress. Returns the streak."""
    if kind not in KINDS:
        raise ValueError(f"unknown post-green no-progress kind {kind!r}")
    if not enabled(runtime) or not _green_engaged(runtime):
        return 0
    count = int(getattr(runtime, _COUNT_ATTR, 0)) + 1
    setattr(runtime, _COUNT_ATTR, count)
    # "kind" is reserved by the journal emit(kind, **fields) signature that
    # _record forwards into; the no-progress kind travels under its own key.
    _record(runtime, "postgreen_no_progress_signal", no_progress_kind=kind,
            streak=count, threshold=threshold(runtime))
    return count


def note_progress(runtime: object) -> None:
    """Reset the streak: a product change, a new verification, or new evidence happened."""
    if int(getattr(runtime, _COUNT_ATTR, 0) or 0) == 0:
        return
    setattr(runtime, _COUNT_ATTR, 0)
    setattr(runtime, _ATTEMPTED_ATTR, False)
    _record(runtime, "postgreen_no_progress_reset")


def streak(runtime: object) -> int:
    return int(getattr(runtime, _COUNT_ATTR, 0) or 0)


def cap_reached(runtime: object) -> bool:
    return enabled(runtime) and streak(runtime) >= threshold(runtime)


def enforce_cap(runtime: object):
    """Settle an eligible submission or advise on a capped no-progress streak.

    ``None`` means the cap is not reached, was already attempted for this
    streak, or hard-budget-only mode queued an advisory so the caller can
    continue. A returned decision is either an accepted submission or the
    legacy typed non-accepted terminal.
    """
    if not cap_reached(runtime):
        return None
    if getattr(runtime, _ATTEMPTED_ATTR, False):
        return None
    setattr(runtime, _ATTEMPTED_ATTR, True)
    count = streak(runtime)

    from kernel_green_then_exhaust import submit_if_ready
    decision = submit_if_ready(runtime, boundary="post_green_stall")
    if decision is not None and bool(getattr(decision, "accepted", False)):
        _record(runtime, "postgreen_no_progress_submitted", streak=count, accepted=True)
        return decision

    completion_decision = getattr(runtime, "completion_decision", None)
    if not callable(completion_decision):
        from kernel_terminal_policy import (
            enabled as hard_budget_only_enabled,
            queue_advisory,
        )
        if hard_budget_only_enabled(runtime):
            queue_advisory(runtime, "postgreen_no_progress")
            _record(runtime, "postgreen_no_progress_advisory", streak=count, accepted=False)
            return None
        return decision
    terminal = completion_decision(
        admissible=False,
        detail=(f"post-green no-progress cap reached: {count} consecutive responses "
                "produced no product change, no new verification and no new evidence"),
        reason_codes=(CAP_REASON,),
        artifact_preserved=True,
    )
    if bool(getattr(terminal, "accepted", False)):
        # A fail-closed authority must never accept an inadmissible decision;
        # if it did, prefer that truth over this module's own bookkeeping.
        _record(runtime, "postgreen_no_progress_submitted", streak=count, accepted=True)
        return terminal
    from kernel_terminal_policy import (
        enabled as hard_budget_only_enabled,
        queue_advisory,
    )
    if hard_budget_only_enabled(runtime):
        queue_advisory(runtime, "postgreen_no_progress")
        _record(runtime, "postgreen_no_progress_advisory", streak=count, accepted=False)
        return None
    setattr(runtime, "system_outcome", "postgreen_no_progress")
    _record(runtime, "postgreen_no_progress_terminal", streak=count, accepted=False)
    return terminal


__all__ = [
    "CAP_REASON", "DEFAULT_THRESHOLD", "KINDS", "MODE_ENV", "PRODUCER",
    "REFUSED_DONE", "REUSED_VERIFICATION", "SCHEMA", "THRESHOLD_ENV",
    "cap_reached", "enabled", "enforce_cap", "mode", "note_no_progress",
    "note_progress", "streak", "threshold",
]
