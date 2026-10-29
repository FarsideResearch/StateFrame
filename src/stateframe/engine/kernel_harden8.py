"""Fail-safe authority-error recording for the V46 hardening lane.

Authority failures are host-control failures, not model observations.  They are
durable only when the runtime journal accepts the typed event; callers still
apply the safe disposition explicitly so an error can never mint completion,
submission, or green state.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence


def _journal_safe_fields(payload):
    """Emit boundary (10-05 L109): rename journal-RESERVED keys; see kernel_continuation_journal."""
    from kernel_continuation_journal import journal_safe_fields

    return journal_safe_fields(payload)


EVENT_SCHEMA = "stateframe-harden8-authority-error/1"
AUTHORITY_EVENT = "harden8_authority_failure"
ADVISORY_EVENT = "harden8_advisory_failure"
_TERMINAL_ACTIONS = frozenset({"done", "submit"})

# Review inventory for catches changed by HARDEN8.  Keep boundary names stable so
# the notes, tests, and journal records use one searchable vocabulary.
AUTHORITY_BOUNDARIES = (
    "readiness.blocked_by_counterexample",
    "route.greenlock_predispatch_hook",
    "route.greenlock_predispatch_authority",
    "green_lock.repair_verifier",
    "completion_authority.effect_liveness_assessment",
    "readiness.menu_projection",
    "green_lock.terminal_menu_sync",
    "completion_basis.public_test_verdict",
    "evidence.redundant_verification_blocker",
    "evidence.redundant_verification_receipt",
    "evidence_pin.append",
    "route.final_owner_presentation",
    "completion_owner.finalization",
    "completion_owner.contract_assertion",
    "completion_owner.finalization_finish",
)
ADVISORY_BOUNDARIES = (
    "project_registries.build",
    "green_evidence_owner.telemetry",
    "passive_retention.continuity_snapshot",
    "passive_retention.decision_cache",
    "passive_retention.rejection_journal",
    "passive_retention.delivery_metric_journal",
    "stop_reason_taxonomy.journal",
    "bind_quantity.optional_receipt_sink",
)


def _safe_error(error: BaseException) -> dict[str, str]:
    message = str(error).splitlines()[0].strip() if str(error).strip() else "<no message>"
    return {
        "error_type": type(error).__name__,
        "message": message[:240],
    }


def _journal(runtime: object | None) -> object | None:
    if runtime is None:
        return None
    for name in ("journal", "audit_journal"):
        candidate = getattr(runtime, name, None)
        if callable(getattr(candidate, "emit", None)):
            return candidate
    return None


def _remember(runtime: object | None, event: Mapping[str, object]) -> None:
    if runtime is None:
        return
    existing = getattr(runtime, "_harden8_events", None)
    if not isinstance(existing, list):
        existing = []
        setattr(runtime, "_harden8_events", existing)
    existing.append(copy.deepcopy(dict(event)))


def record_failure(
        runtime: object | None,
        boundary: str,
        error: BaseException,
        *,
        authority: bool,
        disposition: str,
        denied_actions: Sequence[str] = (),
) -> dict[str, object]:
    """Record one typed authority/advisory failure without exposing payloads."""
    error_fields = _safe_error(error)
    event = {
        "event_schema": EVENT_SCHEMA,
        "producer": "kernel_harden8",
        "boundary": str(boundary),
        "failure_class": "AUTHORITY" if authority else "ADVISORY",
        "disposition": str(disposition),
        "denied_actions": sorted({str(item) for item in denied_actions if str(item)}),
        **error_fields,
        "dispatch": False,
    }
    event["kind"] = AUTHORITY_EVENT if authority else ADVISORY_EVENT
    _remember(runtime, event)
    journal = _journal(runtime)
    if journal is not None:
        kind = str(event.pop("kind"))
        try:
            journal.emit(kind, **_journal_safe_fields(event))
        except Exception as journal_error:  # noqa: BLE001 - retain a typed local failure
            fallback = {
                "kind": "harden8_journal_failure",
                "event_schema": EVENT_SCHEMA,
                "producer": "kernel_harden8",
                "boundary": "journal.emit:" + str(boundary),
                "failure_class": "AUTHORITY" if authority else "ADVISORY",
                "disposition": "hold" if authority else "degrade",
                "denied_actions": event["denied_actions"],
                **_safe_error(journal_error),
                "dispatch": False,
            }
            _remember(runtime, fallback)
    return event


def deny_terminal_actions(tools: object) -> list[dict[str, object]]:
    """Keep recoverable work visible while denying terminal action names."""
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return []
    safe: list[dict[str, object]] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if str(name or "").strip().lower() in _TERMINAL_ACTIONS:
            continue
        safe.append(copy.deepcopy(dict(tool)))
    return safe


def hold_terminal_result(result: object, *, boundary: str) -> dict[str, object]:
    """Convert any terminal projection into an explicit non-submitting hold."""
    held = copy.deepcopy(dict(result)) if isinstance(result, Mapping) else {}
    held.update({
        "result": "INCOMPLETE",
        "outcome": "INCOMPLETE",
        "accepted": False,
        "certified": False,
        "green": False,
        "submit": False,
        "authority_failure": str(boundary),
    })
    return held


__all__ = [
    "ADVISORY_BOUNDARIES", "ADVISORY_EVENT", "AUTHORITY_BOUNDARIES",
    "AUTHORITY_EVENT", "EVENT_SCHEMA",
    "deny_terminal_actions", "hold_terminal_result", "record_failure",
]
