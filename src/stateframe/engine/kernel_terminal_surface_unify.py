"""Small opt-in helpers for the coherent terminal request surface.

The existing green-coherence module owns the verdict projection.  This module
owns only the new refusal vocabulary and the separately ablatable post-green
stall end condition, so the legacy renderer has no import or state change when
coherence is off.
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping


ENV = "STATEFRAME_POST_GREEN_STALL_END"
POST_GREEN_STALL = "POST_GREEN_STALL"
REFUSED_UNCHANGED_GREEN = "REFUSED_UNCHANGED_GREEN"
ADVISORY_OBLIGATIONS_FIELD = "advisory_obligations"
ADVISORY_OBLIGATIONS_LABEL = "ADVISORY OBLIGATIONS"
READY_PENDING_DECISION = "Call done; public checks already PASS at the current revision."
REFUSAL_LEAD = (
    "Public checks already PASS at the current revision; next action: done."
)


def mode() -> str:
    value = os.environ.get(ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    if runtime is not None:
        if getattr(runtime, "stateframe_green_coherence_mode", "off") != "on":
            return False
        configured = getattr(runtime, "stateframe_post_green_stall_end_mode", None)
        if configured is not None:
            return configured == "on"
    return mode() == "on"


def _payload(value: object) -> dict[str, object] | None:
    if isinstance(value, Mapping):
        result = copy.deepcopy(dict(value))
    elif isinstance(value, str):
        text = value.strip()
        start = text.find("{")
        if start < 0:
            return None
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        result = copy.deepcopy(dict(parsed)) if isinstance(parsed, Mapping) else None
    else:
        return None
    if not isinstance(result, dict):
        return None
    if str(result.get("result_class", "")).upper() != REFUSED_UNCHANGED_GREEN:
        return None
    return result


def is_refused_recheck(value: object) -> bool:
    return _payload(value) is not None


def refusal_result_class(value: object) -> str | None:
    payload = _payload(value)
    return REFUSED_UNCHANGED_GREEN if payload is not None else None


def refused_recheck_text(refusal: Mapping[str, object], *, lead: str | None = None) -> str:
    """Put plain next-action language before the machine-readable refusal."""
    payload = copy.deepcopy(dict(refusal))
    payload.setdefault("schema", "stateframe-run-tests-refusal/1")
    payload.setdefault("status", "REFUSED")
    payload["result_class"] = REFUSED_UNCHANGED_GREEN
    return (lead or REFUSAL_LEAD) + "\n" + json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def _host_done_facts_accepted(runtime: object) -> bool:
    if getattr(runtime, "stateframe_host_done_facts_mode", "off") != "on":
        return False
    provider = getattr(runtime, "done_facts_payload", None)
    payload = provider() if callable(provider) else None
    return isinstance(payload, Mapping) and payload.get("accepted") is True


def _terminal_ready(runtime: object) -> bool:
    try:
        from kernel_green_coherence import terminal_status
        status = terminal_status(runtime)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(getattr(status, "green", False) and getattr(status, "state", "") == "READY")


def post_green_stall_allowed(runtime: object) -> bool:
    return enabled(runtime) and _host_done_facts_accepted(runtime) and _terminal_ready(runtime)


def end_post_green_stall(runtime: object, *, refusal_count: int,
                         selector: str | None = None):
    """Ask the normal completion reducer to seal an accepted stall outcome."""
    if not post_green_stall_allowed(runtime):
        return None
    required = getattr(runtime, "required_public_selectors", None)
    selectors = tuple(required()) if callable(required) else ()
    claims = getattr(runtime, "done_facts_claims", None)
    decision = runtime.completion_decision(
        admissible=True,
        detail=("public checks remained PASS after two refused re-checks; "
                "the current product is accepted as-is"),
        reason_codes=(POST_GREEN_STALL,),
        selectors=selectors,
        evidence={"post_green_stall": True, "refusal_count": refusal_count,
                  "selector": selector},
        claims=claims() if callable(claims) else None,
        artifact_preserved=True,
    )
    if getattr(decision, "outcome", None) != POST_GREEN_STALL or not getattr(
            decision, "accepted", False):
        return decision
    runtime.system_outcome = "post_green_stall"
    journal = getattr(runtime, "journal", None)
    if journal is not None and hasattr(journal, "emit"):
        journal.emit(
            "post_green_stall_completion",
            producer="kernel_terminal_surface_unify",
            outcome=POST_GREEN_STALL,
            submission_initiator="host",
            refusal_count=refusal_count,
            selector=selector,
            decision=decision.to_dict(),
            dispatch=False,
        )
    return decision


__all__ = [
    "ADVISORY_OBLIGATIONS_FIELD", "ADVISORY_OBLIGATIONS_LABEL", "ENV",
    "POST_GREEN_STALL", "READY_PENDING_DECISION", "REFUSED_UNCHANGED_GREEN", "REFUSAL_LEAD",
    "enabled", "end_post_green_stall", "is_refused_recheck", "mode",
    "post_green_stall_allowed", "refusal_result_class", "refused_recheck_text",
]
