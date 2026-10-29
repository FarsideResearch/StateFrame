"""Opt-in sweep fixes shared by StateFrame renderers and closure wiring.

The default is deliberately a no-op.  The renderer projection in this module
is fed only by the serialized result of ``CompletionReducer``; public test
verdicts remain evidence, but cannot be rendered as an admission decision in
the sweep arm.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping


MODE_ENV = "STATEFRAME_SWEEP_FIXES"
SCHEMA = "atomic-completion-render/1"
_OUTCOMES = {
    "ACCEPTED", "REFUSED", "BASIS_CHANGED", "BLOCKED",
    "NEEDS_WORK", "BOUNDED", "SEALED", "PENDING",
}


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def pending_completion() -> dict[str, object]:
    """Return the safe render state before the reducer has seen a done attempt."""
    return {
        "schema": SCHEMA,
        "outcome": "PENDING",
        "accepted": False,
        "reason_codes": ["NO_COMPLETION_ATTEMPT"],
        "detail": "done has not been submitted to the completion reducer",
    }


def completion_projection(frame: Mapping[str, object]) -> dict[str, object]:
    """Normalize the reducer result carried by a live frame.

    ``atomic_completion`` is written by the runtime from
    ``CompletionDecision.to_dict()``.  ``completion_decision`` is accepted as
    a compatibility alias for small host fixtures, but no public test fields
    are consulted here.
    """
    raw = frame.get("atomic_completion", frame.get("completion_decision"))
    if not isinstance(raw, Mapping):
        return pending_completion()
    outcome = str(raw.get("outcome", "PENDING")).upper()
    if outcome not in _OUTCOMES:
        outcome = "PENDING"
    reason_codes = raw.get("reason_codes", ())
    if isinstance(reason_codes, (str, bytes)):
        reason_codes = [str(reason_codes)]
    elif isinstance(reason_codes, (list, tuple)):
        reason_codes = [str(code) for code in reason_codes if str(code)]
    else:
        reason_codes = []
    return {
        "schema": SCHEMA,
        "outcome": outcome,
        "accepted": bool(raw.get("accepted") is True or outcome == "ACCEPTED"),
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "detail": str(raw.get("detail", "")),
        **({"basis_digest": str(raw["basis_digest"])}
           if raw.get("basis_digest") is not None else {}),
    }


def phase_for_completion(completion: Mapping[str, object]) -> str:
    """Map the reducer outcome to a model-facing phase without regrading tests."""
    outcome = str(completion.get("outcome", "PENDING")).upper()
    return {
        "ACCEPTED": "completion_accepted",
        "REFUSED": "completion_refused",
        "NEEDS_WORK": "completion_needs_work",
        "BLOCKED": "completion_blocked",
        "BASIS_CHANGED": "completion_basis_changed",
        "BOUNDED": "completion_bounded",
        "SEALED": "completion_sealed",
    }.get(outcome, "completion_pending")


def certification_for_completion(completion: Mapping[str, object]) -> str:
    """Return the compact certification label owned by the reducer."""
    if completion.get("accepted") is True:
        return "ACCEPTED"
    outcome = str(completion.get("outcome", "PENDING")).upper()
    return outcome if outcome in _OUTCOMES else "PENDING"


def admission_for_completion(completion: Mapping[str, object]) -> str:
    """Return a lean/unified admission label owned by the reducer."""
    return "ADMISSIBLE" if completion.get("accepted") is True else "BLOCKED"


def legacy_work_closure_disabled(host_mode: object) -> bool:
    """Whether model-owned work may no longer veto host-owned closure."""
    return enabled() and str(host_mode) == "on"


def decision_projection(decision: object) -> dict[str, object]:
    """Serialize a reducer decision for the next StateFrame render."""
    to_dict = getattr(decision, "to_dict", None)
    raw = to_dict() if callable(to_dict) else decision
    if not isinstance(raw, Mapping):
        return pending_completion()
    return completion_projection({"atomic_completion": copy.deepcopy(dict(raw))})


__all__ = [
    "MODE_ENV", "SCHEMA", "admission_for_completion", "certification_for_completion",
    "completion_projection", "decision_projection", "enabled", "legacy_work_closure_disabled",
    "mode", "pending_completion", "phase_for_completion",
]
