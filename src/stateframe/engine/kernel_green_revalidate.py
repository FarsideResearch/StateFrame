"""Receipt-bound continuation refresh after automatic public verification."""
from __future__ import annotations

import os
from collections.abc import Mapping

from kernel_verdict import SATISFIED, TrustedTestReceipt, Verdict

MODE_ENV = "STATEFRAME_GREEN_REVALIDATE"
DEFERRED_MODE_ENV = "STATEFRAME_GREEN_REVALIDATE_DEFERRED"
SCHEMA = "stateframe-green-revalidate/1"


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def deferred_mode() -> str:
    value = os.environ.get(DEFERRED_MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{DEFERRED_MODE_ENV} must be off or on")
    return value


def trusted_current_receipt(runtime: object, receipt_ref: str) -> tuple[Verdict, TrustedTestReceipt] | None:
    """Select the exact current passing receipt from the shared verdict."""
    verdict_fn = getattr(runtime, "public_test_verdict", None)
    binding_fn = getattr(runtime, "binding", None)
    if not callable(verdict_fn) or not callable(binding_fn):
        return None
    try:
        verdict = verdict_fn()
        current = binding_fn()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if not isinstance(verdict, Verdict) or verdict.state != SATISFIED:
        return None
    if verdict.current_binding != current:
        return None
    for receipt in verdict.trusted_receipts:
        if (isinstance(receipt, TrustedTestReceipt)
                and receipt.receipt_id == receipt_ref
                and receipt.binding == current
                and receipt.after_binding == current
                and receipt.exit_code == 0
                and receipt.binding_stable is True):
            return verdict, receipt
    return None


def effect_is_settled(runtime: object, effect_id: str | None) -> bool:
    continuation = getattr(runtime, "continuation", None)
    state = getattr(getattr(continuation, "engine", None), "state", None)
    effects = getattr(state, "effects", {}) if state is not None else {}
    effect = effects.get(effect_id) if isinstance(effects, Mapping) else None
    return effect is not None and getattr(effect, "status", None) == "applied"


__all__ = ["DEFERRED_MODE_ENV", "SCHEMA", "deferred_mode", "effect_is_settled",
           "mode", "trusted_current_receipt"]
