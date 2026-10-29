"""STATEFRAME_INFLIGHT_FIXES: corrections for in-flight task state.

On:
  1. class-scoped edits accept leading import lines next to the one class; missing imports are added at module top
     (ldev_05 fe: ``import copy`` + class rejected INVALID_CLASS_PAYLOAD 4x).
  3. a post-green STALLED exit submits through green-then-exhaust instead of ending unsubmitted, and a reused
     verification at the same product revision drops run_tests from the post-green menu (one predicate, also used
     by the identical-rerun rule, which is inert under DISPOSITION v2 otherwise).
Off (default): legacy behaviour byte-for-byte.  Fix 2 (multi-definition duplicate names / parse crash) is ungated:
a crash is never a wanted behaviour.
"""
from __future__ import annotations

import os

MODE_ENV = "STATEFRAME_INFLIGHT_FIXES"


def mode() -> str:
    value = str(os.environ.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


EVENT = "inflight_fix_applied"


def _journal(runtime: object) -> object:
    journal = getattr(runtime, "journal", None)
    if journal is None:
        try:
            from kernel_pilot import ACTIVE_RUNTIME
            journal = getattr(ACTIVE_RUNTIME.get(), "journal", None)
        except (ImportError, LookupError):
            journal = None
    return journal


def emit(runtime: object, fix: str, stage: str = "acted", **fields: object) -> None:
    """Journal the switch's own event (fired proof).  ``stage``: evaluated / applicable / acted.  Never raises."""
    if not enabled():
        return
    journal = _journal(runtime)
    emit_fn = getattr(journal, "emit", None)
    if not callable(emit_fn):
        return
    try:
        emit_fn(EVENT, producer="kernel_inflight_fixes", dispatch=False, fix=fix, stage=stage, **fields)
    except Exception:  # noqa: BLE001 - telemetry never changes behaviour
        return


def verification_key(runtime: object) -> object:
    """Reuse is keyed to product, test, and environment bindings, not revision alone."""
    binding = getattr(runtime, "binding", None)
    if callable(binding):
        try:
            value = binding()
        except Exception:  # noqa: BLE001 - fall back to the revision key
            value = None
        if isinstance(value, dict):
            import hashlib
            import json
            key = {name: value.get(name) for name in ("artifact", "environment", "harness_files")}
            if key["artifact"] is not None:
                return "binding:" + hashlib.sha256(
                    json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()
    from kernel_first_green_disclosure import _product_revision
    return _product_revision(runtime, None)


def _norm_selector(selector: object) -> str:
    return " ".join(str(selector or "").split())


def note_reused_revision(measurement: object, revision: object, selector: object = "") -> None:
    """Record that verification of ``selector`` was reused (not re-run) at ``revision``."""
    if revision is None:
        return
    seen = getattr(measurement, "_r45_reused_revisions", None)
    if not isinstance(seen, set):
        seen = set()
        try:
            setattr(measurement, "_r45_reused_revisions", seen)
        except Exception:  # noqa: BLE001 - frozen/foreign measurement objects: nothing to record
            return
    seen.add((str(revision), _norm_selector(selector)))


def _public_selectors(runtime: object) -> tuple[str, ...]:
    """The public check selectors at this state (readiness decision), normalised; () when unknown."""
    for source in (getattr(runtime, "_readiness_request_decision", None), getattr(runtime, "_readiness_decision", None)):
        selectors = getattr(source, "selectors", None)
        if selectors:
            return tuple(sorted({_norm_selector(item) for item in selectors if _norm_selector(item)}))
    return ()


def reused_at(measurement: object, revision: object) -> bool:
    """The one predicate: True only when EVERY public selector's
    verification was already reused at this exact product revision. With the selector set unknown, a reuse of
    the whole-suite request (empty selector) is required -- a reuse of one selector never hides another's rerun."""
    seen = getattr(measurement, "_r45_reused_revisions", None)
    if not (enabled() and revision is not None and isinstance(seen, set)):
        return False
    rev = str(revision)
    reused = {sel for r, sel in seen if r == rev}
    if not reused:
        return False
    selectors = _public_selectors(measurement)
    if selectors:
        return set(selectors) <= reused
    return "" in reused


__all__ = ["EVENT", "emit", "MODE_ENV", "enabled", "mode", "note_reused_revision", "reused_at", "verification_key"]
