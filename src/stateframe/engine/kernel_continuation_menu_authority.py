"""Opt-in composition seam for the green-lock menu and live continuation.

The continuation is a stateful dispatch executive.  This module owns only the
new r44 switch and the small adapter which reads its immutable per-step
allow-set; it never calls ``authorize`` (which creates a dispatch fence and
therefore is not a menu predicate).
"""
from __future__ import annotations

import os
from collections.abc import Iterable

from kernel_bounded_locate_code import OBSERVATION_TOOLS


MODE_ENV = "STATEFRAME_GREENLOCK_MENU_AUTHORITY_CONTINUATION"
# r46: predict was absent from EFFECT_TOOLS, so the continuation's per-step allow set could never
# offer it even though kernel_greenlock_menu_authority.VERIFY_ACTIONS already treats "predict"
# as a verification action alongside run_tests/test/pytest.  Gated: off-parity is byte-identical.
PREDICT_MODE_ENV = "STATEFRAME_CONTINUATION_PREDICT"
CONTROL_TOOLS = frozenset({"done", "continuation_detour", "work_update", "note_update", "work"})
EDIT_TOOLS = frozenset({"edit", "str_replace", "file_create", "insert_before"})
AUTHORED_CHECK_TOOLS = frozenset({"authored_check"})
EFFECT_TOOLS = frozenset({
    *OBSERVATION_TOOLS, "text", *EDIT_TOOLS, "run_tests", *AUTHORED_CHECK_TOOLS,
})


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_greenlock_menu_authority_continuation_mode", None)
    if configured is not None:
        return configured == "on"
    return mode() == "on"


def predict_mode() -> str:
    value = os.environ.get(PREDICT_MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{PREDICT_MODE_ENV} must be off or on")
    return value


def predict_enabled() -> bool:
    """``state_allow_set`` has no runtime handle (``ContinuationExecutive.per_step_allow_set``
    calls it with only ``self.state``), so this is env-only like the rest of this module's
    switches read when no runtime override is threaded through."""
    return predict_mode() == "on"


def _dispatchable_subset(names: Iterable[str], runtime: object | None = None) -> frozenset[str]:
    from kernel_tool_surface import dispatch_admissible
    return frozenset(
        str(name).strip().lower() for name in names
        if str(name).strip() and dispatch_admissible(str(name).strip().lower(), runtime=runtime)
    )


def state_allow_set(state: object) -> frozenset[str]:
    """Return the durable executive's name-level set without changing state.

    Argument/path checks remain at the real dispatch boundary.  The set here
    is the per-intention protocol contract: no active intention permits only
    the native completion control; an unresolved effect permits only that
    control plus a possible typed verification retry.
    """
    identity = getattr(state, "active_id", None)
    intentions = getattr(state, "intentions", {})
    if not isinstance(intentions, dict) or identity not in intentions:
        return _dispatchable_subset({"done", *OBSERVATION_TOOLS})
    item = intentions[identity]
    if getattr(item, "phase", "active") != "active":
        return _dispatchable_subset({"done", *OBSERVATION_TOOLS})
    effects = getattr(state, "effects", {})
    unresolved_ids = {
        effect_id for effect_id, effect in (effects.items() if isinstance(effects, dict) else ())
        if getattr(effect, "intention_id", None) == identity
        and getattr(effect, "status", None) in {"pending", "unknown"}
    }
    if unresolved_ids:
        allowed = {"done", *OBSERVATION_TOOLS}
        if any(getattr(effects[effect_id], "tool", None) == "run_tests"
               for effect_id in unresolved_ids):
            allowed.add("run_tests")
        if predict_enabled() and any(getattr(effects[effect_id], "tool", None) == "predict"
                                     for effect_id in unresolved_ids):
            allowed.add("predict")
        return _dispatchable_subset(allowed)
    if predict_enabled():
        return _dispatchable_subset(CONTROL_TOOLS | EFFECT_TOOLS | OBSERVATION_TOOLS | {"predict"})
    return _dispatchable_subset(CONTROL_TOOLS | EFFECT_TOOLS | OBSERVATION_TOOLS)


def per_step_allow_set(runtime: object | None) -> frozenset[str] | None:
    """Read the continuation's pure name-level allow-set for this request.

    ``None`` means no live continuation is attached.  A live continuation with
    no compatible reader fails closed to an empty set; the green-lock
    authority must never turn a missing executive predicate into permission.
    """
    continuation = getattr(runtime, "continuation", None)
    if continuation is None:
        return None
    getter = getattr(continuation, "per_step_allow_set", None)
    if not callable(getter):
        getter = getattr(continuation, "allow_set", None)
    if not callable(getter):
        return _dispatchable_subset(OBSERVATION_TOOLS, runtime)
    try:
        raw = getter()
    except (AttributeError, TypeError, ValueError):
        return _dispatchable_subset(OBSERVATION_TOOLS, runtime)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        return _dispatchable_subset(OBSERVATION_TOOLS, runtime)
    return _dispatchable_subset(
        {*(str(name).strip().lower() for name in raw if str(name).strip()), *OBSERVATION_TOOLS},
        runtime,
    )


__all__ = [
    "AUTHORED_CHECK_TOOLS", "CONTROL_TOOLS", "EDIT_TOOLS", "EFFECT_TOOLS", "MODE_ENV", "OBSERVATION_TOOLS", "PREDICT_MODE_ENV", "enabled", "mode",
    "per_step_allow_set", "predict_enabled", "predict_mode", "state_allow_set",
]
