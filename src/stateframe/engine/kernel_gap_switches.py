"""Switch and event helpers for lane C capability-gap mechanisms.

Every mechanism is opt-in.  The truth arms have no model filter; the one
guidance arm additionally requires an explicit model allow-list so stronger
models are never silently constrained by advisory text.
"""
from __future__ import annotations

import os
from collections.abc import Mapping


TRUTH_SWITCHES = {
    "witnesses": "STATEFRAME_GAPS_WITNESSES",
    "failure_cards": "STATEFRAME_GAPS_FAILURE_CARDS",
    "navigation": "STATEFRAME_GAPS_NAVIGATION",
    "surface_coverage": "STATEFRAME_GAPS_SURFACE_COVERAGE",
}
GUIDANCE_SWITCH = "STATEFRAME_GAPS_HYPOTHESIS_GUIDANCE"
GUIDANCE_MODELS = "STATEFRAME_GAPS_GUIDANCE_MODELS"

_ON = frozenset({"1", "on"})
_OFF = frozenset({"0", "off", ""})


def _source(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def switch_value(name: str, environ: Mapping[str, str] | None = None) -> str:
    value = str(_source(environ).get(name, "off") or "off").strip().lower()
    if value in _ON:
        return "on"
    if value in _OFF:
        return "off"
    raise ValueError(f"{name} must be off or on")


def truth_enabled(name: str, environ: Mapping[str, str] | None = None) -> bool:
    return switch_value(name, environ) == "on"


def guidance_enabled(model_name: str, environ: Mapping[str, str] | None = None) -> bool:
    source = _source(environ)
    if switch_value(GUIDANCE_SWITCH, source) != "on":
        return False
    model = str(model_name or "").strip().lower()
    if not model:
        return False
    configured = str(source.get(GUIDANCE_MODELS, "") or "")
    models = {item.strip().lower() for item in configured.split(",") if item.strip()}
    return "*" in models or model in models


def any_enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = _source(environ)
    return any(truth_enabled(name, source) for name in TRUTH_SWITCHES.values()) or (
        switch_value(GUIDANCE_SWITCH, source) == "on"
    )


def emit(journal: object | None, event_name: str, *, producer: str,
         enabled_value: bool, **fields: object) -> bool:
    """Emit an owned event without allowing diagnostics to change execution."""
    if not enabled_value:
        return False
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        return False
    try:
        emitter(event_name, producer=producer,
                event_schema="stateframe-gaps/1", dispatch=False, **fields)
    except Exception:  # noqa: BLE001 - observability is fail-open
        return False
    return True


__all__ = [
    "GUIDANCE_MODELS", "GUIDANCE_SWITCH", "TRUTH_SWITCHES", "any_enabled",
    "emit", "guidance_enabled", "switch_value", "truth_enabled",
]
