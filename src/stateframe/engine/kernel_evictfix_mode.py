"""One authority for the EVICTFIX comparison switch.

The release path defaults to the established eviction behavior.  EVICTFIX
remains available as an explicit, case-insensitive opt-in.
"""
from __future__ import annotations

import os


ENV_NAME = "STATEFRAME_EVICTFIX"


def mode() -> str:
    """Return the effective mode; unknown values fail closed to legacy."""
    value = os.environ.get(ENV_NAME, "legacy").strip().lower()
    return value if value in {"on", "legacy"} else "legacy"


def enabled() -> bool:
    return mode() == "on"


def journal_mode(journal: object) -> None:
    """Record mode and one value-free warning for an invalid switch per run."""
    if getattr(journal, "_evictfix_mode_journaled", False):
        return
    raw = os.environ.get(ENV_NAME, "legacy").strip().lower()
    effective = mode()
    emit = getattr(journal, "emit")
    emit("evictfix_mode", mode=effective)
    if raw not in {"on", "legacy"}:
        # The raw environment value may contain secrets; journal only the fault.
        emit("evictfix_mode_warning", reason="invalid_value", mode="legacy")
    journal._evictfix_mode_journaled = True
