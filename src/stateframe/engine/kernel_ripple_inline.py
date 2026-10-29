"""STATEFRAME_RIPPLE_INLINE: ripple preview stays advisory but never holds the edit (r45 token cause 1).

Measured on the r44c frame smoke journals (364/368 ldev_01, 364 ldev_09, 365 ldev_09): 6 of 23 edit calls (26%) were
staged as REFUSED_PREDISPATCH and cost a re-issue turn (7,076 completion tokens); 2 of the 6 re-issues were the
identical call (staging changed nothing).  With the switch on the edit applies immediately; the ripple advisory is
journaled (``ripple_preview_inline``) and still reaches the next request through the advisory frame projection.
Off (default): the one-time stage/re-issue round trip is unchanged.
"""
from __future__ import annotations

import os

MODE_ENV = "STATEFRAME_RIPPLE_INLINE"
SAFETY_ENV = "STATEFRAME_RIPPLE_INLINE_SAFETY"


def mode() -> str:
    value = str(os.environ.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def safety_mode() -> str:
    value = str(os.environ.get(SAFETY_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{SAFETY_ENV} must be off or on")
    return value


def safety_enabled() -> bool:
    return safety_mode() == "on"


__all__ = ["MODE_ENV", "SAFETY_ENV", "enabled", "mode", "safety_enabled", "safety_mode"]
