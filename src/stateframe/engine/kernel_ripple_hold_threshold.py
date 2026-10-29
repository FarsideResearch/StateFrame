"""r47 L4 ripple: inline the ripple advisory and HOLD the edit only when candidates >= 3.

token_bloat_verdict.md section 6 row L4 (lane 4 M4): ripple STAGED costs +10 turns (a re-issue turn per
staged edit, 2 of 6 re-issues identical).  RIPPLE_INLINE removes every hold; this switch keeps the hold
for the case it was built for (a wide blast radius, >= ``MIN_CANDIDATES`` review candidates) and applies
every narrower edit immediately with the advisory journaled and still projected into the next frame.

``STATEFRAME_RIPPLE_HOLD_MIN3`` (default off = the one-time stage/re-issue round trip is unchanged).
STATEFRAME_RIPPLE_INLINE on takes precedence (it never holds).  Own event: ``ripple_hold_threshold``
(decision=inline|hold, candidate_count, min_candidates).
"""
from __future__ import annotations

import os

MODE_ENV = "STATEFRAME_RIPPLE_HOLD_MIN3"
MANIFEST_KEY = "stateframe_ripple_hold_min3"
EVENT = "ripple_hold_threshold"
MIN_CANDIDATES = 3


def mode() -> str:
    value = str(os.environ.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def decide(candidate_count: object) -> str | None:
    """None when off; "hold" when >= MIN_CANDIDATES review candidates, else "inline"."""
    if not enabled():
        return None
    try:
        count = int(candidate_count)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        count = 0
    return "hold" if count >= MIN_CANDIDATES else "inline"


__all__ = ["EVENT", "MANIFEST_KEY", "MIN_CANDIDATES", "MODE_ENV", "decide", "enabled", "mode"]
