"""Fixed-width, model-visible countdown and its canonical budget projection."""
from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping


CANONICAL_TIME_SLOT = "99999.9 (about 999+ minutes left, ~999+ turns at your pace)"
# Live numeric renderings use ASCII and this exact UTF-8 width.
TIME_SLOT_WIDTH = len(CANONICAL_TIME_SLOT)
# The limits line appears at a line start (the plain-text frame, or a hook panel line prefixed
# "| ") and as the value of the [StateFrame] JSON field "continuation_limits" in a message's
# content. Anywhere else, e.g. a source window that quotes it, is file content and stays
# untouched. Callers pass parsed request bodies, never serialized wire bytes.
_TIME_FIELD = re.compile(
    r'(?m)(?:^|(?<="continuation_limits":"))'
    r"((?:\| )?\[CONTINUATION LIMITS \| stateframe-continuation-limits/1\][^\n]*?"
    r"sandbox_seconds_remaining=)[0-9][^;\n]*(;)"
)
_SOURCE_PREFIXES = (
    "[Current source:", "[Working source:", "[Observed source:",
    "[Retained source:", "[Repair source:", "[Paged source:",
    "[Request budget source window]", "[Model source window]", "CURRENT_SOURCE ",
)


def render_time_slot(limits: object, seconds: object) -> str:
    """Give the model bounded seconds, minutes and measured turns at one width."""
    if seconds is None:
        return "unbounded"
    try:
        left = float(seconds)
    except OverflowError:
        left = math.inf
    if math.isnan(left):
        left = 0.0
    left = min(99999.9, max(0.0, left))
    now = time.monotonic()
    start = getattr(limits, "_time_left_clock_start", None)
    if not isinstance(start, float):
        start = now
        try:
            setattr(limits, "_time_left_clock_start", start)
            setattr(limits, "_time_left_decisions_at_start", int(
                getattr(limits, "model_decisions_used", 0) or 0))
        except (AttributeError, TypeError):
            pass
    used = int(getattr(limits, "model_decisions_used", 0) or 0) - int(
        getattr(limits, "_time_left_decisions_at_start", 0) or 0)
    turns = "----"
    if used > 0 and now > start:
        pace = (now - start) / used
        if pace > 0:
            turns = ("999+" if pace < left / 999 else
                     f"{min(999, int(left // pace)):04d}")
    minutes_count = int(round(left / 60))
    minutes = "999+" if minutes_count > 999 else f"{minutes_count:04d}"
    return f"{left:07.1f} (about {minutes} minutes left, ~{turns} turns at your pace)"


def canonical_time_text(text: str) -> str:
    """Project a limits line onto the one slot used for measurement."""
    if "sandbox_seconds_remaining=" not in text or text.startswith(_SOURCE_PREFIXES):
        return text
    projected = _TIME_FIELD.sub(
        lambda match: match[1] + CANONICAL_TIME_SLOT + match[2], text)
    return text if projected == text else projected


def canonical_countdown(value: object) -> object:
    """Replace only the host limits TIME field for sizing, leaving wire data intact."""
    if isinstance(value, str):
        return canonical_time_text(value)
    if isinstance(value, Mapping):
        projected = {key: canonical_countdown(item) for key, item in value.items()}
        return value if all(projected[key] is item for key, item in value.items()) else projected
    if isinstance(value, list):
        projected = [canonical_countdown(item) for item in value]
        return value if all(new is old for new, old in zip(projected, value)) else projected
    if isinstance(value, tuple):
        projected = tuple(canonical_countdown(item) for item in value)
        return value if all(new is old for new, old in zip(projected, value)) else projected
    return value
