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
_COUNTDOWN_MARKER = "sandbox_seconds_remaining="


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
    if _COUNTDOWN_MARKER not in text or text.startswith(_SOURCE_PREFIXES):
        return text
    projected = _TIME_FIELD.sub(
        lambda match: match[1] + CANONICAL_TIME_SLOT + match[2], text)
    return text if projected == text else projected


def contains_countdown_marker(value: object) -> bool:
    """Find the marker with string substring checks before projecting containers."""
    if isinstance(value, str):
        return _COUNTDOWN_MARKER in value
    if isinstance(value, Mapping):
        return any(contains_countdown_marker(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_countdown_marker(item) for item in value)
    return False


def canonical_countdown(value: object, *, marker_verified: bool = False) -> object:
    """Replace only the host limits TIME field for sizing, leaving wire data intact."""
    if not marker_verified and not contains_countdown_marker(value):
        return value
    return _canonical_countdown(value)


def _canonical_countdown(value: object) -> object:
    if isinstance(value, str):
        return canonical_time_text(value)
    if isinstance(value, Mapping):
        projected = None
        for key, item in value.items():
            replacement = _canonical_countdown(item)
            if replacement is not item:
                if projected is None:
                    projected = dict(value)
                projected[key] = replacement
        return value if projected is None else projected
    if isinstance(value, list):
        projected = None
        for index, item in enumerate(value):
            replacement = _canonical_countdown(item)
            if replacement is not item:
                if projected is None:
                    projected = list(value)
                projected[index] = replacement
        return value if projected is None else projected
    if isinstance(value, tuple):
        projected = None
        replacements = []
        for item in value:
            replacement = _canonical_countdown(item)
            replacements.append(replacement)
            if replacement is not item:
                projected = replacements
        return value if projected is None else tuple(replacements)
    return value
