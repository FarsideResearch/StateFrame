"""Step-cap boundary and arithmetic. None means no cap, never a large sentinel."""
from __future__ import annotations

from itertools import count
from collections.abc import Iterator

DEFAULT_STEPS = 60
UNBOUNDED_GUIDANCE = (
    "There is no step limit; finish when the task is done and verified. "
    "If genuinely stuck, preserve the concrete blocker."
)


def parse_step_limit(value: object) -> int | None:
    """Normalize product configuration; generic ledgers can still express zero fuel."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "unbounded":
            return None
        try:
            value = int(value)
        except ValueError as error:
            raise ValueError("step limit must be a non-negative integer or unbounded") from error
    if type(value) is not int or value < 0:
        raise ValueError("step limit must be a non-negative integer or unbounded")
    return value or None


def remaining(limit: int | None, used: int) -> int | None:
    return None if limit is None else max(0, limit - used)


def reached(limit: int | None, used: int) -> bool:
    return limit is not None and used >= limit


def minimum_limit(*limits: int | None) -> int | None:
    bounded = [value for value in limits if value is not None]
    return min(bounded) if bounded else None


def limit_text(limit: int | None) -> str:
    return "unbounded" if limit is None else str(limit)


def step_indices(start: int, limit: int | None) -> Iterator[int]:
    return count(start) if limit is None else iter(range(start, limit))


def unbounded_guidance_mode() -> str:
    """STATEFRAME_STEPS_GUIDANCE_LINE: 'replacement' (default, the true no-cap line) or 'removed' (no line). The removal-only
    arm exists for the required removal-vs-replacement splice before the old step-count line goes."""
    import os
    mode = os.environ.get("STATEFRAME_STEPS_GUIDANCE_LINE", "replacement")
    if mode not in {"replacement", "removed"}:
        raise ValueError("STATEFRAME_STEPS_GUIDANCE_LINE must be replacement or removed")
    return mode


def step_guidance(limit: int | None, *, decisions: bool = False) -> str:
    if limit is None:
        return UNBOUNDED_GUIDANCE if unbounded_guidance_mode() == "replacement" else ""
    return (f"You have at most {limit} decisions." if decisions
            else f"You have {limit} steps.")


def runtime_unbounded(runtime: object) -> bool:
    """Missing runtime attributes are unknown, not evidence of an absent cap."""
    return hasattr(runtime, "step_limit") and runtime.step_limit is None
