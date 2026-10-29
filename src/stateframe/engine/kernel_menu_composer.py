"""Deterministic, explicitly ordered tool-menu composition."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence


# Stages describe the complete per-request menu pipeline.  A stage may be
# omitted for an arm, but an installed stage always runs at this fixed point.
MENU_STAGE_ORDER = (
    "work_record_seed",
    "continuation",
    "generated",
    "decision_state",
    "tool_applicability",
    "repeat_guard",
    "work_record_restore",
    "terminal_green_lock",
    "closure_reserve",
)


def compose_menu(
    tools: Sequence[dict],
    stages: Mapping[str, Callable[[list[dict]], Sequence[dict]]],
) -> list[dict]:
    """Apply declared stages in the one canonical order.

    Registration order is intentionally irrelevant. Unknown stages are an
    error so a new mutator cannot silently bypass the declared pipeline.
    """
    unknown = sorted(set(stages) - set(MENU_STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown menu stages: {unknown}")
    current = list(tools)
    for name in MENU_STAGE_ORDER:
        apply = stages.get(name)
        if apply is None:
            continue
        result = apply(current)
        if result is None:
            raise TypeError(f"menu stage {name!r} returned None")
        current = list(result)
    return current


__all__ = ["MENU_STAGE_ORDER", "compose_menu"]
