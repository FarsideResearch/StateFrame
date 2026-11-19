"""Per-plan retirement for the parent `parallel` tool.

A verified fanout retires only its own plan. The tool stays on the menu, so a
different plan can still be dispatched later in the same run. Retirement state
lives on the run's owner object (one run, one set), never in module globals.
"""
from __future__ import annotations

import hashlib
import json
from typing import Final

# The switch is read by the engine's probes module (the release switch scanner only
# covers manifest-listed files); callers pass the resolved value in as `enabled`.
ALREADY_COMPLETED_OBSERVATION: Final = (
    "REFUSED PLAN_ALREADY_COMPLETED: this exact plan already completed and verified in this run. "
    "Dispatch a different plan for new work."
)
_RETIRED_ATTR: Final = "_parallel_retired_plan_ids"


def plan_identity(arguments: object) -> str:
    """Stable digest of a plan, independent of JSON key order or string-vs-object input."""
    parsed: object = arguments
    if isinstance(arguments, (str, bytes, bytearray)):
        try:
            parsed = json.loads(arguments)
        except ValueError:
            parsed = arguments
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _retired_ids(owner: object) -> set[str]:
    retired = getattr(owner, _RETIRED_ATTR, None)
    if retired is None:
        retired = set()
        setattr(owner, _RETIRED_ATTR, retired)
    return retired


def retire_completed_plan(owner: object, arguments: object, terminal: bool, *, enabled: bool) -> bool:
    """Retire this plan only when its fanout finished fully verified. Returns True if retired."""
    if not terminal or not enabled:
        return False
    _retired_ids(owner).add(plan_identity(arguments))
    return True


def refusal_for(owner: object, arguments: object, *, enabled: bool) -> str | None:
    """Return the refusal observation when this exact plan already completed in this run."""
    if not enabled:
        return None
    if plan_identity(arguments) in _retired_ids(owner):
        return ALREADY_COMPLETED_OBSERVATION
    return None
