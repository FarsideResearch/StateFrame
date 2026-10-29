"""Opt-in Sidekick challenge schema and run-wide challenge charging.

The Sidekick challenge is advisory lifecycle work, but its selector payload is
consumed by the runtime when it renders the challenge.  Keep the repair at the
producer/consumer seam and charge the existing ParentMeter so separate Pilot
workers cannot each spend a local challenge.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_CHALLENGE_SCHEMA"
SCHEMA = "stateframe-sidekick-challenge/1"
CHALLENGE_REASON = "Sidekick missing current check"
CHALLENGE_PREFIX = "[sidekick] Verify the current tree once: "


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _selector_values(value: object) -> list[str]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        values = ()
    selectors: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        selector = item.strip()
        if selector and selector not in seen:
            selectors.append(selector)
            seen.add(selector)
    return selectors


def repair_finding(finding: Mapping[str, object], required_selectors: Sequence[object]) -> dict:
    """Add the required selector list to an enabled CHALLENGE_ONCE finding.

    Existing selector values are preserved when present.  The fallback is the
    host-computed required set, never model text or receipt prose.
    """
    if not enabled() or finding.get("finding") != "CHALLENGE_ONCE":
        return finding
    selectors = _selector_values(finding.get("selectors"))
    if not selectors:
        selectors = _selector_values(required_selectors)
    repaired = dict(finding)
    repaired["selectors"] = selectors
    return repaired


def issue_challenge(
    finding: Mapping[str, object], *, parent: object, sidekick: object,
    remaining_steps: int | None, enforce: bool,
) -> str | None:
    """Charge and render one challenge through the shared ParentMeter."""
    if (not enforce or finding.get("finding") != "CHALLENGE_ONCE"
            or (remaining_steps is not None and remaining_steps <= 0)):
        return None
    selectors = _selector_values(finding.get("selectors"))
    if not selectors:
        return None
    used = getattr(parent, "used", {})
    limits = getattr(parent, "limits", {})
    if used.get("challenges", 0) >= limits.get("challenges", 0):
        return None
    parent.charge("challenges", reason=CHALLENGE_REASON)
    sidekick.challenged = True
    return CHALLENGE_PREFIX + ", ".join(selectors)


def evaluate(
    finding: Mapping[str, object], *, required_selectors: Sequence[object] = (),
) -> dict[str, object]:
    """Evaluate the closed challenge shape without charging or rendering it.

    The live adversary uses this as a producer/consumer compatibility check.
    It deliberately does not turn the opt-in Sidekick flag on and it cannot
    create a refusal; ``issue_challenge`` remains the only charging seam.
    """
    if not isinstance(finding, Mapping):
        return {"schema": SCHEMA, "status": "REJECTED", "reason": "finding_object"}
    if finding.get("finding") != "CHALLENGE_ONCE":
        return {"schema": SCHEMA, "status": "NOT_APPLICABLE", "selectors": []}
    repaired = repair_finding(finding, required_selectors)
    selectors = _selector_values(repaired.get("selectors"))
    # Evaluation is intentionally independent of the opt-in rendering flag:
    # the live adversary must be able to validate the closed shape while the
    # existing issue_challenge seam remains the only charging/refusal owner.
    if not selectors:
        selectors = _selector_values(required_selectors)
    if not selectors:
        return {"schema": SCHEMA, "status": "REJECTED", "reason": "selectors_required"}
    return {
        "schema": SCHEMA,
        "status": "DISABLED" if not enabled() else "ADMITTED",
        "selectors": selectors,
        "charged": False,
        "enforce_with": "issue_challenge",
    }


__all__ = [
    "CHALLENGE_PREFIX", "CHALLENGE_REASON", "MODE_ENV", "SCHEMA",
    "enabled", "evaluate", "issue_challenge", "mode", "repair_finding",
]
