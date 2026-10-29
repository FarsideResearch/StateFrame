"""Strict, opt-in policy for host-owned deterministic follow-up work."""
from __future__ import annotations

import os


MODE_ENV = "STATEFRAME_HOST_AUTOPILOT"
MANIFEST_FIELD = "stateframe_host_autopilot"


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def effective_auto_verify_mode(configured: str) -> str:
    """Let the host switch opt into the existing auto-verify rail only."""
    if configured not in {"off", "on"}:
        raise ValueError("STATEFRAME_AUTO_VERIFY must be off or on")
    return "on" if enabled() else configured
