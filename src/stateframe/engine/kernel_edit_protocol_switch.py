"""STATEFRAME_EDIT_PROTOCOL: one edit interface per run (legacy_safe | handles).

``legacy_safe`` (the default) keeps today's edit tools and request bytes.
``handles`` replaces the legacy text editors on the menu with the
handle-addressed ``edit_source``/``rewrite_source`` tools.
"""
from __future__ import annotations

import os

MODE_ENV = "STATEFRAME_EDIT_PROTOCOL"
LEGACY_SAFE = "legacy_safe"
HANDLES = "handles"
MODES = frozenset({LEGACY_SAFE, HANDLES})


def mode() -> str:
    # An empty value means unset, exactly as the runner's shell allowlist reads it.
    value = os.environ.get(MODE_ENV, "").strip().lower() or LEGACY_SAFE
    if value not in MODES:
        raise ValueError(f"{MODE_ENV} must be legacy_safe or handles")
    return value


def handles_enabled() -> bool:
    return mode() == HANDLES


__all__ = ["HANDLES", "LEGACY_SAFE", "MODES", "MODE_ENV", "handles_enabled", "mode"]
