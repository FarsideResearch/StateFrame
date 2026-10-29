"""STATEFRAME_REREAD_CONTENT: a read the model issues is answered with the current content, never an
"already in frame" stub (r45 token cause 2).

A3B batch 430 ldev_01 (frame): 13 ALREADY_PRESENT stubs, two re-plans, and the CLI item dropped out of the plan after
them; r44d crawl_l_one (67k tokens): 12 stubs after green.  A read is an explicit request for content, so with the
switch on the empirical read-dedup stub is skipped and the ordinary read executes (current bytes).  Off (default):
the dedup path is unchanged.
"""
from __future__ import annotations

import os

MODE_ENV = "STATEFRAME_REREAD_CONTENT"


def mode() -> str:
    value = str(os.environ.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def dedup_allowed() -> bool:
    """May the runner answer a read with the ALREADY_PRESENT stub?  Off: yes (legacy).  On: never."""
    return not enabled()


__all__ = ["MODE_ENV", "dedup_allowed", "enabled", "mode"]
