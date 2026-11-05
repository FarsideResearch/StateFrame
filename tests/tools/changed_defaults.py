"""Switches whose default changed after an older release that tests still compare request bytes against.

A test that runs an older tree and the current tree and compares their request bytes must state the
setting of every switch listed here when it runs the newer tree; otherwise the comparison silently
picks up the new default. tests/test_cross_version_pins.py enforces this.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChangedDefault:
    since: str    # first release with the new default
    pin: str      # value that restores the older request bytes
    keyword: str  # keyword the comparison helpers take for this switch


CHANGED_DEFAULTS = {
    "STATEFRAME_STABLE_TOOL_HEAD": ChangedDefault(since="0.3.13", pin="off", keyword="stable_tool_head"),
    # The release that turns the final cache layout on by default (set at release time).
    "STATEFRAME_FINAL_CACHE_LAYOUT": ChangedDefault(since="next", pin="off", keyword="final_cache_layout"),
}
