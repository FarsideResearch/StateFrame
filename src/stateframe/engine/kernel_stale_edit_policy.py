"""Opt-in policy for repairing a stale continuation without granting completion."""
from __future__ import annotations

import os
from collections.abc import Mapping

SWITCH = "KERNEL_STALE_EDIT_REPAIR"
STALE_REPAIR_TOOLS = frozenset({"edit", "str_replace"})


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether stale product-edit repair is enabled for this host run."""
    values = os.environ if environ is None else environ
    value = values.get(SWITCH, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{SWITCH} must be 0 or 1")
    return value == "1"


def allows_stale_repair(tool: object, *, environ: Mapping[str, str] | None = None) -> bool:
    """Admit only edit effects; evidence-consuming effects remain gated."""
    return enabled(environ) and isinstance(tool, str) and tool in STALE_REPAIR_TOOLS
