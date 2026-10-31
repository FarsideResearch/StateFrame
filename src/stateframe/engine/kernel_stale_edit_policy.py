"""Opt-in policy for repairing a stale continuation without granting completion."""
from __future__ import annotations

import os
from collections.abc import Mapping

from kernel_continuation_menu_authority import EDIT_TOOLS

SWITCH = "KERNEL_STALE_EDIT_REPAIR"
# Product mutations repair stale intentions under the same policy. Keeping a
# second literal list silently withheld new editors from the live tool menu.
STALE_REPAIR_TOOLS = EDIT_TOOLS


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether stale product-edit repair is enabled for this host run."""
    values = os.environ if environ is None else environ
    value = values.get(SWITCH, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{SWITCH} must be 0 or 1")
    return value == "1"


def allows_stale_repair(tool: object, *, environ: Mapping[str, str] | None = None) -> bool:
    """Admit only edit effects; evidence-consuming effects remain gated."""
    if tool == "job_stop":
        return True  # Lifecycle cleanup never consumes product evidence.
    if not enabled(environ) or not isinstance(tool, str):
        return False
    from kernel_agent_tools import service
    host = service()
    return tool in STALE_REPAIR_TOOLS or (host is not None and tool in host.custom)
