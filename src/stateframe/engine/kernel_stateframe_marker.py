"""Shared wire handling for text that begins with the StateFrame marker."""
from __future__ import annotations

import copy
from collections.abc import Mapping

from kernel_stateframe_envelope import STATE_HEADER


def escape_state_marker_data(
        message: Mapping[str, object], *, host_state: bool = False,
        preserve_host_provenance: bool = False,
) -> dict:
    """Copy and escape wire text; retain provenance only for F5's intermediate F4 view."""
    item = copy.deepcopy(dict(message))
    if not preserve_host_provenance:
        item.pop("_stateframe_host_state", None)
    content = str(item.get("content") or "")
    if not host_state and content.startswith(STATE_HEADER):
        # The visible escape is \[StateFrame]. Only the host state starts [StateFrame].
        item["content"] = "\\" + content
    return item
