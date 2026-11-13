"""Content-free, atomic selection authority for stable-wire optional row trims."""
from __future__ import annotations

from collections.abc import Mapping
import json

from kernel_frame_optional_allocation import identity
from kernel_stable_wire_record import MAX_TRIM_CANDIDATES


def removed_candidates(before: Mapping[str, object], after: Mapping[str, object]
                       ) -> list[dict[str, object]] | None:
    """Identify an ordered subset by original index and allocator identity.

    Never advertise incomplete replay authority: a rewrite, insertion, reorder,
    or non-message change is not a row removal and must fail open instead.
    """
    if {key: value for key, value in before.items() if key != "messages"} != {
            key: value for key, value in after.items() if key != "messages"}:
        return None
    retained = iter(after["messages"])
    missing = object()
    next_row = next(retained, missing)
    removed = []
    for index, row in enumerate(before["messages"]):
        if next_row is not missing and row == next_row:
            next_row = next(retained, missing)
        else:
            removed.append({"kind": "message", "index": index, "candidate_id": identity(row)})
    if next_row is not missing or len(removed) > MAX_TRIM_CANDIDATES:
        return None
    # Mapping equality ignores key order; transport compares serialized bytes.
    indices = {item["index"] for item in removed}
    rebuilt = {**before, "messages": [row for index, row in enumerate(before["messages"])
                                     if index not in indices]}
    return removed if json.dumps(rebuilt).encode() == json.dumps(after).encode() else None


def apply_selection(body: dict[str, object], selection: list[dict[str, object]]) -> bool:
    """Validate the entire selection before removing anything; never consult a meter."""
    messages = body["messages"]
    indices = set()
    for candidate in selection:
        index = candidate["index"]
        if (candidate["kind"] != "message" or type(index) is not int
                or index < 0 or index >= len(messages)
                or identity(messages[index]) != candidate["candidate_id"] or index in indices):
            return False
        indices.add(index)
    body["messages"] = [row for index, row in enumerate(messages) if index not in indices]
    return True
