"""Anchor-recovery note on rejected edits (soak CONFIRMED lead C4, fx_anchor_recovery).

Qualification (stacked on restate_req): T 5/12, C 1/12, P 2/12; benchmark scout positive,
benchmark null. Targets the real benchmark failure shape: rejected/ambiguous edits, then a stall.

Placement (not yet qualification screened, so DEFAULT OFF): appended after the rejected edit's
canonical typed result as a ``[kernel]`` note, verbatim. Opt in: STATEFRAME_ANCHOR_RECOVERY=on.
Language-generic: names no language, file type or task.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os

ENV = "STATEFRAME_ANCHOR_RECOVERY"
MESSAGE = ("Base each edit on current source and confirm the target region is unique before "
           "applying it. If an edit is rejected, reread the current region and form a new exact "
           "operation before continuing.")
NOTE = "[kernel] " + MESSAGE


def mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    value = str(values.get(ENV, "off") or "off").strip().lower()
    return value if value in {"on", "off"} else "off"


def annotate_rejection(observation: str, *, rejected: bool) -> str:
    """Append the note to a rejected edit's tool result (idempotent)."""
    if mode() != "on" or not rejected or not isinstance(observation, str) or NOTE in observation:
        return observation
    return observation + "\n" + NOTE


def wire_receipt(wire: bytes) -> dict[str, object] | None:
    """Authenticate the note in exact serialized request bytes."""
    if mode() != "on":
        return None
    try:
        messages = json.loads(wire).get("messages")
    except (TypeError, ValueError, AttributeError):
        return None
    if not isinstance(messages, list):
        return None
    indexes = [index for index, row in enumerate(messages)
               if isinstance(row, Mapping) and isinstance(row.get("content"), str)
               and NOTE in row["content"]]
    if not indexes:
        return None
    return {"message_indexes": indexes[:16], "occurrences": len(indexes),
            "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest()}
