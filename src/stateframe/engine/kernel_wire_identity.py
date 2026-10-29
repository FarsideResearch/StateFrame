"""Stable, compact identities for values that can enter provider messages.

Journal run IDs remain random storage/lease identities. Wire IDs instead use
the task identity and a journal position, so an A/A replay presents the same
bytes while a resumed journal continues past its durable sequence.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from contextlib import nullcontext


WIRE_ID_HEX_LENGTH = 20  # 80 bits; birthday bound at 10^6 IDs is < 4.2e-13.
_ORPHAN_LOCK = threading.Lock()
_ORPHAN_POSITIONS: dict[tuple[str, str], int] = {}
ORPHAN_MINTS = 0
_WORKSPACE_ID = re.compile(r"(?<=continuation-workspaces/)[0-9a-f]{32}(?=/)")
_ABSOLUTE_WORKSPACE = re.compile(
    r"(?<![\w])/(?:[^/\s\"'\\]+/)*continuation-workspaces/[0-9a-f]{32}/repo/"
)
_PYTEST_DURATION = re.compile(r"\bin\s+\d+(?:\.\d+)?s\b")
_OBJECT_ADDRESS = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")


def detfix3_legacy() -> bool:
    """Restore the pre-DETFIX3 model-visible random IDs for controlled comparison."""
    return os.environ.get("SF_LAB_DETFIX3", "on").strip().lower() == "legacy"


_MODE_LINE_EMITTED = False


def mode_line() -> str:
    """One stderr line naming effective history, memory, wire, and eviction modes."""
    from kernel_history_spool import legacy as memgrow2_legacy
    from kernel_memfix_history import legacy as memfix_legacy
    from kernel_evictfix_mode import mode as evictfix_mode
    memgrow2 = "legacy" if memgrow2_legacy() else "on"
    memfix = "legacy" if memfix_legacy() else "on"
    detfix3 = "legacy" if detfix3_legacy() else "on"
    return (f"stateframe: MEMGROW2 mode={memgrow2} DETFIX3 mode={detfix3} "
            f"MEMFIX mode={memfix} EVICTFIX mode={evictfix_mode()}")


def emit_mode_line(stream=None) -> None:
    """Write the mode line once per process. Never touches the journal or the wire."""
    global _MODE_LINE_EMITTED
    if _MODE_LINE_EMITTED:
        return
    _MODE_LINE_EMITTED = True
    import sys
    target = stream if stream is not None else sys.stderr
    print(mode_line(), file=target, flush=True)


def stable_wire_id(task_id: object, kind: str, *position: object) -> str:
    """Derive a domain-separated ID from stable content and a position."""
    identity = [str(task_id or ""), kind, *position]
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:WIRE_ID_HEX_LENGTH]


def wire_run_id(journal: object) -> str:
    """Public alias for a run; never expose its random journal directory name."""
    if detfix3_legacy():
        return str(getattr(journal, "run_id", ""))
    return stable_wire_id(getattr(journal, "task_id", ""), "wire-run")


def public_wire_text(value: str, journal: object) -> str:
    """Normalize volatile model text, leaving host journal and audit data intact."""
    if detfix3_legacy():
        return value
    value = _PYTEST_DURATION.sub("in <duration>", value)
    value = _OBJECT_ADDRESS.sub("0x<addr>", value)
    return _public_identity_text(value, journal)


def _public_identity_text(value: str, journal: object) -> str:
    """Alias host IDs in arbitrary frame fields without rewriting source text."""
    if detfix3_legacy():
        return value
    if journal is None:
        # DETFIX4b: callers holding only stored output (no journal) still must not leak the
        # random continuation-workspace directory; a fixed placeholder is replay-stable.
        value = _ABSOLUTE_WORKSPACE.sub("continuation-workspaces/<workspace>/repo/", value)
        return _WORKSPACE_ID.sub("<workspace>", value)
    workspace_id = stable_wire_id(getattr(journal, "task_id", ""), "continuation-workspace")
    value = _ABSOLUTE_WORKSPACE.sub(
        f"continuation-workspaces/{workspace_id}/repo/", value)
    value = _WORKSPACE_ID.sub(workspace_id, value)
    host_run_id = getattr(journal, "run_id", None)
    if isinstance(host_run_id, str) and host_run_id and host_run_id in value:
        host_token = re.compile(
            rf"(?<!sha256:)(?<![0-9a-f]){re.escape(host_run_id)}(?![0-9a-f])")
        value = host_token.sub(wire_run_id(journal), value)
    return value


def public_wire_value(value: object, journal: object, *,
                      normalize_output: bool = False) -> object:
    """Detach values; normalize output only when the caller owns output fields."""
    transform = public_wire_text if normalize_output else _public_identity_text
    if isinstance(value, str):
        return transform(value, journal)
    if isinstance(value, dict):
        return {_public_identity_text(key, journal) if isinstance(key, str) else key:
                public_wire_value(item, journal, normalize_output=normalize_output)
                for key, item in value.items()}
    if isinstance(value, list):
        return [public_wire_value(item, journal, normalize_output=normalize_output)
                for item in value]
    if isinstance(value, tuple):
        return tuple(public_wire_value(item, journal, normalize_output=normalize_output)
                     for item in value)
    return value


def next_wire_id(journal: object, kind: str, *content: object) -> str:
    """Mint a unique ID at the current durable journal position.

    Multiple captures/receipts before the next append receive increasing local
    ordinals. After restore, the replayed journal sequence moves the position
    beyond every previously recorded provider-visible identity.
    """
    if detfix3_legacy():
        return uuid.uuid4().hex
    if journal is None:
        # Synthetic callers without a journal have no durable replay authority.
        # Live runs never get here: MeasurementRuntime attaches owner.audit_journal
        # before any Bolt receipt (runtime.py ~5348). ORPHAN_MINTS is the tripwire
        # A nonzero count during a real run is a bug.
        global ORPHAN_MINTS
        with _ORPHAN_LOCK:
            ORPHAN_MINTS += 1
            if ORPHAN_MINTS == 1:
                logging.getLogger(__name__).warning(
                    "wire identity minted without a journal (kind=%s); replay-unstable", kind)
            key = (kind, repr(content))
            ordinal = _ORPHAN_POSITIONS.get(key, 0) + 1
            _ORPHAN_POSITIONS[key] = ordinal
        return stable_wire_id("orphan", kind, ordinal, *content)
    lock = getattr(journal, "_lock", None)
    with lock if lock is not None else nullcontext():
        sequence = getattr(journal, "_sequence", 0)
        positions = getattr(journal, "_wire_identity_positions", None)
        if positions is None:
            positions = {}
            setattr(journal, "_wire_identity_positions", positions)
        key = (kind, sequence)
        ordinal = positions.get(key, 0) + 1
        positions[key] = ordinal
        # One active sequence at a time; avoid retaining a long run's positions.
        # Evict only OTHER sequences: dropping a current-sequence key would let a
        # later mint of that kind at this position reuse ordinal 1 (DETFIX2 F1).
        if len(positions) > 8:
            for stale in [k for k in positions if k[1] != sequence]:
                del positions[stale]
        return stable_wire_id(getattr(journal, "task_id", ""), kind,
                              sequence, ordinal, *content)
