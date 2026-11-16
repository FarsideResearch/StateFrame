"""Edit protocol repair hook: the glue between a refused edit_source batch and the next request.

Pure adapter around ``kernel_edit_repair_request.build``: it mints whole-file handles for
the repair request, keeps the refusal fingerprint ledger on the runtime as plain data (so
checkpoints and resume can carry it), and places the rendered repair text as a NEW final
user message so the retained prefix of the previous request is never rewritten.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from kernel_edit_handles import CURRENT_SOURCE_PREFIX, DeliveredSpan, HandleMap, mint_handles
from kernel_edit_refusal_codes import Refusal
from kernel_edit_repair_request import FingerprintLedger, RepairRequest, build

SCHEMA = "stateframe-edit-repair-hook/1"
LEDGER_ATTR = "_edit_repair_ledger"          # runtime attribute holding ledger.as_dict()
PENDING_ATTR = "_pending_edit_repair"        # runtime attribute holding the next user text (str) or None


def repair_minter(request_ordinal: int, path: str, source_text: str) -> HandleMap:
    """Whole-file SPAN handle over the touched file."""
    return mint_handles(request_ordinal, [DeliveredSpan.from_source(path, source_text)])


def load_ledger(runtime: object) -> FingerprintLedger:
    """The runtime's ledger, or an empty one when absent or malformed."""
    data = getattr(runtime, LEDGER_ATTR, None)
    if data is None:
        return FingerprintLedger()
    try:
        return FingerprintLedger.from_dict(data)
    except (TypeError, ValueError):
        return FingerprintLedger()


def store_ledger(runtime: object, ledger: FingerprintLedger) -> None:
    setattr(runtime, LEDGER_ATTR, ledger.as_dict())


def build_repair(*, runtime: object, refusals: Sequence[Refusal], current_bytes: bytes, path: str,
                 proposal: Sequence[Mapping[str, object]], request_ordinal: int) -> RepairRequest:
    """Build the repair request with the runtime ledger; store the ledger after this refusal."""
    repair = build(list(refusals), current_bytes, repair_minter, list(proposal), path=path,
                   request_ordinal=request_ordinal, ledger=load_ledger(runtime))
    store_ledger(runtime, repair.ledger)
    return repair


def place_repair_message(messages: list[dict], text: str) -> None:
    """Append the repair text as a new final user message; earlier messages are never touched."""
    if not isinstance(text, str) or not text:
        raise ValueError("repair text must be a non-empty string")
    if messages:
        last = messages[-1].get("content") if isinstance(messages[-1], Mapping) else None
        if isinstance(last, str) and CURRENT_SOURCE_PREFIX in last:
            raise ValueError("the final message already carries a current-source map")
    messages.append({"role": "user", "content": text})


def journal_fields(repair: RepairRequest) -> dict[str, object]:
    """Journal-safe summary: no source bytes, no handle spellings."""
    return {
        "codes": [code.value for code in repair.codes],
        "fingerprints": list(repair.fingerprints),
        "level": repair.level.value,
        "repeat_count": repair.repeat_count,
        "blocked": repair.blocked,
        "request_ordinal": repair.request_ordinal,
        "path": repair.path,
        "digest": repair.digest(),
    }


SNAPSHOT_KEY = "edit_repair_ledger"          # worker checkpoint key; present only once a refusal was recorded


def snapshot_ledger(runtime: object) -> dict[str, object] | None:
    """The ledger as checkpoint data, or None while nothing was refused (legacy checkpoint shape)."""
    data = getattr(runtime, LEDGER_ATTR, None)
    return dict(data) if isinstance(data, Mapping) and data.get("counts") else None


def valid_snapshot(state: object) -> bool:
    """True when ``state`` is a well-formed ledger snapshot."""
    if not isinstance(state, Mapping) or set(state) != {"counts"}:
        return False
    try:
        return bool(FingerprintLedger.from_dict(state).counts)
    except (TypeError, ValueError):
        return False


def restore_ledger(runtime: object, state: object) -> None:
    if isinstance(state, Mapping) and isinstance(state.get("counts"), Mapping):
        store_ledger(runtime, FingerprintLedger.from_dict(state))


def trip_loop_breaker_if_blocked(runtime: object, repair: RepairRequest, tool: str) -> str | None:
    """On a BLOCKED repair, trip the live loop breaker (if any) via trip_external; else None."""
    if not repair.blocked:
        return None
    breaker = getattr(runtime, "_tool_call_loop_breaker", None)
    if breaker is None or not hasattr(breaker, "trip_external"):
        return None
    return breaker.trip_external(
        tool, f"was refused {repair.repeat_count} times with the same refusal fingerprint.")


__all__ = [
    "LEDGER_ATTR", "PENDING_ATTR", "SCHEMA", "SNAPSHOT_KEY", "build_repair", "journal_fields",
    "load_ledger", "place_repair_message", "repair_minter", "restore_ledger", "snapshot_ledger",
    "store_ledger", "trip_loop_breaker_if_blocked", "valid_snapshot",
]
