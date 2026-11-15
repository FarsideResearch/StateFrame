"""Observational allocator preparation: never owns or replaces submitted bytes."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json

if __package__:
    from .kernel_read_receipts import enabled
    from .kernel_working_set_snapshot import (PreparationSnapshot, SnapshotConstructionError,
                                              freeze_preparation)
else:
    from kernel_read_receipts import enabled
    from kernel_working_set_snapshot import (PreparationSnapshot, SnapshotConstructionError,
                                             freeze_preparation)


@dataclass(frozen=True)
class SnapshotFailure:
    request_id: str
    error_type: str
    reason: str = "PREPARATION_CONTRACT_FAILURE"


def freeze_shadow(body, runtime, *, request_id: str, dialect="llama.cpp",
                  reasoning_mode="on") -> PreparationSnapshot | SnapshotFailure | None:
    """Detach before legacy fitting can remove a required source carrier.

    Diagnostic errors are typed and contain no captured source or exception
    text. A broken shadow must not change legacy dispatch or its result.
    """
    if not enabled():
        return None
    from kernel_request_budget import (
        HARD_MAX_REQUEST_BYTES, ENDPOINT_PROMPT_TOKEN_BUDGET, ENDPOINT_CONTEXT_TOKENS,
        ENDPOINT_THINKING_BUDGET_TOKENS, ENDPOINT_SAFETY_HEADROOM_TOKENS,
    )
    from kernel_context import FRAME_CEILING_TOKENS
    import os
    try:
        # A missing or non-integer ceiling means "no frame limit", never a failed shadow.
        ceiling = getattr(runtime, "frame_ceiling", None)
        frame_limit = (ceiling if type(ceiling) is int and ceiling >= 0
                       and (ceiling != FRAME_CEILING_TOKENS or "STATEFRAME_CEILING" in os.environ)
                       else None)
        snapshot = freeze_preparation(body, runtime, request_id=request_id,
            byte_cap=HARD_MAX_REQUEST_BYTES, prompt_limit=ENDPOINT_PROMPT_TOKEN_BUDGET,
            frame_limit=frame_limit,
            layouts=("plain", "final_cache"), dialect=dialect, reasoning_mode=reasoning_mode)
        return replace(snapshot, context_limit=ENDPOINT_CONTEXT_TOKENS,
                       output_reserve=ENDPOINT_THINKING_BUDGET_TOKENS,
                       safety_reserve=ENDPOINT_SAFETY_HEADROOM_TOKENS)
    except Exception as error:
        return SnapshotFailure(request_id, type(error).__name__,
            error.reason if isinstance(error, SnapshotConstructionError) else
            "CAPTURE_BLOB_UNAVAILABLE" if isinstance(error, FileNotFoundError)
            else "PREPARATION_CONTRACT_FAILURE")


def _due_coverage(snapshot: PreparationSnapshot, legacy_wire: bytes, result) -> dict[str, object]:
    from kernel_working_set_prepare import PreparedCandidate
    body = json.loads(legacy_wire)
    texts = [str(m.get("content", "")) for m in body.get("messages", ())]
    assignments = (dict(result.rendered.certificate.assignment)
                   if isinstance(result, PreparedCandidate) else {})
    due = [slot for slot in snapshot.sources if slot.state == "DUE"]
    # Legacy coverage uses the exact captured page/payload, never a path notice.
    # This diagnostic deliberately does not grant lifecycle exposure authority.
    legacy = [slot.key for slot in due if any(slot.page in text for text in texts)]
    full = [slot.key for slot in due if assignments.get(slot.key) == "FULL"]
    return {"count": len(due), "full": len(full), "legacy_full": len(legacy),
            "full_slot_ids": full, "legacy_full_slot_ids": legacy,
            "fresh_full": len(full) == len(due)}


def journal_shadow(snapshot, legacy_wire: bytes, *, journal, byte_cap: int | None = None,
                   prompt_limit: int | None = None, bytes_per_token: float | None = None):
    """Solve, render and independently prove without publishing lifecycle state."""
    if not enabled():
        return None
    from kernel_working_set_prepare import Blocked, PreparedCandidate, prepare_candidate
    fields = {"request_id": getattr(snapshot, "request_id", ""), "dispatch": False,
              "assignment_digest": None, "floor": 0, "total": 0,
              "cap": byte_cap or getattr(snapshot, "byte_cap", 0),
              "section_breakdown": {}, "due_reads": {}, "blocked_reason": None}
    result = None
    try:
        if isinstance(snapshot, SnapshotFailure):
            raise ValueError(snapshot.error_type)
        if not isinstance(snapshot, PreparationSnapshot):
            raise TypeError("missing frozen shadow snapshot")
        snapshot = replace(snapshot,
            byte_cap=snapshot.byte_cap if byte_cap is None else byte_cap,
            prompt_limit=snapshot.prompt_limit if prompt_limit is None else prompt_limit,
            bytes_per_token=snapshot.bytes_per_token if bytes_per_token is None else bytes_per_token)
        result = prepare_candidate(snapshot)
        fields.update(floor=result.floor_bytes, total=result.total_bytes, cap=result.cap,
                      due_reads=_due_coverage(snapshot, legacy_wire, result))
        if isinstance(result, PreparedCandidate):
            fields.update(assignment_digest=result.assignment_digest,
                section_breakdown=dict(result.rendered.section_breakdown),
                fits_without_degrade=result.fits_without_degrade,
                measurement_method=result.measurement.method,
                prompt_tokens=result.measurement.prompt_tokens,
                frame_tokens=result.measurement.frame_tokens)
        elif isinstance(result, Blocked):
            fields.update(blocked_reason=result.reason.value,
                          assignment_digest=getattr(result, "assignment_digest", None),
                          section_breakdown=dict(result.section_breakdown))
    except Exception as error:
        fields.update(blocked_reason=snapshot.reason if isinstance(snapshot, SnapshotFailure)
                      else "PREPARATION_CONTRACT_FAILURE",
                      error_type=snapshot.error_type if isinstance(snapshot, SnapshotFailure)
                      else type(error).__name__)
        result = None
    journal.emit("working_set_shadow_plan", **fields)
    return result
