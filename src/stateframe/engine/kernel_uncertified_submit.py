"""Explicit, bounded uncertified submission candidate for green public checks."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from kernel_completion_policy import NEVER_ADMIT_UNCERTIFIED_CODES, NON_WAIVABLE_CODES
from kernel_verdict import SATISFIED, Verdict

MODE_ENV = "STATEFRAME_UNCERTIFIED_SUBMIT"
SCHEMA = "stateframe-uncertified-submit/1"
INTERNAL_CODES = frozenset({"HOST_OBLIGATION", "CLAUSE_COVERAGE", "METADATA_OPEN",
                            "INTERNAL_COVERAGE_OPEN", "INTERNAL_METADATA_OPEN"})


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def _green_current(runtime: object) -> bool:
    verdict_fn = getattr(runtime, "public_test_verdict", None)
    binding_fn = getattr(runtime, "binding", None)
    if not callable(verdict_fn) or not callable(binding_fn):
        return False
    try:
        verdict = verdict_fn()
        current = binding_fn()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return (isinstance(verdict, Verdict) and verdict.state == SATISFIED
            and verdict.current_binding == current
            and bool(verdict.trusted_receipts)
            and all(receipt.binding == current and receipt.after_binding == current
                    and receipt.exit_code == 0 and receipt.binding_stable is True
                    for receipt in verdict.trusted_receipts))


def eligible(runtime: object, *, caller_admissible: bool,
             reason_codes: Sequence[str], open_ids: Sequence[str], commit: object) -> bool:
    if mode() != "on" or not _green_current(runtime) or not callable(commit):
        return False
    codes = {str(code) for code in reason_codes if str(code)}
    if not open_ids or not codes or codes - INTERNAL_CODES:
        return False
    forbidden = (codes & NEVER_ADMIT_UNCERTIFIED_CODES) or (codes & NON_WAIVABLE_CODES)
    if forbidden or "PUBLIC_RED_DONE" in codes or "OPEN_COUNTEREXAMPLE" in codes:
        return False
    from kernel_behavior_witness import pending_public_evidence_work
    if pending_public_evidence_work(runtime, open_ids):
        return False
    # This lane is intentionally for a blocked coverage/metadata decision.
    # A caller that was already admissible has no open internal gate to convert.
    return not caller_admissible


def outcome_text(open_ids: Sequence[str]) -> str:
    ids = ", ".join(str(item) for item in open_ids if str(item))
    return f"submit now as uncertified with open IDs [{ids}]; public checks are green on the current binding."


def stage(runtime: object, open_ids: Sequence[str], *, reason_codes: Sequence[str]) -> bool:
    ids = tuple(dict.fromkeys(str(item) for item in open_ids if str(item)))
    if not ids:
        return False
    runtime._v39_uncertified_submit_ids = ids
    runtime._v39_uncertified_submit_text = outcome_text(ids)
    runtime.journal.emit("uncertified_submit_available", producer="kernel_uncertified_submit",
                         event_schema=SCHEMA, open_ids=list(ids), reason_codes=list(reason_codes),
                         outcome=outcome_text(ids), dispatch=False)
    return True


def append_outcome(runtime: object, prepared: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Place the staged candidate in the actual provider message sequence."""
    text = str(getattr(runtime, "_v39_uncertified_submit_text", "") or "")
    outgoing = [dict(message) for message in prepared]
    if not text:
        return outgoing
    if any(message.get("content") == text for message in outgoing):
        return outgoing
    outgoing.append({"role": "user", "content": text})
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("uncertified_submit_rendered", producer="kernel_uncertified_submit",
             event_schema=SCHEMA, text=text, dispatch=False)
    return outgoing


__all__ = ["INTERNAL_CODES", "SCHEMA", "append_outcome", "eligible", "mode",
           "outcome_text", "stage"]
