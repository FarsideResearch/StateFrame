"""Typed fault recording for registered guard handlers, layered on kernel_reason_log.emit_reason.

record_fault classifies a caught error against the handler's HandlerSpec and writes ONE reason row per
(category, handler, disposition, cause) per run (repeats only count). It is logging only: it never raises
from its own failures and never changes control flow. KeyboardInterrupt, SystemExit and CancelledError
raised by the recording path propagate. Such exceptions passed as the error argument are recorded like
any other fault. Callers keep their own passthrough before the broad catch.

Return value: True when the record was written or counted, False when the reason log could not record it
(emit_reason self-error). Step-3 call sites decide what False means (T0 pauses sends).
The self-error check runs under kernel_reason_log._LOCK, so a self-error from another record_fault call
cannot flip this result. Reason-log writes that do not take the lock can still bump the shared counter,
and then the result errs toward False.
"""
from __future__ import annotations

import asyncio
import re

import kernel_reason_log as rl
from kernel_fault_registry import HandlerSpec, lookup_handler

UNREGISTERED = "unregistered"
EMPTY_DETAIL = "<empty message>"
# Typed fault codes are bounded and charset-limited so a hostile or huge code cannot blow up cardinality.
FAULT_CODE_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,40}")
# Same passthrough set the callers keep before their broad except. Never swallowed here.
_PASSTHROUGH = (KeyboardInterrupt, SystemExit, asyncio.CancelledError)
# Unknown dispositions are a wiring bug: they land in the invariant category so they stay visible.
_CATEGORY_BY_DISPOSITION = {
    "fail_closed": "refusal",
    "degraded": "degrade",
    "narrowed": "fallback",
    "boundary": "invariant",
}


def record_fault(handler_id: str, error: BaseException, *, disposition: str, observed_at: str,
                 origin: str | None = None, fallback_id: str | None = None,
                 journal: object | None = None) -> bool:
    try:
        spec = lookup_handler(handler_id)
        error_name = type(error).__name__
        cause_code, code_rejected = _cause_code(error, error_name)
        category = _CATEGORY_BY_DISPOSITION.get(disposition, "invariant")
        expectation = _expectation(spec, error_name)
        op = spec.op if spec is not None else UNREGISTERED
        mismatch = spec is not None and disposition not in spec.dispositions
        cause_key = f"{handler_id}:{disposition}:{cause_code}"
        fields: dict[str, object] = {
            "handler_id": handler_id,
            "op": spec.op if spec is not None else None,
            "cause_code": cause_code,
            "origin": origin,
            "observed_at": observed_at,
            "expectation": expectation,
            "disposition": disposition,
            "disposition_mismatch": mismatch,
            "fault_code_rejected": code_rejected,
            "fallback_id": fallback_id,
            "policy_version": spec.policy_version if spec is not None else None,
        }
        # Hold the reason-log lock across the counter diff: emit_reason re-enters it (RLock), so a
        # self-error from another record_fault call cannot land between the two reads.
        with rl._LOCK:
            before = _self_errors()
            rl.emit_reason(category, cause_key, handler_id,
                           f"{op} {expectation} fault", _detail(error), journal=journal, **fields)
            return _self_errors() == before
    except _PASSTHROUGH:
        raise
    except BaseException:
        return False


def _cause_code(error: BaseException, error_name: str) -> tuple[str, bool]:
    """(cause_code, fault_code_rejected). Only a str fault_code matching FAULT_CODE_PATTERN is appended.

    The rejected value is never logged or formatted: str() is not called on it.
    """
    try:
        code = getattr(error, "fault_code", None)
    except Exception:
        return error_name, True
    if code is None:
        return error_name, False
    if isinstance(code, str) and FAULT_CODE_PATTERN.fullmatch(code):
        return f"{error_name}:{code}", False
    return error_name, True


def _expectation(spec: HandlerSpec | None, error_name: str) -> str:
    if spec is None:
        return UNREGISTERED
    if error_name in spec.expected:
        return "expected"
    if spec.requiredness == "UNCLASSIFIED":
        return "unclassified"
    return "unexpected"


def _detail(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return text or EMPTY_DETAIL


def _self_errors() -> int:
    return int(rl.run_summary().get("self_errors", 0))
