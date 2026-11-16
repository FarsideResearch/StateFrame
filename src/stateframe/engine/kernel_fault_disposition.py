"""Typed fault recording for registered guard handlers, layered on kernel_reason_log.emit_reason.

record_fault classifies a caught error against the handler's HandlerSpec and writes ONE reason row per
(category, handler, disposition, cause) per run (repeats only count). It is logging only: it never raises
from its own failures and never changes control flow. KeyboardInterrupt, SystemExit and CancelledError
raised by the recording path propagate. Such exceptions passed as the error argument are recorded like
any other fault. Callers keep their own passthrough before the broad catch.

Each row's fields form the fault record (persisted in problems.jsonl under "fields"):
  fault_id           sha256(handler_id|disposition|cause_code|observed_at|occurrence)[:16]. No time, uuid or
                     random input: the same inputs in the same call order give the same id.
  kind               registry taxonomy letter (A/B/C/D), or "unregistered".
  exception_class    type name of the caught error.
  exception_message  whitespace-normalized, at most MAX_MESSAGE_CHARS; EMPTY_DETAIL when empty;
                     SECRET_WITHHELD_DETAIL when it holds a registered secret.
  site_id            the handler id (the guarded site).
  required_work_id   HandlerSpec.required_work_id, or None.
  cause_chain        JSON text of the class names along __cause__/__context__, outermost first, at most
                     MAX_CHAIN. The reason log stringifies lists, so this is stored as text.
  primary_fault_id   fault_id computed with the innermost chain class; equals fault_id with no chain.

Return value: True when the record was written or counted, False when the reason log could not record it
(emit_reason self-error). Step-3 call sites decide what False means (T0 pauses sends).
The self-error check runs under kernel_reason_log._LOCK, so a self-error from another record_fault call
cannot flip this result. Reason-log writes that do not take the lock can still bump the shared counter,
and then the result errs toward False.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys

import kernel_reason_log as rl
from kernel_fault_registry import HandlerSpec, lookup_handler

UNREGISTERED = "unregistered"
EMPTY_DETAIL = "<empty message>"
# Stored instead of a message that contains a registered secret or cannot be rendered.
SECRET_WITHHELD_DETAIL = "<message withheld: contains a registered secret>"
UNRENDERABLE_DETAIL = "<message withheld: str() failed>"
MAX_MESSAGE_CHARS = 300
MAX_CHAIN = 5
FAULT_ID_CHARS = 16
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
        detail = _detail(error)
        chain = _cause_chain(error)
        # Hold the reason-log lock across the occurrence read and the counter diff: emit_reason re-enters
        # it (RLock), so a concurrent record_fault cannot change the occurrence index or land a self-error
        # between the reads.
        with rl._LOCK:
            occurrence = _prior_occurrences(category, cause_key)
            fault_id = _fault_id(handler_id, disposition, cause_code, observed_at, occurrence)
            primary_id = (_fault_id(handler_id, disposition, chain[-1], observed_at, occurrence)
                          if chain else fault_id)
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
                "fault_id": fault_id,
                "kind": spec.taxonomy if spec is not None else UNREGISTERED,
                "exception_class": error_name,
                "exception_message": _bounded(detail),
                "site_id": handler_id,
                "required_work_id": spec.required_work_id if spec is not None else None,
                "cause_chain": json.dumps(chain),
                "primary_fault_id": primary_id,
            }
            before = _self_errors()
            rl.emit_reason(category, cause_key, handler_id,
                           f"{op} {expectation} fault", detail, journal=journal, **fields)
            return _self_errors() == before
    except _PASSTHROUGH:
        raise
    except BaseException:
        return False


def _fault_id(handler_id: str, disposition: str, cause_code: str, observed_at: str, occurrence: int) -> str:
    material = "|".join((handler_id, disposition, cause_code, str(observed_at), str(occurrence)))
    return hashlib.sha256(material.encode("utf-8", "surrogatepass")).hexdigest()[:FAULT_ID_CHARS]


def _prior_occurrences(category: str, cause_key: str) -> int:
    """How many times this (category, cause_key) was already recorded in the current reason-log state.

    Read under kernel_reason_log._LOCK. A fresh state (reset_for_tests) restarts at 0, so identical
    inputs in identical order reproduce identical fault ids.
    """
    entry = rl._STATE["causes"].get((category, cause_key))
    return int(entry["count"]) if entry is not None else 0


def _cause_chain(error: BaseException) -> list[str]:
    """Class names along __cause__, else __context__ (unless suppressed), outermost first. Cycle-safe."""
    chain: list[str] = []
    seen = {id(error)}
    current: BaseException = error
    while len(chain) < MAX_CHAIN:
        nxt = current.__cause__
        if nxt is None and not current.__suppress_context__:
            nxt = current.__context__
        if nxt is None or id(nxt) in seen:
            break
        seen.add(id(nxt))
        chain.append(type(nxt).__name__)
        current = nxt
    return chain


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
    """The row's message: never a registered secret, never an exception from a hostile __str__."""
    try:
        text = " ".join(str(error).split())
    except Exception:  # noqa: BLE001 - an error's own __str__ must not break fault recording
        return UNRENDERABLE_DETAIL
    if text and _contains_registered_secret(text):
        return SECRET_WITHHELD_DETAIL
    return text or EMPTY_DETAIL


def _contains_registered_secret(text: str) -> bool:
    """Check the text with the checkpoint byte guard (raw and escaped forms of every registered secret).

    Only an already loaded checkpoint module can hold registered secrets, so this never imports it. When
    the check itself fails the message is treated as secret-bearing (fail closed: withhold it).
    """
    module = sys.modules.get("kernel_checkpoint") or sys.modules.get("stateframe.engine.kernel_checkpoint")
    if module is None or not getattr(module, "_SECRETS", None):
        return False
    try:
        module._check_registered_secrets(text.encode("utf-8", "surrogatepass"))
    except module.CheckpointError:
        return True
    except Exception:  # noqa: BLE001 - unknown guard failure: withhold rather than risk a leak
        return True
    return False


def _bounded(text: str) -> str:
    """At most MAX_MESSAGE_CHARS, ending in an ellipsis when cut."""
    if len(text) <= MAX_MESSAGE_CHARS:
        return text
    return text[: MAX_MESSAGE_CHARS - 3] + "..."


def _self_errors() -> int:
    return int(rl.run_summary().get("self_errors", 0))
