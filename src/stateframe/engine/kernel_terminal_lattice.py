"""Opt-in monotone join for terminal submission labels.

This module guards final terminal projections only. Intermediate runtime outcomes
remain writable so recoverable control flow can clear transient stop signals.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping


MODE_ENV = "STATEFRAME_TERMINAL_LATTICE"
RECEIPT_SCHEMA = "stateframe-terminal-disposition/1"
STORED = "STORED"
NOT_STORED = "NOT_STORED"
MAX_WRITER = 120
MAX_EVENT_TEXT = 100

# Canonical stop labels are ranked by terminal strength. Same-rank labels remain
# stable at their first observed value; evidence and certification are not ranked.
STOP_REASON_RANKS = {
    "UNKNOWN": 0,
    "RUN_INCOMPLETE": 10,
    "RUN_FAILED": 10,
    "ATTEMPT_BUDGET_EXHAUSTED": 10,
    "RESPONSE_BUDGET_EXHAUSTED": 10,
    "TOOL_CALL_CAP": 10,
    "TOOL_CALL_LOOP_BREAKER": 10,
    "CONTEXT_OVERFLOW": 10,
    "STOPPED_INCOMPLETE": 10,
    "CONTINUATION_BLOCKED": 10,
    "BLOCKED": 10,
    "CANCELLED": 10,
    "DEADLINE_EXCEEDED": 10,
    "AUTHORITY_FAILURE": 10,
    "HARNESS_ERROR": 10,
    "ENVIRONMENT_BROKEN": 10,
    "FINALIZATION_ERROR": 10,
    "HARDEN8_CONTRACT_ERROR": 10,
    "HARDEN8_EVIDENCE_PIN_ERROR": 10,
    "HARDEN8_OWNER_FINALIZATION_ERROR": 10,
    "HARDEN8_ROUTE_ERROR": 10,
    "NO_CHANGES": 10,
    "BOUNDED": 10,
    "INVALIDATE_ATTEMPT": 10,
    "STALLED": 10,
    "CEILING_REACHED": 10,
    "POSTGREEN_NO_PROGRESS": 10,
    "EFFECT_BOUNDARY_SHADOW": 10,
    "COMPLETION_OWNER_EXHAUSTED": 10,
    "CONTINUATION_INCOMPLETE": 10,
    "POST_GREEN_STALL": 10,
    "QUARANTINE_HARNESS": 10,
    "STEP_BUDGET_EXHAUSTED": 10,
    "UNCERTIFIED_SUBMIT": 20,
    "SUBMITTED": 30,
    "DONE_ACCEPTED": 40,
}

# Keep aliases explicit so newly introduced writer values are visible in review.
STOP_REASON_ALIASES = {
    "": "UNKNOWN",
    "UNKNOWN": "UNKNOWN",
    "RUN_INCOMPLETE": "RUN_INCOMPLETE",
    "INCOMPLETE": "RUN_INCOMPLETE",
    "NOT_COMPLETED": "RUN_INCOMPLETE",
    "TIMEOUT": "RUN_INCOMPLETE",
    "TIMED_OUT": "RUN_INCOMPLETE",
    "TIMEOUT_EXHAUSTED": "RUN_INCOMPLETE",
    "STOPPED": "RUN_INCOMPLETE",
    "STOPPED_INCOMPLETE": "STOPPED_INCOMPLETE",
    "RUN_FAILED": "RUN_FAILED",
    "FAILED": "RUN_FAILED",
    "FAIL": "RUN_FAILED",
    "FAILURE": "RUN_FAILED",
    "ERROR": "RUN_FAILED",
    "FINALIZATION_ERROR": "FINALIZATION_ERROR",
    "HARDEN8_CONTRACT_ERROR": "HARDEN8_CONTRACT_ERROR",
    "HARDEN8_EVIDENCE_PIN_ERROR": "HARDEN8_EVIDENCE_PIN_ERROR",
    "HARDEN8_OWNER_FINALIZATION_ERROR": "HARDEN8_OWNER_FINALIZATION_ERROR",
    "HARDEN8_ROUTE_ERROR": "HARDEN8_ROUTE_ERROR",
    "HARNESS_ERROR": "HARNESS_ERROR",
    "HARNESS_SUSPECT": "HARNESS_ERROR",
    "ENV_BROKEN": "ENVIRONMENT_BROKEN",
    "ENVIRONMENT_BROKEN": "ENVIRONMENT_BROKEN",
    "REPRO_ENVIRONMENT": "ENVIRONMENT_BROKEN",
    "ATTEMPT_BUDGET_EXHAUSTED": "ATTEMPT_BUDGET_EXHAUSTED",
    "BUDGET_EXHAUSTED": "ATTEMPT_BUDGET_EXHAUSTED",
    "RESPONSE_BUDGET_EXHAUSTED": "RESPONSE_BUDGET_EXHAUSTED",
    "TOOL_CALL_CAP": "TOOL_CALL_CAP",
    "TOOL_CALL_CAP_HIT": "TOOL_CALL_CAP",
    "TOOL_CALL_LOOP_BREAKER": "TOOL_CALL_LOOP_BREAKER",
    "CONTEXT_OVERFLOW": "CONTEXT_OVERFLOW",
    "CONTINUATION_BLOCKED": "CONTINUATION_BLOCKED",
    "BLOCKED": "BLOCKED",
    "NEEDS_WORK": "RUN_INCOMPLETE",
    "RED": "RUN_FAILED",
    "CANCELLED": "CANCELLED",
    "CANCELED": "CANCELLED",
    "DEADLINE_EXCEEDED": "DEADLINE_EXCEEDED",
    "AUTHORITY_FAILURE": "AUTHORITY_FAILURE",
    "NO_CHANGES": "NO_CHANGES",
    "BOUNDED": "BOUNDED",
    "INVALIDATE_ATTEMPT": "INVALIDATE_ATTEMPT",
    "INVALIDATED_ATTEMPT": "INVALIDATE_ATTEMPT",
    "STALLED": "STALLED",
    "CEILING_REACHED": "CEILING_REACHED",
    "POSTGREEN_NO_PROGRESS": "POSTGREEN_NO_PROGRESS",
    "EFFECT_BOUNDARY_SHADOW": "EFFECT_BOUNDARY_SHADOW",
    "COMPLETION_OWNER_EXHAUSTED": "COMPLETION_OWNER_EXHAUSTED",
    "CONTINUATION_INCOMPLETE": "CONTINUATION_INCOMPLETE",
    "POST_GREEN_STALL": "POST_GREEN_STALL",
    "QUARANTINE_HARNESS": "QUARANTINE_HARNESS",
    "STEP_BUDGET_EXHAUSTED": "STEP_BUDGET_EXHAUSTED",
    "SYSTEM_OUTCOME": "UNKNOWN",
    "UNCERTIFIED_SUBMIT": "UNCERTIFIED_SUBMIT",
    "SUBMIT_PENDING": "UNCERTIFIED_SUBMIT",
    "SUBMITTED": "SUBMITTED",
    "SUBMISSION_ACCEPTED": "SUBMITTED",
    "ACCEPTED": "DONE_ACCEPTED",
    "DONE": "DONE_ACCEPTED",
    "DONE_ACCEPTED": "DONE_ACCEPTED",
}
_STATUS_KEYS = (
    "result", "outcome", "result_class", "status", "terminal_status",
    "stop_cause", "stop_reason", "system_outcome",
)
_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


def mode(value: object = None) -> str:
    """Return the strict switch value, failing open to off on invalid input."""
    raw = os.environ.get(MODE_ENV, "off") if value is None else value
    normalized = str(raw or "off").strip().lower()
    return normalized if normalized in {"off", "on"} else "off"


def enabled(runtime: object = None) -> bool:
    """Return whether the terminal lattice is enabled for this runtime."""
    configured = (
        getattr(runtime, "stateframe_terminal_lattice_mode", None)
        if runtime is not None else None
    )
    return mode(configured) == "on" if configured is not None else mode() == "on"


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalise(value: object) -> str:
    return _text(value).upper().replace("-", "_").replace(" ", "_")


def stop_reason_rank(value: object) -> int:
    """Return the explicit terminal rank for a stop reason or writer alias."""
    normalized = _normalise(value)
    canonical = STOP_REASON_ALIASES.get(normalized)
    if canonical is None and normalized.startswith("HARDEN8_"):
        # Hardening failures are stop labels, never successful submissions.
        canonical = (
            "ENVIRONMENT_BROKEN"
            if "ENV" in normalized or "HARNESS" in normalized
            else "FINALIZATION_ERROR"
        )
    return STOP_REASON_RANKS.get(canonical or "UNKNOWN", 0)


def _receipt(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    nested = value.get("terminal_disposition")
    if isinstance(nested, Mapping) and nested.get("schema") == RECEIPT_SCHEMA:
        return nested
    if value.get("schema") == RECEIPT_SCHEMA:
        return value
    return None


def _sources(value: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    nested = _receipt(value)
    if nested is None or nested is value:
        return (value,)
    return (nested, value)


def _statuses(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for source in _sources(value):
        for key in _STATUS_KEYS:
            text = _text(source.get(key))
            if text:
                result.setdefault(key, text)
    return result


def _facts(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {
            "accepted": None, "delivery": None, "artifact": None,
            "stop_reason": "", "stop_rank": 0, "status_values": {},
            "artifact_digest": "",
        }
    sources = _sources(value)
    accept_values: list[bool] = []
    delivery_values: dict[str, list[str]] = {"delivery": [], "artifact": []}
    reasons: list[tuple[int, int, int, str]] = []
    digests: list[str] = []
    statuses: dict[str, str] = {}
    reason_priority = {
        "stop_reason": 4, "stop_cause": 3, "result": 2, "outcome": 2,
        "result_class": 1, "status": 1, "terminal_status": 1,
        "system_outcome": 1,
    }
    reason_order = 0
    for source in sources:
        if source.get("accepted") is True or source.get("acceptance") == "ACCEPTED":
            accept_values.append(True)
        elif source.get("accepted") is False or source.get("acceptance") == "UNACCEPTED":
            accept_values.append(False)
        for key in delivery_values:
            label = _text(source.get(key)).upper()
            if label in {STORED, NOT_STORED}:
                delivery_values[key].append(label)
        for key in _STATUS_KEYS:
            label = _text(source.get(key))
            if label:
                statuses.setdefault(key, label)
                reasons.append((
                    stop_reason_rank(label), reason_priority[key], -reason_order, label,
                ))
                reason_order += 1
        digest = _text(source.get("artifact_digest"))
        if digest:
            digests.append(digest)
    stop_rank, _, _, stop_reason = max(reasons, default=(0, 0, 0, ""))
    accepted = True if True in accept_values else (
        False if False in accept_values else None
    )
    if accepted is True and stop_rank < STOP_REASON_RANKS["SUBMITTED"]:
        stop_reason = "SUBMITTED"
        stop_rank = STOP_REASON_RANKS["SUBMITTED"]
    return {
        "accepted": accepted,
        "delivery": STORED if STORED in delivery_values["delivery"] else (
            NOT_STORED if NOT_STORED in delivery_values["delivery"] else None),
        "artifact": STORED if STORED in delivery_values["artifact"] else (
            NOT_STORED if NOT_STORED in delivery_values["artifact"] else None),
        "stop_reason": stop_reason,
        "stop_rank": stop_rank,
        "status_values": statuses,
        "artifact_digest": digests[0] if digests else "",
    }


def _remember_digest(
    digest_value: object,
    facts: dict[str, object],
    digests: list[str],
) -> None:
    digest = _text(digest_value)
    if not digest:
        return
    digests.append(digest)
    if not _text(facts.get("artifact_digest")):
        facts["artifact_digest"] = digest


def _prior_facts(
    runtime: object,
    explicit_previous: object,
) -> tuple[dict[str, object], Mapping[str, object] | None, tuple[str, ...]]:
    try:
        state = getattr(runtime, "_terminal_lattice_state", None) if runtime is not None else None
    except Exception:
        state = None
    values = [explicit_previous, state]
    if runtime is not None:
        try:
            values.append(getattr(runtime, "_terminal_disposition_receipt", None))
        except Exception:
            pass
    facts = {
        "accepted": None, "delivery": None, "artifact": None,
        "stop_reason": "", "stop_rank": 0, "status_values": {},
        "artifact_digest": "",
    }
    digests: list[str] = []
    prior_receipt: Mapping[str, object] | None = None
    for value in values:
        if not isinstance(value, Mapping):
            continue
        if value is state:
            state_status = value.get("status_values")
            if isinstance(state_status, Mapping):
                for key, label in state_status.items():
                    if key in _STATUS_KEYS and _text(label):
                        facts["status_values"].setdefault(key, _text(label))
            _remember_digest(value.get("artifact_digest"), facts, digests)
            for item in value.get("artifact_digests", ()):
                _remember_digest(item, facts, digests)
            state_receipt = value.get("receipt")
            if isinstance(state_receipt, Mapping):
                prior_receipt = state_receipt
            accepted = value.get("accepted")
            if accepted is True:
                facts["accepted"] = True
            elif accepted is False and facts["accepted"] is None:
                facts["accepted"] = False
            for key in ("delivery", "artifact"):
                if value.get(key) in {STORED, NOT_STORED}:
                    if facts[key] != STORED:
                        facts[key] = value[key]
            label = _text(value.get("stop_reason"))
            rank = stop_reason_rank(label)
            if rank > int(facts["stop_rank"]):
                facts["stop_reason"], facts["stop_rank"] = label, rank
            continue
        candidate = _facts(value)
        if candidate["accepted"] is True:
            facts["accepted"] = True
        elif candidate["accepted"] is False and facts["accepted"] is None:
            facts["accepted"] = False
        for key in ("delivery", "artifact"):
            if candidate[key] == STORED or (
                facts[key] is None and candidate[key] == NOT_STORED
            ):
                facts[key] = candidate[key]
        if int(candidate["stop_rank"]) > int(facts["stop_rank"]):
            facts["stop_reason"] = candidate["stop_reason"]
            facts["stop_rank"] = candidate["stop_rank"]
        for key, label in candidate["status_values"].items():
            facts["status_values"].setdefault(key, label)
        _remember_digest(candidate["artifact_digest"], facts, digests)
        receipt = _receipt(value)
        if receipt is not None and prior_receipt is None:
            prior_receipt = receipt
    return facts, prior_receipt, tuple(dict.fromkeys(digests))


def _safe_event_text(value: object) -> str:
    text = _TOKEN_RE.sub("_", str(value or ""))[:MAX_EVENT_TEXT]
    return text or "unknown"


def _emit_refusal(runtime: object, old: object, new: object, writer: str) -> None:
    try:
        if runtime is None:
            return
        journal = getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if not callable(emit):
            emit = getattr(runtime, "emit", None)
        if not callable(emit):
            return
        emit(
            "terminal_lower_refused",
            **{
                "from": _safe_event_text(old),
                "to": _safe_event_text(new),
                "writer": _safe_event_text(writer)[:MAX_WRITER],
            },
        )
    except Exception:
        # The join is a fail-open projection guard; journaling cannot block it.
        return


def _digest_receipt(receipt: Mapping[str, object]) -> str:
    payload = dict(receipt)
    payload.pop("receipt_digest", None)
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _join_fields(
    candidate: Mapping[str, object],
    previous: Mapping[str, object],
) -> tuple[dict[str, object], list[tuple[object, object]]]:
    joined: dict[str, object] = dict(candidate)
    current = _facts(candidate)
    refusals: list[tuple[object, object]] = []

    if current["accepted"] is True:
        joined["accepted"] = True
        joined["acceptance"] = "ACCEPTED"
    elif previous.get("accepted") is True:
        if current["accepted"] is False:
            refusals.append(("ACCEPTED", "UNACCEPTED"))
        joined["accepted"] = True
        joined["acceptance"] = "ACCEPTED"

    current_reason = _text(current.get("stop_reason"))
    candidate_stop_rank = int(current.get("stop_rank", 0) or 0)
    if current_reason and (
        current["accepted"] is True
        or candidate_stop_rank >= STOP_REASON_RANKS["UNCERTIFIED_SUBMIT"]
    ):
        if stop_reason_rank(_text(joined.get("stop_reason"))) < candidate_stop_rank:
            joined["stop_reason"] = current_reason
        for key in _STATUS_KEYS:
            if key == "stop_reason":
                continue
            value = _text(joined.get(key))
            if value and stop_reason_rank(value) < candidate_stop_rank:
                refusals.append((current_reason, value))
                joined[key] = current_reason

    for key in ("delivery", "artifact"):
        prior_label = previous.get(key)
        candidate_label = current.get(key)
        if prior_label == STORED and candidate_label != STORED:
            if candidate_label == NOT_STORED:
                refusals.append((STORED, NOT_STORED))
            joined[key] = STORED

    prior_reason = _text(previous.get("stop_reason"))
    prior_rank = int(previous.get("stop_rank", stop_reason_rank(prior_reason)) or 0)
    if prior_rank > candidate_stop_rank and prior_reason:
        refusals.append((prior_reason, current_reason or "UNKNOWN"))
        joined["stop_reason"] = prior_reason
    elif prior_rank and not current_reason:
        joined["stop_reason"] = prior_reason

    previous_statuses = previous.get("status_values")
    if isinstance(previous_statuses, Mapping):
        for key in _STATUS_KEYS:
            current_value = _text(joined.get(key))
            current_rank = stop_reason_rank(current_value)
            old_value = _text(previous_statuses.get(key))
            old_rank = stop_reason_rank(old_value)
            if old_value and current_value and old_rank > current_rank:
                if key in {"result", "outcome", "result_class", "status",
                           "terminal_status", "stop_cause", "system_outcome"}:
                    joined[key] = old_value
            elif (old_value and not current_value and key in joined):
                joined[key] = old_value

    if prior_rank > candidate_stop_rank and prior_reason:
        for key in _STATUS_KEYS:
            current_value = _text(joined.get(key))
            if not current_value or stop_reason_rank(current_value) >= prior_rank:
                continue
            old_value = (
                _text(previous_statuses.get(key))
                if isinstance(previous_statuses, Mapping) else ""
            )
            joined[key] = (
                old_value
                if old_value and stop_reason_rank(old_value) >= prior_rank
                else prior_reason
            )

    prior_digest = _text(previous.get("artifact_digest"))
    if prior_digest and not _text(joined.get("artifact_digest")):
        joined["artifact_digest"] = prior_digest
        refusals.append((prior_digest, "MISSING"))

    return joined, refusals


def _join_terminal_impl(
    runtime: object,
    candidate: Mapping[str, object],
    *,
    writer: str,
    previous: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Join a terminal projection with prior accepted/delivery/stop labels.

    With the feature disabled, a dict input is returned unchanged so the
    default-OFF path preserves its existing serialized output byte for byte.
    """
    if not isinstance(candidate, Mapping):
        return {}
    if not enabled(runtime):
        return candidate if isinstance(candidate, dict) else dict(candidate)

    prior, prior_receipt, prior_digests = _prior_facts(runtime, previous)
    joined, refusals = _join_fields(candidate, prior)

    current_receipt = _receipt(candidate)
    receipt_output: Mapping[str, object] | None = None
    if current_receipt is not None and current_receipt is not candidate:
        receipt_joined, receipt_refusals = _join_fields(current_receipt, prior)
        refusals.extend(receipt_refusals)
        receipt_digest = _digest_receipt(receipt_joined)
        if _text(receipt_joined.get("receipt_digest")) != receipt_digest:
            receipt_joined["receipt_digest"] = receipt_digest
        joined["terminal_disposition"] = receipt_joined
        receipt_output = receipt_joined
    elif _text(candidate.get("schema")) == RECEIPT_SCHEMA:
        receipt_digest = _digest_receipt(joined)
        if _text(joined.get("receipt_digest")) != receipt_digest:
            joined["receipt_digest"] = receipt_digest
        receipt_output = joined

    try:
        system_outcome = (
            _text(getattr(runtime, "system_outcome", None)) if runtime is not None else ""
        )
    except Exception:
        system_outcome = ""
    if ("system_outcome" not in candidate and system_outcome
            and stop_reason_rank(system_outcome) < int(prior.get("stop_rank", 0) or 0)):
        refusals.append((prior.get("stop_reason", "UNKNOWN"), system_outcome))
    for old, new in refusals[:16]:
        _emit_refusal(runtime, old, new, writer)

    final_facts = _facts(joined)
    final_digest = _text(final_facts.get("artifact_digest"))
    observed_digests = list(prior_digests)
    if final_digest:
        observed_digests.append(final_digest)
    status_values = dict(prior.get("status_values", {}))
    status_values.update(_statuses(joined))
    state_receipt = receipt_output or prior_receipt
    state = {
        "accepted": (
            True if prior.get("accepted") is True or final_facts["accepted"] is True
            else final_facts["accepted"]
        ),
        "delivery": (
            STORED if prior.get("delivery") == STORED or final_facts["delivery"] == STORED
            else final_facts["delivery"]
        ),
        "artifact": (
            STORED if prior.get("artifact") == STORED or final_facts["artifact"] == STORED
            else final_facts["artifact"]
        ),
        "stop_reason": (
            prior["stop_reason"]
            if int(prior.get("stop_rank", 0) or 0) > int(final_facts["stop_rank"] or 0)
            else final_facts["stop_reason"] or prior.get("stop_reason", "")
        ),
        "stop_rank": max(
            int(prior.get("stop_rank", 0) or 0),
            int(final_facts.get("stop_rank", 0) or 0),
        ),
        "status_values": status_values,
        "artifact_digest": final_digest or prior.get("artifact_digest", ""),
        "artifact_digests": list(dict.fromkeys(observed_digests)),
        "receipt": dict(state_receipt) if isinstance(state_receipt, Mapping) else None,
    }
    try:
        setattr(runtime, "_terminal_lattice_state", state)
        if receipt_output is not None:
            setattr(runtime, "_terminal_disposition_receipt", dict(receipt_output))
        system_outcome = _text(getattr(runtime, "system_outcome", None))
        prior_reason = _text(state.get("stop_reason"))
        if system_outcome and stop_reason_rank(system_outcome) < int(state["stop_rank"]):
            setattr(runtime, "system_outcome", prior_reason)
    except Exception:
        # Host objects may be immutable in adapters; returning the joined result
        # remains useful and must never fail terminal finalization.
        pass
    return joined


def join_terminal(
    runtime: object,
    candidate: Mapping[str, object],
    *,
    writer: str,
    previous: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fail-open public join used by every terminal projection."""
    if not isinstance(candidate, Mapping):
        return {}
    unchanged = candidate if isinstance(candidate, dict) else None
    try:
        return _join_terminal_impl(
            runtime, candidate, writer=writer, previous=previous,
        )
    except Exception:
        if unchanged is not None:
            return unchanged
        try:
            return dict(candidate)
        except Exception:
            return {}
