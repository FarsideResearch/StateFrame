"""Optional, advisory taxonomy for every terminal worker run.

The taxonomy is deliberately a journal-side observer.  It does not change a
completion decision, reopen a run, or add prompt content.  When enabled it
records one bounded ``stop_reason_classified`` event immediately before the
existing ``attempt_terminated`` event so resource stops that followed a green
observation remain visible as potential lost wins.
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

MODE_ENV = "STATEFRAME_STOP_REASON_TAXONOMY"
LAST_N_ENV = "STATEFRAME_STOP_REASON_LAST_N"
SCHEMA = "stateframe-stop-reason-taxonomy/1"
EVENT_KIND = "stop_reason_classified"
PRODUCER = "kernel_stop_reason_taxonomy"
DEFAULT_LAST_N = 8
MAX_LAST_N = 32
MAX_SCAN_EVENTS = 4096
MAX_TEXT = 160

DONE_ACCEPTED = "DONE_ACCEPTED"
RESPONSE_BUDGET_EXHAUSTED = "RESPONSE_BUDGET_EXHAUSTED"
TOOL_CALL_CAP = "TOOL_CALL_CAP"
TOOL_CALL_LOOP_BREAKER = "TOOL_CALL_LOOP_BREAKER"
CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
ATTEMPT_BUDGET_EXHAUSTED = "ATTEMPT_BUDGET_EXHAUSTED"
HARNESS_ERROR = "HARNESS_ERROR"
ENVIRONMENT_BROKEN = "ENVIRONMENT_BROKEN"
STOPPED_INCOMPLETE = "STOPPED_INCOMPLETE"
CONTINUATION_BLOCKED = "CONTINUATION_BLOCKED"
BLOCKED = "BLOCKED"
CANCELLED = "CANCELLED"
DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
RUN_FAILED = "RUN_FAILED"
RUN_INCOMPLETE = "RUN_INCOMPLETE"
AUTHORITY_FAILURE = "AUTHORITY_FAILURE"
UNKNOWN = "UNKNOWN"

BUDGET_STOP_REASONS = frozenset({
    RESPONSE_BUDGET_EXHAUSTED,
    TOOL_CALL_CAP,
    CONTEXT_OVERFLOW,
    ATTEMPT_BUDGET_EXHAUSTED,
})
STOP_REASONS = frozenset({
    DONE_ACCEPTED,
    *BUDGET_STOP_REASONS,
    TOOL_CALL_LOOP_BREAKER,
    HARNESS_ERROR,
    ENVIRONMENT_BROKEN,
    STOPPED_INCOMPLETE,
    CONTINUATION_BLOCKED,
    BLOCKED,
    CANCELLED,
    DEADLINE_EXCEEDED,
    RUN_FAILED,
    RUN_INCOMPLETE,
    AUTHORITY_FAILURE,
    UNKNOWN,
})

_GREEN_VALUES = frozenset({
    "ACCEPTED", "DONE", "DONE_ACCEPTED", "GREEN", "PASS", "PASS_BEHAVIOURAL",
    "SUBMITTED", "SUCCESS", "SATISFIED",
})
_RED_VALUES = frozenset({
    "BLOCKED", "ERROR", "FAIL", "FAILED", "FAILURE", "INCOMPLETE", "NEEDS_WORK",
    "RED", "UNSATISFIED", "VIOLATED",
})
_DECISION_KINDS = frozenset({
    "action_outcome_closed",
    "budget_failure",
    "completion_authority_decision",
    "completion_reducer_decision",
    "done_facts_finalized",
    "history_decision",
    "latest_exchange_decision",
    "progress_guard_decision",
    "public_test_verdict",
    "response_budget_exhausted",
    "supervision_stop",
    "test_subprocess_completed",
    "tool_applicability_decision",
    "tool_call_cap_hit",
    "unified_stop_decision",
    "v2_done_decision",
})
_SAFE_FIELDS = frozenset({
    "accepted", "action", "admissible", "assessment", "behavior", "behaviour",
    "decision", "exit_code", "green", "kind", "mode", "operation", "outcome",
    "passed", "phase", "reason", "reason_code", "reason_codes", "request_id",
    "response_budget", "responses_used", "result", "result_class", "returncode",
    "selected", "selected_action", "sequence", "snapshot_id", "state", "status",
    "step", "system_outcome", "test", "tool", "tool_call_cap", "tool_calls_used",
    "event_id",
})
_NESTED_FIELDS = frozenset({"authority", "completion", "decision", "done_facts", "record"})


def mode(value: object = None) -> str:
    """Return the strict opt-in switch value."""
    raw = os.environ.get(MODE_ENV, "off") if value is None else value
    result = str(raw or "off").strip().lower()
    if result not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return result


def decision_window(value: object = None) -> int:
    """Return the bounded number of terminal decisions to retain."""
    raw = os.environ.get(LAST_N_ENV, str(DEFAULT_LAST_N)) if value is None else value
    if isinstance(raw, bool):
        raise ValueError(f"{LAST_N_ENV} must be an integer from 1 to {MAX_LAST_N}")
    try:
        result = int(str(raw).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{LAST_N_ENV} must be an integer from 1 to {MAX_LAST_N}") from error
    if not 1 <= result <= MAX_LAST_N:
        raise ValueError(f"{LAST_N_ENV} must be an integer from 1 to {MAX_LAST_N}")
    return result


def enabled(runtime: object = None) -> bool:
    marker = (getattr(runtime, "stateframe_stop_reason_taxonomy_mode", None)
              if runtime is not None else None)
    return mode(marker) == "on"


def _normalise(value: object) -> str:
    return (str(value).strip().upper().replace("-", "_").replace(" ", "_")
            if value is not None else "")


def _reason_from_value(value: object) -> str | None:
    key = _normalise(value)
    aliases = {
        "DONE": DONE_ACCEPTED,
        "DONE_ACCEPTED": DONE_ACCEPTED,
        "SUBMITTED": DONE_ACCEPTED,
        "ACCEPTED": DONE_ACCEPTED,
        "GREEN": DONE_ACCEPTED,
        "PASS": DONE_ACCEPTED,
        "PASS_BEHAVIOURAL": DONE_ACCEPTED,
        "SUCCESS": DONE_ACCEPTED,
        "SATISFIED": DONE_ACCEPTED,
        "RESPONSE_BUDGET": RESPONSE_BUDGET_EXHAUSTED,
        "RESPONSE_BUDGET_EXHAUSTED": RESPONSE_BUDGET_EXHAUSTED,
        "RESPONSE_BUDGET_HIT": RESPONSE_BUDGET_EXHAUSTED,
        "TOOL_CALL_CAP": TOOL_CALL_CAP,
        "TOOL_CALL_CAP_HIT": TOOL_CALL_CAP,
        "TOOL_CAP": TOOL_CALL_CAP,
        "TOOL_CALL_LIMIT": TOOL_CALL_CAP,
        "TOOL_CALL_LOOP_BREAKER": TOOL_CALL_LOOP_BREAKER,
        "TOOL_LOOP_BREAKER": TOOL_CALL_LOOP_BREAKER,
        "LOOP_BREAKER": TOOL_CALL_LOOP_BREAKER,
        "CONTEXT_OVERFLOW": CONTEXT_OVERFLOW,
        "CONTEXT_EXHAUSTED": CONTEXT_OVERFLOW,
        "CONTEXT_LIMIT": CONTEXT_OVERFLOW,
        "RENDER_OVERFLOW": CONTEXT_OVERFLOW,
        "CEILING_REACHED": CONTEXT_OVERFLOW,
        "OVERFLOW": CONTEXT_OVERFLOW,
        "FRAME_BUDGET_EXHAUSTED": CONTEXT_OVERFLOW,
        "BUDGET_EXHAUSTED": ATTEMPT_BUDGET_EXHAUSTED,
        "FAILED_BUDGET": ATTEMPT_BUDGET_EXHAUSTED,
        "FUEL_EXHAUSTED": ATTEMPT_BUDGET_EXHAUSTED,
        "BOUNDED": ATTEMPT_BUDGET_EXHAUSTED,
        "ATTEMPT_BUDGET": ATTEMPT_BUDGET_EXHAUSTED,
        "ATTEMPT_BUDGET_EXHAUSTED": ATTEMPT_BUDGET_EXHAUSTED,
        "PARENT_BUDGET_EXHAUSTED": ATTEMPT_BUDGET_EXHAUSTED,
        "HARNESS_ERROR": HARNESS_ERROR,
        "HARNESS_SUSPECT": HARNESS_ERROR,
        "SETUP_ERROR": HARNESS_ERROR,
        "QUARANTINE_HARNESS": HARNESS_ERROR,
        "INVALIDATE_ATTEMPT": HARNESS_ERROR,
        "ENV_BROKEN": ENVIRONMENT_BROKEN,
        "ENVIRONMENT_BROKEN": ENVIRONMENT_BROKEN,
        "STOPPED_INCOMPLETE": STOPPED_INCOMPLETE,
        "UNCERTIFIED_SUBMIT": STOPPED_INCOMPLETE,
        "NO_PROGRESS_EDIT_CAP": STOPPED_INCOMPLETE,
        "STOPPED": STOPPED_INCOMPLETE,
        "SOLVER_STOPPED": STOPPED_INCOMPLETE,
        "STALLED": STOPPED_INCOMPLETE,
        "V2_TERMINAL": STOPPED_INCOMPLETE,
        "CONTINUATION_BLOCKED": CONTINUATION_BLOCKED,
        "DELIVERY_BLOCKED": CONTINUATION_BLOCKED,
        "BLOCKED": BLOCKED,
        "CONTROL_CONFLICT": BLOCKED,
        "CANCELLED": CANCELLED,
        "CANCELED": CANCELLED,
        "DEADLINE": DEADLINE_EXCEEDED,
        "FAILED_DEADLINE": DEADLINE_EXCEEDED,
        "DEADLINE_EXCEEDED": DEADLINE_EXCEEDED,
        "FAIL": RUN_FAILED,
        "FAILED": RUN_FAILED,
        "RUN_FAILED": RUN_FAILED,
        "AUTHORITY_FAILURE": AUTHORITY_FAILURE,
        "HARDEN8_AUTHORITY_FAILURE": AUTHORITY_FAILURE,
        "HARDEN8_CONTRACT_ERROR": AUTHORITY_FAILURE,
        "HARDEN8_EVIDENCE_PIN_ERROR": AUTHORITY_FAILURE,
        "HARDEN8_OWNER_FINALIZATION_ERROR": AUTHORITY_FAILURE,
        "REQUEST_ACTION_CONTRACT_ERROR": AUTHORITY_FAILURE,
        "INCOMPLETE": RUN_INCOMPLETE,
        "NEEDS_WORK": RUN_INCOMPLETE,
        "UNKNOWN": UNKNOWN,
    }
    return aliases.get(key)


def _mapping_values(value: object) -> Iterable[Mapping[str, object]]:
    if not isinstance(value, Mapping):
        return ()
    nested = []
    for name in ("authority", "completion", "completion_decision", "decision",
                 "done_facts", "delivery_manifest", "facts", "public_test_verdict",
                 "projection", "record", "result", "verification", "verdict"):
        child = value.get(name)
        if isinstance(child, Mapping):
            nested.append(child)
    return (value, *nested)


def _status_values(result: Mapping[str, object]) -> tuple[object, ...]:
    values: list[object] = []
    for item in _mapping_values(result):
        for name in ("stop_reason", "system_outcome", "outcome", "result", "result_class",
                     "status", "state", "terminal_status", "code", "behaviour", "behavior",
                     "reason", "reason_codes", "completion_owner_reason"):
            value = item.get(name)
            if isinstance(value, (list, tuple, set, frozenset)):
                values.extend(value)
            else:
                values.append(value)
    return tuple(value for value in values if value is not None)


def _event_reason(event: Mapping[str, object]) -> str | None:
    kind = _normalise(event.get("kind"))
    direct = _reason_from_value(kind)
    if direct is not None and direct != DONE_ACCEPTED:
        return direct
    if (kind in {"BUDGET_FAILURE", "ATTEMPT_TERMINATED", "SUPERVISION_STOP"}
            or kind.lower() in _DECISION_KINDS or kind.endswith("_DECISION")):
        for value in _status_values(event):
            reason = _reason_from_value(value)
            if reason is not None and reason != DONE_ACCEPTED:
                return reason
    return None


def _accepted_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _normalise(value) in {"TRUE", "YES", "ACCEPTED", "GREEN", "PASS", "SATISFIED"}


def _done_seal_follows_latest_done(events: Sequence[Mapping[str, object]]) -> bool:
    """Reject stale seals when journal history contains a later done attempt."""
    latest_done = None
    latest_seal = None
    for index, event in enumerate(events):
        kind = _normalise(event.get("kind"))
        record = event.get("record")
        if (kind == "ACTION_OUTCOME_CLOSED" and isinstance(record, Mapping)
                and record.get("tool") == "done"):
            latest_done = index
        elif kind == "DONE_SEALED":
            latest_seal = index
    return latest_done is None or latest_seal is not None and latest_seal > latest_done


def _acceptance_source(result: Mapping[str, object], events: Sequence[Mapping[str, object]],
                       runtime: object = None) -> str | None:
    """Accept sealed submission as done; otherwise retain legacy acceptance sources."""
    from kernel_done_is_submit_authority import done_sealed

    if (done_sealed(runtime) is not None
            or isinstance(result.get("done_seal"), Mapping)
            and result["done_seal"].get("schema") == "stateframe-done-seal/1"
            and result["done_seal"].get("stored") is True):
        if _done_seal_follows_latest_done(events):
            return "done_seal"
    from kernel_done_is_submit_authority import enabled as done_submit_enabled
    # Default-on done is accepted only through its stored seal. Older decisions
    # remain readable under the explicit OFF parity switch.
    if done_submit_enabled(runtime):
        return None
    if "terminal_disposition" in result:
        from kernel_terminal_disposition import CERTIFIED, VERIFIED_FIXED, receipt_from
        receipt = receipt_from(result)
        if (receipt is not None
                and receipt.get("certification") == CERTIFIED
                and receipt.get("repair") == VERIFIED_FIXED
                and receipt.get("accepted") is True):
            return "terminal_disposition"
        # A receipt is the authority.  Do not fall through to an older event or
        # decision that can re-label the same rejected terminal as accepted.
        return None
    for item in _mapping_values(result):
        values = {_normalise(item.get(key)) for key in
                  ("result", "outcome", "status", "result_class", "terminal_status")}
        if values & {"UNCERTIFIED_SUBMIT", "DONE_ACCEPTED", "ACCEPTED_DONE"}:
            return "accepted_done"
    for event in reversed(events):
        kind = _normalise(event.get("kind"))
        if kind == "SUBMISSION_IDEMPOTENCY" and event.get("phase") == "CONFIRMED":
            return "keyed_submission"
        if (kind == "CONTINUATION_TRANSITION" and event.get("operation") == "complete"
                and isinstance(event.get("data"), Mapping)
                and event["data"].get("submission_key")):
            return "keyed_submission"
        if kind == "ACTION_RECEIPT":
            receipt = event.get("receipt")
            if (isinstance(receipt, Mapping)
                    and _normalise(receipt.get("kind")) == "DONE"
                    and _normalise(receipt.get("result_class")) in {"ACCEPTED", "UNCERTIFIED_SUBMIT"}):
                return "accepted_done"
        if kind in {"COMPLETION_REDUCER_DECISION", "COMPLETION_AUTHORITY_DECISION"}:
            nested = event.get("decision", event.get("authority", event))
            if isinstance(nested, Mapping) and nested.get("accepted") is True:
                return "accepted_done"
    if runtime is not None:
        decision = getattr(runtime, "_last_completion_decision", None)
        if bool(getattr(decision, "accepted", False)):
            return "accepted_done"
    return None


def _green_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _normalise(value) in _GREEN_VALUES


def _accepted_at_stop(result: Mapping[str, object], events: Sequence[Mapping[str, object]],
                      runtime: object = None) -> bool:
    return _acceptance_source(result, events, runtime) is not None


def _trusted_green_source(result: Mapping[str, object], events: Sequence[Mapping[str, object]],
                          runtime: object = None) -> str | None:
    if runtime is not None:
        verdict_fn = getattr(runtime, "public_test_verdict", None)
        binding_fn = getattr(runtime, "binding", None)
        if callable(verdict_fn) and callable(binding_fn):
            try:
                verdict = verdict_fn()
                current = binding_fn()
                if (getattr(verdict, "state", None) == "SATISFIED"
                        and getattr(verdict, "current_binding", None) == current
                        and getattr(verdict, "trusted_receipts", ())):
                    return "runtime_trusted_verdict"
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
    for event in reversed(events):
        kind = _normalise(event.get("kind"))
        if kind not in {"PUBLIC_TEST_VERDICT", "CONTINUATION_PUBLIC_CERTIFICATION"}:
            continue
        values = tuple(item for item in _mapping_values(event))
        for item in values:
            public = item.get("public_test_verdict")
            if (isinstance(public, Mapping)
                    and _normalise(public.get("state")) == "SATISFIED"):
                return "journal_trusted_verdict"
            if kind == "PUBLIC_TEST_VERDICT" and _normalise(item.get("state")) == "SATISFIED":
                return "journal_trusted_verdict"
    public_verdict = result.get("public_test_verdict")
    if isinstance(public_verdict, Mapping) and _normalise(public_verdict.get("state")) == "SATISFIED":
        return "terminal_trusted_verdict"
    return None


def _event_green_state(event: Mapping[str, object]) -> bool | None:
    """Return the latest host verdict carried by an event, if it has one."""
    kind = _normalise(event.get("kind"))
    relevant = (kind in {"ACTION_OUTCOME_CLOSED", "ATTEMPT_TERMINATED",
                         "COMPLETION_AUTHORITY_DECISION",
                         "COMPLETION_REDUCER_DECISION", "DONE_FACTS_FINALIZED",
                         "PUBLIC_TEST_VERDICT", "TEST_SUBPROCESS_COMPLETED"}
                or kind.endswith("_VERDICT") or kind.endswith("_DECISION"))
    if not relevant:
        return None
    for item in _mapping_values(event):
        for key in ("accepted", "green", "passed"):
            if key in item:
                return _accepted_value(item[key])
        for key in ("result", "result_class", "status", "outcome", "behaviour", "behavior"):
            if key not in item:
                continue
            value = _normalise(item[key])
            if value in _GREEN_VALUES:
                return True
            if value in _RED_VALUES:
                return False
        if item.get("exit_code") is not None:
            return item.get("exit_code") == 0
        if item.get("returncode") is not None:
            return item.get("returncode") == 0
    return None


def _result_green_state(result: Mapping[str, object]) -> bool | None:
    for item in _mapping_values(result):
        for key in ("accepted", "admissible", "green", "passed"):
            if key in item:
                return _accepted_value(item[key])
        for key in ("result", "result_class", "status", "outcome", "behaviour", "behavior"):
            if key not in item:
                continue
            value = _normalise(item[key])
            if value in _GREEN_VALUES:
                return True
            if value in _RED_VALUES:
                return False
    return None


def _green_at_stop(result: Mapping[str, object], events: Sequence[Mapping[str, object]],
                   runtime: object = None) -> bool:
    return _trusted_green_source(result, events, runtime) is not None


def _compact_scalar(value: object) -> object:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value[:MAX_TEXT]
    if isinstance(value, (list, tuple)):
        return [_compact_scalar(item) for item in value[:12]
                if isinstance(item, (str, int, float, bool)) or item is None]
    return None


def _compact_mapping(value: Mapping[str, object], *, depth: int = 0) -> dict[str, object]:
    compact: dict[str, object] = {}
    for key, item in value.items():
        if key not in _SAFE_FIELDS and not (depth == 0 and key in _NESTED_FIELDS):
            continue
        if isinstance(item, Mapping) and depth < 1:
            compact[key] = _compact_mapping(item, depth=depth + 1)
        else:
            scalar = _compact_scalar(item)
            if scalar is not None or item is None:
                compact[key] = scalar
    return compact


def _is_decision(event: Mapping[str, object]) -> bool:
    kind = str(event.get("kind", "")).lower()
    return (kind in _DECISION_KINDS or kind.endswith("_decision")
            or "_decision_" in kind)


def _decision_tail(events: Sequence[Mapping[str, object]], last_n: int) -> list[dict[str, object]]:
    decisions = []
    for event in events:
        if not _is_decision(event):
            continue
        compact = _compact_mapping(event)
        kind = event.get("kind")
        if isinstance(kind, str):
            compact = {"kind": kind, **compact}
        decisions.append(compact)
    return decisions[-last_n:]


def _open_id_diagnostic(
        terminal: Mapping[str, object], runtime: object = None,
) -> tuple[list[str], str]:
    """Keep unknown open-work metadata distinct from a verified empty set."""
    explicit = terminal.get("open_ids")
    if isinstance(explicit, (list, tuple)) and not isinstance(explicit, (str, bytes, bytearray)):
        ids = [str(item) for item in explicit if str(item).strip()]
        status = str(terminal.get("open_ids_status") or ("known" if ids else "provided_empty"))
        return ids, status
    if runtime is not None:
        try:
            from kernel_completion_owner import open_ids as owner_open_ids
            ids = [str(item) for item in owner_open_ids(runtime) if str(item).strip()]
            if ids:
                return ids, "known"
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
        snapshot = getattr(runtime, "_completion_owner_last_open_ids", ())
        if isinstance(snapshot, (list, tuple)) and snapshot:
            return [str(item) for item in snapshot if str(item).strip()], "last_authoritative"
        if bool(getattr(runtime, "_completion_owner_open_ids_verified_empty", False)):
            return [], "verified_empty"
    return [], "unknown"


def _classify_reason(result: Mapping[str, object], system_outcome: object,
                     events: Sequence[Mapping[str, object]], *, accepted: bool = False,
                     acceptance_source: str | None = None) -> tuple[str, list[str]]:
    evidence: list[str] = []
    if acceptance_source == "done_seal":
        # A stored model seal is terminal authority; prior refused done attempts
        # remain history and cannot relabel the accepted final submission.
        return DONE_ACCEPTED, evidence
    explicit_values = []
    if system_outcome is not None:
        explicit_values.append(system_outcome)
    explicit_values.extend(_status_values(result))
    for value in explicit_values:
        reason = _reason_from_value(value)
        if reason is None:
            continue
        if reason == DONE_ACCEPTED:
            continue
        return reason, evidence
    for event in reversed(events):
        reason = _event_reason(event)
        if reason is None:
            continue
        kind = event.get("kind")
        if isinstance(kind, str) and kind not in evidence:
            evidence.append(kind)
        return reason, evidence
    if accepted:
        return DONE_ACCEPTED, evidence
    for value in explicit_values:
        key = _normalise(value)
        if key and key not in {"PASS", "DONE", "GREEN"}:
            return (RUN_FAILED if key in {"ERROR", "FAILURE"} else UNKNOWN), evidence
    return UNKNOWN, evidence


def classify_run_end(result: Mapping[str, object] | None, *,
                     system_outcome: object = None,
                     events: Iterable[Mapping[str, object]] = (),
                     runtime: object = None,
                     last_n: object = None) -> dict[str, object]:
    """Classify one terminal result without changing it or its journal."""
    from kernel_terminal_lattice import join_terminal

    terminal = join_terminal(
        runtime, dict(result) if isinstance(result, Mapping) else {},
        writer="kernel_stop_reason_taxonomy.classify_run_end",
    )
    observed = tuple(item for item in events if isinstance(item, Mapping))[-MAX_SCAN_EVENTS:]
    window = decision_window(last_n)
    acceptance_source = _acceptance_source(terminal, observed, runtime)
    green_source = _trusted_green_source(terminal, observed, runtime)
    stop_reason, evidence = _classify_reason(
        terminal, system_outcome, observed, accepted=acceptance_source is not None,
        acceptance_source=acceptance_source)
    green = _green_at_stop(terminal, observed, runtime)
    accepted = acceptance_source is not None
    open_ids, open_ids_status = _open_id_diagnostic(terminal, runtime)
    open_text = ",".join(open_ids) if open_ids else (
        "none" if open_ids_status == "verified_empty" else "unknown"
    )
    fastloop_summary = (
        f"acceptance={'accepted' if accepted else 'not_accepted'}"
        f" source={acceptance_source or 'none'}"
        f" green={'trusted' if green else 'not_trusted'}"
        f" stop={stop_reason}"
        f" open_ids={open_text}"
    )
    classified = {
        "stop_reason": stop_reason if stop_reason in STOP_REASONS else UNKNOWN,
        "stop_reason_classified": True,
        "stop_cause": stop_reason if stop_reason in STOP_REASONS else UNKNOWN,
        "accepted": accepted,
        "acceptance_source": acceptance_source,
        "green_at_stop": green,
        "green_source": green_source,
        "lost_win": green and not accepted and stop_reason in BUDGET_STOP_REASONS,
        "fastloop_summary": fastloop_summary,
        "system_outcome": str(system_outcome) if system_outcome is not None else None,
        "open_ids": open_ids,
        "open_ids_status": open_ids_status,
        "terminal_status": next((str(value)[:MAX_TEXT] for value in _status_values(terminal)
                                  if value is not None), None),
        "evidence_kinds": evidence,
        "decision_count": sum(1 for event in observed if _is_decision(event)),
        "decision_window": window,
        "last_decisions": _decision_tail(observed, window),
    }
    return join_terminal(
        runtime, classified, writer="kernel_stop_reason_taxonomy.classify_run_end",
    )


def _journal_events(path: object) -> list[dict[str, object]]:
    if path is None:
        return []
    records: deque[dict[str, object]] = deque(maxlen=MAX_SCAN_EVENTS)
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except (TypeError, ValueError, UnicodeError):
                    continue
                if isinstance(item, dict):
                    records.append(item)
    except (OSError, TypeError, ValueError, UnicodeError):
        return []
    return list(records)


def _runtime_events(runtime: object) -> list[dict[str, object]]:
    journal = getattr(runtime, "journal", None)
    path = getattr(journal, "path", None)
    if path is not None:
        return _journal_events(path)
    records = getattr(journal, "records", None)
    if callable(records):
        try:
            return [dict(item) for item in records() if isinstance(item, Mapping)][-MAX_SCAN_EVENTS:]
        except (OSError, RuntimeError, TypeError, ValueError):
            return []
    return []


def record(runtime: object, result: Mapping[str, object] | None) -> dict[str, object] | None:
    """Emit one taxonomy event when enabled; observability is fail-open."""
    if not enabled(runtime):
        return None
    try:
        payload = classify_run_end(
            result,
            system_outcome=getattr(runtime, "system_outcome", None),
            events=_runtime_events(runtime),
            runtime=runtime,
            last_n=getattr(runtime, "stateframe_stop_reason_last_n", None),
        )
        journal = getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if not callable(emit):
            return payload
        emit(EVENT_KIND, producer=PRODUCER, payload_schema=SCHEMA,
             mechanism_id="stop_reason_taxonomy", **payload)
        return payload
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError):
        return None


def summarize(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate classifications for a lane report or historical audit."""
    by_reason: dict[str, int] = {}
    green_budget_exhausted = 0
    lost_wins = 0
    green_stops = 0
    unclassified = 0
    runs = 0
    for item in records:
        if not isinstance(item, Mapping):
            continue
        runs += 1
        reason = str(item.get("stop_reason", UNKNOWN))
        by_reason[reason] = by_reason.get(reason, 0) + 1
        green = bool(item.get("green_at_stop", False))
        if green:
            green_stops += 1
        if green and reason in BUDGET_STOP_REASONS:
            green_budget_exhausted += 1
        if bool(item.get("lost_win", False)):
            lost_wins += 1
        if reason == UNKNOWN:
            unclassified += 1
    return {
        "schema": SCHEMA,
        "runs": runs,
        "by_stop_reason": by_reason,
        "green_stops": green_stops,
        "green_budget_exhausted": green_budget_exhausted,
        "lost_wins": lost_wins,
        "unclassified": unclassified,
    }


def summarize_journals(paths: Iterable[str | os.PathLike[str]], *, last_n: object = None) -> dict[str, object]:
    """Classify terminal events from existing JSONL journals without mutation."""
    source_paths = tuple(paths)
    classifications: list[dict[str, object]] = []
    for path in source_paths:
        events = _journal_events(path)
        for index, event in enumerate(events):
            if event.get("kind") != "attempt_terminated":
                continue
            result = event.get("result")
            terminal = result if isinstance(result, Mapping) else {}
            classifications.append(classify_run_end(
                terminal, system_outcome=event.get("system_outcome"),
                events=events[:index + 1], last_n=last_n))
    report = summarize(classifications)
    report["sources"] = len(source_paths)
    return report
