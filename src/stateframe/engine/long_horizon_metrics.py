"""Per-response long-horizon telemetry and assigned-cohort scoring.

Only canonical ``response_received`` events are converted into response rows.
Missing backend counters stay ``None``; in particular, request wall time is
never used as a decode-speed proxy.  The module is usable as an offline reader
for existing journals and as the opt-in annotation hook used by the runtime.
"""
from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from long_horizon_protocol import PROTOCOL_SWITCH, TELEMETRY_SWITCH  # noqa: F401
from kernel_prompt_accounting import accounting_tokens

_MATCHED_PROTOCOL_VALUES = frozenset({"1", "true", "yes", "on"})


def _matched_protocol_enabled() -> bool:
    return str(os.environ.get("STATEFRAME_MATCHED_PROTOCOL", "off")).strip().lower() in \
        _MATCHED_PROTOCOL_VALUES


def _apply_decode_logging_owner(
    row: Mapping[str, Any], event: Mapping[str, Any]
) -> dict[str, Any]:
    """Let the decode-logging lane own optional depth/rate vocabulary."""

    try:
        from long_horizon_decode_logging import enrich_response_record
    except ImportError:
        return dict(row)
    return enrich_response_record(row, event)


CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
_OVERFLOW_VALUES = {
    "context_overflow",
    "context overflow",
    "context-overflow",
    "context_overflowed",
    "context limit",
    "context_limit",
    "prompt too large",
    "prompt_too_large",
    "http 400 context",
}
_SUMMARY_ROLES = {
    "summary",
    "compaction",
    "history_compaction",
    "frame_compaction",
}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(mapping_values: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Any:
    for mapping in mapping_values:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
    return None


def _first_named(
    mapping_values: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> tuple[str | None, Any]:
    """Return the first present field together with its owning field name."""
    for mapping in mapping_values:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return key, mapping[key]
    return None, None


def _int(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _request_role(event: Mapping[str, Any]) -> str:
    for candidate in (
        event.get("request_role"),
        event.get("role"),
        event.get("purpose"),
        _mapping(event.get("request")).get("role"),
        _mapping(event.get("payload")).get("request_role"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    return "solver"


def _request_id(event: Mapping[str, Any]) -> str | None:
    for key in ("request_id", "attempt_id", "transport_attempt_id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _timing_maps(event: Mapping[str, Any], response: Mapping[str, Any], accounting: Mapping[str, Any]):
    direct = _mapping(response.get("timings"))
    raw = _mapping(accounting.get("raw_timings"))
    return (accounting, direct, raw)


def response_record(
    event: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Convert one canonical provider response into a compact metric row."""
    metadata = _mapping(metadata)
    response = _mapping(event.get("response"))
    usage = _mapping(response.get("usage"))
    accounting = _mapping(event.get("accounting"))
    raw_usage = _mapping(accounting.get("raw_usage"))
    timings = _timing_maps(event, response, accounting)

    # ``prompt_n`` is useful accounting even when it came from usage, but it
    # is not proof of the prompt that was actually rendered.  Only the
    # explicit rendered field or a raw backend timing counter may feed the
    # depth gate.  In particular, do not read ``accounting.prompt_n`` as a
    # rendered measurement: prompt-cache accounting intentionally fills that
    # field from usage when a server omits timings.
    explicit_depth = _int(event.get("rendered_prompt_tokens"))
    rendered_prompt_n = explicit_depth
    explicit_source = event.get("rendered_prompt_tokens_source")
    rendered_source = (
        explicit_source.strip()
        if explicit_depth is not None and isinstance(explicit_source, str)
        and explicit_source.strip()
        else "rendered_prompt_tokens" if explicit_depth is not None else None
    )
    if rendered_prompt_n is None:
        timing_name, timing_value = _first_named(
            (timings[1], timings[2]), ("rendered_prompt_tokens", "prompt_n")
        )
        rendered_prompt_n = _int(timing_value)
        if rendered_prompt_n is not None:
            rendered_source = (
                "backend_timing_rendered_prompt_tokens"
                if timing_name == "rendered_prompt_tokens"
                else "backend_timing_prompt_n"
            )

    accounting_prompt_n = _int(accounting.get("prompt_n"))
    usage_prompt_n = _int(_first((usage, raw_usage), ("prompt_tokens",)))
    prompt_n = rendered_prompt_n
    prompt_source = rendered_source
    if prompt_n is None and accounting_prompt_n is not None:
        prompt_n = accounting_prompt_n
        accounting_source = accounting.get("prompt_n_source")
        if isinstance(accounting_source, str) and accounting_source.strip():
            prompt_source = accounting_source.strip()
        elif usage_prompt_n is not None and accounting_prompt_n == usage_prompt_n:
            # Prompt-cache accounting may copy usage.prompt_tokens into its
            # historical prompt_n field.  Keep that provenance in the
            # opt-in long-horizon row without changing the canonical event.
            prompt_source = "usage_prompt_tokens"
        else:
            prompt_source = "accounting_prompt_n"
    if prompt_n is None and usage_prompt_n is not None:
        prompt_n = usage_prompt_n
        prompt_source = "usage_prompt_tokens"
    counts = accounting_tokens(accounting, response)
    cache_n = _int(counts["cached"])
    total_prompt_tokens = _int(counts["total"])
    completion_tokens = _int(
        _first(
            (*timings, usage, raw_usage),
            ("predicted_n", "decode_tokens", "completion_tokens"),
        )
    )
    decode_ms = _number(
        _first(
            (*timings, response),
            ("predicted_ms", "decode_ms", "completion_ms"),
        )
    )
    backend_decode_rate = _number(
        _first(
            (*timings, response),
            ("predicted_per_second", "decode_tok_s", "tokens_per_second"),
        )
    )
    if backend_decode_rate is not None and backend_decode_rate <= 0:
        backend_decode_rate = None
    if backend_decode_rate is None and completion_tokens is not None and decode_ms:
        backend_decode_rate = completion_tokens / (decode_ms / 1000.0)
        decode_rate_source = "derived_decode_tokens_over_decode_ms"
    elif backend_decode_rate is not None:
        decode_rate_source = "backend_decode_timing"
    else:
        decode_rate_source = None

    prompt_ms = _number(_first((*timings, response), ("prompt_ms", "prefill_ms")))
    elapsed_ns = _int(event.get("elapsed_ns"))
    latency_ms = elapsed_ns / 1_000_000.0 if elapsed_ns is not None else None
    uncached_prefill_n = _int(counts["uncached"])
    usable = _int(
        metadata.get(
            "usable_input_tokens",
            event.get("usable_input_tokens", event.get("context_usable_tokens")),
        )
    )
    fraction = (
        rendered_prompt_n / usable
        if rendered_prompt_n is not None and usable
        else None
    )

    task_id = metadata.get("task_id", event.get("task_id", event.get("task")))
    arm = metadata.get("arm", event.get("arm"))
    seed = metadata.get("seed", event.get("seed"))
    request_role = _request_role(event)
    row: dict[str, Any] = {
        "event_id": event.get("event_id"),
        "request_id": _request_id(event),
        "run_id": event.get("run_id", metadata.get("run_id")),
        "task_id": task_id if isinstance(task_id, str) else None,
        "arm": arm if isinstance(arm, str) else None,
        "seed": seed if type(seed) is int else None,
        "step": event.get("step") if type(event.get("step")) is int else None,
        "sequence": event.get("sequence") if type(event.get("sequence")) is int else None,
        "timestamp_ns": event.get("timestamp_ns") if type(event.get("timestamp_ns")) is int else None,
        "request_role": request_role,
        "rendered_prompt_tokens": rendered_prompt_n,
        "rendered_prompt_tokens_source": rendered_source,
        "context_depth_tokens": rendered_prompt_n,
        "rendered_context_depth_tokens": rendered_prompt_n,
        "rendered_context_depth_source": rendered_source,
        "depth_source": (
            rendered_source
            if rendered_prompt_n is not None
            else prompt_source
        ),
        "prompt_n_source": prompt_source,
        "usable_input_tokens": usable,
        "context_fraction": fraction,
        "prompt_n": prompt_n,
        "prompt_tokens": counts,
        "cache_n": cache_n,
        "uncached_prefill_n": uncached_prefill_n,
        "prompt_ms": prompt_ms,
        "decode_tokens": completion_tokens,
        "decode_ms": decode_ms,
        "decode_tok_s": backend_decode_rate,
        "decode_tok_s_source": decode_rate_source,
        "latency_ms": latency_ms,
        "compaction_ordinal": _int(
            event.get("compaction_ordinal", event.get("compaction_count"))
        ),
        "logical_depth_tokens": _int(event.get("logical_depth_tokens")),
        "terminal_result": event.get("terminal_result"),
        "failure_class": event.get("failure_class"),
        "overflow": is_context_overflow(event),
    }
    # A cost proxy is deliberately labeled as tokens.  It is not a dollar
    # estimate and it includes summary/retry responses when those events are
    # supplied to the scorer.
    # Finding 6: a missing prompt OR completion counter makes the proxy
    # UNKNOWN (None), never a silent zero.  The components are kept separate
    # because cached input, uncached input and output have different costs;
    # the unweighted sum is only a token proxy, never "measured compute".
    row["cost_input_cached_tokens"] = cache_n
    row["cost_input_uncached_tokens"] = uncached_prefill_n
    row["cost_output_tokens"] = completion_tokens
    row["cost_proxy_tokens"] = (
        total_prompt_tokens + completion_tokens
        if total_prompt_tokens is not None and completion_tokens is not None
        else None
    )
    row["backend_ms"] = (
        prompt_ms + decode_ms if prompt_ms is not None and decode_ms is not None else None
    )
    row["accounting_complete"] = row["cost_proxy_tokens"] is not None
    for key in ("cumulative_logical_depth_tokens", "cumulative_logical_depth_source",
                "history_summary_ordinal", "frame_epoch_ordinal", "hybrid_crossed",
                "hybrid_crossing", "presentation", "phase"):
        if key in event:
            row[key] = event[key]
    # decode_depth_logging owns the optional depth/rate vocabulary.
    return _apply_decode_logging_owner(row, event)


def iter_response_records(
    events: Iterable[Mapping[str, Any]], *, metadata: Mapping[str, Any] | None = None
) -> Iterable[dict[str, Any]]:
    """Yield one row per ``response_received`` event, never recursive usage hits."""
    metadata = dict(_mapping(metadata))
    request_roles: dict[str, str] = {}
    buffered = list(events)
    for event in buffered:
        if not isinstance(event, Mapping):
            continue
        if event.get("kind") in {"request_prepared", "request_submitted"}:
            request_id = _request_id(event)
            if request_id:
                request_roles[request_id] = _request_role(event)
    for event in buffered:
        if not isinstance(event, Mapping) or event.get("kind") != "response_received":
            continue
        event_metadata = dict(metadata)
        request_id = _request_id(event)
        if request_id and request_id in request_roles and "request_role" not in event:
            event = dict(event)
            event["request_role"] = request_roles[request_id]
        yield response_record(event, metadata=event_metadata)


def annotate_event(
    event: Mapping[str, Any], *, enabled: bool, metadata: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """Opt-in event hook; the disabled path returns the original object."""
    if not enabled or event.get("kind") != "response_received":
        return event
    result = dict(event)
    result["long_horizon"] = response_record(event, metadata=metadata)
    return result


def is_context_overflow(value: object) -> bool:
    """Recognize explicit overflow outcomes without treating all errors as overflow."""
    if isinstance(value, Mapping):
        for key in (
            "terminal_result",
            "failure_class",
            "outcome",
            "result",
            "status",
            "error",
        ):
            if is_context_overflow(value.get(key)):
                return True
        return False
    if isinstance(value, str):
        normalized = " ".join(value.strip().lower().replace("_", " ").split())
        return normalized == CONTEXT_OVERFLOW.lower().replace("_", " ") or normalized in _OVERFLOW_VALUES
    return False


def is_explicit_inconclusive(grade: Mapping[str, Any]) -> bool:
    """An explicit INCONCLUSIVE `outcome` dominates any affirmative field.

    Shared single source of truth (review10 long_horizon_protocol #1): both
    `_grade_correct` below and `grade_long_horizon.adapt_result` route through
    this so stale/conflicting `correct`/`public_pass`/`hidden_pass`/
    `accepted_done` fields can never override an explicit INCONCLUSIVE
    outcome. Does not touch overflow — `is_context_overflow` stays authoritative
    for that and is always checked first by callers.
    """
    return str(grade.get("outcome", "")).strip().upper() == "INCONCLUSIVE"


def _grade_correct(grade: Mapping[str, Any]) -> bool:
    if is_explicit_inconclusive(grade):
        return False
    explicit = grade.get("correct")
    if isinstance(explicit, bool):
        return explicit
    public_pass = grade.get("public_pass") is True
    hidden_pass = grade.get("hidden_pass") is True
    accepted_done = grade.get("accepted_done")
    if public_pass and hidden_pass and accepted_done is not False:
        return True
    return str(grade.get("outcome", "")).upper() in {"PASS", "CORRECT", "SUCCESS"}


def _cost_proxy(run: Mapping[str, Any]) -> int | float | None:
    rows = run.get("responses", run.get("telemetry"))
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        values = [row.get("cost_proxy_tokens") for row in rows if isinstance(row, Mapping)]
        if values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            return sum(values)
        if rows:
            return None
    value = run.get("cost_proxy_tokens")
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def score_run(run: Mapping[str, Any], *, grade: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Score one assigned run; overflow is never an infrastructure exclusion."""
    grade = _mapping(grade)
    overflow = is_context_overflow(run) or is_context_overflow(grade)
    correct = False if overflow else _grade_correct(grade)
    if overflow:
        failure_class = "context_overflow"
    elif correct:
        failure_class = None
    elif not grade:
        failure_class = "missing_grade"
    else:
        failure_class = "incorrect"
    return {
        "task_id": run.get("task_id", run.get("task")),
        "arm": run.get("arm"),
        "seed": run.get("seed"),
        "assigned": True,
        "excluded": False,
        "overflow": overflow,
        "correct": correct,
        "failure_class": failure_class,
        "cost_proxy_tokens": _cost_proxy(run),
    }


def aggregate_scores(scores: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(score) for score in scores]
    assigned = sum(1 for score in rows if score.get("assigned") is True)
    correct = sum(1 for score in rows if score.get("correct") is True)
    known_costs = [
        score.get("cost_proxy_tokens")
        for score in rows
        if isinstance(score.get("cost_proxy_tokens"), (int, float))
        and not isinstance(score.get("cost_proxy_tokens"), bool)
    ]
    total_cost = sum(known_costs) if len(known_costs) == len(rows) and rows else None
    report = {
        "assigned_runs": assigned,
        "correct_runs": correct,
        "overflow_failures": sum(1 for score in rows if score.get("overflow") is True),
        "failure_classes": dict(Counter(score.get("failure_class") for score in rows if score.get("failure_class"))),
        "total_cost_proxy_tokens": total_cost,
        "cost_per_correct": (
            total_cost / correct if total_cost is not None and correct else None
        ),
        "cost_per_correct_defined": total_cost is not None and correct > 0,
    }
    # The historical scorer remains byte-identical when the release-comparison
    # switch is off.  An explicitly matched evaluation additionally proves that
    # every assigned row supplied a cost and that failed rows stayed in the
    # numerator.
    if _matched_protocol_enabled():
        from matched_protocol import validate_score_rows

        report["matched_protocol"] = validate_score_rows(rows, report)
    return report


__all__ = [
    "CONTEXT_OVERFLOW",
    "aggregate_scores",
    "annotate_event",
    "is_context_overflow",
    "iter_response_records",
    "response_record",
    "score_run",
]
