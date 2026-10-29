"""Opt-in per-arm decode-depth performance accounting.

The normal kernel path does not import this module.  When the explicit
``LONG_HORIZON_DECODE_DEPTH_PER_ARM`` switch is on, the runtime records one
compact row for each provider response and one terminal ``run_perf`` record.
The module deliberately treats missing provider/server fields as unknown: it
must never turn an instrumentation gap into a fabricated speed measurement or
make a model request fail.
"""
from __future__ import annotations

import copy
import json
import math
import os
import statistics
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


SWITCH = "LONG_HORIZON_DECODE_DEPTH_PER_ARM"
SCHEMA = "long-horizon-decode-depth-per-arm/1"
SUMMARY_SCHEMA = "long-horizon-decode-depth-summary/1"
_TRUE = frozenset({"on", "1", "true", "yes"})
_DEPTH_KEYS = (
    "rendered_context_depth",
    "rendered_context_depth_tokens",
    "rendered_prompt_tokens",
    "context_depth_tokens",
)
_SLOT_COUNT_KEYS = (
    "slots_busy",
    "busy_slots",
    "active_slots",
    "concurrency",
    "n_busy",
    "busy_count",
    "processing_slots",
)
_SLOT_STATE_KEYS = ("is_processing", "processing", "busy", "active")
_BUSY_STATES = frozenset({"busy", "processing", "generating", "running", "working"})
_PREFILL_RATE_KEYS = (
    "prefill_tok_s",
    "prefill_per_second",
    "prompt_per_second",
    "prompt_tok_s",
    "prompt_tokens_per_second",
)


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether this instrumentation arm is explicitly enabled."""
    values = os.environ if environ is None else environ
    return str(values.get(SWITCH, "off")).strip().lower() in _TRUE


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def _env_integer(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value is not None and value.isdigit() else None


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _first(maps: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> tuple[str | None, Any]:
    for mapping in maps:
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return key, value
    return None, None


def _slot_state_is_busy(slot: Mapping[str, Any]) -> bool | None:
    for key in _SLOT_STATE_KEYS:
        value = slot.get(key)
        if isinstance(value, bool):
            return value
    for key in ("state", "status", "slot_state"):
        value = slot.get(key)
        if isinstance(value, str):
            return value.strip().lower() in _BUSY_STATES
    return None


def slots_busy(value: object) -> int | None:
    """Normalize a server ``/slots`` payload without guessing unknown state."""
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, Mapping):
        for key in _SLOT_COUNT_KEYS:
            count = _integer(value.get(key))
            if count is not None:
                return count
        for key in ("slots", "items", "data"):
            if key in value:
                count = slots_busy(value[key])
                if count is not None:
                    return count
        state = _slot_state_is_busy(value)
        return int(state) if state is not None else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        states = [
            _slot_state_is_busy(item)
            for item in value
            if isinstance(item, Mapping)
        ]
        known = [state for state in states if state is not None]
        return sum(1 for state in known if state) if known else None
    return None


def observe_slots(capture: object, endpoint: str, *, deadline: float | None = None) -> int | None:
    """Best-effort local slot occupancy observation.

    Only loopback endpoints are queried.  A missing endpoint, malformed
    response, timeout, or unsupported shape leaves concurrency unknown and the
    provider request proceeds unchanged.
    """
    if not enabled():
        return None
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            return None
        base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        timeout = min(2.0, deadline - time.monotonic()) if deadline is not None else 2.0
        if timeout <= 0:
            return None
        request = urllib.request.Request(base + "/slots", method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            return None
        count = slots_busy(json.loads(raw))
        if count is not None:
            setattr(capture, "decode_depth_per_arm_slots_busy", count)
            setattr(capture, "decode_depth_per_arm_slots_busy_source", "live_slots")
        return count
    except (OSError, TimeoutError, TypeError, ValueError):
        return None


def request_metadata(
    capture: object,
    messages: Sequence[Mapping[str, Any]],
    tools: object,
    *,
    rendered_context_depth: int,
    logical_messages: Sequence[Mapping[str, Any]] | None = None,
    logical_tools: object = None,
) -> dict[str, Any]:
    """Build journal-only metadata for a prepared request."""
    depth = _integer(rendered_context_depth)
    logical_depth = None
    try:
        from kernel_context import token_estimate

        logical_depth = token_estimate({
            "messages": list(logical_messages if logical_messages is not None else messages),
            "tools": tools if logical_tools is None else logical_tools,
        })
    except (ImportError, TypeError, ValueError):
        pass
    task_id = os.environ.get(
        "LONG_HORIZON_TASK_ID",
        getattr(getattr(capture, "journal", None), "task_id", None),
    )
    arm = _text(os.environ.get("LONG_HORIZON_ARM")) or _text(
        getattr(capture, "decode_depth_per_arm_arm", None)
    )
    live_busy = _integer(getattr(capture, "decode_depth_per_arm_slots_busy", None))
    configured = _env_integer("LONG_HORIZON_CONCURRENCY")
    concurrency = live_busy if live_busy is not None else configured
    concurrency_source = (
        getattr(capture, "decode_depth_per_arm_slots_busy_source", None)
        if live_busy is not None
        else "configured_environment" if configured is not None else None
    )
    return {
        "schema": SCHEMA,
        "request_id": getattr(capture, "request_id", None),
        "task_id": task_id,
        "arm": arm,
        "seed": _env_integer("REACT_SEED"),
        "request_role": getattr(capture, "long_horizon_request_role", "solver"),
        "rendered_context_depth": depth,
        "rendered_context_depth_tokens": depth,
        "rendered_context_depth_source": (
            "runtime_token_estimate" if depth is not None else None
        ),
        "logical_depth_tokens": _integer(logical_depth),
        "slots_busy": live_busy,
        "concurrency": concurrency,
        "concurrency_source": concurrency_source,
        "message_count": len(messages),
        "tool_count": len(tools) if isinstance(tools, Sequence) else None,
    }


def _nested_maps(event: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    response = _mapping(event.get("response"))
    accounting = _mapping(event.get("accounting"))
    return tuple(
        mapping
        for mapping in (
            event,
            metadata,
            _mapping(event.get("decode_depth_per_arm")),
            _mapping(event.get("long_horizon")),
            response,
            _mapping(response.get("timings")),
            accounting,
            _mapping(accounting.get("raw_timings")),
            _mapping(accounting.get("timings")),
        )
        if mapping
    )


def _capture_value(capture: object | None, name: str) -> object:
    return getattr(capture, name, None) if capture is not None else None


def response_record(
    event: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    capture: object | None = None,
) -> dict[str, Any]:
    """Normalize one response into the per-arm performance row."""
    metadata = _mapping(metadata)
    event = dict(event)
    nested = _mapping(event.get("decode_depth_per_arm"))
    for key in ("task_id", "arm", "seed"):
        if key not in event and key in nested:
            event[key] = nested[key]

    from long_horizon_metrics import response_record as canonical_response_record

    base = canonical_response_record(event, metadata=metadata)
    timing_maps = _nested_maps(event, metadata)
    prefill_rate_name, prefill_rate_value = _first(timing_maps, _PREFILL_RATE_KEYS)
    prefill_tok_s = _number(prefill_rate_value)
    prefill_rate_source = (
        "backend_prefill_timing" if prefill_tok_s is not None else None
    )
    prompt_n = _integer(base.get("prompt_n"))
    prefill_ms = _number(_first(timing_maps, ("prefill_ms", "prompt_ms"))[1])
    if prefill_tok_s is None and prompt_n is not None and prefill_ms is not None and prefill_ms > 0:
        prefill_tok_s = prompt_n / (prefill_ms / 1000.0)
        prefill_rate_source = "derived_prefill_tokens_over_prompt_ms"

    depth_name, depth_value = _first(timing_maps, _DEPTH_KEYS)
    depth = _integer(depth_value)
    depth_source = None
    if depth is not None:
        depth_source = _text(
            _first(timing_maps, (f"{depth_name}_source",))[1] if depth_name else None
        )
        depth_source = depth_source or depth_name
    capture_depth = _integer(
        _capture_value(capture, "decode_depth_per_arm_rendered_context_depth")
    )
    capture_depth_source = _text(
        _capture_value(capture, "decode_depth_per_arm_rendered_context_depth_source")
    )
    if capture_depth is not None:
        # Capacity preflight can replace the estimate with the exact
        # template/tokenizer count after prepare() has emitted its request
        # record.  Prefer that host-observed value for the response row.
        depth = capture_depth
        depth_source = capture_depth_source or "capture"
    elif depth is None:
        for name in (
            "decode_depth_per_arm_rendered_context_depth",
            "long_horizon_rendered_prompt_tokens",
        ):
            candidate = _integer(_capture_value(capture, name))
            if candidate is not None:
                depth = candidate
                depth_source = _text(
                    _capture_value(capture, f"{name}_source")
                ) or name
                break

    slot_name, slot_value = _first(timing_maps, _SLOT_COUNT_KEYS)
    concurrency = slots_busy(slot_value)
    concurrency_source = None
    if concurrency is not None:
        concurrency_source = _text(
            _first(timing_maps, (f"{slot_name}_source",))[1] if slot_name else None
        ) or slot_name
    if concurrency is None:
        live_busy = slots_busy(_capture_value(capture, "decode_depth_per_arm_slots_busy"))
        if live_busy is not None:
            concurrency = live_busy
            concurrency_source = _text(
                _capture_value(capture, "decode_depth_per_arm_slots_busy_source")
            ) or "live_slots"
    if concurrency is None:
        configured = _env_integer("LONG_HORIZON_CONCURRENCY")
        if configured is not None:
            concurrency = configured
            concurrency_source = "configured_environment"

    arm = _text(base.get("arm")) or _text(nested.get("arm")) or _text(
        os.environ.get("LONG_HORIZON_ARM")
    )
    task_id = _text(base.get("task_id")) or _text(nested.get("task_id")) or _text(
        os.environ.get("LONG_HORIZON_TASK_ID")
    )
    row = dict(base)
    row.update({
        "schema": SCHEMA,
        "arm": arm,
        "task_id": task_id,
        "prefill_ms": prefill_ms,
        "prefill_tok_s": prefill_tok_s,
        "prefill_tok_s_source": prefill_rate_source or prefill_rate_name,
        "rendered_context_depth": depth,
        "rendered_context_depth_tokens": depth,
        "rendered_context_depth_source": depth_source,
        "slots_busy": concurrency,
        "concurrency": concurrency,
        "concurrency_source": concurrency_source,
    })
    return row


def record_response(journal: object, row: Mapping[str, Any]) -> None:
    """Retain one response row for the terminal accounting record."""
    if not enabled():
        return
    rows = getattr(journal, "_decode_depth_per_arm_rows", None)
    if rows is None:
        rows = []
        setattr(journal, "_decode_depth_per_arm_rows", rows)
    rows.append(copy.deepcopy(dict(row)))


def run_accounting(journal: object) -> dict[str, Any]:
    rows = getattr(journal, "_decode_depth_per_arm_rows", None)
    rows = rows if isinstance(rows, list) else []
    return {"schema": SCHEMA, "response_count": len(rows), "responses": copy.deepcopy(rows)}


def emit_run_accounting(journal: object) -> None:
    """Write one terminal accounting event when the arm is enabled."""
    if not enabled() or getattr(journal, "_decode_depth_per_arm_emitted", False):
        return
    setattr(journal, "_decode_depth_per_arm_emitted", True)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(
            "decode_depth_per_arm_accounting",
            producer="decode_depth_per_arm",
            run_perf=run_accounting(journal),
        )


def iter_response_rows(events: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield rows from response or terminal accounting journal records."""
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if event.get("kind") == "response_received":
            nested = event.get("decode_depth_per_arm")
            if isinstance(nested, Mapping):
                yield response_record(event)
        elif event.get("kind") == "decode_depth_per_arm_accounting":
            run_perf = _mapping(event.get("run_perf"))
            rows = run_perf.get("responses")
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
                for row in rows:
                    if isinstance(row, Mapping) and row.get("schema") == SCHEMA:
                        yield dict(row)


def _metric(values: Iterable[object]) -> dict[str, Any]:
    numbers = sorted(
        float(value)
        for value in values
        if _number(value) is not None
    )
    if not numbers:
        return {"count": 0, "max": None, "mean": None, "median": None, "min": None}
    return {
        "count": len(numbers),
        "max": numbers[-1],
        "mean": sum(numbers) / len(numbers),
        "median": statistics.median(numbers),
        "min": numbers[0],
    }


def _row_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    request_id = _text(row.get("request_id"))
    if request_id is not None:
        # The response event has a journal event_id while the terminal copy is
        # captured before that ID exists.  Request identity is the stable
        # cross-record key and prevents terminal accounting from double-counting
        # the live response row.
        return ("request", row.get("run_id"), request_id, row.get("step"))
    event_id = _text(row.get("event_id"))
    if event_id is not None:
        return ("event", event_id)
    return ("row", json.dumps(dict(row), sort_keys=True, default=str))


def _dedupe(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize response rows by arm, active concurrency, and context depth."""
    supplied = [dict(row) for row in rows if isinstance(row, Mapping)]
    journal_rows = list(iter_response_rows(supplied))
    normalized_rows = [row for row in supplied if row.get("schema") == SCHEMA]
    clean = _dedupe(journal_rows or normalized_rows or supplied)
    by_arm: dict[str, list[dict[str, Any]]] = {}
    by_cell: dict[tuple[str, int | None, int | None], list[dict[str, Any]]] = {}
    for row in clean:
        arm = _text(row.get("arm")) or "UNKNOWN"
        concurrency = _integer(row.get("concurrency", row.get("slots_busy")))
        depth = _integer(
            row.get("rendered_context_depth", row.get("rendered_context_depth_tokens"))
        )
        by_arm.setdefault(arm, []).append(row)
        by_cell.setdefault((arm, concurrency, depth), []).append(row)

    arms: dict[str, Any] = {}
    for arm in sorted(by_arm):
        arm_rows = by_arm[arm]
        cells = []
        cell_keys = sorted(
            (key for key in by_cell if key[0] == arm),
            key=lambda key: (key[1] is None, key[1] if key[1] is not None else -1,
                             key[2] is None, key[2] if key[2] is not None else -1),
        )
        for _arm, concurrency, depth in cell_keys:
            cell_rows = by_cell[(_arm, concurrency, depth)]
            cells.append({
                "arm": arm,
                "concurrency": concurrency,
                "context_depth": depth,
                "responses": len(cell_rows),
                "decode_tok_s": _metric(row.get("decode_tok_s") for row in cell_rows),
                "prefill_ms": _metric(row.get("prefill_ms") for row in cell_rows),
                "prefill_tok_s": _metric(row.get("prefill_tok_s") for row in cell_rows),
            })
        arms[arm] = {
            "responses": len(arm_rows),
            "decode_tok_s": _metric(row.get("decode_tok_s") for row in arm_rows),
            "prefill_ms": _metric(row.get("prefill_ms") for row in arm_rows),
            "prefill_tok_s": _metric(row.get("prefill_tok_s") for row in arm_rows),
            "missing_context_depth": sum(
                _integer(row.get("rendered_context_depth", row.get("rendered_context_depth_tokens")))
                is None for row in arm_rows
            ),
            "missing_concurrency": sum(
                _integer(row.get("concurrency", row.get("slots_busy"))) is None
                for row in arm_rows
            ),
            "cells": cells,
        }
    return {
        "schema": SUMMARY_SCHEMA,
        "row_schema": SCHEMA,
        "response_count": len(clean),
        "arms": arms,
    }


__all__ = [
    "SCHEMA",
    "SUMMARY_SCHEMA",
    "SWITCH",
    "emit_run_accounting",
    "enabled",
    "iter_response_rows",
    "observe_slots",
    "record_response",
    "request_metadata",
    "response_record",
    "slots_busy",
    "summarize_rows",
]
