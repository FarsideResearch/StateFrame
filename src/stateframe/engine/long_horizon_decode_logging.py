"""Opt-in per-response decode/depth accounting for long-horizon runs.

This module is deliberately independent of the transport and journal classes.  It
normalizes the small set of fields needed by offline accounting and supplies the
runtime hooks with a single, conservative schema.  The runtime only calls these
helpers when ``LONG_HORIZON_TELEMETRY`` is enabled.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "long-horizon-decode-depth/1"
_TRUE = {"on", "1", "true", "yes"}
_DEPTH_KEYS = (
    "rendered_context_depth_tokens",
    "rendered_context_depth",
    "rendered_prompt_tokens",
)
_RATE_KEYS = (
    "prefill_tok_s",
    "prefill_per_second",
    "prompt_per_second",
    "prompt_tok_s",
    "prompt_tokens_per_second",
)
_SLOT_COUNT_KEYS = (
    "slots_busy", "busy_slots", "active_slots", "concurrency",
    "n_busy", "busy_count", "processing_slots",
)
_SLOT_STATE_KEYS = ("is_processing", "processing", "busy", "active")
_BUSY_STATES = {"busy", "processing", "generating", "running", "working"}


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether this opt-in annotation path is enabled."""
    values = os.environ if environ is None else environ
    return str(values.get("LONG_HORIZON_TELEMETRY", "off")).strip().lower() in _TRUE


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def _integer(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _first(maps: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> tuple[str | None, Any]:
    for mapping in maps:
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return key, value
    return None, None


def _nested_maps(event: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        mapping for mapping in (
            event,
            _mapping(event.get("decode_depth")),
            _mapping(event.get("long_horizon_decode")),
            _mapping(event.get("long_horizon")),
            _mapping(event.get("run_perf")),
            _mapping(event.get("accounting")),
        )
        if mapping
    )


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
    """Normalize a server ``/slots`` response or a direct occupancy count.

    Unknown shapes return ``None`` rather than guessing that every configured
    slot is busy.  A list is counted only when at least one entry has a known
    busy/idle state.
    """
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
        states = [_slot_state_is_busy(item) for item in value if isinstance(item, Mapping)]
        known = [state for state in states if state is not None]
        return sum(1 for state in known if state) if known else None
    return None


def observe_slots(capture: object, endpoint: str, *, deadline: float | None = None) -> int | None:
    """Best-effort occupancy read for a local study server.

    Slot telemetry must not turn a provider response into a transport failure:
    unsupported endpoints, timeouts, and malformed metadata leave the value
    unknown and the response continues normally.
    """
    if not enabled():
        return None
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
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
        value = json.loads(raw)
        count = slots_busy(value)
        if count is not None:
            capture.decode_depth_slots_busy = count
            capture.decode_depth_slots_busy_source = "live_slots"
        return count
    except (OSError, TimeoutError, TypeError, ValueError):
        return None


def _depth(event: Mapping[str, Any], capture: object | None, row: Mapping[str, Any]):
    maps = _nested_maps(event)
    name, value = _first(maps, _DEPTH_KEYS)
    depth = _integer(value)
    source = None
    if depth is not None:
        for mapping in maps:
            candidate = mapping.get(f"{name}_source") if name else None
            if isinstance(candidate, str) and candidate.strip():
                source = candidate.strip()
                break
        source = source or ("runtime_token_estimate" if name == "rendered_context_depth" else name)
    if depth is None:
        for attr in (
            "long_horizon_rendered_prompt_tokens",
            "decode_depth_rendered_context_depth",
            "rendered_prompt_tokens",
        ):
            candidate = _integer(getattr(capture, attr, None)) if capture is not None else None
            if candidate is not None:
                depth = candidate
                source = getattr(capture, f"{attr}_source", None) or attr
                break
    if depth is None:
        row_source = row.get("depth_source")
        if row_source == "rendered_prompt_tokens":
            depth = _integer(row.get("rendered_prompt_tokens"))
            source = row_source if depth is not None else None
    return depth, source if isinstance(source, str) else None


def _slot_count(event: Mapping[str, Any], capture: object | None) -> tuple[int | None, str | None]:
    maps = _nested_maps(event)
    name, value = _first(maps, _SLOT_COUNT_KEYS)
    count = slots_busy(value)
    if count is not None:
        source = None
        for mapping in maps:
            candidate = mapping.get(f"{name}_source") if name else None
            if isinstance(candidate, str) and candidate.strip():
                source = candidate.strip()
                break
        return count, source or ("event" if name == "slots_busy" else name)
    for attr in ("long_horizon_slots_busy", "decode_depth_slots_busy", "slots_busy"):
        count = slots_busy(getattr(capture, attr, None)) if capture is not None else None
        if count is not None:
            source = getattr(capture, f"{attr}_source", None) or attr
            return count, source
    return None, None


def _timing_maps(event: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    response = _mapping(event.get("response"))
    accounting = _mapping(event.get("accounting"))
    return tuple(
        mapping for mapping in (
            _mapping(response.get("timings")),
            _mapping(accounting.get("raw_timings")),
            _mapping(accounting.get("timings")),
            row,
        )
        if mapping
    )


def enrich_response_record(
    row: Mapping[str, Any],
    event: Mapping[str, Any] | None = None,
    *,
    capture: object | None = None,
) -> dict[str, Any]:
    """Add the canonical decode/depth fields to one response row."""
    event = _mapping(event)
    result = dict(row)
    timings = _timing_maps(event, result)

    rate_name, rate_value = _first(timings, _RATE_KEYS)
    prefill_rate = _number(rate_value)
    rate_source = "backend_prefill_timing" if prefill_rate is not None else None
    if prefill_rate is None:
        prompt_n = _integer(_first(timings, ("prefill_n", "prompt_n", "prefill_tokens"))[1])
        prompt_ms = _number(_first(timings, ("prefill_ms", "prompt_ms"))[1])
        if prompt_n is not None and prompt_ms is not None and prompt_ms > 0:
            prefill_rate = prompt_n / (prompt_ms / 1000.0)
            rate_source = "derived_prefill_tokens_over_prompt_ms"
    result["prefill_tok_s"] = prefill_rate
    result["prefill_tok_s_source"] = rate_source or rate_name

    depth, depth_source = _depth(event, capture, result)
    result["rendered_context_depth"] = depth
    result["rendered_context_depth_tokens"] = depth
    result["rendered_context_depth_source"] = depth_source

    busy, busy_source = _slot_count(event, capture)
    result["slots_busy"] = busy
    result["concurrency"] = busy
    result["concurrency_source"] = busy_source
    return result


def request_metadata(
    capture: object,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    rendered_context_depth: int | None,
    logical_messages: Sequence[Mapping[str, Any]] | None = None,
    logical_tools: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build nested request metadata without changing the canonical event."""
    logical_depth = None
    try:
        from kernel_context import token_estimate
        logical_depth = token_estimate(
            {
                "messages": list(logical_messages if logical_messages is not None else messages),
                "tools": list(logical_tools if logical_tools is not None else tools),
            }
        )
    except (ImportError, TypeError, ValueError):
        pass
    task_id = os.environ.get("LONG_HORIZON_TASK_ID", getattr(capture.journal, "task_id", None))
    arm = os.environ.get("LONG_HORIZON_ARM")
    seed = int(os.environ["REACT_SEED"]) if os.environ.get("REACT_SEED", "").isdigit() else None
    busy, busy_source = _slot_count({}, capture)
    depth = _integer(rendered_context_depth)
    return {
        "schema": SCHEMA,
        "request_id": getattr(capture, "request_id", None),
        "task_id": task_id,
        "arm": arm,
        "seed": seed,
        "request_role": getattr(capture, "long_horizon_request_role", "solver"),
        "rendered_context_depth": depth,
        "rendered_context_depth_tokens": depth,
        "rendered_context_depth_source": "runtime_token_estimate" if depth is not None else None,
        "logical_depth_tokens": _integer(logical_depth),
        "slots_busy": busy,
        "concurrency": busy,
        "concurrency_source": busy_source,
        "message_count": len(messages),
        "tool_count": len(tools) if isinstance(tools, Sequence) else None,
        "usable_input_tokens": _integer(
            int(os.environ["LONG_HORIZON_USABLE_INPUT_TOKENS"])
            if os.environ.get("LONG_HORIZON_USABLE_INPUT_TOKENS", "").isdigit()
            else None
        ),
    }


def record_response(journal: object, row: Mapping[str, Any]) -> None:
    """Retain response rows for the final run accounting event when enabled."""
    if not enabled():
        return
    rows = getattr(journal, "_long_horizon_decode_rows", None)
    if rows is None:
        rows = []
        setattr(journal, "_long_horizon_decode_rows", rows)
    rows.append(copy.deepcopy(dict(row)))


def run_accounting(journal: object) -> dict[str, Any]:
    rows = getattr(journal, "_long_horizon_decode_rows", None)
    rows = rows if isinstance(rows, list) else []
    return {
        "schema": SCHEMA,
        "response_count": len(rows),
        "responses": copy.deepcopy(rows),
    }


def emit_run_accounting(journal: object) -> None:
    """Emit one terminal nested accounting record; disabled path is inert."""
    if not enabled() or getattr(journal, "_long_horizon_decode_emitted", False):
        return
    setattr(journal, "_long_horizon_decode_emitted", True)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(
            "decode_depth_accounting",
            producer="long_horizon_decode_logging",
            run_perf=run_accounting(journal),
        )
