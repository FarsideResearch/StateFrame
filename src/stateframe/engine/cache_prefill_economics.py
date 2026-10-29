"""Opt-in per-step, per-arm cache/prefill accounting for R-032.

This module records backend counters without treating the request text as a
cache-hit signal.  It is intentionally separate from the prompt-cache serving
policy: a study may measure a provider with caching off, unavailable, or
unsupported.  Per-turn response accounting is enabled by either the economics
study or Stage 1 trailing-panel switch; both default off.
"""
from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_CACHE_PREFILL_ECONOMICS"
TRAILING_PANELS_ENV = "STATEFRAME_TRAILING_PANELS"
SCHEMA = "cache-prefill-economics/1"
EVENT_KIND = "cache_prefill_accounting"
RESPONSE_EVENT_KIND = "cache_prefill_response"
_ARMS = ("history", "frame", "unknown")
_VALID_MODES = frozenset({"off", "on"})
_CACHE_KEYS = (
    "cache_n",
    "cached_tokens",
    "prompt_cache_hit_tokens",
)
_SUPPORT_KEYS = (
    "cache_supported",
    "prompt_cache_supported",
    "prompt_cache_reuse_supported",
)


def _values(environ: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def mode(environ: Mapping[str, str] | None = None) -> str:
    value = _values(environ).get(MODE_ENV, "off")
    if value not in _VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def response_accounting_enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = _values(environ)
    economics_mode = mode(values)
    trailing_mode = values.get(TRAILING_PANELS_ENV, "off")
    if trailing_mode not in _VALID_MODES:
        raise ValueError(f"{TRAILING_PANELS_ENV} must be off or on")
    return economics_mode == "on" or trailing_mode == "on"


def _integer(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _maps(accounting: Mapping[str, object], response: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    usage = _mapping(response.get("usage"))
    details = _mapping(usage.get("prompt_tokens_details"))
    raw_usage = _mapping(accounting.get("raw_usage"))
    raw_details = _mapping(raw_usage.get("prompt_tokens_details"))
    return tuple(
        mapping
        for mapping in (
            accounting,
            _mapping(accounting.get("raw_timings")),
            _mapping(accounting.get("timings")),
            _mapping(response.get("timings")),
            response,
            usage,
            details,
            raw_usage,
            raw_details,
        )
        if mapping
    )


def _counter(
    maps: Sequence[Mapping[str, object]], keys: Sequence[str]
) -> tuple[int | None, str | None]:
    for mapping in maps:
        for key in keys:
            value = _integer(mapping.get(key))
            if value is not None:
                return value, key
    return None, None


def _prompt_counter(
    accounting: Mapping[str, object], response: Mapping[str, object]
) -> tuple[int | None, str | None]:
    maps = _maps(accounting, response)
    value, source = _counter(maps, ("prompt_n",))
    if value is not None:
        if source == "prompt_n" and accounting.get("prompt_n") == value:
            raw_timings = _mapping(accounting.get("raw_timings"))
            if _integer(raw_timings.get("prompt_n")) is None:
                if accounting.get("usage_prompt_tokens") == value:
                    return value, "usage_prompt_tokens"
            return value, "accounting.prompt_n"
        return value, source
    value, source = _counter(maps, ("prompt_tokens",))
    return value, "usage.prompt_tokens" if value is not None else source


def _cache_counter(
    accounting: Mapping[str, object], response: Mapping[str, object]
) -> tuple[int | None, str | None]:
    value, source = _counter(_maps(accounting, response), _CACHE_KEYS)
    if value is None:
        return None, None
    if source == "cache_n" and accounting.get("cache_n") == value:
        raw_timings = _mapping(accounting.get("raw_timings"))
        if _integer(raw_timings.get("cache_n")) is None:
            raw_usage = _mapping(accounting.get("raw_usage"))
            details = _mapping(raw_usage.get("prompt_tokens_details"))
            if _integer(details.get("cached_tokens")) == value:
                return value, "usage.prompt_tokens_details.cached_tokens"
        return value, "accounting.cache_n"
    return value, source


def _support_status(
    accounting: Mapping[str, object], response: Mapping[str, object]
) -> str:
    for mapping in _maps(accounting, response):
        for key in _SUPPORT_KEYS:
            value = mapping.get(key)
            if isinstance(value, bool):
                return "supported" if value else "unsupported"
    _cache, cache_source = _cache_counter(accounting, response)
    return "measured" if cache_source is not None else "unreported"


def _arm(capture: object, environ: Mapping[str, str] | None = None) -> str:
    for candidate in (
        getattr(capture, "prompt_cache_arm", None),
        _values(environ).get("LONG_HORIZON_ARM"),
    ):
        if isinstance(candidate, str) and candidate.strip().lower() in {"history", "frame"}:
            return candidate.strip().lower()
    return "unknown"


def _step(capture: object) -> int | None:
    value = getattr(capture, "step", None)
    return value if type(value) is int and value >= 0 else None


def _request_id(capture: object) -> str | None:
    value = getattr(capture, "request_id", None)
    return value if isinstance(value, str) and value else None


def _cache_reuse_fraction(
    counts: Mapping[str, object],
    prompt_source: str | None,
    cache_source: str | None,
) -> tuple[float | None, str | None]:
    total, cached = counts["total"], counts["cached"]
    if type(total) is not int or type(cached) is not int or total <= 0:
        return None, None
    if prompt_source in {"usage.prompt_tokens", "usage_prompt_tokens"}:
        return cached / total, "cached_tokens_over_total_prompt_tokens"
    if prompt_source == "accounting.prompt_n" and cache_source == "accounting.cache_n":
        return cached / total, "separate_cached_and_uncached_prefill_counters"
    return cached / total, "normalized_cached_over_total"


def response_row(
    capture: object,
    accounting: Mapping[str, object] | None,
    *,
    response: Mapping[str, object] | None = None,
    request: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Normalize one backend response without guessing cache reuse.

    ``prompt_n`` and ``cache_n`` intentionally remain separate raw counters.
    Different providers assign different meanings to ``prompt_n``; subtracting
    the counters here would make a llama.cpp-style uncached count incorrect.
    """
    accounting = _mapping(accounting)
    response = _mapping(response)
    request = _mapping(request)
    from kernel_prompt_accounting import accounting_tokens
    counts = accounting_tokens(accounting, response)
    prompt_n, prompt_source = _prompt_counter(accounting, response)
    cache_n, cache_source = _cache_counter(accounting, response)
    prompt_ms, _ = _counter(_maps(accounting, response), ("prompt_ms", "prefill_ms"))
    if prompt_ms is None:
        for mapping in _maps(accounting, response):
            candidate = _number(mapping.get("prompt_ms", mapping.get("prefill_ms")))
            if candidate is not None:
                prompt_ms = candidate
                break
    relation = "unavailable"
    if prompt_n is not None and cache_n is not None:
        relation = "cache_gt_prompt" if cache_n > prompt_n else "cache_le_prompt"
    reuse_fraction, reuse_basis = _cache_reuse_fraction(
        counts, prompt_source, cache_source,
    )
    return {
        "schema": SCHEMA,
        "prompt_tokens": counts,
        "request_id": _request_id(capture),
        "step": _step(capture),
        "arm": _arm(capture, environ),
        "prompt_n": prompt_n,
        "prompt_n_source": prompt_source,
        "cache_n": cache_n,
        "cache_n_source": cache_source,
        "prompt_ms": prompt_ms,
        "cache_measurement": "measured" if cache_n is not None else "unavailable",
        "provider_reuse_status": _support_status(accounting, response),
        "cache_reuse_fraction": reuse_fraction,
        "cache_reuse_basis": reuse_basis,
        "cache_reuse_alarm": reuse_fraction < 0.60 if reuse_fraction is not None else None,
        "request_cache_prompt": request.get("cache_prompt")
        if isinstance(request.get("cache_prompt"), bool)
        else None,
        "counter_relation": relation,
    }


def request_metadata(
    capture: object,
    messages: Sequence[Mapping[str, object]],
    tools: object,
    *,
    rendered_context_depth: int | None,
) -> dict[str, object]:
    """Describe the prepared step so response counters have a live-path key."""
    depth = _integer(rendered_context_depth)
    tool_count = len(tools) if isinstance(tools, Sequence) and not isinstance(
        tools, (str, bytes, bytearray)
    ) else None
    return {
        "schema": SCHEMA,
        "request_id": _request_id(capture),
        "step": _step(capture),
        "arm": _arm(capture),
        "rendered_context_depth": depth,
        "message_count": len(messages),
        "tool_count": tool_count,
        "response_counters": ("prompt_n", "cache_n"),
        "cache_hits_require_backend_counter": True,
    }


def record_response(journal: object, row: Mapping[str, object]) -> None:
    """Retain a response row while either opt-in accounting path is enabled."""
    if not response_accounting_enabled() or not isinstance(row, Mapping):
        return
    rows = getattr(journal, "_cache_prefill_economics_rows", None)
    if not isinstance(rows, list):
        rows = []
        setattr(journal, "_cache_prefill_economics_rows", rows)
    rows.append(copy.deepcopy(dict(row)))
    fraction = _number(row.get("cache_reuse_fraction"))
    emitter = getattr(journal, "emit", None)
    if callable(emitter):
        emitter(
            RESPONSE_EVENT_KIND,
            producer="cache_prefill_economics",
            request_id=row.get("request_id"),
            response_event_id=row.get("response_event_id"),
            step=row.get("step"),
            arm=row.get("arm"),
            cache_n=row.get("cache_n"),
            cache_n_source=row.get("cache_n_source"),
            prompt_n=row.get("prompt_n"),
            prompt_n_source=row.get("prompt_n_source"),
            prompt_ms=row.get("prompt_ms"),
            cache_reuse_fraction=fraction,
            cache_reuse_basis=row.get("cache_reuse_basis"),
            dispatch=False,
        )
    if fraction is not None and fraction < 0.60 and callable(emitter):
        emitter(
            "cache_reuse_alarm",
            producer="cache_prefill_economics",
            request_id=row.get("request_id"),
            step=row.get("step"),
            cache_n=row.get("cache_n"),
            cache_n_source=row.get("cache_n_source"),
            prompt_n=row.get("prompt_n"),
            prompt_n_source=row.get("prompt_n_source"),
            cache_reuse_fraction=fraction,
            cache_reuse_basis=row.get("cache_reuse_basis"),
            threshold=0.60,
            dispatch=False,
        )


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    prompt_values = [_integer(row.get("prompt_n")) for row in rows]
    cache_values = [_integer(row.get("cache_n")) for row in rows]
    prompt_values = [value for value in prompt_values if value is not None]
    cache_values = [value for value in cache_values if value is not None]
    return {
        "responses": len(rows),
        "prompt_measured_responses": len(prompt_values),
        "cache_measured_responses": len(cache_values),
        "prompt_n": sum(prompt_values) if prompt_values else None,
        "cache_n": sum(cache_values) if cache_values else None,
        "unavailable_cache_responses": sum(
            1 for row in rows if row.get("cache_measurement") != "measured"
        ),
        "unsupported_provider_reuse": sum(
            1 for row in rows if row.get("provider_reuse_status") == "unsupported"
        ),
    }


def _group(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        value = row.get(key)
        if key == "step":
            group = str(value) if type(value) is int else "unknown"
        else:
            group = value if value in _ARMS else "unknown"
        grouped.setdefault(group, []).append(row)
    return {group: _aggregate(grouped[group]) for group in sorted(grouped)}


def run_accounting(journal: object) -> dict[str, object]:
    rows = getattr(journal, "_cache_prefill_economics_rows", None)
    rows = rows if isinstance(rows, list) else []
    clean = [dict(row) for row in rows if isinstance(row, Mapping)]
    per_arm = _group(clean, "arm")
    per_step: dict[str, dict[str, dict[str, object]]] = {}
    steps: dict[str, list[Mapping[str, object]]] = {}
    for row in clean:
        step = str(row.get("step")) if type(row.get("step")) is int else "unknown"
        steps.setdefault(step, []).append(row)
    for step, step_rows in sorted(steps.items()):
        per_step[step] = _group(step_rows, "arm")
    return {
        "schema": SCHEMA,
        "response_count": len(clean),
        "responses": copy.deepcopy(clean),
        "per_arm": per_arm,
        "per_step": per_step,
    }


def emit_run_accounting(journal: object) -> dict[str, object] | None:
    """Emit one terminal ``run_perf`` record, never on the disabled path."""
    if not enabled() or getattr(journal, "_cache_prefill_economics_emitted", False):
        return None
    setattr(journal, "_cache_prefill_economics_emitted", True)
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        raise TypeError("journal must expose emit(kind, **fields)")
    return emitter(
        EVENT_KIND,
        producer="cache_prefill_economics",
        run_perf=run_accounting(journal),
    )


__all__ = [
    "EVENT_KIND",
    "MODE_ENV",
    "RESPONSE_EVENT_KIND",
    "SCHEMA",
    "enabled",
    "emit_run_accounting",
    "mode",
    "record_response",
    "request_metadata",
    "response_accounting_enabled",
    "response_row",
    "run_accounting",
]
