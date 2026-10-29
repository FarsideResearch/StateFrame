"""One provider-ingress contract for total, cached and uncached prompt tokens.

llama.cpp timings.prompt_n counts evaluated (uncached) tokens; OpenAI
usage.prompt_tokens counts the total. Missing counters are never zero-filled.
Legacy journals are readable only when raw provider provenance is available.
"""
from __future__ import annotations

from collections.abc import Mapping

MISSING = "MISSING"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _count(value: object) -> int | str:
    return value if type(value) is int and value >= 0 else MISSING


def _validated(total: object, cached: object, uncached: object) -> dict[str, int | str]:
    total, cached, uncached = _count(total), _count(cached), _count(uncached)
    if (isinstance(total, int) and (
            isinstance(cached, int) and cached > total
            or isinstance(uncached, int) and uncached > total
            or isinstance(cached, int) and isinstance(uncached, int)
            and cached + uncached != total)):
        total = cached = uncached = MISSING
    return {"total": total, "cached": cached, "uncached": uncached}


def normalize_prompt_tokens(response: object) -> dict[str, int | str]:
    """Normalize raw provider counters once, preserving unknown and invalid data."""
    response = _mapping(response)
    timings = _mapping(response.get("timings"))
    usage = _mapping(response.get("usage"))
    details = _mapping(usage.get("prompt_tokens_details"))
    total = _count(usage.get("prompt_tokens"))
    uncached = _count(timings.get("prompt_n"))
    cached = _count(timings.get("cache_n"))
    usage_cached = _count(details.get("cached_tokens"))
    if cached == MISSING:
        cached = usage_cached
    elif usage_cached != MISSING and cached != usage_cached:
        return _validated(MISSING, MISSING, MISSING)
    if total == MISSING and isinstance(uncached, int) and isinstance(cached, int):
        total = uncached + cached
    if uncached == MISSING and isinstance(total, int) and isinstance(cached, int):
        if cached > total:
            return _validated(MISSING, MISSING, MISSING)
        uncached = total - cached
    return _validated(total, cached, uncached)


def accounting_tokens(accounting: object, response: object = None) -> dict[str, int | str]:
    """Read normalized ingress; adapt older journals only from explicit raw fields."""
    accounting = _mapping(accounting)
    normalized = accounting.get("prompt_tokens")
    if isinstance(normalized, Mapping):
        return _validated(normalized.get("total"), normalized.get("cached"), normalized.get("uncached"))
    raw_timings = _mapping(accounting.get("raw_timings"))
    raw_usage = _mapping(accounting.get("raw_usage"))
    if raw_timings or raw_usage:
        return normalize_prompt_tokens({"timings": raw_timings, "usage": raw_usage})
    return normalize_prompt_tokens(response)


def compatibility_fields(counts: Mapping[str, object]) -> dict[str, object]:
    """Names retained for pre-normalization v30 consumers, with fixed meanings."""
    return {"logical_prompt_tokens": counts["total"], "cached_tokens": counts["cached"],
            "evaluated_prompt_tokens": counts["uncached"]}
