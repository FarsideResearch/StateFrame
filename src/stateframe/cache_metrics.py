"""Normalize provider prompt-cache counters to total, cached, and uncached tokens."""
from __future__ import annotations

from collections.abc import Mapping


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def normalize(response_usage: object, timings: object) -> dict[str, int | None]:
    """Return prompt token counts without mixing total and uncached counters.

    llama.cpp reports uncached prompt tokens as ``timings.prompt_n`` and reused
    tokens as ``timings.cache_n``. OpenAI-style usage reports a total in
    ``usage.prompt_tokens`` and may report reused tokens in
    ``usage.prompt_tokens_details.cached_tokens``.
    """
    usage = response_usage if isinstance(response_usage, Mapping) else {}
    timing = timings if isinstance(timings, Mapping) else {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, Mapping) else {}

    uncached = _count(timing.get("prompt_n"))
    cached = _count(timing.get("cache_n"))
    usage_total = _count(usage.get("prompt_tokens"))
    usage_cached = _count(details.get("cached_tokens"))

    # Redundant provider fields must agree. Prefer neither source on conflict:
    # choosing one would fabricate confidence in both the total and cache share.
    unknown = {"total": None, "cached": None, "uncached": None}
    if cached is not None and usage_cached is not None and cached != usage_cached:
        return unknown
    reported_cached = cached if cached is not None else usage_cached
    if (usage_total is not None and uncached is not None and reported_cached is not None
            and usage_total != uncached + reported_cached):
        return unknown

    # A consistent llama.cpp timing pair separates reused and newly evaluated tokens.
    if uncached is not None and cached is not None:
        return {"total": uncached + cached, "cached": cached, "uncached": uncached}

    if usage_total is not None:
        reported_cached = cached if cached is not None else usage_cached
        if reported_cached is not None:
            if reported_cached > usage_total:
                return {"total": usage_total, "cached": None, "uncached": None}
            return {"total": usage_total, "cached": reported_cached,
                    "uncached": usage_total - reported_cached}
        if uncached is not None:
            if uncached > usage_total:
                return {"total": usage_total, "cached": None, "uncached": None}
            return {"total": usage_total, "cached": usage_total - uncached,
                    "uncached": uncached}
        return {"total": usage_total, "cached": None, "uncached": None}

    # Retain partial measurements; derive total only when both disjoint parts exist.
    partial_cached = cached if cached is not None else usage_cached
    total = (uncached + partial_cached
             if uncached is not None and partial_cached is not None else None)
    return {"total": total, "cached": partial_cached, "uncached": uncached}
