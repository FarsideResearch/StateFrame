"""Opt-in client policy and accounting for StateFrame prompt-cache reuse.

The switch is deliberately separate from the older cache-layout experiment.
When it is off, callers retain their historical request fields and this module
does not emit accounting events.  When it is on, a caller that already pins a
provider slot also asks that slot to reuse its prompt prefix.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping, MutableMapping, Sequence
from kernel_prompt_accounting import accounting_tokens


MODE_ENV = "STATEFRAME_PROMPT_CACHE"
SCHEMA = "stateframe-prompt-cache/1"
_VALID_MODES = frozenset({"off", "on"})
_ARMS = ("frame", "history")
MISSING = "MISSING"


def _source(environ: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def prompt_cache_mode(environ: Mapping[str, str] | None = None) -> str:
    value = _source(environ).get(MODE_ENV, "off")
    if value not in _VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def mode(environ: Mapping[str, str] | None = None) -> str:
    return prompt_cache_mode(environ)


def prompt_cache_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return prompt_cache_mode(environ) == "on"


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return prompt_cache_enabled(environ)


def configuration(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    return {"prompt_cache": prompt_cache_mode(environ)}


def configure_pinned_slot(body: MutableMapping[str, object], slot: int,
                          environ: Mapping[str, str] | None = None) \
        -> MutableMapping[str, object]:
    """Keep an existing episode slot and couple its cache request to the switch."""
    if not isinstance(body, MutableMapping):
        raise TypeError("request body must be mutable")
    if type(slot) is not int or slot < 0:
        raise ValueError("pinned slot must be a nonnegative integer")
    body["id_slot"] = slot
    body["cache_prompt"] = prompt_cache_enabled(environ)
    return body


def pair_cache_enabled_for_prompt_cache(environ: Mapping[str, str] | None = None) -> bool:
    """Return the new arm's override for the two-call self-plan pair."""
    return prompt_cache_enabled(environ)


def request_arm(messages: object) -> str:
    """Classify a recorded request without relying on task or ticket text."""
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes, bytearray)):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.startswith("[StateFrame]\n"):
                return "frame"
    return "history"


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def uncached_prefill(prompt_n: object, cache_n: object) -> int | str:
    """Compatibility helper: prompt_n is explicitly llama.cpp uncached prefill."""
    prompt = _nonnegative_int(prompt_n)
    cached = _nonnegative_int(cache_n)
    if prompt is None or cached is None:
        return MISSING
    return prompt


def prompt_cache_response_fields(accounting: Mapping[str, object]) -> dict[str, object]:
    """Return the extra response fields owned by this opt-in accounting arm."""
    counts = accounting_tokens(accounting)
    return {
        "cached_tokens": counts["cached"],
        "uncached_prefill_n": counts["uncached"],
    }


def _empty_arm() -> dict[str, int]:
    return {"responses": 0, "measured_responses": 0, "prompt_n": 0,
            "cache_n": 0, "cached_tokens": 0, "uncached_prefill_n": 0}


def record_response_accounting(journal: object, accounting: Mapping[str, object],
                               arm: str) -> None:
    """Accumulate backend measurements in a run-local, non-journal object."""
    if not prompt_cache_enabled():
        return
    if arm not in {*_ARMS, "other"}:
        arm = "other"
    totals = getattr(journal, "_prompt_cache_totals", None)
    if not isinstance(totals, dict):
        totals = {name: _empty_arm() for name in (*_ARMS, "other")}
        setattr(journal, "_prompt_cache_totals", totals)
    bucket = totals.setdefault(arm, _empty_arm())
    bucket["responses"] += 1
    counts = accounting_tokens(accounting)
    prompt_n = _nonnegative_int(counts["total"])
    cache_n = _nonnegative_int(counts["cached"])
    uncached = counts["uncached"]
    if prompt_n is None or cache_n is None or not isinstance(uncached, int):
        return
    bucket["measured_responses"] += 1
    bucket["prompt_n"] += prompt_n
    bucket["cache_n"] += cache_n
    bucket["cached_tokens"] += cache_n
    bucket["uncached_prefill_n"] += uncached


def run_accounting(journal: object) -> dict[str, dict[str, int]]:
    totals = getattr(journal, "_prompt_cache_totals", None)
    result = {name: _empty_arm() for name in (*_ARMS, "other")}
    if isinstance(totals, Mapping):
        for name, bucket in totals.items():
            if isinstance(bucket, Mapping):
                result[name] = {key: int(value) for key, value in bucket.items()
                                if type(value) is int and value >= 0}
                for key, value in _empty_arm().items():
                    result[name].setdefault(key, value)
    return copy.deepcopy(result)


def emit_run_accounting(journal: object) -> dict[str, object] | None:
    """Emit the per-arm total exactly once at the runtime finish boundary."""
    if not prompt_cache_enabled():
        return None
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        raise TypeError("journal must expose emit(kind, **fields)")
    totals = run_accounting(journal)
    return emitter(
        "prompt_cache_accounting",
        accounting_schema=SCHEMA,
        arms={name: totals[name] for name in _ARMS},
        other=totals["other"],
        total_responses=sum(row["responses"] for row in totals.values()),
        measured_responses=sum(row["measured_responses"] for row in totals.values()),
    )


__all__ = [
    "MODE_ENV", "SCHEMA", "configuration", "configure_pinned_slot", "enabled",
    "emit_run_accounting", "mode", "pair_cache_enabled_for_prompt_cache",
    "prompt_cache_enabled", "prompt_cache_mode", "prompt_cache_response_fields",
    "record_response_accounting", "request_arm", "run_accounting", "uncached_prefill",
]
