"""Q6 (E8): zero-information repetition stops the route, not the model.

Semantic progress key = ``(tool, normalised args, workspace revision)`` plus the
digest of the zero result.  The first zero result is returned unchanged.  The
second identical zero result (same key, same result digest, same revision) is
replaced by a ``[REPEATED ZERO]`` observation that restates the host entry
fact, lists the admissible next routes, and suppresses that exact call: a third
identical call is refused with ``REPEATED_ZERO_SUPPRESSED`` until the workspace
revision changes.  A call with different arguments, or any call after an edit
(new revision), is unaffected.

Zero results: grep ``total=0`` (the caller says so), ``where_defined`` /
``who_calls`` / ``references`` status ``NOT_FOUND``/``EMPTY``, an empty read.

``STATEFRAME_REPEATED_ZERO`` (default ``on``; ``off`` restores the raw
observations and never refuses).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping

ENV = "STATEFRAME_REPEATED_ZERO"
HEADER = "[REPEATED ZERO]"
SUPPRESSED = "REPEATED_ZERO_SUPPRESSED"
_STATE_ATTR = "_repeated_zero_state"
_ZERO_STATUSES = frozenset({"NOT_FOUND", "EMPTY"})
_NAV_TOOLS = frozenset({"where_defined", "who_calls", "references"})


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    value = str(values.get(ENV, "on") or "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{ENV} must be on or off")
    return value == "on"


def normalised_args(arguments: object) -> str:
    if not isinstance(arguments, Mapping):
        # Query/pattern bytes are semantic input.  Whitespace normalization
        # made exact and regex searches collide with their meaningful forms.
        return json.dumps(arguments, ensure_ascii=False, default=str)
    # Every serialized argument is potentially semantic.  In particular,
    # regex whitespace, exact query text, cursors, scopes, and schema-defined
    # explanatory fields must not collapse into one repeated-call identity.
    clean = {str(key): value for key, value in arguments.items()}
    return json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str)


def _revision(runtime: object) -> str:
    binding = getattr(runtime, "binding", None)
    try:
        value = binding() if callable(binding) else binding
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return ""
    if isinstance(value, Mapping):
        value = value.get("product_revision") or value.get("artifact") or value
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _key(runtime: object, tool: object, arguments: object) -> tuple[str, str, str]:
    return (str(tool or "").strip().lower(), normalised_args(arguments), _revision(runtime))


def _state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, _STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {"counts": {}, "suppressed": set()}
        try:
            setattr(runtime, _STATE_ATTR, state)
        except (AttributeError, TypeError):
            pass
    return state


def _lookup_status(observation: str) -> str:
    text = str(observation or "")
    start = text.find("{")
    if start < 0:
        return ""
    try:
        payload = json.loads(text[start:])
    except ValueError:
        return ""
    return str(payload.get("status") or "") if isinstance(payload, Mapping) else ""


def is_zero(tool: object, observation: str) -> bool:
    name = str(tool or "").strip().lower()
    if name in _NAV_TOOLS:
        return _lookup_status(observation) in _ZERO_STATUSES
    if name == "read":
        return not str(observation or "").strip()
    return False


def _emit(runtime: object, event: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        try:
            emit(event, producer="kernel_repeated_zero", dispatch=False, **fields)
        except (TypeError, ValueError):
            pass


def _entry_fact(runtime: object, query: str) -> str:
    entry = getattr(runtime, "_site_facts_entry", None)
    if not isinstance(entry, Mapping):
        return ""
    symbol = str(entry.get("symbol") or "")
    short = symbol.rsplit(".", 1)[-1]
    if not short or not re.search(rf"(?<![A-Za-z0-9_]){re.escape(short)}(?![A-Za-z0-9_])", query):
        return ""
    try:
        from kernel_request_source_window import render_entry_verification
        rendered = render_entry_verification(entry)
    except (ImportError, TypeError, ValueError):
        rendered = None
    return rendered or ""


def render(runtime: object, tool: str, arguments: object) -> str:
    query = normalised_args(arguments)
    shown = query if len(query) <= 160 else query[:157] + "..."
    lines = [f"{HEADER} {tool} {shown} returned 0 results twice at this revision "
             "(identical call, identical result)."]
    fact = _entry_fact(runtime, query)
    if fact:
        lines.append(fact)
    lines.append("It is absent from the searched scope; repeating the call cannot change the answer.")
    lines.append("Next routes: use a different bounded public lookup with the current scope/cursor, "
                 "read the relevant source owner, or run the named check. This identical call is "
                 "refused until the workspace changes.")
    return "\n".join(lines)


def observe(runtime: object, tool: object, arguments: object, observation: str, *,
            zero: bool | None = None) -> str:
    """Return ``observation``, or the ``[REPEATED ZERO]`` block on the second identical zero."""
    if not enabled():
        return observation
    name = str(tool or "").strip().lower()
    is_zero_result = is_zero(name, observation) if zero is None else bool(zero)
    state = _state(runtime)
    counts: dict = state["counts"]  # type: ignore[assignment]
    if not is_zero_result:
        # A non-zero answer resets this call's count at every revision (no binding needed).
        identity = (name, normalised_args(arguments))
        for stale in [item for item in counts if item[0][:2] == identity]:
            counts.pop(stale, None)
        return observation
    key = _key(runtime, name, arguments)
    digest = hashlib.sha256(str(observation).encode("utf-8", "replace")).hexdigest()[:16]
    count = int(counts.get((key, digest), 0)) + 1
    counts[(key, digest)] = count
    if count < 2:
        return observation
    state["suppressed"].add(key)  # type: ignore[union-attr]
    _emit(runtime, "repeated_zero", tool=name, arguments=key[1][:240], count=count)
    return render(runtime, name, arguments)


def refusal(runtime: object, tool: object, arguments: object) -> str | None:
    """``REPEATED_ZERO_SUPPRESSED`` for an identical call after the REPEATED ZERO block."""
    if not enabled():
        return None
    state = getattr(runtime, _STATE_ATTR, None)
    if not isinstance(state, dict) or not state.get("suppressed"):
        return None
    key = _key(runtime, tool, arguments)
    if key not in state["suppressed"]:
        return None
    _emit(runtime, "repeated_zero_suppressed", tool=key[0], arguments=key[1][:240])
    return (f"{SUPPRESSED}: this identical {key[0]} call already returned 0 results twice at this "
            "revision. Choose another bounded route: change scope/cursor/pattern, read the source, "
            "or abstain.")


__all__ = ["ENV", "HEADER", "SUPPRESSED", "enabled", "is_zero", "normalised_args",
           "observe", "refusal", "render"]
