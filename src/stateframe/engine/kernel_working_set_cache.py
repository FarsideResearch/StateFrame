"""Bounded transaction-local memoization of immutable renderer inputs.

Identity keys avoid recursively hashing source payloads on every lookup. Strong
references prevent identity reuse; the entire cache is released at transaction
exit, including failures. Independent certificate verification suspends it.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps


@dataclass
class _Cache:
    references: dict[int, object] = field(default_factory=dict)
    values: dict[tuple[object, ...], object] = field(default_factory=dict)
    renders: OrderedDict[tuple[object, ...], object] = field(default_factory=OrderedDict)
    messages: OrderedDict[tuple[object, ...], bytes] = field(default_factory=OrderedDict)
    envelopes: dict[int, tuple[tuple[str, bytes], ...]] = field(default_factory=dict)
    # Snapshot families: a derived snapshot whose provider bytes cannot differ
    # from its base (only the tokenizer identity changed) shares trial wires.
    bases: dict[int, object] = field(default_factory=dict)
    render_bytes: int = 0
    message_bytes: int = 0


_ACTIVE: ContextVar[_Cache | None] = ContextVar("working_set_render_cache", default=None)
MAX_RENDERS = 128
MAX_MESSAGES = 1024
MAX_RENDER_BYTES = 16 * 1024 * 1024
MAX_MESSAGE_BYTES = 4 * 1024 * 1024


def transaction(function):
    """Nested preparation/admission calls share one cache, never live state."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _ACTIVE.get() is not None:
            return function(*args, **kwargs)
        token = _ACTIVE.set(_Cache())
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)
    return wrapped


def independent(function):
    """Recompute proof from authority without trusting proposal memoization."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _ACTIVE.set(None)
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)
    return wrapped


def frozen(function):
    """Memoize a pure function of frozen positional objects by their identity."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        cache = _ACTIVE.get()
        if cache is None:
            return function(*args, **kwargs)
        key = (function, tuple(id(arg) for arg in args),
               tuple(sorted((name, id(value)) for name, value in kwargs.items())))
        if key not in cache.values:
            value = function(*args, **kwargs)
            cache.references.update((id(arg), arg) for arg in args)
            cache.references.update((id(value), value) for value in kwargs.values())
            cache.values[key] = value
        return cache.values[key]
    return wrapped


def source_frozen(function):
    """Reuse slot fragments across floors with the same recovery authority.

    A prefix trial changes its envelope and adds a FULL-only slot. It cannot
    change the existing slots' domains or notices unless one of these frozen
    recovery/protection inputs changes. Do not key by path or slot key alone.
    """
    @wraps(function)
    def wrapped(snapshot, slot):
        cache = _ACTIVE.get()
        if cache is None:
            return function(snapshot, slot)
        key = (function, id(slot), snapshot.recovery_schema, snapshot.recovery_allowed,
               snapshot.recovery_owner_allowed, snapshot.current_revisions,
               snapshot.protected_paths, snapshot.protected_ranges)
        if key not in cache.values:
            value = function(snapshot, slot)
            cache.references[id(slot)] = slot
            cache.values[key] = value
        return cache.values[key]
    return wrapped


def alias(derived, base):
    """Declare that ``derived`` renders the same provider bytes as ``base``.

    Only the tokenizer identity may differ: certificates are still rendered
    against ``derived``; trial wires and search decisions are shared. Outside
    a transaction this is a no-op. Returns ``derived`` for inline use.
    """
    cache = _ACTIVE.get()
    if cache is None or derived is base:
        return derived
    root = cache.bases.get(id(base), base)
    cache.references[id(root)] = root
    cache.references[id(derived)] = derived
    cache.bases[id(derived)] = root
    return derived


def base_of(snapshot):
    """The family root ``snapshot`` was aliased to, or None."""
    cache = _ACTIVE.get()
    return None if cache is None else cache.bases.get(id(snapshot))


def family(snapshot) -> int:
    """Identity key shared by every snapshot in one provider-bytes family."""
    cache = _ACTIVE.get()
    if cache is None:
        return id(snapshot)
    return id(cache.bases.get(id(snapshot), snapshot))


def trial_render(snapshot, assignment, layout, renderer):
    """Reuse repeated whole candidates; eviction affects work, never decisions."""
    cache = _ACTIVE.get()
    if cache is None:
        return renderer(snapshot, assignment, layout)
    modes = getattr(assignment, "modes", assignment)
    key = (renderer, family(snapshot), layout, tuple(sorted(modes.items())))
    if key in cache.renders:
        cache.renders.move_to_end(key)
        result = cache.renders[key]
        rebind = getattr(renderer, "rebind", None)
        if rebind is not None:
            # A sibling snapshot's trial: same bytes, certificate deferred to
            # a render against this snapshot when the candidate is used.
            result = rebind(result, snapshot, modes, layout)
            cache.renders[key] = result
        return result
    result = renderer(snapshot, assignment, layout)
    cache.references[id(snapshot)] = snapshot
    cache.renders[key] = result
    cache.render_bytes += len(result.wire)
    while len(cache.renders) > MAX_RENDERS or cache.render_bytes > MAX_RENDER_BYTES:
        _, removed = cache.renders.popitem(last=False)
        cache.render_bytes -= len(removed.wire)
    return result


def wire_bytes(body, snapshot, layout):
    """Default JSON bytes from cached invariant envelope/whole-row fragments.

    Only messages vary in a plain assignment: normalization of all other fields
    depends exclusively on the frozen snapshot. Preserve dictionary order and
    JSON's exact separators/escaping. Projected layouts and independent proofs
    use the ordinary whole-body serializer.
    """
    cache = _ACTIVE.get()
    if cache is None or layout != "plain":
        return json.dumps(body).encode()
    identity = id(snapshot)
    if identity not in cache.envelopes:
        cache.references[identity] = snapshot
        cache.envelopes[identity] = tuple((key, json.dumps(key).encode() + b": " +
            (json.dumps(value).encode() if key != "messages" else b""))
            for key, value in body.items())
    rows = []
    for message in body["messages"]:
        # Nested native calls/content parts still use the standard serializer.
        # Include value types: Python's True == 1 must not alias JSON bytes.
        key = (tuple((name, type(value), value) for name, value in message.items())
               if all(type(value) in {str, int, bool, type(None)}
                      for value in message.values()) else None)
        if key is None:
            rows.append(json.dumps(message).encode())
            continue
        if key not in cache.messages:
            encoded = json.dumps(message).encode()
            rows.append(encoded)
            cache.messages[key] = encoded
            cache.message_bytes += len(encoded)
            while len(cache.messages) > MAX_MESSAGES or cache.message_bytes > MAX_MESSAGE_BYTES:
                _, removed = cache.messages.popitem(last=False)
                cache.message_bytes -= len(removed)
        else:
            cache.messages.move_to_end(key)
            rows.append(cache.messages[key])
    messages = b"[" + b", ".join(rows) + b"]"
    return b"{" + b", ".join(prefix + (messages if key == "messages" else b"")
                              for key, prefix in cache.envelopes[identity]) + b"}"
