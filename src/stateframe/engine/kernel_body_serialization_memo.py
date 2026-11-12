"""Exact, mutation-safe memo of whole-request serializations.

A request body is serialized several times per prepare (wire bytes, sorted
character count, canonical identity) while usually unchanged between calls.
Entries are guarded by a structural stamp: every container (identity, length)
and every immutable leaf object (identity) in traversal order. Strings are
immutable, so an identical stamp implies identical JSON. Any in-place edit,
replaced leaf, reordered key or changed length produces a different stamp and
a fresh serialization. Unknown value types disable the memo for that call.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping

_MAX_ENTRIES = 8
_MAX_NODES = 200_000
_MAX_TEXT_CHARS = 4_000_000  # larger bodies bypass so cached entries stay small
_LEAVES = frozenset((str, int, float, bool, type(None)))
_LOCK = threading.Lock()
_ENTRIES: "OrderedDict[int, _Entry]" = OrderedDict()


class _Entry:
    __slots__ = ("body", "stamp", "plain", "chars", "digests")

    def __init__(self, body: object, stamp: list[object]) -> None:
        self.body = body  # strong ref keeps id() unique while cached
        self.stamp = stamp
        self.plain: bytes | None = None
        self.chars: int | None = None
        self.digests: dict[Callable[[object], str], str] = {}


def _stamp(body: object) -> list[object] | None:
    """Flatten containers and leaves in order.

    None (bypass the memo) for unknown or subclassed types, repeated containers
    (cycles or shared nodes), more than _MAX_NODES nodes, or more than
    _MAX_TEXT_CHARS of string text.
    """
    out: list[object] = []
    seen: set[int] = set()
    text = 0
    stack: list[object] = [body]
    while stack:
        value = stack.pop()
        kind = type(value)
        if kind is dict or kind is list or kind is tuple:
            if id(value) in seen:
                return None
            seen.add(id(value))
            out.append(value)
            out.append(len(value))
            if kind is dict:
                for key, child in value.items():
                    if type(key) is not str:
                        return None
                    text += len(key)
                    out.append(key)
                    stack.append(child)
            else:
                stack.extend(value)
        elif kind in _LEAVES:
            if kind is str:
                text += len(value)
            out.append(value)
        else:
            return None
        if len(out) > _MAX_NODES or text > _MAX_TEXT_CHARS:
            return None
    return out


def _same(left: list[object], right: list[object]) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a is b:
            continue
        if type(a) is int and type(b) is int and a == b:
            continue
        return False
    return True


def _entry(body: object) -> _Entry | None:
    if type(body) is not dict:
        return None
    stamp = _stamp(body)
    if stamp is None:
        return None
    key = id(body)
    with _LOCK:
        entry = _ENTRIES.get(key)
        if entry is not None and entry.body is body and _same(entry.stamp, stamp):
            _ENTRIES.move_to_end(key)
            return entry
        entry = _Entry(body, stamp)
        _ENTRIES[key] = entry
        _ENTRIES.move_to_end(key)
        while len(_ENTRIES) > _MAX_ENTRIES:
            _ENTRIES.popitem(last=False)
        return entry


def plain_wire(body: Mapping[str, object]) -> bytes:
    """Bytes of ``json.dumps(body).encode()``."""
    entry = _entry(body)
    if entry is None:
        return json.dumps(body).encode()
    if entry.plain is None:
        entry.plain = json.dumps(body).encode()
    return entry.plain


def sorted_chars(body: Mapping[str, object]) -> int:
    """Length of ``json.dumps(body, sort_keys=True, ensure_ascii=False)``."""
    entry = _entry(body)
    if entry is None:
        return len(json.dumps(body, sort_keys=True, ensure_ascii=False))
    if entry.chars is None:
        entry.chars = len(json.dumps(body, sort_keys=True, ensure_ascii=False))
    return entry.chars


def canonical_digest(body: Mapping[str, object], compute: Callable[[object], str]) -> str:
    """Memoize ``compute(body)`` (the validated canonical sha256 identity)."""
    entry = _entry(body)
    if entry is None:
        return compute(body)
    digest = entry.digests.get(compute)
    if digest is None:
        digest = entry.digests[compute] = compute(body)
    return digest
