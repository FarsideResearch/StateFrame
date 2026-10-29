"""Bounded request-local JSON fragment memo for capacity fitting.

Fragments are byte/character accounting, never additive model-token evidence.
Mutable containers are keyed from their current children's content digests.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json


class JsonFragmentMemo:
    def __init__(self, max_chars: int = 1_048_576) -> None:
        self._cache: OrderedDict[bytes, str] = OrderedDict()
        self._chars = 0
        self._max_chars = max_chars

    def dumps(self, value: object) -> str:
        return self._fragment(value)[1]

    def _fragment(self, value: object) -> tuple[bytes, str]:
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            children = [(self._fragment(key), self._fragment(value[key])) for key in sorted(value)]
            key = hashlib.sha256(b"object" + b"".join(
                name[0] + item[0] for name, item in children)).digest()
            def render() -> str:
                return "{" + ",".join(name[1] + ":" + item[1] for name, item in children) + "}"
        elif isinstance(value, (list, tuple)):
            children = [self._fragment(item) for item in value]
            key = hashlib.sha256(b"array" + b"".join(item[0] for item in children)).digest()
            def render() -> str:
                return "[" + ",".join(item[1] for item in children) + "]"
        else:
            # Large source strings are hashed before escaping, so unchanged
            # fragments avoid repeated JSON escaping on every eviction.
            if isinstance(value, str):
                key = hashlib.sha256(b"string" + value.encode("utf-8", "surrogatepass")).digest()
            else:
                text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                key = hashlib.sha256(b"scalar" + text.encode("utf-8", "surrogatepass")).digest()
            def render() -> str:
                return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return key, cached
        text = render()
        if len(text) <= self._max_chars:
            while self._cache and self._chars + len(text) > self._max_chars:
                self._chars -= len(self._cache.popitem(last=False)[1])
            self._cache[key] = text
            self._chars += len(text)
        return key, text


def applied_edit_reductions(edits: list[dict[str, object]]):
    """Yield old rows while promoting the next same-path edit into its slot."""
    from collections import deque

    original = list(edits)
    positions = {id(row): index for index, row in enumerate(original)}
    by_path: dict[object, deque] = {}
    for row in original:
        by_path.setdefault(row.get("file"), deque()).append(row)
    slots = list(original)
    for row in original:
        position = positions.get(id(row))
        if position is None:
            continue
        peers = by_path[row.get("file")]
        if len(peers) < 2:
            continue
        peers.popleft()
        replacement = peers[0]
        slots[positions.pop(id(replacement))] = None
        slots[position] = replacement
        positions.pop(id(row))
        positions[id(replacement)] = position
        edits[:] = [item for item in slots if item is not None]
        yield row
