"""Content-addressed parse cache shared by kernel analyzers (L112 host-time fix).

Live steps were re-parsing the same public sources tens of thousands of times
(41M ``ast.walk`` calls over three steps).  This module memoizes
``ast.parse`` and analyzer-derived maps keyed by ``(filename, sha256(source))``
so an unchanged file is parsed once per process, while ANY content change is a
different key (a stale entry can never hide an edit).

Contract for callers:
  * Trees returned by :func:`parse` are SHARED.  Treat them as read-only; a
    caller that mutates nodes must ``copy.deepcopy`` first.
  * :func:`derived` caches ``build(tree)``.  ``build`` must be a pure function
    of the tree, and its result must be immutable or copied by the caller.
  * A ``SyntaxError`` is never cached; it is re-raised exactly as
    ``ast.parse`` raises it, on every call.

Bounded: least-recently-used eviction at :data:`MAX_TREES` /
:data:`MAX_DERIVED` entries.  Thread-safe.
"""
from __future__ import annotations

import ast
import contextlib
import gc
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import TypeVar

MAX_TREES = 256
MAX_DERIVED = 4096  # several analyzers x ~1k-file repos must not thrash each other

_T = TypeVar("_T")
_LOCK = threading.RLock()
_TREES: "OrderedDict[tuple[str, str], ast.Module]" = OrderedDict()
_DERIVED: "OrderedDict[tuple[Hashable, str, str], object]" = OrderedDict()
_STATS = {"tree_hits": 0, "tree_misses": 0, "derived_hits": 0, "derived_misses": 0}


def source_digest(source: str | bytes) -> str:
    data = source if isinstance(source, bytes) else source.encode("utf-8", "surrogatepass")
    return hashlib.sha256(data).hexdigest()


def _remember(store: OrderedDict, key: object, value: object, limit: int) -> None:
    store[key] = value
    store.move_to_end(key)
    while len(store) > limit:
        store.popitem(last=False)


def parse(source: str, filename: str = "<unknown>") -> ast.Module:
    """``ast.parse(source, filename=filename)``, shared per content (read-only tree)."""
    if not isinstance(source, str):
        return ast.parse(source, filename=filename)
    key = (str(filename), source_digest(source))
    with _LOCK:
        tree = _TREES.get(key)
        if tree is not None:
            _TREES.move_to_end(key)
            _STATS["tree_hits"] += 1
            return tree
    tree = ast.parse(source, filename=filename)  # SyntaxError propagates, never cached
    with _LOCK:
        _STATS["tree_misses"] += 1
        existing = _TREES.get(key)
        if existing is not None:  # a concurrent parse won; keep one shared tree
            return existing
        _remember(_TREES, key, tree, MAX_TREES)
    return tree


def derived(kind: Hashable, source: str, filename: str,
            build: Callable[[ast.Module], _T]) -> _T:
    """Cache ``build(parse(source, filename))`` under ``(kind, filename, sha256)``."""
    key = (kind, str(filename), source_digest(source))
    with _LOCK:
        if key in _DERIVED:
            _DERIVED.move_to_end(key)
            _STATS["derived_hits"] += 1
            return _DERIVED[key]  # type: ignore[return-value]
    value = build(parse(source, filename))
    with _LOCK:
        _STATS["derived_misses"] += 1
        _remember(_DERIVED, key, value, MAX_DERIVED)
    return value


def derived_source(kind: Hashable, source: str, filename: str,
                   build: Callable[[str], _T]) -> _T:
    """Cache ``build(source)`` (no parse) under ``(kind, filename, sha256)``."""
    key = (("source", kind), str(filename), source_digest(source))
    with _LOCK:
        if key in _DERIVED:
            _DERIVED.move_to_end(key)
            _STATS["derived_hits"] += 1
            return _DERIVED[key]  # type: ignore[return-value]
    value = build(source)
    with _LOCK:
        _STATS["derived_misses"] += 1
        _remember(_DERIVED, key, value, MAX_DERIVED)
    return value


def parse_file(path: str | Path, *, encoding: str = "utf-8") -> tuple[str, ast.Module]:
    """Read ``path`` and parse it; the key is the absolute path plus content hash."""
    absolute = Path(path).absolute()
    source = absolute.read_text(encoding=encoding)
    return source, parse(source, str(absolute))


@contextlib.contextmanager
def gc_paused():
    """Pause CYCLIC garbage collection around one bounded bulk index build.

    Building a repository index allocates millions of small, long-lived,
    acyclic objects; generational GC repeatedly re-traverses them for nothing
    (measured: ~40% of a cold ``scan_repository``).  Reference counting still
    frees everything as usual; only cycle detection is deferred, and the prior
    state is restored on exit (nesting-safe).  No result can change.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def clear() -> None:
    with _LOCK:
        _TREES.clear()
        _DERIVED.clear()
        for name in _STATS:
            _STATS[name] = 0


def stats() -> dict[str, int]:
    with _LOCK:
        return {**_STATS, "trees": len(_TREES), "derived": len(_DERIVED)}


__all__ = ["MAX_DERIVED", "MAX_TREES", "clear", "derived", "derived_source", "gc_paused", "parse", "parse_file",
           "source_digest", "stats"]
