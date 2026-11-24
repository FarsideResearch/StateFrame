"""Content-addressed parse cache shared by kernel analyzers (L112 host-time fix).

Live steps were re-parsing the same public sources tens of thousands of times
(41M ``ast.walk`` calls over three steps).  This module memoizes
``ast.parse`` by source digest and analyzer-derived maps by filename and source
digest. A valid tree can be reused by analyzers that identify the same source
with different paths. Content changes always get a different key.

Contract for callers:
  * Trees returned by :func:`parse` are SHARED.  Treat them as read-only; a
    caller that mutates nodes must ``copy.deepcopy`` first.
  * :func:`derived` caches ``build(tree)``.  ``build`` must be a pure function
    of the tree, and its result must be immutable or copied by the caller.
  * A ``SyntaxError`` is never cached; it is re-raised exactly as
    ``ast.parse`` raises it, on every call.

Bounded: least-recently-used tree eviction at :data:`MAX_TREE_RETAINED_BYTES`,
using an allocation-block estimate or an exact walk when that estimate is weak,
with :data:`MAX_TREES` as a count backstop; derived maps have
:data:`MAX_DERIVED` entries.  Thread-safe.
"""
from __future__ import annotations

import ast
import contextlib
import gc
import hashlib
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import TypeVar

if __package__:
    from .kernel_turn_work_counters import count_work
else:
    from kernel_turn_work_counters import count_work

# ``getallocatedblocks`` deltas measured 70-79 retained bytes per block across
# plain, table, dense-line, string, and f-string sources. 96 adds headroom. A
# 3 MB docstring retains roughly its source size while adding only 18 blocks, so
# one source byte per byte is the measured floor for large constants.
MIN_TREE_RETAINED_BYTES = 512
_AST_BYTES_PER_ALLOCATED_BLOCK = 96
_AST_MIN_BYTES_PER_SOURCE_BYTE = 1
MAX_TREE_RETAINED_BYTES = 256 * 1024 * 1024
MAX_TREES = 16_384  # backstop for tiny/empty sources that barely use the byte budget
MAX_DERIVED = 4096  # several analyzers x ~1k-file repos must not thrash each other

_T = TypeVar("_T")
_LOCK = threading.RLock()
_GC_USERS = 0
_GC_WAS_ENABLED = False
_TREES: OrderedDict[str, tuple[ast.Module, int, int]] = OrderedDict()
_TREE_SOURCE_BYTES = 0
_TREE_RETAINED_BYTES = 0
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


def _allocated_blocks() -> int | None:
    """Return a usable interpreter allocation counter, if available."""
    counter = getattr(sys, "getallocatedblocks", None)
    if not callable(counter):
        return None
    try:
        blocks = counter()
    except Exception:
        return None
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0:
        return None
    return blocks


def _estimate_tree_retained_bytes(
    source: bytes, tree: ast.Module, allocated_blocks: int = 0,
) -> int:
    """Estimate retained AST bytes, walking exactly when the block delta is weak.

    The allocation delta is measured immediately around ``ast.parse``. Source
    bytes are also counted because large string constants can consume megabytes
    while adding very few allocated blocks. Weak deltas, including missing
    counters and concurrent frees, use the tree's exact reachable object sizes.
    """
    source_bytes = len(source)
    delta = max(0, allocated_blocks)
    if delta * _AST_BYTES_PER_ALLOCATED_BLOCK < 32 * source_bytes:
        return max(MIN_TREE_RETAINED_BYTES, _object_size_bytes(tree))
    return max(
        MIN_TREE_RETAINED_BYTES,
        source_bytes * _AST_MIN_BYTES_PER_SOURCE_BYTE,
        delta * _AST_BYTES_PER_ALLOCATED_BLOCK + source_bytes,
    )


def _object_size_bytes(root: object) -> int:
    """Count each object in an AST/container result once, without recursion."""
    pending = [root]
    seen: set[int] = set()
    total = 0
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(value)
        if isinstance(value, ast.AST):
            pending.append(vars(value))
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            pending.append(vars(value))
    return total


def parse(source: str, filename: str = "<unknown>", *,
          on_miss: Callable[[], None] | None = None) -> ast.Module:
    """``ast.parse(source, filename=filename)``, shared per content (read-only tree)."""
    global _TREE_SOURCE_BYTES, _TREE_RETAINED_BYTES
    if not isinstance(source, str):
        if on_miss is not None:
            on_miss()
        count_work("parses")
        return ast.parse(source, filename=filename)
    data = source.encode("utf-8", "surrogatepass")
    source_bytes = len(data)
    key = source_digest(data)
    with _LOCK:
        entry = _TREES.get(key)
        if entry is not None:
            _TREES.move_to_end(key)
            _STATS["tree_hits"] += 1
            return entry[0]
    if on_miss is not None:
        on_miss()
    count_work("parses")
    # Avoid automatic cyclic collection changing the process-global counter
    # during this interval; a reduced delta falls back to exact tree sizing.
    with gc_paused():
        blocks_before = _allocated_blocks()
        tree = ast.parse(source, filename=filename)  # SyntaxError propagates, never cached
        blocks_after = _allocated_blocks() if blocks_before is not None else None
    try:
        retained_bytes = _estimate_tree_retained_bytes(
            data, tree,
            max(0, blocks_after - blocks_before)
            if blocks_before is not None and blocks_after is not None else 0,
        )
    except Exception:
        # Sizing is an optimization. Keep the parse result, but do not retain a
        # tree whose memory could not be estimated safely.
        retained_bytes = None
    with _LOCK:
        _STATS["tree_misses"] += 1
        existing = _TREES.get(key)
        if existing is not None:  # a concurrent parse won; keep one shared tree
            _TREES.move_to_end(key)
            return existing[0]
        # A single oversized file must not flush the warm working set.
        if (retained_bytes is None or retained_bytes > MAX_TREE_RETAINED_BYTES
                or MAX_TREES <= 0):
            return tree
        _TREES[key] = (tree, source_bytes, retained_bytes)
        _TREE_SOURCE_BYTES += source_bytes
        _TREE_RETAINED_BYTES += retained_bytes
        while len(_TREES) > MAX_TREES or _TREE_RETAINED_BYTES > MAX_TREE_RETAINED_BYTES:
            _, (_, evicted_source_bytes, evicted_retained_bytes) = _TREES.popitem(last=False)
            _TREE_SOURCE_BYTES -= evicted_source_bytes
            _TREE_RETAINED_BYTES -= evicted_retained_bytes
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
    """Read ``path`` and parse it, using the absolute path for diagnostics."""
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
    global _GC_USERS, _GC_WAS_ENABLED
    with _LOCK:
        if not _GC_USERS:
            _GC_WAS_ENABLED = gc.isenabled()
            gc.disable()
        _GC_USERS += 1
    try:
        yield
    finally:
        with _LOCK:
            _GC_USERS -= 1
            if not _GC_USERS and _GC_WAS_ENABLED:
                gc.enable()


def clear() -> None:
    global _TREE_SOURCE_BYTES, _TREE_RETAINED_BYTES
    with _LOCK:
        _TREES.clear()
        _TREE_SOURCE_BYTES = 0
        _TREE_RETAINED_BYTES = 0
        _DERIVED.clear()
        for name in _STATS:
            _STATS[name] = 0


def stats() -> dict[str, int]:
    with _LOCK:
        return {**_STATS, "trees": len(_TREES), "derived": len(_DERIVED),
                "tree_source_bytes": _TREE_SOURCE_BYTES,
                "tree_retained_bytes": _TREE_RETAINED_BYTES}


__all__ = ["MAX_DERIVED", "MAX_TREES", "MAX_TREE_RETAINED_BYTES",
           "MIN_TREE_RETAINED_BYTES", "clear", "derived", "derived_source",
           "gc_paused", "parse", "parse_file", "source_digest", "stats"]
