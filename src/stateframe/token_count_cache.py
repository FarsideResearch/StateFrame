"""Bounded process-local memo for exact tokenizer counts."""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future


DEFAULT_MAX_ENTRIES = 4096


class ExactTokenCountCache:
    """Share exact counts by tokenizer identity and digest of the full text."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("token-count cache capacity must be a positive integer")
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._inflight: dict[tuple[str, str], Future[int]] = {}
        self._lock = threading.Lock()
        self._lookups = 0
        self._hits = 0
        self._misses = 0
        self._waits = 0
        self._computations = 0

    def get_or_compute(
            self, identity: str, content: str, compute: Callable[[], int], *,
            deadline: float | None = None,
    ) -> int:
        if not isinstance(identity, str) or not identity:
            raise ValueError("tokenizer identity must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError("tokenizer content must be a string")
        if not callable(compute):
            raise TypeError("token-count computation must be callable")
        key = (identity, hashlib.sha256(content.encode("utf-8")).hexdigest())
        with self._lock:
            self._lookups += 1
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return cached
            self._misses += 1
            future = self._inflight.get(key)
            if future is None:
                future = Future()
                self._inflight[key] = future
                self._computations += 1
                leader = True
            else:
                self._waits += 1
                leader = False

        if not leader:
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            if timeout == 0.0:
                raise TimeoutError("token-count cache wait deadline expired")
            return future.result(timeout=timeout)

        try:
            count = compute()
            if type(count) is not int or count < 0:
                raise ValueError("token-count computation must return a non-negative integer")
            with self._lock:
                self._entries[key] = count
                self._entries.move_to_end(key)
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
            future.set_result(count)
            return count
        except BaseException as error:
            with self._lock:
                if self._inflight.get(key) is future:
                    del self._inflight[key]
            future.set_exception(error)
            raise
        finally:
            with self._lock:
                if self._inflight.get(key) is future:
                    del self._inflight[key]

    def cache_info(self) -> dict[str, int]:
        with self._lock:
            return {
                "lookups": self._lookups,
                "hits": self._hits,
                "misses": self._misses,
                "waits": self._waits,
                "computations": self._computations,
                "entries": len(self._entries),
                "inflight": len(self._inflight),
                "max_entries": self._max_entries,
            }


_CACHE = ExactTokenCountCache()


def get_or_compute(
        identity: str, content: str, compute: Callable[[], int], *,
        deadline: float | None = None,
) -> int:
    return _CACHE.get_or_compute(identity, content, compute, deadline=deadline)


def cache_info() -> dict[str, int]:
    return _CACHE.cache_info()
