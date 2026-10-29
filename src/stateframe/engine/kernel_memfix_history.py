"""Bounded decoded views over MEMGROW2's authenticated history spool."""
from __future__ import annotations

import copy
import os
import pickle
import sys
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence

from kernel_history_spool import HistorySpool


EVENT_CACHE_BYTES = 8 * 1024 * 1024
LEDGER_CACHE_BYTES = 2 * 1024 * 1024


def legacy() -> bool:
    return os.environ.get("SF_LAB_MEMFIX", "on").strip().lower() == "legacy"


def _resident_bytes(value: object) -> int:
    """Account for the decoded tree, not its much smaller pickle representation."""
    seen: set[int] = set()

    def visit(item: object) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(item)
        if isinstance(item, dict):
            size += sum(visit(key) + visit(child) for key, child in item.items())
        elif isinstance(item, (list, tuple, set, frozenset)):
            size += sum(visit(child) for child in item)
        return size

    return visit(value)


class MemfixSpool(HistorySpool):
    """HistorySpool whose restore stays spooled if MEMGROW2 is in legacy mode."""

    _memfix_forced = True


class _DecodedCache:
    def __init__(self, budget: int):
        self._budget = budget
        self._cache: OrderedDict[object, tuple[object, int]] = OrderedDict()
        self.cache_bytes = 0

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cache_bytes = 0

    def _cached(self, key: object) -> object | None:
        entry = self._cache.get(key)
        if entry is not None:
            self._cache.move_to_end(key)
            return copy.deepcopy(entry[0])
        return None

    def _remember(self, key: object, value: object) -> None:
        previous = self._cache.pop(key, None)
        if previous is not None:
            self.cache_bytes -= previous[1]
        size = _resident_bytes(value)
        if size > self._budget:
            return
        value = copy.deepcopy(value)
        while self.cache_bytes + size > self._budget:
            _, (_, removed_size) = self._cache.popitem(last=False)
            self.cache_bytes -= removed_size
        self._cache[key] = (value, size)
        self.cache_bytes += size


class SpoolEventMap(_DecodedCache, Mapping):
    """Insertion-ordered event-id map; only IDs and bounded decoded values live in RAM."""

    def __init__(self, directory=None):
        super().__init__(EVENT_CACHE_BYTES)
        self._rows = MemfixSpool(directory=directory)
        self._positions: dict[str, int] = {}

    def append(self, key: str, value: dict) -> None:
        if key in self._positions:
            raise ValueError("duplicate host evidence identity")
        self._rows.append(value)
        self._positions[key] = len(self._rows) - 1
        self._remember(key, value)

    def __getitem__(self, key: str) -> dict:
        position = self._positions[key]
        cached = self._cached(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        value = self._rows[position]
        self._remember(key, value)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._positions)

    def __len__(self) -> int:
        return len(self._positions)

    def __getstate__(self):
        return self._rows, self._positions

    def __setstate__(self, state):
        _DecodedCache.__init__(self, EVENT_CACHE_BYTES)
        self._rows, self._positions = state


class ReceiptAliases(Mapping):
    """Receipt IDs resolve through the owning event map without retaining payloads."""

    def __init__(self, events: SpoolEventMap):
        self._events = events
        self._event_ids: dict[str, str] = {}

    def add(self, receipt_id: str, event_id: str) -> None:
        if receipt_id in self._event_ids:
            raise ValueError("duplicate host receipt identity")
        self._event_ids[receipt_id] = event_id

    def __getitem__(self, receipt_id: str) -> dict:
        return self._events[self._event_ids[receipt_id]]

    def __iter__(self) -> Iterator[str]:
        return iter(self._event_ids)

    def __len__(self) -> int:
        return len(self._event_ids)


class SpoolSequence(_DecodedCache, Sequence):
    """Write-through ledger sequence with slicing and last-row replacement."""

    def __init__(self, directory=None):
        super().__init__(LEDGER_CACHE_BYTES)
        self._rows = MemfixSpool(directory=directory)

    def append(self, value: dict) -> None:
        self._rows.append(value)
        self._remember(len(self._rows) - 1, value)

    def extend(self, values) -> None:
        for value in values:
            self.append(value)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        position = self._rows._normalize(index)
        cached = self._cached(position)
        if cached is not None:
            return cached
        value = self._rows[position]
        self._remember(position, value)
        return value

    def __setitem__(self, index, value: dict) -> None:
        position = self._rows._normalize(index)
        self._rows._replace_encoded(position, pickle.dumps(value, protocol=5))
        self._remember(position, value)

    def __iter__(self) -> Iterator[dict]:
        # Full ledger scans must not decode, size and admit every old row into
        # the small random-access cache on every frame projection.
        yield from self._rows

    def __getstate__(self):
        return (self._rows,)

    def __setstate__(self, state):
        _DecodedCache.__init__(self, LEDGER_CACHE_BYTES)
        (self._rows,) = state


def ledger_records(directory=None):
    return [] if legacy() else SpoolSequence(directory=directory)
