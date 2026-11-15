"""Independent channel I/O and lifecycle lanes for a nonblocking tick façade.

The lifecycle lane retains Popen/pins/reaping authority in this process. Weak owner
references let abandoned supervisors shut their helpers down without retaining runs.
"""
from __future__ import annotations

from collections import deque
import threading
import time
import weakref


class ProtocolErrorLog:
    """Bounded fault log for the service lanes: a 10 ms / 50 ms lane with a persistent fault
    would otherwise append one message per pass without bound. Identical messages collapse
    into a repeat counter; distinct messages keep only the last `capacity` entries, with the
    number of evicted messages accounted in `dropped`."""

    def __init__(self, capacity: int = 256) -> None:
        self._entries: deque[str] = deque(maxlen=capacity)
        self._repeats: dict[str, int] = {}
        self._lock = threading.Lock()  # both lanes and finalize threads append concurrently
        self.total = 0
        self.dropped = 0

    def append(self, message: str) -> None:
        with self._lock:
            self.total += 1
            if message in self._repeats:
                self._repeats[message] += 1
                return
            if len(self._entries) == self._entries.maxlen:
                evicted = self._entries[0]
                self._repeats.pop(evicted, None)
                self.dropped += 1
            self._entries.append(message)
            self._repeats[message] = 1

    def extend(self, messages) -> None:
        for message in messages:
            self.append(message)

    def repeats(self, message: str) -> int:
        return self._repeats.get(message, 0)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(tuple(self._entries))

    def __getitem__(self, index):
        return tuple(self._entries)[index]

    def __contains__(self, message: object) -> bool:
        return message in self._repeats

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ProtocolErrorLog):
            return list(self._entries) == list(other._entries)
        if isinstance(other, (list, tuple)):
            return list(self._entries) == list(other)
        return NotImplemented

    __hash__ = None  # mutable, like the list it replaces

    def __repr__(self) -> str:
        return f"ProtocolErrorLog(total={self.total}, dropped={self.dropped}, entries={list(self._entries)!r})"


def channel_loop(reference: weakref.ReferenceType) -> None:
    while True:
        owner = reference()
        if owner is None or owner._runtime_stop:
            return
        try:
            owner._service_channels()
        except Exception as error:
            owner._protocol_errors.append(f"channel service deferred: {type(error).__name__}: {str(error)[:300]}")
        finally:
            del owner
        time.sleep(0.05)


def lifecycle_loop(reference: weakref.ReferenceType) -> None:
    while True:
        owner = reference()
        if owner is None or owner._runtime_stop:
            return
        try:
            owner._service_lifecycle()
        except Exception as error:
            owner._protocol_errors.append(f"lifecycle service deferred: {type(error).__name__}: {str(error)[:300]}")
        finally:
            del owner
        time.sleep(0.05)
