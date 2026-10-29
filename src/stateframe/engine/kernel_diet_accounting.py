"""Request-local byte accounting for removal-only message fitting.

Callers own an immutable envelope and unchanged retained message objects during
the pass. Rewrites must replace objects or end the pass. These totals are never
model-token evidence; verify the whole body before publishing a selection.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence


class RemovalAccounting:
    def __init__(self, body: Mapping[str, object],
                 wire_bytes: Callable[[object], int]) -> None:
        self._wire_bytes = wire_bytes
        self._sizes: dict[int, tuple[object, int]] = {}
        messages = body.get("messages", [])
        if not isinstance(messages, list):
            raise TypeError("removal accounting requires list messages")
        self._counts = Counter(id(message) for message in messages)
        self._length = len(messages)
        self.total = wire_bytes(body)
        for message in messages:
            self.item_size(message)

    def item_size(self, message: object) -> int:
        key = id(message)
        if key not in self._sizes:
            self._sizes[key] = (message, self._wire_bytes(message))
        return self._sizes[key][1]

    def update(self, messages: Sequence[object]) -> int:
        counts = Counter(id(message) for message in messages)
        for message in messages:
            self.item_size(message)
        for key in self._counts.keys() | counts.keys():
            self.total += (counts[key] - self._counts[key]) * self._sizes[key][1]
        # json.dumps uses ', ' between list members, including non-dict items.
        self.total += 2 * (max(0, len(messages) - 1) - max(0, self._length - 1))
        self._sizes = {key: self._sizes[key] for key in counts}
        self._counts = counts
        self._length = len(messages)
        return self.total

    def remove(self, message: object) -> None:
        """Account for one known pop without traversing retained messages."""
        key = id(message)
        self.total -= self._sizes[key][1] + (2 if self._length > 1 else 0)
        self._length -= 1
        self._counts[key] -= 1
        if not self._counts[key]:
            del self._counts[key]
            del self._sizes[key]

    def verify(self, body: Mapping[str, object]) -> bool:
        return self.total == self._wire_bytes(body)
