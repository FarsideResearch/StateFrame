"""Byte-bounded output retention and streaming literal readiness detection."""
from __future__ import annotations

import codecs
import os
from pathlib import Path


class LiteralMatcher:
    """Retain at most len(text)-1 characters, including one-character patterns."""

    def __init__(self, text: str):
        if not isinstance(text, str) or not text or len(text.encode("utf-8")) > 65536:
            raise ValueError("literal text must contain 1..65536 UTF-8 bytes")
        self.text = text
        self.matched = False
        self._tail = ""

    def feed(self, text: str) -> bool:
        if not self.matched:
            candidate = self._tail + text
            self.matched = self.text in candidate
            keep = len(self.text) - 1
            self._tail = candidate[-keep:] if keep else ""
        return self.matched


class JobOutputCollector:
    """Ring on disk and in memory. Disk size never exceeds capacity bytes.

    The spool is circular, not a chronological log; use snapshot() to read it.
    read_since() exposes loss explicitly for consumers with independent cursors.
    """

    def __init__(self, path: Path, capacity: int = 1 << 20):
        if type(capacity) is not int or not 0 < capacity <= 4 << 20:
            raise ValueError("output capacity must be 1..4194304 bytes")
        self.path = Path(path)
        self.capacity = capacity
        self.total_bytes = 0
        self.lost_bytes = 0
        self._ring = bytearray()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._matchers: list[LiteralMatcher] = []
        self._finished = False
        self._sealed = False
        self._file = self.path.open("xb", buffering=0)

    def watch(self, text: str) -> LiteralMatcher:
        if len(self._matchers) >= 16:
            raise ValueError("at most 16 active literal matchers")
        matcher = LiteralMatcher(text)
        matcher.feed(self.snapshot().decode("utf-8", "replace"))
        self._matchers.append(matcher)
        return matcher

    def unwatch(self, matcher: LiteralMatcher) -> None:
        self._matchers.remove(matcher)

    def feed(self, data: bytes) -> None:
        if self._finished:
            raise ValueError("output collector already finished")
        # Bound temporary decode/matcher state even for a large caller-provided chunk.
        for offset in range(0, len(data), 65536):
            chunk = data[offset:offset + 65536]
            text = self._decoder.decode(chunk)
            for matcher in self._matchers:
                matcher.feed(text)
            start = self.total_bytes % self.capacity
            self.total_bytes += len(chunk)
            retained = chunk[-self.capacity:]
            if len(chunk) >= self.capacity:
                start = (self.total_bytes - len(retained)) % self.capacity
            first = min(len(retained), self.capacity - start)
            self._file.seek(start)
            self._file.write(retained[:first])
            if first < len(retained):
                self._file.seek(0)
                self._file.write(retained[first:])
            self._ring.extend(retained)
            if len(self._ring) > self.capacity:
                del self._ring[:-self.capacity]
            self.lost_bytes = max(0, self.total_bytes - self.capacity)

    def drain(self, fd: int, *, budget: int = 256 << 10) -> bool:
        """Drain available bytes with bounded work; return True on EOF only."""
        os.set_blocking(fd, False)
        remaining = budget
        while remaining > 0:
            try:
                data = os.read(fd, min(65536, remaining))
            except BlockingIOError:
                return False
            if not data:
                self.finish()
                return True
            self.feed(data)
            remaining -= len(data)
        return False

    def finish(self) -> None:
        if not self._finished:
            text = self._decoder.decode(b"", final=True)
            for matcher in self._matchers:
                matcher.feed(text)
            self._finished = True

    def snapshot(self) -> bytes:
        if not self._sealed:
            return bytes(self._ring)
        data = self.path.read_bytes()
        if self.total_bytes >= self.capacity:
            start = self.total_bytes % self.capacity
            return data[start:] + data[:start]
        return data

    def seal(self) -> None:
        """Release memory and the descriptor; historical reads use the disk ring."""
        self.finish()
        self._file.close()
        self._ring.clear()
        self._sealed = True

    def read_since(self, cursor: int) -> tuple[bytes, int, int]:
        if type(cursor) is not int or cursor < 0 or cursor > self.total_bytes:
            raise ValueError("output cursor outside stream")
        data = self.snapshot()
        first = self.total_bytes - len(data)
        lost = max(0, first - cursor)
        return data[max(0, cursor - first):], self.total_bytes, lost

    def close(self) -> None:
        self.seal()
