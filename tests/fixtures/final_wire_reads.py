"""Audit a finished run: every page the model just read must be on the next request's final wire.

The source of truth is the run journal. ``observation_committed`` records the read page the
model was shown, and ``request_submitted`` records the exact payload sent to the provider
(hash-checked here). A read whose bytes are absent from the next submitted payload is a
violation. A read after the last submitted request is not checked: no wire carried it, so
nothing false reached the model (the engine stopped, or refused and logged it).
"""
from __future__ import annotations

import hashlib
import json
import lzma
import re
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_READ_PAGE = re.compile(r"^\[read page\] path=(\S+) revision=(sha256:[0-9a-f]+) lines=(\d+)-(\d+)/(\d+)[^\n]*\n")
_NOTICE = re.compile(r"^(\d+ more(?: lines)?;|\[read\] )")
# Pages shorter than this carry too little to prove anything about the wire.
_MIN_PROOF_CHARS = 80


@dataclass(frozen=True)
class MissingRead:
    path: str
    lines: tuple[int, int]
    observation_sequence: int
    next_request_id: str | None
    missing_chars: int

    def describe(self) -> str:
        target = self.next_request_id or "no later request"
        return (f"{self.path}:{self.lines[0]}-{self.lines[1]} (observation {self.observation_sequence}) "
                f"absent from {target}: {self.missing_chars} chars missing")


class _Journal:
    def __init__(self, run_dir: Path) -> None:
        journals = sorted((run_dir / "worker" / "measurement-runs").glob("*/events.jsonl*"))
        if len(journals) != 1:
            raise AssertionError(f"expected one measurement journal under {run_dir}, found {len(journals)}")
        self.path = journals[0]
        self.blobs = self.path.parent / "blobs"
        index = self.blobs / "pack-0001.idx"
        self.index = json.loads(index.read_text())["records"] if index.is_file() else {}
        opener = lzma.open if self.path.suffix == ".xz" else open
        with opener(self.path, "rt", encoding="utf-8") as handle:
            self.events = [json.loads(line) for line in handle if line.strip()]
        self._cache: dict[str, object] = {}

    def _blob(self, digest: str) -> object:
        if digest in self._cache:
            return self._cache[digest]
        if digest in self.index:
            offset, size = self.index[digest]
            with (self.blobs / "pack-0001.sfpk").open("rb") as pack:
                pack.seek(offset)
                header = pack.read(40)
                magic, length, _ = struct.unpack(">4sI32s", header)
                data = pack.read(length)
            if magic != b"SFPK" or length != size:
                raise AssertionError(f"corrupt journal pack record {digest}")
        else:
            path = self.blobs / digest
            data = (path if path.exists() else self.blobs / f"{digest}.zlib").read_bytes()
        if not data.startswith((b"{", b"[", b'"')):
            data = zlib.decompress(data)
        value = self.hydrate(json.loads(data))
        self._cache[digest] = value
        return value

    def hydrate(self, value: object) -> object:
        if isinstance(value, dict):
            if "$blob" in value:
                return self.hydrate(self._blob(value["$blob"]))
            return {key: self.hydrate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.hydrate(item) for item in value]
        return value


def _strings(value: object) -> Iterator[str]:
    """Every string in a message, including JSON nested inside tool observations."""
    if isinstance(value, str):
        yield value
        for candidate in (value, *(line for line in value.splitlines() if line.startswith(("{", "[")))):
            try:
                decoded = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            if not isinstance(decoded, str):
                yield from _strings(decoded)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _wire_text(payload: str) -> str:
    body = json.loads(payload)
    messages = body.get("messages") if isinstance(body, dict) else None
    return "\n".join(text for message in messages or [] for text in _strings(message))


def _missing_chars(page: str, wire: str) -> int:
    """Characters of the page body not covered by any line that appears on the wire."""
    missing = 0
    for line in page.splitlines():
        if line.strip() and line not in wire:
            missing += len(line)
    return missing


def missing_reads(run_dir: Path) -> tuple[list[MissingRead], int]:
    """Return (violations, reads checked) for a finished run directory."""
    journal = _Journal(run_dir)
    submits: list[tuple[int, str, str]] = []
    for event in journal.events:
        if event.get("kind") != "request_submitted":
            continue
        payload = journal.hydrate(event.get("payload_utf8"))
        if not isinstance(payload, str):
            raise AssertionError(f"request_submitted {event.get('sequence')} has no payload")
        digest = str(event.get("payload_hash", "")).removeprefix("sha256:")
        if hashlib.sha256(payload.encode()).hexdigest() != digest:
            raise AssertionError(f"request_submitted {event.get('sequence')} payload hash mismatch")
        submits.append((int(event["sequence"]), str(event.get("request_id")), payload))

    violations: list[MissingRead] = []
    checked = 0
    for event in journal.events:
        if event.get("kind") != "observation_committed":
            continue
        observation = journal.hydrate(event).get("observation")  # type: ignore[union-attr]
        text = observation.get("observation") if isinstance(observation, dict) else None
        header = _READ_PAGE.match(text) if isinstance(text, str) else None
        if header is None:
            continue
        lines = (int(header[3]), int(header[4]))
        # Only the source lines count, not the cursor and reuse notices around them.
        source = [line for line in text[header.end():].splitlines() if not _NOTICE.match(line)]
        page = "\n".join(source[: lines[1] - lines[0] + 1])
        if len(page.strip()) < _MIN_PROOF_CHARS:
            continue
        sequence = int(event["sequence"])
        following = next(((seq, rid, payload) for seq, rid, payload in submits if seq > sequence), None)
        if following is None:
            # The run ended after this read (tape exhausted or an honest refusal); there is
            # no next request whose wire could have dropped it.
            continue
        checked += 1
        missing = _missing_chars(page, _wire_text(following[2]))
        if missing:
            violations.append(MissingRead(header[1], lines, sequence, following[1], missing))
    return violations, checked
