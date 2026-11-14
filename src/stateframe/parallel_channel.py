"""Polled durable channel writes, isolated from the worker commit FIFO."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import time

from .engine.kernel_parallel_records import canonical_json
from .parallel_journal import (
    Journal, JournalBusy, JournalFault, JournalProtocolError, Ticket, _MAX_PENDING, checked_response,
)

_MAX_WRITE_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 0.05
_MAX_RECORD_BYTES = 1 << 20


def sync_record(payload: dict[str, object]) -> None:
    """Revalidate recovered bytes in the helper without rewriting their representation."""
    from .parallel_host import ParallelHostError, _fsync_directory

    path = Path(str(payload["path"]))
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as record:
        metadata = os.fstat(record.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_RECORD_BYTES:
            raise ParallelHostError("channel record is not a bounded regular file")
        raw = record.read(_MAX_RECORD_BYTES + 1)
        if len(raw) > _MAX_RECORD_BYTES or hashlib.sha256(raw).hexdigest() != payload["digest"]:
            raise ParallelHostError("channel record digest changed before durability validation")
        os.fsync(record.fileno())
        _fsync_directory(path.parent)


def _record(path: Path, label: str) -> tuple[dict[str, object], str] | None:
    from .parallel_host import ParallelHostError, _regular_bytes, _strict_object

    if path.is_symlink():
        raise ParallelHostError(f"{label} is a symlink")
    if not path.exists():
        return None
    raw = _regular_bytes(path, maximum=_MAX_RECORD_BYTES, label=label)
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except (UnicodeError, ValueError) as error:
        raise ParallelHostError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ParallelHostError(f"{label} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def read_synced_record(path: Path, label: str) -> dict[str, object] | None:
    """Foreground/partial supervisors have no lane; synchronously validate captured bytes."""
    record = _record(path, label)
    if record is None:
        return None
    value, digest = record
    sync_record({"path": str(path), "digest": digest})
    return value


@dataclass
class _Write:
    payload: dict[str, object]
    ticket: Ticket | None
    digest: str
    kind: str = "write"
    failures: int = 0
    retry_at: float = 0.0


@dataclass
class _Rejection:
    error_type: type[RuntimeError]
    message: str


class ChannelWrites:
    """Retain at most one journal intent per path until its durability ACK is consumed.

    Only foreground callers may wait. The channel owner polls, leaving an unanswered
    sequence retryable while other requests and lifecycle work continue.
    """

    def __init__(self) -> None:
        self.journal = Journal()
        self._pending: dict[Path, _Write] = {}
        self._acked: OrderedDict[Path, str] = OrderedDict()
        self._rejected: OrderedDict[tuple[Path, str, str], _Rejection] = OrderedDict()
        # Accounting callbacks must survive diagnostic-cache eviction. Reserve each
        # callback at publication admission and release it only when matching bytes ACK.
        self._answer_acks: dict[tuple[Path, str], Callable[[], None]] = {}
        self._fault: str | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        self.journal.close()

    def _settle(self, path: Path) -> Ticket | None:
        """Call under the writer lock. A failed write keeps its original retry intent."""
        from .parallel_host import ParallelHostError

        pending = self._pending.get(path)
        if pending is None:
            return None
        if pending.ticket is None:
            if time.monotonic() < pending.retry_at:
                raise JournalBusy("channel durability retry is backing off")
            pending.ticket = self._submit(pending.kind, pending.payload)
        ticket = pending.ticket
        result = self.journal.take(ticket)
        if result is None:
            return ticket
        pending.ticket = None  # take() returned the journal credit, including failed ACKs
        try:
            checked_response(result)
        except (ParallelHostError, JournalProtocolError) as error:
            self._reject(path, pending, error)
            raise
        except JournalFault as error:
            pending.failures += 1
            fault = getattr(self.journal, "fault", None)
            if fault or result.get("closed") is True or pending.failures >= _MAX_WRITE_ATTEMPTS:
                if fault or result.get("closed") is True or result.get("lost") is True:
                    self._fault = str(fault or error)
                self._reject(path, pending, error)
            else:
                # First retry is immediate; consecutive failures back off and retain one budget
                # across tickets instead of buying a fresh helper restart allowance forever.
                pending.retry_at = time.monotonic() + _RETRY_BACKOFF_SECONDS * (2 ** (pending.failures - 1) - 1)
            raise
        self._acked[path] = pending.digest
        self._acked.move_to_end(path)
        if len(self._acked) > _MAX_PENDING:
            self._acked.popitem(last=False)
        del self._pending[path]
        on_ack = self._answer_acks.pop((path, pending.digest), None)
        if on_ack is not None:
            # Memory-only observer: settlement owns this callback exactly once,
            # including when drain() consumes the ACK between channel passes.
            on_ack()
        return None

    def _submit(self, kind: str, payload: dict[str, object]) -> Ticket:
        fault = self._fault or getattr(self.journal, "fault", None)
        if fault:
            raise JournalFault(str(fault))
        return self.journal.submit(kind, payload)

    def _reject(self, path: Path, pending: _Write, error: RuntimeError) -> None:
        # Cache types and messages, never tracebacks that could retain the supervisor graph.
        key = (path, pending.kind, pending.digest)
        self._rejected[key] = _Rejection(type(error), str(error))
        self._rejected.move_to_end(key)
        if len(self._rejected) > _MAX_PENDING:
            self._rejected.popitem(last=False)
        del self._pending[path]

    def _check_rejection(self, path: Path, kind: str, digest: str) -> None:
        rejected = self._rejected.get((path, kind, digest))
        if rejected is not None:
            raise rejected.error_type(rejected.message)

    def barrier(self, path: Path) -> None:
        self.read(path, "channel record", wait=False)

    def read(self, path: Path, label: str, *, wait: bool) -> dict[str, object] | None:
        """Consume only bytes whose digest has an ACK, under the publication admission lock.

        A missing file returns None. Recovered or changed bytes first need a helper fsync ACK;
        the read then starts over, so neither a prior ACK nor a separate barrier blesses them.
        """
        while True:
            with self._lock:
                ticket = self._settle(path)
                if ticket is None:
                    record = _record(path, label)
                    if record is None:
                        return None
                    value, digest = record
                    if self._acked.get(path) == digest:
                        self._acked.move_to_end(path)
                        return value
                    self._check_rejection(path, "sync", digest)
                    if len(self._pending) >= _MAX_PENDING:
                        raise JournalBusy("channel write reservations are full; retry pending")
                    payload = {"path": str(path), "digest": digest}
                    ticket = self._submit("sync", payload)
                    self._pending[path] = _Write(payload, ticket, digest, kind="sync")
            if not wait:
                raise JournalBusy("channel read awaits its durable ACK; retry pending")
            ticket.done.wait()

    def drain(self) -> tuple[Path, ...]:
        """Return publication ACKs and release credits even for abandoned requests.

        Failed ACKs return journal credits and retain frozen intents for bounded retries,
        independently of visible files or whether the original caller is still scheduled.
        Read revalidation ACKs cannot release another intent's seal-publication fence.
        """
        completed = []
        from .parallel_host import ParallelHostError

        with self._lock:
            for path in tuple(self._pending):
                kind = self._pending[path].kind
                try:
                    if self._settle(path) is None and kind == "write":
                        completed.append(path)
                except (JournalBusy, JournalFault, ParallelHostError, JournalProtocolError):
                    continue  # each path keeps its own retry/rejection state; others still drain
        return tuple(completed)

    def write(self, path: Path, value: object, *, replace_existing: bool, wait: bool,
              on_ack: Callable[[], None] | None = None) -> None:
        """Freeze a nonblocking ACK observer with each accepted intent, across retries."""
        from .parallel_host import ParallelHostError

        encoded = canonical_json(value) + b"\n"
        digest = hashlib.sha256(encoded).hexdigest()
        while True:
            with self._lock:
                self._check_rejection(path, "write", digest)
                ticket = self._settle(path)
                if ticket is None:
                    if path.is_symlink():
                        raise ParallelHostError(f"refusing to write through symlink: {path}")
                    if path.exists():
                        same = path.read_bytes() == encoded
                        # A file recovered after a crash can be visible without a directory
                        # fsync ACK. Only an ACK from this runtime permits skipping the write.
                        if same and self._acked.get(path) == digest:
                            self._acked.move_to_end(path)
                            return
                        if not same and not replace_existing:
                            raise ParallelHostError(f"immutable channel record already differs: {path}")
                    if len(self._pending) >= _MAX_PENDING:
                        raise JournalBusy("channel write reservations are full; retry pending")
                    answer_key = (path, digest)
                    if (on_ack is not None and answer_key not in self._answer_acks
                            and len(self._answer_acks) >= _MAX_PENDING):
                        raise JournalBusy("answer ACK reservations are full; retry pending")
                    # Freeze the value: the helper must retry the accepted bytes, even if the
                    # caller later changes its status view or response payload.
                    payload = {"path": str(path), "value": json.loads(encoded),
                               "replace": replace_existing, "durable": True}
                    ticket = self._submit("write", payload)
                    if on_ack is not None:
                        self._answer_acks.setdefault(answer_key, on_ack)
                    self._pending[path] = _Write(payload, ticket, digest)
            if not wait:
                raise JournalBusy("channel write awaits its durable ACK; retry pending")
            ticket.done.wait()
