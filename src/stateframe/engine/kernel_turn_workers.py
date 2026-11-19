"""One optional producer and a bounded pending queue for a single run."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from kernel_turn_contracts import Artifact, Basis, Key, Publication

_LOGGER = logging.getLogger(__name__)
_MAX_QUEUED = 2
_MAX_OUTPUT_BYTES = 8 * 1024 * 1024


@dataclass(slots=True)
class _Ticket:
    at: Basis
    key: Key
    work: Callable[[], Artifact]
    result: Artifact | None = None
    claimed: bool = False
    cancelled: bool = False


class OverlapWorker:
    """Keep one staged result pinned until the foreground adopts or discards it."""

    def __init__(self, publication: Publication):
        self._publication = publication
        self._changed = threading.Condition()
        self._pending: list[_Ticket] = []
        self._active: _Ticket | None = None
        self._running = False

    def offer(self, at: Basis, key: Key, work: Callable[[], Artifact]) -> None:
        with self._changed:
            for index, pending in enumerate(self._pending):
                if pending.key == key:
                    if pending.at != at:
                        self._pending[index] = _Ticket(at, key, work)
                    return

            active = self._active
            if active is not None and active.key == key and not active.cancelled:
                if active.at == at:
                    return
                active.cancelled = True
                self._changed.notify_all()

            if len(self._pending) >= _MAX_QUEUED:
                return
            self._pending.append(_Ticket(at, key, work))
            if self._running:
                return

            self._running = True
            try:
                threading.Thread(
                    target=self._run, name="stateframe-turn-overlap", daemon=True,
                ).start()
            except RuntimeError:
                # Optional work is unavailable when the process cannot start a worker.
                self._running = False
                self._pending.clear()

    def claim(self, key: Key) -> tuple[_Ticket, Artifact] | None:
        with self._changed:
            ticket = self._active
            if (ticket is None or ticket.key != key or ticket.result is None
                    or ticket.claimed or ticket.cancelled):
                return None
            ticket.claimed = True
            return ticket, ticket.result

    def release(self, ticket: _Ticket) -> bool:
        """Release the producer pin after adoption and test for concurrent cancellation."""
        with self._changed:
            current = not ticket.cancelled
            ticket.claimed = False
            ticket.cancelled = True
            self._changed.notify_all()
            return current

    def cancel(self) -> None:
        with self._changed:
            self._pending.clear()
            if self._active is not None:
                self._active.cancelled = True
            self._changed.notify_all()

    def _run(self) -> None:
        try:
            self._drain()
        except BaseException:
            with self._changed:
                self._running = False
                self._pending.clear()
                self._active = None
                self._changed.notify_all()
            raise

    def _drain(self) -> None:
        while True:
            with self._changed:
                if not self._pending:
                    self._active = None
                    self._running = False
                    self._changed.notify_all()
                    return
                ticket = self._pending.pop(0)
                self._active = ticket
            try:
                self._produce(ticket)
            except Exception:  # noqa: BLE001 - any optional producer failure is a miss.
                _LOGGER.debug("Optional generation-overlap work was discarded")

    def _produce(self, ticket: _Ticket) -> None:
        with self._publication.transaction():
            with self._changed:
                if ticket.cancelled:
                    return

            result = ticket.work()
            size = result.value.size
            if type(size) is not int or not 0 <= size <= _MAX_OUTPUT_BYTES:
                return

            with self._changed:
                if ticket.cancelled:
                    return
                ticket.result = result
                self._changed.notify_all()
                # The claimant acquires its own reader pin before releasing ours.
                self._changed.wait_for(lambda: ticket.cancelled and not ticket.claimed)
