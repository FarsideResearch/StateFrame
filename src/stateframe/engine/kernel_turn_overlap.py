"""Response-independent preparation during provider generation."""
from __future__ import annotations

import os
from collections.abc import Callable

from kernel_turn_contracts import (
    Artifact,
    Basis,
    Context,
    Facts,
    Key,
    Overlap,
    Publication,
    Unavailable,
)

MODE_ENV = "STATEFRAME_TURN_GENERATION_OVERLAP"


def enabled() -> bool:
    return os.environ.get(MODE_ENV, "off") == "on"


class _DisabledOverlap:
    def offer(self, at: Basis, key: Key, work: Callable[[], Artifact]) -> None:
        pass

    def take(self, at: Basis, key: Key) -> Artifact | None:
        return None

    def cancel(self) -> None:
        pass


class _GenerationOverlap:
    def __init__(self, facts: Facts, publication: Publication):
        from kernel_turn_workers import OverlapWorker

        self._facts = facts
        self._publication = publication
        self._worker = OverlapWorker(publication)

    def offer(self, at: Basis, key: Key, work: Callable[[], Artifact]) -> None:
        self._worker.offer(at, key, work)

    def take(self, at: Basis, key: Key) -> Artifact | None:
        claim = self._worker.claim(key)
        if claim is None:
            return None
        ticket, artifact = claim
        accepted = False
        try:
            if self._facts.current(at, artifact.deps) is True:
                self._publication.load(artifact.value)
                accepted = True
        except (Unavailable, OSError):
            pass
        finally:
            current = self._worker.release(ticket)
        return artifact if accepted and current else None

    def cancel(self) -> None:
        self._worker.cancel()


def open_port(ctx: Context, *, facts: Facts, publication: Publication) -> Overlap:
    """Open a transient port using services scoped to the supplied run and attempt.

    Offer only closures over retained immutable inputs, with bounded output and
    complete dependency lists. The owner calls take under the current live
    fence and in its publication transaction, which holds a reader pin through
    use. A miss leaves demand on the synchronous path. Cancel the run's tickets
    on retry, restore and teardown, and discard the port on teardown.
    """
    if not enabled():
        return _DisabledOverlap()
    return _GenerationOverlap(facts, publication)
