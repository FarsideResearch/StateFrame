"""Append-only DecisionEvent stream (JSONL).

There is NO clear(), reset(), truncate(), or delete() on this class, and there
never will be. This is not an oversight and it is not defensive programming
taste — a causal engine whose ledger can be truncated cannot answer a single
counterfactual afterwards, and we have already destroyed one that way. The file
is opened "a" only; the only supported way to lose history is to delete the file
outside this module, deliberately, as a human.

Atomicity: one event is one `write()` of one complete line, flushed, and fsynced
by default. A partially written line therefore cannot be produced by this class;
`read_all` still tolerates a truncated tail from an outside crash by raising
loudly on the offending line rather than silently skipping it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

from .decision_event import DecisionEvent, DecisionEventError


class EventStreamError(RuntimeError):
    pass


class EventStream:
    """Append-only JSONL event log at `path`."""

    #: Names this class refuses to grow. Asserted by the test suite.
    FORBIDDEN_OPERATIONS = ("clear", "reset", "truncate", "delete",
                            "drop", "purge", "rewrite")

    def __init__(self, path: str | os.PathLike, *, fsync: bool = True):
        self.path = Path(path)
        self.fsync = fsync
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ write
    def append(self, event: DecisionEvent) -> DecisionEvent:
        """Validate, then append exactly one line. Returns the event appended."""
        if not isinstance(event, DecisionEvent):
            raise EventStreamError(
                f"only DecisionEvent may be appended, got {type(event)}")
        # Round-trip validation: what we write MUST be readable back as the
        # same event. A record that cannot be re-read is not a record.
        payload = event.to_dict()
        DecisionEvent.from_dict(payload)
        line = json.dumps(payload, sort_keys=True, default=str) + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            if self.fsync:
                os.fsync(fh.fileno())
        return event

    def extend(self, events: Iterable[DecisionEvent]) -> list[DecisionEvent]:
        return [self.append(e) for e in events]

    # ------------------------------------------------------------------- read
    def read_all(self) -> list[DecisionEvent]:
        return list(self.iter_events())

    def iter_events(self) -> Iterator[DecisionEvent]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise EventStreamError(
                        f"{self.path}:{lineno} is not valid JSON ({exc}); "
                        "a corrupt ledger line is reported, never skipped"
                    ) from exc
                try:
                    yield DecisionEvent.from_dict(obj)
                except DecisionEventError as exc:
                    raise EventStreamError(
                        f"{self.path}:{lineno} is not a valid DecisionEvent: {exc}"
                    ) from exc

    def by_correlation(self, correlation_id: str) -> list[DecisionEvent]:
        return [e for e in self.iter_events()
                if e.correlation_id == correlation_id]

    def by_type(self, decision_type: str) -> list[DecisionEvent]:
        return [e for e in self.iter_events()
                if e.decision_type == decision_type]

    def by_causation(self, event_id: str) -> list[DecisionEvent]:
        """Direct effects of one event."""
        return [e for e in self.iter_events() if e.causation_id == event_id]

    def correlation_ids(self) -> list[str]:
        """Distinct runs, in first-seen order. File order IS chronology."""
        seen: list[str] = []
        known = set()
        for e in self.iter_events():
            if e.correlation_id not in known:
                known.add(e.correlation_id)
                seen.append(e.correlation_id)
        return seen

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_events())

    def __iter__(self) -> Iterator[DecisionEvent]:
        return self.iter_events()
