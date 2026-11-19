"""Passive resource counts for the active response-to-submission window.

Only registered engine work sites contribute. The context is empty when turn
timing is disabled, so ordinary requests allocate no counter object.
"""
from __future__ import annotations

import contextvars
import sys
from dataclasses import asdict, dataclass

# The engine is imported both as a package and through its child-process flat
# module path. Both spellings must share the same active context.
_OTHER_NAME = (
    "kernel_turn_work_counters" if __name__ != "kernel_turn_work_counters"
    else "stateframe.engine.kernel_turn_work_counters"
)
sys.modules.setdefault(_OTHER_NAME, sys.modules[__name__])


@dataclass(slots=True)
class WorkCounters:
    files_read: int = 0
    files_statted: int = 0
    files_hashed: int = 0
    parses: int = 0
    index_builds: int = 0
    subprocess_launches: int = 0
    bytes_hashed: int = 0

    def snapshot(self) -> dict[str, int]:
        return asdict(self)


_ACTIVE: contextvars.ContextVar[WorkCounters | None] = contextvars.ContextVar(
    "stateframe_turn_work_counters", default=None,
)


def activate_work_counters(counters: WorkCounters) -> None:
    _ACTIVE.set(counters)


def deactivate_work_counters(counters: WorkCounters | None) -> None:
    if counters is not None and _ACTIVE.get() is counters:
        _ACTIVE.set(None)


def count_work(name: str, amount: int = 1) -> None:
    counters = _ACTIVE.get()
    if counters is not None:
        setattr(counters, name, getattr(counters, name) + amount)
