"""Lightweight coalescing switch and host-only pytest evidence return channel.

Importing this module never imports the coalescer or changes an OFF request. Raw
stdout travels with the string observation, not in its model-facing text or in
mutable last-run state. Selector aggregation must explicitly retain every run.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Self

MODE_ENV: Final = "STATEFRAME_UNIT_COALESCE"


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


@dataclass(frozen=True)
class GateRun:
    stdout: str
    exit_code: int | None
    complete: bool
    collected: tuple[str, ...] | None = None
    source_digest: str = ""


class GateObservation(str):
    """A normal wire string with immutable, attempt-local host evidence attached."""

    unit_gate_runs: tuple[GateRun, ...]

    def __new__(cls, text: str, runs: tuple[GateRun, ...] = ()) -> Self:
        observation = super().__new__(cls, text)
        observation.unit_gate_runs = runs
        return observation
