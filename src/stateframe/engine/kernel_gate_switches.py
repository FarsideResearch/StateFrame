"""Explicit on/off switches for collection gates and requested-byte delivery.

STATEFRAME_IDLE_GOVERNOR  : the read/grep budget (kernel_idle_governor). Default on.
STATEFRAME_REPEAT_GATE    : identical-query replay/escalation in kernel_bolt.observe. Default on.
STATEFRAME_DELIVER_REQUESTED_BYTES: explicit bounded reads return their requested bytes. Default on.

Both exist so an ablation can PROVABLY turn the gate off. Values are strictly
"on" or "off"; anything else raises, so a typo can never silently no-op.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

IDLE_GOVERNOR_ENV = "STATEFRAME_IDLE_GOVERNOR"
REPEAT_GATE_ENV = "STATEFRAME_REPEAT_GATE"
DELIVER_REQUESTED_BYTES_ENV = "STATEFRAME_DELIVER_REQUESTED_BYTES"


def _switch(name: str, environ: Mapping[str, str] | None) -> bool:
    source = os.environ if environ is None else environ
    value = str(source.get(name, "on")).strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{name} must be on or off, got {value!r}")
    from kernel_lean import enabled as lean_enabled
    if lean_enabled(source):
        return False
    return value == "on"


def idle_governor_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _switch(IDLE_GOVERNOR_ENV, environ)


def repeat_gate_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _switch(REPEAT_GATE_ENV, environ)


def deliver_requested_bytes_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return the strict requested-byte delivery mode, independent of LEAN."""
    source = os.environ if environ is None else environ
    value = str(source.get(DELIVER_REQUESTED_BYTES_ENV, "on")).strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{DELIVER_REQUESTED_BYTES_ENV} must be on or off, got {value!r}")
    return value == "on"
