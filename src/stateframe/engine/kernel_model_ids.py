"""Model id resolution for advisory request bodies (MODEL-ROUTING-UNIFY, MR-2).

decide2 and work-frame bodies historically send the tier aliases ``decision-tier`` and
``work-frame`` as their ``model``. Under ``STATEFRAME_ROUTING_SCHEMA=2`` those aliases resolve
to the configured default model id instead:

1. the role's explicit override env (``STATEFRAME_DECIDE2_MODEL`` / ``STATEFRAME_WORK_FRAME_MODEL``),
2. ``STATEFRAME_MODEL_NAME``,
3. ``DS_MODEL``.

If none is set the alias is returned unchanged (fail-open to the legacy body). Schema 1 returns
every input unchanged, so default-off runs stay byte-identical.
"""
from __future__ import annotations

import os
from typing import Mapping

from kernel_routing_schema import routing_v2

TIER_ALIASES = frozenset({"decision-tier", "work-frame"})

_ROLE_OVERRIDE_ENV: Mapping[str, str] = {
    "decision-tier": "STATEFRAME_DECIDE2_MODEL",
    "work-frame": "STATEFRAME_WORK_FRAME_MODEL",
}
_DEFAULT_ENVS = ("STATEFRAME_MODEL_NAME", "DS_MODEL")


def _value(environ: Mapping[str, str], name: str) -> str:
    return str(environ.get(name, "")).strip()


def resolve_model_id(requested: str | None, *, environ: Mapping[str, str] | None = None) -> str | None:
    """Return the model id a request body should carry.

    Schema 1 returns ``requested`` unchanged. Schema 2 replaces ``None``, ``""`` and the tier
    aliases with the role override or configured default; any other explicit id passes through.
    """
    source: Mapping[str, str] = os.environ if environ is None else environ
    if not routing_v2(source):
        return requested
    if requested not in (None, "") and requested not in TIER_ALIASES:
        return requested
    alias = requested if requested else ""
    override = _ROLE_OVERRIDE_ENV.get(alias, "")
    for name in (override, *_DEFAULT_ENVS):
        if name and (resolved := _value(source, name)):
            return resolved
    return requested
