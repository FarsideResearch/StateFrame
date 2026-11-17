"""Routing schema selector (MODEL-ROUTING-UNIFY).

``STATEFRAME_ROUTING_SCHEMA`` picks how a request's model, endpoint and credential are decided:

* ``1`` (default, also unset/empty): the legacy behaviour, byte-identical to earlier releases.
* ``2``: one ``RouteDecision`` per request governs every consumer; advisory paths resolve the
  configured default model id instead of tier aliases; model policy follows the selected deployment.

Default OFF (schema 1) until proven. Anything else fails loud.
"""
from __future__ import annotations

import os
from typing import Mapping

ROUTING_SCHEMA_ENV = "STATEFRAME_ROUTING_SCHEMA"


def routing_schema(environ: Mapping[str, str] | None = None) -> int:
    raw = (os.environ if environ is None else environ).get(ROUTING_SCHEMA_ENV, "")
    value = str(raw).strip()
    if value in {"", "1"}:
        return 1
    if value == "2":
        return 2
    raise ValueError(f"{ROUTING_SCHEMA_ENV} must be 1 or 2, got {raw!r}")


def routing_v2(environ: Mapping[str, str] | None = None) -> bool:
    return routing_schema(environ) == 2
