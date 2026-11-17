"""One RouteDecision per request (MODEL-ROUTING-UNIFY, STATEFRAME_ROUTING_SCHEMA=2).

Precedence is decided here once: an active static frame route pins model, endpoint and
credential; otherwise the adaptive model router chooses; otherwise the request keeps its
caller-supplied destination. Nothing after the decision may swap the destination: the sealed
decision is asserted again before the physical send (see ``RouteDecision.assert_sealed``).

Callers only reach this module when ``routing_v2()`` is true. Schema 1 keeps the legacy
sequential blocks in kernel_measurement untouched.
"""
from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass, field
from typing import Mapping

import kernel_frame_routes as _frame_routes
from kernel_frame_routes import require_local

SOURCE_STATIC_ROUTE = "static_route"
SOURCE_MODEL_ROUTER = "model_router"
SOURCE_DEFAULT = "default"
# Replay of a prepared body whose decision was made in another process: the destination is
# taken as already routed and no router or static route is consulted again.
SOURCE_RECOVERED = "recovered"


@dataclass(frozen=True)
class RouteDecision:
    """The destination a request was decided to use. ``key`` is memory-only and never logged."""

    model: str
    endpoint: str
    key_env: str | None
    source: str
    role: str | None
    request_id: str
    key: str = field(default="", repr=False, compare=False)

    @property
    def endpoint_host(self) -> str | None:
        return urllib.parse.urlsplit(self.endpoint).hostname

    def assert_sealed(self, body: Mapping[str, object], endpoint: str) -> None:
        """Fail closed when the body or endpoint about to be sent differs from the decision."""
        if body.get("model") != self.model or endpoint != self.endpoint:
            raise RuntimeError("route changed after decision")


def _model_router(meter: object) -> object | None:
    router = getattr(meter, "model_router", None)
    if router is None:
        router = getattr(getattr(meter, "journal", None), "model_router", None)
    return router


def _remember(meter: object, decision: RouteDecision) -> None:
    """Cache the decision on the request capture and journal it once per request_id."""
    setattr(meter, "route_decision", decision)
    if getattr(meter, "_route_decision_journaled", None) == decision.request_id:
        return
    setattr(meter, "_route_decision_journaled", decision.request_id)
    meter.journal.emit(  # type: ignore[attr-defined]
        "route_decision", request_id=decision.request_id, model=decision.model,
        endpoint_host=decision.endpoint_host, source=decision.source, role=decision.role,
        dispatch=False)


def _static_route(meter: object, route: object, body: dict[str, object],
                  router: object | None) -> tuple[RouteDecision, dict[str, object], str, str]:
    settings = route.settings  # type: ignore[attr-defined]
    require_local(route.endpoint, allow_remote=bool(settings.get("allow_remote")))  # type: ignore[attr-defined]
    key_name = settings.get("api_key_env")
    key_env = key_name if isinstance(key_name, str) else None
    key = os.environ.get(key_env, "") if key_env is not None else ""
    body["model"] = route.model  # type: ignore[attr-defined]
    # The router is not consulted for the destination, but its fit/budget checks run on the
    # final body so they cover the destination that will actually be sent.
    if router is not None:
        router.validate_final_request(meter, body)  # type: ignore[attr-defined]
    request_id = str(getattr(meter, "request_id", ""))
    if getattr(meter, "_frame_route_journaled", None) != request_id:
        setattr(meter, "_frame_route_journaled", request_id)
        meter.journal.emit(  # type: ignore[attr-defined]
            "frame_route", request_id=request_id, role=route.role,  # type: ignore[attr-defined]
            scope=route.scope, unit_class=route.unit_class,  # type: ignore[attr-defined]
            endpoint_host=urllib.parse.urlsplit(route.endpoint).hostname,  # type: ignore[attr-defined]
            model=route.model, overridden_fields=route.overridden_fields,  # type: ignore[attr-defined]
            escalated_from=route.escalated_from, dispatch=False)  # type: ignore[attr-defined]
    decision = RouteDecision(
        model=str(route.model), endpoint=str(route.endpoint),  # type: ignore[attr-defined]
        key_env=key_env, source=SOURCE_STATIC_ROUTE, role=route.role,  # type: ignore[attr-defined]
        request_id=request_id, key=key)
    return decision, body, decision.endpoint, key


def _router_route(meter: object, router: object, body: dict[str, object], endpoint: str,
                  key: str) -> tuple[RouteDecision, dict[str, object], str, str]:
    prepared = router.prepare_call(meter, body, endpoint, key)  # type: ignore[attr-defined]
    final_body = prepared.body
    router.validate_final_request(meter, final_body)  # type: ignore[attr-defined]
    selected = getattr(getattr(prepared, "decision", None), "endpoint", None)
    key_name = getattr(selected, "api_key_env", None)
    key_env = key_name if isinstance(key_name, str) and key_name else None
    decision = RouteDecision(
        model=str(final_body.get("model", "")), endpoint=str(prepared.endpoint),
        key_env=key_env,
        source=SOURCE_MODEL_ROUTER if getattr(prepared, "applied", False) is True else SOURCE_DEFAULT,
        role=getattr(meter, "model_router_call_purpose", None),
        request_id=str(getattr(meter, "request_id", "")), key=str(prepared.key))
    return decision, final_body, decision.endpoint, decision.key


def decide_route(meter: object, body: dict[str, object], endpoint: str,
                 key: str) -> tuple[RouteDecision, dict[str, object], str, str]:
    """Apply the precedence once: static frame route, then model router, then the default.

    Returns the decision with the body, endpoint and key that must be sent. It does not read
    or write the per-request cache; use ``resolve_route_decision`` for that.
    """
    router = _model_router(meter)
    route = getattr(meter, "frame_route", None)
    if _frame_routes.ROUTES_ACTIVE and route is not None:
        decision, final_body, final_endpoint, final_key = _static_route(
            meter, route, body, router)
    elif router is not None:
        decision, final_body, final_endpoint, final_key = _router_route(
            meter, router, body, endpoint, key)
    else:
        decision = RouteDecision(
            model=str(body.get("model", "")), endpoint=endpoint, key_env=None,
            source=SOURCE_DEFAULT, role=None,
            request_id=str(getattr(meter, "request_id", "")), key=key)
        final_body, final_endpoint, final_key = body, endpoint, key
    _remember(meter, decision)
    return decision, final_body, final_endpoint, final_key


def resolve_route_decision(meter: object, body: dict[str, object], endpoint: str, key: str, *,
                           recovery_prepared: bool) -> tuple[RouteDecision, dict[str, object], str, str]:
    """Per-request entry point: reuse the cached decision or decide once.

    A second body build for the same request_id reuses the cached decision and does not run
    the router again. A recovery replay reuses the cached decision too; without one (replay in a
    new process) the already-routed body is taken as-is and marked ``SOURCE_RECOVERED``.
    """
    request_id = str(getattr(meter, "request_id", ""))
    cached = getattr(meter, "route_decision", None)
    if isinstance(cached, RouteDecision) and cached.request_id == request_id:
        body["model"] = cached.model
        if not recovery_prepared:
            router = _model_router(meter)
            if router is not None:
                router.validate_final_request(meter, body)  # type: ignore[attr-defined]
        return cached, body, cached.endpoint, cached.key
    if recovery_prepared:
        decision = RouteDecision(
            model=str(body.get("model", "")), endpoint=endpoint, key_env=None,
            source=SOURCE_RECOVERED, role=None, request_id=request_id, key=key)
        _remember(meter, decision)
        return decision, body, endpoint, key
    return decide_route(meter, body, endpoint, key)
