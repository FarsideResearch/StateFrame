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
from kernel_routing_schema import routing_v2

SOURCE_STATIC_ROUTE = "static_route"
SOURCE_MODEL_ROUTER = "model_router"
SOURCE_DEFAULT = "default"
# Replay of a prepared body whose decision was made in another process: the destination is
# taken as already routed and no router or static route is consulted again.
SOURCE_RECOVERED = "recovered"
# Cross-process replay that read back the journaled decision for this request_id (FRP).
SOURCE_REPLAYED = "replayed"
# Journal-side index of ``route_decision`` events by request_id, built once per journal.
_ROUTE_INDEX_ATTR = "_route_decision_index"


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


def _runtime_of(meter: object) -> object | None:
    """The runtime that owns this request, found the way kernel_measurement finds it."""
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
    except (ImportError, LookupError):
        runtime = None
    return runtime if runtime is not None else getattr(meter, "completion_owner_runtime", None)


def _journal_endpoint(endpoint: str) -> str | None:
    """The endpoint as journaled for replay, or None when it cannot be stored credential-free.

    Userinfo, query and fragment are never journaled; a replay of such a request falls back to
    the ``recovered`` behaviour instead of sending to a rewritten URL.
    """
    parts = urllib.parse.urlsplit(endpoint)
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        return None
    return endpoint


def _remember(meter: object, decision: RouteDecision) -> None:
    """Cache the decision on the request capture and journal it once per request_id."""
    setattr(meter, "route_decision", decision)
    if routing_v2():
        # MR-3: the model policy follows the decided deployment (policy_for_runtime).
        runtime = _runtime_of(meter)
        if runtime is not None:
            setattr(runtime, "selected_model", decision.model)
    if getattr(meter, "_route_decision_journaled", None) == decision.request_id:
        return
    setattr(meter, "_route_decision_journaled", decision.request_id)
    fields: dict[str, object] = {
        "request_id": decision.request_id, "model": decision.model,
        "endpoint_host": decision.endpoint_host, "source": decision.source, "role": decision.role,
        "key_env": decision.key_env, "dispatch": False,
    }
    # Replay needs the full destination (path included); only the credential-free form is stored.
    journaled_endpoint = _journal_endpoint(decision.endpoint)
    if journaled_endpoint is not None:
        fields["endpoint"] = journaled_endpoint
    journal = meter.journal  # type: ignore[attr-defined]
    journal.emit("route_decision", **fields)
    index = getattr(journal, _ROUTE_INDEX_ATTR, None)
    if index is not None:
        index.setdefault(decision.request_id, []).append({"kind": "route_decision", **fields})


def _journaled_routes(journal: object) -> dict[str, list[dict[str, object]]] | None:
    """``route_decision`` events grouped by request_id, read once per journal object.

    Uses the journal's own validated ``records(rebuild=False)`` pass. The pass is consumed in full
    before any caller emits, because emitting while a replay generator is open is refused.
    """
    index = getattr(journal, _ROUTE_INDEX_ATTR, None)
    if index is not None:
        return index
    records = getattr(journal, "records", None)
    if not callable(records):
        return None
    index = {}
    for event in records(rebuild=False):
        if event.get("kind") != "route_decision":
            continue
        request_id = event.get("request_id")
        if isinstance(request_id, str):
            index.setdefault(request_id, []).append(dict(event))
    setattr(journal, _ROUTE_INDEX_ATTR, index)
    return index


def _replayed_decision(meter: object, request_id: str) -> RouteDecision | None:
    """Rebuild the decision journaled for this request_id, or None when it cannot be trusted.

    Only the destination fields are read back. The credential is resolved from the journaled
    key_env name exactly as ``_static_route`` does; the secret itself is never journaled. Two
    journaled decisions that disagree for one request_id are treated as no decision.
    """
    journal = getattr(meter, "journal", None)
    if journal is None:
        return None
    index = _journaled_routes(journal)
    events = index.get(request_id) if index is not None else None
    if not events:
        return None
    identities = {(event.get("model"), event.get("endpoint"), event.get("key_env"),
                   event.get("role")) for event in events}
    if len(identities) != 1:
        return None
    model, endpoint, key_env, role = identities.pop()
    if not isinstance(model, str) or not model or not isinstance(endpoint, str) or not endpoint:
        return None
    if key_env is not None and not isinstance(key_env, str):
        return None
    key = os.environ.get(key_env, "") if key_env is not None else ""
    return RouteDecision(
        model=model, endpoint=endpoint, key_env=key_env, source=SOURCE_REPLAYED,
        role=role if isinstance(role, str) else None, request_id=request_id, key=key)


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


def _reapply_router_adaptation(meter: object, cached: RouteDecision, body: dict[str, object],
                               endpoint: str, key: str) -> dict[str, object]:
    """Re-run the router's body adaptation on a cached second build, pinned to the decision.

    ``prepare_call`` also proposes a destination; that proposal is discarded and the cached
    model, endpoint and key stay in force. A proposal that differs is journaled once per request.
    """
    router = _model_router(meter)
    if router is None:
        return body
    prepared = router.prepare_call(meter, body, endpoint, key)  # type: ignore[attr-defined]
    adapted = dict(prepared.body)
    wanted_model = str(adapted.get("model", ""))
    drifted = wanted_model != cached.model or str(prepared.endpoint) != cached.endpoint
    if drifted and getattr(meter, "_route_drift_journaled", None) != cached.request_id:
        setattr(meter, "_route_drift_journaled", cached.request_id)
        meter.journal.emit(  # type: ignore[attr-defined]
            "route_drift_pinned", request_id=cached.request_id, wanted_model=wanted_model,
            kept_model=cached.model, dispatch=False)
    # prepare_call stamped its own proposal on the capture; restore the decided values.
    setattr(meter, "model_router_effective_endpoint", cached.endpoint)
    setattr(meter, "model_router_effective_model_id", cached.model)
    return adapted


def _diagnostic_endpoint(endpoint: str) -> str:
    """Scheme, host, port and path only: the drift-diagnostic form (never replay authority)."""
    parts = urllib.parse.urlsplit(endpoint)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))


def _adopt_replayed_decision(meter: object, replayed: RouteDecision, body: dict[str, object],
                             wanted_endpoint: str) -> tuple[RouteDecision, dict[str, object], str, str]:
    """The journaled decision wins over a drifting model or endpoint; drift journals once per request.

    Only the pairs that actually drifted are recorded: the model pair, the endpoint pair, or both.
    """
    wanted_model = str(body.get("model", ""))
    model_drift = wanted_model != replayed.model
    endpoint_drift = wanted_endpoint != replayed.endpoint
    already_journaled = getattr(meter, "_route_drift_journaled", None) == replayed.request_id
    if (model_drift or endpoint_drift) and not already_journaled:
        setattr(meter, "_route_drift_journaled", replayed.request_id)
        fields: dict[str, object] = {"request_id": replayed.request_id, "dispatch": False}
        if model_drift:
            fields.update(wanted_model=wanted_model, kept_model=replayed.model)
        if endpoint_drift:
            fields.update(wanted_endpoint=_diagnostic_endpoint(wanted_endpoint),
                          kept_endpoint=_diagnostic_endpoint(replayed.endpoint))
        meter.journal.emit("route_drift_pinned", **fields)  # type: ignore[attr-defined]
    body["model"] = replayed.model
    _remember(meter, replayed)
    return replayed, body, replayed.endpoint, replayed.key


def resolve_route_decision(meter: object, body: dict[str, object], endpoint: str, key: str, *,
                           recovery_prepared: bool) -> tuple[RouteDecision, dict[str, object], str, str]:
    """Per-request entry point: reuse the cached decision or decide once.

    A second body build for the same request_id reuses the cached decision and does not
    re-decide. A cached model-router decision re-applies the router's body adaptation (its
    proposed model, endpoint and key are discarded). A recovery replay reuses the cached decision too.
    Without one (replay in a new process) schema 2 rebuilds the decision journaled for this request_id
    as ``SOURCE_REPLAYED``. With no usable journaled decision the already-routed body is taken as-is
    and marked ``SOURCE_RECOVERED``.
    """
    request_id = str(getattr(meter, "request_id", ""))
    cached = getattr(meter, "route_decision", None)
    if isinstance(cached, RouteDecision) and cached.request_id == request_id:
        if routing_v2() and not recovery_prepared and cached.source == SOURCE_MODEL_ROUTER:
            body = _reapply_router_adaptation(meter, cached, body, endpoint, key)
        body["model"] = cached.model
        if not recovery_prepared:
            router = _model_router(meter)
            if router is not None:
                router.validate_final_request(meter, body)  # type: ignore[attr-defined]
        return cached, body, cached.endpoint, cached.key
    if recovery_prepared:
        if routing_v2():
            replayed = _replayed_decision(meter, request_id)
            if replayed is not None:
                return _adopt_replayed_decision(meter, replayed, body, endpoint)
        decision = RouteDecision(
            model=str(body.get("model", "")), endpoint=endpoint, key_env=None,
            source=SOURCE_RECOVERED, role=None, request_id=request_id, key=key)
        _remember(meter, decision)
        return decision, body, endpoint, key
    return decide_route(meter, body, endpoint, key)
