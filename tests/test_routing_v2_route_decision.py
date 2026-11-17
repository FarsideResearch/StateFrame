"""MR-1: one RouteDecision per request under STATEFRAME_ROUTING_SCHEMA=2.

Stub meter/router objects stand in for a live RequestCapture and ModelRouter. Each test pins
one clause of the design: precedence, single prepare, replay reuse, seal, provenance, journal.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_frame_routes as frame_routes
import kernel_measurement as measurement
import kernel_route_decision as route_decision
from kernel_route_decision import RouteDecision, decide_route, resolve_route_decision
from kernel_routing_schema import ROUTING_SCHEMA_ENV
import kernel_release_switch_registry as switch_registry

STATIC_BASE = "http://127.0.0.1:9000/v1"
STATIC_URL = STATIC_BASE + "/chat/completions"  # Route.endpoint appends the path
ROUTER_URL = "http://127.0.0.1:9100/v1"
CALLER_URL = "http://127.0.0.1:8000/v1"


class FakeJournal:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict]] = []

    def emit(self, name: str, **fields: object) -> None:
        self.records.append((name, fields))

    def named(self, name: str) -> list[dict]:
        return [fields for event, fields in self.records if event == name]


class FakeRouter:
    """Mimics ModelRouter.prepare_call / validate_final_request without a live lease."""

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.validated_bodies: list[dict] = []

    def prepare_call(self, capture: object, body: dict, endpoint: str, key: str) -> SimpleNamespace:
        self.prepare_calls += 1
        adapted = copy.deepcopy(body)
        adapted["model"] = "router-model"
        return SimpleNamespace(
            body=adapted, endpoint=ROUTER_URL, key="router-secret", applied=True,
            decision=SimpleNamespace(endpoint=SimpleNamespace(api_key_env="ROUTER_KEY")),
        )

    def validate_final_request(self, capture: object, body: dict) -> None:
        self.validated_bodies.append(copy.deepcopy(body))


class FakeMeter:
    def __init__(self, *, request_id: str = "req-1", router: FakeRouter | None = None,
                 frame_route: object | None = None) -> None:
        self.request_id = request_id
        self.journal = FakeJournal()
        self.model_router = router
        self.frame_route = frame_route


def _static_route() -> frame_routes.Route:
    return frame_routes.Route(
        role="draft", scope="main", unit_class="draft",
        settings={"endpoint": STATIC_BASE, "model": "static-model", "api_key_env": "STATIC_KEY",
                  "allow_remote": False},
    )


@pytest.fixture
def v2_routes(monkeypatch):
    monkeypatch.setenv(ROUTING_SCHEMA_ENV, "2")
    monkeypatch.setenv("STATIC_KEY", "static-secret")
    monkeypatch.setattr(frame_routes, "ROUTES_ACTIVE", True)
    yield
    monkeypatch.setattr(frame_routes, "ROUTES_ACTIVE", False)


def _body() -> dict:
    return {"model": "caller-model", "messages": [{"role": "user", "content": "hi"}]}


def test_static_route_and_router_present_sends_exactly_one_destination(v2_routes):
    router = FakeRouter()
    meter = FakeMeter(router=router, frame_route=_static_route())

    decision, body, endpoint, key = decide_route(meter, _body(), CALLER_URL, "caller-key")

    assert decision.source == "static_route"
    assert decision.role == "draft"
    assert endpoint == STATIC_URL
    assert body["model"] == "static-model"
    assert key == "static-secret"
    assert decision.key_env == "STATIC_KEY"
    # The adaptive router never chooses the destination when the static route pins it...
    assert router.prepare_calls == 0
    # ...but its budget/fit checks still run on the final destination body.
    assert router.validated_bodies[-1]["model"] == "static-model"


def test_router_only_source_is_model_router(v2_routes):
    router = FakeRouter()
    meter = FakeMeter(router=router)

    decision, body, endpoint, key = decide_route(meter, _body(), CALLER_URL, "caller-key")

    assert decision.source == "model_router"
    assert endpoint == ROUTER_URL
    assert body["model"] == "router-model"
    assert key == "router-secret"
    assert decision.key_env == "ROUTER_KEY"
    assert router.prepare_calls == 1


def test_no_router_no_route_is_default(v2_routes):
    meter = FakeMeter()

    decision, body, endpoint, key = decide_route(meter, _body(), CALLER_URL, "caller-key")

    assert decision.source == "default"
    assert (endpoint, key, body["model"]) == (CALLER_URL, "caller-key", "caller-model")
    assert decision.key_env is None


def test_repeat_build_for_same_request_reuses_decision_without_rerunning_router(v2_routes):
    router = FakeRouter()
    meter = FakeMeter(router=router)

    first, _, _, _ = resolve_route_decision(meter, _body(), CALLER_URL, "caller-key",
                                            recovery_prepared=False)
    second, body, endpoint, key = resolve_route_decision(meter, _body(), CALLER_URL, "caller-key",
                                                         recovery_prepared=False)

    assert second is first
    assert router.prepare_calls == 1
    assert endpoint == ROUTER_URL and key == "router-secret"
    assert body["model"] == "router-model"


def test_route_decision_is_journaled_once_per_request_without_dispatch(v2_routes):
    meter = FakeMeter(router=FakeRouter())

    resolve_route_decision(meter, _body(), CALLER_URL, "caller-key", recovery_prepared=False)
    resolve_route_decision(meter, _body(), CALLER_URL, "caller-key", recovery_prepared=False)

    records = meter.journal.named("route_decision")
    assert len(records) == 1
    record = records[0]
    assert record["request_id"] == "req-1"
    assert record["model"] == "router-model"
    assert record["endpoint_host"] == "127.0.0.1"
    assert record["source"] == "model_router"
    assert record["dispatch"] is False
    assert "key" not in record and "router-secret" not in repr(record)


def test_static_route_keeps_frame_route_record_once(v2_routes):
    meter = FakeMeter(router=FakeRouter(), frame_route=_static_route())

    resolve_route_decision(meter, _body(), CALLER_URL, "caller-key", recovery_prepared=False)
    resolve_route_decision(meter, _body(), CALLER_URL, "caller-key", recovery_prepared=False)

    frame_records = meter.journal.named("frame_route")
    assert len(frame_records) == 1
    assert frame_records[0]["model"] == "static-model"
    assert frame_records[0]["dispatch"] is False


def test_seal_raises_when_model_changes_after_decision(v2_routes):
    decision, body, endpoint, _ = decide_route(FakeMeter(router=FakeRouter()), _body(),
                                               CALLER_URL, "caller-key")
    decision.assert_sealed(body, endpoint)  # the decided body passes

    body["model"] = "swapped-after-qualification"
    with pytest.raises(RuntimeError, match="route changed after decision"):
        decision.assert_sealed(body, endpoint)


def test_seal_raises_when_endpoint_changes_after_decision(v2_routes):
    decision, body, _, _ = decide_route(FakeMeter(router=FakeRouter()), _body(),
                                        CALLER_URL, "caller-key")

    with pytest.raises(RuntimeError, match="route changed after decision"):
        decision.assert_sealed(body, CALLER_URL)


def test_route_decision_repr_never_contains_the_key():
    decision = RouteDecision(model="m", endpoint=ROUTER_URL, key_env="ROUTER_KEY",
                             source="model_router", role=None, request_id="r", key="sekrit")
    assert "sekrit" not in repr(decision)
    assert decision == RouteDecision(model="m", endpoint=ROUTER_URL, key_env="ROUTER_KEY",
                                     source="model_router", role=None, request_id="r", key="other")


def test_replay_with_cached_decision_reuses_it_under_recovery(v2_routes):
    router = FakeRouter()
    meter = FakeMeter(router=router)
    first, _, _, _ = resolve_route_decision(meter, _body(), CALLER_URL, "caller-key",
                                            recovery_prepared=False)

    replayed, _, endpoint, _ = resolve_route_decision(meter, _body(), CALLER_URL, "caller-key",
                                                      recovery_prepared=True)

    assert replayed is first
    assert endpoint == ROUTER_URL
    assert router.prepare_calls == 1


def test_recovery_without_cached_decision_is_marked_recovered(v2_routes):
    router = FakeRouter()
    meter = FakeMeter(router=router)
    prepared = {"model": "already-routed", "messages": []}

    decision, body, endpoint, key = resolve_route_decision(meter, prepared, ROUTER_URL, "k",
                                                           recovery_prepared=True)

    assert decision.source == route_decision.SOURCE_RECOVERED
    assert (decision.model, endpoint, key) == ("already-routed", ROUTER_URL, "k")
    assert router.prepare_calls == 0


def test_provenance_source_follows_route_decision_under_v2(v2_routes, monkeypatch):
    meter = FakeMeter(router=FakeRouter(), frame_route=_static_route())
    decision, _, endpoint, _ = decide_route(meter, _body(), CALLER_URL, "caller-key")
    meter.route_decision = decision

    measurement._record_probe_destination_provenance(meter, endpoint)

    provenance = meter._provider_probe_provenance
    assert provenance["source"] == "static_route"
    assert provenance["endpoint"] == STATIC_URL
    assert provenance["credential_env"] == "STATIC_KEY"


def test_provenance_keeps_legacy_source_names_when_schema_is_1(monkeypatch):
    monkeypatch.delenv(ROUTING_SCHEMA_ENV, raising=False)
    monkeypatch.setattr(frame_routes, "ROUTES_ACTIVE", True)
    meter = FakeMeter(frame_route=None)
    meter.route_decision = RouteDecision(model="m", endpoint=STATIC_URL, key_env=None,
                                         source="static_route", role=None, request_id="req-1")

    measurement._record_probe_destination_provenance(meter, STATIC_URL)

    assert meter._provider_probe_provenance["source"] == "primary"


def test_seal_hook_is_gated_on_routing_v2_in_measurement():
    # Schema 1 must keep today's sequential route blocks: the v2 hook is the only new path.
    text = Path(measurement.__file__).read_text(encoding="utf-8")
    assert "resolve_route_decision(" in text
    assert "if not unified_routing and not _recovery_prepared and model_router is not None:" in text
    assert "if (not unified_routing and _frame_routes.ROUTES_ACTIVE and not _recovery_prepared" in text


def test_routing_schema_switch_is_registered_in_release_registry():
    names, _digest = switch_registry.release_stateframe_switches(Path(measurement.__file__).parent)
    assert ROUTING_SCHEMA_ENV in names
