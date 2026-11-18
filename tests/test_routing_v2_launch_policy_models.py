"""Routing v2: route-table models belong to the launch policy set (found by the MR-5 end-to-end test).

Under schema 2 a withheld edit is carried to its target route, and that route's model is selected on
the runtime. The router's launch policy set used to hold only its own endpoint ids plus the base model,
so a route-table model was refused with "router selected a model outside the launch policy set".
Schema 1 keeps the old set byte for byte.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import kernel_frame_routes as frame_routes
from kernel_stateframe_model_router import ModelRouter
from stateframe import routes

DEFAULT = {"endpoint": "http://127.0.0.1:8080/v1", "model": "base", "api_key_env": None,
           "temperature": 1.0, "max_tokens": 24000, "timeout_s": 1800.0,
           "slot": None, "context_tokens": 65536, "allow_remote": False}


@pytest.fixture(autouse=True)
def clear_routes(monkeypatch):
    monkeypatch.delenv("STATEFRAME_ROUTING_SCHEMA", raising=False)
    frame_routes.install(None)
    yield
    frame_routes.install(None)


def _install_edit_route() -> None:
    frame_routes.install(routes.resolve(DEFAULT, cli=["read.model=reader", "edit.model=editor"],
                                        environ={}).public())


def _endpoint(model_id: str) -> SimpleNamespace:
    return SimpleNamespace(name=model_id, model_id=model_id, url=DEFAULT["endpoint"])


def test_route_model_ids_lists_every_installed_route_model(monkeypatch):
    assert frame_routes.route_model_ids() == frozenset()
    _install_edit_route()
    ids = frame_routes.route_model_ids()
    assert {"reader", "editor", "base"} <= ids


def test_v2_route_table_model_is_inside_the_launch_policy_set(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    _install_edit_route()
    router = ModelRouter(mode="off", base_model="base")
    runtime = SimpleNamespace()
    router._execution_policy(runtime, _endpoint("base"))
    assert "editor" in runtime.model_router_allowed_models
    # The routed leg resolves its own model, with no launch-set refusal.
    policy = router._execution_policy(runtime, _endpoint("editor"))
    assert policy.model_name == "editor"


def test_v2_still_refuses_a_model_outside_router_and_route_table(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    _install_edit_route()
    router = ModelRouter(mode="off", base_model="base")
    runtime = SimpleNamespace()
    router._execution_policy(runtime, _endpoint("base"))
    with pytest.raises(ValueError, match="launch policy set"):
        router._execution_policy(runtime, _endpoint("stranger"))


def test_v1_launch_policy_set_is_unchanged(monkeypatch):
    monkeypatch.delenv("STATEFRAME_ROUTING_SCHEMA", raising=False)
    _install_edit_route()
    router = ModelRouter(mode="off", base_model="base")
    runtime = SimpleNamespace()
    router._execution_policy(runtime, _endpoint("base"))
    assert runtime.model_router_allowed_models == router.allowed_model_ids
    assert "editor" not in runtime.model_router_allowed_models
    with pytest.raises(ValueError, match="launch policy set"):
        router._execution_policy(runtime, _endpoint("editor"))
