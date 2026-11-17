"""MODEL-ROUTING-UNIFY MR-7: settings-only route changes escalate under routing v2 (schema 2) only.

Schema 1 keeps the legacy (endpoint, model) comparison byte-for-byte. Schema 2 compares the full
effective route identity: endpoint, model, api_key_env and the body settings apply_body writes.
The chosen mechanism returns the target role, so the leg is re-prepared with the target settings.
"""
from __future__ import annotations

import pytest

from stateframe import routes
import kernel_frame_routes as frame_routes

DEFAULT = {"endpoint": "http://127.0.0.1:8080/v1", "model": "base", "api_key_env": None,
           "temperature": 1.0, "max_tokens": 24000, "timeout_s": 1800.0,
           "slot": None, "context_tokens": 65536, "allow_remote": False}

BODY_BASE = {"model": "base", "temperature": 1.0, "max_tokens": 24000}


@pytest.fixture(autouse=True)
def clear_routes(monkeypatch):
    monkeypatch.delenv("STATEFRAME_ROUTING_SCHEMA", raising=False)
    frame_routes.install(None)
    yield
    frame_routes.install(None)


def table(*entries: str) -> dict:
    # EDIT_ONLY_KEY must exist in the environ the table resolves against; the value is never sent.
    return routes.resolve(DEFAULT, cli=list(entries), environ={"EDIT_ONLY_KEY": "test-only"}).public()


def set_schema(monkeypatch, value: str) -> None:
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", value)


def test_v2_temperature_only_change_escalates(monkeypatch):
    # Same endpoint and model on read and edit; only temperature differs. Must escalate under v2.
    set_schema(monkeypatch, "2")
    frame_routes.install(table("edit.temperature=0.2"))
    reader = frame_routes.resolve("read")
    assert frame_routes.escalation_target(reader, [{"function": {"name": "edit", "arguments": "{}"}}]) == "edit"


def test_v2_api_key_env_only_change_escalates(monkeypatch):
    set_schema(monkeypatch, "2")
    frame_routes.install(table("edit.api_key_env=EDIT_ONLY_KEY"))
    reader = frame_routes.resolve("read")
    assert frame_routes.escalation_target(reader, [{"function": {"name": "edit", "arguments": "{}"}}]) == "edit"


def test_v2_identical_route_does_not_escalate(monkeypatch):
    # No role differs from the answering route in any field, so nothing escalates.
    set_schema(monkeypatch, "2")
    frame_routes.install(table())
    reader = frame_routes.resolve("read")
    assert frame_routes.escalation_target(reader, [{"function": {"name": "edit", "arguments": "{}"}}]) is None


def test_v1_temperature_only_change_does_not_escalate(monkeypatch):
    # Legacy schema 1: comparison stays (endpoint, model). A temperature-only change is still ignored.
    set_schema(monkeypatch, "1")
    frame_routes.install(table("edit.temperature=0.2"))
    reader = frame_routes.resolve("read")
    assert frame_routes.escalation_target(reader, [{"function": {"name": "edit", "arguments": "{}"}}]) is None


def test_v1_unset_schema_is_legacy(monkeypatch):
    monkeypatch.delenv("STATEFRAME_ROUTING_SCHEMA", raising=False)
    frame_routes.install(table("edit.api_key_env=EDIT_ONLY_KEY"))
    reader = frame_routes.resolve("read")
    assert frame_routes.escalation_target(reader, [{"function": {"name": "edit", "arguments": "{}"}}]) is None


def test_identity_matches_applied_settings(monkeypatch):
    # The compared identity must be exactly what apply_body writes. Two routes with equal identity
    # produce equal bodies; a differing applied field produces a differing identity.
    set_schema(monkeypatch, "2")
    frame_routes.install(table("edit.temperature=0.2", "edit.api_key_env=EDIT_ONLY_KEY"))
    edit = frame_routes.resolve("edit")
    identity = frame_routes.route_identity(edit)
    assert identity["temperature"] == 0.2
    assert identity["api_key_env"] == "EDIT_ONLY_KEY"
    body = dict(BODY_BASE)
    frame_routes.apply_body(body, edit, default_endpoint=edit.endpoint)
    assert body["temperature"] == identity["temperature"]
    assert body["model"] == identity["model"]
    assert "max_tokens" in body


def test_identity_ignores_fields_apply_body_never_writes(monkeypatch):
    # A temperature equal to the baseline is not written by apply_body, and a None temperature is not
    # written either. Both routes therefore apply the same body, so neither may count as a difference.
    set_schema(monkeypatch, "2")
    baseline = {"model": "base", "temperature": 0.5, "max_tokens": 24000}
    frame_routes.install(table(), baseline=baseline)
    base = dict(frame_routes.resolve("read").settings)
    answering = frame_routes.Route("read", "run", None, {**base, "temperature": None})
    target = frame_routes.Route("edit", "run", None, {**base, "temperature": 0.5})
    assert frame_routes.route_identity(answering) == frame_routes.route_identity(target)
    body_a, body_b = dict(BODY_BASE), dict(BODY_BASE)
    frame_routes.apply_body(body_a, answering, default_endpoint=answering.endpoint)
    frame_routes.apply_body(body_b, target, default_endpoint=target.endpoint)
    assert body_a == body_b
