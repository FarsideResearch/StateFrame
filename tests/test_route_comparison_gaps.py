"""Synthetic regression probes for schema-2 route identity."""
from __future__ import annotations

import hashlib
import json

import pytest

from stateframe import routes
import kernel_frame_routes as frame_routes


DEFAULT = {
    "endpoint": "http://127.0.0.1:8080/v1",
    "model": "base",
    "api_key_env": None,
    "temperature": 1.0,
    "max_tokens": 24000,
    "timeout_s": 1800.0,
    "slot": None,
    "context_tokens": 65536,
    "allow_remote": False,
}

READ = {"id": "read-1", "function": {"name": "read", "arguments": "{}"}}
EDIT = {"id": "edit-1", "function": {"name": "edit", "arguments": '{"path":"sample.py"}'}}


@pytest.fixture(autouse=True)
def schema_two(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    frame_routes.install(None)
    yield
    frame_routes.install(None)


class SyntheticJournal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))
        return fields


def test_schema_two_detects_non_body_route_settings():
    differences = (
        ("timeout_s", "900"),
        ("context_tokens", "32768"),
        ("tokenizer", "http://127.0.0.1:8081/v1"),
        ("allow_remote", "true"),
    )
    undetected = []

    for field, value in differences:
        spec = routes.resolve(DEFAULT, cli=[f"edit.{field}={value}"], environ={}).public()
        frame_routes.install(spec, baseline=DEFAULT)
        answering = frame_routes.resolve("read")
        _kept, withheld, target = frame_routes.route_split(answering, [READ, EDIT])
        if target != "edit" or withheld != [EDIT]:
            undetected.append(field)

    assert undetected == [], f"schema 2 treated operational route differences as equal: {undetected}"


@pytest.mark.parametrize(("field", "value"), (
    ("timeout_s", "900"),
    ("context_tokens", "32768"),
    ("tokenizer", "http://127.0.0.1:8081/v1"),
    ("allow_remote", "true"),
))
def test_schema_two_route_identity_differs_for_each_operational_setting(field, value):
    base_spec = routes.resolve(DEFAULT, environ={}).public()
    changed_spec = routes.resolve(DEFAULT, cli=[f"edit.{field}={value}"], environ={}).public()

    frame_routes.install(base_spec, baseline=DEFAULT)
    base_identity = frame_routes.route_identity(frame_routes.resolve("read"))
    frame_routes.install(changed_spec, baseline=DEFAULT)
    changed_identity = frame_routes.route_identity(frame_routes.resolve("edit"))

    differing = {name for name in base_identity.keys() | changed_identity.keys()
                 if base_identity.get(name) != changed_identity.get(name)}
    assert differing == {field}


def test_schema_two_route_identity_normalizes_explicit_effective_defaults():
    base_spec = routes.resolve(DEFAULT, environ={}).public()
    explicit_spec = routes.resolve(
        DEFAULT,
        cli=[
            "edit.timeout_s=1800",
            "edit.context_tokens=65536",
            "edit.tokenizer=http://127.0.0.1:8080/v1",
            "edit.allow_remote=false",
        ],
        environ={},
    ).public()

    frame_routes.install(base_spec, baseline=DEFAULT)
    base_identity = frame_routes.route_identity(frame_routes.resolve("read"))
    frame_routes.install(explicit_spec, baseline=DEFAULT)
    explicit_identity = frame_routes.route_identity(frame_routes.resolve("edit"))

    assert base_identity == explicit_identity
    assert base_identity["timeout_s"] == 1800.0
    assert base_identity["context_tokens"] == 65536
    assert base_identity["tokenizer"] == DEFAULT["endpoint"]
    assert base_identity["allow_remote"] is False


def test_schema_two_identity_tracks_route_tokenizer_with_shared_environment_set(monkeypatch):
    tokenizer = "http://127.0.0.1:8081/v1"
    monkeypatch.setenv("LLAMA_CPP_TOKENIZE_URL", tokenizer)
    spec = routes.resolve(
        DEFAULT,
        cli=[f"edit.tokenizer={tokenizer}", "finish.model=other"],
        environ={"LLAMA_CPP_TOKENIZE_URL": tokenizer},
    ).public()
    frame_routes.install(spec, baseline=DEFAULT)

    answering = frame_routes.resolve("read")
    target = frame_routes.resolve("edit")

    # Routed requests bind the route tokenizer directly; the global tokenizer cannot mask it.
    assert frame_routes._route_differs(target, answering)


def test_schema_two_resumes_legacy_routes_without_false_escalation():
    old_spec = routes.resolve(DEFAULT, environ={}).public()
    operational_fields = {"timeout_s", "context_tokens", "tokenizer", "allow_remote"}
    for settings in old_spec["roles"].values():
        for field in operational_fields:
            settings.pop(field, None)
    canonical = json.dumps(old_spec["roles"], sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    old_spec["digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # Resume normalizes a pre-extension journal route table against its recorded defaults.
    restored = routes.resolve(DEFAULT, recorded=old_spec, recorded_default=DEFAULT, environ={})
    frame_routes.install(restored.public(), baseline=DEFAULT)
    answering = frame_routes.resolve("read")
    kept, withheld, target = frame_routes.route_split(answering, [READ, EDIT])

    assert kept == [READ, EDIT]
    assert withheld == []
    assert target is None


def test_non_body_route_settings_do_not_enter_the_provider_body():
    spec = routes.resolve(
        DEFAULT,
        cli=[
            "edit.timeout_s=900",
            "edit.context_tokens=32768",
            "edit.tokenizer=http://127.0.0.1:8081/v1",
            "edit.allow_remote=true",
        ],
        environ={},
    ).public()
    frame_routes.install(spec, baseline=DEFAULT)
    original = {"model": "base", "temperature": 1.0, "max_tokens": 24000,
                "messages": [{"role": "user", "content": "synthetic"}]}
    body = dict(original)

    frame_routes.apply_body(body, frame_routes.resolve("edit"),
                            default_endpoint="http://127.0.0.1:8080/v1/chat/completions")

    assert body == original


def test_schema_one_keeps_the_legacy_route_comparison(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "1")
    spec = routes.resolve(DEFAULT, cli=["edit.timeout_s=900"], environ={}).public()
    frame_routes.install(spec, baseline=DEFAULT)

    assert not frame_routes._route_differs(frame_routes.resolve("edit"), frame_routes.resolve("read"))
