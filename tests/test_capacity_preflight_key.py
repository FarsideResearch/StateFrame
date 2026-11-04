"""The study capacity preflight reads /slots with the request's own key."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"))
import capability_capacity_preflight as preflight  # noqa: E402
import kernel_bounded_http  # noqa: E402
import kernel_provider_dialect as provider_dialect  # noqa: E402


class _Stop(Exception):
    pass


@pytest.mark.parametrize("key,expected", [("secret-key", "Bearer secret-key"), ("", None)])
def test_slots_read_sends_authorization_only_with_a_key(monkeypatch, key, expected):
    seen = {}

    def fake_probe(endpoint, selected_key, path, *, timeout):
        seen["key"] = selected_key
        seen["endpoint"] = endpoint
        seen["path"] = path
        seen["timeout"] = timeout
        raise _Stop()

    def fake_capabilities(endpoint, selected_key):
        assert endpoint.endswith("/v1/chat/completions")
        assert selected_key == key
        return provider_dialect.ProviderCapabilities(
            provider_dialect.LLAMA_CPP, slots=True, slot_ids=frozenset({0}),
        )

    monkeypatch.setattr(kernel_bounded_http, "bounded_json_probe", fake_probe)
    monkeypatch.setattr(provider_dialect, "capabilities_for", fake_capabilities)

    class Capture:
        @staticmethod
        def send(capture, body, endpoint, key, *, deadline=None):
            return b""

    preflight.install({"capacity_preflight": {"context_limit": 65536}}, Capture)
    with pytest.raises(_Stop):
        Capture.send(object(), {"id_slot": 0, "max_tokens": 8},
                     "http://127.0.0.1:8080/v1/chat/completions", key)
    assert seen == {
        "key": key,
        "endpoint": "http://127.0.0.1:8080",
        "path": "/slots",
        "timeout": 2.0,
    }
    assert (f"Bearer {seen['key']}" if seen["key"] else None) == expected


def test_capacity_preflight_refuses_strict_provider_before_native_posts(monkeypatch):
    from kernel_provider_dialect import ProviderCapabilityError, ProviderCapabilities

    probes = []
    sent = []
    monkeypatch.setattr(
        provider_dialect, "capabilities_for",
        lambda *_args, **_kwargs: ProviderCapabilities(dialect=provider_dialect.OPENAI_STRICT),
    )
    monkeypatch.setattr(
        kernel_bounded_http, "bounded_json_probe",
        lambda *args, **kwargs: probes.append((args, kwargs)),
    )

    class Capture:
        @staticmethod
        def send(capture, body, endpoint, key, *, deadline=None):
            sent.append((endpoint, key))
            return b""

    preflight.install({"capacity_preflight": {"context_limit": 65536}}, Capture)
    with pytest.raises(ProviderCapabilityError, match="study capacity preflight.*llama.cpp"):
        Capture.send(object(), {"id_slot": 0, "max_tokens": 8},
                     "http://127.0.0.1:8080/v1/chat/completions", "")
    assert probes == []
    assert sent == []
