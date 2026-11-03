"""The study capacity preflight reads /slots with the request's own key."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"))
import capability_capacity_preflight as preflight  # noqa: E402


class _Stop(Exception):
    pass


@pytest.mark.parametrize("key,expected", [("secret-key", "Bearer secret-key"), ("", None)])
def test_slots_read_sends_authorization_only_with_a_key(monkeypatch, key, expected):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["auth"] = request.get_header("Authorization")
        seen["url"] = request.full_url
        raise _Stop()

    monkeypatch.setattr(preflight.urllib.request, "urlopen", fake_urlopen)

    class Capture:
        @staticmethod
        def send(capture, body, endpoint, key, *, deadline=None):
            return b""

    preflight.install({"capacity_preflight": {"context_limit": 65536}}, Capture)
    with pytest.raises(_Stop):
        Capture.send(object(), {"id_slot": 0, "max_tokens": 8},
                     "http://127.0.0.1:8080/v1/chat/completions", key)
    assert seen["url"] == "http://127.0.0.1:8080/slots"
    assert seen["auth"] == expected
