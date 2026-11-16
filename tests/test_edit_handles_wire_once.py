"""The [CURRENT SOURCE] map reaches the provider exactly once per request, in the final user message.

Drives the real worker under STATEFRAME_EDIT_PROTOCOL=handles against the scripted endpoint and reads
the bodies the endpoint actually received (after every send stage: lean finalize, visibility, fit,
seal), so a stage that drops or duplicates the trailing map is caught here and not only in unit tests.
"""
from __future__ import annotations

import json

import pytest

import stateframe.scripted_endpoint as scripted_endpoint
from kernel_edit_handles import CURRENT_SOURCE_PREFIX
from test_edit_protocol_switch_wiring import RULES_PATH, _drive, _edit_source_step


@pytest.fixture
def received(monkeypatch):
    """Bodies the scripted endpoint received, in order."""
    scripts: list[scripted_endpoint.Script] = []
    original = scripted_endpoint.Script

    class Recording(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            scripts.append(self)

    monkeypatch.setattr(scripted_endpoint, "Script", Recording)
    return lambda: [body for script in scripts for body in script.requests]


def _map_messages(body: dict) -> list[int]:
    return [index for index, message in enumerate(body.get("messages", []))
            if CURRENT_SOURCE_PREFIX in json.dumps(message.get("content"))]


def _engine_sends(bodies: list[dict]) -> list[dict]:
    """Tool-offering bodies (the engine's own requests)."""
    return [body for body in bodies if body.get("tools")]


def test_map_follows_fresh_source_once_per_request_in_the_final_user_message(received, monkeypatch, tmp_path):
    # read -> edit (resolves against the map) -> read again (fresh bytes, fresh map).
    _drive(tmp_path, monkeypatch, protocol="handles", stop_at=5,
           steps=[{"tool": "read", "required": True, "arguments": {"path": RULES_PATH}},
                  _edit_source_step("q1h1", "= 25", "= 30"),
                  {"tool": "read", "required": True, "arguments": {"path": RULES_PATH}},
                  {"tool": "read", "required": True, "arguments": {"path": RULES_PATH}}])
    sends = _engine_sends(received())[:4]
    assert len(sends) == 4
    maps = [_map_messages(body) for body in sends]
    # Request 0 delivers no source; request 2 only holds bytes the edit made stale; neither may
    # carry a map. Requests 1 and 3 deliver the file's current bytes and carry exactly one map.
    assert maps[0] == [] and maps[2] == []
    for number in (1, 3):
        messages = sends[number]["messages"]
        last_user = max(i for i, m in enumerate(messages) if m.get("role") == "user")
        assert maps[number] == [last_user], f"request {number}: map not only in the final user message"
        assert json.dumps(messages[last_user]["content"]).count(CURRENT_SOURCE_PREFIX) == 1
    replies = [m["content"] for m in sends[2]["messages"] if m.get("role") == "tool"]
    assert json.loads(replies[-1])["status"] == "APPLIED", "the first edit_source did not resolve"
    assert "q3h1" in json.dumps(sends[3]["messages"])


def test_legacy_safe_wire_never_carries_a_map(received, monkeypatch, tmp_path):
    _drive(tmp_path, monkeypatch, protocol="legacy_safe", stop_at=2,
           steps=[{"tool": "done", "required": True, "arguments": {}}])
    assert all(not _map_messages(body) for body in received())


def test_run_binding_is_replaced_and_cleared_per_run(tmp_path):
    import kernel_edit_request_map as request_map

    first, second = object(), object()
    request_map.bind_run(first, tmp_path / "a")
    request_map.bind_run(second, tmp_path / "b")
    binding = request_map.current_binding()
    assert binding is not None and binding.state is second and binding.root == tmp_path / "b"
    request_map.bind_run(None, tmp_path / "c")  # a legacy_safe run never inherits an earlier state
    assert request_map.current_binding() is None


def test_binding_is_visible_from_a_send_thread_and_rebinds_atomically(tmp_path):
    """The transport sends from a worker thread that does not inherit context variables."""
    import threading

    import kernel_edit_request_map as request_map

    state = object()
    request_map.bind_run(state, tmp_path)
    seen: list[object] = []
    sender = threading.Thread(target=lambda: seen.append(request_map.current_binding()))
    sender.start()
    sender.join()
    assert seen[0] is not None and seen[0].state is state and seen[0].root == tmp_path

    states = [object() for _ in range(8)]
    observed: list[object] = []

    def rebind(value: object) -> None:
        for _ in range(200):
            request_map.bind_run(value, tmp_path / "w")
            binding = request_map.current_binding()
            observed.append(None if binding is None else binding.state)

    workers = [threading.Thread(target=rebind, args=(value,)) for value in states]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    # Every read sees one whole binding from some run, never a torn or foreign value.
    assert set(observed) <= set(states)
    request_map.bind_run(None, tmp_path)
