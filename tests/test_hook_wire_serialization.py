"""Private request classification remains local to final wire preparation."""

import json
from types import SimpleNamespace

from kernel_final_wire_allocation import _prepare_qualified_hook_body

from kernel_measurement import (
    _serialize_final_request_body,
    _without_private_message_keys,
)


def test_private_message_keys_are_removed_from_a_copy():
    body = {
        "messages": [
            {"role": "user", "content": "[USER HOOKS] note", "_request_budget_section": "user_hooks"},
            {"role": "assistant", "content": "answer", "_other": "local"},
        ],
        "model": "scripted",
    }
    stripped = _without_private_message_keys(body)
    assert stripped == {
        "messages": [
            {"role": "user", "content": "[USER HOOKS] note"},
            {"role": "assistant", "content": "answer"},
        ],
        "model": "scripted",
    }
    assert body["messages"][0]["_request_budget_section"] == "user_hooks"
    assert body["messages"][1]["_other"] == "local"


def test_no_hook_serializer_preserves_raw_body_bytes():
    body = {"messages": [{"role": "user", "content": "literal", "_local": "kept"}]}
    assert _serialize_final_request_body(body, None, "request-1") == json.dumps(body).encode()


def test_qualified_hook_body_is_stripped_before_wire_seal():
    calls = []

    class PanelAdmission:
        def admit_panel(self, body, wire, request_id):
            calls.append((body, json.loads(wire), request_id))

    body = {"messages": [
        {"role": "user", "content": "[USER HOOKS] note", "_request_budget_section": "user_hooks"},
    ]}
    qualified = _prepare_qualified_hook_body(
        SimpleNamespace(_hook_wiring=PanelAdmission()), body, "request-1"
    )
    assert calls == [(body, qualified, "request-1")]
    assert qualified["messages"] == [{"role": "user", "content": "[USER HOOKS] note"}]
    assert body["messages"][0]["_request_budget_section"] == "user_hooks"
    assert _prepare_qualified_hook_body(SimpleNamespace(), body, "request-1") is body
