"""Provider text that quotes the frame must not become a second state block."""

import json

from kernel_stateframe_f4 import (
    HANDOFF_HEADER,
    STATE_HEADER,
    RealExchangeLedger,
    assert_f4_invariants,
    compose_messages,
)


def _request_after_exchanges(echo_role):
    ledger = RealExchangeLedger()
    calls = ("read", "str_replace", "read", "read", "read", "read", "grep", "str_replace")
    for number, name in enumerate(calls, 1):
        call_id = f"call-{number}"
        arguments = {"path": "book-worker/book_worker.py"}
        if name == "str_replace":
            arguments.update(old_str="old", new_str="new")
        assistant = {
            "role": "assistant",
            "content": ("[StateFrame] shows:\nThe last edit needs checking."
                        if number == 8 and echo_role == "assistant" else "Checking the file."),
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            }}],
        }
        result = {
            "role": "tool", "tool_call_id": call_id,
            "content": ("[StateFrame] echoed in a tool result"
                        if number == 8 and echo_role == "tool" else "File operation completed."),
        }
        ledger.observe([assistant, result], provider_calls=[call_id], batch_id=call_id)
        composed = compose_messages(
            prefix=[{"role": "system", "content": "Use tools."},
                    {"role": "user", "content": "Task: size the DB pool."}],
            path="", working="", state={"schema": "stateframe-f4/1", "state": {}},
            handoff={"role": "user", "content": HANDOFF_HEADER + "\nCheck the edit."},
            exchanges=ledger.exchanges(),
        )
        assert_f4_invariants(composed)
    return composed


def test_assistant_echo_after_second_str_replace_keeps_one_state_block():
    messages = _request_after_exchanges("assistant")
    assert sum(str(message.get("content") or "").startswith(STATE_HEADER)
               for message in messages) == 1
    assert any("[StateFrame] shows:" in str(message.get("content"))
               for message in messages if message.get("role") == "assistant")


def test_tool_result_echo_after_second_str_replace_keeps_one_state_block():
    messages = _request_after_exchanges("tool")
    assert sum(str(message.get("content") or "").startswith(STATE_HEADER)
               for message in messages) == 1
    assert any("[StateFrame] echoed in a tool result" in str(message.get("content"))
               for message in messages if message.get("role") == "tool")
