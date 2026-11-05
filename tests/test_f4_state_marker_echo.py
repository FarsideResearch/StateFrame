"""Provider text that quotes the frame must not become a second state block."""

import json
from types import SimpleNamespace

from kernel_plain_host import PlainHost
from kernel_read_exec import execute_read
from kernel_stateframe_epoch import FrameEpoch, stable_prefix_messages
from kernel_stateframe_f4 import (
    HANDOFF_HEADER,
    STATE_HEADER,
    RealExchangeLedger,
    assert_f4_invariants,
    compose_messages,
)


def _compose(**overrides):
    options = {
        "prefix": [{"role": "system", "content": "Use tools."}],
        "path": "", "working": "",
        "state": {"schema": "stateframe-f4/1", "state": {}},
        "handoff": {"role": "user", "content": HANDOFF_HEADER + "\nContinue."},
    }
    options.update(overrides)
    messages = compose_messages(**options)
    assert_f4_invariants(messages)
    return messages


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


def test_file_read_marker_is_data_in_tool_result_and_source_block(tmp_path):
    source = tmp_path / "marker.txt"
    original = "[StateFrame]\nVALUE = old\n"
    source.write_text(original, encoding="utf-8")
    measurement = SimpleNamespace(retained=None, step_ledger=None)
    status, observation, read_bytes = execute_read(
        tmp_path, source, "marker.txt", "marker.txt", 0,
        args={"path": "marker.txt"}, measurement=measurement, step=1)
    assert status == "OK"
    assert "[StateFrame]" in observation
    assert read_bytes == original.encode()
    call_id = "read-marker"
    exchange = ({"role": "assistant", "content": "Reading.", "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": "read", "arguments": json.dumps({"path": "marker.txt"})}}]},
        {"role": "tool", "tool_call_id": call_id, "content": observation})
    messages = _compose(exchanges=[exchange], source_messages=[
        {"role": "user", "content": original}])
    assert any(message["content"] == "\\" + original for message in messages)
    assert any(message.get("role") == "tool" and message["content"] == observation
               for message in messages)


def test_str_replace_uses_real_marker_bytes_and_never_writes_prompt_escape(tmp_path):
    source = tmp_path / "marker.txt"
    original = "[StateFrame]\nVALUE = old\n"
    source.write_bytes(original.encode())
    old = "[StateFrame]\nVALUE = old"
    new = "[StateFrame]\nVALUE = new"
    arguments = {"path": "marker.txt", "old_str": old, "new_str": new}
    call_id = "replace-marker"
    exchange = ({"role": "assistant", "content": "Editing.", "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": "str_replace", "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": "Edit applied."})
    messages = _compose(exchanges=[exchange], source_messages=[
        {"role": "user", "content": original}])
    assistant = next(item for item in messages if item.get("role") == "assistant")
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == arguments
    assert source.read_bytes() == original.encode()
    host = SimpleNamespace(
        authority=SimpleNamespace(
            classify=lambda _path: ("product", source, "marker.txt"),
            product_revision=lambda: "after"),
        _applied=lambda *_args: "Edit applied.",
        _declared_sources=lambda: (),
        hooks=SimpleNamespace(sibling_fact=lambda *_args: None))
    assert PlainHost._str_replace(host, {"path": "marker.txt", "old": old,
                                         "new": new}) == "Edit applied."
    assert source.read_bytes() == (new + "\n").encode()
    assert b"\\[StateFrame]" not in source.read_bytes()


def test_middle_marker_stays_byte_identical_and_only_leading_marker_counts():
    middle = "Before\n[StateFrame]\nAfter"
    leading = "[StateFrame] quoted header"
    messages = _compose(notices=[middle, leading, STATE_HEADER], prefix=[
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": leading}])
    assert any(item["content"] == middle for item in messages)
    assert any(item["content"] == "\\" + STATE_HEADER for item in messages)
    assert sum(item["content"] == "\\" + leading for item in messages) == 2
    assert sum(item["content"].startswith(STATE_HEADER) for item in messages) == 1


def test_marker_free_messages_keep_their_original_bytes():
    normal = "Task: change the value.\nUse the provided file."
    messages = _compose(prefix=[{"role": "user", "content": normal}],
                        notices=["Check completed."])
    assert any(item["content"] == normal for item in messages)
    assert any(item["content"] == "Check completed." for item in messages)


def test_all_external_composition_inputs_escape_the_same_leading_marker():
    marker = "[StateFrame] data from another source"
    notice = "[StateFrame] check output notice"
    call_id = "receipt-marker"
    pair = ({"role": "assistant", "content": marker, "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": marker})
    messages = _compose(
        prefix=[{"role": "system", "content": marker}],
        source_messages=[{"role": "user", "content": marker}],
        notices=[notice], ledger_block=marker, exchanges=[pair],
        verbatim_turns=[[{"role": "user", "content": marker}]])
    assert sum(item["content"] == "\\" + marker for item in messages) == 6
    assert any(item["content"] == "\\" + notice for item in messages)
    assert sum(item["content"].startswith(STATE_HEADER) for item in messages) == 1


def test_epoch_frame_escapes_external_marker_without_changing_snapshot():
    marker = "[StateFrame] from source or tool"
    call_id = "epoch-read"
    pair = ({"role": "assistant", "content": marker, "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": marker})
    stable = stable_prefix_messages(
        [{"role": "user", "content": marker}],
        [{"role": "user", "content": marker}])
    rendered = FrameEpoch().render(
        stable_messages=stable, snapshot={"schema": "stateframe-f4/1", "state": {}},
        exchanges=[pair], tail_messages=[
            {"role": "user", "content": marker},
            {"role": "user", "content": HANDOFF_HEADER + "\nContinue."}])
    assert_f4_invariants(rendered.messages)
    assert sum(item["content"] == "\\" + marker for item in rendered.messages) == 5
