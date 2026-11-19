import copy
import json
from pathlib import Path

from kernel_cache_layout_view import semantic_messages
from stateframe.inspect_run import frame_of, show_send


WIRE_FIXTURE = Path(__file__).parent / "fixtures/parallel/off_wire_main_73a6a78a/0001.request.json"


def test_frame_view_includes_authority_carried_by_the_recorded_request(tmp_path):
    wire = WIRE_FIXTURE.read_bytes()
    body = json.loads(wire)
    original_messages = copy.deepcopy(body["messages"])
    semantic = frame_of({"messages": semantic_messages(body["messages"])})
    actual = frame_of(body)

    run_wire = tmp_path / "wire"
    run_wire.mkdir()
    (run_wire / "0001.request.json").write_bytes(wire)

    assert semantic is not None
    assert actual is not None
    for key in ("public_checks", "done_permission", "submission"):
        assert actual["state"].get(key) == semantic["state"][key]

    assert semantic["state"]["public_checks"]["state"] == "NOT_RUN"
    assert semantic["state"]["done_permission"]["state"] == "DENY"
    assert semantic["state"]["submission"]["state"] == "BLOCKED"
    assert json.loads(show_send(tmp_path, 1, what="frame")) == semantic
    assert show_send(tmp_path, 1, what="bytes") == wire
    assert body["messages"] == original_messages

    expected_message_lines = []
    for index, message in enumerate(original_messages):
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        calls = message.get("tool_calls")
        expected_message_lines.extend((
            f"--- [{index}] {message.get('role')} ({len(text or '')} chars)"
            + (f" tool_calls={json.dumps(calls)}" if calls else ""),
            text or "",
        ))
    assert show_send(tmp_path, 1, what="messages") == "\n".join(expected_message_lines)
