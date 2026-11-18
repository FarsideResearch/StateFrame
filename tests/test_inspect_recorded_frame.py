import json
from pathlib import Path

from kernel_cache_layout_view import semantic_messages
from stateframe.inspect_run import frame_of


WIRE_FIXTURE = Path(__file__).parent / "fixtures/parallel/off_wire_main_73a6a78a/0001.request.json"


def test_frame_view_includes_authority_carried_by_the_recorded_request():
    body = json.loads(WIRE_FIXTURE.read_bytes())
    semantic = frame_of({"messages": semantic_messages(body["messages"])})
    actual = frame_of(body)

    assert semantic is not None
    assert actual is not None
    for key in ("public_checks", "done_permission", "submission"):
        assert actual["state"].get(key) == semantic["state"][key]

    assert semantic["state"]["public_checks"]["state"] == "NOT_RUN"
    assert semantic["state"]["done_permission"]["state"] == "DENY"
    assert semantic["state"]["submission"]["state"] == "BLOCKED"
