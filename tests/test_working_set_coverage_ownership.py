"""Host-looking opaque messages never confer coverage repair authority."""
import copy
import json
from types import SimpleNamespace

import pytest

from kernel_working_set_prepare import PreparedCandidate, prepare_candidate
from kernel_working_set_proof import verify_certificate
from kernel_working_set_snapshot import freeze_preparation
from test_working_set_render_boundaries import envelope


@pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
@pytest.mark.parametrize("text", [
    "[Source coverage]\nopaque content with no recognized path λ",
    "[Source coverage]\nghost.py (9 lines, unverified): all 9 lines are shown below.",
])
def test_allocator_preserves_opaque_coverage_markers(role, text):
    opaque = {"role": role, "content": text}
    suffix = [opaque] if role == "user" else []
    body = envelope(suffix)
    body["messages"][2]["content"] = '[StateFrame]\n{"state":{"framefit5":{"mode":"FRAME_ONLY"}}}'
    if role != "user":
        target = next(message for message in body["messages"] if message["role"] == role)
        target["content"] = text
        opaque = copy.deepcopy(target)
    original = copy.deepcopy(body)
    rt = SimpleNamespace(_active_capture=SimpleNamespace(_framefit5_newest_user_suffix=suffix))
    frozen = freeze_preparation(body, rt, request_id="opaque-coverage", byte_cap=100_000)
    prepared = prepare_candidate(frozen, measure=lambda wire: (1, 1, "test"))
    assert isinstance(prepared, PreparedCandidate)
    assert opaque in json.loads(prepared.wire)["messages"]
    assert verify_certificate(prepared.wire, prepared.rendered.certificate)
    assert body == original


def test_only_explicitly_owned_coverage_coordinate_is_healed():
    from kernel_frame_source_snapshot import reconcile_source_coverage

    text = "[Source coverage]\nghost.py (1 line, unverified): the single line is shown below."
    messages = [{"role": "user", "content": text},
                {"role": "user", "content": text},
                {"role": "tool", "tool_call_id": "opaque", "content": text}]
    opaque = copy.deepcopy(messages[1:])
    reconcile_source_coverage(messages, owned_indexes=(0,))
    assert "ghost.py: not shown;" in messages[0]["content"]
    assert messages[1:] == opaque
