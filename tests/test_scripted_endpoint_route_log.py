"""Optional scripted endpoint controls used by routed host proofs."""
from __future__ import annotations

import pytest

from stateframe.scripted_endpoint import Script


def test_repeat_step_keeps_same_reply_for_a_resumed_pending_send():
    script = Script({"model": "repeat", "steps": [
        {"tool": "file_create", "arguments": {"path": "accepted.txt"}, "repeat": True}]})
    assert script.next_call(["file_create"])["tool"] == "file_create"
    assert script.next_call(["file_create"])["tool"] == "file_create"
    script.assert_consumed()


def test_repeat_optional_unoffered_step_does_not_loop():
    script = Script({"steps": [{"tool": "file_create", "repeat": True, "required": False}]})
    assert script.next_call(["read"])["tool"] == "done"


@pytest.mark.parametrize("value", [-1, 31, "5", True])
def test_invalid_script_delay_refused(value):
    with pytest.raises(ValueError, match="delay_s"):
        Script({"steps": [{"tool": "read", "delay_s": value}]})
