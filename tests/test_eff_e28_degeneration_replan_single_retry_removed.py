import json
from types import SimpleNamespace

import pytest

import kernel_measurement as measurement
from kernel_response_truncation import MODEL_NOTE
from probes import react_repair


DONE_TOOL = {"type": "function", "function": {
    "name": "done", "description": "Finish the task.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}}
DONE_CALL = {"id": "done-1", "type": "function",
             "function": {"name": "done", "arguments": "{}"}}


class ProviderFixture:
    """Exercise the real decision to resubmit from ask_tools' transport boundary."""
    _v43g_note_usage = measurement.RequestCapture._v43g_note_usage
    _v43g_retry_degenerate = measurement.RequestCapture._v43g_retry_degenerate
    request_kind = "framefit5_primary"
    request_id = "r1"
    step = 1
    v43g_degradation_retry_used = False
    v43g_logical_completion_tokens = v43g_logical_prompt_tokens = 0

    def __init__(self, *, terminal_done=False):
        self.posts = []
        self.terminal_done = terminal_done
        self.journal = SimpleNamespace(emit=lambda *args, **kwargs: {"event_id": "detected"})

    def send(self, body, endpoint, key, **kwargs):
        self.qualified_request = SimpleNamespace(body=body)
        self.posts.append(json.dumps(body).encode())
        if len(self.posts) == 2:
            if self.terminal_done:
                return {"message": {"content": None, "tool_calls": [DONE_CALL]},
                        "finish_reason": "tool_calls"}
            return {"message": {"content": "recovered"}, "finish_reason": "stop"}
        choice = {"finish_reason": "length", "message": {
            "content": "same phrase in a capped output " * 40,
            "tool_calls": [{"id": "bad", "function": {"name": "edit", "arguments": "{}"}}]}}
        retry = self._v43g_retry_degenerate({"choices": [choice]}, body, endpoint, key, deadline=None)
        return retry or choice

    send_with_frozen_timeout_retry = send


def run(monkeypatch, mode, initial, *, temperature=0, terminal_done=False):
    monkeypatch.setenv("STATEFRAME_DEGENERATION_GUARD", "on")
    monkeypatch.setenv("STATEFRAME_DEGENERATION_REPLAN", mode)
    monkeypatch.setenv("REACT_INITIAL_OUTPUT_TOKENS", str(initial))
    monkeypatch.setenv("REACT_MAX_OUTPUT_TOKENS", "128")
    capture = ProviderFixture(terminal_done=terminal_done)
    token = measurement.ACTIVE_REQUEST.set(capture)
    try:
        result = react_repair.ask_tools([{"role": "user", "content": "go"}],
                                        tools=[DONE_TOOL] if terminal_done else [],
                                        temp=temperature)
    finally:
        measurement.ACTIVE_REQUEST.reset(token)
    return result, capture.posts


def test_capped_degenerate_primary_posts_once_and_dispatches_nothing(monkeypatch):
    control, original = run(monkeypatch, "off", 128)
    result, posts = run(monkeypatch, "on", 128)
    assert control == ("recovered", [])
    assert original == posts * 2
    assert len(posts) == 1
    assert result == (MODEL_NOTE, [])


def test_sampling_temperature_keeps_retry_that_can_return_final_done(monkeypatch):
    control, control_posts = run(monkeypatch, "off", 128, temperature=1.0, terminal_done=True)
    result, posts = run(monkeypatch, "on", 128, temperature=1.0, terminal_done=True)

    assert control == ("", [DONE_CALL])
    assert len(control_posts) == 2
    assert result == control
    assert posts == control_posts


@pytest.mark.parametrize("mode", ["on", "off"])
def test_below_ceiling_keeps_identical_retry(monkeypatch, mode):
    result, posts = run(monkeypatch, mode, 64)
    assert result == ("recovered", [])
    assert len(posts) == 2 and posts[0] == posts[1]
