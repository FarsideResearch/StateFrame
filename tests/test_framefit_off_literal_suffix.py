"""Provider-wire coverage for literal task text that quotes a host fact."""

import copy
import json
import urllib.request
from types import SimpleNamespace

import pytest

import kernel_greenfact
from _timeouts import scaled
from kernel_greenfact import GREENFACT_TEXT
from kernel_measurement import _serialize_final_request_body
from kernel_stateframe_framefit5 import (
    PRODUCT_MODE,
    STATE_HEADER,
    VOLATILE_TAIL_MARKER,
    assemble_framefit5_request,
)
from kernel_wire_truth_facts import render_fact_messages
from stateframe.scripted_endpoint import Script, serve


TASK = "Task: Retain this quotation verbatim. Café."
FOLLOWUP = "Also retain this example. Café."
SYSTEM = {"role": "system", "content": "Review the supplied text."}
STATE = {
    "role": "user",
    "content": STATE_HEADER + "\n" + json.dumps({"state": {"framefit5": {"mode": PRODUCT_MODE}}}),
}
EXCHANGE = [
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "sample"},
]


def _request(task, followup):
    task_row = {"role": "user", "content": task}
    transcript = [SYSTEM, task_row, *EXCHANGE, {"role": "user", "content": followup}]
    messages = assemble_framefit5_request([SYSTEM, task_row, STATE], transcript)
    return {"model": "scripted", "messages": messages}, transcript


@pytest.fixture
def scripted_provider():
    script = Script({"model": "scripted"})
    server = serve(script)
    try:
        yield script, f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    finally:
        server.shutdown()
        server.server_close()


def _send(body, scripted_provider):
    script, endpoint = scripted_provider
    wire = _serialize_final_request_body(body, None, "suffix-check")
    request = urllib.request.Request(
        endpoint, data=wire, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=scaled(5)) as response:
        assert response.status == 200
    assert script.requests[-1] == body
    return wire


@pytest.mark.host_sandbox
@pytest.mark.parametrize("row", ["task", "followup", "multipart"])
def test_greenfact_off_preserves_literal_wire_suffix(monkeypatch, scripted_provider, row):
    monkeypatch.setenv("STATEFRAME_GREENFACT", "off")
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    control_followup = (
        [{"type": "text", "text": FOLLOWUP}] if row == "multipart" else FOLLOWUP
    )
    control, _ = _request(TASK, control_followup)
    assert control["messages"] == [
        SYSTEM, {"role": "user", "content": TASK}, STATE,
        {"role": "user", "content": control_followup}, *EXCHANGE,
        {"role": "user", "content": VOLATILE_TAIL_MARKER},
    ]
    task = TASK + GREENFACT_TEXT if row == "task" else TASK
    followup = {
        "task": FOLLOWUP,
        "followup": FOLLOWUP + GREENFACT_TEXT,
        "multipart": [{"type": "text", "text": FOLLOWUP + GREENFACT_TEXT}],
    }[row]
    actual, _ = _request(task, followup)
    assert not kernel_greenfact.maybe_append_greenfact(actual["messages"], None)
    expected = copy.deepcopy(control)
    expected["messages"][1]["content"] = task
    expected["messages"][3]["content"] = followup

    assert _send(actual, scripted_provider) == json.dumps(expected).encode("utf-8")


@pytest.mark.host_sandbox
@pytest.mark.parametrize("green_revision", [None, "receipt-current"], ids=["unverified", "verified"])
def test_greenfact_on_refit_preserves_literal_and_only_renders_host_fact(
    monkeypatch, scripted_provider, green_revision,
):
    monkeypatch.setenv("STATEFRAME_GREENFACT", "on")
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    monkeypatch.setattr(kernel_greenfact, "_green_revision", lambda _runtime: (green_revision, None))
    task, followup = TASK + GREENFACT_TEXT, FOLLOWUP + GREENFACT_TEXT
    body, transcript = _request(task, followup)
    assert kernel_greenfact.maybe_append_greenfact(
        body["messages"], SimpleNamespace(), request_id="suffix-check",
    ) is (green_revision is not None)
    host_fact, = render_fact_messages({"green_state": "fixture"})
    body["messages"].append(host_fact)
    body["messages"] = assemble_framefit5_request(body["messages"], transcript)

    host_tail = VOLATILE_TAIL_MARKER + "\n\n" + host_fact["content"]
    if green_revision is not None:
        host_tail += GREENFACT_TEXT
    expected = {"model": "scripted", "messages": [
        SYSTEM, {"role": "user", "content": task}, STATE,
        {"role": "user", "content": followup}, *EXCHANGE,
        {"role": "user", "content": host_tail},
    ]}
    assert _send(body, scripted_provider) == json.dumps(expected).encode("utf-8")
