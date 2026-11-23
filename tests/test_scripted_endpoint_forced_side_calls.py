"""Unserved forced calls must leave scripted main-loop steps available."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest

from stateframe.scripted_endpoint import Script, make_handler, serve


SIDE_TOOL = "side_answer"


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _post(server, offered: list[str], *, forced: bool = False, stream: bool = False):
    body = {
        "model": "scripted",
        "messages": [{"role": "user", "content": "continue"}],
        "tools": [_tool(name) for name in offered],
        "stream": stream,
    }
    if forced:
        body["tool_choice"] = {
            "type": "function",
            "function": {"name": SIDE_TOOL},
        }
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read()
    else:
        status = response.status
        raw = response.read()
        response.close()
    return status, json.loads(raw)


def _assert_explicit_error(result) -> None:
    status, body = result
    assert status == 409
    assert isinstance(body.get("error"), dict)
    assert isinstance(body["error"].get("message"), str)
    assert body["error"]["message"]


def _assert_scripted_call(result, expected_tool: str, expected_arguments: dict) -> None:
    status, body = result
    assert status == 200
    call = body["choices"][0]["message"]["tool_calls"][0]["function"]
    assert call["name"] == expected_tool
    assert json.loads(call["arguments"]) == expected_arguments


@pytest.mark.parametrize("stream", [False, True])
def test_unserved_forced_call_between_steps_preserves_next_step(stream):
    script = Script({"steps": [
        {"tool": "read", "arguments": {"position": "before"}},
        {"tool": "write", "arguments": {"position": "after"}},
    ]})
    server = serve(script)
    try:
        _assert_scripted_call(_post(server, ["read", "write"], stream=False),
                              "read", {"position": "before"})

        _assert_explicit_error(
            _post(server, [SIDE_TOOL], forced=True, stream=stream))

        _assert_scripted_call(_post(server, ["read", "write"], stream=False),
                              "write", {"position": "after"})
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()


def test_several_unserved_forced_calls_do_not_consume_next_step():
    script = Script({"steps": [
        {"tool": "read", "arguments": {"position": "before"}},
        {"tool": "write", "arguments": {"position": "after"}},
    ]})
    server = serve(script)
    try:
        _assert_scripted_call(_post(server, ["read", "write"]),
                              "read", {"position": "before"})

        for _ in range(3):
            _assert_explicit_error(_post(server, [SIDE_TOOL], forced=True))

        _assert_scripted_call(_post(server, ["read", "write"]),
                              "write", {"position": "after"})
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()


def test_concurrent_unserved_forced_calls_do_not_consume_next_step():
    script = Script({"steps": [
        {"tool": "read", "arguments": {"position": "before"}},
        {"tool": "write", "arguments": {"position": "after"}},
    ]})
    server = serve(script)
    try:
        _assert_scripted_call(_post(server, ["read", "write"]),
                              "read", {"position": "before"})

        count = 6
        barrier = threading.Barrier(count)

        def request_side_call():
            barrier.wait()
            return _post(server, [SIDE_TOOL], forced=True)

        with ThreadPoolExecutor(max_workers=count) as pool:
            results = list(pool.map(lambda _: request_side_call(), range(count)))

        for result in results:
            _assert_explicit_error(result)
        _assert_scripted_call(_post(server, ["read", "write"]),
                              "write", {"position": "after"})
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()


def test_unserved_forced_call_after_script_end_returns_explicit_error():
    script = Script({"steps": [
        {"tool": "read", "arguments": {"position": "last"}},
    ]})
    server = serve(script)
    try:
        _assert_scripted_call(_post(server, ["read"]),
                              "read", {"position": "last"})

        _assert_explicit_error(_post(server, [SIDE_TOOL], forced=True))
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()


def test_served_forced_call_consumes_its_scripted_step():
    script = Script({"steps": [
        {"tool": "read", "arguments": {"position": "before"}},
        {"tool": SIDE_TOOL, "arguments": {"answer": "served"}},
        {"tool": "write", "arguments": {"position": "after"}},
    ]})
    server = serve(script)
    try:
        _assert_scripted_call(_post(server, ["read", SIDE_TOOL]),
                              "read", {"position": "before"})
        _assert_scripted_call(_post(server, [SIDE_TOOL], forced=True),
                              SIDE_TOOL, {"answer": "served"})
        _assert_scripted_call(_post(server, ["write"]),
                              "write", {"position": "after"})
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()


def _post_forced_in_memory(script: Script, name: str, offered: list[str]):
    body = json.dumps({
        "tools": [_tool(tool) for tool in offered],
        "tool_choice": {"type": "function", "function": {"name": name}},
    }).encode()
    request = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: localhost\r\nConnection: close\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
    )
    response = BytesIO()

    class Handler(make_handler(script)):
        def setup(self):
            self.rfile = BytesIO(request)
            self.wfile = response

        def finish(self):
            self.rfile.close()

    Handler(None, ("127.0.0.1", 0), None)
    headers, _, payload = response.getvalue().partition(b"\r\n\r\n")
    return int(headers.split()[1]), json.loads(payload)


@pytest.mark.parametrize("steps", [[], [{"tool": "read", "required": False}]])
def test_forced_done_keeps_the_script_completion_fallback(steps):
    script = Script({"steps": steps})

    result = _post_forced_in_memory(script, "done", ["done"])

    _assert_scripted_call(result, "done", {"summary": "scripted run finished"})
    assert script.position == len(steps)
    script.assert_consumed()


@pytest.mark.parametrize("offered_required_step", [False, True])
def test_unserved_forced_call_error_does_not_echo_request_or_script_text(offered_required_step):
    request_marker = "private request marker"
    script_marker = "private script marker"
    script = Script({"steps": [{"tool": script_marker}]})
    offered = [request_marker]
    if offered_required_step:
        offered.append(script_marker)

    result = _post_forced_in_memory(script, request_marker, offered)

    _assert_explicit_error(result)
    message = result[1]["error"]["message"]
    assert request_marker not in message
    assert script_marker not in message
    assert script.position == 0
