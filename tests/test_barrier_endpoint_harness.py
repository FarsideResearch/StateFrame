"""Harness contract for BarrierEndpoint: steps after a terminal done and client disconnects."""
from __future__ import annotations

from email.message import Message
import io
import json
import logging

import pytest

from fixtures.barrier_endpoint import BarrierEndpoint


def _body(task: str = "Task: fix tax") -> bytes:
    return json.dumps({"messages": [{"role": "user", "content": task}],
                       "tools": [{"function": {"name": "read"}}, {"function": {"name": "done"}}]}).encode()


class _DisconnectingWriter(io.BytesIO):
    """Raises the given error on the Nth write, like a peer that vanished mid-response."""

    def __init__(self, error: type[OSError], fail_on_write: int) -> None:
        super().__init__()
        self._error = error
        self._fail_on_write = fail_on_write
        self._writes = 0

    def write(self, value: bytes) -> int:
        self._writes += 1
        if self._writes == self._fail_on_write:
            raise self._error("client disconnected")
        return super().write(value)


def _handler(endpoint: BarrierEndpoint, method: str, error: type[OSError], fail_on_write: int):
    handler_type = endpoint._handler()
    handler = handler_type.__new__(handler_type)
    body = _body()
    handler.command = method
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{method} / HTTP/1.1"
    handler.close_connection = False
    handler.path = "/v1/models" if method == "GET" else "/v1/chat/completions"
    handler.rfile = io.BytesIO(body)
    handler.headers = Message()
    handler.headers["Content-Length"] = str(len(body))
    handler.wfile = _DisconnectingWriter(error, fail_on_write)
    return handler


# --- (1) steps after a terminal done are not required -------------------------

def test_steps_after_terminal_done_are_not_required_for_assert_consumed():
    endpoint = BarrierEndpoint([{"tool": "read"}, {"tool": "done"}, {"tool": "read"}], {})
    assert endpoint._step("parent", ["read", "done"])["tool"] == "read"
    assert endpoint._step("parent", ["read", "done"])["tool"] == "done"

    endpoint.assert_consumed()


def test_steps_after_terminal_done_are_not_required_for_subframes():
    endpoint = BarrierEndpoint([], {"fix_tax": [{"tool": "done"}, {"tool": "read"}]})
    assert endpoint._step("fix_tax", ["done", "read"])["tool"] == "done"

    endpoint.assert_consumed()


def test_required_step_before_terminal_done_is_still_reported():
    endpoint = BarrierEndpoint([{"tool": "read"}, {"tool": "done"}, {"tool": "read"}], {})

    with pytest.raises(AssertionError, match=r"parent step 1 \('read'\)"):
        endpoint.assert_consumed()


def test_terminal_done_itself_stays_required_until_consumed():
    endpoint = BarrierEndpoint([{"tool": "done"}, {"tool": "read"}], {})

    with pytest.raises(AssertionError, match=r"parent step 1 \('done'\)"):
        endpoint.assert_consumed()


def test_script_caller_dicts_are_not_mutated():
    steps = [{"tool": "done"}, {"tool": "read"}]
    BarrierEndpoint(steps, {})

    assert steps == [{"tool": "done"}, {"tool": "read"}]


# --- (2) client disconnect while writing a response is logged, not raised --

@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError])
@pytest.mark.parametrize("fail_on_write", [1, 2], ids=["headers", "body"])
def test_post_response_disconnect_is_logged_and_swallowed(error, fail_on_write, caplog):
    endpoint = BarrierEndpoint([{"tool": "done"}], {})
    handler = _handler(endpoint, "POST", error, fail_on_write)

    with caplog.at_level(logging.WARNING, logger="fixtures.barrier_endpoint"):
        handler.do_POST()

    assert error.__name__ in caplog.text
    assert handler.close_connection is True
    assert endpoint.errors == ()
    assert endpoint._active == 0


@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError])
def test_get_response_disconnect_is_logged_and_swallowed(error, caplog):
    endpoint = BarrierEndpoint([], {})
    handler = _handler(endpoint, "GET", error, 1)

    with caplog.at_level(logging.WARNING, logger="fixtures.barrier_endpoint"):
        handler.do_GET()

    assert error.__name__ in caplog.text
    assert handler.close_connection is True


def test_other_os_errors_on_response_write_still_propagate_from_get():
    endpoint = BarrierEndpoint([], {})
    handler = _handler(endpoint, "GET", OSError, 1)

    with pytest.raises(OSError, match="client disconnected"):
        handler.do_GET()
