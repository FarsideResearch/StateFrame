"""Scripted OpenAI-compatible endpoint for parallel subframe integration tests."""
from __future__ import annotations

from collections import Counter
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import re
import threading
import time
from typing import TypeAlias


_SUBFRAME_MARKER = re.compile(r"\ATask: Subframe ([a-z][a-z0-9_]{0,23}):")
JSONValue: TypeAlias = type(None) | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
_LOG = logging.getLogger(__name__)


def _script_steps(steps: list[dict[str, object]]) -> list[dict[str, object]]:
    """Copy a script, marking steps after the first terminal ``done`` as optional.

    Once ``done`` is served the client run is over, so later steps can never be
    consumed. Copies keep the caller's dicts unmodified.
    """
    copied: list[dict[str, object]] = []
    terminal_seen = False
    for step in steps:
        item = dict(step)
        if terminal_seen:
            item["required"] = False
        copied.append(item)
        if item.get("tool") == "done":
            terminal_seen = True
    return copied


def _usage(body: dict[str, object]) -> dict[str, int]:
    size = len(json.dumps(body.get("messages") or []))
    prompt = max(1, size // 4)
    return {"prompt_tokens": prompt, "completion_tokens": 16, "total_tokens": prompt + 16}


class BarrierEndpoint:
    """Serve separate scripts for the parent and each marked subframe task.

    ``hold_until`` maps a subframe id to a sibling id. The first request from
    the key waits until a request from the sibling is in flight, making overlap
    observable from request timestamps instead of relying on elapsed-time guesses.
    """

    def __init__(self, parent_steps: list[dict[str, object]],
                 subframe_steps: dict[str, list[dict[str, object]]], *,
                 slots: int = 16, hold_until: dict[str, str] | None = None,
                 hold_until_all: dict[str, tuple[str, ...]] | None = None,
                 hold_subframe_request: dict[str, int] | None = None,
                 hold_parent_request: int | None = None,
                 barrier_timeout: float = 30.0) -> None:
        if type(slots) is not int or slots < 1:
            raise ValueError("slots must be a positive integer")
        if barrier_timeout <= 0:
            raise ValueError("barrier_timeout must be positive")
        if hold_parent_request is not None and hold_parent_request < 1:
            raise ValueError("hold_parent_request must be positive")
        self.parent_steps = _script_steps(parent_steps)
        self.subframe_steps = {key: _script_steps(value) for key, value in subframe_steps.items()}
        self.hold_until = dict(hold_until or {})
        self.hold_until_all = {key: tuple(value) for key, value in (hold_until_all or {}).items()}
        self.hold_subframe_request = dict(hold_subframe_request or {})
        if any(type(count) is not int or count < 1 for count in self.hold_subframe_request.values()):
            raise ValueError("held subframe request numbers must be positive integers")
        self.hold_parent_request = hold_parent_request
        self._parent_released = False
        self._released_subframes: set[str] = set()
        self.slots = slots
        self.barrier_timeout = barrier_timeout
        self._positions: Counter[str] = Counter()
        self._arrivals: Counter[str] = Counter()
        self._requests: list[dict[str, object]] = []
        self._errors: list[str] = []
        self._active = 0
        self._peak_active = 0
        self._condition = threading.Condition()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("endpoint is not started")
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def start(self) -> "BarrierEndpoint":
        if self._server is not None:
            raise RuntimeError("endpoint is already started")
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        with self._condition:
            self._parent_released = True
            self._released_subframes.update(self.subframe_steps)
            self._condition.notify_all()
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("endpoint thread did not stop")

    def __enter__(self) -> "BarrierEndpoint":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def requests(self) -> tuple[dict[str, object], ...]:
        with self._condition:
            return tuple(dict(item) for item in self._requests)

    @property
    def errors(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(self._errors)

    @property
    def peak_active_requests(self) -> int:
        with self._condition:
            return self._peak_active

    def intervals(self, marker: str) -> tuple[tuple[float, float], ...]:
        with self._condition:
            return tuple((item["started"], item["finished"])
                         for item in self._requests
                         if item["marker"] == marker and item.get("finished") is not None)

    def assert_consumed(self) -> None:
        missing: list[str] = []
        with self._condition:
            for marker, steps in [("parent", self.parent_steps), *self.subframe_steps.items()]:
                position = self._positions[marker]
                for index, step in enumerate(steps[position:], start=position + 1):
                    if step.get("required", True):
                        missing.append(f"{marker} step {index} ({step.get('tool')!r})")
            errors = list(self._errors)
        if errors:
            raise AssertionError("endpoint errors: " + "; ".join(errors))
        if missing:
            raise AssertionError("required endpoint steps were not consumed: " + "; ".join(missing))

    def release_parent_request(self) -> None:
        with self._condition:
            self._parent_released = True
            self._condition.notify_all()

    def release_subframe(self, marker: str) -> None:
        with self._condition:
            self._released_subframes.add(marker)
            self._condition.notify_all()

    def wait_for_requests(self, marker: str, count: int = 1, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._arrivals[marker] < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    @staticmethod
    def _marker(body: dict[str, object]) -> str:
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise ValueError("request messages must be an array")
        for message in messages:
            if (not isinstance(message, dict) or message.get("role") != "user"
                    or not isinstance(message.get("content"), str)):
                continue
            match = _SUBFRAME_MARKER.match(message["content"])
            if match:
                return match.group(1)
        tools = body.get("tools")
        if isinstance(tools, list) and any(
                isinstance(tool, dict) and isinstance(tool.get("function"), dict)
                and tool["function"].get("name") == "unit_blocked" for tool in tools):
            raise ValueError("cannot identify subframe task from Task: Subframe <id>:")
        return "parent"

    def _step(self, marker: str, offered: list[str]) -> dict[str, object]:
        with self._condition:
            steps = self.parent_steps if marker == "parent" else self.subframe_steps.get(marker)
            if steps is None:
                raise ValueError(f"no script is configured for subframe {marker!r}")
            while self._positions[marker] < len(steps):
                index = self._positions[marker]
                step = steps[index]
                self._positions[marker] += 1
                tool = step.get("tool")
                if not isinstance(tool, str) or not tool:
                    raise ValueError(f"{marker} script step {index + 1} has no tool")
                if tool in offered or step.get("force") is True:
                    return step
                if step.get("required", True):
                    raise ValueError(f"required {marker} script step {index + 1} ({tool!r}) is not offered")
            return {"tool": "done", "arguments": {"summary": "scripted subframe run finished"}}

    def _arrive_and_wait(self, marker: str) -> tuple[int, float]:
        started = time.monotonic()
        with self._condition:
            self._arrivals[marker] += 1
            self._active += 1
            self._peak_active = max(self._peak_active, self._active)
            self._condition.notify_all()
            sibling = self.hold_until.get(marker)
            required = self.hold_until_all.get(marker, ())
            deadline = started + self.barrier_timeout
            try:
                if sibling is not None:
                    while self._arrivals[sibling] < 1 and marker not in self._released_subframes:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f"{marker} timed out waiting for subframe {sibling}")
                        self._condition.wait(remaining)
                while (required and any(self._arrivals[item] < 1 for item in required)
                       and marker not in self._released_subframes):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"{marker} timed out waiting for subframe group {required!r}")
                    self._condition.wait(remaining)
                if (marker != "parent"
                        and self._arrivals[marker] == self.hold_subframe_request.get(marker)
                        and marker not in self._released_subframes):
                    while marker not in self._released_subframes:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f"{marker} request timed out at the configured hold")
                        self._condition.wait(remaining)
                if (marker == "parent" and self.hold_parent_request is not None
                        and self._arrivals[marker] == self.hold_parent_request):
                    while not self._parent_released:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("parent request timed out at the configured hold")
                        self._condition.wait(remaining)
            except TimeoutError:
                self._active -= 1
                self._condition.notify_all()
                raise
            return self._active, started

    def _depart(self, record: dict[str, object]) -> None:
        with self._condition:
            record["finished"] = time.monotonic()
            self._active -= 1
            self._condition.notify_all()

    def _handler(self):
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                return

            def _send(self, status: int, payload: JSONValue) -> None:
                raw = json.dumps(payload).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError) as error:
                    # The client gave up before reading the response; the script
                    # step was already consumed, so this is not a script failure.
                    self.close_connection = True
                    _LOG.warning("client disconnected during endpoint response: %s", type(error).__name__)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/") == "/v1/models":
                    self._send(200, {"object": "list", "data": [{"id": "scripted", "object": "model"}]})
                elif self.path.rstrip("/") == "/slots":
                    self._send(200, [{"id": index, "n_ctx": 65536, "is_processing": False}
                                     for index in range(endpoint.slots)])
                else:
                    self._send(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    raw = self.rfile.read(length)
                    if self.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    if self.path.rstrip("/").endswith("/tokenize"):
                        document = json.loads(raw)
                        content = str(document.get("content") or "")
                        self._send(200, {"tokens": list(range(max(1, len(content) // 4)))})
                        return
                    if self.path.rstrip("/") != "/v1/chat/completions":
                        self._send(404, {"error": {"message": "not found"}})
                        return
                    body = json.loads(raw)
                    if not isinstance(body, dict):
                        raise ValueError("request body must be an object")
                    marker = endpoint._marker(body)
                    record: dict[str, object] = {"marker": "parent" if marker == "parent" else marker,
                                                 "started": time.monotonic(), "request": body,
                                                 "raw_request": raw}
                    with endpoint._condition:
                        endpoint._requests.append(record)
                    active = False
                    try:
                        endpoint._arrive_and_wait(marker)
                        active = True
                        offered = [tool["function"]["name"]
                                   for tool in body.get("tools") or []
                                   if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
                                   and isinstance(tool["function"].get("name"), str)]
                        step = endpoint._step(marker, offered)
                    except (TimeoutError, TypeError, ValueError) as error:
                        with endpoint._condition:
                            endpoint._errors.append(str(error))
                        if active:
                            endpoint._depart(record)
                        self._send(409, {"error": {"message": str(error)}})
                        return
                    request_id = len(endpoint.requests)
                    call = {"id": f"call_{request_id}", "type": "function",
                            "function": {"name": step["tool"],
                                         "arguments": json.dumps(step.get("arguments") or {})}}
                    try:
                        self._send(200, {"id": f"chatcmpl-{request_id}", "object": "chat.completion",
                                         "model": "scripted", "choices": [{"index": 0,
                                         "finish_reason": "tool_calls", "message": {"role": "assistant",
                                         "content": "", "tool_calls": [call]}}], "usage": _usage(body)})
                    finally:
                        endpoint._depart(record)
                except (OSError, UnicodeError, ValueError, gzip.BadGzipFile) as error:
                    if isinstance(error, (BrokenPipeError, ConnectionResetError)):
                        return
                    with endpoint._condition:
                        endpoint._errors.append(f"request failed: {type(error).__name__}: {error}")
                    try:
                        self._send(400, {"error": {"message": "invalid request"}})
                    except OSError:
                        return

        return Handler
