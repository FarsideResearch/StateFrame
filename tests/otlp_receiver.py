"""Loopback OTLP/HTTP JSON collector and hostile response helpers for tests."""
from __future__ import annotations

import base64
import binascii
import copy
import dataclasses
import ipaddress
import json
import math
import re
import socket
import ssl
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from typing import TypeAlias
from urllib.parse import urlsplit


MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_REPLY_BYTES = 2 * 1024 * 1024
OVERSIZED_REPLY_BYTES = 1024 * 1024 + 1
MAX_CAPTURED_REQUESTS = 2048
_MAX_HANG_SECONDS = 2.0
_MAX_DRIP_SECONDS = 0.5
_MAX_DRIP_BODY_BYTES = 4096
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_UINT64_MAX = 2**64 - 1
_UINT32_MAX = 2**32 - 1
_LOWER_CAMEL = re.compile(r"^[a-z][A-Za-z0-9]*$")
_DECIMAL = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_HEX = re.compile(r"^[0-9a-fA-F]+$")


_STATUS_MODES = frozenset({
    "http_400", "http_401", "http_413", "http_429",
    "http_500", "http_502", "http_503", "http_504",
})
_RESPONSE_MODES = frozenset({
    "accept", "partial_success", "malformed_json", "empty_body",
    "redirect_same_origin", "redirect_cross_origin",
    "hang_before_headers", "hang_mid_body", "slow_drip",
    "connection_reset", "oversized_reply", "duplicate_delivery",
}) | _STATUS_MODES


@dataclass(frozen=True, slots=True)
class ResponseStep:
    """One scripted collector response. Delays and bodies are deliberately capped."""

    mode: str = "accept"
    retry_after: str | None = None
    target_url: str | None = None
    body: bytes | None = None
    delay_seconds: float = 0.1
    oversized_bytes: int = OVERSIZED_REPLY_BYTES

    def __post_init__(self) -> None:
        if self.mode not in _RESPONSE_MODES:
            raise ValueError("unknown OTLP receiver response mode")
        if self.retry_after is not None and (
            not isinstance(self.retry_after, str)
            or len(self.retry_after) > 128
            or "\r" in self.retry_after
            or "\n" in self.retry_after
        ):
            raise ValueError("Retry-After must be a short single-line string")
        if self.body is not None and not isinstance(self.body, bytes):
            raise ValueError("response body must be bytes")
        if self.body is not None and len(self.body) > MAX_REPLY_BYTES:
            raise ValueError("response body exceeds the test helper limit")
        if self.mode == "slow_drip" and self.body is not None and len(self.body) > _MAX_DRIP_BODY_BYTES:
            raise ValueError("slow drip body exceeds the test helper limit")
        if self.mode == "hang_mid_body" and self.body is not None and len(self.body) < 2:
            raise ValueError("mid-body hang needs at least two response bytes")
        if (
            isinstance(self.delay_seconds, bool)
            or not isinstance(self.delay_seconds, (int, float))
            or not math.isfinite(self.delay_seconds)
            or not 0 <= self.delay_seconds <= _MAX_DRIP_SECONDS
        ):
            raise ValueError("delay must be finite and between zero and half a second")
        if (
            type(self.oversized_bytes) is not int
            or not OVERSIZED_REPLY_BYTES <= self.oversized_bytes <= MAX_REPLY_BYTES
        ):
            raise ValueError("oversized reply length is outside the bounded test range")
        if self.mode == "redirect_cross_origin" and self.target_url is not None:
            _validate_loopback_url(self.target_url)


ResponseInput: TypeAlias = ResponseStep | str


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    """A bounded copy of one collector arrival, including raw bytes for leak scans."""

    arrival_order: int
    method: str
    path: str
    headers: Mapping[str, str]
    decoded_json: object | None
    raw_body: bytes
    decode_error: str | None = None
    duplicate_of: int | None = None


@dataclass(frozen=True, slots=True)
class DuplicateSpan:
    trace_id: str
    span_id: str
    first_arrival_order: int | None
    duplicate_arrival_order: int | None


class ClosedPort:
    """An ephemeral loopback port that is bound and closed before it is exposed."""

    def __init__(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._socket.bind(("127.0.0.1", 0))
            self.port = int(self._socket.getsockname()[1])
        finally:
            self._socket.close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/traces"

    @property
    def socket_closed(self) -> bool:
        return self._socket.fileno() == -1


class _TrackedHTTPServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], owner: OTLPReceiver):
        self.owner = owner
        super().__init__(address, handler)

    def process_request_thread(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        self.owner._handler_started()
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.owner._handler_finished()

    def handle_error(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        # Test traffic can intentionally reset or time out; never print request data.
        return


class OTLPReceiver:
    """Scriptable, loopback-only OTLP/HTTP JSON receiver with bounded teardown."""

    def __init__(
        self,
        responses: Sequence[ResponseInput] = (),
        *,
        cross_origin_url: str | None = None,
        hang_seconds: float = 0.75,
        port: int = 0,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if isinstance(responses, (str, bytes, bytearray)):
            raise ValueError("responses must be a sequence of response steps")
        if not isinstance(responses, Sequence):
            raise ValueError("responses must be a sequence of response steps")
        if (
            isinstance(hang_seconds, bool)
            or not isinstance(hang_seconds, (int, float))
            or not math.isfinite(hang_seconds)
            or not 0 <= hang_seconds <= _MAX_HANG_SECONDS
        ):
            raise ValueError("hang duration must be finite and no more than two seconds")
        if cross_origin_url is not None:
            _validate_loopback_url(cross_origin_url)

        self._responses = tuple(_response_step(value) for value in responses)
        self._cross_origin_url = cross_origin_url
        self._hang_seconds = float(hang_seconds)
        self._lock = threading.Lock()
        self._active = threading.Condition(self._lock)
        self._requests: list[RecordedRequest] = []
        self._network_request_count = 0
        self._arrival_order = 0
        self._active_handlers = 0
        self._started = False
        self._closed = False
        self._release = threading.Event()
        self._mode_entered = threading.Event()
        self._server = _TrackedHTTPServer(("127.0.0.1", port), self._handler(), self)
        self._ssl_context = ssl_context
        if ssl_context is not None:
            self._server.socket = ssl_context.wrap_socket(self._server.socket, server_side=True)
        try:
            self._closed_port = ClosedPort()
        except BaseException:
            self._server.server_close()
            raise
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        scheme = "https" if self._ssl_context is not None else "http"
        return f"{scheme}://127.0.0.1:{self._server.server_address[1]}"

    @property
    def refused_url(self) -> str:
        """A local endpoint that refuses connections because its socket is closed."""
        return self._closed_port.url

    @property
    def refused_socket_closed(self) -> bool:
        return self._closed_port.socket_closed

    @property
    def requests(self) -> tuple[RecordedRequest, ...]:
        with self._lock:
            return tuple(_copy_record(record) for record in self._requests)

    @property
    def active_handlers(self) -> int:
        with self._lock:
            return self._active_handlers

    @property
    def socket_closed(self) -> bool:
        return self._server.fileno() == -1

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def thread_stopped(self) -> bool:
        return self._thread is None or not self._thread.is_alive()

    def start(self) -> OTLPReceiver:
        if self._closed:
            raise RuntimeError("receiver is closed")
        if self._started:
            raise RuntimeError("receiver is already running")
        thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.01},
            name="otlp-test-receiver",
            daemon=False,
        )
        self._thread = thread
        thread.start()
        self._started = True
        return self

    def wait_for_scripted_mode(self, timeout: float = 1.0) -> bool:
        """Wait until the currently handled response mode reaches its delay point."""
        return self._mode_entered.wait(timeout)

    def release_hangs(self) -> None:
        """Release hanging and slow-drip handlers early for deterministic cleanup."""
        self._release.set()

    def wait_for_handlers(self, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._active:
            while self._active_handlers:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._active.wait(remaining)
            return True

    def close(self) -> None:
        if self._closed:
            return
        self._release.set()
        if self._started:
            self._server.shutdown()
        self._server.server_close()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("OTLP receiver serve thread did not stop")
        if not self.wait_for_handlers(timeout=2.0):
            raise RuntimeError("OTLP receiver request handlers did not stop")
        self._closed = True

    def __enter__(self) -> OTLPReceiver:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _handler_started(self) -> None:
        with self._active:
            self._active_handlers += 1
            self._active.notify_all()

    def _handler_finished(self) -> None:
        with self._active:
            self._active_handlers -= 1
            self._active.notify_all()

    def _capture(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        raw_body: bytes,
        decode_error: str | None,
    ) -> tuple[RecordedRequest, int]:
        decoded: object | None = None
        if raw_body and decode_error is None:
            try:
                decoded = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                decode_error = "invalid JSON"
        elif not raw_body and decode_error is None:
            decode_error = "empty body"
        with self._lock:
            self._network_request_count += 1
            self._arrival_order += 1
            order = self._arrival_order
            record = RecordedRequest(
                arrival_order=order,
                method=method,
                path=path,
                headers=MappingProxyType(dict(headers)),
                decoded_json=decoded,
                raw_body=bytes(raw_body),
                decode_error=decode_error,
            )
            if len(self._requests) < MAX_CAPTURED_REQUESTS:
                self._requests.append(record)
            return record, self._network_request_count

    def _duplicate_capture(self, record: RecordedRequest) -> None:
        with self._lock:
            if len(self._requests) >= MAX_CAPTURED_REQUESTS:
                return
            self._arrival_order += 1
            self._requests.append(dataclasses.replace(
                _copy_record(record),
                arrival_order=self._arrival_order,
                duplicate_of=record.arrival_order,
            ))

    def _next_step(self, network_request_number: int) -> ResponseStep:
        index = network_request_number - 1
        if index < len(self._responses):
            return self._responses[index]
        return ResponseStep()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                self.close_connection = True
                raw_body, read_error = self._read_body()
                headers = {key.lower(): value for key, value in self.headers.items()}
                record, network_request_number = receiver._capture("POST", self.path, headers, raw_body, read_error)
                step = receiver._next_step(network_request_number)
                if step.mode == "duplicate_delivery":
                    receiver._duplicate_capture(record)
                self._respond(step, record)

            def _read_body(self) -> tuple[bytes, str | None]:
                header = self.headers.get("Content-Length")
                if header is None:
                    return b"", "missing content length"
                try:
                    length = int(header)
                except ValueError:
                    return b"", "invalid content length"
                if length < 0:
                    return b"", "invalid content length"
                if length > MAX_REQUEST_BYTES:
                    self.close_connection = True
                    return self.rfile.read(MAX_REQUEST_BYTES), "request body exceeds capture limit"
                raw = self.rfile.read(length)
                if len(raw) != length:
                    self.close_connection = True
                    return raw, "incomplete request body"
                return raw, None

            def _respond(self, step: ResponseStep, record: RecordedRequest) -> None:
                mode = step.mode
                if mode == "connection_reset":
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    try:
                        self.connection.close()
                    except OSError:
                        pass
                    return
                if mode == "hang_before_headers":
                    receiver._mode_entered.set()
                    receiver._release.wait(receiver._hang_seconds)
                    self._send(200, b"{}")
                    return
                if mode == "hang_mid_body":
                    self._send_in_parts(step.body or b'{"accepted":true}', pause=True)
                    return
                if mode == "slow_drip":
                    self._send_in_parts(step.body or b"{}", pause=False, delay=step.delay_seconds)
                    return
                if mode == "oversized_reply":
                    self._send(200, b"x" * step.oversized_bytes)
                    return
                if mode in _STATUS_MODES:
                    body = b'{"error":"collector rejected request"}'
                    self._send(int(mode[5:]), body, retry_after=step.retry_after)
                    return
                if mode == "partial_success":
                    body = json.dumps(
                        {"partialSuccess": {"rejectedSpans": "1", "errorMessage": "rejected"}},
                        separators=(",", ":"),
                    ).encode("ascii")
                    self._send(200, body)
                    return
                if mode == "malformed_json":
                    self._send(200, b'{"partialSuccess":')
                    return
                if mode == "empty_body":
                    self._send(200, b"")
                    return
                if mode == "redirect_same_origin":
                    self._send(307, b"", location=f"{receiver.url}/v1/redirected")
                    return
                if mode == "redirect_cross_origin":
                    target = step.target_url or receiver._cross_origin_url
                    if target is None or _origin(target) == _origin(receiver.url):
                        self._send(500, b'{"error":"cross-origin target unavailable"}')
                    else:
                        self._send(307, b"", location=target)
                    return
                self._send(200, step.body if step.body is not None else b"{}")

            def _send_in_parts(self, body: bytes, *, pause: bool, delay: float = 0.0) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                first = max(1, len(body) // 2) if pause else 1
                try:
                    self.wfile.write(body[:first])
                    self.wfile.flush()
                    receiver._mode_entered.set()
                    if pause and len(body) > first:
                        receiver._release.wait(receiver._hang_seconds)
                        self.wfile.write(body[first:])
                        self.wfile.flush()
                        return
                    for index in range(first, len(body)):
                        if delay and receiver._release.wait(delay):
                            self.wfile.write(body[index:])
                            break
                        self.wfile.write(body[index:index + 1])
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            def _send(
                self,
                status: int,
                body: bytes,
                *,
                retry_after: str | None = None,
                location: str | None = None,
            ) -> None:
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    if retry_after is not None:
                        self.send_header("Retry-After", retry_after)
                    if location is not None:
                        self.send_header("Location", location)
                    self.end_headers()
                    if body:
                        self.wfile.write(body)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        return Handler


def _response_step(value: ResponseInput) -> ResponseStep:
    if isinstance(value, ResponseStep):
        return value
    if not isinstance(value, str):
        raise ValueError("each response must be a mode name or ResponseStep")
    return ResponseStep(value)


def _copy_record(record: RecordedRequest) -> RecordedRequest:
    return dataclasses.replace(
        record,
        headers=MappingProxyType(dict(record.headers)),
        decoded_json=copy.deepcopy(record.decoded_json),
        raw_body=bytes(record.raw_body),
    )


def _validate_loopback_url(url: str) -> None:
    if not isinstance(url, str):
        raise ValueError("redirect URL must be a loopback HTTP URL")
    parsed = urlsplit(url)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
    except ValueError:
        if (parsed.hostname or "").lower() != "localhost":
            raise ValueError("redirect URL must be a loopback HTTP URL")
    else:
        if not address.is_loopback:
            raise ValueError("redirect URL must be a loopback HTTP URL")
    if parsed.scheme != "http" or parsed.port is None or parsed.username or parsed.password:
        raise ValueError("redirect URL must be a loopback HTTP URL")


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port


class _JSONMappingValidator:
    """Strict test-side OTLP/JSON schema checks with value-free diagnostics."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def obj(
        self,
        value: object,
        path: str,
        allowed: Iterable[str],
        required: Iterable[str] = (),
    ) -> Mapping[str, object] | None:
        if not isinstance(value, Mapping):
            self.errors.append(f"{path}: expected object")
            return None
        allowed_set = frozenset(allowed)
        for key in value:
            if not isinstance(key, str):
                self.errors.append(f"{path}: object key must be a string")
                continue
            if not _LOWER_CAMEL.fullmatch(key):
                self.errors.append(f"{path}: key is not lowerCamelCase")
            if key not in allowed_set:
                self.errors.append(f"{path}: unknown key")
        for key in required:
            if key not in value:
                self.errors.append(f"{path}: missing required field")
        return value

    def array(self, value: object, path: str) -> Sequence[object] | None:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            self.errors.append(f"{path}: expected array")
            return None
        return value

    def string(self, value: object, path: str) -> None:
        if not isinstance(value, str):
            self.errors.append(f"{path}: expected string")

    def integer(self, value: object, path: str, minimum: int, maximum: int) -> None:
        if type(value) is not int or not minimum <= value <= maximum:
            self.errors.append(f"{path}: expected bounded integer")

    def number(self, value: object, path: str, minimum: float | None = None, maximum: float | None = None) -> None:
        try:
            finite = math.isfinite(value) if not isinstance(value, bool) and isinstance(value, (int, float)) else False
        except OverflowError:
            finite = False
        if not finite:
            self.errors.append(f"{path}: expected finite number")
            return
        if minimum is not None and value < minimum:
            self.errors.append(f"{path}: number below minimum")
        if maximum is not None and value > maximum:
            self.errors.append(f"{path}: number above maximum")

    def int_string(self, value: object, path: str, *, unsigned: bool = False) -> int | None:
        if not isinstance(value, str) or not _DECIMAL.fullmatch(value):
            self.errors.append(f"{path}: int64 must be a decimal string")
            return None
        try:
            parsed = int(value)
        except ValueError:
            self.errors.append(f"{path}: int64 string is out of range")
            return None
        lower = 0 if unsigned else _INT64_MIN
        upper = _UINT64_MAX if unsigned else _INT64_MAX
        if not lower <= parsed <= upper:
            self.errors.append(f"{path}: int64 string is out of range")
            return None
        return parsed

    def identifier(self, value: object, path: str, digits: int, *, empty_allowed: bool = False) -> None:
        if empty_allowed and value == "":
            return
        if not isinstance(value, str) or len(value) != digits or not _HEX.fullmatch(value):
            self.errors.append(f"{path}: identifier must be {digits} hexadecimal characters")
            return
        if not any(char != "0" for char in value):
            self.errors.append(f"{path}: identifier must be non-zero")

    def key_values(self, value: object, path: str) -> None:
        items = self.array(value, path)
        if items is None:
            return
        for index, item in enumerate(items):
            item_path = f"{path}[{index}]"
            obj = self.obj(item, item_path, ("key", "value"), ("key", "value"))
            if obj is None:
                continue
            self.string(obj.get("key"), f"{item_path}.key")
            if "value" in obj:
                self.any_value(obj["value"], f"{item_path}.value")

    def any_value(self, value: object, path: str) -> None:
        fields = ("stringValue", "boolValue", "intValue", "doubleValue", "arrayValue", "kvlistValue", "bytesValue")
        obj = self.obj(value, path, fields)
        if obj is None:
            return
        present = [field for field in fields if field in obj]
        if len(present) != 1:
            self.errors.append(f"{path}: AnyValue must contain exactly one value")
            return
        field = present[0]
        item = obj[field]
        if field == "stringValue":
            self.string(item, f"{path}.{field}")
        elif field == "boolValue":
            if type(item) is not bool:
                self.errors.append(f"{path}.{field}: expected boolean")
        elif field == "intValue":
            self.int_string(item, f"{path}.{field}")
        elif field == "doubleValue":
            self.number(item, f"{path}.{field}")
        elif field == "bytesValue":
            self.string(item, f"{path}.{field}")
            if isinstance(item, str):
                try:
                    base64.b64decode(item, validate=True)
                except (ValueError, binascii.Error):
                    self.errors.append(f"{path}.{field}: expected base64 string")
        elif field == "arrayValue":
            array_obj = self.obj(item, f"{path}.{field}", ("values",), ("values",))
            if array_obj is not None:
                values = self.array(array_obj.get("values"), f"{path}.{field}.values")
                if values is not None:
                    for index, nested in enumerate(values):
                        self.any_value(nested, f"{path}.{field}.values[{index}]")
        elif field == "kvlistValue":
            list_obj = self.obj(item, f"{path}.{field}", ("values",), ("values",))
            if list_obj is not None:
                self.key_values(list_obj.get("values"), f"{path}.{field}.values")

    def resource(self, value: object, path: str) -> None:
        obj = self.obj(value, path, ("attributes", "droppedAttributesCount"))
        if obj is None:
            return
        if "attributes" in obj:
            self.key_values(obj["attributes"], f"{path}.attributes")
        if "droppedAttributesCount" in obj:
            self.integer(obj["droppedAttributesCount"], f"{path}.droppedAttributesCount", 0, _UINT32_MAX)

    def scope(self, value: object, path: str) -> None:
        obj = self.obj(value, path, ("name", "version", "attributes", "droppedAttributesCount"), ("name",))
        if obj is None:
            return
        self.string(obj.get("name"), f"{path}.name")
        for field in ("version",):
            if field in obj:
                self.string(obj[field], f"{path}.{field}")
        if "attributes" in obj:
            self.key_values(obj["attributes"], f"{path}.attributes")
        if "droppedAttributesCount" in obj:
            self.integer(obj["droppedAttributesCount"], f"{path}.droppedAttributesCount", 0, _UINT32_MAX)

    def trace_request(self, value: object) -> None:
        root = self.obj(value, "traceRequest", ("resourceSpans",), ("resourceSpans",))
        if root is None:
            return
        resources = self.array(root.get("resourceSpans"), "resourceSpans")
        if resources is None:
            return
        for resource_index, resource_span in enumerate(resources):
            path = f"resourceSpans[{resource_index}]"
            obj = self.obj(resource_span, path, ("resource", "scopeSpans", "schemaUrl"), ("scopeSpans",))
            if obj is None:
                continue
            if "resource" in obj:
                self.resource(obj["resource"], f"{path}.resource")
            if "schemaUrl" in obj:
                self.string(obj["schemaUrl"], f"{path}.schemaUrl")
            scopes = self.array(obj.get("scopeSpans"), f"{path}.scopeSpans")
            if scopes is None:
                continue
            for scope_index, scope_span in enumerate(scopes):
                scope_path = f"{path}.scopeSpans[{scope_index}]"
                scope_obj = self.obj(scope_span, scope_path, ("scope", "spans", "schemaUrl"), ("spans",))
                if scope_obj is None:
                    continue
                if "scope" in scope_obj:
                    self.scope(scope_obj["scope"], f"{scope_path}.scope")
                if "schemaUrl" in scope_obj:
                    self.string(scope_obj["schemaUrl"], f"{scope_path}.schemaUrl")
                spans = self.array(scope_obj.get("spans"), f"{scope_path}.spans")
                if spans is None:
                    continue
                for span_index, span in enumerate(spans):
                    self.span(span, f"{scope_path}.spans[{span_index}]")

    def span(self, value: object, path: str) -> None:
        fields = (
            "traceId", "spanId", "parentSpanId", "traceState", "name", "kind",
            "startTimeUnixNano", "endTimeUnixNano", "attributes", "droppedAttributesCount",
            "events", "droppedEventsCount", "links", "droppedLinksCount", "status", "flags",
        )
        required = ("traceId", "spanId", "name", "kind", "startTimeUnixNano", "endTimeUnixNano")
        obj = self.obj(value, path, fields, required)
        if obj is None:
            return
        if "traceId" in obj:
            self.identifier(obj["traceId"], f"{path}.traceId", 32)
        if "spanId" in obj:
            self.identifier(obj["spanId"], f"{path}.spanId", 16)
        if "parentSpanId" in obj:
            self.identifier(obj["parentSpanId"], f"{path}.parentSpanId", 16, empty_allowed=True)
        for field in ("traceState", "name"):
            if field in obj:
                self.string(obj[field], f"{path}.{field}")
        if "kind" in obj:
            self.integer(obj["kind"], f"{path}.kind", 0, 5)
        for field in ("startTimeUnixNano", "endTimeUnixNano"):
            if field in obj:
                self.int_string(obj[field], f"{path}.{field}", unsigned=True)
        for field in ("attributes",):
            if field in obj:
                self.key_values(obj[field], f"{path}.{field}")
        for field in ("droppedAttributesCount", "droppedEventsCount", "droppedLinksCount", "flags"):
            if field in obj:
                self.integer(obj[field], f"{path}.{field}", 0, _UINT32_MAX)
        if "events" in obj:
            events = self.array(obj["events"], f"{path}.events")
            if events is not None:
                for index, event in enumerate(events):
                    event_path = f"{path}.events[{index}]"
                    event_obj = self.obj(event, event_path,
                                         ("timeUnixNano", "name", "attributes", "droppedAttributesCount"),
                                         ("timeUnixNano", "name"))
                    if event_obj is None:
                        continue
                    if "timeUnixNano" in event_obj:
                        self.int_string(event_obj["timeUnixNano"], f"{event_path}.timeUnixNano", unsigned=True)
                    if "name" in event_obj:
                        self.string(event_obj["name"], f"{event_path}.name")
                    if "attributes" in event_obj:
                        self.key_values(event_obj["attributes"], f"{event_path}.attributes")
                    if "droppedAttributesCount" in event_obj:
                        self.integer(
                            event_obj["droppedAttributesCount"],
                            f"{event_path}.droppedAttributesCount", 0, _UINT32_MAX,
                        )
        if "links" in obj:
            links = self.array(obj["links"], f"{path}.links")
            if links is not None:
                for index, link in enumerate(links):
                    link_path = f"{path}.links[{index}]"
                    link_fields = (
                        "traceId", "spanId", "traceState", "attributes", "droppedAttributesCount", "flags",
                    )
                    link_obj = self.obj(link, link_path, link_fields, ("traceId", "spanId"))
                    if link_obj is None:
                        continue
                    if "traceId" in link_obj:
                        self.identifier(link_obj["traceId"], f"{link_path}.traceId", 32)
                    if "spanId" in link_obj:
                        self.identifier(link_obj["spanId"], f"{link_path}.spanId", 16)
                    if "traceState" in link_obj:
                        self.string(link_obj["traceState"], f"{link_path}.traceState")
                    if "attributes" in link_obj:
                        self.key_values(link_obj["attributes"], f"{link_path}.attributes")
                    for field in ("droppedAttributesCount", "flags"):
                        if field in link_obj:
                            self.integer(link_obj[field], f"{link_path}.{field}", 0, _UINT32_MAX)
        if "status" in obj:
            status = self.obj(obj["status"], f"{path}.status", ("message", "code"))
            if status is not None:
                if "message" in status:
                    self.string(status["message"], f"{path}.status.message")
                if "code" in status:
                    self.integer(status["code"], f"{path}.status.code", 0, 2)


def validate_traces(payload: object) -> tuple[str, ...]:
    """Return value-free errors for a strict OTLP/HTTP JSON trace request."""
    validator = _JSONMappingValidator()
    try:
        validator.trace_request(payload)
    except RecursionError:
        return ("traceRequest: nesting exceeds validator depth",)
    return tuple(validator.errors)


def assert_valid_traces(payload: object) -> None:
    errors = validate_traces(payload)
    if errors:
        raise AssertionError("invalid OTLP trace JSON mapping: " + "; ".join(errors))


def planted_capture_leaks(
    captured: object,
    planted_values: Iterable[str | bytes],
    *,
    allowed_headers: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """Return planted-value indexes found anywhere in captured request data."""
    allowed = {key.lower(): value for key, value in (allowed_headers or {}).items()}
    haystack = "\n".join(_capture_strings(captured, allowed_headers=allowed))
    leaked: list[int] = []
    for index, value in enumerate(planted_values):
        if isinstance(value, bytes):
            raw = value
            text = value.decode("utf-8", "replace")
        elif isinstance(value, str):
            text = value
            raw = value.encode("utf-8")
        else:
            raise TypeError("planted values must be strings or bytes")
        if not raw:
            raise ValueError("planted values cannot be empty")
        variants = (
            text,
            json.dumps(text, ensure_ascii=True)[1:-1],
            base64.b64encode(raw).decode("ascii"),
            raw.hex(),
        )
        if any(candidate and candidate in haystack for candidate in variants):
            leaked.append(index)
    return tuple(leaked)


def assert_capture_clean(
    captured: object,
    planted_values: Iterable[str | bytes],
    *,
    allowed_headers: Mapping[str, str] | None = None,
) -> None:
    leaks = planted_capture_leaks(captured, planted_values, allowed_headers=allowed_headers)
    if leaks:
        raise AssertionError("captured request contains planted private data")


def _capture_strings(value: object, *, allowed_headers: Mapping[str, str] | None = None) -> list[str]:
    if isinstance(value, RecordedRequest):
        parts = [
            str(value.arrival_order), value.method, value.path,
            value.raw_body.decode("utf-8", "replace"), value.raw_body.hex(),
            value.decode_error or "", str(value.duplicate_of or ""),
        ]
        allowed = allowed_headers or {}
        parts.extend(f"{key}:{item}" for key, item in value.headers.items()
                     if allowed.get(key.lower()) != item)
        if value.decoded_json is not None:
            parts.extend(_capture_strings(value.decoded_json, allowed_headers=allowed_headers))
        return parts
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _capture_strings(dataclasses.asdict(value), allowed_headers=allowed_headers)
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, item in value.items():
            parts.extend(_capture_strings(key, allowed_headers=allowed_headers))
            parts.extend(_capture_strings(item, allowed_headers=allowed_headers))
        try:
            parts.append(json.dumps(value, ensure_ascii=True, sort_keys=True, default=_json_default))
        except (TypeError, ValueError, RecursionError):
            pass
        return parts
    if isinstance(value, (list, tuple, set, frozenset)):
        parts = []
        for item in value:
            parts.extend(_capture_strings(item, allowed_headers=allowed_headers))
        return parts
    if isinstance(value, bytes):
        return [value.decode("utf-8", "replace"), value.hex(), base64.b64encode(value).decode("ascii")]
    if isinstance(value, str):
        return [value, json.dumps(value, ensure_ascii=True)[1:-1]]
    if value is None or type(value) in (bool, int, float):
        return [str(value)]
    return [type(value).__name__]


def _json_default(value: object) -> object:
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    if isinstance(value, Mapping):
        return dict(value)
    return type(value).__name__


def find_duplicate_spans(captured: Iterable[object]) -> tuple[DuplicateSpan, ...]:
    """Find repeated trace/span IDs across captured requests or decoded OTLP payloads."""
    first_seen: dict[tuple[str, str], int | None] = {}
    duplicates: list[DuplicateSpan] = []
    for item in captured:
        payload = item.decoded_json if isinstance(item, RecordedRequest) else item
        arrival = item.arrival_order if isinstance(item, RecordedRequest) else None
        if not isinstance(payload, Mapping):
            continue
        resource_spans = payload.get("resourceSpans")
        if not isinstance(resource_spans, Sequence) or isinstance(resource_spans, (str, bytes)):
            continue
        for resource_span in resource_spans:
            if not isinstance(resource_span, Mapping):
                continue
            scope_spans = resource_span.get("scopeSpans")
            if not isinstance(scope_spans, Sequence) or isinstance(scope_spans, (str, bytes)):
                continue
            for scope_span in scope_spans:
                if not isinstance(scope_span, Mapping):
                    continue
                spans = scope_span.get("spans")
                if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
                    continue
                for span in spans:
                    if not isinstance(span, Mapping):
                        continue
                    trace_id, span_id = span.get("traceId"), span.get("spanId")
                    if not isinstance(trace_id, str) or not isinstance(span_id, str):
                        continue
                    key = trace_id.lower(), span_id.lower()
                    if key in first_seen:
                        duplicates.append(DuplicateSpan(key[0], key[1], first_seen[key], arrival))
                    else:
                        first_seen[key] = arrival
    return tuple(duplicates)


def assert_no_duplicate_spans(captured: Iterable[object]) -> None:
    if find_duplicate_spans(captured):
        raise AssertionError("duplicate span delivery detected")
