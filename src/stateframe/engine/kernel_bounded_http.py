"""Optional killable HTTP transport; no model or filesystem effects in the child.

The child keeps the existing non-streamed response assembly. The parent therefore
uses provider-origin health probes during the wait instead of a first-byte timeout.
"""
from __future__ import annotations

from collections.abc import Callable
import functools
import http.client
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from kernel_http_fallback import send_with_gzip_fallback

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_ERROR_BYTES = 8 * 1024
RELAY_ERROR_LABELS = frozenset({
    "upstream_reset", "upstream_timeout", "upstream_refused", "upstream_error",
    "downstream_disconnect", "relay_wall_cap", "relay_ttft_timeout",
})


DEFAULT_PROVIDER_LIVENESS_S = 90.0


class ProviderLivenessTimeout(TimeoutError):
    """No healthy provider-origin probe arrived before the liveness deadline."""


class BoundedTransportError(ConnectionError):
    """Safe child-process transport metadata, without response bodies or headers."""

    def __init__(self, *, error_type: str, http_status: int | None,
                 relay_error: str | None, server_error: str | None = None) -> None:
        suffix = f" HTTP {http_status}" if http_status is not None else ""
        super().__init__(f"bounded HTTP child failed: {error_type}{suffix}")
        self.error_type = error_type
        self.http_status = http_status
        self.relay_error = relay_error
        # Redacted, <= 2 KB provider error text (see safe_server_error).
        self.server_error = server_error


SERVER_ERROR_BYTES = 2048
# Server bodies that prove the REQUEST is wrong (deterministic, never retried).
_CONTRACT_MARKERS = (
    "json schema error", "failed to parse grammar", "invalid_request",
    "cannot set both", "invalid tool", "unsupported",
)
_CAPACITY_MARKERS = ("context size", "context length", "exceed_context", "n_ctx",
                     "exceeds the available context")
_SECRET_PATTERN = None


def safe_server_error(body: bytes | None, key: str = "") -> str | None:
    """First 2 KB of a provider error body, decoded, with credentials redacted."""
    if not body:
        return None
    global _SECRET_PATTERN
    if _SECRET_PATTERN is None:
        import re
        _SECRET_PATTERN = re.compile(
            r"(?i)(bearer\s+|authorization\W+|api[_-]?key\W+|token\W+)[A-Za-z0-9._\-]{6,}")
    text = body[:SERVER_ERROR_BYTES].decode("utf-8", errors="replace")
    if key and len(key) >= 4:
        text = text.replace(key, "[redacted]")
    return _SECRET_PATTERN.sub(lambda match: match.group(1) + "[redacted]", text)


def transport_class(http_status: int | None, server_error: str | None = None,
                    error_type: str | None = None) -> str:
    """``contract`` (request is wrong), ``capacity`` or ``upstream``; never model.

    contract: 400/422 and other 4xx, plus a 500 whose body names a schema/grammar
    defect. capacity: 413/503/429 or a context-overflow body. upstream: 5xx gateway
    failures and every status-less connection/timeout failure.
    """
    text = (server_error or "").lower()
    if any(marker in text for marker in _CAPACITY_MARKERS) or http_status in {413, 429, 503}:
        return "capacity"
    if http_status is None:
        return "upstream"
    if 400 <= http_status < 500:
        return "contract"
    if http_status == 500 and any(marker in text for marker in _CONTRACT_MARKERS):
        return "contract"
    return "upstream"


def _relay_error_label(body: bytes) -> str | None:
    """Accept only the relay's small, closed label vocabulary."""
    if len(body) > MAX_ERROR_BYTES:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    label = error.get("label") if isinstance(error, dict) else None
    return label if isinstance(label, str) and label in RELAY_ERROR_LABELS else None


def provider_liveness_timeout_seconds() -> float:
    """Return the allowed silence between healthy provider-origin probes."""
    raw = os.environ.get("STATEFRAME_PROVIDER_LIVENESS_S")
    if raw is None:
        return DEFAULT_PROVIDER_LIVENESS_S
    try:
        seconds = float(raw)
    except ValueError as error:
        raise ValueError(
            "STATEFRAME_PROVIDER_LIVENESS_S must be finite and positive") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("STATEFRAME_PROVIDER_LIVENESS_S must be finite and positive")
    return seconds


def _provider_probe_origin(endpoint: str) -> tuple[str, str, int | None] | None:
    """Validate a provider URL and return only its origin components."""
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            return parsed.scheme, parsed.hostname, parsed.port
    except ValueError:
        pass
    return None


def _close_probe_connection(connection: http.client.HTTPConnection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except OSError:
        pass


def _provider_probe_status(endpoint: str, path: str, key: str,
                           timeout: float) -> int | None:
    """Get one bounded provider-origin health response, returning none on failure."""
    origin = _provider_probe_origin(endpoint)
    if origin is None:
        return None
    scheme, host, port = origin
    connection_type = (http.client.HTTPSConnection if scheme == "https"
                       else http.client.HTTPConnection)
    connection = None
    try:
        connection = connection_type(host, port, timeout=timeout)
        headers = {"Accept": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        status = response.status
        response.close()
        return status
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        _close_probe_connection(connection)


def _probe_liveness(endpoint: str, key: str, timeout: float) -> bool:
    """Probe known provider health routes on the request's authenticated origin."""
    for path in ("/slots", "/health"):
        status = _provider_probe_status(endpoint, path, key, timeout)
        # Missing routes (404/405) do not prove the provider is alive behind a relay.
        if status is not None and 200 <= status < 300:
            return True
    return False



def _terminate_child(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
        process.communicate()


def _communicate_with_liveness(
        process: subprocess.Popen[bytes], packet: bytes, endpoint: str, key: str, *,
        deadline: float, liveness_timeout: float) -> tuple[bytes, bytes]:
    """Drain the child while checking the provider origin until either deadline."""
    probe_interval = min(10.0, max(0.01, liveness_timeout / 3))
    last_liveness = time.monotonic()
    next_probe = last_liveness + probe_interval
    communication_started = False
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            _terminate_child(process)
            raise TimeoutError("HTTP absolute deadline exceeded")
        wait_for = min(remaining, max(0.01, next_probe - now))
        try:
            if communication_started:
                return process.communicate(timeout=wait_for)
            communication_started = True
            return process.communicate(packet, timeout=wait_for)
        except subprocess.TimeoutExpired:
            communication_started = True
            now = time.monotonic()
            if now >= deadline:
                _terminate_child(process)
                raise TimeoutError("HTTP absolute deadline exceeded")
            if now < next_probe:
                continue
            probe_timeout = min(
                2.0, max(0.01, liveness_timeout / 4), max(0.01, deadline - now))
            alive = _probe_liveness(endpoint, key, probe_timeout)
            probed_at = time.monotonic()
            if alive:
                last_liveness = probed_at
            if probed_at - last_liveness >= liveness_timeout:
                _terminate_child(process)
                raise ProviderLivenessTimeout(
                    "provider-origin liveness lost before the total request deadline")
            next_probe = probed_at + probe_interval


def _child_transport_error(diagnostic: bytes, key: str) -> BoundedTransportError:
    try:
        failure = json.loads(diagnostic[:4 * SERVER_ERROR_BYTES + 1024])
    except (ValueError, UnicodeError):
        failure = {}
    kind = failure.get("error_type", "ChildProcessError")
    if not isinstance(kind, str) or not kind.isidentifier() or len(kind) > 80:
        kind = "ChildProcessError"
    status = failure.get("http_status")
    relay_error = failure.get("relay_error")
    if not isinstance(relay_error, str) or relay_error not in RELAY_ERROR_LABELS:
        relay_error = None
    server_error = failure.get("server_error")
    server_error = (safe_server_error(server_error.encode("utf-8"), key)
                    if isinstance(server_error, str) else None)
    return BoundedTransportError(
        error_type=kind,
        http_status=status if type(status) is int and 100 <= status <= 599 else None,
        relay_error=relay_error,
        server_error=server_error,
    )


def request_bytes(wire: bytes, endpoint: str, key: str, *,
                  deadline: float, socket_timeout: float,
                  liveness_timeout: float | None = None,
                  record_attempts: Callable[[list[dict[str, object]]], None] | None = None) -> bytes:
    """Bound the full HTTP operation and abort it after provider liveness is lost."""
    if not math.isfinite(deadline) or not math.isfinite(socket_timeout) or socket_timeout <= 0:
        raise ValueError("invalid HTTP deadline or socket timeout")
    if liveness_timeout is None:
        liveness_timeout = provider_liveness_timeout_seconds()
    if (isinstance(liveness_timeout, bool)
            or not isinstance(liveness_timeout, (int, float))
            or not math.isfinite(liveness_timeout) or liveness_timeout <= 0):
        raise ValueError("provider liveness timeout must be finite and positive")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("HTTP deadline expired before submission")
    packet = json.dumps({"wire": wire.decode("utf-8"), "endpoint": endpoint, "key": key,
                         "deadline": deadline,
                         "socket_timeout": min(socket_timeout, remaining),
                         "gzip": request_gzip_enabled()}).encode()
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--http-child"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        output, diagnostic = _communicate_with_liveness(
            process, packet, endpoint, key, deadline=deadline,
            liveness_timeout=float(liveness_timeout))
        if record_attempts is not None and diagnostic:
            try:
                attempts = json.loads(diagnostic).get("transport_attempts")
            except (ValueError, AttributeError):
                attempts = None
            if isinstance(attempts, list):
                record_attempts(attempts)
        if process.returncode != 0:
            raise _child_transport_error(diagnostic, key)
        return output
    finally:
        _terminate_child(process)


REQUEST_GZIP_ENV = "STATEFRAME_HTTP_GZIP"
REQUEST_GZIP_MIN_BYTES = 32 * 1024


def request_gzip_enabled(environ: dict[str, str] | None = None) -> bool:
    """``STATEFRAME_HTTP_GZIP`` off (default) | on; anything else fails closed."""
    mode = str((environ if environ is not None else os.environ).get(REQUEST_GZIP_ENV, "off")).strip().lower()
    if mode not in {"on", "off"}:
        raise ValueError(f"{REQUEST_GZIP_ENV} must be 'on' or 'off'")
    return mode == "on"


def encode_request(wire: bytes, *, gzip_enabled: bool) -> tuple[bytes, dict[str, str]]:
    """Identity wire -> (HTTP body, extra headers).

    10-13: the container -> socat sidecar -> host path truncates request bodies at ~149,326 bytes,
    and the relay -> provider leg zero-fills random bytes. Large bodies are gzipped (the lab relay
    inflates them) and every body carries X-Request-Body-SHA256 over the IDENTITY bytes so the relay
    can refuse a corrupted upload. The wire identity used by proofs is unchanged.
    """
    import hashlib
    headers = {"X-Request-Body-SHA256": hashlib.sha256(wire).hexdigest()}
    if gzip_enabled and len(wire) >= REQUEST_GZIP_MIN_BYTES:
        import gzip
        return gzip.compress(wire, compresslevel=6, mtime=0), {"Content-Encoding": "gzip", **headers}
    return wire, headers


class _AbortiveConnectionTracker:
    """Keep child HTTP sockets reset-capable until the parent accepts a full response."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connections: list[_AbortiveConnectionMixin] = []
        self._aborted = False

    @property
    def aborted(self) -> bool:
        with self._lock:
            return self._aborted

    def register(self, connection: "_AbortiveConnectionMixin") -> None:
        with self._lock:
            self._connections.append(connection)
            aborted = self._aborted
        if aborted:
            connection.abort()

    def abort_all(self) -> None:
        with self._lock:
            self._aborted = True
            connections = tuple(self._connections)
        for connection in connections:
            connection.abort()

    def finish_all(self) -> None:
        with self._lock:
            connections = tuple(self._connections)
        for connection in connections:
            connection.finish()

    def set_timeout(self, seconds: float) -> None:
        with self._lock:
            connections = tuple(self._connections)
        for connection in connections:
            connection.set_timeout(seconds)


class _AbortiveConnectionMixin:
    def __init__(self, *args, tracker: _AbortiveConnectionTracker, **kwargs) -> None:
        self._tracker = tracker
        self._armed = True
        self._active_socket: socket.socket | None = None
        super().__init__(*args, **kwargs)
        tracker.register(self)

    def connect(self) -> None:
        if self._tracker.aborted:
            raise ConnectionError("HTTP request was aborted")
        super().connect()
        active_socket = self.sock
        if active_socket is not None:
            try:
                active_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            except OSError:
                super().close()
                raise
            self._active_socket = active_socket
        if self._tracker.aborted:
            self.abort()
            raise ConnectionError("HTTP request was aborted")

    def close(self) -> None:
        # urllib closes its HTTPConnection as soon as it sees Connection: close, before the
        # response body is consumed. Defer that close so a parent timeout can still reset the TCP peer.
        if not self._armed:
            super().close()

    def set_timeout(self, seconds: float) -> None:
        active_socket = self._active_socket or self.sock
        # A finished earlier attempt (the gzip 415 resend) leaves a closed socket registered.
        if active_socket is not None and active_socket.fileno() != -1:
            active_socket.settimeout(seconds)

    def abort(self) -> None:
        if not self._armed:
            return
        self._armed = False
        active_socket = self._active_socket or self.sock
        if active_socket is not None:
            try:
                active_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            except OSError:
                pass
            try:
                active_socket.shutdown(socket.SHUT_RD)
            except OSError:
                pass
        try:
            super().close()
        except OSError:
            pass

    def finish(self) -> None:
        if not self._armed:
            return
        self._armed = False
        active_socket = self._active_socket or self.sock
        if active_socket is not None:
            try:
                active_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 0, 0))
            except OSError:
                pass
        try:
            super().close()
        except OSError:
            pass


class _AbortiveHTTPConnection(_AbortiveConnectionMixin, http.client.HTTPConnection):
    pass


class _AbortiveHTTPSConnection(_AbortiveConnectionMixin, http.client.HTTPSConnection):
    pass


class _AbortiveHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, tracker: _AbortiveConnectionTracker) -> None:
        self._tracker = tracker
        super().__init__()

    def http_open(self, request):
        factory = functools.partial(_AbortiveHTTPConnection, tracker=self._tracker)
        return self.do_open(factory, request)


class _AbortiveHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, tracker: _AbortiveConnectionTracker) -> None:
        self._tracker = tracker
        super().__init__()

    def https_open(self, request):
        factory = functools.partial(_AbortiveHTTPSConnection, tracker=self._tracker)
        return self.do_open(
            factory, request, context=self._context, check_hostname=self._check_hostname)


def _read_response_body(
        response: object, tracker: _AbortiveConnectionTracker, *,
        deadline: float, socket_timeout: float, max_bytes: int) -> bytes:
    """Read at most max_bytes plus one byte under the request's absolute deadline."""
    reader = response
    read_once = getattr(response, "read1", None)
    if not callable(read_once):
        reader = getattr(response, "fp", None) or response
        read_once = getattr(reader, "read1", None)
    if not callable(read_once):
        read_once = getattr(reader, "read", None)
    if not callable(read_once):
        raise TypeError("bounded HTTP response has no readable body stream")

    data = bytearray()
    while len(data) <= max_bytes:
        if getattr(reader, "length", None) == 0:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("HTTP absolute deadline exceeded during response read")
        tracker.set_timeout(min(socket_timeout, remaining))
        read_size = min(64 * 1024, max_bytes + 1 - len(data))
        try:
            chunk = read_once(read_size)
        except (socket.timeout, TimeoutError) as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "HTTP absolute deadline exceeded during response read") from error
            raise
        if time.monotonic() >= deadline:
            raise TimeoutError("HTTP absolute deadline exceeded during response read")
        if not chunk:
            break
        data.extend(chunk)

    result = bytes(data)
    remaining_length = getattr(reader, "length", None)
    if len(result) <= max_bytes and remaining_length is not None and remaining_length > 0:
        raise http.client.IncompleteRead(result, len(result) + remaining_length)
    return result


def _child() -> None:
    packet = json.load(sys.stdin)
    wire = packet["wire"].encode()
    body, encoding = encode_request(wire, gzip_enabled=packet.get("gzip") is True)
    identity, identity_headers = encode_request(wire, gzip_enabled=False)
    tracker = _AbortiveConnectionTracker()
    opener = urllib.request.build_opener(
        _AbortiveHTTPHandler(tracker), _AbortiveHTTPSHandler(tracker))
    response_complete = False
    attempts: list[dict[str, object]] = []

    def send(data: bytes, headers: dict[str, str]) -> bytes:
        import hashlib
        request = urllib.request.Request(packet["endpoint"], data=data,
            headers={"Content-Type": "application/json", **headers,
                     "Authorization": "Bearer " + packet["key"]})
        attempt = {"content_encoding": headers.get("Content-Encoding"),
                   "body_bytes": len(data), "body_sha256": hashlib.sha256(data).hexdigest(),
                   "http_status": None}
        attempts.append(attempt)
        try:
            with opener.open(request, timeout=packet["socket_timeout"]) as response:
                attempt["http_status"] = response.status
                if response.length is not None and response.length > MAX_RESPONSE_BYTES:
                    raise ValueError("HTTP response exceeds bounded transport limit")
                result = _read_response_body(
                    response, tracker, deadline=packet["deadline"],
                    socket_timeout=packet["socket_timeout"], max_bytes=MAX_RESPONSE_BYTES)
                if len(result) > MAX_RESPONSE_BYTES:
                    raise ValueError("HTTP response exceeds bounded transport limit")
                tracker.finish_all()
            return result
        except urllib.error.HTTPError as error:
            attempt["http_status"] = error.code
            error_body = _read_response_body(
                error, tracker, deadline=packet["deadline"],
                socket_timeout=packet["socket_timeout"], max_bytes=MAX_ERROR_BYTES)
            error.relay_error = _relay_error_label(error_body)
            error.server_error = safe_server_error(error_body, packet["key"])
            raise

    def on_refused(_error: urllib.error.HTTPError) -> bool:
        if time.monotonic() >= packet["deadline"]:
            return False
        tracker.finish_all()
        return True

    try:
        data = send_with_gzip_fallback(
            send, body, encoding, identity, identity_headers, on_refused)
        response_complete = True
    except Exception as error:
        if len(attempts) > 1:
            error.transport_attempts = attempts
        raise
    finally:
        if not response_complete:
            tracker.abort_all()
    if len(attempts) > 1:
        sys.stderr.write(json.dumps({"transport_attempts": attempts}))
    sys.stdout.buffer.write(data)


if __name__ == "__main__":
    if sys.argv[1:] != ["--http-child"]:
        raise SystemExit("HTTP child invocation required")
    try:
        _child()
    except Exception as error:
        # No exception message, URL, or credentials cross this channel; only the
        # redacted, 2 KB-bounded provider error body (server_error) does.
        status = getattr(error, "code", None)
        relay_error = getattr(error, "relay_error", None)
        server_error = getattr(error, "server_error", None)
        sys.stderr.write(json.dumps({"error_type": type(error).__name__,
                                     "http_status": status if type(status) is int else None,
                                     "relay_error": relay_error
                                     if isinstance(relay_error, str)
                                     and relay_error in RELAY_ERROR_LABELS else None,
                                     "server_error": server_error
                                     if isinstance(server_error, str) else None,
                                     **({"transport_attempts": error.transport_attempts}
                                        if hasattr(error, "transport_attempts") else {})}))
        raise SystemExit(1) from None
