"""Optional killable HTTP transport; no model or filesystem effects in the child.

The child keeps the existing non-streamed response assembly. The parent therefore
uses provider-origin health probes during the wait instead of a first-byte timeout.
"""
from __future__ import annotations

from collections.abc import Callable
import errno
import functools
import ipaddress
import http.client
import json
import math
import os
from pathlib import Path
import re
import select
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import TypeVar

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_ERROR_BYTES = 8 * 1024
RELAY_ERROR_LABELS = frozenset({
    "upstream_reset", "upstream_timeout", "upstream_refused", "upstream_error",
    "downstream_disconnect", "relay_wall_cap", "relay_ttft_timeout",
})


DEFAULT_PROVIDER_LIVENESS_S = 90.0
DEFAULT_QUEUE_WAIT_MAX_S = 1800.0
DEFAULT_SEND_STALL_S = 90.0


class ProviderLivenessTimeout(TimeoutError):
    """No provider-origin HTTP response arrived before the liveness deadline."""


class RequestBodyDeadlineTimeout(TimeoutError):
    """The child did not finish its upload within the request deadline."""

    def __init__(self) -> None:
        super().__init__("HTTP request deadline exceeded before body completion")


class SendStallTimeout(TimeoutError):
    """Request-body bytes stopped moving before the body was fully sent."""

    error_type = "SendStallTimeout"
    http_status = None
    relay_error = None

    def __init__(self, bytes_sent: int, bytes_total: int, seconds_stalled: float,
                 origin: str = "unknown") -> None:
        super().__init__("HTTP request body send stalled")
        self.bytes_sent = bytes_sent
        self.bytes_total = bytes_total
        self.seconds_stalled = seconds_stalled
        self.origin = origin


class BoundedTransportError(ConnectionError):
    """Safe child-process transport metadata, without response bodies or headers."""

    def __init__(self, *, error_type: str, http_status: int | None,
                 relay_error: str | None, server_error: str | None = None,
                 error_detail: str | None = None) -> None:
        suffix = f" HTTP {http_status}" if http_status is not None else ""
        detail_suffix = f": {error_detail}" if error_detail else ""
        super().__init__(f"bounded HTTP child failed: {error_type}{suffix}{detail_suffix}")
        self.error_type = error_type
        self.http_status = http_status
        self.relay_error = relay_error
        self.error_detail = error_detail
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


_DETAIL_URL = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>\"']+")
_DETAIL_USERINFO = re.compile(r"[^\s/@:]+:[^\s/@]+@[^\s/]+")
_DETAIL_QUERY = re.compile(r"\?[^\s<>\"']+")
_DETAIL_CREDENTIAL = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|token|key|secret|password)"
    r"(\s*[:=]\s*)(?:(?:bearer|basic|token)\s+)?[^\s,;]+")
_DETAIL_BEARER = re.compile(r"(?i)\bbearer\s+[^\s,;]+")


def safe_transport_detail(error: BaseException | str, key: str = "") -> str | None:
    """Return a short diagnostic without destinations or credentials.

    Called in the HTTP child before crossing the pipe and again in the parent
    with the endpoint key, which is deliberately unavailable to the child guard.
    """
    reason: object = error.reason if isinstance(error, urllib.error.URLError) else error
    if isinstance(reason, ssl.SSLCertVerificationError):
        reason = reason.verify_message or reason.reason or reason
    elif isinstance(reason, ssl.SSLError):
        reason = reason.reason or reason
    detail = str(reason)[:4096]
    detail = _DETAIL_URL.sub("[redacted URL]", detail)
    detail = _DETAIL_USERINFO.sub("[redacted userinfo]", detail)
    detail = _DETAIL_QUERY.sub("?[redacted query]", detail)
    detail = _DETAIL_CREDENTIAL.sub(
        lambda match: match.group(1) + match.group(2) + "[redacted]", detail)
    detail = _DETAIL_BEARER.sub("Bearer [redacted]", detail)
    if key and len(key) >= 4:
        detail = detail.replace(key, "[redacted]")
    detail = safe_server_error(detail.encode("utf-8", errors="replace"), key) or ""
    detail = " ".join(detail.split())[:300]
    return detail or None


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


def queue_wait_max_seconds() -> float:
    """Maximum total time for a silent, reachable provider request."""
    raw = os.environ.get("STATEFRAME_QUEUE_WAIT_MAX_S")
    if raw is None:
        return DEFAULT_QUEUE_WAIT_MAX_S
    try:
        seconds = float(raw)
    except ValueError as error:
        raise ValueError("STATEFRAME_QUEUE_WAIT_MAX_S must be finite and positive") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("STATEFRAME_QUEUE_WAIT_MAX_S must be finite and positive")
    return seconds


def send_stall_seconds() -> float:
    """Maximum silence between successful request-body socket writes."""
    raw = os.environ.get("STATEFRAME_SEND_STALL_S")
    if raw is None:
        return DEFAULT_SEND_STALL_S
    try:
        seconds = float(raw)
    except ValueError as error:
        raise ValueError("STATEFRAME_SEND_STALL_S must be between 10 and 600 seconds") from error
    if not math.isfinite(seconds) or not 10 <= seconds <= 600:
        raise ValueError("STATEFRAME_SEND_STALL_S must be between 10 and 600 seconds")
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


def _numeric_http_origin(origin: tuple[str, str, int | None] | None) -> bool:
    if origin is None or origin[0] != "http":
        return False
    try:
        ipaddress.ip_address(origin[1])
    except ValueError:
        return False
    return True


def _close_probe_connection(connection: http.client.HTTPConnection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except OSError:
        pass


def _provider_probe_status(origin: tuple[str, str, int | None], path: str,
                           key: str, timeout: float, read_ok: bool = False) -> tuple[int, bool] | None:
    """Probe in a disposable child; the parent enforces the whole-operation limit."""
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
        healthy = b"ok" in response.read(1024) if read_ok else False
        response.close()
        return status, healthy
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        _close_probe_connection(connection)


def _provider_probe_json_response(origin: tuple[str, str, int | None], path: str,
                                  key: str, timeout: float) -> list[object] | None:
    """Read one small JSON probe response without redirecting or leaking credentials."""
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
        raw = response.read(65537)
        response.close()
        if len(raw) > 65536:
            return [status, None]
        if status != 200:
            return [status, None]
        try:
            return [status, json.loads(raw)]
        except (UnicodeDecodeError, ValueError):
            # The endpoint answered, but its payload cannot prove a dialect.
            return [status, None]
    except (OSError, http.client.HTTPException):
        return None
    finally:
        _close_probe_connection(connection)


def bounded_json_probe(endpoint: str, key: str, path: str, *,
                       timeout: float = 2.0,
                       deadline: float | None = None) -> tuple[int, object] | None:
    """Probe the request origin in the existing killable child with a hard deadline."""
    origin = _provider_probe_origin(endpoint)
    if origin is None or path not in {"/props", "/slots"}:
        return None
    stop_at = time.monotonic() + timeout
    if deadline is not None:
        stop_at = min(stop_at, deadline)
    if stop_at <= time.monotonic():
        return None
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--probe-child"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    packet = json.dumps({"origin": origin, "paths": [path], "key": key,
                         "stop_at": stop_at, "json_body": True}).encode()
    try:
        remaining = stop_at - time.monotonic()
        if remaining <= 0:
            return None
        output, _ = process.communicate(packet, timeout=remaining)
        if process.returncode == 0 and output:
            result = json.loads(output)
            if isinstance(result, list) and len(result) == 2 and type(result[0]) is int:
                return result[0], result[1]
    except (subprocess.TimeoutExpired, ValueError):
        pass
    finally:
        _terminate_child(process)
    return None


def _direct_http_probe(origin: tuple[str, str, int | None], path: str,
                       key: str, stop_at: float, read_ok: bool) -> tuple[int, bool] | None:
    """Bound a numeric HTTP origin without process startup on short liveness windows."""
    _, host, port = origin
    try:
        address = ipaddress.ip_address(host)
        if "\r" in key or "\n" in key:
            return None
        authority = f"[{host}]" if address.version == 6 else host
        authority += f":{port}" if port is not None else ""
        headers = (f"GET {path} HTTP/1.1\r\nHost: {authority}\r\n"
                   "Accept: application/json\r\nConnection: close\r\n"
                   + (f"Authorization: Bearer {key}\r\n" if key else "")
                   + "\r\n").encode("latin-1")
    except (ValueError, UnicodeEncodeError):
        return None
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    target = (host, 80 if port is None else port)
    try:
        with socket.socket(family, socket.SOCK_STREAM) as connection:
            connection.setblocking(False)
            code = connection.connect_ex(target)
            if code not in (0, errno.EINPROGRESS, errno.EWOULDBLOCK):
                return None
            if code:
                remaining = stop_at - time.monotonic()
                if remaining <= 0 or not select.select([], [connection], [], remaining)[1]:
                    return None
                if connection.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR):
                    return None
            sent = 0
            while sent < len(headers):
                remaining = stop_at - time.monotonic()
                if remaining <= 0 or not select.select([], [connection], [], remaining)[1]:
                    return None
                try:
                    written = connection.send(headers[sent:])
                except BlockingIOError:
                    continue
                if written == 0:
                    return None
                sent += written
            data = bytearray()
            header_end = -1
            status = None
            while len(data) < 64 * 1024 + 1024:
                remaining = stop_at - time.monotonic()
                if remaining <= 0 or not select.select([connection], [], [], remaining)[0]:
                    return None
                try:
                    chunk = connection.recv(4096)
                except BlockingIOError:
                    continue
                if not chunk:
                    return (status, False) if status is not None else None
                data.extend(chunk)
                if status is None and b"\r\n" in data:
                    first_line = bytes(data).split(b"\r\n", 1)[0].split(b" ", 2)
                    if len(first_line) < 2 or not first_line[0].startswith(b"HTTP/"):
                        return None
                    try:
                        status = int(first_line[1])
                    except ValueError:
                        return None
                    if not 100 <= status <= 599:
                        return None
                    if not read_ok:
                        return status, False
                if status is not None and header_end < 0:
                    header_end = data.find(b"\r\n\r\n")
                if header_end >= 0 and b"ok" in data[header_end + 4:]:
                    return status, True
    except OSError:
        return None
    return None


def _bounded_probe(endpoint: str, key: str, stop_at: float, *,
                   paths: tuple[str, ...], read_ok: bool = False) -> tuple[int, bool] | None:
    """Kill a trickling probe at its absolute deadline, including DNS and body reads."""
    origin = _provider_probe_origin(endpoint)
    if origin is None or stop_at <= time.monotonic():
        return None
    if _numeric_http_origin(origin):
        for path in paths:
            if time.monotonic() >= stop_at:
                break
            result = _direct_http_probe(origin, path, key, stop_at, read_ok)
            if result is not None:
                return result
        return None
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--probe-child"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    packet = json.dumps({"origin": origin, "paths": paths, "key": key,
                         "stop_at": stop_at, "read_ok": read_ok}).encode()
    try:
        remaining = stop_at - time.monotonic()
        if remaining <= 0:
            return None
        output, _ = process.communicate(packet, timeout=remaining)
        if process.returncode == 0:
            result = json.loads(output)
            if (isinstance(result, list) and len(result) == 2
                    and type(result[0]) is int and type(result[1]) is bool):
                return result[0], result[1]
    except (subprocess.TimeoutExpired, ValueError):
        pass
    finally:
        _terminate_child(process)
    return None


def _probe_liveness(endpoint: str, key: str, timeout: float) -> int | None:
    """Any HTTP status from the request's origin proves the process is reachable."""
    result = _bounded_probe(endpoint, key, time.monotonic() + timeout,
                            paths=("/slots", "/health"))
    return result[0] if result is not None else None


def provider_health_ok(endpoint: str, key: str, stop_at: float) -> bool:
    """Check recovery health without following redirects or escaping the deadline."""
    result = _bounded_probe(endpoint, key, stop_at, paths=("/health",), read_ok=True)
    return result is not None and 200 <= result[0] < 300 and result[1]


def _safe_origin(endpoint: str) -> str:
    origin = _provider_probe_origin(endpoint)
    if origin is None:
        return "unknown"
    scheme, host, port = origin
    host = f"[{host}]" if ":" in host else host
    return f"{scheme}://{host}" + (f":{port}" if port is not None else "")



def _terminate_child(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
        process.communicate()


def _child_diagnostic(diagnostic: bytes) -> dict[str, object]:
    """The child may precede its final diagnostic with a flushed send marker."""
    try:
        detail = json.loads(diagnostic.rsplit(b"\n", 1)[-1])
    except (ValueError, UnicodeError):
        return {}
    return detail if isinstance(detail, dict) else {}


def _body_completion_state(
        diagnostic: bytes, attempt: int, completed_at: float | None,
) -> tuple[int, float | None]:
    """Use only the newest send attempt's timed completion marker."""
    for line in diagnostic.splitlines():
        try:
            marker = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if not isinstance(marker, dict) or type(marker.get("body_attempt")) is not int:
            continue
        marker_attempt = marker["body_attempt"]
        if marker_attempt < 1 or marker_attempt < attempt:
            continue
        if marker_attempt > attempt:
            attempt, completed_at = marker_attempt, None
        if marker.get("body_fully_sent") is False:
            completed_at = None
        elif marker.get("body_fully_sent") is True:
            value = marker.get("completed_at")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                completed_at = float(value)
    return attempt, completed_at


def _deliver_packet(process: subprocess.Popen[bytes], packet: bytes) -> threading.Event | None:
    """Write the child's packet on a thread with blocking writes; set the event once all of it is in the pipe.

    ``communicate(input=...)`` feeds stdin 512 bytes per select and only while the parent is
    inside ``communicate``, so a multi-MiB wire can take longer than the request deadline on a
    loaded host. Taking stdin away from the Popen object leaves ``communicate`` draining
    stdout and stderr only. Returns None for a process object without a stdin pipe.
    """
    stdin = getattr(process, "stdin", None)
    if stdin is None:
        return None
    process.stdin = None
    delivered = threading.Event()

    def write() -> None:
        try:
            stdin.write(packet)
            stdin.flush()
            delivered.set()
        except (OSError, ValueError):
            pass  # the child exited or was terminated; its own exit status reports why
        finally:
            try:
                stdin.close()
            except (OSError, ValueError):
                pass

    threading.Thread(target=write, name="bounded-http-packet", daemon=True).start()
    return delivered


def _communicate_with_liveness(
        process: subprocess.Popen[bytes], packet: bytes, endpoint: str, key: str, *,
        deadline: float, hard_deadline: float, liveness_timeout: float,
        probe_state: dict[str, int | float | None]) -> tuple[bytes, bytes]:
    """Extend a silent send only after the selected provider answers a probe."""
    probe_interval = min(10.0, max(0.01, liveness_timeout / 3))
    probe_floor = 0.01 if _numeric_http_origin(_provider_probe_origin(endpoint)) else 0.25
    last_liveness = time.monotonic()
    first_probe_margin = min(0.05, max(0.0, (deadline - last_liveness) / 3))
    next_probe = min(last_liveness + probe_interval,
                     max(last_liveness, deadline - first_probe_margin))
    communication_started = False
    delivered: threading.Event | None = None

    def record_markers(diagnostic: bytes | None) -> None:
        if diagnostic:
            attempt, completed_at = _body_completion_state(
                diagnostic, int(probe_state["body_attempt"] or 0),
                probe_state["completed_at"])
            probe_state["body_attempt"] = attempt
            probe_state["completed_at"] = completed_at

    while True:
        now = time.monotonic()
        remaining = hard_deadline - now
        if remaining <= 0:
            _terminate_child(process)
            raise TimeoutError("HTTP absolute deadline exceeded")
        if now >= deadline and probe_state["status"] is None:
            _terminate_child(process)
            raise TimeoutError("HTTP request deadline exceeded before body completion or provider liveness")
        if now >= deadline and delivered is not None and not delivered.is_set():
            # The child cannot have sent a body it has not yet received: no verdict to wait for.
            _terminate_child(process)
            raise TimeoutError("HTTP request deadline exceeded before body completion")
        if now >= deadline and communication_started:
            # A synchronous probe can span the cap. Drain what is available, but
            # let the child decide whether its final socket send beat the cap.
            try:
                output, diagnostic = process.communicate(timeout=min(0.01, remaining))
            except subprocess.TimeoutExpired as error:
                record_markers(error.stderr)
            else:
                record_markers(diagnostic)
                if (process.returncode == 0 and (probe_state["completed_at"] is None
                                                or probe_state["completed_at"] > deadline)):
                    raise TimeoutError("HTTP request deadline exceeded before body completion")
                return output, diagnostic
        completed_at = probe_state["completed_at"]
        body_sent_before_cap = isinstance(completed_at, float) and completed_at <= deadline
        wait_for = min(remaining, max(0.01, next_probe - now))
        if (probe_state["status"] is None or not body_sent_before_cap) and now < deadline:
            wait_for = min(wait_for, deadline - now)
        try:
            if communication_started:
                output, diagnostic = process.communicate(timeout=wait_for)
            else:
                communication_started = True
                delivered = _deliver_packet(process, packet)
                output, diagnostic = process.communicate(
                    None if delivered is not None else packet, timeout=wait_for)
            record_markers(diagnostic)
            return output, diagnostic
        except subprocess.TimeoutExpired as error:
            communication_started = True
            record_markers(error.stderr)
            now = time.monotonic()
            if now >= hard_deadline:
                _terminate_child(process)
                raise TimeoutError("HTTP absolute deadline exceeded")
            if now < next_probe:
                continue
            probe_timeout = min(
                2.0, max(probe_floor, liveness_timeout / 4),
                hard_deadline - now)
            status = _probe_liveness(endpoint, key, probe_timeout)
            probed_at = time.monotonic()
            if status is not None:
                probe_state["status"] = status
                last_liveness = probed_at
            if probed_at - last_liveness >= liveness_timeout:
                _terminate_child(process)
                raise ProviderLivenessTimeout(
                    "provider-origin liveness lost before the total request deadline")
            next_probe = probed_at + probe_interval


def _child_transport_error(diagnostic: bytes, key: str, endpoint: str) -> Exception:
    failure = _child_diagnostic(diagnostic)
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
    error_detail = failure.get("error_detail")
    error_detail = (safe_transport_detail(error_detail, key)
                    if isinstance(error_detail, str) else None)
    if kind == "SendStallTimeout":
        sent = failure.get("bytes_sent")
        total = failure.get("bytes_total")
        stalled = failure.get("seconds_stalled")
        if (type(sent) is int and type(total) is int and 0 <= sent < total <= 256 * 1024 * 1024
                and isinstance(stalled, (int, float)) and math.isfinite(stalled)
                and stalled >= 0):
            return SendStallTimeout(sent, total, float(stalled), _safe_origin(endpoint))
    if kind == "RequestBodyDeadlineTimeout":
        return TimeoutError("HTTP request deadline exceeded before body completion")
    if kind == "TimeoutError":
        # The child's own absolute deadline can fire just before the parent's: same typed error
        # either way (kernel_measurement classifies both by error_type "TimeoutError").
        return TimeoutError("HTTP absolute deadline exceeded")
    return BoundedTransportError(
        error_type=kind,
        http_status=status if type(status) is int and 100 <= status <= 599 else None,
        relay_error=relay_error,
        server_error=server_error,
        error_detail=error_detail,
    )


def _require_routed_local(url: str, allow_remote: bool) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("routed HTTP destination is invalid")
    try:
        local = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        local = parsed.hostname.lower() == "localhost"
    if not local and not allow_remote and os.environ.get("STATEFRAME_ALLOW_REMOTE_ENDPOINT") != "1":
        raise ValueError("routed HTTP destination is not on this machine")


class _RoutedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_remote: bool) -> None:
        self.allow_remote = allow_remote

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_routed_local(newurl, self.allow_remote)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def request_bytes(wire: bytes, endpoint: str, key: str, *,
                   deadline: float, socket_timeout: float,
                   run_deadline: float | None = None,
                   liveness_timeout: float | None = None,
                  route_meta: dict[str, object] | None = None,
                  route_allow_remote: bool = False,
                   record_attempts: Callable[[list[dict[str, object]]], None] | None = None,
                   record_queue_wait: Callable[[float, int, str], None] | None = None) -> bytes:
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
    if route_meta is not None:
        _require_routed_local(endpoint, route_allow_remote)
    hard_deadline = deadline
    if run_deadline is not None:
        if not math.isfinite(run_deadline):
            raise ValueError("invalid run deadline")
        hard_deadline = min(run_deadline,
                            time.monotonic() + queue_wait_max_seconds())
    if hard_deadline <= time.monotonic():
        raise TimeoutError("HTTP run deadline expired before submission")
    packet = json.dumps({"wire": wire.decode("utf-8"), "endpoint": endpoint, "key": key,
                         "deadline": hard_deadline,
                         "request_deadline": deadline,
                         "socket_timeout": min(socket_timeout, remaining),
                         "send_stall_seconds": send_stall_seconds(),
                         "report_body_sent": True,
                         "gzip": request_gzip_enabled(),
                         **({"route_allow_remote": route_allow_remote} if route_meta is not None else {})}).encode()
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--http-child"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    probe_state: dict[str, int | float | None] = {
        "status": None, "body_attempt": 0, "completed_at": None}
    diagnostic = b""
    try:
        output, diagnostic = _communicate_with_liveness(
            process, packet, endpoint, key, deadline=deadline,
            hard_deadline=hard_deadline, liveness_timeout=float(liveness_timeout),
            probe_state=probe_state)
        if record_attempts is not None and diagnostic:
            try:
                attempts = _child_diagnostic(diagnostic).get("transport_attempts")
            except (ValueError, AttributeError):
                attempts = None
            if isinstance(attempts, list):
                record_attempts(attempts)
        if process.returncode != 0:
            raise _child_transport_error(diagnostic, key, endpoint)
        return output
    finally:
        try:
            if (record_queue_wait is not None and probe_state["status"] is not None
                    and isinstance(probe_state["completed_at"], float)
                    and probe_state["completed_at"] <= deadline):
                detail = _child_diagnostic(diagnostic)
                first_response_at = detail.get("first_response_at") if isinstance(detail, dict) else None
                finished_at = (first_response_at if isinstance(first_response_at, (int, float))
                               and math.isfinite(first_response_at) else time.monotonic())
                waited = max(0.0, min(finished_at, hard_deadline) - deadline)
                if waited > 0:
                    record_queue_wait(waited, probe_state["status"], _safe_origin(endpoint))
        finally:
            _terminate_child(process)


REQUEST_GZIP_ENV = "STATEFRAME_HTTP_GZIP"
REQUEST_GZIP_MIN_BYTES = 32 * 1024
_Response = TypeVar("_Response")


def send_with_gzip_fallback(
        send: Callable[[bytes, dict[str, str]], _Response],
        body: bytes, headers: dict[str, str], identity: bytes,
        identity_headers: dict[str, str],
        on_refused: Callable[[urllib.error.HTTPError], bool]) -> _Response:
    """Retry only the first HTTP 415 to a request actually encoded as gzip."""
    try:
        return send(body, headers)
    except urllib.error.HTTPError as error:
        if (error.code != 415 or headers.get("Content-Encoding") != "gzip"
                or not on_refused(error)):
            raise
    return send(identity, identity_headers)


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
        self.request_deadline = 0.0
        self.hard_deadline = 0.0
        self.socket_timeout = 0.0
        self.response_deadline = 0.0
        self.first_response_at: float | None = None
        self.send_stall_seconds = DEFAULT_SEND_STALL_S
        self.report_body_sent = False
        self.body_attempt = 0

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
        self._sending_output = False
        self._header_sent = False
        super().__init__(*args, **kwargs)
        tracker.register(self)

    def connect(self) -> None:
        if self._tracker.aborted:
            raise ConnectionError("HTTP request was aborted")
        if self._tracker.report_body_sent:
            remaining = self._tracker.request_deadline - time.monotonic()
            if remaining <= 0:
                raise RequestBodyDeadlineTimeout()
            self.timeout = min(self.timeout, remaining)
        try:
            super().connect()
        except TimeoutError as error:
            if (self._tracker.report_body_sent
                    and time.monotonic() >= self._tracker.request_deadline):
                raise RequestBodyDeadlineTimeout() from error
            raise
        if (self._tracker.report_body_sent
                and time.monotonic() >= self._tracker.request_deadline):
            raise RequestBodyDeadlineTimeout()
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

    def _send_output(self, message_body=None, encode_chunked=False):
        if self._tracker.report_body_sent:
            if time.monotonic() >= self._tracker.request_deadline:
                raise RequestBodyDeadlineTimeout()
            self._tracker.body_attempt += 1
            attempt = self._tracker.body_attempt
            self._body_sent_at = None
            sys.stderr.write(json.dumps({"body_attempt": attempt,
                                         "body_fully_sent": False}) + "\n")
            sys.stderr.flush()
        self._sending_output = True
        self._header_sent = False
        try:
            result = super()._send_output(message_body, encode_chunked=encode_chunked)
        finally:
            self._sending_output = False
        # Flushed before waiting for a response, so the parent can distinguish
        # provider queue time from an unfinished upload at the request deadline.
        if self._tracker.report_body_sent:
            completed_at = self._body_sent_at
            if completed_at is None or completed_at >= self._tracker.request_deadline:
                raise RequestBodyDeadlineTimeout()
            sys.stderr.write(json.dumps({"body_attempt": attempt,
                                         "body_fully_sent": True,
                                         "completed_at": completed_at}) + "\n")
            sys.stderr.flush()
        return result

    def send(self, data):
        if not self._sending_output or not self._header_sent:
            if (self._tracker.report_body_sent and self._sending_output
                    and time.monotonic() >= self._tracker.request_deadline):
                raise RequestBodyDeadlineTimeout()
            if self._tracker.report_body_sent and self._sending_output and self.sock is not None:
                self.sock.settimeout(max(0.001, self._tracker.request_deadline - time.monotonic()))
            try:
                result = super().send(data)
            except TimeoutError as error:
                if (self._tracker.report_body_sent and self._sending_output
                        and time.monotonic() >= self._tracker.request_deadline):
                    raise RequestBodyDeadlineTimeout() from error
                raise
            if (self._tracker.report_body_sent and self._sending_output
                    and time.monotonic() >= self._tracker.request_deadline):
                raise RequestBodyDeadlineTimeout()
            if self._sending_output:
                self._header_sent = True
            return result
        if not isinstance(data, (bytes, bytearray, memoryview)):
            return super().send(data)
        body = memoryview(data)
        total = len(body)
        sent = 0
        last_progress = time.monotonic()
        active_socket = self.sock
        if active_socket is None:
            raise ConnectionError("HTTP socket closed before request body")
        while sent < total:
            now = time.monotonic()
            if self._tracker.report_body_sent and now >= self._tracker.request_deadline:
                raise RequestBodyDeadlineTimeout()
            stalled = now - last_progress
            remaining = min(self._tracker.send_stall_seconds - stalled,
                            self._tracker.hard_deadline - now)
            if self._tracker.report_body_sent:
                remaining = min(remaining, self._tracker.request_deadline - now)
            if stalled >= self._tracker.send_stall_seconds:
                raise SendStallTimeout(sent, total, stalled)
            if remaining <= 0:
                raise TimeoutError("HTTP absolute deadline exceeded during request body send")
            active_socket.settimeout(min(0.25, remaining))
            try:
                moved = active_socket.send(body[sent:])
            except (socket.timeout, BlockingIOError):
                continue
            if moved <= 0:
                raise ConnectionError("HTTP socket closed during request body send")
            sent += moved
            last_progress = time.monotonic()
        if self._tracker.report_body_sent:
            self._body_sent_at = last_progress
            if last_progress >= self._tracker.request_deadline:
                raise RequestBodyDeadlineTimeout()

    def getresponse(self):
        """Wait for the first response byte without a premature socket timeout."""
        active_socket = self.sock
        response_deadline = self._tracker.request_deadline
        if active_socket is not None:
            remaining = self._tracker.hard_deadline - time.monotonic()
            if remaining <= 0 or not select.select([active_socket], [], [], remaining)[0]:
                raise TimeoutError("HTTP absolute deadline exceeded before response")
            arrived_at = time.monotonic()
            if self._tracker.first_response_at is None:
                self._tracker.first_response_at = arrived_at
            if arrived_at >= response_deadline:
                response_deadline = arrived_at + self._tracker.socket_timeout
            response_deadline = min(response_deadline, self._tracker.hard_deadline)
            self._tracker.response_deadline = response_deadline
            active_socket.settimeout(max(0.001, min(
                self._tracker.socket_timeout,
                response_deadline - arrived_at)))
        expired = threading.Event()
        finished = threading.Event()

        def stop_header_read() -> None:
            if not finished.is_set():
                expired.set()
                self.abort()

        timer = threading.Timer(max(0.001, response_deadline - time.monotonic()),
                                stop_header_read)
        timer.daemon = True
        timer.start()
        try:
            response = super().getresponse()
            if expired.is_set() or time.monotonic() >= response_deadline:
                raise TimeoutError("HTTP response header deadline exceeded")
            return response
        except Exception as error:
            if expired.is_set() or time.monotonic() >= response_deadline:
                raise TimeoutError("HTTP response header deadline exceeded") from error
            raise
        finally:
            finished.set()
            timer.cancel()

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
        # Built lazily on the first https send: a default context loads the CA store (~80 ms on 3.11
        # under load), and 3.12+ HTTPSHandler.__init__ builds one eagerly even for plain-http sends.
        # Never read HTTPSHandler private state: 3.12+ removed _check_hostname and broke every https send.
        self._tls_context: ssl.SSLContext | None = None
        urllib.request.AbstractHTTPHandler.__init__(self)

    def https_open(self, request):
        factory = functools.partial(_AbortiveHTTPSConnection, tracker=self._tracker)
        if self._tls_context is None:
            self._tls_context = ssl.create_default_context()
        return self.do_open(factory, request, context=self._tls_context)


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
    tracker.request_deadline = packet.get("request_deadline", packet["deadline"])
    tracker.hard_deadline = packet["deadline"]
    tracker.socket_timeout = packet["socket_timeout"]
    tracker.send_stall_seconds = packet.get("send_stall_seconds", DEFAULT_SEND_STALL_S)
    tracker.report_body_sent = packet.get("report_body_sent") is True
    if "route_allow_remote" in packet:
        _require_routed_local(packet["endpoint"], packet["route_allow_remote"])
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _AbortiveHTTPHandler(tracker), _AbortiveHTTPSHandler(tracker),
            _RoutedRedirects(packet["route_allow_remote"]))
    else:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _AbortiveHTTPHandler(tracker), _AbortiveHTTPSHandler(tracker))
    response_complete = False
    attempts: list[dict[str, object]] = []

    def send(data: bytes, headers: dict[str, str]) -> bytes:
        import hashlib
        request_headers = {"Content-Type": "application/json", **headers}
        request_headers["Authorization"] = "Bearer " + packet["key"]
        request = urllib.request.Request(packet["endpoint"], data=data,
                                         headers=request_headers)
        attempt = {"content_encoding": headers.get("Content-Encoding"),
                   "body_bytes": len(data), "body_sha256": hashlib.sha256(data).hexdigest(),
                   "http_status": None}
        attempts.append(attempt)
        try:
            remaining = packet["deadline"] - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("HTTP absolute deadline exceeded before send")
            if tracker.report_body_sent and time.monotonic() >= tracker.request_deadline:
                raise RequestBodyDeadlineTimeout()
            with opener.open(request, timeout=remaining) as response:
                attempt["http_status"] = response.status
                if response.length is not None and response.length > MAX_RESPONSE_BYTES:
                    raise ValueError("HTTP response exceeds bounded transport limit")
                result = _read_response_body(
                    response, tracker, deadline=tracker.response_deadline,
                    socket_timeout=packet["socket_timeout"], max_bytes=MAX_RESPONSE_BYTES)
                if len(result) > MAX_RESPONSE_BYTES:
                    raise ValueError("HTTP response exceeds bounded transport limit")
                tracker.finish_all()
            return result
        except urllib.error.HTTPError as error:
            attempt["http_status"] = error.code
            error_body = _read_response_body(
                error, tracker, deadline=tracker.response_deadline,
                socket_timeout=packet["socket_timeout"], max_bytes=MAX_ERROR_BYTES)
            error.relay_error = _relay_error_label(error_body)
            error.server_error = safe_server_error(error_body, packet["key"])
            raise
        except urllib.error.URLError as error:
            if isinstance(error.reason, (SendStallTimeout, RequestBodyDeadlineTimeout)):
                raise error.reason from error
            raise

    def on_refused(_error: urllib.error.HTTPError) -> bool:
        if time.monotonic() >= tracker.request_deadline:
            return False
        tracker.finish_all()
        # The resend has its own first byte. Its queue time must not inherit the 415.
        tracker.first_response_at = None
        tracker.response_deadline = tracker.request_deadline
        return True

    try:
        data = send_with_gzip_fallback(
            send, body, encoding, identity, identity_headers, on_refused)
        response_complete = True
    except Exception as error:
        if len(attempts) > 1:
            error.transport_attempts = attempts
        error.first_response_at = tracker.first_response_at
        raise
    finally:
        if not response_complete:
            tracker.abort_all()
    if len(attempts) > 1 or tracker.first_response_at is not None:
        sys.stderr.write(json.dumps({
            **({"transport_attempts": attempts} if len(attempts) > 1 else {}),
            "first_response_at": tracker.first_response_at}))
    sys.stdout.buffer.write(data)


def _probe_child() -> None:
    packet = json.load(sys.stdin)
    origin = tuple(packet["origin"])
    for path in packet["paths"]:
        remaining = packet["stop_at"] - time.monotonic()
        if remaining <= 0:
            break
        if packet.get("json_body"):
            result = _provider_probe_json_response(origin, path, packet["key"], remaining)
        else:
            result = _provider_probe_status(
                origin, path, packet["key"], remaining, packet["read_ok"])
        if result is not None:
            sys.stdout.write(json.dumps(result))
            return


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe-child"]:
        _probe_child()
        raise SystemExit(0)
    if sys.argv[1:] != ["--http-child"]:
        raise SystemExit("HTTP child invocation required")
    try:
        _child()
    except Exception as error:
        # Only a short sanitized reason crosses the pipe; the parent redacts
        # again using the endpoint key before exposing it to the run log.
        status = getattr(error, "code", None)
        relay_error = getattr(error, "relay_error", None)
        server_error = getattr(error, "server_error", None)
        sys.stderr.write(json.dumps({"error_type": type(error).__name__,
                                     "error_detail": safe_transport_detail(error),
                                     "http_status": status if type(status) is int else None,
                                     "relay_error": relay_error
                                     if isinstance(relay_error, str)
                                     and relay_error in RELAY_ERROR_LABELS else None,
                                     "server_error": server_error
                                     if isinstance(server_error, str) else None,
                                      "first_response_at": getattr(error, "first_response_at", None),
                                      **({"bytes_sent": error.bytes_sent,
                                          "bytes_total": error.bytes_total,
                                          "seconds_stalled": error.seconds_stalled}
                                         if isinstance(error, SendStallTimeout) else {}),
                                     **({"transport_attempts": error.transport_attempts}
                                        if hasattr(error, "transport_attempts") else {})}))
        raise SystemExit(1) from None
