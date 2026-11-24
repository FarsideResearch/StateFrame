"""Direct, bounded OTLP/HTTP delivery with pinned DNS and finite retry windows."""
from __future__ import annotations

import datetime as _datetime
import email.utils
import gzip
import hashlib
import hmac
import http.client
import ipaddress
import math
import os
import re
import socket
import ssl
import stat
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from . import codec
from .types import ExportResult, Reason, ResultKind, SignalBatch


MAX_ENDPOINT_BYTES = 2048
MAX_HEADER_COUNT = 32
MAX_HEADER_BYTES = 8 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_RESPONSE_HEADERS = 64
MAX_RESPONSE_HEADER_BYTES = 16 * 1024
MAX_RESPONSE_LINE_BYTES = 4096
MAX_RETRIES = 4
MAX_RETRY_DELAY_S = 5.0
MAX_TLS_FILE_BYTES = 1024 * 1024
_HEADER_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_PERCENT_BAD = re.compile(r"%(?![0-9A-Fa-f]{2})")
_NUMERIC_HOST = re.compile(r"^[0-9.]+$")
_FORBIDDEN_HEADERS = frozenset({
    "host", "content-length", "content-type", "content-encoding", "transfer-encoding", "connection",
    "keep-alive", "upgrade", "te", "trailer", "proxy-authorization", "proxy-connection", "expect",
    "accept", "accept-encoding",
})
_METADATA_HOSTS = frozenset({"metadata.google.internal", "instance-data"})
_METADATA_ADDRESSES = frozenset({"169.254.169.254", "100.100.100.200"})


@dataclass(frozen=True, slots=True)
class TransportSettings:
    endpoint: str = field(repr=False)
    protocol: Literal["http/protobuf", "http/json"] = field(repr=False)
    headers: tuple[tuple[str, str], ...] = field(repr=False)
    compression: Literal["none", "gzip"] = field(repr=False)
    timeout_ms: int = field(repr=False)
    ca: Path | None = field(repr=False)
    cert: Path | None = field(repr=False)
    key_file: Path | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Endpoint:
    scheme: str
    host: str
    port: int
    path: str
    host_header: str
    request_target: str


class _TransportFailure(Exception):
    def __init__(self, reason: Reason, *, permanent: bool, may_have_delivered: bool = False):
        super().__init__(reason.value)
        self.reason = reason
        self.permanent = permanent
        self.may_have_delivered = may_have_delivered


def destination_id(key: bytes, settings: TransportSettings) -> bytes:
    """Return an opaque identity for the effective destination and wire format."""
    return destination_ids(key, settings, frozenset())[0]


def destination_ids(key: bytes, settings: TransportSettings,
                    recorded: frozenset[bytes]) -> tuple[bytes, frozenset[bytes]]:
    try:
        _validate_settings(settings)
    except _TransportFailure as error:
        raise ValueError(error.reason.value) from None
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("invalid_config")
    endpoint = _parse_endpoint(settings.endpoint)
    parts = [
        b"sf-otel/destination/1",
        endpoint.scheme.encode("ascii"),
        endpoint.host.lower().encode("idna"),
        str(endpoint.port).encode("ascii"),
        endpoint.request_target.encode("utf-8"),
        settings.protocol.encode("ascii"),
        settings.compression.encode("ascii"),
    ]
    suffix = [_header_fingerprint(settings.headers)]
    for path, is_key in ((settings.ca, False), (settings.cert, False), (settings.key_file, True)):
        suffix.append(_tls_file_digest(path, is_key=is_key) if path is not None else b"")
    digest = hmac.new(key, b"".join(struct.pack(">I", len(part)) + part for part in parts), hashlib.sha256)
    tail = b"".join(struct.pack(">I", len(part)) + part for part in suffix)

    def fingerprint(timeout: bytes) -> bytes:
        candidate = digest.copy()
        candidate.update(struct.pack(">I", len(timeout)) + timeout + tail)
        return candidate.digest()[:16]

    # Old rows contain only keyed fingerprints, so match them against the former timeout domain.
    current = fingerprint(b"10000")
    prior = recorded - {current}
    matches = (fingerprint(str(timeout).encode("ascii")) for timeout in range(100, 30_001)) if prior else ()
    return current, frozenset(value for value in matches if value in prior)


def send(batch: SignalBatch, settings: TransportSettings, deadline_ns: int) -> ExportResult:
    """Send a stable batch within one timeout/deadline budget; never follow redirects."""
    start_ns = time.monotonic_ns()
    if type(deadline_ns) is not int or deadline_ns <= start_ns:
        return ExportResult(ResultKind.RETRY, None, None, Reason.EXPIRED, 0.0, False)
    try:
        _validate_settings(settings)
        budget_end_ns = min(deadline_ns, start_ns + settings.timeout_ms * 1_000_000)
        payload = codec.encode(batch, settings.protocol)
        endpoint = _parse_endpoint(settings.endpoint)
        addresses = _resolve(endpoint)
    except _TransportFailure as error:
        return _permanent(error.reason, batch, error.may_have_delivered)
    except (ValueError, OSError, UnicodeError, TypeError):
        return _permanent(Reason.INVALID_CONFIG, batch)

    if time.monotonic_ns() >= budget_end_ns:
        return ExportResult(ResultKind.RETRY, None, None, Reason.EXPIRED, 0.0, False)
    wire = payload
    headers: list[tuple[str, str]] = list(settings.headers)
    if settings.compression == "gzip":
        wire = gzip.compress(payload, compresslevel=6, mtime=0)
    content_type = "application/x-protobuf" if settings.protocol == "http/protobuf" else "application/json"
    lost_attempt = False
    last_reason = Reason.CONNECTION_FAILED

    for attempt in range(MAX_RETRIES):
        if time.monotonic_ns() >= budget_end_ns:
            return ExportResult(ResultKind.RETRY, None, None, Reason.EXPIRED, 0.0, lost_attempt)
        request_started = False
        conn: http.client.HTTPConnection | None = None
        try:
            sock = _connect(endpoint, addresses, attempt, settings, budget_end_ns)
            conn = http.client.HTTPConnection(endpoint.host, endpoint.port, timeout=_remaining_s(budget_end_ns))
            conn.sock = sock
            sock.settimeout(_remaining_s(budget_end_ns))
            conn.putrequest("POST", endpoint.request_target, skip_host=True, skip_accept_encoding=True)
            conn.putheader("Host", endpoint.host_header)
            conn.putheader("Content-Type", content_type)
            conn.putheader("Content-Length", str(len(wire)))
            conn.putheader("Connection", "close")
            if settings.compression == "gzip":
                conn.putheader("Content-Encoding", "gzip")
            for name, value in headers:
                conn.putheader(name, value)
            request_started = True
            conn.endheaders(wire)
            response_status, response_headers, body = _read_response(conn.sock, budget_end_ns)
            conn.close()
        except ssl.SSLCertVerificationError:
            if conn is not None:
                conn.close()
            return _permanent(Reason.TLS_FAILED, batch, False)
        except ssl.SSLError:
            if conn is not None:
                conn.close()
            return _permanent(Reason.TLS_FAILED, batch, False)
        except _TransportFailure as error:
            if conn is not None:
                conn.close()
            last_reason = error.reason
            lost_attempt = lost_attempt or error.may_have_delivered or request_started
            if error.permanent:
                return _permanent(error.reason, batch, lost_attempt)
            retry_result = _retry_result(batch, error.reason, lost_attempt)
            if attempt + 1 >= MAX_RETRIES:
                return retry_result
            if not _wait_for_retry(_backoff(attempt, wire), budget_end_ns):
                return retry_result
            continue
        except (socket.timeout, TimeoutError):
            if conn is not None:
                conn.close()
            last_reason = Reason.TIMEOUT
            lost_attempt = lost_attempt or request_started
            retry_result = _retry_result(batch, last_reason, lost_attempt)
            if attempt + 1 >= MAX_RETRIES:
                return retry_result
            if not _wait_for_retry(_backoff(attempt, wire), budget_end_ns):
                return retry_result
            continue
        except (ConnectionError, OSError, http.client.HTTPException):
            if conn is not None:
                conn.close()
            last_reason = Reason.CONNECTION_FAILED
            lost_attempt = lost_attempt or request_started
            retry_result = _retry_result(batch, last_reason, lost_attempt)
            if attempt + 1 >= MAX_RETRIES:
                return retry_result
            if not _wait_for_retry(_backoff(attempt, wire), budget_end_ns):
                return retry_result
            continue
        except Exception:
            # The sender is disposable; preserve only a fixed reason at its boundary.
            if conn is not None:
                conn.close()
            return _permanent(Reason.PEER_FAILURE, batch, request_started)

        if 300 <= response_status < 400:
            return _permanent(Reason.REDIRECT, batch, False)
        if response_status == 200:
            media = response_headers.get("content-type", "").partition(";")[0].strip().lower()
            try:
                if media != content_type:
                    raise ValueError("bad_response")
                result = codec.decode_success(body, settings.protocol, batch.signal, _item_count(batch))
            except ValueError:
                return ExportResult(ResultKind.UNKNOWN, None, None, Reason.BAD_RESPONSE, None, True)
            if lost_attempt:
                return ExportResult(
                    result.kind, result.accepted, result.rejected, result.reason, result.retry_delay_s, True,
                )
            return result
        if response_status in (429, 502, 503, 504):
            last_reason = Reason.THROTTLED if response_status == 429 else Reason.PEER_FAILURE
            lost_attempt = True
            delay = _retry_after(response_headers.get("retry-after"), time.monotonic_ns())
            retry_result = _retry_result(batch, last_reason, True, delay)
            if attempt + 1 >= MAX_RETRIES or not _wait_for_retry(
                    max(_backoff(attempt, wire), delay or 0.0), budget_end_ns):
                return retry_result
            continue
        if response_status in (401, 403):
            return _permanent(Reason.AUTH_FAILED, batch)
        if response_status == 413:
            return _permanent(Reason.PAYLOAD_REJECTED, batch)
        if 400 <= response_status < 500:
            return _permanent(Reason.PAYLOAD_REJECTED, batch)
        return _permanent(Reason.PEER_FAILURE, batch, True)
    return _retry_result(batch, last_reason, lost_attempt)


def _validate_settings(settings: TransportSettings) -> None:
    if type(settings) is not TransportSettings:
        raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
    endpoint = _parse_endpoint(settings.endpoint)
    if settings.protocol not in ("http/protobuf", "http/json"):
        raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
    if settings.compression not in ("none", "gzip"):
        raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
    if type(settings.timeout_ms) is not int or not 100 <= settings.timeout_ms <= 30_000:
        raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
    if type(settings.headers) is not tuple or len(settings.headers) > MAX_HEADER_COUNT:
        raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
    if (settings.cert is None) != (settings.key_file is None):
        raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
    if (settings.ca is not None or settings.cert is not None) and endpoint.scheme != "https":
        raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
    seen: set[str] = set()
    total = 0
    for pair in settings.headers:
        if type(pair) is not tuple or len(pair) != 2:
            raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
        name, value = pair
        if type(name) is not str or type(value) is not str or not name.isascii() or not value.isascii():
            raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
        if not _HEADER_TOKEN.fullmatch(name) or name.casefold() in _FORBIDDEN_HEADERS:
            raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
        if any(ord(ch) < 0x20 and ch != " " or ord(ch) > 0x7E for ch in value):
            raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
        folded = name.casefold()
        if folded in seen:
            raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
        seen.add(folded)
        total += len(name) + len(value) + 2
    if total > MAX_HEADER_BYTES:
        raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
    if endpoint.scheme == "http" and settings.headers:
        raise _TransportFailure(Reason.AUTH_FAILED, permanent=True)
    if settings.ca is not None:
        _tls_file_digest(settings.ca, is_key=False)
    if settings.cert is not None:
        _tls_file_digest(settings.cert, is_key=False)
        _tls_file_digest(settings.key_file, is_key=True)


def _parse_endpoint(value: str) -> _Endpoint:
    if type(value) is not str or not value or len(value.encode("utf-8", "ignore")) > MAX_ENDPOINT_BYTES:
        raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
    if any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value) or _PERCENT_BAD.search(value):
        raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True)
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        if (scheme not in ("http", "https") or parsed.username is not None or parsed.password is not None
                or "@" in parsed.netloc or "?" in value or "#" in value):
            raise ValueError
        if not parsed.hostname or "%" in parsed.hostname:
            raise ValueError
        port = parsed.port if parsed.port is not None else (443 if scheme == "https" else 80)
        if not 1 <= port <= 65535 or _NUMERIC_HOST.fullmatch(parsed.hostname) and not _valid_ip(parsed.hostname):
            raise ValueError
        host = parsed.hostname
        if ":" in host:
            host = ipaddress.IPv6Address(host).compressed
        else:
            host = host.encode("idna").decode("ascii").lower()
        if host.casefold().rstrip(".") in _METADATA_HOSTS:
            raise ValueError
        if scheme == "http":
            address = ipaddress.ip_address(host)
            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
                address = address.ipv4_mapped
            if not address.is_loopback:
                raise ValueError
        path = parsed.path or "/"
        if not path.startswith("/"):
            raise ValueError
        host_display = f"[{host}]" if ":" in host else host
        host_header = host_display if port == (443 if scheme == "https" else 80) else f"{host_display}:{port}"
        return _Endpoint(scheme, host, port, path, host_header, path)
    except (ValueError, UnicodeError):
        raise _TransportFailure(Reason.INVALID_CONFIG, permanent=True) from None


def _resolve(endpoint: _Endpoint) -> tuple[tuple[int, int, int, tuple], ...]:
    try:
        literal = ipaddress.ip_address(endpoint.host)
    except ValueError:
        literal = None
    if literal is not None:
        if not _allowed_address(literal):
            raise _TransportFailure(Reason.DNS_FAILED, permanent=True)
        if literal.version == 4:
            return ((socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, (str(literal), endpoint.port)),)
        return ((socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP,
                 (str(literal), endpoint.port, 0, literal.scope_id if hasattr(literal, "scope_id") else 0)),)
    try:
        results = socket.getaddrinfo(endpoint.host, endpoint.port, type=socket.SOCK_STREAM)
    except (OSError, socket.gaierror):
        raise _TransportFailure(Reason.DNS_FAILED, permanent=True) from None
    if not results:
        raise _TransportFailure(Reason.DNS_FAILED, permanent=True)
    addresses: list[tuple[int, int, int, tuple]] = []
    seen: set[tuple] = set()
    for family, socktype, proto, _canon, sockaddr in results[:16]:
        if family not in (socket.AF_INET, socket.AF_INET6) or socktype not in (0, socket.SOCK_STREAM):
            raise _TransportFailure(Reason.DNS_FAILED, permanent=True)
        try:
            address = ipaddress.ip_address(sockaddr[0])
        except (ValueError, IndexError, TypeError):
            raise _TransportFailure(Reason.DNS_FAILED, permanent=True) from None
        if not _allowed_address(address):
            raise _TransportFailure(Reason.DNS_FAILED, permanent=True)
        candidate = (family, socktype or socket.SOCK_STREAM, proto or socket.IPPROTO_TCP, tuple(sockaddr))
        if candidate[3] not in seen:
            seen.add(candidate[3])
            addresses.append(candidate)
    if not addresses:
        raise _TransportFailure(Reason.DNS_FAILED, permanent=True)
    return tuple(addresses)


def _allowed_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return (
        not address.is_link_local
        and not address.is_unspecified
        and not address.is_multicast
        and str(address) not in _METADATA_ADDRESSES
        and not (isinstance(address, ipaddress.IPv4Address) and int(address) == 0xFFFFFFFF)
    )


def _connect(endpoint: _Endpoint, addresses: tuple[tuple[int, int, int, tuple], ...], attempt: int,
             settings: TransportSettings, deadline_ns: int) -> socket.socket:
    last_error: OSError | None = None
    candidates = addresses[attempt % len(addresses):] + addresses[:attempt % len(addresses)]
    for family, socktype, proto, sockaddr in candidates:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.settimeout(_remaining_s(deadline_ns))
            sock.connect(sockaddr)
            if endpoint.scheme == "https":
                context = ssl.create_default_context()
                context.check_hostname = True
                context.verify_mode = ssl.CERT_REQUIRED
                if settings.ca is not None:
                    context.load_verify_locations(cafile=str(settings.ca))
                if settings.cert is not None:
                    context.load_cert_chain(certfile=str(settings.cert), keyfile=str(settings.key_file))
                sock.settimeout(_remaining_s(deadline_ns))
                sock = context.wrap_socket(sock, server_hostname=endpoint.host)
                sock.settimeout(_remaining_s(deadline_ns))
            return sock
        except (ssl.SSLError, socket.timeout, TimeoutError):
            sock.close()
            raise
        except OSError as error:
            last_error = error
            sock.close()
    if last_error is not None:
        raise last_error
    raise OSError("connection failed")


def _read_response(sock: socket.socket, deadline_ns: int) -> tuple[int, dict[str, str], bytes]:
    reader = sock.makefile("rb")
    try:
        status_line = _readline(reader, sock, deadline_ns)
        parts = status_line.rstrip(b"\r\n").split(b" ", 2)
        if len(parts) < 2 or parts[0] not in (b"HTTP/1.0", b"HTTP/1.1") or len(parts[1]) != 3 or not parts[1].isdigit():
            raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
        status = int(parts[1])
        headers: dict[str, str] = {}
        total = len(status_line)
        while True:
            line = _readline(reader, sock, deadline_ns)
            total += len(line)
            if total > MAX_RESPONSE_HEADER_BYTES:
                raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
            if line in (b"\r\n", b"\n"):
                break
            if line[:1] in (b" ", b"\t") or b":" not in line:
                raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
            name, value = line.split(b":", 1)
            try:
                key = name.decode("ascii").lower()
                text = value.decode("ascii").strip()
            except UnicodeDecodeError:
                raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True) from None
            if not _HEADER_TOKEN.fullmatch(key) or any(ord(ch) < 0x20 and ch != "\t" for ch in text):
                raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
            if key in headers:
                raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
            headers[key] = text
            if len(headers) > MAX_RESPONSE_HEADERS:
                raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
        if "transfer-encoding" in headers:
            if headers["transfer-encoding"].lower() != "chunked" or "content-length" in headers:
                raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
            transmitted = _read_chunked(reader, sock, deadline_ns)
        elif "content-length" in headers:
            length_text = headers["content-length"]
            if not length_text.isascii() or not length_text.isdecimal() or len(length_text) > 10:
                raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
            length = int(length_text)
            if length > MAX_RESPONSE_BYTES:
                raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
            transmitted = _read_exact(reader, sock, length, deadline_ns)
        else:
            transmitted = _read_until_close(reader, sock, deadline_ns)
        return status, headers, _decode_response_content(transmitted, headers.get("content-encoding"))
    finally:
        reader.close()


def _readline(reader, sock: socket.socket, deadline_ns: int) -> bytes:
    sock.settimeout(_remaining_s(deadline_ns))
    line = reader.readline(MAX_RESPONSE_LINE_BYTES + 1)
    if not line or len(line) > MAX_RESPONSE_LINE_BYTES or not line.endswith(b"\n"):
        raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
    return line


def _read_exact(reader, sock: socket.socket, length: int, deadline_ns: int) -> bytes:
    body = bytearray()
    while len(body) < length:
        sock.settimeout(_remaining_s(deadline_ns))
        part = reader.read(min(8192, length - len(body)))
        if not part:
            raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
        body.extend(part)
    return bytes(body)


def _read_chunked(reader, sock: socket.socket, deadline_ns: int) -> bytes:
    body = bytearray()
    chunks = 0
    while True:
        line = _readline(reader, sock, deadline_ns)
        size_text = line.rstrip(b"\r\n").split(b";", 1)[0]
        if not size_text or len(size_text) > 16 or not re.fullmatch(rb"[0-9A-Fa-f]+", size_text):
            raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
        size = int(size_text, 16)
        chunks += 1
        if chunks > 4096 or len(body) + size > MAX_RESPONSE_BYTES:
            raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
        if size == 0:
            trailer_bytes = 0
            trailer_count = 0
            while True:
                trailer = _readline(reader, sock, deadline_ns)
                trailer_bytes += len(trailer)
                if trailer_bytes > MAX_RESPONSE_HEADER_BYTES:
                    raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
                if trailer in (b"\r\n", b"\n"):
                    return bytes(body)
                trailer_count += 1
                if trailer_count > MAX_RESPONSE_HEADERS or b":" not in trailer:
                    raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
        body.extend(_read_exact(reader, sock, size, deadline_ns))
        if _read_exact(reader, sock, 2, deadline_ns) != b"\r\n":
            raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)


def _read_until_close(reader, sock: socket.socket, deadline_ns: int) -> bytes:
    body = bytearray()
    while len(body) <= MAX_RESPONSE_BYTES:
        sock.settimeout(_remaining_s(deadline_ns))
        part = reader.read(min(8192, MAX_RESPONSE_BYTES + 1 - len(body)))
        if not part:
            return bytes(body)
        body.extend(part)
    raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)


def _decode_response_content(body: bytes, content_encoding: str | None) -> bytes:
    if content_encoding is None or content_encoding.lower() in ("", "identity"):
        if len(body) > MAX_RESPONSE_BYTES:
            raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
        return body
    if content_encoding.lower() != "gzip":
        raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    output = bytearray()
    try:
        for offset in range(0, len(body), 4096):
            chunk = body[offset:offset + 4096]
            output.extend(inflater.decompress(chunk, MAX_RESPONSE_BYTES + 1 - len(output)))
            if len(output) > MAX_RESPONSE_BYTES or inflater.unconsumed_tail:
                raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
        output.extend(inflater.flush(MAX_RESPONSE_BYTES + 1 - len(output)))
    except zlib.error:
        raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True) from None
    if len(output) > MAX_RESPONSE_BYTES or not inflater.eof or inflater.unused_data:
        raise _TransportFailure(Reason.BAD_RESPONSE, permanent=True, may_have_delivered=True)
    return bytes(output)


def _retry_after(value: str | None, received_mono_ns: int) -> float | None:
    if value is None or len(value) > 128:
        return None
    if value.isascii() and (value.isdecimal() or re.fullmatch(r"[0-9]+\.[0-9]+", value)):
        try:
            return min(float(value), 86400.0)
        except (ValueError, OverflowError):
            return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_datetime.timezone.utc)
        wall_now = _datetime.datetime.now(_datetime.timezone.utc)
        delta = max(0.0, (parsed - wall_now).total_seconds())
        # Convert once at response receipt; later wall-clock changes cannot extend this delay.
        monotonic_deadline = received_mono_ns + int(delta * 1_000_000_000)
        return max(0.0, (monotonic_deadline - received_mono_ns) / 1_000_000_000)
    except (TypeError, ValueError, OverflowError):
        return None


def _backoff(attempt: int, seed: bytes) -> float:
    ceiling = min(MAX_RETRY_DELAY_S, 0.25 * (2 ** attempt))
    digest = hashlib.sha256(b"stateframe-otel-retry/1\0" + attempt.to_bytes(4, "big") + seed).digest()
    return ceiling * int.from_bytes(digest[:8], "big") / (1 << 64)


def _wait_for_retry(delay: float, deadline_ns: int) -> bool:
    if not math_is_finite(delay) or delay < 0:
        return False
    remaining = _remaining_ns(deadline_ns)
    delay_ns = int(delay * 1_000_000_000)
    if delay_ns >= remaining:
        return False
    if delay_ns:
        time.sleep(delay_ns / 1_000_000_000)
    return time.monotonic_ns() < deadline_ns


def _remaining_ns(deadline_ns: int) -> int:
    return max(0, deadline_ns - time.monotonic_ns())


def _remaining_s(deadline_ns: int) -> float:
    remaining = _remaining_ns(deadline_ns)
    if remaining <= 0:
        raise _TransportFailure(Reason.TIMEOUT, permanent=False, may_have_delivered=True)
    return max(0.001, remaining / 1_000_000_000)


def _retry_result(batch: SignalBatch, reason: Reason, uncertain: bool, delay: float | None = None) -> ExportResult:
    return ExportResult(ResultKind.RETRY, None, None, reason, delay, uncertain)


def _permanent(reason: Reason, batch: SignalBatch, uncertain: bool = False) -> ExportResult:
    return ExportResult(ResultKind.PERMANENT, None, None, reason, None, uncertain)


def _item_count(batch: SignalBatch) -> int:
    return len(batch.spans)


def _valid_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _header_fingerprint(headers: tuple[tuple[str, str], ...]) -> bytes:
    digest = hashlib.sha256()
    for name, value in headers:
        for part in (name.casefold().encode("ascii"), value.encode("ascii")):
            digest.update(struct.pack(">I", len(part)))
            digest.update(part)
    return digest.digest()


def _tls_file_digest(path: Path | None, *, is_key: bool) -> bytes:
    if path is None or not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("invalid_config")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid():
                raise ValueError("invalid_config")
            if before.st_mode & (stat.S_IWGRP | stat.S_IWOTH) or is_key and stat.S_IMODE(before.st_mode) != 0o600:
                raise ValueError("invalid_config")
            if before.st_size <= 0 or before.st_size > MAX_TLS_FILE_BYTES:
                raise ValueError("invalid_config")
            chunks = bytearray()
            while len(chunks) <= MAX_TLS_FILE_BYTES:
                part = os.read(fd, min(65536, MAX_TLS_FILE_BYTES + 1 - len(chunks)))
                if not part:
                    break
                chunks.extend(part)
            after = os.fstat(fd)
            path_after = os.stat(path, follow_symlinks=False)
            identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            path_identity = (path_after.st_dev, path_after.st_ino, path_after.st_size,
                             path_after.st_mtime_ns, path_after.st_ctime_ns)
            if identity_before != identity_after or identity_before != path_identity or len(chunks) != before.st_size:
                raise ValueError("invalid_config")
            if len(chunks) > MAX_TLS_FILE_BYTES:
                raise ValueError("invalid_config")
            return hashlib.sha256(chunks).digest()
        finally:
            os.close(fd)
    except (OSError, ValueError):
        raise ValueError("invalid_config") from None


def math_is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


__all__ = ["TransportSettings", "destination_id", "send"]
