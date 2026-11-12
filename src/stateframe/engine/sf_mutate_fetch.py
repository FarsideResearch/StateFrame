"""Disposable, bounded HTTP fetch broker for mutation units."""
from __future__ import annotations

import ipaddress
import json
import socket
import sys
from pathlib import Path
from urllib.parse import urlsplit

if __package__:
    from .kernel_web_fetch import WebPolicy, fetch_raw
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kernel_web_fetch import WebPolicy, fetch_raw


MAX_REQUEST_BYTES = 16 << 10
MAX_RESPONSE_BYTES = 8 << 20
MAX_TIMEOUT_S = 20.0
MAX_ACCEPT_BYTES = 8 << 10
MAX_HEADER_BYTES = 8 << 10
MAX_DOMAINS = 64


class _InvalidRequest(ValueError):
    pass


def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidRequest("duplicate request key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise _InvalidRequest("non-finite request number")


def _domain(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 253:
        raise _InvalidRequest("invalid domain allowlist")
    normalized = value.lower().rstrip(".")
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        try:
            normalized = normalized.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise _InvalidRequest("invalid domain allowlist") from exc
        labels = normalized.split(".")
        if any(not label or len(label) > 63 or label[0] == "-" or label[-1] == "-" or
               any(not (character.isalnum() or character == "-") for character in label)
               for label in labels):
            raise _InvalidRequest("invalid domain allowlist")
    else:
        normalized = str(address)
    return normalized


def _request(data: bytes) -> tuple[str, str | None, int, float, tuple[str, ...] | None]:
    if len(data) > MAX_REQUEST_BYTES:
        raise _InvalidRequest("request too large")
    try:
        value = json.loads(data.decode("utf-8", errors="strict"),
                           object_pairs_hook=_pairs_no_duplicates,
                           parse_constant=_invalid_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise _InvalidRequest("invalid JSON request") from exc
    if not isinstance(value, dict) or set(value) != {
            "url", "accept", "max_bytes", "timeout_s", "domains"}:
        raise _InvalidRequest("invalid request fields")
    url = value["url"]
    accept = value["accept"]
    max_bytes = value["max_bytes"]
    timeout_s = value["timeout_s"]
    domains = value["domains"]
    if not isinstance(url, str) or not url:
        raise _InvalidRequest("invalid URL")
    try:
        url_size = len(url.encode("utf-8", errors="strict"))
    except UnicodeError as exc:
        raise _InvalidRequest("invalid URL") from exc
    if url_size > 2048:
        raise _InvalidRequest("invalid URL")
    if accept is not None and not isinstance(accept, str):
        raise _InvalidRequest("invalid Accept header")
    if isinstance(accept, str):
        try:
            accept_size = len(accept.encode("utf-8", errors="strict"))
        except UnicodeError as exc:
            raise _InvalidRequest("invalid Accept header") from exc
    else:
        accept_size = 0
    if accept_size > MAX_ACCEPT_BYTES or (isinstance(accept, str) and
                                          ("\r" in accept or "\n" in accept)):
        raise _InvalidRequest("invalid Accept header")
    if (isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or
            not 1 <= max_bytes <= MAX_RESPONSE_BYTES):
        raise _InvalidRequest("invalid response byte limit")
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (float, int)) or
            not 0 < timeout_s <= MAX_TIMEOUT_S):
        raise _InvalidRequest("invalid timeout")
    if domains is None:
        allow_domains = None
    elif isinstance(domains, list) and 1 <= len(domains) <= MAX_DOMAINS:
        allow_domains = tuple(_domain(item) for item in domains)
    else:
        raise _InvalidRequest("invalid domain allowlist")
    return url, accept, max_bytes, float(timeout_s), allow_domains


def _frame(metadata: dict[str, object], body: bytes) -> bytes:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return b"SFMB1 " + str(len(encoded)).encode("ascii") + b" " + \
        str(len(body)).encode("ascii") + b"\n" + encoded + body


def _error(code: str) -> bytes:
    return _frame({"status": None, "headers": {}, "final_url": None,
                   "redirects": [], "error": code}, b"")


def run_once(source, destination, *, allow_private_for_tests: bool = False,
             resolver=socket.getaddrinfo) -> int:
    """Read one request and emit exactly one SFMB1 response frame."""
    request = source.read(MAX_REQUEST_BYTES + 1)
    try:
        url, accept, max_bytes, timeout_s, domains = _request(request)
    except _InvalidRequest:
        destination.write(_error("invalid_request"))
        destination.flush()
        return 0

    allowed_ports: tuple[int, ...] = ()
    if allow_private_for_tests:
        try:
            port = urlsplit(url).port
        except ValueError:
            port = None
        if port is not None and port not in (80, 443):
            allowed_ports = (port,)
    policy = WebPolicy(
        () if domains is None else domains,
        max_body=max_bytes,
        max_decoded=max_bytes,
        deadline=timeout_s,
        allowed_ports=allowed_ports,
        allow_private_for_tests=allow_private_for_tests)
    try:
        result = fetch_raw(url, policy, accept=accept, max_bytes=max_bytes,
                           broker_role=True, resolver=resolver)
        metadata: dict[str, object] = {
            "status": result.status,
            "headers": result.headers,
            "final_url": result.final_url,
            "redirects": list(result.redirects),
            "error": result.error,
        }
        body = result.body if result.error is None else b""
        header_size = sum(len((key + ": " + value + "\r\n").encode("utf-8"))
                          for key, value in result.headers.items())
        if header_size > MAX_HEADER_BYTES:
            destination.write(_error("headers_too_large"))
            destination.flush()
            return 0
        destination.write(_frame(metadata, body))
        destination.flush()
        return 0
    except Exception:
        destination.write(_error("broker_error"))
        destination.flush()
        return 0


def main() -> int:
    return run_once(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
