"""Bounded web observations with direct, vetted-IP connections."""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import http.client
import io
import ipaddress
import socket
import ssl
import threading
import time
from typing import Callable, Mapping
from urllib.parse import urljoin, urlsplit, urlunsplit
import zlib


@dataclass(frozen=True)
class WebPolicy:
    allow_domains: tuple[str, ...]
    max_body: int = 2 << 20
    max_decoded: int = 8 << 20
    deadline: float = 20.0
    max_chars: int = 12000
    allowed_ports: tuple[int, ...] = ()
    # For isolated transport tests only; never set by the CLI or runtime.
    allow_private_for_tests: bool = False


@dataclass(frozen=True)
class WebResult:
    ok: bool
    status: int | None
    final_url: str | None
    content_type: str | None
    text: str
    total_bytes: int
    refusal: str | None


class WebRefusal(ValueError):
    pass


_DNS_SLOTS = threading.BoundedSemaphore(4)


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if ip.ipv4_mapped:
        return ip.ipv4_mapped
    if ip.sixtofour:
        return ip.sixtofour
    if ip.teredo:
        return ip.teredo[1]
    if ip in ipaddress.ip_network("64:ff9b::/96") or ip in ipaddress.ip_network("64:ff9b:1::/48"):
        return ipaddress.IPv4Address(int(ip) & 0xffffffff)
    return None


def public_address(value: str) -> bool:
    """Refuse non-global addresses, including embedded private IPv4 forms."""
    try:
        ip = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    if (not ip.is_global or ip.is_multicast or ip.is_reserved or
            (isinstance(ip, ipaddress.IPv6Address) and ip.is_site_local)):
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(ip)
        if embedded is not None and not embedded.is_global:
            return False
    return True


def _host_allowed(host: str, policy: WebPolicy) -> bool:
    if not policy.allow_domains:
        return True
    for domain in policy.allow_domains:
        domain = domain.lower().rstrip(".")
        if host == domain or host.endswith("." + domain):
            return True
    return False


def _parse_url(url: str, policy: WebPolicy) -> tuple[str, str, int, str]:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise WebRefusal(f"invalid URL: {exc}") from exc
    if parts.scheme.lower() not in {"http", "https"}:
        raise WebRefusal("URL scheme must be http or https")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise WebRefusal("URL userinfo is not allowed")
    if not parts.hostname:
        raise WebRefusal("URL host is required")
    host = parts.hostname.lower().rstrip(".")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise WebRefusal("invalid URL hostname") from exc
    if not _host_allowed(host, policy):
        raise WebRefusal(f"host is outside the web allow list: {host}")
    port = port or (443 if parts.scheme.lower() == "https" else 80)
    if port not in (80, 443, *policy.allowed_ports):
        raise WebRefusal(f"port is not allowed: {port}")
    path = urlunsplit(("", "", parts.path or "/", parts.query, ""))
    return parts.scheme.lower(), host, port, path


def _addresses(host: str, port: int, resolver: Callable, policy: WebPolicy,
               deadline: float) -> list[tuple[int, str]]:
    if not _DNS_SLOTS.acquire(blocking=False):
        raise WebRefusal("DNS resolver capacity exhausted")
    completed = threading.Event()
    outcome: dict[str, object] = {}

    def resolve() -> None:
        try:
            outcome["answers"] = resolver(host, port, type=socket.SOCK_STREAM)
        except Exception as exc:
            outcome["error"] = exc
        finally:
            _DNS_SLOTS.release()
            completed.set()

    threading.Thread(target=resolve, daemon=True, name="stateframe-web-dns").start()
    if not completed.wait(max(0, deadline - time.monotonic())):
        raise WebRefusal("web fetch total deadline exceeded during DNS resolution")
    if "error" in outcome:
        raise WebRefusal(f"DNS resolution failed: {outcome['error']}")
    answers = outcome["answers"]
    if not answers:
        raise WebRefusal("DNS returned no addresses")
    result: list[tuple[int, str]] = []
    for family, socktype, _, _, sockaddr in answers:
        if family not in (socket.AF_INET, socket.AF_INET6) or socktype != socket.SOCK_STREAM:
            raise WebRefusal("DNS returned an unsupported address family")
        address = sockaddr[0]
        test_loopback = policy.allow_private_for_tests and address in {"127.0.0.1", "::1"}
        if not test_loopback and not public_address(address):
            raise WebRefusal(f"DNS returned a non-public address: {address}")
        result.append((family, address))
    return result


class _PinnedHTTP(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, address: str, deadline: float):
        super().__init__(host, port)
        self._address = address
        self._deadline = deadline
        self.transport: socket.socket | None = None

    def _remaining(self) -> float:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("web fetch total deadline exceeded")
        return remaining

    def connect(self) -> None:
        raw = socket.create_connection((self._address, self.port), self._remaining())
        self.transport = raw
        self.sock = _DeadlineSocket(raw, self._deadline)

class _DeadlineReader(io.RawIOBase):
    def __init__(self, sock: socket.socket, deadline: float):
        self.sock = sock
        self.deadline = deadline

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("web fetch total deadline exceeded")
        self.sock.settimeout(remaining)
        return self.sock.recv_into(buffer)


class _DeadlineSocket:
    def __init__(self, sock: socket.socket, deadline: float):
        self._sock = sock
        self._deadline = deadline

    def makefile(self, mode: str = "rb", buffering: int = -1) -> io.BufferedReader:
        if mode != "rb":
            raise ValueError("web transport only supports binary reads")
        return io.BufferedReader(_DeadlineReader(self._sock, self._deadline))

    def close(self) -> None:
        # HTTPConnection closes its socket after headers with Connection: close;
        # the response still owns the body stream. fetch closes the raw socket.
        pass

    def __getattr__(self, name: str):
        return getattr(self._sock, name)


class _PinnedHTTPS(_PinnedHTTP, http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, address: str, deadline: float):
        _PinnedHTTP.__init__(self, host, port, address, deadline)
        self._context = ssl.create_default_context()

    def connect(self) -> None:
        raw = socket.create_connection((self._address, self.port), self._remaining())
        try:
            secured = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise
        self.transport = secured
        self.sock = _DeadlineSocket(secured, self._deadline)


class _ReadableHTML(HTMLParser):
    _BREAKS = frozenset({"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"})

    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.parts: list[str] = []
        self.hidden = 0
        self.links: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1
        if self.hidden:
            return
        if tag in self._BREAKS:
            self.parts.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            url = urljoin(self.base, href) if href else None
            self.links.append(url if url and urlsplit(url).scheme in {"http", "https"} else None)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden = max(0, self.hidden - 1)
            return
        if self.hidden:
            return
        if tag == "a" and self.links:
            href = self.links.pop()
            if href:
                self.parts.append(f" [{href}]")
        if tag in self._BREAKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)

    def text(self) -> str:
        return "\n".join(" ".join(line.split()) for line in "".join(self.parts).splitlines() if line.strip())


def _read_body(response: http.client.HTTPResponse, connection: _PinnedHTTP,
               policy: WebPolicy, encoding: str) -> tuple[bytes, int]:
    if encoding not in {"", "identity", "gzip", "deflate"}:
        raise WebRefusal(f"unsupported content encoding: {encoding}")
    decoder = (zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else
               zlib.decompressobj() if encoding == "deflate" else None)
    chunks: list[bytes] = []
    raw_total = decoded_total = 0
    while True:
        assert connection.transport is not None
        connection.transport.settimeout(connection._remaining())
        chunk = response.read1(8192)
        if not chunk:
            break
        raw_total += len(chunk)
        if raw_total > policy.max_body:
            raise WebRefusal(f"response body exceeds {policy.max_body} bytes")
        if decoder is not None:
            try:
                block = decoder.decompress(chunk, policy.max_decoded - decoded_total + 1)
            except zlib.error as exc:
                raise WebRefusal(f"invalid compressed response: {exc}") from exc
        else:
            block = chunk
        decoded_total += len(block)
        if decoded_total > policy.max_decoded:
            raise WebRefusal(f"decoded response exceeds {policy.max_decoded} bytes")
        chunks.append(block)
    if decoder is not None:
        try:
            tail = decoder.flush(policy.max_decoded - decoded_total + 1)
        except zlib.error as exc:
            raise WebRefusal(f"invalid compressed response: {exc}") from exc
        decoded_total += len(tail)
        if decoded_total > policy.max_decoded or not decoder.eof:
            raise WebRefusal("compressed response exceeds limit or is incomplete")
        chunks.append(tail)
    return b"".join(chunks), raw_total


def fetch(url: str, policy: WebPolicy, *, resolver: Callable = socket.getaddrinfo) -> WebResult:
    """Fetch without proxy environment use; policy refusals are typed results."""
    started = time.monotonic()
    deadline = started + policy.deadline
    current = url
    try:
        for redirect_count in range(6):
            scheme, host, port, path = _parse_url(current, policy)
            addresses = _addresses(host, port, resolver, policy, deadline)
            # Every answer is vetted before any connection. Use only the first vetted IP.
            _, address = addresses[0]
            cls = _PinnedHTTPS if scheme == "https" else _PinnedHTTP
            conn = cls(host, port, address, deadline)
            try:
                default_port = 443 if scheme == "https" else 80
                conn.request("GET", path, headers={"Host": host if port == default_port else f"{host}:{port}",
                                                  "Accept": "text/html,text/plain,application/json,application/xml,text/*",
                                                  "Accept-Encoding": "gzip, deflate", "User-Agent": "StateFrame/1"})
                response = conn.getresponse()
                status = response.status
                if status in {301, 302, 303, 307, 308}:
                    location = response.getheader("Location")
                    if not location:
                        raise WebRefusal("redirect has no Location header")
                    if redirect_count == 5:
                        raise WebRefusal("too many redirects (maximum 5)")
                    current = urljoin(current, location)
                    continue
                content_type_header = response.getheader("Content-Type", "")
                content_type = content_type_header.split(";", 1)[0].strip().lower()
                if not (content_type.startswith("text/") or content_type in {"application/json", "application/xml", "application/xhtml+xml"}):
                    raise WebRefusal(f"content type is not readable text: {content_type or '(missing)'}")
                body, raw_total = _read_body(response, conn, policy,
                                             response.getheader("Content-Encoding", "identity").lower().strip())
                charset = "utf-8"
                for parameter in content_type_header.split(";")[1:]:
                    key, _, value = parameter.strip().partition("=")
                    if key.lower() == "charset" and value:
                        charset = value.strip().strip('"')
                try:
                    decoded = body.decode(charset, errors="replace")
                except LookupError as exc:
                    raise WebRefusal(f"unsupported charset: {charset}") from exc
                if content_type in {"text/html", "application/xhtml+xml"}:
                    parser = _ReadableHTML(current)
                    parser.feed(decoded)
                    decoded = parser.text()
                shown = decoded[:policy.max_chars]
                note = (f"\n[truncated to {policy.max_chars} characters; {len(body)} decoded bytes, "
                        f"{raw_total} transferred bytes]" if len(decoded) > policy.max_chars else
                        f"\n[{len(body)} decoded bytes, {raw_total} transferred bytes]")
                text = f"UNTRUSTED EXTERNAL CONTENT from {current}\n{shown}{note}\nEND UNTRUSTED EXTERNAL CONTENT"
                return WebResult(True, status, current, content_type, text, raw_total, None)
            finally:
                conn.close()
                if conn.transport is not None:
                    conn.transport.close()
        raise WebRefusal("too many redirects")
    except (WebRefusal, ValueError, OSError, TimeoutError, http.client.HTTPException, ssl.SSLError) as exc:
        reason = str(exc) or type(exc).__name__
        return WebResult(False, None, None, None, "", 0, reason)


def handle_web_call(policy: WebPolicy, arguments: Mapping[str, object]) -> tuple[str, dict]:
    if set(arguments) != {"url"} or not isinstance(arguments.get("url"), str):
        return "web fetch refused: url must be a string", {"ok": False, "refusal": "invalid_request"}
    result = fetch(arguments["url"], policy)
    if not result.ok:
        return f"web fetch refused: {result.refusal}", {"ok": False, "refusal": result.refusal}
    return result.text, {"ok": True, "status": result.status, "final_url": result.final_url,
                         "content_type": result.content_type, "total_bytes": result.total_bytes,
                         "replay_class": "observation"}
