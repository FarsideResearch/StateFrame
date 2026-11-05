"""OpenAI-compatible endpoint configuration and a small stdlib client.

The engine does its own provider sends (one sealed body per request); this module only
normalizes what the user typed, reads the API key from the environment, and probes the
server before a run starts so a typo fails in one second instead of on the first turn.

Keys are read from the environment only and never printed, logged or written to disk.
"""
from __future__ import annotations

import json
import ipaddress
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

CHAT_SUFFIX = "/chat/completions"


MAX_JSON_BYTES = 4 * 1024 * 1024


class EndpointError(RuntimeError):
    """The endpoint is unusable for a run (bad URL, unreachable, auth refused)."""


def _read_bounded(response, deadline: float) -> bytes:
    """At most MAX_JSON_BYTES of the body, raising TimeoutError once ``deadline`` passes."""
    data = bytearray()
    read = getattr(response, "read1", None) or response.read
    while len(data) < MAX_JSON_BYTES:
        if time.monotonic() > deadline:
            raise TimeoutError("server response exceeded the probe deadline")
        chunk = read(min(65536, MAX_JSON_BYTES - len(data)))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def _is_loopback(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    return (isinstance(address, ipaddress.IPv4Address) and address.is_loopback) or str(address) == "::1"


def _require_local(url: str, allow_remote: bool = False) -> None:
    if not _is_loopback(url) and not allow_remote and os.environ.get("STATEFRAME_ALLOW_REMOTE_ENDPOINT") != "1":
        raise EndpointError("local-only release: endpoint must use 127.0.0.0/8, ::1 or localhost; "
                            "remote endpoints require STATEFRAME_ALLOW_REMOTE_ENDPOINT=1")


class _LocalRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_remote: bool = False) -> None:
        self.allow_remote = allow_remote

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_local(newurl, self.allow_remote)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        from stateframe.engine.kernel_redirect_policy import (
            REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE,
            REDIRECT_SECRET_HEADERS,
            scope_redirect_credentials,
        )

        credential_origin = getattr(req, REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE, req.full_url)
        scope_redirect_credentials(
            redirected,
            credential_origin=credential_origin,
            secret_headers=REDIRECT_SECRET_HEADERS,
        )
        return redirected


def normalize_base_url(raw: str) -> str:
    """Return the OpenAI-style base URL (ending in /v1 or the user's own prefix).

    Accepted inputs: ``http://host:8080``, ``http://host:8080/v1``, or a full
    ``.../v1/chat/completions`` URL. A bare host gets ``/v1`` appended, which is what
    llama.cpp, vLLM and OpenAI all serve.
    """
    value = (raw or "").strip()
    if not value:
        raise EndpointError("--endpoint is required (an OpenAI-compatible base URL)")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise EndpointError("--endpoint has an invalid host or port") from error
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise EndpointError("--endpoint must be an http(s) URL, for example http://127.0.0.1:8080/v1")
    if parsed.username or parsed.password:
        raise EndpointError("--endpoint must not embed credentials; use --api-key-env")
    if parsed.query or parsed.fragment:
        raise EndpointError("--endpoint must not carry a query string or fragment")
    if not parsed.hostname or port == 0 or any(c.isspace() or ord(c) < 32 for c in value):
        raise EndpointError("--endpoint has an invalid host or port")
    path = parsed.path.rstrip("/")
    if path.endswith(CHAT_SUFFIX):
        path = path[: -len(CHAT_SUFFIX)]
    if not path:
        path = "/v1"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def redact_url(url: str) -> str:
    """Scheme, host and path only (no userinfo, query or fragment)."""
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_s: float = 10.0
    allow_remote: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
        self.check_locality()
        if os.environ.get("STATEFRAME_ALLOW_REMOTE_ENDPOINT") == "1":
            print("WARNING: remote endpoint override enabled; source code will leave the machine "
                  "when using a remote endpoint. A loopback URL can still forward elsewhere.", file=sys.stderr)

    def check_locality(self) -> None:
        """Validate without DNS resolution, before any transport can run."""
        _require_local(self.base_url, self.allow_remote)

    @classmethod
    def from_args(cls, endpoint: str, model: str, api_key_env: str | None,
                  timeout_s: float = 10.0) -> "Endpoint":
        if not (model or "").strip():
            raise EndpointError("--model is required (the name the server serves)")
        if api_key_env is not None:
            if not api_key_env.replace("_", "").isalnum() or api_key_env[0].isdigit():
                raise EndpointError("--api-key-env must be an environment variable NAME, not a key")
            if api_key_env not in os.environ:
                raise EndpointError(f"--api-key-env {api_key_env} is not set in the environment")
        return cls(normalize_base_url(endpoint), model.strip(), api_key_env, timeout_s)

    @property
    def chat_url(self) -> str:
        return self.base_url + CHAT_SUFFIX

    def api_key(self) -> str:
        """The key, read fresh from the environment. Empty when no variable was named."""
        if self.api_key_env is None:
            return ""
        return os.environ.get(self.api_key_env, "")

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        key = self.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _read_json(self, opener: urllib.request.OpenerDirector, url: str) -> object:
        """GET ``url`` as JSON, retrying once when the server stalls past the timeout.

        A loaded or shared server can stall one request for longer than the timeout and then
        answer the next at once. Only a timeout is retried; an HTTP answer or a refused
        connection fails immediately.
        """
        for attempt in (1, 2):
            request = urllib.request.Request(url, headers=self._headers(), method="GET")
            # The socket timeout bounds each wait, not the whole read: a server that trickles bytes
            # could hold one attempt open forever. The body read stops at a wall-clock deadline, so an
            # attempt is bounded (connect, headers and urllib's capped redirects each by one socket
            # timeout, the body by the deadline plus at most one more wait), not exactly timeout_s.
            deadline = time.monotonic() + self.timeout_s
            try:
                with opener.open(request, timeout=self.timeout_s) as response:
                    return json.loads(_read_bounded(response, deadline))
            except (TimeoutError, urllib.error.URLError) as error:
                timed_out = isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None),
                                                                           TimeoutError)
                if attempt == 2 or not timed_out or isinstance(error, urllib.error.HTTPError):
                    raise
        raise AssertionError("unreachable")

    def get_json(self, path: str) -> object:
        self.check_locality()
        # Do not route loopback requests through an inherited HTTP proxy. Redirects
        # must pass the same local policy before urllib can make another request.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _LocalRedirects(self.allow_remote))
        return self._read_json(opener, self.base_url + path)

    def get_slots(self) -> list[object] | None:
        """Read llama.cpp's root ``/slots`` endpoint when this endpoint is loopback.

        Slot discovery is optional: OpenAI-compatible servers commonly do not expose
        it. In that case return ``None`` so callers can continue without pinning.
        """
        if not _is_loopback(self.base_url):
            return None
        parsed = urllib.parse.urlsplit(self.base_url)
        url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/slots", "", ""))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _LocalRedirects())
        try:
            payload = self._read_json(opener, url)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                ConnectionError, OSError, EndpointError, json.JSONDecodeError, UnicodeError):
            return None
        return payload if isinstance(payload, list) else None

    def probe(self) -> dict[str, object]:
        """Check the server answers and, when it lists models, that ours is one of them.

        Sends no repository content. Returns a small public description of what was found.
        """
        try:
            listing = self.get_json("/models")
        except urllib.error.HTTPError as error:
            if error.code in {401, 403}:
                raise EndpointError(
                    f"endpoint refused the credentials (HTTP {error.code}); check --api-key-env") from None
            if error.code == 404:
                return {"reachable": True, "models_listed": False, "model_found": None}
            raise EndpointError(f"endpoint answered HTTP {error.code} on GET /models") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise EndpointError(f"endpoint unreachable at {redact_url(self.base_url)}: "
                                f"{type(reason).__name__}") from None
        except json.JSONDecodeError:
            raise EndpointError("endpoint GET /models did not return JSON") from None
        names = []
        if isinstance(listing, dict) and isinstance(listing.get("data"), list):
            names = [str(item.get("id")) for item in listing["data"] if isinstance(item, dict)]
        found = (self.model in names) if names else None
        return {"reachable": True, "models_listed": bool(names), "model_found": found,
                "served_models": names[:20]}

    def public(self) -> dict[str, object]:
        """What may be printed or stored: never the key itself."""
        return {"base_url": redact_url(self.base_url), "model": self.model,
                "api_key_env": self.api_key_env, "api_key_present": bool(self.api_key())}
