"""Credential handling shared by the existing HTTP redirect handlers."""
from __future__ import annotations

import ipaddress
import sys
import urllib.parse
import urllib.request
from collections.abc import Iterable


if __name__ == "kernel_redirect_policy":
    sys.modules.setdefault("stateframe.engine.kernel_redirect_policy", sys.modules[__name__])
elif __name__ == "stateframe.engine.kernel_redirect_policy":
    sys.modules.setdefault("kernel_redirect_policy", sys.modules[__name__])


REDIRECT_SECRET_HEADERS = (
    "Authorization",
    "Proxy-Authorization",
    "Cookie",
    "Cookie2",
    "X-Api-Key",
    "Api-Key",
    "X-Auth-Token",
)
REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE = "_stateframe_redirect_credential_origin"
# None is terminal: later hops stay stripped even if the target returns to the origin.
_UNSET = object()

Origin = tuple[str, str, int]


def _normalized_host(host: str) -> str | None:
    if any(char.isspace() or ord(char) < 33 or char in "/?#@[]\\" for char in host):
        return None
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        if ":" in host:
            return None
        try:
            return host.encode("idna").decode("ascii").lower()
        except UnicodeError:
            return None


def _origin(value: object) -> Origin | None:
    if isinstance(value, tuple):
        if len(value) != 3:
            return None
        scheme, host, port = value
        if (not isinstance(scheme, str) or not isinstance(host, str)
                or type(port) is not int or not 1 <= port <= 65535):
            return None
    elif isinstance(value, str):
        try:
            parsed = urllib.parse.urlsplit(value)
            if parsed.username is not None or parsed.password is not None:
                return None
            scheme = parsed.scheme
            host = parsed.hostname
            port = parsed.port
        except ValueError:
            return None
        if not host:
            return None
        if port is None:
            port = 443 if scheme.lower() == "https" else 80
    else:
        return None

    scheme = scheme.lower()
    if scheme not in {"http", "https"} or not host:
        return None
    normalized_host = _normalized_host(host)
    if normalized_host is None:
        return None
    if type(port) is not int or not 1 <= port <= 65535:
        return None
    return scheme, normalized_host, port


def _secret_header_names(secret_headers: Iterable[str]) -> frozenset[str]:
    if isinstance(secret_headers, (str, bytes)):
        raise TypeError("secret_headers must be an iterable of header names")
    try:
        names = tuple(secret_headers)
    except TypeError as error:
        raise TypeError("secret_headers must be an iterable of header names") from error
    if not names:
        raise ValueError("secret_headers must not be empty")
    if any(not isinstance(name, str) or not name or name != name.strip()
           or any(char in name for char in "\r\n:") for name in names):
        raise ValueError("secret_headers must contain valid, nonempty header names")
    return frozenset(name.casefold() for name in names)


def scope_redirect_credentials(
        request: urllib.request.Request, *, credential_origin: object,
        secret_headers: Iterable[str]) -> urllib.request.Request:
    """Strip registered credentials after their first redirect away from their origin."""
    if not isinstance(request, urllib.request.Request):
        raise TypeError("redirect request must be a urllib Request")

    names = _secret_header_names(secret_headers)
    stored_origin = getattr(request, REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE, _UNSET)
    origin = _origin(credential_origin if stored_origin is _UNSET else stored_origin)
    target_origin = _origin(request.full_url)

    if origin is None or target_origin is None or target_origin != origin:
        for collection in (request.headers, request.unredirected_hdrs):
            for name in tuple(collection):
                if not isinstance(name, str):
                    raise TypeError("redirect request header names must be strings")
                if name.casefold() in names:
                    del collection[name]
        setattr(request, REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE, None)
    else:
        setattr(request, REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE, origin)
    return request
