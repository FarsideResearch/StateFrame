"""Credential handling shared by HTTP redirects and explicitly scoped openers."""
from __future__ import annotations

import copy
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
CHILD_DIRECT_OPENER_ATTRIBUTE = "_stateframe_child_direct_opener"
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


def _remove_proxy_authorization(request: urllib.request.Request) -> None:
    for headers in (request.headers, request.unredirected_hdrs):
        for name in tuple(headers):
            if name.casefold() == "proxy-authorization":
                del headers[name]


def _proxy_without_credentials(proxy: str) -> str:
    """Keep the proxy address but remove URL userinfo for an untrusted hop."""
    parsed = urllib.parse.urlsplit(proxy)
    if parsed.netloc:
        if "@" not in parsed.netloc:
            return proxy
        return urllib.parse.urlunsplit(parsed._replace(netloc=parsed.netloc.rsplit("@", 1)[1]))
    return proxy.rsplit("@", 1)[1] if "@" in proxy else proxy


class _CredentialScopedProxyHandler(urllib.request.ProxyHandler):
    """Apply proxy URL credentials only while the request remains at its origin."""

    def __init__(self, credential_origin: Origin) -> None:
        self.credential_origin = credential_origin
        super().__init__()  # The sole environment-proxy read in this contract.

    def proxy_open(self, request, proxy, proxy_type):
        # urllib runs proxy_open after redirect_request and can add this header again.
        _remove_proxy_authorization(request)
        stored_origin = getattr(request, REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE, _UNSET)
        allowed_origin = self.credential_origin if stored_origin is _UNSET else _origin(stored_origin)
        if _origin(request.full_url) != allowed_origin or allowed_origin is None:
            proxy = _proxy_without_credentials(proxy)
        return super().proxy_open(request, proxy, proxy_type)


class _ProxyHeaderScrubber(urllib.request.BaseHandler):
    """Prevent caller-supplied proxy credentials from reaching an origin."""

    handler_order = 99

    def http_request(self, request):
        _remove_proxy_authorization(request)
        return request

    https_request = http_request


def _scoped_redirect_handler(
        handler: urllib.request.HTTPRedirectHandler, credential_origin: Origin,
) -> urllib.request.HTTPRedirectHandler:
    scoped = copy.copy(handler)
    original = scoped.redirect_request

    def redirect_request(req, fp, code, msg, headers, newurl):
        redirected = original(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        origin = getattr(req, REDIRECT_CREDENTIAL_ORIGIN_ATTRIBUTE, credential_origin)
        return scope_redirect_credentials(
            redirected, credential_origin=origin, secret_headers=REDIRECT_SECRET_HEADERS,
        )

    scoped.redirect_request = redirect_request
    scoped.handler_order = min(scoped.handler_order, urllib.request.HTTPRedirectHandler.handler_order - 1)
    return scoped


def build_credential_scoped_opener(
        *, proxy_mode: str, credential_origin: object | None = None,
        handlers: Iterable[urllib.request.BaseHandler] = (),
) -> urllib.request.OpenerDirector:
    """Build an opener with explicit proxy routing and redirect credential scope.

    ``direct`` never consults proxy environment variables. ``environment`` uses
    urllib's proxy discovery and NO_PROXY bypass. The engine child's installed
    direct opener makes even an environment request resolve to direct mode.
    """
    if proxy_mode not in {"direct", "environment"}:
        raise ValueError("proxy_mode must be 'direct' or 'environment'")
    origin = _origin(credential_origin) if credential_origin is not None else None
    if credential_origin is not None and origin is None:
        raise ValueError("credential_origin must be an HTTP(S) URL or origin")
    selected = tuple(handlers)
    if any(not isinstance(handler, urllib.request.BaseHandler)
           or isinstance(handler, urllib.request.ProxyHandler) for handler in selected):
        raise TypeError("handlers must be urllib handlers other than ProxyHandler")

    child_opener = getattr(urllib.request, "_opener", None)
    use_environment = (proxy_mode == "environment"
                       and getattr(child_opener, CHILD_DIRECT_OPENER_ATTRIBUTE, False) is not True)
    if use_environment and origin is None:
        raise ValueError("environment mode requires credential_origin")

    # The literal handler also keeps the existing proxy-free opener guard true.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    opener.add_handler(_ProxyHeaderScrubber())
    if use_environment:
        assert origin is not None
        opener.add_handler(_CredentialScopedProxyHandler(origin))

    has_redirect_handler = False
    for handler in selected:
        configured = copy.copy(handler)
        if use_environment and isinstance(configured, urllib.request.HTTPRedirectHandler):
            assert origin is not None
            configured = _scoped_redirect_handler(configured, origin)
            has_redirect_handler = True
        elif isinstance(configured, (urllib.request.HTTPHandler, urllib.request.HTTPSHandler,
                                     urllib.request.HTTPRedirectHandler)):
            # Defaults were installed above; custom transports must run first.
            configured.handler_order = min(configured.handler_order, 499)
        opener.add_handler(configured)
    if use_environment and not has_redirect_handler:
        assert origin is not None
        opener.add_handler(_scoped_redirect_handler(urllib.request.HTTPRedirectHandler(), origin))
    return opener
