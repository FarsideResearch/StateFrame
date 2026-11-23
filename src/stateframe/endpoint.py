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
from dataclasses import dataclass, field
from types import ModuleType

CHAT_SUFFIX = "/chat/completions"


MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES_ENV = "STATEFRAME_ENDPOINT_MAX_JSON_BYTES"
SLOT_OBSERVATION_TTL = 1.0
_PROVIDER_FACT_TTL = 30.0
_PROVIDER_EPOCH_TIMEOUT = 0.25


class EndpointError(RuntimeError):
    """The endpoint is unusable for a run (bad URL, unreachable, auth refused)."""


class EndpointProbeLimitError(EndpointError):
    """A probe response exceeds its configured JSON byte limit."""

    def __init__(self, path: str, limit: int) -> None:
        self.path, self.limit = path, limit
        super().__init__(f"endpoint GET {path} response exceeds {limit}-byte limit ({MAX_JSON_BYTES_ENV})")


class EndpointProbeConfigError(EndpointError):
    """The endpoint probe JSON byte limit is invalid."""


def _max_json_bytes() -> int:
    value = os.environ.get(MAX_JSON_BYTES_ENV, str(MAX_JSON_BYTES)).strip()
    try:
        limit = int(value)
    except ValueError:
        limit = 0
    if not value.isascii() or not value.isdecimal() or limit < 1:
        raise EndpointProbeConfigError(f"{MAX_JSON_BYTES_ENV} must be a positive integer")
    return limit


def _is_timeout(error: BaseException) -> bool:
    return isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None), TimeoutError)


def _report_timeout(what: str, deadline_s: float | None, elapsed_s: float | None = None) -> None:
    """Write one reason line for a probe timeout. Logging must never change the probe outcome."""
    try:
        try:
            from kernel_reason_log import emit_timeout
        except ImportError:
            from stateframe.engine.kernel_reason_log import emit_timeout
        emit_timeout("endpoint", what=what, why="endpoint probe timed out",
                     deadline_s=deadline_s, elapsed_s=elapsed_s, cause_key="endpoint:timeout")
    except Exception:  # noqa: BLE001 - the log helper is best effort by contract
        return


def _read_bounded(response, deadline: float, limit: int = MAX_JSON_BYTES) -> bytes:
    """Read at most one byte over the limit to distinguish a complete body from a clipped one."""
    data = bytearray()
    read = getattr(response, "read1", None) or response.read
    while len(data) <= limit:
        if time.monotonic() > deadline:
            raise TimeoutError("server response exceeded the probe deadline")
        chunk = read(min(65536, limit + 1 - len(data)))
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


def _facts_digest(facts: object) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(
        facts, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")).hexdigest()


def _model_facts(record: object, model: str) -> bool:
    if not isinstance(record, dict) or record.keys() != {
        "reachable", "models_listed", "model_found", "served_models",
    }:
        return False
    names = record["served_models"]
    return (record["reachable"] is True and record["models_listed"] is True
            and record["model_found"] is True
            and isinstance(names, list) and 0 < len(names) <= 20
            and all(isinstance(name, str) and 0 < len(name) <= 512 for name in names)
            and (model in names or len(names) == 20))


def _static_slot_facts(slots: object) -> list[dict[str, int]] | None:
    if not isinstance(slots, list) or not 0 < len(slots) <= 256:
        return None
    result = []
    seen = set()
    for slot in slots:
        if not isinstance(slot, dict):
            return None
        slot_id = slot.get("id")
        if type(slot_id) is not int or slot_id < 0:
            slot_id = slot.get("id_slot")
        size = slot.get("n_ctx")
        if (type(slot_id) is not int or slot_id < 0 or slot_id in seen
                or type(size) is not int or size <= 0):
            return None
        seen.add(slot_id)
        result.append({"id": slot_id, "n_ctx": size})
    return result


@dataclass
class ProviderEpoch:
    """A live, launch-local commitment to selected static preflight facts.

    The provider gives a fresh instance ID on restart and changes its configuration
    or model revision on relevant reconfiguration. Each fact digest also binds the
    exact canonical JSON projection, including when facts change within one epoch.
    The store supplies installation integrity. The scope binds the host and boot,
    and records carry the monotonic time of the original observation.
    """

    endpoint: Endpoint
    observed_at: float
    _scope: str
    _auth: str | None = field(repr=False)
    _digests: dict[str, str] = field(repr=False)
    _store: ModuleType = field(repr=False)
    _invalidated: bool = field(default=False, init=False, repr=False)

    def matches(self, endpoint: str, key: str) -> bool:
        from stateframe.engine.kernel_request_auth_identity import auth_binding_for_key

        return (endpoint in (self.endpoint.base_url, self.endpoint.chat_url)
                and self._auth == auth_binding_for_key(key or None))

    def invalidate(self) -> None:
        self._invalidated = True

    @property
    def invalidated(self) -> bool:
        return self._invalidated

    @property
    def covers_slots(self) -> bool:
        return "slots" in self._digests

    def _key(self, kind: str) -> str | None:
        if self._invalidated or kind not in self._digests:
            return None
        return _facts_digest([self._scope, kind])

    def _current(self) -> bool:
        return not self._invalidated and 0 <= time.monotonic() - self.observed_at < SLOT_OBSERVATION_TTL

    def lookup(self, kind: str) -> object:
        if not self._current():
            return None
        fingerprint = self._key(kind)
        if fingerprint is None:
            return None
        max_age = SLOT_OBSERVATION_TTL if kind == "slots" else _PROVIDER_FACT_TTL
        try:
            entry = self._store.lookup("provider-" + kind, fingerprint, max_age=max_age)
            if not isinstance(entry, dict) or entry.keys() != {"observed_ns", "facts"}:
                return None
            observed_ns, record = entry["observed_ns"], entry["facts"]
            if (type(observed_ns) is int and 0 <= time.monotonic_ns() - observed_ns < max_age * 1e9
                    and _facts_digest(record) == self._digests[kind]
                    and self._current()):
                return record
        except (OSError, ValueError, TypeError, RecursionError):
            pass
        return None

    def observe(self, kind: str, record: object, *, publish: bool = True) -> None:
        expected = self._digests.get(kind)
        if expected is None:
            return
        if record is None or _facts_digest(record) != expected:
            self.invalidate()
            return
        if not publish:
            return
        # The live fact can seed a later lookup even if its probe took longer
        # than the commitment's admission window. That lookup needs a new proof.
        fingerprint = self._key(kind)
        if fingerprint is not None:
            try:
                self._store.publish("provider-" + kind, fingerprint, {
                    "observed_ns": time.monotonic_ns(), "facts": record,
                })
            except (OSError, ValueError, TypeError):
                pass

    def observe_slots(self, slots: object, *, publish: bool = True) -> None:
        self.observe("slots", _static_slot_facts(slots), publish=publish)


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
        _max_json_bytes()
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
        limit = _max_json_bytes()
        for attempt in (1, 2):
            request = urllib.request.Request(url, headers=self._headers(), method="GET")
            # The socket timeout bounds each wait, not the whole read: a server that trickles bytes
            # could hold one attempt open forever. The body read stops at a wall-clock deadline, so an
            # attempt is bounded (connect, headers and urllib's capped redirects each by one socket
            # timeout, the body by the deadline plus at most one more wait), not exactly timeout_s.
            deadline = time.monotonic() + self.timeout_s
            try:
                with opener.open(request, timeout=self.timeout_s) as response:
                    body = _read_bounded(response, deadline, limit)
                    if len(body) > limit:
                        path = urllib.parse.urlsplit(url).path or "/"
                        raise EndpointProbeLimitError(path, limit)
                    return json.loads(body)
            except (TimeoutError, urllib.error.URLError) as error:
                timed_out = isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None),
                                                                           TimeoutError)
                if timed_out and attempt == 2:
                    _report_timeout(f"GET {redact_url(url)}", self.timeout_s,
                                    time.monotonic() - (deadline - self.timeout_s))
                if attempt == 2 or not timed_out or isinstance(error, urllib.error.HTTPError):
                    raise
        raise AssertionError("unreachable")

    def get_json(self, path: str) -> object:
        self.check_locality()
        # Do not route loopback requests through an inherited HTTP proxy. Redirects
        # must pass the same local policy before urllib can make another request.
        from stateframe.engine.kernel_redirect_policy import build_credential_scoped_opener
        opener = build_credential_scoped_opener(
            proxy_mode="direct", handlers=(_LocalRedirects(self.allow_remote),),
        )
        return self._read_json(opener, self.base_url + path)

    def provider_epoch(self, route: str) -> ProviderEpoch | None:
        """Read an optional provider commitment for the selected model and auth.

        GET <base>/provider_epoch?model=NAME&nonce=NONCE returns version (1),
        the echoed nonce, instance, configuration, model_revision, and facts
        (kind to lowercase SHA-256).
        Emit a fact only if its discovery request would currently succeed for
        this caller. Canonicalization is the compact, sorted, ASCII JSON used
        by _facts_digest. Hash the public model probe record, the capability
        record with the default preflight settings, or the ordered list of
        static id/n_ctx slot pairs, respectively. This proof is host-only.
        """
        from . import _startup_optimization_enabled

        if not _startup_optimization_enabled():
            return None

        from http.client import HTTPException
        from importlib import import_module

        try:
            send1_probe_cache = import_module("stateframe.send1_probe_cache")
        except ModuleNotFoundError as error:
            if error.name != "stateframe.send1_probe_cache":
                raise
            return None
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme != "http" or parsed.hostname == "localhost" or not _is_loopback(self.base_url):
            return None
        host = send1_probe_cache.host_fingerprint()
        if host is None:
            return None
        try:
            time_namespace = os.readlink("/proc/self/ns/time")
        except OSError:
            return None
        from stateframe.engine.kernel_request_auth_identity import auth_binding_for_key

        timeout = min(self.timeout_s, _PROVIDER_EPOCH_TIMEOUT)
        if timeout <= 0:
            return None
        observed_at = time.monotonic()
        headers = self._headers()
        auth = auth_binding_for_key(headers.get("Authorization", "").removeprefix("Bearer ") or None)
        headers["Cache-Control"] = "no-cache, no-store"
        nonce = os.urandom(16).hex()
        query = urllib.parse.urlencode({"model": self.model, "nonce": nonce})
        request = urllib.request.Request(self.base_url + "/provider_epoch?" + query, headers=headers)

        # The commitment must come from this exact loopback HTTP endpoint.
        opener = urllib.request.OpenerDirector()
        opener.add_handler(urllib.request.HTTPHandler())
        try:
            with opener.open(request, timeout=timeout) as response:
                if response.status != 200:
                    return None
                body = _read_bounded(response, observed_at + timeout, 2049)
                if response.length not in (None, 0):
                    return None
            if len(body) > 2048 or time.monotonic() - observed_at >= timeout:
                return None
            proof = json.loads(body)
        except (OSError, HTTPException, ValueError, RecursionError):
            return None
        if (not isinstance(proof, dict) or proof.get("nonce") != nonce
                or type(proof.get("version")) is not int or proof["version"] != 1):
            return None
        revisions = [proof.get(name) for name in ("instance", "configuration", "model_revision")]
        if not all(isinstance(value, str) and 0 < len(value) <= 128 and value.isascii()
                   and value.isprintable() and value.strip() == value for value in revisions):
            return None
        facts = proof.get("facts")
        if (not isinstance(facts, dict) or not facts or facts.keys() - {"models", "capabilities", "slots"}
                or not all(isinstance(digest, str) and len(digest) == 64
                           and all(char in "0123456789abcdef" for char in digest)
                           for digest in facts.values())):
            return None
        scope = _facts_digest([
            "provider-preflight-v1", host, time_namespace, self.base_url, self.model, route,
            self.api_key_env, auth, "direct-literal-loopback-http", self.allow_remote, revisions, facts,
        ])
        return ProviderEpoch(self, observed_at, scope, auth, facts, send1_probe_cache)

    def _matching_epoch(self, epoch: ProviderEpoch | None) -> ProviderEpoch | None:
        if (epoch is not None and self.model == epoch.endpoint.model
                and self.api_key_env == epoch.endpoint.api_key_env
                and self.allow_remote == epoch.endpoint.allow_remote
                and epoch.matches(self.base_url, self.api_key())):
            return epoch
        return None

    @dataclass(frozen=True)
    class SlotObservation:
        """One live slot response shared by the adjacent startup checks."""

        slots: list[object]
        observed_at: float
        base_url: str
        api_key: str = field(repr=False)
        model: str | None = None

        def current_slots(self, endpoint: Endpoint) -> list[object] | None:
            # Startup may stall between checks. Limit how long idle status can be reused.
            if (0 <= time.monotonic() - self.observed_at < SLOT_OBSERVATION_TTL
                    and self.base_url == endpoint.base_url and self.api_key == endpoint.api_key()
                    and (self.model is None or self.model == endpoint.model)):
                return self.slots
            return None

    def get_slots(self, *, epoch: ProviderEpoch | None = None, static: bool = False) -> list[object] | None:
        """Read llama.cpp's root ``/slots`` endpoint when this endpoint is loopback.

        Slot discovery is optional: OpenAI-compatible servers commonly do not expose
        it. In that case return ``None`` so callers can continue without pinning.
        The static view is for capacity and explicit membership checks. Automatic
        slot selection uses the default live view, which includes occupancy.
        """
        epoch = self._matching_epoch(epoch)
        if epoch is not None and static:
            cached = epoch.lookup("slots")
            if cached is not None and _static_slot_facts(cached) == cached:
                return cached
        if not _is_loopback(self.base_url):
            return None
        parsed = urllib.parse.urlsplit(self.base_url)
        url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/slots", "", ""))
        from stateframe.engine.kernel_redirect_policy import build_credential_scoped_opener
        opener = build_credential_scoped_opener(proxy_mode="direct", handlers=(_LocalRedirects(),))
        try:
            payload = self._read_json(opener, url)
        except EndpointProbeConfigError:
            raise
        except EndpointProbeLimitError as error:
            if epoch is not None:
                epoch.invalidate()
            print(f"stateframe: warning: optional endpoint GET {error.path} response exceeds "
                  f"{error.limit}-byte limit; ignoring probe", file=sys.stderr, flush=True)
            return None
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                ConnectionError, OSError, EndpointError, json.JSONDecodeError, UnicodeError) as error:
            if epoch is not None:
                epoch.invalidate()
            if _is_timeout(error):
                _report_timeout(f"optional slots probe {redact_url(url)}", self.timeout_s)
            return None
        if epoch is not None and self._matching_epoch(epoch) is not None:
            epoch.observe_slots(payload, publish=static)
        return payload if isinstance(payload, list) else None

    def probe(self, *, epoch: ProviderEpoch | None = None) -> dict[str, object]:
        """Check the server answers and, when it lists models, that ours is one of them.

        Sends no repository content. Returns a small public description of what was found.
        """
        epoch = self._matching_epoch(epoch)
        if epoch is None:
            return self._probe_live()
        cached = epoch.lookup("models")
        if _model_facts(cached, self.model):
            return cached
        try:
            result = self._probe_live()
        except (EndpointError, UnicodeError):
            epoch.invalidate()
            raise
        if result.get("model_found") is not True:
            epoch.invalidate()
        elif self._matching_epoch(epoch) is not None:
            key = self.api_key()
            publishable = _model_facts(result, self.model) and not any(
                key and key in name for name in result["served_models"])
            epoch.observe("models", result if publishable else None)
        return result

    def _probe_live(self) -> dict[str, object]:
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
            if _is_timeout(error):
                _report_timeout(f"GET /models at {redact_url(self.base_url)}", self.timeout_s)
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
