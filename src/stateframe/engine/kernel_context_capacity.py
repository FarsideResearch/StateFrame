"""Live context-capacity invariant at the request seam (audit 2026-10-05 #8).

Invariant: ``prompt_estimate * 1.02 + max_output <= n_ctx``.  The chars/4
estimate measured 0.98-1.11x of real prompt tokens (median 1.017), so 2% is
the guard band the audit sized.  When a request would overflow we shrink its
output budget; when even the floor does not fit we refuse typed.  Nothing that
overflows is ever sent.

``n_ctx`` comes from ``STATEFRAME_N_CTX`` when set, else one cached probe of the
llama.cpp endpoint (``GET /slots``, then ``GET /props``).  Unknown -> no-op.
``STATEFRAME_CONTEXT_PREFLIGHT=off`` disables the check.
"""
from __future__ import annotations

import http.client
import json
import math
import os
import threading
import urllib.parse
from collections.abc import Mapping

ESTIMATE_MARGIN = 1.02
DEFAULT_MIN_OUTPUT_TOKENS = 512
_PROBE_TIMEOUT_S = 3.0
_OUTPUT_KEYS = ("max_tokens", "max_completion_tokens", "n_predict")
_cache: dict[str, int | None] = {}
_lock = threading.Lock()


class ContextCapacityError(ValueError):
    """The prompt alone leaves less than the output floor: harness-owned, terminal."""

    outcome = "HARNESS_CONTRACT"
    failure_origin = "harness"
    harness_contract = True
    transport_retryable = False
    transport_class = "capacity"


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit() and int(value) > 0:
        return int(value)
    return None


def _base_url(endpoint: str) -> str | None:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _get_json(base: str, path: str, key: str) -> object:
    parsed = urllib.parse.urlsplit(base)
    connection_type = (http.client.HTTPSConnection if parsed.scheme == "https"
                       else http.client.HTTPConnection)
    connection = connection_type(parsed.hostname, parsed.port, timeout=_PROBE_TIMEOUT_S)
    try:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        data = response.read(1024 * 1024)
        if response.status != 200:
            return None
        return json.loads(data)
    finally:
        connection.close()


def _n_ctx_from(payload: object) -> int | None:
    if isinstance(payload, list):  # /slots
        sizes = [_positive_int(slot.get("n_ctx")) for slot in payload if isinstance(slot, Mapping)]
        sizes = [size for size in sizes if size is not None]
        return min(sizes) if sizes else None
    if isinstance(payload, Mapping):  # /props
        settings = payload.get("default_generation_settings")
        if isinstance(settings, Mapping):
            return _positive_int(settings.get("n_ctx"))
        return _positive_int(payload.get("n_ctx"))
    return None


def probe_n_ctx(endpoint: str, key: str = "") -> int | None:
    """One cached probe per server base URL; any failure caches ``None``."""
    base = _base_url(endpoint)
    if base is None:
        return None
    with _lock:
        if base in _cache:
            return _cache[base]
    value: int | None = None
    for path in ("/slots", "/props"):
        try:
            value = _n_ctx_from(_get_json(base, path, key))
        except (OSError, ValueError, http.client.HTTPException):
            value = None
        if value is not None:
            break
    with _lock:
        _cache[base] = value
    return value


def resolve_n_ctx(endpoint: str, key: str = "") -> int | None:
    explicit = _positive_int(os.environ.get("STATEFRAME_N_CTX", ""))
    if explicit is not None:
        return explicit
    return probe_n_ctx(endpoint, key)


def check_capacity(prompt_estimate: int, max_output: int, n_ctx: int, *,
                   min_output: int = DEFAULT_MIN_OUTPUT_TOKENS) -> tuple[str, int]:
    """Return ``("ok"|"shrink"|"refuse", allowed_output)`` for the invariant."""
    reserved = math.ceil(prompt_estimate * ESTIMATE_MARGIN)
    if reserved + max_output <= n_ctx:
        return "ok", max_output
    allowed = n_ctx - reserved
    if allowed >= min(min_output, max_output):
        return "shrink", allowed
    return "refuse", max(allowed, 0)


def enforce_context_capacity(capture: object, body: dict[str, object], *,
                             endpoint: str | None = None, key: str = "") -> None:
    """Shrink ``body``'s output budget or raise :class:`ContextCapacityError`."""
    from kernel_endpoint_size import ENDPOINT_SIZE
    endpoint = endpoint if endpoint is not None else getattr(capture, "capacity_endpoint", None)
    live_slot = None
    if ENDPOINT_SIZE.configured:
        if not isinstance(endpoint, str):
            raise ContextCapacityError(
                f"STATEFRAME_CONTEXT_TOKENS={ENDPOINT_SIZE.context_tokens} requires a live "
                f"server slot of {ENDPOINT_SIZE.context_tokens} tokens; endpoint is missing. "
                "Set a serving endpoint or unset the context setting.")
        live_slot = probe_n_ctx(endpoint, key)
        if live_slot is None or live_slot != ENDPOINT_SIZE.context_tokens:
            raise ContextCapacityError(
                f"STATEFRAME_CONTEXT_TOKENS={ENDPOINT_SIZE.context_tokens} requires a live "
                f"server slot of {ENDPOINT_SIZE.context_tokens} tokens; observed "
                f"{live_slot}. Set the server slot to {ENDPOINT_SIZE.context_tokens} "
                "or unset the context setting.")
    if os.environ.get("STATEFRAME_CONTEXT_PREFLIGHT", "on").strip().lower() in {
            "off", "0", "false", "no"}:
        return
    output_key = next((name for name in _OUTPUT_KEYS if _positive_int(body.get(name))), None)
    if output_key is None or not isinstance(body.get("messages"), list):
        return
    if not isinstance(endpoint, str):
        return
    n_ctx = live_slot if ENDPOINT_SIZE.configured else resolve_n_ctx(endpoint, key)
    if n_ctx is None:
        return
    from kernel_measurement import _prompt_token_estimate
    estimate = _prompt_token_estimate(body)
    max_output = int(body[output_key])
    min_output = _positive_int(os.environ.get("STATEFRAME_MIN_OUTPUT_TOKENS", "")) or \
        DEFAULT_MIN_OUTPUT_TOKENS
    verdict, allowed = check_capacity(estimate, max_output, n_ctx, min_output=min_output)
    if verdict == "ok":
        return
    journal = getattr(capture, "journal", None)
    fields = dict(request_id=getattr(capture, "request_id", None), step=getattr(capture, "step", None),
                  prompt_estimate=estimate, margin=ESTIMATE_MARGIN, max_output=max_output,
                  allowed_output=allowed, n_ctx=n_ctx, output_key=output_key, dispatch=False)
    if verdict == "shrink":
        body[output_key] = allowed
        if journal is not None:
            journal.emit("context_capacity_shrunk", **fields)
        return
    if journal is not None:
        journal.emit("harness_contract_error", producer="kernel_context_capacity",
                     problems=[f"prompt estimate {estimate} leaves {allowed} < "
                               f"{min(min_output, max_output)} output tokens of n_ctx {n_ctx}"],
                     outcome="HARNESS_CONTRACT", failure_origin="harness",
                     transport_class="capacity", **fields)
    raise ContextCapacityError(
        f"request overflows n_ctx {n_ctx}: estimate {estimate}*{ESTIMATE_MARGIN} + "
        f"{max_output} output; only {allowed} output tokens fit")
