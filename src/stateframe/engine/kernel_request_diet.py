"""Evidence-preserving request-body diet for StateFrame provider requests.

The request budget owns the final ceiling and tool-menu policy.  This module
owns only model-visible message presentation: section attribution, duplicate
and stale-read removal, stage allowlists, and lossless JSON compaction for
host receipts.  A request is rejected before transport when protected content
cannot fit its stage budget.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
import hashlib
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence

from kernel_context import token_estimate
from kernel_measurement import profiled_function_span
from kernel_endpoint_size import ENDPOINT_SIZE


SCHEMA = "stateframe-request-diet/1"
ENDPOINT_CONTEXT_TOKENS = ENDPOINT_SIZE.context_tokens
ENDPOINT_PROMPT_TOKEN_BUDGET = ENDPOINT_SIZE.prompt_budget
INITIAL_BYTES_PER_TOKEN = 2.46
# SPEED 2026-10-22: Stage6 relay RTT is 0.14-0.15 s for tiny bodies and
# 0.68-0.77 s for 90 KB. b1569's ~211 ms and product benchmark's ~203 ms were
# the old 0.2 s timeout firing, not successful-tokenization latency.
TOKENIZE_REQUEST_TIMEOUT_SECONDS = 2.0
TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS = 10.0
TOKENIZE_RELAY_REFERENCE_BYTES = 90_000
TOKENIZE_RELAY_REFERENCE_SECONDS = 0.77
TOKENIZE_RELAY_RTT_SECONDS = 0.15
TOKENIZE_RELAY_LOAD_MARGIN = 3.0
TOKENIZE_SLOW_SECONDS = 1.0  # Telemetry only: never discard a valid count.
TOKENIZE_MAX_CONSECUTIVE_FAILURES = 3
TEMPLATE_OVERHEAD_TOKENS = 1_148
TEMPLATE_OVERHEAD_EVIDENCE = (
    "calibration-batch-927 request d402be5a823d4026a7d6b6066403fe84: "
    "prompt_n 22042 - /tokenize 20894; chat-template "
    "55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c"
)
FALLBACK_SAFETY_MARGIN = 0.10
# The captured GGUF configuration pins a tokenizer hash but omits byte-fallback
# metadata. Keep this conservative rejection bound in diagnostics; admission
# uses the calibrated count plus the independent serialized-byte envelope.
FALLBACK_MAX_TOKENS_PER_WIRE_BYTE = 2
_TOKENIZER_URL_ENV = (
    "LLAMA_CPP_TOKENIZE_URL", "LLAMA_CPP_BASE_URL", "A3B_ENDPOINT",
    "A3B_BASE_URL", "OPENAI_API_BASE", "OPENAI_BASE_URL",
    # 10-08 r2b 932/933: the live harness sets only DS_URL (react_repair.API, the serving endpoint itself);
    # without it /tokenize was never tried and the diet ran uncalibrated bytes. Last: explicit vars win.
    "DS_URL",
)
_TOKENIZER_ENDPOINT_ATTRS = (
    "model_router_effective_endpoint", "effective_endpoint", "endpoint",
    "model_endpoint", "provider_endpoint",
    "api_url", "base_url", "api_base",
)
_MAX_TOKENIZE_RESPONSE_BYTES = 2 * 1024 * 1024
# The cache stores only a digest and integer per exact prompt projection. A
# pressure walk can exceed 64 distinct bodies, then revisit them on another
# budget pass. Keep the complete walk without retaining prompt text.
_EXACT_TOKEN_COUNT_CACHE_ENTRIES = 512


def resolve_tokenize_url(
        runtime: object | None = None,
        capture: object | None = None,
        *, endpoint: str | None = None,
) -> str | None:
    """Resolve the llama.cpp tokenizer route from the request's endpoint."""
    # TOKURL: an explicitly set LLAMA_CPP_TOKENIZE_URL (process env, then an owner's declared environment) wins over every
    # derived endpoint, so tokenization can be co-located while generation stays remote. It only routes /tokenize; model id, template
    # and generation endpoints are untouched. Unset => the original order below, unchanged.
    explicit: list[object] = [os.environ.get("LLAMA_CPP_TOKENIZE_URL")]
    for owner in (capture, runtime):
        try:
            declared = getattr(owner, "declared_environment", None) if owner is not None else None
        except Exception:
            declared = None
        if isinstance(declared, Mapping):
            explicit.append(declared.get("LLAMA_CPP_TOKENIZE_URL"))
    candidates: list[object] = [value for value in explicit if isinstance(value, str) and value.strip()]
    candidates.append(endpoint)
    for owner in (capture, runtime):
        if owner is None:
            continue
        for name in _TOKENIZER_ENDPOINT_ATTRS:
            try:
                candidates.append(getattr(owner, name, None))
            except Exception:
                continue
        for name in ("backend_identity", "provider_config", "model_config"):
            try:
                value = getattr(owner, name, None)
            except Exception:
                continue
            if isinstance(value, Mapping):
                candidates.extend(value.get(key) for key in _TOKENIZER_ENDPOINT_ATTRS)
        try:
            declared = getattr(owner, "declared_environment", None)
        except Exception:
            declared = None
        if isinstance(declared, Mapping):
            candidates.extend(declared.get(name) for name in _TOKENIZER_URL_ENV)
    candidates.extend(os.environ.get(name) for name in _TOKENIZER_URL_ENV)

    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        raw = candidate.strip()
        try:
            parsed = urllib.parse.urlsplit(raw)
            hostname = parsed.hostname
            has_credentials = parsed.username is not None or parsed.password is not None
        except ValueError:
            continue
        if parsed.scheme not in {"http", "https"} or not hostname or has_credentials:
            continue
        if parsed.path.rstrip("/").endswith("/tokenize"):
            return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
        path = parsed.path.rstrip("/")
        for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
            if path.endswith(suffix):
                path = path[:-len(suffix)]
                break
        tokenize_path = f"{path}/tokenize" if path else "/tokenize"
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, tokenize_path, "", ""))
    return None


def tokenize_route_label(endpoint: object) -> str | None:
    """scheme://host[:port] of a tokenize URL, with path, query and credentials removed."""
    if not isinstance(endpoint, str) or not endpoint:
        return None
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        return None
    if not parsed.scheme or not parsed.hostname:
        return None
    return f"{parsed.scheme}://{parsed.hostname}" + (f":{port}" if port else "")


def _tokenization_content(body: Mapping[str, object]) -> str:
    from kernel_request_measurement import prompt_projection
    return prompt_projection(body)


def tokenizer_contract(runtime: object | None, capture: object | None) -> str:
    """Known served/configured tokenizer identity, without endpoint secrets."""
    fields = ("model", "server_build", "tokenizer", "chat_template")
    manifest = getattr(getattr(capture, "journal", None), "manifest", None)
    contract = {key: manifest[key] for key in fields if isinstance(manifest, Mapping) and key in manifest}
    for owner in (runtime, capture):
        backend = getattr(owner, "backend_identity", None)
        if isinstance(backend, Mapping):
            contract.update({key: backend[key] for key in fields if key in backend})
    return json.dumps(contract, sort_keys=True, separators=(",", ":"))


def _http_tokenize(content: str, url: str, *, deadline: float | None = None) -> Sequence[int]:
    payload = json.dumps({
        "content": content, "add_special": False, "parse_special": True,
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST",
    )
    # Budget the actual UTF-8 JSON payload, including escaping. Three times
    # SPEED's upper measured relay cost covers the b1598 134 KB load case.
    timeout = min(TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS, max(
        TOKENIZE_REQUEST_TIMEOUT_SECONDS,
        TOKENIZE_RELAY_RTT_SECONDS + TOKENIZE_RELAY_LOAD_MARGIN
        * TOKENIZE_RELAY_REFERENCE_SECONDS * len(payload) / TOKENIZE_RELAY_REFERENCE_BYTES,
    ))
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("tokenizer request deadline expired")
        timeout = min(timeout, remaining)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(_MAX_TOKENIZE_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_TOKENIZE_RESPONSE_BYTES:
        raise ValueError("oversized tokenizer response")
    decoded = json.loads(raw)
    tokens = decoded.get("tokens") if isinstance(decoded, Mapping) else None
    if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("invalid llama.cpp tokenizer response")
    return tokens


class RequestTokenMeter:
    """Run-level cache/calibration with monotone fallback within each request."""

    def __init__(
            self, tokenizer: Callable[[str], object] | None = None,
            *, endpoint: str | None = None,
            template_overhead_tokens: int = TEMPLATE_OVERHEAD_TOKENS,
    ) -> None:
        if type(template_overhead_tokens) is not int or template_overhead_tokens < 0:
            raise ValueError("template overhead must be a non-negative integer")
        self._tokenizer = tokenizer
        endpoint = resolve_tokenize_url(endpoint=endpoint) if endpoint else None
        self.endpoint = endpoint
        self.template_overhead_tokens = template_overhead_tokens
        self._availability = (
            "unknown" if tokenizer is not None or endpoint else "unavailable"
        )
        self._failure_reason = "endpoint_unavailable" if endpoint is None and tokenizer is None else None
        self._last_latency_ms: float | None = None
        self._last_attempts = 0
        self._consecutive_failures = 0
        self._fallback_pending = False
        self._fallback_reported = False
        self._request_id: str | None = None
        self._request_aliases: dict[str, str] = {}
        self._tokenizer_calls_by_request: dict[str | None, int] = {}
        self._deadline: float | None = None
        self._namespace: str | None = None
        self.provider_identity = "{}"
        # Keep failed request identities until this run ends: replacement
        # validation may revisit an earlier request after a newer one starts.
        self._request_failures: dict[str, tuple[str | None, float | None, int, bool, int | None]] = {}
        self._minimum_bytes_per_token: float | None = None
        self._calibration_samples = 0
        self._last_fallback_rejection_bound: int | None = None
        self._pending_bytes: OrderedDict[str, int] = OrderedDict()
        self._attached_journals: set[int] = set()
        self._subscriber_name = f"request_token_meter_{id(self):x}"
        self._journal: object | None = None
        self._observer_failure_count = 0
        self._pending_observer_failures = 0
        self._last_observer_failure_type: str | None = None
        self._last_observer_event_kind: str | None = None

    @property
    def measure_method(self) -> str:
        return "tokenize" if self._availability == "available" else "calibrated_bytes"

    def bind_request(self, request_id: str, logical_request_id: str,
                     *, deadline: float | None = None) -> None:
        self._request_aliases[request_id] = logical_request_id
        same_request = self._request_id == logical_request_id
        self.begin_request(logical_request_id)
        if not same_request:
            self._deadline = deadline
        elif deadline is not None:
            self._deadline = deadline if self._deadline is None else min(self._deadline, deadline)

    def begin_request(self, request_id: object) -> None:
        """Retry transient unavailability only at a new request boundary."""
        if isinstance(request_id, str):
            request_id = self._request_aliases.get(request_id, request_id)
        if not isinstance(request_id, str) or not request_id or request_id == self._request_id:
            return
        if self._request_id is not None and self._availability == "unavailable":
            self._request_failures[self._request_id] = (
                self._failure_reason, self._last_latency_ms,
                self._consecutive_failures, self._fallback_reported,
                self._last_fallback_rejection_bound,
            )
        self._request_id = request_id
        self._deadline = None
        self._last_attempts = 0
        failure = self._request_failures.get(request_id)
        if failure is not None:
            self._availability = "unavailable"
            (self._failure_reason, self._last_latency_ms,
             self._consecutive_failures, self._fallback_reported,
             self._last_fallback_rejection_bound) = failure
            self._fallback_pending = True
            return
        configured = self._tokenizer is not None or self.endpoint is not None
        self._availability = "unknown" if configured else "unavailable"
        self._failure_reason = None if configured else "endpoint_unavailable"
        self._last_latency_ms = None
        self._last_attempts = 0
        self._consecutive_failures = 0
        self._last_fallback_rejection_bound = None
        self._fallback_pending = False
        self._fallback_reported = False

    def tokenizer_identity(self, body: Mapping[str, object], *, endpoint: str | None = None) -> str:
        from kernel_request_measurement import PROJECTION_VERSION, digest
        contract = {
            "endpoint": endpoint or self.endpoint, "model": body.get("model"),
            "template": body.get("chat_template"),
            "template_options": body.get("chat_template_kwargs"),
            "projection": PROJECTION_VERSION,
            "template_overhead": self.template_overhead_tokens,
            "template_reference": TEMPLATE_OVERHEAD_EVIDENCE,
            "provider_contract": self.provider_identity,
            "injected_tokenizer": id(self._tokenizer) if self._tokenizer is not None else None,
        }
        return digest(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())

    def _select_namespace(self, body: Mapping[str, object]) -> None:
        namespace = self.tokenizer_identity(body)
        if self._namespace is not None and namespace != self._namespace:
            self.__dict__.pop("_count_cache", None)
            self._minimum_bytes_per_token = None
            self._calibration_samples = 0
            self._pending_bytes.clear()
        self._namespace = namespace

    def bind_measurement(self, body: Mapping[str, object], *, tokens: int, method: str,
                         request_id: str, max_request_bytes: int, max_prompt_tokens: int):
        from kernel_request_measurement import RequestMeasurement
        return RequestMeasurement.bind(
            body, tokenizer_identity=self.tokenizer_identity(body),
            request_id=request_id, logical_request_id=self._request_id or request_id,
            tokens=tokens, method=method, template_overhead_tokens=self.template_overhead_tokens,
            max_request_bytes=max_request_bytes, max_prompt_tokens=max_prompt_tokens,
        )

    def _count_tokens(self, content: str) -> int:
        # Identical content is tokenized once per meter: re-measuring an unchanged (or normalized-to-identical)
        # request must not cost a second /tokenize round trip on the request path.
        key = hashlib.sha256(content.encode("utf-8")).hexdigest()
        cache = self.__dict__.setdefault("_count_cache", OrderedDict())
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        count = self._count_tokens_uncached(content)
        cache[key] = count
        while len(cache) > _EXACT_TOKEN_COUNT_CACHE_ENTRIES:
            cache.popitem(last=False)
        return count

    def _count_tokens_uncached(self, content: str) -> int:
        key = self._request_id
        self._tokenizer_calls_by_request[key] = self._tokenizer_calls_by_request.get(key, 0) + 1
        if self._tokenizer is not None:
            result = self._tokenizer(content)
        elif self.endpoint is not None:
            result = _http_tokenize(content, self.endpoint, deadline=self._deadline)
        else:
            raise OSError("tokenizer endpoint unavailable")
        if type(result) is int:
            count = result
        elif isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
            if any(type(token) is not int or token < 0 for token in result):
                raise ValueError("invalid injected tokenizer response")
            count = len(result)
        else:
            raise TypeError("tokenizer must return a token count or token sequence")
        if count < 0:
            raise ValueError("tokenizer returned a negative count")
        return count

    def count_text_exact(self, content: str) -> int | None:
        """Count cache blocks within the current request's bounded retry policy.

        Raw block counts omit the prompt projection/template reserve. They are
        cache occupancy evidence only, never final-body measurement authority.
        """
        if not isinstance(content, str) or self._availability != "available":
            return None
        start = time.monotonic()
        self._last_attempts = 0
        for _attempt in range(TOKENIZE_MAX_CONSECUTIVE_FAILURES):
            if self._deadline is not None and time.monotonic() >= self._deadline:
                raise TimeoutError("tokenizer request deadline expired")
            self._last_attempts += 1
            try:
                count = self._count_tokens(content)
            except Exception as error:
                self._consecutive_failures += 1
                self._failure_reason = "tokenize_unavailable"
                self._last_tokenizer_error = type(error).__name__
                continue
            self._last_latency_ms = (time.monotonic() - start) * 1000.0
            if self._deadline is not None and time.monotonic() >= self._deadline:
                raise TimeoutError("tokenizer request deadline expired")
            self._failure_reason = None
            self._consecutive_failures = 0
            return count
        self._last_latency_ms = (time.monotonic() - start) * 1000.0
        self._availability = "unavailable"
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TimeoutError("tokenizer request deadline expired")
        self._fallback_pending = True
        self._emit_fallback()
        return None

    def _fallback_ratio(self) -> tuple[float, float]:
        if self._minimum_bytes_per_token is None:
            return INITIAL_BYTES_PER_TOKEN, 0.0
        return self._minimum_bytes_per_token * (1.0 - FALLBACK_SAFETY_MARGIN), FALLBACK_SAFETY_MARGIN

    def measure(self, body: Mapping[str, object]) -> tuple[int, str]:
        return self._measure_with_wire_bytes(body, None)

    def _measure_with_wire_bytes(
            self, body: Mapping[str, object], wire_bytes: int | None,
    ) -> tuple[int, str]:
        """Internal fitting count; supplied bytes require whole-body verification."""
        self._select_namespace(body)
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TimeoutError("tokenizer request deadline expired")
        self._last_attempts = 0
        if self._availability != "unavailable":
            from kernel_request_budget import (
                ENDPOINT_VERIFIED_BYTES_PER_TOKEN, FREE_STAGE, _lean_request_limits,
            )
            byte_limit, token_limit = _lean_request_limits(body, FREE_STAGE, exact_tokenizer=True)
            size = _wire_bytes(body) if wire_bytes is None else wire_bytes
            if size > byte_limit:
                # A rejection bound, never an exact count or endpoint failure.
                # Smaller candidates can still tokenize within this request.
                return max(token_limit + 1, math.ceil(size / ENDPOINT_VERIFIED_BYTES_PER_TOKEN)), "verified_byte_bound"
            start = time.monotonic()
            # Resolve transient errors before publishing a measurement method.
            # Diet/finalizer fitting relies on a monotone switch to fallback;
            # retries on later measurements of this request violate that contract.
            for _attempt in range(TOKENIZE_MAX_CONSECUTIVE_FAILURES):
                if self._deadline is not None and time.monotonic() >= self._deadline:
                    raise TimeoutError("tokenizer request deadline expired")
                self._last_attempts += 1
                try:
                    token_count = self._count_tokens(_tokenization_content(body))
                except Exception as error:
                    self._consecutive_failures += 1
                    self._failure_reason = "tokenize_unavailable"
                    self._last_tokenizer_error = type(error).__name__
                    continue
                self._last_latency_ms = (time.monotonic() - start) * 1000.0
                if self._deadline is not None and time.monotonic() >= self._deadline:
                    raise TimeoutError("tokenizer request deadline expired")
                self._availability = "available"
                self._failure_reason = None
                self._consecutive_failures = 0
                return token_count + self.template_overhead_tokens, "tokenize"
            self._last_latency_ms = (time.monotonic() - start) * 1000.0
            self._availability = "unavailable"
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TimeoutError("tokenizer request deadline expired")
        # All fallback consumers compare this count to the prompt ceiling. A
        # tokenizer-independent rejection bound is diagnostic, not that count:
        # otherwise even the irreducible 58 KB benchmark request cannot fit.
        size = _wire_bytes(body) if wire_bytes is None else wire_bytes
        ratio, _margin = self._fallback_ratio()
        estimated = math.ceil(size / ratio) + self.template_overhead_tokens
        self._last_fallback_rejection_bound = (
            FALLBACK_MAX_TOKENS_PER_WIRE_BYTE * size + self.template_overhead_tokens
        )
        self._fallback_pending = True
        self._emit_fallback()
        return estimated, "calibrated_bytes"

    def _emit_fallback(self) -> None:
        """Publish once per affected request, or after late journal attachment."""
        if not self._fallback_pending or self._fallback_reported:
            return
        emit = getattr(self._journal, "emit", None)
        if not callable(emit):
            return
        try:
            emit("request_token_meter_fallback", request_id=self._request_id,
                 measure_method="calibrated_bytes",
                 **self.journal_fields("calibrated_bytes"))
        except Exception as error:
            self._record_observer_failure("request_token_meter_fallback", error)
            return
        self._fallback_reported = True

    def observe_response(self, request_bytes: int, prompt_n: int, cache_n: int) -> None:
        """Compatibility entry point for explicit llama.cpp uncached/cache counts."""
        if (type(request_bytes) is not int or request_bytes <= 0
                or type(prompt_n) is not int or prompt_n < 0
                or type(cache_n) is not int or cache_n < 0):
            return
        self._observe_prompt_total(request_bytes, prompt_n + cache_n)

    def _observe_prompt_total(self, request_bytes: int, token_total: int) -> None:
        if (type(request_bytes) is not int or request_bytes <= 0
                or type(token_total) is not int or token_total <= 0):
            return
        ratio = request_bytes / token_total
        if not math.isfinite(ratio) or ratio <= 0:
            return
        self._minimum_bytes_per_token = (
            ratio if self._minimum_bytes_per_token is None
            else min(self._minimum_bytes_per_token, ratio)
        )
        self._calibration_samples += 1

    def observe_record(self, record: Mapping[str, object]) -> None:
        """Consume request/response journal rows without raising into the writer."""
        observed_kind: str | None = None
        try:
            kind = record.get("kind")
            observed_kind = kind if isinstance(kind, str) else None
            request_id = record.get("request_id") or record.get("provider_request_id")
            if not isinstance(request_id, str) or not request_id:
                return
            if kind == "request_diet_applied":
                value = record.get("request_bytes_after")
                if type(value) is int and value > 0:
                    self._remember_request_bytes(request_id, value)
                return
            if kind == "request_submitted":
                payload_utf8 = record.get("payload_utf8")
                if isinstance(payload_utf8, str):
                    size = len(payload_utf8.encode("utf-8"))
                    self._remember_request_bytes(request_id, size)
                    return
                payload = record.get("payload")
                if isinstance(payload, Mapping):
                    self._remember_request_bytes(request_id, len(json.dumps(payload).encode("utf-8")))
                return
            if kind != "response_received":
                return
            request_bytes = self._pending_bytes.pop(request_id, None)
            accounting = record.get("accounting")
            response = record.get("response")
            if not isinstance(accounting, Mapping):
                accounting = {}
            if not isinstance(response, Mapping):
                response = {}
            from kernel_prompt_accounting import accounting_tokens
            total = accounting_tokens(accounting, response)["total"]
            if type(request_bytes) is int and type(total) is int:
                self._observe_prompt_total(request_bytes, total)
        except Exception as error:
            self._record_observer_failure(observed_kind, error)

    def _record_observer_failure(self, event_kind: str | None, error: Exception) -> None:
        self._observer_failure_count += 1
        self._pending_observer_failures += 1
        self._last_observer_failure_type = type(error).__name__
        self._last_observer_event_kind = event_kind

    def _flush_observer_failures(self) -> None:
        pending = self._pending_observer_failures
        if pending <= 0:
            return
        try:
            journal_emit = getattr(self._journal, "emit", None)
            if not callable(journal_emit):
                return
            self._pending_observer_failures = 0
            journal_emit(
                "request_token_meter_observer_failed",
                subscriber=self._subscriber_name,
                failure_count=pending,
                event_kind=self._last_observer_event_kind,
                error_type=self._last_observer_failure_type,
            )
        except Exception:
            # Retain the count for a later safe journal boundary. Observer
            # callbacks never write reentrantly while the journal is publishing.
            self._pending_observer_failures += pending

    def _remember_request_bytes(self, request_id: str, request_bytes: int) -> None:
        self._pending_bytes[request_id] = request_bytes
        self._pending_bytes.move_to_end(request_id)
        while len(self._pending_bytes) > 128:
            self._pending_bytes.popitem(last=False)

    def attach_journal(self, journal: object | None) -> None:
        if journal is None or id(journal) in self._attached_journals:
            return
        self._journal = journal
        subscribe = getattr(journal, "subscribe", None)
        if not callable(subscribe):
            self._emit_fallback()
            return
        try:
            subscribe(
                self._subscriber_name, self.observe_record,
                kinds=("request_diet_applied", "request_submitted", "response_received"),
            )
        except Exception as error:
            self._record_observer_failure("journal_subscribe", error)
            self._flush_observer_failures()
            return
        self._attached_journals.add(id(journal))
        self._flush_observer_failures()
        self._emit_fallback()

    def journal_fields(self, measure_method: str) -> dict[str, object]:
        self._flush_observer_failures()
        ratio, margin = self._fallback_ratio()
        return {
            "tokenizer_identity": self._namespace,
            # TOKURL: effective tokenize route, scheme://host:port only (never path/query/credentials). A field on the
            # existing measurement events rather than a new event, so journal event ids (which reach the wire) do not shift.
            "tokenize_route": tokenize_route_label(self.endpoint),
            "measurement_contract": "projected_prompt_plus_template_reserve",
            "template_overhead_tokens": (
                self.template_overhead_tokens if measure_method == "tokenize" else 0
            ),
            "template_overhead_reference_tokens": self.template_overhead_tokens,
            "template_overhead_source": TEMPLATE_OVERHEAD_EVIDENCE,
            "tokenize_availability": self._availability,
            "tokenize_failure_reason": self._failure_reason,
            "tokenize_latency_ms": self._last_latency_ms,
            # Zero means no attempt this measurement; latency then describes
            # the last attempted measurement, not a new network sample.
            "tokenize_attempts": self._last_attempts,
            # Count actual uncached tokenizer invocations, including retries.
            # Replacement generations share their logical request's counter.
            "tokenizer_calls": self._tokenizer_calls_by_request.get(self._request_id, 0),
            "tokenize_consecutive_failures": self._consecutive_failures,
            "tokenize_slow": (
                self._last_latency_ms is not None
                and self._last_latency_ms > TOKENIZE_SLOW_SECONDS * 1000.0
            ),
            "fallback_bytes_per_token": ratio,
            "fallback_safety_margin": margin,
            "fallback_calibration_samples": self._calibration_samples,
            **({"fallback_rejection_bound": self._last_fallback_rejection_bound}
               if measure_method == "calibrated_bytes" else {}),
            "journal_observer_failures": self._observer_failure_count,
        }
SECTION_NAMES = (
    "system_prompt",
    "task",
    "stateframe",
    "settled_state",
    "ship_tree",
    "failure_packet",
    "environment_state",
    "get_sure_state",
    "coordination_ledger",
    "passive_conclusions",
    "file_excerpts",
    "file_reads",
    "history_ledger",
    "menus",
    "completion_contract",
    "continuation_limits",
    "stage_evidence",
    "file_index",
    "other",
    "schemas",
    "request_framing",
)

# Stage allowlists still shape the operational surface, but read evidence is
# always admitted and is bounded separately by the request-scoped LRU.
STAGE_SECTION_ALLOWLIST = {
    # No IDEALPATH owner means native/free mode.  Preserve normal byte and
    # tool ceilings, but never apply an S1 section policy to that request.
    "FREE": frozenset(SECTION_NAMES),
    "S1": frozenset({
        "system_prompt", "task", "stateframe", "settled_state", "ship_tree",
        "failure_packet", "environment_state", "get_sure_state",
        "coordination_ledger", "passive_conclusions", "menus",
        "completion_contract", "continuation_limits", "stage_evidence", "file_excerpts",
    }),
    "S2": frozenset({
        "system_prompt", "task", "stateframe", "settled_state", "failure_packet",
        "environment_state", "get_sure_state", "menus", "completion_contract",
        "continuation_limits", "stage_evidence", "file_excerpts", "file_reads",
    }),
    "S3": frozenset({
        "system_prompt", "task", "stateframe", "failure_packet",
        "environment_state", "get_sure_state", "menus", "completion_contract",
        "continuation_limits", "stage_evidence", "file_excerpts",
    }),
    "S4": frozenset({
        "system_prompt", "task", "stateframe", "settled_state", "environment_state",
        "get_sure_state", "menus", "completion_contract", "continuation_limits",
        "stage_evidence", "passive_conclusions", "file_excerpts",
    }),
    "S5": frozenset({
        "system_prompt", "task", "stateframe", "settled_state", "environment_state",
        "get_sure_state", "menus", "completion_contract", "continuation_limits",
        "stage_evidence", "passive_conclusions", "file_excerpts",
    }),
    "S6": frozenset({
        "system_prompt", "task", "stateframe", "settled_state", "environment_state",
        "get_sure_state", "menus", "completion_contract", "continuation_limits",
        "stage_evidence", "passive_conclusions", "file_excerpts",
    }),
    "S7": frozenset({
        "system_prompt", "task", "stateframe", "settled_state", "environment_state",
        "get_sure_state", "menus", "completion_contract", "continuation_limits",
        "stage_evidence", "passive_conclusions", "file_excerpts",
    }),
}

_PROTECTED_SECTIONS = frozenset({
    "system_prompt", "task", "stateframe", "settled_state", "menus",
    "completion_contract", "continuation_limits", "stage_evidence",
})
_READ_EVIDENCE_SECTIONS = frozenset({"file_excerpts", "file_reads"})
# An over-limit request may shed old read-page results only after optional
# narratives; retain the two most recent result spans as a useful floor.
_MIN_READ_SPANS_TO_RETAIN = 2
_DROPPED_SOURCE_RANGES_HEADER = "[Request diet: dropped source ranges]"
TRUTHFUL_READ_PANELS_ENV = "STATEFRAME_TRUTHFUL_READ_PANELS"
DUPLICATED_SOURCE_EVICTION_ENV = "STATEFRAME_DIET_EVICT_DUPLICATED_ONLY"


def truthful_read_panels_enabled(environ: Mapping[str, object] | None = None) -> bool:
    """Enable truthful source diet panels only by explicit opt-in."""
    source = os.environ if environ is None else environ
    return str(source.get(TRUTHFUL_READ_PANELS_ENV, "off")).strip().lower() == "on"


DIET_INCR_ENV = "SF_LAB_DIET_INCR"


def diet_incr_enabled(environ: Mapping[str, object] | None = None) -> bool:
    """Incremental removal byte accounting (DIET_INCR); SF_LAB_DIET_INCR=legacy restores the exact-only path."""
    source = os.environ if environ is None else environ
    return str(source.get(DIET_INCR_ENV, "on")).strip().lower() != "legacy"


def duplicated_source_eviction_enabled(
        environ: Mapping[str, object] | None = None,
) -> bool:
    """Enable duplicate-only read eviction only by explicit opt-in."""
    source = os.environ if environ is None else environ
    return str(source.get(DUPLICATED_SOURCE_EVICTION_ENV, "off")).strip().lower() == "on"


_DROP_ORDER = (
    # Preserve the required eviction priority; reads are a separate last resort.
    "history_ledger", "passive_conclusions", "coordination_ledger",
    "get_sure_state", "environment_state", "ship_tree", "failure_packet",
)
_OPTIONAL_EXPLICIT_SECTIONS = frozenset({
    "optional", "advice", "playbook", "diagnostic", "scratch",
})
_SECTION_ALIASES = {
    "system": "system_prompt",
    "developer": "system_prompt",
    "frame": "stateframe",
    "state": "stateframe",
    "history": "history_ledger",
    "ledger": "history_ledger",
    "menu": "menus",
    "completion": "completion_contract",
    "contract": "completion_contract",
    "evidence": "stage_evidence",
}
_ID_PATTERN = re.compile(
    r"\b(?:sha256:[0-9a-f]{8,}|R[0-9]+|"
    r"(?:obl|ev|evidence|receipt)[:-][A-Za-z0-9_.:-]+)\b",
    re.IGNORECASE,
)
_SOURCE_PATTERN = re.compile(
    r"\[(?:Current|Working|Observed|Retained|Repair|Paged) source:\s*([^\]\n]+)\]",
    re.IGNORECASE,
)
_CURRENT_SOURCE_BLOCK_PATTERN = re.compile(
    r"(?m)^\[Current source:\s*([^\]\n]+)\]$",
)
_LEGACY_SOURCE_PATTERN = re.compile(r"^CURRENT_SOURCE\s+([^\s]+)", re.IGNORECASE)
_READ_PATH_PATTERN = re.compile(r"\bpath=(.*?)\s+revision=")
_REVISION_PATTERN = re.compile(r"\brevision=([^\s]+)")
_SOURCE_RANGE_REVERSE_PATTERN = re.compile(
    r"\bend_line\s*[:=]\s*(\d+).*?\bstart_line\s*[:=]\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
_SOURCE_RANGE_PATTERNS = (
    re.compile(r"---\s*source\s+lines\s+(\d+)-(\d+)\s*---", re.IGNORECASE),
    re.compile(r"\blines\s*[:=]\s*(\d+)\s*-\s*(\d+)(?:/\d+)?", re.IGNORECASE),
    re.compile(r"\bcoverage:\s*lines\s+(\d+)-(\d+)", re.IGNORECASE),
    re.compile(
        r"\bstart_line\s*[:=]\s*(\d+).*?\bend_line\s*[:=]\s*(\d+)",
        re.IGNORECASE | re.DOTALL,
    ),
    _SOURCE_RANGE_REVERSE_PATTERN,
)
_MENU_PATTERN = re.compile(r"action_menu=\[([^\]]*)\]")


class RequestDietDeferred(RuntimeError):
    """Raised when the protected request and minimum read set cannot fit."""

    def __init__(self, detail: Mapping[str, object]):
        self.detail = dict(detail)
        super().__init__(str(self.detail.get("reason", "request diet deferred")))


def _wire_bytes(value: object) -> int:
    return len(json.dumps(value).encode())


def _message_content(message: Mapping[str, object]) -> str:
    value = message.get("content", "")
    return value if isinstance(value, str) else str(value)


def _is_framefit5_stateframe(message: Mapping[str, object]) -> bool:
    content = _message_content(message)
    return (
        message.get("role") == "user"
        and content.startswith("[StateFrame]\n")
        and ('"framefit5"' in content or "stateframe-framefit5/1" in content)
    )


def _explicit_section(message: Mapping[str, object]) -> str | None:
    value = message.get("_request_budget_section")
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower()
    return _SECTION_ALIASES.get(normalized, normalized)


def _tool_call_name(call: Mapping[str, object]) -> str:
    function = call.get("function")
    name = call.get("name") or (
        function.get("name") if isinstance(function, Mapping) else ""
    )
    return str(name).strip().casefold()


def _assistant_calls_read(message: Mapping[str, object]) -> bool:
    calls = message.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
        return False
    for call in calls:
        if isinstance(call, Mapping) and _tool_call_name(call) == "read":
            return True
    return False


def section_name(message: Mapping[str, object], index: int = 0) -> str:
    """Classify one rendered message without trusting model-authored prose."""
    role = str(message.get("role", ""))
    content = _message_content(message)
    stripped = content.lstrip()
    lowered = stripped.lower()
    explicit = _explicit_section(message)
    if explicit in SECTION_NAMES or explicit in _OPTIONAL_EXPLICIT_SECTIONS:
        return explicit
    if role in {"system", "developer"} and index == 0:
        return "system_prompt"
    if role in {"system", "developer"}:
        return "system_prompt"
    if lowered.startswith("task:") or lowered.startswith("brief:"):
        return "task"
    if lowered.startswith("[files in frame]"):
        return "file_index"
    if _SOURCE_PATTERN.match(stripped):
        return "file_excerpts"
    if _LEGACY_SOURCE_PATTERN.match(stripped):
        return "file_excerpts"
    if any(marker in content for marker in (
            "[Current source:", "[Working source:", "[Observed source:",
            "[Retained source:", "[Repair source:", "[Paged source:",
            "CURRENT_SOURCE ", "[Source metadata]", "[Source content:",
    )):
        return "file_excerpts"
    if lowered.startswith("[read page]") or lowered.startswith("[read "):
        return "file_reads"
    if role == "assistant" and _assistant_calls_read(message):
        return "file_reads"
    if "[reasoning ledger" in lowered or lowered.startswith("[history"):
        return "history_ledger"
    if "action_menu=[" in content or "[evidencepin]" in lowered:
        return "menus"
    if lowered.startswith("[continuation limits") or "continuation limits" in lowered:
        return "continuation_limits"
    if lowered.startswith("[stateframe]"):
        return "stateframe"
    if lowered.startswith("<settled_state>"):
        return "settled_state"
    if lowered.startswith("[ship-tree"):
        return "ship_tree"
    if lowered.startswith("[raising-h"):
        return "failure_packet"
    if lowered.startswith("[environment state]"):
        return "environment_state"
    if lowered.startswith("[get-sure state]"):
        return "get_sure_state"
    if lowered.startswith("[coordinated change ledger]"):
        return "coordination_ledger"
    if lowered.startswith("[passive conclusions"):
        return "passive_conclusions"
    if lowered.startswith("[delivery_manifest") or lowered.startswith("[delivery manifest"):
        return "passive_conclusions"
    if message.get("_stateframe_owner_block") or "completion contract" in lowered:
        return "completion_contract"
    if role == "tool":
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, Mapping) and any(
                key in parsed for key in ("stage_exit", "next_action", "state_epoch")):
            return "stage_evidence"
        return "history_ledger"
    if "unified stop authority" in lowered or "delivery_manifest" in lowered:
        return "completion_contract"
    if role == "user":
        return "stateframe"
    return "other"


def _repeat_unit(text: str) -> str | None:
    stripped = text.strip()
    if len(stripped) < 128:
        return None
    for repetitions in range(8, 1, -1):
        if len(stripped) % repetitions:
            continue
        unit_length = len(stripped) // repetitions
        unit = stripped[:unit_length]
        if len(unit) >= 64 and len(set(unit)) > 4 and unit * repetitions == stripped:
            return unit
    return None


def _dedupe_text(text: str) -> str:
    repeated = _repeat_unit(text)
    if repeated is not None:
        return repeated
    markers = re.compile(
        r"(?m)(?=^\[(?:StateFrame|Decision handoff|Reasoning ledger|Continuation limits))"
    )
    blocks = [block for block in markers.split(text) if block]
    if len(blocks) < 2:
        return text.strip()
    seen: set[str] = set()
    unique: list[str] = []
    for block in blocks:
        key = " ".join(block.split())
        if key in seen:
            continue
        seen.add(key)
        unique.append(block)
    return "".join(unique).strip()


def _compact_json_text(text: str) -> str:
    stripped = text.strip()
    if not ((stripped.startswith("{") and stripped.endswith("}"))
            or (stripped.startswith("[") and stripped.endswith("]"))):
        return text
    try:
        value = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _compact_json_lines(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            compacted = _compact_json_text(stripped)
            lines.append(compacted if compacted != stripped else line)
        else:
            lines.append(line)
    return "\n".join(lines)


def _dedupe_instruction_lines(text: str) -> str:
    """Remove exact repeated numbered instructions from one system prompt."""
    seen: set[str] = set()
    result: list[str] = []
    for line in text.splitlines():
        key = re.sub(r"^\s*\d+\.\s*", "", line).strip()
        if len(key) >= 48 and key in seen:
            continue
        if key:
            seen.add(key)
        result.append(line)
    return "\n".join(result)


def _native_tool_names(tools: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    names: list[str] = []
    for tool in tools:
        direct = tool.get("name")
        function = tool.get("function")
        name = direct or (function.get("name") if isinstance(function, Mapping) else "")
        if name and str(name) not in names:
            names.append(str(name))
    return tuple(names)


def _compact_menu_text(
        text: str, evidence_ids: Sequence[str],
        native_tools: Sequence[Mapping[str, object]] = (),
) -> str:
    """Keep evidence once; native schemas carry the complete action contract."""
    actions = menu_actions(({"content": text},))
    source = text
    if "[EVIDENCEPIN]" in source:
        source = source.split("[EVIDENCEPIN]", 1)[1]
        source = source.split("[/EVIDENCEPIN]", 1)[0]
    retained: list[str] = []
    prefixes = (
        "check_status=", "freshness=",
        "readiness_claim=", "witness_operation=", "requirements_source=",
        "requirements:", "- R",
    )
    native_names = _native_tool_names(native_tools)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("action_menu=") and native_names:
            continue
        if stripped.startswith("action_menu=") or stripped.startswith(prefixes):
            pieces = [piece.strip() for piece in stripped.split(";")]
            pieces = [piece for piece in pieces
                      if not piece.startswith(("menu_revision=", "measured_revision=",
                                               "retain_until_revision="))]
            normalized = "; ".join(piece for piece in pieces if piece)
            if normalized and normalized not in retained:
                retained.append(normalized)
    if actions and not native_names:
        retained.insert(0, "action_menu=[" + ",".join(actions) + "]")
    if evidence_ids:
        retained.append("required_evidence_ids=" + ",".join(evidence_ids))
    if not retained:
        if native_names:
            return "[Decision handoff]"
        return _dedupe_text(text)
    return "[Decision handoff]\n[EVIDENCEPIN]\n" + "\n".join(retained) + "\n[/EVIDENCEPIN]"


def _id_values(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    return set(_ID_PATTERN.findall(value))


def required_evidence_ids(messages: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    ids: set[str] = set()
    for message in messages:
        content = _message_content(message)
        lowered = content.lower()
        if ("evidencepin" not in lowered and "required_evidence" not in lowered
                and message.get("_stateframe_owner_block") is None):
            continue
        ids.update(_id_values(content))
    return tuple(sorted(ids))


def menu_actions(messages: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    actions: list[str] = []
    for message in messages:
        for match in _MENU_PATTERN.finditer(_message_content(message)):
            actions.extend(item.strip() for item in match.group(1).split(",") if item.strip())
    return tuple(dict.fromkeys(actions))


def _source_identity(
        message: Mapping[str, object],
) -> tuple[str, str, str, tuple[int, int] | None] | None:
    certified = message.get("_framebuild_evidence_envelope")
    if isinstance(certified, Mapping):
        path_value = certified.get("source_ref", certified.get("path"))
        revision_value = certified.get("source_revision", certified.get("revision", ""))
        if isinstance(path_value, str) and path_value.strip():
            return "certified", path_value.replace("\\", "/"), str(revision_value), None
    content = _message_content(message)
    match = _SOURCE_PATTERN.search(content)
    if match is not None:
        path = match.group(1).strip()
    else:
        legacy_match = _LEGACY_SOURCE_PATTERN.match(content.lstrip())
        if legacy_match is not None:
            path = legacy_match.group(1).strip()
        elif content.lstrip().lower().startswith("[read "):
            path_match = _READ_PATH_PATTERN.search(content)
            path = path_match.group(1).strip() if path_match else ""
            kind = "read"
        else:
            return None
    if match is not None or _LEGACY_SOURCE_PATTERN.match(content.lstrip()):
        kind = "source"
    if not path:
        return None
    revision_match = _REVISION_PATTERN.search(content)
    source_range = None
    for pattern in _SOURCE_RANGE_PATTERNS:
        range_match = pattern.search(content)
        if range_match is None:
            continue
        first, second = (int(value) for value in range_match.groups())
        start_line, end_line = (
            (second, first) if pattern is _SOURCE_RANGE_REVERSE_PATTERN else (first, second)
        )
        if start_line > 0 and end_line >= start_line:
            source_range = (start_line, end_line)
        break
    if (source_range is None and match is not None
            and match.group(0).casefold().startswith("[current source:")):
        # Legacy Current source blocks contain the complete file body but no
        # explicit range header. Their visible body is anchored at line 1.
        lines = content.splitlines()
        body_start = 1
        while body_start < len(lines) and lines[body_start].casefold().startswith(
                ("revision=", "source_revision=")):
            body_start += 1
        body_line_count = len(lines) - body_start
        if body_line_count > 0:
            source_range = (1, body_line_count)
    return (
        kind,
        path.replace("\\", "/"),
        revision_match.group(1) if revision_match else "",
        source_range,
    )


def _current_source_segments(
        messages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Extract exact, revision-bound byte spans from visible Current source blocks."""
    from kernel_request_source_window import (
        WINDOW_END_MARKER,
        _model_window_segments,
        _window_segments,
        parse_model_source_window,
        parse_window,
    )

    blocks: list[tuple[str, str, Mapping[str, object] | None]] = []
    current_revisions: dict[str, set[str]] = {}
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = _message_content(message)
        headers = tuple(_CURRENT_SOURCE_BLOCK_PATTERN.finditer(content))
        for index, header in enumerate(headers):
            block_end = headers[index + 1].start() if index + 1 < len(headers) else len(content)
            block = content[header.start():block_end]
            path = header.group(1).strip().replace("\\", "/").removeprefix("./")
            if not path:
                continue
            metadata: Mapping[str, object] | None = None
            for parser in (parse_window, parse_model_source_window):
                parsed = parser(block)
                if not isinstance(parsed, Mapping):
                    continue
                metadata = parsed
                metadata_path = str(parsed.get("path", "")).replace("\\", "/").removeprefix("./")
                metadata_revision = str(parsed.get("revision", ""))
                if (metadata_path == path and metadata_revision
                        and parsed.get("sha256") in (
                            None, metadata_revision.removeprefix("sha256:")
                        )):
                    current_revisions.setdefault(path, set()).add(metadata_revision)
                break
            blocks.append((path, block, metadata))

    segments: list[dict[str, object]] = []
    for path, block, metadata in blocks:
        if metadata is not None:
            metadata_path = str(metadata.get("path", "")).replace("\\", "/").removeprefix("./")
            revision = str(metadata.get("revision", ""))
            if (metadata_path != path or not revision
                    or metadata.get("sha256") not in (
                        None, revision.removeprefix("sha256:")
                    )):
                continue
            segmenter = (
                _window_segments if metadata.get("schema") == "stateframe-request-source-window/1"
                else _model_window_segments
            )
            try:
                rows = segmenter(block, metadata)
            except (TypeError, ValueError, KeyError):
                continue
            for row, text in rows:
                start = row.get("start_line")
                end = row.get("end_line")
                if (type(start) is not int or type(end) is not int
                        or start < 1 or end < start):
                    continue
                trailing_newline = row.get("trailing_newline")
                if trailing_newline is False and text.endswith("\n"):
                    text = text[:-1]
                elif trailing_newline is True and not text.endswith("\n"):
                    continue
                if len(text.splitlines(keepends=True)) != end - start + 1:
                    continue
                segments.append({
                    "path": path,
                    "revision": revision,
                    "start_line": start,
                    "end_line": end,
                    "text": text,
                })
            continue

        identity = _source_identity({"content": block})
        if identity is None or identity[0] != "source":
            continue
        identity_path = identity[1].replace("\\", "/").removeprefix("./")
        if identity_path != path:
            continue
        revision = identity[2]
        if not revision:
            revisions = current_revisions.get(path, set())
            if len(revisions) > 1:
                continue
            if len(revisions) == 1:
                revision = next(iter(revisions))

        lines = block.splitlines(keepends=True)
        source_range = identity[3]
        range_marker = re.compile(r"^--- source lines (\d+)-(\d+) ---\r?\n?$")
        coverage_marker = re.compile(r"^Coverage: lines (\d+)-(\d+)(?:[,; ].*)?\r?\n?$")
        markers = [
            (line_index, match)
            for line_index, line in enumerate(lines)
            for match in (range_marker.match(line),)
            if match is not None
        ]
        if markers:
            for marker_index, match in markers:
                start, end = (int(value) for value in match.groups())
                body_lines: list[str] = []
                for line in lines[marker_index + 1:]:
                    if (range_marker.match(line) or coverage_marker.match(line)
                            or line.rstrip("\r\n") == WINDOW_END_MARKER):
                        break
                    body_lines.append(line)
                text = "".join(body_lines)
                if len(text.splitlines(keepends=True)) == end - start + 1:
                    segments.append({
                        "path": path,
                        "revision": revision,
                        "start_line": start,
                        "end_line": end,
                        "text": text,
                    })
            continue

        if source_range is None:
            continue
        body_start = 1
        while body_start < len(lines):
            metadata_line = lines[body_start].strip().casefold()
            if metadata_line.startswith(("revision=", "source_revision=", "total_lines=", "sha256=")):
                body_start += 1
                continue
            if metadata_line.startswith("coverage: lines"):
                body_start += 1
                continue
            break
        body_end = len(lines)
        last_content_line = body_end - 1
        while last_content_line >= body_start and not lines[last_content_line].strip():
            last_content_line -= 1
        if last_content_line >= body_start:
            footer = coverage_marker.match(lines[last_content_line])
            if footer is not None:
                footer_range = tuple(int(value) for value in footer.groups())
                if source_range != footer_range:
                    continue
                body_end = last_content_line
        text = "".join(lines[body_start:body_end])
        start, end = source_range
        if len(text.splitlines(keepends=True)) == end - start + 1:
            segments.append({
                "path": path,
                "revision": revision,
                "start_line": start,
                "end_line": end,
                "text": text,
            })
    return segments


def _read_page_source_segment(
        message: Mapping[str, object],
) -> dict[str, object] | None:
    """Read a complete, unmodified whole-line source page with exact body bytes."""
    content = _message_content(message)
    header, first_separator, remainder = content.partition("\n")
    _status_line, second_separator, body = remainder.partition("\n")
    if not first_separator or not second_separator:
        return None
    try:
        from kernel_request_source_window import _READ_PAGE_HEADER

        match = _READ_PAGE_HEADER.fullmatch(header.removesuffix("\r"))
        if match is None:
            return None
        fields = match.groupdict()
        clip_match = re.search(r"\bclip=(\S+)", header)
        start = int(fields["start"])
        end = int(fields["end"])
        effective_start = int(fields["effective_start"])
        effective_end = int(fields["effective_end"])
        chars = int(fields["chars"])
        byte_count = int(fields["bytes"])
        start_column = int(fields["start_column"])
        end_column = int(fields["end_column"])
        if len(body) not in {chars, chars - 1}:
            return None
        page = body
        encoded = page.encode("utf-8")
    except (ImportError, TypeError, ValueError, UnicodeError):
        return None
    identity = _source_identity(message)
    if (identity is None or identity[0] != "read"
            or fields["truncated"] != "false"
            or fields["replaced"] != "false"
            or clip_match is None or clip_match.group(1) != "none"
            or start != effective_start or end != effective_end
            or start_column != 0 or start < 1 or end < start
            or identity[1].replace("\\", "/").removeprefix("./")
            != fields["path"].replace("\\", "/").removeprefix("./")
            or identity[2] != fields["revision"]
            or identity[3] != (start, end)
            or len(page) not in {chars, chars - 1}
            or len(page.splitlines(keepends=True)) != end - start + 1):
        return None
    if len(page) != chars:
        # Some serialized tool results omit only the final line feed even though
        # the authenticated page header includes it. Reconstruct that byte only
        # when its character and byte totals, complete-line count, and declared
        # final-column length all agree; exact comparison with Current source is
        # still required before eviction.
        final_line = page.rsplit("\n", 1)[-1]
        if (page.endswith("\n") or len(page) + 1 != chars
                or len(encoded) + 1 != byte_count
                or len(final_line) + 1 != end_column):
            return None
        page += "\n"
        encoded = page.encode("utf-8")
    if len(encoded) != byte_count:
        return None
    return {
        "path": fields["path"].replace("\\", "/").removeprefix("./"),
        "revision": fields["revision"],
        "start_line": start,
        "end_line": end,
        "bytes": encoded,
    }


def _current_source_duplicate_location(
        messages: Sequence[Mapping[str, object]],
        read_message: Mapping[str, object],
        current_sources: Sequence[Mapping[str, object]] | None = None,
        *,
        allow_unversioned_current_source: bool = False,
) -> dict[str, object] | None:
    """Return same-wire source proof; unversioned blocks require exact read-byte binding."""
    try:
        read = _read_page_source_segment(read_message)
        if read is None:
            return None
        read_path = str(read["path"])
        read_revision = str(read["revision"])
        if not read_revision:
            return None
        read_start = int(read["start_line"])
        read_end = int(read["end_line"])
        read_bytes = read["bytes"]
        sources = _current_source_segments(messages) if current_sources is None else current_sources
        explicit_revisions = {
            str(source.get("revision", ""))
            for source in sources
            if source.get("path") == read_path and source.get("revision")
        }
        if explicit_revisions and read_revision not in explicit_revisions:
            return None
        for source in sources:
            source_revision = str(source.get("revision", ""))
            revision_matches = source_revision == read_revision
            unversioned_match_allowed = (
                allow_unversioned_current_source and not source_revision
            )
            if (source.get("path") != read_path
                    or not (revision_matches or unversioned_match_allowed)
                    or int(source["start_line"]) > read_start
                    or int(source["end_line"]) < read_end):
                continue
            source_lines = str(source["text"]).splitlines(keepends=True)
            offset = read_start - int(source["start_line"])
            matched = "".join(source_lines[offset:offset + read_end - read_start + 1])
            if matched.encode("utf-8") == read_bytes:
                return {
                    "path": read_path,
                    "revision": read_revision,
                    "start_line": read_start,
                    "end_line": read_end,
                }
    except Exception:
        # This is an optimization only; uncertain evidence must keep the bytes.
        return None
    return None


def _duplicate_source_location_record(
        location: Mapping[str, object],
) -> dict[str, object]:
    """Keep location coordinates here; the outer source record carries revision."""
    return {
        "path": location["path"],
        "start_line": location["start_line"],
        "end_line": location["end_line"],
    }


def _source_delivery_ranges(
        messages: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Collect exact path/revision line spans that remain in the request."""
    from kernel_request_source_window import (
        _model_window_segments,
        _window_segments,
        parse_model_source_window,
        parse_window,
    )

    ranges: dict[tuple[str, str], list[tuple[int, int]]] = {}

    def add(path: object, revision: object, start: object, end: object) -> None:
        if not isinstance(path, str) or not path.strip():
            return
        if (not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)
                or start < 1 or end < start):
            return
        key = (path.replace("\\", "/").strip().removeprefix("./"),
               str(revision or ""))
        ranges.setdefault(key, []).append((start, end))

    for message in messages:
        identity = _source_identity(message)
        if identity is None:
            continue
        kind, path, revision, source_range = identity
        if source_range is not None:
            add(path, revision, source_range[0], source_range[1])

        envelope = message.get("_framebuild_evidence_envelope")
        if kind == "certified" and isinstance(envelope, Mapping):
            declared = envelope.get("covered_intervals", envelope.get("ranges", ()))
            if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
                for interval in declared:
                    if isinstance(interval, Mapping):
                        add(path, revision, interval.get("start_line", interval.get("start")),
                            interval.get("end_line", interval.get("end")))
                    elif (isinstance(interval, Sequence) and not isinstance(interval, (str, bytes))
                          and len(interval) >= 2):
                        add(path, revision, interval[0], interval[1])

        content = _message_content(message)
        for parser, segmenter in (
                (parse_window, _window_segments),
                (parse_model_source_window, _model_window_segments)):
            metadata = parser(content)
            if metadata is None:
                continue
            window_path = metadata.get("path", path)
            window_revision = metadata.get("revision", revision)
            for row, _text in segmenter(content, metadata):
                add(window_path, window_revision,
                    row.get("start_line"), row.get("end_line"))
            break

    for key, spans in tuple(ranges.items()):
        merged: list[tuple[int, int]] = []
        for start, end in sorted(set(spans)):
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        ranges[key] = merged
    return ranges


def _fully_delivered_source_paths(
        messages: Sequence[Mapping[str, object]],
) -> set[str]:
    """Return paths with a complete source block or complete revision-bound span."""
    from kernel_request_source_window import (
        _model_window_segments,
        _window_segments,
        parse_model_source_window,
        parse_window,
    )

    delivered: set[str] = set()
    ranges = _source_delivery_ranges(messages)
    for message in messages:
        identity = _source_identity(message)
        if identity is None:
            continue
        kind, path, revision, _source_range = identity
        content = _message_content(message)
        normalized_path = path.replace("\\", "/").removeprefix("./")
        if (kind == "source" and content.lstrip().startswith("[Current source:")
                and parse_window(content) is None
                and parse_model_source_window(content) is None):
            delivered.add(normalized_path)
            continue

        if kind == "read" and content.lstrip().startswith("[read page]"):
            header = content.splitlines()[0] if content.splitlines() else ""
            match = re.search(r"\blines=(\d+)-(\d+)/(\d+)\b", header)
            if (match is not None and match.group(1) == "1"
                    and match.group(2) == match.group(3)
                    and "truncated=false" in header):
                delivered.add(normalized_path)
                continue

        for parser, segmenter in (
                (parse_window, _window_segments),
                (parse_model_source_window, _model_window_segments)):
            metadata = parser(content)
            if metadata is None:
                continue
            total = metadata.get("total_lines", metadata.get("source_line_count"))
            if not isinstance(total, int) or isinstance(total, bool) or total < 1:
                break
            window_path = (str(metadata.get("path", normalized_path))
                           .replace("\\", "/").strip().removeprefix("./"))
            window_revision = str(metadata.get("revision", revision) or "")
            coverage = ranges.get((window_path, window_revision), ())
            if coverage and coverage[0][0] == 1 and coverage[-1][1] >= total:
                delivered.add(window_path)
            for row, _text in segmenter(content, metadata):
                row_total = row.get("total_lines", row.get("total"))
                if (row_total == total and row.get("start_line") == 1
                        and row.get("end_line") == total):
                    delivered.add(window_path)
            break
    return delivered


def _uncovered_source_ranges(
        start: int, end: int, coverage: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    remaining: list[tuple[int, int]] = []
    cursor = start
    for covered_start, covered_end in coverage:
        if covered_end < cursor:
            continue
        if covered_start > end:
            break
        if covered_start > cursor:
            remaining.append((cursor, min(end, covered_start - 1)))
        cursor = max(cursor, covered_end + 1)
        if cursor > end:
            break
    if cursor <= end:
        remaining.append((cursor, end))
    return remaining


def _certified_scope(message: Mapping[str, object]) -> str:
    """Return exact scope/revision identity for an EC01 evidence envelope."""

    envelope = message.get("_framebuild_evidence_envelope")
    if not isinstance(envelope, Mapping):
        return ""
    scope = envelope.get("scope", envelope.get("namespace_id", ""))
    identity = {
        "store_id": envelope.get("store_id", ""),
        "namespace_id": envelope.get("namespace_id", scope),
        "repo_id": envelope.get("repo_id", ""),
        "worktree_id": envelope.get("worktree_id", ""),
        "branch_id": envelope.get("branch_id", ""),
        "source_revision": envelope.get("source_revision", ""),
        "source_sha256": envelope.get("source_sha256", ""),
        "covered_intervals": envelope.get("covered_intervals", envelope.get("ranges", ())),
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _compact_stage_evidence(value: Mapping[str, object], ids: Sequence[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    top_keys = (
        "accepted", "action", "code", "detail", "next_action", "schema", "stage",
        "stage_exit", "state_epoch", "status",
    )
    for key in top_keys:
        if key in value:
            result[key] = copy.deepcopy(value[key])
    stage_exit = value.get("stage_exit")
    if isinstance(stage_exit, Mapping):
        result["stage_exit"] = {
            key: copy.deepcopy(stage_exit[key])
            for key in (
                "accepted", "admitted_actions", "action", "code", "reason", "stage",
                "state_epoch", "required_evidence_ids", "evidence_ids",
            )
            if key in stage_exit
        }
        continuation = stage_exit.get("continuation")
        next_menu = continuation.get("next_menu") if isinstance(continuation, Mapping) else None
        if isinstance(next_menu, Mapping):
            result["stage_exit"]["next_menu"] = {
                key: copy.deepcopy(next_menu[key])
                for key in ("actions", "backtrack_targets", "answer_fields")
                if key in next_menu
            }
    evidence = value.get("evidence")
    if isinstance(evidence, Mapping):
        result["evidence"] = {
            key: copy.deepcopy(evidence[key])
            for key in (
                "stage", "required_evidence_ids", "evidence_ids", "receipt_ids",
                "row_ids", "clause_ids", "status",
            )
            if key in evidence
        }
    if ids:
        result["required_evidence_ids"] = list(ids)
    return result or dict(value)


def _normalize_content(message: Mapping[str, object], section: str,
                       evidence_ids: Sequence[str],
                       native_tools: Sequence[Mapping[str, object]] = ()) -> str:
    content = _dedupe_text(_message_content(message))
    if section == "stage_evidence":
        try:
            value = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = None
        if isinstance(value, Mapping):
            content = json.dumps(
                _compact_stage_evidence(value, evidence_ids),
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            )
    elif section in {"stateframe", "settled_state"}:
        content = _compact_json_lines(content)
    elif section == "system_prompt":
        content = _dedupe_instruction_lines(content)
    elif section == "menus":
        content = _compact_menu_text(content, evidence_ids, native_tools)
    else:
        content = _compact_json_text(content)
    return content


def _message_key(message: Mapping[str, object]) -> str:
    comparable = {key: value for key, value in message.items() if key != "content"}
    comparable["role"] = message.get("role")
    comparable["content"] = _message_content(message)
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def source_window_path(content: str) -> str | None:
    """Return the path of a bounded source window message, else ``None``."""
    head = content[:512]
    if "[Request budget source window]" not in head and "[Model source window]" not in head:
        return None
    match = _SOURCE_PATTERN.search(head)
    if match is not None:
        return match.group(1).strip().replace("\\", "/")
    path = re.search(r'"path":"([^"]+)"', head)
    return path.group(1).replace("\\", "/") if path else ""


def _is_protected_message(
        message: Mapping[str, object], protected_contents: Sequence[str],
        protected_messages: Sequence[Mapping[str, object]],
        required_source_paths: frozenset[str] | None = None,
) -> bool:
    content = _message_content(message)
    if (message.get("role") == "user"
            and content.lstrip().lower().startswith("[stateframe]")):
        return True
    if _is_framefit5_stateframe(message):
        return True
    if message.get("_framebuild_certified_evidence") is True or isinstance(
        message.get("_framebuild_evidence_envelope"), Mapping
    ):
        return True
    if _is_complete_anchored_source(message):
        return True
    window_path = source_window_path(content)
    if required_source_paths is None and window_path is not None:
        return True
    if required_source_paths is not None:
        identity = _source_identity(message)
        source_path = window_path or (identity[1] if identity is not None else None)
        if (section_name(message) == "file_excerpts"
                and source_path is not None
                and source_path.replace("\\", "/") in required_source_paths):
            # Required closure excerpts can use Current, Paged, or legacy source
            # framing. Protect by the parsed source identity, not one renderer.
            return True
    if any(value and value in content for value in protected_contents):
        return True
    explicitly_protected = any(
        message is protected or _message_key(message) == _message_key(protected)
        for protected in protected_messages
        if isinstance(protected, Mapping)
    )
    if explicitly_protected:
        return True
    return False


def _paired_tool_exchange_messages(
        messages: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Return complete assistant/tool exchanges that must be diet-atomic.

    OpenAI-compatible tool protocols treat an assistant ``tool_calls`` message
    and each matching ``tool`` result as one exchange.  Dropping only the
    result leaves an invalid provider transcript; dropping only the call loses
    the evidence that explains the result.  Incomplete exchanges are not
    promoted to evidence, but their existing messages remain available for the
    final protocol validator to reject honestly.
    """
    calls: dict[str, tuple[int, Mapping[str, object]]] = {}
    calls_by_message: dict[int, list[str]] = {}
    results: dict[str, list[tuple[int, Mapping[str, object]]]] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role", "")) == "assistant":
            raw_calls = message.get("tool_calls")
            if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, (str, bytes, bytearray)):
                for call in raw_calls:
                    if isinstance(call, Mapping):
                        call_id = str(call.get("id", "")).strip()
                        if call_id:
                            calls[call_id] = (index, message)
                            calls_by_message.setdefault(index, []).append(call_id)
        if str(message.get("role", "")) == "tool":
            call_id = str(message.get("tool_call_id", "")).strip()
            if call_id:
                results.setdefault(call_id, []).append((index, message))
    protected: list[Mapping[str, object]] = []
    for assistant_index, call_ids in calls_by_message.items():
        matching = [(call_id, result_index, result)
                    for call_id in call_ids
                    for result_index, result in results.get(call_id, ())]
        if not matching:
            continue
        complete_exchange = all(results.get(call_id) for call_id in call_ids)
        assistant = calls[call_ids[0]][1]
        raw_calls = assistant.get("tool_calls")
        read_only_calls = (
            isinstance(raw_calls, Sequence)
            and not isinstance(raw_calls, (str, bytes, bytearray))
            and len(raw_calls) == len(call_ids)
            and all(
                isinstance(call, Mapping)
                and str(call.get("id", "")).strip()
                and _tool_call_name(call) == "read"
                for call in raw_calls
            )
        )
        read_only_results = all(
            section_name(result, result_index) == "file_reads"
            for _, result_index, result in matching
        )
        if complete_exchange and read_only_calls and read_only_results:
            # Read-page exchanges remain atomic, but the LRU may evict the
            # assistant call and all paired results together when necessary.
            continue
        protected.append(assistant)
        protected.extend(result for _, _, result in matching)
    return tuple(protected)


def _is_complete_anchored_source(message: Mapping[str, object]) -> bool:
    """Keep a complete, anchored source span intact during budget reduction."""
    for key in ("_source_span_complete", "source_span_complete", "anchored_source_complete"):
        if message.get(key) is True:
            return True
    for key in ("anchored_source_span", "source_span"):
        value = message.get(key)
        if isinstance(value, Mapping) and value.get("complete") is True \
                and value.get("anchored") is True:
            return True
    return False


def _range_contains(
        outer: tuple[int, int], inner: tuple[int, int],
) -> bool:
    return outer[0] <= inner[0] and outer[1] >= inner[1]


def _source_range_record(
        message: Mapping[str, object], *, reason: str, protected: bool = False,
) -> dict[str, object] | None:
    identity = _source_identity(message)
    if identity is None:
        return None
    kind, path, revision, source_range = identity
    if source_range is None:
        return None
    return {
        "path": path,
        "start_line": source_range[0],
        "end_line": source_range[1],
        "revision": revision,
        "reason": reason,
        "origin": "tool_read" if kind == "read" else "auto_delivered",
        "protected": protected,
    }


def _parse_dropped_source_notice(content: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in content.splitlines():
        match = re.match(
            r"^- (.+):(\d+)-(\d+) revision=(\S+) reason=(\S+)$", line.strip(),
        )
        if match is None:
            continue
        path, start_line, end_line, revision, reason = match.groups()
        records.append({
            "path": path,
            "start_line": int(start_line),
            "end_line": int(end_line),
            "revision": "" if revision == "unknown" else revision,
            "reason": reason,
        })
    return records


def _upsert_dropped_source_notice(
        messages: list[dict[str, object]], records: Sequence[Mapping[str, object]],
) -> None:
    if duplicated_source_eviction_enabled():
        try:
            _upsert_duplicated_source_notice(messages, records)
        except Exception:
            # Malformed history must never fail request submission or retain a
            # legacy reread instruction whose range could not be verified.
            messages[:] = [
                message for message in messages
                if not (
                    isinstance(message, Mapping)
                    and _message_content(message).startswith(_DROPPED_SOURCE_RANGES_HEADER)
                )
            ]
            return
        return
    if not truthful_read_panels_enabled():
        _upsert_dropped_source_notice_r2fo(messages, records)
        return
    _upsert_truthful_dropped_source_notice(messages, records)


def _upsert_duplicated_source_notice(
        messages: list[dict[str, object]], records: Sequence[Mapping[str, object]],
) -> None:
    """Render only source locations proven present in the same request."""
    record_map: dict[
        tuple[str, int, int, str, str], dict[str, object]
    ] = {}
    panel_indices: list[int] = []
    try:
        current_sources = _current_source_segments(messages)
    except Exception:
        current_sources = []
    line_pattern = re.compile(
        r"^- (.+):(\d+)-(\d+) revision=(\S+) reason=(\S+)"
        r"(?:; present above at \[Current source: ([^\]]+)\] lines (\d+)-(\d+))?$"
    )

    def add(record: Mapping[str, object]) -> None:
        path = str(record.get("path", "")).replace("\\", "/").strip().removeprefix("./")
        start = record.get("start_line")
        end = record.get("end_line")
        if (not path or type(start) is not int or type(end) is not int
                or start < 1 or end < start):
            return
        normalized: dict[str, object] = {
            "path": path,
            "start_line": start,
            "end_line": end,
            "revision": str(record.get("revision") or ""),
            "reason": str(record.get("reason") or "budget_eviction"),
        }
        location = record.get("duplicate_current_source")
        if isinstance(location, Mapping):
            location_path = str(location.get("path", "")).replace("\\", "/").strip().removeprefix("./")
            location_start = location.get("start_line")
            location_end = location.get("end_line")
            if (location_path and type(location_start) is int and type(location_end) is int
                    and location_start >= 1 and location_end >= location_start):
                normalized["duplicate_current_source"] = {
                    "path": location_path,
                    "start_line": location_start,
                    "end_line": location_end,
                }
        key = (
            path, start, end, str(normalized["revision"]), str(normalized["reason"]),
        )
        record_map[key] = normalized

    for index, message in enumerate(messages):
        content = _message_content(message)
        if not content.startswith(_DROPPED_SOURCE_RANGES_HEADER):
            continue
        panel_indices.append(index)
        for line in content.splitlines():
            match = line_pattern.match(line.strip())
            if match is None:
                continue
            path, start, end, revision, reason, location_path, location_start, location_end = match.groups()
            if max(
                    len(start), len(end), len(location_start or ""), len(location_end or ""),
            ) > 10:
                continue
            parsed: dict[str, object] = {
                "path": path,
                "start_line": int(start),
                "end_line": int(end),
                "revision": "" if revision == "unknown" else revision,
                "reason": reason,
            }
            if location_path is not None:
                parsed["duplicate_current_source"] = {
                    "path": location_path,
                    "start_line": int(location_start),
                    "end_line": int(location_end),
                }
                location_is_current = (
                    location_path.replace("\\", "/").removeprefix("./")
                    == parsed["path"].replace("\\", "/").removeprefix("./")
                    and int(location_start) == parsed["start_line"]
                    and int(location_end) == parsed["end_line"]
                    and any(
                        source.get("path") == parsed["path"]
                        and source.get("revision") == parsed["revision"]
                        and int(source.get("start_line", 0)) <= int(location_start)
                        and int(source.get("end_line", 0)) >= int(location_end)
                        for source in current_sources
                    )
                )
                if location_is_current:
                    add(parsed)
    for record in records:
        add(record)

    if not record_map:
        if panel_indices:
            generated = set(panel_indices)
            messages[:] = [
                message for index, message in enumerate(messages) if index not in generated
            ]
        return
    lines = [
        _DROPPED_SOURCE_RANGES_HEADER,
        "Previously read source ranges omitted from this request:",
    ]
    for key in sorted(record_map):
        record = record_map[key]
        revision = str(record["revision"]) or "unknown"
        line = (
            f"- {record['path']}:{record['start_line']}-{record['end_line']} "
            f"revision={revision} reason={record['reason']}"
        )
        location = record.get("duplicate_current_source")
        if isinstance(location, Mapping):
            line += (
                "; present above at [Current source: "
                f"{location['path']}] lines {location['start_line']}-{location['end_line']}"
            )
        lines.append(line)
    notice = {
        "role": "user",
        "content": "\n".join(lines),
        "_request_budget_section": "stage_evidence",
    }
    if panel_indices:
        first = panel_indices[0]
        generated = set(panel_indices)
        messages[:] = [
            notice if index == first else message
            for index, message in enumerate(messages)
            if index == first or index not in generated
        ]
    else:
        messages.append(notice)


def _upsert_dropped_source_notice_r2fo(
        messages: list[dict[str, object]], records: Sequence[Mapping[str, object]],
) -> None:
    """Flag OFF: the r2FO function verbatim (byte-identical legacy panel and placement)."""
    if not records:
        return
    all_records: dict[tuple[str, int, int, str, str], dict[str, object]] = {}
    existing_index: int | None = None
    for index, message in enumerate(messages):
        content = _message_content(message)
        if not content.startswith(_DROPPED_SOURCE_RANGES_HEADER):
            continue
        existing_index = index
        for record in _parse_dropped_source_notice(content):
            key = (str(record["path"]), int(record["start_line"]),
                   int(record["end_line"]), str(record["revision"]),
                   str(record["reason"]))
            all_records[key] = record
    for record in records:
        path = str(record.get("path", "")).replace("\\", "/")
        start_line = record.get("start_line")
        end_line = record.get("end_line")
        if (not path or not isinstance(start_line, int) or isinstance(start_line, bool)
                or not isinstance(end_line, int) or isinstance(end_line, bool)
                or start_line < 1 or end_line < start_line):
            continue
        normalized = {
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "revision": str(record.get("revision") or ""),
            "reason": str(record.get("reason") or "budget_eviction"),
        }
        key = (path, start_line, end_line, normalized["revision"], normalized["reason"])
        all_records[key] = normalized
    if not all_records:
        return
    lines = [
        _DROPPED_SOURCE_RANGES_HEADER,
        "The source text for these previously read ranges is absent from this request.",
        "Pinned host range anchors (path:start-end); re-read before relying on the text:",
    ]
    for key in sorted(all_records):
        record = all_records[key]
        revision = (
            "unknown" if record["reason"] == "stale_revision"
            else str(record["revision"]) or "unknown"
        )
        lines.append(
            f"- {record['path']}:{record['start_line']}-{record['end_line']} "
            f"revision={revision} reason={record['reason']}"
        )
    notice = {
        "role": "user",
        "content": "\n".join(lines),
        "_request_budget_section": "stage_evidence",
    }
    if existing_index is None:
        messages.append(notice)
    else:
        messages[existing_index] = notice


def _upsert_truthful_dropped_source_notice(
        messages: list[dict[str, object]], records: Sequence[Mapping[str, object]],
) -> None:
    all_records: dict[tuple[str, int, int, str, str], dict[str, object]] = {}
    user_panel_indices: list[int] = []
    for index, message in enumerate(messages):
        content = _message_content(message)
        if not content.startswith(_DROPPED_SOURCE_RANGES_HEADER):
            continue
        if message.get("role") == "user":
            user_panel_indices.append(index)
        for record in _parse_dropped_source_notice(content):
            normalized_path = (str(record["path"]).replace("\\", "/")
                               .strip().removeprefix("./"))
            normalized_record = {**record, "path": normalized_path}
            key = (normalized_path, int(record["start_line"]),
                   int(record["end_line"]), str(record["revision"]),
                   str(record["reason"]))
            all_records[key] = normalized_record
    for record in records:
        path = (str(record.get("path", "")).replace("\\", "/")
                .strip().removeprefix("./"))
        start_line = record.get("start_line")
        end_line = record.get("end_line")
        if (not path or not isinstance(start_line, int) or isinstance(start_line, bool)
                or not isinstance(end_line, int) or isinstance(end_line, bool)
                or start_line < 1 or end_line < start_line):
            continue
        normalized = {
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "revision": str(record.get("revision") or ""),
            "reason": str(record.get("reason") or "budget_eviction"),
        }
        key = (path, start_line, end_line, normalized["revision"], normalized["reason"])
        all_records[key] = normalized
    if not truthful_read_panels_enabled():
        _upsert_legacy_dropped_source_notice(messages, all_records, user_panel_indices)
        return
    generated_indices = user_panel_indices
    if not all_records:
        if generated_indices:
            messages[:] = [
                message for index, message in enumerate(messages)
                if index not in set(generated_indices)
            ]
        return

    delivered = _source_delivery_ranges(messages)
    remaining_records: dict[tuple[str, int, int, str, str], dict[str, object]] = {}
    for record in all_records.values():
        path = str(record["path"])
        start = int(record["start_line"])
        end = int(record["end_line"])
        revision = str(record["revision"])
        reason = str(record["reason"])
        for uncovered_start, uncovered_end in _uncovered_source_ranges(
                start, end, delivered.get((path, revision), ())):
            remainder = {
                "path": path,
                "start_line": uncovered_start,
                "end_line": uncovered_end,
                "revision": revision,
                "reason": reason,
            }
            key = (path, uncovered_start, uncovered_end, revision, reason)
            remaining_records[key] = remainder

    if not remaining_records:
        if generated_indices:
            generated = set(generated_indices)
            messages[:] = [message for index, message in enumerate(messages) if index not in generated]
        return

    lines = [
        _DROPPED_SOURCE_RANGES_HEADER,
        "Source ranges delivered in this request:",
    ]
    delivered_rows = [
        (path, revision, start, end)
        for (path, revision), spans in sorted(delivered.items())
        for start, end in spans
    ]
    if delivered_rows:
        for path, revision, start, end in delivered_rows:
            lines.append(
                f"- {path}:{start}-{end} revision={revision or 'unknown'}"
            )
    else:
        lines.append("- none")
    lines.append("Ranges omitted by request budgeting:")
    for key in sorted(remaining_records):
        record = remaining_records[key]
        revision = str(record["revision"]) or "unknown"
        lines.append(
            f"- {record['path']}:{record['start_line']}-{record['end_line']} "
            f"revision={revision} reason={record['reason']}"
        )
    notice = {
        "role": "user",
        "content": "\n".join(lines),
        "_request_budget_section": "stage_evidence",
    }
    if generated_indices:
        first = generated_indices[0]
        generated = set(generated_indices)
        messages[:] = [
            notice if index == first else message
            for index, message in enumerate(messages)
            if index == first or index not in generated
        ]
    else:
        messages.append(notice)


def _upsert_legacy_dropped_source_notice(
        messages: list[dict[str, object]],
        records: Mapping[tuple[str, int, int, str, str], Mapping[str, object]],
        generated_indices: Sequence[int],
) -> None:
    """Keep the pre-item-9 panel bytes when truthful panels are not enabled."""
    if not records:
        if generated_indices:
            generated = set(generated_indices)
            messages[:] = [
                message for index, message in enumerate(messages) if index not in generated
            ]
        return
    lines = [
        _DROPPED_SOURCE_RANGES_HEADER,
        "The source text for these previously read ranges is absent from this request.",
        "Pinned host range anchors (path:start-end); re-read before relying on the text:",
    ]
    for key in sorted(records):
        record = records[key]
        revision = str(record.get("revision") or "unknown")
        lines.append(
            f"- {record['path']}:{record['start_line']}-{record['end_line']} "
            f"revision={revision} reason={record['reason']}"
        )
    notice = {
        "role": "user",
        "content": "\n".join(lines),
        "_request_budget_section": "stage_evidence",
    }
    if generated_indices:
        first = generated_indices[0]
        generated = set(generated_indices)
        messages[:] = [
            notice if index == first else message
            for index, message in enumerate(messages)
            if index == first or index not in generated
        ]
    else:
        messages.append(notice)


def _has_read_tool(body: Mapping[str, object]) -> bool:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, Mapping) else tool.get("name")
        if str(name or "") == "read":
            return True
    return False


def _required_source_excerpt_path(
        message: Mapping[str, object], index: int,
) -> str | None:
    content = _message_content(message)
    if (message.get("role") == "user"
            and content.lstrip().lower().startswith("[stateframe]")):
        identity = _source_identity(message)
        if identity is not None and identity[0] == "source":
            return identity[1].replace("\\", "/")
        return source_window_path(content)
    if section_name(message, index) != "file_excerpts":
        return None
    if message.get("_framebuild_certified_evidence") is True or isinstance(
        message.get("_framebuild_evidence_envelope"), Mapping
    ):
        return None
    identity = _source_identity(message)
    if identity is not None and identity[0] == "source":
        return identity[1].replace("\\", "/")
    return source_window_path(content)


def _defer_required_source_paths(
        body: dict[str, object], *, required_source_paths: frozenset[str] | None,
        required_source_order: Sequence[str], deferred_source_paths: Sequence[str],
        measure_after: Callable[[Mapping[str, object]], int],
        over_limit: Callable[[Mapping[str, object], int], bool],
        current_tokens: int,
) -> tuple[int, tuple[str, ...]]:
    """Remove least-relevant required excerpts only with an executable read route."""
    if (required_source_paths is None or not required_source_order
            or not _has_read_tool(body)):
        return current_tokens, tuple(deferred_source_paths)
    from kernel_request_source_window import upsert_deferred_source_notice

    deferred = list(deferred_source_paths)
    deferred_set = set(deferred)
    for path in reversed(required_source_order):
        if path not in required_source_paths or path in deferred_set:
            continue
        messages = body.get("messages")
        if not isinstance(messages, list):
            break
        removed_indices = {
            index for index, message in enumerate(messages)
            if isinstance(message, Mapping)
            and not (
                message.get("role") == "user"
                and _message_content(message).lstrip().lower().startswith("[stateframe]")
            )
            and _required_source_excerpt_path(message, index) == path
        }
        if not removed_indices:
            continue
        trial = copy.deepcopy(body)
        trial_messages = trial.get("messages")
        if not isinstance(trial_messages, list):
            continue
        trial["messages"] = [
            message for index, message in enumerate(trial_messages)
            if index not in removed_indices
        ]
        trial_deferred = [*deferred, path]
        upsert_deferred_source_notice(trial["messages"], trial_deferred)
        measured = measure_after(trial)
        body.clear()
        body.update(trial)
        deferred = trial_deferred
        deferred_set.add(path)
        current_tokens = measured
        if not over_limit(body, current_tokens):
            break
    return current_tokens, tuple(deferred)


def _owner_first_replace_optional_context(
        body: dict[str, object], *, max_request_bytes: int,
        current_tokens: int,
        measure_after: Callable[[Mapping[str, object]], int],
        protected_contents: Sequence[str],
        protected_messages: Sequence[Mapping[str, object]],
        required_source_paths: frozenset[str] | None,
        allocation_plan: dict[str, object],
        dropped_source_ranges: list[dict[str, object]],
) -> int:
    """Replace optional dependency source with revision-bound read handles first."""
    from kernel_owner_first import optional_recall_record, render_recall_inventory
    from kernel_request_source_window import parse_window

    messages = body.get("messages")
    if not isinstance(messages, list):
        return current_tokens
    requested_paths = {str(path).replace("\\", "/")
                       for path in allocation_plan.get("requested_paths", ()) or ()}
    candidates: list[tuple[int, dict[str, object]]] = []
    for index, raw in enumerate(messages):
        if (not isinstance(raw, Mapping) or raw.get("tool_calls")
                or str(raw.get("role", "user")) in {"assistant", "tool"}):
            continue
        content = _message_content(raw)
        is_optional_head = content.startswith((
            "[MODULE HEAD", "[IMPORT GRAPH FACT", "[INITIALIZATION / CONSTRUCTION FACT",
        ))
        metadata = parse_window(content)
        is_optional_window = (
            isinstance(metadata, Mapping)
            and str(metadata.get("path", "")).replace("\\", "/") not in requested_paths
        )
        if not (is_optional_head or is_optional_window):
            continue
        if _is_protected_message(raw, protected_contents, protected_messages,
                                 required_source_paths):
            continue
        record = optional_recall_record(raw, allocation_plan)
        if record is not None:
            candidates.append((index, record))
    if not candidates:
        return current_tokens

    current_bytes = _wire_bytes(body)
    inventory = list(allocation_plan.get("recall_inventory", ()) or ())
    replaced_ranges: list[dict[str, object]] = []
    candidate_indices: set[int] = set()
    revised: list[dict[str, object]] = []
    for index, record in candidates:
        candidate_indices.add(index)
        for span in record.get("ranges", ()) or ():
            if isinstance(span, Mapping):
                replaced_ranges.append({
                    "path": record["path"], "revision": record["revision"],
                    "start_line": span["start_line"], "end_line": span["end_line"],
                    "reason": "replaced_by_revision_bound_recall_inventory",
                })
    inventory.extend(record for _index, record in candidates)
    inventory_text = render_recall_inventory(inventory)
    first_index = min(candidate_indices)
    for index, raw in enumerate(messages):
        if index == first_index:
            revised.append({"role": "user", "content": inventory_text,
                            "_request_budget_section": "environment_state"})
        if index not in candidate_indices:
            revised.append(dict(raw) if isinstance(raw, Mapping) else raw)
    _upsert_dropped_source_notice(revised, [*dropped_source_ranges, *replaced_ranges])
    candidate_body = dict(body)
    candidate_body["messages"] = revised
    candidate_bytes = _wire_bytes(candidate_body)
    candidate_tokens = measure_after(candidate_body)
    if (candidate_bytes >= current_bytes and candidate_tokens >= current_tokens):
        return current_tokens
    allocation_plan["recall_inventory"] = inventory
    allocation_plan["optional_context_replaced"] = len(candidates)
    allocation_plan["optional_context_paths"] = sorted({
        str(record["path"]) for _index, record in candidates
    })
    allocation_plan["dropped_ranges"] = [
        *list(allocation_plan.get("dropped_ranges", ()) or ()), *replaced_ranges,
    ]
    dropped_source_ranges.extend(replaced_ranges)
    body["messages"] = revised
    return candidate_tokens


def _dropped_source_notice_text(messages: Sequence[Mapping[str, object]]) -> str | None:
    for message in messages:
        content = _message_content(message)
        if content.startswith(_DROPPED_SOURCE_RANGES_HEADER):
            return content
    return None


def _normalized_messages(
        messages: Sequence[Mapping[str, object]], evidence_ids: Sequence[str],
        protected_contents: Sequence[str] = (),
        protected_messages: Sequence[Mapping[str, object]] = (),
        native_tools: Sequence[Mapping[str, object]] = (),
        required_source_paths: frozenset[str] | None = None,
) -> tuple[list[dict[str, object]], list[str], list[int], list[dict[str, object]]]:
    identities: dict[int, tuple[str, str, str, tuple[int, int] | None, str]] = {}
    latest_revisions: dict[tuple[str, str], str] = {}
    for index, message in enumerate(messages):
        if _is_protected_message(message, protected_contents, protected_messages, required_source_paths):
            continue
        identity = _source_identity(message)
        if identity is None:
            continue
        identities[index] = identity
        kind, path, revision, _source_range = identity
        scope = _certified_scope(message)
        identities[index] = (kind, path, revision, _source_range, scope)
        if revision:
            latest_revisions[(path, scope)] = revision
    stale_indices: set[int] = set()
    stale_reasons: dict[int, str] = {}
    retained_ranges: dict[tuple[str, str, str, str], list[tuple[int, int]]] = {}
    for index in range(len(messages) - 1, -1, -1):
        identity = identities.get(index)
        if identity is None:
            continue
        # Tool reads remain individual LRU units. Their freshness and overlap
        # are handled only after the two-span floor is known for this request.
        if section_name(messages[index], index) == "file_reads":
            continue
        kind, path, revision, source_range, scope = identity
        latest_revision = latest_revisions.get((path, scope))
        if (revision and latest_revision is not None
                and revision != latest_revision):
            # A newer revision supersedes this path even when a legacy header
            # does not provide enough body metadata to anchor a line range.
            stale_indices.add(index)
            stale_reasons[index] = "stale_revision"
            continue
        same_revision_key = (kind, path, revision, scope)
        prior_ranges = retained_ranges.setdefault(same_revision_key, [])
        if source_range is not None and any(
                _range_contains(prior, source_range) for prior in prior_ranges):
            stale_indices.add(index)
            stale_reasons[index] = "range_already_retained"
        elif source_range is not None:
            # Partial overlap still carries uncovered lines; keep it.
            prior_ranges.append(source_range)
    seen: set[str] = set()
    normalized: list[dict[str, object]] = []
    duplicate_sections: list[str] = []
    dropped_source_ranges: list[dict[str, object]] = []
    for index, raw in enumerate(messages):
        if not isinstance(raw, Mapping):
            continue
        section = section_name(raw, index)
        if index in stale_indices:
            duplicate_sections.append(f"stale:{section}")
            if stale_reasons.get(index) == "stale_revision":
                record = _source_range_record(raw, reason="stale_revision")
                if record is not None:
                    dropped_source_ranges.append(record)
            continue
        message = copy.deepcopy(dict(raw))
        if section == "file_reads":
            normalized.append(message)
            continue
        if isinstance(message.get("content"), str):
            if not _is_protected_message(message, protected_contents, protected_messages, required_source_paths):
                message["content"] = _normalize_content(message, section, evidence_ids, native_tools)
        key = _message_key(message)
        if key in seen:
            duplicate_sections.append(f"duplicate:{section}")
            record = _source_range_record(message, reason="duplicate_message")
            if record is not None:
                dropped_source_ranges.append(record)
            continue
        seen.add(key)
        normalized.append(message)
    return normalized, duplicate_sections, sorted(stale_indices), dropped_source_ranges


def _read_working_set_groups(
        messages: Sequence[Mapping[str, object]], *,
        protected_contents: Sequence[str],
        protected_messages: Sequence[Mapping[str, object]],
        required_source_paths: frozenset[str] | None,
) -> list[dict[str, object]]:
    """Return source spans in LRU order, keeping read tool exchanges atomic."""
    duplicate_only = duplicated_source_eviction_enabled()
    current_sources: list[dict[str, object]] = []
    if duplicate_only:
        try:
            current_sources = _current_source_segments(messages)
        except Exception:
            # Uncertain source parsing keeps tool-result bytes in the request.
            current_sources = []

    assistant_calls: dict[int, list[str]] = {}
    call_results: dict[str, list[int]] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", ""))
        if role == "assistant":
            raw_calls = message.get("tool_calls")
            if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, (str, bytes, bytearray)):
                for call in raw_calls:
                    if not isinstance(call, Mapping):
                        continue
                    call_id = str(call.get("id", "")).strip()
                    if call_id:
                        assistant_calls.setdefault(index, []).append(call_id)
        elif role == "tool":
            call_id = str(message.get("tool_call_id", "")).strip()
            if call_id:
                call_results.setdefault(call_id, []).append(index)

    groups: list[dict[str, object]] = []
    paired_indices: set[int] = set()
    for assistant_index, call_ids in assistant_calls.items():
        result_indices = list(dict.fromkeys(
            result_index for call_id in call_ids
            for result_index in call_results.get(call_id, ())
        ))
        read_indices = [result_index for result_index in result_indices
                        if section_name(messages[result_index], result_index) == "file_reads"]
        if not read_indices:
            continue
        complete = all(call_results.get(call_id) for call_id in call_ids)
        read_only = all(
            section_name(messages[result_index], result_index) == "file_reads"
            for result_index in result_indices
        )
        indices = tuple(sorted({assistant_index, *result_indices}))
        records: list[dict[str, object]] = []
        duplicate_proof_missing = False
        for result_index in read_indices:
            result = messages[result_index]
            if not isinstance(result, Mapping):
                duplicate_proof_missing = duplicate_only
                continue
            record = _source_range_record(result, reason="lru_eviction")
            if record is None:
                duplicate_proof_missing = duplicate_only
                continue
            if duplicate_only:
                identity = _source_identity(result)
                location = (
                    _current_source_duplicate_location(
                        messages, result, current_sources,
                    )
                    if result.get("role") == "tool"
                    and identity is not None and identity[0] == "read"
                    else None
                )
                if location is None:
                    duplicate_proof_missing = True
                else:
                    record["duplicate_current_source"] = _duplicate_source_location_record(location)
            records.append(record)
        span_count = len(read_indices)
        protected = (
            not complete or not read_only or len(records) != span_count
            or duplicate_proof_missing
            or any(_is_protected_message(
                messages[index], protected_contents, protected_messages,
                required_source_paths,
            ) for index in indices)
        )
        groups.append({
            "indices": indices,
            "recency": max(indices),
            "records": records,
            "span_count": span_count,
            "read_span_count": span_count,
            "protected": protected,
        })
        paired_indices.update(indices)

    for index, message in enumerate(messages):
        if index in paired_indices or not isinstance(message, Mapping):
            continue
        section = section_name(message, index)
        if section not in _READ_EVIDENCE_SECTIONS:
            continue
        record = _source_range_record(message, reason="lru_eviction")
        if duplicate_only:
            identity = _source_identity(message)
            location = (
                _current_source_duplicate_location(
                    messages, message, current_sources,
                )
                if message.get("role") == "tool"
                and identity is not None and identity[0] == "read"
                else None
            )
            if record is not None and location is not None:
                record["duplicate_current_source"] = _duplicate_source_location_record(location)
        protected = (
            record is None or _is_protected_message(
                message, protected_contents, protected_messages,
                required_source_paths,
            )
            or (
                duplicate_only
                and not (
                    message.get("role") == "tool"
                    and isinstance(record, Mapping)
                    and isinstance(record.get("duplicate_current_source"), Mapping)
                )
            )
        )
        groups.append({
            "indices": (index,),
            "recency": index,
            "records": [record] if record is not None else [],
            "span_count": 1,
            "read_span_count": 1 if section == "file_reads" else 0,
            "protected": protected,
        })
    return sorted(groups, key=lambda group: int(group["recency"]))


def _read_spans_retained(messages: Sequence[Mapping[str, object]]) -> int:
    groups = _read_working_set_groups(
        messages,
        protected_contents=(),
        protected_messages=(),
        required_source_paths=None,
    )
    return sum(
        int(group["read_span_count"])
        for group in groups
        if isinstance(group.get("read_span_count"), int)
    )


def _protected_working_source_paths(
        messages: Sequence[Mapping[str, object]], *,
        protected_contents: Sequence[str],
        protected_messages: Sequence[Mapping[str, object]],
        required_source_paths: frozenset[str] | None,
) -> list[str]:
    paths = set(required_source_paths or ())
    for message in messages:
        if not _is_protected_message(
                message, protected_contents, protected_messages, required_source_paths):
            continue
        path = source_window_path(_message_content(message))
        identity = _source_identity(message)
        if path:
            paths.add(path.replace("\\", "/"))
        elif identity is not None:
            paths.add(identity[1].replace("\\", "/"))
    return sorted(paths)


def _diet_contract_fields(
        meter: RequestTokenMeter, *, measure_method: str,
        request_tokens_before: int, request_tokens_after: int,
        max_request_bytes: int, token_budget: int, read_spans_available: int,
        read_spans_retained: int, protected_paths: Sequence[str],
) -> dict[str, object]:
    return {
        "measure_method": measure_method,
        "request_tokens_before": request_tokens_before,
        "request_tokens_after": request_tokens_after,
        "max_request_bytes": max_request_bytes,
        "token_budget": token_budget,
        "over_budget": request_tokens_before > token_budget,
        "protected_paths": list(protected_paths),
        "read_spans_available": read_spans_available,
        "read_spans_retained": read_spans_retained,
        **meter.journal_fields(measure_method),
    }


class _MeasurementModeChanged(RuntimeError):
    """Restart a diet with one stable measurement method after tokenizer failure."""


def _protected_request_sizes(
        body: Mapping[str, object], *,
        protected_contents: Sequence[str],
        protected_messages: Sequence[Mapping[str, object]],
        required_source_paths: frozenset[str] | None,
) -> tuple[int, dict[str, int]]:
    messages = body.get("messages", [])
    message_list = messages if isinstance(messages, list) else []
    retained: list[dict[str, object]] = []
    section_bytes: dict[str, int] = {}
    for index, message in enumerate(message_list):
        if not isinstance(message, Mapping):
            continue
        section = section_name(message, index)
        if (section not in _PROTECTED_SECTIONS
                and not _is_protected_message(
                    message, protected_contents, protected_messages,
                    required_source_paths,
                )):
            continue
        retained.append(dict(message))
        section_bytes[section] = section_bytes.get(section, 0) + _wire_bytes(message)
    minimum = dict(body)
    minimum["messages"] = retained
    return _wire_bytes(minimum), section_bytes


def _section_for_messages(messages: Sequence[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped = {name: [] for name in SECTION_NAMES}
    for index, message in enumerate(messages):
        section = section_name(message, index)
        grouped.setdefault(section, []).append(dict(message))
    return grouped


def _section_stats(body: Mapping[str, object]) -> dict[str, dict[str, int]]:
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    message_list = ([message for message in messages if isinstance(message, Mapping)]
                    if isinstance(messages, list) else [])
    tool_list = ([tool for tool in tools if isinstance(tool, Mapping)]
                 if isinstance(tools, list) else [])
    grouped = _section_for_messages(message_list)
    result = {
        name: {
            "bytes": sum(_wire_bytes(message) for message in grouped.get(name, [])),
            "tokens": sum(token_estimate(message) for message in grouped.get(name, [])),
            "messages": len(grouped.get(name, [])),
        }
        for name in SECTION_NAMES
        if name not in {"schemas", "request_framing"}
    }
    message_array = _wire_bytes(message_list)
    message_items = sum(_wire_bytes(message) for message in message_list)
    tool_array = _wire_bytes(tool_list)
    tool_items = sum(_wire_bytes(tool) for tool in tool_list)
    result["schemas"] = {
        "bytes": tool_array,
        "tokens": token_estimate(tool_list),
        "messages": len(tool_list),
        "canonical_bytes": len(json.dumps(
            tool_list, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode()),
    }
    result["request_framing"] = {
        "bytes": max(0, _wire_bytes(body) - message_array - tool_array),
        "tokens": max(0, token_estimate(body) - token_estimate(message_list)
                    - token_estimate(tool_list)),
        "messages": 0,
        "array_overhead_bytes": (message_array - message_items) + (tool_array - tool_items),
    }
    return result


def breakdown(body: Mapping[str, object], stage: str = "S1") -> dict[str, object]:
    """Return a byte/token breakdown suitable for the REQDIET report."""
    if stage not in STAGE_SECTION_ALLOWLIST:
        raise ValueError(f"unknown request-diet stage: {stage}")
    stats = _section_stats(body)
    return {
        "schema": SCHEMA,
        "stage": stage,
        "request_bytes": _wire_bytes(body),
        "request_tokens_proxy": token_estimate(body),
        "sections": stats,
        "allowlist": sorted(STAGE_SECTION_ALLOWLIST[stage]),
    }


def _audit(before: Mapping[str, object], after: Mapping[str, object], stage: str,
           *, dropped_sections: Sequence[str], dropped_items: Sequence[str],
           duplicate_items: Sequence[str], stale_indices: Sequence[int],
           evidence_ids: Sequence[str], menu: Sequence[str], reason: str,
           dropped_source_ranges: Sequence[Mapping[str, object]] = ()) -> dict[str, object]:
    before_stats = _section_stats(before)
    after_stats = _section_stats(after)
    source_headers: dict[str, list[str]] = {}
    for label, body in (("before", before), ("after", after)):
        messages = body.get("messages", [])
        if not isinstance(messages, list):
            continue
        source_headers[label] = [
            _message_content(message).splitlines()[0][:160]
            for index, message in enumerate(messages)
            if isinstance(message, Mapping)
            and section_name(message, index) == "file_excerpts"
            and _message_content(message).splitlines()
        ]
    rows = []
    for name in SECTION_NAMES:
        left = before_stats.get(name, {"bytes": 0, "tokens": 0, "messages": 0})
        right = after_stats.get(name, {"bytes": 0, "tokens": 0, "messages": 0})
        rows.append({
            "section": name,
            "before_bytes": left.get("bytes", 0),
            "after_bytes": right.get("bytes", 0),
            "before_tokens": left.get("tokens", 0),
            "after_tokens": right.get("tokens", 0),
            "before_messages": left.get("messages", 0),
            "after_messages": right.get("messages", 0),
        })
    return {
        "event_schema": SCHEMA,
        "stage": stage,
        "action": ("noop" if before == after
                   else "diet" if (dropped_sections or dropped_items or dropped_source_ranges or stale_indices)
                   else "normalize"),  # 10-08: normalize = boilerplate compaction only, nothing evicted
        "reason": reason,
        "request_bytes_before": _wire_bytes(before),
        "request_bytes_after": _wire_bytes(after),
        "request_tokens_before": token_estimate(before),
        "request_tokens_after": token_estimate(after),
        "dropped_sections": list(dict.fromkeys(dropped_sections)),
        "dropped_items": list(dropped_items),
        "duplicate_items": list(duplicate_items),
        "stale_read_indices": list(stale_indices),
        "allowlist": sorted(STAGE_SECTION_ALLOWLIST[stage]),
        "required_evidence_ids": list(evidence_ids),
        "menu_actions": list(menu),
        "source_headers": source_headers,
        "dropped_source_ranges": [dict(record) for record in dropped_source_ranges],
        "read_spans_retained": _read_spans_retained(
            [message for message in after.get("messages", [])
             if isinstance(message, Mapping)]
        ),
        "dropped_ranges_notice_text": _dropped_source_notice_text(
            [message for message in after.get("messages", [])
             if isinstance(message, Mapping)]
        ),
        "sections": rows,
        "producer": "kernel_request_diet",
    }


def _history_messages_for_append_only(
    messages: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        message for message in messages
        if message.get("role") in {"assistant", "tool"}
    )


@profiled_function_span("request_diet")
def apply_diet(
        body: dict[str, object], *, stage: str, max_request_bytes: int,
        emit: Callable[..., object] | None = None, request_id: str | None = None,
        step: int | None = None, protected_contents: Sequence[str] = (),
        protected_messages: Sequence[Mapping[str, object]] = (),
        required_source_paths: Sequence[str] | None = None,
        required_source_order: Sequence[str] = (),
        deferred_source_paths: Sequence[str] = (),
        token_budget: int | None = None,
        tokenizer: Callable[[str], object] | None = None,
        tokenizer_endpoint: str | None = None,
        token_meter: RequestTokenMeter | None = None,
        allocation_plan: dict[str, object] | None = None,
        _measurement_retry: bool = True,
        _incremental_accounting: bool = True,
) -> dict[str, object]:
    """Diet requests that exceed either the token budget or serialized byte cap.

    ``required_source_paths`` names source windows reserved through ordinary
    trimming. If those excerpts still prevent admission, the lowest-relevance
    paths may be deferred only when the request carries the ``read`` tool and an
    exact host notice. ``None`` keeps the legacy bounded-window protection.
    """
    initial_protected_messages = protected_messages
    required_source_order = tuple(dict.fromkeys(
        str(path).replace("\\", "/").strip()
        for path in required_source_order if str(path).strip()
    ))
    deferred_paths = list(dict.fromkeys(
        str(path).replace("\\", "/").strip()
        for path in deferred_source_paths if str(path).strip()
    ))
    if required_source_paths is not None:
        required_source_paths = frozenset(
            str(path).replace("\\", "/").strip()
            for path in required_source_paths if str(path).strip()
        )
        if not required_source_order:
            required_source_order = tuple(sorted(required_source_paths))
    if not isinstance(body, dict):
        raise TypeError("request diet requires a mutable request dictionary")
    if stage not in STAGE_SECTION_ALLOWLIST:
        raise ValueError(f"unknown request-diet stage: {stage}")
    if max_request_bytes <= 0:
        raise ValueError("request diet budget must be positive")
    if token_budget is None:
        token_budget = max(1, round(max_request_bytes / INITIAL_BYTES_PER_TOKEN))
    if type(token_budget) is not int or token_budget <= 0:
        raise ValueError("request diet token budget must be a positive integer")
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise TypeError("request diet requires messages to be a list")
    from kernel_cache_layout import trailing_panels_enabled
    if trailing_panels_enabled():
        protected = list(protected_messages)
        for message in _history_messages_for_append_only(messages):
            if message not in protected:
                protected.append(message)
        protected_messages = tuple(protected)
    before = copy.deepcopy(body)
    meter = token_meter or RequestTokenMeter(
        tokenizer=tokenizer,
        endpoint=resolve_tokenize_url(endpoint=tokenizer_endpoint),
    )
    meter.begin_request(request_id)
    read_spans_available = _read_spans_retained(
        [message for message in messages if isinstance(message, Mapping)]
    )
    protected_paths = _protected_working_source_paths(
        [message for message in messages if isinstance(message, Mapping)],
        protected_contents=protected_contents,
        protected_messages=protected_messages,
        required_source_paths=(
            frozenset(required_source_paths) if required_source_paths is not None else None
        ),
    )
    request_tokens_before, measure_method = meter.measure(before)
    removal_accounting = None
    byte_rejected_without_measurement = False
    # Retained message objects (identity, not ids) at the last real measurement
    # of `after`; lets the final verify reuse that count instead of re-measuring.
    last_measured_messages: list[object] | None = None

    def restart_with_stable_method() -> dict[str, object]:
        if not _measurement_retry:
            raise RuntimeError("request diet tokenizer changed measurement method twice")
        return apply_diet(
            body,
            stage=stage,
            max_request_bytes=max_request_bytes,
            emit=emit,
            request_id=request_id,
            step=step,
            protected_contents=protected_contents,
            protected_messages=protected_messages,
            required_source_paths=required_source_paths,
            required_source_order=required_source_order,
            deferred_source_paths=deferred_paths,
            token_budget=token_budget,
            token_meter=meter,
            allocation_plan=allocation_plan,
            _measurement_retry=False,
            _incremental_accounting=_incremental_accounting,
        )

    def measure_after(candidate: Mapping[str, object]) -> int:
        nonlocal measure_method, byte_rejected_without_measurement, last_measured_messages
        last_measured_messages = None
        if removal_accounting is not None and candidate is after:
            size = removal_accounting.update(candidate["messages"])
            if size > max_request_bytes:
                if meter._deadline is not None and time.monotonic() >= meter._deadline:
                    raise TimeoutError("tokenizer request deadline expired")
                # Exact byte rejection is sufficient to continue the same pop
                # sequence. This sentinel is not measurement evidence; the
                # complete candidate is measured when the removal pass ends.
                byte_rejected_without_measurement = True
                return token_budget + 1
        byte_rejected_without_measurement = False
        if (removal_accounting is not None and candidate is after
                and type(meter) is RequestTokenMeter
                and "measure" not in meter.__dict__
                and meter._availability == "unavailable"):
            size = removal_accounting.update(candidate["messages"])
            measured, method = meter._measure_with_wire_bytes(candidate, size)
        else:
            measured, method = meter.measure(candidate)
        # Bounds reject oversized candidates without selecting a measurement
        # mode. Only actual tokenize -> fallback consumes the one restart.
        if method == "verified_byte_bound":
            return measured
        if measure_method == "verified_byte_bound":
            measure_method = method
        if method != measure_method:
            raise _MeasurementModeChanged
        if candidate is after:
            last_measured_messages = list(candidate["messages"])
        return measured

    def over_limit(candidate: Mapping[str, object], measured_tokens: int) -> bool:
        if measured_tokens > token_budget:
            return True
        size = (removal_accounting.update(candidate["messages"])
                if removal_accounting is not None and candidate is after
                else _wire_bytes(candidate))
        return size > max_request_bytes

    if allocation_plan is None:
        from kernel_owner_first import allocation_enabled, build_priority_plan
        if allocation_enabled():
            try:
                from kernel_pilot import ACTIVE_RUNTIME
                active_runtime = ACTIVE_RUNTIME.get()
            except (ImportError, LookupError):
                active_runtime = None
            allocation_plan = build_priority_plan(body, active_runtime)
    owner_first_changed = False
    owner_first_dropped_ranges: list[dict[str, object]] = []
    if allocation_plan is not None:
        from kernel_owner_first import allocation_enabled, protected_priority_messages
        if allocation_enabled():
            protected_messages = tuple(protected_messages) + protected_priority_messages(
                body, allocation_plan,
            )
            current_bytes = _wire_bytes(body)
            if request_tokens_before > token_budget or current_bytes > max_request_bytes:
                try:
                    _owner_first_replace_optional_context(
                        body,
                        max_request_bytes=max_request_bytes,
                        current_tokens=request_tokens_before,
                        measure_after=measure_after,
                        protected_contents=protected_contents,
                        protected_messages=protected_messages,
                        required_source_paths=required_source_paths,
                        allocation_plan=allocation_plan,
                        dropped_source_ranges=owner_first_dropped_ranges,
                    )
                except _MeasurementModeChanged:
                    return restart_with_stable_method()
                owner_first_changed = body != before
                if owner_first_changed:
                    messages = body.get("messages", [])
                    if not isinstance(messages, list):
                        raise TypeError("owner-first request diet requires list messages")

    evidence_ids = required_evidence_ids(messages)
    menu = menu_actions(messages)
    native_tools = tuple(
        tool for tool in body.get("tools", ()) if isinstance(tool, Mapping)
    )
    # The caller may protect source/menu messages explicitly. Keep non-read
    # tool exchanges intact; read exchanges are removable only as an LRU unit.
    protected_messages = tuple(protected_messages) + _paired_tool_exchange_messages(messages)
    normalized, duplicate_items, stale_indices, stale_source_ranges = _normalized_messages(
        messages, evidence_ids, protected_contents, protected_messages,
        native_tools, required_source_paths,
    )
    allowlist = STAGE_SECTION_ALLOWLIST[stage]
    kept: list[dict[str, object]] = []
    dropped_sections: list[str] = []
    dropped_items: list[str] = []
    dropped_source_ranges = [*stale_source_ranges, *owner_first_dropped_ranges]
    for index, message in enumerate(normalized):
        section = section_name(message, index)
        if (section not in allowlist and section not in _READ_EVIDENCE_SECTIONS
                and not _is_protected_message(message, protected_contents, protected_messages, required_source_paths)):
            dropped_sections.append(section)
            dropped_items.append(f"{section}:{index}")
            continue
        kept.append(message)

    after = dict(body)
    after["messages"] = kept
    _upsert_dropped_source_notice(after["messages"], dropped_source_ranges)
    if deferred_paths:
        from kernel_request_source_window import upsert_deferred_source_notice
        upsert_deferred_source_notice(after["messages"], deferred_paths)
    if not set(evidence_ids).issubset(required_evidence_ids(kept)):
        missing = sorted(set(evidence_ids) - set(required_evidence_ids(kept)))
        kept.append({
            "role": "user",
            "content": "[REQDIET required evidence] ids=" + ",".join(missing),
        })
        after["messages"] = kept

    if (after == before and not owner_first_changed
            and request_tokens_before <= token_budget
            and _wire_bytes(before) <= max_request_bytes):
        # Reuse only an unchanged body's count. Even shrinking normalization
        # changes projection identity; TR's cache owns any safe count reuse.
        request_tokens_after = request_tokens_before
    else:
        try:
            request_tokens_after = measure_after(after)
        except _MeasurementModeChanged:
            return restart_with_stable_method()
    if (_incremental_accounting and diet_incr_enabled()
            and over_limit(after, request_tokens_after)):
        from kernel_diet_accounting import RemovalAccounting
        removal_accounting = RemovalAccounting(after, _wire_bytes)
    # Remove narrative sections first. Source evidence is handled below as an
    # LRU of individual, revision-bound spans rather than a removable section.
    for section in _DROP_ORDER:
        if not over_limit(after, request_tokens_after):
            break
        removable = [
            message for index, message in enumerate(after["messages"])
            if section_name(message, index) == section
            and section not in _PROTECTED_SECTIONS
            and not _is_protected_message(message, protected_contents, protected_messages, required_source_paths)
        ]
        if not removable:
            continue
        dropped_sections.append(section)
        dropped_items.extend(f"{section}:{index}" for index, message in enumerate(after["messages"])
                            if section_name(message, index) == section
                            and not _is_protected_message(
                                message, protected_contents, protected_messages,
                                required_source_paths))
        after["messages"] = [
            message for index, message in enumerate(after["messages"])
            if (section_name(message, index) != section
                    or _is_protected_message(message, protected_contents, protected_messages, required_source_paths))
        ]
        try:
            request_tokens_after = measure_after(after)
        except _MeasurementModeChanged:
            return restart_with_stable_method()

    while True:
        groups = _read_working_set_groups(
            after["messages"],
            protected_contents=protected_contents,
            protected_messages=protected_messages,
            required_source_paths=required_source_paths,
        )
        # 10-08 (BUILDLEAD merge fix): auto-delivered source excerpts ([Current source: ...], read_span_count == 0)
        # are evidence too - they are the working source the frame arm presents. Counting them as 0 let the diet drop
        # EVERY source span under a tight cap (L104 fixture: only the StateFrame + notice survived), which is the
        # 918/921 zero-frame failure class. Every source-evidence group now weighs >= 1 toward the floor.
        # Rule: an auto-delivered source excerpt (read_span_count == 0) is expendable ONLY while a protected source
        # group (the presentation's working source) remains; tool read spans keep the 2-span floor. Otherwise the
        # request is infeasible and defers (journaled), never silently stripped of all source.
        read_span_count = sum(int(group["read_span_count"]) for group in groups)
        protected_source_remains = any(group["protected"] for group in groups)
        if not over_limit(after, request_tokens_after):
            break
        candidates = [
            group for group in groups
            if not group["protected"]
            and (
                (int(group["read_span_count"]) == 0 and protected_source_remains)
                or (int(group["read_span_count"]) > 0
                    and read_span_count - int(group["read_span_count"]) >= _MIN_READ_SPANS_TO_RETAIN)
            )
        ]
        if not candidates:
            break
        # Optional source excerpts are still expendable before read results;
        # among actual read spans, preserve the two most recent as a floor.
        oldest = min(
            candidates,
            key=lambda group: (
                int(group["read_span_count"]) > 0,
                int(group["recency"]),
            ),
        )
        removed_indices = set(oldest["indices"])
        for index in sorted(removed_indices):
            if index >= len(after["messages"]):
                continue
            section = section_name(after["messages"][index], index)
            if section in _READ_EVIDENCE_SECTIONS and section not in dropped_sections:
                dropped_sections.append(section)
            if section in _READ_EVIDENCE_SECTIONS:
                dropped_items.append(f"{section}:{index}")
        after["messages"] = [
            message for index, message in enumerate(after["messages"])
            if index not in removed_indices
        ]
        new_records = [record for record in oldest["records"]
                       if isinstance(record, Mapping)]
        dropped_source_ranges.extend(dict(record) for record in new_records)
        _upsert_dropped_source_notice(after["messages"], new_records)
        try:
            request_tokens_after = measure_after(after)
        except _MeasurementModeChanged:
            return restart_with_stable_method()

    if removal_accounting is not None:
        removal_accounting.update(after["messages"])
        verified = removal_accounting.verify(after)
        removal_accounting = None
        if verified:
            had_byte_rejection = byte_rejected_without_measurement
            current = after["messages"]
            unchanged = (not had_byte_rejection and last_measured_messages is not None
                         and len(last_measured_messages) == len(current)
                         and all(a is b for a, b in zip(last_measured_messages, current)))
            if unchanged:
                # DIET_INCR review item 2: the pass ended on a real
                # measurement of these same retained objects (bytes just
                # verified), so a re-measure is a redundant cache hit.
                exact_tokens = request_tokens_after
            else:
                try:
                    exact_tokens = measure_after(after)
                except _MeasurementModeChanged:
                    return restart_with_stable_method()
            verified = had_byte_rejection or exact_tokens == request_tokens_after
            request_tokens_after = exact_tokens
        if not verified:
            # Replay from the original body, not the possibly over-trimmed
            # candidate. Keep the same logical meter/fallback failure budget.
            trial = copy.deepcopy(before)
            result = apply_diet(
                trial, stage=stage, max_request_bytes=max_request_bytes,
                emit=emit, request_id=request_id, step=step,
                protected_contents=protected_contents,
                protected_messages=initial_protected_messages,
                required_source_paths=required_source_paths,
                required_source_order=required_source_order,
                deferred_source_paths=deferred_source_paths,
                token_budget=token_budget, token_meter=meter,
                allocation_plan=allocation_plan,
                _measurement_retry=_measurement_retry,
                _incremental_accounting=False,
            )
            body.clear()
            body.update(trial)
            return result
    after_ids = set(required_evidence_ids(after["messages"]))
    missing = sorted(set(evidence_ids) - after_ids)
    if missing:
        raise RequestDietDeferred({
            "event_schema": SCHEMA,
            "stage": stage,
            "action": "defer",
            "reason": "required_evidence_removed",
            "required_evidence_ids": list(evidence_ids),
            "missing_evidence_ids": missing,
        })
    read_spans_retained = _read_spans_retained(
        [message for message in after["messages"] if isinstance(message, Mapping)]
    )
    if read_spans_retained < min(_MIN_READ_SPANS_TO_RETAIN, read_spans_available):
        raise RequestDietDeferred({
            "event_schema": SCHEMA,
            "stage": stage,
            "action": "defer",
            "reason": "request_diet_read_span_floor_violation",
            **_diet_contract_fields(
                meter,
                measure_method=measure_method,
                request_tokens_before=request_tokens_before,
                request_tokens_after=request_tokens_after,
                max_request_bytes=max_request_bytes,
                token_budget=token_budget,
                read_spans_available=read_spans_available,
                read_spans_retained=read_spans_retained,
                protected_paths=protected_paths,
            ),
        })
    if over_limit(after, request_tokens_after):
        try:
            request_tokens_after, deferred_paths = _defer_required_source_paths(
                after,
                required_source_paths=required_source_paths,
                required_source_order=required_source_order,
                deferred_source_paths=deferred_paths,
                measure_after=measure_after,
                over_limit=over_limit,
                current_tokens=request_tokens_after,
            )
        except _MeasurementModeChanged:
            return restart_with_stable_method()

    if over_limit(after, request_tokens_after):
        protected_request_bytes, protected_sections_bytes = _protected_request_sizes(
            after,
            protected_contents=protected_contents,
            protected_messages=protected_messages,
            required_source_paths=required_source_paths,
        )
        after_sections = _section_stats(after)
        retained_read_messages = sum(
            after_sections[name]["messages"] for name in _READ_EVIDENCE_SECTIONS
        )
        request_bytes_after = _wire_bytes(after)
        if protected_request_bytes > max_request_bytes:
            infeasibility_reason = "protected_sections_exceed_max_request_bytes"
        elif retained_read_messages and request_bytes_after > max_request_bytes:
            infeasibility_reason = "minimum_read_working_set_exceeds_max_request_bytes"
        elif request_bytes_after > max_request_bytes:
            infeasibility_reason = "non_dietable_request_content_exceeds_max_request_bytes"
        elif retained_read_messages:
            infeasibility_reason = "minimum_read_working_set_exceeds_token_budget"
        else:
            infeasibility_reason = "non_dietable_request_content_exceeds_token_budget"
        detail = {
            "event_schema": SCHEMA,
            "stage": stage,
            "action": "defer",
            "reason": "request_diet_cap_infeasible",
            "infeasibility_reason": infeasibility_reason,
            **_diet_contract_fields(
                meter,
                measure_method=measure_method,
                request_tokens_before=request_tokens_before,
                request_tokens_after=request_tokens_after,
                max_request_bytes=max_request_bytes,
                token_budget=token_budget,
                read_spans_available=read_spans_available,
                read_spans_retained=read_spans_retained,
                protected_paths=protected_paths,
            ),
            "request_bytes_before": _wire_bytes(before),
            "request_bytes_after": request_bytes_after,
            "protected_request_bytes": protected_request_bytes,
            "protected_sections_bytes": protected_sections_bytes,
            "retained_read_messages": retained_read_messages,
            "dropped_ranges_notice_text": _dropped_source_notice_text(
                [message for message in after["messages"]
                 if isinstance(message, Mapping)]
            ),
            "dropped_source_ranges": [dict(record) for record in dropped_source_ranges],
            "required_evidence_ids": list(evidence_ids),
            "menu_actions": list(menu),
            "dropped_sections": list(dict.fromkeys(dropped_sections)),
            "sections": after_sections,
        }
        if emit is not None:
            emit(
                "request_diet_cap_infeasible",
                request_id=request_id,
                step=step,
                **{**detail, "action": "error"},
            )
            from kernel_continuation_journal import journal_safe_fields
            emit(
                "request_diet_applied", request_id=request_id, step=step,
                **journal_safe_fields(detail),
            )
        raise RequestDietDeferred(detail)

    body["messages"] = after["messages"]
    audit = _audit(
        before, body, stage,
        dropped_sections=dropped_sections,
        dropped_items=dropped_items,
        duplicate_items=duplicate_items,
        stale_indices=stale_indices,
        evidence_ids=evidence_ids,
        menu=menu,
        reason="trimmed" if before != body else "within_budget",
        dropped_source_ranges=dropped_source_ranges,
    )
    if any(record.get("protected") is True for record in dropped_source_ranges):
        raise RuntimeError("request diet attempted to drop a protected source range")
    audit.update(_diet_contract_fields(
        meter,
        measure_method=measure_method,
        request_tokens_before=request_tokens_before,
        request_tokens_after=request_tokens_after,
        max_request_bytes=max_request_bytes,
        token_budget=token_budget,
        read_spans_available=read_spans_available,
        read_spans_retained=read_spans_retained,
        protected_paths=protected_paths,
    ))
    audit.update({
        "request_id": request_id,
        "step": step,
        "deferred_source_paths": tuple(deferred_paths),
    })
    if emit is not None:
        from kernel_continuation_journal import journal_safe_fields  # emit boundary (reserved keys)
        emit("request_diet_applied", **journal_safe_fields(audit))
    return audit


__all__ = [
    "RequestDietDeferred", "RequestTokenMeter", "SCHEMA", "SECTION_NAMES",
    "STAGE_SECTION_ALLOWLIST", "TEMPLATE_OVERHEAD_TOKENS", "resolve_tokenize_url",
    "apply_diet", "breakdown", "menu_actions", "required_evidence_ids", "section_name",
]
