"""Deterministic, request-local limits for one provider context-overflow retry."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import re


# The retry keeps seven eighths of the first submitted envelope. This leaves
# room for tokenizer/template disagreement while preserving protected content.
RETRY_NUMERATOR = 7
RETRY_DENOMINATOR = 8
# A context rejection can be caused entirely by the completion reserve. Keep
# enough room for useful tool calls while removing the 24K default reservation.
MAX_RETRY_OUTPUT_TOKENS = 8_192
_MAX_REPORTED_TOKENS = 1_000_000_000
# Smaller provider limits cannot leave useful room for the protected request
# and completion. Ignore them for both the immediate retry and later requests.
MIN_USABLE_CONTEXT_TOKENS = 1_024
CAPACITY_400_PATTERNS = (
    ("maximum_context_length", re.compile(
        r"\bmaximum context length (?:is|of)\s+(?P<limit>\d+)\s+tokens\b", re.I)),
    ("context_window_exceeded", re.compile(
        r"\b(?:input|request|prompt) exceeds? the context window\b", re.I)),
    ("available_context_size", re.compile(
        r"\brequest exceeds the available context size\b", re.I)),
    ("prompt_token_maximum", re.compile(
        r"\bprompt is too long:\s*\d+\s+tokens\s*>\s*(?P<limit>\d+)\s+maximum\b", re.I)),
)
_LEARNED_EVENT = "capacity_context_limit_learned"


def reported_context_limit(server_error: str | None) -> int | None:
    """Extract a token limit from the message used by capacity_400_reason."""
    if not isinstance(server_error, str):
        return None
    message = server_error
    try:
        payload = json.loads(server_error)
    except ValueError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        value = payload["error"].get("message")
        message = value if isinstance(value, str) else ""
    for _reason, pattern in CAPACITY_400_PATTERNS:
        match = pattern.search(message)
        if match is None:
            continue
        if "limit" not in match.groupdict():
            continue
        try:
            limit = int(match.group("limit"))
        except ValueError:
            return None
        return limit if 0 < limit <= _MAX_REPORTED_TOKENS else None
    return None


@dataclass(frozen=True, slots=True)
class RetryLimits:
    max_request_bytes: int
    max_prompt_tokens: int
    provider_limit_tokens: int | None
    max_output_tokens: int | None = None


def _scope_digest(endpoint: str, model: str, key: str) -> str:
    """Bind a ceiling to the actual destination and private credential identity."""
    from kernel_request_auth_identity import auth_binding_for_key
    scope = json.dumps([endpoint, model, auth_binding_for_key(key)],
                       ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(scope.encode("utf-8")).hexdigest()


def learned_context_limit(journal: object, endpoint: str, model: str, key: str) -> int | None:
    """Fold durable provider rejections once per journal, including on resume."""
    limits = getattr(journal, "_capacity_context_limits", None)
    if limits is None:
        limits = {}
        records = getattr(journal, "records", None)
        if callable(records):
            history = records(rebuild=False)
        else:
            from kernel_measurement import iter_journal_records
            history = iter_journal_records(journal.path)
        for event in history:
            if event.get("kind") != _LEARNED_EVENT or event.get("producer") != "kernel_capacity_retry":
                continue
            scope, value = event.get("scope_sha256"), event.get("provider_limit_tokens")
            if (isinstance(scope, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", scope)
                    and type(value) is int
                    and MIN_USABLE_CONTEXT_TOKENS <= value <= _MAX_REPORTED_TOKENS):
                limits[scope] = min(limits.get(scope, value), value)
        setattr(journal, "_capacity_context_limits", limits)
    if not limits:
        return None
    return limits.get(_scope_digest(endpoint, model, key))


def learn_context_limit(journal: object, endpoint: str, model: str, key: str,
                        limit: int | None) -> None:
    """Persist a validated lower ceiling before using it for another request."""
    if type(limit) is not int or not 0 < limit <= _MAX_REPORTED_TOKENS:
        return
    if limit < MIN_USABLE_CONTEXT_TOKENS:
        journal.emit("capacity_context_limit_ignored", producer="kernel_capacity_retry",
                     fact_owner="HOST", reason="below_minimum_usable_context_tokens",
                     provider_limit_tokens=limit,
                     minimum_usable_context_tokens=MIN_USABLE_CONTEXT_TOKENS,
                     dispatch=False)
        return
    previous = learned_context_limit(journal, endpoint, model, key)
    if previous is not None and previous <= limit:
        return
    scope = _scope_digest(endpoint, model, key)
    journal.emit(_LEARNED_EVENT, producer="kernel_capacity_retry", fact_owner="HOST",
                 scope_sha256=scope, provider_limit_tokens=limit, dispatch=False)
    journal._capacity_context_limits[scope] = limit


def learned_request_limits(provider_limit: int, output_tokens: int) -> RetryLimits:
    """Pre-fit prompt and completion under the learned total context ceiling."""
    from kernel_request_budget import HARD_MAX_REQUEST_BYTES
    total = provider_limit * RETRY_NUMERATOR // RETRY_DENOMINATOR
    output = min(output_tokens, MAX_RETRY_OUTPUT_TOKENS, max(1, total // 4))
    if output < 1 or output >= total:
        raise ValueError("learned provider context limit has no prompt room")
    return RetryLimits(HARD_MAX_REQUEST_BYTES, total - output, provider_limit, output)


def apply_learned_limit(capture: object, body: dict[str, object],
                        endpoint: str, key: str) -> None:
    """Constrain a later routed request before its complete-envelope fit."""
    model = body.get("model")
    if not isinstance(model, str):
        return
    known = learned_context_limit(capture.journal, endpoint, model, key)
    output = body.get("max_tokens")
    if known is None or type(output) is not int or output < 1:
        return
    limits = learned_request_limits(known, output)
    body["max_tokens"] = limits.max_output_tokens
    capture._capacity_retry_limits = limits


def compact_tool_prose(body: dict[str, object]) -> None:
    """Keep every callable schema while removing prose before a bounded refit."""
    essential_descriptions = {
        "done": "Complete the task.",
        "edit": "Edit a file.",
        "str_replace": "Replace text in a file.",
        "run_tests": "Run tests.",
    }
    def strip_schema_prose(schema: object) -> None:
        if not isinstance(schema, dict):
            return
        schema.pop("description", None)
        for name in ("properties", "patternProperties", "$defs", "definitions",
                     "dependentSchemas"):
            children = schema.get(name)
            if isinstance(children, dict):
                for child in children.values():
                    strip_schema_prose(child)
        for name in ("items", "additionalItems", "additionalProperties",
                     "unevaluatedItems", "unevaluatedProperties", "contains", "not",
                     "if", "then", "else", "propertyNames", "contentSchema"):
            strip_schema_prose(schema.get(name))
        for name in ("allOf", "anyOf", "oneOf", "prefixItems"):
            children = schema.get(name)
            if isinstance(children, list):
                for child in children:
                    strip_schema_prose(child)

    tools = body.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        strip_schema_prose(function.get("parameters"))
        description = function.get("description")
        if isinstance(description, str):
            function["description"] = essential_descriptions.get(function.get("name"), "")


def retry_limits(request_bytes: int, prompt_tokens: int,
                 provider_limit_tokens: int | None,
                 output_tokens: int | None = None) -> RetryLimits:
    """Choose strict lower byte/token ceilings from the rejected request."""
    if request_bytes < 2 or prompt_tokens < 2:
        raise ValueError("capacity retry requires a shrinkable measured request")
    byte_limit = max(1, request_bytes * RETRY_NUMERATOR // RETRY_DENOMINATOR)
    token_limit = max(1, prompt_tokens * RETRY_NUMERATOR // RETRY_DENOMINATOR)
    output_limit = None
    if provider_limit_tokens is not None and provider_limit_tokens < 1:
        raise ValueError("provider context limit must be positive")
    if (provider_limit_tokens is not None
            and provider_limit_tokens < MIN_USABLE_CONTEXT_TOKENS):
        provider_limit_tokens = None
    if provider_limit_tokens is not None:
        total_limit = provider_limit_tokens * RETRY_NUMERATOR // RETRY_DENOMINATOR
        if type(output_tokens) is int and output_tokens > 0:
            output_limit = min(output_tokens, MAX_RETRY_OUTPUT_TOKENS,
                               max(1, total_limit // 4))
            if output_limit >= total_limit:
                raise ValueError("provider context limit has no prompt room")
            token_limit = min(token_limit, total_limit - output_limit)
        else:
            token_limit = min(token_limit, max(1, total_limit))
    return RetryLimits(min(byte_limit, request_bytes - 1),
                       min(token_limit, prompt_tokens - 1), provider_limit_tokens,
                       output_limit)


def reseal_capacity_retry(capture: object, original_body: dict[str, object],
                          endpoint: str, key: str, deadline: float | None,
                          limits: RetryLimits) -> tuple[dict[str, object], bytes, RetryLimits]:
    """Fit a fresh detached candidate and publish it only after full qualification."""
    from kernel_final_wire_allocation import build_qualified_request
    from kernel_frame_allocation_capture import FrameFit5ImpossibleMinimum
    from kernel_request_seal import RequestSealError

    previous = getattr(capture, "qualified_request", None)
    runtime = getattr(capture, "completion_owner_runtime", None)
    if previous is None or runtime is None:
        raise RequestSealError("capacity retry requires a sealed primary request")
    def build(body: dict[str, object], ceiling: RetryLimits, *, compact: bool = False):
        replacement = copy.copy(capture)
        replacement.qualified_request = None
        replacement.frame_request_allocation = None
        replacement._frame_request_allocation_candidate = None
        replacement._frame_candidate_counter = None
        replacement._frame_candidate_inputs = None
        replacement._frame_candidate_measurement = None
        replacement._request_measurement = None
        replacement._capacity_retry_limits = ceiling
        replacement._capacity_retry_compact_tool_prose = compact
        active = getattr(runtime, "_active_capture", None)
        runtime._active_capture = replacement
        try:
            record = build_qualified_request(
                replacement, runtime, body,
                endpoint=endpoint, key=key, deadline=deadline,
            )
        finally:
            runtime._active_capture = active
        return replacement, record

    candidate_body = copy.deepcopy(original_body)
    if limits.max_output_tokens is not None:
        candidate_body["max_tokens"] = limits.max_output_tokens
    # Capacity recovery needs the schema-preserving compact surface in the
    # protected-minimum case. Author it before the first full qualification so
    # a rejected prose-heavy candidate does not pay for a second FrameFit5 fit.
    compact_tool_prose(candidate_body)
    try:
        replacement, record = build(candidate_body, limits, compact=True)
    except FrameFit5ImpossibleMinimum:
        # The compact protected minimum can still exceed the provider ceiling.
        # Lower the completion reserve only when it changes the candidate,
        # then pass the result through the same fitter and seal.
        previous_output = previous.measurement.output_tokens
        if previous_output < 2:
            raise
        output_limit = min(previous_output * RETRY_NUMERATOR // RETRY_DENOMINATOR,
                           MAX_RETRY_OUTPUT_TOKENS,
                           limits.max_output_tokens or MAX_RETRY_OUTPUT_TOKENS)
        if limits.provider_limit_tokens is not None:
            output_limit = min(output_limit, max(1, limits.provider_limit_tokens // 4))
        if len(str(output_limit)) >= len(str(previous_output)):
            output_limit = min(output_limit, 10 ** (len(str(previous_output)) - 1) - 1)
        if output_limit < 1:
            raise
        fallback_body = copy.deepcopy(original_body)
        fallback_body["max_tokens"] = output_limit
        compact_tool_prose(fallback_body)
        total_limit = (limits.provider_limit_tokens * RETRY_NUMERATOR // RETRY_DENOMINATOR
                       if limits.provider_limit_tokens is not None else None)
        fallback_prompt_limit = limits.max_prompt_tokens
        if total_limit is not None:
            fallback_prompt_limit = min(fallback_prompt_limit, total_limit - output_limit)
        fallback_limits = RetryLimits(
            limits.max_request_bytes, fallback_prompt_limit,
            limits.provider_limit_tokens, output_limit,
        )
        if fallback_body == candidate_body and fallback_limits == limits:
            raise
        replacement, record = build(fallback_body, fallback_limits, compact=True)
        candidate_body = fallback_body
        limits = fallback_limits
    if (len(record.wire) >= len(previous.wire)
            or len(record.wire) > limits.max_request_bytes
            or record.measurement.prompt_tokens is None
            or record.measurement.prompt_tokens > limits.max_prompt_tokens
            or (limits.provider_limit_tokens is not None
                and record.measurement.prompt_tokens + record.measurement.output_tokens
                > limits.provider_limit_tokens * RETRY_NUMERATOR // RETRY_DENOMINATOR)):
        raise RequestSealError("capacity retry did not produce a smaller qualified request")
    capture.__dict__.update(replacement.__dict__)
    capture._capacity_retry_input_body = copy.deepcopy(candidate_body)
    return record.body, record.wire, limits
