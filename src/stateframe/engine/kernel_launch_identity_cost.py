"""Pure launch-identity, provider-boundary, and host-cost contracts.

This module intentionally has no provider, router, parser, or model dependency.
The final wire owner supplies the effective model/configuration and the exact
serialized body.  A router classification can request a transformation, but it
cannot certify that a request was reasoning-off.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


SCHEMA = "stateframe-launch-identity-cost/1"
CERTIFICATE_SCHEMA = "stateframe-run-certificate/1"
OPENAI_REASONING_EFFORTS = frozenset({
    "none", "minimal", "low", "medium", "high", "xhigh", "max",
})
REASONING_PROFILE_ENV = "STATEFRAME_REASONING_PROFILE"
REASONING_MODE_ENV = "STATEFRAME_REASONING_MODE"
UNKNOWN = "UNKNOWN"
ENVIRONMENT = "ENVIRONMENT"
PROVIDER_MALFORMED = "PROVIDER_MALFORMED"
SEMANTIC_ERROR = "SEMANTIC_ERROR"
HARNESS_CONTRACT = "HARNESS_CONTRACT"
PHASES = ("extraction", "binding_indexing", "journaling", "request_delivery")
MAX_COUNTER = 10_000_000
MAX_ELAPSED_NS = 86_400_000_000_000

# OpenAI Chat Completions create schema (official generated SDK source):
# https://github.com/openai/openai-python/blob/main/src/openai/types/chat/completion_create_params.py
# Keep this reviewed request-field set local so endpoint extensions require proof.
OPENAI_CHAT_BODY_KEYS = frozenset({
    "model", "messages", "tools", "tool_choice", "parallel_tool_calls",
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "n",
    "stop", "stream", "stream_options", "seed", "logprobs", "top_logprobs",
    "response_format", "frequency_penalty", "presence_penalty", "logit_bias",
    "user", "reasoning_effort", "service_tier", "store", "metadata",
    "modalities", "audio", "moderation", "prediction", "functions", "function_call",
    "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "safety_identifier",
    "verbosity", "web_search_options",
})
_MESSAGE_KEYS = frozenset({"role", "content", "name", "tool_call_id", "tool_calls",
                           "function_call", "refusal", "audio"})
_CONTENT_PART_KEYS = frozenset({"type", "text", "image_url", "input_audio",
                                "file", "refusal", "prompt_cache_breakpoint"})
_TOOL_KEYS = frozenset({"type", "function"})
_FUNCTION_KEYS = frozenset({"name", "description", "parameters", "strict"})
_LEGACY_FUNCTION_KEYS = frozenset({"name", "description", "parameters"})
_TOOL_CALL_KEYS = frozenset({"id", "type", "function"})
_TOOL_CALL_FUNCTION_KEYS = frozenset({"name", "arguments"})
_MESSAGE_FUNCTION_CALL_KEYS = frozenset({"name", "arguments"})
_IMAGE_URL_KEYS = frozenset({"url", "detail"})
_INPUT_AUDIO_KEYS = frozenset({"data", "format"})
_FILE_KEYS = frozenset({"file_data", "file_id", "filename"})
_AUDIO_MESSAGE_KEYS = frozenset({"id"})
_AUDIO_REQUEST_KEYS = frozenset({"voice", "format"})
_TOOL_CHOICE_KEYS = frozenset({"type", "function"})
_TOOL_CHOICE_FUNCTION_KEYS = frozenset({"name"})
_RESPONSE_FORMAT_KEYS = frozenset({"type", "json_schema"})
_JSON_SCHEMA_FORMAT_KEYS = frozenset({"name", "description", "strict", "schema"})
_STREAM_OPTIONS_KEYS = frozenset({"include_usage"})
_PREDICTION_KEYS = frozenset({"type", "content"})
_WEB_SEARCH_KEYS = frozenset({"search_context_size", "user_location"})
_USER_LOCATION_KEYS = frozenset({"type", "approximate"})
_APPROXIMATE_LOCATION_KEYS = frozenset({"city", "country", "region", "timezone"})
_CACHE_BREAKPOINT_KEYS = frozenset({"mode", "ttl"})
_PROMPT_CACHE_OPTIONS_KEYS = frozenset({"mode", "ttl"})
_MODERATION_KEYS = frozenset({"model", "policy"})
_MODERATION_POLICY_KEYS = frozenset({"input", "output"})
_MODERATION_MODE_KEYS = frozenset({"mode"})
_JSON_SCHEMA_KEYS = frozenset({
    # JSON Schema keywords used inside OpenAI function parameters and json_schema.
    "$schema", "$id", "$ref", "$anchor", "$dynamicRef", "$dynamicAnchor",
    "$defs", "$comment", "$vocabulary", "definitions", "type", "properties",
    "patternProperties", "additionalProperties", "unevaluatedProperties",
    "propertyNames", "required", "dependentRequired", "dependentSchemas",
    "items", "prefixItems", "additionalItems", "contains", "minContains",
    "maxContains", "unevaluatedItems", "enum", "const", "anyOf", "oneOf",
    "allOf", "not", "if", "then", "else", "title", "description",
    "default", "deprecated", "readOnly", "writeOnly", "examples", "format",
    "contentEncoding", "contentMediaType", "contentSchema", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength",
    "pattern", "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties",
})


def _openai_schema(value: object) -> object:
    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    result: dict[str, object] = {}
    for key, item in value.items():
        if key not in _JSON_SCHEMA_KEYS:
            continue
        if key in {"properties", "$defs", "definitions", "patternProperties",
                   "dependentSchemas"} and isinstance(item, Mapping):
            result[key] = {name: _openai_schema(schema) for name, schema in item.items()}
        elif key in {"items", "additionalItems", "additionalProperties", "not", "if",
                     "then", "else", "contains", "propertyNames", "unevaluatedItems",
                     "unevaluatedProperties", "contentSchema"} and isinstance(item, Mapping):
            result[key] = _openai_schema(item)
        elif key in {"items", "prefixItems"} and isinstance(item, list):
            result[key] = [_openai_schema(schema) for schema in item]
        elif key in {"anyOf", "oneOf", "allOf"} and isinstance(item, list):
            result[key] = [_openai_schema(schema) for schema in item]
        else:
            result[key] = copy.deepcopy(item)
    return result


def _openai_nested(value: object, allowed: frozenset[str]) -> object:
    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    return {key: copy.deepcopy(item) for key, item in value.items() if key in allowed}


def _openai_function(value: object) -> object:
    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    return {
        key: (_openai_schema(item) if key == "parameters" else copy.deepcopy(item))
        for key, item in value.items() if key in _FUNCTION_KEYS
    }


def _openai_content_part(value: object) -> object:
    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    part = {key: copy.deepcopy(item) for key, item in value.items()
            if key in _CONTENT_PART_KEYS}
    for field, allowed in (("image_url", _IMAGE_URL_KEYS),
                           ("input_audio", _INPUT_AUDIO_KEYS), ("file", _FILE_KEYS),
                           ("prompt_cache_breakpoint", _CACHE_BREAKPOINT_KEYS)):
        nested = part.get(field)
        if isinstance(nested, Mapping):
            part[field] = {key: copy.deepcopy(item) for key, item in nested.items()
                           if key in allowed}
    return part


def _openai_message(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    message = {key: copy.deepcopy(item) for key, item in value.items()
               if key in _MESSAGE_KEYS}
    content = message.get("content")
    if isinstance(content, list):
        message["content"] = [_openai_content_part(part) for part in content]
    audio = message.get("audio")
    if isinstance(audio, Mapping):
        message["audio"] = {key: copy.deepcopy(item) for key, item in audio.items()
                             if key in _AUDIO_MESSAGE_KEYS}
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        message["tool_calls"] = [
            {key: ({} if not isinstance(item, Mapping) else
                   {nested: copy.deepcopy(v) for nested, v in item.items()
                     if nested in _TOOL_CALL_FUNCTION_KEYS})
              if key == "function" else copy.deepcopy(item)
              for key, item in call.items() if key in _TOOL_CALL_KEYS}
            if isinstance(call, Mapping) else call for call in calls
        ]
    function_call = message.get("function_call")
    if isinstance(function_call, Mapping):
        message["function_call"] = {
            key: copy.deepcopy(item) for key, item in function_call.items()
            if key in _MESSAGE_FUNCTION_CALL_KEYS
        }
    return message


def _openai_body(
    body: Mapping[str, object],
    selected: "ReasoningProfile",
    *,
    environ: Mapping[str, str] | None = None,
    side_call: bool = False,
) -> dict[str, object]:
    source = os.environ if environ is None else environ
    raw_effort = source.get("REACT_REASONING_EFFORT")
    explicit_effort = raw_effort.strip().lower() if isinstance(raw_effort, str) else None
    if not explicit_effort:
        explicit_effort = None
    if side_call:
        # Registered auxiliary calls require no thinking independently of the
        # solver profile. Carry that policy into the strict provider dialect.
        explicit_effort = "none"
    if selected.off and explicit_effort != "none":
        raise ProviderBoundaryContractError(
            "reasoning_off_unproven",
            "strict endpoint needs operator-confirmed REACT_REASONING_EFFORT=none for global reasoning-off")
    result = {key: copy.deepcopy(value) for key, value in body.items()
              if key in OPENAI_CHAT_BODY_KEYS}
    if explicit_effort is None:
        result.pop("reasoning_effort", None)
    elif explicit_effort not in OPENAI_REASONING_EFFORTS:
        raise ProviderBoundaryContractError(
            "invalid_reasoning_effort",
            "REACT_REASONING_EFFORT must be one of none|minimal|low|medium|high|xhigh|max")
    else:
        # The selected policy explicitly admits this OpenAI schema field.
        result["reasoning_effort"] = explicit_effort
    if "n_probs" in body:
        count = body["n_probs"]
        if count is not None and (type(count) is not int or count < 0 or count > 20):
            raise ProviderBoundaryContractError(
                "invalid_n_probs", "OpenAI logprobs mapping requires n_probs=0 or an integer from 1 to 20")
        if type(count) is int and 0 < count <= 20:
            result["logprobs"] = True
            result.setdefault("top_logprobs", count)
    messages = result.get("messages")
    if isinstance(messages, list):
        result["messages"] = [_openai_message(message) for message in messages]
    tools = result.get("tools")
    if isinstance(tools, list):
        result["tools"] = [
            {key: (_openai_function(item) if key == "function"
                   else copy.deepcopy(item))
             for key, item in tool.items() if key in _TOOL_KEYS}
            if isinstance(tool, Mapping) else copy.deepcopy(tool) for tool in tools
        ]
    functions = result.get("functions")
    if isinstance(functions, list):
        result["functions"] = [
            ({key: (_openai_schema(value) if key == "parameters" else copy.deepcopy(value))
              for key, value in item.items() if key in _LEGACY_FUNCTION_KEYS}
             if isinstance(item, Mapping) else copy.deepcopy(item))
            for item in functions
        ]
    function_call = result.get("function_call")
    if isinstance(function_call, Mapping):
        result["function_call"] = {
            key: copy.deepcopy(value) for key, value in function_call.items()
            if key == "name"
        }
    if isinstance(result.get("stream_options"), Mapping):
        result["stream_options"] = _openai_nested(result["stream_options"], _STREAM_OPTIONS_KEYS)
    if isinstance(result.get("audio"), Mapping):
        result["audio"] = _openai_nested(result["audio"], _AUDIO_REQUEST_KEYS)
    if isinstance(result.get("prompt_cache_options"), Mapping):
        result["prompt_cache_options"] = _openai_nested(
            result["prompt_cache_options"], _PROMPT_CACHE_OPTIONS_KEYS,
        )
    moderation = result.get("moderation")
    if isinstance(moderation, Mapping):
        moderation = _openai_nested(moderation, _MODERATION_KEYS)
        policy = moderation.get("policy")
        if isinstance(policy, Mapping):
            policy = _openai_nested(policy, _MODERATION_POLICY_KEYS)
            for channel in ("input", "output"):
                if isinstance(policy.get(channel), Mapping):
                    policy[channel] = _openai_nested(policy[channel], _MODERATION_MODE_KEYS)
            moderation["policy"] = policy
        result["moderation"] = moderation
    if isinstance(result.get("prediction"), Mapping):
        result["prediction"] = _openai_nested(result["prediction"], _PREDICTION_KEYS)
    web_search = result.get("web_search_options")
    if isinstance(web_search, Mapping):
        web_search = _openai_nested(web_search, _WEB_SEARCH_KEYS)
        location = web_search.get("user_location")
        if isinstance(location, Mapping):
            location = _openai_nested(location, _USER_LOCATION_KEYS)
            approximate = location.get("approximate")
            if isinstance(approximate, Mapping):
                location["approximate"] = _openai_nested(approximate, _APPROXIMATE_LOCATION_KEYS)
            web_search["user_location"] = location
        result["web_search_options"] = web_search
    tool_choice = result.get("tool_choice")
    if isinstance(tool_choice, Mapping):
        result["tool_choice"] = {
            key: (_openai_nested(item, _TOOL_CHOICE_FUNCTION_KEYS)
                  if key == "function" else copy.deepcopy(item))
            for key, item in tool_choice.items() if key in _TOOL_CHOICE_KEYS
        }
    response_format = result.get("response_format")
    if isinstance(response_format, Mapping):
        formatted = {key: copy.deepcopy(item) for key, item in response_format.items()
                     if key in _RESPONSE_FORMAT_KEYS}
        schema_format = formatted.get("json_schema")
        if isinstance(schema_format, Mapping):
            formatted["json_schema"] = {
                key: (_openai_schema(item) if key == "schema" else copy.deepcopy(item))
                for key, item in schema_format.items() if key in _JSON_SCHEMA_FORMAT_KEYS
            }
        result["response_format"] = formatted
    return result


def assert_openai_chat_body(body: Mapping[str, object]) -> None:
    """Refuse any top-level or message/tool extension at the strict wire seam."""
    if not isinstance(body, Mapping):
        raise ProviderBoundaryContractError(
            "request_not_mapping", "provider request must be a mapping")
    unexpected = set(body) - OPENAI_CHAT_BODY_KEYS
    if unexpected:
        raise ProviderBoundaryContractError(
            "openai_body_schema_violation",
            "unknown OpenAI Chat Completions request keys: " + ", ".join(sorted(map(str, unexpected))))

    def check_keys(value: Mapping[str, object], allowed: frozenset[str], path: str) -> None:
        unknown = set(value) - allowed
        if unknown:
            raise ProviderBoundaryContractError(
                "openai_body_schema_violation",
                f"unknown OpenAI keys at {path}: " + ", ".join(sorted(map(str, unknown))))

    def check_schema(schema: object, path: str) -> None:
        if isinstance(schema, bool):
            return
        if not isinstance(schema, Mapping):
            raise ProviderBoundaryContractError(
                "openai_body_schema_violation", f"{path} must be a JSON schema object")
        check_keys(schema, _JSON_SCHEMA_KEYS, path)
        for field in ("properties", "$defs", "definitions", "patternProperties",
                      "dependentSchemas"):
            nested = schema.get(field)
            if isinstance(nested, Mapping):
                for name, child in nested.items():
                    check_schema(child, f"{path}.{field}.{name}")
        for field in ("items", "additionalItems", "additionalProperties", "not", "if",
                      "then", "else", "contains", "propertyNames", "unevaluatedItems",
                      "unevaluatedProperties", "contentSchema"):
            nested = schema.get(field)
            if isinstance(nested, Mapping):
                check_schema(nested, f"{path}.{field}")
        prefix_items = schema.get("prefixItems")
        if isinstance(prefix_items, list):
            for index, child in enumerate(prefix_items):
                check_schema(child, f"{path}.prefixItems[{index}]")
        for field in ("anyOf", "oneOf", "allOf"):
            nested = schema.get(field)
            if isinstance(nested, list):
                for index, child in enumerate(nested):
                    check_schema(child, f"{path}.{field}[{index}]")

    def check_message(message: object, path: str) -> None:
        if not isinstance(message, Mapping):
            raise ProviderBoundaryContractError(
                "openai_body_schema_violation", f"{path} must be an object")
        check_keys(message, _MESSAGE_KEYS, path)
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise ProviderBoundaryContractError(
                "openai_body_schema_violation", f"{path}.content must be a string or content-part array")
        if isinstance(content, list):
            for index, part in enumerate(content):
                if not isinstance(part, Mapping):
                    raise ProviderBoundaryContractError(
                        "openai_body_schema_violation", f"{path}.content[{index}] must be an object")
                part_path = f"{path}.content[{index}]"
                check_keys(part, _CONTENT_PART_KEYS, part_path)
                for field, allowed in (("image_url", _IMAGE_URL_KEYS),
                                       ("input_audio", _INPUT_AUDIO_KEYS),
                                       ("file", _FILE_KEYS),
                                       ("prompt_cache_breakpoint", _CACHE_BREAKPOINT_KEYS)):
                    nested = part.get(field)
                    if isinstance(nested, Mapping):
                        check_keys(nested, allowed, f"{part_path}.{field}")
        audio = message.get("audio")
        if isinstance(audio, Mapping):
            check_keys(audio, _AUDIO_MESSAGE_KEYS, f"{path}.audio")
        function_call = message.get("function_call")
        if isinstance(function_call, Mapping):
            check_keys(function_call, _MESSAGE_FUNCTION_CALL_KEYS, f"{path}.function_call")
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for index, call in enumerate(calls):
                if not isinstance(call, Mapping):
                    raise ProviderBoundaryContractError(
                        "openai_body_schema_violation", f"{path}.tool_calls[{index}] must be an object")
                check_keys(call, _TOOL_CALL_KEYS, f"{path}.tool_calls[{index}]")
                function = call.get("function")
                if not isinstance(function, Mapping):
                    raise ProviderBoundaryContractError(
                        "openai_body_schema_violation",
                        f"{path}.tool_calls[{index}].function must be an object")
                check_keys(function, _TOOL_CALL_FUNCTION_KEYS,
                           f"{path}.tool_calls[{index}].function")

    messages = body.get("messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            check_message(message, f"messages[{index}]")
    tools = body.get("tools")
    if isinstance(tools, list):
        for index, tool in enumerate(tools):
            if not isinstance(tool, Mapping):
                raise ProviderBoundaryContractError(
                    "openai_body_schema_violation", f"tools[{index}] must be an object")
            check_keys(tool, _TOOL_KEYS, f"tools[{index}]")
            function = tool.get("function")
            if not isinstance(function, Mapping):
                raise ProviderBoundaryContractError(
                    "openai_body_schema_violation", f"tools[{index}].function must be an object")
            check_keys(function, _FUNCTION_KEYS, f"tools[{index}].function")
            if "parameters" in function:
                check_schema(function["parameters"], f"tools[{index}].function.parameters")
    functions = body.get("functions")
    if isinstance(functions, list):
        for index, function in enumerate(functions):
            if not isinstance(function, Mapping):
                raise ProviderBoundaryContractError(
                    "openai_body_schema_violation", f"functions[{index}] must be an object")
            check_keys(function, _LEGACY_FUNCTION_KEYS, f"functions[{index}]")
            if "parameters" in function:
                check_schema(function["parameters"], f"functions[{index}].parameters")
    choice = body.get("tool_choice")
    if isinstance(choice, Mapping):
        check_keys(choice, _TOOL_CHOICE_KEYS, "tool_choice")
        function = choice.get("function")
        if isinstance(function, Mapping):
            check_keys(function, _TOOL_CHOICE_FUNCTION_KEYS, "tool_choice.function")
    function_call = body.get("function_call")
    if isinstance(function_call, Mapping):
        check_keys(function_call, frozenset({"name"}), "function_call")
    response_format = body.get("response_format")
    if isinstance(response_format, Mapping):
        check_keys(response_format, _RESPONSE_FORMAT_KEYS, "response_format")
        schema_format = response_format.get("json_schema")
        if isinstance(schema_format, Mapping):
            check_keys(schema_format, _JSON_SCHEMA_FORMAT_KEYS, "response_format.json_schema")
            schema = schema_format.get("schema")
            if schema is not None:
                check_schema(schema, "response_format.json_schema.schema")
    for name, allowed in (("stream_options", _STREAM_OPTIONS_KEYS),
                          ("audio", _AUDIO_REQUEST_KEYS),
                          ("prompt_cache_options", _PROMPT_CACHE_OPTIONS_KEYS),
                          ("prediction", _PREDICTION_KEYS),
                          ("web_search_options", _WEB_SEARCH_KEYS)):
        value = body.get(name)
        if isinstance(value, Mapping):
            check_keys(value, allowed, name)
    moderation = body.get("moderation")
    if isinstance(moderation, Mapping):
        check_keys(moderation, _MODERATION_KEYS, "moderation")
        policy = moderation.get("policy")
        if isinstance(policy, Mapping):
            check_keys(policy, _MODERATION_POLICY_KEYS, "moderation.policy")
            for channel in ("input", "output"):
                value = policy.get(channel)
                if isinstance(value, Mapping):
                    check_keys(value, _MODERATION_MODE_KEYS, f"moderation.policy.{channel}")
    search = body.get("web_search_options")
    if isinstance(search, Mapping) and isinstance(search.get("user_location"), Mapping):
        location = search["user_location"]
        check_keys(location, _USER_LOCATION_KEYS, "web_search_options.user_location")
        if isinstance(location.get("approximate"), Mapping):
            check_keys(location["approximate"], _APPROXIMATE_LOCATION_KEYS,
                       "web_search_options.user_location.approximate")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def wire_digest(wire: bytes) -> str:
    """Digest exact serialized provider bytes, not a reconstructed object."""
    if not isinstance(wire, bytes):
        raise TypeError("wire must be bytes")
    return "sha256:" + hashlib.sha256(wire).hexdigest()


class LaunchIdentityError(ValueError):
    """Base error for invalid launch identity or provider-boundary data."""


class ProviderBoundaryContractError(LaunchIdentityError):
    """A host-owned request contract failed before provider dispatch."""

    outcome = HARNESS_CONTRACT
    failure_origin = "harness"
    transport_retryable = False
    harness_contract = True

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code).strip() or "provider_boundary_contract"
        self.detail = " ".join(str(detail).split())[:400]
        super().__init__(f"{self.code}: {self.detail}")


class ProviderMalformedResponseError(RuntimeError):
    """The provider returned a response that cannot enter semantic parsing."""

    outcome = PROVIDER_MALFORMED
    failure_origin = "provider"
    transport_retryable = False

    def __init__(self, detail: str) -> None:
        self.detail = " ".join(str(detail).split())[:400]
        super().__init__(self.detail)


@dataclass(frozen=True)
class ReasoningProfile:
    """Host-declared request policy; every default is an explicit mode."""

    mode: str = "on"
    source: str = "default"

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"off", "on", "passthrough"}:
            raise ValueError("reasoning profile must be off, on or passthrough")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "source", str(self.source).strip() or "unknown")

    @property
    def off(self) -> bool:
        return self.mode == "off"

    def to_dict(self) -> dict[str, str]:
        return {"mode": self.mode, "source": self.source}


def _profile_value(value: object) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("mode", value.get("reasoning_mode"))
    if value is None:
        return None
    result = str(value).strip().lower()
    return result or None


def reasoning_profile(
    *,
    manifest: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> ReasoningProfile:
    """Resolve an explicit host profile without inspecting labels or responses."""
    if manifest is not None:
        for key in ("reasoning_profile", "reasoning_mode"):
            value = _profile_value(manifest.get(key))
            if value is not None:
                return ReasoningProfile(value, "manifest:" + key)
    source = os.environ if environ is None else environ
    profile = _profile_value(source.get(REASONING_PROFILE_ENV))
    mode = _profile_value(source.get(REASONING_MODE_ENV))
    if profile is not None and mode is not None and profile != mode:
        raise ValueError(
            f"{REASONING_PROFILE_ENV} and {REASONING_MODE_ENV} disagree"
        )
    if profile is not None:
        return ReasoningProfile(profile, "environment:" + REASONING_PROFILE_ENV)
    if mode is not None:
        return ReasoningProfile(mode, "environment:" + REASONING_MODE_ENV)
    return ReasoningProfile()


def apply_global_reasoning_off(body: Mapping[str, object]) -> dict[str, object]:
    """Return an explicit no-thinking copy for the final provider boundary."""
    if not isinstance(body, Mapping):
        raise ProviderBoundaryContractError(
            "request_not_mapping", "provider request must be a mapping"
        )
    result = copy.deepcopy(dict(body))
    raw_kwargs = result.get("chat_template_kwargs")
    if raw_kwargs is None:
        kwargs: dict[str, object] = {}
    elif isinstance(raw_kwargs, Mapping):
        kwargs = copy.deepcopy(dict(raw_kwargs))
    else:
        raise ProviderBoundaryContractError(
            "malformed_template_kwargs",
            "chat_template_kwargs must be a mapping",
        )
    kwargs["enable_thinking"] = False
    result["chat_template_kwargs"] = kwargs
    result["reasoning_effort"] = "none"
    result["reasoning_budget_tokens"] = 0
    result["reasoning_mode"] = "off"
    return result


def assert_global_reasoning_off(body: Mapping[str, object]) -> None:
    """Prove all required off fields are present in the exact final body."""
    if not isinstance(body, Mapping):
        raise ProviderBoundaryContractError(
            "request_not_mapping", "provider request must be a mapping"
        )
    kwargs = body.get("chat_template_kwargs")
    if not isinstance(kwargs, Mapping) or kwargs.get("enable_thinking") is not False:
        raise ProviderBoundaryContractError(
            "reasoning_template_not_off",
            "final provider body must set chat_template_kwargs.enable_thinking=false",
        )
    if body.get("reasoning_effort") != "none":
        raise ProviderBoundaryContractError(
            "reasoning_effort_not_off",
            "final provider body must set reasoning_effort=none",
        )
    if body.get("reasoning_budget_tokens") != 0:
        raise ProviderBoundaryContractError(
            "reasoning_budget_not_off",
            "final provider body must set reasoning_budget_tokens=0",
        )
    if body.get("reasoning_mode") != "off":
        raise ProviderBoundaryContractError(
            "reasoning_mode_not_off",
            "final provider body must set reasoning_mode=off",
        )


def enforce_provider_body(
    body: Mapping[str, object],
    *,
    profile: ReasoningProfile | None = None,
    dialect: str = "llama.cpp",
    environ: Mapping[str, str] | None = None,
    side_call: bool = False,
) -> dict[str, object]:
    """Apply and prove the explicit profile at the last mutable boundary."""
    selected = profile or reasoning_profile(environ=environ)
    if dialect == "openai-strict":
        if not isinstance(body, Mapping):
            raise ProviderBoundaryContractError(
                "request_not_mapping", "provider request must be a mapping")
        result = _openai_body(body, selected, environ=environ, side_call=side_call)
        assert_openai_chat_body(result)
        return result
    if dialect != "llama.cpp":
        raise ValueError("unknown provider dialect")
    if selected.off:
        result = apply_global_reasoning_off(body)
        assert_global_reasoning_off(result)
        return result
    if selected.mode == "passthrough":
        raise ProviderBoundaryContractError(
            "reasoning_profile_unset",
            "final provider body cannot inherit an unspecified reasoning policy",
        )
    if not isinstance(body, Mapping):
        raise ProviderBoundaryContractError(
            "request_not_mapping", "provider request must be a mapping"
        )
    return copy.deepcopy(dict(body))


_UNKNOWN_IDENTITY_VALUES = frozenset({"", "MISSING", "UNKNOWN", "ENVIRONMENT", "PASSTHROUGH"})


def _identity_value(value: object) -> object:
    """Normalize sentinel identity values so incomplete evidence stays incomplete."""
    if value is None:
        return UNKNOWN
    if isinstance(value, str) and value.strip().upper() in _UNKNOWN_IDENTITY_VALUES:
        return UNKNOWN
    return value


def deployed_source_digest(manifest: Mapping[str, object] | None) -> str:
    """Return a manifest-bound source identity, never a shell/config label."""
    if not isinstance(manifest, Mapping):
        return UNKNOWN
    source_digests = manifest.get("source_digests")
    if isinstance(source_digests, Mapping) and source_digests:
        if any(_identity_value(value) == UNKNOWN for value in source_digests.values()):
            return UNKNOWN
        explicit = manifest.get("deployed_source_digest")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        return _digest({str(key): str(value) for key, value in source_digests.items()})
    explicit = manifest.get("deployed_source_digest")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return UNKNOWN


@dataclass(frozen=True)
class RunCertificate:
    """One receipt identity joining effective config and exact request bytes."""

    run_id: str
    effective_model: str
    deployed_source_digest: str
    idealpath_mode: str
    reasoning_mode: str
    request_body_digest: str
    served_config: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        fields = {
            "schema": CERTIFICATE_SCHEMA,
            "run_id": self.run_id,
            "effective_model": self.effective_model,
            "deployed_source_digest": self.deployed_source_digest,
            "idealpath_mode": self.idealpath_mode,
            "reasoning_mode": self.reasoning_mode,
            "request_body_digest": self.request_body_digest,
            "served_config": copy.deepcopy(dict(self.served_config)),
        }
        complete = all(
            _identity_value(fields[name]) != UNKNOWN
            for name in (
                "run_id",
                "effective_model",
                "deployed_source_digest",
                "idealpath_mode",
                "reasoning_mode",
                "request_body_digest",
            )
        )
        fields["identity_status"] = "COMPLETE" if complete else UNKNOWN
        return fields


def build_run_certificate(
    *,
    run_id: str,
    body: Mapping[str, object],
    wire: bytes,
    manifest: Mapping[str, object] | None,
    backend_identity: Mapping[str, object] | None,
    profile: ReasoningProfile | None = None,
) -> RunCertificate:
    """Join only host-observed identity with the exact final serialized body."""
    selected = profile or reasoning_profile(manifest=manifest)
    manifest_value = manifest if isinstance(manifest, Mapping) else {}
    backend = backend_identity if isinstance(backend_identity, Mapping) else {}
    configured = backend.get("configured_identity")
    configured = configured if isinstance(configured, Mapping) else {}
    served_config: dict[str, object] = {}
    for key in ("model", "server_build", "tokenizer", "chat_template", "cache_policy"):
        value = backend.get(key, configured.get(key, manifest_value.get(key, UNKNOWN)))
        served_config[key] = _identity_value(value)
    explicit_model = backend.get("effective_model")
    if explicit_model not in (None, ""):
        model = explicit_model
    elif backend.get("router_applied") is True:
        model = backend.get("model", body.get("model", UNKNOWN))
    else:
        # With no independently observed routed backend, the model in the
        # exact final body is the effective request identity.  A configured
        # manifest model must not mask a per-request override.
        model = body.get("model", served_config.get("model", UNKNOWN))
    if model in (None, ""):
        model = UNKNOWN
    idealpath = manifest_value.get(
        "stateframe_idealpath", manifest_value.get("idealpath_mode", UNKNOWN)
    )
    idealpath = str(idealpath).strip().lower() if idealpath not in (None, "") else UNKNOWN
    if idealpath == "environment":
        idealpath = "ENVIRONMENT"
    elif idealpath in {"unknown", "missing"}:
        idealpath = UNKNOWN
    elif idealpath not in {"on", "off"}:
        idealpath = UNKNOWN
    return RunCertificate(
        run_id=str(run_id).strip() or UNKNOWN,
        effective_model=str(_identity_value(model)),
        deployed_source_digest=str(_identity_value(deployed_source_digest(manifest_value))),
        idealpath_mode=idealpath.upper() if idealpath == UNKNOWN.lower() else idealpath,
        reasoning_mode=selected.mode,
        request_body_digest=wire_digest(wire),
        served_config=served_config,
    )


class PhaseMeter:
    """Fixed-shape bounded host instrumentation; no per-fact timing list."""

    def __init__(self) -> None:
        self._rows = {
            phase: {"count": 0, "items": 0, "cache_hits": 0, "elapsed_ns": 0}
            for phase in PHASES
        }

    @staticmethod
    def _bounded(value: object, limit: int = MAX_COUNTER) -> int:
        try:
            numeric = int(value)
        except (OverflowError, TypeError, ValueError):
            return 0
        return max(0, min(limit, numeric))

    def record(
        self,
        phase: str,
        elapsed_ns: int,
        *,
        count: int = 1,
        items: int = 0,
        cache_hit: bool = False,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown host-cost phase: {phase}")
        row = self._rows[phase]
        row["count"] = min(MAX_COUNTER, row["count"] + self._bounded(count))
        row["items"] = min(MAX_COUNTER, row["items"] + self._bounded(items))
        row["cache_hits"] = min(MAX_COUNTER, row["cache_hits"] + int(cache_hit))
        row["elapsed_ns"] = min(
            MAX_ELAPSED_NS,
            row["elapsed_ns"] + self._bounded(elapsed_ns, MAX_ELAPSED_NS),
        )

    @contextmanager
    def measure(
        self,
        phase: str,
        *,
        count: int = 1,
        items: int = 0,
        cache_hit: bool = False,
    ) -> Iterator[None]:
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record(
                phase,
                time.perf_counter_ns() - started,
                count=count,
                items=items,
                cache_hit=cache_hit,
            )

    def merge(self, value: Mapping[str, object]) -> None:
        phases = value.get("phases") if isinstance(value, Mapping) else None
        if not isinstance(phases, Mapping):
            return
        for phase in PHASES:
            row = phases.get(phase)
            if not isinstance(row, Mapping):
                continue
            current = self._rows[phase]
            current["count"] = min(
                MAX_COUNTER,
                current["count"] + self._bounded(row.get("count", 0)),
            )
            current["items"] = min(
                MAX_COUNTER,
                current["items"] + self._bounded(row.get("items", 0)),
            )
            current["cache_hits"] = min(
                MAX_COUNTER,
                current["cache_hits"] + self._bounded(row.get("cache_hits", 0)),
            )
            current["elapsed_ns"] = min(
                MAX_ELAPSED_NS,
                current["elapsed_ns"]
                + self._bounded(row.get("elapsed_ns", 0), MAX_ELAPSED_NS),
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "bounded": True,
            "timing_scope": "instrumented_phase_only",
            "phases": {phase: dict(row) for phase, row in self._rows.items()},
        }


__all__ = [
    "CERTIFICATE_SCHEMA",
    "ENVIRONMENT",
    "HARNESS_CONTRACT",
    "MAX_COUNTER",
    "PHASES",
    "PROVIDER_MALFORMED",
    "SEMANTIC_ERROR",
    "ProviderBoundaryContractError",
    "ProviderMalformedResponseError",
    "ReasoningProfile",
    "RunCertificate",
    "PhaseMeter",
    "UNKNOWN",
    "apply_global_reasoning_off",
    "assert_global_reasoning_off",
    "build_run_certificate",
    "deployed_source_digest",
    "enforce_provider_body",
    "reasoning_profile",
    "wire_digest",
]
