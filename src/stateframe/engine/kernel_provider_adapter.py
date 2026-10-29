"""Opt-in provider/native-continuation contract for one OpenAI-compatible path.

This module is deliberately narrow.  It describes the exact message dialect the
runtime can prove today; it is not a provider registry or a translation layer.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_PROVIDER_ADAPTER"
SCHEMA = "stateframe-provider-adapter/1"
ADAPTER = "openai-compatible-v1"

_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_SUPPORTED_TEMPLATE_NAMES = frozenset({
    "openai", "openai-compatible", "openai-compatible-v1",
})
_TEMPLATE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FORBIDDEN_REASONING_FIELDS = frozenset({
    "response_reasoning_content", "reasoning", "analysis", "thinking",
})
_FORBIDDEN_TOP_LEVEL_REASONING_FIELDS = _FORBIDDEN_REASONING_FIELDS | {
    "reasoning_content",
}
_TEXT_CONTENT_BLOCK = "text"


class ProviderAdapterError(ValueError):
    """Raised when a request cannot be represented by the supported adapter."""


def adapter_mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict, opt-in adapter mode."""
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on, got {value!r}")
    return value


def configuration() -> dict[str, object]:
    """Return the portable capability contract recorded in a run manifest."""
    return {
        "schema": SCHEMA,
        "adapter": ADAPTER,
        "template_family": "openai-compatible",
        "roles": ["system", "user", "assistant", "tool"],
        "tool_calls": {
            "assistant_field": "tool_calls",
            "result_role": "tool",
            "result_id_field": "tool_call_id",
            "function_arguments": "json-object-string",
        },
        "reasoning": {
            "request_field": "reasoning_content",
            "response_field": "reasoning_content",
            "prefill_marker": "continue_final_message=reasoning_content",
            "host_aliases": [],
        },
        "cache": {
            "eligibility": "cache_prompt requires nonnegative id_slot",
            "cold_resume": "provider cache is never reused",
        },
        "transport": {
            "deadline": "absolute_monotonic",
            "cancel": "cooperative_pre_dispatch",
        },
        "cold_resume": {
            "history": "journal-native-continuation",
            "provider_context_reused": False,
        },
    }


def _fail(message: str) -> None:
    raise ProviderAdapterError(f"provider adapter: {message}")


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        _fail(f"{label} must be a sequence")
    return value


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _template_identity(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().upper() == "MISSING":
        _fail("template identity is required")
    identity = value.strip()
    normalized = identity.lower()
    if normalized not in _SUPPORTED_TEMPLATE_NAMES and not _TEMPLATE_DIGEST.fullmatch(normalized):
        _fail(f"unsupported template identity {identity!r}")
    return normalized


def _validate_content(message: Mapping[str, object], *, index: int,
                      role: str) -> None:
    """Accept only the text content forms this adapter can preserve exactly."""
    has_content = "content" in message
    content = message.get("content")
    continuation_without_content = (
        role == "assistant" and
        ("tool_calls" in message or "reasoning_content" in message))
    if not has_content:
        if continuation_without_content:
            return
        _fail(f"message {index} is missing content")
    if content is None:
        if continuation_without_content:
            return
        _fail(f"message {index} has unsupported null content")
    if isinstance(content, str):
        return
    if (isinstance(content, Sequence)
            and not isinstance(content, (str, bytes, bytearray, Mapping))):
        for block_index, block in enumerate(content):
            if (not isinstance(block, Mapping)
                    or block.get("type") != _TEXT_CONTENT_BLOCK
                    or not isinstance(block.get("text"), str)):
                _fail(f"message {index} content block {block_index} is not supported text")
        return
    _fail(f"message {index} content must be text or text blocks")


def _validate_tools(tools: object) -> tuple[set[str], list[str]]:
    if tools is None:
        return set(), []
    rows = _sequence(tools, label="tools")
    names: set[str] = set()
    ordered: list[str] = []
    for index, tool in enumerate(rows):
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            _fail(f"tool {index} is not an OpenAI function tool")
        function = tool.get("function")
        if not isinstance(function, Mapping):
            _fail(f"tool {index} has no function definition")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            _fail(f"tool {index} has no function name")
        if name in names:
            _fail(f"duplicate tool name {name!r}")
        description = function.get("description")
        if description is not None and not isinstance(description, str):
            _fail(f"tool {index} description is not text")
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            _fail(f"tool {index} parameters are not a JSON schema object")
        strict = function.get("strict")
        if strict is not None and type(strict) is not bool:
            _fail(f"tool {index} strict must be boolean")
        names.add(name)
        ordered.append(name)
    return names, ordered


def _validate_tool_choice(tool_choice: object, *, tool_names: set[str]) -> dict[str, object] | str | None:
    """Validate OpenAI choices, including V43G's terminal function form."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice not in {"auto", "none", "required"}:
            _fail(f"unsupported tool_choice {tool_choice!r}")
        return tool_choice
    if not isinstance(tool_choice, Mapping):
        _fail("tool_choice must be auto, none, required, or a function object")
    if tool_choice.get("type") != "function":
        _fail("tool_choice function object has unsupported type")
    function = tool_choice.get("function")
    if (not isinstance(function, Mapping)
            or not isinstance(function.get("name"), str)
            or function.get("name") not in tool_names):
        _fail("tool_choice function object names an undeclared tool")
    return {"type": "function", "function": {"name": function["name"]}}


def _validate_call(call: object, *, tool_names: set[str], seen_ids: set[str]) -> str:
    if not isinstance(call, Mapping) or call.get("type") != "function":
        _fail("tool call is not an OpenAI function call")
    call_id = call.get("id")
    if (not isinstance(call_id, str) or not call_id.strip()
            or call_id != call_id.strip()):
        _fail("tool call has no id")
    if call_id in seen_ids:
        _fail(f"duplicate tool call id {call_id!r}")
    index = call.get("index")
    if index is not None and (type(index) is not int or index < 0):
        _fail(f"tool call {call_id!r} index is not a nonnegative integer")
    function = call.get("function")
    if not isinstance(function, Mapping):
        _fail(f"tool call {call_id!r} has no function")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        _fail(f"tool call {call_id!r} has no function name")
    if name not in tool_names:
        _fail(f"tool call {call_id!r} names undeclared tool {name!r}")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        _fail(f"tool call {call_id!r} arguments are not a JSON string")
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError) as error:
        _fail(f"tool call {call_id!r} arguments are not JSON: {error}")
    if not isinstance(parsed, Mapping):
        _fail(f"tool call {call_id!r} arguments are not a JSON object")
    seen_ids.add(call_id)
    return call_id


def _validate_messages(messages: object, *, tool_names: set[str],
                       allow_reasoning_prefill: bool) -> dict[str, object]:
    rows = _sequence(messages, label="messages")
    if not rows:
        _fail("messages cannot be empty")
    roles: list[str] = []
    pending: dict[str, int] = {}
    seen_ids: set[str] = set()
    results = 0
    reasoning_field: str | None = None
    prefill = False
    call_count = 0
    for index, message in enumerate(rows):
        if not isinstance(message, Mapping):
            _fail(f"message {index} is not an object")
        role = message.get("role")
        if not isinstance(role, str) or role not in _ALLOWED_ROLES:
            _fail(f"message {index} has unsupported role {role!r}")
        _validate_content(message, index=index, role=role)
        if "name" in message and (
                not isinstance(message.get("name"), str)
                or not str(message.get("name")).strip()):
            _fail(f"message {index} name is not non-empty text")
        if role == "tool" and "tool_call_id" not in message:
            _fail(f"tool result at message {index} has no matching tool call")
        if role != "tool" and "tool_call_id" in message:
            _fail(f"message {index} has a tool_call_id outside tool role")
        if role != "assistant" and "tool_calls" in message:
            _fail(f"message {index} has tool_calls outside assistant role")
        roles.append(role)
        if role == "tool":
            if pending == {}:
                _fail(f"orphan tool result at message {index}")
            result_id = message.get("tool_call_id")
            if not isinstance(result_id, str) or result_id not in pending:
                _fail(f"tool result at message {index} has no matching tool call")
            pending.pop(result_id)
            results += 1
        elif pending:
            _fail("tool call must be immediately followed by its tool result")

        for field in _FORBIDDEN_REASONING_FIELDS:
            if field in message:
                _fail(f"unsupported reasoning field {field}")
        if role == "assistant" and "reasoning_content" in message:
            value = message.get("reasoning_content")
            if not isinstance(value, str) or not value:
                _fail("reasoning_content must be a non-empty string")
            if index != len(rows) - 1 or message.get("tool_calls"):
                _fail("reasoning prefill must be on the final assistant message")
            if message.get("content") not in (None, ""):
                _fail("reasoning prefill cannot include assistant content")
            if not allow_reasoning_prefill:
                _fail("reasoning prefill requires the wire continuation marker")
            reasoning_field = "reasoning_content"
            prefill = True
        elif "reasoning_content" in message:
            _fail("reasoning_content is only valid on an assistant message")

        if role == "assistant" and "tool_calls" in message:
            calls = _sequence(message.get("tool_calls"), label="assistant tool_calls")
            if not calls:
                _fail("assistant tool_calls cannot be empty")
            if pending:
                _fail("assistant tool calls cannot start before prior tool results")
            for call in calls:
                call_id = _validate_call(call, tool_names=tool_names, seen_ids=seen_ids)
                pending[call_id] = index
                call_count += 1
    if pending:
        _fail("tool call has no matching tool result")
    return {
        "roles": roles,
        "role_digest": _digest(roles),
        "tool_pairing": {"calls": call_count, "results": results,
                          "complete": call_count == results},
        "reasoning": {"field": reasoning_field, "prefill": prefill},
    }


def _manifest_contract(manifest: Mapping[str, object]) -> str:
    if not isinstance(manifest, Mapping):
        _fail("manifest must be an object")
    if manifest.get("provider_adapter_schema") != SCHEMA:
        _fail("manifest provider adapter schema does not match")
    config = manifest.get("provider_adapter")
    if not isinstance(config, Mapping) or dict(config) != configuration():
        _fail("manifest provider adapter configuration does not match")
    return _template_identity(manifest.get("chat_template"))


def _flag(manifest: Mapping[str, object], name: str) -> bool:
    value = manifest.get(name, False)
    if type(value) is not bool:
        _fail(f"manifest {name} must be boolean")
    return value


def _continuation_contract(manifest: Mapping[str, object], *, cold_resume: bool) -> dict[str, bool]:
    continuation = _flag(manifest, "continuation_executive")
    native_batches = _flag(manifest, "native_tool_batches")
    native_prefix = _flag(manifest, "native_prefix_authority")
    restart = _flag(manifest, "continuation_restart")
    if native_batches and not continuation:
        _fail("native tool batches require continuation executive")
    if native_prefix and (not continuation or not native_batches):
        _fail("native prefix authority requires continuation and native tool batches")
    if restart and not continuation:
        _fail("continuation restart requires continuation executive")
    if cold_resume and (not continuation or not restart):
        _fail("cold resume requires continuation restart")
    return {
        "enabled": continuation,
        "native_tool_batches": native_batches,
        "native_prefix_authority": native_prefix,
        "restart": restart,
    }


def _cache_contract(*, cache_prompt: object, id_slot: object,
                    cold_resume: bool) -> dict[str, object]:
    if cache_prompt is not None and type(cache_prompt) is not bool:
        _fail("cache_prompt must be boolean")
    if id_slot is not None and (type(id_slot) is not int or id_slot < 0):
        _fail("id_slot must be a nonnegative integer")
    requested = cache_prompt is True
    if requested and id_slot is None:
        _fail("cache_prompt requires a pinned slot id_slot")
    if cold_resume and requested:
        _fail("provider cache is not eligible during cold resume")
    return {
        "requested": requested,
        "eligible": requested and not cold_resume,
        "id_slot": id_slot,
        "reason": "cold_resume_forces_cold_prefix" if cold_resume
        else ("pinned_slot" if requested else "not_requested"),
    }


def transport_contract(*, deadline: float | None = None, now: float | None = None,
                       cancel_requested: bool = False,
                       cancel_scope: str = "request") -> dict[str, object]:
    if not isinstance(cancel_scope, str) or cancel_scope != "request":
        _fail("cancel scope must be request-scoped")
    if type(cancel_requested) is not bool:
        _fail("cancel_requested must be boolean")
    if cancel_requested:
        _fail("request cancellation was requested before dispatch")
    if deadline is None:
        deadline_report = {"kind": "none", "absolute_monotonic": None}
    else:
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            _fail("deadline must be a finite monotonic timestamp")
        if not math.isfinite(deadline):
            _fail("deadline must be a finite monotonic timestamp")
        current = time.monotonic() if now is None else now
        if (isinstance(current, bool) or not isinstance(current, (int, float))
                or not math.isfinite(current)):
            _fail("monotonic clock reading must be finite")
        if deadline <= current:
            _fail("deadline has expired")
        deadline_report = {"kind": "absolute_monotonic",
                           "absolute_monotonic": float(deadline),
                           "remaining": float(deadline - current)}
    return {
        "deadline": deadline_report,
        "cancel": {"scope": cancel_scope, "requested": False,
                    "semantics": "cooperative_pre_dispatch"},
    }


def validate_prepared_request(messages: object, tools: object, *,
                              template_identity: object,
                              continuation_enabled: bool = False,
                              native_tool_batches: bool = False,
                              native_prefix_authority: bool = False,
                              cold_resume: bool = False,
                              restart_enabled: bool = False) -> dict[str, object]:
    """Validate the rendered request before it can be handed to a provider."""
    if type(continuation_enabled) is not bool:
        _fail("continuation_enabled must be boolean")
    if type(native_tool_batches) is not bool:
        _fail("native_tool_batches must be boolean")
    if type(native_prefix_authority) is not bool:
        _fail("native_prefix_authority must be boolean")
    if type(cold_resume) is not bool:
        _fail("cold_resume must be boolean")
    if type(restart_enabled) is not bool:
        _fail("restart_enabled must be boolean")
    if native_tool_batches and not continuation_enabled:
        _fail("native tool batches require continuation executive")
    if native_prefix_authority and (not continuation_enabled or not native_tool_batches):
        _fail("native prefix authority requires continuation and native tool batches")
    if cold_resume and (not continuation_enabled or not restart_enabled):
        _fail("cold resume requires continuation restart")
    template = _template_identity(template_identity)
    tool_names, _ = _validate_tools(tools)
    message_report = _validate_messages(
        messages, tool_names=tool_names,
        allow_reasoning_prefill=True,
    )
    return {
        "accepted": True,
        "schema": SCHEMA,
        "adapter": ADAPTER,
        "template_identity": template,
        "rendered_roles": message_report["roles"],
        "rendered_role_digest": message_report["role_digest"],
        "tool_pairing": message_report["tool_pairing"],
        "reasoning": message_report["reasoning"],
        "cache": _cache_contract(cache_prompt=False, id_slot=None,
                                  cold_resume=cold_resume),
        "continuation": {
            "enabled": continuation_enabled,
            "native_tool_batches": native_tool_batches,
            "native_prefix_authority": native_prefix_authority,
            "restart": restart_enabled,
        },
        "cold_resume": bool(cold_resume),
        "transport": transport_contract(),
    }


def validate_wire_request(body: Mapping[str, object], *,
                          manifest: Mapping[str, object],
                          deadline: float | None = None,
                          cold_resume: bool = False,
                          cancel_requested: bool = False) -> dict[str, object]:
    """Validate the final serialized request after all body mutators ran."""
    if not isinstance(body, Mapping):
        _fail("wire request must be an object")
    template = _manifest_contract(manifest)
    continuation = _continuation_contract(manifest, cold_resume=cold_resume)
    for field in _FORBIDDEN_TOP_LEVEL_REASONING_FIELDS:
        if field in body:
            _fail(f"unsupported reasoning field {field}")
    tool_names, _ = _validate_tools(body.get("tools"))
    normalized_tool_choice = _validate_tool_choice(
        body.get("tool_choice"), tool_names=tool_names)
    continuation_marker = body.get("continue_final_message")
    if continuation_marker is not None and continuation_marker != "reasoning_content":
        _fail("unsupported reasoning continuation marker")
    message_report = _validate_messages(
        body.get("messages"), tool_names=tool_names,
        allow_reasoning_prefill=continuation_marker == "reasoning_content",
    )
    if message_report["reasoning"]["prefill"] and continuation_marker != "reasoning_content":
        _fail("reasoning prefill requires continue_final_message=reasoning_content")
    if continuation_marker == "reasoning_content" and not message_report["reasoning"]["prefill"]:
        _fail("reasoning continuation marker has no reasoning prefill")
    return {
        "accepted": True,
        "schema": SCHEMA,
        "adapter": ADAPTER,
        "template_identity": template,
        "rendered_roles": message_report["roles"],
        "rendered_role_digest": message_report["role_digest"],
        "tool_pairing": message_report["tool_pairing"],
        "tool_choice": normalized_tool_choice,
        "reasoning": message_report["reasoning"],
        "cache": _cache_contract(cache_prompt=body.get("cache_prompt"),
                                  id_slot=body.get("id_slot"),
                                  cold_resume=cold_resume),
        "continuation": {
            "enabled": continuation["enabled"],
            "native_tool_batches": continuation["native_tool_batches"],
            "native_prefix_authority": continuation["native_prefix_authority"],
            "restart": continuation["restart"],
        },
        "cold_resume": bool(cold_resume),
        "transport": transport_contract(deadline=deadline,
                                         cancel_requested=cancel_requested),
    }


def validate_response(response: Mapping[str, object], tools: object = None) -> dict[str, object]:
    """Validate the provider response fields consumed by native continuation."""
    if not isinstance(response, Mapping):
        _fail("provider response must be an object")
    for field in _FORBIDDEN_TOP_LEVEL_REASONING_FIELDS:
        if field in response:
            _fail(f"unsupported reasoning field {field}")
    choices = _sequence(response.get("choices"), label="provider choices")
    if not choices:
        _fail("provider response has no choices")
    if len(choices) != 1:
        _fail("provider adapter supports exactly one response choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        _fail("provider choice is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        _fail("provider choice has no message")
    role = message.get("role")
    if role != "assistant":
        _fail("provider response message role must be explicit assistant")
    _validate_content(message, index=0, role="assistant")
    for field in _FORBIDDEN_REASONING_FIELDS:
        if field in message:
            _fail(f"unsupported reasoning field {field}")
    reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        _fail("provider reasoning_content must be a string")
    tool_names, _ = _validate_tools(tools)
    raw_calls = message.get("tool_calls", ())
    calls = _sequence(raw_calls, label="provider tool_calls")
    seen_ids: set[str] = set()
    for call in calls:
        _validate_call(call, tool_names=tool_names, seen_ids=seen_ids)
    finish_reason = choice.get("finish_reason")
    if finish_reason == "tool_calls" and not calls:
        _fail("tool_calls finish reason has no tool calls")
    return {
        "accepted": True,
        "schema": SCHEMA,
        "adapter": ADAPTER,
        "reasoning_field": "reasoning_content" if reasoning is not None else None,
        "tool_calls": len(calls),
        "finish_reason": finish_reason,
    }


def cold_resume_contract(manifest: Mapping[str, object], records: object) -> dict[str, object]:
    """Prove a resumed run starts from journal state, never provider cache state."""
    template = _manifest_contract(manifest)
    continuation = _continuation_contract(manifest, cold_resume=True)
    rows = _sequence(records, label="continuation records")
    prepared_count = 0
    for record in rows:
        if not isinstance(record, Mapping) or record.get("kind") != "request_prepared":
            continue
        recorded_contract = record.get("provider_adapter")
        provider_request = record.get("provider_request", True)
        if type(provider_request) is not bool:
            _fail("continuation record provider_request must be boolean")
        if not provider_request:
            if recorded_contract is not None:
                _fail("non-provider continuation record has a provider contract")
            continue
        if (not isinstance(recorded_contract, Mapping)
                or recorded_contract.get("schema") != SCHEMA
                or recorded_contract.get("adapter") != ADAPTER):
            _fail("continuation record is missing the provider adapter contract")
        validate_prepared_request(
            record.get("messages"), record.get("tools", ()),
            template_identity=template,
            continuation_enabled=continuation["enabled"],
            native_tool_batches=continuation["native_tool_batches"],
            native_prefix_authority=continuation["native_prefix_authority"],
            cold_resume=True,
            restart_enabled=continuation["restart"],
        )
        prepared_count += 1
    return {
        "schema": SCHEMA,
        "adapter": ADAPTER,
        "template_identity": template,
        "cold_resume": True,
        "provider_context_reused": False,
        "cache_prompt": False,
        "replay": "journal-native-continuation",
        "continuation": continuation,
        "validated_prepared_requests": prepared_count,
    }
