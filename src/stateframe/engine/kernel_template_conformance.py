"""Endpoint-template conformance evidence for the R-120/R-121 slice.

The runtime owns the messages and the selected endpoint, but only the endpoint
owns the final chat-template rendering.  This module records a small, safe
contract at the final prepare boundary and, when the request is submitted,
optionally asks the endpoint for its rendered prompt and token IDs.  It never
claims that a JSON body is equivalent to a rendered prompt.

``STATEFRAME_TEMPLATE_CONFORMANCE`` is deliberately off by default.  The
normal request and journal paths do not import or call this module when it is
off.  ``on`` is an observation arm: a renderer can be injected by a release
runner, or the local endpoint's ``/apply-template`` and ``/tokenize`` routes
are used.  Rendering failures remain explicit evidence rather than silently
turning into a passing comparison.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence


def _journal_safe_fields(payload):
    """Emit boundary (10-05 L109): rename journal-RESERVED keys; see kernel_continuation_journal."""
    from kernel_continuation_journal import journal_safe_fields

    return journal_safe_fields(payload)


MODE_ENV = "STATEFRAME_TEMPLATE_CONFORMANCE"
SCHEMA = "stateframe-template-conformance/1"
VALID_MODES = frozenset({"off", "on"})
_ACTION_KEYS = frozenset({
    "action", "intended_action", "next_action", "proposed_action", "tool",
})
_STATE_PREFIX = "[StateFrame]"
_MAX_ANCHOR_LENGTH = 160
_MAX_TOKEN_IDS = 250_000
_MAX_LIVE_RESPONSE_BYTES = 16 * 1024 * 1024


class ConformanceError(ValueError):
    """The endpoint rendering did not preserve the prepared contract."""


def mode(value: str | Mapping[str, str] | None = None) -> str:
    """Return the validated switch, which is ``off`` unless explicitly enabled."""
    if isinstance(value, Mapping):
        raw = value.get(MODE_ENV, "off")
    else:
        raw = os.environ.get(MODE_ENV, "off") if value is None else value
    result = str(raw).strip().lower() or "off"
    if result not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return result


def _public(value: object) -> object:
    """Make a bounded JSON-safe copy for a journal contract."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _public(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_public(child) for child in value]
    return str(value)


def _canonical(value: object) -> bytes:
    return json.dumps(_public(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _content(message: Mapping[str, object]) -> str:
    value = message.get("content", "")
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(_public(value), sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _message_roles(messages: Sequence[Mapping[str, object]]) -> list[str]:
    return [str(message.get("role", "unknown")) for message in messages]


def _messages(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [message for message in value if isinstance(message, Mapping)]


def _tool_name(call: Mapping[str, object]) -> str:
    function = call.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
    else:
        name = call.get("name")
    return str(name).strip() if name not in (None, "") else "UNKNOWN_TOOL"


def _tool_id(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _tool_calls(messages: Sequence[Mapping[str, object]]) -> list[tuple[str, str, int]]:
    calls: list[tuple[str, str, int]] = []
    for index, message in enumerate(messages):
        value = message.get("tool_calls")
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            continue
        for call in value:
            if isinstance(call, Mapping):
                calls.append((_tool_id(call.get("id")), _tool_name(call), index))
    return calls


def _tool_result_pairing(messages: Sequence[Mapping[str, object]]) -> dict[str, object]:
    calls = _tool_calls(messages)
    by_id: dict[str, tuple[str, int]] = {}
    duplicate_calls: list[str] = []
    for call_id, name, index in calls:
        if not call_id or call_id in by_id:
            duplicate_calls.append(call_id or "MISSING_CALL_ID")
        else:
            by_id[call_id] = (name, index)
    pairs: list[dict[str, object]] = []
    missing_results: list[str] = []
    seen_results: set[str] = set()
    orphaned_results: list[str] = []
    out_of_order_results: list[str] = []
    for call_id, (name, call_index) in by_id.items():
        result_indexes = [
            index for index, message in enumerate(messages)
            if str(message.get("role", "")) == "tool"
            and _tool_id(message.get("tool_call_id")) == call_id
        ]
        if len(result_indexes) != 1:
            missing_results.append(call_id)
        elif result_indexes[0] <= call_index:
            out_of_order_results.append(call_id)
        else:
            seen_results.add(call_id)
            pairs.append({"call_id": call_id, "tool_name": name,
                          "result_index": result_indexes[0]})
    for message in messages:
        if str(message.get("role", "")) != "tool":
            continue
        call_id = _tool_id(message.get("tool_call_id"))
        if call_id not in by_id or call_id in seen_results and len([
                item for item in messages
                if str(item.get("role", "")) == "tool"
                and _tool_id(item.get("tool_call_id")) == call_id]) > 1:
            orphaned_results.append(call_id or "MISSING_CALL_ID")
    pairs.sort(key=lambda item: int(item["result_index"]))
    return {
        "valid": (not missing_results and not orphaned_results and not duplicate_calls
                  and not out_of_order_results),
        "pairs": pairs,
        "missing_results": missing_results,
        "orphaned_results": orphaned_results,
        "out_of_order_results": out_of_order_results,
    }


def _state_object(content: str) -> object | None:
    if not content.startswith(_STATE_PREFIX):
        return None
    _, _, tail = content.partition("\n")
    if not tail.strip():
        return None
    try:
        value = json.loads(tail)
    except (TypeError, ValueError, UnicodeError):
        return None
    return value if isinstance(value, (Mapping, list)) else None


def _state_anchors(content: str) -> tuple[list[str], list[str]]:
    """Return state-preservation anchors and action anchors separately."""
    if not content.startswith(_STATE_PREFIX):
        return [], []
    anchors = [_STATE_PREFIX]
    actions: list[str] = []
    value = _state_object(content)
    if isinstance(value, Mapping):
        def visit(key: object, child: object) -> None:
            key_text = str(key)
            if key_text in _ACTION_KEYS:
                if isinstance(child, (str, int)) and not isinstance(child, bool):
                    action = str(child).strip()
                    if action:
                        actions.append(action[:_MAX_ANCHOR_LENGTH])
                return
            if isinstance(child, Mapping):
                for nested_key, nested_value in child.items():
                    visit(nested_key, nested_value)
                return
            if isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
                for nested in child:
                    visit(key_text, nested)
                return
            if isinstance(child, (str, int, float)) and not isinstance(child, bool):
                text = str(child).strip()
                if text and len(text) <= _MAX_ANCHOR_LENGTH:
                    anchors.append(key_text[:_MAX_ANCHOR_LENGTH])
                    anchors.append(text)
        for key, child in value.items():
            visit(key, child)
    else:
        # A non-JSON StateFrame is still identifiable by its header and a short
        # content fragment; this avoids treating an opaque state tail as proof
        # of the intended action alone.
        fragment = content[len(_STATE_PREFIX):].strip()
        if fragment:
            anchors.append(fragment[:_MAX_ANCHOR_LENGTH])
    unique_anchors = list(dict.fromkeys(anchors))
    unique_actions = list(dict.fromkeys(actions))
    return unique_anchors, unique_actions


def _state_metadata(messages: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], list[str]]:
    rows = []
    anchors: list[str] = []
    state_actions: list[str] = []
    for index, message in enumerate(messages):
        content = _content(message)
        if not content.startswith(_STATE_PREFIX):
            continue
        state_anchors, actions = _state_anchors(content)
        rows.append({"index": index, "role": str(message.get("role", "unknown")),
                     "content": content})
        anchors.extend(state_anchors)
        state_actions.extend(actions)
    state = {
        "message_count": len(rows),
        "message_indexes": [row["index"] for row in rows],
        "message_digest": _digest(rows),
        "anchors": list(dict.fromkeys(anchors)),
    }
    return state, list(dict.fromkeys(state_actions))


def _adapter_identity(adapter: object) -> dict[str, object]:
    if hasattr(adapter, "as_dict") and callable(adapter.as_dict):
        adapter = adapter.as_dict()
    if not isinstance(adapter, Mapping):
        return {"name": str(adapter) if adapter not in (None, "") else "UNKNOWN"}
    result = _public(adapter)
    return result if isinstance(result, dict) else {"name": str(result)}


def _identity_value(adapter: Mapping[str, object], *names: str) -> str:
    for name in names:
        value = adapter.get(name)
        if value not in (None, "", "UNKNOWN", "MISSING"):
            return str(value)
    return "UNKNOWN"


def build_contract(messages: object, tools: object, *, arm: str, endpoint: str,
                   model_id: str, adapter: object, request_id: str,
                   step: int | None, hop: bool, resumed: bool,
                   intended_action: str | None = None) -> dict[str, object]:
    """Build the public contract for one exact request boundary."""
    normalized_messages = _messages(messages)
    adapter_public = _adapter_identity(adapter)
    state, state_actions = _state_metadata(normalized_messages)
    pairing = _tool_result_pairing(normalized_messages)
    actions = list(state_actions)
    for call_id, name, _index in _tool_calls(normalized_messages):
        if name not in actions:
            actions.append(name)
    if intended_action not in (None, "") and str(intended_action) not in actions:
        actions.append(str(intended_action))
    return {
        "payload_schema": SCHEMA,
        "arm": str(arm),
        "endpoint": str(endpoint),
        "model_id": str(model_id),
        "adapter": adapter_public,
        "chat_template": _identity_value(adapter_public, "chat_template", "template"),
        "tokenizer": _identity_value(adapter_public, "tokenizer", "tokenizer_name"),
        "request_id": str(request_id),
        "step": step,
        "hop": bool(hop),
        "resumed": bool(resumed),
        "message_roles": _message_roles(normalized_messages),
        "message_count": len(normalized_messages),
        "messages_digest": _digest(normalized_messages),
        "tools_digest": _digest(tools),
        "state": state,
        "intended_actions": actions,
        "tool_result_pairing": pairing,
    }


def _validated_token_ids(token_ids: object) -> list[int]:
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes, bytearray)):
        raise ConformanceError("rendered token IDs must be a sequence")
    if len(token_ids) > _MAX_TOKEN_IDS:
        raise ConformanceError("rendered token IDs exceed the bounded capture size")
    result = []
    for token in token_ids:
        if type(token) is not int or token < 0:
            raise ConformanceError("rendered token IDs must be non-negative integers")
        result.append(token)
    if not result:
        raise ConformanceError("endpoint returned no rendered token IDs")
    return result


def verify_rendered(contract: Mapping[str, object], messages: object,
                    rendered_text: str, token_ids: object, *,
                    fail_closed: bool = False) -> dict[str, object]:
    """Verify semantic anchors against the endpoint's actual rendered prompt."""
    if not isinstance(rendered_text, str) or not rendered_text.strip():
        raise ConformanceError("endpoint returned an empty rendered prompt")
    rendered_tokens = _validated_token_ids(token_ids)
    state = contract.get("state")
    state_anchors = state.get("anchors", []) if isinstance(state, Mapping) else []
    if not isinstance(state_anchors, Sequence) or isinstance(state_anchors, (str, bytes)):
        state_anchors = []
    folded = rendered_text.casefold()
    state_preserved = all(str(anchor).casefold() in folded for anchor in state_anchors)
    actions = contract.get("intended_actions", [])
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        actions = []
    intended_action_preserved = all(str(action).casefold() in folded for action in actions)
    pairing = contract.get("tool_result_pairing")
    pairing_valid = isinstance(pairing, Mapping) and pairing.get("valid") is True
    pairing_preserved, pairing_observations = _rendered_tool_pairing(
        messages, pairing if isinstance(pairing, Mapping) else {}, rendered_text)
    valid = bool(state_preserved and intended_action_preserved and pairing_valid
                 and pairing_preserved)
    result = {
        "rendered_token_ids": rendered_tokens,
        "rendered_token_count": len(rendered_tokens),
        "rendered_prompt_sha256": "sha256:" + hashlib.sha256(
            rendered_text.encode("utf-8")).hexdigest(),
        "state_preserved": state_preserved,
        "intended_action_preserved": intended_action_preserved,
        "tool_result_pairing_valid": pairing_valid,
        "tool_result_pairing_preserved": pairing_preserved,
        "tool_result_pairing_observations": pairing_observations,
        "valid": valid,
    }
    if fail_closed and not valid:
        failed = []
        if not state_preserved:
            failed.append("state")
        if not intended_action_preserved:
            failed.append("intended action")
        if not pairing_valid:
            failed.append("tool-result pairing")
        elif not pairing_preserved:
            failed.append("rendered tool-result pairing")
        raise ConformanceError("rendered contract lost " + ", ".join(failed))
    return result


def _identity_report(contract: Mapping[str, object], rendered: Mapping[str, object]) -> dict[str, object]:
    """Compare endpoint-reported template/tokenizer identity when supplied."""
    expected_template = str(contract.get("chat_template", "UNKNOWN"))
    expected_tokenizer = str(contract.get("tokenizer", "UNKNOWN"))
    actual_template = rendered.get("chat_template", rendered.get("template"))
    actual_tokenizer = rendered.get("tokenizer", rendered.get("tokenizer_sha256"))
    template_match = (
        actual_template in (None, "") or expected_template == "UNKNOWN"
        or str(actual_template) == expected_template)
    tokenizer_match = (
        actual_tokenizer in (None, "") or expected_tokenizer == "UNKNOWN"
        or str(actual_tokenizer) == expected_tokenizer)
    return {
        "template_identity_match": template_match,
        "tokenizer_identity_match": tokenizer_match,
        "observed_chat_template": (
            str(actual_template)[:_MAX_ANCHOR_LENGTH] if actual_template not in (None, "")
            else "UNREPORTED"),
        "observed_tokenizer": (
            str(actual_tokenizer)[:_MAX_ANCHOR_LENGTH] if actual_tokenizer not in (None, "")
            else "UNREPORTED"),
    }


def _tool_result_anchors(content: str) -> tuple[str, ...]:
    """Return small stable content anchors that survive JSON reformatting."""
    normalized = " ".join(str(content).split())
    if not normalized:
        return ()
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError, UnicodeError):
        return (normalized[:_MAX_ANCHOR_LENGTH],)

    values: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                collect(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                collect(child)
        elif value is None or isinstance(value, (str, int, float, bool)):
            text = "null" if value is None else str(value)
            if text and len(text) <= _MAX_ANCHOR_LENGTH:
                values.append(text)

    collect(parsed)
    if not values:
        return (normalized[:_MAX_ANCHOR_LENGTH],)
    return tuple(dict.fromkeys(values))[:16]


def _rendered_tool_pairing(messages: object, pairing: Mapping[str, object],
                           rendered_text: str) -> tuple[bool, list[dict[str, object]]]:
    """Check that each valid source pair has call/result evidence in the render."""
    if pairing.get("valid") is not True:
        return False, [{"status": "source_pairing_invalid"}]
    normalized_messages = _messages(messages)
    calls = {call_id: (name, call_index)
             for call_id, name, call_index in _tool_calls(normalized_messages)}
    pairs = pairing.get("pairs", ())
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes, bytearray)):
        return False, [{"status": "pairs_missing"}]
    folded = rendered_text.casefold()
    cursor = 0
    observations: list[dict[str, object]] = []
    preserved = True
    for pair in pairs:
        if not isinstance(pair, Mapping):
            preserved = False
            observations.append({"status": "pair_malformed"})
            continue
        call_id = str(pair.get("call_id", ""))
        call = calls.get(call_id)
        if call is None:
            preserved = False
            observations.append({"call_id": call_id, "status": "source_call_missing"})
            continue
        name, _call_index = call
        name_folded = name.casefold()
        name_index = folded.find(name_folded, cursor) if name_folded else cursor
        if name_index < 0:
            preserved = False
            observations.append({"call_id": call_id, "tool_name": name,
                                 "status": "call_not_rendered"})
            continue
        result_message = next(
            (message for message in normalized_messages
             if str(message.get("role", "")) == "tool"
             and _tool_id(message.get("tool_call_id")) == call_id),
            None,
        )
        anchors = _tool_result_anchors(
            _content(result_message) if result_message is not None else "")
        result_index = max(name_index, cursor)
        result_positions = [folded.find(anchor.casefold(), name_index + len(name_folded))
                            for anchor in anchors]
        result_preserved = (not anchors or all(position >= 0 for position in result_positions))
        if result_preserved and result_positions:
            result_index = max(result_positions)
        ordered = result_preserved and result_index >= name_index
        if not ordered:
            preserved = False
        observations.append({
            "call_id": call_id,
            "tool_name": name,
            "result_index": pair.get("result_index"),
            "call_rendered": True,
            "result_rendered": result_preserved,
            "ordered": ordered,
        })
        if ordered:
            cursor = result_index + 1
    return preserved, observations


def _endpoint_base(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(str(endpoint))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConformanceError("live template rendering requires an HTTP endpoint")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def _post_json(url: str, payload: Mapping[str, object], key: str, *,
              deadline: float | None) -> Mapping[str, object]:
    wire = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=wire, headers={"Content-Type": "application/json",
                                 "Authorization": "Bearer " + str(key)})
    timeout = 15.0
    if deadline is not None:
        timeout = min(timeout, deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("template conformance deadline expired")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(_MAX_LIVE_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_LIVE_RESPONSE_BYTES:
        raise ConformanceError("template endpoint response exceeds bounded size")
    parsed = json.loads(raw)
    if not isinstance(parsed, Mapping):
        raise ConformanceError("template endpoint returned a non-object response")
    return parsed


def render_endpoint_request(body: Mapping[str, object], endpoint: str, key: str,
                            *, deadline: float | None = None,
                            post: Callable[[str, Mapping[str, object]], Mapping[str, object]] | None = None
                            ) -> dict[str, object]:
    """Render and tokenize one body through the selected endpoint's routes."""
    base = _endpoint_base(endpoint) if post is None else ""
    post_json = post or (lambda route, payload: _post_json(base + route, payload, key,
                                                            deadline=deadline))
    rendered_response = post_json("/apply-template", {
        **dict(body), "add_generation_prompt": True,
    })
    rendered_text = next((rendered_response.get(name) for name in ("prompt", "rendered", "text")
                          if isinstance(rendered_response.get(name), str)), None)
    if not isinstance(rendered_text, str) or not rendered_text:
        raise ConformanceError("template endpoint returned no rendered prompt")
    token_response = post_json("/tokenize", {
        "content": rendered_text, "add_special": False, "parse_special": True,
    })
    token_ids = next((token_response.get(name) for name in ("tokens", "token_ids", "ids")
                      if isinstance(token_response.get(name), list)), None)
    result = {"rendered_text": rendered_text, "token_ids": _validated_token_ids(token_ids)}
    for key_name in ("template", "chat_template", "tokenizer", "tokenizer_sha256"):
        if token_response.get(key_name) is not None:
            result[key_name] = _public(token_response[key_name])
        elif rendered_response.get(key_name) is not None:
            result[key_name] = _public(rendered_response[key_name])
    return result


def _selected_identity(capture: object, endpoint: str) -> tuple[str, str, object]:
    decision = getattr(capture, "model_router_decision", None)
    selected = getattr(decision, "endpoint", None)
    endpoint_name = getattr(capture, "model_router_selected_endpoint", None)
    if not endpoint_name:
        endpoint_name = getattr(selected, "name", None)
    endpoint_name = str(endpoint_name or "")
    model_id = getattr(capture, "model_router_effective_model_id", None)
    if not model_id:
        model_id = getattr(selected, "model_id", None)
    if not model_id:
        model_id = getattr(capture, "journal", None)
        model_id = getattr(model_id, "manifest", {}).get("model", "UNKNOWN")
    adapter = getattr(selected, "adapter", None)
    if adapter is None:
        adapter = getattr(capture, "template_conformance_adapter", None)
    if adapter in (None, {}):
        existing = getattr(capture, "template_conformance_contract", None)
        if isinstance(existing, Mapping):
            existing_adapter = existing.get("adapter")
            if isinstance(existing_adapter, Mapping) and any(
                    value not in (None, "", "UNKNOWN", "MISSING")
                    for value in (existing_adapter.get("chat_template"),
                                  existing_adapter.get("tokenizer"))):
                adapter = existing_adapter
            else:
                adapter = {
                    "chat_template": existing.get("chat_template", "UNKNOWN"),
                    "tokenizer": existing.get("tokenizer", "UNKNOWN"),
                }
    if adapter in (None, {}, {"chat_template": "UNKNOWN", "tokenizer": "UNKNOWN"}):
        manifest = getattr(getattr(capture, "journal", None), "manifest", {})
        if isinstance(manifest, Mapping):
            adapter = {
                "chat_template": manifest.get("chat_template", "UNKNOWN"),
                "tokenizer": manifest.get("tokenizer", "UNKNOWN"),
            }
    if adapter is None:
        adapter = {}
    if not endpoint_name:
        existing = getattr(capture, "template_conformance_contract", None)
        if isinstance(existing, Mapping) and existing.get("endpoint") not in (None, ""):
            endpoint_name = str(existing["endpoint"])
    if not endpoint_name:
        # A contract made at prepare time is authoritative; URL parsing is only
        # a fallback for standalone RequestCapture usage.
        parsed = urllib.parse.urlsplit(str(endpoint))
        endpoint_name = parsed.hostname or str(endpoint)
    return endpoint_name, str(model_id), adapter


def _arm(capture: object, runtime: object | None) -> str:
    explicit = getattr(capture, "template_conformance_arm", None)
    if explicit in (None, "") and runtime is not None:
        explicit = getattr(runtime, "template_conformance_arm", None)
    if explicit in (None, ""):
        explicit = os.environ.get("STATEFRAME_TEMPLATE_CONFORMANCE_ARM")
    if explicit in (None, ""):
        explicit = os.environ.get("LONG_HORIZON_ARM")
    if explicit in (None, ""):
        explicit = "router" if getattr(capture, "model_router_decision", None) is not None else "default"
    return str(explicit)


def capture_prepared(runtime: object, capture: object, messages: object,
                     tools: object) -> dict[str, object] | None:
    """Record the exact final prepared request when the switch is enabled."""
    configured = getattr(runtime, "stateframe_template_conformance_mode", None)
    if configured is None:
        configured = mode()
    if mode(configured) == "off":
        return None
    endpoint_name, model_id, adapter = _selected_identity(capture, "")
    previous = getattr(runtime, "_template_conformance_last_endpoint", None)
    hop = previous is not None and previous != endpoint_name
    runtime._template_conformance_last_endpoint = endpoint_name
    resumed = bool(getattr(runtime, "_template_conformance_resumed", False))
    arm = _arm(capture, runtime)
    contract = build_contract(
        messages, tools, arm=arm, endpoint=endpoint_name, model_id=model_id,
        adapter=adapter, request_id=str(getattr(capture, "request_id", "UNKNOWN")),
        step=getattr(capture, "step", None), hop=hop, resumed=resumed,
    )
    capture.template_conformance_contract = contract
    capture.template_conformance_mode = mode(configured)
    capture.template_conformance_runtime = runtime
    capture.template_conformance_messages = copy.deepcopy(_messages(messages))
    capture.template_conformance_tools = copy.deepcopy(tools)
    event = runtime.journal.emit(
        "template_conformance_prepared", producer="kernel_template_conformance",
        dispatch=False, **contract,
    )
    return event


def _renderer_result(renderer: Callable[..., object], body: Mapping[str, object],
                     endpoint: str, key: str, deadline: float | None) -> Mapping[str, object]:
    try:
        signature = inspect.signature(renderer)
    except (TypeError, ValueError):
        signature = None
    accepts_deadline = signature is None or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or (parameter.name == "deadline"
            and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY})
        for parameter in (signature.parameters.values() if signature is not None else ()))
    deadline_is_positional = signature is not None and any(
        parameter.name == "deadline"
        and parameter.kind == inspect.Parameter.POSITIONAL_ONLY
        for parameter in signature.parameters.values())
    if accepts_deadline:
        result = renderer(body, endpoint, key, deadline=deadline)
    elif deadline_is_positional:
        result = renderer(body, endpoint, key, deadline)
    else:
        result = renderer(body, endpoint, key)
    if not isinstance(result, Mapping):
        raise ConformanceError("template renderer returned a non-object result")
    return result


def capture_submitted(capture: object, body: Mapping[str, object], endpoint: str,
                      key: str, *, mode_value: str | None = None,
                      runtime: object | None = None,
                      renderer: Callable[..., object] | None = None,
                      deadline: float | None = None) -> dict[str, object] | None:
    """Capture endpoint rendering evidence for the exact adapted request body."""
    selected_mode = mode(mode_value if mode_value is not None else getattr(
        capture, "template_conformance_mode", None) or mode())
    if selected_mode == "off":
        return None
    base_contract = getattr(capture, "template_conformance_contract", None)
    wire_messages = body.get("messages") if isinstance(body, Mapping) else []
    wire_tools = body.get("tools", []) if isinstance(body, Mapping) else []
    endpoint_name, model_id, adapter = _selected_identity(capture, endpoint)
    if isinstance(base_contract, Mapping):
        arm = str(base_contract.get("arm", _arm(capture, runtime)))
        hop = bool(base_contract.get("hop", False))
        resumed = bool(base_contract.get("resumed", False))
    else:
        arm = _arm(capture, runtime)
        hop = False
        resumed = bool(getattr(runtime, "_template_conformance_resumed", False)) if runtime else False
    contract = build_contract(
        wire_messages, wire_tools, arm=arm, endpoint=endpoint_name,
        model_id=model_id, adapter=adapter,
        request_id=str(getattr(capture, "request_id", "UNKNOWN")),
        step=getattr(capture, "step", None), hop=hop, resumed=resumed,
    )
    capture.template_conformance_contract = contract
    render = renderer or getattr(capture, "template_conformance_renderer", None)
    if render is None:
        render = render_endpoint_request
    try:
        rendered = (_renderer_result(render, body, endpoint, key, deadline)
                    if render is not render_endpoint_request
                    else render_endpoint_request(body, endpoint, key, deadline=deadline))
        verification = verify_rendered(
            contract, wire_messages, rendered.get("rendered_text", ""),
            rendered.get("token_ids", ()), fail_closed=False)
        identity = _identity_report(contract, rendered)
        verification.update(identity)
        verification["valid"] = bool(
            verification["valid"] and identity["template_identity_match"]
            and identity["tokenizer_identity_match"])
        render_status = "verified" if verification["valid"] else "mismatch"
    except Exception as error:  # noqa: BLE001 - evidence must record live failures
        rendered = {}
        verification = {
            "rendered_token_ids": [], "rendered_token_count": 0,
            "rendered_prompt_sha256": None, "state_preserved": False,
            "intended_action_preserved": False,
            "tool_result_pairing_valid": contract["tool_result_pairing"]["valid"],
            "tool_result_pairing_preserved": False,
            "tool_result_pairing_observations": [],
            "template_identity_match": False,
            "tokenizer_identity_match": False,
            "observed_chat_template": "UNREPORTED",
            "observed_tokenizer": "UNREPORTED",
            "valid": False,
        }
        render_status = "render_failed"
        error_type = type(error).__name__
    else:
        error_type = None
    journal = getattr(capture, "journal", None)
    if journal is None or not callable(getattr(journal, "emit", None)):
        raise ConformanceError("template conformance requires an audit journal")
    fields = {
        **contract,
        "producer": "kernel_template_conformance",
        "dispatch": False,
        "endpoint_url": str(endpoint),
        "render_status": render_status,
        **verification,
    }
    if error_type is not None:
        fields["render_error_type"] = error_type
    for key_name in ("template", "chat_template", "tokenizer", "tokenizer_sha256"):
        if key_name in rendered and key_name not in fields:
            fields[key_name] = _public(rendered[key_name])
    return journal.emit("template_conformance_submitted", **_journal_safe_fields(fields))


__all__ = [
    "ConformanceError", "MODE_ENV", "SCHEMA", "build_contract",
    "capture_prepared", "capture_submitted", "mode", "render_endpoint_request",
    "verify_rendered",
]
