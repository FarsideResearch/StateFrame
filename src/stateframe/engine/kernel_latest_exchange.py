"""Bounded, provenance-checked presentation of one completed tool exchange.

This is a presentation diagnostic.  It never treats model prose or raw tool
output as evidence; only the already-public StepLedger projection is eligible.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kernel_tool_reply import outcome_from_receipt, tool_reply


_MODES = frozenset({"off", "quoted", "roles"})
_ALLOWED_TOOLS = frozenset({"read", "grep", "run_tests"})
_PUBLIC_FEEDBACK_LIMIT = 2000
_PAIR_FEEDBACK_LIMIT = 1600
_SERIALIZED_CHAR_CAP = 6000
_CALL_ID_PREFIX = "latest_"


@dataclass(frozen=True)
class LatestExchange:
    """The bounded candidate and the reason it was omitted, if any."""

    mode: str
    messages: tuple[dict[str, object], ...]
    pair_digest: str | None
    omission_reason: str | None


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _matches_digest(value: object, expected: str) -> bool:
    try:
        return _digest(value) == expected
    except (TypeError, ValueError):
        return False


def _bounded_feedback(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n[... middle omitted ...]\n"
    available = limit - len(marker)
    return text[:available // 2] + marker + text[-(available - available // 2):]


def _omit(mode: str, reason: str) -> LatestExchange:
    return LatestExchange(mode=mode, messages=(), pair_digest=None, omission_reason=reason)


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _nonempty_text(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _valid_receipts(record: Mapping[str, object]) -> bool:
    receipts = record.get("receipt_ids")
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)) or not receipts:
        return False
    return all(isinstance(receipt, str) and bool(receipt) for receipt in receipts)


def _public_projection_matches(record: Mapping[str, object], public_action: Mapping[str, object]) -> bool:
    action_parameters = _mapping(record.get("action_parameters"))
    observation = record.get("observation")
    if action_parameters is None or not isinstance(observation, str):
        return False
    if any(public_action.get(key) != record.get(record_key) for key, record_key in (
        ("step", "step"), ("tool", "tool"), ("result", "result"))):
        return False
    if public_action.get("parameters") != action_parameters:
        return False
    return public_action.get("feedback") == _bounded_feedback(observation, _PUBLIC_FEEDBACK_LIMIT)


def _tail_pair(messages: object, expected_tool: str) -> tuple[dict[str, object], str] | None:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or len(messages) < 2:
        return None
    assistant = _mapping(messages[-2])
    tool_result = _mapping(messages[-1])
    if assistant is None or tool_result is None or assistant.get("role") != "assistant":
        return None
    if tool_result.get("role") != "tool":
        return None
    calls = assistant.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)) or len(calls) != 1:
        return None
    call = _mapping(calls[0])
    if call is None:
        return None
    call_id = _nonempty_text(call.get("id"))
    function = _mapping(call.get("function"))
    if (call_id is None or call.get("type") != "function" or function is None
            or function.get("name") != expected_tool):
        return None
    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str):
        return None
    try:
        parsed_arguments = json.loads(raw_arguments)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed_arguments, Mapping):
        return None
    if tool_result.get("tool_call_id") != call_id or "content" not in tool_result:
        return None
    sanitized = {key: value for key, value in parsed_arguments.items()
                 if key not in {"why", "reason"}}
    return dict(sanitized), call_id


def _arguments_are_public(record: Mapping[str, object], parsed_arguments: Mapping[str, object]) -> bool:
    public_parameters = _mapping(record.get("action_parameters"))
    return public_parameters is not None and dict(parsed_arguments) == dict(public_parameters)


def _canonical_messages(
    *,
    mode: str,
    tool: str,
    arguments: Mapping[str, object],
    record_ref: str,
    result_class: str,
    feedback: str,
) -> tuple[tuple[dict[str, object], ...], str]:
    call_id = _CALL_ID_PREFIX + hashlib.sha256(record_ref.encode("utf-8")).hexdigest()[:24]
    result = {
        "record_ref": record_ref,
        "result_class": result_class,
        "binding_status": "current",
        "feedback": feedback,
    }
    reply_outcome, reply_receipt = outcome_from_receipt(result)
    result_text = _canonical_bytes(result).decode("utf-8")
    roles = (
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": tool, "arguments": _canonical_bytes(arguments).decode("utf-8")},
            }],
        },
        tool_reply(call_id, tool, reply_outcome, result_text, receipt=reply_receipt),
    )
    roles_serialized = _canonical_bytes(list(roles)).decode("utf-8")
    quoted = ({"role": "user", "content": roles_serialized},)
    quoted_serialized = _canonical_bytes(list(quoted)).decode("utf-8")
    if mode == "roles":
        return roles, quoted_serialized
    return quoted, quoted_serialized


def build_latest_exchange(
    *,
    mode: str,
    record: Mapping[str, object] | None,
    public_action: Mapping[str, object] | None,
    messages: Sequence[Mapping[str, object]],
    current_binding: Mapping[str, object] | None,
    public_identity: str = "run",
) -> LatestExchange | None:
    """Build an OFF/quoted/roles candidate from one current closed action."""
    if public_identity not in {"run", "semantic"}:
        raise ValueError("public_identity must be run or semantic")
    if mode not in _MODES:
        raise ValueError("mode must be off, quoted or roles")
    if mode == "off":
        return None
    record_map = _mapping(record)
    public_map = _mapping(public_action)
    current_map = _mapping(current_binding)
    if record_map is None:
        return _omit(mode, "missing_record")
    if public_map is None:
        # A host-owned refusal (any REFUSED_* class, not only unchanged-green)
        # is itself the public action.  Refusal paths do not carry the
        # ordinary ``last_action`` copy, but the closed StepLedger record
        # still contains the exact bounded projection needed here.
        if (str(record_map.get("result", "")).upper().startswith("REFUSED_")
                and isinstance(record_map.get("action_parameters"), Mapping)
                and isinstance(record_map.get("observation"), str)):
            public_map = {
                "step": record_map.get("step"),
                "tool": record_map.get("tool"),
                "result": record_map.get("result"),
                "parameters": dict(record_map["action_parameters"]),
                "feedback": _bounded_feedback(record_map["observation"], _PUBLIC_FEEDBACK_LIMIT),
            }
        else:
            return _omit(mode, "missing_public_action")
    if current_map is None:
        return _omit(mode, "missing_current_binding")

    tool = _nonempty_text(record_map.get("tool"))
    record_ref = _nonempty_text(record_map.get("step_record_id"))
    result_class = _nonempty_text(record_map.get("result"))
    after_binding = _mapping(record_map.get("after_binding"))
    if tool is None or record_ref is None or result_class is None or after_binding is None:
        return _omit(mode, "incomplete_record")
    if tool not in _ALLOWED_TOOLS:
        return _omit(mode, "tool_not_allowed")
    if not _valid_receipts(record_map):
        return _omit(mode, "missing_receipts")
    if dict(after_binding) != dict(current_map):
        return _omit(mode, "stale_binding")
    if not _public_projection_matches(record_map, public_map):
        return _omit(mode, "public_projection_mismatch")

    arguments_hash = _nonempty_text(record_map.get("arguments_hash"))
    if arguments_hash is None:
        return _omit(mode, "arguments_hash_mismatch")
    parsed = _tail_pair(messages, tool)
    if parsed is None:
        return _omit(mode, "invalid_latest_tool_pair")
    parsed_arguments, _provider_call_id = parsed
    if not _matches_digest(parsed_arguments, arguments_hash):
        return _omit(mode, "arguments_hash_mismatch")
    if not _arguments_are_public(record_map, parsed_arguments):
        return _omit(mode, "arguments_not_public")

    public_feedback = public_map.get("feedback")
    if not isinstance(public_feedback, str) or len(public_feedback) > _PUBLIC_FEEDBACK_LIMIT:
        return _omit(mode, "feedback_not_bounded")
    feedback = _bounded_feedback(public_feedback, _PAIR_FEEDBACK_LIMIT)
    if public_identity == "semantic":
        # Presentation only: existing receipt, binding and projection checks above
        # remain authoritative. Never derive a public handle from host receipt IDs.
        record_ref = "exchange:" + _digest({
            "step": record_map.get("step"), "tool": tool,
            "arguments": dict(parsed_arguments), "result": result_class,
            "feedback": feedback,
        })
    pair = {
        "tool": tool,
        "args": dict(parsed_arguments),
        "record_ref": record_ref,
        "result_class": result_class,
        "binding_status": "current",
        "feedback": feedback,
    }
    try:
        pair_digest = _digest(pair)
        candidate, serialized = _canonical_messages(
            mode=mode, tool=tool, arguments=parsed_arguments, record_ref=record_ref,
            result_class=result_class, feedback=feedback)
    except (TypeError, ValueError):
        return _omit(mode, "arguments_not_serializable")
    if len(serialized) > _SERIALIZED_CHAR_CAP:
        return _omit(mode, "serialized_pair_exceeds_cap")
    return LatestExchange(mode=mode, messages=candidate,
                          pair_digest=pair_digest, omission_reason=None)
