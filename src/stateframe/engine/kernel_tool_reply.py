"""Typed, fail-open construction and accounting for tool replies."""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
import copy
from dataclasses import dataclass
from enum import Enum
import json
import os
import re


MODE_ENV = "STATEFRAME_TOOL_REPLY_CONSTRUCTOR"
_INTERNAL_RECEIPT_KEY = "_stateframe_tool_reply_receipt"
_HOST_NO_RESULT_RECOVERY = (
    "nothing changed; inspect the current state before retrying this call."
)
_SAFE_TOOL = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,79}$")
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_CAUSES = {
    "duplicate_reply", "empty_reply", "exact_receipt", "legacy_reply",
    "missing_reply", "prepared_reply",
}


class Outcome(str, Enum):
    APPLIED = "APPLIED"
    NO_OP = "NO_OP"
    HITS = "HITS"
    NO_HITS = "NO_HITS"
    READ = "READ"
    GREEN = "GREEN"
    RED = "RED"
    REFUSED = "REFUSED"
    ACCEPTED = "ACCEPTED"
    ERROR = "ERROR"


class ToolReplyMessage(dict[str, object]):
    """A provider-shaped mapping with non-serialized host receipt metadata."""

    def __init__(self, payload: Mapping[str, object], receipt: Mapping[str, object]):
        super().__init__(payload)
        self._stateframe_tool_reply_receipt = dict(receipt)


@dataclass(frozen=True)
class _RenderedOutcome:
    outcome: Outcome
    label: str
    receipt: dict[str, object]


def enabled(environ: Mapping[str, object] | None = None) -> bool:
    """Return whether typed formatting is active; invalid values safely leave it off."""
    source = os.environ if environ is None else environ
    value = source.get(MODE_ENV, "off")
    return isinstance(value, str) and value.strip().lower() == "on"


def _safe_tool(tool: object) -> str:
    value = str(tool or "").strip()
    return value if _SAFE_TOOL.fullmatch(value) else "tool"


def _safe_code(value: object, *, default: str) -> str:
    code = str(value or "").strip().lower()
    return code if _SAFE_CODE.fullmatch(code) else default


def _render_outcome(outcome: Outcome, receipt: Mapping[str, object] | None) -> _RenderedOutcome:
    fields = receipt if isinstance(receipt, Mapping) else {}
    if not isinstance(outcome, Outcome):
        outcome = Outcome.ERROR
        fields = {"code": "invalid_outcome"}

    if outcome is Outcome.HITS:
        count = fields.get("hit_count", fields.get("count"))
        if type(count) is int and count >= 0:
            return _RenderedOutcome(outcome, f"HITS {count}", {
                "kind": outcome.value, "count": count,
            })
        outcome = Outcome.ACCEPTED
    elif outcome is Outcome.READ:
        start = fields.get("start_line")
        end = fields.get("end_line")
        if (type(start) is int and type(end) is int and start >= 1 and end >= start):
            return _RenderedOutcome(outcome, f"READ lines {start}-{end}", {
                "kind": outcome.value, "start_line": start, "end_line": end,
            })
        outcome = Outcome.ACCEPTED
    elif outcome in {Outcome.REFUSED, Outcome.ERROR}:
        code = _safe_code(fields.get("code"), default="unspecified")
        return _RenderedOutcome(outcome, f"{outcome.value} {code}", {
            "kind": outcome.value, "code": code,
        })

    return _RenderedOutcome(outcome, outcome.value, {"kind": outcome.value})


def outcome_from_receipt(
        receipt: Mapping[str, object] | None,
        call: Mapping[str, object] | None = None,
) -> tuple[Outcome, dict[str, object]]:
    """Map structured receipt/call fields to the closed reply outcome set."""
    row = receipt if isinstance(receipt, Mapping) else {}
    stored = row.get("tool_outcome")
    if isinstance(stored, Mapping):
        try:
            outcome = Outcome(str(stored.get("kind", "")))
        except ValueError:
            outcome = Outcome.ACCEPTED
        return outcome, {
            key: stored[key] for key in ("code", "count", "start_line", "end_line")
            if key in stored
        }

    result_class = ""
    for key in ("result_class", "result", "status"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            result_class = value.strip().upper()
            break
    if result_class in {"APPLIED", "CREATED", "CHANGED"}:
        return Outcome.APPLIED, {}
    if result_class in {"NO_OP", "UNCHANGED", "NO_CHANGE"}:
        return Outcome.NO_OP, {}
    if result_class in {"NO_HITS", "NO_MATCHES", "GREP_NO_HITS"}:
        return Outcome.NO_HITS, {}
    if result_class in {"GREEN", "PASS", "PASSED"}:
        return Outcome.GREEN, {}
    if result_class in {"RED", "FAIL", "FAILED_CHECK"}:
        return Outcome.RED, {}
    if result_class in {"REFUSED", "REJECTED", "UNKNOWN_TOOL", "UNSUPPORTED_ACTION"}:
        return Outcome.REFUSED, {"code": "tool_refused"}
    if result_class in {"ERROR", "HARNESS_ERROR", "HOST_ERROR", "FAILED"}:
        return Outcome.ERROR, {"code": "tool_error"}

    count = row.get("hit_count", row.get("match_count"))
    if type(count) is int and count >= 0 and str(row.get("kind", "")).lower() in {
            "grep", "search"}:
        return (Outcome.HITS, {"hit_count": count}) if count else (Outcome.NO_HITS, {})

    if isinstance(call, Mapping):
        function = call.get("function")
        name = function.get("name") if isinstance(function, Mapping) else call.get("name")
        arguments = function.get("arguments") if isinstance(function, Mapping) else call.get("arguments")
        if str(name or "").strip() == "read" and isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, Mapping):
                start = parsed.get("start_line", parsed.get("line"))
                end = parsed.get("end_line", start)
                if type(start) is int and type(end) is int and start >= 1 and end >= start:
                    return Outcome.READ, {"start_line": start, "end_line": end}
    return Outcome.ACCEPTED, {}


def _reply_receipt(
        call_id: str, tool: str, rendered: _RenderedOutcome,
        receipt: Mapping[str, object] | None,
) -> dict[str, object]:
    marker: dict[str, object] = {
        "tool_call_id": call_id,
        "tool": tool,
        "outcome": rendered.receipt,
    }
    cause = receipt.get("cause") if isinstance(receipt, Mapping) else None
    if isinstance(cause, str) and cause in _SAFE_CAUSES:
        marker["cause"] = cause
    return marker


def tool_reply(
        tool_call_id: str,
        tool: str,
        outcome: Outcome,
        text: str,
        *,
        receipt: MutableMapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one reply; OFF preserves the legacy message mapping and content exactly."""
    if not enabled():
        return {"role": "tool", "tool_call_id": tool_call_id, "content": text}

    safe_tool = _safe_tool(tool)
    visible = text if isinstance(text, str) else ""
    receipt_fields: Mapping[str, object] = receipt if isinstance(receipt, Mapping) else {}
    if not visible.strip():
        outcome = Outcome.ERROR
        receipt_fields = {**dict(receipt_fields), "code": "host_no_result"}
        visible = _HOST_NO_RESULT_RECOVERY
        if isinstance(receipt, MutableMapping):
            receipt["cause"] = receipt.get("cause") or "empty_reply"

    rendered = _render_outcome(outcome, receipt_fields)
    safe_call_id = tool_call_id if isinstance(tool_call_id, str) else str(tool_call_id or "")
    marker = _reply_receipt(safe_call_id, safe_tool, rendered, receipt_fields)
    if isinstance(receipt, MutableMapping):
        receipt["tool_outcome"] = dict(rendered.receipt)
    return ToolReplyMessage({
        "role": "tool",
        "tool_call_id": safe_call_id,
        "content": f"{safe_tool} {rendered.label}: {visible.strip()}",
    }, marker)


def _tool_call_for_id(messages: Sequence[object], tool_call_id: object) -> Mapping[str, object] | None:
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            continue
        for call in reversed(calls):
            if isinstance(call, Mapping) and call.get("id") == tool_call_id:
                return call
    return None


def tool_name_for_call(messages: Sequence[object], tool_call_id: object) -> str:
    """Resolve a tool name from its structural assistant call, never from reply prose."""
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return "tool"
    call = _tool_call_for_id(messages, tool_call_id)
    if call is None:
        return "tool"
    function = call.get("function")
    name = function.get("name") if isinstance(function, Mapping) else call.get("name")
    return _safe_tool(name)


def tool_reply_for_call(
        messages: Sequence[object],
        tool_call_id: object,
        text: str,
        *,
        tool: object = None,
        outcome: Outcome | None = None,
        receipt: MutableMapping[str, object] | None = None,
        no_id_role: str = "tool",
        legacy_key_order: Sequence[str] | None = None,
) -> dict[str, object]:
    """Convenience adapter for transcript builders with optional call IDs."""
    call_id = tool_call_id if isinstance(tool_call_id, str) else str(tool_call_id or "")
    if not call_id and no_id_role == "user":
        return {"role": "user", "content": text}
    tool_name = _safe_tool(tool) if tool is not None else tool_name_for_call(messages, call_id)
    if outcome is None:
        if enabled():
            inferred, inferred_receipt = outcome_from_receipt(
                receipt, _tool_call_for_id(messages, call_id),
            )
            outcome = inferred
            if receipt is None:
                receipt = inferred_receipt
            elif inferred_receipt:
                if isinstance(receipt, MutableMapping):
                    receipt.update(inferred_receipt)
                else:
                    receipt = {**dict(receipt), **inferred_receipt}
        else:
            outcome = Outcome.ACCEPTED
    message = tool_reply(call_id, tool_name, outcome, text, receipt=receipt)
    if not enabled() and legacy_key_order:
        ordered = {key: message[key] for key in legacy_key_order if key in message}
        ordered.update({key: value for key, value in message.items() if key not in ordered})
        return ordered
    return message


def compose_tool_replies(
        messages: object,
        journal: object = None,
        *,
        request_id: object = None,
        step: object = None,
        defer_receipts: bool = False,
) -> list[object]:
    """Type and fill replies; final-wire callers may defer receipt stripping."""
    if not enabled() or not isinstance(messages, list):
        return messages if isinstance(messages, list) else []

    calls: dict[str, Mapping[str, object]] = {}
    results: dict[str, list[Mapping[str, object]]] = {}
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "assistant":
            raw_calls = message.get("tool_calls")
            if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, (str, bytes)):
                for call in raw_calls:
                    if isinstance(call, Mapping):
                        call_id = call.get("id")
                        if isinstance(call_id, str) and call_id and call_id not in calls:
                            calls[call_id] = call
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                results.setdefault(call_id, []).append(message)

    composed: list[object] = []
    emitted_ids: set[str] = set()
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            call_id = call_id if isinstance(call_id, str) else ""
            if call_id not in calls:
                composed.append(copy_tool_reply_message(message))
            continue

        composed.append(copy_tool_reply_message(message))
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
            continue
        for call in raw_calls:
            if not isinstance(call, Mapping):
                continue
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                continue
            if call_id in emitted_ids:
                continue
            tool = tool_name_for_call(messages, call_id)
            candidates = results.get(call_id, [])
            chosen = next((item for item in candidates
                           if isinstance(item.get("content"), str)
                           and item.get("content", "").strip()),
                          candidates[0] if candidates else None)
            if chosen is None:
                composed.append(tool_reply(
                    call_id, tool, Outcome.ERROR, "",
                    receipt={"cause": "missing_reply"},
                ))
            else:
                content = chosen.get("content")
                has_marker = isinstance(chosen, ToolReplyMessage) and isinstance(
                    getattr(chosen, "_stateframe_tool_reply_receipt", None), Mapping)
                if not isinstance(content, str) or not content.strip():
                    composed.append(tool_reply(
                        call_id, tool, Outcome.ERROR, "",
                        receipt={"cause": "empty_reply"},
                    ))
                elif has_marker:
                    composed.append(copy_tool_reply_message(chosen))
                else:
                    composed.append(tool_reply(
                        call_id, tool, Outcome.ACCEPTED, content,
                        receipt={"cause": "legacy_reply"},
                    ))
            if len(candidates) > 1:
                emit = getattr(journal, "emit", None)
                write = getattr(journal, "write", None)
                try:
                    fields = {
                        "producer": "kernel_tool_reply",
                        "request_id": request_id,
                        "step": step,
                        "tool_call_id": call_id,
                        "tool": tool,
                        "removed_count": len(candidates) - 1,
                        "dispatch": False,
                    }
                    if callable(emit):
                        emit("tool_reply_duplicate_removed", **fields)
                    elif callable(write):
                        write("tool_reply_duplicate_removed", **fields)
                except Exception:
                    pass
            emitted_ids.add(call_id)

    if not defer_receipts:
        record_and_strip_receipts(
            composed, journal, request_id=request_id, step=step,
        )
    return composed


def copy_tool_reply_message(message: Mapping[str, object]) -> dict[str, object]:
    """Copy provider mappings while preserving typed receipt attributes on subclasses."""
    try:
        copied = copy.deepcopy(message)
    except Exception:
        copied = dict(message)
    return copied if isinstance(copied, dict) else dict(message)


def record_and_strip_receipts(
        messages: object,
        journal: object,
        *,
        request_id: object = None,
        step: object = None,
) -> int:
    """Journal typed reply receipts and remove private metadata before provider serialization."""
    if not isinstance(messages, list):
        return 0
    recorded = 0
    emit = getattr(journal, "emit", None)
    write = getattr(journal, "write", None)
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        marker = getattr(message, "_stateframe_tool_reply_receipt", None)
        if marker is not None:
            try:
                delattr(message, "_stateframe_tool_reply_receipt")
            except AttributeError:
                pass
        else:
            marker = message.pop(_INTERNAL_RECEIPT_KEY, None)
        if not isinstance(marker, Mapping):
            continue
        recorded += 1
        fields: dict[str, object] = {
            "producer": "kernel_tool_reply",
            "request_id": request_id,
            "step": step,
            "tool_call_id": marker.get("tool_call_id"),
            "tool": marker.get("tool"),
            "outcome": marker.get("outcome"),
            "dispatch": False,
        }
        try:
            if callable(emit):
                emit("tool_reply_receipt", **fields)
                cause = marker.get("cause")
                if cause in {"empty_reply", "missing_reply", "duplicate_reply"}:
                    emit("tool_reply_filled", cause=cause, **{
                        key: value for key, value in fields.items()
                        if key != "dispatch"
                    }, dispatch=False)
            elif callable(write):
                write("tool_reply_receipt", **fields)
                cause = marker.get("cause")
                if cause in {"empty_reply", "missing_reply", "duplicate_reply"}:
                    write("tool_reply_filled", cause=cause, **{
                        key: value for key, value in fields.items()
                        if key != "dispatch"
                    }, dispatch=False)
        except Exception:
            # A receipt failure must never turn reply construction into a refusal.
            continue
    return recorded


__all__ = [
    "MODE_ENV", "Outcome", "compose_tool_replies", "enabled", "outcome_from_receipt",
    "copy_tool_reply_message", "record_and_strip_receipts", "tool_name_for_call",
    "tool_reply", "tool_reply_for_call",
]
