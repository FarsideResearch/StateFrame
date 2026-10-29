"""Opt-in masking of old tool observations in the history presentation.

The history arm must retain the model-authored transcript and tool-call wire
shape.  This module therefore changes only the ``content`` field of a tool
result, and only after its call/result pairing has been checked.  The stateful
masker keeps the already-rendered prefix and appends new turns verbatim until
``turns`` more assistant turns have arrived; at that boundary it renders the
whole transcript again.  That batch rule is what keeps the submitted prefix
stable between remask points.  The batch state is process-local; enabled
observation masking is therefore deliberately unsupported with continuation
restart, whose restore contract resumes without chat history.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_OBS_MASK"
DEFAULT_TURNS = 10
SCHEMA = "stateframe-observation-mask/1"
PLACEHOLDER_PREFIX = "[observation elided: "


def configured_turns(environ: Mapping[str, str] | None = None) -> int | None:
    """Return the enabled retention window, or ``None`` for the byte path.

    The switch is normally an integer (``STATEFRAME_OBS_MASK=10``).  ``on``
    is an explicit spelling for the documented default window; ``off`` and
    ``0`` retain the disabled path.  Invalid values fail closed at runtime
    construction instead of silently choosing a different retention policy.
    """
    source = os.environ if environ is None else environ
    raw = source.get(MODE_ENV)
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"", "off", "0"}:
        return None
    if value == "on":
        return DEFAULT_TURNS
    try:
        turns = int(value, 10)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{MODE_ENV} must be off, on, or a positive integer") from error
    if turns < 1:
        raise ValueError(f"{MODE_ENV} must be off, on, or a positive integer")
    return turns


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return configured_turns(environ) is not None


def mode(environ: Mapping[str, str] | None = None) -> str:
    turns = configured_turns(environ)
    return "off" if turns is None else str(turns)


def _copy_messages(messages: Sequence[Mapping[str, object]]) -> list[dict]:
    if isinstance(messages, (str, bytes)):
        raise TypeError("observation history must be a sequence of messages")
    copied: list[dict] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("observation history messages must be mappings")
        copied.append(copy.deepcopy(dict(message)))
    return copied


def _tool_name(call: Mapping[str, object]) -> str:
    function = call.get("function")
    if isinstance(function, Mapping) and function.get("name"):
        return str(function["name"])
    if call.get("name"):
        return str(call["name"])
    return "tool"


def _pair_owners(messages: Sequence[Mapping[str, object]]) -> tuple[dict[int, tuple[int, str]], int]:
    """Validate the native transcript and map each tool result to its turn."""
    pending: dict[str, tuple[int, str]] = {}
    owners: dict[int, tuple[int, str]] = {}
    assistant_turn = 0
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise AssertionError("tool result is missing its tool_call_id")
            owner = pending.pop(call_id, None)
            if owner is None:
                raise AssertionError("tool result has no matching tool call")
            owners[index] = owner
            continue
        if pending:
            raise AssertionError("tool call is missing its tool result before the next message")
        if role != "assistant":
            continue
        calls = message.get("tool_calls") or ()
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            raise AssertionError("assistant tool_calls must be a sequence")
        for call in calls:
            if not isinstance(call, Mapping):
                raise AssertionError("assistant tool_calls must contain mappings")
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise AssertionError("tool call is missing its id")
            if call_id in pending:
                raise AssertionError("duplicate tool call id")
            pending[call_id] = (assistant_turn, _tool_name(call))
        assistant_turn += 1
    if pending:
        raise AssertionError("tool call is missing its tool result")
    return owners, assistant_turn


def _scalar(value: object) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _one_line(value: object) -> str:
    """Normalize one interpolated placeholder component to one physical line."""
    return " ".join(str(value).split())


def _first_field(message: Mapping[str, object], names: Sequence[str]) -> str:
    for name in names:
        value = message.get(name)
        if value is not None and value != "":
            return _scalar(value)
    return ""


def _result_excerpt(message: Mapping[str, object]) -> str:
    value = message.get("content")
    if value is None:
        value = message.get("output", message.get("detail", ""))
    if not isinstance(value, str):
        value = _scalar(value)
    # The contract is one line.  Normalize before taking the bounded excerpt
    # so a newline cannot turn a placeholder into multiple prompt messages.
    return _one_line(value)[:120]


def _placeholder(message: Mapping[str, object], *, tool: str, call_id: str) -> str:
    tool = _one_line(tool) or "tool"
    receipt = _one_line(
        _first_field(message, ("receipt_id", "receipt", "id", "tool_call_id"))
        or call_id
    ) or "unknown"
    exit_code = _one_line(_first_field(message, ("exit_code", "exit", "returncode")))
    status = _one_line(_first_field(message, ("status", "result_class", "outcome", "state")))
    exit_status = "/".join(value for value in (exit_code, status) if value) or "unknown"
    return f"{PLACEHOLDER_PREFIX}{tool} {receipt} {exit_status} {_result_excerpt(message)}]"


def _mask_messages(
    messages: Sequence[Mapping[str, object]],
    *,
    turns: int,
        owners: Mapping[int, tuple[int, str]],
        assistant_turns: int,
        protected_source_spans: Sequence[str] = (),
) -> tuple[list[dict], int]:
    cutoff = max(0, assistant_turns - turns)
    masked: list[dict] = []
    masked_count = 0
    for index, original in enumerate(messages):
        message = copy.deepcopy(dict(original))
        owner = owners.get(index)
        if owner is not None and owner[0] < cutoff:
            result_content = str(message.get("content") or "")
            if any(span and span in result_content for span in protected_source_spans):
                masked.append(message)
                continue
            call_id = str(message.get("tool_call_id") or "")
            message["content"] = _placeholder(message, tool=owner[1], call_id=call_id)
            masked_count += 1
        masked.append(message)
    return masked, masked_count


@dataclasses.dataclass(frozen=True)
class ObservationMaskReport:
    schema: str
    assistant_turns: int
    masked_tool_results: int
    remasked: bool
    reused_prefix: bool
    next_boundary: int


class ObservationMasker:
    """Stateful batch renderer for one history-arm runtime."""

    def __init__(self, turns: int = DEFAULT_TURNS) -> None:
        if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
            raise ValueError("observation mask turns must be a positive integer")
        self.turns = turns
        self._raw: tuple[dict, ...] | None = None
        self._masked: tuple[dict, ...] | None = None
        self._assistant_turns = 0
        self._next_boundary = turns
        self._protected_source_digest = ""
        self.last_report = ObservationMaskReport(
            SCHEMA, 0, 0, False, False, turns)

    def apply(self, messages: Sequence[Mapping[str, object]], *,
              protected_source_spans: Sequence[str] = ()) -> list[dict]:
        """Return a masked copy while preserving the native message pairing."""
        raw = _copy_messages(messages)
        protected = tuple(dict.fromkeys(
            span for span in protected_source_spans
            if isinstance(span, str) and span
        ))
        protected_digest = hashlib.sha256(
            "\0".join(protected).encode("utf-8")).hexdigest()
        owners, assistant_turns = _pair_owners(raw)
        previous_raw = self._raw
        prefix_matches = (
            previous_raw is not None
            and len(previous_raw) <= len(raw)
            and list(previous_raw) == raw[:len(previous_raw)]
        )
        can_reuse = (
            prefix_matches
            and assistant_turns >= self._assistant_turns
            and assistant_turns < self._next_boundary
            and self._masked is not None
            and protected_digest == self._protected_source_digest
        )
        if can_reuse:
            masked = [*copy.deepcopy(list(self._masked)),
                      *copy.deepcopy(raw[len(previous_raw):])]
            remasked = False
            reused_prefix = True
            masked_count = sum(
                1 for message in masked
                if isinstance(message.get("content"), str)
                and message["content"].startswith(PLACEHOLDER_PREFIX))
        else:
            masked, masked_count = _mask_messages(
                raw, turns=self.turns, owners=owners,
                assistant_turns=assistant_turns,
                protected_source_spans=protected)
            remasked = True
            reused_prefix = bool(prefix_matches and self._masked is not None)
            self._next_boundary = assistant_turns + self.turns

        self._raw = tuple(copy.deepcopy(raw))
        self._masked = tuple(copy.deepcopy(masked))
        self._assistant_turns = assistant_turns
        self._protected_source_digest = protected_digest
        self.last_report = ObservationMaskReport(
            SCHEMA, assistant_turns, masked_count, remasked,
            reused_prefix, self._next_boundary)
        return copy.deepcopy(masked)


def mask_history(
    messages: Sequence[Mapping[str, object]],
    *,
    turns: int | None = None,
    state: ObservationMasker | None = None,
) -> list[dict]:
    """Apply the configured arm; disabled operation is a deep-copy identity."""
    configured = configured_turns() if turns is None else turns
    if configured is None:
        return _copy_messages(messages)
    masker = state if state is not None else ObservationMasker(configured)
    if masker.turns != configured:
        raise ValueError("observation mask state and configured window differ")
    return masker.apply(messages)


mask_messages = mask_history


__all__ = [
    "DEFAULT_TURNS", "MODE_ENV", "PLACEHOLDER_PREFIX", "SCHEMA",
    "ObservationMaskReport", "ObservationMasker", "configured_turns",
    "enabled", "mask_history", "mask_messages", "mode",
]
