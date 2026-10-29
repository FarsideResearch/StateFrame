"""Opt-in, model-authored compaction for the legacy ReAct history arm.

This module deliberately owns only transcript mechanics.  It does not inspect the
workspace, receipts, tests, or a host-side state frame.  The caller supplies the
ordinary transcript to the summary model and receives the resulting continuation
summary back as one ordinary user message.

The feature is disabled unless ``REACT_HISTORY_COMPACTION=on``.  In the disabled
case :class:`HistoryCompactor` returns the exact input list object and does not call
the summary callback, charge a budget, or journal an event.
"""
from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


HISTORY_SUMMARY_MARKER = "[history-compaction-summary]"
DEFAULT_CONTEXT_TOKENS = 65_536
DEFAULT_THRESHOLD = 0.70
DEFAULT_KEEP_TURNS = 4

_CONTEXT_ENV_NAMES = (
    "REACT_CONTEXT_TOKENS",
    "REACT_CONTEXT_SIZE",
    "REACT_CONTEXT_WINDOW",
    "REACT_MAX_CONTEXT",
    "REACT_MODEL_CONTEXT_TOKENS",
    "REACT_MODEL_CONTEXT",
    "STATEFRAME_CONTEXT_SIZE",
    "STATEFRAME_MODEL_CONTEXT_SIZE",
    "STATEFRAME_CONTEXT_TOKENS",
    "STATEFRAME_MODEL_CONTEXT_TOKENS",
    "STATEFRAME_CONTEXT_WINDOW",
    "STATEFRAME_MAX_CONTEXT",
    "N_CTX",
)


def enabled(value: object = None) -> bool:
    """Resolve the feature switch without treating an absent value as a mutation."""
    raw = os.environ.get("REACT_HISTORY_COMPACTION", "off") if value is None else value
    normalized = str(raw).strip().lower()
    if normalized in {"on", "1", "true", "yes"}:
        return True
    if normalized in {"off", "0", "false", "no", ""}:
        return False
    raise ValueError("REACT_HISTORY_COMPACTION must be on or off")


def _positive_int(value: object, name: str) -> int:
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _token_count(value: object, name: str) -> int:
    """Parse ordinary integer token counts plus convenient ``64k``/``1m`` forms."""
    text = str(value).strip().lower().replace("_", "")
    multiplier = 1
    if text.endswith("k"):
        multiplier, text = 1_000, text[:-1]
    elif text.endswith("m"):
        multiplier, text = 1_000_000, text[:-1]
    try:
        result = int(float(text) * multiplier)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive token count") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive token count")
    return result


def configured_context_tokens(context_tokens: object = None) -> int:
    """Return the configured per-slot context, defaulting to the measured 64k slot."""
    if context_tokens is not None:
        return _token_count(context_tokens, "context_tokens")
    for name in _CONTEXT_ENV_NAMES:
        if name in os.environ:
            return _token_count(os.environ[name], name)
    return DEFAULT_CONTEXT_TOKENS


def _threshold_value(value: object = None) -> float | int:
    raw = os.environ.get("REACT_COMPACTION_THRESHOLD", str(DEFAULT_THRESHOLD)) \
        if value is None else value
    text = str(raw).strip().lower()
    if text.endswith("%"):
        try:
            parsed = float(text[:-1]) / 100.0
        except ValueError as error:
            raise ValueError("REACT_COMPACTION_THRESHOLD must be a ratio or token count") from error
        if not 0 < parsed <= 1:
            raise ValueError("REACT_COMPACTION_THRESHOLD percentage must be between 0 and 100")
        return parsed
    try:
        parsed = float(text)
    except ValueError as error:
        raise ValueError("REACT_COMPACTION_THRESHOLD must be a ratio or token count") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("REACT_COMPACTION_THRESHOLD must be positive")
    if parsed <= 1:
        return parsed
    if parsed != int(parsed):
        raise ValueError("absolute REACT_COMPACTION_THRESHOLD must be an integer")
    return int(parsed)


def threshold_tokens(context_tokens: object = None, threshold: object = None) -> int:
    """Resolve a ratio (the documented form) or an explicit absolute token threshold."""
    context = configured_context_tokens(context_tokens)
    value = _threshold_value(threshold)
    if isinstance(value, float):
        return max(1, math.ceil(context * value))
    return value


def configured_keep_turns(keep_turns: object = None) -> int:
    if keep_turns is not None:
        result = int(str(keep_turns).strip())
    else:
        raw = next((os.environ[name] for name in (
            "REACT_COMPACTION_KEEP_TURNS",
            "REACT_HISTORY_COMPACTION_KEEP_TURNS",
            "REACT_COMPACTION_LAST_K",
        ) if name in os.environ), str(DEFAULT_KEEP_TURNS))
        try:
            result = int(str(raw).strip())
        except (TypeError, ValueError) as error:
            raise ValueError("compaction keep-turns must be a nonnegative integer") from error
    if result < 0:
        raise ValueError("compaction keep-turns must be a nonnegative integer")
    return result


def estimate_prompt_tokens(messages: Sequence[Mapping[str, object]],
                           tools: object = None) -> int:
    """Use the repository's serialized-character/4 estimate for a next request."""
    payload = {"messages": list(messages), "tools": tools}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return (len(encoded) + 3) // 4


def _tool_calls(message: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    calls = message.get("tool_calls") or ()
    if not isinstance(calls, (list, tuple)):
        raise AssertionError("tool_calls must be a sequence")
    return calls


def assert_tool_call_pairs(messages: Sequence[Mapping[str, object]]) -> None:
    """Reject any transcript that would submit an orphaned tool result or call."""
    pending: set[str] = set()
    for message in messages:
        if not isinstance(message, Mapping):
            raise AssertionError("transcript messages must be mappings")
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise AssertionError("orphaned or duplicate tool result ID")
            pending.remove(call_id)
            continue
        if pending:
            raise AssertionError("tool calls missing results before next message")
        if role == "assistant":
            for call in _tool_calls(message):
                if not isinstance(call, Mapping):
                    raise AssertionError("tool call must be a mapping")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in pending:
                    raise AssertionError("missing or duplicate tool call ID")
                pending.add(call_id)
    if pending:
        raise AssertionError("tool calls missing results at transcript end")


def _is_summary_message(message: Mapping[str, object]) -> bool:
    return (message.get("role") == "user"
            and str(message.get("content") or "").startswith(HISTORY_SUMMARY_MARKER))


def split_turns(messages: Sequence[Mapping[str, object]], *, prefix_count: int = 2
                ) -> tuple[tuple[Mapping[str, object], ...], ...]:
    """Group dialogue into assistant-led turns without splitting tool exchanges."""
    if not 0 <= prefix_count <= len(messages):
        raise ValueError("prefix_count is outside the transcript")
    turns: list[list[Mapping[str, object]]] = []
    leading: list[Mapping[str, object]] = []
    current: list[Mapping[str, object]] = []
    for message in messages[prefix_count:]:
        # A prior compaction summary is replaced by the next summary.  It is not
        # one of the verbatim turns and is deliberately omitted from this split.
        if not current and _is_summary_message(message):
            continue
        if message.get("role") == "assistant":
            if current:
                turns.append(current)
            current = [message]
        elif current:
            current.append(message)
        else:
            leading.append(message)
    if current:
        turns.append(current)
    if leading:
        if turns:
            turns[0] = leading + turns[0]
        else:
            turns.append(leading)
    return tuple(tuple(turn) for turn in turns)


def compact_messages(messages: Sequence[Mapping[str, object]], summary: str, *,
                     keep_turns: int = DEFAULT_KEEP_TURNS,
                     original_prefix: Sequence[Mapping[str, object]] | None = None
                     ) -> list[dict[str, object]]:
    """Replace older complete turns with one model-authored continuation summary."""
    assert_tool_call_pairs(messages)
    keep = configured_keep_turns(keep_turns)
    turns = split_turns(messages)
    if len(turns) <= keep:
        return [copy.deepcopy(dict(message)) for message in messages]
    text = str(summary or "").strip()
    if not text:
        raise ValueError("a nonempty summary is required")
    prefix_source = (original_prefix if original_prefix is not None
                     else messages[:2])
    prefix = [copy.deepcopy(dict(message)) for message in prefix_source]
    if len(prefix) < 2:
        raise ValueError("history compaction requires system prompt and original brief")
    retained = turns[-keep:] if keep else ()
    result = prefix + [{"role": "user", "content": HISTORY_SUMMARY_MARKER + "\n" + text}]
    result.extend(copy.deepcopy(dict(message)) for turn in retained for message in turn)
    assert_tool_call_pairs(result)
    return result


SUMMARY_INSTRUCTION = """Summarize the work so far for another coding-agent turn.
Use only information present in the supplied conversation; do not invent host facts,
test results, file changes, or completion claims. Cover exactly these continuation
fields in concise plain text:
- Task
- Decisions
- Files changed
- Current test status
- Open problems
- Next steps
Do not call tools or emit a proposed action. Return only the continuation summary."""


def summary_messages(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Build the extra model call without adding any host-observed facts."""
    assert_tool_call_pairs(messages)
    prompt = [copy.deepcopy(dict(message)) for message in messages]
    prompt.append({"role": "user", "content": SUMMARY_INSTRUCTION})
    return prompt


def summary_request(messages: Sequence[Mapping[str, object]], *, model: str) -> dict[str, object]:
    """Wire body for the extra same-model, thinking-enabled, uncapped call."""
    effort = os.environ.get("REACT_COMPACTION_REASONING_EFFORT", "high").strip().lower()
    if effort not in {"high", "xhigh"}:
        raise ValueError("REACT_COMPACTION_REASONING_EFFORT must be high or xhigh")
    body: dict[str, object] = {
        "model": model,
        "messages": [copy.deepcopy(dict(message)) for message in messages],
        "temperature": 0,
        "reasoning_effort": effort,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if "REACT_SEED" in os.environ:
        body["seed"] = int(os.environ["REACT_SEED"])
    if os.environ.get("LONG_HORIZON_PROTOCOL", "off").strip().lower() in {
            "on", "1", "true", "yes"}:
        raw_budget = os.environ.get("LONG_HORIZON_SUMMARY_OUTPUT_TOKENS", "4096")
        try:
            summary_budget = int(raw_budget)
        except (TypeError, ValueError) as error:
            raise ValueError("LONG_HORIZON_SUMMARY_OUTPUT_TOKENS must be positive") from error
        if summary_budget <= 0:
            raise ValueError("LONG_HORIZON_SUMMARY_OUTPUT_TOKENS must be positive")
        body["max_tokens"] = summary_budget
        raw_slot = os.environ.get("REACT_SLOT_ID")
        if raw_slot is not None:
            try:
                slot = int(raw_slot)
            except ValueError as error:
                raise ValueError("REACT_SLOT_ID must be a nonnegative integer") from error
            if slot < 0:
                raise ValueError("REACT_SLOT_ID must be a nonnegative integer")
            body["id_slot"] = slot
        body["cache_prompt"] = os.environ.get(
            "STATEFRAME_PROMPT_CACHE", "off").strip().lower() in {
                "on", "1", "true", "yes"
            }
    return body


@dataclass(frozen=True)
class CompactionResult:
    messages: Sequence[Mapping[str, object]]
    applied: bool
    triggered: bool
    tokens_before: int
    tokens_after: int
    summary_tokens: int = 0
    turn_index: int = 0
    summary_request_id: str | None = None


def _summary_parts(value: object) -> tuple[str, int | None, str | None]:
    if isinstance(value, str):
        return value, None, None
    if isinstance(value, tuple) and value:
        text = value[0] if isinstance(value[0], str) else ""
        tokens = value[1] if len(value) > 1 and isinstance(value[1], int) else None
        request_id = value[2] if len(value) > 2 and isinstance(value[2], str) else None
        return text, tokens, request_id
    if isinstance(value, Mapping):
        text = value.get("text")
        if not isinstance(text, str):
            choice = value.get("choice")
            message = choice.get("message") if isinstance(choice, Mapping) else value.get("message")
            if isinstance(message, Mapping):
                text = message.get("content") or message.get("reasoning_content")
        text = text if isinstance(text, str) else ""
        tokens = next((value.get(key) for key in (
            "summary_tokens", "completion_tokens", "output_tokens")
                       if isinstance(value.get(key), int) and value.get(key) >= 0), None)
        request_id = next((value.get(key) for key in ("request_id", "summary_request_id")
                           if isinstance(value.get(key), str)), None)
        return text, tokens, request_id
    return "", None, None


class HistoryCompactor:
    """Run-scoped compaction policy and accounting boundary."""

    def __init__(self, *, enabled: bool | None = None,
                 context_tokens: object = None, threshold: object = None,
                 keep_turns: object = None,
                 summarize: Callable[[Sequence[Mapping[str, object]]], object] | None = None,
                 charge: Callable[[str, str], object] | None = None,
                 journal: object = None):
        self.enabled = globals()["enabled"]() if enabled is None else bool(enabled)
        self.context_tokens = configured_context_tokens(context_tokens)
        self.threshold = threshold_tokens(self.context_tokens, threshold)
        self.keep_turns = configured_keep_turns(keep_turns)
        self.summarize = summarize
        self.charge = charge
        self.journal = journal

    @classmethod
    def from_environment(cls, **kwargs) -> "HistoryCompactor":
        return cls(enabled=globals()["enabled"](), **kwargs)

    def _emit(self, kind: str, **fields: object) -> None:
        emit = getattr(self.journal, "emit", None)
        if callable(emit):
            emit(kind, **fields)
        elif callable(self.journal):
            self.journal(kind, **fields)

    def maybe_compact(self, messages: list[dict[str, object]], tools: object = None, *,
                      original_prefix: Sequence[Mapping[str, object]] | None = None,
                      turn_index: int = 0, force: bool = False) -> CompactionResult:
        before = estimate_prompt_tokens(messages, tools)
        unchanged = CompactionResult(messages, False, False, before, before,
                                     turn_index=turn_index)
        if not self.enabled or (before <= self.threshold and not force):
            return unchanged
        if self.summarize is None:
            return CompactionResult(messages, False, True, before, before,
                                    turn_index=turn_index)
        try:
            # Validate before charging.  A malformed pre-existing transcript is
            # a harness problem; compaction must never manufacture a repair.
            assert_tool_call_pairs(messages)
            turns = split_turns(messages)
            if len(turns) <= self.keep_turns:
                return CompactionResult(messages, False, True, before, before,
                                        turn_index=turn_index)
            if self.charge is not None:
                self.charge("model_calls", "history compaction summary")
            raw = self.summarize(summary_messages(messages))
            text, measured_tokens, request_id = _summary_parts(raw)
            if not text.strip():
                self._emit("history_compaction_failed", turn_index=turn_index,
                           tokens_before=before, reason="empty_summary")
                return unchanged
            compacted = compact_messages(
                messages, text, keep_turns=self.keep_turns,
                original_prefix=original_prefix)
            after = estimate_prompt_tokens(compacted, tools)
            summary_count = (measured_tokens if measured_tokens is not None
                             else estimate_prompt_tokens(
                                 [{"role": "assistant", "content": text}]))
            self._emit(
                "history_compaction",
                tokens_before=before,
                tokens_after=after,
                summary_tokens=summary_count,
                turn_index=turn_index,
                summary_request_id=request_id,
                kept_turns=self.keep_turns,
                threshold_tokens=self.threshold,
                context_tokens=self.context_tokens,
            )
            return CompactionResult(compacted, True, True, before, after,
                                    summary_tokens=summary_count,
                                    turn_index=turn_index,
                                    summary_request_id=request_id)
        except Exception as error:  # compaction is recoverable; preserve the old request
            self._emit("history_compaction_failed", turn_index=turn_index,
                       tokens_before=before, reason=type(error).__name__)
            return unchanged


__all__ = [
    "CompactionResult", "DEFAULT_CONTEXT_TOKENS", "HistoryCompactor",
    "HISTORY_SUMMARY_MARKER", "SUMMARY_INSTRUCTION", "assert_tool_call_pairs",
    "compact_messages", "configured_context_tokens", "configured_keep_turns",
    "enabled", "estimate_prompt_tokens", "split_turns", "summary_messages",
    "summary_request", "threshold_tokens",
]
