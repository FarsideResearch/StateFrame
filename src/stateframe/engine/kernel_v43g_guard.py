"""V43G terminal-choice and degenerate-output guard.

The guard is deliberately host-side and bounded. It retries only a response
that hit its token cap and ended in a repeated suffix, and it records usage for
every provider attempt.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import os
from collections.abc import Mapping, Sequence


DEGENERATION_ENV = "STATEFRAME_DEGENERATION_GUARD"
TERMINAL_CHOICE_ENV = "STATEFRAME_TERMINAL_TOOL_CHOICE"
TERMINAL_FORMAT_ENV = "STATEFRAME_TERMINAL_TOOL_CHOICE_FORMAT"
SCHEMA = "stateframe-v43g-guard/1"
TRAILING_WINDOW_CHARS = 131072
MIN_REPEAT_SPAN_CHARS = 8
MIN_REPETITIONS = 20
MAX_REPEAT_SPAN_CHARS = 6144
RETRY_TEMPERATURE = 0.6
TOKEN_CAP_REASONS = ("length", "max_tokens", "max_output_tokens")


def _mode(name: str, environ: Mapping[str, str] | None, default: str = "off") -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(name, default) or default).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{name} must be off or on")
    return value


def degeneration_mode(environ: Mapping[str, str] | None = None) -> str:
    return _mode(DEGENERATION_ENV, environ)


def terminal_choice_mode(environ: Mapping[str, str] | None = None) -> str:
    return _mode(TERMINAL_CHOICE_ENV, environ)


def terminal_choice_format(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(TERMINAL_FORMAT_ENV, "required") or "required").strip().lower()
    if value not in {"required", "structured_function"}:
        raise ValueError(f"{TERMINAL_FORMAT_ENV} must be required or structured_function")
    return value


@dataclass(frozen=True)
class RepetitionFinding:
    """Safe, text-free evidence that a response tail is degenerate."""

    source: str
    total_chars: int
    stop_chars: int
    trailing_window_chars: int
    repeated_span_chars: int
    repetition_count: int
    repeated_chars: int
    span_hash: str

    @property
    def saved_chars(self) -> int:
        return max(self.total_chars - self.stop_chars, 0)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "source": self.source,
            "total_chars": self.total_chars,
            "stop_chars": self.stop_chars,
            "saved_chars": self.saved_chars,
            "trailing_window_chars": self.trailing_window_chars,
            "repeated_span_chars": self.repeated_span_chars,
            "repetition_count": self.repetition_count,
            "repeated_chars": self.repeated_chars,
            "span_hash": self.span_hash,
        }


def _normalize_whitespace(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            if not normalized or normalized[-1] != " ":
                normalized.append(" ")
                offsets.append(index)
        else:
            normalized.append(char)
            offsets.append(index)
    offsets.append(len(text))
    return "".join(normalized), offsets


def _z_values(text: str) -> list[int]:
    """Return prefix-match lengths for each suffix in linear work."""
    values = [0] * len(text)
    left = right = 0
    for index in range(1, len(text)):
        if index < right:
            values[index] = min(right - index, values[index - left])
        while (index + values[index] < len(text)
               and text[values[index]] == text[index + values[index]]):
            values[index] += 1
        if index + values[index] > right:
            left, right = index, index + values[index]
    return values


def _best_run(text: str, *, source: str, trailing_window: int,
              min_repetitions: int, min_span: int,
              max_span: int) -> RepetitionFinding | None:
    if not text:
        return None
    window_start = max(0, len(text) - trailing_window)
    tail = text[window_start:]
    normalized, offsets = _normalize_whitespace(tail)
    if len(normalized) < (min_repetitions - 1) * min_span + 1:
        return None
    if not normalized.strip():
        return None
    matches = _z_values(normalized[::-1])
    max_period = min(max_span, (len(normalized) - 1) // (min_repetitions - 1))
    best: tuple[int, int, int, int, int, int] | None = None
    for period in range(min_span, max_period + 1):
        repeated_length = period + matches[period]
        repetitions, partial = divmod(repeated_length, period)
        if repetitions < min_repetitions and not (
                repetitions == min_repetitions - 1 and partial > 0):
            continue
        start = len(normalized) - repeated_length
        if not normalized[start:start + period].strip():
            continue
        stop = window_start + offsets[min(len(normalized), start + min_repetitions * period)]
        loop_start = window_start + offsets[start]
        repeated_chars = len(text) - loop_start
        span_chars = offsets[start + period] - offsets[start]
        candidate = (stop, -repeated_chars, span_chars, repetitions, start, period)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    stop, neg_repeated, span_chars, repetitions, start, period = best
    span = normalized[start:start + period]
    return RepetitionFinding(
        source=source,
        total_chars=len(text),
        stop_chars=stop,
        trailing_window_chars=len(tail),
        repeated_span_chars=span_chars,
        repetition_count=repetitions,
        repeated_chars=-neg_repeated,
        span_hash="sha256:" + hashlib.sha256(span.encode("utf-8", "surrogatepass")).hexdigest(),
    )


def detect_text(text: object, *, source: str = "text",
                trailing_window: int = TRAILING_WINDOW_CHARS,
                min_repetitions: int = MIN_REPETITIONS,
                min_span: int = MIN_REPEAT_SPAN_CHARS) -> RepetitionFinding | None:
    """Detect a contiguous repeated span in a bounded trailing window."""
    if not isinstance(text, str):
        return None
    if trailing_window < min_span * min_repetitions:
        raise ValueError("trailing_window is too small for the repetition threshold")
    return _best_run(
        text, source=source, trailing_window=trailing_window,
        min_repetitions=min_repetitions, min_span=min_span,
        max_span=min(MAX_REPEAT_SPAN_CHARS, trailing_window // min_repetitions),
    )


def _message_texts(response: Mapping[str, object]) -> tuple[str, str, tuple[str, ...]]:
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes, bytearray)):
        return "", "", ()
    choice = choices[0] if choices else None
    if not isinstance(choice, Mapping):
        return "", "", ()
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return "", "", ()

    def text_value(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return "".join(
                str(block.get("text", ""))
                for block in value
                if isinstance(block, Mapping) and isinstance(block.get("text"), str)
            )
        return ""

    tool_arguments: list[str] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes, bytearray)):
        for call in tool_calls:
            function = call.get("function") if isinstance(call, Mapping) else None
            arguments = function.get("arguments") if isinstance(function, Mapping) else None
            if isinstance(arguments, str):
                tool_arguments.append(arguments)
    legacy_call = message.get("function_call")
    legacy_arguments = legacy_call.get("arguments") if isinstance(legacy_call, Mapping) else None
    if isinstance(legacy_arguments, str):
        tool_arguments.append(legacy_arguments)

    return (
        text_value(message.get("reasoning_content")),
        text_value(message.get("content")),
        tuple(tool_arguments),
    )


def detect_response(response: Mapping[str, object]) -> RepetitionFinding | None:
    """Find a repeated suffix only on a token-capped response."""
    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    choice = (choices[0] if isinstance(choices, Sequence)
              and not isinstance(choices, (str, bytes, bytearray)) and choices else None)
    if not isinstance(choice, Mapping) or choice.get("finish_reason") not in TOKEN_CAP_REASONS:
        return None
    reasoning, content, tool_arguments = _message_texts(response)
    candidates = [
        finding for finding in (
            detect_text(reasoning, source="reasoning"),
            detect_text(content, source="content"),
            detect_text(reasoning + content, source="reasoning+content"),
            *(detect_text(argument, source="tool_arguments") for argument in tool_arguments),
        ) if finding is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda finding: (finding.stop_chars, -finding.repeated_chars))


def usage_counts(response: Mapping[str, object]) -> dict[str, int | None]:
    usage = response.get("usage") if isinstance(response, Mapping) else None
    usage = usage if isinstance(usage, Mapping) else {}

    def count(name: str) -> int | None:
        value = usage.get(name)
        return value if type(value) is int and value >= 0 else None

    prompt = count("prompt_tokens")
    completion = count("completion_tokens")
    total = count("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def retry_body(body: Mapping[str, object]) -> dict[str, object]:
    """Copy a request and lower sampling temperature without touching its budget."""
    result = copy.deepcopy(dict(body))
    current = result.get("temperature", RETRY_TEMPERATURE)
    temperature = float(current) if isinstance(current, (int, float)) and not isinstance(current, bool) else RETRY_TEMPERATURE
    if not math.isfinite(temperature) or temperature < 0:
        temperature = RETRY_TEMPERATURE
    result["temperature"] = min(temperature, RETRY_TEMPERATURE)
    return result


def forced_done_tool_choice(tools: object, *, environ: Mapping[str, str] | None = None) -> str | dict[str, object] | None:
    """Return an exact function choice when supported, else llama.cpp-safe required."""
    if terminal_choice_mode(environ) != "on":
        return None
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return None
    names = [
        str(tool.get("function", {}).get("name", ""))
        for tool in tools
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
    ]
    if names != ["done"]:
        return None
    if terminal_choice_format(environ) == "structured_function":
        return {"type": "function", "function": {"name": "done"}}
    return "required"


def response_journal_fields(finding: RepetitionFinding, response: Mapping[str, object], *,
                            retry_temperature: float, logical_prompt_tokens: int,
                            logical_completion_tokens: int) -> dict[str, object]:
    counts = usage_counts(response)
    return {
        "schema": SCHEMA,
        "counts": counts,
        "logical_prompt_tokens": logical_prompt_tokens,
        "logical_completion_tokens": logical_completion_tokens,
        "logical_total_tokens": logical_prompt_tokens + logical_completion_tokens,
        "finish_reason": (
            response.get("choices", [{}])[0].get("finish_reason")
            if isinstance(response.get("choices"), list)
            and response.get("choices") and isinstance(response["choices"][0], Mapping)
            else None
        ),
        "retry_temperature": retry_temperature,
        **finding.as_dict(),
    }


__all__ = [
    "DEGENERATION_ENV", "MIN_REPEAT_SPAN_CHARS", "MIN_REPETITIONS",
    "RepetitionFinding", "SCHEMA",
    "TERMINAL_CHOICE_ENV", "TERMINAL_FORMAT_ENV", "detect_response", "detect_text",
    "degeneration_mode", "forced_done_tool_choice", "response_journal_fields",
    "retry_body", "terminal_choice_format", "terminal_choice_mode", "usage_counts",
]
