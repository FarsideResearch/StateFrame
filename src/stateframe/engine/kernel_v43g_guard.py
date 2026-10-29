"""V43G terminal-choice and degenerate-output guard.

The guard is deliberately host-side and bounded.  It never limits ordinary
reasoning; it only intervenes after a contiguous repeated span is provably
degenerate, and it records usage for every provider attempt.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import math
import os
from collections.abc import Mapping, Sequence


DEGENERATION_ENV = "STATEFRAME_DEGENERATION_GUARD"
TERMINAL_CHOICE_ENV = "STATEFRAME_TERMINAL_TOOL_CHOICE"
TERMINAL_FORMAT_ENV = "STATEFRAME_TERMINAL_TOOL_CHOICE_FORMAT"
SCHEMA = "stateframe-v43g-guard/1"
TRAILING_WINDOW_CHARS = 4096
MIN_REPEAT_SPAN_CHARS = 8
MIN_REPETITIONS = 20
MAX_REPEAT_SPAN_CHARS = 512
RETRY_TEMPERATURE = 0.6


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


def _best_run(text: str, *, source: str, trailing_window: int,
              min_repetitions: int, min_span: int,
              max_span: int) -> RepetitionFinding | None:
    if not text:
        return None
    window_start = max(0, len(text) - trailing_window)
    tail = text[window_start:]
    if len(tail) < min_span * min_repetitions:
        return None
    max_period = min(max_span, len(tail) // min_repetitions)
    earliest: tuple[int, int, int, int, str] | None = None
    # The observed loop alternates two long phrases, so a longer repeated span
    # is frequent without forming one contiguous period.  Shorter non-contiguous
    # n-grams are intentionally excluded: ordinary prose repeats 8-16 character
    # fragments and must not trip the guard.
    frequency_lengths = tuple(
        length for length in (24, 32, 48, 64, 96, 128)
        if min_span <= length <= max_period
    )
    for length in frequency_lengths:
        positions: dict[str, list[int]] = {}
        for start in range(len(tail) - length + 1):
            span = tail[start:start + length]
            if not span.strip() or len(set(span.replace(" ", ""))) < 3:
                continue
            positions.setdefault(span, []).append(start)
        for span, occurrences in positions.items():
            if len(occurrences) < min_repetitions:
                continue
            count = len(occurrences)
            candidate = (
                window_start + occurrences[min_repetitions - 1] + length,
                -(count * length),
                length,
                count,
                "sha256:" + hashlib.sha256(span.encode("utf-8")).hexdigest(),
            )
            if earliest is None or candidate < earliest:
                earliest = candidate
    for period in range(min_span, max_period + 1):
        span = tail[-period:]
        if not span.strip() or not any(not char.isspace() for char in span):
            continue
        repetitions = 1
        while repetitions * period <= len(tail):
            begin = len(tail) - (repetitions + 1) * period
            if begin < 0 or tail[begin:begin + period] != span:
                break
            repetitions += 1
        if repetitions < min_repetitions:
            continue
        repeated_chars = repetitions * period
        start = len(tail) - repeated_chars
        stop = window_start + start + min_repetitions * period
        candidate = (
            stop,
            -repeated_chars,
            period,
            repetitions,
            "sha256:" + hashlib.sha256(span.encode("utf-8")).hexdigest(),
        )
        if earliest is None or candidate < earliest:
            earliest = candidate
    if earliest is None:
        return None
    stop, neg_repeated, period, repetitions, span_hash = earliest
    return RepetitionFinding(
        source=source,
        total_chars=len(text),
        stop_chars=stop,
        trailing_window_chars=len(tail),
        repeated_span_chars=period,
        repetition_count=repetitions,
        repeated_chars=-neg_repeated,
        span_hash=span_hash,
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


def _message_texts(response: Mapping[str, object]) -> tuple[str, str]:
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes, bytearray)):
        return "", ""
    choice = choices[0] if choices else None
    if not isinstance(choice, Mapping):
        return "", ""
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return "", ""

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

    return text_value(message.get("reasoning_content")), text_value(message.get("content"))


def detect_response(response: Mapping[str, object]) -> RepetitionFinding | None:
    """Inspect reasoning and content without retaining or returning their text."""
    if not isinstance(response, Mapping):
        return None
    reasoning, content = _message_texts(response)
    candidates = [
        finding for finding in (
            detect_text(reasoning, source="reasoning"),
            detect_text(content, source="content"),
            detect_text(reasoning + content, source="reasoning+content"),
        ) if finding is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda finding: (finding.stop_chars, -finding.repeated_chars))


class StreamingRepetitionGuard:
    """Incremental guard for providers that expose SSE/delta chunks."""

    def __init__(self, *, trailing_window: int = TRAILING_WINDOW_CHARS) -> None:
        self.trailing_window = trailing_window
        self._reasoning = ""
        self._content = ""
        self._total_chars = 0
        self.finding: RepetitionFinding | None = None

    def feed(self, *, reasoning: str = "", content: str = "") -> RepetitionFinding | None:
        if not isinstance(reasoning, str) or not isinstance(content, str):
            raise TypeError("stream chunks must be text")
        self._total_chars += len(reasoning) + len(content)
        self._reasoning = (self._reasoning + reasoning)[-self.trailing_window:]
        self._content = (self._content + content)[-self.trailing_window:]
        if self.finding is not None:
            return self.finding
        finding = detect_response({
            "choices": [{"message": {
                "role": "assistant", "reasoning_content": self._reasoning,
                "content": self._content,
            }}],
        })
        if finding is None:
            return None
        retained_chars = len(self._reasoning) + len(self._content)
        prefix_chars = max(self._total_chars - retained_chars, 0)
        self.finding = replace(
            finding, total_chars=self._total_chars,
            stop_chars=prefix_chars + finding.stop_chars,
        )
        return self.finding


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
    "RepetitionFinding", "SCHEMA", "StreamingRepetitionGuard",
    "TERMINAL_CHOICE_ENV", "TERMINAL_FORMAT_ENV", "detect_response", "detect_text",
    "degeneration_mode", "forced_done_tool_choice", "response_journal_fields",
    "retry_body", "terminal_choice_format", "terminal_choice_mode", "usage_counts",
]
