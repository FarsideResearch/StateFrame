"""Small opt-in metadata helpers for the long-horizon live request path."""
from __future__ import annotations

import os
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def enabled() -> bool:
    return os.environ.get("LONG_HORIZON_TELEMETRY", "off").strip().lower() in {
        "on", "1", "true", "yes",
    }


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value is not None and value.isdigit() else None


CUMULATIVE_DEPTH_SOURCE = "appended_messages_chars_div_4"


def _message_key(message: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(message), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _message_tokens(message: Mapping[str, Any]) -> int:
    """Use the lane's serialized-message chars/4 estimate for one append."""
    from kernel_context import token_estimate
    return token_estimate(dict(message))


class AppendOnlyLogicalDepth:
    """Count transcript appends without treating compaction as a reset.

    The live loop passes copied prompt prefixes on every request.  A sequence
    suffix therefore distinguishes a genuinely appended duplicate message from
    a repeated prefix, while ``rebase`` marks a compacted rendering as derived
    rather than charging its summary a second time.
    """

    source = CUMULATIVE_DEPTH_SOURCE

    def __init__(self) -> None:
        self.tokens = 0
        self._last_sequence: tuple[str, ...] = ()

    @property
    def cumulative_tokens(self) -> int:
        return self.tokens

    def observe(self, messages: Sequence[Mapping[str, Any]]) -> int:
        """Charge only the suffix appended to the prior live transcript."""
        material = [message for message in messages if isinstance(message, Mapping)]
        sequence = tuple(_message_key(message) for message in material)
        if (len(self._last_sequence) <= len(sequence)
                and sequence[:len(self._last_sequence)] == self._last_sequence):
            start = len(self._last_sequence)
        else:
            # A caller that replaces a view without first calling ``rebase`` is
            # still treated conservatively: preserve the common prefix and do
            # not charge repeated retained messages as new prompt prefixes.
            start = 0
            while (start < len(self._last_sequence) and start < len(sequence)
                   and self._last_sequence[start] == sequence[start]):
                start += 1
        for message in material[start:]:
            self.tokens += _message_tokens(message)
        self._last_sequence = sequence
        return self.tokens

    def rebase(self, messages: Sequence[Mapping[str, Any]]) -> int:
        """Mark a derived compacted view without charging its replacement summary."""
        self._last_sequence = tuple(
            _message_key(message) for message in messages if isinstance(message, Mapping))
        return self.tokens

    def add_message(self, message: Mapping[str, Any]) -> int:
        """Charge one logical append that is not represented by the live list yet."""
        item = dict(message)
        self.tokens += _message_tokens(item)
        return self.tokens


def request_metadata(
    capture: object,
    messages: Sequence[Mapping[str, Any]],
    tools: object,
    *,
    rendered_prompt_tokens: int,
    logical_messages: Sequence[Mapping[str, Any]] | None = None,
    logical_tools: object = None,
) -> dict[str, Any]:
    """Describe the final prepared request without changing its payload."""
    if type(rendered_prompt_tokens) is not int or rendered_prompt_tokens < 0:
        raise ValueError("rendered_prompt_tokens must be a non-negative integer")
    from kernel_context import token_estimate

    logical = logical_messages if logical_messages is not None else messages
    logical_depth = token_estimate({
        "messages": list(logical),
        "tools": tools if logical_tools is None else logical_tools,
    })
    tracker = getattr(capture, "long_horizon_depth_tracker", None)
    cumulative = getattr(capture, "long_horizon_cumulative_logical_depth_tokens", None)
    cumulative_source = getattr(
        capture, "long_horizon_cumulative_logical_depth_source", None)
    if isinstance(tracker, AppendOnlyLogicalDepth):
        cumulative = tracker.cumulative_tokens
        cumulative_source = tracker.source
    presentation = getattr(capture, "long_horizon_presentation", None)
    direct_presentation = getattr(capture, "presentation", None)
    if (not isinstance(presentation, str) or
            (presentation == "history" and direct_presentation not in {None, "history"})):
        presentation = direct_presentation or "history"
    history_summary_ordinal = max(
        int(getattr(capture, "long_horizon_history_summary_ordinal", 0) or 0),
        int(getattr(capture, "history_summary_ordinal", 0) or 0),
    )
    frame_epoch_ordinal = max(
        int(getattr(capture, "long_horizon_frame_epoch_ordinal", 0) or 0),
        int(getattr(capture, "frame_epoch_ordinal", 0) or 0),
    )
    return {
        "request_id": getattr(capture, "request_id", None),
        "task_id": os.environ.get("LONG_HORIZON_TASK_ID"),
        "arm": os.environ.get("LONG_HORIZON_ARM"),
        "seed": _env_int("REACT_SEED"),
        "request_role": getattr(capture, "long_horizon_request_role", "solver"),
        "rendered_prompt_tokens": rendered_prompt_tokens,
        "rendered_prompt_tokens_source": "runtime_token_estimate",
        "depth_source": "runtime_token_estimate",
        "logical_depth_tokens": logical_depth,
        # Legacy compatibility: this is now the sum of the two distinct
        # run-scoped mechanisms, not a single always-zero counter.
        "compaction_ordinal": history_summary_ordinal + frame_epoch_ordinal,
        "history_summary_ordinal": history_summary_ordinal,
        "frame_epoch_ordinal": frame_epoch_ordinal,
        "hybrid_crossed": bool(
            getattr(capture, "long_horizon_hybrid_crossed", False)
            or getattr(capture, "hybrid_crossed", False)),
        "hybrid_crossing_request_id": (
            getattr(capture, "long_horizon_hybrid_crossing_request_id", None)
            or getattr(capture, "hybrid_crossing_request_id", None)),
        "hybrid_crossing": bool(
            getattr(capture, "long_horizon_hybrid_crossing", False)
            or getattr(capture, "hybrid_crossing", False)),
        "phase": int(max(
            int(getattr(capture, "long_horizon_phase", 1) or 1),
            int(getattr(capture, "phase", 1) or 1))),
        "presentation": presentation,
        "cumulative_logical_depth_tokens": (
            cumulative if cumulative is not None else getattr(
                capture, "cumulative_logical_depth_tokens", None)),
        "cumulative_logical_depth_source": (
            cumulative_source if cumulative_source is not None else getattr(
                capture, "cumulative_logical_depth_source", None)),
        "usable_input_tokens": _env_int("LONG_HORIZON_USABLE_INPUT_TOKENS"),
        "context_window_tokens": _env_int("LONG_HORIZON_CONTEXT_LIMIT_TOKENS"),
    }


__all__ = [
    "AppendOnlyLogicalDepth", "CUMULATIVE_DEPTH_SOURCE", "enabled", "request_metadata",
]
