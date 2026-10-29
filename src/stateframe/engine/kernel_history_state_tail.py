"""An optional, replaceable current-state message for history requests.

The module owns only tail placement and accounting.  Callers provide blocks
already rendered by the normal frame and receipt renderers.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from collections.abc import Mapping, Sequence

from kernel_context import token_estimate
from kernel_measurement import digest


MODE_ENV = "STATEFRAME_HISTORY_STATE_TAIL"
SCHEMA = "history-state-tail/1"
STATE_TAIL_HEADER = "[StateFrame Current State]"


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strictly validated switch value."""
    source = os.environ if environ is None else environ
    value = source.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def is_state_tail(message: Mapping[str, object]) -> bool:
    """Recognize only messages emitted by this module."""
    return (
        message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].startswith(STATE_TAIL_HEADER + "\n")
    )


def strip_state_tail(messages: Sequence[Mapping[str, object]]) -> list[dict]:
    """Copy a history while removing any previously emitted state tail."""
    return [copy.deepcopy(dict(message)) for message in messages if not is_state_tail(message)]


def _block_text(block: object) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, Mapping) and isinstance(block.get("content"), str):
        return block["content"]
    raise TypeError("state-tail blocks must be strings or content messages")


def render_state_tail(blocks: Sequence[object]) -> str:
    """Put already-rendered frame blocks into the one state-tail message."""
    rendered = []
    for block in blocks:
        text = _block_text(block)
        if text:
            rendered.append(text)
    return STATE_TAIL_HEADER + ("\n\n" + "\n\n".join(rendered) if rendered else "")


@dataclasses.dataclass(frozen=True)
class TailResult:
    messages: tuple[dict, ...]
    report: dict[str, object]


def _unchanged_prefix(before: Sequence[Mapping[str, object]], after: Sequence[Mapping[str, object]]) -> bool:
    return list(before) == list(after)


def replace_state_tail(
    messages: Sequence[Mapping[str, object]],
    blocks: Sequence[object],
    *,
    mode: str = "off",
    provider_request: bool = True,
) -> TailResult:
    """Replace one prior tail, or return the original message sequence.

    The disabled and non-provider paths deliberately do not add a message.
    Every non-tail message is copied without normalization or rewriting.
    """
    if mode not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")

    original = [copy.deepcopy(dict(message)) for message in messages]
    prefix = strip_state_tail(original)
    replaced_count = len(original) - len(prefix)
    active = mode == "on" and provider_request
    if not active:
        return TailResult(
            tuple(original),
            {
                "schema": SCHEMA,
                "prefix_hash": digest(original),
                "tail_hash": None,
                "prefix_message_count": len(original),
                "tail_tokens": 0,
                "tail_chars": 0,
                "replaced_tail_count": 0,
                "prefix_unchanged": True,
            },
        )

    content = render_state_tail(blocks)
    tail = {"role": "user", "content": content}
    result = tuple(prefix + [tail])
    report = {
        "schema": SCHEMA,
        "prefix_hash": digest(prefix),
        "tail_hash": digest(tail),
        "prefix_message_count": len(prefix),
        "tail_tokens": token_estimate(tail),
        "tail_chars": len(content),
        "replaced_tail_count": replaced_count,
        "prefix_unchanged": _unchanged_prefix(prefix, result[:-1]),
    }
    return TailResult(result, report)


__all__ = [
    "MODE_ENV",
    "SCHEMA",
    "STATE_TAIL_HEADER",
    "TailResult",
    "enabled",
    "is_state_tail",
    "mode",
    "render_state_tail",
    "replace_state_tail",
    "strip_state_tail",
]
