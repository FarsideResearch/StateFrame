"""Stable, task-independent harness literacy prefix for StateFrame requests."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_HARNESS_CARD"
SCHEMA = "stateframe-harness-card/1"

# Keep the harness capability description consistent with its actual tools.
HARNESS_CARD = (
    "StateFrame is your current working state. Requirements define success; current host receipts "
    "establish only their recorded observations. A receipt binds an action/result to a revision; "
    "edits can stale it. Model notes are hypotheses; advisories neither grant permission nor prove failure.\n"
    "Obligations are requirements awaiting evidence, not necessarily bugs. Use their stated evidence "
    "action; do not invent missing meanings.\n"
    "A witness is a proposed input and expected result for a named requirement. Submit through the offered "
    "typed probe; the host runs it. Derive expectations from requirements, not observed output.\n"
    "Public PASS, done permission, acceptance, and submission are separate. Call done to request completion; "
    "inspect any refusal. Use its available clearing action. Repair demonstrated failures; do not edit merely "
    "to satisfy uncertainty. Retry done only when explicitly offered. Uncertified submission preserves open "
    "evidence gaps; it does not certify correctness."
)


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def system_prefix(content: str) -> str:
    """Return one stable card followed by the existing generic system content."""
    text = str(content or "")
    if text.startswith(HARNESS_CARD):
        return text
    return HARNESS_CARD + ("\n" + text if text else "")


def ensure_system_prefix(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Insert the card before task-specific content without mutating ``messages``."""
    projected = [dict(message) for message in messages]
    if not enabled():
        return projected
    for index, message in enumerate(projected):
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = system_prefix(content)
                return projected
            # Preserve structured/multimodal system content byte-for-byte;
            # put the stable text card in its own preceding system message.
            return [{"role": "system", "content": HARNESS_CARD}, *projected]
    return [{"role": "system", "content": HARNESS_CARD}, *projected]


def card_token_delta(before: str) -> int:
    """Use the local byte/4 accounting used by the request fitting prompt budget."""
    return (len(HARNESS_CARD.encode("utf-8")) + 3) // 4 - ((len(str(before).encode("utf-8")) + 3) // 4)


__all__ = ["HARNESS_CARD", "MODE_ENV", "SCHEMA", "card_token_delta", "enabled", "ensure_system_prefix", "mode", "system_prefix"]
