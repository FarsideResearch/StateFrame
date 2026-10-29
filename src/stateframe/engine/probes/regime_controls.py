"""Neutral context-furniture controls for ReAct regime experiments."""

from __future__ import annotations

import os


def approx_tokens(text: str) -> int:
    """Use the harness' documented ~4 characters/token approximation."""
    return len(text) // 4


_FURNITURE = (
    ("RULES DIRECTIVE", "Workspace boundaries are durable; retain the originating workspace when records are displayed."),
    ("RULES DIRECTIVE", "Display labels are presentation metadata; stable identifiers remain authoritative for records."),
    ("CONSTITUTION", "Durable state has an owner, a lifecycle, and an audit trail; each is independent of display order."),
    ("CONSTITUTION", "A neutral status is allowed when a record is incomplete; absence of a value is not a value."),
    ("MEMORY ENTRY", "The operations workspace uses explicit names for queues, collections, and saved views."),
    ("MEMORY ENTRY", "Historical entries remain attributable to their source workspace and retain their recorded timestamp."),
    ("APP CONTEXT", "This internal operations surface contains workspaces, queues, saved views, and activity records."),
    ("APP CONTEXT", "Navigation can expose the same record through several views without changing its durable identity."),
)


def render_neutral_preamble(target_tokens: int) -> str:
    """Render deterministic MC-like furniture with no task, target, or solution facts."""
    if target_tokens <= 0:
        return ""
    target_chars = target_tokens * 4
    paragraphs: list[str] = []
    index = 0
    while len("\n\n".join(paragraphs)) < target_chars:
        kind, text = _FURNITURE[index % len(_FURNITURE)]
        paragraphs.append(f"{kind} {index + 1:04d}: {text}")
        index += 1
    rendered = "\n\n".join(paragraphs)
    if len(rendered) > target_chars:
        rendered = rendered[:target_chars].rsplit(" ", 1)[0]
    return rendered


def pad_system_prompt(system_prompt: str, target_tokens: int) -> str:
    """Append preamble to the system message, the portion preserved by history trim."""
    preamble = render_neutral_preamble(target_tokens)
    if not preamble:
        return system_prompt
    return system_prompt + "\n\n[Mission Control context furniture]\n" + preamble


def nonnegative_env_int(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
