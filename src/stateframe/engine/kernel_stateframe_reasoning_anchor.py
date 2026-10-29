"""Keep a stable reasoning anchor across task-state updates.

The anchor retains the current problem and supporting evidence so unrelated
observations cannot silently replace the active objective.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

from kernel_context import token_estimate


MODE_ENV = "STATEFRAME_REASONING_ANCHOR"
SCHEMA = "stateframe-reasoning-anchor/1"
ANCHOR_HEADER = f"[Settled findings | this run | schema={SCHEMA}]"
MAX_ENTRIES = 4
MAX_BLOCK_TOKENS = 160

# Ordered so the most specific marker wins when a result string mentions more than one
# (e.g. "ALREADY_PRESENT already_in_frame_as IN_FRAME" must resolve to ALREADY_PRESENT,
# not the generic IN_FRAME binding note).
_OUTCOME_MARKERS: tuple[tuple[str, str], ...] = (
    ("ALREADY_PRESENT", "ALREADY_PRESENT"),
    ("REUSED_VERIFICATION", "REUSED_VERIFICATION"),
    ("READ_UNCHANGED", "READ_UNCHANGED"),
)


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict opt-in switch value."""
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _target(call: Mapping[str, object]) -> str | None:
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping):
        return None
    raw = function.get("arguments")
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, Mapping):
        return None
    for key in ("path", "selector", "file", "target"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _tool_name(call: Mapping[str, object]) -> str | None:
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else None


def extract_conclusion(call: Mapping[str, object], result: object) -> dict[str, str] | None:
    """Return one typed {"tool", "target", "outcome"} row, or None if nothing recognized.

    Only literal, known markers are matched -- never the model's own prose. ``result`` is the
    tool-result message (a Mapping with "content") or a plain string content.
    """
    tool = _tool_name(call)
    target = _target(call)
    if not tool or not target:
        return None
    content = result.get("content") if isinstance(result, Mapping) else result
    if not isinstance(content, str):
        return None
    for marker, tag in _OUTCOME_MARKERS:
        if marker in content:
            return {"tool": tool, "target": target, "outcome": tag}
    return None


class AnchorRing:
    """Bounded, deduplicated ring of settled conclusions for one run."""

    def __init__(self) -> None:
        self._entries: list[dict[str, str]] = []

    def observe_pair(self, pair: Sequence[Mapping[str, object]]) -> bool:
        """Offer one evicted (call, result) tool exchange pair. Returns True if kept."""
        if not isinstance(pair, Sequence) or len(pair) != 2:
            return False
        assistant, result = pair
        if not isinstance(assistant, Mapping) or not isinstance(result, Mapping):
            return False
        calls = assistant.get("tool_calls")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
            return False
        kept = False
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            entry = extract_conclusion(call, result)
            if entry is None:
                continue
            self._append(entry)
            kept = True
        return kept

    def _append(self, entry: dict[str, str]) -> None:
        self._entries = [row for row in self._entries
                         if not (row["target"] == entry["target"] and row["tool"] == entry["tool"])]
        self._entries.append(entry)
        self._entries = self._entries[-MAX_ENTRIES:]

    def entries(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(entry) for entry in self._entries)


def validate_entries(entries: Sequence[Mapping[str, object]]) -> None:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        raise ValueError("reasoning anchor entries must be a list")
    if not entries:
        raise ValueError("reasoning anchor entries must not be empty")
    if len(entries) > MAX_ENTRIES:
        raise ValueError(f"reasoning anchor entries exceed MAX_ENTRIES={MAX_ENTRIES}")
    required = frozenset({"tool", "target", "outcome"})
    known_outcomes = frozenset(tag for _marker, tag in _OUTCOME_MARKERS)
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("reasoning anchor entry must be a mapping")
        if set(entry) != required:
            raise ValueError("reasoning anchor entry must have exactly tool/target/outcome")
        for key in required:
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise ValueError(f"reasoning anchor entry.{key} must be a non-empty string")
        if entry["outcome"] not in known_outcomes:
            raise ValueError("reasoning anchor entry.outcome is not a known marker")


def render_block(entries: Sequence[Mapping[str, object]]) -> str:
    """Render the typed ring as one compact, deterministic, auditable block."""
    validate_entries(entries)
    body = json.dumps(list(entries), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    block = ANCHOR_HEADER + "\n" + body
    estimate = token_estimate(list(entries))
    if estimate > MAX_BLOCK_TOKENS:
        raise ValueError(f"reasoning anchor block is {estimate} tokens; budget is {MAX_BLOCK_TOKENS}")
    return block


__all__ = [
    "ANCHOR_HEADER", "MAX_ENTRIES", "MAX_BLOCK_TOKENS", "MODE_ENV", "SCHEMA",
    "AnchorRing", "enabled", "extract_conclusion", "mode", "render_block", "validate_entries",
]
