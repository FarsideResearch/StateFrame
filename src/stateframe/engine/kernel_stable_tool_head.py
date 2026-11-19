"""Stable tool head (STATEFRAME_STABLE_TOOL_HEAD): one append-only tool list on the wire for a whole run.

Why: llama.cpp reuses the longest common token prefix between consecutive requests on a slot, and the chat
template renders the tool list before the system prompt. When the per-phase menu changes membership or order,
nothing after the tool list can be reused, so every turn re-reads the whole prompt.

With the switch on, which is the default when the variable is unset, the final tool surface separates two
things the engine used to keep identical:

* the LISTED tools (what the provider sees): the current callable menu followed on request 1 by the run's
  registered offerable schemas, with seed-only tools in name order. Later unseen tools append in first-seen
  order. A parameter change replaces a schema in place; description-only changes keep the first schema.
* the CALLABLE tools (what may execute this turn): exactly the menu the engine authored, unchanged.

Dispatch keeps using the callable names (the request contract's ``advertised_actions``), so a listed but gated
call is refused with the existing "not offered for the current state; choose one of: ..." reply, and a short note
in the volatile tail names the callable tools. Set the variable to ``off`` to keep the per-step tool list.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence

from kernel_request_tool_surface import CONSTANT_MENU_ATTR
from kernel_tokenlab_cache_stable import ToolLayout
from kernel_wire_stability import VOLATILE_TAIL_MARKER

MODE_ENV = "STATEFRAME_STABLE_TOOL_HEAD"
CONSTANT_MENU_ENV = "STATEFRAME_CONSTANT_TOOL_MENU"
EVENT = "stable_tool_head_applied"
NOTE_HEADER = "[Tool menu | this turn]"
GUIDANCE_NOTE_HEADER = "[Tool guidance | this turn]"
_STATE_ATTR = "_stable_tool_head_layout"
_OFFERABLE_ATTR = "_stable_tool_head_offerable_tools"
_MODES = frozenset({"off", "on"})


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "on")).strip().lower()
    if value not in _MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def constant_menu_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(CONSTANT_MENU_ENV, "off")).strip().lower()
    if value not in _MODES:
        raise ValueError(f"{CONSTANT_MENU_ENV} must be off or on")
    return value


def constant_menu_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return constant_menu_mode(environ) == "on"


def constant_menu_active(runtime: object, *, minimal_launch: bool = False) -> bool:
    return constant_menu_enabled() and enabled() and (
        minimal_launch or getattr(runtime, CONSTANT_MENU_ATTR, None) is not None
    )


def _name(tool: object) -> str:
    if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping):
        name = tool["function"].get("name")
        return name if isinstance(name, str) else ""
    return ""


def remember_offerable_tools(runtime: object, tools: object) -> None:
    """Capture the first already-filtered request menu before state projections narrow it."""
    if not enabled() or getattr(runtime, _OFFERABLE_ATTR, None) is not None:
        return
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return
    captured = [copy.deepcopy(dict(tool)) for tool in tools
                if isinstance(tool, Mapping) and _name(tool)]
    if captured:
        setattr(runtime, _OFFERABLE_ATTR, captured)


def run_offerable_tools(runtime: object) -> list[dict]:
    """Combine per-run admitted surfaces, preferring the completion owner's schemas."""
    merged: list[dict] = []
    seen: set[str] = set()
    for attr in ("_completion_owner_full_tools", "_readiness_full_tools",
                 _OFFERABLE_ATTR, "available_tools"):
        tools = getattr(runtime, attr, ())
        if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
            continue
        for tool in tools:
            name = _name(tool)
            if not name or name in seen or not isinstance(tool, Mapping):
                continue
            seen.add(name)
            merged.append(copy.deepcopy(dict(tool)))
    return merged


def listed_tools(
        runtime: object,
        callable_tools: Sequence[Mapping[str, object]],
        offerable_tools: Sequence[Mapping[str, object]] = (),
) -> list[dict]:
    """The run's listed tools after observing this request's callable menu.

    Idempotent within a request: observing the same callable menu again leaves the registry unchanged, so
    every pass that authors the surface for one request (fit, re-plan, seal) produces the same bytes. The
    first call also seeds run-level offerable tools after the current menu; later unseen tools remain append-only.
    """
    registry = getattr(runtime, _STATE_ATTR, None)
    if not isinstance(registry, ToolLayout):
        registry = ToolLayout()
        setattr(runtime, _STATE_ATTR, registry)
    registry.observe(callable_tools)
    registry.seed(offerable_tools)
    return registry.tools()


def constant_menu_tools(runtime: object, offerable_tools: Sequence[Mapping[str, object]]) -> list[dict]:
    """Keep first-seen schemas byte-stable, including parameter changes in later phases."""
    snapshot = getattr(runtime, CONSTANT_MENU_ATTR, None)
    if snapshot is None:
        snapshot = [copy.deepcopy(dict(tool)) for tool in offerable_tools
                    if isinstance(tool, Mapping) and _name(tool)]
        setattr(runtime, CONSTANT_MENU_ATTR, snapshot)
    if not isinstance(snapshot, list):
        return []
    return copy.deepcopy(snapshot)


def callable_subset(listed: Sequence[object], names: Sequence[str]) -> list[dict]:
    """Listed schemas restricted to the callable names, in listed order (the dispatch authority)."""
    allowed = set(names)
    return [copy.deepcopy(dict(tool)) for tool in listed
            if isinstance(tool, Mapping) and _name(tool) in allowed]


def apply_menu_note(
        body: dict, listed: Sequence[object], names: Sequence[str], *,
        guidance: object = None, volatile_tail: bool = False,
) -> None:
    """Keep current menu notes at the tail (before a continuation prefill, if any).

    Re-authoring the surface for the same request replaces the note instead of stacking a second one.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    headers = (NOTE_HEADER, GUIDANCE_NOTE_HEADER) if volatile_tail else (NOTE_HEADER,)
    kept = [message for message in messages
            if not (isinstance(message, Mapping) and message.get("role") == "user"
                    and str(message.get("content", "")).startswith(headers))]
    notes = []
    menu = menu_note(listed, names)
    if menu is not None:
        notes.append(menu)
    if isinstance(guidance, str) and guidance.strip():
        notes.append(f"{GUIDANCE_NOTE_HEADER} {guidance.strip()}")
    if volatile_tail:
        for index, message in enumerate(kept):
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.startswith(VOLATILE_TAIL_MARKER):
                continue
            # Keep the notes inside the tail so complete-envelope fitting retains them.
            lines = [line for line in content.split("\n") if not line.startswith(headers)]
            lines[1:1] = [" ".join(note.splitlines()) for note in notes]
            kept[index] = {**message, "content": "\n".join(lines)}
            body["messages"] = kept
            return
    for note in notes:
        entry = {"role": "user", "content": note}
        last = kept[-1] if kept else None
        if isinstance(last, Mapping) and last.get("role") == "assistant" and not last.get("tool_calls"):
            kept.insert(len(kept) - 1, entry)
        else:
            kept.append(entry)
    body["messages"] = kept


def menu_note(listed: Sequence[object], names: Sequence[str]) -> str | None:
    """Volatile-tail note naming the callable tools when the listed set is wider."""
    callable_names = [str(name) for name in names]
    gated = [_name(tool) for tool in listed if _name(tool) and _name(tool) not in set(callable_names)]
    if not gated:
        return None
    return (f"{NOTE_HEADER} callable now: {', '.join(callable_names) or 'none'}. "
            f"Listed but not callable this turn (a call is refused): {', '.join(gated)}.")
