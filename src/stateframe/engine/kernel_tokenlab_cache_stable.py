"""Token-lab v8: KV-cache-friendly F4 request layout (STATEFRAME_TOKENLAB_CACHE_STABLE).

Why: llama.cpp reuses the longest common TOKEN prefix between consecutive requests on a slot.
The r46/v6 F4 order puts per-turn blocks (StateFrame, handoff) before the real-exchange ring
and trails slow notices after it, and the tool list (rendered inside the system turn by the
MiniCPM5 template) changes whenever the menu is gated -- so the cache breaks early and the
whole prompt is re-prefilled every turn (lane 2: 21.8% message-prefix reuse vs the boss's 61.4%).

This module is presentation-only and pure (no runtime import), so the offline journal replay
exercises the exact code the live arm runs:

* :func:`layout_messages` -- a permutation by change-frequency tier (measured on the 703 v6a
  journals): system, task, constant notices, slow notices, source blocks, medium notices, the
  real-exchange ring (assistant/tool order untouched), per-turn blocks, StateFrame, handoff.
  Unknown blocks default to the medium tier, before the ring.  A trailing assistant prefill
  (no tool calls) stays last.
* :class:`ToolLayout` -- a per-run append-only registry of every tool schema the run has been
  offered, in first-seen order.  A tool whose PARAMETERS change is replaced in place; a
  description-only change (the menu flips tools between their verbose description and a terse
  "Host X action." form: 5 flips in the ldev_12 v6a run) keeps the first-seen schema.  The PROVIDER
  body carries the registry; authorization is untouched -- dispatch still validates against the
  request's real menu, so a listed-but-gated tool is refused as "currently unavailable".  When
  the menu is narrower than the registry a short trailing note names the callable tools.

With the switch off nothing here runs (the runtime hook and the ask_tools override are both
guarded), so requests are byte-identical to v6a.
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence

MODE_ENV = "STATEFRAME_TOKENLAB_CACHE_STABLE"
MENU_TOOL_LIST_MODE_ENV = "STATEFRAME_TOKENLAB_MENU_TOOL_LIST"
EVENT = "tokenlab_cache_stable_applied"
V15_EVENT = "tokenlab_v15_cache_prefill_protected"
MENU_NOTE_HEADER = "[Tool menu | this turn]"

# Tier table.  Order inside a tier = original order.  Rates are the fraction of consecutive
# requests in which the block changed (703 v6a ldev_04/05/12, 24 pairs).
_CONSTANT = ("[SHIP-TREE SMOKE]", "[Decisive factor]")                      # 0.00
_SOURCE = ("[Working source:", "[Current source:", "[Observed source:",
           "[Retained source:", "[Source content:", "CURRENT_SOURCE ")     # 0.00-0.11
_SLOW = ("[VERIFICATION CERTIFICATE]", "[DELIVERY_MANIFEST", "[Unified stop authority]",
         "<settled_state>", "[RAISING-H FAILURE PACKET]", "GREEN LOCK")    # 0.08-0.18
_PER_TURN = ("[Working sets", "[Pending intent", "[Reasoning ledger")       # 1.00 (snapshot ids)
_STATE = "[StateFrame]"
_HANDOFF = "[Decision handoff]"

# Slow notices BEFORE sources: offline replay (v8_replay.py, 27 v6a requests) measured 48.8% token-prefix
# reuse with this order vs 45.9% sources-first and 48.5% all-notices-first (v6a as run: 38.3%).  New source
# blocks (a first read of a file) are the most frequent non-tool break, so they sit as late as possible.
T_SYSTEM, T_TASK, T_CONST, T_SLOW, T_SOURCE, T_MEDIUM, T_RING, T_TURN, T_STATE, T_HANDOFF, \
    T_NOTE, T_PREFILL = range(12)


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def menu_tool_list_mode() -> str:
    value = os.environ.get(MENU_TOOL_LIST_MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MENU_TOOL_LIST_MODE_ENV} must be off or on")
    return value


def menu_tool_list_enabled() -> bool:
    return menu_tool_list_mode() == "on"


def _content(message: Mapping[str, object]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _is_terminal_prefill(message: Mapping[str, object]) -> bool:
    return (message.get("role") == "assistant" and not message.get("tool_calls"))


def tier(message: Mapping[str, object]) -> int:
    role = message.get("role")
    text = _content(message)
    if role in {"system", "developer"}:
        return T_SYSTEM
    if role in {"assistant", "tool"}:
        return T_RING
    if text.startswith(MENU_NOTE_HEADER):
        return T_NOTE
    if text.startswith(("Task:", "[FRAME_CLARITY_BOOTSTRAP]")):
        return T_TASK
    if text.startswith(_CONSTANT):
        return T_CONST
    if text.startswith(_SOURCE):
        return T_SOURCE
    if text.startswith(_SLOW):
        return T_SLOW
    if text.startswith(_PER_TURN):
        return T_TURN
    if text.startswith(_STATE):
        return T_STATE
    if text.startswith(_HANDOFF):
        return T_HANDOFF
    return T_MEDIUM


def layout_messages(messages: Sequence[Mapping[str, object]]) -> list[dict]:
    """Stable-first / volatile-last permutation.  Never adds, drops or edits a message."""
    items = [dict(message) for message in messages]
    prefill = None
    if items and _is_terminal_prefill(items[-1]):
        prefill = items.pop()     # continuation prefill must stay the final message
    ordered = [item for _i, item in sorted(enumerate(items), key=lambda p: (tier(p[1]), p[0]))]
    if prefill is not None:
        ordered.append(prefill)
    return ordered


def _canon(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def is_permutation(before: Sequence[Mapping[str, object]],
                   after: Sequence[Mapping[str, object]]) -> bool:
    return sorted(map(_canon, before)) == sorted(map(_canon, after))


def ring_pairs_intact(messages: Sequence[Mapping[str, object]]) -> bool:
    """Every tool message is preceded (within the ring) by the assistant that issued its id."""
    issued: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or ():
                if isinstance(call, Mapping) and isinstance(call.get("id"), str):
                    issued.add(call["id"])
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str) and call_id not in issued:
                return False
    return True


def _tool_name(tool: object) -> str:
    if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping):
        name = tool["function"].get("name")
        return name if isinstance(name, str) else ""
    return ""


def _strip_descriptions(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _strip_descriptions(item) for key, item in value.items() if key != "description"}
    if isinstance(value, list):
        return [_strip_descriptions(item) for item in value]
    return value


def _parameters(tool: Mapping[str, object]) -> object:
    """The argument contract: parameter schema with every prose ``description`` removed."""
    function = tool.get("function") if isinstance(tool, Mapping) else None
    return _strip_descriptions(function.get("parameters")) if isinstance(function, Mapping) else None


class ToolLayout:
    """Append-only, first-seen-order registry of the tool schemas a run has been offered."""

    def __init__(self) -> None:
        self._order: list[str] = []
        self._schemas: dict[str, dict] = {}
        self.additions = 0
        self.replacements = 0

    def observe(self, tools: Sequence[Mapping[str, object]]) -> None:
        for tool in tools or ():
            name = _tool_name(tool)
            if not name:
                continue
            if name not in self._schemas:
                self._order.append(name)
                self.additions += 1
            elif _canon(_parameters(self._schemas[name])) != _canon(_parameters(tool)):
                self.replacements += 1      # the argument CONTRACT changed: the model must see it
            else:
                continue                    # description-only churn (verbose <-> "Host X action."): keep
            self._schemas[name] = copy.deepcopy(dict(tool))

    def tools(self) -> list[dict]:
        return [copy.deepcopy(self._schemas[name]) for name in self._order]

    def names(self) -> list[str]:
        return list(self._order)


def menu_note(offered: Sequence[str], listed: Sequence[str]) -> str | None:
    """Short trailing note when the callable menu is narrower than the listed tool set."""
    gated = [name for name in listed if name not in set(offered)]
    if not gated:
        return None
    return (f"{MENU_NOTE_HEADER} callable now: {', '.join(offered) or 'none'}. "
            f"Not callable this turn (a call is refused): {', '.join(gated)}.")


def apply(messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]] | None,
          registry: ToolLayout | None, *, protect_prefill: bool = False
          ) -> tuple[list[dict], list[dict] | None, dict]:
    """(messages, provider_tools, audit).  provider_tools is None when the menu is not changed."""
    offered = [_tool_name(tool) for tool in (tools or ()) if _tool_name(tool)]
    provider_tools = None
    note = None
    if registry is not None and tools is not None:
        registry.observe(tools)
        provider_tools = registry.tools()
        if menu_tool_list_enabled():
            offered_set = set(offered)
            provider_tools = [tool for tool in provider_tools if _tool_name(tool) in offered_set]
            listed = [_tool_name(tool) for tool in provider_tools]
        else:
            listed = registry.names()
        note = menu_note(offered, listed)
    candidate = [dict(message) for message in messages]
    terminal_prefill = candidate[-1] if candidate and _is_terminal_prefill(candidate[-1]) else None
    if note is not None:
        note_message = {"role": "user", "content": note}
        if protect_prefill and terminal_prefill is not None:
            candidate.insert(len(candidate) - 1, note_message)
        else:
            candidate.append(note_message)
    ordered = layout_messages(candidate)
    if not is_permutation(candidate, ordered) or not ring_pairs_intact(ordered):
        raise AssertionError("cache-stable layout changed the message set or split an exchange")
    if protect_prefill and terminal_prefill is not None:
        if not ordered or ordered[-1] != terminal_prefill:
            raise AssertionError("cache-stable layout moved the continuation prefill")
    audit = {
        "moved": sum(1 for a, b in zip(candidate, ordered) if a is not b and a != b),
        "tiers": [tier(message) for message in ordered],
        "offered_tools": offered,
        "listed_tools": listed if registry is not None and tools is not None else offered,
        "menu_note": note is not None,
        "registry_additions": registry.additions if registry is not None else 0,
        "registry_replacements": registry.replacements if registry is not None else 0,
        "prefill_protected": bool(protect_prefill and terminal_prefill is not None),
    }
    if registry is not None and tools is not None and menu_tool_list_enabled():
        provider_names = [_tool_name(tool) for tool in (provider_tools or ())]
        audit["callable_tool_list_filtered"] = True
        audit["filtered_tools"] = [name for name in registry.names() if name not in provider_names]
        audit["provider_tools"] = provider_names
    return ordered, provider_tools, audit
