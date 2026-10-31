"""Stable model-visible recovery capabilities for active StateFrame requests."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Callable

MODE_ENV = "STATEFRAME_CAPABILITY_FLOOR"
CAPABILITY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("read",),
    ("grep",),
    ("str_replace", "edit"),
    ("file_delete", "file_move"),
    ("shell",),
    ("run_tests", "authored_check"),
    ("read_check_receipt", "read_receipt"),
    ("done",),
)
CAPABILITY_NAMES = frozenset(name for group in CAPABILITY_GROUPS for name in group)
SCOPE_TURN_PREFERRED_TOOLS = ("str_replace", "edit", "run_tests", "authored_check")


def enabled(runtime: object | None = None,
            environ: Mapping[str, str] | None = None) -> bool:
    """Read the default-on floor switch; invalid values fail loudly."""
    configured = getattr(runtime, "stateframe_capability_floor_mode", None)
    source = os.environ if environ is None else environ
    raw = configured if configured is not None else source.get(MODE_ENV, "on")
    value = str(raw or "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{MODE_ENV} must be on or off")
    return value == "on"


def _name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    return str(function.get("name", "")).strip() if isinstance(function, Mapping) else ""


def _tools(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, Mapping):
        items = value.values()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = value
    else:
        return []
    return [item for item in items if isinstance(item, Mapping) and _name(item)]


def admissible_floor_names(
        available_tools: object,
        admissible: Callable[[str], bool] | None = None,
) -> tuple[str, ...]:
    """Return only floor capabilities backed by a current, dispatchable schema."""
    available = {_name(tool) for tool in _tools(available_tools)}
    allowed = {
        name for name in CAPABILITY_NAMES
        if name in available and (admissible is None or _safe_admission(admissible, name))
    }
    return tuple(name for group in CAPABILITY_GROUPS for name in group if name in allowed)


def rank_tools(
        selected: Sequence[Mapping[str, object]],
        preferred_names: Sequence[str],
) -> list[dict[str, object]]:
    """Move recommended actions forward while retaining every selected schema."""
    preferred = tuple(dict.fromkeys(
        name.strip() for name in preferred_names if isinstance(name, str) and name.strip()
    ))
    preferred_rank = {name: index for index, name in enumerate(preferred)}
    ordered = sorted(
        (tool for tool in selected if _name(tool) in preferred_rank),
        key=lambda tool: preferred_rank[_name(tool)],
    )
    remaining = [tool for tool in selected if _name(tool) not in preferred_rank]
    return [copy.deepcopy(dict(tool)) for tool in (*ordered, *remaining)]


def scope_turns_ranked_receipt(
        wire: bytes, *, runtime: object | None = None,
) -> dict[str, object] | None:
    """Prove a scoped edit continuation reached the final wire with edit actions first."""
    if not enabled(runtime):
        return None
    try:
        payload = json.loads(wire)
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    messages = payload.get("messages")
    tools = payload.get("tools")
    if not isinstance(messages, list) or not isinstance(tools, list):
        return None
    try:
        from kernel_release_wire_claims import scope_turns_on_wire
        from kernel_step_contract import edit_nturn_scope_turns_enabled

        if not edit_nturn_scope_turns_enabled():
            return None
        scope_turns = scope_turns_on_wire(messages)
    except (ImportError, AttributeError, ValueError):
        return None
    if not scope_turns:
        return None
    names = [_name(tool) for tool in tools]
    if any(not name for name in names) or len(names) != len(set(names)):
        return None
    preferred = tuple(
        name for name in SCOPE_TURN_PREFERRED_TOOLS if name in names
    )
    if len(preferred) < 2 or names[:len(preferred)] != list(preferred):
        return None
    return {
        "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest(),
        "ranked_event": "autopsy_r2_scope_turns_ranked",
        "recommended_tools": list(preferred),
        "scope_turns": scope_turns,
    }


def scope_turns_notice_receipt(
        wire: bytes, *, runtime: object | None = None,
) -> dict[str, object] | None:
    """Prove scoped continuation guidance is on-wire even when admission withholds edits."""
    if not enabled(runtime):
        return None
    try:
        payload = json.loads(wire)
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    messages = payload.get("messages")
    tools = payload.get("tools")
    if not isinstance(messages, list) or not isinstance(tools, list):
        return None
    try:
        from kernel_release_wire_claims import scope_turns_on_wire
        from kernel_step_contract import edit_nturn_scope_turns_enabled

        if not edit_nturn_scope_turns_enabled():
            return None
        scope_turns = scope_turns_on_wire(messages)
    except (ImportError, AttributeError, TypeError, ValueError):
        return None
    if not scope_turns:
        return None
    names = [_name(tool) for tool in tools]
    if any(not name for name in names) or len(names) != len(set(names)):
        return None
    return {
        "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest(),
        "notice_event": "autopsy_r2_scope_turns_notice_fired",
        "admitted_scope_actions": [
            name for name in SCOPE_TURN_PREFERRED_TOOLS if name in names
        ],
        "admitted_tools": names,
        "scope_turns": scope_turns,
    }


def _safe_admission(admissible: Callable[[str], bool], name: str) -> bool:
    try:
        return bool(admissible(name))
    except Exception:  # noqa: BLE001 - admission failures must fail closed
        return False


def apply_floor(
        selected: Sequence[Mapping[str, object]],
        available_tools: object,
        *,
        admissible: Callable[[str], bool] | None = None,
        active: bool = True,
        runtime: object | None = None,
) -> list[dict[str, object]]:
    """Add admissible floor schemas without narrowing the caller's selected menu.

    ``selected`` is already projected by its owning menu/dispatch authority, so
    existing schemas preserve their exact scope and order. Only missing floor
    actions are checked here and copied from the caller's current full schema set.
    Keeping admission out of the existing-menu pass makes this projection
    monotone: a planning/recovery floor can add choices, but cannot revoke an
    action selected by the upstream owner.
    """
    if not active or not enabled(runtime):
        return selected if isinstance(selected, list) else list(selected)

    available = {_name(tool): tool for tool in _tools(available_tools)}
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for tool in selected:
        name = _name(tool)
        if not name or name in seen:
            continue
        result.append(copy.deepcopy(dict(tool)))
        seen.add(name)

    for name in admissible_floor_names(available, admissible):
        if name not in seen:
            result.append(copy.deepcopy(dict(available[name])))
            seen.add(name)
    return result


__all__ = [
    "CAPABILITY_GROUPS", "CAPABILITY_NAMES", "MODE_ENV", "SCOPE_TURN_PREFERRED_TOOLS",
    "admissible_floor_names", "apply_floor", "enabled", "rank_tools",
    "scope_turns_notice_receipt", "scope_turns_ranked_receipt",
]
