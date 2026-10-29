"""Shared bounded action restriction for current-source interventions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

ENV = "STATEFRAME_REJECTION_TEMPLATE_RESTRICT"
TURNS_ENV = "STATEFRAME_REJECTION_TEMPLATE_RESTRICT_TURNS"
STATE_ATTR = "_rejection_template_restrict_state"
PROJECTION_RELEASE_ATTR = "_rejection_template_restrict_projection_release"
MAX_TURNS = 5
RESTRICTED_TOOL = "str_replace"
READ_POINTER_STATE_ATTR = "_read_pointer_edit_menu_state"
NOREAD_STATE_ATTR = "_noread_menu_state"
READ_POINTER_REQUIRED_TOOLS = ("str_replace", "edit")
READ_POINTER_OPTIONAL_TOOLS = ("authored_check", "run_tests")
READ_POINTER_ALLOWED_TOOLS = READ_POINTER_REQUIRED_TOOLS + READ_POINTER_OPTIONAL_TOOLS
READ_POINTER_MAX_TURNS = 8
NOREAD_MAX_TURNS = 8


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """The rejection restriction is controlled by its independent sub-flag."""
    values = os.environ if environ is None else environ
    return str(values.get(ENV, "off") or "off").strip().lower() == "on"


def turn_limit(environ: Mapping[str, str] | None = None) -> int:
    """Return a safe request count in the documented inclusive range 1..5."""
    values = os.environ if environ is None else environ
    raw = str(values.get(TURNS_ENV, "1") or "1").strip()
    try:
        requested = int(raw, 10)
    except (TypeError, ValueError):
        requested = 1
    return max(1, min(MAX_TURNS, requested))


def _request_id(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _emit(
    runtime: object,
    kind: str,
    *,
    producer: str = "kernel_rejection_template_restrict",
    **fields: object,
) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(kind, producer=producer, dispatch=False, **fields)


def _tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    return str(function.get("name", "")) if isinstance(function, Mapping) else ""


def _copy_tools(tools: object) -> list[dict[str, object]]:
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return []
    return [copy.deepcopy(dict(tool)) for tool in tools if isinstance(tool, Mapping)]


def applied_product_batch_mutation(status: object, changed_paths: object) -> bool:
    """Recognize a committed batch with at least one non-test workspace path."""
    if str(status or "").strip().upper() != "APPLIED":
        return False
    if not isinstance(changed_paths, Sequence) or isinstance(
            changed_paths, (str, bytes, bytearray)):
        return False
    for value in changed_paths:
        path = str(value or "").replace("\\", "/").strip("/")
        parts = [part for part in path.split("/") if part]
        if path and "tests" not in parts and not any(
                part.startswith("test_") for part in parts):
            return True
    return False


def _serialized_tool_names(payload: Mapping[str, object]) -> list[str] | None:
    """Read native tools and the strict content-only menu from exact body bytes."""
    menus: list[list[str]] = []
    if "tools" in payload:
        native = payload.get("tools")
        if not isinstance(native, list):
            return None
        menus.append([_tool_name(tool) for tool in native])

    response_format = payload.get("response_format")
    json_schema = (response_format.get("json_schema")
                   if isinstance(response_format, Mapping) else None)
    contract_name = (json_schema.get("name")
                     if isinstance(json_schema, Mapping) else None)
    if contract_name in {"kernel_step", "stateframe_action_envelope"}:
        schema = json_schema.get("schema")
        variants_key, action_key = (
            ("anyOf", "tool") if contract_name == "kernel_step"
            else ("oneOf", "action")
        )
        variants = schema.get(variants_key) if isinstance(schema, Mapping) else None
        if not isinstance(variants, list) or not variants:
            return None
        names: list[str] = []
        for variant in variants:
            properties = variant.get("properties") if isinstance(variant, Mapping) else None
            declaration = properties.get(action_key) if isinstance(properties, Mapping) else None
            if not isinstance(declaration, Mapping):
                return None
            name = declaration.get("const")
            enum = declaration.get("enum")
            if not isinstance(name, str) and isinstance(enum, list) and len(enum) == 1:
                name = enum[0]
            if not isinstance(name, str) or not name:
                return None
            names.append(name)
        if len(set(names)) != len(names):
            return None
        menus.append(names)

    if not menus or not menus[0] or len(set(menus[0])) != len(menus[0]):
        return None
    if any(menu != menus[0] for menu in menus[1:]):
        return None
    return menus[0]


def _wire_tool_names(wire: bytes) -> list[str] | None:
    try:
        payload = json.loads(wire)
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return _serialized_tool_names(payload) if isinstance(payload, Mapping) else None


def arm(runtime: object, *, source_request_id: object = None) -> bool:
    """Arm a restriction after a rejected definition edit, with or without a notice."""
    if not enabled():
        return False
    state: dict[str, object] = {
        "source_request_id": _request_id(source_request_id),
        "turn_limit": turn_limit(),
        "requests_sent": 0,
        "sent_request_ids": [],
        "active_request_id": None,
        "allowed_tools": [RESTRICTED_TOOL],
    }
    setattr(runtime, STATE_ATTR, state)
    setattr(runtime, PROJECTION_RELEASE_ATTR, None)
    _emit(
        runtime,
        "rejection_template_restrict_armed",
        source_request_id=state["source_request_id"],
        turn_limit=state["turn_limit"],
    )
    return True


def rejected_definition_edit_eligible(
    *,
    rejected: bool,
    root: Path,
    path: object,
    proposed: object,
    path_allowed: Callable[[Path], bool] | None = None,
) -> bool:
    """Require a relevant current-definition diff before arming the independent policy."""
    if not rejected or not enabled():
        return False
    try:
        from kernel_rejection_template import has_definition_diff
        return has_definition_diff(
            root, path, proposed, path_allowed=path_allowed
        )
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False


def arm_on_rejected_edit(
    runtime: object,
    *,
    rejected: bool,
    source_request_id: object = None,
) -> bool:
    """Arm on a whole-file/definition edit rejection even when its notice is disabled."""
    return arm(runtime, source_request_id=source_request_id) if rejected else False


def arm_read_pointer_menu(
    runtime: object,
    *,
    source_request_id: object = None,
) -> bool:
    """Arm the read-pointer policy through this shared owner-menu restriction path."""
    try:
        from kernel_read_pointer import edit_menu_enabled, edit_menu_turn_limit
    except ImportError:
        return False
    if not edit_menu_enabled():
        return False
    if isinstance(getattr(runtime, READ_POINTER_STATE_ATTR, None), Mapping):
        return True
    state: dict[str, object] = {
        "source_request_id": _request_id(source_request_id),
        "turn_limit": edit_menu_turn_limit(),
        "requests_sent": 0,
        "sent_request_ids": [],
        "active_request_id": None,
        "allowed_tools": list(READ_POINTER_REQUIRED_TOOLS),
    }
    setattr(runtime, READ_POINTER_STATE_ATTR, state)
    setattr(runtime, PROJECTION_RELEASE_ATTR, None)
    _emit(
        runtime,
        "read_pointer_edit_menu_armed",
        producer="kernel_read_pointer",
        source_request_id=state["source_request_id"],
        offered_tools=list(READ_POINTER_REQUIRED_TOOLS),
        turn_limit=state["turn_limit"],
    )
    return True


def is_active_request(runtime: object | None, request_id: object = None) -> bool:
    """Tell owner-contract checks whether this exact rejection request is restricted."""
    if runtime is None or not enabled():
        return False
    state = getattr(runtime, STATE_ATTR, None)
    expected = _request_id(request_id)
    return (
        isinstance(state, Mapping)
        and expected is not None
        and state.get("active_request_id") == expected
    )


def is_read_pointer_menu_active_request(
    runtime: object | None, request_id: object = None
) -> bool:
    if runtime is None:
        return False
    try:
        from kernel_read_pointer import edit_menu_enabled
    except ImportError:
        return False
    if not edit_menu_enabled():
        return False
    state = getattr(runtime, READ_POINTER_STATE_ATTR, None)
    expected = _request_id(request_id)
    return (
        isinstance(state, Mapping)
        and expected is not None
        and state.get("active_request_id") == expected
    )


def is_noread_menu_active_request(
    runtime: object | None, request_id: object = None
) -> bool:
    if runtime is None:
        return False
    try:
        from kernel_noread import enabled as noread_enabled
    except ImportError:
        return False
    if not noread_enabled():
        return False
    state = getattr(runtime, NOREAD_STATE_ATTR, None)
    expected = _request_id(request_id)
    return (
        isinstance(state, Mapping)
        and expected is not None
        and state.get("active_request_id") == expected
        and state.get("read_removed") is True
    )


def rejection_pending(runtime: object | None) -> bool:
    return (
        runtime is not None
        and enabled()
        and isinstance(getattr(runtime, STATE_ATTR, None), Mapping)
    )


def read_pointer_menu_pending(runtime: object | None) -> bool:
    if runtime is None:
        return False
    try:
        from kernel_read_pointer import edit_menu_enabled
    except ImportError:
        return False
    return (
        edit_menu_enabled()
        and isinstance(getattr(runtime, READ_POINTER_STATE_ATTR, None), Mapping)
    )


def noread_menu_pending(runtime: object | None) -> bool:
    """Return a live noread policy, releasing it if its triggering file changed."""
    if runtime is None:
        return False
    from kernel_noread import pending as noread_pending, source_content_changed

    if not noread_pending(runtime):
        return False
    if source_content_changed(runtime):
        state = getattr(runtime, NOREAD_STATE_ATTR, None)
        active_id = state.get("active_request_id") if isinstance(state, Mapping) else None
        release_noread(runtime, request_id=active_id, reason="source_content_changed")
        return False
    return True


def pending(runtime: object | None) -> bool:
    """Return whether any shared-path menu restriction awaits an owner request."""
    return (
        rejection_pending(runtime)
        or read_pointer_menu_pending(runtime)
        or noread_menu_pending(runtime)
    )


def owner_projection_release_pending(runtime: object | None) -> bool:
    """Whether the next owner-built request follows a released or exhausted menu."""
    if runtime is None:
        return False
    marker = getattr(runtime, PROJECTION_RELEASE_ATTR, None)
    if isinstance(marker, Mapping) and bool(marker.get("request_id")):
        return True
    states = [
        (getattr(runtime, STATE_ATTR, None), MAX_TURNS, enabled()),
        (
            getattr(runtime, READ_POINTER_STATE_ATTR, None),
            READ_POINTER_MAX_TURNS,
            read_pointer_menu_pending(runtime),
        ),
        (
            getattr(runtime, NOREAD_STATE_ATTR, None),
            NOREAD_MAX_TURNS,
            noread_menu_pending(runtime),
        ),
    ]
    for state, hard_limit, is_enabled in states:
        if not is_enabled or not isinstance(state, Mapping):
            continue
        try:
            sent = int(state.get("requests_sent", 0))
            maximum = max(1, min(hard_limit, int(state.get("turn_limit", 1))))
        except (TypeError, ValueError):
            continue
        if sent >= maximum:
            return True
    return False


def note_owner_projection_finalized(runtime: object | None) -> None:
    """Consume the release marker after the next request menu is assembled."""
    if runtime is not None:
        setattr(runtime, PROJECTION_RELEASE_ATTR, None)


def _clear(runtime: object, state_attr: str) -> None:
    setattr(runtime, state_attr, None)


def _release_policy(
    runtime: object,
    state: Mapping[str, object],
    *,
    state_attr: str,
    event_prefix: str,
    producer: str,
    request_id: str | None,
    reason: str,
    tool_name: str | None = None,
) -> None:
    _clear(runtime, state_attr)
    setattr(runtime, PROJECTION_RELEASE_ATTR, {
        "request_id": request_id,
        "restricted_request_id": state.get("active_request_id"),
        "reason": reason,
    })
    _emit(
        runtime,
        f"{event_prefix}_released",
        producer=producer,
        request_id=request_id,
        source_request_id=state.get("source_request_id"),
        restricted_request_id=state.get("active_request_id"),
        turn_limit=state.get("turn_limit"),
        requests_sent=state.get("requests_sent", 0),
        reason=reason,
        tool_name=tool_name,
    )


def release_noread(
    runtime: object,
    *,
    request_id: object = None,
    reason: str,
    tool_name: str | None = None,
) -> bool:
    """Release the shared noread menu state after content or lifecycle changes."""
    state = getattr(runtime, NOREAD_STATE_ATTR, None)
    if not isinstance(state, Mapping):
        return False
    _release_policy(
        runtime, state, state_attr=NOREAD_STATE_ATTR, event_prefix="noread",
        producer="kernel_noread", request_id=_request_id(request_id),
        reason=reason, tool_name=tool_name,
    )
    return True


def _skip_policy(
    runtime: object,
    state: Mapping[str, object],
    *,
    state_attr: str,
    event_prefix: str,
    producer: str,
    request_id: str | None,
    reason: str,
) -> None:
    if state.get("active_request_id") is not None:
        _release_policy(
            runtime, state, state_attr=state_attr, event_prefix=event_prefix,
            producer=producer, request_id=request_id,
            reason="restricted_action_not_admissible",
        )
    else:
        _clear(runtime, state_attr)
    _emit(
        runtime,
        f"{event_prefix}_skipped",
        producer=producer,
        request_id=request_id,
        source_request_id=state.get("source_request_id"),
        reason=reason,
    )


def _policy_admission(
    policy: str,
    default_admissible: bool,
    default_reason: str | None,
    overrides: Mapping[str, object] | None,
) -> tuple[bool, str | None]:
    value = overrides.get(policy) if isinstance(overrides, Mapping) else None
    if isinstance(value, Mapping):
        reason = value.get("reason")
        return value.get("admissible") is True, str(reason) if reason else default_reason
    return default_admissible, default_reason


def _effective_policy_tools(
    policy: str, required: Sequence[str], tools: Sequence[Mapping[str, object]],
    overrides: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """Retain required tools and only optional tools admitted by the owner."""
    optional_by_policy = {
        "rejection_template_restrict": {"run_tests"},
        "read_pointer_edit_menu": {"authored_check", "run_tests"},
    }
    value = overrides.get(policy) if isinstance(overrides, Mapping) else None
    requested = value.get("optional_tools") if isinstance(value, Mapping) else ()
    requested = requested if isinstance(requested, Sequence) and not isinstance(
        requested, (str, bytes, bytearray)) else ()
    permitted = optional_by_policy.get(policy, set())
    optional = {str(name) for name in requested if str(name) in permitted}
    names = [_tool_name(tool) for tool in tools]
    if any(names.count(name) != 1 for name in required):
        return tuple(required)
    selected = set(required) | {
        name for name in optional if names.count(name) == 1
    }
    return tuple(name for name in names if name in selected)


def _apply_policy(
    runtime: object,
    tools: list[dict[str, object]],
    *,
    state_attr: str,
    event_prefix: str,
    producer: str,
    allowed_tools: Sequence[str],
    hard_limit: int,
    request_id: object,
    admissible: bool,
    reason: str | None,
    emit_applied_event: bool = True,
    floor_ranked: bool = False,
) -> tuple[list[dict[str, object]], bool]:
    state = getattr(runtime, state_attr, None)
    if not isinstance(state, Mapping):
        return tools, False
    current_id = _request_id(request_id)
    source_id = _request_id(state.get("source_request_id"))
    if current_id is None:
        _skip_policy(
            runtime, state, state_attr=state_attr, event_prefix=event_prefix,
            producer=producer, request_id=None, reason="request_id_missing",
        )
        return tools, False
    if source_id is not None and current_id == source_id:
        return tools, False

    active_id = _request_id(state.get("active_request_id"))
    try:
        sent = int(state.get("requests_sent", 0))
        maximum = max(1, min(hard_limit, int(state.get("turn_limit", 1))))
    except (TypeError, ValueError):
        sent, maximum = 0, 1
    if active_id is not None and active_id != current_id and sent >= maximum:
        _release_policy(
            runtime, state, state_attr=state_attr, event_prefix=event_prefix,
            producer=producer, request_id=current_id, reason="request_limit",
        )
        return tools, False

    names = [_tool_name(tool) for tool in tools]
    all_offered = all(names.count(name) == 1 for name in allowed_tools)
    if not admissible or not all_offered:
        default_skip = (
            "required_menu_tool_not_offered" if not all_offered
            else "owner_route_not_actionable"
        )
        _skip_policy(
            runtime, state, state_attr=state_attr, event_prefix=event_prefix,
            producer=producer, request_id=current_id,
            reason=reason or default_skip,
        )
        return tools, False

    selected = [tool for tool in tools if _tool_name(tool) in allowed_tools]
    updated = dict(state)
    updated["active_request_id"] = current_id
    if floor_ranked:
        names = [_tool_name(tool) for tool in tools]
        updated["floor_ranked"] = True
        updated["priority_tools"] = list(allowed_tools)
        updated["advertised_tools"] = names
        updated["allowed_tools"] = names
    else:
        updated["floor_ranked"] = False
        updated.pop("priority_tools", None)
        updated.pop("advertised_tools", None)
        updated["allowed_tools"] = list(allowed_tools)
    if event_prefix == "noread":
        updated["read_removed"] = not floor_ranked
    setattr(runtime, state_attr, updated)
    state = updated
    if active_id != current_id and emit_applied_event:
        event_fields: dict[str, object] = {
            "request_id": current_id,
            "source_request_id": state.get("source_request_id"),
            "offered_tools": names if floor_ranked else list(allowed_tools),
            "turn_limit": maximum,
            "requests_sent": sent,
        }
        if floor_ranked:
            event_fields.update({
                "menu_mode": "ranked",
                "recommended_tools": list(allowed_tools),
            })
        _emit(
            runtime,
            f"{event_prefix}_applied",
            producer=producer,
            **event_fields,
        )
    if floor_ranked:
        from kernel_capability_floor import rank_tools
        return rank_tools(tools, allowed_tools), True
    return selected, True


def _apply_configured_policy(
    runtime: object,
    current: list[dict[str, object]],
    *,
    state_attr: str,
    prefix: str,
    producer: str,
    allowed_tools: Sequence[str],
    hard_limit: int,
    request_id: object,
    admissible: bool,
    reason: str | None,
    policy_admissibility: Mapping[str, object] | None,
    floor_ranked: bool = False,
) -> list[dict[str, object]] | None:
    policy_ok, policy_reason = _policy_admission(
        prefix, admissible, reason, policy_admissibility
    )
    selected, applied = _apply_policy(
        runtime, current, state_attr=state_attr, event_prefix=prefix,
        producer=producer, allowed_tools=allowed_tools, hard_limit=hard_limit,
        request_id=request_id, admissible=policy_ok, reason=policy_reason,
        emit_applied_event=prefix != "noread",
        floor_ranked=floor_ranked,
    )
    return selected if applied else None


def _noread_allowed_tools(
    runtime: object,
    tools: Sequence[Mapping[str, object]],
    *,
    request_id: object,
    policy_admissibility: Mapping[str, object] | None = None,
) -> tuple[str, ...] | None:
    names = tuple(_tool_name(tool) for tool in tools)
    if "read" not in names:
        state = getattr(runtime, NOREAD_STATE_ATTR, None)
        if isinstance(state, Mapping):
            _skip_policy(
                runtime, state, state_attr=NOREAD_STATE_ATTR, event_prefix="noread",
                producer="kernel_noread", request_id=_request_id(request_id),
                reason="read_not_offered",
            )
        return None
    value = policy_admissibility.get("noread") if isinstance(
        policy_admissibility, Mapping) else None
    optional = value.get("optional_tools", ()) if isinstance(value, Mapping) else ()
    optional = optional if isinstance(optional, Sequence) and not isinstance(
        optional, (str, bytes, bytearray)) else ()
    admitted_optional = {str(name) for name in optional if str(name) in {"run_tests"}}
    allowed = tuple(
        name for name in names
        if name != "read" and (name != "run_tests" or name in admitted_optional)
    )
    if not any(name != "done" for name in allowed):
        state = getattr(runtime, NOREAD_STATE_ATTR, None)
        if isinstance(state, Mapping):
            _skip_policy(
                runtime, state, state_attr=NOREAD_STATE_ATTR, event_prefix="noread",
                producer="kernel_noread", request_id=_request_id(request_id),
                reason="read_is_only_non_done_action",
            )
        return None
    return allowed


def apply_menu(
    runtime: object,
    tools: object,
    *,
    request_id: object,
    admissible: bool,
    reason: str | None = None,
    policy_admissibility: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, object]], bool]:
    """Apply shared restrictions through one owner-menu and dispatch projection."""
    from kernel_capability_floor import enabled as capability_floor_enabled

    current = _copy_tools(tools)
    floor_ranked = capability_floor_enabled(runtime)
    if not pending(runtime):
        note_owner_projection_finalized(runtime)
        return current, False

    accepted_menus: list[list[dict[str, object]]] = []
    policies = (
        (
            enabled(),
            STATE_ATTR,
            "rejection_template_restrict",
            "kernel_rejection_template_restrict",
            (RESTRICTED_TOOL,),
            MAX_TURNS,
        ),
        (
            read_pointer_menu_pending(runtime),
            READ_POINTER_STATE_ATTR,
            "read_pointer_edit_menu",
            "kernel_read_pointer",
            READ_POINTER_REQUIRED_TOOLS,
            READ_POINTER_MAX_TURNS,
        ),
    )
    for is_enabled, state_attr, prefix, producer, allowed, hard_limit in policies:
        if not is_enabled:
            continue
        effective_allowed = _effective_policy_tools(
            prefix, allowed, current, policy_admissibility)
        selected = _apply_configured_policy(
            runtime, current, state_attr=state_attr, prefix=prefix,
            producer=producer, allowed_tools=effective_allowed, hard_limit=hard_limit,
            request_id=request_id, admissible=admissible, reason=reason,
            policy_admissibility=policy_admissibility,
            floor_ranked=floor_ranked,
        )
        if selected is not None:
            accepted_menus.append(selected)

    if noread_menu_pending(runtime):
        allowed = _noread_allowed_tools(
            runtime, current, request_id=request_id,
            policy_admissibility=policy_admissibility)
        if allowed is not None:
            selected = _apply_configured_policy(
                runtime, current, state_attr=NOREAD_STATE_ATTR,
                prefix="noread", producer="kernel_noread",
                allowed_tools=allowed, hard_limit=NOREAD_MAX_TURNS,
                request_id=request_id, admissible=admissible, reason=reason,
                policy_admissibility=policy_admissibility,
                floor_ranked=floor_ranked,
            )
            if selected is not None:
                accepted_menus.append(selected)

    note_owner_projection_finalized(runtime)
    if not accepted_menus:
        return current, False

    if floor_ranked:
        current_id = _request_id(request_id)
        preferred: list[str] = []
        for state_attr in (STATE_ATTR, READ_POINTER_STATE_ATTR, NOREAD_STATE_ATTR):
            state = getattr(runtime, state_attr, None)
            if (not isinstance(state, Mapping)
                    or state.get("floor_ranked") is not True
                    or state.get("active_request_id") != current_id):
                continue
            names = state.get("priority_tools")
            if isinstance(names, Sequence) and not isinstance(names, (str, bytes, bytearray)):
                preferred.extend(str(name) for name in names if str(name))
        from kernel_capability_floor import rank_tools
        return rank_tools(current, preferred), True

    allowed_names = {_tool_name(tool) for tool in accepted_menus[0]}
    for selected in accepted_menus[1:]:
        allowed_names.intersection_update(_tool_name(tool) for tool in selected)
    return [tool for tool in current if _tool_name(tool) in allowed_names], True


def release_after_product_mutation(
    runtime: object,
    *,
    request_id: object,
    tool_name: object,
) -> bool:
    """Release active shared-path policies after the first applied product mutation."""
    current_id = _request_id(request_id)
    name = str(tool_name or "").strip()
    mutation_tools = {
        "edit", "str_replace", "file_create", "insert_before",
        "replace_module", "write", "replace",
    }
    if current_id is None or name not in mutation_tools:
        return False

    policies = (
        (
            enabled(),
            STATE_ATTR,
            "rejection_template_restrict",
            "kernel_rejection_template_restrict",
        ),
        (
            read_pointer_menu_pending(runtime),
            READ_POINTER_STATE_ATTR,
            "read_pointer_edit_menu",
            "kernel_read_pointer",
        ),
        (
            isinstance(getattr(runtime, NOREAD_STATE_ATTR, None), Mapping),
            NOREAD_STATE_ATTR,
            "noread",
            "kernel_noread",
        ),
    )
    released = False
    for is_enabled, state_attr, prefix, producer in policies:
        state = getattr(runtime, state_attr, None)
        if (not is_enabled or not isinstance(state, Mapping)
                or state.get("active_request_id") != current_id):
            continue
        _release_policy(
            runtime, state, state_attr=state_attr, event_prefix=prefix,
            producer=producer, request_id=current_id,
            reason="first_applied_product_mutation", tool_name=name,
        )
        released = True
    return released


def release_noread_after_mutation(
    runtime: object,
    *,
    request_id: object,
    tool_name: object,
) -> bool:
    """Disarm only noread after an applied edit, including edits to test files."""
    current_id = _request_id(request_id)
    name = str(tool_name or "").strip()
    if current_id is None or name not in {
        "edit", "str_replace", "file_create", "insert_before",
        "replace_module", "write", "replace",
    }:
        return False
    state = getattr(runtime, NOREAD_STATE_ATTR, None)
    if (not isinstance(state, Mapping)
            or state.get("active_request_id") != current_id):
        return False
    return release_noread(
        runtime, request_id=current_id, reason="first_applied_mutation", tool_name=name,
    )


def release_after_test_run(
    runtime: object,
    *,
    request_id: object,
    tool_name: object = "run_tests",
) -> bool:
    """Disarm noread after a dispatched test action on its active request."""
    current_id = _request_id(request_id)
    state = getattr(runtime, NOREAD_STATE_ATTR, None)
    if (current_id is None or not isinstance(state, Mapping)
            or state.get("active_request_id") != current_id):
        return False
    return release_noread(
        runtime, request_id=current_id, reason="tests_run",
        tool_name=str(tool_name or "run_tests"),
    )


def _wire_receipt_for_policy(
    wire: bytes,
    *,
    runtime: object | None,
    request_id: object,
    state_attr: str,
    enabled_now: bool,
    allowed_tools: Sequence[str],
    exact_order: bool,
    ranked_event: str | None = None,
) -> dict[str, object] | None:
    if runtime is None or not enabled_now:
        return None
    state = getattr(runtime, state_attr, None)
    current_id = _request_id(request_id)
    if (not isinstance(state, dict) or current_id is None
            or state.get("active_request_id") != current_id):
        return None
    try:
        payload = json.loads(wire)
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    names = _serialized_tool_names(payload)
    if names is None:
        return None
    from kernel_capability_floor import enabled as capability_floor_enabled
    floor_now = capability_floor_enabled(runtime)
    floor_ranked = state.get("floor_ranked") is True
    if floor_now != floor_ranked:
        return None
    if floor_ranked:
        preferred = state.get("priority_tools")
        advertised = state.get("advertised_tools")
        matches = (
            isinstance(preferred, Sequence)
            and not isinstance(preferred, (str, bytes, bytearray))
            and isinstance(advertised, Sequence)
            and not isinstance(advertised, (str, bytes, bytearray))
            and bool(preferred)
            and names[:len(preferred)] == [str(name) for name in preferred]
            and set(str(name) for name in advertised).issubset(names)
        )
    elif exact_order:
        matches = names == list(allowed_tools)
    else:
        matches = len(names) == len(allowed_tools) and set(names) == set(allowed_tools)
    if not matches:
        return None

    sent_ids = state.get("sent_request_ids")
    if not isinstance(sent_ids, list):
        sent_ids = []
    if current_id not in sent_ids:
        sent_ids.append(current_id)
        state["sent_request_ids"] = sent_ids
        try:
            state["requests_sent"] = int(state.get("requests_sent", 0)) + 1
        except (TypeError, ValueError):
            state["requests_sent"] = 1
    receipt: dict[str, object] = {
        "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest(),
        "offered_tools": names,
        "source_request_id": state.get("source_request_id"),
        "request_ordinal": state.get("requests_sent"),
        "turn_limit": state.get("turn_limit"),
    }
    if floor_ranked and ranked_event:
        receipt["ranked_event"] = ranked_event
        receipt["recommended_tools"] = list(state.get("priority_tools", ()))
    return receipt


def wire_receipt(
    wire: bytes,
    *,
    runtime: object | None,
    request_id: object,
) -> dict[str, object] | None:
    """Prove the rejection restriction was present in exact submitted bytes."""
    state = getattr(runtime, STATE_ATTR, None)
    allowed = state.get("allowed_tools") if isinstance(state, Mapping) else None
    if not isinstance(allowed, Sequence) or isinstance(allowed, (str, bytes, bytearray)):
        allowed = (RESTRICTED_TOOL,)
    from kernel_rejection_template import enabled as rejection_template_enabled

    return _wire_receipt_for_policy(
        wire, runtime=runtime, request_id=request_id, state_attr=STATE_ATTR,
        enabled_now=enabled(), allowed_tools=tuple(str(name) for name in allowed),
        exact_order=False,
        ranked_event=(
            "rejection_template_restrict_ranked" if rejection_template_enabled()
            else "rejection_restrict_only_ranked"
        ),
    )


def read_pointer_menu_wire_receipt(
    wire: bytes,
    *,
    runtime: object | None,
    request_id: object,
) -> dict[str, object] | None:
    """Prove the complete read-pointer menu was present in exact submitted bytes."""
    try:
        from kernel_read_pointer import edit_menu_enabled
    except ImportError:
        return None
    state = getattr(runtime, READ_POINTER_STATE_ATTR, None)
    allowed = state.get("allowed_tools") if isinstance(state, Mapping) else None
    if not isinstance(allowed, Sequence) or isinstance(allowed, (str, bytes, bytearray)):
        return None
    return _wire_receipt_for_policy(
        wire, runtime=runtime, request_id=request_id,
        state_attr=READ_POINTER_STATE_ATTR,
        enabled_now=edit_menu_enabled(),
        allowed_tools=tuple(str(name) for name in allowed),
        exact_order=False,
        ranked_event="read_pointer_edit_menu_ranked",
    )


def noread_wire_receipt(
    wire: bytes,
    *,
    runtime: object | None,
    request_id: object,
) -> dict[str, object] | None:
    """Prove this armed request's serialized menu excludes read."""
    if runtime is None:
        return None
    from kernel_noread import enabled as noread_enabled

    if not noread_enabled() or not noread_menu_pending(runtime):
        return None
    state = getattr(runtime, NOREAD_STATE_ATTR, None)
    current_id = _request_id(request_id)
    if (not isinstance(state, dict) or current_id is None
            or state.get("active_request_id") != current_id
            or state.get("read_removed") is not (state.get("floor_ranked") is not True)):
        return None
    if state.get("floor_ranked") is True:
        preferred = state.get("priority_tools")
        if not isinstance(preferred, Sequence) or isinstance(preferred, (str, bytes, bytearray)):
            return None
        return _wire_receipt_for_policy(
            wire, runtime=runtime, request_id=request_id,
            state_attr=NOREAD_STATE_ATTR, enabled_now=noread_enabled(),
            allowed_tools=tuple(str(name) for name in preferred), exact_order=True,
            ranked_event="noread_ranked",
        )
    names = _wire_tool_names(wire)
    effective_tools = active_menu_tools(runtime, current_id)
    if (names is None or "read" in names or effective_tools is None
            or len(names) != len(effective_tools)
            or set(names) != set(effective_tools)):
        return None

    sent_ids = state.get("sent_request_ids")
    if not isinstance(sent_ids, list):
        sent_ids = []
    if current_id not in sent_ids:
        sent_ids.append(current_id)
        state["sent_request_ids"] = sent_ids
        try:
            state["requests_sent"] = int(state.get("requests_sent", 0)) + 1
        except (TypeError, ValueError):
            state["requests_sent"] = 1
    return {
        "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest(),
        "offered_tools": names,
        "source_request_id": state.get("source_request_id"),
        "request_ordinal": state.get("requests_sent"),
        "turn_limit": state.get("turn_limit"),
    }


def active_menu_tools(
    runtime: object | None, request_id: object
) -> tuple[str, ...] | None:
    """Return effective restricted names for route and dispatch contract checks."""
    if runtime is None:
        return None
    current_id = _request_id(request_id)
    if current_id is None:
        return None
    allowed_sets: list[tuple[str, ...]] = []
    if is_active_request(runtime, current_id):
        state = getattr(runtime, STATE_ATTR, None)
        names = state.get("allowed_tools") if isinstance(state, Mapping) else None
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes, bytearray)):
            allowed_sets.append(tuple(str(name) for name in names if str(name)))
    if is_read_pointer_menu_active_request(runtime, current_id):
        state = getattr(runtime, READ_POINTER_STATE_ATTR, None)
        names = state.get("allowed_tools") if isinstance(state, Mapping) else None
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes, bytearray)):
            allowed_sets.append(tuple(str(name) for name in names if str(name)))
    if is_noread_menu_active_request(runtime, current_id):
        state = getattr(runtime, NOREAD_STATE_ATTR, None)
        names = state.get("allowed_tools") if isinstance(state, Mapping) else None
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes, bytearray)):
            allowed_sets.append(tuple(str(name) for name in names if str(name)))
    if not allowed_sets:
        return None
    common = set(allowed_sets[0])
    for allowed in allowed_sets[1:]:
        common.intersection_update(allowed)
    return tuple(name for name in allowed_sets[0] if name in common)


def active_policy_records(
    runtime: object | None, request_id: object
) -> list[dict[str, object]]:
    """Describe active policies for owner-synchronized tool instructions."""
    if runtime is None:
        return []
    current_id = _request_id(request_id)
    rows: list[dict[str, object]] = []
    if is_active_request(runtime, current_id):
        state = getattr(runtime, STATE_ATTR, None)
        allowed = state.get("allowed_tools") if isinstance(state, Mapping) else None
        ranked = isinstance(state, Mapping) and state.get("floor_ranked") is True
        row: dict[str, object] = {
            "component": "rejection_template_restrict",
            "reason": "A rejected whole-file edit requires the focused replacement action.",
        }
        if ranked:
            row.update({
                "menu_mode": "ranked",
                "recommended_tools": list(state.get("priority_tools", ())),
                "reason": (
                    "The rejected whole-file edit omitted an unchanged definition; "
                    "prefer a focused replacement."
                ),
            })
        else:
            row["allowed_tools"] = (
                list(allowed) if isinstance(allowed, list) else [RESTRICTED_TOOL]
            )
        rows.append(row)
    if is_read_pointer_menu_active_request(runtime, current_id):
        state = getattr(runtime, READ_POINTER_STATE_ATTR, None)
        allowed = state.get("allowed_tools") if isinstance(state, Mapping) else None
        ranked = isinstance(state, Mapping) and state.get("floor_ranked") is True
        row = {
            "component": "read_pointer_edit_menu",
            "reason": "Keep this request focused on the named source: edit or verify it without more reads.",
        }
        if ranked:
            row.update({
                "menu_mode": "ranked",
                "recommended_tools": list(state.get("priority_tools", ())),
                "reason": (
                    "The requested definitions were already supplied; prefer an "
                    "edit or verification action next."
                ),
            })
        else:
            row["allowed_tools"] = (
                list(allowed) if isinstance(allowed, list)
                else list(READ_POINTER_REQUIRED_TOOLS)
            )
        rows.append(row)
    if is_noread_menu_active_request(runtime, current_id):
        state = getattr(runtime, NOREAD_STATE_ATTR, None)
        allowed = state.get("allowed_tools") if isinstance(state, Mapping) else None
        if isinstance(allowed, list):
            ranked = state.get("floor_ranked") is True
            row = {
                "component": "noread",
                "reason": "A repeat read returned unchanged content; continue with any other offered action.",
            }
            if ranked:
                row.update({
                    "menu_mode": "ranked",
                    "recommended_tools": list(state.get("priority_tools", ())),
                    "reason": (
                        "A repeat read returned unchanged content; another action is "
                        "recommended."
                    ),
                })
            else:
                row["allowed_tools"] = list(allowed)
            rows.append(row)
    return rows
