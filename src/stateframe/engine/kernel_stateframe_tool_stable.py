"""Stable tool descriptions, a versioned wire catalog, and current admission.

The admitted action set remains the current N4 projection. Menu freeze keeps the
non-done serialized catalog stable until an enabled semantic-identity rebase;
dispatch and the signed request contract continue to use this turn's admitted set.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace


TOOL_MODE_ENV = "STATEFRAME_TOOL_STABLE"
MENU_MODE_ENV = "STATEFRAME_MENU_FREEZE"
MENU_REBASE_MODE_ENV = "STATEFRAME_MENU_FREEZE_REBASE"
TOOL_EVENT_PREFIX = "stateframe_tool_stable"
MENU_EVENT_PREFIX = "stateframe_menu_freeze"
MENU_REBASE_EVENT_PREFIX = "stateframe_menu_freeze_rebase"
_TOOL_MODES = frozenset({"off", "shadow", "on"})
_MENU_MODES = frozenset({"off", "shadow", "on"})
_MENU_REBASE_MODES = frozenset({"off", "shadow", "on"})
_TOOL_STATE_ATTR = "_stateframe_tool_stable_state"
_PENDING_ATTR = "_stateframe_tool_surface_pending"
_CANONICAL_DESCRIPTION = (
    "Current behavior and constraints are provided in this request's [tool guidance] section."
)


class ToolSurfaceError(ValueError):
    """The stable tool surface could not be validated or projected safely."""


@dataclass(frozen=True)
class ToolSurfaceState:
    run_id: str
    order: tuple[str, ...]
    schemas: tuple[tuple[str, dict[str, object]], ...]
    generation: int
    menu_catalog_initialized: bool = False
    menu_catalog_order: tuple[str, ...] = ()
    menu_catalog_schemas: tuple[tuple[str, dict[str, object]], ...] = ()
    menu_done_index: int | None = None
    menu_catalog_version: int = 0


@dataclass(frozen=True)
class ToolSurfaceProjection:
    wire_tools: list[dict[str, object]]
    admitted_tools: list[dict[str, object]]
    admitted_names: tuple[str, ...]
    catalog_names: tuple[str, ...]
    frozen_catalog_tools: list[dict[str, object]]
    frozen_catalog_names: tuple[str, ...]
    guidance: tuple[str, ...]
    state: ToolSurfaceState
    stable_mode: str
    menu_mode: str
    stable_shadow_mismatch_fields: tuple[str, ...]
    shadow_mismatch_fields: tuple[str, ...]
    selected_menu_mode: str = "off"
    rebase_mode: str = "off"
    rebase_from_version: int | None = None
    rebase_candidate_version: int | None = None
    rebase_changed_names: tuple[str, ...] = ()
    rebase_added_names: tuple[str, ...] = ()
    rebase_candidate_identity: str = ""
    rebase_shadow_mismatch_fields: tuple[str, ...] = ()
    rebase_failed_open_reason: str | None = None
    menu_freeze_failed_open_reason: str | None = None


@dataclass(frozen=True)
class ToolSurfacePending:
    runtime: object
    previous_state: ToolSurfaceState | None
    candidate_state: ToolSurfaceState
    request_id: str | None
    stable_mode: str
    menu_mode: str
    selected_menu_mode: str = "off"
    rebase_mode: str = "off"
    rebase_from_version: int | None = None
    rebase_candidate_version: int | None = None
    rebase_changed_names: tuple[str, ...] = ()
    rebase_added_names: tuple[str, ...] = ()
    rebase_candidate_identity: str = ""
    rebase_shadow_mismatch_fields: tuple[str, ...] = ()
    rebase_failed_open_reason: str | None = None
    menu_freeze_failed_open_reason: str | None = None


def tool_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    selected = str(source.get(TOOL_MODE_ENV, "off")).strip().lower()
    return selected if selected in _TOOL_MODES else "off"


def menu_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    selected = str(source.get(MENU_MODE_ENV, "off")).strip().lower()
    return selected if selected in _MENU_MODES else "off"


def menu_rebase_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    selected = str(source.get(MENU_REBASE_MODE_ENV, "off")).strip().lower()
    return selected if selected in _MENU_REBASE_MODES else "off"


def _tool_name(tool: Mapping[str, object]) -> str:
    function = tool.get("function")
    name = function.get("name") if isinstance(function, Mapping) else None
    if not isinstance(name, str) or not name.strip():
        raise ToolSurfaceError("tool schema has no stable function name")
    return name.strip()


def _schema_map(state: ToolSurfaceState | None) -> dict[str, dict[str, object]]:
    if state is None:
        return {}
    return {name: copy.deepcopy(schema) for name, schema in state.schemas}


def _literal_schema_value(path: tuple[str, ...], key: str) -> bool:
    # JSON Schema values are data, not annotation-bearing schemas. A property
    # named "default" still contains a schema and must be traversed normally.
    named_maps = {"properties", "patternProperties", "$defs", "definitions",
                  "dependentSchemas", "dependencies"}
    return key in {"default", "const", "enum", "examples"} and (
        not path or path[-1] not in named_maps)


def _description_rows(value: object, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], str]:
    rows: dict[tuple[str, ...], str] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            part = str(key)
            child_path = (*path, part)
            if _literal_schema_value(path, part):
                continue
            if part == "description" and isinstance(child, str):
                rows[child_path] = child
            else:
                rows.update(_description_rows(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            rows.update(_description_rows(child, (*path, str(index))))
    return rows


def _json_pointer(parts: Sequence[str]) -> str:
    return "/" + "/".join(
        part.replace("~", "~0").replace("/", "~1") for part in parts
    )


def _merge_stable_descriptions(
        current: object, previous: object, path: tuple[str, ...] = (),
) -> object:
    if isinstance(current, Mapping):
        prior = previous if isinstance(previous, Mapping) else {}
        merged: dict[str, object] = {}
        for raw_key, child in current.items():
            key = str(raw_key)
            prior_child = prior.get(raw_key)
            child_path = (*path, key)
            if _literal_schema_value(path, key):
                merged[key] = copy.deepcopy(child)
            elif key == "description" and isinstance(child, str):
                if isinstance(prior_child, str):
                    merged[key] = prior_child
                # Descriptions absent at first sight stay absent for the run.
            else:
                merged[key] = _merge_stable_descriptions(child, prior_child, child_path)
        if ("description" not in merged
                and isinstance(prior.get("description"), str)):
            merged["description"] = prior["description"]
        return merged
    if isinstance(current, list):
        prior_list = previous if isinstance(previous, list) else []
        return [
            _merge_stable_descriptions(
                child, prior_list[index] if index < len(prior_list) else None,
                (*path, str(index)),
            )
            for index, child in enumerate(current)
        ]
    return copy.deepcopy(current)


def _canonicalize_descriptions(value: object, path: tuple[str, ...] = ()) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if _literal_schema_value(path, key):
                result[key] = copy.deepcopy(child)
                continue
            result[key] = (
                _CANONICAL_DESCRIPTION
                if key == "description" and isinstance(child, str)
                else _canonicalize_descriptions(child, (*path, key))
            )
        return result
    if isinstance(value, list):
        return [_canonicalize_descriptions(child, (*path, str(index))) for index, child in enumerate(value)]
    return copy.deepcopy(value)


def _semantic_schema(value: object, path: tuple[str, ...] = ()) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): (copy.deepcopy(child) if _literal_schema_value(path, str(key))
                       else _semantic_schema(child, (*path, str(key))))
            for key, child in value.items()
            if not (str(key) == "description" and isinstance(child, str))
        }
    if isinstance(value, list):
        return [_semantic_schema(child, (*path, str(index))) for index, child in enumerate(value)]
    return copy.deepcopy(value)


def _wire_schema_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _guidance_rows(name: str, current: object, previous: object) -> list[str]:
    current_rows = _description_rows(current)
    previous_rows = _description_rows(previous)
    result: list[str] = []
    for path in sorted(set(current_rows) | set(previous_rows)):
        row: dict[str, object] = {"tool": name, "field": _json_pointer(path)}
        if path in current_rows:
            row["text"] = current_rows[path]
        else:
            row["present"] = False
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        result.append("[tool guidance] " + payload)
    return result


def _run_id(runtime: object, capture: object) -> str:
    for owner in (capture, runtime):
        journal = getattr(owner, "journal", None)
        value = getattr(journal, "run_id", None)
        if isinstance(value, str) and value:
            return value
    return ""


def _catalog_identity(names: Sequence[str], schemas: Mapping[str, object]) -> str:
    payload = [
        {"name": name, "schema": _semantic_schema(schemas[name])}
        for name in names if name in schemas
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _project_tools_core(
        runtime: object,
        capture: object,
        current_tools: Sequence[object],
        *,
        selected_tool_mode: str | None = None,
        selected_menu_mode: str | None = None,
        rebase_catalog: bool = False,
) -> ToolSurfaceProjection:
    """Keep current admission separate from the first-request wire catalog."""
    stable_mode = tool_mode() if selected_tool_mode is None else selected_tool_mode
    menu_selected = menu_mode() if selected_menu_mode is None else selected_menu_mode
    if stable_mode not in {"on", "shadow"}:
        raise ToolSurfaceError("tool stable mode is not on or shadow")
    if menu_selected not in _MENU_MODES:
        menu_selected = "off"
    raw_state = getattr(runtime, _TOOL_STATE_ATTR, None)
    run_id = _run_id(runtime, capture)
    if not run_id:
        raise ToolSurfaceError("request run identity is unavailable")
    previous = raw_state if isinstance(raw_state, ToolSurfaceState) else None
    if previous is not None and previous.run_id != run_id:
        previous = None
    previous_schemas = _schema_map(previous)
    order = list(previous.order) if previous is not None else []
    current_by_name: dict[str, dict[str, object]] = {}
    semantic_changes: set[str] = set()
    guidance: list[str] = []
    schemas = copy.deepcopy(previous_schemas)

    for raw_tool in current_tools:
        if not isinstance(raw_tool, Mapping):
            raise ToolSurfaceError("current tool schema is malformed")
        current = copy.deepcopy(dict(raw_tool))
        name = _tool_name(current)
        if name in current_by_name:
            raise ToolSurfaceError("current tool names are not unique")
        prior_schema = schemas.get(name)
        guidance.extend(_guidance_rows(name, current, prior_schema))
        canonical = _canonicalize_descriptions(current)
        if not isinstance(canonical, dict):
            raise ToolSurfaceError("canonical tool schema is malformed")
        if prior_schema is None:
            stable = canonical
            order.append(name)
            schemas[name] = copy.deepcopy(stable)
        else:
            stable = _merge_stable_descriptions(canonical, prior_schema)
            if _semantic_schema(stable) != _semantic_schema(prior_schema):
                semantic_changes.add(name)
                # OFF preserves the legacy state transition exactly. Enabled rebase
                # mode advances the comparison baseline with the committed identity.
                if rebase_catalog and menu_selected == "on":
                    schemas[name] = copy.deepcopy(stable)
        if not isinstance(stable, dict):
            raise ToolSurfaceError("stable tool schema is malformed")
        current_by_name[name] = copy.deepcopy(stable)

    admitted_names = tuple(name for name in order if name in current_by_name)
    if not admitted_names:
        raise ToolSurfaceError("current admitted tool menu is empty")
    admitted_tools = [copy.deepcopy(current_by_name[name]) for name in admitted_names]

    # C freezes only the non-done catalog from its first participating request.
    # The done slot is remembered separately so membership remains per-turn.
    menu_catalog_initialized = bool(previous and previous.menu_catalog_initialized)
    had_menu_catalog = menu_catalog_initialized
    menu_catalog_order = previous.menu_catalog_order if menu_catalog_initialized else ()
    menu_catalog_schemas = (
        {name: copy.deepcopy(schema) for name, schema in previous.menu_catalog_schemas}
        if menu_catalog_initialized else {}
    )
    menu_done_index = previous.menu_done_index if menu_catalog_initialized else None
    menu_catalog_version = (
        max(1, int(getattr(previous, "menu_catalog_version", 0)))
        if menu_catalog_initialized and previous is not None else 0
    )
    previous_catalog_version = menu_catalog_version if menu_catalog_initialized else None
    if menu_selected in {"on", "shadow"} and not menu_catalog_initialized:
        menu_catalog_order = tuple(name for name in current_by_name if name != "done")
        menu_catalog_schemas = {
            name: copy.deepcopy(current_by_name[name]) for name in menu_catalog_order
        }
        current_order = tuple(current_by_name)
        if "done" in current_order:
            menu_done_index = sum(1 for name in current_order[:current_order.index("done")]
                                  if name != "done")
        menu_catalog_initialized = True
        menu_catalog_version = 1

    frozen_catalog_names = tuple(menu_catalog_order)
    frozen_catalog_tools = [
        copy.deepcopy(menu_catalog_schemas[name]) for name in frozen_catalog_names
    ]
    frozen_name_set = set(frozen_catalog_names)
    unrepresented_names = tuple(
        name for name in current_by_name if name != "done" and name not in frozen_name_set
    ) if had_menu_catalog else ()
    changed_catalog_names = tuple(
        name for name in semantic_changes
        if name != "done" and name in frozen_name_set
    ) if had_menu_catalog else ()
    changed_catalog_names = tuple(
        name for name in current_by_name if name in changed_catalog_names
    )
    added_catalog_names = tuple(
        name for name in current_by_name
        if name != "done" and name not in frozen_name_set
    ) if had_menu_catalog else ()
    rebase_changed_names: tuple[str, ...] = ()
    rebase_added_names: tuple[str, ...] = ()
    rebase_from_version = previous_catalog_version
    if (rebase_catalog and menu_selected == "on"
            and previous_catalog_version is not None
            and (changed_catalog_names or added_catalog_names)):
        rebase_changed_names = changed_catalog_names
        rebase_added_names = added_catalog_names
        for name in rebase_changed_names:
            menu_catalog_schemas[name] = copy.deepcopy(current_by_name[name])
        for name in rebase_added_names:
            menu_catalog_order = (*menu_catalog_order, name)
            menu_catalog_schemas[name] = copy.deepcopy(current_by_name[name])
        menu_catalog_version += 1
        frozen_catalog_names = tuple(menu_catalog_order)
        frozen_catalog_tools = [
            copy.deepcopy(menu_catalog_schemas[name]) for name in frozen_catalog_names
        ]
        frozen_name_set = set(frozen_catalog_names)
        unrepresented_names = ()
        changed_catalog_names = ()
    changed_catalog_bytes = tuple(
        name for name in current_by_name
        if name != "done" and name in frozen_name_set
        and _wire_schema_bytes(current_by_name[name])
            != _wire_schema_bytes(menu_catalog_schemas[name])
    ) if menu_catalog_initialized else ()
    # The frozen non-done wire catalog and current dispatch admission are
    # independent authorities. Only an explicit rebase may change the catalog.
    catalog_names_list = list(frozen_catalog_names)
    catalog_tools = copy.deepcopy(frozen_catalog_tools)
    if menu_catalog_initialized and "done" in current_by_name:
        done_index = (menu_done_index if menu_done_index is not None
                      else len(catalog_names_list))
        done_index = min(max(done_index, 0), len(catalog_names_list))
        catalog_names_list.insert(done_index, "done")
        catalog_tools.insert(done_index, copy.deepcopy(current_by_name["done"]))
    catalog_names = tuple(catalog_names_list) if menu_catalog_initialized else admitted_names

    stable_mismatch_fields: list[str] = []
    if stable_mode == "shadow":
        raw_names = tuple(_tool_name(tool) for tool in current_tools if isinstance(tool, Mapping))
        if set(raw_names) != set(admitted_names):
            stable_mismatch_fields.append("tool_membership")
        elif raw_names != admitted_names:
            stable_mismatch_fields.append("tool_order")
        current_raw_by_name = {
            _tool_name(tool): dict(tool) for tool in current_tools if isinstance(tool, Mapping)
        }
        if any(
                _semantic_schema(current_raw_by_name.get(name))
                != _semantic_schema(current_by_name.get(name))
                for name in admitted_names):
            stable_mismatch_fields.append("argument_semantics")
        if any(
                _description_rows(current_raw_by_name.get(name))
                != _description_rows(current_by_name.get(name))
                for name in admitted_names):
            stable_mismatch_fields.append("tool_descriptions")

    mismatch_fields: list[str] = []
    wire_tools = admitted_tools
    if menu_selected == "shadow":
        if set(catalog_names) != set(admitted_names):
            mismatch_fields.append("tool_membership")
        elif tuple(catalog_names) != admitted_names:
            mismatch_fields.append("tool_order")
        if unrepresented_names or changed_catalog_names:
            mismatch_fields.append("catalog_schema_changed")
        if changed_catalog_bytes:
            mismatch_fields.append("catalog_schema_bytes")
    elif menu_selected == "on":
        wire_tools = catalog_tools

    candidate_state = ToolSurfaceState(
        run_id=run_id,
        order=tuple(order),
        schemas=tuple((name, copy.deepcopy(schemas[name])) for name in order),
        generation=(previous.generation + 1) if previous is not None else 1,
        menu_catalog_initialized=menu_catalog_initialized,
        menu_catalog_order=tuple(menu_catalog_order),
        menu_catalog_schemas=tuple(
            (name, copy.deepcopy(menu_catalog_schemas[name]))
            for name in menu_catalog_order
        ),
        menu_done_index=menu_done_index,
        menu_catalog_version=menu_catalog_version,
    )
    identity_schemas = {
        name: schema for name, schema in menu_catalog_schemas.items()
    }
    return ToolSurfaceProjection(
        wire_tools=copy.deepcopy(wire_tools),
        admitted_tools=copy.deepcopy(admitted_tools),
        admitted_names=admitted_names,
        catalog_names=catalog_names,
        frozen_catalog_tools=copy.deepcopy(frozen_catalog_tools),
        frozen_catalog_names=frozen_catalog_names,
        guidance=tuple(guidance),
        state=candidate_state,
        stable_mode=stable_mode,
        menu_mode=menu_selected,
        stable_shadow_mismatch_fields=tuple(stable_mismatch_fields),
        shadow_mismatch_fields=tuple(mismatch_fields),
        selected_menu_mode=menu_selected,
        rebase_from_version=rebase_from_version,
        rebase_candidate_version=menu_catalog_version if menu_catalog_initialized else None,
        rebase_changed_names=rebase_changed_names,
        rebase_added_names=rebase_added_names,
        rebase_candidate_identity=(
            _catalog_identity(frozen_catalog_names, identity_schemas)
            if menu_catalog_initialized else ""
        ),
    )


def _base_menu_projection(
        runtime: object,
        capture: object,
        current_tools: Sequence[object],
        *,
        selected_tool_mode: str,
        selected_menu_mode: str,
) -> ToolSurfaceProjection:
    """Project current admission against the catalog without applying a rebase."""
    return _project_tools_core(
        runtime, capture, current_tools,
        selected_tool_mode=selected_tool_mode,
        selected_menu_mode=selected_menu_mode,
    )


def _rebase_projection_mismatches(
        candidate: ToolSurfaceProjection,
        legacy: ToolSurfaceProjection,
) -> tuple[str, ...]:
    comparisons = (
        ("wire_tools", candidate.wire_tools, legacy.wire_tools),
        ("admitted_tools", candidate.admitted_tools, legacy.admitted_tools),
        ("admitted_names", candidate.admitted_names, legacy.admitted_names),
        ("menu_mode", candidate.menu_mode, legacy.menu_mode),
        ("selected_menu_mode", candidate.selected_menu_mode, legacy.selected_menu_mode),
        ("stable_mode", candidate.stable_mode, legacy.stable_mode),
        ("guidance", candidate.guidance, legacy.guidance),
        ("stable_shadow_mismatch_fields", candidate.stable_shadow_mismatch_fields,
         legacy.stable_shadow_mismatch_fields),
        ("menu_shadow_mismatch_fields", candidate.shadow_mismatch_fields,
         legacy.shadow_mismatch_fields),
        ("catalog_names", candidate.catalog_names, legacy.catalog_names),
        ("frozen_catalog_tools", candidate.frozen_catalog_tools, legacy.frozen_catalog_tools),
        ("frozen_catalog_names", candidate.frozen_catalog_names, legacy.frozen_catalog_names),
        ("tool_state_order", candidate.state.order, legacy.state.order),
        ("tool_state_schemas", candidate.state.schemas, legacy.state.schemas),
        ("state_generation", candidate.state.generation, legacy.state.generation),
        ("catalog_initialized", candidate.state.menu_catalog_initialized,
         legacy.state.menu_catalog_initialized),
        ("catalog_order", candidate.state.menu_catalog_order,
         legacy.state.menu_catalog_order),
        ("catalog_state", candidate.state.menu_catalog_schemas,
         legacy.state.menu_catalog_schemas),
        ("catalog_done_index", candidate.state.menu_done_index,
         legacy.state.menu_done_index),
        ("catalog_version", candidate.state.menu_catalog_version,
         legacy.state.menu_catalog_version),
        ("run_id", candidate.state.run_id, legacy.state.run_id),
        ("catalog_identity", candidate.rebase_candidate_identity,
         legacy.rebase_candidate_identity),
    )
    return tuple(field for field, left, right in comparisons if left != right)


def project_tools(
        runtime: object,
        capture: object,
        current_tools: Sequence[object],
        *,
        selected_tool_mode: str | None = None,
        selected_menu_mode: str | None = None,
        selected_rebase_mode: str | None = None,
) -> ToolSurfaceProjection:
    """Project the frozen catalog, optionally rebasing semantic identities."""
    stable_selected = tool_mode() if selected_tool_mode is None else selected_tool_mode
    menu_selected = menu_mode() if selected_menu_mode is None else selected_menu_mode
    rebase_selected = (
        menu_rebase_mode() if selected_rebase_mode is None else selected_rebase_mode
    )
    if rebase_selected not in _MENU_REBASE_MODES:
        rebase_selected = "off"
    if rebase_selected == "off":
        return replace(
            _base_menu_projection(
                runtime, capture, current_tools,
                selected_tool_mode=stable_selected,
                selected_menu_mode=menu_selected,
            ),
            rebase_mode="off",
        )

    if rebase_selected == "on":
        try:
            candidate = _project_tools_core(
                runtime, capture, current_tools,
                selected_tool_mode=stable_selected,
                selected_menu_mode=menu_selected,
                rebase_catalog=True,
            )
            return replace(candidate, rebase_mode="on")
        except Exception as error:
            # The new path is optional; retain today's projection if its candidate fails.
            legacy = _base_menu_projection(
                runtime, capture, current_tools,
                selected_tool_mode=stable_selected,
                selected_menu_mode=menu_selected,
            )
            return replace(
                legacy,
                rebase_mode="on",
                rebase_failed_open_reason=type(error).__name__,
            )

    try:
        candidate = _project_tools_core(
            runtime, capture, current_tools,
            selected_tool_mode=stable_selected,
            selected_menu_mode=menu_selected,
            rebase_catalog=True,
        )
        legacy = _base_menu_projection(
            runtime, capture, current_tools,
            selected_tool_mode=stable_selected,
            selected_menu_mode=menu_selected,
        )
        mismatch_fields = _rebase_projection_mismatches(candidate, legacy)
        return replace(
            legacy,
            rebase_mode="shadow",
            rebase_from_version=candidate.rebase_from_version,
            rebase_candidate_version=candidate.state.menu_catalog_version,
            rebase_changed_names=candidate.rebase_changed_names,
            rebase_added_names=candidate.rebase_added_names,
            rebase_candidate_identity=candidate.rebase_candidate_identity,
            rebase_shadow_mismatch_fields=mismatch_fields,
        )
    except Exception as error:
        legacy = _base_menu_projection(
            runtime, capture, current_tools,
            selected_tool_mode=stable_selected,
            selected_menu_mode=menu_selected,
        )
        return replace(
            legacy,
            rebase_mode="shadow",
            rebase_shadow_mismatch_fields=("candidate_projection",),
            rebase_failed_open_reason=type(error).__name__,
        )


def append_volatile_lines(
        body: dict[str, object], guidance: Sequence[str], *, availability: Sequence[str] | None = None,
) -> None:
    """Append dynamic guidance; the shared final-message composer owns GREENFACT order."""
    lines = list(guidance)
    if availability is not None:
        lines.append("[tool availability] " + ", ".join(str(name) for name in availability))
    if not lines:
        return
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ToolSurfaceError("tool guidance has no request message array")
    if any(not isinstance(message, Mapping) for message in messages):
        raise ToolSurfaceError("tool guidance request message is malformed")
    present_lines = {
        line for message in messages
        if isinstance(message, Mapping) and isinstance(message.get("content"), str)
        for line in message["content"].splitlines()
    }
    fresh_lines = [line for line in lines if line not in present_lines]
    if not fresh_lines:
        return
    body["messages"] = [copy.deepcopy(dict(message)) for message in messages]
    body["messages"].append({"role": "user", "content": "\n".join(fresh_lines)})


def make_pending(
        runtime: object,
        projection: ToolSurfaceProjection,
        *,
        request_id: object = None,
) -> ToolSurfacePending:
    previous = getattr(runtime, _TOOL_STATE_ATTR, None)
    if not isinstance(previous, ToolSurfaceState) or previous.run_id != projection.state.run_id:
        previous = None
    return ToolSurfacePending(
        runtime=runtime,
        previous_state=previous,
        candidate_state=projection.state,
        request_id=str(request_id) if request_id is not None else None,
        stable_mode=projection.stable_mode,
        menu_mode=projection.menu_mode,
        selected_menu_mode=projection.selected_menu_mode,
        rebase_mode=projection.rebase_mode,
        rebase_from_version=projection.rebase_from_version,
        rebase_candidate_version=projection.rebase_candidate_version,
        rebase_changed_names=projection.rebase_changed_names,
        rebase_added_names=projection.rebase_added_names,
        rebase_candidate_identity=projection.rebase_candidate_identity,
        rebase_shadow_mismatch_fields=projection.rebase_shadow_mismatch_fields,
        rebase_failed_open_reason=projection.rebase_failed_open_reason,
        menu_freeze_failed_open_reason=projection.menu_freeze_failed_open_reason,
    )


def _emit(journal: object | None, kind: str, **fields: object) -> None:
    emit = getattr(journal, "emit", None)
    if callable(emit):
        try:
            emit(kind, producer=__name__, dispatch=False, **fields)
        except Exception:
            return


def fail_open_pending(capture: object, journal: object | None, *, reason: str) -> None:
    pending = getattr(capture, _PENDING_ATTR, None)
    if not isinstance(pending, ToolSurfacePending):
        return
    setattr(capture, _PENDING_ATTR, None)
    _emit(journal, "stateframe_tool_stable_failed_open", request_id=pending.request_id, reason=reason)
    if pending.menu_mode != "off":
        _emit(journal, "stateframe_menu_freeze_failed_open", request_id=pending.request_id, reason=reason)
    if pending.rebase_mode != "off":
        _emit(
            journal, "stateframe_menu_freeze_rebase_failed_open",
            request_id=pending.request_id, mode=pending.rebase_mode, reason=reason,
        )


def commit_after_response(capture: object, journal: object | None) -> None:
    """Commit first-seen tool identity only after response_received is durable."""
    pending = getattr(capture, _PENDING_ATTR, None)
    if not isinstance(pending, ToolSurfacePending):
        return
    runtime = pending.runtime
    current = getattr(runtime, _TOOL_STATE_ATTR, None)
    if pending.previous_state is None:
        if (isinstance(current, ToolSurfaceState)
                and current.run_id == pending.candidate_state.run_id):
            fail_open_pending(capture, journal, reason="concurrent_tool_state_advance")
            return
    elif (not isinstance(current, ToolSurfaceState)
          or current.run_id != pending.previous_state.run_id
          or current.generation != pending.previous_state.generation):
        fail_open_pending(capture, journal, reason="concurrent_tool_state_advance")
        return
    try:
        setattr(runtime, _TOOL_STATE_ATTR, copy.deepcopy(pending.candidate_state))
    except (AttributeError, TypeError, ValueError):
        fail_open_pending(capture, journal, reason="tool_state_commit_failed")
        return
    setattr(capture, _PENDING_ATTR, None)
    facts = {
        "request_id": pending.request_id,
        "stable_mode": pending.stable_mode,
        "menu_mode": pending.menu_mode,
        "tool_count": (
            len(pending.candidate_state.menu_catalog_order)
            if (pending.menu_mode != "off"
                and pending.candidate_state.menu_catalog_initialized)
            else len(pending.candidate_state.order)
        ),
        "generation": pending.candidate_state.generation,
    }
    _emit(journal, "stateframe_tool_stable_committed", **facts)
    if pending.menu_mode != "off":
        _emit(journal, "stateframe_menu_freeze_committed", **facts)
    if pending.rebase_mode != "off":
        previous_version = pending.rebase_from_version
        committed_version = pending.candidate_state.menu_catalog_version
        rebased = (
            pending.rebase_mode == "on"
            and previous_version is not None
            and committed_version > previous_version
        )
        outcome = (
            "failed_open" if pending.rebase_failed_open_reason else
            "shadow_mismatch" if pending.rebase_shadow_mismatch_fields else
            "rebased" if rebased else
            "unchanged"
        )
        rebase_facts = {
            "request_id": pending.request_id,
            "mode": pending.rebase_mode,
            "outcome": outcome,
            "previous_catalog_version": previous_version,
            "catalog_version": committed_version,
            "candidate_catalog_version": pending.rebase_candidate_version,
            "catalog_identity": pending.rebase_candidate_identity,
            "changed_tools": list(pending.rebase_changed_names),
            "added_tools": list(pending.rebase_added_names),
            "cache_miss": bool(rebased),
            "would_cache_miss": bool(
                pending.rebase_mode == "shadow"
                and previous_version is not None
                and pending.rebase_candidate_version is not None
                and pending.rebase_candidate_version > previous_version
            ),
        }
        evaluated_facts = {
            **rebase_facts,
            "cache_miss": False,
            "cache_miss_candidate": bool(rebased),
        }
        _emit(journal, "stateframe_menu_freeze_rebase_evaluated", **evaluated_facts)
        if rebased:
            _emit(journal, "stateframe_menu_freeze_rebased", **rebase_facts)
        for field in pending.rebase_shadow_mismatch_fields:
            _emit(
                journal, "stateframe_menu_freeze_rebase_shadow_mismatch",
                field=field,
                **rebase_facts,
            )
        if pending.rebase_failed_open_reason:
            _emit(
                journal, "stateframe_menu_freeze_rebase_failed_open",
                request_id=pending.request_id,
                mode=pending.rebase_mode,
                reason=pending.rebase_failed_open_reason,
                previous_catalog_version=previous_version,
                candidate_catalog_version=pending.rebase_candidate_version,
            )


__all__ = [
    "MENU_MODE_ENV", "MENU_REBASE_MODE_ENV", "TOOL_MODE_ENV",
    "ToolSurfaceError", "ToolSurfacePending",
    "ToolSurfaceProjection", "ToolSurfaceState", "append_volatile_lines",
    "commit_after_response", "fail_open_pending", "make_pending", "menu_mode",
    "menu_rebase_mode", "project_tools", "tool_mode",
]
