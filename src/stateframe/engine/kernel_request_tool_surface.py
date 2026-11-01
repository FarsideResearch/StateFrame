"""Author the request's final provider tool surface at the send boundary."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence


_MISSING = object()
SURFACE_SCHEMA = "stateframe-final-tool-surface/1"
REQUEST_TOOL_FLOOR_ENV = "STATEFRAME_REQUEST_TOOL_FLOOR"
TOOL_MENU_MEMO_ENV = "STATEFRAME_TOOL_MENU_MEMO"
_REQUEST_TOOL_FLOOR = ("read", "grep", "edit", "str_replace", "file_delete", "file_move", "shell", "run_tests", "done")
_REQUEST_TOOL_FLOOR_MAX_SCHEMA_BYTES = 131072
_RUNTIME_FIELDS = (
    "_greenlock_frozen_menu",
    "_greenlock_menu_authority_upstream",
    "completion_menu",
    "offered_tools",
    "_question_done_available",
    "_completion_owner_request_contract",
    "_completion_owner_dispatch_tools",
    "_stateframe_current_request_tool_names",
    "_completion_owner_owner_blocks",
    "_completion_owner_actionable",
    "_request_tool_floor_schema_cache_run",
    "_request_tool_floor_schema_cache",
)
_CAPTURE_FIELDS = (
    "_stateframe_tool_surface_pending",
    "_stateframe_tool_guidance_pending",
    "final_tool_surface",
    "request_action_contract",
    "request_budget_final_wire",
    "_stateframe_tool_menu_memo_frame_digest",
    "_stateframe_tool_menu_memo_frame_snapshot",
    "_stateframe_tool_menu_memo_key",
    "_stateframe_tool_menu_memo_surface_bytes",
    "_stateframe_tool_menu_memo_shadow_match",
)


class _SurfaceUnavailable(Exception):
    """A safe, bounded signal that the current path should continue unchanged."""


def _tool_names(tools: Sequence[object]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if isinstance(name, str):
            names.append(name.strip())
        else:
            names.append("")
    return names


def request_tool_floor_enabled(runtime: object | None = None) -> bool:
    """Return the strict, default-off request-final tool-floor switch."""
    configured = (
        getattr(runtime, "stateframe_request_tool_floor_mode", None)
        if runtime is not None else None
    )
    value = str(configured if configured is not None else
                os.environ.get(REQUEST_TOOL_FLOOR_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{REQUEST_TOOL_FLOOR_ENV} must be off or on")
    return value == "on"


def _schema_index(tools: object) -> dict[str, Mapping[str, object]]:
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return {}
    indexed: dict[str, Mapping[str, object]] = {}
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        names = _tool_names((tool,))
        if names and names[0] and names[0] not in indexed:
            indexed[names[0]] = tool
    return indexed


def _canonical_schema_bytes(tool: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(tool), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _request_tool_floor_schemas(
        runtime: object, incoming_tools: Sequence[object],
) -> tuple[tuple[str, ...], dict[str, dict[str, object]], str]:
    """Cache bounded canonical base schemas for one runtime run."""
    base_tools = getattr(runtime, "_completion_owner_full_tools", None)
    if not isinstance(base_tools, Sequence) or isinstance(base_tools, (str, bytes, bytearray)):
        base_tools = getattr(runtime, "_readiness_full_tools", None)
    if not isinstance(base_tools, Sequence) or isinstance(base_tools, (str, bytes, bytearray)):
        base_tools = getattr(runtime, "available_tools", None)
    if not isinstance(base_tools, Sequence) or isinstance(base_tools, (str, bytes, bytearray)):
        base_tools = incoming_tools

    base_by_name = _schema_index(base_tools)
    current_by_name = _schema_index(incoming_tools)
    floor_names = list(_REQUEST_TOOL_FLOOR)
    if "list_files" in base_by_name:
        floor_names.append("list_files")
    from kernel_code_bookmarks import enabled as bookmarks_enabled
    if bookmarks_enabled() and getattr(runtime, "_read_lockout_notice", None):
        from kernel_read_lockout import _PURE_SOURCE_TOOLS
        floor_names = [name for name in floor_names if name not in _PURE_SOURCE_TOOLS]

    journal = getattr(runtime, "journal", None)
    run_id = getattr(journal, "run_id", None)
    if run_id in (None, ""):
        run_id = getattr(runtime, "run_id", None)
    run_key = str(run_id) if run_id not in (None, "") else f"runtime:{id(runtime)}"
    cached_run = getattr(runtime, "_request_tool_floor_schema_cache_run", None)
    cache = getattr(runtime, "_request_tool_floor_schema_cache", None)
    if cached_run != run_key or not isinstance(cache, dict):
        cache = {}
        runtime._request_tool_floor_schema_cache_run = run_key
        runtime._request_tool_floor_schema_cache = cache

    schemas: dict[str, dict[str, object]] = {}
    digest = hashlib.sha256()
    for name in floor_names:
        payload = cache.get(name)
        if not isinstance(payload, bytes):
            source = base_by_name.get(name) or current_by_name.get(name)
            if not isinstance(source, Mapping):
                raise _SurfaceUnavailable("capability_floor_schema_missing")
            payload = _canonical_schema_bytes(source)
            if len(payload) > _REQUEST_TOOL_FLOOR_MAX_SCHEMA_BYTES:
                raise _SurfaceUnavailable("capability_floor_schema_too_large")
            cache[name] = payload
        try:
            schema = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _SurfaceUnavailable("capability_floor_schema_invalid") from error
        if not isinstance(schema, dict) or _tool_names((schema,)) != [name]:
            raise _SurfaceUnavailable("capability_floor_schema_invalid")
        schemas[name] = schema
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return tuple(floor_names), schemas, "sha256:" + digest.hexdigest()


def _apply_request_tool_floor(
        tools: Sequence[object], floor_names: Sequence[str],
        floor_schemas: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        names = _tool_names((tool,))
        name = names[0] if names else ""
        schema = floor_schemas.get(name)
        if schema is not None:
            result.append(copy.deepcopy(dict(schema)))
            seen.add(name)
        else:
            result.append(copy.deepcopy(dict(tool)))
    for name in floor_names:
        if name not in seen:
            result.append(copy.deepcopy(dict(floor_schemas[name])))
    return result


def _emit_capability_floor_restored(
        capture: object, runtime: object, request_id: str,
        removed_by: Mapping[str, str],
) -> None:
    journal = getattr(capture, "journal", None) or getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    for tool, producer in removed_by.items():
        try:
            emit(
                "capability_floor_restored", producer=__name__, request_id=request_id,
                tool=tool, removed_by=producer, dispatch=False,
            )
        except Exception:
            # Journal failures never change the final request surface or send path.
            continue


def _snapshot(owner: object, fields: Sequence[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in fields:
        value = getattr(owner, field, _MISSING)
        if value is _MISSING:
            result[field] = _MISSING
            continue
        try:
            result[field] = copy.deepcopy(value)
        except Exception:
            result[field] = value
    return result


def _restore(owner: object, state: Mapping[str, object]) -> None:
    for field, value in state.items():
        if value is _MISSING:
            try:
                delattr(owner, field)
            except Exception:
                pass
        else:
            try:
                setattr(owner, field, value)
            except Exception:
                pass


def _record_failure(capture: object, runtime: object, request_id: str,
                    reason: str, error: BaseException | None = None) -> None:
    try:
        journal = getattr(capture, "journal", None) or getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if not callable(emit):
            return
        fields: dict[str, object] = {
            "producer": __name__,
            "request_id": request_id,
            "reason_code": reason,
            "dispatch": False,
        }
        if error is not None:
            fields["error_type"] = type(error).__name__
        emit("final_tool_surface_authoring_failed", **fields)
    except Exception:
        pass


def _surface_matches_frozen(frozen: object, tools: list[dict[str, object]],
                            names: Sequence[str]) -> bool:
    if not isinstance(frozen, Mapping):
        return False
    raw_names = frozen.get("allow_set")
    raw_tools = frozen.get("tools")
    return (
        isinstance(raw_names, Sequence)
        and not isinstance(raw_names, (str, bytes, bytearray))
        and tuple(str(name) for name in raw_names) == tuple(names)
        and raw_tools == tools
    )


def _tool_menu_memo_mode(runtime: object | None = None) -> str:
    try:
        configured = getattr(runtime, "stateframe_tool_menu_memo_mode", None)
        value = str(configured if configured is not None else
                    os.environ.get(TOOL_MENU_MEMO_ENV, "off")).strip().lower()
    except Exception:
        return "off"
    return value if value in {"off", "shadow", "on"} else "off"


def _tool_surface_bytes(tools: object) -> bytes | None:
    if not isinstance(tools, list):
        return None
    try:
        return json.dumps(
            tools, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        return None


def _tool_menu_frame_digest(messages: object) -> str | None:
    try:
        from kernel_measurement import digest

        return digest(messages)
    except Exception:
        return None


def _tool_menu_memo_key(
        runtime: object, body: Mapping[str, object], *, request_id: str,
        tool_schema_digest: str, candidate_exists: bool, done_enabled: bool,
        floor_enabled: bool, floor_schema_digest: str | None,
        frame_digest: object = _MISSING,
) -> str | None:
    """Bind final-surface reuse to the exact frame, evidence, artifact and shared decision."""
    try:
        from kernel_check_receipts import current_wire_revision
        from kernel_done_readiness_authority import cached_decision
        from kernel_measurement import digest

        journal = getattr(runtime, "journal", None)
        receipt_generation = getattr(journal, "receipt_generation", None)
        product_revision = current_wire_revision(runtime)
        if type(receipt_generation) is not int or receipt_generation < 0:
            return None
        if (not isinstance(product_revision, str) or not product_revision
                or product_revision == "UNKNOWN"):
            return None
        shared = cached_decision(runtime)
        shared_data: object = None
        if shared is not None:
            to_dict = getattr(shared, "to_dict", None)
            if not callable(to_dict):
                return None
            shared_data = {
                "key": list(getattr(shared, "key", ())),
                "decision": to_dict(),
            }
        bound_frame_digest = (
            digest(body.get("messages")) if frame_digest is _MISSING else frame_digest
        )
        if not isinstance(bound_frame_digest, str) or not bound_frame_digest:
            return None
        return digest({
            "schema": "stateframe-tool-menu-memo-key/1",
            "request_id": request_id,
            "frame_digest": bound_frame_digest,
            "tool_surface_digest": tool_schema_digest,
            "receipt_generation": receipt_generation,
            "product_revision": product_revision,
            "shared_done_decision": shared_data,
            "candidate_exists": candidate_exists,
            "done_enabled": done_enabled,
            "capability_floor_enabled": floor_enabled,
            "capability_floor_schema_digest": floor_schema_digest,
        })
    except Exception:
        return None


def restore_authored_action_contract(capture: object, contract: object) -> None:
    """Put back a contract that author_final_tool_surface already produced for an earlier snapshot.

    N1 one-snapshot yield (STATEFRAME_ONE_SNAPSHOT) restores the FIRST qualified wire bytes when a later
    revalidation disagrees; the action contract must be restored with those bytes so menu == gate == wire
    still holds. This is a restore of authored output, never a new surface, so it lives beside the single
    author and is listed in the donewire guard's allowed writers.
    """
    try:
        capture.request_action_contract = copy.deepcopy(contract)
    except Exception:
        capture.request_action_contract = contract


def _store_tool_menu_memo_state(
        capture: object,
        mode: str,
        *,
        frame_digest: str | None,
        frame_snapshot: object,
        memo_key: str | None,
        surface: Mapping[str, object],
        shadow_match: bool | None,
) -> None:
    """Keep request-local menu hints only when the existing memo switch is enabled."""
    if mode == "off":
        return
    try:
        capture._stateframe_tool_menu_memo_frame_digest = frame_digest
        if frame_snapshot is _MISSING:
            try:
                delattr(capture, "_stateframe_tool_menu_memo_frame_snapshot")
            except AttributeError:
                pass
        else:
            capture._stateframe_tool_menu_memo_frame_snapshot = frame_snapshot
        capture._stateframe_tool_menu_memo_key = memo_key
        capture._stateframe_tool_menu_memo_surface_bytes = _tool_surface_bytes(
            surface["tools"],
        )
        capture._stateframe_tool_menu_memo_shadow_match = shadow_match
    except Exception:
        pass


def _emit_tool_menu_memo_shadow_mismatch(
        capture: object, runtime: object, request_id: str, field: str,
) -> None:
    try:
        journal = getattr(capture, "journal", None) or getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if callable(emit):
            emit(
                "tool_menu_memo_shadow_mismatch",
                producer=__name__, request_id=request_id, field=field, dispatch=False,
            )
    except Exception:
        # Shadow telemetry must not change final-surface authoring or admission.
        pass


def _clear_current_request_tool_names(
        capture: object, runtime: object | None, request_key: str, *, side_call: bool,
) -> bool:
    if runtime is None or side_call:
        return True
    try:
        runtime._stateframe_current_request_tool_names = ()
        return True
    except Exception as error:  # noqa: BLE001 - a stale surface must never authorize this request
        _record_failure(capture, runtime, request_key, "request_surface_state_clear_failed", error)
        return False


def author_final_tool_surface(
        capture: object,
        runtime: object | None,
        body: dict[str, object],
        *,
        request_id: object,
        tools_override: object = None,
        side_call: bool = False,
        bind_current_surface: bool = False,
        stabilize_cache_tools: bool = False,
) -> Mapping[str, object] | None:
    """Project once, then bind the exact schemas, names and digest to each gate."""
    request_key = str(request_id or "")
    if not _clear_current_request_tool_names(
            capture, runtime, request_key, side_call=side_call):
        return None
    if not isinstance(body, dict):
        return None
    original_body_had_tools = "tools" in body
    original_body_tools = body.get("tools", _MISSING)
    override_tools: list[object] | None = None
    if tools_override is not None:
        if not isinstance(tools_override, list):
            return None
        try:
            # Keep the prior recovery path's shallow list copy as the fail-open body.
            override_tools = list(tools_override)
            body["tools"] = override_tools
        except Exception as error:
            _record_failure(capture, runtime, request_key, "override_copy_failed", error)
            return None
    if runtime is None or side_call:
        return None
    try:
        from kernel_done_is_submit_authority import (
            candidate_exists as done_candidate_exists,
            enabled as done_submit_enabled,
            project_final_candidate_tools,
        )

        done_enabled = done_submit_enabled(runtime)
        floor_enabled = request_tool_floor_enabled(runtime)
        if not done_enabled and not floor_enabled and not bind_current_surface:
            return None
        raw_tools = tools_override if tools_override is not None else body.get("tools")
        if (not isinstance(raw_tools, list)
                or any(not isinstance(tool, Mapping) for tool in raw_tools)):
            return None
        incoming_tools = [copy.deepcopy(dict(tool)) for tool in raw_tools]
        runtime_before = _snapshot(runtime, _RUNTIME_FIELDS)
        capture_before = _snapshot(capture, _CAPTURE_FIELDS)
        candidate_exists = bool(done_candidate_exists(runtime)) if done_enabled else False
        incoming_names = set(_tool_names(incoming_tools))
        floor_names: tuple[str, ...] = ()
        floor_schemas: dict[str, dict[str, object]] = {}
        floor_schema_digest: str | None = None
        if floor_enabled:
            floor_names, floor_schemas, floor_schema_digest = _request_tool_floor_schemas(
                runtime, incoming_tools,
            )
        from kernel_wire_truth_gate import schema_digest
        incoming_digest = schema_digest(incoming_tools)
        previous = getattr(capture, "final_tool_surface", None)
        memo_surface_bytes = getattr(capture, "_stateframe_tool_menu_memo_surface_bytes", None)
        previous_surface_bytes = (
            _tool_surface_bytes(previous.get("tools"))
            if isinstance(previous, Mapping) else None
        )
        memo_mode = _tool_menu_memo_mode(runtime)
        memo_frame_digest: str | None = None
        memo_frame_snapshot: object = _MISSING
        if memo_mode != "off":
            try:
                cached_frame_snapshot = getattr(
                    capture, "_stateframe_tool_menu_memo_frame_snapshot", _MISSING,
                )
                cached_frame_digest = getattr(
                    capture, "_stateframe_tool_menu_memo_frame_digest", None,
                )
                current_messages = body.get("messages")
                if (cached_frame_snapshot is not _MISSING
                        and isinstance(cached_frame_digest, str)
                        and cached_frame_snapshot == current_messages):
                    memo_frame_digest = cached_frame_digest
                    memo_frame_snapshot = cached_frame_snapshot
                else:
                    memo_frame_digest = _tool_menu_frame_digest(current_messages)
                    if memo_frame_digest is not None:
                        memo_frame_snapshot = copy.deepcopy(current_messages)
            except Exception:
                memo_frame_digest = None
                memo_frame_snapshot = _MISSING
        memo_key = _tool_menu_memo_key(
            runtime, body, request_id=request_key,
            tool_schema_digest=incoming_digest, candidate_exists=candidate_exists,
            done_enabled=done_enabled, floor_enabled=floor_enabled,
            floor_schema_digest=floor_schema_digest,
            frame_digest=memo_frame_digest,
        ) if memo_mode != "off" else None
        previous_memo_key = getattr(capture, "_stateframe_tool_menu_memo_key", None)
        memo_previous_eligible = (
            memo_key is not None and previous_memo_key == memo_key
            and isinstance(previous, Mapping)
            and previous.get("schema") == SURFACE_SCHEMA
            and previous.get("request_id") == request_key
            and previous.get("candidate_exists") is candidate_exists
            and previous.get("capability_floor_enabled", False) is floor_enabled
            and incoming_digest in {
                previous.get("source_tool_schema_digest"),
                previous.get("tool_schema_digest"),
            }
            and isinstance(previous.get("tools"), list)
        )
        memo_previous_matches = (
            memo_previous_eligible
            and isinstance(memo_surface_bytes, bytes)
            and memo_surface_bytes == previous_surface_bytes
        )
        legacy_previous_matches = (
            isinstance(previous, Mapping)
            and previous.get("schema") == SURFACE_SCHEMA
            and previous.get("request_id") == request_key
            and previous.get("candidate_exists") is candidate_exists
            and previous.get("capability_floor_enabled", False) is floor_enabled
            and incoming_digest in {
                previous.get("source_tool_schema_digest"),
                previous.get("tool_schema_digest"),
            }
            and isinstance(previous.get("tools"), list)
        )
        reuse_previous = (
            (memo_mode == "on" and memo_previous_matches)
            or (memo_mode == "off" and legacy_previous_matches)
        )
        shadow_match: bool | None = None

        removed_by: dict[str, str] = {}
        greenlock_projection_ran = False
        if reuse_previous:
            final_tools = copy.deepcopy(previous["tools"])
            after_candidate_names = set(_tool_names(final_tools))
        else:
            final_tools = incoming_tools
            if candidate_exists:
                final_tools, missing_tools = project_final_candidate_tools(
                    runtime,
                    final_tools,
                    request_id=request_key,
                    full_tools=getattr(runtime, "_completion_owner_full_tools", None),
                )
                if missing_tools:
                    raise _SurfaceUnavailable("candidate_tool_missing")
            after_candidate_names = set(_tool_names(final_tools))

            lock = getattr(runtime, "green_lock", None)
            frozen = getattr(runtime, "_greenlock_frozen_menu", None)
            needs_menu_finalize = candidate_exists or (
                isinstance(frozen, Mapping)
                and not _surface_matches_frozen(
                    frozen, final_tools, _tool_names(final_tools),
                )
            )
            if needs_menu_finalize:
                from kernel_greenlock_menu_authority import (
                    applies, finalize_wire_menu_authority,
                )
                if applies(lock):
                    finalized_names = finalize_wire_menu_authority(runtime, final_tools)
                    greenlock_projection_ran = True
                    if finalized_names and tuple(finalized_names) != tuple(_tool_names(final_tools)):
                        raise _SurfaceUnavailable("greenlock_projection_unbound")

        if floor_enabled:
            before_floor_names = set(_tool_names(final_tools))
            for name in floor_names:
                if name not in incoming_names:
                    removed_by[name] = "upstream_menu_projection"
                elif name not in after_candidate_names:
                    removed_by[name] = "done_is_submit_candidate_projection"
                elif name not in before_floor_names:
                    removed_by[name] = (
                        "greenlock_menu_projection"
                        if greenlock_projection_ran else "pre_request_menu_projection"
                    )
            final_tools = _apply_request_tool_floor(final_tools, floor_names, floor_schemas)
            final_by_name = _schema_index(final_tools)
            for name in floor_names:
                actual = final_by_name.get(name)
                if not isinstance(actual, Mapping) or dict(actual) != floor_schemas[name]:
                    raise _SurfaceUnavailable("capability_floor_not_applied")
                cached_bytes = runtime._request_tool_floor_schema_cache.get(name)
                serialized = json.dumps(
                    dict(actual), ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")
                if serialized != cached_bytes:
                    raise _SurfaceUnavailable("capability_floor_schema_bytes_unbound")

        if memo_mode == "shadow" and memo_previous_eligible:
            try:
                output_matches = previous["tools"] == final_tools
                cache_integrity_matches = (
                    isinstance(memo_surface_bytes, bytes)
                    and memo_surface_bytes == previous_surface_bytes
                )
                shadow_match = output_matches and cache_integrity_matches
                if not output_matches:
                    _emit_tool_menu_memo_shadow_mismatch(
                        capture, runtime, request_key, "tools",
                    )
                if not cache_integrity_matches:
                    _emit_tool_menu_memo_shadow_mismatch(
                        capture, runtime, request_key, "cached_surface_integrity",
                    )
            except Exception:
                shadow_match = False
                _emit_tool_menu_memo_shadow_mismatch(
                    capture, runtime, request_key, "comparison",
                )

        if stabilize_cache_tools:
            from kernel_cache_layout_candidate import prepare_stable_tools
            final_tools = prepare_stable_tools(capture, runtime, final_tools, body)
        from kernel_read_lockout import filter_tools as filter_locked_reads
        final_tools = filter_locked_reads(runtime, final_tools)
        names = _tool_names(final_tools)
        if (not names or any(not name for name in names)
                or len(names) != len(set(names))):
            raise _SurfaceUnavailable("invalid_tool_surface")
        digest = schema_digest(final_tools)
        surface: dict[str, object] = {
            "schema": SURFACE_SCHEMA,
            "request_id": request_key,
            "candidate_exists": candidate_exists,
            "source_tool_schema_digest": incoming_digest,
            "tools": copy.deepcopy(final_tools),
            "names": list(names),
            "tool_schema_digest": digest,
        }
        if floor_enabled:
            surface["capability_floor_enabled"] = True
            surface["capability_floor_schema_digest"] = floor_schema_digest

        # The body and every authority below are consumers of this one surface.
        body["tools"] = copy.deepcopy(surface["tools"])
        runtime._completion_owner_dispatch_tools = copy.deepcopy(surface["tools"])
        # Ownerless completion reads this finalized request-local menu directly.
        runtime._stateframe_current_request_tool_names = tuple(names)
        runtime.completion_menu = tuple(names)
        runtime.offered_tools = tuple(names)
        runtime._question_done_available = "done" in names
        allocation = getattr(capture, "frame_request_allocation", None)
        if allocation is not None:
            from kernel_frame_optional_allocation import FrameAllocationReplanNeeded, identity
            allocation.validate()
            if identity(surface["tools"]) != allocation.capacity["tool_schema_digest"]:
                raise FrameAllocationReplanNeeded("final tool schema changed after frame allocation")
        from kernel_completion_owner import bind_final_tool_surface
        if not bind_final_tool_surface(
                runtime, capture, surface, messages=body.get("messages")):
            raise _SurfaceUnavailable("completion_owner_contract_unbound")

        frozen = getattr(runtime, "_greenlock_frozen_menu", None)
        from kernel_greenlock_menu_authority import canonical_action
        menu_payload: dict[str, object] = {
            "schema": "stateframe-greenlock-final-menu/1",
            "allow_set": list(surface["names"]),
            "tools": copy.deepcopy(surface["tools"]),
        }
        if isinstance(frozen, Mapping) and "done_decision_key" in frozen:
            menu_payload["done_decision_key"] = copy.deepcopy(frozen["done_decision_key"])
        menu_bytes = json.dumps(
            menu_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        runtime._greenlock_frozen_menu = {
            **menu_payload,
            "digest": "sha256:" + hashlib.sha256(menu_bytes).hexdigest(),
        }
        runtime._greenlock_menu_authority_upstream = frozenset(
            canonical_action(name) for name in names if canonical_action(name)
        )

        receipt = getattr(capture, "request_budget_final_wire", None)
        budget = dict(receipt) if isinstance(receipt, Mapping) else {}
        budget.update({
            "tool_names": list(surface["names"]),
            "tool_schema_digest": digest,
        })
        if floor_enabled and floor_schema_digest is not None:
            budget["capability_floor_schema_digest"] = floor_schema_digest
        capture.request_budget_final_wire = budget
        capture.final_tool_surface = surface

        if isinstance(getattr(capture, "request_action_contract", None), Mapping):
            from kernel_final_wire_budget import reconcile_frozen_budget
            budget = reconcile_frozen_budget(capture, runtime, body)
            if (budget.get("tool_schema_digest") != digest
                    or tuple(budget.get("tool_names", ())) != tuple(names)):
                raise _SurfaceUnavailable("request_budget_surface_unbound")
        if floor_enabled and removed_by:
            _emit_capability_floor_restored(capture, runtime, request_key, removed_by)
        if memo_mode != "off":
            try:
                final_messages = body.get("messages")
                if (memo_frame_snapshot is _MISSING
                        or memo_frame_snapshot != final_messages):
                    memo_frame_digest = _tool_menu_frame_digest(final_messages)
                    memo_frame_snapshot = (
                        copy.deepcopy(final_messages)
                        if memo_frame_digest is not None else _MISSING
                    )
            except Exception:
                memo_frame_digest = None
                memo_frame_snapshot = _MISSING
            memo_key = _tool_menu_memo_key(
                runtime, body, request_id=request_key,
                tool_schema_digest=incoming_digest,
                candidate_exists=candidate_exists,
                done_enabled=done_enabled,
                floor_enabled=floor_enabled,
                floor_schema_digest=floor_schema_digest,
                frame_digest=memo_frame_digest,
            )
        _store_tool_menu_memo_state(
            capture, memo_mode,
            frame_digest=memo_frame_digest,
            frame_snapshot=memo_frame_snapshot,
            memo_key=memo_key,
            surface=surface,
            shadow_match=shadow_match,
        )
        return surface
    except _SurfaceUnavailable as error:
        _restore(runtime, runtime_before if "runtime_before" in locals() else {})
        _restore(capture, capture_before if "capture_before" in locals() else {})
        if tools_override is not None:
            body["tools"] = override_tools if override_tools is not None else tools_override
        elif original_body_had_tools:
            body["tools"] = original_body_tools
        else:
            body.pop("tools", None)
        _record_failure(capture, runtime, request_key, str(error))
        return None
    except Exception as error:  # N1: author failure must not become a send-time exception.
        _restore(runtime, runtime_before if "runtime_before" in locals() else {})
        _restore(capture, capture_before if "capture_before" in locals() else {})
        if tools_override is not None:
            body["tools"] = override_tools if override_tools is not None else tools_override
        elif original_body_had_tools:
            body["tools"] = original_body_tools
        else:
            body.pop("tools", None)
        _record_failure(capture, runtime, request_key, "surface_authoring_failed", error)
        from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
        if isinstance(error, FrameAllocationReplanNeeded):
            raise
        return None
