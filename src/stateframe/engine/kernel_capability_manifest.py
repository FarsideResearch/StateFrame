"""Opt-in R-014 manifest for live request and dispatch capabilities.

Importing this module is inert unless ``STATEFRAME_CAPABILITY_MANIFEST=on``.
The entries reuse the effect-boundary ``CapabilityEntry`` contract; the
runtime declares them at startup and appends observed status only at live
menu/request boundaries when this lane's switch is enabled.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

from kernel_history_spool import history, legacy
from kernel_capability_projection import CapabilityProjection, JournalPrefix, fold_menu
from kernel_measurement import profile_active_host_span

from kernel_effect_boundary import (
    CapabilityEntry,
    ComponentKind,
    ConsequenceType,
    capability_manifest as _registered_capabilities,
    register_capability,
)


MODE_ENV = "STATEFRAME_CAPABILITY_MANIFEST"
SCHEMA = "stateframe-capability-manifest/1"
STATUS_SCHEMA = "stateframe-capability-status/1"
STATUS_EVENT = "capability_manifest_status"
_MENU_EVENT = "green_lock_menu_authority"
_ASSIGNMENT_EVENT = "model_router_assignment"
_HANDOFF_HEADER = "[StateFrame model handoff]"
_NOTICE_MARKERS = ("[unverified checks]", "[coverage reminders]",
                   "[Named-layer coverage | turn 1]")
_BOUNDARY_BY_CAPABILITY = {
    "menu_authority": frozenset({"tool_menu"}),
    "notice_dedup": frozenset({"request_prepared"}),
    "model_router": frozenset({"request_prepared"}),
}

_COMPONENT_SWITCHES = {
    "menu_authority": "STATEFRAME_GREENLOCK_MENU_AUTHORITY",
    "notice_dedup": "STATEFRAME_NOTICE_DEDUP",
    "model_router": "STATEFRAME_MODEL_ROUTER",
}
_COMPONENT_ATTRIBUTES = {
    "menu_authority": "stateframe_greenlock_menu_authority_mode",
    "notice_dedup": "stateframe_notice_dedup_mode",
    "model_router": "stateframe_model_router_mode",
}

_CAPABILITY_ROWS = (
    CapabilityEntry(
        name="menu_authority",
        trigger="StateFrameRuntime.tool_menu() projects the GreenLock surface",
        typed_output="serialized provider tool set plus GreenLock dispatch verdict",
        intended_consumer="runner provider menu and GreenLock action/mutation gates",
        required_phase="request composition and dispatch",
        allowed_effects=(ConsequenceType.CONTROL, ConsequenceType.INFORMATION),
        kind=ComponentKind.EXPERIMENTAL_CONTROL,
        documentation_version="R-014-2026-09-27",
        integration_test="probes/test_capability_manifest.py::"
        "test_live_menu_capability_reaches_request_and_dispatch",
    ),
    CapabilityEntry(
        name="notice_dedup",
        trigger="StateFrameRuntime.prepare() renders the notice bundle",
        typed_output="serialized notice body with revision-bound dedup state",
        intended_consumer="provider-bound prepared message list",
        required_phase="request composition",
        allowed_effects=(ConsequenceType.INFORMATION, ConsequenceType.MEMORY),
        kind=ComponentKind.EXPERIMENTAL_CONTROL,
        documentation_version="R-014-2026-09-27",
        integration_test="probes/test_capability_manifest.py::"
        "test_live_notice_capability_reaches_serialized_prepare_request",
    ),
    CapabilityEntry(
        name="model_router",
        trigger="StateFrameRuntime.prepare() composes the model handoff",
        typed_output="serialized handoff message and routed tool payload",
        intended_consumer="provider-bound prepared request and model-router journal",
        required_phase="request composition",
        allowed_effects=(ConsequenceType.CONTROL, ConsequenceType.INFORMATION),
        kind=ComponentKind.EXPERIMENTAL_CONTROL,
        documentation_version="R-014-2026-09-27",
        integration_test="probes/test_capability_manifest.py::"
        "test_live_router_capability_reaches_serialized_prepare_request",
    ),
)

_REGISTERED = False


def mode(value: str | None = None) -> str:
    """Return the strict opt-in mode for the capability manifest."""
    selected = os.environ.get(MODE_ENV, "off") if value is None else value
    selected = str(selected).strip().lower() or "off"
    if selected not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return selected


def enabled(runtime: object | None = None) -> bool:
    selected = getattr(runtime, "stateframe_capability_manifest_mode", None)
    return (str(selected).strip().lower() == "on"
            if selected is not None else mode() == "on")


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    for entry in _CAPABILITY_ROWS:
        register_capability(entry)
    _REGISTERED = True


def entries() -> dict[str, CapabilityEntry]:
    """Return the R-014 entries only while the manifest switch is enabled."""
    if mode() != "on":
        return {}
    _ensure_registered()
    registered = _registered_capabilities()
    return {entry.name: registered[entry.name] for entry in _CAPABILITY_ROWS}


def _component_mode(name: str, runtime: object | None) -> str:
    attribute = _COMPONENT_ATTRIBUTES[name]
    selected = getattr(runtime, attribute, None) if runtime is not None else None
    if selected is None:
        selected = os.environ.get(_COMPONENT_SWITCHES[name], "off")
    return str(selected).strip().lower() or "off"


def _journal_records(runtime: object | None) -> tuple[Mapping[str, object], ...]:
    records, _menu, _menu_refs, _assignment_refs = _journal_projection(runtime)
    return tuple(records)


def _journal_projection(runtime: object | None) -> tuple[
        Sequence[Mapping[str, object]], Sequence[Mapping[str, object]],
        tuple[str, ...], tuple[str, ...]]:
    """Read only the journal suffix not already projected for this runtime.

    Capability status is rendered at every request boundary.  Re-reading and
    decoding the complete append-only journal at each boundary made this
    bookkeeping path O(history) per step.  The writer lease gives us a safe
    append-only invariant; inode/size/mtime changes invalidate the projection
    and force a complete rebuild rather than allowing stale authority facts.
    """
    journal = getattr(runtime, "journal", None) if runtime is not None else None
    path = getattr(journal, "path", None)
    if path is None:
        return (), (), (), ()
    try:
        stat_result = path.stat()
        signature = (str(path), stat_result.st_ino, stat_result.st_size,
                     stat_result.st_mtime_ns)
    except (AttributeError, OSError, TypeError):
        # Test doubles and non-file journals retain the old safe behavior.
        records: list[Mapping[str, object]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if isinstance(record, Mapping):
                    records.append(record)
        except (OSError, TypeError, ValueError):
            return (), (), ()
        menu = tuple(record for record in records if record.get("kind") == _MENU_EVENT)
        def refs_for(kind: str) -> tuple[str, ...]:
            return tuple(dict.fromkeys(
                str(record.get("event_id") or record.get("id") or
                    f"{record.get('kind')}:{record.get('sequence')}" ).strip()
                for record in records
                if record.get("kind") == kind
                and (record.get("event_id") or record.get("id")
                     or record.get("sequence") is not None)
            ))
        return tuple(records), menu, refs_for(_MENU_EVENT), refs_for(_ASSIGNMENT_EVENT)

    cache = getattr(runtime, "_capability_manifest_journal_projection", None)
    if isinstance(cache, dict) and isinstance(cache.get("records"), JournalPrefix) == legacy():
        cache = None  # Re-read when changing the lab arm, including checkpoint restore.
    if not isinstance(cache, dict) or cache.get("path") != signature[0]:
        cache = {"path": signature[0], "inode": signature[1], "size": 0,
                 "mtime_ns": 0, "records": ([] if legacy() else JournalPrefix(str(path))), "menu": [], "menu_refs": history(directory=path.parent),
                 "assignment_refs": history(directory=path.parent),
                 "invalid": False}
        if not legacy():
            cache = CapabilityProjection(cache)
        setattr(runtime, "_capability_manifest_journal_projection", cache)
    cached_size = int(cache.get("size", 0))
    same_file = (cache.get("inode") == signature[1]
                 and signature[2] >= cached_size
                 and (signature[2] > cached_size or cache.get("mtime_ns") == signature[3]))
    if cache.get("invalid") and same_file and signature[2] == cache.get("size"):
        return (), (), ()
    if not same_file:
        cache.update({"inode": signature[1], "size": 0, "mtime_ns": 0,
                      "records": ([] if legacy() else JournalPrefix(str(path))), "menu": [], "menu_refs": history(directory=path.parent),
                      "assignment_refs": history(directory=path.parent), "invalid": False})

    start = int(cache.get("size", 0))
    try:
        with profile_active_host_span('capability_journal_suffix_read'):
            with path.open("rb") as handle:
                handle.seek(start)
                chunk = handle.read()
        if chunk and not chunk.endswith(b"\n"):
            raise ValueError("journal suffix is not newline terminated")
        for line in chunk.decode("utf-8").splitlines():
            record = json.loads(line) if line.strip() else None
            if isinstance(cache["records"], JournalPrefix):
                cache["records"] = cache["records"].advance(line, record=isinstance(record, Mapping))
            if not isinstance(record, Mapping):
                continue
            if isinstance(cache["records"], list):
                cache["records"].append(record)
            if record.get("kind") == _MENU_EVENT:
                if isinstance(cache["records"], JournalPrefix):
                    cache["menu"] = fold_menu(cache["menu"], record)
                else:
                    cache["menu"].append(record)
                value = record.get("event_id") or record.get("id")
                if value is None and record.get("sequence") is not None:
                    value = f"{record.get('kind')}:{record.get('sequence')}"
                if isinstance(value, str) and value.strip() and value not in cache["menu_refs"]:
                    cache["menu_refs"].append(value.strip())
            if record.get("kind") == _ASSIGNMENT_EVENT:
                value = record.get("event_id") or record.get("id")
                if value is None and record.get("sequence") is not None:
                    value = f"{record.get('kind')}:{record.get('sequence')}"
                if (isinstance(value, str) and value.strip()
                        and value not in cache["assignment_refs"]):
                    cache["assignment_refs"].append(value.strip())
        if isinstance(cache["records"], JournalPrefix):
            cache["records"] = cache["records"].finish(signature[2])
        cache.update({"size": signature[2], "mtime_ns": signature[3], "invalid": False})
    except (OSError, TypeError, ValueError):
        # A partial or malformed suffix is not evidence.  Do not reuse a
        # previous projection until the file changes and can be re-read.
        cache.update({"size": signature[2], "mtime_ns": signature[3], "invalid": True,
                      "records": ([] if legacy() else JournalPrefix(str(path))), "menu": [], "menu_refs": history(directory=path.parent),
                      "assignment_refs": history(directory=path.parent)})
        return (), (), (), ()
    return ((cache["records"] if isinstance(cache["records"], JournalPrefix) else tuple(cache["records"])),
            tuple(cache["menu"]),
            tuple(cache["menu_refs"]), tuple(cache["assignment_refs"]))


def _serialized_text(messages: object) -> str:
    parts: list[str] = []
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes, bytearray)):
        for message in messages:
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif content is not None:
                    parts.append(json.dumps(content, ensure_ascii=False, default=str))
    return "\n\x00\n".join(parts)


def _menu_observation(runtime: object | None) -> tuple[bool, bool]:
    """Return (consumer-called, output-changed) from the authority's own event."""
    _records, records, _menu_refs, _assignment_refs = _journal_projection(runtime)
    successful = [record for record in records if not record.get("error_type")]
    if not successful:
        return False, False
    changed = any(record.get("added") or record.get("removed") for record in successful)
    return True, bool(changed)


def _observed(name: str, runtime: object | None, messages: object) -> tuple[bool, bool]:
    """Return (fired, behavior-changing) from a real consumer-facing result."""
    component_mode = _component_mode(name, runtime)
    if component_mode == "off":
        return False, False
    if name == "menu_authority":
        fired, changed = _menu_observation(runtime)
        return fired, changed and component_mode == "on"
    text = _serialized_text(messages)
    if name == "notice_dedup":
        delivered = any(marker in text for marker in _NOTICE_MARKERS)
        return delivered, delivered and component_mode == "on"
    if name == "model_router":
        delivered = _HANDOFF_HEADER in text
        return delivered, delivered and component_mode == "on"
    return False, False


def _event_refs(name: str, runtime: object | None) -> tuple[str, ...]:
    """Return durable producer references without treating them as delivery."""
    event_kinds = {
        "menu_authority": frozenset({_MENU_EVENT}),
        "notice_dedup": frozenset(),
        "model_router": frozenset({_ASSIGNMENT_EVENT}),
    }[name]
    if not event_kinds:
        return ()
    _records, _menu, menu_refs, assignment_refs = _journal_projection(runtime)
    if _MENU_EVENT in event_kinds:
        return menu_refs
    if _ASSIGNMENT_EVENT in event_kinds:
        return assignment_refs
    return ()


def _status(
    *,
    effective: bool,
    eligible: bool,
    fired: bool,
    exposed: bool,
    delivered: bool,
) -> str:
    if not effective:
        return "disabled"
    if not eligible:
        return "not_eligible"
    if not fired:
        return "not_fired"
    if not exposed:
        return "fired_not_exposed"
    if not delivered:
        return "exposed_not_delivered"
    return "delivered"


def _entry_dict(
    entry: CapabilityEntry,
    runtime: object | None,
    messages: object,
    *,
    boundary: str | None = None,
    tools: object = (),
    request_id: str | None = None,
    delivered: bool = False,
) -> dict[str, object]:
    component_mode = _component_mode(entry.name, runtime)
    fired, behavior_changing = _observed(entry.name, runtime, messages)
    effective = component_mode != "off"
    eligible = effective and boundary in _BOUNDARY_BY_CAPABILITY[entry.name]
    if entry.name == "menu_authority":
        exposed = bool(
            fired
            and eligible
            and isinstance(tools, Sequence)
            and not isinstance(tools, (str, bytes, bytearray))
            and len(tools) > 0
        )
    else:
        exposed = bool(fired and eligible)
    delivered = bool(delivered and exposed)
    return {
        "status_schema": STATUS_SCHEMA,
        "name": entry.name,
        "trigger": entry.trigger,
        "typed_output": entry.typed_output,
        "intended_consumer": entry.intended_consumer,
        "required_phase": entry.required_phase,
        "allowed_effects": [effect.value for effect in entry.allowed_effects],
        "kind": entry.kind.value,
        "documentation_version": entry.documentation_version,
        "integration_test": entry.integration_test,
        "switch": _COMPONENT_SWITCHES[entry.name],
        "mode": component_mode,
        "installed": True,
        # ``enabled`` is retained for existing consumers; the status fields
        # below make each lifecycle distinction explicit.
        "enabled": effective,
        "configured": effective,
        "effective": effective,
        "eligible": eligible,
        "fired": fired,
        "exposed": exposed,
        "delivered": delivered,
        "delivery_state": _status(
            effective=effective,
            eligible=eligible,
            fired=fired,
            exposed=exposed,
            delivered=delivered,
        ),
        "event_refs": list(_event_refs(entry.name, runtime)),
        "boundary": boundary,
        "request_id": request_id,
        "behavior_changing": behavior_changing,
    }


def manifest_payload(
    runtime: object | None = None,
    *,
    messages: object = (),
    boundary: str | None = None,
    tools: object = (),
    request_id: str | None = None,
    delivered: bool = False,
) -> dict[str, object]:
    """Serialize the manifest and current liveness facts for a run manifest."""
    if not enabled(runtime):
        return {}
    _ensure_registered()
    if messages == () and runtime is not None:
        messages = getattr(runtime, "_capability_manifest_last_messages", ())
    return {
        "schema": SCHEMA,
        "status_schema": STATUS_SCHEMA,
        "boundary": boundary,
        "request_id": request_id,
        "entries": [
            _entry_dict(
                entry,
                runtime,
                messages,
                boundary=boundary,
                tools=tools,
                request_id=request_id,
                delivered=delivered,
            )
            for entry in _CAPABILITY_ROWS
        ],
    }


def publish_status(
    runtime: object,
    *,
    boundary: str,
    messages: object = (),
    tools: object = (),
    request_id: str | None = None,
    delivered: bool = False,
) -> dict[str, object] | None:
    """Persist observed capability status at a real menu/request boundary.

    The startup manifest remains an immutable declaration.  This event is the
    later evidence record: it is emitted only with the capability switch on,
    after the consumer has produced its actual menu or serialized request.
    """
    if not enabled(runtime):
        return None
    payload = manifest_payload(
        runtime,
        messages=messages,
        boundary=boundary,
        tools=tools,
        request_id=request_id,
        delivered=delivered,
    )
    setattr(runtime, "_capability_manifest_last_messages", messages)
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return payload
    emit(
        STATUS_EVENT,
        producer="kernel_capability_manifest",
        boundary=boundary,
        request_id=request_id,
        delivered=delivered,
        manifest=payload,
        tools=tools,
        dispatch=False,
    )
    return payload


__all__ = [
    "MODE_ENV", "SCHEMA", "STATUS_EVENT", "STATUS_SCHEMA", "enabled", "entries",
    "manifest_payload", "mode", "publish_status",
]
