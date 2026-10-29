"""Host-only completion-boundary diagnosis for the r45 R-001/R-004 slice.

The diagnostic records the wire menu, the runner's dispatchable subset, open
obligations, and the action adopted by the completion reducer.  It is a
journal-side witness, not another prompt block: the existing F4 handoff is
referenced by digest when it is present.  The switch is deliberately OFF by
default so legacy provider payloads and semantic journal events are untouched;
source-provenance digests still truthfully reflect this module's host changes.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os


MODE_ENV = "STATEFRAME_COMPLETION_DIAGNOSIS"
SCHEMA = "stateframe-completion-diagnosis/1"
MAX_TOOLS = 64
MAX_OBLIGATIONS = 64
MAX_TEXT = 360


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _field(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _text(value: object, limit: int = MAX_TEXT) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _bounded(value: object, *, depth: int = 0) -> object:
    """Copy only small JSON-shaped host facts into a journal event."""
    if depth >= 4:
        return _text(value, 160)
    if isinstance(value, Mapping):
        return {
            _text(key, 80): _bounded(item, depth=depth + 1)
            for key, item in list(value.items())[:32]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_bounded(item, depth=depth + 1) for item in list(value)[:32]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _text(value) if isinstance(value, str) else value
    return _text(value)


def _tool_name(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, Mapping):
        return ""
    function = value.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
    else:
        name = value.get("name", value.get("tool", value.get("kind")))
    return str(name).strip() if name is not None else ""


def _names(tools: object) -> list[str]:
    if isinstance(tools, Mapping):
        tools = tools.values()
    if isinstance(tools, (str, bytes, bytearray)):
        tools = (tools,)
    if not isinstance(tools, Iterable):
        return []
    result: list[str] = []
    for item in tools:
        name = _tool_name(item)
        if name and name not in result:
            result.append(name)
        if len(result) >= MAX_TOOLS:
            break
    return result


def _sequence(value: object) -> tuple[object, ...]:
    """Normalize host assessment collections without iterating mapping keys."""
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _revision(runtime: object) -> str:
    binding = getattr(runtime, "binding", None)
    try:
        value = binding() if callable(binding) else binding
    except (AttributeError, TypeError, ValueError):
        value = None
    for key in ("product_revision", "artifact", "revision", "workspace_revision",
                "artifact_hash", "sha"):
        candidate = _field(value, key)
        if candidate not in (None, ""):
            return _text(candidate, 180)
    return "UNKNOWN"


def _dispatch_allow_set(runtime: object, offered: Sequence[str]) -> tuple[list[str], list[str]]:
    """Mirror the runner's name-level dispatch branch without changing it."""
    from kernel_tool_surface import runner_dispatchable, tool_allowed

    allowed: list[str] = []
    blocked: list[str] = []
    for name in offered:
        if runner_dispatchable(name, runtime=runtime) and tool_allowed(name, runtime=runtime):
            allowed.append(name)
        else:
            blocked.append(name)
    return allowed, blocked


def _journal_emit(runtime: object, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        try:
            emit(kind, **fields)
        except Exception:  # noqa: BLE001 - diagnosis is advisory and fail-open
            return


def capture_request(runtime: object, tools: object, *, request_id: object = None) -> dict[str, object] | None:
    """Capture the final provider-bound tool names at the real prepare seam."""
    if not enabled():
        return None
    offered = _names(tools)
    allow_set, blocked = _dispatch_allow_set(runtime, offered)
    request_revision = _revision(runtime)
    request = {
        "payload_schema": SCHEMA,
        "request_id": _text(request_id, 180) if request_id is not None else None,
        "request_revision": request_revision,
        "offered_tools": offered,
        "dispatch_allow_set": allow_set,
        "dispatch_blocked": blocked,
    }
    runtime._completion_diagnosis_request = request
    _journal_emit(
        runtime,
        "completion_diagnosis_request",
        producer="kernel_completion_diagnosis",
        payload_schema=SCHEMA,
        request_id=request["request_id"],
        request_revision=request_revision,
        offered_tools=offered,
        dispatch_allow_set=allow_set,
        dispatch_blocked=blocked,
        wire_bound=True,
        dispatch=False,
    )
    return request


def _action(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    name = value.get("tool", value.get("name", value.get("kind")))
    if not isinstance(name, str) or not name.strip():
        return None
    result: dict[str, object] = {"tool": _text(name, 120)}
    arguments = value.get("arguments", value.get("args"))
    if isinstance(arguments, Mapping):
        result["arguments"] = _bounded(arguments)
    for key in ("reason", "why", "selector_scope"):
        if key in value:
            result[key] = _text(value[key], 220)
    return result


def _obligation_rows(decision: object, runtime: object = None
                     ) -> tuple[list[str], list[object], list[object]]:
    ids: list[str] = []
    rows: list[object] = []
    context_rows: list[object] = []

    try:
        from kernel_completion_owner import _assessment_is_context_only
    except ImportError:
        def _assessment_is_context_only(_value: object) -> bool:
            return False

    def add_id(value: object) -> None:
        if value not in (None, ""):
            item = _text(value, 180)
            if item and item not in ids and len(ids) < MAX_OBLIGATIONS:
                ids.append(item)

    def assessment_status(value: object) -> str:
        for key in ("verdict", "status", "underlying_status", "result", "outcome"):
            candidate = _field(value, key)
            if candidate not in (None, ""):
                return str(candidate).strip().upper()
        return ""

    for item in _field(decision, "needs_work_ids", ()) or ():
        add_id(item)
    for raw in _field(decision, "obligation_assessments", ()) or ():
        row = raw.to_dict() if hasattr(raw, "to_dict") else raw
        if isinstance(row, Mapping):
            if _assessment_is_context_only(row):
                context_rows.append(_bounded(row))
                continue
            # A reducer keeps every host assessment for auditability, but only
            # non-SATISFIED verdicts are open obligations.  Missing status is
            # intentionally unresolved so an old/partial adapter fails safe.
            if assessment_status(row) != "SATISFIED":
                add_id(row.get("obligation_id", row.get("id", row.get("check_id"))))
            rows.append(_bounded(row))
        if len(rows) >= MAX_OBLIGATIONS:
            break
    if runtime is not None:
        context_keys = {
            str(row.get("obligation_id", row.get("id", ""))).strip()
            for row in context_rows if isinstance(row, Mapping)
            and str(row.get("obligation_id", row.get("id", ""))).strip()
        }
        for source_name in (
                "host_closure_assessments", "behavior_witness_assessments", "obligation_assessments"):
            for raw in _sequence(getattr(runtime, source_name, ())):
                row = raw.to_dict() if hasattr(raw, "to_dict") else raw
                if not isinstance(row, Mapping) or not _assessment_is_context_only(row):
                    continue
                key = str(row.get("obligation_id", row.get("id", ""))).strip()
                if not key or key not in context_keys:
                    context_rows.append(_bounded(row))
                    if key:
                        context_keys.add(key)
                if len(context_rows) >= MAX_OBLIGATIONS:
                    break
            if len(context_rows) >= MAX_OBLIGATIONS:
                break
    payload = _field(decision, "needs_work_payload")
    if isinstance(payload, Mapping):
        add_id(payload.get("focus_obligation"))
    context_ids = {
        str(row.get("obligation_id", row.get("id", ""))).strip()
        for row in context_rows if isinstance(row, Mapping)
    }
    ids = [item for item in ids if item not in context_ids]
    return ids, rows, context_rows


def _handoff_descriptor(runtime: object) -> dict[str, object]:
    handoff = getattr(runtime, "_f4_last_handoff", None)
    if not isinstance(handoff, Mapping) or "content" not in handoff:
        return {"renderer": "none"}
    content = handoff.get("content")
    if not isinstance(content, str):
        return {"renderer": "none"}
    return {
        "renderer": "stateframe_f4",
        "handoff_digest": _digest({
            "content": content,
            "token_estimate": handoff.get("token_estimate"),
            "budget_note": handoff.get("budget_note"),
        }),
        "token_estimate": handoff.get("token_estimate"),
        "budget_note": handoff.get("budget_note"),
    }


def record_completion(runtime: object, decision: object) -> dict[str, object] | None:
    """Record one reducer result, separating authority from selected action."""
    if not enabled():
        return None
    request = getattr(runtime, "_completion_diagnosis_request", None)
    if not isinstance(request, Mapping):
        offered = _names(getattr(runtime, "completion_menu", ()))
        allow_set, blocked = _dispatch_allow_set(runtime, offered)
        request = {
            "payload_schema": SCHEMA,
            "request_id": None,
            "request_revision": _revision(runtime),
            "offered_tools": offered,
            "dispatch_allow_set": allow_set,
            "dispatch_blocked": blocked,
        }
    basis = _field(decision, "basis")
    basis_dict = basis.to_dict() if hasattr(basis, "to_dict") else _bounded(basis)
    basis_digest = getattr(basis, "digest", None)
    if not isinstance(basis_digest, str):
        basis_digest = _digest(basis_dict)
    selected = _action(_field(decision, "repair_action"))
    requested = _action(getattr(runtime, "current_action", None))
    open_ids, obligation_rows, context_rows = _obligation_rows(decision, runtime)
    reason_codes = [_text(item, 160) for item in (_field(decision, "reason_codes", ()) or ())]
    outcome = _text(_field(decision, "outcome", "UNKNOWN"), 120)
    accepted = bool(_field(decision, "accepted", False))
    adopted = {
        "outcome": outcome,
        "accepted": accepted,
        "reason_codes": reason_codes[:32],
        "basis_digest": basis_digest,
        "selected_action": selected,
    }
    handoff = _handoff_descriptor(runtime)
    route_decision: dict[str, object] | None = None
    try:
        from kernel_completion_owner import completion_route_decision
        route = completion_route_decision(
            runtime, decision,
            tools=getattr(runtime, "_completion_owner_full_tools", ())
            or getattr(runtime, "completion_menu", ()),
        )
        route_decision = route.to_dict()
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        route_decision = None
    record = {
        "payload_schema": SCHEMA,
        "request_id": request.get("request_id"),
        "request_revision": request.get("request_revision", "UNKNOWN"),
        "current_revision": _revision(runtime),
        "offered_tools": list(request.get("offered_tools", ()))[:MAX_TOOLS],
        "dispatch_allow_set": list(request.get("dispatch_allow_set", ()))[:MAX_TOOLS],
        "dispatch_blocked": list(request.get("dispatch_blocked", ()))[:MAX_TOOLS],
        "open_obligations": open_ids,
        "obligation_assessments": obligation_rows,
        "context_assessments": context_rows,
        "selected_action": selected,
        "requested_action": requested,
        "adopted_decision": adopted,
        "adopted_decision_handoff": {
            "decision": adopted,
            "completion_basis": _bounded(basis_dict),
            "f4_handoff": handoff,
        },
        "completion_basis": _bounded(basis_dict),
        "handoff": handoff,
        "next_action_decision": route_decision,
        "fired": True,
        "dispatch": False,
    }
    runtime._completion_diagnosis_last = record
    _journal_emit(runtime, "completion_diagnosed", producer="kernel_completion_diagnosis", **record)
    return record


__all__ = ["MODE_ENV", "SCHEMA", "capture_request", "enabled", "mode", "record_completion"]
