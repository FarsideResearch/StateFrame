"""Append-only, state-only frame deltas for provider prefix-cache stability.

The feature stores only one typed StateFrame base and its bounded typed deltas
on the run-scoped runtime. It never retains assistant/tool transcript messages.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kernel_stateframe_message_layout import (
    StateFrameMessageLayoutError,
    stateframe_message_index,
    validate_stateframe_message_order,
)


MODE_ENV = "STATEFRAME_FRAME_APPEND"
SCHEMA = "stateframe-frame-append/1"
DELTA_SCHEMA = "stateframe-frame-delta/1"
STATE_HEADER = "[StateFrame]"
DELTA_HEADER = "[StateFrame delta]"
PRODUCT_MODE = "FRAME_ONLY"
DEFAULT_REBASE_TURNS = 6
DEFAULT_DELTA_BYTES = 8 * 1024
_RUNTIME_STATE_ATTR = "_stateframe_frame_append_state"
_RUNTIME_LOCK = threading.RLock()
_VALID_MODES = frozenset({"off", "shadow", "on"})
_MAX_DELTA_SECTIONS = 256


class FrameAppendError(ValueError):
    """The append representation cannot preserve the current frame exactly."""


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the literal registered mode; invalid values fail safely to OFF."""
    source = os.environ if environ is None else environ
    selected = str(source.get(MODE_ENV, "off")).strip().lower()
    return selected if selected in _VALID_MODES else "off"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _wire_sha256(value: object) -> str:
    """Hash the exact standard JSON serialization used at the provider seam."""
    return "sha256:" + hashlib.sha256(_wire_json(value)).hexdigest()


def _wire_json(value: object) -> bytes:
    return json.dumps(value).encode("utf-8")


def _same(left: object, right: object) -> bool:
    return _canonical(left) == _canonical(right)


def _state_payload(message: Mapping[str, object]) -> dict[str, object]:
    content = message.get("content")
    if not isinstance(content, str) or not content.startswith(STATE_HEADER + "\n"):
        raise FrameAppendError("frame append requires one typed StateFrame message")
    try:
        payload = json.loads(content.split("\n", 1)[1])
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise FrameAppendError("frame append StateFrame JSON is invalid") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise FrameAppendError("frame append StateFrame payload is malformed")
    return payload


def _state_index(messages: Sequence[object]) -> int:
    try:
        return stateframe_message_index(messages)
    except StateFrameMessageLayoutError as error:
        raise FrameAppendError(str(error)) from error


def _is_delta_message(message: object) -> bool:
    return (
        isinstance(message, Mapping)
        and message.get("role") == "user"
        and message.get("name") == "stateframe_frame_delta"
        and isinstance(message.get("content"), str)
        and str(message.get("content")).startswith(DELTA_HEADER + "\n")
    )


def _identity_parts(
    body: Mapping[str, object], messages: Sequence[object],
    *, require_task_label: bool, page_log_layout: bool,
) -> tuple[str, str, str]:
    try:
        layout = validate_stateframe_message_order(
            messages,
            require_task_label=require_task_label,
            allow_user_context_before_stateframe=page_log_layout,
        )
    except StateFrameMessageLayoutError as error:
        raise FrameAppendError(str(error)) from error
    task = messages[layout.task_index]
    if not isinstance(task, Mapping):
        raise FrameAppendError("frame append requires the original task before StateFrame")
    return (
        _wire_sha256(list(messages[:layout.system_end])),
        _wire_sha256(dict(task)),
        _wire_sha256(body.get("tools")),
    )


def _section_type(path: Sequence[str]) -> str:
    lowered = ".".join(path).lower()
    if "verdict" in lowered:
        return "verdict"
    if "site" in lowered:
        return "site"
    if "ledger" in lowered or "reads" in lowered or "edits" in lowered:
        return "ledger_entry"
    if "receipt" in lowered or "applied_edits" in lowered:
        return "receipt"
    if "module" in lowered and "head" in lowered:
        return "module_head"
    return "fact"


def _section_id(path: Sequence[str]) -> str:
    """Use a stable JSON-pointer identity for one typed frame section."""
    return "/" + "/".join(
        part.replace("~", "~0").replace("/", "~1") for part in path
    )


def _section(
    path: tuple[str, ...], operation: str, prior: object, value: object = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "type": _section_type(path),
        "id": _section_id(path),
        "path": list(path),
        "operation": operation,
        "supersedes": "absent" if prior is _ABSENT else _sha256(prior),
    }
    if operation != "delete":
        result["value"] = copy.deepcopy(value)
        if operation == "append":
            result["digest"] = _sha256([*prior, *value])
        else:
            result["digest"] = _sha256(value)
    else:
        result["digest"] = _sha256({"deleted": True})
    return result


_ABSENT = object()


def _diff(
    before: object, after: object, path: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    if _same(before, after):
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        if any(not isinstance(key, str) for key in (*before.keys(), *after.keys())):
            raise FrameAppendError("frame append only supports string object keys")
        changes: list[dict[str, object]] = []
        for key in sorted(before.keys() - after.keys()):
            changes.append(_section((*path, key), "delete", before[key]))
        for key in sorted(after):
            if key not in before:
                changes.append(_section((*path, key), "replace", _ABSENT, after[key]))
            else:
                changes.extend(_diff(before[key], after[key], (*path, key)))
        return changes
    if (isinstance(before, list) and isinstance(after, list)
            and len(after) > len(before)
            and _same(before, after[:len(before)])):
        additions = after[len(before):]
        return [_section(path, "append", before, additions)]
    if not path:
        raise FrameAppendError("frame append cannot replace the StateFrame payload root")
    return [_section(path, "replace", before, after)]


def _lookup_path(root: object, path: Sequence[str]) -> tuple[bool, object]:
    current = root
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, _ABSENT
        current = current[key]
    return True, current


def _set_path(root: dict[str, object], path: Sequence[str], value: object) -> None:
    if not path:
        raise FrameAppendError("frame append cannot replace the StateFrame payload root")
    current: dict[str, object] = root
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise FrameAppendError("frame append path crosses a non-object section")
        current = child
    current[path[-1]] = copy.deepcopy(value)


def _delete_path(root: dict[str, object], path: Sequence[str]) -> None:
    if not path:
        raise FrameAppendError("frame append cannot delete the StateFrame payload root")
    current: dict[str, object] = root
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise FrameAppendError("frame append delete path crosses a non-object section")
        current = child
    if path[-1] not in current:
        raise FrameAppendError("frame append delete target is absent")
    del current[path[-1]]


def _apply_delta(payload: dict[str, object], delta_message: Mapping[str, object]) -> None:
    if not _is_delta_message(delta_message):
        raise FrameAppendError("frame append delta message identity is invalid")
    content = delta_message.get("content")
    if not isinstance(content, str) or not content.startswith(DELTA_HEADER + "\n"):
        raise FrameAppendError("frame append delta message header is invalid")
    try:
        delta = json.loads(content.split("\n", 1)[1])
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise FrameAppendError("frame append delta JSON is invalid") from error
    if not isinstance(delta, Mapping) or delta.get("schema") != DELTA_SCHEMA:
        raise FrameAppendError("frame append delta schema is invalid")
    sections = delta.get("sections")
    if not isinstance(sections, list):
        raise FrameAppendError("frame append delta sections are malformed")
    for section in sections:
        if not isinstance(section, Mapping):
            raise FrameAppendError("frame append delta section is malformed")
        path = section.get("path")
        operation = section.get("operation")
        if (not isinstance(path, list) or not path
                or any(not isinstance(key, str) for key in path)):
            raise FrameAppendError("frame append delta path is malformed")
        if section.get("id") != _section_id(path):
            raise FrameAppendError("frame append delta section identity is invalid")
        if section.get("type") != _section_type(path):
            raise FrameAppendError("frame append delta section type is invalid")
        exists, prior = _lookup_path(payload, path)
        prior_marker = "absent" if not exists else _sha256(prior)
        if section.get("supersedes") != prior_marker:
            raise FrameAppendError("frame append delta supersession digest does not match")
        if operation == "delete":
            if section.get("digest") != _sha256({"deleted": True}):
                raise FrameAppendError("frame append delete digest does not match")
            _delete_path(payload, path)
            continue
        value = section.get("value", _ABSENT)
        if value is _ABSENT:
            raise FrameAppendError("frame append delta value is missing")
        if operation == "append":
            if not exists or not isinstance(prior, list) or not isinstance(value, list):
                raise FrameAppendError("frame append list section is malformed")
            updated = [*prior, *value]
        elif operation == "replace":
            updated = value
        else:
            raise FrameAppendError("frame append delta operation is unknown")
        if section.get("digest") != _sha256(updated):
            raise FrameAppendError("frame append delta content digest does not match")
        _set_path(payload, path, updated)


def materialize_payload(
    base_message: Mapping[str, object],
    delta_messages: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply typed deltas to a base frame and return the composed payload."""
    payload = _state_payload(base_message)
    for message in delta_messages:
        _apply_delta(payload, message)
    return payload


def materialize_messages(messages: Sequence[object]) -> list[dict[str, object]]:
    """Return the continuity-gate view: one materialized StateFrame, no deltas."""
    if any(not isinstance(message, Mapping) for message in messages):
        raise FrameAppendError("frame append continuity messages are malformed")
    state_index = _state_index(messages)
    base = messages[state_index]
    if not isinstance(base, Mapping):
        raise FrameAppendError("frame append base message is malformed")
    delta_messages = [
        message for message in messages[state_index + 1:]
        if _is_delta_message(message)
    ]
    payload = materialize_payload(base, delta_messages)
    composed = [copy.deepcopy(dict(message)) for message in messages
                if isinstance(message, Mapping)
                and not _is_delta_message(message)]
    composed_state_index = _state_index(composed)
    state_message = composed[composed_state_index]
    state_message["content"] = STATE_HEADER + "\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return composed


def validate_composed_product(
    body: Mapping[str, object], runtime: object,
) -> dict[str, object]:
    """Run final FrameFit5/F4 checks against the fully materialized append state."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise FrameAppendError("frame append final validation messages are malformed")
    composed = materialize_messages(messages)
    from kernel_context import token_estimate
    from kernel_stateframe_framefit5 import validate_product_request

    ceiling = getattr(runtime, "frame_ceiling", None)
    if type(ceiling) is not int or ceiling <= 0:
        raise FrameAppendError("frame append final validation has no valid token ceiling")
    if token_estimate({"messages": composed, "tools": body.get("tools")}) > ceiling:
        raise FrameAppendError("frame append composed request exceeds its token ceiling")
    validation = validate_product_request(composed, runtime, tools=body.get("tools"))
    from kernel_stateframe_f4 import assert_f4_invariants

    assert_f4_invariants(composed)
    return validation


@dataclass(frozen=True)
class AppendState:
    identity_parts: tuple[str, str, str]
    identity_sha256: str
    base_message: dict[str, object]
    base_payload: dict[str, object]
    current_payload: dict[str, object]
    delta_messages: tuple[dict[str, object], ...]
    turns_since_base: int
    delta_bytes: int
    generation: int


@dataclass(frozen=True)
class Projection:
    body: dict[str, object]
    state: AppendState
    previous_state: AppendState | None
    action: str
    rebase_reason: str | None
    delta_sha256: str | None
    delta_bytes: int
    prefix_preserved: bool


def _make_delta(
    previous_payload: dict[str, object], current_payload: dict[str, object], turn: int,
) -> dict[str, object]:
    sections = _diff(previous_payload, current_payload)
    if len(sections) > _MAX_DELTA_SECTIONS:
        raise FrameAppendError("frame append delta exceeds its section-count bound")
    sections.sort(key=lambda section: tuple(str(part) for part in section.get("path", ())))
    content = DELTA_HEADER + "\n" + json.dumps(
        {"schema": DELTA_SCHEMA, "turn": turn, "sections": sections},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return {
        "role": "user", "name": "stateframe_frame_delta", "content": content,
    }


def _identity_reason(
    previous: AppendState, current: tuple[str, str, str],
) -> str | None:
    labels = ("system", "task", "tool_surface")
    for label, old, new in zip(labels, previous.identity_parts, current):
        if old != new:
            return f"{label}_changed"
    return None


def project_request(
    body: dict[str, object],
    previous: AppendState | None = None,
    *,
    interval: int = DEFAULT_REBASE_TURNS,
    byte_bound: int = DEFAULT_DELTA_BYTES,
    require_product: bool = True,
    page_log_layout: bool = False,
) -> Projection:
    """Build one append-only request; caller commits the returned state after response."""
    if type(interval) is not int or interval < 2:
        raise FrameAppendError("frame append rebase interval must be at least two turns")
    if type(byte_bound) is not int or byte_bound < 1:
        raise FrameAppendError("frame append delta bound must be positive")
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        raise FrameAppendError("frame append request messages are malformed")
    if require_product:
        from kernel_stateframe_framefit5 import is_frame_only
        if not is_frame_only(raw_messages):
            raise FrameAppendError("frame append only accepts a validated frame-only request")
    state_index = _state_index(raw_messages)
    current_message = raw_messages[state_index]
    if not isinstance(current_message, Mapping):
        raise FrameAppendError("frame append StateFrame message is malformed")
    payload = _state_payload(current_message)
    framefit5 = payload.get("state", {}).get("framefit5")
    if require_product and (
            not isinstance(framefit5, Mapping) or framefit5.get("mode") != PRODUCT_MODE):
        raise FrameAppendError("frame append requires the FrameFit5 product state")
    identity_parts = _identity_parts(
        body,
        raw_messages,
        require_task_label=require_product,
        page_log_layout=page_log_layout,
    )
    identity_sha = _sha256(list(identity_parts))

    action = "base"
    reason: str | None = None
    delta_message: dict[str, object] | None = None
    prior_deltas: tuple[dict[str, object], ...] = ()
    turns_since_base = 1
    cumulative_delta_bytes = 0
    base_message = copy.deepcopy(dict(current_message))
    base_payload = copy.deepcopy(payload)
    if previous is not None:
        reason = _identity_reason(previous, identity_parts)
        if reason is None:
            try:
                composed_prior = materialize_payload(
                    previous.base_message, previous.delta_messages,
                )
                if not _same(composed_prior, previous.current_payload):
                    reason = "prior_state_mismatch"
            except FrameAppendError:
                reason = "prior_state_unmaterializable"
        if reason is None and previous.turns_since_base + 1 >= interval:
            reason = "turn_interval"
        if reason is None:
            try:
                delta_message = _make_delta(
                    previous.current_payload, payload, previous.turns_since_base + 1,
                )
            except (FrameAppendError, TypeError, ValueError):
                reason = "delta_unrepresentable"
        if reason is None and delta_message is not None:
            delta_size = len(_wire_json(delta_message))
            if previous.delta_bytes + delta_size > byte_bound:
                reason = "delta_byte_limit"
        if reason is None and delta_message is not None:
            action = "delta"
            base_message = copy.deepcopy(previous.base_message)
            base_payload = copy.deepcopy(previous.base_payload)
            prior_deltas = previous.delta_messages
            turns_since_base = previous.turns_since_base + 1
            cumulative_delta_bytes = previous.delta_bytes + len(_wire_json(delta_message))
        else:
            action = "rebase"
            base_message = copy.deepcopy(dict(current_message))
            base_payload = copy.deepcopy(payload)

    if action == "delta" and delta_message is not None:
        delta_messages = (*prior_deltas, delta_message)
        delta_sha = "sha256:" + hashlib.sha256(
            str(delta_message["content"]).encode("utf-8"),
        ).hexdigest()
        delta_size = len(_wire_json(delta_message))
    else:
        delta_messages = ()
        delta_sha = None
        delta_size = 0

    candidate_messages = [copy.deepcopy(dict(message)) for message in raw_messages
                          if isinstance(message, Mapping)]
    if len(candidate_messages) != len(raw_messages):
        raise FrameAppendError("frame append request contains a malformed message")
    candidate_messages[state_index] = copy.deepcopy(base_message)
    candidate_messages[state_index + 1:state_index + 1] = [
        copy.deepcopy(message) for message in delta_messages
    ]
    candidate_body = copy.deepcopy(body)
    candidate_body["messages"] = candidate_messages

    materialized = materialize_payload(base_message, delta_messages)
    if not _same(materialized, payload):
        raise FrameAppendError("frame append deltas do not reconstruct the current state")
    original_tail = raw_messages[state_index + 1:]
    projected_tail = candidate_messages[state_index + 1 + len(delta_messages):]
    if _wire_json(original_tail) != _wire_json(projected_tail):
        raise FrameAppendError("frame append changed a newest user message")
    if _wire_json(body.get("tools")) != _wire_json(candidate_body.get("tools")):
        raise FrameAppendError("frame append changed N4-authored tool schemas")
    prefix_preserved = previous is not None and action == "delta" and (
        _wire_json([base_message, *prior_deltas])
        == _wire_json([previous.base_message, *previous.delta_messages])
    )
    candidate_state = AppendState(
        identity_parts=identity_parts,
        identity_sha256=identity_sha,
        base_message=base_message,
        base_payload=base_payload,
        current_payload=copy.deepcopy(payload),
        delta_messages=tuple(copy.deepcopy(message) for message in delta_messages),
        turns_since_base=turns_since_base,
        delta_bytes=cumulative_delta_bytes,
        generation=(previous.generation + 1) if previous is not None else 1,
    )
    return Projection(
        body=candidate_body,
        state=candidate_state,
        previous_state=previous,
        action=action,
        rebase_reason=reason,
        delta_sha256=delta_sha,
        delta_bytes=delta_size,
        prefix_preserved=prefix_preserved,
    )


@dataclass(frozen=True)
class PendingCommit:
    runtime: object
    previous_state: AppendState | None
    candidate_state: AppendState
    mode: str
    request_id: str | None
    step: int | None


def _emit(journal: object | None, kind: str, **fields: object) -> None:
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    try:
        emit(kind, producer="kernel_stateframe_frame_append", dispatch=False, **fields)
    except Exception:
        # Observability must not prevent the existing request path from running.
        return


def _state_facts(projection: Projection, selected_mode: str) -> dict[str, object]:
    return {
        "mode": selected_mode,
        "action": projection.action,
        "rebase_reason": projection.rebase_reason,
        "turns_since_base": projection.state.turns_since_base,
        "base_sha256": _sha256(projection.state.base_message.get("content")),
        "state_sha256": _sha256(projection.state.current_payload),
        "delta_sha256": projection.delta_sha256,
        "delta_bytes": projection.delta_bytes,
        "cumulative_delta_bytes": projection.state.delta_bytes,
        "delta_count": len(projection.state.delta_messages),
        "identity_sha256": projection.state.identity_sha256,
        "prefix_preserved": projection.prefix_preserved,
    }


def prepare_for_capture(
    capture: object,
    body: dict[str, object],
    runtime: object | None,
    journal: object | None,
    *,
    selected_mode: str | None = None,
) -> tuple[dict[str, object], PendingCommit | None]:
    """Prepare ON bytes or SHADOW facts without advancing the runtime state."""
    selected = mode() if selected_mode is None else selected_mode
    if selected not in _VALID_MODES:
        selected = "off"
    if selected == "off":
        return body, None
    setattr(capture, "_frame_append_mode", selected)
    setattr(capture, "_frame_append_active", False)
    raw_messages = body.get("messages")
    try:
        from kernel_stateframe_framefit5 import has_framefit5_state
        if (not isinstance(raw_messages, Sequence)
                or isinstance(raw_messages, (str, bytes, bytearray))
                or not has_framefit5_state(raw_messages)):
            return body, None
    except Exception as error:
        _emit(
            journal, "stateframe_frame_append_failed_open",
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None), mode=selected,
            reason="product_detection_failed", error_type=type(error).__name__,
        )
        return body, None
    if runtime is None:
        _emit(
            journal, "stateframe_frame_append_failed_open",
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None), mode=selected,
            reason="runtime_unavailable",
        )
        return body, None
    try:
        with _RUNTIME_LOCK:
            previous = getattr(runtime, _RUNTIME_STATE_ATTR, None)
            if previous is not None and not isinstance(previous, AppendState):
                previous = None
            projection = project_request(body, previous)
        try:
            validate_composed_product(projection.body, runtime)
        except Exception as error:
            _emit(
                journal, "stateframe_frame_append_failed_open",
                request_id=getattr(capture, "request_id", None),
                step=getattr(capture, "step", None), mode=selected,
                reason="composed_product_validation", error_type=type(error).__name__,
            )
            return body, None
        pending = PendingCommit(
            runtime=runtime,
            previous_state=projection.previous_state,
            candidate_state=projection.state,
            mode=selected,
            request_id=(str(getattr(capture, "request_id", "")) or None),
            step=(getattr(capture, "step", None)
                  if type(getattr(capture, "step", None)) is int else None),
        )
        setattr(capture, "_frame_append_active", True)
        facts = _state_facts(projection, selected)
        _emit(
            journal, "stateframe_frame_append_prepared",
            request_id=pending.request_id, step=pending.step, **facts,
        )
        if projection.action == "rebase" and projection.rebase_reason:
            _emit(
                journal, "stateframe_frame_append_rebased",
                request_id=pending.request_id, step=pending.step,
                reason=projection.rebase_reason, **facts,
            )
        if selected == "shadow":
            mismatches: list[str] = []
            try:
                materialized = materialize_messages(projection.body["messages"])
                if _wire_json(materialized) != _wire_json(
                        materialize_messages(body.get("messages", ()))):
                    mismatches.append("composed_state")
            except Exception:
                mismatches.append("composed_state")
            for field in mismatches:
                _emit(
                    journal, "stateframe_frame_append_shadow_mismatch",
                    request_id=pending.request_id, step=pending.step,
                    field=field, **facts,
                )
            return body, pending
        return projection.body, pending
    except Exception as error:
        _emit(
            journal, "stateframe_frame_append_failed_open",
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None), mode=selected,
            reason="projection_error", error_type=type(error).__name__,
        )
        return body, None


def fail_open_pending(
    capture: object,
    journal: object | None,
    *,
    reason: str,
    error_type: str | None = None,
) -> None:
    """Discard a not-yet-committed projection after a later wire guard refuses it."""
    pending = getattr(capture, "_frame_append_pending", None)
    if not isinstance(pending, PendingCommit):
        return
    setattr(capture, "_frame_append_pending", None)
    setattr(capture, "_frame_append_active", False)
    _emit(
        journal, "stateframe_frame_append_failed_open",
        request_id=pending.request_id, step=pending.step, mode=pending.mode,
        reason=reason, error_type=error_type,
    )


def commit_after_response(capture: object, journal: object | None) -> None:
    """Advance run state only after a response was durably captured."""
    pending = getattr(capture, "_frame_append_pending", None)
    if not isinstance(pending, PendingCommit):
        return
    setattr(capture, "_frame_append_pending", None)
    try:
        with _RUNTIME_LOCK:
            current = getattr(pending.runtime, _RUNTIME_STATE_ATTR, None)
            if current is pending.candidate_state:
                return
            if current is not pending.previous_state:
                _emit(
                    journal, "stateframe_frame_append_failed_open",
                    request_id=pending.request_id, step=pending.step, mode=pending.mode,
                    reason="concurrent_state_change",
                )
                return
            setattr(pending.runtime, _RUNTIME_STATE_ATTR, pending.candidate_state)
        _emit(
            journal, "stateframe_frame_append_committed",
            request_id=pending.request_id, step=pending.step, mode=pending.mode,
            generation=pending.candidate_state.generation,
            state_sha256=_sha256(pending.candidate_state.current_payload),
            turns_since_base=pending.candidate_state.turns_since_base,
            delta_count=len(pending.candidate_state.delta_messages),
        )
    except Exception as error:
        _emit(
            journal, "stateframe_frame_append_failed_open",
            request_id=pending.request_id, step=pending.step, mode=pending.mode,
            reason="state_commit_failed", error_type=type(error).__name__,
        )


def response_cache_share(
    journal: object,
    accounting: Mapping[str, object],
    *,
    request_id: object = None,
    step: object = None,
    selected_mode: str,
) -> dict[str, object] | None:
    """Emit cached-token share from the normalized provider-ingress counters."""
    if selected_mode not in {"shadow", "on"}:
        return None
    from kernel_prompt_accounting import accounting_tokens
    counts = accounting_tokens(accounting)
    prompt_value, cache_value, total_prefill = (
        counts["uncached"], counts["cached"], counts["total"])
    if not all(type(value) is int for value in (prompt_value, cache_value, total_prefill)):
        return None
    if total_prefill <= 0:
        return None
    turn = step + 1 if type(step) is int and step >= 0 else None
    return {
        "cache_share_schema": "stateframe-prefix-cache-share/2",
        "request_id": request_id,
        "step": step if type(step) is int else None,
        "turn": turn,
        "prompt_n": prompt_value,
        "prompt_tokens": counts,
        "cache_n": cache_value,
        "cached_share": cache_value / total_prefill,
        "total_prefill_n": total_prefill,
        "basis": "normalized_prompt_tokens.cached_over_total",
    }


__all__ = [
    "DEFAULT_DELTA_BYTES", "DEFAULT_REBASE_TURNS", "DELTA_HEADER", "DELTA_SCHEMA",
    "FrameAppendError", "MODE_ENV", "PRODUCT_MODE", "SCHEMA", "AppendState",
    "PendingCommit", "Projection", "commit_after_response", "fail_open_pending",
    "materialize_messages", "materialize_payload", "mode", "prepare_for_capture",
    "project_request", "response_cache_share", "validate_composed_product",
]
