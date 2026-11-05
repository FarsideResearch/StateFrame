"""Opt-in append-only projection for source windows and volatile frame state."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass


MODE_ENV = "STATEFRAME_STABLE_WIRE"
HEADROOM_ENV = "STATEFRAME_STABLE_WIRE_HEADROOM"
DEFAULT_HEADROOM_TOKENS = 16_000
MIN_HEADROOM_TOKENS = 6_000
STATE_ATTR = "_stable_wire_state"
STATE_SCHEMA = "stateframe-stable-wire/1"
VOLATILE_NAME = "stateframe_stable_wire_volatile"
STATUS_NAME = "stateframe_stable_wire_status"
VOLATILE_HEADER = "[StateFrame volatile]"
STATUS_HEADER = "[Source windows status]"
_VOLATILE_PATHS = (
    ("state", "framefit5", "action_ledger"),
    ("state", "framefit5", "headcarry"),
    ("state", "framefit5", "open_sites"),
    ("state", "framefit5", "code_bookmarks"),
    ("recorded_events",),
    ("state", "recorded_events"),
    ("state", "framefit5", "recorded_events"),
)


class StableWireError(ValueError):
    """A stable-wire projection cannot prove its representation is faithful."""


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"invalid {MODE_ENV} mode")
    return value


def enabled_for_replay(environ: Mapping[str, str] | None = None) -> bool:
    """Keep a recorded stable projection active even if the current env was edited."""
    if mode(environ) == "on":
        return True
    from kernel_cache_layout_candidate import replay_stable_wire_decision
    return replay_stable_wire_decision() == "applied"


def headroom_tokens(environ: Mapping[str, str] | None = None) -> int:
    source = os.environ if environ is None else environ
    raw = str(source.get(HEADROOM_ENV, str(DEFAULT_HEADROOM_TOKENS))).strip()
    if not raw.isdecimal() or int(raw) <= 0:
        raise ValueError(f"{HEADROOM_ENV} must be a positive integer")
    value = int(raw)
    if value < MIN_HEADROOM_TOKENS:
        raise ValueError(
            f"{HEADROOM_ENV} below {MIN_HEADROOM_TOKENS} is refused because compactions every "
            "2-3 requests erase the gain (measured)"
        )
    return value


def is_stable_wire_projection(messages: object) -> bool:
    return (isinstance(messages, Sequence) and not isinstance(messages, (str, bytes))
            and any(isinstance(row, Mapping)
                    and (row.get("name") == VOLATILE_NAME
                         or str(row.get("content", "")).startswith(VOLATILE_HEADER + "\n"))
                    for row in messages))


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _frame_index(messages: Sequence[object]) -> int:
    indexes = [index for index, row in enumerate(messages)
               if isinstance(row, Mapping)
               and isinstance(row.get("content"), str)
               and row["content"].startswith("[StateFrame]\n")]
    if len(indexes) != 1:
        raise StableWireError("stable wire requires exactly one StateFrame")
    return indexes[0]


def _parse_frame(content: object) -> dict[str, object]:
    if not isinstance(content, str) or not content.startswith("[StateFrame]\n"):
        raise StableWireError("stable wire requires a canonical StateFrame")
    try:
        payload = json.loads(content.split("\n", 1)[1])
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise StableWireError("stable wire StateFrame JSON is invalid") from error
    if not isinstance(payload, dict):
        raise StableWireError("stable wire StateFrame payload is malformed")
    return payload


def _lookup(root: object, path: Sequence[str]) -> tuple[bool, object]:
    current = root
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _delete(root: dict[str, object], path: Sequence[str]) -> None:
    parent: object = root
    for key in path[:-1]:
        if not isinstance(parent, dict) or key not in parent:
            raise StableWireError("stable wire volatile frame path is malformed")
        parent = parent[key]
    if not isinstance(parent, dict) or path[-1] not in parent:
        raise StableWireError("stable wire volatile frame path is absent")
    del parent[path[-1]]


def _set(root: dict[str, object], path: Sequence[str], value: object) -> None:
    parent: dict[str, object] = root
    for key in path[:-1]:
        child = parent.get(key)
        if not isinstance(child, dict):
            raise StableWireError("stable wire volatile frame path is malformed")
        parent = child
    parent[path[-1]] = copy.deepcopy(value)


def _insert(root: dict[str, object], path: Sequence[str], value: object) -> None:
    parent = root
    for key in path[:-1]:
        child = parent.setdefault(key, {})
        if not isinstance(child, dict):
            raise StableWireError("stable wire volatile payload paths overlap")
        parent = child
    parent[path[-1]] = copy.deepcopy(value)


def _volatile_fields(payload: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    stable = copy.deepcopy(payload)
    volatile: dict[str, object] = {}
    for path in _VOLATILE_PATHS:
        present, value = _lookup(stable, path)
        if present:
            _insert(volatile, path, value)
            _delete(stable, path)
    return stable, volatile


def _volatile_message(data: Mapping[str, object]) -> dict[str, object]:
    content = VOLATILE_HEADER + "\n" + _canonical_json({
        "schema": "stateframe-stable-wire-volatile/1",
        "data": data,
    })
    return {"role": "user", "name": VOLATILE_NAME, "content": content}


def _parse_volatile_message(messages: Sequence[object], frame_index: int
                            ) -> dict[str, object]:
    volatile_index = frame_index + 1
    if volatile_index >= len(messages):
        raise StableWireError("stable wire volatile message must immediately follow StateFrame")
    row = messages[volatile_index]
    content = row.get("content") if isinstance(row, Mapping) else None
    if (not isinstance(content, str) or not content.startswith(VOLATILE_HEADER + "\n")
            or row.get("name") != VOLATILE_NAME):
        raise StableWireError("stable wire volatile message must immediately follow StateFrame")
    try:
        value = json.loads(content.split("\n", 1)[1])
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise StableWireError("stable wire volatile JSON is invalid") from error
    if (not isinstance(value, Mapping) or value.get("schema") != "stateframe-stable-wire-volatile/1"
            or set(value) != {"schema", "data"} or not isinstance(value.get("data"), Mapping)):
        raise StableWireError("stable wire volatile payload is malformed")
    data = value["data"]
    normalized: dict[str, object] = {}
    for path in _VOLATILE_PATHS:
        present, field_value = _lookup(data, path)
        if present:
            _insert(normalized, path, field_value)
    if normalized != data:
        raise StableWireError("stable wire volatile payload contains an unpermitted section")
    return normalized


def materialize_stable_frame(messages: Sequence[object]) -> str:
    """Rebuild the exact canonical frame from its stable and volatile messages."""
    frame_index = _frame_index(messages)
    payload = _parse_frame(messages[frame_index].get("content"))
    volatile = _parse_volatile_message(messages, frame_index)
    for path in _VOLATILE_PATHS:
        present, value = _lookup(volatile, path)
        if not present:
            continue
        retained, _value = _lookup(payload, path)
        if retained:
            raise StableWireError("volatile frame key remains in the stable StateFrame part")
        _set(payload, path, value)
    return "[StateFrame]\n" + _canonical_json(payload)


def _identity(content: str) -> dict[str, object]:
    from kernel_request_source_window import parse_model_source_window, parse_window

    parsed = parse_window(content)
    if parsed is None:
        parsed = parse_model_source_window(content)
    if not isinstance(parsed, Mapping):
        raise StableWireError("stable wire encountered a malformed current source window")
    path = parsed.get("path")
    revision = parsed.get("revision")
    ranges = parsed.get("ranges")
    if (not isinstance(path, str) or not path or not isinstance(revision, str)
            or not isinstance(ranges, list) or not ranges):
        raise StableWireError("stable wire source window identity is incomplete")
    normalized = []
    for item in ranges:
        if (not isinstance(item, Mapping) or type(item.get("start_line")) is not int
                or type(item.get("end_line")) is not int or item["start_line"] <= 0
                or item["end_line"] < item["start_line"]):
            raise StableWireError("stable wire source window line range is invalid")
        normalized.append([item["start_line"], item["end_line"]])
    return {"path": path, "revision": revision, "ranges": normalized}


def _is_current_window(content: object) -> bool:
    from kernel_request_source_window import is_source_window, parse_model_source_window
    return (isinstance(content, str) and content.startswith("[Current source:")
            and (is_source_window(content) or parse_model_source_window(content) is not None))


def _identity_key(identity: Mapping[str, object]) -> str:
    return _canonical_json(identity)


def _window_entries(messages: Sequence[object]) -> list[dict[str, object]]:
    entries = []
    seen: set[str] = set()
    for row in messages:
        if not isinstance(row, Mapping):
            raise StableWireError("stable wire requires message objects")
        content = row.get("content")
        if not isinstance(content, str) or not content.startswith("[Current source:"):
            continue
        if not _is_current_window(content):
            raise StableWireError("stable wire encountered an invalid current source window")
        if row.get("role") != "user":
            raise StableWireError("stable wire source windows must be user messages")
        identity = _identity(content)
        key = _identity_key(identity)
        if key in seen:
            raise StableWireError("stable wire input repeats a current source window")
        seen.add(key)
        entries.append({"identity": identity, "key": key, "content": content,
                        "message": copy.deepcopy(dict(row))})
    return entries


def _validated_state(value: object, run_id: str) -> dict[str, object] | None:
    if value is None:
        return None
    if (not isinstance(value, Mapping) or value.get("schema") != STATE_SCHEMA
            or type(value.get("epoch")) is not int or value["epoch"] < 1
            or not isinstance(value.get("delivered"), list)):
        raise StableWireError("stable wire checkpoint state is malformed")
    if value.get("run_id") != run_id:
        return None
    delivered = []
    seen = set()
    for item in value["delivered"]:
        if (not isinstance(item, Mapping) or not isinstance(item.get("identity"), Mapping)
                or not isinstance(item.get("message"), Mapping)
                or not isinstance(item.get("content"), str)):
            raise StableWireError("stable wire checkpoint window is malformed")
        parsed_identity = _identity(item["content"])
        if _identity_key(parsed_identity) != _identity_key(item["identity"]):
            raise StableWireError("stable wire checkpoint window metadata changed")
        key = _identity_key(item["identity"])
        if (key in seen or item["message"].get("content") != item["content"]
                or item["message"].get("role") != "user"):
            raise StableWireError("stable wire checkpoint window identity is inconsistent")
        seen.add(key)
        delivered.append({"identity": copy.deepcopy(dict(item["identity"])), "key": key,
                          "content": item["content"], "message": copy.deepcopy(dict(item["message"]))})
    raw_seen = value.get("seen_keys", [str(row["key"]) for row in delivered])
    if (not isinstance(raw_seen, list) or any(not isinstance(key, str) for key in raw_seen)
            or len(raw_seen) != len(set(raw_seen)) or not seen.issubset(set(raw_seen))):
        raise StableWireError("stable wire checkpoint first-seen window order is malformed")
    return {"schema": STATE_SCHEMA, "run_id": run_id, "epoch": value["epoch"],
            "delivered": delivered,
            "seen_keys": list(raw_seen),
            "structure": value.get("structure") if isinstance(value.get("structure"), str) else None}


def validate_checkpoint_state(value: object) -> dict[str, object] | None:
    """Validate the JSON-safe state field before accepting it from a checkpoint."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or not isinstance(value.get("run_id"), str):
        raise StableWireError("stable wire checkpoint state is malformed")
    validated = _validated_state(value, value["run_id"])
    if validated is None:
        raise StableWireError("stable wire checkpoint state has no run identity")
    return validated


def _is_moved_row(row: Mapping[str, object]) -> bool:
    content = row.get("content")
    if not isinstance(content, str):
        return False
    source_summary = (
        content.startswith("kept ")
        and content.endswith(" chars of source fit")
        and (" selected source windows; about " in content
             or " recent reads; about " in content)
    )
    return (source_summary or content.startswith((
        "[Host fact]", "[tool guidance]", "[Tool menu | this turn]", "[Source coverage]",
    )))


def _status_atom(value: object) -> str:
    """Escape path and revision punctuation so status text has no dash characters."""
    encoded = json.dumps(str(value), ensure_ascii=True)[1:-1]
    return encoded.replace("-", "\\u002d")


def _window_ranges(identity: Mapping[str, object]) -> str:
    ranges = identity.get("ranges")
    if not isinstance(ranges, list):
        raise StableWireError("stable wire source window ranges are malformed")
    rendered = []
    for row in ranges:
        if not isinstance(row, list) or len(row) != 2:
            raise StableWireError("stable wire source window ranges are malformed")
        rendered.append(f"lines {row[0]} to {row[1]}")
    return " and ".join(rendered)


def _status_line(row: Mapping[str, object], *, is_current: bool) -> str:
    identity = row["identity"]
    label = ": current" if is_current else ""
    return (f"{_status_atom(identity['path'])}{label}, revision "
            f"{_status_atom(identity['revision'])}, {_window_ranges(identity)}")


def _status_content(epoch: int, delivered: Sequence[Mapping[str, object]], current: set[str]) -> str:
    current_rows = [row for row in delivered if str(row["key"]) in current]
    retained_rows = [row for row in delivered if str(row["key"]) not in current]
    lines = [STATUS_HEADER, f"Epoch {epoch}", "Current windows:"]
    lines.extend(_status_line(row, is_current=True) for row in current_rows)
    if retained_rows:
        lines.append("Not current, do not edit from these:")
        lines.extend(_status_line(row, is_current=False) for row in retained_rows)
    content = "\n".join(lines)
    if "\u2014" in content or "\u2013" in content:
        raise StableWireError("stable wire status contains dash punctuation")
    return content


def _source_tokens(meter: object, rows: Sequence[Mapping[str, object]]) -> tuple[int, str]:
    if not rows:
        return 0, "tokenize"
    measure = getattr(meter, "measure", None)
    if not callable(measure):
        return 0, "unavailable"
    try:
        value = measure({"messages": [copy.deepcopy(dict(row)) for row in rows]})
    except Exception:
        return 0, "unavailable"
    if (not isinstance(value, tuple) or len(value) != 2 or type(value[0]) is not int
            or value[0] < 0 or not isinstance(value[1], str)):
        return 0, "unavailable"
    return value


def _measured_request(body: Mapping[str, object], meter: object, capture: object = None):
    try:
        if capture is None:
            value = meter.measure(body)
        else:
            from kernel_frame_allocation_capture import final_request_tokens
            value = final_request_tokens(body, meter, capture)
    except Exception:
        return None
    if (not isinstance(value, tuple) or len(value) != 2 or type(value[0]) is not int
            or value[0] < 0 or not isinstance(value[1], str) or not value[1]):
        return None
    try:
        size = len(json.dumps(body).encode("utf-8"))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return value[0], value[1], size


def _request_fits(body: Mapping[str, object], meter: object,
                  inputs: Mapping[str, object], capture: object = None) -> bool:
    measured = _measured_request(body, meter, capture)
    if measured is None:
        return False
    tokens, method, size = measured
    expected_method = inputs.get("measurement_identity")
    if expected_method is not None and method != expected_method:
        return False
    try:
        return (tokens <= int(inputs["token_ceiling"])
                and size <= int(inputs["byte_ceiling"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _record_disposition(capture: object, disposition: str, reason: str | None = None) -> None:
    from kernel_cache_layout_candidate import _record_stable_wire
    _record_stable_wire(capture, disposition, reason)


def _failed_open(capture: object, body: object, reason: str):
    _emit(capture, "stable_wire_failed_open", request_id=getattr(capture, "request_id", None),
          reason=reason, vital=True)
    _record_disposition(capture, "failed_open", reason)
    return body


def _request_retry_reserve(body: Mapping[str, object], meter: object,
                           inputs: Mapping[str, object], capture: object = None) -> tuple[int, int] | None:
    measured = _measured_request(body, meter, capture)
    if measured is None:
        return None
    tokens, _method, size = measured
    try:
        from kernel_cache_layout_candidate import _final_layout_reserve
        return _final_layout_reserve(
            tokens, int(inputs["token_ceiling"]), size, int(inputs["byte_ceiling"]))
    except (ImportError, KeyError, TypeError, ValueError, OverflowError):
        return None


def _project_body(body: Mapping[str, object], delivered: Sequence[Mapping[str, object]],
                  current: Sequence[Mapping[str, object]], epoch: int
                  ) -> tuple[dict[str, object], str, str]:
    messages = body.get("messages")
    if not isinstance(messages, list) or any(not isinstance(row, Mapping) for row in messages):
        raise StableWireError("stable wire requires a message array")
    frame_index = _frame_index(messages)
    original_frame = messages[frame_index].get("content")
    payload = _parse_frame(original_frame)
    stable_payload, volatile_fields = _volatile_fields(payload)
    frame_content = "[StateFrame]\n" + _canonical_json(stable_payload)
    volatile_message = _volatile_message(volatile_fields)

    current_by_key = {str(row["key"]): row for row in current}
    delivered_by_key = {str(row["key"]): row for row in delivered}
    for key, row in current_by_key.items():
        previous = delivered_by_key.get(key)
        if previous is not None and previous["content"] != row["content"]:
            raise StableWireError("stable wire refuses a re-rendered delivered source window")
    final_delivered = [copy.deepcopy(dict(row)) for row in delivered]
    for row in current:
        if str(row["key"]) not in delivered_by_key:
            final_delivered.append(copy.deepcopy(dict(row)))

    # Collect once across the whole input so moved rows retain their recorded order.
    moved = [copy.deepcopy(dict(row)) for row in messages
             if isinstance(row, Mapping) and _is_moved_row(row)]
    prefix = []
    source_indexes = set()
    for index, row in enumerate(messages):
        content = row.get("content")
        if _is_current_window(content):
            source_indexes.add(index)
        elif index < frame_index and not _is_moved_row(row):
            prefix.append(copy.deepcopy(dict(row)))
    suffix = []
    for index, row in enumerate(messages[frame_index + 1:], frame_index + 1):
        content = row.get("content")
        if index in source_indexes or _is_moved_row(row):
            continue
        suffix.append(copy.deepcopy(dict(row)))

    frame_row = copy.deepcopy(dict(messages[frame_index]))
    frame_row["content"] = frame_content
    source_rows = [copy.deepcopy(dict(row["message"])) for row in final_delivered]
    status_content = _status_content(epoch, final_delivered, set(current_by_key))
    status_row = {"role": "user", "name": STATUS_NAME, "content": status_content}
    projected = copy.deepcopy(dict(body))
    # End the cacheable prefix after the append-only source block. Status and
    # frame state can change on every request, so both follow that boundary.
    projected["messages"] = [*prefix, *source_rows, frame_row,
                              volatile_message, status_row, *moved, *suffix]

    if materialize_stable_frame(projected["messages"]) != original_frame:
        raise StableWireError("stable wire frame reconstruction differs byte-for-byte")
    _prove_projection(messages, projected["messages"], frame_index, original_frame,
                      final_delivered, current_by_key, status_content, epoch)
    return projected, status_content, frame_content


def _prove_projection(input_messages: Sequence[object], output_messages: Sequence[object],
                      frame_index: int, original_frame: str,
                      delivered: Sequence[Mapping[str, object]],
                      current: Mapping[str, object], status_content: str, epoch: int) -> None:
    def content_counts(rows: Sequence[object], *, skip: int | None = None) -> Counter[str]:
        counts: Counter[str] = Counter()
        for index, row in enumerate(rows):
            if index == skip or not isinstance(row, Mapping):
                continue
            try:
                content = _canonical_json(row.get("content"))
            except (TypeError, ValueError, OverflowError) as error:
                raise StableWireError("stable wire message content is not JSON-safe") from error
            counts[content] += 1
        return counts

    output_contents = content_counts(output_messages)
    expected = content_counts(input_messages, skip=frame_index)
    expected.update({
        _canonical_json(_volatile_content(output_messages)): 1,
        _canonical_json(status_content): 1,
    })
    stable_frame = next(row.get("content") for row in output_messages
                        if isinstance(row, Mapping)
                        and isinstance(row.get("content"), str)
                        and row["content"].startswith("[StateFrame]\n"))
    expected[_canonical_json(stable_frame)] += 1
    for row in delivered:
        identity_key = str(row["key"])
        if identity_key not in current:
            expected[_canonical_json(row["content"])] += 1
    rebuilt = materialize_stable_frame(output_messages)
    if rebuilt != original_frame:
        raise StableWireError("stable wire frame proof failed")
    if output_contents != expected:
        raise StableWireError("stable wire projection added or lost an unapproved message")
    status_rows = [row for row in output_messages
                   if isinstance(row, Mapping) and row.get("name") == STATUS_NAME]
    if (len(status_rows) != 1 or status_rows[0].get("content") != status_content
            or "\u2014" in status_content or "\u2013" in status_content):
        raise StableWireError("stable wire status message is missing")
    frame_position = _frame_index(output_messages)
    if (frame_position + 1 >= len(output_messages)
            or not isinstance(output_messages[frame_position + 1], Mapping)
            or output_messages[frame_position + 1].get("name") != VOLATILE_NAME
            or frame_position + 2 >= len(output_messages)
            or not isinstance(output_messages[frame_position + 2], Mapping)
            or output_messages[frame_position + 2].get("name") != STATUS_NAME):
        raise StableWireError("stable wire post-frame messages are out of order")
    if _status_content(epoch, delivered, set(current)) != status_content:
        raise StableWireError("stable wire status does not name every delivered source window")


def _volatile_content(messages: Sequence[object]) -> str:
    rows = [row for row in messages if isinstance(row, Mapping) and row.get("name") == VOLATILE_NAME]
    if len(rows) != 1 or not isinstance(rows[0].get("content"), str):
        raise StableWireError("stable wire volatile message is missing")
    return rows[0]["content"]


def _emit(capture: object, kind: str, **fields: object) -> None:
    journal = getattr(capture, "journal", None)
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    try:
        emit(kind, producer="kernel_stable_wire", dispatch=False, **fields)
        from kernel_stateframe_page_log import _observe_run_vitals_event
        _observe_run_vitals_event(journal, kind, fields)
    except Exception:
        return


def _run_id(capture: object, runtime: object) -> str:
    journal = getattr(capture, "journal", None)
    value = getattr(journal, "run_id", None) or getattr(runtime, "run_id", None)
    return value if isinstance(value, str) else ""


def _task_text(capture: object, messages: Sequence[object], inputs: Mapping[str, object]) -> str:
    value = next((row.get("content") for row in messages
                  if isinstance(row, Mapping) and isinstance(row.get("content"), str)
                  and row["content"].startswith(("Task:", "[Task]"))), None)
    if not isinstance(value, str):
        value = getattr(capture, "_framefit5_task_message", None)
        if isinstance(value, Mapping):
            value = value.get("content")
    if not isinstance(value, str):
        value = inputs.get("task_text")
    if not isinstance(value, str):
        value = next((row.get("content") for row in messages
                      if isinstance(row, Mapping) and isinstance(row.get("content"), str)
                      and row["content"].startswith(("Task:", "[Task]"))), "")
    if not isinstance(value, str):
        raise StableWireError("stable wire task text is malformed")
    return value


def _authority_shape(value: object) -> dict[str, object]:
    """Keep only explicit authority fields, not per-request source-plan contents."""
    names = {"edit_authority", "editable_paths", "editable_set", "workspace_authorization"}
    found: dict[str, object] = {}
    if not isinstance(value, Mapping):
        return found
    for label in ("inputs", "source_plan", "capture", "runtime", "body"):
        node = value.get(label)
        if not isinstance(node, Mapping):
            continue
        for key in names:
            if key not in node:
                continue
            selected = _authority_value(node[key])
            if key == "workspace_authorization" and isinstance(selected, Mapping):
                selected = {name: selected[name] for name in ("scope", "editable_paths")
                            if name in selected}
            found[f"{label}.{key}"] = selected
    return found


def _authority_value(value: object) -> object:
    if is_dataclass(value):
        return _authority_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _authority_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_authority_value(child) for child in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return copy.deepcopy(value)
    return type(value).__qualname__


def _tool_heads(body: Mapping[str, object]) -> list[str]:
    tools = body.get("tools", [])
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        raise StableWireError("stable wire tool menu is malformed")
    heads = []
    for row in tools:
        if not isinstance(row, Mapping):
            raise StableWireError("stable wire tool menu is malformed")
        function = row.get("function")
        name = function.get("name") if isinstance(function, Mapping) else row.get("name")
        if not isinstance(name, str) or not name:
            raise StableWireError("stable wire tool head is missing")
        heads.append(name)
    return heads


def _structural_signature(capture: object, runtime: object, body: Mapping[str, object],
                          inputs: Mapping[str, object], messages: Sequence[object]) -> str:
    profile = inputs.get("profile", inputs.get("profile_name"))
    if profile is None:
        profile = getattr(runtime, "profile_name", None)
    if profile is None:
        profile = os.environ.get("STATEFRAME_PROFILE", "")
    if not isinstance(profile, (str, int, float, bool, type(None))):
        if isinstance(profile, Mapping):
            profile = copy.deepcopy(dict(profile))
        elif isinstance(profile, (list, tuple)):
            profile = copy.deepcopy(list(profile))
        else:
            profile = getattr(profile, "name", type(profile).__qualname__)
    switches = inputs.get("switch_set", inputs.get("switches"))
    if switches is None:
        secret_markers = ("_KEY", "_TOKEN", "_URL", "_ENDPOINT", "_MODEL")
        switches = {key: value for key, value in os.environ.items()
                    if key.startswith("STATEFRAME_")
                    and not key.endswith(secret_markers)}
    if isinstance(switches, (set, frozenset)):
        switches = sorted(str(value) for value in switches)
    runtime_plan = getattr(runtime, "_request_source_closure", {})
    raw_runtime_authorization = (runtime_plan.get("workspace_authorization")
                                 if isinstance(runtime_plan, Mapping) else None)
    runtime_authorization = (
        {key: raw_runtime_authorization[key] for key in ("scope", "editable_paths")
         if key in raw_runtime_authorization}
        if isinstance(raw_runtime_authorization, Mapping) else None
    )
    frame_index = _frame_index(messages)
    frame_payload = _parse_frame(messages[frame_index].get("content"))
    stable_frame, _volatile = _volatile_fields(frame_payload)
    structure = {
        "task": _task_text(capture, messages, inputs),
        "authority": _authority_shape({
            "inputs": inputs,
            "source_plan": inputs.get("source_plan"),
            "capture": {"editable_paths": getattr(capture, "editable_paths", None)},
            "runtime": {"editable_paths": getattr(runtime, "editable_paths", None),
                        "workspace_authorization": runtime_authorization},
            "body": {key: body.get(key) for key in ("edit_authority", "editable_paths")
                     if key in body},
        }),
        "tool_heads": _tool_heads(body),
        "profile": profile,
        "switches": switches,
        # The stable frame is itself part of the immutable message prefix.
        # A changed frame starts a fresh epoch so message-boundary reuse stays exact.
        "stable_frame": stable_frame,
    }
    try:
        encoded = _canonical_json(structure).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise StableWireError("stable wire structural inputs are not JSON-safe") from error
    return hashlib.sha256(encoded).hexdigest()


def project_stable_wire(capture, runtime, body, *, meter, inputs,
                        trim_callback=None, replay_decision=None):
    """Project one final-layout request while proving exact frame/source delivery."""
    if replay_decision is None:
        from kernel_cache_layout_candidate import replay_stable_wire_decision
        replay_decision = replay_stable_wire_decision()
    if replay_decision == "plain":
        _record_disposition(capture, "plain", "recorded_plain_wire")
        return body
    if mode() == "off" and replay_decision != "applied":
        _record_disposition(capture, "off")
        return body
    if not isinstance(body, Mapping) or not isinstance(inputs, Mapping):
        raise StableWireError("stable wire requires a fitted request body and inputs")
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise StableWireError("stable wire requires a message array")
    headroom = headroom_tokens()
    run_id = _run_id(capture, runtime)
    previous = _validated_state(getattr(runtime, STATE_ATTR, None), run_id)
    current = _window_entries(messages)
    structure = _structural_signature(capture, runtime, body, inputs, messages)
    first_epoch = previous is None
    structural_rebase = previous is not None and previous.get("structure") != structure
    epoch = 1 if previous is None else previous["epoch"] + int(structural_rebase)
    previous_delivered = previous["delivered"] if previous is not None else []
    seen_keys = list(previous.get("seen_keys", ())) if previous is not None else []

    prior_by_key = {str(row["key"]): row for row in previous_delivered}
    for row in current:
        key = str(row["key"])
        prior = prior_by_key.get(key)
        if prior is not None and prior["content"] != row["content"]:
            raise StableWireError("stable wire refuses a re-rendered delivered source window")
        if key not in seen_keys:
            seen_keys.append(key)

    delivered = [] if first_epoch or structural_rebase else previous_delivered
    candidate_delivered = copy.deepcopy(delivered)
    known = {str(row["key"]): row for row in candidate_delivered}
    for row in current:
        key = str(row["key"])
        prior = known.get(key)
        if prior is not None and prior["content"] != row["content"]:
            raise StableWireError("stable wire refuses a re-rendered delivered source window")
        if prior is None:
            candidate_delivered.append(copy.deepcopy(dict(row)))
            known[key] = row

    current_keys = {str(row["key"]) for row in current}
    retained = [row for row in candidate_delivered if str(row["key"]) not in current_keys]
    retained_tokens, retained_method = _source_tokens(
        meter, [row["message"] for row in retained])
    headroom_rebase = (not first_epoch and not structural_rebase
                       and (retained_method == "unavailable" or retained_tokens > headroom))

    reason = "start" if first_epoch else "structural" if structural_rebase else None
    must_relayout = first_epoch or structural_rebase
    if headroom_rebase:
        epoch += 1
        must_relayout = True
        reason = "headroom"

    first_seen_rank = {key: index for index, key in enumerate(seen_keys)}
    if structural_rebase or headroom_rebase:
        candidate_delivered = sorted(
            copy.deepcopy(current), key=lambda row: first_seen_rank[str(row["key"])])

    projected, status_content, _frame = _project_body(body, candidate_delivered, current, epoch)

    if replay_decision != "applied" and not _request_fits(projected, meter, inputs, capture):
        if not must_relayout:
            epoch += 1
            candidate_delivered = sorted(
                copy.deepcopy(current), key=lambda row: first_seen_rank[str(row["key"])])
            must_relayout = True
            reason = "envelope"
            projected, status_content, _frame = _project_body(
                body, candidate_delivered, current, epoch)
        if not _request_fits(projected, meter, inputs, capture):
            reserve = _request_retry_reserve(projected, meter, inputs, capture)
            if reserve is not None and callable(trim_callback):
                trimmed = trim_callback(*reserve)
                if trimmed is not False and isinstance(inputs, dict):
                    inputs["_stable_wire_retry_after_trim"] = True
                    return body
            return _failed_open(capture, body, "fresh_epoch_exceeds_measured_envelope")

    source_tokens, source_method = _source_tokens(
        meter, [row["message"] for row in candidate_delivered])
    if first_epoch or must_relayout:
        _emit(capture, "stable_wire_epoch", request_id=getattr(capture, "request_id", None),
              epoch=epoch, action="start" if first_epoch else "relayout",
              reason=reason or "structural", source_window_count=len(candidate_delivered),
              source_tokens=source_tokens, retained_tokens=retained_tokens,
              headroom=headroom)

    try:
        from kernel_request_source_window import source_delivery_satisfied, verify_final_source_delivery
        receipt = verify_final_source_delivery(
            inputs["source_plan"], projected["messages"], payload=projected,
            sources=inputs.get("sources"),
        )
        if not source_delivery_satisfied(receipt):
            raise StableWireError("stable wire source delivery proof failed")
    except StableWireError:
        raise
    except Exception as error:
        raise StableWireError("stable wire source delivery proof failed") from error

    setattr(runtime, STATE_ATTR, {
        "schema": STATE_SCHEMA,
        "run_id": run_id,
        "epoch": epoch,
        "delivered": copy.deepcopy(candidate_delivered),
        "seen_keys": seen_keys,
        "structure": structure,
    })
    inputs["receipt"] = receipt
    if capture is not None:
        capture.request_source_final_delivery = receipt
    _record_disposition(capture, "applied")
    return projected


def canonicalize_stable_wire(messages: Sequence[object]) -> tuple[list[dict[str, object]], dict[int, int]]:
    """Return the canonical final-layout view and physical-to-canonical index map."""
    if not is_stable_wire_projection(messages):
        return [copy.deepcopy(dict(row)) for row in messages if isinstance(row, Mapping)], {
            index: index for index, row in enumerate(messages) if isinstance(row, Mapping)}
    frame_index = _frame_index(messages)
    full_frame = materialize_stable_frame(messages)
    prefix: list[tuple[int, dict[str, object]]] = []
    source: list[tuple[int, dict[str, object]]] = []
    moved: list[tuple[int, dict[str, object]]] = []
    suffix: list[tuple[int, dict[str, object]]] = []
    for index, row in enumerate(messages):
        if not isinstance(row, Mapping):
            continue
        content = row.get("content")
        if index == frame_index:
            continue
        if row.get("name") in {VOLATILE_NAME, STATUS_NAME}:
            continue
        copied = copy.deepcopy(dict(row))
        if _is_current_window(content):
            source.append((index, copied))
        elif _is_moved_row(row):
            moved.append((index, copied))
        elif index < frame_index:
            prefix.append((index, copied))
        else:
            suffix.append((index, copied))
    frame_row = copy.deepcopy(dict(messages[frame_index]))
    frame_row["content"] = full_frame
    canonical_rows = [*prefix, *source, *moved, (frame_index, frame_row), *suffix]
    return [row for _physical, row in canonical_rows], {
        physical: canonical for canonical, (physical, _row) in enumerate(canonical_rows)
    }


__all__ = [
    "DEFAULT_HEADROOM_TOKENS", "HEADROOM_ENV", "MODE_ENV", "STATE_ATTR", "STATUS_HEADER",
    "STATUS_NAME", "StableWireError", "VOLATILE_HEADER", "VOLATILE_NAME",
    "canonicalize_stable_wire", "headroom_tokens", "is_stable_wire_projection", "materialize_stable_frame",
    "mode", "project_stable_wire", "validate_checkpoint_state",
]
