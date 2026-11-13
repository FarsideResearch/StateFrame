"""Opt-in append-only projection for source windows and volatile frame state."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass

from kernel_stable_wire_ownership import (
    CanonicalMessages, ProjectionOrder, RequestOwnership, projection_order, request_ownership,
    request_exchange_start, request_frame_index,
)


MODE_ENV = "STATEFRAME_STABLE_WIRE"
HEADROOM_ENV = "STATEFRAME_STABLE_WIRE_HEADROOM"
RETAIN_PREFIX_ENV = "STATEFRAME_STABLE_WIRE_RETAIN_PREFIX"
DEFAULT_HEADROOM_TOKENS = 16_000
MIN_HEADROOM_TOKENS = 6_000
STATE_ATTR = "_stable_wire_state"
STATE_SCHEMA = "stateframe-stable-wire/1"
VOLATILE_NAME = "stateframe_stable_wire_volatile"
STATUS_NAME = "stateframe_stable_wire_status"
VOLATILE_HEADER = "[StateFrame volatile]"
STATUS_HEADER = "[Source windows status]"
# Keys the caller shares with project_stable_wire through ``inputs`` while it retries a trimmed body.
RETRY_AFTER_TRIM_KEY = "_stable_wire_retry_after_trim"
TRIM_COUNT_KEY = "_stable_wire_trim_count"
DIET_NOTICE_HEADER = "[Request diet: dropped source ranges]"
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


def retain_prefix(environ: Mapping[str, str] | None = None) -> bool:
    """Whether a relayout may keep not-current (retained) windows inside its kept prefix."""
    source = os.environ if environ is None else environ
    value = str(source.get(RETAIN_PREFIX_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"invalid {RETAIN_PREFIX_ENV} mode")
    return value == "on"


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


def is_stable_wire_projection(messages: object, *, recorded: bool = False) -> bool:
    """Recognize controls; recorded inspection never consults replay providers."""
    if (isinstance(messages, CanonicalMessages)
            or not isinstance(messages, Sequence) or isinstance(messages, (str, bytes))):
        return False
    if not recorded and not enabled_for_replay():
        # Off mode retains even the historical plain-layout fallback decisions.
        return any(isinstance(row, Mapping)
                   and (row.get("name") == VOLATILE_NAME
                        or str(row.get("content", "")).startswith(VOLATILE_HEADER + "\n"))
                   for row in messages)
    # A native/host row can use any name or quote any header. The split-frame
    # protocol owns only the typed row immediately following the frame.
    return _split_frame_index(messages) is not None


def recorded_window_contents(messages: object) -> list[str] | None:
    """Source window contents of a recorded stable-wire request, in order, for replay."""
    if not is_stable_wire_projection(messages, recorded=True):
        return None
    frame_index = _frame_index(messages)
    order = projection_order(messages, frame_index)
    ownership = _recorded_window_ownership(order, frame_index)
    return [row["content"] for index, row in enumerate(
                messages[order.prefix_end:frame_index], start=order.prefix_end)
            if _is_current_window(row, index, ownership=ownership)]


def recorded_window_layout(messages: object) -> tuple[list[str], int] | None:
    """Return the recorded windows AND the epoch parsed from the status row."""
    windows = recorded_window_contents(messages)
    if windows is None:
        return None
    # recorded_window_contents validated the owned control position. Other rows
    # may use the same name or quote a status without acquiring its authority.
    status = messages[_frame_index(messages) + 2].get("content")
    if not isinstance(status, str):
        return None
    lines = status.splitlines()
    if len(lines) < 2 or lines[0] != STATUS_HEADER:
        return None
    prefix = "Epoch "
    epoch = lines[1][len(prefix):] if lines[1].startswith(prefix) else ""
    if not epoch.isascii() or not epoch.isdigit() or len(epoch) > 16 or int(epoch) < 1:
        return None
    return windows, int(epoch)


def replay_mismatch(capture, body, *, reason="replay_layout_mismatch", **fields):
    """Refuse an unproven replay layout without crashing or publishing stable state."""
    _emit(capture, "stable_wire_replay_mismatch",
          request_id=getattr(capture, "request_id", None), vital=True, **fields)
    return _failed_open(capture, body, reason, **fields)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _split_frame_index(messages: Sequence[object]) -> int | None:
    """Recognize controls at the end of the host's contiguous source prefix.

    The host's controls precede the entire tail (including captured users), so
    no later quoted frame or control can acquire ownership. Keep recognizing
    legacy controls without order metadata so strict readback can reject them.
    """
    if isinstance(messages, CanonicalMessages) or any(
            not isinstance(row, Mapping) for row in messages):
        return None
    try:
        end = request_exchange_start(messages)
    except ValueError:
        return None
    index = next((i for i, row in enumerate(messages) if row.get("role") != "system"), end) + 1
    while (index < end and messages[index].get("role") == "user"
           and str(messages[index].get("content", "")).startswith("[Current source:")):
        index += 1
    if index + 1 < end:
        row, volatile = messages[index:index + 2]
        if (row.get("role") == "user"
                and str(row.get("content", "")).startswith("[StateFrame]\n")
                and volatile.get("role") == "user"
                and volatile.get("name") == VOLATILE_NAME
                and str(volatile.get("content", "")).startswith(VOLATILE_HEADER + "\n")):
            return index
    return None


def _frame_index(messages: Sequence[object], *, capture: object = None,
                 exchange_start: int | None = None, newest_count: int = 0) -> int:
    split = _split_frame_index(messages)
    if split is not None:
        return split
    if any(not isinstance(row, Mapping) for row in messages):
        raise StableWireError("stable wire requires a message array")
    try:
        return request_frame_index(messages, capture=capture, exchange_start=exchange_start,
                                   newest_count=newest_count)
    except ValueError as error:
        raise StableWireError(str(error)) from error


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
    try:
        projection_order(messages, frame_index)
    except ValueError as error:
        raise StableWireError(str(error)) from error
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


def _host_window_row(row: object, index: int, ownership: RequestOwnership | None) -> bool:
    """Only proven host user rows before the frame may reach a source parser."""
    return (ownership is not None and index < ownership.frame_index
            and ownership.host_owned(index) and isinstance(row, Mapping)
            and row.get("role") == "user")


def _recorded_window_ownership(order: ProjectionOrder, frame_index: int) -> RequestOwnership:
    # Only the recorded source block belongs to this inventory. Everything
    # after the split frame (panels, captured users, exchange) remains opaque.
    return RequestOwnership(order.prefix_end, frame_index, frame_index + 1,
                            frame_index, frame_index)


def _is_current_window(row: object, index: int, *, ownership: RequestOwnership | None) -> bool:
    if not isinstance(row, Mapping) or not _host_window_row(row, index, ownership):
        return False
    from kernel_request_source_window import is_source_window, parse_model_source_window
    content = row.get("content")
    return (isinstance(content, str) and content.startswith("[Current source:")
            and (is_source_window(content) or parse_model_source_window(content) is not None))


def _identity_key(identity: Mapping[str, object]) -> str:
    return _canonical_json(identity)


def _rerendered_keys(delivered: Sequence[Mapping[str, object]],
                     current: Sequence[Mapping[str, object]]) -> set[str]:
    """Find delivered windows whose bytes were re-rendered under the same key.

    Owner, continuation and anchor metadata can change for an unchanged
    path/revision/range, and so can the rendered source lines (renderer drift,
    not a file edit: the revision is the whole-file digest). Both get the same
    answer, a fresh epoch. The relayout never resends the delivered bytes:
    ``_prefix_cap`` stops the kept prefix before the first changed row and
    ``_relayout_order`` takes every later row from the current render, so the
    final ``_project_body`` proof and ``verify_final_source_delivery`` still
    hold against the bytes actually sent. Refusing here ended the run (body 8).
    """
    delivered_by_key = {str(row["key"]): row for row in delivered}
    return {str(row["key"]) for row in current
            if (prior := delivered_by_key.get(str(row["key"]))) is not None
            and prior["content"] != row["content"]}


def _prefix_cap(previous_delivered: Sequence[Mapping[str, object]],
                current: Sequence[Mapping[str, object]], *, retain: bool = True) -> int:
    """Longest previous-layout prefix a relayout may keep.

    It stops at the first current row whose content changed (a re-rendered window). With
    ``retain`` off it also stops at the first row that is no longer current, so the kept
    prefix holds only current, byte-unchanged rows.
    """
    current_by_key = {str(row["key"]): row for row in current}
    for index, row in enumerate(previous_delivered):
        now = current_by_key.get(str(row["key"]))
        if now is None:
            if not retain:
                return index
        elif now["content"] != row["content"]:
            return index
    return len(previous_delivered)


def _relayout_order(previous_delivered: Sequence[Mapping[str, object]],
                    current: Sequence[Mapping[str, object]],
                    first_seen_rank: Mapping[str, int], keep: int
                    ) -> list[dict[str, object]]:
    """Build the relayout candidate that keeps the first ``keep`` delivered rows.

    Kept rows retain their delivered bytes whether or not they are still current, so the
    cacheable window prefix stays byte-identical. Every current row outside that prefix
    follows in run first-seen order. ``keep=0`` is a plain first-seen sort of the current rows.
    """
    kept = [copy.deepcopy(dict(row)) for row in previous_delivered[:keep]]
    kept_keys = {str(row["key"]) for row in kept}
    rest = sorted((row for row in current if str(row["key"]) not in kept_keys),
                  key=lambda row: first_seen_rank[str(row["key"])])
    return [*kept, *(copy.deepcopy(dict(row)) for row in rest)]


def _window_entries(messages: Sequence[object], ownership: RequestOwnership | None = None
                    ) -> list[dict[str, object]]:
    """Read proven host windows; absent ownership makes every row opaque."""
    entries = []
    seen: set[str] = set()
    for index, row in enumerate(messages):
        if not isinstance(row, Mapping) or not _host_window_row(row, index, ownership):
            continue
        content = row.get("content")
        if not isinstance(content, str) or not content.startswith("[Current source:"):
            continue
        if not _is_current_window(row, index, ownership=ownership):
            raise StableWireError("stable wire encountered an invalid current source window")
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


def _is_moved_row(row: Mapping[str, object], *, host_owned: bool = False) -> bool:
    """Only a host-owned panel can move; text and native metadata confer no ownership."""
    return host_owned


def _is_pre_window_notice(row: Mapping[str, object], *, host_owned: bool = False) -> bool:
    """Per-send notices the plain layout puts ahead of the windows.

    They change on most sends, so stable wire keeps them in the volatile tail. The canonical
    view (canonicalize_stable_wire) puts them back where the plain layout has them.
    """
    from kernel_request_source_window import DEFERRED_SOURCE_NOTICE_HEADER

    content = row.get("content")
    return (_is_moved_row(row, host_owned=host_owned) and isinstance(content, str)
            and content.startswith((DEFERRED_SOURCE_NOTICE_HEADER + "\n", DIET_NOTICE_HEADER + "\n")))


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
    if is_current:
        # A current window's own message already states its path, revision and ranges, so the
        # status only names it. Repeating a 64 character revision per window cost about 130
        # bytes each and pushed relayouts over the envelope.
        return f"{_status_atom(identity['path'])}: current"
    return (f"{_status_atom(identity['path'])}, revision "
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


def _fit_report(body: Mapping[str, object], meter: object,
                inputs: Mapping[str, object], capture: object = None) -> dict[str, object]:
    """Measure one request against its envelope. ``cause`` names why it does not fit."""
    measured = _measured_request(body, meter, capture)
    if measured is None:
        return {"fits": False, "cause": "measurement_unavailable"}
    tokens, method, size = measured
    report: dict[str, object] = {"fits": False, "prompt_tokens": tokens, "bytes": size,
                                 "measurement": method}
    expected_method = inputs.get("measurement_identity")
    try:
        token_ceiling, byte_ceiling = int(inputs["token_ceiling"]), int(inputs["byte_ceiling"])
    except (KeyError, TypeError, ValueError, OverflowError):
        report["cause"] = "ceiling_unavailable"
        return report
    report.update(token_ceiling=token_ceiling, byte_ceiling=byte_ceiling)
    if expected_method is not None and method != expected_method:
        report["cause"] = "measurement_method_mismatch"
        return report
    over_tokens, over_bytes = tokens > token_ceiling, size > byte_ceiling
    report["fits"] = not (over_tokens or over_bytes)
    report["cause"] = (None if report["fits"] else "tokens_and_bytes" if over_tokens and over_bytes
                       else "tokens" if over_tokens else "bytes")
    return report


def _record_disposition(capture: object, disposition: str, reason: str | None = None) -> None:
    from kernel_cache_layout_candidate import _record_stable_wire
    _record_stable_wire(capture, disposition, reason)


def _failed_open(capture: object, body: object, reason: str, **measured: object):
    """Fall back to the plain layout. ``measured`` carries numbers and flags only, never content."""
    _emit(capture, "stable_wire_failed_open", request_id=getattr(capture, "request_id", None),
          reason=reason, vital=True, **measured)
    _record_disposition(capture, "failed_open", reason)
    return body


def _request_retry_reserve(body: Mapping[str, object], meter: object,
                           inputs: Mapping[str, object], capture: object = None,
                           report: Mapping[str, object] | None = None) -> tuple[int, int] | None:
    """Reserve to trim for a request over the envelope; ``report`` reuses a measurement."""
    if report is not None and report.get("prompt_tokens") is not None \
            and report.get("bytes") is not None:
        tokens, size = int(report["prompt_tokens"]), int(report["bytes"])
    else:
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
                  current: Sequence[Mapping[str, object]], epoch: int, *,
                  ownership: RequestOwnership, prove: bool = True
                  ) -> tuple[dict[str, object], str, str]:
    """Project one layout. ``prove=False`` builds a probe whose bytes are only measured."""
    messages = body.get("messages")
    if not isinstance(messages, list) or any(not isinstance(row, Mapping) for row in messages):
        raise StableWireError("stable wire requires a message array")
    frame_index = ownership.frame_index
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

    # The host owns panels after the task, except the captured newest user turn.
    # Keep that turn in recorded order with the panels and contiguous with the
    # native exchange. Neither a row's name nor its content establishes ownership.
    first_tail = ownership.prefix_end
    exchange_start = ownership.exchange_start
    prefix, tail, suffix = [], [], []
    notices = []
    for index, row in enumerate(messages):
        host_owned = ownership.host_owned(index)
        if index == frame_index or _is_current_window(row, index, ownership=ownership):
            continue
        copied = copy.deepcopy(dict(row))
        if index < first_tail:
            prefix.append(copied)
        elif index < exchange_start:
            if _is_pre_window_notice(row, host_owned=host_owned):
                notices.append(len(tail))
            tail.append(copied)
        else:
            suffix.append(copied)

    frame_row = copy.deepcopy(dict(messages[frame_index]))
    frame_row["content"] = frame_content
    source_rows = [copy.deepcopy(dict(row["message"])) for row in final_delivered]
    status_content = _status_content(epoch, final_delivered, set(current_by_key))
    order = ProjectionOrder(
        len(prefix), len(tail), tuple(notices), ownership.newest_end - ownership.newest_start)
    status_content += order.encode()
    status_row = {"role": "user", "name": STATUS_NAME, "content": status_content}
    projected = copy.deepcopy(dict(body))
    # End the cacheable prefix after the append-only source block. Status and
    # frame state can change on every request, so both follow that boundary.
    projected["messages"] = [*prefix, *source_rows, frame_row,
                              volatile_message, status_row, *tail, *suffix]

    if prove:
        if materialize_stable_frame(projected["messages"]) != original_frame:
            raise StableWireError("stable wire frame reconstruction differs byte-for-byte")
        _prove_projection(messages, projected["messages"], frame_index, original_frame,
                          final_delivered, current_by_key, status_content, epoch, order)
    return projected, status_content, frame_content


def _prove_projection(input_messages: Sequence[object], output_messages: Sequence[object],
                      frame_index: int, original_frame: str,
                      delivered: Sequence[Mapping[str, object]],
                      current: Mapping[str, object], status_content: str, epoch: int,
                      order: ProjectionOrder) -> None:
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
    frame_position = order.prefix_end + len(delivered)
    stable_frame = output_messages[frame_position].get("content")
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
    if (frame_position + 1 >= len(output_messages)
            or not isinstance(output_messages[frame_position + 1], Mapping)
            or output_messages[frame_position + 1].get("name") != VOLATILE_NAME
            or frame_position + 2 >= len(output_messages)
            or not isinstance(output_messages[frame_position + 2], Mapping)
            or output_messages[frame_position + 2].get("name") != STATUS_NAME):
        raise StableWireError("stable wire post-frame messages are out of order")
    status_row = output_messages[frame_position + 2]
    if (status_row.get("content") != status_content
            or "\u2014" in status_content or "\u2013" in status_content):
        raise StableWireError("stable wire status message is missing")
    if _status_content(epoch, delivered, set(current)) + order.encode() != status_content:
        raise StableWireError("stable wire status does not name every delivered source window")
    if projection_order(output_messages, frame_position) != order:
        raise StableWireError("stable wire recorded row ownership differs from the projection")


def _volatile_content(messages: Sequence[object]) -> str:
    index = _frame_index(messages) + 1
    row = messages[index] if index < len(messages) else None
    if (not isinstance(row, Mapping) or row.get("name") != VOLATILE_NAME
            or not isinstance(row.get("content"), str)):
        raise StableWireError("stable wire volatile message is missing")
    return row["content"]


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
                          inputs: Mapping[str, object], messages: Sequence[object], *,
                          ownership: RequestOwnership) -> str:
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
    structure = {
        # Old checkpoints may have cached host panels or post-frame source text.
        # Changing ownership rules must establish a new prefix epoch on upgrade.
        "row_ownership": 2,
        # Whole rows preserve exact content, metadata and row order in the
        # cacheable preface. Host panels after this boundary live in the tail.
        "task": _task_text(capture, messages[:ownership.prefix_end], inputs),
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
        # The stable frame is deliberately NOT part of this signature: it is laid out
        # after the source windows, so a frame change only invalidates bytes from the
        # frame onward and must never relayout (rebase) the delivered windows.
    }
    try:
        # Preserve message-key order too: canonicalizing the rows themselves
        # would hide a byte change in the provider's serialized prefix.
        structure["preface"] = json.dumps(
            messages[:ownership.prefix_end], ensure_ascii=False, separators=(",", ":"),
            allow_nan=False)
        encoded = _canonical_json(structure).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise StableWireError("stable wire structural inputs are not JSON-safe") from error
    return hashlib.sha256(encoded).hexdigest()


_SIZE_CAUSES = frozenset({"tokens", "bytes", "tokens_and_bytes"})


def _recorded_prefix(previous_delivered, current, first_seen_rank,
                     recorded: Sequence[str] | None) -> int | None:
    """The kept prefix length that reproduces a recorded request's window rows, if any."""
    if recorded is None:
        return None
    # Any policy may have recorded it, so look at the widest prefix the rows allow.
    cap = _prefix_cap(previous_delivered, current, retain=True)
    shared = 0
    while (shared < min(cap, len(recorded))
           and str(previous_delivered[shared]["content"]) == recorded[shared]):
        shared += 1
    for keep in range(shared, -1, -1):
        order = _relayout_order(previous_delivered, current, first_seen_rank, keep)
        if [str(row["content"]) for row in order] == list(recorded):
            return keep
    return None


def _choose_relayout(body, meter, inputs, capture, previous_delivered, current,
                     first_seen_rank, epoch, *, ownership: RequestOwnership, measure: bool,
                     headroom_limit: int | None = None, retained_unavailable: bool = False,
                     rejected: tuple[list[str], int] | None = None, retain: bool = True,
                     recorded: Sequence[str] | None = None):
    """Pick the longest kept window prefix that fits the envelope and the retention headroom.

    Returns (candidate_delivered, projected, status_content, kept_prefix_windows, report).
    ``report`` is the chosen layout's fit report, or None when nothing was measured (replay).

    fits(k) = the non-current rows inside previous_delivered[:k] measure within
    ``headroom_limit`` (skipped when None) and the projected request fits. Fit is treated as
    monotone in k: keeping fewer prior rows never grows the request or its retained tokens.
    One binary search from the cap down therefore needs about log2(cap) meter calls. When the
    meter cannot measure retained rows the prefix stops at the first non-current row.
    ``rejected`` is (window key order, epoch) of an append-only layout the caller already
    measured over the envelope; the same bytes are not measured again. ``retain`` off keeps
    only current, unchanged rows in the prefix. ``recorded`` (replay) is the window contents
    of the recorded request: the k that reproduces it wins over any search.
    Probes are built without the full proof; only the chosen layout is proven.
    """
    current_keys = {str(row["key"]) for row in current}
    cap = _prefix_cap(previous_delivered, current, retain=retain)
    if retained_unavailable:
        cap = min(cap, next((index for index, row in enumerate(previous_delivered)
                             if str(row["key"]) not in current_keys), len(previous_delivered)))
    built: dict[int, tuple[list[dict[str, object]], dict[str, object], str]] = {}
    reports: dict[int, dict[str, object]] = {}

    def build(keep: int):
        if keep not in built:
            candidate = _relayout_order(previous_delivered, current, first_seen_rank, keep)
            projected, status, _frame = _project_body(
                body, candidate, current, epoch, ownership=ownership, prove=False)
            built[keep] = (candidate, projected, status)
        return built[keep]

    def retained_ok(keep: int) -> bool:
        if headroom_limit is None:
            return True
        rows = [row["message"] for row in previous_delivered[:keep]
                if str(row["key"]) not in current_keys]
        tokens, method = _source_tokens(meter, rows)
        return method != "unavailable" and tokens <= headroom_limit

    def already_rejected(keep: int) -> bool:
        if rejected is None or keep != len(previous_delivered):
            return False
        order = [str(row["key"]) for row in _relayout_order(
            previous_delivered, current, first_seen_rank, keep)]
        # Same window bytes and the same epoch digits mean the same status length.
        return order == rejected[0] and len(str(epoch)) == len(str(rejected[1]))

    def fits(keep: int) -> bool:
        if not retained_ok(keep):
            return False
        if not measure:
            return True
        reports[keep] = _fit_report(build(keep)[1], meter, inputs, capture)
        return bool(reports[keep]["fits"])

    chosen = _recorded_prefix(previous_delivered, current, first_seen_rank, recorded)
    if chosen is None:
        chosen = 0
        if measure and cap > 0:
            if not already_rejected(cap) and fits(cap):
                chosen = cap
            elif cap in reports and reports[cap].get("cause") not in _SIZE_CAUSES:
                # Not a size problem (method mismatch, no ceilings): fewer rows cannot help.
                chosen = cap
            else:
                low, high = 0, cap - 1
                while low < high:
                    mid = (low + high + 1) // 2
                    if fits(mid):
                        low = mid
                    else:
                        high = mid - 1
                chosen = low
    candidate, projected, status = build(chosen)
    report = reports.get(chosen)
    if measure and report is None:
        report = _fit_report(projected, meter, inputs, capture)
    # Probes skipped the proof; the layout that will be sent gets the full one.
    projected, status, _frame = _project_body(body, candidate, current, epoch, ownership=ownership)
    return candidate, projected, status, chosen, report


def project_stable_wire(capture, runtime, body, *, meter, inputs,
                        trim_callback=None, replay_decision=None, exchange_start=None):
    """Fail open on an unprovable wire; configuration/replan errors stay explicit."""
    from kernel_cache_layout_view import CacheLayoutError

    try:
        return _project_stable_wire(
            capture, runtime, body, meter=meter, inputs=inputs, trim_callback=trim_callback,
            replay_decision=replay_decision, exchange_start=exchange_start)
    except (StableWireError, CacheLayoutError):
        # Text can reach semantic validators through host-rendered frame values
        # as well as opaque rows. A failed proof must never end the run or
        # publish a partial epoch. Strict recorded-wire readers still reject
        # corrupt controls instead of silently inventing a semantic view.
        return _failed_open(capture, body, "unproven_stable_wire")


def _project_stable_wire(capture, runtime, body, *, meter, inputs,
                         trim_callback=None, replay_decision=None, exchange_start=None):
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
    recorded = None
    recorded_epoch = None
    if replay_decision == "applied":
        from kernel_cache_layout_candidate import replay_stable_wire_windows
        try:
            layout = replay_stable_wire_windows()
        except ValueError:
            # Legacy recordings cannot prove which source-shaped rows the host
            # owned. Keep the fitted plain request; exact-byte replay will stop
            # at this divergence without borrowing the old model response.
            return _failed_open(capture, body, "recorded_row_ownership_unavailable")
        # "applied" without recorded windows cannot prove any layout. Fail open explicitly,
        # including older recordings with an absent/malformed status row; never guess or crash.
        if layout is None:
            return replay_mismatch(capture, body)
        recorded, recorded_epoch = layout
    if not isinstance(body, Mapping) or not isinstance(inputs, Mapping):
        raise StableWireError("stable wire requires a fitted request body and inputs")
    messages = body.get("messages")
    if not isinstance(messages, list) or any(not isinstance(row, Mapping) for row in messages):
        raise StableWireError("stable wire requires a message array")
    try:
        frame_index = request_frame_index(messages, capture=capture, exchange_start=exchange_start)
    except ValueError as error:
        raise StableWireError(str(error)) from error
    ownership = request_ownership(
        messages, frame_index, capture=capture, exchange_start=exchange_start)
    headroom = headroom_tokens()
    run_id = _run_id(capture, runtime)
    previous = _validated_state(getattr(runtime, STATE_ATTR, None), run_id)
    current = _window_entries(messages, ownership)
    structure = _structural_signature(capture, runtime, body, inputs, messages, ownership=ownership)
    first_epoch = previous is None
    structural_rebase = previous is not None and previous.get("structure") != structure
    epoch = 1 if previous is None else previous["epoch"] + int(structural_rebase)
    previous_delivered = previous["delivered"] if previous is not None else []
    seen_keys = list(previous.get("seen_keys", ())) if previous is not None else []

    # A structural rebase discards the previous layout, so its rows are never compared.
    rerender_rebase = (not first_epoch and not structural_rebase
                       and bool(_rerendered_keys(previous_delivered, current)))
    epoch += int(rerender_rebase)
    for row in current:
        key = str(row["key"])
        if key not in seen_keys:
            seen_keys.append(key)

    delivered = ([] if first_epoch or structural_rebase or rerender_rebase
                 else previous_delivered)
    candidate_delivered = copy.deepcopy(delivered)
    known = {str(row["key"]): row for row in candidate_delivered}
    for row in current:
        key = str(row["key"])
        prior = known.get(key)
        if prior is not None and prior["content"] != row["content"]:
            # Unreachable: any changed row emptied ``delivered`` through rerender_rebase
            # (or a structural/first epoch). Kept as the append-only layout's invariant.
            raise StableWireError("stable wire refuses a re-rendered delivered source window")
        if prior is None:
            candidate_delivered.append(copy.deepcopy(dict(row)))
            known[key] = row

    current_keys = {str(row["key"]) for row in current}
    retained = [row for row in candidate_delivered if str(row["key"]) not in current_keys]
    measure = replay_decision != "applied"
    retained_tokens, retained_method = (_source_tokens(
        meter, [row["message"] for row in retained]) if measure else (0, "recorded"))
    headroom_rebase = (not first_epoch and not structural_rebase and not rerender_rebase
                       and (retained_method == "unavailable" or retained_tokens > headroom))

    reason = ("start" if first_epoch else "structural" if structural_rebase
              else "rerender" if rerender_rebase else None)
    must_relayout = first_epoch or structural_rebase or rerender_rebase
    if headroom_rebase:
        epoch += 1
        must_relayout = True
        reason = "headroom"

    first_seen_rank = {key: index for index, key in enumerate(seen_keys)}
    if structural_rebase:
        # The prefix ahead of the windows changed, so no window bytes are reusable.
        candidate_delivered = sorted(
            copy.deepcopy(current), key=lambda row: first_seen_rank[str(row["key"])])
    kept_prefix = 0
    report = None
    retain = retain_prefix()
    # On replay the recorded request tells which prefix the original send kept.
    # Meter-driven headroom/envelope relayouts are reconstructed from the recorded epoch.
    # Window equality alone misses a relayout that keeps k equal to all windows.
    replay_epoch_relayout = (recorded_epoch is not None and not must_relayout
                             and recorded_epoch == epoch + 1)
    if replay_epoch_relayout:
        epoch += 1
        must_relayout = True
        reason = "envelope"
    if (headroom_rebase or rerender_rebase or replay_epoch_relayout) and not structural_rebase:
        candidate_delivered, projected, status_content, kept_prefix, report = _choose_relayout(
            body, meter, inputs, capture, previous_delivered, current, first_seen_rank, epoch,
            ownership=ownership, measure=measure, headroom_limit=headroom,
            retained_unavailable=retained_method == "unavailable", retain=retain,
            recorded=recorded)
    else:
        projected, status_content, _frame = _project_body(
            body, candidate_delivered, current, epoch, ownership=ownership)
        replayed_relayout = (
            recorded is not None and not must_relayout
            and [str(row["content"]) for row in candidate_delivered] != list(recorded))
        if measure:
            report = _fit_report(projected, meter, inputs, capture)
        if ((measure and not report["fits"] and not must_relayout
                and report.get("cause") in _SIZE_CAUSES) or replayed_relayout):
            rejected = ([str(row["key"]) for row in candidate_delivered], epoch)
            epoch += 1
            must_relayout = True
            reason = "envelope"
            (candidate_delivered, projected, status_content, kept_prefix,
             report) = _choose_relayout(
                body, meter, inputs, capture, previous_delivered, current,
                first_seen_rank, epoch, ownership=ownership, measure=measure, rejected=rejected,
                retain=retain, recorded=recorded)

    if recorded is not None:
        # Replay must send the recorded window bytes or nothing stable at all. Every path
        # (append-only, envelope, structural, rerender, headroom) lands here before the send.
        projected_rows = [str(row["content"]) for row in candidate_delivered]
        from kernel_cache_layout_candidate import replay_stable_wire_reserves
        reserves_record = replay_stable_wire_reserves()
        if (projected_rows != list(recorded)
                or (recorded_epoch is not None and epoch != recorded_epoch)
                or (reserves_record is not None and epoch != reserves_record["epoch"])):
            shared = 0
            while (shared < min(len(projected_rows), len(recorded))
                   and projected_rows[shared] == recorded[shared]):
                shared += 1
            mismatch = dict(epoch=epoch, layout_reason=reason or "append_only",
                            relayout=bool(first_epoch or must_relayout),
                            kept_prefix_windows=kept_prefix,
                            recorded_window_count=len(recorded),
                            projected_window_count=len(projected_rows),
                            shared_prefix_windows=shared)
            if recorded_epoch is not None:
                mismatch["recorded_epoch"] = recorded_epoch
            return replay_mismatch(capture, body, **mismatch)

    order_failure = _order_failure_reason(messages, projected["messages"], ownership=ownership)
    if order_failure is not None:
        # Never hand allocation a wire whose canonical view reorders the request:
        # send the plain final layout instead, with the reason on record.
        return _failed_open(capture, body, order_failure)

    if report is not None and not report["fits"]:
        # Only a size miss can be cured by trimming; other causes fail open as measured.
        reserve = (_request_retry_reserve(projected, meter, inputs, capture, report)
                   if report.get("cause") in _SIZE_CAUSES else None)
        if reserve is not None and callable(trim_callback) and isinstance(inputs, dict):
            # The caller reads this count, so the event reports trims from earlier retries too.
            inputs[TRIM_COUNT_KEY] = int(inputs.get(TRIM_COUNT_KEY, 0)) + 1
            trimmed = trim_callback(*reserve)
            if trimmed is not False:
                inputs[RETRY_AFTER_TRIM_KEY] = True
                return body
        measured = {name: report[name] for name in (
            "cause", "prompt_tokens", "token_ceiling", "bytes", "byte_ceiling", "measurement")
            if report.get(name) is not None}
        trims = int(inputs.get(TRIM_COUNT_KEY, 0)) if isinstance(inputs, Mapping) else 0
        measured.update(trim_count=trims, trim_attempted=trims > 0,
                        trim_available=callable(trim_callback),
                        reserve_available=reserve is not None)
        return _failed_open(capture, body, "fresh_epoch_exceeds_measured_envelope", **measured)

    source_tokens, source_method = _source_tokens(
        meter, [row["message"] for row in candidate_delivered])
    if first_epoch or must_relayout:
        # The chosen layout decides what stays retained, so measure that, not the pre-relayout set.
        retained_tokens, _method = _source_tokens(meter, [
            row["message"] for row in candidate_delivered if str(row["key"]) not in current_keys])
        _emit(capture, "stable_wire_epoch", request_id=getattr(capture, "request_id", None),
              epoch=epoch, action="start" if first_epoch else "relayout",
              reason=reason or "structural", source_window_count=len(candidate_delivered),
              source_tokens=source_tokens, retained_tokens=retained_tokens,
              headroom=headroom, kept_prefix_windows=kept_prefix,
              retain_prefix=retain)

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


def _order_preserved(input_messages: Sequence[object], projected: Sequence[object], *,
                     ownership: RequestOwnership | None = None) -> bool:
    """Compare whole rows including the reconstructed frame's relative position."""
    return _order_failure_reason(input_messages, projected, ownership=ownership) is None


def _order_failure_reason(input_messages: Sequence[object], projected: Sequence[object], *,
                          ownership: RequestOwnership | None = None) -> str | None:
    """Prove order for whole rows, exempting only proven host windows/notices.

    Captured user rows and the native exchange always participate, regardless of
    their text, role or name. Recorded positions also make this proof available
    to callers that have only the input and serialized output messages.
    """
    stable = is_stable_wire_projection(projected)
    try:
        output_frame = _frame_index(projected)
        order = projection_order(projected, output_frame) if stable else None
        frame_index = (ownership.frame_index if ownership is not None else _frame_index(
            input_messages, newest_count=order.newest_count if order is not None else 0))
    except ValueError:
        # A broken native exchange or unprovable control position is itself an
        # order-proof failure; never recover its frame by scanning native text.
        return "projection_reorders_messages"
    if ownership is None and order is not None:
        ownership = request_ownership(
            input_messages, frame_index,
            newest_count=order.newest_count)
    input_skip = set()
    for index, row in enumerate(input_messages):
        if ownership is not None and ownership.host_owned(index) and isinstance(row, Mapping):
            if (_is_current_window(row, index, ownership=ownership)
                    or (stable and _is_pre_window_notice(row, host_owned=True))):
                input_skip.add(index)
    canonical, index_map = canonicalize_stable_wire(projected)
    canonical_frame = index_map[output_frame]
    output_skip = set()
    if order is not None:
        recorded_ownership = _recorded_window_ownership(order, output_frame)
        output_skip.update(index_map[i] for i, row in enumerate(projected)
                           if _is_current_window(row, i, ownership=recorded_ownership))
        output_skip.update(index_map[output_frame + 3 + i] for i in order.notices)
    else:
        output_skip.update(i for i, row in enumerate(canonical)
                           if _is_current_window(row, i, ownership=ownership))
    before = [_canonical_json(row) for i, row in enumerate(input_messages) if i not in input_skip]
    after = [_canonical_json(row) for i, row in enumerate(canonical) if i not in output_skip]
    if before == after:
        return None
    without_input_frame = [_canonical_json(row) for i, row in enumerate(input_messages)
                           if i not in input_skip and i != frame_index]
    without_output_frame = [_canonical_json(row) for i, row in enumerate(canonical)
                            if i not in output_skip and i != canonical_frame]
    input_position = frame_index - sum(i < frame_index for i in input_skip)
    output_position = canonical_frame - sum(i < canonical_frame for i in output_skip)
    if input_position != output_position and without_input_frame == without_output_frame:
        return "projection_moves_frame"
    return "projection_reorders_messages"


def canonicalize_stable_wire(messages: Sequence[object]) -> tuple[list[dict[str, object]], dict[int, int]]:
    """Rebuild the canonical view from host-recorded positions, never notice text."""
    if not is_stable_wire_projection(messages):
        return [copy.deepcopy(dict(row)) for row in messages if isinstance(row, Mapping)], {
            index: index for index, row in enumerate(messages) if isinstance(row, Mapping)}
    frame_index = _frame_index(messages)
    full_frame = materialize_stable_frame(messages)
    try:
        order = projection_order(messages, frame_index)
    except ValueError as error:
        raise StableWireError(str(error)) from error
    tail_start = frame_index + 3
    exchange_start = tail_start + order.tail_count
    notice_indexes = {tail_start + offset for offset in order.notices}
    physical_order = [
        *range(order.prefix_end),
        *sorted(notice_indexes),
        *range(order.prefix_end, frame_index),
        *(i for i in range(tail_start, exchange_start) if i not in notice_indexes),
        frame_index,
        *range(exchange_start, len(messages)),
    ]
    canonical = CanonicalMessages()
    for index in physical_order:
        row = messages[index]
        if not isinstance(row, Mapping):
            raise StableWireError("stable wire contains a malformed message")
        copied = copy.deepcopy(dict(row))
        if index == frame_index:
            copied["content"] = full_frame
        canonical.append(copied)
    canonical.frame_index = physical_order.index(frame_index)
    return canonical, {physical: index for index, physical in enumerate(physical_order)}


__all__ = [
    "DEFAULT_HEADROOM_TOKENS", "HEADROOM_ENV", "MODE_ENV", "STATE_ATTR", "STATUS_HEADER",
    "STATUS_NAME", "StableWireError", "VOLATILE_HEADER", "VOLATILE_NAME",
    "canonicalize_stable_wire", "headroom_tokens", "is_stable_wire_projection", "materialize_stable_frame",
    "mode", "project_stable_wire", "validate_checkpoint_state",
    "RETAIN_PREFIX_ENV", "RETRY_AFTER_TRIM_KEY", "TRIM_COUNT_KEY",
    "recorded_window_contents", "recorded_window_layout", "replay_mismatch", "retain_prefix",
]
