"""Read-only materialization of the versioned cache wire, preserving locations.

The submitted bytes remain the authority for source delivery and allocation.
This view restores the current typed frame for semantic validators only.
"""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence

from kernel_stateframe_final_message import AUTHORITY_OVERLAY_HEADER, PAGE_LOG_FINAL_MESSAGE_NAME
from kernel_stateframe_frame_append import DELTA_HEADER, materialize_payload
from kernel_stateframe_message_layout import validate_stateframe_message_order


FINAL_LAYOUT_NAME = "stateframe_final_cache_1"


class CacheLayoutError(ValueError):
    pass


def is_cache_layout(messages: object) -> bool:
    return (isinstance(messages, Sequence) and not isinstance(messages, (str, bytes))
            and bool(messages) and isinstance(messages[-1], Mapping)
            and messages[-1].get("name") in {PAGE_LOG_FINAL_MESSAGE_NAME, FINAL_LAYOUT_NAME})


def _final_layout(messages):
    """Validate and rebuild the original message order from the submitted wire."""
    from kernel_stateframe_page_log import (
        _partition_authority, _payload_has_authority_keys, _restore_authority,
    )
    from kernel_stateframe_message_layout import stateframe_message_index

    if not isinstance(messages, list) or len(messages) < 4:
        raise CacheLayoutError("final cache layout has no message array")
    if any(not isinstance(row, Mapping) for row in messages):
        raise CacheLayoutError("final cache layout contains a malformed message")
    final = messages[-1]
    marker = AUTHORITY_OVERLAY_HEADER + "\n"
    content = final.get("content")
    if final.get("role") != "user" or not isinstance(content, str) or content.count(marker) != 1:
        raise CacheLayoutError("final cache layout requires one authority overlay")
    marker_index = content.index(marker)
    wire_prefix = content[:marker_index]
    encoded = content[marker_index + len(marker):]
    try:
        overlay, consumed = json.JSONDecoder().raw_decode(encoded)
        state_index = stateframe_message_index(messages)
        base = json.loads(messages[state_index]["content"].split("\n", 1)[1])
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise CacheLayoutError("final cache layout has invalid typed data") from error
    if (not isinstance(overlay, dict) or overlay.get("schema") != "stateframe-authority-overlay/1"
            or overlay.get("layout") != FINAL_LAYOUT_NAME
            or not isinstance(overlay.get("fields"), list)
            or _payload_has_authority_keys(base, final=True)):
        raise CacheLayoutError("final cache layout authority is malformed")
    from kernel_greenfact import GREENFACT_TEXT
    green = GREENFACT_TEXT if encoded[consumed:] == GREENFACT_TEXT else ""
    if encoded[consumed:] not in {"", GREENFACT_TEXT}:
        raise CacheLayoutError("final cache layout has an unrecognized overlay suffix")
    system_end = next((i for i, row in enumerate(messages) if row.get("role") != "system"), len(messages))
    if (system_end < 1 or system_end + 1 > state_index
            or not str(messages[system_end].get("content", "")).startswith("Task:")
            or any(row.get("role") != "user" for row in messages[system_end:state_index + 1])):
        raise CacheLayoutError("final cache layout must put stable user rows before the frame")
    from kernel_stateframe_framefit5 import _validated_exchange_and_volatile_tail
    _exchange, exchange_start, _tail = _validated_exchange_and_volatile_tail(messages)
    if state_index != exchange_start - 1:
        raise CacheLayoutError("final cache frame must precede the native exchange")
    if (sum(str(row.get("content", "")).count("[CONTINUATION LIMITS |") for row in messages) != 1
            or "[CONTINUATION LIMITS |" not in content
            or any(AUTHORITY_OVERLAY_HEADER in str(row.get("content", "")) for row in messages[:-1])):
        raise CacheLayoutError("final cache authority is not unique or physically final")

    order = overlay.get("order")
    moved_rows: dict[int, dict[str, object]] = {}
    field_values: dict[str, object] = {}
    frame_original_index = None
    original_prefix_count = None
    prefix_length = None
    if order is not None:
        if not isinstance(order, Mapping):
            raise CacheLayoutError("final cache order metadata is malformed")
        prefix_length = order.get("prefix_length")
        original_prefix_count = order.get("original_prefix_count")
        frame_original_index = order.get("frame_index")
        moved = order.get("moved")
        if (type(prefix_length) is not int or not 0 <= prefix_length <= len(wire_prefix)
                or type(original_prefix_count) is not int
                or type(frame_original_index) is not int
                or not isinstance(moved, list)):
            raise CacheLayoutError("final cache order metadata is malformed")
        cursor = prefix_length
        last_start = -1
        for item in moved:
            if not isinstance(item, Mapping):
                raise CacheLayoutError("final cache moved row is malformed")
            original_index, start, end = item.get("index"), item.get("start"), item.get("end")
            if (type(original_index) is not int or type(start) is not int or type(end) is not int
                    or original_index in moved_rows or start < cursor or end <= start
                    or end > len(wire_prefix) or start <= last_start
                    or wire_prefix[cursor:start] != "\n\n"):
                raise CacheLayoutError("final cache moved row location is malformed")
            extras = item.get("extra", {})
            if (not isinstance(extras, Mapping)
                    or any(not isinstance(key, str) or key in {"role", "content"} for key in extras)):
                raise CacheLayoutError("final cache moved row metadata is malformed")
            moved_rows[original_index] = {
                "role": "user", "content": wire_prefix[start:end],
                **copy.deepcopy(dict(extras)),
            }
            cursor, last_start = end, start
        if wire_prefix[cursor:] not in {"", "\n\n"}:
            raise CacheLayoutError("final cache moved row spans do not cover the appended rows")
        if (original_prefix_count != exchange_start + len(moved_rows)
                or not 0 <= frame_original_index < original_prefix_count
                or frame_original_index in moved_rows):
            raise CacheLayoutError("final cache original order metadata is inconsistent")
        if any(index < system_end or index >= original_prefix_count for index in moved_rows):
            raise CacheLayoutError("final cache moved row is outside the user prefix")
        if len(messages[system_end:state_index]) != (
                original_prefix_count - system_end - len(moved_rows) - 1):
            raise CacheLayoutError("final cache stable row count differs from its order metadata")
        for row in overlay["fields"]:
            if not isinstance(row, Mapping):
                raise CacheLayoutError("final cache authority field is malformed")
            path = row.get("path")
            if set(row) == {"path", "message"}:
                reference = row.get("message")
                if type(reference) is not int or reference not in moved_rows:
                    raise CacheLayoutError("final cache authority message reference is invalid")
                field_values[json.dumps(path)] = moved_rows[reference]["content"]
            elif set(row) == {"path", "message", "span"}:
                reference, span = row.get("message"), row.get("span")
                if (type(reference) is not int or reference not in moved_rows
                        or not isinstance(span, list) or len(span) != 2
                        or any(type(value) is not int for value in span)):
                    raise CacheLayoutError("final cache authority substring reference is invalid")
                content = moved_rows[reference]["content"]
                start, end = span
                if not isinstance(content, str) or not 0 <= start < end <= len(content):
                    raise CacheLayoutError("final cache authority substring span is invalid")
                field_values[json.dumps(path)] = content[start:end]
            elif set(row) == {"path", "value"}:
                field_values[json.dumps(path)] = row["value"]
            else:
                raise CacheLayoutError("final cache authority field is malformed")
    else:
        for row in overlay["fields"]:
            if not isinstance(row, Mapping) or set(row) != {"path", "value"}:
                raise CacheLayoutError("legacy final cache authority fields are malformed")
            field_values[json.dumps(row.get("path"))] = row.get("value")

    paths = [row.get("path") for row in overlay["fields"] if isinstance(row, Mapping)]
    if (len(paths) != len(overlay["fields"])
            or any(not isinstance(path, list) or not path for path in paths)
            or len({json.dumps(path) for path in paths}) != len(paths)
            or len(field_values) != len(paths)):
        raise CacheLayoutError("final cache authority paths are malformed or duplicated")
    fields = [{"path": row["path"], "value": field_values[json.dumps(row["path"])]}
              for row in overlay["fields"]]
    restored = _restore_authority(base, fields)
    repartitioned, canonical_fields = _partition_authority(restored, final=True)
    if (repartitioned != base
            or {json.dumps(row["path"]): row["value"] for row in canonical_fields}
            != field_values):
        raise CacheLayoutError("final cache authority overlay is not canonical")

    if order is None:
        physical_order = list(range(len(messages)))
        physical_order.pop(state_index)
        physical_order.insert(system_end + 1, state_index)
        canonical = [copy.deepcopy(dict(messages[index])) for index in physical_order]
        canonical_state_index = physical_order.index(state_index)
        canonical[canonical_state_index]["content"] = "[StateFrame]\n" + json.dumps(
            restored, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        canonical[-1]["content"] = wire_prefix.rstrip() + green
        canonical[-1].pop("name", None)
        return state_index, restored, canonical, {
            physical: index for index, physical in enumerate(physical_order)}

    occupied = set(moved_rows) | {frame_original_index}
    stable_indexes = [index for index in range(system_end, original_prefix_count)
                      if index not in occupied]
    if len(stable_indexes) != state_index - system_end:
        raise CacheLayoutError("final cache stable order cannot be reconstructed")
    prefix_rows: list[dict[str, object] | None] = [None] * original_prefix_count
    physical_to_canonical: dict[int, int] = {}
    for index in range(system_end):
        prefix_rows[index] = copy.deepcopy(dict(messages[index]))
        physical_to_canonical[index] = index
    for physical, original_index in zip(range(system_end, state_index), stable_indexes):
        prefix_rows[original_index] = copy.deepcopy(dict(messages[physical]))
        physical_to_canonical[physical] = original_index
    prefix_rows[frame_original_index] = copy.deepcopy(dict(messages[state_index]))
    physical_to_canonical[state_index] = frame_original_index
    for original_index, row in moved_rows.items():
        prefix_rows[original_index] = copy.deepcopy(row)
    if any(row is None for row in prefix_rows):
        raise CacheLayoutError("final cache original prefix has a missing message")
    canonical = [row for row in prefix_rows if row is not None]
    canonical_frame_index = next(index for index, row in enumerate(canonical)
                                 if str(row.get("content", "")).startswith("[StateFrame]\n"))
    canonical[canonical_frame_index]["content"] = "[StateFrame]\n" + json.dumps(
        restored, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for offset, row in enumerate(messages[exchange_start:-1]):
        canonical.append(copy.deepcopy(dict(row)))
        physical_to_canonical[exchange_start + offset] = original_prefix_count + offset
    canonical_final = copy.deepcopy(dict(final))
    canonical_final["content"] = wire_prefix[:prefix_length] + green
    canonical_final.pop("name", None)
    canonical.append(canonical_final)
    physical_to_canonical[len(messages) - 1] = len(canonical) - 1
    return state_index, restored, canonical, physical_to_canonical


def semantic_index_map(messages):
    """Map submitted locations to the canonical validation view."""
    if not is_cache_layout(messages) or messages[-1].get("name") != FINAL_LAYOUT_NAME:
        return {index: index for index in range(len(messages))}
    _state_index, _payload, _canonical, physical_to_canonical = _final_layout(messages)
    return physical_to_canonical


def inspect_layout(messages):
    """Validate log/index/authority and return current payload and page locations."""
    if messages[-1].get("name") == FINAL_LAYOUT_NAME:
        state_index, payload, _canonical, _index_map = _final_layout(messages)
        return state_index, payload, frozenset(), state_index
    from kernel_greenfact import GREENFACT_TEXT
    from kernel_stateframe_page_log import (
        _page_index_content, _payload_has_authority_keys, _restore_authority, _authority_path,
    )

    layout = validate_stateframe_message_order(
        messages, require_task_label=False, allow_native_exchange=True)
    end = layout.exchange_start if layout.exchange_start is not None else len(messages) - 1
    final = messages[-1]
    if final.get("role") != "user" or not isinstance(final.get("content"), str):
        raise CacheLayoutError("cache layout requires one text final user")
    content = final["content"]
    marker = AUTHORITY_OVERLAY_HEADER + "\n"
    if content.count(marker) != 1:
        raise CacheLayoutError("cache layout requires one authority overlay")
    encoded = content.split(marker, 1)[1]
    overlay, consumed = json.JSONDecoder().raw_decode(encoded)
    if (not isinstance(overlay, dict) or overlay.get("schema") != "stateframe-authority-overlay/1"
            or not isinstance(overlay.get("fields"), list)):
        raise CacheLayoutError("invalid cache authority overlay")
    deltas, pages, locations, versions = [], {}, {}, {}
    base = json.loads(messages[layout.state_index]["content"].split("\n", 1)[1])
    if _payload_has_authority_keys(base):
        raise CacheLayoutError("authority leaked into cache base")
    for index in range(layout.state_index + 1, end):
        row = messages[index]
        text = row.get("content")
        if not isinstance(text, str):
            raise CacheLayoutError("cache log contains non-text data")
        if row.get("name") == "stateframe_frame_delta" and text.startswith(DELTA_HEADER + "\n"):
            delta = json.loads(text.split("\n", 1)[1])
            sections = delta.get("sections") if isinstance(delta, Mapping) else None
            if not isinstance(sections, list) or any(
                    not isinstance(section, Mapping)
                    or not isinstance(section.get("path"), list)
                    or _authority_path(section["path"], section.get("value"))
                    for section in sections):
                raise CacheLayoutError("authority leaked into cache delta")
            deltas.append(row)
        elif text.startswith("[page retired] "):
            key = text.removeprefix("[page retired] ")
            if key not in pages:
                raise CacheLayoutError("cache retirement does not name a live page")
            pages.pop(key)
            locations.pop(key)
        elif text.startswith("[page "):
            header, separator, literal = text.partition("\n")
            match = re.fullmatch(r"\[page (.+) v([1-9][0-9]*)\]", header)
            if not separator or match is None:
                raise CacheLayoutError("malformed cache page")
            key, raw_version = match.groups()
            version = int(raw_version)
            if version <= versions.get(key, 0):
                raise CacheLayoutError("cache page version did not advance")
            versions[key] = version
            pages[key] = (version, literal)
            locations[key] = index
        else:
            raise CacheLayoutError("untyped content in cache log")
    cacheable = materialize_payload(messages[layout.state_index], deltas)
    if _payload_has_authority_keys(cacheable):
        raise CacheLayoutError("authority leaked into cache frame")
    remainder = encoded[consumed:]
    expected = _page_index_content(pages)
    if remainder != ("\n\n" + expected if expected else "") + (
            GREENFACT_TEXT if content.endswith(GREENFACT_TEXT) else ""):
        raise CacheLayoutError("cache page index differs from replayed log")
    paths = [row.get("path") for row in overlay["fields"] if isinstance(row, Mapping)]
    if len(paths) != len(overlay["fields"]) or len({json.dumps(p) for p in paths}) != len(paths):
        raise CacheLayoutError("cache authority paths are malformed or duplicated")
    payload = _restore_authority(cacheable, overlay["fields"])
    from kernel_stateframe_page_log import _partition_authority
    repartitioned, fields = _partition_authority(payload)
    if (repartitioned != cacheable
            or {tuple(row["path"]): row["value"] for row in fields}
            != {tuple(row["path"]): row["value"] for row in overlay["fields"]}):
        raise CacheLayoutError("noncanonical cache authority overlay")
    return layout.state_index, payload, frozenset(locations.values()), end


def semantic_messages(messages):
    """Same message positions, current frame, opaque native exchange, no stale pages."""
    if not is_cache_layout(messages):
        return messages
    if messages[-1].get("name") == FINAL_LAYOUT_NAME:
        _state_index, payload, view, _index_map = _final_layout(messages)
        canonical_index = next(index for index, row in enumerate(view)
                               if str(row.get("content", "")).startswith("[StateFrame]\n"))
        view[canonical_index]["content"] = "[StateFrame]\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return view
    state_index, payload, current_pages, end = inspect_layout(messages)
    view = copy.deepcopy(list(messages))
    view[state_index]["content"] = "[StateFrame]\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for index in range(state_index + 1, end):
        if index not in current_pages:
            view[index] = {"role": "user", "content": ""}
    # The current authority now lives in the typed frame in this validation
    # view. It remains ONLY in [V] in the physical wire.
    from kernel_greenfact import GREENFACT_TEXT
    green = GREENFACT_TEXT if view[-1]["content"].endswith(GREENFACT_TEXT) else ""
    view[-1]["content"] = view[-1]["content"].split(AUTHORITY_OVERLAY_HEADER + "\n", 1)[0].rstrip() + green
    return view


def volatile_blocks(messages):
    """Return literal volatile blocks only after validating the complete layout."""
    if not is_cache_layout(messages):
        return ()
    if messages[-1].get("name") == FINAL_LAYOUT_NAME:
        _final_layout(messages)
        return tuple(messages[-1]["content"].split(AUTHORITY_OVERLAY_HEADER + "\n", 1)[0].rstrip().split("\n\n"))
    inspect_layout(messages)
    prefix = messages[-1]["content"].split(AUTHORITY_OVERLAY_HEADER + "\n", 1)[0]
    return tuple(prefix.rstrip().split("\n\n"))


def source_messages(messages, *, unwrap_pages=False):
    """Mask stale rows; optionally expose validated literals to source parsers.

    Positions never change. Callers proving delivery must bind the original
    physical message, including its page header, to the submitted wire.
    """
    if not is_cache_layout(messages):
        return messages
    if messages[-1].get("name") == FINAL_LAYOUT_NAME:
        _final_layout(messages)
        return messages
    state_index, _payload, current_pages, end = inspect_layout(messages)
    result = []
    for index, row in enumerate(messages):
        if state_index < index < end and index not in current_pages:
            result.append({"role": "user", "content": ""})
        elif unwrap_pages and index in current_pages:
            # inspect_layout already proved this exact typed page and its live
            # index entry. Remove only its framing, never search for a header
            # inside arbitrary source or a native tool reply.
            result.append({**row, "content": row["content"].split("\n", 1)[1]})
        else:
            result.append(row)
    return result
