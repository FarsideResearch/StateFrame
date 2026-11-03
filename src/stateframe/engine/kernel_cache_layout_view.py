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
    """Return the physical frame index, restored payload, and canonical index map."""
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
    encoded = content.split(marker, 1)[1]
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
    paths = [row.get("path") for row in overlay["fields"] if isinstance(row, Mapping)]
    if (len(paths) != len(overlay["fields"])
            or any(not isinstance(path, list) or not path for path in paths)
            or len({json.dumps(path) for path in paths}) != len(paths)):
        raise CacheLayoutError("final cache authority paths are malformed or duplicated")
    restored = _restore_authority(base, overlay["fields"])
    repartitioned, fields = _partition_authority(restored, final=True)
    if (repartitioned != base
            or {json.dumps(row["path"]): row["value"] for row in fields}
            != {json.dumps(row["path"]): row["value"] for row in overlay["fields"]}):
        raise CacheLayoutError("final cache authority overlay is not canonical")
    canonical_order = list(range(len(messages)))
    canonical_order.pop(state_index)
    canonical_order.insert(system_end + 1, state_index)
    return state_index, restored, canonical_order


def semantic_index_map(messages):
    """Map submitted locations to the canonical validation view."""
    if not is_cache_layout(messages) or messages[-1].get("name") != FINAL_LAYOUT_NAME:
        return {index: index for index in range(len(messages))}
    _state_index, _payload, order = _final_layout(messages)
    return {physical: canonical for canonical, physical in enumerate(order)}


def inspect_layout(messages):
    """Validate log/index/authority and return current payload and page locations."""
    if messages[-1].get("name") == FINAL_LAYOUT_NAME:
        state_index, payload, _order = _final_layout(messages)
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
        state_index, payload, order = _final_layout(messages)
        view = [copy.deepcopy(dict(messages[index])) for index in order]
        canonical_index = order.index(state_index)
        view[canonical_index]["content"] = "[StateFrame]\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        from kernel_greenfact import GREENFACT_TEXT
        content = view[-1]["content"]
        green = GREENFACT_TEXT if content.endswith(GREENFACT_TEXT) else ""
        view[-1]["content"] = content.split(AUTHORITY_OVERLAY_HEADER + "\n", 1)[0].rstrip() + green
        view[-1].pop("name", None)
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
