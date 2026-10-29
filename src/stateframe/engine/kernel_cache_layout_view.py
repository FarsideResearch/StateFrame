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


class CacheLayoutError(ValueError):
    pass


def is_cache_layout(messages: object) -> bool:
    return (isinstance(messages, Sequence) and not isinstance(messages, (str, bytes))
            and bool(messages) and isinstance(messages[-1], Mapping)
            and messages[-1].get("name") == PAGE_LOG_FINAL_MESSAGE_NAME)


def inspect_layout(messages):
    """Validate log/index/authority and return current payload and page locations."""
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
