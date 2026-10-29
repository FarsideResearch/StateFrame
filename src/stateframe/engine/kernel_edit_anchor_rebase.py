"""Revision-bound evidence for rejected edits; this module never writes source."""
from __future__ import annotations

import base64
import difflib
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence


FLAG = "STATEFRAME_EDIT_ANCHOR_REBASE"
SCHEMA = "edit-anchor-rebase/1"
_MAX_CONTEXT_BYTES = 4096
_MAX_PAYLOAD_BYTES = 8192
_MAX_DELTA_BYTES = 4096
_MAX_DELTA_SPANS = 8
_MIN_PRIOR_ANCHOR_MATCH_BYTES = 8


def enabled() -> bool:
    value = os.environ.get(FLAG, "on").strip().lower()
    if value in {"on", "1", "true"}:
        return True
    if value in {"off", "0", "false"}:
        return False
    raise ValueError(f"{FLAG} must be on or off")


def revision(source: bytes) -> str:
    return "sha256:" + hashlib.sha256(source).hexdigest()


def _lines(source: bytes) -> tuple[list[bytes], list[int]]:
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    if not lines and source:
        lines = [source]
        offsets = [0, len(source)]
    return lines, offsets


def _line_range(offsets: Sequence[int], start: int, end: int) -> tuple[int, int]:
    if len(offsets) <= 1:
        return (1, 1)
    start_index = max(0, min(len(offsets) - 2, _bisect_right(offsets, start) - 1))
    end_probe = max(start, end - 1)
    end_index = max(start_index, min(len(offsets) - 2, _bisect_right(offsets, end_probe) - 1))
    return start_index + 1, end_index + 1


def _bisect_right(values: Sequence[int], target: int) -> int:
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if target < values[middle]:
            high = middle
        else:
            low = middle + 1
    return low


def _slice_bytes(source: bytes, start: int, end: int, focus: int,
                 max_bytes: int) -> tuple[int, int]:
    """Clip a context range by byte column while keeping valid UTF-8 boundaries."""
    start = max(0, min(len(source), int(start)))
    end = max(start, min(len(source), int(end)))
    if end - start <= max_bytes:
        return start, end
    if max_bytes <= 0:
        return start, start

    focus = max(start, min(end, int(focus)))
    left = max(start, focus - max_bytes // 2)
    right = min(end, left + max_bytes)
    if right == end and right - left < max_bytes:
        left = max(start, right - max_bytes)

    while left < right and (source[left] & 0xC0) == 0x80:
        left += 1
    while right > left and right < len(source) and (source[right] & 0xC0) == 0x80:
        right -= 1
    return left, right


def _whole_line_range(source: bytes, start: int, end: int) -> bool:
    starts_on_line = start == 0 or source[start - 1:start] == b"\n"
    ends_on_line = end == len(source) or end > 0 and source[end - 1:end] == b"\n"
    return starts_on_line and ends_on_line


def _span(path: str, source: bytes, start: int, end: int, *, kind: str,
          max_bytes: int = _MAX_CONTEXT_BYTES) -> dict[str, object]:
    start = max(0, min(len(source), int(start)))
    end = max(start, min(len(source), int(end)))
    context_start, context_end = _context_around(
        source, start, end, max_bytes=max_bytes,
    )
    start, end = _slice_bytes(source, context_start, context_end, start, max_bytes)
    lines, offsets = _lines(source)
    first_line, last_line = _line_range(offsets, start, end)
    exact = source[start:end]
    record: dict[str, object] = {
        "path": path,
        "range": f"{first_line}-{last_line}",
        "path_range_rev": f"{path}:{first_line}-{last_line}@{revision(source)}",
        "line_range": [first_line, last_line],
        "byte_range": [start, end],
        "revision": revision(source),
        "kind": kind,
        "bytes_sha256": revision(exact),
        "truncated": (start != context_start or end != context_end
                      or not _whole_line_range(source, context_start, context_end)),
    }
    try:
        record["old_str"] = exact.decode("utf-8")
    except UnicodeDecodeError:
        record["bytes_base64"] = base64.b64encode(exact).decode("ascii")
    return record


def _occurrences(source: bytes, needle: bytes) -> list[int]:
    if not needle:
        return []
    found: list[int] = []
    cursor = 0
    while True:
        offset = source.find(needle, cursor)
        if offset < 0:
            return found
        found.append(offset)
        cursor = offset + 1


def _context_around(source: bytes, start: int, end: int, *, max_bytes: int = _MAX_CONTEXT_BYTES) -> tuple[int, int]:
    """Return unique whole-line context; oversized lines go to the byte slicer."""
    lines, offsets = _lines(source)
    if not lines:
        return 0, 0
    left = max(0, min(len(source), start))
    right = max(left, min(len(source), end))
    left_index = min(len(lines) - 1, max(0, _bisect_right(offsets, left) - 1))
    right_index = min(
        len(lines) - 1,
        max(left_index, _bisect_right(offsets, max(left, right - 1)) - 1),
    )
    left, right = offsets[left_index], offsets[min(right_index + 1, len(offsets) - 1)]
    if right == left and left < len(source):
        right = offsets[min(left_index + 1, len(offsets) - 1)]
    if right - left > max_bytes:
        return left, right
    while right - left <= max_bytes:
        snippet = source[left:right]
        if snippet and source.count(snippet) == 1:
            return left, right
        expanded = False
        if left_index > 0 and right - offsets[left_index - 1] <= max_bytes:
            left_index -= 1
            left = offsets[left_index]
            expanded = True
        if right_index + 1 < len(lines) and offsets[right_index + 2] - left <= max_bytes:
            right_index += 1
            right = offsets[right_index + 1]
            expanded = True
        if not expanded:
            break
    return max(0, min(len(source), start)), max(0, min(len(source), end))


def _delta_spans(before: bytes, after: bytes) -> list[dict[str, object]]:
    before_lines, before_offsets = _lines(before)
    after_lines, after_offsets = _lines(after)
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
    result: list[dict[str, object]] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_bytes = before[before_offsets[old_start]:before_offsets[old_end]]
        new_bytes = after[after_offsets[new_start]:after_offsets[new_end]]
        result.append({
            "before_byte_range": [before_offsets[old_start], before_offsets[old_end]],
            "after_byte_range": [after_offsets[new_start], after_offsets[new_end]],
            "before_base64": base64.b64encode(old_bytes).decode("ascii"),
            "after_base64": base64.b64encode(new_bytes).decode("ascii"),
            "before_sha256": revision(old_bytes),
            "after_sha256": revision(new_bytes),
            "before_bytes_utf8": old_bytes.decode("utf-8", errors="replace"),
            "after_bytes_utf8": new_bytes.decode("utf-8", errors="replace"),
        })
    return result


def applied_delta(path: str, before: bytes, after: bytes) -> dict[str, object]:
    """Build the exact before/after delta stored on the existing action receipt."""
    full_spans = _delta_spans(before, after)
    spans: list[dict[str, object]] = []
    retained_bytes = 0
    for item in full_spans[:_MAX_DELTA_SPANS]:
        row = dict(item)
        old_bytes = base64.b64decode(str(row.pop("before_base64")), validate=True)
        new_bytes = base64.b64decode(str(row.pop("after_base64")), validate=True)
        needed = len(old_bytes) + len(new_bytes)
        if retained_bytes + needed <= _MAX_DELTA_BYTES:
            row["before_base64"] = base64.b64encode(old_bytes).decode("ascii")
            row["after_base64"] = base64.b64encode(new_bytes).decode("ascii")
            retained_bytes += needed
        else:
            row["content_omitted"] = True
            row["before_bytes_utf8"] = None
            row["after_bytes_utf8"] = None
        spans.append(row)
    return {
        "schema": SCHEMA,
        "path": path,
        "before_revision": revision(before),
        "after_revision": revision(after),
        "changed_spans": spans,
        "changed_span_count": len(full_spans),
        "changed_spans_omitted": max(0, len(full_spans) - len(spans)),
        "change_content_omitted": any(bool(item.get("content_omitted")) for item in spans)
        or len(full_spans) > len(spans),
    }


def _latest_prior_delta(receipts: object, path: str) -> Mapping[str, object] | None:
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes, bytearray)):
        return None
    for row in reversed(receipts):
        if not isinstance(row, Mapping) or row.get("kind") != "edit" \
                or row.get("result_class") != "APPLIED":
            continue
        delta = row.get("edit_anchor_delta")
        if isinstance(delta, Mapping) and str(delta.get("path", "")) == path:
            return delta
        deltas = row.get("edit_anchor_deltas", ())
        if isinstance(deltas, Sequence) and not isinstance(deltas, (str, bytes, bytearray)):
            for item in deltas:
                if isinstance(item, Mapping) and str(item.get("path", "")) == path:
                    return item
    return None


def _prior_locations(source: bytes, delta: Mapping[str, object] | None) -> list[tuple[int, int]]:
    if not isinstance(delta, Mapping):
        return []
    spans = delta.get("changed_spans", ())
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes, bytearray)):
        return []
    current_revision = revision(source)
    locations: list[tuple[int, int]] = []
    for item in spans:
        if not isinstance(item, Mapping):
            continue
        raw_range = item.get("after_byte_range")
        if (delta.get("after_revision") == current_revision
                and isinstance(raw_range, Sequence) and len(raw_range) == 2
                and all(isinstance(value, int) for value in raw_range)):
            start, end = int(raw_range[0]), int(raw_range[1])
            if 0 <= start <= end <= len(source):
                locations.append((start, end))
                continue
        encoded = item.get("after_base64")
        if isinstance(encoded, str):
            try:
                new_bytes = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError):
                continue
            found = _occurrences(source, new_bytes)
            if found:
                locations.extend((offset, offset + len(new_bytes)) for offset in found[:3])
    return locations


def _closest_locations(
        source: bytes, stale: bytes, *, context_radius: int = 1,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], bool, bool]:
    source_lines, offsets = _lines(source)
    stale_lines, _ = _lines(stale)
    if not source_lines:
        return [(0, 0)], [], True, False
    source_nonblank = [(index, line) for index, line in enumerate(source_lines) if line.strip()]
    stale_nonblank = [(index, line) for index, line in enumerate(stale_lines) if line.strip()]
    matcher = difflib.SequenceMatcher(
        a=[line for _index, line in stale_nonblank],
        b=[line for _index, line in source_nonblank],
    )
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if blocks:
        best = max(block.size for block in blocks)
        locations: list[tuple[int, int]] = []
        matched_locations: list[tuple[int, int]] = []
        for block in blocks:
            if block.size != best:
                continue
            first_matching_line = source_nonblank[block.b][0]
            last_matching_line = source_nonblank[block.b + block.size - 1][0]
            matched_locations.append((
                offsets[first_matching_line], offsets[last_matching_line + 1],
            ))
            start_line = max(0, first_matching_line - context_radius)
            end_line = min(len(source_lines), last_matching_line + context_radius + 1)
            locations.append((offsets[start_line], offsets[end_line]))
            if len(locations) == 3:
                break
        return locations, matched_locations, False, False

    # Generic line similarity is useful when a stale anchor differs only by
    # whitespace or one identifier. Keep it language-agnostic: normalize text,
    # compare line sequences, and use token overlap only to break weak matches.
    stale_text = [line.decode("utf-8", errors="replace") for line in stale_lines]
    stale_text = [line for line in stale_text if line.strip()]
    current_text = [line.decode("utf-8", errors="replace") for line in source_lines]
    stale_tokens = [
        set(re.findall(r"\w+", " ".join(line.split()).casefold()))
        for line in stale_text
    ]
    ranked: list[tuple[float, int]] = []
    for source_index, current_line in enumerate(current_text):
        normalized_current = " ".join(current_line.split()).casefold()
        if not normalized_current:
            continue
        current_tokens = set(re.findall(r"\w+", normalized_current))
        best_score = 0.0
        for old_line, old_tokens in zip(stale_text, stale_tokens):
            normalized_old = " ".join(old_line.split()).casefold()
            if not normalized_old:
                continue
            ratio = difflib.SequenceMatcher(
                a=normalized_old, b=normalized_current, autojunk=False,
            ).ratio()
            union = old_tokens | current_tokens
            overlap = (len(old_tokens & current_tokens) / len(union)) if union else 0.0
            best_score = max(best_score, ratio, overlap * 0.85)
        ranked.append((best_score, source_index))
    if ranked:
        best_score = max(score for score, _index in ranked)
        best_candidates = [
            index for score, index in ranked
            if score == best_score and best_score >= 0.62
        ][:3]
        if best_candidates:
            locations: list[tuple[int, int]] = []
            matched_locations: list[tuple[int, int]] = []
            for line_index in best_candidates:
                matched_locations.append((offsets[line_index], offsets[line_index + 1]))
                start_line = max(0, line_index - context_radius)
                end_line = min(len(source_lines), line_index + context_radius + 1)
                locations.append((offsets[start_line], offsets[end_line]))
            return locations, matched_locations, False, True
    # The module head is only a bounded fallback; it is not a closest match.
    end_line = min(len(source_lines), max(1, min(8, len(stale_lines) or 4)))
    return [(0, offsets[end_line])], [], True, False


def _anchor(path: str, source: bytes, start: int, end: int, *, kind: str,
            max_bytes: int = _MAX_CONTEXT_BYTES,
            target_range: tuple[int, int] | None = None) -> dict[str, object]:
    result = _span(path, source, start, end, kind=kind, max_bytes=max_bytes)
    exact_start, exact_end = result["byte_range"]
    exact = source[int(exact_start):int(exact_end)]
    result["unique_exact_match"] = bool(exact) and source.count(exact) == 1
    target_start, target_end = target_range if target_range is not None else (start, end)
    result["target_byte_range"] = [target_start, target_end]
    return result


def _span_reference(anchor: Mapping[str, object], index: int,
                    *, kind: str) -> dict[str, object]:
    source_fields = {"old_str", "bytes_base64"}
    result = {key: value for key, value in anchor.items() if key not in source_fields}
    result["kind"] = kind
    result["source_ref"] = f"anchor_suggestion.anchors[{index}]"
    return result


def _location_reference(path: str, source: bytes, start: int, end: int,
                        index: int, *, kind: str) -> dict[str, object]:
    lines, offsets = _lines(source)
    first_line, last_line = _line_range(offsets, start, end)
    return {
        "path": path,
        "range": f"{first_line}-{last_line}",
        "path_range_rev": f"{path}:{first_line}-{last_line}@{revision(source)}",
        "line_range": [first_line, last_line],
        "byte_range": [start, end],
        "revision": revision(source),
        "kind": kind,
        "source_ref": f"anchor_suggestion.anchors[{index}]",
    }


def _payload_size(payload: Mapping[str, object]) -> int:
    return len(encode(payload).encode("utf-8"))


def _fit_payload(build: Callable[[int, int], dict[str, object]],
                 *, item_count: int) -> dict[str, object]:
    """Choose the largest source slice whose complete encoded receipt fits."""
    make = build
    for count in range(item_count, 0, -1):
        low, high = 0, _MAX_CONTEXT_BYTES
        best = make(0, count)
        if _payload_size(best) > _MAX_PAYLOAD_BYTES:
            continue
        while low <= high:
            middle = (low + high) // 2
            candidate = make(middle, count)
            if _payload_size(candidate) <= _MAX_PAYLOAD_BYTES:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    # The structured receipt remains useful without excerpts if caller-supplied
    # metadata is unusually large; retain one location when its metadata fits.
    minimal = make(0, 1)
    if _payload_size(minimal) <= _MAX_PAYLOAD_BYTES:
        return minimal
    anchors = minimal.get("anchor_suggestion")
    if isinstance(anchors, dict):
        anchors["anchors"] = []
    minimal["closest_current_spans"] = []
    minimal["hint"] = ""
    minimal["alternatives"] = []
    changed = minimal.get("changed_since_model_last_saw")
    if isinstance(changed, dict):
        changed["changed_spans"] = []
        changed.pop("path", None)
    path = minimal.get("path")
    if isinstance(path, str) and len(path) > 512:
        minimal["path"] = path[:512]
        minimal["path_truncated"] = True
    for field in ("expected_scope", "candidate_scope"):
        value = minimal.get(field)
        if isinstance(value, str) and len(value) > 128:
            minimal[field] = value[:128]
    if _payload_size(minimal) <= _MAX_PAYLOAD_BYTES:
        return minimal
    fallback: dict[str, object] = {
        "schema": SCHEMA,
        "status": "REJECTED",
        "code": str(minimal.get("code", "EDIT_REJECTED"))[:64],
        "path": str(minimal.get("path", ""))[:256],
        "current_revision": str(minimal.get("current_revision", ""))[:80],
        "source_context_omitted": True,
        "reason": "Receipt metadata exceeded the encoded payload budget.",
    }
    return fallback


def _delta_explains_anchor(stale: bytes,
                           delta: Mapping[str, object] | None,
                           prior_locations: Sequence[tuple[int, int]],
                           target_locations: Sequence[tuple[int, int]],
                           *, target_is_fallback: bool) -> bool:
    """Accept a prior edit only when its bytes or current range identify this target."""
    if not isinstance(delta, Mapping) or not prior_locations:
        return False
    spans = delta.get("changed_spans", ())
    if isinstance(spans, Sequence) and not isinstance(spans, (str, bytes, bytearray)):
        for item in spans:
            if not isinstance(item, Mapping):
                continue
            for field in ("before_base64", "after_base64"):
                encoded = item.get(field)
                if not isinstance(encoded, str):
                    continue
                try:
                    changed = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError):
                    continue
                if stale and changed and stale in changed:
                    # A generic byte shared by a stale anchor and an unrelated hunk
                    # cannot establish that the hunk explains the rejected target.
                    distinctive = b"".join(stale.split())
                    if len(distinctive) >= _MIN_PRIOR_ANCHOR_MATCH_BYTES:
                        return True

    if target_is_fallback:
        return False
    for prior_start, prior_end in prior_locations:
        for target_start, target_end in target_locations:
            overlaps = (prior_start < target_end and target_start < prior_end)
            if prior_start == prior_end:
                overlaps = target_start <= prior_start <= target_end
            if overlaps:
                return True
    return False


def _public_delta_spans(delta: Mapping[str, object] | None) -> list[dict[str, object]]:
    if not isinstance(delta, Mapping):
        return []
    spans = delta.get("changed_spans", ())
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes, bytearray)):
        return []
    result: list[dict[str, object]] = []
    for item in spans[:_MAX_DELTA_SPANS]:
        if not isinstance(item, Mapping):
            continue
        row: dict[str, object] = {}
        for key in ("before_byte_range", "after_byte_range",
                    "before_sha256", "after_sha256"):
            value = item.get(key)
            if key.endswith("_byte_range"):
                if (isinstance(value, Sequence) and len(value) == 2
                        and all(isinstance(part, int) for part in value)):
                    row[key] = [int(value[0]), int(value[1])]
            elif isinstance(value, str):
                row[key] = value
        if row:
            result.append(row)
    return result


def rejection_payload(*, path: str, source: bytes, old_text: str,
                      code: str, receipts: object = (),
                      hint: str = "", context_radius: int = 1) -> dict[str, object]:
    """Create exact current-revision context for a no-match or duplicate rejection."""
    if type(context_radius) is not int or not 0 <= context_radius <= 12:
        raise ValueError("context_radius must be an integer from 0 through 12")
    stale = old_text.encode("utf-8")
    matches = _occurrences(source, stale)
    prior = _latest_prior_delta(receipts, path)
    if matches:
        locations = [(offset, offset + len(stale)) for offset in matches[:8]]
        span_locations = locations
        selection = "exact_duplicate_occurrences"
        fallback = False
    else:
        target_locations, matched_locations, fallback, approximate = _closest_locations(
            source, stale, context_radius=context_radius)
        prior_locations = _prior_locations(source, prior)
        if _delta_explains_anchor(
                stale, prior, prior_locations, matched_locations,
                target_is_fallback=fallback):
            locations = prior_locations
            span_locations = prior_locations
            selection = "prior_applied_edit"
        else:
            locations = target_locations
            span_locations = matched_locations or target_locations
            selection = (
                "module_head_fallback" if fallback else
                "closest_line_similarity_context" if approximate else
                "closest_exact_current_context"
            )

    changed: dict[str, object]
    if prior is None:
        changed = {
            "status": "NO_PRIOR_APPLIED_EDIT",
            "detail": "No earlier applied edit receipt for this path is available.",
        }
    else:
        changed = {
            "status": "PRIOR_APPLIED_EDIT",
            "path": path,
            "model_seen_revision": prior.get("before_revision"),
            "before_revision": prior.get("before_revision"),
            "after_revision": prior.get("after_revision"),
            "current_revision": revision(source),
            "delta_is_current": prior.get("after_revision") == revision(source),
            "changed_spans": _public_delta_spans(prior),
        }

    def build(max_bytes: int, item_count: int) -> dict[str, object]:
        selected_locations = locations[:item_count]
        selected_span_locations = span_locations[:item_count]
        anchors = [
            _anchor(path, source, start, end, kind="unique_exact_anchor",
                    max_bytes=max_bytes,
                    target_range=(selected_span_locations[index]
                                  if index < len(selected_span_locations) else (start, end)))
            for index, (start, end) in enumerate(selected_locations)
        ]
        spans = [
            _location_reference(path, source, start, end, index, kind=selection)
            for index, (start, end) in enumerate(selected_span_locations)
        ]
        suggestion: dict[str, object] = {
            "tool": "str_replace",
            "match_count": len(matches),
            "selection_basis": selection,
            "anchors": anchors,
            "instruction": (
                "Use the listed current bytes as an exact anchor at the stated revision. "
                "Keep the replacement limited to the intended target and preserve any "
                "surrounding anchor bytes unchanged. Do not apply a fuzzy match."
            ),
        }
        if selection == "module_head_fallback":
            suggestion["fallback_reason"] = (
                "No stale line matched; this is bounded module-head context, not a closest match."
            )
        elif selection == "closest_line_similarity_context":
            suggestion["similarity_basis"] = (
                "Generic line-level whitespace normalization, difflib similarity, and token overlap; "
                "no language-specific parsing or fuzzy edit was applied."
            )
        return {
            "schema": SCHEMA,
            "status": "REJECTED",
            "code": code,
            "path": path,
            "current_revision": revision(source),
            "closest_current_spans": spans,
            "changed_since_model_last_saw": changed,
            "anchor_suggestion": suggestion,
            "hint": str(hint)[:600],
            "alternatives": [{
                "tool": "str_replace",
                "path": path,
                "hint": "Retry with an exact current anchor; the host did not change source bytes.",
            }],
        }

    return _fit_payload(build, item_count=min(8, len(locations)))


def scope_mismatch_payload(*, path: str, source: bytes, expected_scope: str,
                           candidate_scope: str, start: int = 0, end: int | None = None) -> dict[str, object]:
    """Return a typed refusal for a whole-file/region contract mismatch."""
    end = len(source) if end is None else end

    def build(max_bytes: int, _item_count: int) -> dict[str, object]:
        anchor = _anchor(path, source, start, end, kind="scope_mismatch_current_source",
                         max_bytes=max_bytes)
        return {
            "schema": SCHEMA,
            "status": "REJECTED",
            "code": "EDIT_SCOPE_MISMATCH",
            "path": path,
            "current_revision": revision(source),
            "expected_scope": expected_scope,
            "candidate_scope": candidate_scope,
            "closest_current_spans": [_span_reference(
                anchor, 0, kind="scope_mismatch_current_source")],
            "changed_since_model_last_saw": {
                "status": "NO_PRIOR_APPLIED_EDIT",
                "detail": "Scope mismatch is based on the current source revision.",
            },
            "anchor_suggestion": {
                "tool": "str_replace",
                "anchors": [anchor],
                "instruction": "Use an exact current anchor at the stated revision; keep the edit within the requested region.",
            },
            "alternatives": [{
                "tool": "str_replace",
                "path": path,
                "hint": "Use an exact region replacement, or submit a complete file only when whole-file scope is declared.",
            }],
        }

    return _fit_payload(build, item_count=1)


def encode(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


__all__ = [
    "FLAG", "SCHEMA", "applied_delta", "enabled", "encode", "rejection_payload",
    "revision", "scope_mismatch_payload",
]
