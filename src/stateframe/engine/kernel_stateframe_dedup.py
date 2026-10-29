"""Deterministic, presentation-only StateFrame deduplication.

The host-owned frame remains the input of record.  This module only projects
that frame for a model request when ``STATEFRAME_DEDUP=on``; it never mutates
the host frame, receipts, journal state, or the completion authorities that
consume them.  The resulting request-prepared presentation is, of course,
recorded as the request's delivered messages.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence

from kernel_measurement import canonical_bytes


MODE_ENV = "STATEFRAME_DEDUP"

# These are the fields needed by the model for source identity, trust,
# currentness, range/pagination, and explicit recall.  The native-prefix
# authority additionally reads path, revision, freshness, start_line,
# start_column, omitted_chars_in_selection, and more_content_available.
# ``span_id`` is deliberately not a source-header field: it is a working-set
# identity already present in the typed StateFrame projection.
RETAINED_SOURCE_METADATA_FIELDS = (
    "authority",
    "path",
    "reference",
    "revision",
    "freshness",
    "current_reference",
    "start_line",
    "start_column",
    "selected_end_line",
    "shown_end_line",
    "total_lines",
    "omitted_chars_in_selection",
    "more_content_available",
    "more_lines_available",
    "continuation",
    "requested_delivery",
    "range_status",
    "decoded_with_replacement",
)

_RECORD_FIELDS = (
    "recent_negative_actions",
    "recent_test_feedback",
    "recent_changes",
)


def dedup_mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return dedup_mode() == "on"


def retained_source_metadata(evidence: Mapping[str, object]) -> dict[str, object]:
    """Return only the model/kernel-used retained-source header fields."""
    return {key: copy.deepcopy(evidence[key])
            for key in RETAINED_SOURCE_METADATA_FIELDS if key in evidence}


def _record_key(record: Mapping[str, object]) -> bytes | None:
    try:
        return canonical_bytes(record)
    except (TypeError, ValueError):
        # A non-JSON record is not a typed record we can safely canonicalize;
        # keep it rather than invent an equivalence relation.
        return None


def dedupe_identical_records(value: object) -> object:
    """Keep the newest exact record for each value and attach its count.

    The order is the order of newest occurrences.  No field is summarized or
    selected, so a latest test's selector, binding, and exit status stay in the
    retained record verbatim.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return copy.deepcopy(value)
    newest: list[tuple[bytes | None, object, int]] = []
    positions: dict[bytes, int] = {}
    for raw in reversed(value):
        if not isinstance(raw, Mapping):
            newest.append((None, copy.deepcopy(raw), 1))
            continue
        record = copy.deepcopy(dict(raw))
        key = _record_key(record)
        if key is None:
            newest.append((None, record, 1))
            continue
        position = positions.get(key)
        if position is None:
            positions[key] = len(newest)
            newest.append((key, record, 1))
        else:
            existing = newest[position]
            newest[position] = (existing[0], existing[1], existing[2] + 1)
    result: list[object] = []
    for _key, record, count in newest:
        if isinstance(record, dict):
            record["count"] = count
        result.append(record)
    return list(reversed(result))


def _complete_current_retained_sources(
    retained_evidence: Sequence[Mapping[str, object]],
) -> dict[tuple[object, object], dict[str, object]]:
    sources: dict[tuple[object, object], dict[str, object]] = {}
    for row in retained_evidence:
        descriptor = _complete_current_retained_source(row)
        if descriptor is None:
            continue
        sources[(descriptor["path"], descriptor["revision"])] = descriptor
    return sources


def _complete_current_retained_source(row: object) -> dict[str, object] | None:
    if not isinstance(row, Mapping) or not isinstance(row.get("content"), str):
        return None
    if row.get("freshness") != "current":
        return None
    if row.get("start_line") != 1 or row.get("start_column", 0) != 0:
        return None
    if row.get("omitted_chars_in_selection") != 0:
        return None
    if row.get("more_content_available") is not False:
        return None
    path, revision = row.get("path"), row.get("revision")
    reference = row.get("reference")
    if path is None or revision is None or reference is None:
        return None
    return {
        "path": path,
        "revision": revision,
        "reference": reference,
        "span_id": row.get("span_id"),
    }


def complete_current_retained_source(row: object) -> bool:
    """Whether a retained message is complete enough to back a frame pointer."""
    return _complete_current_retained_source(row) is not None


def _compact_earlier_reads(
    value: object,
    sources: Mapping[tuple[object, object], Mapping[str, object]],
) -> object:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return copy.deepcopy(value)
    result: list[object] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            result.append(copy.deepcopy(raw))
            continue
        row = copy.deepcopy(dict(raw))
        source = sources.get((row.get("path"), row.get("file_revision")))
        if source is not None:
            # The exact current source is already delivered in the retained
            # source message.  The read record is therefore only a typed
            # pointer.  This also handles rows whose content was already
            # removed by cover_observed_reads before this projector runs.
            result.append({
                "path": row.get("path"),
                "file_revision": row.get("file_revision"),
                "retained_source_ref": source["reference"],
            })
            continue
        result.append(row)
    return result


def _compact_retained_context(value: object) -> object:
    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    result = copy.deepcopy(dict(value))
    catalog = result.get("catalog")
    if isinstance(catalog, Sequence) and not isinstance(catalog, (str, bytes, bytearray)):
        compact_catalog = []
        for raw in catalog:
            if not isinstance(raw, Mapping):
                compact_catalog.append(copy.deepcopy(raw))
                continue
            # The catalog is a typed source index.  Bodies are not allowed in
            # it; source bytes live only in the selected retained message.
            compact_catalog.append({key: copy.deepcopy(raw[key]) for key in (
                "path", "revision", "reference", "freshness", "in_working_source"
            ) if key in raw})
        result["catalog"] = compact_catalog
    return result


def project_frame(
    frame: Mapping[str, object],
    *,
    retained_evidence: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Build the compact typed model-facing frame projection."""
    if not isinstance(frame, Mapping):
        raise TypeError("StateFrame dedup requires a frame mapping")
    result = copy.deepcopy(dict(frame))
    for key in _RECORD_FIELDS:
        if key in result:
            result[key] = dedupe_identical_records(result[key])
    if "retained_context" in result:
        result["retained_context"] = _compact_retained_context(result["retained_context"])
    sources = _complete_current_retained_sources(retained_evidence)
    if "earlier_other_file_reads" in result:
        result["earlier_other_file_reads"] = _compact_earlier_reads(
            result["earlier_other_file_reads"], sources)
    return result


__all__ = [
    "MODE_ENV",
    "RETAINED_SOURCE_METADATA_FIELDS",
    "dedup_mode",
    "dedupe_identical_records",
    "complete_current_retained_source",
    "enabled",
    "project_frame",
    "retained_source_metadata",
]
