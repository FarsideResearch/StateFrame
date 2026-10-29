"""Opt-in bounded paging for retained source evidence.

The archive remains the lossless host store.  This module only adds a small,
typed index to the model-facing projection and caps explicit cold reads.  It
does not participate in the default retained-context path.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

from kernel_measurement import digest


SWITCH_ENV = "KERNEL_LOSSLESS_STORE_PAGING"
SCHEMA = "retained-source-paging/1"
STABLE_REFERENCE_PREFIX = "source-v1:"
PAGING_SCHEMA_FIELD = "paging_schema"
REVISION_REF_FIELD = "revision_ref"
APPLICABILITY_FIELD = "applicability"
OMITTED_REFERENCE_INDEX = "omitted_reference_index"
OMITTED_REFERENCE_TOTAL = "omitted_reference_index_total"
OMITTED_REFERENCE_NEXT = "omitted_reference_index_next"
MANIFEST_ENABLED_FIELD = "retained_lossless_store_paging"
MANIFEST_SCHEMA_FIELD = "retained_source_paging_schema"
INDEX_LIMIT = 64
MAX_PAGE_LINES = 120
MAX_PAGE_CHARS = 6000


def enabled() -> bool:
    value = os.environ.get(SWITCH_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{SWITCH_ENV} must be 0 or 1")
    return value == "1"


def stable_revision_ref(archive, reference: str) -> str:
    value = archive.revisions[reference]
    return STABLE_REFERENCE_PREFIX + digest({
        "path": value["path"], "revision": value["revision"],
    })


def resolve_stable_reference(archive, reference: str) -> str:
    if not isinstance(reference, str) or not reference.startswith(STABLE_REFERENCE_PREFIX):
        raise KeyError(reference)
    matches = [candidate for candidate in archive.revisions
               if stable_revision_ref(archive, candidate) == reference]
    if len(matches) != 1:
        raise KeyError(reference)
    return matches[0]


def _index_row(archive, reference: str, current_hashes: Mapping[str, str]) -> dict:
    value = archive.revisions[reference]
    current = current_hashes.get(value["path"]) == value["revision"]
    applicability = "current" if current else "historical"
    return {
        "reference": reference,
        # Keep an explicit revision reference so consumers cannot confuse a
        # path handle with the immutable archive identity.
        REVISION_REF_FIELD: stable_revision_ref(archive, reference),
        "path": value["path"],
        "revision": value["revision"],
        "freshness": applicability,
        APPLICABILITY_FIELD: applicability,
        "authority": "HOST_OBSERVED_SOURCE",
        "evidence_ref": value["evidence_ref"],
    }


def attach_index(view: Mapping[str, object], archive, current_hashes: Mapping[str, str]) -> dict:
    """Attach a bounded index for source revisions absent from the visible catalog.

    Only metadata is rendered.  The bytes stay in ``archive.blobs`` and are
    fetched by ``cold_page`` when a model chooses one of these references.
    """
    result = dict(view)
    catalog = result.get("catalog", ())
    visible = {
        str(row["reference"])
        for row in [*catalog, *result.get("selected", ())]
        if isinstance(row, Mapping) and row.get("reference") in archive.revisions
    }
    omitted = [reference for reference in archive.revisions if reference not in visible]
    result[OMITTED_REFERENCE_INDEX] = [
        _index_row(archive, reference, current_hashes)
        for reference in omitted[:INDEX_LIMIT]
    ]
    result[OMITTED_REFERENCE_TOTAL] = len(omitted)
    result[OMITTED_REFERENCE_NEXT] = INDEX_LIMIT if len(omitted) > INDEX_LIMIT else None
    result[PAGING_SCHEMA_FIELD] = SCHEMA
    return result


def cold_page(archive, reference: str, current_hashes: Mapping[str, str], *,
              start_line: int = 1, start_column: int = 0, symbol: str = "") -> dict:
    """Fetch one bounded page from retained bytes, preserving archive labels."""
    page = archive.recall(
        reference,
        current_hashes,
        start_line=start_line,
        start_column=start_column,
        symbol=symbol,
        max_lines=MAX_PAGE_LINES,
        max_chars=MAX_PAGE_CHARS,
        continuation=True,
    )
    return {
        **page,
        REVISION_REF_FIELD: stable_revision_ref(archive, reference),
        APPLICABILITY_FIELD: page["freshness"],
        PAGING_SCHEMA_FIELD: SCHEMA,
    }


__all__ = [
    "INDEX_LIMIT",
    "APPLICABILITY_FIELD",
    "MAX_PAGE_CHARS",
    "MAX_PAGE_LINES",
    "MANIFEST_ENABLED_FIELD",
    "MANIFEST_SCHEMA_FIELD",
    "OMITTED_REFERENCE_INDEX",
    "OMITTED_REFERENCE_NEXT",
    "OMITTED_REFERENCE_TOTAL",
    "PAGING_SCHEMA_FIELD",
    "REVISION_REF_FIELD",
    "SCHEMA",
    "STABLE_REFERENCE_PREFIX",
    "SWITCH_ENV",
    "attach_index",
    "cold_page",
    "enabled",
    "resolve_stable_reference",
    "stable_revision_ref",
]
