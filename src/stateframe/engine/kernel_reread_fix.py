"""STATEFRAME_REREAD_FIX: make a source block the model can see count as delivered.

Root cause (r35e ldev_02, 57 reads / 1 edit): with STATEFRAME_SOURCE_LABELS=plain every
source block is rendered as ``[Current source: <path>]``.  The native-prefix edit authority
(``kernel_native_prefix.delivered_source``) only recognised ``[Working source: ...]`` and
``[Retained source: ...]`` headers, so a correct multi-file ``str_replace`` batch applied its
first edit and then refused every following edit with "complete target source was not
delivered" -- although the exact bytes were in the submitted request.  The model then tried to
"deliver" the source by reading it, the read receipt answered ALREADY_PRESENT, and with no
transcript history each fresh frame re-planned "read first, then edit" forever.

Two host mechanisms, one switch:

1. ``current_source_candidates``: the native-prefix delivery check also accepts the plain
   ``[Current source: <path>]`` label (bare body, ``Coverage:`` body, or retained-metadata
   body).  Every candidate is still hash-verified against the bound revision by the caller, so
   this can never accept bytes the model did not receive.
2. ``enrich_read_receipt``: a READ_UNCHANGED receipt states, as typed fields, that the
   in-frame bytes are the edit precondition source and how many times this unchanged revision
   has already been re-read.  No task text, no directive.

Switch values: ``on`` | ``off`` | ``auto`` (default).  ``auto`` is ON exactly when plain
source labels are enabled (the frame arm that renders ``[Current source: ...]``); with legacy
labels no plain blocks exist, so behaviour is byte-identical.  ``off`` restores the baseline
everywhere.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping

from kernel_read_fixes import PLAIN_SOURCE_LABEL

REREAD_FIX_ENV = "STATEFRAME_REREAD_FIX"
PLAIN_PREFIX = f"[{PLAIN_SOURCE_LABEL}: "


def reread_fix_mode() -> str:
    value = os.environ.get(REREAD_FIX_ENV, "auto")
    if value not in {"on", "off", "auto"}:
        raise ValueError(f"{REREAD_FIX_ENV} must be on, off, or auto")
    return value


def enabled() -> bool:
    mode = reread_fix_mode()
    if mode == "auto":
        from kernel_read_fixes import plain_source_labels_enabled
        return plain_source_labels_enabled()
    return mode == "on"


def current_source_candidates(content: str, path: str) -> list[str]:
    """Raw-body candidates for a plain ``[Current source: path]`` block (unverified)."""
    from kernel_source_presentation import strip_callsite_census
    content = strip_callsite_census(content)
    header = f"{PLAIN_PREFIX}{path}]\n"
    if not isinstance(content, str) or not content.startswith(header):
        return []
    body = content[len(header):]
    out: list[str] = []
    if body.startswith("[Source metadata]\n"):
        marker, end = f"\n[Source content: {path}]\n", f"\n[End source content: {path}]"
        if marker in body and body.endswith(end):
            metadata, raw = body[len("[Source metadata]\n"):].split(marker, 1)
            try:
                meta = json.loads(metadata)
            except ValueError:
                return []
            if (isinstance(meta, Mapping) and meta.get("path") == path
                    and meta.get("freshness") == "current" and meta.get("start_line") == 1
                    and meta.get("start_column", 0) == 0
                    and meta.get("omitted_chars_in_selection") == 0
                    and meta.get("more_content_available") is False):
                out.append(raw[:-len(end)])
        return out
    if body.startswith("Coverage: COMPLETE FILE;"):
        out.append(body.partition("\n")[2])
        return out
    out.append(body)
    # Frame blocks are newline-terminated by the renderer; the file may not be.
    if body.endswith("\n"):
        out.append(body[:-1])
    return out


def enrich_read_receipt(receipt: dict, runtime: object) -> dict:
    """Add typed re-read facts to a READ_UNCHANGED receipt (mutates + returns)."""
    if not isinstance(receipt, dict) or receipt.get("status") != "READ_UNCHANGED":
        return receipt
    counts = getattr(runtime, "_reread_fix_counts", None)
    if not isinstance(counts, dict):
        counts = {}
        try:
            setattr(runtime, "_reread_fix_counts", counts)
        except (AttributeError, TypeError):
            pass
    key = (str(receipt.get("path")), str(receipt.get("revision")))
    counts[key] = counts.get(key, 0) + 1
    receipt["unchanged_rereads_of_this_revision"] = counts[key]
    receipt["edit_precondition_source"] = "IN_FRAME"
    receipt["message"] = (
        str(receipt.get("message", ""))
        + f"\nThose in-frame bytes are the source edit preconditions are checked against; "
        f"this unchanged revision has been re-read {counts[key]} time(s)."
    )
    return receipt


__all__ = ["REREAD_FIX_ENV", "current_source_candidates", "enabled",
           "enrich_read_receipt", "reread_fix_mode"]
