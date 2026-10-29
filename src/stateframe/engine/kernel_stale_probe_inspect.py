"""Opt-in, on-demand inspection of host-owned probe receipts.

The ordinary receipt sidecar intentionally does not expose the whole probe
history once this switch is enabled.  A caller selects one receipt by its
host-issued ID; this module then compares the receipt's observed revision with
the current frame binding and labels the result as evidence, never as a
completion proof.
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from typing import TypeAlias


MODE_ENV = "STATEFRAME_STALE_PROBE_INSPECT"
REQUEST_ENV = "STATEFRAME_STALE_PROBE_INSPECT_RECEIPT"
REQUEST_FIELD = "stale_probe_inspect_receipt_id"
INSPECTION_SCHEMA = "stateframe-stale-probe-inspection/1"
MAX_RECEIPT_ID_CHARS = 256
MAX_RENDERED_VALUE_CHARS = 1_200
JSONMapping: TypeAlias = Mapping[str, object]
ReceiptRow: TypeAlias = dict[str, object]


def stale_probe_inspect_mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return stale_probe_inspect_mode() == "on"


def _bounded_text(value: object, *, limit: int = MAX_RECEIPT_ID_CHARS) -> str:
    text = str(value).strip()
    return text[:limit]


def _as_rows(value: object) -> list[ReceiptRow]:
    if isinstance(value, Mapping):
        value = list(value.values())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _receipt_id(row: JSONMapping) -> str:
    return _bounded_text(
        row.get("receipt_id") or row.get("event_id") or row.get("id") or "UNKNOWN_RECEIPT"
    )


def _probe_rows(frame: JSONMapping) -> list[ReceiptRow]:
    rows: list[ReceiptRow] = []
    for key in ("probe_receipts", "probe_results", "recent_probe_receipts"):
        rows.extend(_as_rows(frame.get(key)))
    for row in _as_rows(frame.get("receipts")):
        if str(row.get("kind", row.get("tool", ""))).lower() == "probe":
            rows.append(row)
    return rows


def requested_receipt_id(frame: JSONMapping) -> str | None:
    """Return the single host-requested receipt, if any.

    The frame field is the normal embedding seam.  The environment selector
    is retained for runner/preflight arms that only expose process-scoped
    configuration and is deliberately consulted only while the feature is on.
    """
    for key in (REQUEST_FIELD, "inspect_probe_receipt_id"):
        value = frame.get(key)
        if value is not None and _bounded_text(value):
            return _bounded_text(value)
    value = os.environ.get(REQUEST_ENV)
    return _bounded_text(value) if value and _bounded_text(value) else None


def _revision(value: object, *, path: str | None = None) -> object:
    if isinstance(value, Mapping):
        if path is not None:
            for key in ("artifact_revisions", "product_files", "harness_files"):
                revisions = value.get(key)
                if isinstance(revisions, Mapping) and revisions.get(path) is not None:
                    return revisions[path]
        for key in (
            "observed_revision", "source_revision", "file_revision", "current_revision",
            "revision", "product_revision", "artifact_revision", "binding_revision",
        ):
            if value.get(key) is not None:
                return value[key]
        for key in ("binding", "after_binding", "current_binding", "revision_binding_state"):
            found = _revision(value.get(key), path=path)
            if found is not None:
                return found
        for key in ("artifact_revisions", "product_files", "harness_files"):
            revisions = value.get(key)
            if isinstance(revisions, Mapping) and len(revisions) == 1:
                return next(iter(revisions.values()))
    return value if value is not None and not isinstance(value, Mapping) else None


def _current_binding(frame: JSONMapping) -> ReceiptRow | None:
    for key in ("current_binding", "binding"):
        value = frame.get(key)
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))
    binding: ReceiptRow = {}
    for key in (
        "product_revision", "test_harness_revision", "environment_revision",
        "current_revision", "revision", "artifact", "environment", "snapshot_id",
    ):
        if frame.get(key) is not None:
            binding[key] = copy.deepcopy(frame[key])
    revisions = frame.get("artifact_revisions")
    if isinstance(revisions, Mapping):
        binding["artifact_revisions"] = copy.deepcopy(dict(revisions))
    for key in ("product_files", "harness_files"):
        value = frame.get(key)
        if isinstance(value, Mapping):
            binding[key] = copy.deepcopy(dict(value))
    return binding or None


def _current_revision(frame: JSONMapping, row: JSONMapping) -> object:
    path = row.get("path") or row.get("target_path")
    path_text = str(path) if path is not None else None
    for value in (frame.get("current_binding"), frame.get("binding")):
        found = _revision(value, path=path_text)
        if found is not None:
            return found
    for key in ("current_revision", "product_revision", "artifact_revision", "revision"):
        if frame.get(key) is not None:
            found = _revision(frame[key], path=path_text)
            if found is not None:
                return found
    revisions = frame.get("artifact_revisions")
    if isinstance(revisions, Mapping):
        if path_text is not None and revisions.get(path_text) is not None:
            return revisions[path_text]
        if len(revisions) == 1:
            return next(iter(revisions.values()))
    for key in ("product_files", "harness_files"):
        revisions = frame.get(key)
        if isinstance(revisions, Mapping):
            if path_text is not None and revisions.get(path_text) is not None:
                return revisions[path_text]
            if len(revisions) == 1:
                return next(iter(revisions.values()))
    return None


def _observed_revision(row: JSONMapping) -> object:
    return _revision(row, path=(str(row["path"]) if row.get("path") is not None else None))


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(value)


def _stale_reason(
    row: JSONMapping,
    frame: JSONMapping,
    observed_revision: object,
    current_revision: object,
    current_binding: Mapping[str, object] | None,
) -> str:
    status = next(
        (str(row[key]).upper() for key in ("status", "result", "result_class", "outcome", "verdict")
         if row.get(key) is not None),
        "",
    )
    if status == "STALE" or row.get("stale") is True:
        return "probe_reported_stale"
    observed_binding = next(
        (row.get(key) for key in ("observed_binding", "binding", "after_binding")
         if isinstance(row.get(key), Mapping)),
        None,
    )
    if observed_binding is not None:
        if current_binding is None:
            return "current_binding_unknown"
        if _canonical(observed_binding) != _canonical(current_binding):
            return "current_binding_mismatch"
    if observed_revision is None:
        return "observed_revision_missing"
    if current_revision is None:
        revisions = frame.get("artifact_revisions")
        if isinstance(revisions, Mapping) and revisions and not any(
                str(observed_revision) == str(value) for value in revisions.values()
        ):
            return "observed_revision_mismatch"
        return "current_binding_unknown"
    if str(observed_revision) != str(current_revision):
        return "observed_revision_mismatch"
    return "current_binding_matches"


def _reproduction_reference(row: JSONMapping, receipt_id: str) -> str:
    for key in ("reproduction_reference", "reproduction_ref", "reference"):
        value = row.get(key)
        if value is not None and _bounded_text(value):
            return _bounded_text(value)
    parts = []
    if row.get("definition_id") is not None:
        parts.append(f"definition={_bounded_text(row['definition_id'])}")
    if row.get("snapshot_id") is not None:
        parts.append(f"snapshot={_bounded_text(row['snapshot_id'])}")
    return "; ".join(parts) or f"receipt={receipt_id}"


def inspect_probe_receipt(frame: JSONMapping) -> ReceiptRow | None:
    """Build one immutable inspection result for the requested receipt."""
    if not enabled():
        return None
    requested = requested_receipt_id(frame)
    if requested is None:
        return None
    rows = _probe_rows(frame)
    row = next((candidate for candidate in rows if _receipt_id(candidate) == requested), None)
    binding = _current_binding(frame)
    if row is None:
        return {
            "schema": INSPECTION_SCHEMA,
            "receipt_id": requested,
            "found": False,
            "observed_revision": None,
            "current_binding": binding,
            "stale_reason": "receipt_not_found",
            "applicability": "UNKNOWN",
            "proof_status": "NO_CURRENT_PROOF",
            "reproduction_reference": f"receipt={requested}",
        }
    observed = _observed_revision(row)
    current = _current_revision(frame, row)
    reason = _stale_reason(row, frame, observed, current, binding)
    stale = reason != "current_binding_matches"
    return {
        "schema": INSPECTION_SCHEMA,
        "receipt_id": requested,
        "found": True,
        "observed_revision": observed,
        "current_revision": current,
        "current_binding": binding,
        "stale_reason": reason,
        "applicability": "STALE" if stale else "CURRENT",
        "proof_status": "NOT_CURRENT_PROOF" if stale else "EVIDENCE_ONLY",
        "reproduction_reference": _reproduction_reference(row, requested),
        "status": row.get("status", row.get("result", "UNKNOWN")),
        "reason": row.get("reason", ""),
        "output": row.get("output", row.get("captured_output", "")),
    }


def _render_value(value: object, *, limit: int = MAX_RENDERED_VALUE_CHARS) -> str:
    if isinstance(value, Mapping) or isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        text = _canonical(value)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text[:limit]


def _render_or_unknown(value: object) -> str:
    return _render_value("UNKNOWN" if value is None else value)


def render_inspection_lines(frame: JSONMapping) -> list[str]:
    """Render the selected host receipt as bounded, non-authoritative text."""
    inspection = inspect_probe_receipt(frame)
    if inspection is None:
        return []
    if not inspection["found"]:
        return [
            "ON-DEMAND STALE PROBE INSPECTION (host facts):",
            f"receipt: {inspection['receipt_id']}",
            "observed revision: UNKNOWN",
            f"current binding: {_render_or_unknown(inspection['current_binding'])}",
            f"stale reason: {inspection['stale_reason']}",
            "applicability: UNKNOWN; no current proof",
            f"reproduction reference: {inspection['reproduction_reference']}",
        ]
    if inspection["applicability"] == "STALE":
        applicability = "STALE; not current proof"
    else:
        applicability = "CURRENT; evidence only, not completion proof"
    lines = [
        "ON-DEMAND STALE PROBE INSPECTION (host facts):",
        f"receipt: {inspection['receipt_id']}",
        f"observed revision: {_render_or_unknown(inspection['observed_revision'])}",
        f"current binding: {_render_or_unknown(inspection['current_binding'])}",
        f"stale reason: {inspection['stale_reason']}",
        f"applicability: {applicability}",
        f"reproduction reference: {inspection['reproduction_reference']}",
        f"observed outcome: {_render_value(inspection['status'])}",
    ]
    if inspection.get("reason"):
        lines.append(f"host reason: {_render_value(inspection['reason'])}")
    if inspection.get("output"):
        lines.append(f"observed output: {_render_value(inspection['output'])}")
    return lines


__all__ = [
    "INSPECTION_SCHEMA",
    "MODE_ENV",
    "REQUEST_ENV",
    "REQUEST_FIELD",
    "enabled",
    "inspect_probe_receipt",
    "render_inspection_lines",
    "requested_receipt_id",
    "stale_probe_inspect_mode",
]
