"""Graceful over-cap fallback for the host-closure presentation (r43).

``bounded_host_closure_projection`` used to raise "host closure mandatory
projection exceeds render cap" once satisfied rows were shed and the view was
still too large; the child crashed (r41 3/28 frame runs, r42 341/345
crawl_l_two and ldev_11).  With ``STATEFRAME_HOST_CLOSURE_RENDER_DEGRADE=on``
(the runtime passes ``degrade=True``) the projection degrades instead:

1. ``rows_active_detail`` / ``rows_minimal`` - shed optional detail (check file,
   receipts, failing check ids, and finally the active counterexample detail);
   every mandatory row stays as ``{obligation_id, status}``.
2. ``summary`` - mandatory rows become ``mandatory_summary`` with exact
   ``count``/``counts`` and as many ids as fit.  Omission metadata
   (``ids_omitted``, ``omitted_from``, ``omitted_digest``) is RESERVED at its
   worst-case size before any id is fitted, so dropped ids are always counted
   and identified (ordered positions ``omitted_from..count-1`` plus a digest;
   the journal lists every omitted id).  The optional envelope (snapshot,
   schema, mode) is shed only if the reserved summary would not fit.

``MIN_RENDER_CAP`` is the smallest cap for which the minimal summary fits for
any realistic row count; :func:`validate_render_cap` enforces it before any
request is rendered (runtime init) and on every degrading call.  The result
never exceeds the cap.  Decisions never read this projection (the reducer's
typed assessments stay authoritative).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_HOST_CLOSURE_RENDER_DEGRADE"
DEGRADATION_EVENT = "host_closure_render_degraded"
STATUSES = ("SATISFIED", "UNKNOWN", "UNSATISFIED")
DIGEST_CHARS = 16
_MAX_ROWS = 10**12 - 1  # sizing bound for MIN_RENDER_CAP (12 digits)


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _size(value: object) -> int:
    # Must match kernel_obligation_assess._projection_json_size exactly.
    return len(json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), default=str))


def _summary(*, count: int, counts: Mapping[str, int], ids: Sequence[str],
             omitted: int, omitted_from: int, digest: str) -> dict[str, object]:
    return {
        "count": count,
        "counts": dict(counts),
        "ids": list(ids),
        "ids_omitted": omitted,
        "omitted_from": omitted_from,
        "omitted_digest": digest,
    }


def _minimal_envelope(summary: Mapping[str, object]) -> dict[str, object]:
    return {"degraded": True, "assessments": [], "receipts": [],
            "mandatory_summary": dict(summary)}


def _reserved(count: int, counts: Mapping[str, int], ids: Sequence[str]) -> dict[str, object]:
    """Worst-case omission metadata for ``count`` rows (sizes only grow here)."""
    return _summary(count=count, counts=counts, ids=ids, omitted=count,
                    omitted_from=count, digest="f" * DIGEST_CHARS)


MIN_RENDER_CAP = _size(_minimal_envelope(_reserved(
    _MAX_ROWS, {status: _MAX_ROWS for status in STATUSES}, ())))


def validate_render_cap(cap: object) -> int:
    if type(cap) is not int or cap < MIN_RENDER_CAP:
        raise ValueError(
            f"host closure render cap {cap!r} is below the minimum usable cap {MIN_RENDER_CAP}")
    return cap


def _digest(ids: Sequence[str]) -> str:
    if not ids:
        return ""
    encoded = json.dumps(list(ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def _status(row: Mapping[str, object]) -> str:
    value = str(row.get("status", "UNKNOWN") or "UNKNOWN").upper()
    return value if value in STATUSES else "UNKNOWN"


def _ordered_ids(rows: Sequence[Mapping[str, object]], active_id: object) -> list[str]:
    ids = [str(row.get("obligation_id", "")) for row in rows]
    active = str(active_id or "")
    if active and active in ids:
        ids.remove(active)
        ids.insert(0, active)
    return ids


def _fit_summary(envelope: Mapping[str, object], ordered: Sequence[str],
                 counts: Mapping[str, int], cap: int) -> dict[str, object] | None:
    """Fit ids into ``envelope`` under ``cap`` with omission metadata reserved."""
    count = len(ordered)

    def measured(ids: Sequence[str]) -> int:
        return _size({**envelope, "mandatory_summary": _reserved(count, counts, ids)})

    if measured(()) > cap:
        return None
    kept: list[str] = []
    for ident in ordered:
        if measured([*kept, ident]) > cap:
            break
        kept.append(ident)
    omitted = list(ordered[len(kept):])
    summary = _summary(count=count, counts=counts, ids=kept, omitted=len(omitted),
                       omitted_from=len(kept), digest=_digest(omitted))
    result = {**envelope, "mandatory_summary": summary}
    if _size(result) > cap:  # unreachable: actual metadata <= reserved metadata
        return None
    return result


def degrade_projection(view: Mapping[str, object], *, cap: int, active_id: object = None,
                       journal: object = None) -> dict[str, object]:
    """Return a cap-fitting presentation of ``view`` and journal the degradation."""
    validate_render_cap(cap)
    rows = [dict(row) for row in view.get("assessments", ()) or () if isinstance(row, Mapping)]
    ordered = _ordered_ids(rows, active_id)
    counts: dict[str, int] = {}
    for row in rows:
        counts[_status(row)] = counts.get(_status(row), 0) + 1
    counts = dict(sorted(counts.items()))
    omitted_satisfied = view.get("omitted_satisfied", 0)
    base = {key: view[key] for key in ("schema", "mode", "snapshot") if key in view}
    base["degraded"] = True
    if omitted_satisfied:
        base["omitted_satisfied"] = omitted_satisfied
    active = str(active_id or "")

    def minimal(row: Mapping[str, object], keep_detail: bool) -> dict[str, object]:
        if keep_detail and str(row.get("obligation_id", "")) == active:
            return {key: value for key, value in row.items() if key != "failing_check_ids"}
        return {"obligation_id": row.get("obligation_id", ""), "status": row.get("status")}

    result: dict[str, object] | None = None
    stage = ""
    shed = ["check_file", "receipts", "failing_check_ids"]
    for stage_name, keep_detail in (("rows_active_detail", True), ("rows_minimal", False)):
        candidate = {**base, "assessments": [minimal(row, keep_detail) for row in rows],
                     "receipts": []}
        if _size(candidate) <= cap:
            result, stage = candidate, stage_name
            if not keep_detail:
                shed.append("active_counterexample_detail")
            break
    if result is None:
        shed = [*shed, "active_counterexample_detail", "assessment_rows"]
        envelopes = [
            {**base, "assessments": [], "receipts": []},
            {**{k: v for k, v in base.items() if k != "snapshot"},
             "assessments": [], "receipts": []},
            _minimal_envelope({})
        ]
        envelopes[-1].pop("mandatory_summary")
        for index, envelope in enumerate(envelopes):
            result = _fit_summary(envelope, ordered, counts, cap)
            if result is not None:
                stage = "summary"
                if index >= 1:
                    shed.append("snapshot")
                if index >= 2:
                    shed.extend(["schema", "mode", "omitted_satisfied"])
                break
    if result is None:  # cap >= MIN_RENDER_CAP makes this unreachable
        raise ValueError("host closure render cap below minimum usable cap")
    summary = result.get("mandatory_summary")
    rendered = list(summary["ids"]) if isinstance(summary, Mapping) else list(ordered)
    omitted_ids = list(ordered[len(rendered):]) if isinstance(summary, Mapping) else []
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(
            DEGRADATION_EVENT,
            producer="stateframe_host_closure",
            reason="mandatory_projection_exceeds_render_cap",
            stage=stage,
            cap=cap,
            rendered_chars=_size(result),
            mandatory_count=len(ordered),
            status_counts=counts,
            rendered_ids=rendered,
            omitted_count=len(omitted_ids),
            omitted_ids=omitted_ids,
            omitted_digest=_digest(omitted_ids),
            optional_detail_shed=shed,
            dispatch=False,
        )
    return result


__all__ = [
    "DEGRADATION_EVENT", "MIN_RENDER_CAP", "MODE_ENV", "degrade_projection", "enabled",
    "mode", "validate_render_cap",
]
