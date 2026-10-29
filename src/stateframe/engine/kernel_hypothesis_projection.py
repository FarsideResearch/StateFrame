"""Revision-bound presentation for carried model hypotheses (R-142).

The runtime's historical ``hypotheses_not_facts`` field is useful continuity,
but a carried claim formed against an older source revision stayed "active"
after the premise was edited.  Under ``STATEFRAME_HYPOTHESES_NOT_FACTS=on``:

* the host binds each native hypothesis at its causal creation point to the
  product revision its request was built against (``bind_hypotheses``, called
  by ``kernel_live_projection``), and publishes the current product revision in
  ``HOST_REVISION_FIELD``;
* the projection compares that binding against the host revision only, never
  against revision/status values carried on the hypothesis row itself;
* eliminated hypotheses stay inactive and keep their rejection;
* evidence references resolve only against host-owned receipt rows; inline
  evidence payloads never become evidence, unresolved references are labelled.

Off is byte-identical: every entry point returns its input unchanged.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping, MutableMapping, Sequence


MODE_ENV = "STATEFRAME_HYPOTHESES_NOT_FACTS"
SCHEMA = "stateframe-hypotheses-not-facts/2"
BLOCK_HEADER = "[HYPOTHESES_NOT_FACTS | carried claims are hypotheses, not facts]"
ACTIVE_FIELD = "hypotheses_not_facts"
STALE_FIELD = "stale_hypotheses_not_facts"
META_FIELD = "hypotheses_not_facts_meta"
HOST_REVISION_FIELD = "hypothesis_host_revision"
BINDING_FIELD = "basis_binding"
REVISION_DOMAIN = "product_revision"
HYPOTHESIS_LABEL = "HYPOTHESIS"
STALE_HYPOTHESIS_LABEL = "STALE_HYPOTHESIS_DROPPED"
ELIMINATED_HYPOTHESIS_LABEL = "ELIMINATED_HYPOTHESIS_DROPPED"
PRESENTATION_HYPOTHESIS = HYPOTHESIS_LABEL
AUTHORITY_MODEL_REASONING = "MODEL_REASONING"
STATUS_UNVERIFIED = "UNVERIFIED"
STATUS_STALE = "STALE"
STATUS_ELIMINATED = "ELIMINATED"
STALENESS_CURRENT = "CURRENT"
STALENESS_STALE = STATUS_STALE
STALENESS_INVALIDATED = "INVALIDATED"
STALENESS_UNKNOWN = "UNKNOWN"
EVIDENCE_RECORDED = "RECORDED"
EVIDENCE_UNRESOLVED_ONLY = "UNRESOLVED_REFERENCES_ONLY"
EVIDENCE_NONE_RECORDED = "NONE_RECORDED"
EVIDENCE_UNRESOLVED = "UNRESOLVED"
MAX_HYPOTHESES = 12
MAX_EVIDENCE = 8
MAX_PATHS = 16


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _rows(value: object) -> list[object]:
    if isinstance(value, Mapping):
        return list(value.values())
    return list(value) if _sequence(value) else []


def _text(value: object, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _claim(value: object) -> str:
    if isinstance(value, Mapping):
        for key in ("claim", "text", "content", "statement", "intent", "summary"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return _text(candidate)
        return ""
    return _text(value) if isinstance(value, str) else ""


# --------------------------------------------------------------- host binding
def bind_hypotheses(records: object, revision_by_step: Mapping[int, str],
                    initial_revision: str) -> None:
    """Stamp unbound native hypothesis records with their causal revision.

    A hypothesis event recorded at trace turn ``t`` (folded ``step == t + 1``)
    was reasoned against the request of step ``t``.  When that step's revision
    is unknown the record binds to the initial revision, so any later product
    edit invalidates it (conservative).  Existing bindings are never rewritten.
    """
    if not enabled() or not _sequence(records):
        return
    for record in records:
        if not isinstance(record, MutableMapping) or isinstance(record.get(BINDING_FIELD), Mapping):
            continue
        step = record.get("step")
        revision = (revision_by_step.get(step - 1) if isinstance(step, int)
                    and not isinstance(step, bool) else None)
        record[BINDING_FIELD] = {
            "domain": REVISION_DOMAIN,
            "value": revision if revision is not None else initial_revision,
            "bound_at": "request_revision" if revision is not None else "initial_revision",
        }


def _host_revision(*frames: object) -> dict[str, str] | None:
    """Current revision from host-authored state only (never from a row)."""
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        value = frame.get(HOST_REVISION_FIELD)
        if (isinstance(value, Mapping) and value.get("domain") == REVISION_DOMAIN
                and value.get("value") is not None):
            result = {"domain": REVISION_DOMAIN, "value": str(value["value"])}
            if value.get("initial") is not None:
                result["initial"] = str(value["initial"])
            return result
    return None


def _basis(item: Mapping[str, object]) -> str | None:
    binding = item.get(BINDING_FIELD)
    if (isinstance(binding, Mapping) and binding.get("domain") == REVISION_DOMAIN
            and binding.get("value") is not None):
        return str(binding["value"])
    return None


# ----------------------------------------------------------------- premises
def _normal_path(value: object) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def _paths(value: object) -> list[str]:
    if isinstance(value, str):
        path = _normal_path(value)
        return [path] if path else []
    if _sequence(value):
        result: list[str] = []
        for item in value:
            result.extend(_paths(item))
        return result
    if not isinstance(value, Mapping):
        return []
    result = []
    for key, child in value.items():
        if str(key) in {"path", "target_path", "file", "file_path", "source_path",
                        "paths", "files", "changed_files", "premise_paths",
                        "premise_files", "targets"}:
            result.extend(_paths(child))
    return list(dict.fromkeys(path for path in result if path))[:MAX_PATHS]


def _premise_paths(item: Mapping[str, object]) -> list[str]:
    result: list[str] = []
    for key in ("premise_paths", "premise_files", "paths", "files", "path",
                "target_path", "file", "file_path"):
        result.extend(_paths(item.get(key)))
    return list(dict.fromkeys(result))[:MAX_PATHS]


def _applied_paths(frame: Mapping[str, object]) -> set[str]:
    result: set[str] = set()
    for change in _rows(frame.get("recent_changes")):
        if not isinstance(change, Mapping):
            continue
        status = str(change.get("observed_result", change.get(
            "result_class", change.get("result", change.get("status", ""))))).upper()
        if status == "APPLIED":
            result.update(_paths(change) or ["<unknown-path>"])
    return result


def _path_changed(path: str, changed: set[str]) -> bool:
    return any(path == other or path.endswith("/" + other) or other.endswith("/" + path)
               for other in changed)


# ----------------------------------------------------------------- evidence
def _evidence_rows(frame: Mapping[str, object]) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for key in ("receipts", "test_receipts", "recent_test_feedback", "probe_receipts",
                "host_check_receipts"):
        rows.extend(value for value in _rows(frame.get(key)) if isinstance(value, Mapping))
    return rows


def _reference(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key in ("reference", "ref", "receipt_id", "id", "span_id", "event_id"):
            if value.get(key) is not None:
                return str(value[key])
        return None
    return None if value is None else str(value)


def _evidence_refs(item: Mapping[str, object]) -> list[object]:
    refs: list[object] = []
    for key in ("evidence_refs", "supporting_evidence", "receipt_refs", "receipt_ids",
                "basis_refs", "evidence"):
        value = item.get(key)
        if isinstance(value, (Mapping, str)):
            refs.append(value)
        elif _sequence(value):
            refs.extend(value)
    return refs[:MAX_EVIDENCE]


def _evidence(item: Mapping[str, object], host_rows: Sequence[Mapping[str, object]]
              ) -> list[dict[str, object]]:
    """Resolve references against host rows; inline payload fields are ignored."""
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for ref in _evidence_refs(item):
        reference = _reference(ref)
        if not reference or reference in seen:
            continue
        seen.add(reference)
        row = next((row for row in host_rows if _reference(row) == reference), None)
        if row is None:
            result.append({"reference": reference, "resolution": EVIDENCE_UNRESOLVED})
            continue
        entry: dict[str, object] = {"reference": reference, "resolution": "HOST_RECEIPT"}
        for name, keys in (("kind", ("kind", "type")),
                           ("status", ("status", "result_class", "result")),
                           ("path", ("path", "target_path", "file_path")),
                           ("revision", ("revision",))):
            value = next((row[key] for key in keys if row.get(key) is not None), None)
            if value is not None:
                entry[name] = _normal_path(value) if name == "path" else _text(value, 80)
        result.append(entry)
    return result


def _provenance(item: Mapping[str, object]) -> list[str]:
    result = []
    for ref in _rows(item.get("source_refs")):
        if isinstance(ref, Mapping) and ref.get("event_id") is not None:
            result.append(f"{_text(ref.get('kind', 'trace'), 40)}:{_text(ref['event_id'], 80)}")
    return result[:MAX_EVIDENCE]


# --------------------------------------------------------------- projection
def _eliminated(item: Mapping[str, object]) -> bool:
    return (str(item.get("status", "")).upper() == STATUS_ELIMINATED
            or str(item.get("lifecycle", "")).lower() == "eliminated")


def _normalise(item: object, *, host: dict[str, str] | None, host_rows,
               changed: set[str], index: int) -> dict[str, object]:
    source = item if isinstance(item, Mapping) else {"claim": item}
    evidence = _evidence(source, host_rows)
    paths = _premise_paths(source)
    basis = _basis(source)
    current = host["value"] if host else None
    premise_changed = any(_path_changed(path, changed) for path in paths)
    eliminated = _eliminated(source)
    explicit_stale = (source.get("stale") is True or source.get("invalidated") is True
                      or str(source.get("status", "")).upper() == STATUS_STALE)
    if eliminated:
        freshness, status = STALENESS_INVALIDATED, STATUS_ELIMINATED
    else:
        if basis is not None and current is not None:
            revision_stale = basis != current
        elif basis is None:
            # Unbound claim: conservatively stale after any applied product edit.
            revision_stale = bool(changed) or bool(
                host and host.get("initial") is not None and host["initial"] != current)
        else:
            revision_stale = False
        stale = explicit_stale or premise_changed or revision_stale
        if stale:
            freshness = STALENESS_STALE
        elif basis is not None and current is not None:
            freshness = STALENESS_CURRENT
        else:
            freshness = STALENESS_UNKNOWN
        status = STATUS_STALE if stale else STATUS_UNVERIFIED
    resolved = [row for row in evidence if row.get("resolution") != EVIDENCE_UNRESOLVED]
    row: dict[str, object] = {
        "hypothesis_id": str(source.get("hypothesis_id", source.get(
            "event_id", source.get("id", f"h{index}")))),
        "claim": _claim(source),
        "presentation": PRESENTATION_HYPOTHESIS,
        "authority": AUTHORITY_MODEL_REASONING,
        "status": status,
        "evidence": evidence,
        "evidence_status": (EVIDENCE_RECORDED if resolved else
                            EVIDENCE_UNRESOLVED_ONLY if evidence else EVIDENCE_NONE_RECORDED),
        "basis_revision": basis if basis is not None else "UNBOUND",
        "current_revision": current if current is not None else "UNKNOWN",
        "staleness": freshness,
        "premises_changed": premise_changed,
        "premise_paths": paths,
    }
    provenance = _provenance(source)
    if provenance:
        row["provenance"] = provenance
    if eliminated:
        row["invalidation"] = "eliminated; rejection retained, do not reuse as a premise"
    if source.get("step") is not None:
        row["recorded_step"] = source["step"]
    return row


def project_frame(frame: Mapping[str, object], *, current_revision: object = None,
                  evidence_frame: Mapping[str, object] | None = None) -> dict[str, object]:
    """Project carried hypotheses into explicit, host-revision-bound records.

    ``current_revision`` is accepted for call-site compatibility but ignored:
    the comparison domain is the host's ``HOST_REVISION_FIELD`` only.
    """
    del current_revision
    public = copy.deepcopy(dict(frame))
    if not enabled():
        return public
    host_frame = evidence_frame if isinstance(evidence_frame, Mapping) else public
    raw = public.get(ACTIVE_FIELD)
    if not _sequence(raw):
        raw = public.get("hypotheses", ())
    values = list(raw) if _sequence(raw) else []
    source_values = host_frame.get("hypotheses") if host_frame is not public else None
    if _sequence(source_values) and values:
        # ``operational_view`` keeps only claim/status; restore the host record
        # (binding, lifecycle, provenance) by tail position.
        source_values = list(source_values)
        if len(source_values) >= len(values):
            tail = source_values[-len(values):]
            values = [({**source_item, **item}
                       if isinstance(source_item, Mapping) and isinstance(item, Mapping)
                       else item) for source_item, item in zip(tail, values)]
    host = _host_revision(host_frame, public)
    host_rows = _evidence_rows(host_frame)
    changed = _applied_paths(public) | (_applied_paths(host_frame) if host_frame is not public else set())
    active: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = [
        dict(row) for row in _rows(public.get(STALE_FIELD)) if isinstance(row, Mapping)]
    for index, item in enumerate(values[-MAX_HYPOTHESES:]):
        row = _normalise(item, host=host, host_rows=host_rows, changed=changed, index=index)
        (active if row["staleness"] in {STALENESS_CURRENT, STALENESS_UNKNOWN}
         else dropped).append(row)
    if not active and not dropped and not changed:
        # r44 F4 R6: nothing carried -> no empty ACTIVE list and no meta block on
        # every request (pure per-turn boilerplate, r42 regression class).  Only
        # the host-internal revision field is removed, as below.
        public.pop(HOST_REVISION_FIELD, None)
        return public
    public.pop("hypotheses", None)
    public.pop(HOST_REVISION_FIELD, None)
    public[ACTIVE_FIELD] = active
    if dropped:
        public[STALE_FIELD] = dropped[-MAX_HYPOTHESES:]
    else:
        public.pop(STALE_FIELD, None)
    public[META_FIELD] = {
        "schema": SCHEMA,
        "current_revision": host["value"] if host else "UNKNOWN",
        "revision_domain": REVISION_DOMAIN,
        "active_count": len(active),
        "stale_count": sum(1 for row in dropped if row.get("status") != STATUS_ELIMINATED),
        "eliminated_count": sum(1 for row in dropped if row.get("status") == STATUS_ELIMINATED),
        "premises_changed_paths": sorted(changed)[:MAX_PATHS],
        "evidence_policy": "host receipts only; model text and inline payloads are not evidence",
    }
    return public


def _evidence_text(rows: object) -> str:
    values = []
    for row in _rows(rows):
        if not isinstance(row, Mapping):
            continue
        details = [f"{key}={_text(row[key], 80)}" for key in
                   ("resolution", "kind", "status", "revision", "path") if row.get(key) is not None]
        values.append(str(row.get("reference", "UNKNOWN")) + (" (" + ", ".join(details) + ")"
                                                             if details else ""))
    return ", ".join(values) if values else "none recorded"


def _render_row(prefix: str, row: Mapping[str, object]) -> str:
    line = (f"{prefix}: {_text(row.get('claim'))} | status={row.get('status', 'UNKNOWN')} | "
            f"staleness={row.get('staleness', 'UNKNOWN')} | "
            f"basis_revision={row.get('basis_revision', 'UNBOUND')} | "
            f"current_revision={row.get('current_revision', 'UNKNOWN')} | "
            f"evidence={_evidence_text(row.get('evidence'))}")
    provenance = row.get("provenance")
    if _sequence(provenance) and provenance:
        line += " | provenance=" + ", ".join(map(str, provenance))
    return line


def render_block(frame: Mapping[str, object], *, current_revision: object = None) -> str:
    """Render one auditable model-facing block, or nothing while disabled."""
    if not enabled():
        return ""
    meta = frame.get(META_FIELD)
    projected = (frame if isinstance(meta, Mapping) and meta.get("schema") == SCHEMA
                 else project_frame(frame, current_revision=current_revision))
    meta = projected.get(META_FIELD, {})
    lines = [
        BLOCK_HEADER + " | schema=" + SCHEMA,
        "current_revision: " + str(meta.get("current_revision", "UNKNOWN")),
        "active_hypotheses: " + str(meta.get("active_count", 0)),
        "stale_hypotheses: " + str(meta.get("stale_count", 0)),
        "eliminated_hypotheses: " + str(meta.get("eliminated_count", 0)),
    ]
    for row in _rows(projected.get(ACTIVE_FIELD)):
        if isinstance(row, Mapping):
            lines.append(_render_row(HYPOTHESIS_LABEL, row))
    for row in _rows(projected.get(STALE_FIELD)):
        if isinstance(row, Mapping):
            lines.append(_render_row(ELIMINATED_HYPOTHESIS_LABEL
                                     if row.get("status") == STATUS_ELIMINATED
                                     else STALE_HYPOTHESIS_LABEL, row))
    changed = meta.get("premises_changed_paths")
    if _sequence(changed) and changed:
        lines.append("premises_changed_paths: " + ", ".join(map(str, changed)))
    if len(lines) == 5:
        # r44 F4 R6: nothing carried (no active/stale/eliminated rows, no changed
        # premises) -> no block at all.  A "none carried" boilerplate block on
        # every request under the default projector was a pure per-turn cost
        # (r42 regression class) with zero information.
        return ""
    return "\n".join(lines)


def append_block(rendered: str, frame: Mapping[str, object], *, current_revision: object = None,
                  runtime: object = None) -> str:
    """Append the block once, preserving the caller's off-path bytes.

    Under ``STATEFRAME_PROMPT_DIET=on`` and when a ``runtime`` is given, the
    full block is appended only the first time for that runtime, or again
    once its rendered content changes; an unchanged block is replaced with a
    short reference notice instead.  ``runtime=None`` (the default, used by
    every call site that predates this switch) is byte-identical to the
    original renderer regardless of the switch.
    """
    if not enabled() or BLOCK_HEADER in rendered:
        return rendered
    block = render_block(frame, current_revision=current_revision)
    if not block:
        return rendered
    if runtime is not None:
        from kernel_prompt_diet import (
            content_digest, enabled as diet_enabled, get_gate, unchanged_notice,
        )
        if diet_enabled():
            gate = get_gate(runtime)
            digest = content_digest(block)
            if not gate.should_render("hypotheses_not_facts", digest):
                notice = unchanged_notice("HYPOTHESES_NOT_FACTS", digest)
                return rendered.rstrip() + "\n\n" + notice
            gate.commit("hypotheses_not_facts", digest)
    return rendered.rstrip() + "\n\n" + block
