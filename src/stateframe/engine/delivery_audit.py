"""Host-side information-delivery diagnostics for the delivery contract.

This module is an extension point for ``kernel_delivery_manifest``.  It does
not build a second payload or decide whether a task is complete.  It audits a
payload/manifest that the delivery-contract lane already produced and keeps
the four useful observations separate:

* required spans and their current source revisions;
* explicit versus silent omissions;
* provider tool schemas versus the host dispatch allow-set; and
* readiness values projected by the different public surfaces.

The optional runtime arm is ``STATEFRAME_DELIVERY_AUDIT=off|on`` and defaults
to ``off``.  ``audit_events`` is also available as an offline replay helper;
it is intentionally usable without enabling the runtime arm so a recorded
journal can be diagnosed without changing the run that produced it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_DELIVERY_AUDIT"
SCHEMA = "stateframe-delivery-audit/1"
AUDIT_HEADER = f"[DELIVERY_AUDIT {SCHEMA}]"
PRODUCER = "delivery_audit"

_READY = frozenset({
    "ACCEPTED", "COMPLETE", "COMPLETED", "DONE", "GREEN", "PASS", "PASSED",
    "READY", "SUCCESS", "SUCCEEDED",
})
_BLOCKED = frozenset({
    "BLOCKED", "CHALLENGED", "ERROR", "FAIL", "FAILED", "INCOMPLETE",
    "NEEDS_WORK", "RED", "REFUSED", "UNKNOWN", "UNVERIFIED",
})
_REVISION_FIELD_RE = re.compile(
    r"(?:revision|rev|product_revision|workspace_revision)\s*[:=]\s*"
    r"(?P<value>sha256:[0-9a-f]{4,}(?:\.\.\.)?|[0-9a-f]{4,}(?:\.\.\.)?)",
    re.IGNORECASE,
)
_RETAINED_OMISSION_RE = re.compile(
    r"\[Retained source excerpt omitted:\s*(?P<reference>[^\]\n]+)\]",
    re.IGNORECASE,
)


def delivery_audit_mode() -> str:
    """Return the strict two-state runtime switch."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def mode() -> str:
    return delivery_audit_mode()


def enabled() -> bool:
    return delivery_audit_mode() == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _safe(value: object) -> object:
    """Return JSON-safe diagnostic data without granting it authority."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


def _items(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return ()


def _texts(value: object) -> list[str]:
    result: list[str] = []
    for item in _items(value):
        if isinstance(item, (bytes, bytearray)):
            text = bytes(item).decode("utf-8", "replace").strip()
        else:
            text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _first(mapping: Mapping[str, object], *keys: str, default: object = None) -> object:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _payload_mapping(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        return {}
    nested = payload.get("payload")
    return nested if isinstance(nested, Mapping) else payload


def _content_text(value: object) -> str:
    if isinstance(value, (str, bytes, bytearray)):
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("utf-8", "replace")
        return value
    if isinstance(value, Mapping):
        if value.get("type") in {"text", "input_text", "output_text"}:
            return str(value.get("text", value.get("content", "")))
        return " ".join(_content_text(item) for item in value.values())
    if isinstance(value, Sequence):
        return " ".join(_content_text(item) for item in value)
    return str(value) if value is not None else ""


def _message_text(payload: object) -> str:
    mapping = _payload_mapping(payload)
    messages = mapping.get("messages", payload if isinstance(payload, Sequence) else ())
    chunks: list[str] = []
    for message in _items(messages):
        if isinstance(message, Mapping):
            chunks.append(_content_text(message.get("content", message)))
        else:
            chunks.append(_content_text(message))
    return "\n".join(chunk for chunk in chunks if chunk)


def _tool_names(value: object) -> set[str]:
    names: set[str] = set()
    for item in _items(value):
        if isinstance(item, Mapping):
            if "tools" in item and len(item) == 1:
                names.update(_tool_names(item["tools"]))
                continue
            function = item.get("function")
            if isinstance(function, Mapping) and function.get("name") is not None:
                names.add(str(function["name"]).strip())
                continue
            name = _first(item, "name", "tool", "action", "id")
            if (name is not None and not isinstance(name, Mapping)
                    and not (isinstance(name, Sequence)
                             and not isinstance(name, (str, bytes, bytearray)))):
                names.add(str(name).strip())
                continue
            for key in ("schemas", "tool_schemas", "allowed_tools", "tools"):
                if key in item:
                    names.update(_tool_names(item[key]))
        elif str(item).strip():
            names.add(str(item).strip())
    return {name for name in names if name}


def _permission_names(value: object) -> set[str] | None:
    """Normalize common host allow-set shapes; ``None`` means not supplied."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("allow_set", "allowed", "allowed_tools", "dispatch_permissions",
                    "permissions", "actions", "tools"):
            if key in value:
                return _permission_names(value[key])
        names: set[str] = set()
        for key, flag in value.items():
            if isinstance(flag, bool):
                if flag:
                    names.add(str(key).strip())
            elif isinstance(flag, Mapping):
                allowed = _first(flag, "allowed", "enabled", "permitted", default=False)
                if bool(allowed):
                    names.add(str(key).strip())
            elif flag is not None and str(key) in {"name", "tool", "action"}:
                names.add(str(flag).strip())
        return {name for name in names if name}
    return _tool_names(value)


def _revision_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return False
    a = str(left).strip().lower()
    b = str(right).strip().lower()
    if a.startswith("sha256:"):
        a = a.removeprefix("sha256:")
    if b.startswith("sha256:"):
        b = b.removeprefix("sha256:")
    a_truncated = a.endswith("...")
    b_truncated = b.endswith("...")
    a = a.removesuffix("...")
    b = b.removesuffix("...")
    if not a or not b:
        return False
    if a == b:
        return True
    # Prefix matching is valid only for an explicitly ellipsized digest.  An
    # opaque revision such as ``r1`` must not equal ``r10`` by accident.
    return ((a_truncated and b.startswith(a))
            or (b_truncated and a.startswith(b)))


def _normal_revisions(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, revision in value.items():
        if revision is None or isinstance(revision, (Mapping, Sequence)) and not isinstance(
                revision, (str, bytes, bytearray)):
            continue
        result[str(key)] = str(revision)
    return result


def _span_rows(value: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(_items(value)):
        if isinstance(raw, Mapping):
            row = dict(raw)
        else:
            row = {"reference": str(raw)}
        reference = _first(row, "reference", "span_id", "id", "handle", "path")
        if reference is None:
            reference = f"span-{index}"
        row["span_id"] = str(_first(row, "span_id", "id", default=reference))
        row["reference"] = str(reference)
        path = _first(row, "path", "file", "source_path")
        if path is not None:
            row["path"] = str(path)
        row["required"] = _flag(row.get("required"), default=True)
        rows.append(row)
    return rows


def _explicit_references(value: object) -> set[str]:
    refs: set[str] = set()
    for item in _items(value):
        if isinstance(item, Mapping):
            for key in ("span_id", "reference", "id", "handle", "path"):
                candidate = item.get(key)
                if candidate is not None:
                    refs.add(str(candidate))
            for key in ("omissions", "omitted_items", "items", "references"):
                if key in item:
                    refs.update(_explicit_references(item[key]))
        elif str(item).strip():
            refs.add(str(item).strip())
    return refs


def _reference_list(value: object) -> list[str]:
    """Keep omission/reference identifiers without stringifying whole mappings."""
    result: list[str] = []
    for item in _items(value):
        if isinstance(item, Mapping):
            refs = _explicit_references(item)
            if not refs:
                fallback = _first(item, "reason", "message", "omission")
                if fallback is not None:
                    refs.add(str(fallback).strip())
        else:
            refs = {str(item).strip()}
        for reference in refs:
            if reference and reference not in result:
                result.append(reference)
    return result


def _text_omissions(text: str) -> tuple[str, str]:
    """Return known omitted span references and text safe for presence checks."""
    references: list[str] = []

    def replace(match: re.Match[str]) -> str:
        reference = match.group("reference").strip()
        if reference and reference not in references:
            references.append(reference)
        return ""

    presence_text = _RETAINED_OMISSION_RE.sub(replace, text)
    return tuple(references), presence_text


def _flag(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "required"}:
        return True
    if text in {"0", "false", "no", "off", "optional"}:
        return False
    return default


def _span_audit(
    payload: object,
    required_spans: object,
    current_revisions: object,
    omissions: object,
    delivered_spans: object,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = _span_rows(required_spans)
    current = _normal_revisions(current_revisions)
    text = _message_text(payload)
    text_omissions, presence_text = _text_omissions(text)
    explicit_refs = _explicit_references(delivered_spans)
    omission_text = set(_reference_list(omissions)) | set(text_omissions)
    audited: list[dict[str, object]] = []
    for row in rows:
        span_id = str(row["span_id"])
        reference = str(row["reference"])
        path = str(row.get("path", ""))
        literal = _first(row, "content", "text", "literal")
        present = bool(
            span_id in explicit_refs
            or reference in explicit_refs
            or (path and path in explicit_refs)
            or reference in presence_text
            or span_id in presence_text
            or (literal is not None and str(literal) in presence_text)
        )
        expected = _first(row, "revision", "current_revision", "product_revision")
        actual = current.get(path) if path else None
        revision_status = "unknown"
        if expected is not None and actual is not None:
            revision_status = "current" if _revision_equal(expected, actual) else "stale"
        elif expected is not None:
            revision_status = "unverified"
        elif actual is not None:
            revision_status = "current_unpinned"
        explicit_omission = any(
            candidate in omission_text for candidate in (span_id, reference, path) if candidate
        )
        if present and revision_status == "stale":
            status = "stale"
        elif present and revision_status == "unverified":
            status = "unverified"
        elif present:
            status = "delivered"
        elif explicit_omission:
            status = "omitted"
        else:
            status = "missing"
        audited.append({
            "span_id": span_id,
            "reference": reference,
            **({"path": path} if path else {}),
            "required": bool(row.get("required", True)),
            "expected_revision": expected,
            "current_revision": actual,
            "revision_status": revision_status,
            "status": status,
        })
    required_rows = [row for row in audited if row["required"]]
    if not required_rows:
        status = "not_applicable"
    elif any(row["status"] == "stale" for row in required_rows):
        status = "contradicted"
    elif any(row["status"] == "missing" for row in required_rows):
        status = "contradicted"
    elif any(row["status"] == "unverified" for row in required_rows):
        status = "unknown"
    elif any(row["status"] == "omitted" for row in required_rows):
        status = "dropped"
    else:
        status = "delivered"
    return {
        "status": status,
        "required_count": len(required_rows),
        "delivered_count": sum(row["status"] == "delivered" for row in required_rows),
        "omitted_count": sum(row["status"] == "omitted" for row in required_rows),
        "stale_count": sum(row["status"] == "stale" for row in required_rows),
        "missing_count": sum(row["status"] == "missing" for row in required_rows),
        "spans": audited,
    }, audited


def _omission_audit(
    omissions: object,
    spans: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    explicit = _reference_list(omissions)
    required = [row for row in spans if bool(row.get("required", True))]
    omitted = [str(row["span_id"]) for row in required if row.get("status") == "omitted"]
    explicit_set = set(explicit)
    def row_references(row: Mapping[str, object]) -> set[str]:
        return {
            str(row[key])
            for key in ("span_id", "reference", "path")
            if row.get(key)
        }

    silent = [
        str(row["span_id"])
        for row in required
        if row.get("status") == "missing"
        and not row_references(row).intersection(explicit_set)
    ]
    delivered_with_omission = [
        str(row["span_id"])
        for row in required
        if row.get("status") == "delivered"
        and row_references(row).intersection(explicit_set)
    ]
    if silent or delivered_with_omission:
        status = "contradicted"
    elif explicit or omitted:
        status = "recorded"
    else:
        status = "clear"
    return {
        "status": status,
        "items": explicit,
        "required_spans_omitted": omitted,
        "silent_required_spans": silent,
        "omission_matches_delivered": delivered_with_omission,
    }


def _tool_audit(
    payload: object,
    tool_schemas: object,
    dispatch_permissions: object,
) -> dict[str, object]:
    mapping = _payload_mapping(payload)
    schemas = tool_schemas
    if schemas is None:
        schemas = _first(mapping, "tools", "tool_schemas", default=())
    schema_names = _tool_names(schemas)
    permissions = dispatch_permissions
    if permissions is None:
        permissions = _first(
            mapping,
            "dispatch_permissions",
            "allow_set",
            "allowed_tools",
            "permissions",
        )
    allowed_names = _permission_names(permissions)
    if allowed_names is None:
        return {
            "status": "unknown",
            "schema_names": sorted(schema_names),
            "dispatch_allow_set": None,
            "schema_without_permission": [],
            "permission_without_schema": [],
            "permission_source": "not_supplied",
        }
    schema_without_permission = sorted(schema_names - allowed_names)
    permission_without_schema = sorted(allowed_names - schema_names)
    status = "contradicted" if schema_without_permission or permission_without_schema else "delivered"
    return {
        "status": status,
        "schema_names": sorted(schema_names),
        "dispatch_allow_set": sorted(allowed_names),
        "schema_without_permission": schema_without_permission,
        "permission_without_schema": permission_without_schema,
        "permission_source": "supplied",
    }


def _normalize_readiness(value: object) -> str:
    if isinstance(value, bool):
        return "READY" if value else "BLOCKED"
    if value is None:
        return "UNKNOWN"
    if isinstance(value, Mapping):
        nested = _first(value, "readiness", "status", "state", "result", "ready")
        return _normalize_readiness(nested)
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if text in _READY or text.endswith("_READY") or text.endswith("_GREEN"):
        return "READY"
    if text in _BLOCKED or text.endswith("_BLOCKED") or text.endswith("_FAIL"):
        return "BLOCKED"
    return "UNKNOWN"


def _readiness_rows(value: object) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if any(key in value for key in ("surface", "name", "id")):
            surface = str(_first(value, "surface", "name", "id", default="surface"))
            return [{"surface": surface, "readiness": _normalize_readiness(value)}]
        if any(key in value for key in ("readiness", "status", "state", "result", "ready")):
            return [{"surface": "surface", "readiness": _normalize_readiness(value)}]
        return [
            {"surface": str(key), "readiness": _normalize_readiness(item)}
            for key, item in value.items()
        ]
    rows: list[dict[str, str]] = []
    for index, item in enumerate(_items(value)):
        if isinstance(item, Mapping):
            surface = str(_first(item, "surface", "name", "id", default=f"surface-{index}"))
        else:
            surface = f"surface-{index}"
        rows.append({"surface": surface, "readiness": _normalize_readiness(item)})
    return rows


def _readiness_audit(value: object) -> dict[str, object]:
    rows = _readiness_rows(value)
    values = [row["readiness"] for row in rows]
    known = [item for item in values if item != "UNKNOWN"]
    if not rows or not known:
        status = "unknown"
        normalized = "UNKNOWN"
    elif len(set(known)) > 1:
        status = "contradicted"
        normalized = "CONFLICT"
    elif len(known) != len(values):
        status = "unknown"
        normalized = known[0]
    else:
        status = "delivered"
        normalized = known[0]
    return {
        "status": status,
        "normalized": normalized,
        "surfaces": rows,
        "surface_count": len(rows),
    }


def audit_request(
    request: object = None,
    *,
    payload: object = None,
    required_spans: object = None,
    current_revisions: object = None,
    omissions: object = None,
    delivered_spans: object = None,
    tool_schemas: object = None,
    dispatch_permissions: object = None,
    readiness: object = None,
    readiness_surfaces: object = None,
    request_id: object = None,
    manifest_id: object = None,
) -> dict[str, object]:
    """Audit one provider-bound request without changing it.

    ``request`` is accepted as a positional convenience for callers that
    already have a prepared request.  A delivery-contract manifest can be
    passed directly too; its nested ``payload`` and diagnostic fields are
    discovered conservatively.
    """
    source = payload if payload is not None else request
    source_mapping = source if isinstance(source, Mapping) else {}
    required = required_spans
    if required is None:
        required = _first(source_mapping, "required_spans", "required_span", "spans", default=())
    revisions = current_revisions
    if revisions is None:
        revisions = _first(
            source_mapping,
            "current_revisions",
            "artifact_revisions",
            "source_revisions",
            default={},
        )
    omission_values = omissions
    if omission_values is None:
        omission_values = _first(source_mapping, "omissions", "omitted_items", default=())
    explicit_delivered = delivered_spans
    if explicit_delivered is None:
        explicit_delivered = _first(
            source_mapping,
            "delivered_spans",
            "rendered_spans",
            "selected_spans",
            default=(),
        )
    span_check, span_rows = _span_audit(
        source,
        required,
        revisions,
        omission_values,
        explicit_delivered,
    )
    omission_check = _omission_audit(omission_values, span_rows)
    tool_check = _tool_audit(source, tool_schemas, dispatch_permissions)
    readiness_value = readiness_surfaces if readiness_surfaces is not None else readiness
    if readiness_value is None:
        readiness_value = _first(
            source_mapping,
            "readiness_surfaces",
            "surface_readiness",
            "readiness",
            default=None,
        )
    readiness_check = _readiness_audit(readiness_value)
    checks = {
        "required_spans": span_check,
        "omissions": omission_check,
        "tool_schema_vs_permission": tool_check,
        "readiness": readiness_check,
    }
    statuses = [str(check.get("status")) for check in checks.values()]
    if "contradicted" in statuses:
        status = "contradicted"
    elif "dropped" in statuses or "recorded" in statuses:
        status = "dropped"
    elif "unknown" in statuses:
        status = "unknown"
    else:
        status = "delivered"
    result: dict[str, object] = {
        "schema": SCHEMA,
        "producer": PRODUCER,
        "request_id": request_id if request_id is not None else source_mapping.get("request_id"),
        "manifest_id": manifest_id if manifest_id is not None else source_mapping.get("manifest_id"),
        "status": status,
        "checks": checks,
        "required_spans": span_rows,
        "omissions": omission_check["items"],
        "tool_schema_vs_permission": tool_check,
        "readiness": readiness_check,
        "host_only": True,
        "claims_block_completion": False,
    }
    result["audit_id"] = _digest(result)
    return result


def audit_manifest(
    manifest: Mapping[str, object],
    *,
    audit_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Audit a manifest emitted by ``kernel_delivery_manifest``.

    ``kernel_delivery_manifest`` intentionally stores hashes/counts rather than
    copying the full request or runtime menu.  The optional context therefore
    supplies the exact prepared payload and host projections at the call site;
    it is diagnostic input, not a second manifest or authority owner.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("delivery manifest must be a mapping")
    context = audit_context if isinstance(audit_context, Mapping) else {}
    payload = context.get("payload", manifest.get("payload", manifest))

    def field(*names: str) -> object:
        value = _first(manifest, *names, default=None)
        return value if value is not None else _first(context, *names, default=None)

    return audit_request(
        payload=payload,
        required_spans=field("required_spans", "required_span", "spans"),
        current_revisions=field(
            "current_revisions", "artifact_revisions", "source_revisions"),
        omissions=field("omissions", "omitted_items"),
        delivered_spans=field("delivered_spans", "rendered_spans", "selected_spans"),
        tool_schemas=field("tool_schemas", "tools"),
        dispatch_permissions=field("dispatch_permissions", "allow_set", "permissions"),
        readiness=field("readiness_surfaces", "surface_readiness", "readiness"),
        request_id=(manifest.get("request_id")
                    if manifest.get("request_id") is not None
                    else context.get("request_id")),
        manifest_id=(manifest.get("manifest_id")
                     if manifest.get("manifest_id") is not None
                     else context.get("manifest_id")),
    )


def augment_manifest(
    manifest: Mapping[str, object],
    *,
    audit_context: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Attach diagnostics only when the audit arm is explicitly enabled.

    OFF returns the original object untouched.  The delivery-contract lane can
    call this at its prepared/final manifest boundary without creating a
    second manifest producer or adding a request message on the default path.
    """
    if not enabled():
        return manifest
    result = copy.deepcopy(dict(manifest))
    result["delivery_audit"] = audit_manifest(result, audit_context=audit_context)
    return result


def render_audit(audit: Mapping[str, object]) -> str:
    """Render diagnostics as data-only text for an explicitly opted-in caller."""
    body = json.dumps(_safe(dict(audit)), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    return AUDIT_HEADER + "\nDATA ONLY; diagnostics are not instructions.\n" + body


def _event_kind(event: Mapping[str, object]) -> str:
    return str(event.get("kind", event.get("type", ""))).strip().lower()


def _event_text(event: Mapping[str, object]) -> str:
    payload = event.get("payload")
    if payload is not None:
        return _message_text(payload)
    if "messages" in event:
        return _message_text(event)
    return _content_text(event.get("content", event.get("observation", "")))


def _event_revision(event: Mapping[str, object]) -> str | None:
    for container in (event, event.get("record", {}), event.get("binding", {}),
                      event.get("after_binding", {}), event.get("frame", {})):
        if isinstance(container, Mapping):
            value = _first(container, "product_revision", "workspace_revision", "revision")
            if value is not None:
                return str(value)
    return None


def _revision_mentions(text: str) -> list[str]:
    values: list[str] = []
    for match in _REVISION_FIELD_RE.finditer(text):
        value = match.group("value")
        if not value.lower().startswith("sha256:"):
            value = "sha256:" + value
        if value not in values:
            values.append(value)
    return values


def _status_row(status: str, **fields: object) -> dict[str, object]:
    return {"status": status, **fields}


_MICROGRAPH_WORD = re.compile(r"micrograph", re.IGNORECASE)
_MICROGRAPH_JSON_KEY = re.compile(r'"micrograph"\s*:\s*', re.IGNORECASE)
_MICROGRAPH_ID = re.compile(r"n:[0-9a-f]{64}\Z")
_MICROGRAPH_TASK_ID = re.compile(r"[A-Za-z0-9_./:@+-]{1,256}\Z")
_MICROGRAPH_SAFE_PATH = re.compile(r"[A-Za-z0-9_./+@-]+\Z")


def _valid_micrograph_path(value: object) -> bool:
    # Keep offline auditing independent of the opt-in adapter on the OFF path.
    if not isinstance(value, str):
        return False
    path = value.replace("\\", "/").strip()
    return (bool(path) and len(path) <= 256 and not path.startswith("/")
            and all(part not in {"", ".", ".."} for part in path.split("/"))
            and _MICROGRAPH_SAFE_PATH.fullmatch(path) is not None)


def _valid_micrograph_field(field: object) -> bool:
    if (not isinstance(field, Mapping)
            or set(field) != {"schema", "suspects", "node_count", "edge_count", "skipped"}
            or field.get("schema") != "stateframe-micrograph-wire/1"
            or not isinstance(field.get("suspects"), list)
            or len(field["suspects"]) > 5):
        return False
    if any(type(field.get(key)) is not int or field[key] < 0
           for key in ("node_count", "edge_count", "skipped")):
        return False
    if field["node_count"] > 256 or field["edge_count"] > 384:
        return False
    for row in field["suspects"]:
        if (not isinstance(row, Mapping)
                or set(row) != {"path", "score", "evidence_ids"}
                or not _valid_micrograph_path(row.get("path"))
                or type(row.get("score")) is not int or not 1 <= row["score"] <= 5
                or not isinstance(row.get("evidence_ids"), list)
                or not 1 <= len(row["evidence_ids"]) <= 4
                or any(not isinstance(ref, str) or _MICROGRAPH_ID.fullmatch(ref) is None
                       for ref in row["evidence_ids"])):
            return False
    return True


def _micrograph_json_violation(value: object, *, key: str = "") -> bool:
    """Check a typed JSON carrier without treating task identifiers as prose."""
    if key == "micrograph":
        return not _valid_micrograph_field(value)
    if isinstance(value, Mapping):
        return any((not isinstance(name, str)
                    or (name != "micrograph" and _MICROGRAPH_WORD.search(name) is not None)
                    or _micrograph_json_violation(item, key=name))
                   for name, item in value.items())
    if isinstance(value, list):
        return any(_micrograph_json_violation(item) for item in value)
    if isinstance(value, str):
        if key in {"task_id", "required_by"} and _MICROGRAPH_TASK_ID.fullmatch(value):
            return False
        return _MICROGRAPH_WORD.search(value) is not None
    return False


def _micrograph_free_text(text: str) -> bool:
    """Admit only complete typed fields; scan all remaining request text."""
    remaining = []
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("micrograph:"):
            try:
                field = json.loads(stripped.split(":", 1)[1].strip())
            except (json.JSONDecodeError, ValueError):
                return True
            if not _valid_micrograph_field(field):
                return True
            continue
        if _MICROGRAPH_WORD.search(stripped) and stripped.startswith(("{", "[")):
            try:
                carrier = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                pass
            else:
                if _micrograph_json_violation(carrier):
                    return True
                continue
        # F4 can carry the field as a JSON member inside a larger typed state.
        cursor = 0
        spans = []
        for match in _MICROGRAPH_JSON_KEY.finditer(line):
            if match.start() < cursor:
                continue
            try:
                field, end = decoder.raw_decode(line, match.end())
            except (json.JSONDecodeError, ValueError):
                return True
            if not _valid_micrograph_field(field):
                return True
            spans.append(line[cursor:match.start()])
            cursor = end
        remaining.append("".join(spans) + line[cursor:])
    return _MICROGRAPH_WORD.search("\n".join(remaining)) is not None


def _micrograph_request_violation(request: Mapping[str, object]) -> bool:
    payload = request.get("payload", request)
    if not isinstance(payload, Mapping):
        return _micrograph_free_text(_content_text(payload))
    if "micrograph" in payload and not _valid_micrograph_field(payload["micrograph"]):
        return True
    # Tool descriptions are instructions, never a typed StateFrame field.
    if _MICROGRAPH_WORD.search(_content_text(payload.get("tools", ()))):
        return True
    # Preserve message boundaries and contents. Flattening whole message mappings
    # prefixes a typed field with its role ("user micrograph:"), falsely rejecting it.
    surfaces = (_message_text(payload),
                payload.get("content", ""), request.get("content", ""))
    return any(_micrograph_free_text(_content_text(surface)) for surface in surfaces)


def _audit_replayed_request(
    text: str,
    request: Mapping[str, object],
    state: Mapping[str, object],
    spec: Mapping[str, object],
) -> dict[str, object]:
    current_revision = state.get("current_revision")
    current_revisions = state.get("current_revisions", {})
    accepted_revisions = [current_revision] if current_revision is not None else []
    if isinstance(current_revisions, Mapping):
        accepted_revisions.extend(current_revisions.values())

    def matches_current(value: object) -> bool:
        return any(_revision_equal(value, revision)
                   for revision in accepted_revisions if revision is not None)

    receipts = state.get("receipts", ())
    required_selectors = _texts(
        _first(spec.get("public_row", {}), "public_check_paths", "selectors", default=())
        if isinstance(spec.get("public_row", {}), Mapping) else ()
    )
    latest = next(
        (item for item in reversed(tuple(receipts))
         if isinstance(item, Mapping)
         and str(item.get("result_class", item.get("status", ""))).upper()
         in {"GREEN", "PASS", "PASSED", "SUCCESS"}),
        None,
    )
    mentions = _revision_mentions(text)
    stale_mentions = [
        mention for mention in mentions
        if accepted_revisions and not matches_current(mention)
    ]
    public_receipt_bad = bool(
        re.search(r"public_receipt\s*:[^\n]*\b(?:selector|selectors)\s*=\s*UNKNOWN", text,
                  re.IGNORECASE)
        or re.search(r"public_receipt\s*:[^\n]*\brevision\s*=\s*(?:UNKNOWN|sha256:[0-9a-f]{4,}(?:\.\.\.)?)",
                     text, re.IGNORECASE)
    )
    if (latest is None or not accepted_revisions
            or not matches_current(_event_revision(latest))):
        latest_status = "missing"
    elif public_receipt_bad or stale_mentions:
        latest_status = "contradicted"
    else:
        latest_status = "delivered"
    if current_revision is None:
        revision_status = "unknown"
    elif stale_mentions:
        revision_status = "contradicted"
    elif any(matches_current(mention) for mention in mentions):
        revision_status = "delivered"
    else:
        revision_status = "missing"

    prior_request = state.get("last_request")
    request_payload = request.get("payload", {})
    identical = isinstance(prior_request, Mapping) and prior_request.get("payload") == request_payload
    blocks = [
        "\n".join(part.splitlines()).strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip().startswith("[") and part.strip()
    ]
    duplicated = [block for block, count in Counter(blocks).items() if count > 1]
    if duplicated:
        duplicate_status = "contradicted"
    else:
        duplicate_status = "clear"

    has_green_lock = "GREEN LOCK" in text.upper()
    if has_green_lock and identical:
        green_diff_status = "dropped"
    elif has_green_lock and state.get("last_edit") is not None:
        green_diff_status = "delivered"
    elif has_green_lock:
        green_diff_status = "missing"
    else:
        green_diff_status = "not_applicable"

    last_refusal = state.get("last_refusal")
    if last_refusal:
        refusal_text = str(last_refusal)
        refusal_delivered = refusal_text in text or (
            "REFUSED" in text.upper() or "CHALLENGED" in text.upper())
        refusal_status = "delivered" if refusal_delivered else "dropped"
        has_failing_audit = "FAIL_WITH_WITNESS" in text or "[CONTRACT AUDIT]" in text.upper()
        failing_status = "delivered" if has_failing_audit else "dropped"
    else:
        refusal_status = "not_applicable"
        failing_status = "not_applicable"

    return {
        "sequence": request.get("sequence"),
        "request_id": request.get("request_id"),
        "step": request.get("step"),
        "checks": {
            "latest_test_receipt": _status_row(
                latest_status,
                receipt_id=(latest.get("receipt_id") if isinstance(latest, Mapping) else None),
                receipt_revision=(_event_revision(latest) if isinstance(latest, Mapping) else None),
                current_revision=current_revision,
                current_revisions=(dict(current_revisions)
                                   if isinstance(current_revisions, Mapping) else {}),
                required_selectors=required_selectors,
            ),
            "current_product_revision": _status_row(
                revision_status,
                current_revision=current_revision,
                current_revisions=(dict(current_revisions)
                                   if isinstance(current_revisions, Mapping) else {}),
                reported_revisions=mentions,
            ),
            "byte_identical_consecutive_request": _status_row(
                "contradicted" if identical else "clear",
                compared=bool(prior_request),
            ),
            "green_lock_diff": _status_row(green_diff_status),
            "duplicated_blocks": _status_row(
                duplicate_status,
                count=len(duplicated),
            ),
            "failing_audit_after_refusal": _status_row(failing_status),
            "previous_refusal_or_challenge": _status_row(refusal_status),
            "micrograph_free_text": _status_row(
                "contradicted" if _micrograph_request_violation(request) else "clear"),
        },
    }


def audit_events(events: Sequence[Mapping[str, object]], *, spec: Mapping[str, object] | None = None) -> dict[str, object]:
    """Replay public journal events and diagnose what each request delivered."""
    configuration = dict(spec or {})
    state: dict[str, object] = {
        "current_revision": None,
        "current_revisions": {},
        "receipts": [],
        "last_edit": None,
        "last_refusal": None,
        "last_request": None,
    }
    requests: list[dict[str, object]] = []
    for raw_event in events:
        if not isinstance(raw_event, Mapping):
            continue
        event = dict(raw_event)
        kind = _event_kind(event)
        if kind == "snapshot_committed":
            frame = event.get("frame")
            if isinstance(frame, Mapping):
                revisions = _normal_revisions(frame.get("artifact_revisions", {}))
                state["current_revisions"] = revisions
                if revisions:
                    state["current_revision"] = next(iter(revisions.values()))
            revision = _event_revision(event)
            if revision is not None:
                state["current_revision"] = revision
        elif kind == "action_outcome_closed":
            record = event.get("record")
            record = record if isinstance(record, Mapping) else event
            outcome = str(_first(record, "result", "result_class", "status", default="")).upper()
            if outcome == "APPLIED":
                state["last_edit"] = record
                revision = _event_revision(record)
                if revision is not None:
                    state["current_revision"] = revision
                    revisions = state.get("current_revisions")
                    changes = record.get("changes") if isinstance(record, Mapping) else None
                    if isinstance(revisions, Mapping) and isinstance(changes, Sequence):
                        updated_revisions = dict(revisions)
                        changed_path = False
                        for change in changes:
                            if isinstance(change, Mapping):
                                path = _first(change, "path", "file", "target")
                                if path is not None:
                                    updated_revisions[str(path)] = revision
                                    changed_path = True
                        if changed_path:
                            state["current_revisions"] = updated_revisions
            if "REFUSED" in outcome or "CHALLENG" in outcome or "REFUSED" in str(
                    _first(record, "observation", "message", default="")).upper():
                state["last_refusal"] = _first(record, "observation", "message", default=outcome)
        elif kind in {"test_subprocess_completed", "check_completed", "verification_completed"}:
            receipts = state["receipts"]
            if isinstance(receipts, list):
                receipts.append(event)
            revision = _event_revision(event)
            if revision is not None and str(event.get("result_class", "")).upper() in {
                    "GREEN", "PASS", "PASSED", "SUCCESS"}:
                # A green receipt is evidence of the revision it names, not a
                # new product revision.  Keep the edit/snapshot binding above.
                event["receipt_revision"] = revision
        elif kind == "request_submitted":
            request = event
            row = _audit_replayed_request(
                _event_text(request), request, state, configuration)
            if isinstance(request.get("delivery_manifest"), Mapping):
                row["delivery_manifest"] = audit_manifest(request["delivery_manifest"])
            requests.append(row)
            state["last_request"] = request
    return {
        "schema": SCHEMA,
        "producer": PRODUCER,
        "requests": requests,
        "summary": {
            "request_count": len(requests),
            "contradicted_requests": sum(
                any(check.get("status") == "contradicted"
                    for check in row.get("checks", {}).values())
                for row in requests
            ),
        },
    }


# Names used by adjacent host-side diagnostics and by early lane probes.
audit_delivery = audit_request
build_audit = audit_request
audit_delivery_manifest = audit_manifest
attach_audit = augment_manifest


__all__ = [
    "AUDIT_HEADER", "MODE_ENV", "PRODUCER", "SCHEMA", "attach_audit",
    "audit_delivery", "audit_delivery_manifest", "audit_events", "audit_manifest",
    "audit_request", "build_audit", "augment_manifest", "delivery_audit_mode",
    "enabled", "mode", "render_audit",
]
