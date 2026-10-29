"""Host-truth provability classification for the completion projection.

This module is deliberately a pure adapter.  It classifies host-supplied
requirement, obligation, and clause frames from the current capability matrix;
it never turns a classification into completion authority.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


MODE_ENV = "STATEFRAME_PROVABILITY_ADVISORY"
SCHEMA = "stateframe-provability/1"
PROVABLE_NOW = "PROVABLE_NOW"
PROVABLE_AFTER_ACTION = "PROVABLE_AFTER_ACTION"
UNPROVABLE_HERE = "UNPROVABLE_HERE"
ADVISORY_START = "[TOKENS provability advisory]"
ADVISORY_END = "[/TOKENS provability advisory]"

_CONTAINER_KINDS = {
    "requirements": "requirement",
    "obligations": "obligation",
    "assessments": "obligation",
    "clauses": "clause",
    "items": "requirement",
}
_ID_KEYS = (
    "requirement_id", "obligation_id", "clause_id", "id", "name", "check_id",
)
_STATUS_KEYS = ("verdict", "status", "result", "result_class", "disposition")
_CHECK_KEYS = (
    "check_ids", "check_refs", "checks", "check_id", "check_identity",
    "check_identities", "check_ref",
)
_RECEIPT_KEYS = ("receipts", "evidence_refs", "receipt_ids", "evidence")
_CLAUSE_KEYS = (
    "clause_id", "clause_ids", "clause_ref", "clause_refs",
    "requirement_id", "requirement_ids", "obligation_id", "obligation_ids",
)
_PUBLIC_SELECTOR_KEYS = (
    "required_public_checks", "public_checks", "public_check_selectors",
    "check_selectors", "check_selector", "selector", "selectors",
)
_ROW_SELECTOR_KEYS = (
    "required_public_checks", "public_checks", "public_check_selectors",
    "check_selectors", "check_selector", "public_check_selector",
)
_PASS_RESULTS = {"PASS", "PASSED", "GREEN", "OK", "SATISFIED", "SUCCESS"}


def _text(value: object) -> str:
    return str(value).strip() if isinstance(value, (str, int, float, bool)) else ""


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            candidate = to_dict()
        except (AttributeError, TypeError, ValueError):
            candidate = None
        if isinstance(candidate, Mapping):
            return candidate
    fields = (
        *_ID_KEYS, *_STATUS_KEYS, *_CHECK_KEYS, *_RECEIPT_KEYS,
        *_CLAUSE_KEYS, *_PUBLIC_SELECTOR_KEYS,
        "reason", "clause", "current", "snapshot_digest", "revision",
        "environment", "fixture", "coverage_disposition", "workspace_revision",
        "snapshot_id", "binding", "after_binding", "current_binding",
        "receipt_id", "evidence_ref", "executed", "exit_code", "binding_stable",
        "waived", "waiver", "event", "kind", "display_status", "execution",
    )
    result = {key: getattr(value, key) for key in fields if hasattr(value, key)}
    return result


def _values(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    if value is None:
        return ()
    return (value,)


def _strings(value: object) -> tuple[str, ...]:
    result: list[str] = []
    for item in _values(value):
        candidate = _mapping(item)
        if candidate and "id" in candidate and not isinstance(item, (str, int, float, bool)):
            item = candidate.get("id")
        text = _text(item)
        if text:
            result.append(text)
    return tuple(dict.fromkeys(result))


def _first(row: Mapping[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = _text(row.get(key))
        if value:
            return value
    return ""


def _current_revision(row: Mapping[str, object], fallback: str) -> str:
    direct = _first(row, (
        "snapshot_digest", "revision", "environment_revision", "snapshot_id",
    ))
    if direct:
        return direct
    # Public subprocess receipts carry the candidate identity in the binding
    # envelope rather than duplicating it at top level.
    for key in ("after_binding", "current_binding", "binding"):
        nested = _mapping(row.get(key))
        if nested:
            value = _first(nested, (
                "snapshot_digest", "revision", "environment_revision", "workspace_revision",
                "snapshot_id", "artifact",
            ))
            if value:
                return value
    # Numeric workspace revisions are a compatibility fallback; when a receipt
    # also carries a digest in its binding envelope, the digest is authoritative.
    workspace_revision = _text(row.get("workspace_revision"))
    if workspace_revision:
        return workspace_revision
    return fallback


def _frame_rows(frames: object) -> list[tuple[str, Mapping[str, object]]]:
    rows: list[tuple[str, Mapping[str, object]]] = []
    ordinal = 0

    def visit(value: object, default_kind: str = "requirement") -> None:
        nonlocal ordinal
        if isinstance(value, Mapping):
            container_seen = False
            for key, kind in _CONTAINER_KINDS.items():
                if key in value:
                    container_seen = True
                    for item in _values(value[key]):
                        visit(item, kind)
            if container_seen:
                for key in ("frame_tracks", "tracks"):
                    for item in _values(value.get(key)):
                        visit(item, default_kind)
                return
            row = dict(value)
        else:
            row = dict(_mapping(value))
        if not row:
            row = {"id": f"{default_kind}:unidentified:{ordinal}"}
        row.setdefault("kind", default_kind)
        if not _first(row, _ID_KEYS):
            row["id"] = f"{default_kind}:unidentified:{ordinal}"
        rows.append((default_kind, row))
        ordinal += 1
        nested = row.get("clause")
        if isinstance(nested, Mapping) or nested is not None and not isinstance(nested, (str, bytes)):
            visit(nested, "clause")

    visit(frames)
    return rows


def _row_id(kind: str, row: Mapping[str, object], ordinal: int) -> str:
    return _first(row, _ID_KEYS) or f"{kind}:unidentified:{ordinal}"


def _row_checks(row: Mapping[str, object]) -> tuple[str, ...]:
    result: list[str] = []
    for key in _CHECK_KEYS:
        result.extend(_strings(row.get(key)))
    return tuple(dict.fromkeys(result))


def _row_clause_ids(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return explicit clause identities; never infer a clause from prose."""
    result: list[str] = []
    for key in _CLAUSE_KEYS:
        result.extend(_strings(row.get(key)))
    result.append(_row_id(_text(row.get("kind")) or "requirement", row, 0))
    return tuple(dict.fromkeys(item for item in result if item))


def _row_selectors(row: Mapping[str, object]) -> tuple[str, ...]:
    result: list[str] = []
    for key in _ROW_SELECTOR_KEYS:
        result.extend(_strings(row.get(key)))
    return tuple(dict.fromkeys(result))


def _receipt_checks(receipt: Mapping[str, object]) -> tuple[str, ...]:
    result: list[str] = []
    for key in _CHECK_KEYS:
        result.extend(_strings(receipt.get(key)))
    return tuple(dict.fromkeys(result))


def _receipt_clause_ids(receipt: Mapping[str, object]) -> tuple[str, ...]:
    result: list[str] = []
    for key in _CLAUSE_KEYS:
        result.extend(_strings(receipt.get(key)))
    return tuple(dict.fromkeys(result))


def _receipt_selectors(receipt: Mapping[str, object]) -> tuple[str, ...]:
    result: list[str] = []
    for key in _PUBLIC_SELECTOR_KEYS:
        result.extend(_strings(receipt.get(key)))
    return tuple(dict.fromkeys(result))


def _receipt_is_passing(receipt: Mapping[str, object]) -> bool:
    """Require an executed, non-waived passing check before binding a clause."""
    if receipt.get("executed") is False or receipt.get("waived") is True:
        return False
    waiver = _text(receipt.get("waiver"))
    if waiver:
        return False
    exit_code = receipt.get("exit_code")
    if exit_code is not None:
        try:
            if int(exit_code) != 0:
                return False
        except (TypeError, ValueError):
            return False
    status = _first(receipt, (
        "status", "display_status", "result", "result_class", "execution", "verdict",
    )).upper()
    if status:
        return status in _PASS_RESULTS
    # test_subprocess_completed records encode success as exit_code=0 and do
    # not repeat a second status field.
    return exit_code == 0


def _receipt_binds_row(row: Mapping[str, object], receipt: Mapping[str, object]) -> bool:
    """Match only explicit check/clause/selector relations."""
    row_checks = set(_row_checks(row))
    receipt_checks = set(_receipt_checks(receipt))
    if row_checks and row_checks.intersection(receipt_checks):
        return True
    row_clauses = set(_row_clause_ids(row))
    if row_clauses.intersection(_receipt_clause_ids(receipt)):
        return True
    row_selectors = set(_row_selectors(row))
    if row_selectors and row_selectors.intersection(_receipt_selectors(receipt)):
        return True
    # Host-issued identities such as check:ul:requirement:R1@... explicitly
    # name the clause.  Parse that structured identity; do not use a loose
    # substring match that could bind R1 to R10.
    item_ids = {item for item in row_clauses if item}
    for identity in receipt_checks:
        for item_id in item_ids:
            if any(token in identity for token in (
                    f":{item_id}:", f":{item_id}@", f"/{item_id}/", f"/{item_id}@")):
                return True
    return False


def _row_receipts(row: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    result: list[Mapping[str, object]] = []
    for key in _RECEIPT_KEYS:
        for item in _values(row.get(key)):
            mapped = _mapping(item)
            if mapped:
                result.append(mapped)
            elif _text(item):
                result.append({"receipt_id": _text(item)})
    return tuple(result)


def _merge_rows(rows: Sequence[tuple[str, Mapping[str, object]]]) -> tuple[tuple[str, dict[str, object]], ...]:
    merged: dict[tuple[str, str], dict[str, object]] = {}
    for ordinal, (kind, source) in enumerate(rows):
        row = dict(source)
        item_id = _row_id(kind, row, ordinal)
        key = (kind, item_id)
        prior = merged.get(key)
        if prior is None:
            row["id"] = item_id
            row["kind"] = kind
            row["check_ids"] = list(_row_checks(row))
            row["receipts"] = [dict(item) for item in _row_receipts(row)]
            merged[key] = row
            continue
        checks = list(dict.fromkeys((*_strings(prior.get("check_ids")), *_row_checks(row))))
        receipts = [*(_mapping(item) for item in _values(prior.get("receipts"))),
                    *_row_receipts(row)]
        prior.update({key_name: value for key_name, value in row.items()
                      if key_name not in {"check_ids", "receipts"} and value not in (None, "")})
        prior["check_ids"] = checks
        prior["receipts"] = [dict(item) for item in receipts if item]
    return tuple((kind, row) for (kind, _), row in merged.items())


def _offered_tools(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    result: list[str] = []
    source = (value,) if isinstance(value, Mapping) and (
        "function" in value or "name" in value) else _values(value)
    for item in source:
        row = _mapping(item)
        function = row.get("function") if isinstance(row, Mapping) else None
        if isinstance(function, Mapping):
            name = _text(function.get("name"))
        else:
            name = _text(row.get("name")) if row else _text(item)
        if name:
            result.append(name)
    return tuple(dict.fromkeys(result))


def _capability_rows(value: object) -> tuple[Mapping[str, object], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        for key in ("capabilities", "actions", "checks"):
            if key in value:
                return tuple(_mapping(item) for item in _values(value[key]) if _mapping(item))
        if any(key in value for key in ("action_id", "tool", "check_id", "executable")):
            return (dict(value),)
    return tuple(_mapping(item) for item in _values(value) if _mapping(item))


def _receipt_matches(row: Mapping[str, object], receipt: Mapping[str, object], revision: str) -> bool:
    if not _receipt_binds_row(row, receipt):
        return False
    receipt_revision = _current_revision(receipt, "")
    if not revision or not receipt_revision or receipt_revision != revision:
        return False
    receipt_id = _first(receipt, ("receipt_id", "id", "evidence_ref"))
    return bool(receipt_id and _receipt_is_passing(receipt))


def _capability_match(row: Mapping[str, object], capability: Mapping[str, object]) -> bool:
    item_id = _first(row, ("id", "obligation_id", "requirement_id", "clause_id"))
    row_checks = set(_strings(row.get("check_ids")))
    target_ids = set(_strings(capability.get("item_ids")))
    target_ids.update(_strings(capability.get("obligation_ids")))
    target_ids.update(_strings(capability.get("requirement_ids")))
    target_ids.update(_strings(capability.get("clause_ids")))
    if target_ids and item_id not in target_ids:
        return False
    check_id = _first(capability, ("check_id", "check"))
    return bool(check_id and row_checks and check_id in row_checks)


def _action(capability: Mapping[str, object], offered: tuple[str, ...] | None,
            row: Mapping[str, object]) -> tuple[dict[str, object] | None, str]:
    tool = _first(capability, ("tool", "action_tool"))
    action_id = _first(capability, ("action_id", "id"))
    check_id = _first(capability, ("check_id", "check"))
    arguments = capability.get("arguments")
    if not action_id or not check_id:
        return None, "NO_BOUND_CHECK"
    if not isinstance(arguments, Mapping):
        return None, "UNEXECUTABLE_CHECK"
    if capability.get("executable") is not True:
        return None, "UNEXECUTABLE_CHECK"
    if capability.get("available") is False:
        return None, "CAPABILITY_UNAVAILABLE"
    if offered is None:
        return None, "HOST_CAPABILITY_MATRIX_UNKNOWN"
    if not tool or tool not in offered:
        return None, "TOOL_NOT_OFFERED"
    return {
        "action_id": action_id,
        "check_id": check_id,
        "tool": tool,
        "arguments": dict(arguments),
        "item_id": _first(row, ("id", "obligation_id", "requirement_id", "clause_id")),
        "receipt_binding": _first(capability, ("receipt_binding", "revision")),
        "expected_result": _first(capability, ("expected_result", "expected")),
    }, ""


@dataclass(frozen=True)
class ProvabilityItem:
    item_id: str
    kind: str
    classification: str
    reason: str
    check_ids: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()
    action: Mapping[str, object] | None = None

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.item_id}"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.item_id,
            "kind": self.kind,
            "classification": self.classification,
            "reason": self.reason,
            "check_ids": list(self.check_ids),
            "receipt_ids": list(self.receipt_ids),
            "action": dict(self.action) if isinstance(self.action, Mapping) else None,
        }


@dataclass(frozen=True)
class ProvabilityReport:
    items: tuple[ProvabilityItem, ...]
    basis_digest: str
    current_revision: str
    host_known: bool

    @property
    def now_items(self) -> tuple[ProvabilityItem, ...]:
        return tuple(item for item in self.items if item.classification == PROVABLE_NOW)

    @property
    def after_action_items(self) -> tuple[ProvabilityItem, ...]:
        return tuple(item for item in self.items if item.classification == PROVABLE_AFTER_ACTION)

    @property
    def unprovable_items(self) -> tuple[ProvabilityItem, ...]:
        return tuple(item for item in self.items if item.classification == UNPROVABLE_HERE)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "basis_digest": self.basis_digest,
            "current_revision": self.current_revision,
            "host_known": self.host_known,
            "items": [item.to_dict() for item in self.items],
        }


def classify_frames(frames: object, *, capabilities: object = None,
                    offered_tools: object = None, current_revision: object = "",
                    receipts: object = None) -> ProvabilityReport:
    """Classify each host frame using only current receipts and exposed actions."""
    revision = _text(current_revision)
    offered = _offered_tools(offered_tools)
    capability_rows = _capability_rows(capabilities)
    receipt_source = ((receipts,) if isinstance(receipts, Mapping)
                      and any(key in receipts for key in ("receipt_id", "check_id", "status"))
                      else _values(receipts))
    receipt_rows = tuple(_mapping(item) for item in receipt_source if _mapping(item))
    merged = _merge_rows(_frame_rows(frames))
    items: list[ProvabilityItem] = []
    for kind, row in merged:
        item_id = _row_id(kind, row, len(items))
        row_receipts = (*_row_receipts(row), *receipt_rows)
        matching_receipts = tuple(item for item in row_receipts if _receipt_matches(row, item, revision))
        receipt_ids = tuple(dict.fromkeys(
            _first(item, ("receipt_id", "id", "evidence_ref"))
            for item in matching_receipts
            if _first(item, ("receipt_id", "id", "evidence_ref"))))
        checks = _row_checks(row)
        if matching_receipts:
            items.append(ProvabilityItem(item_id, kind, PROVABLE_NOW,
                                         "CURRENT_RECEIPT", checks, receipt_ids))
            continue
        matching_caps = [capability for capability in capability_rows
                         if _capability_match({**row, "id": item_id, "check_ids": checks}, capability)]
        selected_action: dict[str, object] | None = None
        failure_reason = "NO_AVAILABLE_TOOL"
        for capability in matching_caps:
            selected_action, failure_reason = _action(capability, offered, {**row, "id": item_id})
            if selected_action is not None:
                break
        if selected_action is not None:
            items.append(ProvabilityItem(item_id, kind, PROVABLE_AFTER_ACTION,
                                         "EXACT_HOST_ACTION", checks, (), selected_action))
        else:
            if not capability_rows and offered is None:
                failure_reason = "HOST_CAPABILITY_MATRIX_UNKNOWN"
            elif not checks and not matching_caps:
                failure_reason = "NO_BOUND_CHECK"
            items.append(ProvabilityItem(item_id, kind, UNPROVABLE_HERE,
                                         failure_reason, checks, ()))
    basis = {
        "schema": SCHEMA,
        "revision": revision,
        "offered_tools": offered,
        "items": [item.to_dict() for item in items],
    }
    encoded = json.dumps(basis, sort_keys=True, ensure_ascii=False, default=str,
                         separators=(",", ":"))
    digest = "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return ProvabilityReport(tuple(items), digest, revision, bool(merged) and offered is not None)


def _clip(value: object, limit: int) -> str:
    text = " ".join(_text(value).split())
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def advisory_for(report: ProvabilityReport, *, accepted: bool,
                 authority_known: bool = True,
                 seen: Sequence[str] = ()) -> tuple[str, tuple[str, ...]]:
    """Return one bounded advisory and the keys it consumed.

    The caller owns the seen-set because deduplication is presentation state;
    classification itself remains side-effect free.
    """
    seen_keys = set(_text(item) for item in seen if _text(item))
    consumed: list[str] = []
    if not report.host_known:
        key = f"abstain:unknown:{report.basis_digest}"
        if key in seen_keys:
            return "", ()
        return (f"{ADVISORY_START}\nProvability advisory: abstain; the host capability matrix is unknown. "
                "Completion authority remains authoritative.\n{ADVISORY_END}"), (key,)

    new_unprovable = [item for item in report.unprovable_items if item.key not in seen_keys]
    after = report.after_action_items
    if new_unprovable:
        details = "; ".join(
            f"{item.item_id} ({item.reason})" for item in new_unprovable[:6])
        if len(new_unprovable) > 6:
            details += f"; +{len(new_unprovable) - 6} more"
        consumed.extend(item.key for item in new_unprovable)
        text = (f"{ADVISORY_START}\nUnprovable here: {details}. "
                "No available host tool can establish these items; they are not counted as satisfied.")
        if after:
            item = after[0]
            action = item.action or {}
            text += (f" One receipt-bound check remains: do {action.get('tool', 'the host action')} "
                     f"for {item.item_id} ({action.get('check_id', 'bound check')}).")
        text += f"\n{ADVISORY_END}"
        consumed.append(f"unprovable:{report.basis_digest}")
        return text, tuple(consumed)

    if report.unprovable_items and not accepted and not after:
        return "", ()

    if after:
        item = after[0]
        action = item.action or {}
        key = f"check:{report.basis_digest}:{item.key}"
        if key in seen_keys:
            return "", ()
        consumed.append(key)
        args = json.dumps(action.get("arguments", {}), sort_keys=True, separators=(",", ":"))
        text = (f"{ADVISORY_START}\nProvability advisory (receipt-bound): do this one specific check: "
                f"{_clip(action.get('tool'), 60)}({args}) for {item.item_id}; bind its receipt to "
                f"{_clip(action.get('check_id'), 80)}. Completion authority remains authoritative.\n"
                f"{ADVISORY_END}")
        return text, tuple(consumed)

    if accepted:
        key = f"submit:{report.basis_digest}"
        if key in seen_keys:
            return "", ()
        consumed.append(key)
        unprovable_count = len(report.unprovable_items)
        suffix = (f" with {unprovable_count} explicitly unprovable item(s)"
                  if unprovable_count else "")
        return (f"{ADVISORY_START}\nProvability advisory (receipt-bound): submit now{suffix}. "
                "Completion authority and safeguards remain authoritative.\n{ADVISORY_END}"), tuple(consumed)

    key = f"abstain:authority:{report.basis_digest}"
    if key in seen_keys:
        return "", ()
    consumed.append(key)
    authority_text = "unknown" if not authority_known else "not authorized"
    return (f"{ADVISORY_START}\nProvability advisory: abstain; completion authority is {authority_text}. "
            "No submission claim is made.\n{ADVISORY_END}"), tuple(consumed)


def completion_record(report: ProvabilityReport, *, accepted: bool) -> dict[str, object]:
    """Build an honest terminal record without changing completion state."""
    items = report.unprovable_items
    details = "; ".join(f"{item.item_id} ({item.reason})" for item in items)
    label = "submitted" if accepted else "not submitted"
    text = (f"{label} with unknown provability state"
            if not report.host_known else f"{label} with {len(items)} unprovable items")
    if details:
        text += f": {details}"
    if not report.host_known:
        text += ": host capability matrix unknown"
    return {
        "schema": SCHEMA,
        "text": text,
        "accepted": accepted,
        "unprovable_count": len(items),
        "unprovable_items": [item.to_dict() for item in items],
        "basis_digest": report.basis_digest,
    }


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    return "on" if value in {"1", "on", "true", "yes"} else "off"


def _assessment_frame(assessment: object) -> dict[str, object]:
    row = dict(_mapping(assessment))
    if not row:
        return {}
    row.setdefault("id", _first(row, ("obligation_id", "id")))
    row.setdefault("kind", "obligation")
    return row


def runtime_report(runtime: object, decision: object = None, *, frame: object = None,
                   tools: object = None) -> ProvabilityReport:
    """Project only runtime-owned frames and capabilities into the pure classifier."""
    sources: dict[str, object] = {}

    def append_source(name: str, value: object) -> None:
        if value is None:
            return
        existing = sources.get(name)
        if existing is None:
            sources[name] = value
        else:
            sources[name] = [*_values(existing), *_values(value)]

    ledger = getattr(runtime, "requirements_ledger", None)
    if isinstance(ledger, Mapping):
        append_source("requirements", ledger.get("requirements", ledger))
    assessments = getattr(runtime, "host_closure_assessments", ())
    decision_assessments = getattr(decision, "obligation_assessments", ())
    append_source("obligations", [*_values(assessments), *_values(decision_assessments)])
    frame_row = _mapping(frame)
    for key, kind in _CONTAINER_KINDS.items():
        if key in frame_row:
            append_source(key, frame_row[key])
    for envelope_key in ("host_closure", "completion_contract"):
        envelope = frame_row.get(envelope_key)
        if isinstance(envelope, Mapping):
            for key in _CONTAINER_KINDS:
                if key in envelope:
                    append_source(key, envelope[key])
    needs_work = _strings(getattr(decision, "needs_work_ids", ()))
    if needs_work:
        append_source("clauses", [{"id": item, "status": "OPEN"} for item in needs_work])
    capabilities: list[dict[str, object]] = []
    payloads = (
        getattr(decision, "repair_action", None),
        (getattr(decision, "needs_work_payload", None) or {}).get("next_action")
        if isinstance(getattr(decision, "needs_work_payload", None), Mapping) else None,
        (getattr(runtime, "host_closure_payload", None) or {}).get("next_action")
        if isinstance(getattr(runtime, "host_closure_payload", None), Mapping) else None,
    )
    checks: list[str] = []
    for item in _values(assessments):
        checks.extend(_strings(_mapping(item).get("check_ids")))
    check_id = checks[0] if len(set(checks)) == 1 else ""
    item_ids = [*_strings(getattr(decision, "needs_work_ids", ())),
                *[_first(_mapping(item), ("obligation_id", "id")) for item in _values(assessments)]]
    for ordinal, payload in enumerate(payloads):
        row = _mapping(payload)
        if not row:
            continue
        tool = _first(row, ("tool", "name", "action"))
        arguments = row.get("arguments", row.get("parameters", {}))
        if tool and isinstance(arguments, Mapping) and check_id:
            capabilities.append({
                "action_id": _first(row, ("action_id", "id")) or f"host-action:{ordinal}",
                "check_id": _first(row, ("check_id", "check")) or check_id,
                "tool": tool,
                "arguments": dict(arguments),
                "item_ids": [item for item in item_ids if item],
                "executable": True,
                "available": True,
                "expected_result": _first(row, ("expected_result", "expected")),
                "receipt_binding": _text(getattr(runtime, "snapshot_id", "")),
            })
    snapshot_value = getattr(runtime, "snapshot_id", "")
    snapshot_row = _mapping(getattr(runtime, "host_closure_snapshot", None))
    snapshot = (_text(snapshot_value) or
                _first(snapshot_row, ("digest", "snapshot_digest", "content_digest", "revision")))
    revision = snapshot or _text(getattr(getattr(runtime, "host_closure_snapshot", None), "digest", ""))
    all_receipts = [*getattr(runtime, "host_check_receipts", ()),
                    *getattr(runtime, "host_repair_check_receipts", ()),
                    *getattr(runtime, "test_receipts", ()),
                    *getattr(runtime, "probe_receipts", ())]
    if frame is not None:
        runtime._provability_last_frame = frame
    if tools is not None:
        runtime._provability_last_tools = tools
    return classify_frames(sources, capabilities=capabilities, offered_tools=tools,
                           current_revision=revision, receipts=all_receipts)


__all__ = [
    "ADVISORY_END", "ADVISORY_START", "PROVABLE_AFTER_ACTION", "PROVABLE_NOW",
    "SCHEMA", "UNPROVABLE_HERE", "ProvabilityItem", "ProvabilityReport",
    "advisory_for", "classify_frames", "completion_record", "mode", "runtime_report",
]
