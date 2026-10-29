"""Revision-bound, typed findings carried from history into a StateFrame.

History's useful residue is not a transcript-sized explanation.  It is the
settled boundary claim, the exact oracle or failure that supports it, and the
next decision.  This module keeps that residue host-owned and projects only a
small, revision-bound ledger.  It deliberately has no model-facing free-form
reasoning field.
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence


ENV = "STATEFRAME_HISTORY_FINDINGS"
SCHEMA = "stateframe-history-findings/1"
FRAME_FIELD = "history_findings"
MAX_FINDINGS = 8
MAX_EVIDENCE_REFS = 4
MAX_SUPERSEDES = 8
MAX_TEXT = 260
MAX_SCOPE = 120
MAX_PROJECTION_TEXT = 80
MAX_DETAIL_TEXT = MAX_TEXT
# Aggregate serialized budget for the model-facing projection (board item:
# 220-440 typed tokens per finding pair; ~4 chars/token -> 1760 chars total).
MAX_PROJECTION_CHARS = 1760

KINDS = frozenset({
    "root_cause",
    "test_oracle",
    "failure",
    "decision",
    "semantic_contract",
})

_COMMON_FIELDS = frozenset({
    "id", "kind", "claim", "scope", "source_revision", "evidence_refs", "supersedes",
})
_DETAIL_FIELDS = frozenset({
    "oracle", "check", "verdict", "key_assertion", "implication", "invalidates",
    "decision", "semantic_contract",
})


def mode() -> str:
    """Return the explicit opt-in mode for the history-findings projection."""
    value = os.environ.get(ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{ENV} must be off or on")
    return value


def _text(value: object, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"history finding {field} must be a non-empty string")
    value = value.strip()
    return value[:maximum]


def _strings(value: object, *, field: str, maximum: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"history finding {field} must be a list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"history finding {field} must contain non-empty strings")
        item = item.strip()
        if item not in result:
            result.append(item[:maximum])
    return result


def _json_value(value: object, *, field: str, depth: int = 0) -> object:
    """Copy JSON data while bounding model-visible nested detail."""
    if depth > 4:
        raise ValueError(f"history finding {field} is nested too deeply")
    if isinstance(value, str):
        return value[:MAX_DETAIL_TEXT]
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise ValueError(f"history finding {field} must be strict JSON")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in list(value.items())[:12]:
            if not isinstance(key, str):
                raise ValueError(f"history finding {field} has a non-string key")
            result[key[:80]] = _json_value(child, field=field, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(child, field=field, depth=depth + 1) for child in list(value)[:12]]
    raise ValueError(f"history finding {field} must be strict JSON")


def _is_list(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


# Kind-specific collection shapes the projection iterates.  Validated on
# admission AND restore so a malformed record can never reach request building.
_SHAPES: dict[str, dict[str, type | str]] = {
    "oracle": {"cases": "list_of_mappings"},
    "decision": {"unmet": "list_of_strings"},
    "semantic_contract": {"symbols": "mapping", "equations": "list_of_strings",
                          "discriminating_examples": "list_of_strings"},
}


def _check_shape(field: str, value: object) -> None:
    if field not in _SHAPES:
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"history finding {field} must be a mapping")
    for key, shape in _SHAPES[field].items():
        if key not in value:
            continue
        child = value[key]
        if shape == "mapping" and not isinstance(child, Mapping):
            raise ValueError(f"history finding {field}.{key} must be a mapping")
        if shape == "list_of_strings" and not (
                _is_list(child) and all(isinstance(item, str) for item in child)):
            raise ValueError(f"history finding {field}.{key} must be a list of strings")
        if shape == "list_of_mappings" and not (
                _is_list(child) and all(isinstance(item, Mapping) for item in child)):
            raise ValueError(f"history finding {field}.{key} must be a list of mappings")


def _normalise(finding: Mapping[str, object]) -> dict[str, object]:
    if "reasoning" in finding or "rationale" in finding:
        raise ValueError("history findings carry typed claims, not free-form reasoning")
    missing = _COMMON_FIELDS.difference(finding)
    if missing:
        raise ValueError("history finding missing " + ", ".join(sorted(missing)))
    ident = _text(finding["id"], field="id", maximum=100)
    kind = _text(finding["kind"], field="kind", maximum=40)
    if kind not in KINDS:
        raise ValueError(f"history finding kind must be one of {sorted(KINDS)}")
    source_revision = _text(finding["source_revision"], field="source_revision", maximum=180)
    evidence_refs = _strings(
        finding["evidence_refs"], field="evidence_refs", maximum=180)[:MAX_EVIDENCE_REFS]
    if not evidence_refs:
        raise ValueError("history finding evidence_refs must not be empty")
    row: dict[str, object] = {
        "id": ident,
        "kind": kind,
        "claim": _text(finding["claim"], field="claim", maximum=MAX_TEXT),
        "scope": _text(finding["scope"], field="scope", maximum=MAX_SCOPE),
        "source_revision": source_revision,
        "evidence_refs": evidence_refs,
        "supersedes": _strings(
            finding.get("supersedes", ()), field="supersedes", maximum=100)[:MAX_SUPERSEDES],
    }
    if finding.get("product_binding") is not None:
        # r44 F4 item 8: a test-derived finding is bound to the test receipt's FULL
        # product/dependency binding, not only the target-source sha.  Optional,
        # so target-local findings keep their target-only binding.
        row["product_binding"] = _text(
            finding["product_binding"], field="product_binding", maximum=180)
    for field in sorted(_DETAIL_FIELDS.intersection(finding)):
        value = finding[field]
        if field in {"check", "verdict", "key_assertion", "implication"}:
            row[field] = _text(value, field=field, maximum=MAX_TEXT)
        elif field == "invalidates":
            row[field] = _strings(value, field=field, maximum=MAX_TEXT)[:6]
        else:
            _check_shape(field, value)
            row[field] = _json_value(value, field=field)
    return row


def _copy(value: object) -> object:
    return copy.deepcopy(value)


def _projection_value(value: object, *, depth: int = 0) -> object:
    """Keep nested oracle/decision detail useful without replaying a transcript."""
    if isinstance(value, str):
        return value[:MAX_PROJECTION_TEXT]
    if depth >= 2:
        return _copy(value) if not isinstance(value, (Mapping, Sequence)) else str(value)[:MAX_PROJECTION_TEXT]
    if isinstance(value, Mapping):
        return {
            str(key)[:60]: _projection_value(child, depth=depth + 1)
            for key, child in list(value.items())[:6]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_projection_value(child, depth=depth + 1) for child in list(value)[:3]]
    return _copy(value)


def _projection_detail(field: str, value: object) -> object:
    """Project the decisive part of each typed detail, not its raw source."""
    if field == "oracle" and isinstance(value, Mapping):
        cases = value.get("cases", ())
        case = cases[0] if isinstance(cases, Sequence) and cases else {}
        case = case if isinstance(case, Mapping) else {}
        return {
            "selector": str(value.get("selector", ""))[:MAX_PROJECTION_TEXT],
            "case": str(case.get("inputs", ""))[:MAX_PROJECTION_TEXT],
            "expected": _projection_value(case.get("expected")),
            "negative_control": str(case.get("negative_control", ""))[:MAX_PROJECTION_TEXT],
        }
    if field == "decision" and isinstance(value, Mapping):
        return {
            "phase": str(value.get("phase", ""))[:60],
            "unmet": [str(item)[:MAX_PROJECTION_TEXT]
                      for item in list(value.get("unmet", ()))[:3]],
            "next_action": str(value.get("next_action", ""))[:MAX_PROJECTION_TEXT],
            "basis": str(value.get("completion_basis", ""))[:MAX_PROJECTION_TEXT],
        }
    if field == "semantic_contract" and isinstance(value, Mapping):
        return {
            "symbols": _projection_value(value.get("symbols", {})),
            "equations": [str(item)[:MAX_PROJECTION_TEXT]
                          for item in list(value.get("equations", ()))[:2]],
            "examples": [str(item)[:MAX_PROJECTION_TEXT]
                         for item in list(value.get("discriminating_examples", ()))[:2]],
        }
    return _projection_value(value)


def _active_records(records: Mapping[str, Mapping[str, object]]) -> list[Mapping[str, object]]:
    superseded = {
        str(target)
        for row in records.values()
        for target in row.get("supersedes", ())
        if isinstance(target, str)
    }
    return [row for ident, row in records.items() if ident not in superseded]


class HistoryFindingsStore:
    """Host-owned store for compact, evidence-backed historical findings."""

    def __init__(self, state: Mapping[str, object] | None = None):
        self._records: dict[str, dict[str, object]] = {}
        self._order: list[str] = []
        self.restore_rejected = 0
        if state is not None:
            self._restore(state)

    @classmethod
    def from_state(cls, state: Mapping[str, object] | None) -> "HistoryFindingsStore":
        return cls(state if isinstance(state, Mapping) else None)

    def _restore(self, state: Mapping[str, object]) -> None:
        raw = state.get("records", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise ValueError("history findings state records must be a list")
        for item in raw:
            if not isinstance(item, Mapping):
                self.restore_rejected += 1
                continue
            try:
                row = _normalise(item)
            except ValueError:
                # A malformed persisted record is dropped, never replayed into
                # every later request (review finding 3).
                self.restore_rejected += 1
                continue
            ident = str(row["id"])
            if ident not in self._records:
                self._order.append(ident)
            self._records[ident] = row

    def record(self, finding: Mapping[str, object], *, current_revision: str,
               available_refs: Sequence[str] | set[str]) -> dict[str, object]:
        """Admit one settled finding only when its evidence is present now."""
        if not isinstance(finding, Mapping):
            raise ValueError("history finding must be a mapping")
        row = _normalise(finding)
        current_revision = _text(current_revision, field="current_revision", maximum=180)
        refs = {str(item) for item in available_refs}
        missing = [ref for ref in row["evidence_refs"] if ref not in refs]
        if missing:
            raise ValueError("history finding evidence refs unavailable: " + ", ".join(missing))
        ident = str(row["id"])
        if ident in self._records:
            if self._records[ident] == row:
                return {"accepted": False, "duplicate": True, "id": ident, "schema": SCHEMA}
            raise ValueError(f"history finding id already exists: {ident}")
        if row["source_revision"] != current_revision:
            raise ValueError("history finding source_revision must match the current revision")
        self._records[ident] = row
        self._order.append(ident)
        return {"accepted": True, "duplicate": False, "id": ident, "schema": SCHEMA}

    def has_product_bindings(self) -> bool:
        return any(row.get("product_binding") is not None for row in self._records.values())

    def snapshot(self) -> dict[str, object]:
        return {
            "payload_schema": SCHEMA,
            "records": [_copy(self._records[ident]) for ident in self._order
                         if ident in self._records],
        }

    def project(self, *, current_revision: str,
                available_refs: Sequence[str] | set[str] | None = None,
                max_chars: int = MAX_PROJECTION_CHARS,
                current_product_binding: str | None = None) -> dict[str, object]:
        """Return the compact frame payload, demoting stale evidence explicitly.

        ``current_product_binding`` (r44 F4 item 8): when given, a finding that
        carries a ``product_binding`` is SETTLED only if it matches; an edit to a
        dependency (target sha unchanged) therefore demotes a test-derived pass to
        STALE, consistent with work_carry's demotion and the vacuity sentinel.
        """
        current_revision = _text(current_revision, field="current_revision", maximum=180)
        known_refs = None if available_refs is None else {str(item) for item in available_refs}
        rows: list[dict[str, object]] = []
        active = _active_records(self._records)[-MAX_FINDINGS:]
        scopes = {str(row.get("scope", "")) for row in active}
        shared_scope = next(iter(scopes)) if len(scopes) == 1 and scopes else None
        for raw in active:
            status = "SETTLED"
            if str(raw.get("source_revision")) != current_revision:
                status = "STALE"
            elif (current_product_binding is not None
                  and raw.get("product_binding") is not None
                  and str(raw.get("product_binding")) != str(current_product_binding)):
                status = "STALE"
            elif known_refs is not None and not set(raw.get("evidence_refs", ())).issubset(known_refs):
                status = "UNBOUND"
            row: dict[str, object] = {}
            for key, value in raw.items():
                if key == "supersedes" and not value:
                    continue
                if key == "product_binding":
                    continue  # host binding identity, not model-facing content
                if key == "source_revision" and status == "SETTLED":
                    # The enclosing projection revision is the binding for a
                    # current finding; stale rows retain their old revision.
                    continue
                if key == "claim":
                    row[key] = str(value)[:MAX_PROJECTION_TEXT]
                elif key == "scope" and shared_scope is not None:
                    continue
                elif key == "scope":
                    row[key] = str(value)[:80]
                elif key in {"id", "kind", "source_revision", "evidence_refs", "supersedes"}:
                    row[key] = _copy(value)
                else:
                    try:
                        row[key] = _projection_detail(key, value)
                    except (TypeError, ValueError, AttributeError):
                        row[key] = {"detail_unrenderable": True}
            row["status"] = status
            if status != "SETTLED":
                row["revalidate"] = True
            rows.append(row)
        projection = {
            "schema": SCHEMA,
            "revision": current_revision,
            "findings": rows,
        }
        if shared_scope is not None:
            projection["scope"] = shared_scope[:80]
        return _fit_budget(projection, max_chars=max_chars)


def _serialized_len(value: object) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


_CORE_KEYS = frozenset({"id", "kind", "claim", "status", "revalidate", "evidence_refs",
                        "source_revision", "scope"})


def _fit_budget(projection: dict[str, object], *, max_chars: int) -> dict[str, object]:
    """Enforce the aggregate budget: shed detail oldest-first, then whole findings."""
    if _serialized_len(projection) <= max_chars:
        return projection
    rows: list[dict[str, object]] = projection["findings"]  # type: ignore[assignment]
    projection["truncated"] = True
    for row in rows:  # oldest first
        detail = sorted(set(row) - _CORE_KEYS - {"supersedes"})
        if detail:
            for key in detail:
                row.pop(key)
            row["detail_omitted"] = detail
        if _serialized_len(projection) <= max_chars:
            return projection
    omitted = 0
    while rows and _serialized_len(projection) > max_chars:
        rows.pop(0)
        omitted += 1
        projection["omitted_findings"] = omitted
    return projection


def finding_from_test_observation(*, receipt_id: str, selectors: Sequence[object],
                                  exit_code: int, source_revision: str,
                                  supersedes: Sequence[str] = (),
                                  product_binding: str | None = None) -> dict[str, object]:
    """Production producer: one host test observation -> one typed finding.

    The host can only assert what it measured: which public check ran on which
    source revision and how it exited.  A non-zero exit is a ``failure``
    (history's "prior failing assertion"); exit 0 is a ``test_oracle`` that
    the check held on that revision.  No model text enters the record.
    """
    names = [str(item) for item in selectors if str(item).strip()][:4] or ["<public checks>"]
    scope = ",".join(names)
    passed = exit_code == 0
    extra = {"product_binding": str(product_binding)} if product_binding else {}
    return {
        **extra,
        "id": "check:" + str(receipt_id)[:80],
        "kind": "test_oracle" if passed else "failure",
        "claim": (f"public check {scope} {'passed' if passed else 'failed'} "
                  f"(exit {exit_code}) on this source revision"),
        "scope": scope,
        "source_revision": source_revision,
        "evidence_refs": [str(receipt_id)],
        "supersedes": [str(item) for item in supersedes][:MAX_SUPERSEDES],
        "check": scope,
        "verdict": "PASSED" if passed else "FAILED",
    }


def superseded_by_scope(store: HistoryFindingsStore, scope: str) -> list[str]:
    """Active host check findings with the same scope, replaced by a newer run."""
    return [ident for ident, row in store._records.items()
            if ident.startswith("check:") and row.get("scope") == scope[:MAX_SCOPE]
            and ident in {str(r["id"]) for r in _active_records(store._records)}]


def project_frame(frame: dict, *, current_revision: str,
                  store: HistoryFindingsStore | None = None,
                  available_refs: Sequence[str] | set[str] | None = None) -> dict:
    """Attach the field only in the opt-in arm; OFF returns the same object."""
    if mode() == "off":
        return frame
    if store is None:
        return frame
    projected = dict(frame)
    projected[FRAME_FIELD] = store.project(
        current_revision=current_revision, available_refs=available_refs)
    return projected


def validate_projection(payload: Mapping[str, object]) -> None:
    """Keep the shared producer/consumer field contract auditable in tests."""
    if payload.get("schema") != SCHEMA:
        raise ValueError("history findings projection schema mismatch")
    if not isinstance(payload.get("findings"), list):
        raise ValueError("history findings projection findings must be a list")
    json.dumps(payload, sort_keys=True, allow_nan=False)
