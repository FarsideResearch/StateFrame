"""Bounded refusal and evidence panels at the actual provider request boundary."""
from __future__ import annotations

import json
import hashlib
import os
import re
from collections.abc import Mapping, Sequence

MODE_ENV = "STATEFRAME_REFUSAL_COMPACT"
START = "[STATEFRAME compact refusal]"
END = "[/STATEFRAME compact refusal]"
MAX_BYTES = 1200
EVIDENCE_START = "[EVIDENCEPIN]"
EVIDENCE_END = "[/EVIDENCEPIN]"
EVIDENCE_MAX_BYTES = 5600
_FAILURE_STATUSES = frozenset({
    "BLOCKED", "COUNTEREXAMPLE", "ERROR", "FAIL", "FAILED", "NEEDS_WORK",
    "RED", "REJECTED", "REFUSED", "UNSATISFIED", "VIOLATED",
})
_PASS_STATUSES = frozenset({
    "CERTIFIED", "CURRENT", "GREEN", "GREEN_CURRENT", "OK", "PASS", "PASSED",
    "REUSED", "SATISFIED", "SUCCESS", "VERIFIED",
})
_CHECK_FIELDS = (
    "host_check_receipts", "host_repair_check_receipts", "test_receipts",
    "recent_test_feedback", "probe_receipts", "receipts",
)


def enabled() -> bool:
    return mode() == "on"


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def _text(value: object, limit: int = 240) -> str:
    return " ".join(str(value or "").split())[:limit]


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
        return result if isinstance(result, Mapping) else None
    return None


def _rows(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        if any(key in value for key in (
                "status", "result", "result_class", "kind", "action", "observation",
                "reason", "exact_clause", "clause_text")):
            return (value,)
        return tuple(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _json(value: object, limit: int = 420) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        rendered = str(value)
    return _clip(rendered, limit)


def _clip(value: object, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:max(0, limit - 24)] + " [... bounded excerpt]"


def _number(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _revision(value: object) -> str:
    row = _as_mapping(value)
    if row is not None:
        for key in (
                "revision", "product_revision", "current_revision", "binding_revision", "artifact",
                "artifact_revision", "source_revision", "snapshot_digest"):
            item = row.get(key)
            if item not in (None, ""):
                return str(item)
        for key in ("binding", "after_binding", "current_binding", "revision_binding_state"):
            nested = _revision(row.get(key))
            if nested != "UNKNOWN":
                return nested
    return "UNKNOWN"


def _current_revision(runtime: object, frame: Mapping[str, object]) -> str:
    # Public check receipts are product-wide.  A target file hash is useful
    # evidence, but comparing it with a product binding creates a false stale
    # result and can steer the owner into an unnecessary rerun/refusal.
    frame_product = {
        key: frame[key]
        for key in ("product_revision", "current_product_revision", "binding_revision")
        if frame.get(key) not in (None, "")
    }
    for source in (frame.get("product_binding"), frame_product):
        revision = _revision(source)
        if revision != "UNKNOWN":
            return revision
    try:
        from kernel_check_receipts import _current_revision as receipt_revision
        current = receipt_revision(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        current = "UNKNOWN"
    if current != "UNKNOWN":
        return current
    for source in (
            frame.get("product_binding"), frame.get("binding"), frame,
            getattr(runtime, "prepared_binding", None), getattr(runtime, "binding", None)):
        if callable(source):
            try:
                source = source()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                source = None
        revision = _revision(source)
        if revision != "UNKNOWN":
            return revision
    return "UNKNOWN"


def _status(value: object) -> str:
    row = _as_mapping(value)
    if row is not None:
        for key in ("status", "result_class", "result", "verdict", "outcome", "state"):
            if row.get(key) not in (None, ""):
                return _status(row[key])
        return "UNKNOWN"
    text = str(value or "").strip().upper().replace("-", "_")
    if text in _FAILURE_STATUSES:
        return text
    if text in _PASS_STATUSES:
        return text
    return text or "UNKNOWN"


def _event_status(row: Mapping[str, object]) -> str:
    """Use the same public result markers as ctx_audit for journal events."""
    action = _as_mapping(row.get("action"))
    candidates = (
        row.get("status"), row.get("result_class"), row.get("result"),
        row.get("verdict"), row.get("outcome"),
        row.get("check_status"), row.get("observation_certificate"),
        action.get("result_kind") if action is not None else None,
        action.get("status") if action is not None else None,
        action.get("result") if action is not None else None,
        action.get("outcome") if action is not None else None,
        action.get("check_status") if action is not None else None,
        row.get("kind"),
    )
    for candidate in candidates:
        status = _receipt_status(candidate)
        if status in _FAILURE_STATUSES or status in _PASS_STATUSES:
            return status
    exit_code = row.get("exit_code", row.get("returncode"))
    if type(exit_code) is int:
        return "PASS" if exit_code == 0 else "FAIL"
    combined = " ".join(str(value or "") for value in candidates).upper()
    for marker in (*_FAILURE_STATUSES, *_PASS_STATUSES):
        if marker in combined.replace("-", "_"):
            return marker
    return _status(row)


def _binding_fields(row: Mapping[str, object]) -> dict[str, object]:
    """Extract explicit action identity without guessing from prose."""
    action = _as_mapping(row.get("action"))
    arguments = _as_mapping(_action_arguments(row))
    nested = (action, arguments)

    def first(keys: Sequence[str]) -> str:
        for source in (row, *nested):
            if source is None:
                continue
            for key in keys:
                value = source.get(key)
                if value not in (None, "") and not isinstance(value, Mapping):
                    return str(value)
        return ""

    return {
        "path": first(("path", "target_path", "file")),
        "operation": first(("operation", "action_kind", "tool", "verb")),
        "failed_condition": first(("failed_condition", "failure_condition", "condition")),
        "revision": _revision(row),
        "receipt_id": first(("receipt_id", "check_receipt_id", "action_receipt_id", "full_output_handle")),
        "producer_kind": first(("kind", "event_kind", "producer")),
    }


def _canonical_check_rows(frame: Mapping[str, object]) -> list[dict[str, object]]:
    """Read the existing progress/truth projection, never raw step history."""
    try:
        from kernel_progress_ledger import project_progress
        envelope = frame.get("progress_inputs", frame)
        view = project_progress(envelope)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return []
    result: list[dict[str, object]] = []
    for item in getattr(view, "verifications", ()):
        row = item.mapping() if callable(getattr(item, "mapping", None)) else {}
        state = str(row.get("state", "UNKNOWN"))
        if state == "NO_CHECK_RECORDED":
            continue
        outcome = str(row.get("outcome", "UNKNOWN"))
        status = {
            "CHECK_PASS": "PASS",
            "CHECK_FAIL": "FAIL",
            "CHECK_NOT_REACHED": "UNEXECUTED",
            "STALE": "STALE",
            "BINDING_UNKNOWN": "UNKNOWN",
        }.get(state, outcome or "UNKNOWN")
        result.append({
            "status": status,
            "text": (
                f"state={state}; outcome={outcome}; "
                f"phase={row.get('failure_phase', 'UNKNOWN')}"
            ),
            "revision": row.get("execution_revision") or "UNKNOWN",
            "ordinal": row.get("sequence", -1),
            "source": "progress_ledger",
            "path": row.get("path", ""),
            "operation": row.get("operation", ""),
            "failed_condition": row.get("failed_condition", ""),
            "receipt_id": row.get("execution_identity", ""),
            "producer_kind": row.get("provenance", "HOST_EXECUTION"),
        })
    return result


def _receipt_status(value: object) -> str:
    """Normalize a runner receipt without serializing opaque mappings."""
    row = _as_mapping(value)
    if row is None:
        return _status(value)
    try:
        from kernel_host_done_facts import _status as receipt_status
        normalized = receipt_status(row)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        normalized = "UNKNOWN"
    return normalized if normalized != "UNKNOWN" else _status(row)


def _public_check_text(text: str) -> str:
    """Model-visible check text: stored receipt summaries are the lossless private copy, so
    durations and object addresses are normalized here (DETFIX4b). Workspace aliasing needs the
    journal and is already applied where receipts are projected."""
    from kernel_wire_identity import public_wire_text
    return public_wire_text(text, None)


def _check_text(row: Mapping[str, object]) -> str:
    action = _as_mapping(row.get("action"))
    if action is not None:
        observation = action.get("observation")
        if isinstance(observation, str) and observation.strip():
            return observation
    for key in ("summary", "text", "feedback", "observation", "output", "message", "detail", "reason"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("counterexample", "observed", "expected", "observed_values"):
        if row.get(key) not in (None, "", {}, []):
            return _json({key: row[key]}, 1800)
    return _status(row)


def _check_rows(
        runtime: object, frame: Mapping[str, object], *,
        include_framefit_journal: bool = False,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = _canonical_check_rows(frame)
    sources: list[tuple[str, object]] = []
    for key in _CHECK_FIELDS:
        sources.append((key, frame.get(key)))
        sources.append((key, getattr(runtime, key, None)))
    contract = _as_mapping(frame.get("completion_contract"))
    if contract is not None:
        sources.append(("current_public_checks", contract.get("current_public_checks")))
    journal = getattr(runtime, "journal", None)
    records = getattr(journal, "records", None)
    if callable(records):
        try:
            sources.append(("journal", records()))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
    step_ledger = getattr(runtime, "step_ledger", None)
    step_records = getattr(step_ledger, "records", None)
    if isinstance(step_records, Sequence) and not isinstance(step_records, (str, bytes, bytearray)):
        sources.append(("step_ledger", step_records))
    if include_framefit_journal:
        action_context = getattr(runtime, "_framefit_action_context", None)
        try:
            from kernel_stateframe_frame_continuity import journal_check_receipts
            pinned_rows = journal_check_receipts(action_context)
        except (ImportError, AttributeError, TypeError, ValueError):
            pinned_rows = ()
        if pinned_rows:
            sources.append(("framefit5_pinned_journal", pinned_rows))
    for source, values in sources:
        for index, raw in enumerate(_rows(values)):
            row = _as_mapping(raw)
            if row is None:
                continue
            action = _as_mapping(row.get("action"))
            kind = str(row.get("kind", "")).lower()
            action_kind = str(action.get("action_kind", "")).lower() if action else ""
            checkish = source in _CHECK_FIELDS or source in {"current_public_checks"} \
                or kind in {"test_subprocess_completed", "host_repair_check", "check_receipt"} \
                or action_kind in {"run_tests", "gate", "check"} \
                or str(row.get("tool", "")).lower() in {"run_tests", "gate", "check", "pytest", "test"}
            if not checkish:
                continue
            status = _event_status(row)
            normalized = {
                "status": status,
                "text": _public_check_text(_check_text(row)),
                "revision": _revision(row),
                "ordinal": max(_number(row.get("sequence")), _number(row.get("step")),
                                _number(row.get("turn")), index),
                "source": source,
            }
            normalized.update(_binding_fields(row))
            for key in (
                    "full_output_handle", "receipt_id", "stdout_bytes", "stderr_bytes",
                    "stdout_omitted_bytes", "stderr_omitted_bytes",
                    "assertion_context_omitted_bytes"):
                if row.get(key) not in (None, ""):
                    normalized[key] = row[key]
            for key in ("execution_id", "host_execution_id"):
                if row.get(key) not in (None, ""):
                    normalized["execution_id"] = str(row[key])
                    break
            result.append(normalized)
    return _collapse_to_canonical_receipts(runtime, result)


def _collapse_to_canonical_receipts(
        runtime: object, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Every view of one execution reports the ONE canonical receipt.

    Wrapper, journal, and test-receipt views of an execution used to be
    re-decoded independently (b857: PASS/46 -> UNKNOWN, FAIL/26 -> UNKNOWN), and
    the highest-ordinal view won.  A view that names a canonical receipt (by
    execution id or receipt id) now carries that receipt's status and text, and
    duplicate views collapse to one row at the latest ordinal.
    """
    store = getattr(runtime, "_check_output_receipts", None)
    if not isinstance(store, Mapping) or not store:
        return rows
    by_receipt: dict[str, Mapping[str, object]] = {}
    by_execution: dict[str, Mapping[str, object]] = {}
    for handle, private in store.items():
        if not isinstance(private, Mapping):
            continue
        by_receipt[str(handle)] = private
        execution = str(private.get("execution_id") or "")
        if execution:
            by_execution[execution] = private
    collapsed: dict[str, dict[str, object]] = {}
    passthrough: list[dict[str, object]] = []
    for row in rows:
        canonical = by_execution.get(str(row.get("execution_id") or "")) \
            or by_receipt.get(str(row.get("receipt_id") or "")) \
            or by_receipt.get(str(row.get("full_output_handle") or ""))
        if canonical is None:
            passthrough.append(row)
            continue
        handle = str(canonical.get("receipt_id"))
        merged = dict(row)
        merged.update({
            "status": str(canonical.get("status") or "UNKNOWN").upper(),
            "text": _public_check_text(str(canonical.get("summary") or row.get("text") or "")),
            "revision": canonical.get("revision") or row.get("revision"),
            "receipt_id": handle,
            "full_output_handle": canonical.get("full_output_handle") or handle,
            "execution_id": str(canonical.get("execution_id") or ""),
            "selector_scope": list(canonical.get("selector_scope") or ()),
        })
        for key in ("stdout_bytes", "stderr_bytes", "stdout_omitted_bytes",
                    "stderr_omitted_bytes", "assertion_context_omitted_bytes"):
            if canonical.get(key) is not None:
                merged[key] = canonical[key]
        previous = collapsed.get(handle)
        if previous is None or int(merged.get("ordinal", -1)) > int(previous.get("ordinal", -1)):
            if previous is not None:
                merged["ordinal"] = max(int(merged.get("ordinal", -1)), int(previous.get("ordinal", -1)))
            collapsed[handle] = merged
    return passthrough + list(collapsed.values())


def _action_arguments(row: Mapping[str, object]) -> object:
    for key in ("rejected_arguments", "action_parameters", "canonical_arguments", "parameters", "arguments"):
        if row.get(key) not in (None, ""):
            return row[key]
    action = _as_mapping(row.get("action"))
    if action is not None:
        return _action_arguments(action)
    return {}


def _refusal_rows(runtime: object, frame: Mapping[str, object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    sources: list[object] = []
    for key in ("recent_negative_actions", "previous_actions", "open_work", "action_receipts", "step_records"):
        sources.extend(_rows(frame.get(key)))
        sources.extend(_rows(getattr(runtime, key, None)))
    records = getattr(getattr(runtime, "journal", None), "records", None)
    if callable(records):
        try:
            sources.extend(_rows(records()))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
    step_ledger = getattr(runtime, "step_ledger", None)
    step_records = getattr(step_ledger, "records", None)
    if isinstance(step_records, Sequence) and not isinstance(step_records, (str, bytes, bytearray)):
        sources.extend(step_records)
    decision = _as_mapping(getattr(runtime, "_last_completion_decision", None))
    if decision is not None and decision.get("accepted") is not True:
        decision_row = {
            "status": "REFUSED",
            "reason": decision.get("detail") or decision.get("reason_codes") or "completion decision refused",
            "arguments": _action_arguments(decision),
            "ordinal": 10**9,
        }
        decision_row.update(_binding_fields(decision))
        result.append(decision_row)
    rejected = _as_mapping(getattr(runtime, "rejected_candidate", None))
    if rejected is not None:
        rejected_row = {
            "status": "REJECTED",
            "reason": rejected.get("reason") or rejected.get("detail") or "candidate rejected",
            "arguments": _action_arguments(rejected),
            "ordinal": 10**9,
        }
        rejected_row.update(_binding_fields(rejected))
        result.append(rejected_row)
    for index, raw in enumerate(sources):
        row = _as_mapping(raw)
        if row is None:
            continue
        action = _as_mapping(row.get("action"))
        combined = " ".join(str(row.get(key, "")) for key in ("status", "result", "outcome", "reason", "detail", "observation"))
        if action is not None:
            combined += " " + " ".join(str(action.get(key, "")) for key in ("status", "result", "outcome", "reason", "detail", "observation"))
        if not any(marker in combined.upper() for marker in ("REFUSED", "REJECTED", "NO_OP", "NO EXECUTION RECEIPT", "TOOL_REFUSED")):
            continue
        normalized = {
            "status": _status(row),
            "reason": _public_check_text(str(
                (action.get("observation") if action is not None else None)
                or row.get("observation") or row.get("reason") or row.get("detail")
                or row.get("message") or combined)),
            "arguments": _action_arguments(row),
            "ordinal": max(_number(row.get("sequence")), _number(row.get("step")), index),
        }
        normalized.update(_binding_fields(row))
        result.append(normalized)
    return result


def _action_history(runtime: object, frame: Mapping[str, object]) -> list[dict[str, object]]:
    """Collect action outcomes for an explicit refusal-resolution join."""
    values: list[object] = []
    for key in (
            "recent_changes", "applied_actions", "action_receipts", "previous_actions",
            "step_records"):
        values.extend(_rows(frame.get(key)))
        values.extend(_rows(getattr(runtime, key, None)))
    records = getattr(getattr(runtime, "journal", None), "records", None)
    if callable(records):
        try:
            values.extend(_rows(records()))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
    result: list[dict[str, object]] = []
    for index, raw in enumerate(values):
        row = _as_mapping(raw)
        if row is None:
            continue
        status = _event_status(row)
        normalized = {
            "status": status,
            "ordinal": max(_number(row.get("sequence")), _number(row.get("step")), index),
        }
        normalized.update(_binding_fields(row))
        normalized["reason"] = _check_text(row)
        result.append(normalized)
    return result


def _linked_success(
        refusal: Mapping[str, object], candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Resolve only a later success with all three explicit action bindings."""
    required = ("path", "operation", "failed_condition")
    refusal_binding = tuple(str(refusal.get(key, "")).strip() for key in required)
    if not all(refusal_binding):
        return None
    refusal_ordinal = _number(refusal.get("ordinal"))
    matches: list[Mapping[str, object]] = []
    for candidate in candidates:
        candidate_binding = tuple(str(candidate.get(key, "")).strip() for key in required)
        if candidate_binding != refusal_binding:
            continue
        if _number(candidate.get("ordinal")) <= refusal_ordinal:
            continue
        status = _status(candidate)
        if status in _PASS_STATUSES or status in {"APPLIED", "ACCEPTED", "CURRENT"}:
            matches.append(candidate)
    return max(matches, key=lambda row: _number(row.get("ordinal")), default=None)


def _missing_evidence_category(row: Mapping[str, object], assessment: Mapping[str, object] | None) -> str:
    for source in (row, assessment or {}):
        for key in ("missing_evidence_category", "missing_category", "open_reason"):
            value = source.get(key)
            if value not in (None, ""):
                return str(value).upper().replace("-", "_")
    if row.get("definition_missing") is True or row.get("defined") is False:
        return "MISSING_DEFINITION"
    if row.get("check_executed") is False or row.get("executed") is False:
        return "UNEXECUTED_CHECK"
    if row.get("receipt_available") is False or row.get("receipt_id") in (None, "") \
            and row.get("check_required") is True:
        return "UNAVAILABLE_RECEIPT"
    if row.get("binding_known") is False or row.get("binding_state") in {"UNKNOWN", "BINDING_UNKNOWN"}:
        return "UNKNOWN_BINDING"
    return "UNKNOWN"


def _observed_value(row: Mapping[str, object], assessment: Mapping[str, object] | None) -> str:
    for source in (assessment or {}, row):
        for key in ("observed_status", "observed", "observation", "result", "outcome"):
            value = source.get(key)
            if value not in (None, "", {}, []):
                return _clip(_json(value, 300), 300)
    return "UNKNOWN"


def _requirement_rows(runtime: object, frame: Mapping[str, object]) -> tuple[str, list[dict[str, str]]]:
    ledger = _as_mapping(getattr(runtime, "requirements_ledger", None))
    if ledger is None:
        ledger = _as_mapping(frame.get("requirements_ledger"))
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    if ledger is not None:
        for raw in _rows(ledger.get("requirements")):
            row = _as_mapping(raw)
            if row is None:
                continue
            clause = row.get("exact_clause") or row.get("clause_text") or row.get("clause") or row.get("scope")
            if not isinstance(clause, str) or not clause.strip():
                continue
            assessment = row.get("assessment")
            assessment_map = assessment if isinstance(assessment, Mapping) else None
            requirement_id = str(row.get("id", f"R{len(rows) + 1}"))
            source = _json(row.get("source_span_ids", row.get("source", "ticket")), 160)
            dedupe_key = (requirement_id, clause, source)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            status = row.get("status", row.get("state", "UNKNOWN"))
            if assessment_map is not None:
                status = assessment_map.get("status", status)
            rows.append({
                "id": requirement_id,
                "clause": clause,
                "source": source,
                "reference": _json(row.get("reference", row.get("source_span", source)), 200),
                "status": str(status or "UNKNOWN").upper(),
                "observed": _observed_value(row, assessment_map),
                "missing_category": _missing_evidence_category(row, assessment_map),
            })
    if not rows:
        try:
            from kernel_clause_review import open_clauses
            for raw in open_clauses(runtime):
                row = _as_mapping(raw)
                if row is None:
                    continue
                clause = row.get("clause") or row.get("exact_clause") or row.get("clause_text")
                if isinstance(clause, str) and clause.strip():
                    requirement_id = str(row.get("clause_id", f"R{len(rows) + 1}"))
                    source = "clause_review"
                    dedupe_key = (requirement_id, clause, source)
                    if dedupe_key not in seen:
                        seen.add(dedupe_key)
                        rows.append({
                            "id": requirement_id,
                            "clause": clause,
                            "source": source,
                            "reference": source,
                            "status": str(row.get("status", "UNKNOWN")).upper(),
                            "observed": _observed_value(row, None),
                            "missing_category": _missing_evidence_category(row, None),
                        })
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
    task_row = _as_mapping(getattr(getattr(runtime, "owner", None), "task_row", None)) or {}
    brief = next((task_row.get(key) for key in ("msg", "message", "brief", "requirements")
                  if isinstance(task_row.get(key), str) and task_row.get(key).strip()), "")
    if not rows and isinstance(brief, str) and brief.strip():
        clauses = [part.strip() for part in re.split(r"(?<=[.!?])\s+|;\s+|\n+", brief) if part.strip()]
        rows = [{"id": f"R{index}", "clause": clause, "source": "task_row.msg",
                 "reference": "task_row.msg", "status": "UNKNOWN", "observed": "UNKNOWN",
                 "missing_category": "UNKNOWN"}
                for index, clause in enumerate(clauses[:5], 1)]
    # The old five-row slice was unsafe: it removed active negative or
    # compatibility clauses before the byte bound was even considered.  Keep
    # the complete active set here; render_evidence_pin decides what fits and
    # records every omitted ID plus its retrieval route when it cannot fit.
    return str(brief or ""), rows


def _requirements_retrieval_handle(
        runtime: object, frame: Mapping[str, object], rows: Sequence[Mapping[str, str]]) -> str:
    """Return the host ticket handle used by the admitted read_ticket route."""
    ledger = _as_mapping(getattr(runtime, "requirements_ledger", None))
    if ledger is None:
        ledger = _as_mapping(frame.get("requirements_ledger"))
    candidates = (ledger, frame, getattr(runtime, "owner", None))
    for source in candidates:
        row = _as_mapping(source)
        if row is None:
            continue
        for key in (
                "retrieval_handle", "requirements_handle", "ticket_handle",
                "source_id", "ticket_source_id", "requirements_source_id"):
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
    revision = _current_revision(runtime, frame)
    ids = "|".join(str(row.get("id", "")) for row in rows)
    digest = hashlib.sha256(f"{revision}|{ids}".encode("utf-8")).hexdigest()[:20]
    return f"requirements:{digest}"


def _pin_bytes(lines: Sequence[str]) -> int:
    return len("\n".join(lines).encode("utf-8"))


def _fit_pin_core(lines: list[str], *, end: str) -> list[str]:
    """Bound explanatory fields without ever cutting a normative marker."""
    limits = (900, 600, 360, 220, 120)
    variable_prefixes = (
        "check_output=", "historical_failure=", "prior_refusal=",
        "rejected_arguments=", "requirements_source=",
    )
    for limit in limits:
        if _pin_bytes([*lines, end]) <= EVIDENCE_MAX_BYTES:
            return lines
        for index, line in enumerate(lines):
            prefix = next((item for item in variable_prefixes if line.startswith(item)), None)
            if prefix is not None:
                lines[index] = prefix + _clip(line[len(prefix):], limit)
    return lines


def _menu_names(tools: object) -> tuple[str, ...]:
    return _tool_names(tools)


def _witness_pin_row(runtime: object, tools: object) -> str | None:
    """Keep the bounded witness episode and receipts inside the owner pin."""
    if "probe" not in _tool_names(tools):
        return None
    try:
        from kernel_clause_review import state as witness_state
        state = witness_state(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return "schema=stateframe-clause-witness-contract/1; state=UNAVAILABLE"
    if not isinstance(state, Mapping):
        return "state=UNKNOWN"
    state_name = _clip(state.get("state", "UNKNOWN"), 80)
    receipt_ids = ",".join(
        str(value) for value in _rows(state.get("receipt_ids")) if str(value).strip())
    clause_ids = ",".join(
        str(value) for value in _rows(state.get("clause_ids")) if str(value).strip())
    gaps = ",".join(
        str(value) for value in _rows(state.get("remaining_gaps")) if str(value).strip())
    active = state_name not in {"IDLE", "UNAVAILABLE", "UNKNOWN"} \
        or bool(clause_ids or receipt_ids or gaps or state.get("episode_id"))
    if state_name == "IDLE" and not active:
        return None
    fields = ["state=" + state_name]
    for name, value, limit in (
            ("episode_id", state.get("episode_id"), 160),
            ("goal_revision", state.get("goal_revision"), 160),
            ("clause_ids", clause_ids, 400),
            ("receipt_ids", receipt_ids, 800),
            ("remaining_gaps", gaps, 800)):
        if value not in (None, "", "none"):
            fields.append(name + "=" + _clip(value, limit))
    if active:
        fields.append(
            "budget=round:" + str(state.get("round", 0))[:40]
            + ",decisions:" + str(state.get("used_decisions", 0))[:40]
            + ",cases:" + str(state.get("used_cases", 0))[:40]
            + ",repairs:" + str(state.get("used_repairs", 0))[:40]
            + ",terminal_reserve:" + str(state.get("terminal_reserve", 0))[:40]
        )
    return "; ".join(fields)


_MAX_ACTION_CHILDREN = 4


def _action_check_output(selected: Mapping[str, object], checks: Sequence[Mapping[str, object]]) -> str:
    """The selected check, or every child of its multi-selector action.

    One automatic action runs one child per selector; showing only the last
    child hid its siblings' counts and verdicts.  Children are canonical rows
    sharing the selected row's selector scope and revision, in run order.
    """
    scope = selected.get("selector_scope")
    if not isinstance(scope, list) or len(scope) < 2 or not selected.get("execution_id"):
        return _clip(selected.get("text", "NO CHECK OUTPUT RECORDED"), 1400)
    siblings = sorted(
        (row for row in checks
         if row.get("execution_id") and row.get("selector_scope") == scope
         and row.get("revision") == selected.get("revision")),
        key=lambda row: int(row.get("ordinal", -1)),
    )[-_MAX_ACTION_CHILDREN:]
    if len(siblings) < 2:
        return _clip(selected.get("text", "NO CHECK OUTPUT RECORDED"), 1400)
    return "\n".join(_clip(row.get("text", ""), 1400) for row in siblings)


def render_evidence_pin(
        runtime: object, tools: object, *, frame: Mapping[str, object] | None = None,
        include_framefit_journal: bool | None = None,
) -> str:
    """Render one bounded join of existing check, revision, refusal, and clause records."""
    frame_map = frame if isinstance(frame, Mapping) else {}
    if include_framefit_journal is None:
        try:
            from kernel_stateframe_frame_continuity import mode as continuity_mode
            include_framefit_journal = continuity_mode() == "on"
        except ImportError:
            include_framefit_journal = False
    current_revision = _current_revision(runtime, frame_map)
    checks = _check_rows(
        runtime, frame_map, include_framefit_journal=include_framefit_journal)
    failures = [row for row in checks if str(row.get("status", "")).upper() in _FAILURE_STATUSES]
    current_checks = [
        row for row in checks
        if current_revision != "UNKNOWN" and str(row.get("revision")) == current_revision
    ]
    # Current status and unresolved historical failure are separate facts.  A
    # stale failure must not hide a newer PASS, and a newer unrelated PASS must
    # not erase the older failure that still needs attention.
    selected = max(current_checks or checks, key=lambda row: int(row.get("ordinal", -1))) \
        if (current_checks or checks) else None
    if selected is None:
        selected = {"status": "NO_CHECK", "text": "NO CHECK OUTPUT RECORDED", "revision": "UNKNOWN"}
    measured_revision = str(selected.get("revision") or "UNKNOWN")
    if current_revision == "UNKNOWN" or measured_revision in {"", "UNKNOWN"}:
        freshness = "UNKNOWN"
        readiness = "UNKNOWN_PRODUCT_REVISION"
    elif measured_revision == current_revision:
        freshness = "FRESH"
        readiness = "VALID_ONLY_FOR_MEASURED_REVISION"
    else:
        freshness = "STALE"
        readiness = "INVALIDATED_BY_PRODUCT_REVISION_CHANGE"

    refusal_candidates = _refusal_rows(runtime, frame_map)
    state = _as_mapping(getattr(runtime, "_evidencepin_refusal_state", None))
    if refusal_candidates:
        latest_refusal = max(refusal_candidates, key=lambda row: int(row.get("ordinal", -1)))
        state = dict(latest_refusal)
        setattr(runtime, "_evidencepin_refusal_state", dict(state))
    refusal_history = _action_history(runtime, frame_map)
    refusal_history.extend(checks)
    latest_refusal = state
    resolved = _linked_success(latest_refusal, refusal_history) if latest_refusal else None
    if state is None:
        refusal_text = "NO PRIOR REFUSAL RECORDED"
        refusal_arguments = "{}"
    elif resolved is not None:
        refusal_text = (
            "SUCCESS linked=" + _clip(resolved.get("receipt_id") or "none", 180)
            + "; path=" + _clip(state.get("path") or "unknown", 180)
            + "; operation=" + _clip(state.get("operation") or "unknown", 120)
            + "; condition=" + _clip(state.get("failed_condition") or "unknown", 240)
        )
        refusal_arguments = "{}"
    else:
        refusal_text = _clip(state.get("reason"), 800)
        if state.get("receipt_id") not in (None, ""):
            refusal_text += "; receipt=" + _clip(state.get("receipt_id"), 180)
        refusal_arguments = _json(state.get("arguments", {}), 800)

    brief, requirements = _requirement_rows(runtime, frame_map)
    menu_names = _menu_names(tools)
    retrieval_handle = _requirements_retrieval_handle(runtime, frame_map, requirements)
    retrieval_action = "read_ticket" if "read_ticket" in menu_names else "none"
    lines = [
        EVIDENCE_START,
        f"check_status={selected.get('status', 'UNKNOWN')}; freshness={freshness}",
        f"readiness_claim={readiness}",
        "check_output=" + _action_check_output(selected, checks),
        ("resolved_outcome=" if resolved is not None else "prior_refusal=") + refusal_text,
    ]
    # Native tool schemas are the complete action contract; do not repeat their
    # names in the model-facing evidence pin.  A missing schema surface remains
    # explicit rather than being replaced by invented advice.
    if not menu_names:
        lines.insert(1, "action_menu=[none]")
    if resolved is None:
        lines.append("rejected_arguments=" + refusal_arguments)
    full_output_handle = selected.get("full_output_handle") or selected.get("receipt_id")
    if full_output_handle:
        lines.insert(
            5,
            "check_output_receipt=" + _clip(full_output_handle, 180)
            + "; stdout_omitted_bytes=" + str(selected.get("stdout_omitted_bytes", 0))
            + "; stderr_omitted_bytes=" + str(selected.get("stderr_omitted_bytes", 0))
            + "; assertion_context_omitted_bytes="
            + str(selected.get("assertion_context_omitted_bytes", 0)),
        )
    historical_failures = [row for row in failures if row is not selected]
    if historical_failures:
        historical = max(historical_failures, key=lambda row: int(row.get("ordinal", -1)))
        historical_revision = str(historical.get("revision") or "UNKNOWN")
        historical_handle = historical.get("full_output_handle") or historical.get("receipt_id")
        lines.append(
            "historical_failure=" + _clip(historical.get("text", "UNKNOWN"), 1000)
            + "; freshness=" + (
                "UNKNOWN" if historical_revision == "UNKNOWN" or current_revision == "UNKNOWN"
                else "FRESH" if historical_revision == current_revision else "STALE"
            )
            + ("; receipt=" + _clip(historical_handle, 180) if historical_handle else "")
        )
    witness_row = _witness_pin_row(runtime, tools)
    if witness_row:
        lines.append("witness_operation=" + witness_row)
    if brief and not requirements:
        lines.append("requirements_source=" + _clip(brief, 1400))
    requirement_lines = [
        f"- {row['id']} verbatim={row['clause']} "
        f"reference={row.get('reference', row['source'])} source={row['source']} "
        f"status={row.get('status', 'UNKNOWN')} observed={row.get('observed', 'UNKNOWN')} "
        f"missing_evidence={row.get('missing_category', 'UNKNOWN')}"
        for row in requirements
    ]
    if requirements:
        lines.append("requirements:")
    else:
        lines.append("requirements=NO REQUIREMENT ROWS RECORDED")
    full_candidate = [*lines, *requirement_lines, EVIDENCE_END]
    if _pin_bytes(full_candidate) <= EVIDENCE_MAX_BYTES:
        return "\n".join(full_candidate)

    all_ids = [str(row["id"]) for row in requirements]
    selected_lines: list[str] = []
    for requirement_line in requirement_lines:
        candidate_selected_lines = [*selected_lines, requirement_line]
        candidate_selected_ids = [
            line.split(" ", 2)[1] for line in candidate_selected_lines
        ]
        omitted_ids = [identifier for identifier in all_ids if identifier not in candidate_selected_ids]
        marker = (
            "pin_truncated=true; requirements_omitted_ids="
            + ",".join(omitted_ids or ("none",))
            + "; requirements_retrieval_handle=" + retrieval_handle
            + "; retrieval_action=" + retrieval_action
        )
        candidate = [*lines, *selected_lines, requirement_line, marker, EVIDENCE_END]
        candidate = _fit_pin_core(candidate[:-1], end=EVIDENCE_END)
        if _pin_bytes([*candidate, EVIDENCE_END]) <= EVIDENCE_MAX_BYTES:
            selected_lines.append(requirement_line)

    selected_ids = [line.split(" ", 2)[1] for line in selected_lines]
    omitted_ids = [identifier for identifier in all_ids if identifier not in selected_ids]
    marker = (
        "pin_truncated=true; requirements_omitted_ids="
        + ",".join(omitted_ids or ("none",))
        + "; requirements_retrieval_handle=" + retrieval_handle
        + "; retrieval_action=" + retrieval_action
    )
    bounded_lines = _fit_pin_core([*lines, *selected_lines, marker], end=EVIDENCE_END)
    if _pin_bytes([*bounded_lines, EVIDENCE_END]) > EVIDENCE_MAX_BYTES:
        # This is a pathological all-core-overflow case.  Preserve every
        # authoritative locator and make the loss explicit rather than using a
        # lossy UTF-8 byte slice that can remove the marker or its handle.
        bounded_lines = [
            EVIDENCE_START,
            "pin_truncated=true; core_evidence_omitted=true",
            "check_output_receipt=" + _clip(full_output_handle or "none", 180),
            "requirements_omitted_ids=" + ",".join(omitted_ids or ("none",)),
            "requirements_retrieval_handle=" + retrieval_handle,
            "retrieval_action=" + retrieval_action,
        ]
    bounded_lines.append(EVIDENCE_END)
    return "\n".join(bounded_lines)


def _emit_continuity_event(runtime: object, event: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if not callable(emit):
        return
    try:
        emit(event, producer="kernel_stateframe_frame_continuity",
             dispatch=False, **fields)
    except Exception:
        return


def _is_stateframe_message(message: Mapping[str, object]) -> bool:
    content = message.get("content")
    return isinstance(content, str) and content.startswith("[StateFrame]\n")


def _replace_framefit5_owner_pin(
        message: dict[str, object], value: str,
) -> tuple[bool, bool]:
    """Update the typed FRAMEFIT5 owner pin without editing serialized JSON as text."""
    content = message.get("content")
    if not isinstance(content, str) or not content.startswith("[StateFrame]\n"):
        return False, False
    try:
        payload = json.loads(content.split("\n", 1)[1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, False
    state = payload.get("state") if isinstance(payload, dict) else None
    framefit5 = state.get("framefit5") if isinstance(state, dict) else None
    owner = framefit5.get("owner_protocol") if isinstance(framefit5, dict) else None
    if (not isinstance(framefit5, dict)
            or framefit5.get("schema") != "stateframe-framefit5/1"
            or not isinstance(owner, dict)
            or owner.get("schema") != "stateframe-framefit5-owner/1"):
        return False, False
    changed = owner.get("evidence_pin") != value
    if changed:
        owner["evidence_pin"] = value
        message["content"] = "[StateFrame]\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return True, changed


def _update_typed_owner_pin(messages: list[dict[str, object]], value: str) -> bool:
    for message in messages:
        handled, _changed = _replace_framefit5_owner_pin(message, value)
        if handled:
            return True
    return False


def _remove_evidence_pin_blocks(
        messages: list[dict[str, object]], *, preserve_stateframe: bool,
) -> None:
    source_prefixes = (
        "[Working source: ", "[Current source: ", "CURRENT_SOURCE ",
        "[Model source window]",
    )
    for message in messages:
        content = message.get("content")
        if preserve_stateframe and _is_stateframe_message(message):
            continue
        if not isinstance(content, str) or EVIDENCE_START not in content:
            continue
        if content.startswith(source_prefixes):
            source_prefix = content.split(EVIDENCE_START, 1)[0]
            message["content"] = (
                source_prefix[:-2] if source_prefix.endswith("\n\n") else source_prefix
            )
            continue
        while EVIDENCE_START in content:
            start = content.find(EVIDENCE_START)
            end = content.find(EVIDENCE_END, start + len(EVIDENCE_START))
            if end < 0:
                break
            content = (
                content[:start].rstrip() + "\n"
                + content[end + len(EVIDENCE_END):].lstrip()
            ).strip()
        message["content"] = content


def _append_evidence_pin_panel(
        messages: list[dict[str, object]], panel: str, *, preserve_stateframe: bool,
) -> None:
    source_prefixes = (
        "[Working source: ", "[Current source: ", "CURRENT_SOURCE ",
        "[Model source window]",
    )
    target = next((
        message for message in reversed(messages)
        if message.get("role") == "user"
        and not (preserve_stateframe and _is_stateframe_message(message))
    ), None)
    content = target.get("content") if isinstance(target, Mapping) else None
    if target is None or not isinstance(content, str) or content.startswith(source_prefixes):
        messages.append({"role": "user", "content": panel})
    else:
        target["content"] = content + "\n\n" + panel


def _has_evidence_pin(messages: Sequence[Mapping[str, object]], *,
                      preserve_stateframe: bool) -> bool:
    return any(
        EVIDENCE_START in str(message.get("content", ""))
        and not (preserve_stateframe and _is_stateframe_message(message))
        for message in messages
    )


def append_evidence_pin(runtime: object, messages: Sequence[dict], tools: object, *,
                        frame: Mapping[str, object] | None = None,
                        request_id: str | None = None) -> tuple[list[dict], str]:
    """Place the evidence pin after the owner-selected menu and before hashing."""
    try:
        from kernel_stateframe_frame_continuity import (
            evidence_pin_mismatch_fields, mode as continuity_mode,
        )
        selected_mode = continuity_mode()
    except Exception:
        selected_mode = "off"
    if selected_mode == "shadow":
        panel = render_evidence_pin(
            runtime, tools, frame=frame, include_framefit_journal=False)
        try:
            candidate = render_evidence_pin(
                runtime, tools, frame=frame, include_framefit_journal=True)
            mismatch_fields = evidence_pin_mismatch_fields(panel, candidate)
        except Exception as error:
            _emit_continuity_event(
                runtime, "frame_continuity_evidence_pin_failed",
                error_type=type(error).__name__, mode=selected_mode)
            mismatch_fields = ()
        for field in mismatch_fields:
            _emit_continuity_event(
                runtime, f"frame_continuity_shadow_mismatch_{field}",
                field=field)
    elif selected_mode == "on":
        try:
            panel = render_evidence_pin(
                runtime, tools, frame=frame, include_framefit_journal=True)
        except Exception as error:
            _emit_continuity_event(
                runtime, "frame_continuity_evidence_pin_failed",
                error_type=type(error).__name__, mode=selected_mode)
            panel = render_evidence_pin(
                runtime, tools, frame=frame, include_framefit_journal=False)
    else:
        panel = render_evidence_pin(
            runtime, tools, frame=frame, include_framefit_journal=False)
    result = [dict(message) for message in messages]
    preserve_stateframe = selected_mode == "on"
    _remove_evidence_pin_blocks(result, preserve_stateframe=preserve_stateframe)
    typed_pin_handled = (
        _update_typed_owner_pin(result, panel) if preserve_stateframe else False
    )
    if not typed_pin_handled:
        _append_evidence_pin_panel(
            result, panel, preserve_stateframe=preserve_stateframe,
        )
    delivered = typed_pin_handled or _has_evidence_pin(
        result, preserve_stateframe=preserve_stateframe,
    )
    if delivered:
        from kernel_completion_owner import record_owner_block
        record_owner_block(
            runtime,
            "evidence_pin",
            panel,
            block_id=f"evidence:{request_id or 'unbound'}",
            request_id=request_id,
        )
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("evidence_pin_rendered", producer="kernel_refusal_compact", request_id=request_id,
             content_bytes=len(panel.encode("utf-8")), delivered=delivered, dispatch=False)
    return result, panel if delivered else ""


def ensure_final_evidence_pin(runtime: object, messages: Sequence[Mapping[str, object]]) -> tuple[list[dict], bool]:
    """Restore the exact typed pin after late presentation filters."""
    from kernel_completion_owner import _owner_block_value
    from kernel_request_source_window import DEFERRED_SOURCE_NOTICE_HEADER

    value = _owner_block_value(runtime, "evidence_pin")
    if not isinstance(value, str) or EVIDENCE_START not in value or EVIDENCE_END not in value:
        return [dict(message) for message in messages], False
    try:
        from kernel_stateframe_frame_continuity import mode as continuity_mode
        continuity_on = continuity_mode() == "on"
    except Exception:
        continuity_on = False
    result = [dict(message) for message in messages]
    if continuity_on:
        for message in result:
            handled, changed = _replace_framefit5_owner_pin(message, value)
            if handled:
                return result, changed
    wire_values: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping) or str(message.get("role", "")) not in {
                "system", "user", "tool"}:
            continue
        if continuity_on and _is_stateframe_message(message):
            continue
        content = str(message.get("content") or "")
        if content.startswith(DEFERRED_SOURCE_NOTICE_HEADER):
            continue
        start = 0
        while True:
            begin = content.find(EVIDENCE_START, start)
            if begin < 0:
                break
            finish = content.find(EVIDENCE_END, begin + len(EVIDENCE_START))
            if finish < 0:
                break
            wire_values.append(content[begin:finish + len(EVIDENCE_END)])
            start = finish + len(EVIDENCE_END)
    if len(wire_values) == 1 and wire_values[0] == value:
        return result, False
    # Final admission is allowed to restore, replace, or deduplicate the pin;
    # leaving two pins in place makes the model-facing contract ambiguous.
    for message in result:
        content = message.get("content")
        if not isinstance(content, str) or EVIDENCE_START not in content:
            continue
        if continuity_on and _is_stateframe_message(message):
            continue
        if content.startswith(("[Working source: ", "[Current source: ", "CURRENT_SOURCE ",
                               "[Model source window]")):
            source_prefix = content.split(EVIDENCE_START, 1)[0]
            if source_prefix.endswith("\n\n"):
                source_prefix = source_prefix[:-2]
            message["content"] = source_prefix
            continue
        while EVIDENCE_START in content:
            begin = content.find(EVIDENCE_START)
            finish = content.find(EVIDENCE_END, begin + len(EVIDENCE_START))
            if finish < 0:
                content = content[:begin].rstrip()
                break
            content = (content[:begin].rstrip() + "\n"
                       + content[finish + len(EVIDENCE_END):].lstrip()).strip()
        message["content"] = content
    target = next((message for message in reversed(result)
                   if message.get("role") == "user"
                   and isinstance(message.get("content"), str)
                   and not (continuity_on and _is_stateframe_message(message))
                   and not str(message.get("content", "")).startswith((
                       "[Working source: ", "[Current source: ", "CURRENT_SOURCE ",
                       "[Model source window]", DEFERRED_SOURCE_NOTICE_HEADER))), None)
    if target is None:
        result.append({"role": "user", "content": value})
    else:
        content = str(target["content"]).rstrip()
        target["content"] = value if not content else content + "\n\n" + value
    return result, True


def _finding(runtime: object) -> Mapping[str, object]:
    continuation = getattr(runtime, "continuation", None)
    method = getattr(continuation, "completion_finding", None)
    try:
        result = method() if callable(method) else {}
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        result = {}
    return result if isinstance(result, Mapping) else {}


def _tool_names(tools: object) -> tuple[str, ...]:
    if isinstance(tools, Mapping):
        tools = (tools,)
    names: list[str] = []
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)):
        for tool in tools:
            function = tool.get("function") if isinstance(tool, Mapping) else None
            name = _text(function.get("name"), 80) if isinstance(function, Mapping) else ""
            if name and name not in names:
                names.append(name)
    return tuple(names)


def _stale_item(runtime: object, decision: object, finding: Mapping[str, object]) -> str:
    stale = finding.get("stale_dependency")
    if isinstance(stale, Mapping) and stale:
        return _text(next(iter(stale)), 180)
    for value in (getattr(decision, "needs_work_ids", ()), finding.get("selectors")):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and value:
            return _text(value[0], 180)
    return "continuation binding/dependencies"


def _refresh_action(finding: Mapping[str, object], tools: object) -> str:
    offered = set(_tool_names(tools))
    selectors = finding.get("selectors")
    selector = (_text(selectors[0], 180) if isinstance(selectors, Sequence)
                and not isinstance(selectors, (str, bytes, bytearray)) and selectors
                else "the current public check")
    if "read" in offered:
        return f"read {selector}"
    if "run_tests" in offered:
        return f"run_tests {selector}"
    for candidate in ("grep", "probe", "str_replace"):
        if candidate in offered:
            return f"use {candidate} on the stale item"
    return "preserve the refusal until a refresh action is offered"


def render(runtime: object, decision: object, tools: object) -> str:
    if not enabled():
        return ""
    finding = _finding(runtime)
    reason_codes = tuple(getattr(decision, "reason_codes", ()) or ())
    code = _text(finding.get("code"), 100).upper()
    if not code:
        code = _text(next(iter(reason_codes), "COMPLETION_REFUSED"), 100).upper()
    if "STALE" in code or "STALE" in _text(finding.get("detail"), 240).upper():
        code = "STALE_EVIDENCE"
    if getattr(decision, "accepted", False) and not finding:
        return ""
    item = _stale_item(runtime, decision, finding)
    action = _refresh_action(finding, tools)
    text = (f"{START}\ncode={code}; stale item={item}; available refresh action={action}\n{END}")
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_BYTES:
        return text
    suffix = f"; available refresh action={action}\n{END}".encode("utf-8")
    prefix = f"{START}\ncode={code}; stale item=".encode("utf-8")
    room = max(0, MAX_BYTES - len(prefix) - len(suffix))
    item_bytes = item.encode("utf-8")[:room]
    return (prefix + item_bytes + suffix).decode("utf-8", "ignore")


def append_final(runtime: object, messages: Sequence[dict], tools: object, *,
                 request_id: str | None = None) -> tuple[list[dict], str]:
    """Insert the panel after all request projections, before hashing/submission."""
    decision = getattr(runtime, "_last_completion_decision", None)
    panel = render(runtime, decision, tools)
    if not panel:
        return [dict(message) for message in messages], ""
    result = [dict(message) for message in messages]
    for message in result:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if message.get("role") == "tool":
            # A tool result is the model's own settled observation.  Stripping
            # a refusal block out of it left the refused call paired with ""
            # (live J857 response 57); the observation is carried verbatim.
            continue
        while START in content:
            start = content.find(START)
            end = content.find(END, start + len(START))
            if end < 0:
                break
            content = (content[:start].rstrip() + "\n" +
                       content[end + len(END):].lstrip()).strip()
        message["content"] = content
    target = next((message for message in reversed(result)
                   if message.get("role") == "user"), None)
    if target is None or not isinstance(target.get("content"), str):
        result.append({"role": "user", "content": panel})
    else:
        target["content"] = target["content"] + "\n\n" + panel
    delivered = any(START in str(message.get("content", "")) for message in result)
    runtime.journal.emit("refusal_compact_rendered", producer="kernel_refusal_compact",
                         request_id=request_id, content_bytes=len(panel.encode("utf-8")),
                         delivered=delivered, marker_present=delivered, dispatch=False)
    return result, panel if delivered else ""


__all__ = [
    "END", "EVIDENCE_END", "EVIDENCE_MAX_BYTES", "EVIDENCE_START", "MAX_BYTES",
    "MODE_ENV", "START", "append_evidence_pin", "append_final", "enabled", "mode",
    "render", "render_evidence_pin",
]
