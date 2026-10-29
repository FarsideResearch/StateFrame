"""Bounded, opt-in continuity fields for the F4 model-facing StateFrame."""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence

from kernel_read_working_set import _read_ranges, evictfix_enabled

MODE_ENV = "STATEFRAME_FRAME_CONTINUITY"
SCHEMA = "stateframe-frame-continuity/1"
_EMPTY_HYPOTHESES_SCHEMA = "stateframe-hypotheses/1"

_SEQUENCE_LIMITS = {
    "hypotheses": 8,
    "model_claims": 4,
    "read_memory": 3,
    "negative_work": 8,
}
_SEQUENCE_OMISSIONS = {
    "hypotheses": "hypotheses_omitted",
    "model_claims": "model_claims_omitted",
    "read_memory": "read_memory_omitted",
    "negative_work": "negative_work_omitted",
    "required_public_checks": "required_public_checks_omitted",
}
_SEQUENCE_BYTE_LIMITS = {
    "hypotheses": 4096,
    "model_claims": 3072,
    "read_memory": 3072,
    "negative_work": 4096,
}
_VALUE_OMISSIONS = {
    "objective": "objective_omitted",
    "obligations": "obligations_omitted",
    "hypothesis_state": "hypothesis_state_omitted",
    "facts": "facts_omitted",
    "world": "world_omitted",
    "action_prerequisites": "action_prerequisites_omitted",
    "completion_contract": "completion_contract_omitted",
    "source_access": "source_access_omitted",
}
_VALUE_BYTE_LIMITS = {
    "objective": 1024,
    "obligations": 2048,
    "hypothesis_state": 1024,
    "facts": 2048,
    "world": 2048,
    "action_prerequisites": 2048,
    "completion_contract": 3072,
    "source_access": 3072,
}
_HISTORY_FINDINGS_BYTE_LIMIT = 4096
_MAX_SELECTOR_BYTES = 512
_MAX_JSON_DEPTH = 6
_MAX_JSON_ITEMS = 32
_MAX_NESTED_ITEM_BYTES = 1024
_COPY_FIELDS = tuple(_VALUE_OMISSIONS)
_PLACEHOLDER_SELECTORS = frozenset({
    "", "not recorded", "unknown", "none", "n/a", "na", "null",
})
_CHECK_STATUSES = frozenset({
    "BLOCKED", "CERTIFIED", "CURRENT", "COUNTEREXAMPLE", "ERROR", "FAIL",
    "FAILED", "GREEN", "GREEN_CURRENT", "NEEDS_WORK", "OK", "PASS", "PASSED",
    "RED", "REJECTED", "REFUSED", "REUSED", "SATISFIED", "STALE", "SUCCESS",
    "UNEXECUTED", "UNKNOWN", "UNSATISFIED", "VERIFIED", "VIOLATED",
})
_EVIDENCE_PIN_FIELDS = (
    "check_status", "freshness", "readiness_claim", "check_output",
    "check_output_receipt", "historical_failure", "prior_refusal", "resolved_outcome",
)


def mode(value: object = None) -> str:
    """Read the rollout switch; invalid values fail open to the legacy projection."""
    raw = os.environ.get(MODE_ENV, "off") if value is None else value
    normalized = str(raw or "off").strip().lower()
    return normalized if normalized in {"off", "shadow", "on"} else "off"


def _sequence(value: object) -> list[object] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    return list(value)


def _omitted_count(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _strings(value: object) -> list[str]:
    sequence = _sequence(value)
    values = sequence if sequence is not None else [value]
    result: list[str] = []
    for raw in values:
        if isinstance(raw, Mapping):
            raw = next((raw.get(key) for key in (
                "selector", "check_id", "test_selector", "path", "name"
            ) if isinstance(raw.get(key), str) and raw.get(key).strip()), None)
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if text.casefold() in _PLACEHOLDER_SELECTORS or text in result:
            continue
        result.append(text)
    return result


def _required_selectors(frame: Mapping[str, object],
                        state: Mapping[str, object],
                        authoritative_selectors: object = ()) -> tuple[list[str], int]:
    contract = frame.get("completion_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    world = frame.get("world")
    world = world if isinstance(world, Mapping) else {}
    public_checks = state.get("public_checks")
    public_checks = public_checks if isinstance(public_checks, Mapping) else {}

    collected: list[str] = []
    oversized: set[str] = set()
    # LiveProjection/live_contract and the explicit public authority lead; the
    # pinned FRAMEFIT5 target is supplied only when those authorities are empty.
    for value in (
        frame.get("required_public_checks"),
        contract.get("required_public_checks"),
        authoritative_selectors,
        world.get("required_selectors"),
        public_checks.get("selectors"),
    ):
        for selector in _strings(value):
            encoded_size = _json_size(selector)
            if encoded_size is None or encoded_size > _MAX_SELECTOR_BYTES:
                oversized.add(selector)
                continue
            if selector not in collected:
                collected.append(selector)
    inherited_omitted = max(
        _omitted_count(frame.get("required_public_checks_omitted")),
        _omitted_count(contract.get("required_public_checks_omitted")),
        _omitted_count(world.get("required_selectors_omitted")),
        _omitted_count(public_checks.get("selectors_omitted")),
    )
    omitted = inherited_omitted + len(oversized) + max(0, len(collected) - 8)
    return collected[:8], omitted


def framefit5_public_selectors(action_context: object) -> tuple[str, ...]:
    """Read the target from a pinned FRAMEFIT5 verdict with matching revision."""
    context = action_context if isinstance(action_context, Mapping) else {}
    framefit5 = context.get("framefit5")
    if not isinstance(framefit5, Mapping):
        return ()
    verdict = framefit5.get("last_verdict")
    ledger = framefit5.get("action_ledger")
    ledger = ledger if isinstance(ledger, Mapping) else {}
    last_test = ledger.get("last_test")
    if not isinstance(verdict, Mapping) or not isinstance(last_test, Mapping):
        return ()
    receipt_id = verdict.get("check_receipt_id")
    revision = verdict.get("tested_revision")
    target = last_test.get("target")
    if (not isinstance(receipt_id, str) or not receipt_id.strip()
            or not isinstance(revision, str) or not revision.strip()
            or last_test.get("revision") != revision
            or not isinstance(target, str) or not target.strip()):
        return ()
    return (target.strip(),)


def _json_size(value: object) -> int | None:
    try:
        return len(json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8"))
    except (TypeError, ValueError, UnicodeEncodeError):
        return None


def _empty_shape(value: object) -> object:
    if isinstance(value, Mapping):
        return {}
    if _sequence(value) is not None:
        return []
    if isinstance(value, str):
        return ""
    return None


def _omitted_units(value: object) -> int:
    if isinstance(value, Mapping):
        return max(1, len(value))
    rows = _sequence(value)
    if rows is not None:
        return max(1, len(rows))
    return 1


def _bounded_string(value: str, byte_limit: int) -> tuple[str, int]:
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        candidate_size = _json_size(value[:middle])
        if candidate_size is not None and candidate_size <= byte_limit:
            low = middle
        else:
            high = middle - 1
    return value[:low], int(low < len(value))


def _bounded_mapping(
        value: Mapping[object, object], byte_limit: int, *, depth: int,
) -> tuple[dict[str, object], int]:
    priority = (
        "schema", "status", "state", "objective", "required_public_checks",
        "selectors", "revision", "path", "target", "claim", "kind", "id",
        "source", "evidence", "obligations",
    )
    priority_rank = {key: index for index, key in enumerate(priority)}
    keys = sorted(
        (key for key in value if isinstance(key, str)),
        key=lambda key: (priority_rank.get(key, len(priority)), key),
    )
    omitted = len(value) - len(keys) + max(0, len(keys) - _MAX_JSON_ITEMS)
    result: dict[str, object] = {}
    for key in keys[:_MAX_JSON_ITEMS]:
        result_size = _json_size(result) or 2
        key_size = _json_size(key)
        available = byte_limit - result_size - (key_size or byte_limit) - 2
        if available < 2:
            omitted += _omitted_units(value[key])
            continue
        bounded, child_omitted = _bounded_json_value(
            value[key], min(_MAX_NESTED_ITEM_BYTES, available), depth=depth + 1,
        )
        candidate = {**result, key: bounded}
        candidate_size = _json_size(candidate)
        if candidate_size is None or candidate_size > byte_limit:
            omitted += _omitted_units(value[key])
            continue
        result = candidate
        if child_omitted:
            omitted += 1
    return result, omitted


def _bounded_sequence(
        rows: Sequence[object], byte_limit: int, *, depth: int,
) -> tuple[list[object], int]:
    omitted = max(0, len(rows) - _MAX_JSON_ITEMS)
    retained: list[object] = []
    for row in reversed(rows[-_MAX_JSON_ITEMS:]):
        available = byte_limit - (_json_size(retained) or 2) - 2
        if available < 2:
            omitted += _omitted_units(row)
            continue
        bounded, child_omitted = _bounded_json_value(
            row, min(_MAX_NESTED_ITEM_BYTES, available), depth=depth + 1,
        )
        candidate = [bounded, *retained]
        candidate_size = _json_size(candidate)
        if candidate_size is None or candidate_size > byte_limit:
            omitted += _omitted_units(row)
            continue
        retained = candidate
        if child_omitted:
            omitted += 1
    return retained, omitted


def _bounded_json_value(
        value: object, byte_limit: int, *, depth: int = 0,
) -> tuple[object, int]:
    """Keep JSON data within a byte cap and report omitted rows/values."""
    size = _json_size(value)
    if size is not None and size <= byte_limit:
        try:
            return copy.deepcopy(value), 0
        except Exception:
            return _empty_shape(value), _omitted_units(value)
    if byte_limit < 2 or depth >= _MAX_JSON_DEPTH:
        return _empty_shape(value), _omitted_units(value)
    if isinstance(value, str):
        return _bounded_string(value, byte_limit)
    if isinstance(value, Mapping):
        return _bounded_mapping(value, byte_limit, depth=depth)
    rows = _sequence(value)
    if rows is not None:
        return _bounded_sequence(rows, byte_limit, depth=depth)
    return _empty_shape(value), _omitted_units(value)


def _copy_sequence_field(
        source: Mapping[str, object], target: dict[str, object], key: str,
        *, limit: int,
) -> int:
    rows = _sequence(source.get(key))
    count_key = _SEQUENCE_OMISSIONS[key]
    inherited = _omitted_count(source.get(count_key))
    if rows is None:
        target[key] = []
        removed = int(source.get(key) is not None)
        target[count_key] = inherited + removed
        return removed
    selected = rows[-limit:] if limit else []
    removed = max(0, len(rows) - len(selected))
    if key == "read_memory" and evictfix_enabled():
        selected = [_compact_read_memory_row(row) for row in selected]
    bounded, clipped = _bounded_json_value(
        selected, _SEQUENCE_BYTE_LIMITS[key],
    )
    if not isinstance(bounded, list):
        bounded = []
        clipped = max(1, clipped)
    target[key] = bounded
    target[count_key] = inherited + removed + clipped
    return removed + clipped


def _compact_read_memory_row(value: object) -> dict[str, object]:
    """Keep provenance and a short note, never a copied source body."""
    if not isinstance(value, Mapping):
        return {"summary": "Prior read recorded; revalidate source before editing."}
    row: dict[str, object] = {}
    path = value.get("path") or value.get("target")
    if isinstance(path, str) and path:
        row["path"] = path
    ranges = _read_ranges(value)
    if ranges:
        row["ranges"] = ranges
    revision = value.get("revision")
    if isinstance(revision, str) and revision:
        row["revision"] = revision
    summary = value.get("summary")
    row["summary"] = (summary[:160] if isinstance(summary, str) and summary.strip()
                      else "Prior read recorded; revalidate source before editing.")
    return row


def _copy_live_sequence_fields(
        frame: Mapping[str, object], state: dict[str, object],
) -> tuple[set[str], list[str], dict[str, int]]:
    changed: set[str] = set()
    carried: list[str] = []
    omitted: dict[str, int] = {}
    for key, limit in _SEQUENCE_LIMITS.items():
        if key not in frame:
            continue
        count_key = _SEQUENCE_OMISSIONS[key]
        before_rows, before_count = state.get(key), state.get(count_key)
        _copy_sequence_field(frame, state, key, limit=limit)
        if before_rows != state.get(key):
            changed.add(key)
        if before_count != state.get(count_key):
            changed.add(count_key)
        carried.append(key)
        count = _omitted_count(state.get(count_key))
        if count:
            omitted[key] = count
    return changed, carried, omitted


def _copy_live_value_fields(
        frame: Mapping[str, object], state: dict[str, object],
) -> tuple[set[str], list[str], dict[str, int]]:
    changed: set[str] = set()
    carried: list[str] = []
    omitted: dict[str, int] = {}
    for key in _COPY_FIELDS:
        if key not in frame:
            continue
        count_key = _VALUE_OMISSIONS[key]
        inherited = _omitted_count(frame.get(count_key))
        bounded, clipped = _bounded_json_value(
            frame.get(key), _VALUE_BYTE_LIMITS[key],
        )
        count = inherited + clipped
        if state.get(key) != bounded or key not in state:
            changed.add(key)
        if state.get(count_key) != count:
            changed.add(count_key)
        state[key] = bounded
        state[count_key] = count
        carried.append(key)
        if count:
            omitted[key] = count
    return changed, carried, omitted


def _ensure_hypotheses_carrier(
        frame: Mapping[str, object], state: dict[str, object],
) -> bool:
    """Represent a journal-backed empty hypothesis set without inventing a carrier."""
    current = state.get("hypotheses")
    source_rows = _sequence(frame.get("hypotheses"))
    if source_rows is None:
        return False
    if source_rows:
        return False
    omitted = _omitted_count(state.get("hypotheses_omitted"))
    source_state = state.get("hypothesis_state")
    source_status = source_state.get("status") if isinstance(source_state, Mapping) else None
    if omitted:
        status = "PARTIAL"
    elif source_rows is not None and isinstance(source_status, str) \
            and source_status.strip().casefold() == "none_recorded":
        status = "NONE_RECORDED"
    elif isinstance(source_status, str) \
            and source_status.strip().casefold() == "unknown":
        status = "UNKNOWN"
    else:
        state.pop("hypotheses", None)
        return current is not None
    state["hypotheses"] = {
        "schema": _EMPTY_HYPOTHESES_SCHEMA,
        "status": status,
        "items": [],
        "omitted_count": omitted,
    }
    state.setdefault("hypotheses_omitted", omitted)
    return True


def _project_required_selectors(
        frame: Mapping[str, object], state: dict[str, object],
        authoritative_selectors: object = (),
) -> tuple[set[str], int]:
    required, omitted = _required_selectors(frame, state, authoritative_selectors)
    changed: set[str] = set()
    if state.get("required_public_checks") != required:
        changed.add("required_public_checks")
    if state.get("required_public_checks_omitted") != omitted:
        changed.add("required_public_checks_omitted")
    state["required_public_checks"] = required
    state["required_public_checks_omitted"] = omitted

    current_checks = state.get("public_checks")
    if isinstance(current_checks, Mapping):
        checks = copy.deepcopy(dict(current_checks))
        if checks.get("selectors") != required:
            changed.add("public_checks_selectors")
        if checks.get("selectors_omitted") != omitted:
            changed.add("public_checks_selectors_omitted")
        checks["selectors"] = list(required)
        checks["selectors_omitted"] = omitted
    else:
        checks = {
            "state": "UNKNOWN",
            "selectors": list(required),
            "selectors_omitted": omitted,
        }
        changed.update(("public_checks_selectors", "public_checks_selectors_omitted"))
    state["public_checks"] = checks
    return changed, omitted


def _carry_history_findings(
        frame: Mapping[str, object], state: dict[str, object],
) -> tuple[bool, bool, int]:
    old_history = state.get("history_findings")
    old_omitted = state.get("history_findings_omitted")
    history = frame.get("history_findings")
    if isinstance(history, Mapping):
        try:
            from kernel_stateframe_history_findings import validate_projection
            validate_projection(history)
            size = _json_size(history)
            if size is None or size > _HISTORY_FINDINGS_BYTE_LIMIT:
                state["history_findings"] = None
                state["history_findings_omitted"] = 1
                changed = old_history is not None or old_omitted != 1
                return changed, True, 1
            copied = copy.deepcopy(dict(history))
            state["history_findings"] = copied
            state["history_findings_omitted"] = 0
            return old_history != copied or old_omitted != 0, True, 0
        except (ImportError, AttributeError, TypeError, ValueError):
            state["history_findings"] = None
            state["history_findings_omitted"] = 1
            changed = old_history is not None or old_omitted != 1
            return changed, True, 1
    state["history_findings"] = None
    state["history_findings_omitted"] = 0
    changed = old_history is not None or old_omitted != 0
    return changed, True, 0


def project_continuity(
        frame: Mapping[str, object],
        projection: Mapping[str, object],
        *,
        authoritative_selectors: object = (),
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Copy the allowlisted live view into F4 state and report changed field names."""
    result = copy.deepcopy(dict(projection))
    current_state = result.get("state")
    if not isinstance(current_state, Mapping):
        raise ValueError("F4 projection has no state mapping")
    state = copy.deepcopy(dict(current_state))
    sequence_changes, sequence_fields, omitted = _copy_live_sequence_fields(frame, state)
    value_changes, value_fields, value_omitted = _copy_live_value_fields(frame, state)
    omitted.update(value_omitted)
    if _ensure_hypotheses_carrier(frame, state) and "hypotheses" not in sequence_fields:
        sequence_fields.append("hypotheses")
    selector_changes, selector_omissions = _project_required_selectors(
        frame, state, authoritative_selectors,
    )
    history_omitted_before = state.get("history_findings_omitted")
    history_changed, history_carried, history_omissions = _carry_history_findings(frame, state)

    changed = sequence_changes | value_changes | selector_changes
    if history_changed:
        changed.add("history_findings")
    if state.get("history_findings_omitted") != history_omitted_before:
        changed.add("history_findings_omitted")
    carried = sequence_fields + value_fields
    carried.append("required_public_checks")
    if history_carried:
        carried.append("history_findings")
        if history_omissions:
            omitted["history_findings"] = history_omissions
    if selector_omissions:
        omitted["required_public_checks"] = selector_omissions

    state["frame_continuity"] = {
        "schema": SCHEMA,
        "source": "live_runtime_frame",
        "carried_fields": sorted(set(carried)),
        "omitted_counts": omitted,
    }
    result["state"] = state
    return result, tuple(sorted(changed))


def _emit(journal: object, event: str, **fields: object) -> None:
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    try:
        emit(event, producer="kernel_stateframe_frame_continuity",
             dispatch=False, **fields)
    except Exception:
        return


def apply_f4_projection(
        frame: Mapping[str, object],
        projection: Mapping[str, object],
        *,
        journal: object = None,
        authoritative_selectors: object = (),
) -> Mapping[str, object]:
    """Apply the bounded projection only when enabled; shadow emits field-only events."""
    selected_mode = mode()
    if selected_mode == "off":
        return projection
    try:
        candidate, mismatches = project_continuity(
            frame, projection, authoritative_selectors=authoritative_selectors,
        )
    except Exception as error:
        _emit(journal, "frame_continuity_projection_failed",
              error_type=type(error).__name__, mode=selected_mode)
        return projection
    if selected_mode == "shadow":
        for field in mismatches:
            _emit(
                journal,
                f"frame_continuity_shadow_mismatch_{field}",
                field=field,
            )
        return projection
    _emit(
        journal,
        "frame_continuity_projected",
        field_count=len(candidate.get("state", {}).get(
            "frame_continuity", {}).get("carried_fields", ())),
        mismatch_fields=list(mismatches),
    )
    return candidate


def journal_check_receipts(action_context: object) -> tuple[dict[str, object], ...]:
    """Normalize the already-pinned FRAMEFIT5 check receipts for owner evidence rendering."""
    context = action_context if isinstance(action_context, Mapping) else {}
    records = _sequence(context.get("journal_receipt_records")) or []
    result: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping) or record.get("kind") != "check_receipt":
            continue
        status_value = record.get("status")
        if not isinstance(status_value, str):
            continue
        status = status_value.strip().upper().replace("-", "_")
        if (status not in _CHECK_STATUSES
                and not status.endswith(("_PASS", "_GREEN", "_FAIL", "_RED"))):
            continue
        revision = next((record.get(key) for key in (
            "revision", "tested_rev", "tested_revision", "source_revision"
        ) if isinstance(record.get(key), str) and record.get(key).strip()), None)
        receipt_id = next((record.get(key) for key in (
            "receipt_id", "check_receipt_id", "event_id"
        ) if isinstance(record.get(key), str) and record.get(key).strip()), None)
        row: dict[str, object] = {
            "kind": "check_receipt",
            "status": status,
            "text": "journal check receipt status=" + status,
            "source": "framefit5_pinned_journal",
        }
        if isinstance(revision, str):
            row["revision"] = revision
            row["text"] += " revision=" + revision
        if isinstance(receipt_id, str):
            row["receipt_id"] = receipt_id
        for key in ("selector", "execution_id", "request_id", "sequence", "ordinal"):
            value = record.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                row[key] = value
        event_id = record.get("source_event_id", record.get("event_id"))
        if isinstance(event_id, str) and event_id:
            row["source_event_id"] = event_id
        result.append(row)
    return tuple(result)


def _evidence_status_fields(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in line.split(";"):
        key, separator, item = part.partition("=")
        if separator and key.strip() in {"check_status", "freshness"}:
            result[key.strip()] = item.strip()
    return result


def _evidence_pin_fields(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    in_block = False
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if line == "[EVIDENCEPIN]":
            in_block = True
            continue
        if line == "[/EVIDENCEPIN]":
            break
        if not in_block:
            continue
        if line.startswith("check_status="):
            result.update(_evidence_status_fields(line))
        else:
            key, separator, item = line.partition("=")
            if separator and key.strip() in _EVIDENCE_PIN_FIELDS:
                result[key.strip()] = item.strip()
    return result


def evidence_pin_mismatch_fields(before: str, after: str) -> tuple[str, ...]:
    """Compare allowlisted check fields; never return or journal their values."""
    left, right = _evidence_pin_fields(before), _evidence_pin_fields(after)
    return tuple(key for key in _EVIDENCE_PIN_FIELDS if left.get(key) != right.get(key))


__all__ = [
    "MODE_ENV",
    "SCHEMA",
    "apply_f4_projection",
    "evidence_pin_mismatch_fields",
    "framefit5_public_selectors",
    "journal_check_receipts",
    "mode",
    "project_continuity",
]
