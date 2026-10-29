"""Adaptive, lossless StateFrame projections.

The gear layer is deliberately a projection layer.  A caller hands it the
complete host-owned frame and it returns a new JSON-compatible mapping; it
never edits the supplied mapping and never decides whether a requirement is
satisfied.  ``STATEFRAME_GEARS=off`` is handled by returning a deep copy so a
caller can use the same seam without changing the legacy wire payload.

The controller owns only presentation state (the current monotone gear and
debounced trigger observations).  Candidate, obligation, evidence, and
counter values used at a promotion boundary are copied into the promotion
snapshot and can be checked/replayed without consulting the current
workspace.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


MODE_ENV = "STATEFRAME_GEARS"
REQUIREMENTS_LEDGER_ENV = "STATEFRAME_REQUIREMENTS_LEDGER"
SCHEMA = "stateframe-gears/1"
PROMOTION_SCHEMA = "stateframe-gear-promotion/1"
VALID_MODES = frozenset({"off", "adaptive", "g0", "g1", "g2", "g3"})
GEAR_ORDER = {"g0": 0, "g1": 1, "g2": 2, "g3": 3}
PUBLIC_GEARS = frozenset({"g1", "g2", "g3"})

_PASS = frozenset({"GREEN", "PASS", "PASSED", "SUCCESS", "OK", "SATISFIED", "CERTIFIED"})
_FAIL = frozenset({"RED", "FAIL", "FAILED", "ERROR", "TIMEOUT", "VIOLATED", "CONFLICT"})
_TERMINAL_OPEN = frozenset({"open", "OPEN", "pending", "PENDING", "unresolved", "UNRESOLVED",
                             "blocked", "BLOCKED", "counterexample", "COUNTEREXAMPLE"})
_TERMINAL_CLOSED = frozenset({"satisfied", "SATISFIED", "closed", "CLOSED", "done", "DONE"})


class GearError(ValueError):
    """Invalid gear configuration or promotion record."""


class AtomicBoundaryError(GearError):
    """A promotion was requested between durable events."""


class PromotionReplayError(GearError):
    """A promotion snapshot was changed or is incomplete."""


def _deep(value: object) -> object:
    return copy.deepcopy(value)


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _as_list(value: object) -> list:
    if _sequence(value):
        return list(value)
    if value is None:
        return []
    return [value]


def _first(state: Mapping[str, object], *names: str, default: object = None) -> object:
    for name in names:
        if name in state and state[name] is not None:
            return state[name]
    return default


def _bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "active", "present"}
    return bool(value)


def _int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def gear_mode(value: object = None) -> str:
    """Return the validated switch value; the default is the legacy OFF arm."""
    raw = os.environ.get(MODE_ENV, "off") if value is None else value
    if not isinstance(raw, str):
        raise ValueError(f"{MODE_ENV} must be off, adaptive, g0, g1, g2 or g3")
    normalized = raw.strip().lower()
    if normalized not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off, adaptive, g0, g1, g2 or g3")
    return normalized


mode = gear_mode


def enabled(value: object = None) -> bool:
    return gear_mode(value) != "off"


def requirements_ledger_mode(value: object = None) -> str:
    raw = os.environ.get(REQUIREMENTS_LEDGER_ENV, "off") if value is None else value
    if not isinstance(raw, str) or raw.strip().lower() not in {"off", "on"}:
        raise ValueError(f"{REQUIREMENTS_LEDGER_ENV} must be off or on")
    return raw.strip().lower()


def requirements_ledger_enabled(value: object = None) -> bool:
    return requirements_ledger_mode(value) == "on"


@dataclass(frozen=True)
class EntryDecision:
    gear: str
    reason: str
    facts: dict[str, object]


def _entry_inputs(state: Mapping[str, object] | None, **overrides: object) -> dict[str, object]:
    values = dict(state) if isinstance(state, Mapping) else {}
    values.update(overrides)
    return values


def choose_entry(state: Mapping[str, object] | None = None, **overrides: object) -> EntryDecision:
    """Apply the cheap host-only entry rule from entry assessment.

    No text generation, model response, or model-authored claim is consulted.
    Explicit coordinated resume/plan state wins over the small-ticket rules.
    """
    values = _entry_inputs(state, **overrides)
    coordinated = _bool(_first(
        values, "resuming_coordinated_job", "coordinated_job", "structured_plan",
        "resuming", "resume", "background_jobs", "background_operation",
        default=False))
    if coordinated:
        return EntryDecision("g3", "coordinated_resume_or_structured_plan", {
            "coordinated": True,
        })
    answer_only = _bool(_first(
        values, "trusted_answer_only", "recognized_answer_only", "answer_only",
        "answer_task", default=False))
    if answer_only:
        return EntryDecision("g0", "trusted_answer_only", {"answer_only": True})
    files = _first(values, "target_files", "explicit_target_files", "files", default=())
    if isinstance(files, Mapping):
        file_count = len(files)
    else:
        file_count = len(_as_list(files))
    packages = _first(values, "packages", "target_packages", "explicit_packages", default=())
    package_count = len(packages) if isinstance(packages, Mapping) else len(_as_list(packages))
    source_tokens = _int(_first(
        values, "known_required_source_tokens", "required_source_tokens",
        "source_tokens", default=0))
    if file_count >= 3 or package_count >= 2 or source_tokens > 6000:
        return EntryDecision("g2", "broad_initial_scope", {
            "explicit_target_files": file_count,
            "explicit_packages": package_count,
            "known_required_source_tokens": source_tokens,
        })
    return EntryDecision("g1", "local_ticket_default", {
        "explicit_target_files": file_count,
        "explicit_packages": package_count,
        "known_required_source_tokens": source_tokens,
    })


def entry_gear(state: Mapping[str, object] | None = None, **overrides: object) -> str:
    return choose_entry(state, **overrides).gear


def entry_rule(state: Mapping[str, object] | None = None, **overrides: object) -> dict[str, object]:
    decision = choose_entry(state, **overrides)
    return {"gear": decision.gear, "reason": decision.reason, "facts": _deep(decision.facts)}


def _meaningful_product_files(state: Mapping[str, object]) -> set[str]:
    direct = _first(state, "meaningful_product_files", "edited_product_files", default=None)
    if direct is not None:
        return {str(item) for item in (direct.keys() if isinstance(direct, Mapping) else _as_list(direct))
                if str(item)}
    found: set[str] = set()
    rows = _first(state, "recent_changes", "changes", "edit_history", default=())
    for row in _as_list(rows):
        if not isinstance(row, Mapping):
            continue
        result = str(_first(row, "result", "result_class", "observed_result", default="APPLIED")).upper()
        if result not in {"APPLIED", "OK", "SUCCESS", "GREEN"}:
            continue
        if row.get("meaningful") is False or row.get("role") in {"test", "harness"}:
            continue
        nested = row.get("changed_files")
        candidates = _as_list(nested) if nested is not None else [row]
        for change in candidates:
            if isinstance(change, str) and change:
                found.add(change.replace("\\", "/").lstrip("./"))
                continue
            if not isinstance(change, Mapping):
                continue
            role = str(change.get("role", row.get("role", "product"))).lower()
            if role in {"test", "harness", "fixture"} or change.get("meaningful") is False:
                continue
            path = change.get("path", row.get("path", row.get("target_path", "")))
            if path:
                found.add(str(path))
    return found


def _working_set_tokens(state: Mapping[str, object]) -> int:
    direct = _first(state, "working_set_tokens", "working_set_token_count", default=None)
    if direct is not None:
        return max(0, _int(direct))
    manifest = _first(state, "working_set_manifest", "working_set", "retained_context",
                      "read_working_set", default=None)
    if isinstance(manifest, Mapping):
        return max(0, _int(_first(manifest, "token_count", "tokens", "estimated_tokens", default=0)))
    return 0


def _status(value: object) -> str:
    if isinstance(value, Mapping):
        value = _first(value, "status", "state", "result", "verdict", default="UNKNOWN")
    return str(value or "UNKNOWN").strip().upper()


def _status_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _status(item) for key, item in value.items()}


def _previous_pass_now_failed(state: Mapping[str, object]) -> bool:
    if _bool(_first(state, "previously_passing_applicable_check_now_fails",
                   "previous_pass_now_fail", "check_regressed", default=False)):
        return True
    previous = _status_mapping(_first(state, "previous_check_statuses", "prior_check_statuses", default={}))
    current = _status_mapping(_first(state, "applicable_check_statuses", "check_statuses", default={}))
    if not current:
        contract = _mapping(state.get("completion_contract"))
        if contract is not None:
            current = _status_mapping(contract.get("current_public_checks"))
    applicable = _first(state, "applicable_checks", "required_checks", default=None)
    allowed = {str(item) for item in _as_list(applicable)} if applicable is not None else None
    return any(
        (allowed is None or key in allowed)
        and old in _PASS and now in _FAIL
        for key, old in previous.items() for now in [current.get(key, "UNKNOWN")]
    )


def _normalized_call(row: Mapping[str, object]) -> str:
    explicit = row.get("normalized")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    tool = str(_first(row, "tool", "kind", "name", "action", default="unknown")).lower()
    args = _first(row, "arguments", "parameters", default=None)
    if not isinstance(args, Mapping):
        args = {key: row[key] for key in ("path", "target", "selector", "line") if key in row}
    args = {str(key): value for key, value in args.items() if key not in {"why", "reason"}}
    try:
        encoded = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        encoded = str(args)
    return f"{tool}:{encoded}"


def _same_normalized_read_or_test(state: Mapping[str, object]) -> bool:
    rows = _first(state, "recent_calls", "normalized_calls", "read_test_history",
                  "recent_actions", "call_history", default=())
    rows = [row for row in _as_list(rows) if isinstance(row, Mapping)][-4:]
    if len(rows) < 3:
        return False
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        tool = str(_first(row, "tool", "kind", "name", "action", default="")).lower()
        if tool not in {"read", "test", "run_tests", "pytest", "recall_evidence"}:
            continue
        groups.setdefault(_normalized_call(row), []).append(row)
    return any(
        len(items) >= 3 and all(not _bool(_first(item, "new_evidence", "evidence_added",
                                                 "new_receipt", default=False)) for item in items)
        for items in groups.values()
    )


def _shadow_trigger(state: Mapping[str, object]) -> bool:
    tokens = _int(_first(state, "shadow_transcript_tokens", "shadow_tokens", default=0))
    boundaries = _first(state, "shadow_boundaries", "shadow_transcript_boundaries", default=0)
    boundary_count = len(boundaries) if _sequence(boundaries) else _int(boundaries)
    return tokens > 12000 and boundary_count >= 2


def _g2_trigger(state: Mapping[str, object], name: str) -> bool:
    if _bool(state.get(name)):
        return True
    aliases = {
        "cross_invalidating_workstreams": ("cross_invalidating", "workstreams_cross_invalidate"),
        "background_operation": ("background_op_coexisting", "background_job_coexisting", "active_background_job"),
        "conflicting_evidence": ("evidence_conflict", "conflicting_current_evidence"),
    }
    if any(_bool(state.get(alias)) for alias in aliases.get(name, ())):
        return True
    if name == "background_operation":
        jobs = _first(state, "background_jobs", "jobs", default=())
        return any(_status(job) not in {"DONE", "CLOSED", "INACTIVE"}
                   for job in _as_list(jobs) if job is not None)
    if name == "conflicting_evidence":
        rows = _as_list(_first(state, "evidence", "relevant_receipts", default=()))
        by_id: dict[str, set[str]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            identity = str(_first(row, "id", "evidence_id", "receipt_id", default=""))
            value = _status(row)
            if identity:
                by_id.setdefault(identity, set()).add(value)
        return any(any(item in _PASS for item in values) and any(item in _FAIL for item in values)
                   for values in by_id.values())
    return False


def _failed_cycles(state: Mapping[str, object]) -> int:
    value = _first(state, "failed_edit_retest_cycles", "distinct_failed_edit_retest_cycles",
                   default=0)
    if _sequence(value):
        return len(value)
    if isinstance(value, Mapping):
        return _int(_first(value, "count", "total", default=0))
    return _int(value)


def escalation_triggers(state: Mapping[str, object] | None = None, *, current_gear: str = "g1") -> tuple[str, ...]:
    """Return deterministic escalation reasons for one atomic observation."""
    values = state if isinstance(state, Mapping) else {}
    if current_gear not in {"g0", "g1", "g2", "g3"}:
        raise GearError(f"unknown current gear: {current_gear}")
    reasons: list[str] = []
    if current_gear == "g1":
        if len(_meaningful_product_files(values)) > 2:
            reasons.append("scope>2_meaningfully_edited_product_files")
        if _working_set_tokens(values) > 6000:
            reasons.append("working_set>6000_tokens")
        failed = _failed_cycles(values)
        if failed >= 2:
            reasons.append("failed_edit_retest_cycles>=2")
        if _previous_pass_now_failed(values):
            reasons.append("previously_passing_applicable_check_now_fails")
        if _same_normalized_read_or_test(values):
            reasons.append("same_normalized_read/test>=3_of_4_without_new_evidence")
        if _shadow_trigger(values):
            reasons.append("shadow_transcript>12000_tokens_for_2_boundaries")
    elif current_gear == "g2":
        if _g2_trigger(values, "cross_invalidating_workstreams"):
            reasons.append("cross_invalidating_workstreams")
        if _g2_trigger(values, "background_operation"):
            reasons.append("background_operation_coexisting")
        if _g2_trigger(values, "conflicting_evidence"):
            reasons.append("conflicting_evidence")
        failed = _failed_cycles(values)
        if failed >= 4:
            reasons.append("additional_failed_edit_retest_cycles>=2")
    return tuple(reasons)


def _ticket(frame: Mapping[str, object]) -> object:
    value = _first(frame, "ticket", "task_brief", "brief", "msg", "objective", default="")
    return _deep(value)


def _revision_map(frame: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("current_revisions", "artifact_revisions", "product_files"):
        value = frame.get(key)
        if isinstance(value, Mapping):
            return value
    binding = _mapping(frame.get("binding"))
    if binding is not None and isinstance(binding.get("product_files"), Mapping):
        return binding["product_files"]
    return {}


def _coverage(row: Mapping[str, object], revisions: Mapping[str, object]) -> tuple[str, str]:
    explicit = row.get("coverage") or row.get("coverage_status")
    if isinstance(explicit, str) and explicit.strip():
        normalized = explicit.strip().lower()
        if normalized in {"complete", "full", "current_complete"}:
            return "complete", "CURRENT_COMPLETE"
        if normalized in {"partial", "excerpt", "incomplete"}:
            return "partial", "CURRENT_PARTIAL"
        if normalized in {"stale", "historical"}:
            return "stale", "STALE_REVISION"
    path = str(row.get("path") or "")
    revision = row.get("revision", row.get("file_revision", row.get("source_revision")))
    if row.get("freshness") in {"stale", "historical"} or row.get("stale") is True:
        return "stale", "STALE_REVISION"
    if path and revision is not None and path in revisions and revisions[path] != revision:
        return "stale", "STALE_REVISION"
    omitted = _int(row.get("omitted_chars", row.get("omitted_chars_in_selection", 0)))
    more = row.get("more_content_available", row.get("more_lines_available", False))
    total = row.get("total_lines")
    end = row.get("selected_end_line", row.get("end_line"))
    if omitted == 0 and not _bool(more) and (total is None or end is None or _int(end) >= _int(total)):
        return "complete", "CURRENT_COMPLETE"
    return "partial", "CURRENT_PARTIAL"


def _source_spans(frame: Mapping[str, object], context: Mapping[str, object]) -> list[dict]:
    raw = ()
    for name in ("current_source_spans", "relevant_source_spans", "source_spans",
                 "recent_reads", "current_sources", "observed_sources"):
        value = frame.get(name)
        if value not in (None, [], {}):
            raw = value
            break
    rows = [row for row in _as_list(raw) if isinstance(row, Mapping)]
    revisions = _revision_map(frame)
    result: list[dict] = []
    for raw_row in rows:
        row = copy.deepcopy(dict(raw_row))
        path = str(row.get("path") or row.get("requested_path") or "")
        if path:
            row["path"] = path.replace("\\", "/").lstrip("./")
        coverage, marker = _coverage(row, revisions)
        row["coverage"] = coverage
        row["coverage_marker"] = marker
        result.append(row)
    context_rows = context.get("source_spans")
    if not result and _sequence(context_rows):
        for row in context_rows:
            if isinstance(row, Mapping):
                item = copy.deepcopy(dict(row))
                coverage, marker = _coverage(item, revisions)
                item["coverage"] = coverage
                item["coverage_marker"] = marker
                result.append(item)
    path = context.get("path")
    source = context.get("source")
    if isinstance(path, str) and path and isinstance(source, str) and not any(
            row.get("path") == path for row in result):
        lines = source.splitlines()
        result.append({"path": path, "start_line": 1 if lines else 0,
                       "end_line": len(lines), "selected_end_line": len(lines),
                       "revision": revisions.get(path, context.get("current_revision")),
                       "coverage": "complete", "coverage_marker": "CURRENT_COMPLETE",
                       "content": source})
    return result


def _current_diff(frame: Mapping[str, object], context: Mapping[str, object]) -> object:
    value = None
    for name in ("current_diff", "diff", "current_changes", "recent_changes", "changes"):
        candidate = frame.get(name)
        if candidate not in (None, "", [], {}):
            value = candidate
            break
    if value in (None, "", [], {}):
        value = context.get("current_diff", "")
    return _deep(value if value is not None else "")


def _last_action(frame: Mapping[str, object]) -> object:
    value = _first(frame, "last_action", "action", "latest_action", default=None)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return _deep(value)
    action = copy.deepcopy(dict(value))
    outcome = _first(action, "exact_outcome", "outcome", "result_class", "result", "status",
                     "observed_result", default=None)
    if outcome is not None:
        action.setdefault("exact_outcome", copy.deepcopy(outcome))
    return action


def _check_statuses(frame: Mapping[str, object]) -> object:
    terminal = frame.get("terminal_status")
    if isinstance(terminal, Mapping):
        return {
            "status": terminal.get("public_check_status", "BLOCKED"),
            "selectors": _deep(terminal.get("selectors", ())),
            "revision": terminal.get("current_revision"),
        }
    value = _first(frame, "applicable_check_statuses", "check_statuses", "public_check_statuses",
                   default=None)
    if value is not None:
        return _deep(value)
    contract = _mapping(frame.get("completion_contract"))
    if contract is not None and contract.get("current_public_checks") is not None:
        return _deep(contract["current_public_checks"])
    value = _first(frame, "verification", "test_receipts", "recent_test_feedback", default={})
    return _deep(value)


def _open_work(frame: Mapping[str, object]) -> list:
    terminal = frame.get("terminal_status")
    if isinstance(terminal, Mapping) and "open_ids" in terminal:
        return [{"obligation_id": str(item), "source": "terminal_status"}
                for item in terminal.get("open_ids", ()) if str(item).strip()]
    raw = None
    for name in ("open_work", "pending_multi_definition_edits", "unresolved_obligations",
                 "obligations"):
        value = frame.get(name)
        if value not in (None, [], {}):
            raw = value
            break
    if isinstance(raw, Mapping):
        if _sequence(raw.get("items")):
            raw = raw["items"]
        elif _sequence(raw.get("open")):
            raw = raw["open"]
        elif any(key in raw for key in ("id", "obligation_id", "status", "state", "summary", "reason")):
            raw = [raw]
        else:
            raw = [{"id": str(key), **(dict(value) if isinstance(value, Mapping) else {"summary": value})}
                   for key, value in raw.items()]
    result = []
    for item in _as_list(raw):
        if isinstance(item, Mapping):
            status = str(_first(item, "status", "state", "assessment", default="open"))
            if status in _TERMINAL_CLOSED:
                continue
            result.append(copy.deepcopy(dict(item)))
        elif item is not None:
            result.append(item)
    if not result:
        host = _mapping(frame.get("host_closure"))
        if host is not None:
            assessments = host.get("assessments", host.get("open_work", ()))
            result = [copy.deepcopy(item) for item in _as_list(assessments)
                      if not isinstance(item, Mapping) or _status(item) not in _TERMINAL_CLOSED]
    return result


def _pending_intent(frame: Mapping[str, object]) -> object:
    direct = frame.get("pending_intent")
    if isinstance(direct, Mapping):
        return _deep(direct)
    if direct is not None:
        return _deep(direct)
    values = _first(frame, "pending_intents", "intents", default=())
    values = _as_list(values)
    return _deep(values[0]) if values else None


def _completion(frame: Mapping[str, object]) -> object:
    terminal = frame.get("terminal_status")
    if isinstance(terminal, Mapping):
        return _deep(terminal)
    value = _first(frame, "compact_completion_status", "completion_status", "atomic_completion",
                   "completion_verdict", default=None)
    if value is not None:
        return _deep(value)
    contract = _mapping(frame.get("completion_contract"))
    if contract is not None:
        checks = contract.get("current_public_checks")
        return {"status": _status(checks), "checks": _deep(checks)}
    return _deep(_first(frame, "phase", "verification", default={}))


def _working_manifest(frame: Mapping[str, object], source_spans: Sequence[Mapping[str, object]]) -> object:
    value = _first(frame, "working_set_manifest", "working_set", "retained_context",
                   "read_working_set", default=None)
    if value is not None:
        return _deep(value)
    return {"paths": [row.get("path") for row in source_spans if row.get("path")],
            "span_count": len(source_spans),
            "token_count": _working_set_tokens(frame)}


def _g1_working_set_manifest(frame: Mapping[str, object]) -> object:
    """Project the opt-in retained set without widening the legacy g1 arm."""
    manifest = frame.get("working_set_manifest")
    if manifest is None:
        return None
    from kernel_working_sets import project_g1_manifest
    return project_g1_manifest(manifest)


def _flat_ledger(frame: Mapping[str, object]) -> list[dict]:
    raw = ()
    for name in ("requirements_ledger", "requirements", "requirement_items", "obligations"):
        value = frame.get(name)
        if value not in (None, [], {}):
            raw = value
            break
    if isinstance(raw, Mapping):
        raw = raw.get("items", raw.get("requirements", raw.get("obligations", raw)))
        if isinstance(raw, Mapping):
            raw = [{"id": str(key), **(dict(item) if isinstance(item, Mapping) else {"summary": item})}
                   for key, item in raw.items()]
    result: list[dict] = []
    for item in _as_list(raw):
        if not isinstance(item, Mapping):
            result.append({"id": str(item), "status": "open"})
            continue
        row = {}
        for key, value in item.items():
            if key in {"children", "depends_on", "dependencies"}:
                continue
            row[str(key)] = _deep(value)
        identifier = _first(row, "id", "requirement_id", "obligation_id", default=None)
        if identifier is not None:
            row["id"] = str(identifier)
        result.append(row)
    return result


def _evidence_map(frame: Mapping[str, object], ledger: Sequence[Mapping[str, object]]) -> object:
    direct = _first(frame, "requirement_evidence_map", "req_evidence_map", "requirement_to_evidence",
                    default=None)
    if direct is not None:
        return _deep(direct)
    result: dict[str, list] = {}
    for row in ledger:
        identifier = _first(row, "id", "requirement_id", "obligation_id", default=None)
        refs = _first(row, "evidence_refs", "evidence", "receipts", default=())
        if identifier is not None and refs not in (None, [], {}):
            result[str(identifier)] = _deep(_as_list(refs))
    return result


def _relevant_receipts(frame: Mapping[str, object]) -> list:
    value = ()
    for name in ("relevant_receipts", "receipts", "test_receipts", "recent_test_feedback"):
        candidate = frame.get(name)
        if candidate not in (None, [], {}):
            value = candidate
            break
    return _deep(_as_list(value))


def _retry_counters(frame: Mapping[str, object]) -> object:
    value = None
    for name in ("retry_no_progress_counters", "retry_counters", "counters"):
        candidate = frame.get(name)
        if candidate not in (None, [], {}):
            value = candidate
            break
    if value in (None, [], {}):
        progress = _mapping(frame.get("frame_series"))
        value = progress.get("progress", {}) if progress is not None else {}
    return _deep(value if value is not None else {})


def _unresolved_questions(frame: Mapping[str, object]) -> object:
    return _deep(_first(frame, "unresolved_questions", "open_questions", default=[]))


def project_frame(frame: Mapping[str, object], gear: str, *, context: Mapping[str, object] | None = None,
                  requirements_ledger: object = None) -> dict:
    """Project one complete durable frame into G0/G1/G2/G3.

    ``off`` and ``g3`` are exact deep copies.  The copy makes the API safe for
    callers that subsequently redact presentation-only fields; the runtime's
    OFF path does not call this function at all and therefore preserves the
    original object/serialization path byte-for-byte.
    """
    if not isinstance(frame, Mapping):
        raise TypeError("gear projection requires a mapping frame")
    requested = gear_mode(gear) if isinstance(gear, str) and gear in VALID_MODES else gear
    if requested == "adaptive":
        requested = str(frame.get("gear", entry_gear(frame)))
    if requested == "off" or requested == "g3":
        return copy.deepcopy(dict(frame))
    if requested not in GEAR_ORDER:
        raise GearError(f"unknown projection gear: {gear}")
    ctx = dict(context) if isinstance(context, Mapping) else {}
    spans = _source_spans(frame, ctx)
    projected: dict[str, object] = {
        "schema": SCHEMA,
        "gear": requested,
        "task_id": _deep(frame.get("task_id")),
        "ticket": _ticket(frame),
        "current_source_spans": spans,
        "current_diff": _current_diff(frame, ctx),
        "last_action": _last_action(frame),
        "applicable_check_statuses": _check_statuses(frame),
        "open_work": _open_work(frame),
        "pending_intent": _pending_intent(frame),
        "completion_status": _completion(frame),
    }
    # These are host facts needed by the source/evidence binding and do not
    # duplicate the durable kernel object itself.
    revisions = _revision_map(frame)
    if revisions:
        projected["current_revisions"] = _deep(revisions)
    if requested == "g0":
        return {
            "schema": SCHEMA, "gear": "g0", "task_id": _deep(frame.get("task_id")),
            "ticket": _ticket(frame), "completion_status": _completion(frame),
            "answer_only": True,
        }
    if requested == "g1":
        working_set_manifest = _g1_working_set_manifest(frame)
        if working_set_manifest is not None:
            projected["working_set_manifest"] = working_set_manifest
    if requested == "g2":
        projected.update({
            "working_set_manifest": _working_manifest(frame, spans),
            "relevant_receipts": _relevant_receipts(frame),
            "retry_no_progress_counters": _retry_counters(frame),
            "unresolved_questions": _unresolved_questions(frame),
            "requirement_evidence_map": {},
        })
        ledger = _flat_ledger(frame) if requirements_ledger is not False and requirements_ledger_enabled() else None
        if (requirements_ledger is not False and requirements_ledger is not None
                and requirements_ledger_enabled()):
            ledger = _flat_ledger({"requirements_ledger": requirements_ledger})
        if ledger is not None:
            projected["requirements_ledger"] = ledger
        projected["requirement_evidence_map"] = _evidence_map(
            frame, projected.get("requirements_ledger", []))
    if requested in {"g1", "g2"}:
        history_findings = _g1_history_findings(frame)
        if history_findings is not None:
            projected["history_findings"] = history_findings
        settled_findings = _g1_settled_findings(frame)
        if settled_findings is not None:
            projected["settled_findings"] = settled_findings
    return projected


def _g1_history_findings(frame: Mapping[str, object]) -> object:
    """r44 F4 R5: carry the small, host-owned, already budgeted reverse-history
    payload through g1/g2 (it was silently dropped by the allowlist).  Opt-in
    only: with STATEFRAME_HISTORY_FINDINGS off the projection is unchanged."""
    value = frame.get("history_findings")
    if value in (None, {}, []):
        return None
    from kernel_stateframe_history_findings import mode as history_findings_mode
    if history_findings_mode() != "on":
        return None
    return _deep(value)


def _g1_settled_findings(frame: Mapping[str, object]) -> object:
    """Carry the selected r43 replay payload through the g1/g2 allowlist."""
    value = frame.get("settled_findings")
    if value in (None, {}, []):
        return None
    from kernel_stateframe_settled_findings import mode as settled_findings_mode
    if settled_findings_mode() != "on":
        return None
    return _deep(value)


def render_projection(frame: Mapping[str, object], gear: str, *, context: Mapping[str, object] | None = None,
                      requirements_ledger: object = None) -> str:
    return "[StateFrame]\n" + _canonical(project_frame(
        frame, gear, context=context, requirements_ledger=requirements_ledger)).decode("utf-8")


def _token_count(value: object) -> int:
    if isinstance(value, str):
        text = value
    else:
        text = _canonical(value).decode("utf-8")
    return int(math.ceil(len(text.encode("utf-8")) / 4)) if text else 0


def token_accounting(frame: Mapping[str, object]) -> dict[str, object]:
    """Return deterministic per-block accounting; block totals always reconcile."""
    if not isinstance(frame, Mapping):
        raise TypeError("token accounting requires a mapping")
    gear = str(frame.get("gear", "g3"))
    budget_projection = frame.get("schema") == "stateframe-frame-budget/1"
    if gear == "g3" and not budget_projection:
        blocks = {"full_frame": _token_count(frame)}
    else:
        block_names = [
            "ticket", "current_source_spans", "current_diff", "last_action",
            "applicable_check_statuses", "open_work", "pending_intent", "completion_status",
        ]
        if gear == "g2":
            block_names.extend([
                "working_set_manifest", "requirements_ledger", "requirement_evidence_map",
                "relevant_receipts", "retry_no_progress_counters", "unresolved_questions",
            ])
        if gear == "g1" and "working_set_manifest" in frame:
            from kernel_working_sets import enabled as working_sets_enabled
            if working_sets_enabled():
                block_names.append("working_set_manifest")
        if gear in {"g1", "g2"} and "history_findings" in frame and not budget_projection:
            block_names.append("history_findings")
        if gear in {"g1", "g2"} and "settled_findings" in frame and not budget_projection:
            block_names.append("settled_findings")
        if budget_projection:
            # Budget projections add typed phase/truth blocks.  Keep this list
            # conditional so legacy projections retain their exact block order
            # and totals while the new arm uses the same accountant.
            block_names.extend([
                "phase", "truth", "assumptions", "request", "repo_map", "needed_files",
                "source_slices", "where_used", "requested_files", "working_set", "edit_receipts",
                "verification_receipts", "done_allowed",
            ])
        blocks = {name: _token_count(frame.get(name)) for name in block_names if name in frame}
    order = list(blocks)
    total = sum(blocks.values())
    base = {"ticket", "current_source_spans", "current_diff", "relevant_receipts"}
    scaffold = sum(value for name, value in blocks.items() if name not in base)
    return {
        "schema": "stateframe-gear-token-accounting/1",
        "gear": gear,
        "block_order": order,
        "blocks": blocks,
        "total_tokens": total,
        "scaffold_tokens": scaffold,
        "accounting_complete": total == sum(blocks.values()),
    }


def component_allowed(gear: str, component: str) -> bool:
    """Return whether an optional derived presentation block belongs in a gear."""
    if gear in {"off", "g3"}:
        return True
    if gear == "g0":
        return component in {"frame", "source", "questions"}
    if gear == "g1":
        return component in {"frame", "source", "observed_source", "questions"}
    if gear == "g2":
        return component in {"frame", "source", "observed_source", "retained_source",
                             "questions", "repair_packet"}
    raise GearError(f"unknown component gear: {gear}")


def _candidate(frame: Mapping[str, object]) -> object:
    value = _first(frame, "candidate", "candidate_state", "binding", default=None)
    if value is None:
        value = _first(frame, "artifact_revisions", "current_revisions", default=None)
    return _deep(value)


def _obligations(frame: Mapping[str, object]) -> object:
    for name in ("open_work", "obligations", "requirements_ledger"):
        value = frame.get(name)
        if value not in (None, [], {}):
            return _deep(value)
    return []


def _evidence(frame: Mapping[str, object]) -> object:
    for name in ("evidence", "relevant_receipts", "receipts", "test_receipts"):
        value = frame.get(name)
        if value not in (None, [], {}):
            return _deep(value)
    return []


def _evidence_freshness(rows: object) -> dict[str, list[str]]:
    result = {"current": [], "stale": []}
    for row in _as_list(rows):
        if not isinstance(row, Mapping):
            continue
        identity = str(_first(row, "id", "evidence_id", "receipt_id", "reference", default=""))
        if not identity:
            continue
        stale = row.get("stale") is True or str(row.get("freshness", "")).lower() in {"stale", "historical"}
        result["stale" if stale else "current"].append(identity)
    return result


def _counters(frame: Mapping[str, object]) -> object:
    return _retry_counters(frame)


def mandatory_object_manifest(frame: Mapping[str, object], *, counters: object = None) -> dict[str, object]:
    """Identify all objects that a promotion boundary must carry forward."""
    candidate = _candidate(frame)
    obligations = _obligations(frame)
    evidence = _evidence(frame)
    counter_value = _deep(_counters(frame) if counters is None else counters)
    return {
        "schema": "stateframe-gear-mandatory-manifest/1",
        "objects": {
            "candidate": {"present": candidate is not None, "digest": _digest(candidate)},
            "obligations": {"present": obligations not in (None, [], {}), "digest": _digest(obligations)},
            "evidence": {"present": evidence not in (None, [], {}), "digest": _digest(evidence)},
            "counters": {"present": counter_value not in (None, [], {}), "digest": _digest(counter_value)},
        },
        "evidence_by_freshness": _evidence_freshness(evidence),
    }


def promotion_snapshot(frame: Mapping[str, object], *, from_gear: str, to_gear: str,
                       reason: str, step: int, counters: object = None) -> dict[str, object]:
    if from_gear not in GEAR_ORDER or to_gear not in GEAR_ORDER:
        raise GearError("promotion snapshot requires known gears")
    if GEAR_ORDER[to_gear] <= GEAR_ORDER[from_gear]:
        raise GearError("promotion must be monotone")
    candidate = _candidate(frame)
    obligations = _obligations(frame)
    evidence = _evidence(frame)
    counter_value = _deep(_counters(frame) if counters is None else counters)
    manifest = mandatory_object_manifest(frame, counters=counter_value)
    return {
        "schema": PROMOTION_SCHEMA,
        "from": from_gear, "to": to_gear, "reason": str(reason), "step": _int(step),
        "candidate": candidate, "obligations": obligations, "evidence": evidence,
        "counters": counter_value,
        "evidence_by_freshness": _deep(manifest["evidence_by_freshness"]),
        "mandatory_object_manifest": manifest,
    }


def _verify_digest(value: object, expected: object, name: str) -> None:
    if expected is not None and expected != _digest(value):
        raise PromotionReplayError(f"promotion {name} changed before replay")


def replay_promotion(snapshot: Mapping[str, object]) -> dict[str, object]:
    """Replay a promotion using only its atomic snapshot; reject silent loss."""
    if not isinstance(snapshot, Mapping) or snapshot.get("schema") != PROMOTION_SCHEMA:
        raise PromotionReplayError("invalid promotion snapshot")
    from_gear, to_gear = snapshot.get("from"), snapshot.get("to")
    if (from_gear not in GEAR_ORDER or to_gear not in GEAR_ORDER
            or GEAR_ORDER[to_gear] <= GEAR_ORDER[from_gear]):
        raise PromotionReplayError("promotion is not monotone")
    manifest = snapshot.get("mandatory_object_manifest")
    if not isinstance(manifest, Mapping):
        raise PromotionReplayError("promotion is missing the mandatory-object manifest")
    objects = manifest.get("objects")
    if not isinstance(objects, Mapping):
        raise PromotionReplayError("promotion manifest is missing object digests")
    values = {
        "candidate": _deep(snapshot.get("candidate")),
        "obligations": _deep(snapshot.get("obligations", [])),
        "evidence": _deep(snapshot.get("evidence", [])),
        "counters": _deep(snapshot.get("counters", {})),
    }
    for name, value in values.items():
        row = objects.get(name)
        if not isinstance(row, Mapping) or "digest" not in row:
            raise PromotionReplayError(f"promotion manifest lacks {name}")
        _verify_digest(value, row.get("digest"), name)
    freshness = _evidence_freshness(values["evidence"])
    expected_freshness = manifest.get("evidence_by_freshness")
    if expected_freshness is not None and freshness != expected_freshness:
        raise PromotionReplayError("promotion freshness classification changed")
    return {
        "candidate": values["candidate"], "obligations": values["obligations"],
        "evidence": values["evidence"], "counters": values["counters"],
        "evidence_by_freshness": freshness,
        "from": from_gear, "to": to_gear,
        "reason": snapshot.get("reason"), "step": snapshot.get("step"),
    }


replay_lossless_promotion = replay_promotion


class GearController:
    """Host-only monotone gear selector with atomic promotion journal hooks."""

    def __init__(self, configured: str | None = None, *, context: Mapping[str, object] | None = None,
                 journal: object = None, step: int = 0):
        self.configured_mode = gear_mode(configured)
        self.context = copy.deepcopy(dict(context)) if isinstance(context, Mapping) else {}
        self.entry = choose_entry(self.context)
        self.entry_gear = self.entry.gear
        self.entry_reason = self.entry.reason
        self.current = self.entry_gear if self.configured_mode == "adaptive" else self.configured_mode
        self.auto_promote = self.configured_mode == "adaptive"
        self.counters: dict[str, object] = {}
        self.pending_reason: str | None = None
        self.last_promotion: dict[str, object] | None = None
        self._entry_recorded = False
        self.journal = journal
        if journal is not None:
            self.record_entry(step=step)

    @property
    def gear(self) -> str:
        return self.current

    def entry_payload(self, *, step: int = 0) -> dict[str, object]:
        return {
            "configured_mode": self.configured_mode,
            "gear": self.current,
            "entry_gear": self.entry_gear,
            "entry_reason": self.entry_reason,
            "entry_facts": _deep(self.entry.facts),
            "step": _int(step),
        }

    def record_entry(self, *, step: int = 0) -> dict[str, object] | None:
        if self._entry_recorded:
            return None
        payload = self.entry_payload(step=step)
        if self.journal is not None and hasattr(self.journal, "emit"):
            self.journal.emit("gear_entry", **payload)
        self._entry_recorded = True
        return payload

    def promote(self, to_gear: str, *, reason: str, step: int, state: Mapping[str, object] | None = None,
                atomic_boundary: bool = True) -> dict[str, object] | None:
        if to_gear not in GEAR_ORDER:
            raise GearError(f"unknown target gear: {to_gear}")
        if GEAR_ORDER[to_gear] <= GEAR_ORDER[self.current]:
            return None
        if not atomic_boundary:
            raise AtomicBoundaryError("gear promotion requires an atomic event boundary")
        source = state if isinstance(state, Mapping) else {}
        if not self.counters:
            extracted = _counters(source)
            if isinstance(extracted, Mapping):
                self.counters = copy.deepcopy(dict(extracted))
        snapshot = promotion_snapshot(
            source, from_gear=self.current, to_gear=to_gear, reason=reason,
            step=step, counters=self.counters)
        payload = {
            "from": self.current, "to": to_gear, "reason": str(reason), "step": _int(step),
            "snapshot": snapshot,
            "mandatory_object_manifest": _deep(snapshot["mandatory_object_manifest"]),
            "counters": _deep(self.counters),
        }
        old = self.current
        self.current = to_gear
        self.pending_reason = None
        self.last_promotion = _deep(payload)
        if self.journal is not None and hasattr(self.journal, "emit"):
            self.journal.emit("gear_promotion", **payload)
        return payload

    def observe(self, state: Mapping[str, object] | None = None, *, step: int = 0,
                atomic_boundary: bool = True) -> dict[str, object] | None:
        values = state if isinstance(state, Mapping) else {}
        extracted = _retry_counters(values)
        if extracted:
            self.counters = copy.deepcopy(dict(extracted)) if isinstance(extracted, Mapping) else {
                "value": copy.deepcopy(extracted)}
        reasons = escalation_triggers(values, current_gear=self.current) if self.current in {"g1", "g2"} else ()
        if self.current == "g0" and (reasons or _open_work(values) or _last_action(values) is not None):
            reasons = ("g0_requires_local_frame",)
        if not reasons:
            return None
        self.pending_reason = reasons[0]
        if not self.auto_promote or not atomic_boundary or self.current == "g3":
            return None
        target = "g1" if self.current == "g0" else "g2" if self.current == "g1" else "g3"
        return self.promote(target, reason=self.pending_reason, step=step,
                            state=values, atomic_boundary=True)

    def project(self, frame: Mapping[str, object], *, context: Mapping[str, object] | None = None) -> dict:
        return project_frame(frame, self.current, context=context)

    def account(self, projection: Mapping[str, object]) -> dict[str, object]:
        return token_accounting(projection)

    def restore(self, events: Sequence[Mapping[str, object]]) -> None:
        """Restore controller presentation state from a trusted journal prefix."""
        for event in events:
            if not isinstance(event, Mapping):
                continue
            kind = event.get("kind")
            if kind == "gear_entry":
                self._entry_recorded = True
                if event.get("entry_gear") in GEAR_ORDER:
                    self.entry_gear = str(event["entry_gear"])
                if event.get("gear") in GEAR_ORDER:
                    self.current = str(event["gear"])
                if event.get("entry_reason"):
                    self.entry_reason = str(event["entry_reason"])
            elif kind == "gear_promotion":
                target = event.get("to")
                if target not in GEAR_ORDER or GEAR_ORDER[target] < GEAR_ORDER[self.current]:
                    raise GearError("journal contains a non-monotone gear promotion")
                self.current = str(target)
                counters = event.get("counters")
                if isinstance(counters, Mapping):
                    self.counters = copy.deepcopy(dict(counters))
                snapshot = event.get("snapshot")
                if isinstance(snapshot, Mapping):
                    replay_promotion(snapshot)


def gear_entry_event(controller: GearController, *, step: int = 0) -> dict[str, object]:
    return controller.entry_payload(step=step)


__all__ = [
    "AtomicBoundaryError", "EntryDecision", "GEAR_ORDER", "GearController", "GearError",
    "PromotionReplayError", "PUBLIC_GEARS", "REQUIREMENTS_LEDGER_ENV", "SCHEMA",
    "VALID_MODES", "choose_entry", "component_allowed", "enabled", "entry_gear", "entry_rule",
    "escalation_triggers", "gear_entry_event", "gear_mode", "mandatory_object_manifest", "mode", "project_frame",
    "promotion_snapshot", "render_projection", "replay_lossless_promotion", "replay_promotion",
    "requirements_ledger_enabled", "requirements_ledger_mode", "token_accounting",
]
