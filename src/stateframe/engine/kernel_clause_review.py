"""Bounded model-authored clause review over the existing typed witness rail.

This module is deliberately a reducer, not a completion authority.  It records
which original clauses were reviewed, asks the existing witness machinery to
admit and execute each case, and keeps model assessment separate from host
receipt status.  The completion owner decides whether the resulting episode
permits a terminal request.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_CLAUSE_REVIEW"
SCHEMA = "stateframe-clause-review/1"
MAX_MODEL_DECISIONS = 6
MAX_CASES_PER_ROUND = 8
MAX_ROUNDS = 2
MAX_REPAIRS = 2
MAX_TERMINAL_RESERVE = 1
PROPOSE_MODE = "clause_review_propose"
ASSESS_MODE = "clause_review_assess"
_OPEN_VERDICTS = frozenset({"", "UNKNOWN", "OPEN", "UNSATISFIED", "UNVERIFIED"})
_OPEN_REASONS = frozenset({
    "MISSING_CHECKS", "UNBOUND_OBSERVABLE", "NO_VERIFIER", "REQUIRED_EVIDENCE_GAP",
    "CLAUSE_COVERAGE", "COVERAGE_OPEN", "UNKNOWN", "INCONCLUSIVE", "NOT_EXECUTED",
})
_JUDGMENTS = frozenset({"SUPPORTED", "CONTRADICTED", "UNKNOWN"})
_PASS_RESULTS = frozenset({"PASS", "PASSED", "COMPLETED", "SUCCESS", "OK"})
_REPAIR_RESULTS = frozenset({"APPLIED", "PASS", "PASSED", "SUCCESS", "OK"})
_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")


def _text(value: object, limit: int = 1200) -> str:
    text = value if isinstance(value, str) else str(value or "")
    return text[:limit]


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def mode(runtime: object | None = None) -> str:
    configured = getattr(runtime, "stateframe_clause_review_mode", None) if runtime is not None else None
    value = str(configured if configured is not None else os.environ.get(MODE_ENV, "off"))
    value = value.strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    return mode(runtime) == "on"


def _emit(runtime: object, kind: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_clause_review", event_schema=SCHEMA,
             dispatch=False, **fields)


def _binding(runtime: object) -> object:
    binding = getattr(runtime, "binding", None)
    try:
        return binding() if callable(binding) else binding
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    result: dict[str, object] = {}
    for name in (
            "obligation_id", "id", "clause", "exact_clause", "clause_text", "verdict",
            "status", "reason", "reason_code", "entrypoint", "public_entrypoint",
            "check_ids", "checks", "receipts", "receipt_refs", "kind"):
        item = getattr(value, name, None)
        if item is not None:
            result[name] = item
    return result


def _assessment_rows(runtime: object) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    try:
        # Public clauses are obligations even when no failing check produced an
        # assessment.  The discriminator owns coverage truth; this reducer only
        # exposes those open rows to the bounded witness episode.
        from kernel_clause_discriminator import open_clauses
        for clause in open_clauses(runtime):
            clause_id = _text(clause.get("clause_ref", clause.get("obligation_id", "")), 240).strip()
            if not clause_id:
                continue
            rows.append({
                "obligation_id": clause_id, "clause_id": clause_id,
                "clause": _text(clause.get("clause", "")),
                "verdict": "UNKNOWN", "reason": "CLAUSE_COVERAGE",
                "entrypoint": _text(clause.get("entrypoint", ""), 240),
                "check_ids": list(_sequence(clause.get("check_ids", ()))),
                "receipt_refs": list(_sequence(clause.get("evidence_ids", ()))),
                "coverage_required": True,
            })
            seen.add(clause_id)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    collections = (
        "host_closure_assessments", "behavior_witness_assessments", "obligation_assessments",
    )
    for name in collections:
        values = getattr(runtime, name, ())
        for raw in _sequence(values):
            mapped = _mapping(raw)
            if not mapped:
                continue
            clause_id = _text(mapped.get("obligation_id", mapped.get("id", "")), 240).strip()
            if not clause_id or clause_id in seen:
                continue
            verdict = _text(mapped.get("verdict", mapped.get("status", "")), 80).strip().upper()
            reason = _text(mapped.get("reason", mapped.get("reason_code", "")), 100).strip().upper()
            if verdict not in _OPEN_VERDICTS and reason not in _OPEN_REASONS:
                continue
            clause = _text(mapped.get("clause", mapped.get("exact_clause", mapped.get("clause_text", ""))))
            if not clause:
                clause = clause_id
            rows.append({
                "obligation_id": clause_id,
                "clause_id": clause_id,
                "clause": clause,
                "verdict": verdict or "UNKNOWN",
                "reason": reason or "UNKNOWN",
                "entrypoint": _text(mapped.get("entrypoint", mapped.get("public_entrypoint", "")), 240),
                "check_ids": list(_sequence(mapped.get("check_ids", mapped.get("checks", ())))),
                "receipt_refs": list(_sequence(mapped.get("receipts", mapped.get("receipt_refs", ())))),
            })
            seen.add(clause_id)
    return tuple(rows)


def open_clauses(runtime: object) -> tuple[dict[str, object], ...]:
    """Return every original open clause, including compatibility/output rows."""
    return _assessment_rows(runtime)


def _state(runtime: object) -> dict[str, object]:
    current = getattr(runtime, "_clause_review_state", None)
    if isinstance(current, dict) and current.get("schema") == SCHEMA:
        return current
    state: dict[str, object] = {
        "schema": SCHEMA,
        "state": "IDLE",
        "episode_id": "",
        "goal_revision": "",
        "clause_ids": [],
        "clauses": [],
        "round": 0,
        "used_decisions": 0,
        "used_cases": 0,
        "used_repairs": 0,
        "terminal_reserve": MAX_TERMINAL_RESERVE,
        "receipt_ids": [],
        "assessments": [],
        "remaining_gaps": [],
        "reason": "",
    }
    setattr(runtime, "_clause_review_state", state)
    return state


def state(runtime: object) -> dict[str, object]:
    return copy.deepcopy(_state(runtime))


def _episode_key(runtime: object, rows: Sequence[Mapping[str, object]]) -> str:
    return _digest({
        "revision": _binding(runtime),
        "clauses": [(str(row.get("clause_id", "")), str(row.get("clause", ""))) for row in rows],
    })


def _typed_probe_available(tools: object) -> bool:
    for tool in _sequence(tools):
        if not isinstance(tool, Mapping) or str(tool.get("function", {}).get("name", "")) != "probe":
            continue
        function = tool.get("function")
        parameters = function.get("parameters") if isinstance(function, Mapping) else None
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        required = parameters.get("required") if isinstance(parameters, Mapping) else None
        if not isinstance(properties, Mapping) or not isinstance(required, Sequence):
            continue
        if {"mode", "episode_id"}.issubset(properties) or {
                "clause_id", "entrypoint", "input", "expected", "rationale",
        }.issubset(properties):
            return True
    return False


def _tool_names(tools: object) -> set[str]:
    return {
        str(tool.get("function", {}).get("name", ""))
        for tool in _sequence(tools)
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
    }


def _probe_budget_remaining(runtime: object) -> int | None:
    value = getattr(runtime, "probe_remaining", None)
    if type(value) is int:
        return value
    try:
        from kernel_route_budget import _budget
        remaining = _budget(runtime).get("remaining")
        return int(remaining) if type(remaining) is int else None
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def ensure_episode(runtime: object) -> dict[str, object] | None:
    if not enabled(runtime):
        return None
    rows = open_clauses(runtime)
    if not rows:
        return None
    current = _state(runtime)
    key = _episode_key(runtime, rows)
    if current.get("episode_id") != key:
        current.clear()
        current.update({
            "schema": SCHEMA,
            "state": "PROPOSE",
            "episode_id": key,
            "goal_revision": _digest(_binding(runtime)),
            "clause_ids": [str(row["clause_id"]) for row in rows],
            "clauses": copy.deepcopy(list(rows)),
            "round": 0,
            "used_decisions": 0,
            "used_cases": 0,
            "used_repairs": 0,
            "terminal_reserve": MAX_TERMINAL_RESERVE,
            "receipt_ids": [],
            "assessments": [],
            "remaining_gaps": [],
            "reason": "",
        })
        _emit(runtime, "clause_review_episode_started", episode_id=key,
               clause_ids=list(current["clause_ids"]), round=0,
               terminal_reserve=MAX_TERMINAL_RESERVE)
    return current


def _finish_unfunded(runtime: object, reason: str) -> None:
    current = _state(runtime)
    if current.get("state") in {"REVIEW_NOT_FUNDED", "EXHAUSTED"}:
        return
    current["state"] = "REVIEW_NOT_FUNDED"
    current["reason"] = reason
    current["remaining_gaps"] = list(current.get("clause_ids", ()))
    _emit(runtime, "clause_review_not_funded", episode_id=current.get("episode_id"),
           reason=reason, remaining_gaps=list(current["remaining_gaps"]))


def _review_action(runtime: object, tools: object, phase: str) -> dict[str, object] | None:
    current = ensure_episode(runtime)
    if current is None or current.get("state") != phase:
        return None
    if current.get("used_decisions", 0) >= MAX_MODEL_DECISIONS:
        current["state"] = "EXHAUSTED"
        current["reason"] = "model_decision_cap"
        current["remaining_gaps"] = list(current.get("clause_ids", ()))
        _emit(runtime, "clause_review_exhausted", episode_id=current.get("episode_id"),
               reason=current["reason"], remaining_gaps=list(current["remaining_gaps"]))
        return None
    remaining = _probe_budget_remaining(runtime)
    if remaining is not None and remaining <= 0:
        _finish_unfunded(runtime, "probe_budget_unavailable")
        return None
    if not _typed_probe_available(tools):
        _finish_unfunded(runtime, "typed_probe_not_on_current_menu")
        return None
    args: dict[str, object] = {
        "mode": PROPOSE_MODE if phase == "PROPOSE" else ASSESS_MODE,
        "episode_id": str(current["episode_id"]),
        "schema": SCHEMA,
        "clause_ids": list(current["clause_ids"]),
        "clauses": copy.deepcopy(current["clauses"]),
        "max_cases": MAX_CASES_PER_ROUND,
        "max_rounds": MAX_ROUNDS,
    }
    if phase == "ASSESS":
        args["receipt_ids"] = list(current.get("receipt_ids", ()))
        args["assembled_result"] = copy.deepcopy(getattr(runtime, "_clause_review_last_result", {}))
    return {
        "tool": "probe",
        "arguments": args,
        "reason": "review every original open clause through the existing typed witness rail",
        "clears": "the bounded clause-review episode or its truthful incomplete disposition",
    }


def next_actions(runtime: object, tools: object = ()) -> tuple[dict[str, object], ...]:
    if not enabled(runtime):
        return ()
    current = ensure_episode(runtime)
    if current is None:
        return ()
    phase = str(current.get("state", ""))
    if phase in {"PROPOSE", "ASSESS"}:
        action = _review_action(runtime, tools, phase)
        return (action,) if action is not None else ()
    if phase == "REPAIR":
        if int(current.get("used_repairs", 0) or 0) >= MAX_REPAIRS:
            current["state"] = "EXHAUSTED"
            current["reason"] = "repair_cap"
            current["remaining_gaps"] = list(current.get("clause_ids", ()))
            _emit(runtime, "clause_review_exhausted", episode_id=current.get("episode_id"),
                   reason=current["reason"], remaining_gaps=list(current["remaining_gaps"]))
            return ()
        names = _tool_names(tools)
        return tuple({
            "tool": name,
            "arguments": {},
            "reason": "apply the model-identified clause repair before reassessment",
            "clears": "the demonstrated clause mismatch",
        } for name in ("edit", "str_replace", "read") if name in names)
    return ()


def _clause_map(runtime: object) -> dict[str, dict[str, object]]:
    current = ensure_episode(runtime)
    rows = current.get("clauses", ()) if current is not None else ()
    return {str(row.get("clause_id", "")): dict(row) for row in _sequence(rows)
            if isinstance(row, Mapping) and str(row.get("clause_id", ""))}


def _case_input(case: Mapping[str, object]) -> object:
    if "input" in case:
        return copy.deepcopy(case.get("input"))
    return {
        "args": copy.deepcopy(list(_sequence(case.get("args", ())))),
        "kwargs": copy.deepcopy(dict(case.get("kwargs", {})))
        if isinstance(case.get("kwargs"), Mapping) else {},
    }


def _case_proposal(case: Mapping[str, object]) -> dict[str, object]:
    proposal = {
        "clause_id": _text(case.get("clause_id"), 240),
        "entrypoint": _text(case.get("entrypoint"), 240),
        "input": _case_input(case),
        "expected": copy.deepcopy(case.get("expected")),
        "rationale": _text(case.get("rationale"), 1000),
    }
    if "args" in case:
        proposal["args"] = copy.deepcopy(list(_sequence(case.get("args"))))
    if "kwargs" in case and isinstance(case.get("kwargs"), Mapping):
        proposal["kwargs"] = copy.deepcopy(dict(case["kwargs"]))
    if "projection" in case:
        proposal["projection"] = copy.deepcopy(case["projection"])
    return proposal


def _packet_error(current: Mapping[str, object], reason: str, **extra: object) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "episode_id": current.get("episode_id"),
        "status": "PACKET_REJECTED",
        "reason": reason,
        **extra,
    }


def execute_packet(runtime: object, packet: object, runner: object) -> dict[str, object]:
    """Admit and execute a complete model-authored witness packet."""
    current = ensure_episode(runtime)
    if current is None:
        return {"schema": SCHEMA, "status": "REVIEW_OFF_OR_NO_OPEN_CLAUSES"}
    if current.get("state") != "PROPOSE":
        return _packet_error(current, "proposal_not_expected", state=current.get("state"))
    current["used_decisions"] = int(current.get("used_decisions", 0) or 0) + 1
    if int(current.get("used_decisions", 0)) > MAX_MODEL_DECISIONS:
        current["state"] = "EXHAUSTED"
        return _packet_error(current, "model_decision_cap")
    if not isinstance(packet, Mapping):
        current["remaining_gaps"] = list(current.get("clause_ids", ()))
        return _packet_error(current, "packet_must_be_object")
    episode_id = _text(packet.get("episode_id"), 240)
    if episode_id != str(current.get("episode_id")):
        return _packet_error(current, "episode_id_mismatch")
    declared_ids = tuple(str(item) for item in _sequence(packet.get("clause_ids", ())) if str(item))
    expected_ids = tuple(str(item) for item in current.get("clause_ids", ()))
    if declared_ids != expected_ids:
        return _packet_error(current, "clause_id_set_mismatch", expected=list(expected_ids), received=list(declared_ids))
    cases = _sequence(packet.get("cases", ()))
    if not cases or len(cases) > MAX_CASES_PER_ROUND:
        return _packet_error(current, "case_count_out_of_bounds", max_cases=MAX_CASES_PER_ROUND)
    case_clause_ids = tuple(str(item.get("clause_id", "")) for item in cases
                            if isinstance(item, Mapping))
    if set(case_clause_ids) != set(expected_ids) or len(case_clause_ids) != len(expected_ids):
        return _packet_error(current, "per_clause_case_set_mismatch",
                             expected=list(expected_ids), received=list(case_clause_ids))
    clause_map = _clause_map(runtime)
    case_ids: set[str] = set()
    records: list[dict[str, object]] = []
    from kernel_behavior_witness import admit_proposal, context_for_runtime, execute_admitted

    for raw_case in cases:
        if not isinstance(raw_case, Mapping):
            return _packet_error(current, "case_must_be_object")
        case_id = _text(raw_case.get("case_id"), 240).strip()
        clause_id = _text(raw_case.get("clause_id"), 240).strip()
        if not case_id or case_id in case_ids:
            return _packet_error(current, "duplicate_or_missing_case_id")
        case_ids.add(case_id)
        row = clause_map.get(clause_id)
        if row is None:
            return _packet_error(current, "case_clause_id_not_open", clause_id=clause_id)
        entrypoint = _text(raw_case.get("entrypoint"), 240).strip()
        if not _ENTRYPOINT_RE.fullmatch(entrypoint):
            return _packet_error(current, "entrypoint_not_public_shape", case_id=case_id)
        brief, interface, _unused = context_for_runtime(runtime, row, clause_id)
        proposal = _case_proposal(raw_case)
        try:
            admission = admit_proposal(
                proposal, clause_id=clause_id, clause=str(row.get("clause", "")),
                brief=brief, interface=interface, revision=_binding(runtime),
                open_obligation=row,
                open_obligations=tuple(current.get("clauses", ())), journal=getattr(runtime, "journal", None),
            )
            if admission.get("admitted") is True:
                record = execute_admitted(runtime, admission, runner,
                                          question="clause review " + case_id)
            else:
                record = {"schema": SCHEMA, "case_id": case_id, "clause_id": clause_id,
                          "result": "UNRUNNABLE", "admission": admission}
        except Exception as error:  # public runner failures become truthful receipts, not owner crashes
            record = {
                "schema": SCHEMA,
                "case_id": case_id,
                "clause_id": clause_id,
                "result": "UNRUNNABLE",
                "reason": "host_probe_error:" + type(error).__name__,
            }
            _emit(runtime, "clause_review_case_unrunnable", episode_id=current.get("episode_id"),
                  case_id=case_id, clause_id=clause_id, error_type=type(error).__name__)
        record["case_id"] = case_id
        record["purpose"] = _text(raw_case.get("purpose", "normal"), 80)
        records.append(record)

    current["used_cases"] = int(current.get("used_cases", 0) or 0) + len(records)
    current["state"] = "ASSESS"
    current["round"] = int(current.get("round", 0) or 0) + 1
    receipt_ids = [str(record.get("receipt_id")) for record in records if str(record.get("receipt_id", ""))]
    current["receipt_ids"] = list(dict.fromkeys([*current.get("receipt_ids", ()), *receipt_ids]))
    assembled = {
        "binding": copy.deepcopy(_binding(runtime)),
        "clauses": copy.deepcopy(list(current.get("clauses", ()))),
        "cases": copy.deepcopy(records),
        "receipt_ids": list(current["receipt_ids"]),
    }
    result = {
        "schema": SCHEMA, "status": "CASES_EXECUTED", "phase": "ASSESS",
        "episode_id": current.get("episode_id"), "round": current.get("round"),
        "receipts": copy.deepcopy(records), "assembled_result": assembled,
        "open_clause_ids": list(current.get("clause_ids", ())),
    }
    setattr(runtime, "_clause_review_last_result", copy.deepcopy(result))
    _emit(runtime, "clause_review_packet_executed", episode_id=current.get("episode_id"),
           round=current.get("round"), clause_ids=list(current.get("clause_ids", ())),
           case_count=len(records), receipt_ids=list(current["receipt_ids"]))
    return result


def assess_packet(runtime: object, assessment: object) -> dict[str, object]:
    """Record the model's receipt-backed judgement without host certifying it."""
    current = ensure_episode(runtime)
    if current is None:
        return {"schema": SCHEMA, "status": "REVIEW_OFF_OR_NO_OPEN_CLAUSES"}
    if int(current.get("used_decisions", 0) or 0) >= MAX_MODEL_DECISIONS:
        current["state"] = "EXHAUSTED"
        current["reason"] = "model_decision_cap"
        current["remaining_gaps"] = list(current.get("clause_ids", ()))
        return _packet_error(current, "model_decision_cap")
    current["used_decisions"] = int(current.get("used_decisions", 0) or 0) + 1
    if current.get("state") != "ASSESS":
        return _packet_error(current, "assessment_not_expected", state=current.get("state"))
    if not isinstance(assessment, Mapping):
        return _packet_error(current, "assessment_must_be_object")
    if _text(assessment.get("episode_id"), 240) != str(current.get("episode_id")):
        return _packet_error(current, "episode_id_mismatch")
    raw_rows = _sequence(assessment.get("assessments", assessment.get("clause_assessments", ())))
    expected_ids = tuple(str(item) for item in current.get("clause_ids", ()))
    received_ids = tuple(_text(row.get("clause_id", row.get("id", "")), 240)
                         for row in raw_rows if isinstance(row, Mapping))
    if received_ids != expected_ids:
        return _packet_error(current, "assessment_clause_id_set_mismatch",
                             expected=list(expected_ids), received=list(received_ids))
    known_receipts = {str(item) for item in current.get("receipt_ids", ())}
    normalized: list[dict[str, object]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            return _packet_error(current, "assessment_row_must_be_object")
        judgment = _text(raw_row.get("judgment", ""), 40).strip().upper()
        if judgment not in _JUDGMENTS:
            return _packet_error(current, "invalid_judgment", judgment=judgment)
        refs = [str(item) for item in _sequence(
            raw_row.get("current_receipt_refs", raw_row.get("receipt_refs", ()))) if str(item)]
        if judgment != "UNKNOWN" and not refs:
            return _packet_error(current, "named_evidence_required", clause_id=raw_row.get("clause_id"))
        if not set(refs).issubset(known_receipts):
            return _packet_error(current, "unknown_receipt_reference", clause_id=raw_row.get("clause_id"))
        normalized.append({
            "clause_id": _text(raw_row.get("clause_id", raw_row.get("id", "")), 240),
            "judgment": judgment, "current_receipt_refs": refs,
            "explanation": _text(raw_row.get("explanation", "")),
            "uncovered_parts": list(_sequence(raw_row.get("uncovered_parts", ()))),
            "proposed_repair": copy.deepcopy(raw_row.get("proposed_repair")),
        })
    root_review = assessment.get("root_review")
    if not isinstance(root_review, Mapping):
        return _packet_error(current, "root_review_required")
    current["assessments"] = copy.deepcopy(normalized)
    current["remaining_gaps"] = [
        str(row["clause_id"]) for row in normalized if row["judgment"] == "UNKNOWN"
    ]
    contradicted = [row for row in normalized if row["judgment"] == "CONTRADICTED"]
    if contradicted:
        current["state"] = "REPAIR"
        current["remaining_gaps"] = list(dict.fromkeys(
            [str(row["clause_id"]) for row in contradicted] + list(current["remaining_gaps"])))
        current["pending_repairs"] = [copy.deepcopy(row.get("proposed_repair")) for row in contradicted]
        disposition = "REPAIR_REQUIRED"
    elif current["remaining_gaps"]:
        current["state"] = "SETTLED"
        current["reason"] = "unknown_clause_support"
        disposition = "REVIEW_INCOMPLETE"
    else:
        current["state"] = "SETTLED"
        current["reason"] = "model_assessed_all_clauses"
        disposition = "REVIEW_COMPLETE"
    result = {
        "schema": SCHEMA, "status": disposition, "episode_id": current.get("episode_id"),
        "state": current.get("state"), "assessments": copy.deepcopy(normalized),
        "root_review": copy.deepcopy(dict(root_review)),
        "remaining_gaps": list(current.get("remaining_gaps", ())),
        "model_assessment_only": True,
    }
    setattr(runtime, "_clause_review_assessment", copy.deepcopy(result))
    _emit(runtime, "clause_review_assessed", episode_id=current.get("episode_id"),
           judgments=[str(row["judgment"]) for row in normalized],
           remaining_gaps=list(current.get("remaining_gaps", ())), disposition=disposition)
    return result


def note_action_result(runtime: object, record: Mapping[str, object]) -> None:
    """Reopen a bounded proposal round after an admitted product repair."""
    if not enabled(runtime):
        return
    current = _state(runtime)
    if current.get("state") != "REPAIR":
        return
    tool = _text(record.get("tool", record.get("action_kind", "")), 80).lower()
    result = _text(record.get("result", record.get("result_class", "")), 80).upper()
    if tool not in {"edit", "str_replace", "replace_module", "file_create", "write"} \
            or result not in _REPAIR_RESULTS:
        return
    current["used_repairs"] = int(current.get("used_repairs", 0) or 0) + 1
    if int(current["used_repairs"]) > MAX_REPAIRS or int(current.get("round", 0) or 0) >= MAX_ROUNDS:
        current["state"] = "EXHAUSTED"
        current["reason"] = "repair_or_round_cap"
        current["remaining_gaps"] = list(current.get("clause_ids", ()))
    else:
        current["state"] = "PROPOSE"
        current["reason"] = "product_repair_admitted_recheck_required"
    _emit(runtime, "clause_review_repair_admitted", episode_id=current.get("episode_id"),
           tool=tool, result=result, state=current.get("state"),
           used_repairs=current.get("used_repairs"))


def terminal_projection(runtime: object) -> dict[str, object] | None:
    if not enabled(runtime):
        return None
    current = _state(runtime)
    if not current.get("episode_id"):
        return None
    remaining = list(current.get("remaining_gaps", ()))
    complete = current.get("state") == "SETTLED" and not remaining
    projected = {
        "schema": SCHEMA,
        "episode_id": current.get("episode_id"),
        "state": current.get("state"),
        "reviewed": complete,
        "model_assessment_only": True,
        "remaining_gaps": remaining,
        "reason": current.get("reason", ""),
    }
    if current.get("state") in {"PROPOSE", "ASSESS", "REPAIR", "EXHAUSTED", "REVIEW_NOT_FUNDED"} or remaining:
        projected.update({"truthful_exit": True, "accepted": False, "certified": False})
    return projected


def render_result(result: Mapping[str, object]) -> str:
    return "clause review " + json.dumps(dict(result), sort_keys=True, ensure_ascii=False, default=str)[:12000]


__all__ = [
    "ASSESS_MODE", "MAX_CASES_PER_ROUND", "MAX_MODEL_DECISIONS", "MAX_REPAIRS", "MAX_ROUNDS",
    "MODE_ENV", "PROPOSE_MODE", "SCHEMA", "assess_packet", "enabled", "ensure_episode",
    "execute_packet", "mode", "next_actions", "note_action_result", "open_clauses",
    "render_result", "state", "terminal_projection",
]
