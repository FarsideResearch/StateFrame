"""Completion policy grounded in current host observations.

Keep unknown, violated, and satisfied states distinct. Decisions are bound to
their source snapshot; stale decisions must not be rendered as current evidence.
Completion policy does not replace qualification or finalization.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

MODE_ENV = "STATEFRAME_UNIFIED_STOP"
DECLARED_PATH_DONE_GATE_ENV = "STATEFRAME_DECLARED_PATH_DONE_GATE"
POLICY_VERSION = "stateframe-unified-stop/2"
DECISION_EVENT = "unified_stop_decision"
RENDER_HEADER = "[Unified stop authority]"

SATISFIED = "SATISFIED"
VIOLATED = "VIOLATED"
UNKNOWN = "UNKNOWN"

CODE_VIOLATED = "UNIFIED_VIOLATED"
CODE_STOPPED = "STOPPED_INCOMPLETE"          # existing terminal (runner + finaliser honour it)
CODE_DELIVERY_BLOCKED = "UNIFIED_DELIVERY_BLOCKED"
CODE_UNKNOWN_ADMITTED = "UNIFIED_UNKNOWN_UNCERTIFIED"

# Runner gate codes that are host-measured violations (bound, executed evidence).  Every
# other gate code is a refusal WITHOUT clause evidence and is UNKNOWN.
HOST_VIOLATION_CODES = frozenset({"ORACLE_TESTS_RED", "OPEN_COUNTEREXAMPLE",
                                  "BOUND_COUNTEREXAMPLE",
                                  # r44 fix DONE/D2: N14 ship-tree smoke is a host-executed
                                  # import failure of the candidate (kernel_ship_tree_smoke).
                                  "SHIP_TREE_SMOKE"})
# r44 fix DONE/D1 -- the ONE shared vocabulary of the done path (kernel_done_challenge_once
# imports these; never re-list them).  TERMINAL_CODES are non-accepted terminals of the shared
# stop decision; V2_OWNED_CODES are refusals disposition-v2 owns end to end (its one
# recoverable witnessed refusal and its INCOMPLETE terminal).
TERMINAL_CODES = frozenset({"STOPPED_INCOMPLETE", "V2_TERMINAL"})
V2_OWNED_CODES = frozenset({"V2_DISPOSITION", "V2_TERMINAL"})
WITNESSED_VIOLATION_CODES = HOST_VIOLATION_CODES | frozenset({"V2_DISPOSITION"})
# A request carrying any of these is NEVER admitted uncertified (a code missing here would be
# silently converted into a submission -- the #181c silent-drop class).
NEVER_ADMIT_UNCERTIFIED_CODES = TERMINAL_CODES | WITNESSED_VIOLATION_CODES
# Witnessed failures the post-reducer arbiter may refuse (bounded) but never waive by cap.
NON_WAIVABLE_CODES = frozenset({"SHIP_TREE_SMOKE"})
# Edit-class tools: a menu offering one of these can repair a violated clause.
REPAIR_TOOLS = frozenset({"edit", "str_replace", "write", "file_create", "replace",
                          "replace_module"})

# The provider inventory (verified in the r44 lane report) is documentation only.  The
# providers WIRED here are the three above; the rest stay independent legacy gates whose
# refusals reach this arbiter as reason codes.
WIRED_PROVIDERS = ("host_closure_assessments", "public_test_verdict", "runner_reason_codes")


def mode(value: object = None) -> str:
    raw = os.environ.get(MODE_ENV, "off") if value is None else value
    result = str(raw or "off").strip().lower()
    if result not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return result


def enabled(runtime: object = None) -> bool:
    marker = getattr(runtime, "stateframe_unified_stop_mode", None) if runtime is not None else None
    return mode(marker) == "on"


def declared_path_done_gate_mode(value: object = None) -> str:
    raw = os.environ.get(DECLARED_PATH_DONE_GATE_ENV, "off") if value is None else value
    result = str(raw or "off").strip().lower()
    if result not in {"off", "on"}:
        raise ValueError(f"{DECLARED_PATH_DONE_GATE_ENV} must be off or on")
    return result


def declared_path_done_gate_enabled() -> bool:
    return declared_path_done_gate_mode() == "on"


@dataclass(frozen=True)
class ClauseAssessment:
    clause_id: str
    state: str
    provenance: tuple[str, ...] = ()
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in {SATISFIED, VIOLATED, UNKNOWN}:
            raise ValueError(f"unknown clause state: {self.state!r}")

    def to_dict(self) -> dict[str, object]:
        return {"clause_id": self.clause_id, "state": self.state,
                "provenance": list(self.provenance), "reason": self.reason,
                "evidence_refs": list(self.evidence_refs)}


@dataclass(frozen=True)
class StopDecision:
    snapshot_id: str
    assessment: str
    clauses: tuple[ClauseAssessment, ...]
    action: str                  # "pass" | "admit_uncertified" | "challenge" | "stop" | "delivery_blocked"
    certified: bool
    challenged_before: bool
    repair_on_menu: bool
    menu: tuple[str, ...] = ()
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def violation_ids(self) -> tuple[str, ...]:
        return tuple(row.clause_id for row in self.clauses if row.state == VIOLATED)

    @property
    def unknown_ids(self) -> tuple[str, ...]:
        return tuple(row.clause_id for row in self.clauses if row.state == UNKNOWN)

    def to_dict(self) -> dict[str, object]:
        return {"policy_version": POLICY_VERSION, "snapshot_id": self.snapshot_id,
                "assessment": self.assessment, "action": self.action,
                "certified": self.certified, "challenged_before": self.challenged_before,
                "repair_on_menu": self.repair_on_menu, "menu": list(self.menu),
                "reasons": list(self.reasons),
                "clauses": [row.to_dict() for row in self.clauses]}


def aggregate_state(clauses: Iterable[ClauseAssessment]) -> str:
    rows = tuple(clauses)
    if any(row.state == VIOLATED for row in rows):
        return VIOLATED
    if rows and all(row.state == SATISFIED for row in rows):
        return SATISFIED
    return UNKNOWN


# --------------------------------------------------------------------------- evidence admission

def _strings(value: object) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes)):
        return (str(value),) if isinstance(value, str) and value.strip() else ()
    if isinstance(value, Iterable):
        return tuple(dict.fromkeys(str(item) for item in value if str(item).strip()))
    return ()


def _host_row(item: object) -> ClauseAssessment | None:
    """A host-closure ObligationAssessment (or its to_dict) -> clause.  Production objects expose
    ``.verdict`` (SATISFIED / UNSATISFIED / UNKNOWN); ``.status`` is its alias."""
    if isinstance(item, Mapping):
        get = item.get
    else:
        def get(key, default=None, _item=item):
            return getattr(_item, key, default)
    ident = str(get("obligation_id") or get("id") or "").strip()
    if not ident:
        return None
    verdict = str(get("verdict") or get("status") or "").strip().upper()
    refs = _strings(get("evidence_refs"))
    receipts = get("receipts") or ()
    bound = bool(refs) or bool(receipts)
    provenance = (f"host_closure:{ident}",)
    reason = str(get("reason") or "")
    if verdict == "SATISFIED" and bound:
        return ClauseAssessment(ident, SATISFIED, provenance, reason, refs)
    if verdict in {"UNSATISFIED", "VIOLATED"} and bound:
        return ClauseAssessment(ident, VIOLATED, provenance, reason or "UNSATISFIED", refs)
    unbound = "" if bound else "UNBOUND_EVIDENCE"
    return ClauseAssessment(ident, UNKNOWN, provenance, reason or unbound or verdict, refs)


def _public_row(runtime: object) -> ClauseAssessment | None:
    verdict_fn = getattr(runtime, "public_test_verdict", None)
    if not callable(verdict_fn):
        return None
    try:
        verdict = verdict_fn()
    except Exception:  # noqa: BLE001 - an unavailable verdict is UNKNOWN, never a veto
        return ClauseAssessment("public.required_tests", UNKNOWN, ("public_test_verdict",),
                                "VERDICT_UNAVAILABLE")
    state = str(getattr(verdict, "state", "") or "").upper()
    refs = _strings(getattr(verdict, "supporting_evidence", ()))
    codes = ",".join(_strings(getattr(verdict, "reason_codes", ())))
    if state == SATISFIED and refs:
        return ClauseAssessment("public.required_tests", SATISFIED, ("public_test_verdict",),
                                codes, refs)
    if state == VIOLATED and refs:
        return ClauseAssessment("public.required_tests", VIOLATED, ("public_test_verdict",),
                                codes or "PUBLIC_TESTS_RED", refs)
    return ClauseAssessment("public.required_tests", UNKNOWN, ("public_test_verdict",),
                            codes or state or "UNRESOLVED", refs)


def _merge(rows: list[ClauseAssessment], row: ClauseAssessment | None) -> None:
    """Same clause from two admitted sources: agreement merges provenance, SATISFIED vs
    VIOLATED is an unresolved CONFLICT (UNKNOWN), UNKNOWN never overrides a truth value."""
    if row is None:
        return
    for index, old in enumerate(rows):
        if old.clause_id != row.clause_id:
            continue
        provenance = tuple(dict.fromkeys((*old.provenance, *row.provenance)))
        refs = tuple(dict.fromkeys((*old.evidence_refs, *row.evidence_refs)))
        if {old.state, row.state} == {SATISFIED, VIOLATED}:
            rows[index] = ClauseAssessment(old.clause_id, UNKNOWN, provenance, "CONFLICT", refs)
        elif old.state == UNKNOWN and row.state != UNKNOWN and old.reason != "CONFLICT":
            rows[index] = ClauseAssessment(old.clause_id, row.state, provenance, row.reason, refs)
        else:
            rows[index] = ClauseAssessment(old.clause_id, old.state, provenance, old.reason, refs)
        return
    rows.append(row)


_PRODUCT_EDIT_TOOLS = frozenset({
    "edit", "str_replace", "write", "file_create", "replace", "replace_module",
})
_NON_PRODUCT_PATH_PARTS = frozenset({
    ".git", ".pytest_cache", "__pycache__", "tests", "test", "probes",
    "scratch", "tmp", "private", "hidden_tests",
})


def _manifest_source_paths(value: object, *, depth: int = 0) -> tuple[str, ...]:
    if depth > 5:
        return ()
    found: list[str] = []
    if isinstance(value, Mapping):
        source_path = value.get("source_path")
        if isinstance(source_path, str) and source_path.strip():
            found.append(source_path)
        for key in ("SNAPSHOT_MANIFEST", "snapshot_manifest", "manifest", "sources", "entries", "files"):
            if key in value:
                found.extend(_manifest_source_paths(value[key], depth=depth + 1))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value[:256]:
            found.extend(_manifest_source_paths(item, depth=depth + 1))
    return tuple(found)


def _declared_path_values(runtime: object) -> list[object]:
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    row = row if isinstance(row, Mapping) else {}
    values: list[object] = []
    scopes = (getattr(runtime, "scope", None), getattr(owner, "scope", None), row.get("scope"))
    for scope in scopes:
        if not isinstance(scope, Mapping):
            continue
        declared = scope.get("declared_paths", ())
        if isinstance(declared, Mapping):
            values.extend(declared.keys())
        elif isinstance(declared, Sequence) and not isinstance(declared, (str, bytes, bytearray)):
            values.extend(declared)
        values.extend(_manifest_source_paths(scope))
    values.extend(_manifest_source_paths(row))
    for key in ("source_path",):
        if isinstance(row.get(key), str):
            values.append(row[key])
    for key in ("source_paths", "declared_paths", "editable_paths", "target_files", "files"):
        candidate = row.get(key)
        if isinstance(candidate, Mapping):
            values.extend(candidate.keys())
        elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
            values.extend(candidate)
    return values[:512]


def _safe_relative_path(value: object, base: Path) -> str | None:
    raw = str(value or "").strip().replace("\\", "/").removeprefix("./")
    if (not raw or raw.startswith("/") or "\x00" in raw
            or (len(raw) > 1 and raw[1] == ":")):
        return None
    relative = PurePosixPath(raw)
    parts = relative.parts
    if (not parts or any(part in {"..", "."} for part in parts)
            or any(part.casefold() in _NON_PRODUCT_PATH_PARTS or part.startswith(".") for part in parts)):
        return None
    normalized = relative.as_posix()
    try:
        (base / normalized).resolve().relative_to(base)
    except (OSError, ValueError):
        return None
    return normalized


def _safe_declared_paths(runtime: object) -> tuple[str, ...]:
    root = getattr(getattr(runtime, "owner", None), "root", None)
    if root is None:
        return ()
    try:
        base = Path(root).resolve()
    except (OSError, TypeError, ValueError):
        return ()
    paths = []
    for value in _declared_path_values(runtime):
        normalized = _safe_relative_path(value, base)
        if normalized and normalized not in paths:
            paths.append(normalized)
    return tuple(paths[:128])


def _applied_product_paths(runtime: object) -> set[str]:
    ledger = getattr(runtime, "step_ledger", None)
    records = getattr(ledger, "records", ())
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        return set()
    applied: set[str] = set()
    for record in records:
        if (not isinstance(record, Mapping)
                or str(record.get("tool", "")).strip() not in _PRODUCT_EDIT_TOOLS
                or str(record.get("result", "")).upper() != "APPLIED"):
            continue
        changes = record.get("changes", ())
        if not isinstance(changes, Sequence) or isinstance(changes, (str, bytes, bytearray)):
            continue
        for change in changes:
            if (isinstance(change, Mapping) and change.get("role") == "product"
                    and isinstance(change.get("path"), str)):
                applied.add(change["path"].replace("\\", "/").removeprefix("./"))
    return applied


def _no_change_justifications(value: object, declared: Sequence[str]) -> dict[str, str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return {}
    allowed = set(declared)
    result: dict[str, str] = {}
    for item in value[:128]:
        if not isinstance(item, Mapping):
            continue
        raw_path = item.get("path")
        raw_reason = item.get("reason")
        if not isinstance(raw_path, str) or not isinstance(raw_reason, str):
            continue
        path = raw_path.strip().replace("\\", "/").removeprefix("./")
        reason = raw_reason.strip()
        if path in allowed and reason and path not in result:
            result[path] = reason[:400]
    return result


def _declared_path_clause(runtime: object, no_change_paths: object = None) -> ClauseAssessment | None:
    if not declared_path_done_gate_enabled():
        return None
    paths = _safe_declared_paths(runtime)
    if not paths:
        return None
    applied = _applied_product_paths(runtime)
    justifications = _no_change_justifications(no_change_paths, paths)
    unedited = tuple(path for path in paths if path not in applied and path not in justifications)
    provenance = ("task_scope.declared_paths", "host_step_ledger")
    if unedited:
        rendered = ", ".join(unedited)
        reason = (f"UNSATISFIED; unedited=[{rendered}]; apply a product edit "
                  "or provide a nonempty no_change_paths reason for each named path.")
        return ClauseAssessment("scope.declared_paths", VIOLATED, provenance, reason)
    applied_text = ", ".join(path for path in paths if path in applied) or "none"
    justified_text = ", ".join(path for path in paths if path in justifications) or "none"
    return ClauseAssessment(
        "scope.declared_paths", SATISFIED, provenance,
        f"applied=[{applied_text}]; explicitly justified no change=[{justified_text}]",
    )


def collect_clauses(runtime: object, reason_codes: Sequence[str] = (), *,
                    no_change_paths: object = None) -> tuple[ClauseAssessment, ...]:
    rows: list[ClauseAssessment] = []
    for item in getattr(runtime, "host_closure_assessments", ()) or ():
        _merge(rows, _host_row(item))
    _merge(rows, _public_row(runtime))
    _merge(rows, _declared_path_clause(runtime, no_change_paths))
    for code in _strings(reason_codes):
        normal = code.upper()
        if normal in {CODE_VIOLATED, CODE_STOPPED, CODE_DELIVERY_BLOCKED, CODE_UNKNOWN_ADMITTED}:
            continue                                    # our own codes are not evidence
        state = VIOLATED if normal in HOST_VIOLATION_CODES else UNKNOWN
        _merge(rows, ClauseAssessment(f"gate.{normal.lower()}", state, ("runner_gate",), normal))
    return tuple(rows)


# --------------------------------------------------------------------------- the decision

def _menu(runtime: object) -> tuple[str, ...]:
    value = getattr(runtime, "completion_menu", ()) or ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item).strip().lower() for item in value if str(item).strip())
    return ()


def _binding(runtime: object) -> object:
    try:
        return runtime.binding()
    except Exception:  # noqa: BLE001
        return None


def snapshot_id(runtime: object, clauses: Sequence[ClauseAssessment]) -> str:
    payload = {"binding": _binding(runtime),
               "clauses": [[row.clause_id, row.state] for row in clauses]}
    return "unified:" + hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:24]


def _repeatable_declared_path_violation(clauses: Sequence[ClauseAssessment]) -> bool:
    violated = [row for row in clauses if row.state == VIOLATED]
    return len(violated) == 1 and violated[0].clause_id == "scope.declared_paths"


def _ledger(runtime: object) -> dict[str, object]:
    ledger = getattr(runtime, "_unified_stop_state", None)
    if not isinstance(ledger, dict):
        ledger = {}
        try:
            setattr(runtime, "_unified_stop_state", ledger)
        except (AttributeError, TypeError):
            pass
    ledger.setdefault("challenged", [])
    return ledger


def evaluate(runtime: object, *, admissible: bool = True, reason_codes: Sequence[str] = (),
             delivery_feasible: bool = True, no_change_paths: object = None) -> StopDecision:
    """Pure: the SAME function render and dispatch use (no ledger mutation)."""
    clauses = collect_clauses(runtime, reason_codes, no_change_paths=no_change_paths)
    snap = snapshot_id(runtime, clauses)
    assessment = aggregate_state(clauses)
    menu = _menu(runtime)
    repair = bool(set(menu) & REPAIR_TOOLS) or not menu
    challenged = snap in set(_ledger(runtime).get("challenged") or ())
    repeatable_declared_path = _repeatable_declared_path_violation(clauses)
    if assessment == VIOLATED:
        action = "challenge" if repair and (not challenged or repeatable_declared_path) else "stop"
    elif admissible:
        action = "pass"
    elif delivery_feasible:
        action = "admit_uncertified"
    else:
        action = "delivery_blocked"
    return StopDecision(snap, assessment, clauses, action,
                        certified=(assessment == SATISFIED), challenged_before=challenged,
                        repair_on_menu=repair, menu=menu,
                        reasons=tuple(row.reason for row in clauses if row.state != SATISFIED))


def _detail(decision: StopDecision) -> str:
    ids = ", ".join(decision.violation_ids)
    declared = next((row.reason for row in decision.clauses
                     if row.clause_id == "scope.declared_paths" and row.state == VIOLATED), "")
    suffix = f"; task obligation: {declared}" if declared else ""
    if decision.action == "challenge":
        return (f"{RENDER_HEADER} done challenged once at this snapshot: known violation(s) "
                f"{ids}{suffix}. Repair them (edit/str_replace), then run_tests.")
    if decision.action == "stop":
        return (f"{CODE_STOPPED}: {RENDER_HEADER} known violation(s) {ids} remain{suffix} "
                "and no further repair is possible here; the attempt stops unsubmitted.")
    if decision.action == "delivery_blocked":
        return (f"{RENDER_HEADER} no clause is violated, but the completion delivery was not "
                "prepared; the gate refusal stands.")
    return ""


def transform(runtime: object, admissible: bool, kwargs: dict) -> tuple[bool, dict]:
    """The completion_decision hook: rewrite the caller's admissibility from the stop decision."""
    codes = tuple(kwargs.get("reason_codes", ()) or ())
    upper = {str(code).strip().upper() for code in codes}
    owned = upper & (TERMINAL_CODES | V2_OWNED_CODES)
    if owned:
        # r44 fix DONE/D1: a terminal or a disposition-v2 refusal is owned by its producer;
        # unified stop passes it through unchanged (refusal + codes + detail preserved).
        _emit_pass_through(runtime, admissible, codes, sorted(owned))
        return bool(admissible), dict(kwargs)
    continuation = getattr(runtime, "continuation", None)
    feasible = continuation is None or callable(kwargs.get("commit"))
    no_change_paths = kwargs.get("no_change_paths")
    decision = evaluate(runtime, admissible=bool(admissible), reason_codes=codes,
                        delivery_feasible=feasible, no_change_paths=no_change_paths)
    ledger = _ledger(runtime)
    out = dict(kwargs)
    new_admissible = bool(admissible)
    if decision.action == "challenge":
        ledger["challenged"] = list(dict.fromkeys(
            (*list(ledger.get("challenged") or ()), decision.snapshot_id)))
        new_admissible = False
        out["reason_codes"] = tuple(dict.fromkeys((*codes, CODE_VIOLATED)))
        out["detail"] = _detail(decision)
    elif decision.action == "stop":
        new_admissible = False
        out["reason_codes"] = tuple(dict.fromkeys((*codes, CODE_VIOLATED, CODE_STOPPED)))
        out["detail"] = _detail(decision)
        try:
            setattr(runtime, "system_outcome", CODE_STOPPED)
        except (AttributeError, TypeError):
            pass
    elif decision.action == "admit_uncertified" and upper & NEVER_ADMIT_UNCERTIFIED_CODES:
        # Defence in depth (unreachable while every witnessed code maps to VIOLATED): a
        # witnessed/terminal refusal is never converted into a submission.
        pass
    elif decision.action == "admit_uncertified":
        new_admissible = True
        out["reason_codes"] = ()
        out["detail"] = ""
    elif decision.action == "delivery_blocked":
        out["reason_codes"] = tuple(dict.fromkeys((*codes, CODE_DELIVERY_BLOCKED)))
        out["detail"] = "\n".join(filter(None, (str(kwargs.get("detail") or ""),
                                                _detail(decision))))
    if decision.action in {"challenge", "stop"}:
        declared = next((row for row in decision.clauses
                         if row.clause_id == "scope.declared_paths" and row.state == VIOLATED), None)
        if declared is not None:
            emit_gate = getattr(getattr(runtime, "journal", None), "emit", None)
            if callable(emit_gate):
                unedited = declared.reason.partition("unedited=[")[2].partition("]")[0]
                emit_gate("declared_path_done_gate_fired",
                          producer="kernel_completion_policy", dispatch=False,
                          action=decision.action, snapshot_id=decision.snapshot_id,
                          unedited_paths=[item.strip() for item in unedited.split(",") if item.strip()])
    ledger["last"] = decision.to_dict()
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit(DECISION_EVENT, producer="kernel_completion_policy", dispatch=False,
             caller_admissible=bool(admissible), admissible=new_admissible,
             policy_version=POLICY_VERSION, snapshot_id=decision.snapshot_id,
             assessment=decision.assessment, action=decision.action,
             certified=decision.certified, challenged_before=decision.challenged_before,
             repair_on_menu=decision.repair_on_menu, menu=list(decision.menu),
             reasons=list(decision.reasons),
             clauses=[row.to_dict() for row in decision.clauses])
    return new_admissible, out


def _emit_pass_through(runtime: object, admissible: object, codes: Sequence[str],
                       owned: Sequence[str]) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit(DECISION_EVENT, producer="kernel_completion_policy", dispatch=False,
             caller_admissible=bool(admissible), admissible=bool(admissible),
             policy_version=POLICY_VERSION, action="pass_through_owned",
             reasons=list(codes), owned_codes=list(owned))


def render(runtime: object) -> str:
    decision = evaluate(runtime)
    lines = [RENDER_HEADER,
             f"snapshot={decision.snapshot_id}",
             f"assessment={decision.assessment} (certified={'true' if decision.certified else 'false'})"]
    if decision.assessment == VIOLATED:
        lines.append("done now: " + ("challenged once, repair first" if decision.action == "challenge"
                                     else "stops the attempt unsubmitted (STOPPED_INCOMPLETE)"))
    else:
        lines.append("done now: submits" + ("" if decision.certified else " uncertified"))
    for row in decision.clauses:
        reason = f"; reason={row.reason}" if row.reason else ""
        lines.append(f"- {row.clause_id}: {row.state}; provenance={','.join(row.provenance)}{reason}")
    return "\n".join(lines)


__all__ = [
    "CODE_DELIVERY_BLOCKED", "CODE_STOPPED", "CODE_UNKNOWN_ADMITTED", "CODE_VIOLATED",
    "ClauseAssessment", "DECLARED_PATH_DONE_GATE_ENV", "DECISION_EVENT", "HOST_VIOLATION_CODES", "NEVER_ADMIT_UNCERTIFIED_CODES",
    "NON_WAIVABLE_CODES", "TERMINAL_CODES", "V2_OWNED_CODES", "WITNESSED_VIOLATION_CODES", "MODE_ENV", "POLICY_VERSION",
    "RENDER_HEADER", "REPAIR_TOOLS", "SATISFIED", "StopDecision", "UNKNOWN", "VIOLATED",
    "WIRED_PROVIDERS", "aggregate_state", "collect_clauses", "declared_path_done_gate_enabled",
    "declared_path_done_gate_mode", "enabled", "evaluate", "mode",
    "render", "snapshot_id", "transform",
]
