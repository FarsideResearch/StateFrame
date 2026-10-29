"""Typed state gates for keeping a repair on its stated behavioral target.

QGATES is deliberately a small adapter around the StateFrame runtime.  It does
not interpret a ticket into a second plan and it never dispatches a check itself;
it exposes one typed menu action, records the route, and hands witness work to
the existing REPROFIRST/probe surfaces.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


EVENT_SCHEMA = "stateframe-qgates/1"
FLAG = "STATEFRAME_QGATES"
TOOL_NAME = "question_gate"
CHECK_ROUTE = "REPROFIRST_WITNESS_CHECK"
GATES = ("intake", "before_first_product_edit", "first_green", "before_done")
EDIT_KINDS = frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace"})
CHECK_KINDS = frozenset({"reproduce", "probe", "witness_check", "run_tests", "test"})
BUDGETS = {
    "max_answer_submissions": 12,
    "max_forced_checks": 4,
    "max_belief_watches": 8,
    "belief_silent_results": 6,
    "max_cost_tokens": 512,
    "max_field_chars": 480,
}
_PREDICT_RE = re.compile(r"^\s*PREDICT\s*:\s*(?P<value>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{3,}")
_FORBIDDEN = re.compile(r"(?:grader|hidden|gold|backup|backups|production\s+database)", re.IGNORECASE)
_STOPWORDS = frozenset({
    "about", "after", "again", "before", "being", "could", "does", "during", "from",
    "have", "into", "must", "only", "that", "their", "there", "these", "this", "will",
    "with", "would", "should", "where", "when", "then", "than", "what", "which",
})


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    status: str
    gate: str | None
    route: str
    detail: str
    next_gate: str | None = None
    forced_check: Mapping[str, object] | None = None
    reanchor: Mapping[str, object] | None = None
    cost: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": EVENT_SCHEMA,
            "accepted": self.accepted,
            "status": self.status,
            "gate": self.gate,
            "route": self.route,
            "detail": self.detail,
        }
        if self.next_gate is not None:
            result["next_gate"] = self.next_gate
        if self.forced_check is not None:
            result["forced_check"] = dict(self.forced_check)
        if self.reanchor is not None:
            result["reanchor"] = dict(self.reanchor)
        if self.cost is not None:
            result["cost"] = dict(self.cost)
        return result


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    return str(source.get(FLAG, "off")).strip().lower() in {"on", "1", "true", "yes"}


def _clip(value: object, limit: int = 480) -> str:
    text = str(value or "").replace("\x00", " ").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _emit(runtime: object, kind: str, payload: Mapping[str, object]) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_qgates", event_schema=EVENT_SCHEMA,
             payload=dict(payload), dispatch=False)


def _notice(runtime: object, text: str) -> None:
    runtime._qgates_notice = _clip(text, 520)


def _safe_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > BUDGETS["max_field_chars"] or _FORBIDDEN.search(text):
        return None
    return text


def _ticket_clause(row: Mapping[str, object] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    for key in ("_bridge_ordinary_brief", "msg", "brief", "symptom", "goal"):
        value = _safe_text(row.get(key))
        if value:
            return value
    return ""


def _new_state(ticket_clause: str = "") -> dict[str, object]:
    return {
        "schema": EVENT_SCHEMA,
        "pending": None,
        "completed": [],
        "answers": {},
        "ticket_clause": ticket_clause,
        "dodges": {},
        "forced_checks": 0,
        "open_check": None,
        "transitions": [],
        "cost": {"answer_submissions": 0, "estimated_tokens": 0, "budget_tokens": BUDGETS["max_cost_tokens"]},
        "beliefs": [],
        "belief_receipts": [],
        "belief_path_writes": {},
        "drift_events": [],
    }


def state(runtime: object) -> dict[str, object]:
    current = getattr(runtime, "_qgates_state", None)
    if not isinstance(current, dict):
        current = _new_state()
        runtime._qgates_state = current
    return current


def _pending(runtime: object) -> str | None:
    value = state(runtime).get("pending")
    return value if isinstance(value, str) and value in GATES else None


def _step(runtime: object) -> int:
    owner = getattr(runtime, "owner", None)
    value = getattr(owner, "current_step", 0)
    return int(value) if isinstance(value, int) else 0


def _transition(runtime: object, gate: str, reason: str) -> bool:
    current = _pending(runtime)
    if current == gate:
        return False
    data = state(runtime)
    data["pending"] = gate
    transition = {"gate": gate, "from": current, "reason": reason, "step": _step(runtime)}
    transitions = data.setdefault("transitions", [])
    if isinstance(transitions, list):
        transitions.append(transition)
        del transitions[:-16]
    _emit(runtime, "qgate_transition", transition)
    _notice(runtime, f"[qgate] state={gate}; submit the typed action before continuing")
    return True


def task_start(runtime: object, row: Mapping[str, object] | None = None) -> None:
    """Arm intake once for a task; resume calls never reset answered gates."""
    if not enabled() or not isinstance(row, Mapping):
        return
    data = state(runtime)
    if not data.get("ticket_clause"):
        data["ticket_clause"] = _ticket_clause(row)
    if not data.get("transitions") and not data.get("completed") and not _pending(runtime):
        _transition(runtime, "intake", "task_start")


def _answer_keys(gate: str) -> tuple[str, ...]:
    return {
        "intake": ("broken_behavior", "triggering_condition", "observable_outcome"),
        "before_first_product_edit": ("behavior_to_change", "observation_that_shows_it"),
        "first_green": ("requirement_to_assertion",),
        "before_done": ("untested_risk", "why_acceptable", "check_to_run"),
    }[gate]


def _coverage_mapping(answer: Mapping[str, object]) -> Mapping[object, object] | None:
    wrapped = answer.get("requirement_to_assertion")
    if len(answer) == 1 and isinstance(wrapped, Mapping):
        return wrapped
    return answer


def validate_answer(gate: object, answer: object) -> tuple[bool, str]:
    if not isinstance(gate, str) or gate not in GATES:
        return False, "gate must name one active state"
    if not isinstance(answer, Mapping):
        return False, "answer must be a structured object"
    keys = set(answer)
    if gate == "before_done":
        if keys != {"untested_risk", "why_acceptable"} and keys != {"untested_risk", "check_to_run"}:
            return False, "before_done requires untested_risk plus exactly one completion disposition"
        for key in keys:
            if _safe_text(answer.get(key)) is None:
                return False, f"{key} must be a bounded safe string"
        return True, ""
    if gate in {"intake", "before_first_product_edit"}:
        required = set(_answer_keys(gate))
        if keys != required:
            return False, "answer fields do not match the typed gate schema"
        if any(_safe_text(answer.get(key)) is None for key in required):
            return False, "all gate fields must be bounded safe strings"
        return True, ""
    mapping = _coverage_mapping(answer)
    if not isinstance(mapping, Mapping) or not mapping:
        return False, "first_green requires a non-empty requirement-to-assertion map"
    if len(mapping) > 32:
        return False, "requirement_to_assertion exceeds the bounded map size"
    for requirement, assertion in mapping.items():
        if _safe_text(requirement) is None or not isinstance(assertion, str):
            return False, "map keys and assertions must be bounded strings"
        if not assertion.strip() or len(assertion) > BUDGETS["max_field_chars"]:
            return False, "map assertions must be non-empty and bounded"
        if assertion.strip().upper() != "UNEXERCISED" and _FORBIDDEN.search(assertion):
            return False, "map assertion contains a forbidden evidence class"
    return True, ""


def _words(value: object) -> set[str]:
    return {
        word.lower() for word in re.findall(r"[a-z][a-z0-9_/-]{2,}", str(value or "").lower())
        if word.lower() not in _STOPWORDS
    }


def _answer_text(answer: Mapping[str, object]) -> str:
    values: list[str] = []
    for value in answer.values():
        if isinstance(value, Mapping):
            values.extend(str(item) for item in value)
            values.extend(str(item) for item in value.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return " ".join(values)


def _drift(runtime: object, gate: str, answer: Mapping[str, object]) -> Mapping[str, object] | None:
    if gate == "intake":
        return None
    data = state(runtime)
    intake = data.get("answers", {})
    intake = intake.get("intake", {}) if isinstance(intake, Mapping) else {}
    anchor_text = " ".join(str(intake.get(key, "")) for key in ("broken_behavior", "triggering_condition", "observable_outcome"))
    anchor_text += " " + str(data.get("ticket_clause", ""))
    anchor = _words(anchor_text)
    claim = _words(_answer_text(answer))
    if not anchor or not claim:
        return None
    overlap = anchor & claim
    explicit_rename = bool(re.search(r"\b(rename|renamed|alias|relabel|no\s+behavior|cosmetic)\b", _answer_text(answer), re.IGNORECASE))
    if explicit_rename or (len(anchor) >= 4 and not overlap):
        return {
            "anchor_overlap": sorted(overlap),
            "intake_answer": dict(intake) if isinstance(intake, Mapping) else {},
            "ticket_clause": data.get("ticket_clause", ""),
        }
    return None


def _cost(data: Mapping[str, object]) -> dict[str, object]:
    raw = data.get("cost", {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def _charge(runtime: object, answer: object) -> Mapping[str, object]:
    data = state(runtime)
    cost = _cost(data)
    submissions = int(cost.get("answer_submissions", 0) or 0) + 1
    estimate = max(1, len(json.dumps(answer, sort_keys=True, ensure_ascii=False, default=str)) // 4)
    total = int(cost.get("estimated_tokens", 0) or 0) + estimate
    cost.update({"answer_submissions": submissions, "estimated_tokens": total,
                 "budget_tokens": BUDGETS["max_cost_tokens"],
                 "within_budget": submissions <= BUDGETS["max_answer_submissions"] and total <= BUDGETS["max_cost_tokens"]})
    data["cost"] = cost
    _emit(runtime, "qgate_cost_report", cost)
    return cost


def _screen_allows(runtime: object, name: str) -> bool:
    checker = getattr(runtime, "tool_surface_allows", None)
    if not callable(checker):
        return True
    try:
        return bool(checker(name))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _check_tool(runtime: object) -> str:
    offered = getattr(runtime, "completion_menu", ())
    if isinstance(offered, Sequence):
        for name in ("reproduce", "probe", "witness_check", "run_tests"):
            if name in offered and _screen_allows(runtime, name):
                return name
    for name in ("reproduce", "probe", "witness_check", "run_tests"):
        if _screen_allows(runtime, name):
            return name
    # The screen gate has withheld every executable check. Keep the typed route
    # alive and let the next menu projection reopen the first admitted surface.
    return "probe"


def _open_check(runtime: object, gate: str, *, reason: str, expected: str = "") -> Mapping[str, object]:
    data = state(runtime)
    forced = int(data.get("forced_checks", 0) or 0)
    if reason == "second_dodge":
        forced += 1
        data["forced_checks"] = forced
    check = {
        "route": CHECK_ROUTE,
        "tool": _check_tool(runtime),
        "gate": gate,
        "reason": reason,
        "expected_observation": _clip(expected or "one concrete receipt that tests the unexercised condition", 220),
        "attempt": forced,
        "bounded": forced <= BUDGETS["max_forced_checks"],
    }
    data["open_check"] = check
    _emit(runtime, "qgate_check_opened", check)
    _notice(runtime, f"[qgate] route={CHECK_ROUTE}; run the offered witness/probe check")
    return check


def _advance(runtime: object, gate: str) -> str | None:
    data = state(runtime)
    completed = data.setdefault("completed", [])
    if isinstance(completed, list) and gate not in completed:
        completed.append(gate)
    data["pending"] = None
    if gate == "intake":
        _transition(runtime, "before_first_product_edit", "intake_answered")
        return "before_first_product_edit"
    if gate == "before_first_product_edit":
        data["armed_first_green"] = True
        return None
    if gate == "first_green":
        data["armed_before_done"] = True
        return None
    return None


def _result(runtime: object, result: GateResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)


def _blocked(runtime: object, gate: str, *, detail: str, route: str = "QUESTION_GATE_REQUIRED",
             forced_check: Mapping[str, object] | None = None) -> str:
    data = state(runtime)
    return _result(runtime, GateResult(False, "BLOCKED", gate, route, detail,
                                       forced_check=forced_check, cost=_cost(data)))


def _dodge(runtime: object, gate: str, action: str) -> str:
    data = state(runtime)
    dodges = data.setdefault("dodges", {})
    count = int(dodges.get(gate, 0) or 0) + 1 if isinstance(dodges, dict) else 1
    if isinstance(dodges, dict):
        dodges[gate] = count
    _emit(runtime, "qgate_dodged", {"gate": gate, "action": action, "count": count, "step": _step(runtime)})
    if count >= 2:
        check = _open_check(runtime, gate, reason="second_dodge")
        return _blocked(runtime, gate, detail="second gate dodge converted to a forced concrete check",
                        route=CHECK_ROUTE, forced_check=check)
    return _blocked(runtime, gate, detail="the typed gate is required before this action")


def answer(runtime: object, gate: object, answer_value: object) -> str:
    """Validate one menu submission and return a compact structured receipt."""
    if not enabled():
        return _result(runtime, GateResult(False, "INACTIVE", None, "NOOP", "QGATES is disabled"))
    active = _pending(runtime)
    gate_name = gate if isinstance(gate, str) else None
    if gate_name != active:
        return _result(runtime, GateResult(False, "BLOCKED", gate_name, "QUESTION_GATE_REQUIRED",
                                           "answer the currently active state gate", next_gate=active,
                                           cost=_cost(state(runtime))))
    valid, error = validate_answer(gate_name, answer_value)
    if not valid:
        return _dodge(runtime, gate_name, "invalid_question_gate_answer")
    drift = _drift(runtime, gate_name, answer_value)
    if drift is not None:
        data = state(runtime)
        events = data.setdefault("drift_events", [])
        if isinstance(events, list):
            events.append({"gate": gate_name, "step": _step(runtime), "anchor": drift})
            del events[:-8]
        _emit(runtime, "qgate_drift_reanchored", {"gate": gate_name, "anchor": drift})
        _notice(runtime, "[qgate] drift detected; re-anchor to the intake answer and ticket clause")
        return _result(runtime, GateResult(False, "REANCHORED", gate_name, "REANCHOR_INTAKE",
                                           "answer contradicts the intake or ticket anchor",
                                           reanchor=drift, cost=_charge(runtime, answer_value)))
    data = state(runtime)
    answers = data.setdefault("answers", {})
    if isinstance(answers, dict):
        answers[gate_name] = dict(answer_value) if isinstance(answer_value, Mapping) else answer_value
    cost = _charge(runtime, answer_value)
    if not bool(cost.get("within_budget", True)):
        return _blocked(runtime, gate_name, detail="bounded QGATES answer budget exhausted", route="BUDGET_EXHAUSTED")
    check_value = answer_value.get("check_to_run") if isinstance(answer_value, Mapping) else None
    unexercised = False
    if gate_name == "first_green" and isinstance(answer_value, Mapping):
        mapping = _coverage_mapping(answer_value) or {}
        unexercised = isinstance(mapping, Mapping) and any(
            str(value).strip().upper() == "UNEXERCISED" for value in mapping.values())
    if isinstance(check_value, str) or unexercised:
        check = _open_check(runtime, gate_name, reason="unexercised" if unexercised else "check_to_run",
                            expected=str(check_value or ""))
        return _result(runtime, GateResult(True, "CHECK_REQUIRED", gate_name, CHECK_ROUTE,
                                           "typed answer accepted; witness check is now required",
                                           forced_check=check, cost=cost))
    next_gate = _advance(runtime, gate_name)
    _emit(runtime, "qgate_answered", {"gate": gate_name, "answer_digest": _digest(answer_value),
                                       "next_gate": next_gate, "cost": cost})
    return _result(runtime, GateResult(True, "ACCEPTED", gate_name, "CONTINUE",
                                       "typed answer accepted", next_gate=next_gate, cost=cost))


def tool_schema() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Submit the required structured answer for the currently active state gate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gate": {"type": "string", "enum": list(GATES)},
                    "answer": {"type": "object", "description": "Typed fields for the active gate."},
                },
                "required": ["gate", "answer"],
                "additionalProperties": False,
            },
        },
    }


def _has_tool(tools: Sequence[object], name: str) -> bool:
    return any(isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
               and tool["function"].get("name") == name for tool in tools)


def add_menu_tool(tools: object, runtime: object) -> list[dict[str, object]]:
    result = list(tools) if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)) else []
    if enabled() and _pending(runtime) is not None and not _has_tool(result, TOOL_NAME):
        result.append(tool_schema())
    return result


def project_menu(selected: object, tools: object, runtime: object) -> list[dict[str, object]]:
    result = list(selected) if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes, bytearray)) else []
    if enabled() and _pending(runtime) is not None and not _has_tool(result, TOOL_NAME):
        candidates = tools if isinstance(tools, Sequence) else ()
        if _has_tool(candidates, TOOL_NAME):
            result.append(next(tool for tool in candidates if isinstance(tool, Mapping)
                              and isinstance(tool.get("function"), Mapping)
                              and tool["function"].get("name") == TOOL_NAME))
        else:
            result.append(tool_schema())
        runtime.completion_menu = tuple(
            tool.get("function", {}).get("name") for tool in result if isinstance(tool, Mapping))
    return result


def handle_action(runtime: object, kind: str, arguments: Mapping[str, object]) -> str | None:
    if str(kind or "").strip().lower() != TOOL_NAME:
        return None
    if not isinstance(arguments, Mapping):
        return _dodge(runtime, _pending(runtime) or "intake", TOOL_NAME)
    return answer(runtime, arguments.get("gate"), arguments.get("answer"))


def guard_action(runtime: object, kind: str, arguments: Mapping[str, object] | None = None) -> str | None:
    """Block edits/terminal actions until the state gate for that transition is answered."""
    if not enabled():
        return None
    action = str(kind or "").strip().lower()
    data = state(runtime)
    completed = data.get("completed", ())
    completed = completed if isinstance(completed, Sequence) else ()
    if action in EDIT_KINDS and "before_first_product_edit" not in completed:
        if _pending(runtime) is None:
            _transition(runtime, "before_first_product_edit", "first_product_edit_boundary")
        return _dodge(runtime, _pending(runtime) or "before_first_product_edit", action)
    if action == "done":
        if "first_green" not in completed:
            if _pending(runtime) is None:
                _transition(runtime, "first_green", "terminal_boundary")
            return _dodge(runtime, _pending(runtime) or "first_green", action)
        if "before_done" not in completed:
            if _pending(runtime) is None:
                _transition(runtime, "before_done", "terminal_boundary")
            return _dodge(runtime, _pending(runtime) or "before_done", action)
    if data.get("open_check") and action not in CHECK_KINDS:
        return _blocked(runtime, _pending(runtime) or str(data.get("open_check", {}).get("gate", "")),
                        detail="the routed witness check must settle before continuing", route=CHECK_ROUTE,
                        forced_check=data.get("open_check") if isinstance(data.get("open_check"), Mapping) else None)
    return None


def _green_observation(action: object, observation: object) -> bool:
    text = f"{action} {observation}".lower()
    return any(token in text for token in ("pass", "passed", "green", "exit 0", "exit_code=0", "exit_code: 0"))


def after_observation(runtime: object, action: str, observation: str) -> None:
    if not enabled():
        return
    data = state(runtime)
    open_check = data.get("open_check")
    if isinstance(open_check, Mapping) and str(action or "").split(None, 1)[0].lower() in CHECK_KINDS:
        receipt = {"action": _clip(action, 160), "observation": _clip(observation, 360), "step": _step(runtime)}
        receipts = data.setdefault("belief_receipts", [])
        if isinstance(receipts, list):
            receipts.append(receipt)
            del receipts[:-16]
        if _green_observation(action, observation):
            gate = str(open_check.get("gate", ""))
            data["open_check"] = None
            if gate in GATES:
                data["pending"] = gate
                _advance(runtime, gate)
            _emit(runtime, "qgate_check_settled", {"check": dict(open_check), "receipt": receipt, "status": "GREEN"})
        else:
            _emit(runtime, "qgate_check_observed", {"check": dict(open_check), "receipt": receipt, "status": "OPEN"})
    _observe_beliefs(runtime, observation, action=action)
    if "first_green" not in data.get("completed", ()) and _green_observation(action, observation):
        if _pending(runtime) is None and data.get("open_check") is None:
            _transition(runtime, "first_green", "first_green_observation")


def before_completion(runtime: object, kwargs: Mapping[str, object] | None = None) -> dict[str, object] | None:
    if not enabled():
        return None
    blocked = guard_action(runtime, "done", kwargs or {})
    if blocked is None:
        return None
    return {
        "reason_code": "QGATE_BLOCKED",
        "finding": "typed state gate is unresolved",
        "next_action": blocked,
    }


def _distinctive_symbols(value: str) -> tuple[str, ...]:
    symbols = []
    for symbol in _SYMBOL_RE.findall(value):
        if symbol.lower() in _STOPWORDS or len(symbol) < 5:
            continue
        if any(char in symbol for char in "_./-") or any(char.isupper() for char in symbol[1:]):
            if symbol not in symbols:
                symbols.append(symbol)
    return tuple(symbols[:8])


def extract_predictions(text: object) -> tuple[dict[str, object], ...]:
    if not isinstance(text, str):
        return ()
    results = []
    for value in _PREDICT_RE.findall(text):
        symbols = _distinctive_symbols(value)
        if symbols:
            results.append({"prediction": _clip(value, 320), "symbols": list(symbols)})
    return tuple(results[:BUDGETS["max_belief_watches"]])


def record_statement(runtime: object, reasoning: object = None, content: object = None) -> None:
    if not enabled():
        return
    data = state(runtime)
    beliefs = data.setdefault("beliefs", [])
    if not isinstance(beliefs, list):
        beliefs = []
        data["beliefs"] = beliefs
    existing = {str(item.get("prediction")) for item in beliefs if isinstance(item, Mapping)}
    combined = "\n".join(value for value in (reasoning, content) if isinstance(value, str))
    for prediction in extract_predictions(combined):
        if len(beliefs) >= BUDGETS["max_belief_watches"] or prediction["prediction"] in existing:
            continue
        beliefs.append({**prediction, "opened_step": _step(runtime), "silent_results": 0,
                        "satisfied": False, "fired": False})
        existing.add(str(prediction["prediction"]))
        _emit(runtime, "qgate_belief_opened", prediction)


def _observe_beliefs(runtime: object, observation: object, *, action: object = "") -> None:
    data = state(runtime)
    beliefs = data.get("beliefs", [])
    if not isinstance(beliefs, list) or not str(observation or "").strip():
        return
    text = str(observation)
    action_text = str(action or "")
    path_writes = data.setdefault("belief_path_writes", {})
    if isinstance(path_writes, dict):
        action_head, _, action_tail = action_text.partition(" ")
        if action_head.strip().lower() in EDIT_KINDS:
            path = action_tail.strip().split(None, 1)[0] if action_tail.strip() else action_head
            path_writes[path] = int(path_writes.get(path, 0) or 0) + 1
    writes = data.setdefault("belief_receipts", [])
    receipt = {"action": _clip(action, 160), "observation": _clip(text, 360), "step": _step(runtime)}
    if isinstance(writes, list):
        writes.append(receipt)
        del writes[:-16]
    for belief in beliefs:
        if not isinstance(belief, dict) or belief.get("satisfied") or belief.get("fired"):
            continue
        symbols = tuple(str(item) for item in belief.get("symbols", ()) if str(item))
        if symbols and any(symbol.lower() in text.lower() for symbol in symbols):
            belief["satisfied"] = True
            belief["silent_results"] = 0
            continue
        belief["silent_results"] = int(belief.get("silent_results", 0) or 0) + 1
        if belief["silent_results"] < BUDGETS["belief_silent_results"]:
            continue
        if not any(int(count or 0) >= 2 for count in path_writes.values()):
            continue
        if belief.get("fired"):
            continue
        belief["fired"] = True
        contradiction = {
            "prediction": belief.get("prediction", ""),
            "symbols": list(symbols),
            "silent_results": belief["silent_results"],
            "contradicting_receipts": list(writes[-2:]),
            "route": "RECALL",
        }
        _emit(runtime, "qgate_belief_falsified", contradiction)
        _notice(runtime, "[belief] a prediction lacks its expected receipt; inspect the contradicting evidence")


def belief_report(runtime: object) -> dict[str, object]:
    data = state(runtime)
    beliefs = data.get("beliefs", [])
    return {"schema": EVENT_SCHEMA, "watches": list(beliefs) if isinstance(beliefs, list) else [],
            "budget": {"max": BUDGETS["max_belief_watches"]}}


__all__ = [
    "BUDGETS", "CHECK_ROUTE", "EVENT_SCHEMA", "FLAG", "GATES", "GateResult", "TOOL_NAME",
    "add_menu_tool", "after_observation", "answer", "belief_report", "before_completion",
    "enabled", "extract_predictions", "guard_action", "handle_action", "project_menu",
    "record_statement", "state", "task_start", "tool_schema", "validate_answer",
]
