"""Bounded host-owned state carried with the work being resumed.

The F4 decision slots remain the owner of hypotheses, evidence, rejected
alternatives, and the next probe.  This module adds only the association that
makes those slots belong to a work item, plus compact host observations about
pending checks and closed actions.  It never stores or renders a transcript.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_WORK_CARRY"
SCHEMA = "stateframe-work-carry/1"
MAX_WORK_ID = 120
MAX_REF = 180
MAX_TEXT = 180
MAX_PENDING_CHECKS = 4
MAX_ACTION_DISPOSITIONS = 4
MAX_EVIDENCE_REFS = 3

_TERMINAL_CHECK_STATES = frozenset({
    "closed", "complete", "completed", "done", "pass", "passed", "resolved",
    "satisfied", "superseded", "retired",
})
_EDIT_TOOLS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})
_APPLICABILITY = frozenset({"current", "stale", "unknown"})
_ACTION_DISPOSITIONS = frozenset({
    "applied", "blocked", "failed", "not_run", "observed", "verified",
    "staged",  # M3
})


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _clip(value: object, maximum: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= maximum:
        return text
    return text[: max(0, maximum - 1)] + "…"


def _text(value: object, maximum: int, *, default: str = "") -> str:
    if not isinstance(value, str):
        return default
    value = " ".join(value.split())
    return _clip(value, maximum) if value else default


def _first(mapping: Mapping[str, object], names: Sequence[str]) -> object:
    for name in names:
        value = mapping.get(name)
        if value is not None and value != "":
            return value
    return None


def _sequence(value: object, *keys: str) -> Sequence[object]:
    if isinstance(value, Mapping):
        for key in keys:
            nested = value.get(key)
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
                return nested
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    if isinstance(value, str) and value.strip():
        return (value,)
    return ()


def _pending_checks(value: object) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index, raw in enumerate(_sequence(value, "pending_checks", "checks", "items"), start=1):
        if isinstance(raw, Mapping):
            status = _text(_first(raw, ("status", "state", "verdict")), 32,
                           default="pending").lower()
            if status in _TERMINAL_CHECK_STATES:
                continue
            ident = _text(_first(raw, (
                "id", "check_id", "obligation_id", "work_id", "subject", "name")),
                MAX_REF, default=f"check-{index}")
            question = _text(_first(raw, ("question", "text", "title", "detail")),
                             MAX_TEXT)
            action = _first(raw, ("action", "next_action", "repair_action"))
            if isinstance(action, Mapping):
                action = _first(action, ("text", "description", "tool", "kind"))
            reopen = _text(_first(raw, (
                "reopen_if", "reconsider_condition", "reopen_condition")), MAX_TEXT)
            row = {"id": ident, "status": status or "pending"}
            if question:
                row["question"] = question
            if isinstance(action, str) and action.strip():
                row["action"] = _text(action, MAX_TEXT)
            if reopen:
                row["reopen_if"] = reopen
        elif isinstance(raw, str) and raw.strip():
            row = {"id": f"check-{index}", "status": "pending",
                   "question": _text(raw, MAX_TEXT)}
        else:
            continue
        if row["id"] not in {item["id"] for item in result}:
            result.append(row)
        if len(result) >= MAX_PENDING_CHECKS:
            break
    return result


def _evidence_refs(value: object) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
    else:
        values = []
    result: list[str] = []
    for item in values:
        ref = _text(item, MAX_REF)
        if ref and ref not in result:
            result.append(ref)
        if len(result) >= MAX_EVIDENCE_REFS:
            break
    return result


def _action_ref(record: Mapping[str, object]) -> str:
    for key in ("action_ref", "step_record_id", "record_id", "receipt_id"):
        value = _text(record.get(key), MAX_REF)
        if value:
            return value
    refs = _evidence_refs(record.get("receipt_ids", record.get("evidence_refs")))
    return refs[0] if refs else "action"


def _path(record: Mapping[str, object], pending: Mapping[str, object] | None) -> str:
    for source in (record, pending if isinstance(pending, Mapping) else {}):
        value = _first(source, ("target_path", "path", "file"))
        if isinstance(value, str) and value.strip():
            return _clip(value.replace("\\", "/"), MAX_REF)
        arguments = source.get("arguments", source.get("action_parameters"))
        if isinstance(arguments, Mapping):
            value = _first(arguments, ("target_path", "path", "file"))
            if isinstance(value, str) and value.strip():
                return _clip(value.replace("\\", "/"), MAX_REF)
    return ""


def _norm_path(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def _disposition(result: str) -> str:
    if result == "APPLIED":
        return "applied"
    if result in {"PASS", "PASSED", "GREEN", "SUCCESS", "OK", "VERIFIED"}:
        return "verified"
    if result == "STAGED":  # M3 (only ever produced with STAGED_CLASS on)
        return "staged"
    if result in {"REFUSED", "REFUSED_PREDISPATCH", "BLOCKED"}:
        return "blocked"
    if result in {"NOT_RUN", "SKIPPED", "UNAVAILABLE"}:
        return "not_run"
    if result in {"FAIL", "FAILED", "ERROR", "REJECTED", "RED"}:
        return "failed"
    return "observed"


def _restored_applicability(value: object) -> str:
    text = _text(value, 24, default="current").lower()
    return text if text in _APPLICABILITY else "unknown"


def _restore_actions(value: object, work_id: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for raw in _sequence(value, "action_dispositions", "actions", "items"):
        if not isinstance(raw, Mapping):
            continue
        action_ref = _text(_first(raw, ("action_ref", "record_ref", "record_id")),
                           MAX_REF, default="action")
        observation = _text(_first(raw, ("observation", "result", "result_class")),
                            48, default="UNKNOWN").upper()
        disposition = _text(raw.get("disposition"), 32, default=_disposition(observation)).lower()
        if disposition not in _ACTION_DISPOSITIONS:
            disposition = _disposition(observation)
        row: dict[str, object] = {
            "action_ref": action_ref,
            "work_id": _text(raw.get("work_id"), MAX_WORK_ID, default=work_id) or work_id,
            "tool": _text(_first(raw, ("tool", "kind")), 64, default="unknown"),
            "invocation": _text(raw.get("invocation"), 24, default="completed").lower(),
            "observation": observation,
            "applicability": _restored_applicability(raw.get("applicability")),
            "disposition": disposition,
            "evidence_refs": _evidence_refs(
                raw.get("evidence_refs", raw.get("receipt_refs", raw.get("receipt_ids")))),
        }
        scope = _text(raw.get("scope"), MAX_REF)
        if scope:
            row["scope"] = scope
        result.append(row)
        if len(result) >= MAX_ACTION_DISPOSITIONS:
            break
    return result[-MAX_ACTION_DISPOSITIONS:]


def _state(value: Mapping[str, object] | None) -> dict[str, object]:
    raw = value if isinstance(value, Mapping) else {}
    nested = raw.get("work_carry")
    if isinstance(nested, Mapping):
        raw = nested
    work_id = _text(_first(raw, ("work_id", "id", "work")), MAX_WORK_ID)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "work_id": work_id,
        "objective_ref": _text(raw.get("objective_ref"), MAX_REF),
        "focus_ref": _text(raw.get("focus_ref"), MAX_REF),
        "status": _text(raw.get("status"), 32, default="active").lower(),
        "pending_checks": _pending_checks(raw.get("pending_checks", raw.get("checks", ()))),
        "action_dispositions": _restore_actions(
            raw.get("action_dispositions", raw.get("actions", ())), work_id),
    }
    return result


class WorkCarryStore:
    """A bounded work association and host action/disposition ledger."""

    def __init__(self, state: Mapping[str, object] | None = None):
        self._state = _state(state)

    @classmethod
    def from_state(cls, state: Mapping[str, object] | None) -> "WorkCarryStore":
        return cls(state)

    @property
    def has_content(self) -> bool:
        return bool(self._state.get("work_id") or self._state.get("pending_checks")
                    or self._state.get("action_dispositions"))

    @property
    def work_id(self) -> str:
        return str(self._state.get("work_id") or "")

    def snapshot(self) -> dict[str, object]:
        return copy.deepcopy(self._state)

    def set_work_carry(self, value: Mapping[str, object] | None) -> dict[str, object]:
        if not isinstance(value, Mapping):
            return {"accepted": False, "error": "work carry must be an object"}
        incoming = _state(value)
        work_id = incoming["work_id"]
        if not work_id:
            return {"accepted": False, "error": "work_id is required"}
        previous_id = self.work_id
        changed: list[str] = []
        if previous_id and previous_id != work_id:
            self._state["action_dispositions"] = []
            self._state["objective_ref"] = ""
            self._state["focus_ref"] = ""
            self._state["status"] = "active"
            self._state["pending_checks"] = []
            changed.append("work_changed")
        for key in ("work_id", "objective_ref", "focus_ref", "status"):
            if incoming[key] and self._state.get(key) != incoming[key]:
                self._state[key] = copy.deepcopy(incoming[key])
                changed.append(key)
        if "pending_checks" in value or "checks" in value:
            if self._state.get("pending_checks") != incoming["pending_checks"]:
                self._state["pending_checks"] = copy.deepcopy(incoming["pending_checks"])
                changed.append("pending_checks")
        if not previous_id:
            changed.append("work_bound")
        return {"accepted": True, "changed": list(dict.fromkeys(changed)),
                "work_id": self.work_id}

    bind = set_work_carry

    def set_pending_checks(self, value: object) -> dict[str, object]:
        checks = _pending_checks(value)
        if checks == self._state["pending_checks"]:
            return {"accepted": True, "changed": []}
        self._state["pending_checks"] = checks
        return {"accepted": True, "changed": ["pending_checks"]}

    def record_action(self, record: Mapping[str, object], *,
                      pending: Mapping[str, object] | None = None) -> bool:
        if not isinstance(record, Mapping) or not self.work_id:
            return False
        pending = pending if isinstance(pending, Mapping) else {}
        result = _text(_first(record, ("result", "result_class", "status")), 48,
                       default="UNKNOWN").upper()
        invocation = _text(record.get("invocation"), 24).lower()
        if invocation not in {"completed", "refused", "not_run"}:
            invocation = ("refused" if "REFUSED" in result or result == "BLOCKED"
                          else "not_run" if result in {"NOT_RUN", "SKIPPED"}
                          else "completed")
        applicability = _text(record.get("applicability"), 24, default="current").lower()
        if applicability not in {"current", "stale", "unknown"}:
            applicability = "unknown"
        refs = _evidence_refs(record.get(
            "receipt_ids", record.get("evidence_refs", record.get("receipt_id"))))
        row: dict[str, object] = {
            "action_ref": _action_ref(record),
            "work_id": self.work_id,
            "tool": _text(_first(record, ("tool", "kind")) or pending.get("tool"),
                           64, default="unknown"),
            "invocation": invocation,
            "observation": result,
            "applicability": applicability,
            "disposition": _disposition(result),
            "evidence_refs": refs,
        }
        scope = _path(record, pending)
        if scope:
            row["scope"] = scope
        step = record.get("step")
        if isinstance(step, (int, str)) and str(step).strip():
            row["step"] = step
        rows = []
        for item in self._state["action_dispositions"]:
            same_ref = item.get("action_ref") == row["action_ref"]
            old_refs = set(_evidence_refs(item.get("evidence_refs")))
            same_receipt = bool(refs and old_refs.intersection(refs))
            if not same_ref and not same_receipt:
                rows.append(item)
        rows.append(row)
        self._state["action_dispositions"] = rows[-MAX_ACTION_DISPOSITIONS:]
        if (row["disposition"] == "applied"
                and str(row["tool"]).lower() in _EDIT_TOOLS):
            self.invalidate_after_edit(str(row.get("scope") or ""),
                                       exclude_ref=str(row["action_ref"]))
        return True

    def invalidate_after_edit(self, path: str, *, exclude_ref: str = "") -> list[str]:
        """Demote earlier outcomes that an applied edit may have made historical.

        An outcome scoped to the edited path becomes ``stale``.  Any other
        non-edit outcome still marked ``current`` becomes ``unknown``: the host
        holds no dependency map proving it unaffected, so it must not keep
        claiming present applicability.  Returns the demoted action refs.
        """
        edit_path = _norm_path(path)
        demoted: list[str] = []
        for action in self._state["action_dispositions"]:
            if exclude_ref and action.get("action_ref") == exclude_ref:
                continue
            if action.get("disposition") == "applied":
                continue
            scope = _norm_path(action.get("scope"))
            before = action.get("applicability")
            if edit_path and scope and scope == edit_path:
                action["applicability"] = "stale"
            elif before == "current":
                action["applicability"] = "unknown"
            if action.get("applicability") != before:
                demoted.append(str(action.get("action_ref")))
        return demoted

    def drop_dangling_refs(self, existing_refs: set[str] | Sequence[str] = ()) -> list[str]:
        """Keep restored action outcomes only when their host receipts remain known."""
        known = {str(ref) for ref in (existing_refs or ()) if str(ref).strip()}
        kept_actions: list[dict[str, object]] = []
        dropped: list[str] = []
        for action in self._state["action_dispositions"]:
            refs = _evidence_refs(action.get("evidence_refs"))
            valid = [ref for ref in refs if ref in known]
            dropped.extend(ref for ref in refs if ref not in known)
            if refs and not valid:
                continue
            action["evidence_refs"] = valid
            kept_actions.append(action)
        self._state["action_dispositions"] = kept_actions[-MAX_ACTION_DISPOSITIONS:]
        return list(dict.fromkeys(dropped))


def _check_detail(check: Mapping[str, object]) -> str:
    detail = _clip(check.get("id"), 36)
    question = _clip(check.get("question"), 74)
    action = _clip(check.get("action"), 54)
    reopen = _clip(check.get("reopen_if"), 62)
    if question:
        detail += " — " + question
    if action:
        detail += "; next " + action
    if reopen:
        detail += "; reopen if " + reopen
    return detail


def _action_detail(action: Mapping[str, object]) -> str:
    tool = _clip(action.get("tool"), 32) or "unknown"
    observation = _clip(action.get("observation"), 20) or "UNKNOWN"
    disposition = _clip(action.get("disposition"), 20) or "observed"
    applicability = _clip(action.get("applicability"), 16) or "unknown"
    refs = _evidence_refs(action.get("evidence_refs"))
    if disposition == "staged":
        from kernel_tokenlab_v12 import STAGED_DISPOSITION  # M3 legible handoff wording
        return f"{tool} -> {STAGED_DISPOSITION} ({applicability})"
    detail = f"{tool} -> {disposition}/{observation} ({applicability}"
    if refs:
        detail += "; refs " + ",".join(_clip(ref, 30) for ref in refs[:2])
    return detail + ")"


_ACTIONS_HEAD = "Action dispositions: "
_CHECKS_HEAD = "Pending checks: "
_MIN_CHECK_ROOM = 40


def render_extension(value: object, *, max_chars: int = 360) -> str:
    """Render work identity and typed host outcomes as handoff lines.

    Space is allocated, not first-come: the newest complete action outcome
    (with its reference) is reserved first, then work identity, then as many
    whole pending checks as fit.  Omitted entries are counted explicitly so a
    bounded render never silently drops the action ledger or checks.
    """
    state = _state(value if isinstance(value, Mapping) else None)
    max_chars = max(0, int(max_chars))
    actions = [a for a in state.get("action_dispositions", ()) if isinstance(a, Mapping)]
    checks = [c for c in state.get("pending_checks", ()) if isinstance(c, Mapping)]

    action_details = [_action_detail(a) for a in actions[-2:]]
    action_line = ""
    if action_details:
        newest = action_details[-1]
        older = len(actions) - 1
        suffix = f" (+{older} earlier)" if older else ""
        action_line = _clip(_ACTIONS_HEAD + newest + suffix + ".", max_chars)
    remaining = max_chars - (len(action_line) + 1 if action_line else 0)

    work_line = ""
    work_id = state.get("work_id")
    if work_id and remaining > 0:
        full = f"Work: {_clip(work_id, 72)}"
        objective = _clip(state.get("objective_ref"), 64)
        focus = _clip(state.get("focus_ref"), 64)
        if objective:
            full += f"; objective {objective}"
        if focus:
            full += f"; focus {focus}"
        full += "."
        # Keep room for at least one check line when checks exist.
        work_room = remaining - (_MIN_CHECK_ROOM + 1 if checks else 0)
        if len(full) > work_room:
            full = _clip(f"Work: {_clip(work_id, 72)}.", max(0, work_room))
        work_line = full
        remaining -= len(work_line) + 1 if work_line else 0

    check_line = ""
    if checks and remaining > 0:
        shown: list[str] = []
        for check in checks:
            candidate = _CHECKS_HEAD + "; ".join([*shown, _check_detail(check)])
            omitted = len(checks) - len(shown) - 1
            tail = f" (+{omitted} more omitted)." if omitted else "."
            if len(candidate) + len(tail) > remaining:
                break
            shown.append(_check_detail(check))
        omitted = len(checks) - len(shown)
        tail = f" (+{omitted} more omitted)." if omitted else "."
        if shown:
            check_line = _CHECKS_HEAD + "; ".join(shown) + tail
        else:
            # No whole check fits: keep a clipped first check plus the count.
            check_line = _clip(_CHECKS_HEAD + _check_detail(checks[0]),
                               max(0, remaining - len(tail))) + tail
            if len(check_line) > remaining:
                check_line = ""
        remaining -= len(check_line) + 1 if check_line else 0

    # Spend any leftover on the second-newest action outcome.
    if len(action_details) == 2 and action_line:
        older = len(actions) - 2
        suffix = f" (+{older} earlier)" if older else ""
        both = _ACTIONS_HEAD + action_details[0] + "; " + action_details[1] + suffix + "."
        if len(both) - len(action_line) <= remaining:
            action_line = both

    return "\n".join(line for line in (work_line, check_line, action_line) if line)


__all__ = [
    "MAX_ACTION_DISPOSITIONS", "MAX_PENDING_CHECKS", "MODE_ENV", "SCHEMA",
    "WorkCarryStore", "enabled", "mode", "render_extension",
]
