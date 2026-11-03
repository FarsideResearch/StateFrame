"""Requirement recheck: refuse the model's first done once and ask it to re-check the task.

Opt-in (STATEFRAME_REQUIREMENT_RECHECK=on; off by default, and off leaves request bytes unchanged). On the first
model done of a run, the host answers with the task's own sentences as a numbered checklist and asks the model to
check each against its change, fix what is unmet, and call done again. It refuses at most once per run, never on
the last step, and never when the task text has fewer than two clauses. Deterministic: no model call, no judgment
of the change; the checklist comes from the task text only.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

from kernel_lifecycle_table import clauses

SWITCH = "STATEFRAME_REQUIREMENT_RECHECK"
REASON_CODE = "REQUIREMENT_RECHECK"
TASK_KEYS = ("_bridge_ordinary_brief", "msg", "brief", "symptom")
MAX_CLAUSES = 20
MAX_CLAUSE_CHARS = 240
MIN_CLAUSES = 2
MIN_STEPS_LEFT = 2
_COUNTER = "_requirement_recheck_refusals"


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    value = str(values.get(SWITCH, "off") or "off").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{SWITCH} must be on or off")
    return value == "on"


def task_text(task_row: Mapping[str, object]) -> str:
    """The original task text: the first non-empty string among the brief keys."""
    for key in TASK_KEYS:
        value = task_row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def checklist(text: str) -> list[str]:
    items = []
    for clause in clauses(text)[:MAX_CLAUSES]:
        items.append(clause if len(clause) <= MAX_CLAUSE_CHARS else clause[:MAX_CLAUSE_CHARS - 3].rstrip() + "...")
    return items


def render(items: list[str]) -> str:
    lines = [f"REFUSED {REASON_CODE}: before submitting, re-read the task and check every requirement and edge "
             "case it names against your change. Fix anything unmet and run the tests, then call done again. "
             "This check runs once per run.", "Task requirements:"]
    lines.extend(f"{number}. {item}" for number, item in enumerate(items, 1))
    return "\n".join(lines)


def closing_refusal(measurement: object, task_row: Mapping[str, object], steps_left: int | None,
                    environ: Mapping[str, str] | None = None) -> str | None:
    """The refusal text for this done, or None to let it proceed.

    ``steps_left`` counts model steps remaining after this one (None when the run has no step cap).
    """
    if not enabled(environ):
        return None
    if int(getattr(measurement, _COUNTER, 0) or 0) >= 1:
        return None
    if steps_left is not None and steps_left < MIN_STEPS_LEFT:
        return None
    items = checklist(task_text(task_row))
    if len(items) < MIN_CLAUSES:
        return None
    setattr(measurement, _COUNTER, 1)
    journal = getattr(measurement, "journal", None)
    if journal is not None and callable(getattr(journal, "emit", None)):
        journal.emit("requirement_recheck_refused", clauses=len(items), dispatch=False)
    return render(items)
