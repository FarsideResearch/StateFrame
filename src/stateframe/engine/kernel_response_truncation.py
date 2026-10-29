"""STATEFRAME_RESPONSE_TRUNCATION: a length-truncated completion is never parsed into a proposal (rule 11d).

Every arm, same policy (r45; crawl_l_one-924103-r44d-f4 spent 24,000 completion tokens on one truncated response that
became an ``invalid_proposal``).  When a response stops on ``finish_reason == "length"`` AT the output ceiling (no
larger budget exists -- budgets are never lowered or capped here):
  1. journal ``response_truncated`` (result_class RESPONSE_TRUNCATED) with the request id and attempt;
  2. end the step as a typed truncation (``TRUNCATED_MARKER``) on that FIRST truncation, never an invalid edit.
     No identical retry (coordinator 09-28 #10, o4 H1 / o5 H3): the request is seeded, so resending it reproduces the
     degenerate loop at ~+24k completion tokens.
``response_truncation_checked`` is journaled once per run when the consumer first evaluates a response (fired proof).
Off (default): the legacy path is byte-identical.
"""
from __future__ import annotations

import os

MODE_ENV = "STATEFRAME_RESPONSE_TRUNCATION"
RESULT_CLASS = "RESPONSE_TRUNCATED"
EVENT = "response_truncated"
CHECKED_EVENT = "response_truncation_checked"
TRUNCATED_MARKER = "[response-truncated]"
TRUNCATED_ACTION = "response_truncated"
MODEL_NOTE = ("[response-truncated] Your previous response reached the output limit before completing a tool "
              "call, so nothing was executed. Continue from the current state and return one complete tool call.")


def mode() -> str:
    value = str(os.environ.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def is_truncated(content: object) -> bool:
    return isinstance(content, str) and content.startswith(TRUNCATED_MARKER)


def note_checked(journal: object) -> None:
    """Journal CHECKED_EVENT once per journal (the consumer reached its decision point)."""
    if journal is None or getattr(journal, "_r45_truncation_checked", False):
        return
    try:
        journal.emit(CHECKED_EVENT)
        setattr(journal, "_r45_truncation_checked", True)
    except Exception:  # noqa: BLE001 - telemetry never changes the call path
        return


__all__ = ["CHECKED_EVENT", "EVENT", "MODEL_NOTE", "note_checked", "MODE_ENV", "RESULT_CLASS", "TRUNCATED_ACTION", "TRUNCATED_MARKER",
           "enabled", "is_truncated", "mode"]
