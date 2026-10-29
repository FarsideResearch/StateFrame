"""Reduce redundant output after checks pass without changing completion authority."""
from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence

MODE_ENV = "STATEFRAME_TOKENLAB_POSTGREEN_QUIET"
MANIFEST_KEY = "stateframe_tokenlab_postgreen_quiet"
EVENT = "postgreen_quiet_applied"
PRODUCER = "kernel_postgreen_quiet"

DISCLOSURE_PREFIX = "GREEN LOCK / open counterexamples disclosure:"   # kernel_first_green_disclosure.DISCLOSURE_PREFIX
STOP_HEADER = "[Unified stop authority]"                                # kernel_completion_policy.RENDER_HEADER
UNBOUND_REASONS = frozenset({"UNBOUND_OBSERVABLE", "AMBIGUOUS_CONTRACT"})
_HINT_RE = re.compile(r"\s*If one remains, make a real source edit, read a file not read since the challenge, "
                      r"or run a new probe before done\.?")
_STOP_ROW_RE = re.compile(r"^- (?P<id>\S+): (?P<state>[A-Z_]+); provenance=[^;]*(?:; reason=(?P<reason>\S+))?$")
CURRENT_CHECK_NOTE = "not certified by the current host checks; review the ticket and gather evidence if needed"
QUIET_NOTE = CURRENT_CHECK_NOTE


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _green_disclosure(text: str) -> bool:
    return (text.startswith(DISCLOSURE_PREFIX) and "public tests passed" in text
            and "open counterexamples []" in text)


def _current_revision(runtime: object) -> str | None:
    binding = getattr(runtime, "binding", None)
    if callable(binding):
        try:
            value = binding()
        except (AttributeError, OSError, TypeError, ValueError):
            value = None
        if isinstance(value, Mapping):
            for key in ("product_revision", "artifact", "revision"):
                if value.get(key) not in (None, "", "UNKNOWN", "MISSING"):
                    return str(value[key])
    for key in ("product_revision", "current_revision", "revision"):
        value = getattr(runtime, key, None)
        if value not in (None, "", "UNKNOWN", "MISSING"):
            return str(value)
    return None


def _structured_quiet_guard(runtime: object | None) -> bool:
    """Require current structured evidence; rendered text is never the authority for this guard."""
    if runtime is None:
        return False
    revision = _current_revision(runtime)
    raw = getattr(runtime, "host_closure_assessments", None)
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    rows = [dict(row) for row in (raw or ()) if isinstance(row, Mapping)]
    if not rows:
        return False
    open_rows = []
    closed = {"SATISFIED", "PASS", "PASSED", "GREEN", "SUCCESS", "OK", "RESOLVED", "CLOSED", "COMPLETE"}
    for row in rows:
        state = str(next((row.get(key) for key in
                          ("state", "status", "verdict", "result_class", "result")
                          if row.get(key) not in (None, "")), "")).strip().upper()
        if state in closed:
            continue
        row_revision = next((row.get(key) for key in
                             ("revision", "current_revision", "product_revision", "binding_revision")
                             if row.get(key) not in (None, "")), None)
        if revision is None or row_revision is None or str(row_revision) != revision:
            return False
        open_rows.append(row)
        if state != "UNKNOWN":
            return False
        reason = str(row.get("reason") or row.get("classification") or "").strip().upper()
        explicitly_unbound = row.get("bound") is False or row.get("observable") is False
        if reason not in UNBOUND_REASONS and not explicitly_unbound:
            return False
    if not open_rows:
        return False
    try:
        from kernel_live_counterexample_challenge import live_rows
        if live_rows(runtime):
            return False
        if getattr(runtime, "live_counterexamples", None) or getattr(runtime, "open_counterexamples", None):
            return False
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        for key in ("live_counterexamples", "open_counterexamples"):
            value = getattr(runtime, key, None)
            if value:
                return False
    return True


def quiet_disclosure(text: str, runtime: object | None = None) -> str:
    """Keep disclosure facts and IDs; replace only generic action guidance after structured proof."""
    if not _green_disclosure(text) or not _structured_quiet_guard(runtime):
        return text
    return _HINT_RE.sub(f"; {CURRENT_CHECK_NOTE}", text, count=1)


def quiet_stop_authority(text: str, runtime: object | None = None) -> str:
    """Append current-check guidance only after structured current-revision proof."""
    if (not text.startswith(STOP_HEADER) or "done now: submits uncertified" not in text
            or not _structured_quiet_guard(runtime)):
        return text
    if QUIET_NOTE in text:
        return text
    return text + "\n" + QUIET_NOTE


def quiet_messages(messages: Sequence[Mapping[str, object]], runtime: object | None = None) -> tuple[list[dict], list[str]]:
    """Pure: returns (new messages, changed block names) after structured proof."""
    original = [dict(m) for m in messages]
    if not _structured_quiet_guard(runtime):
        return original, []
    if not any(isinstance(m.get("content"), str) and _green_disclosure(m["content"])
               for m in original if m.get("role") == "user"):
        return original, []
    changed: list[str] = []
    out: list[dict] = []
    for message in original:
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str):
            if content.startswith(DISCLOSURE_PREFIX):
                new = quiet_disclosure(content, runtime)
                if new != content:
                    changed.append("first_green_disclosure")
                    message = {**message, "content": new}
            elif content.startswith(STOP_HEADER):
                new = quiet_stop_authority(content, runtime)
                if new != content:
                    changed.append("unified_stop_authority")
                    message = {**message, "content": new}
        out.append(message)
    return out, changed


def emit_applied(journal: object, changed: Sequence[str], *, request_id: object) -> bool:
    if not changed or not hasattr(journal, "emit"):
        return False
    journal.emit(EVENT, producer=PRODUCER, dispatch=False, request_id=request_id, blocks=list(changed))
    return True


__all__ = ["CURRENT_CHECK_NOTE", "EVENT", "MANIFEST_KEY", "MODE_ENV", "PRODUCER", "QUIET_NOTE", "UNBOUND_REASONS", "emit_applied",
           "enabled", "mode", "quiet_disclosure", "quiet_messages", "quiet_stop_authority"]
