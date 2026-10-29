"""Q8: a navigation tool's receipt is the symbol answer, never a gate's text.

``where_defined`` / ``who_calls`` / ``references`` are host-registered read
effects (``kernel_bounded_locate_code.READ_TOOLS``).  A completion/green-lock
hold may restrict writes or certification, but it must not substitute its own
refusal text ("done refused -- public checks are BLOCKED ...") for the lookup
result.  When a pre-dispatch gate refuses a navigation call, the host still runs
the read-only ``kernel_repo_intel`` query and returns its result first; the gate
text survives only as an ``advisories`` entry.

``STATEFRAME_NAV_RECEIPT_FIRST`` (default ``on``; ``off`` restores the old
substitution).
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping

ENV = "STATEFRAME_NAV_RECEIPT_FIRST"
KIND = "symbol_lookup"
NAVIGATION_TOOLS = frozenset({"where_defined", "who_calls", "references"})
_ADVISORY_LIMIT = 1200


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    value = str(values.get(ENV, "on") or "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{ENV} must be on or off")
    return value == "on"


def is_navigation(name: object) -> bool:
    return str(name or "").strip().lower() in NAVIGATION_TOOLS


def _status_line(payload: Mapping[str, object]) -> str:
    action = str(payload.get("action") or "lookup")
    symbol = str(payload.get("symbol") or "")
    status = str(payload.get("status") or "UNKNOWN")
    head = f"{action} `{symbol}`: "
    if status == "NOT_FOUND":
        return head + "NOT FOUND (0 definitions in the indexed repository)"
    definition = payload.get("definition")
    if status == "FOUND" and isinstance(definition, Mapping):
        location = f"{definition.get('file')}:{definition.get('lineno') or definition.get('line')}"
        return head + f"FOUND {location}"
    if isinstance(payload.get("results"), list):
        return head + f"{status} ({len(payload['results'])} result(s))"
    return head + status


def compose(lookup: str, advisory: str | None = None) -> str:
    """Stamp ``kind=symbol_lookup`` and carry any gate text as an advisory only.

    The observation starts with the symbol answer (``FOUND`` / ``NOT FOUND``);
    a non-JSON lookup is returned unchanged apart from the appended advisory.
    """
    try:
        payload = json.loads(lookup)
    except (TypeError, ValueError):
        payload = None
    note = str(advisory or "").strip()[:_ADVISORY_LIMIT]
    if not isinstance(payload, dict):
        return lookup if not note else f"{lookup}\nadvisory (not part of the lookup): {note}"
    payload["kind"] = KIND
    if note:
        payload["advisories"] = [note]
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{_status_line(payload)}\n{body}" if note else body


def _receipt_detail(result: object) -> str:
    text = str(result or "")
    start = text.find("{")
    if start < 0:
        return text[:240]
    try:
        payload = json.loads(text[start:])
    except (TypeError, ValueError):
        return text[:240]
    if not isinstance(payload, Mapping):
        return text[:240]
    parts = [str(payload.get("status") or "HOST_OBSERVED"),
             str(payload.get("action") or "repository navigation"),
             str(payload.get("symbol") or "")]
    definition = payload.get("definition")
    if isinstance(definition, Mapping):
        file_path = str(definition.get("file") or "")
        line = definition.get("lineno") or definition.get("line") or ""
        parts.append(f"{file_path}:{line}")
    count = payload.get("count")
    if type(count) is int:
        parts.append(f"count={count}")
    return " ".join(part for part in parts if part)[:240]


def record_execution_receipt(bolt: object, name: object, arguments: object, result: object) -> bool:
    """Attach the typed host receipt for one executed repository navigation query."""
    normalized = str(name or "").strip().lower()
    if not is_navigation(normalized):
        return False
    receipt = getattr(bolt, "receipt", None)
    if not callable(receipt):
        raise TypeError("navigation execution receipt producer is unavailable")
    args = dict(arguments) if isinstance(arguments, Mapping) else {}
    receipt(
        normalized,
        f"repository navigation {normalized}",
        str(args.get("symbol", "")),
        "HOST_OBSERVED",
        detail=_receipt_detail(result),
    )
    return True


def answer_despite_gate(name: object, arguments: object, gate_text: str | None,
                        lookup: Callable[[str, dict], str | None] | None) -> str | None:
    """Return the composed lookup when ``gate_text`` refused a navigation call.

    ``None`` means "keep the gate's own observation": the switch is off, the
    tool is not a navigation tool, nothing refused, or no lookup is available.
    """
    if (not gate_text or "no implemented execution receipt contract" in gate_text
            or not is_navigation(name) or not callable(lookup) or not enabled()):
        return None
    args = dict(arguments) if isinstance(arguments, Mapping) else {}
    try:
        result = lookup(str(name).strip().lower(), args)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if not isinstance(result, str) or not result:
        return None
    return compose(result, gate_text)


__all__ = ["ENV", "KIND", "NAVIGATION_TOOLS", "answer_despite_gate", "compose",
           "enabled", "is_navigation", "record_execution_receipt"]
