"""Opt-in, host-signalled gating for rarely used frame capabilities.

The default is deliberately inert: ``apply`` returns the exact input list and
the budget guidance remains enabled.  When enabled, a capability is retained
only when its host-owned signal is present.  Signals are structured runtime
facts; this module never searches messages or task text for them.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping


MODE_ENV = "STATEFRAME_TOOL_SURFACE"

# The mapping is the policy boundary: a capability names the structured host
# condition that makes it useful in the current frame.  Other tools pass
# through unchanged, including tools already gated by their own organ.
TOOL_SIGNALS = {
    "read_ticket": "ticket_needed",
    "dispute": "requirements_dispute",
    "restore_candidate": "candidate_recovery_available",
}
HOST_SIGNALS = frozenset(TOOL_SIGNALS.values())


def _switch() -> bool:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value == "on"


def enabled() -> bool:
    return _switch()


def budget_guidance_enabled() -> bool:
    """Whether the legacy response-budget guidance belongs in the prompt."""
    return not enabled()


def _field(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _status(value: object) -> str:
    for key in ("underlying_status", "status", "result", "result_class"):
        candidate = _field(value, key, "")
        if candidate:
            return str(candidate).upper()
    return ""


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()


def _has_current_dispute(runtime: object) -> bool:
    """Derive the dispute signal from structured host admissions only."""
    ledger = getattr(runtime, "requirements_ledger", None)
    if not isinstance(ledger, Mapping):
        return False
    checks = {
        str(_field(row, "id", "")): row
        for row in _sequence(ledger.get("checks", ()))
        if _field(row, "id", "")
    }
    decisions = {
        str(_field(row, "check_id", "")): row
        for row in _sequence(getattr(runtime, "requirements_admissions", ()))
        if _field(row, "check_id", "")
    }
    receipts = (
        *tuple(getattr(runtime, "host_check_receipts", ()) or ()),
        *tuple(getattr(runtime, "host_repair_check_receipts", ()) or ()),
    )
    # Freshness: a receipt bound to a snapshot other than the current host
    # closure snapshot is stale evidence and cannot keep a dispute visible
    # (a repair COUNTEREXAMPLE listed after a later PASS would otherwise win).
    current_snapshot = getattr(runtime, "host_closure_snapshot", None)
    if current_snapshot:
        from kernel_obligation_assess import _identity
        current_identity = _identity(current_snapshot)

        def _fresh(receipt: object) -> bool:
            snap = _field(receipt, "snapshot_digest", None) or _field(receipt, "snapshot", None)
            return not snap or _identity(snap) == current_identity

        receipts = tuple(receipt for receipt in receipts if _fresh(receipt))
    latest = {
        str(_field(receipt, "check_id", "")): receipt
        for receipt in receipts if _field(receipt, "check_id", "")
    }
    for receipt in latest.values():
        if _status(receipt) != "COUNTEREXAMPLE":
            continue
        check_id = str(_field(receipt, "check_id", ""))
        check = checks.get(check_id, {})
        admission = decisions.get(check_id, _field(check, "admission", {}))
        if str(_field(admission, "tier", "")).upper() == "GATING":
            return True
    return False


def signals_for(runtime: object | None = None) -> frozenset[str]:
    """Return explicit host signals plus the current structured dispute signal."""
    if runtime is None:
        return frozenset()
    raw = getattr(runtime, "_tool_surface_signals", ())
    signals = {str(signal) for signal in raw if str(signal) in HOST_SIGNALS}
    if _has_current_dispute(runtime):
        signals.add("requirements_dispute")
    if runtime is not None:
        try:
            from kernel_candidate_preservation import restore_available
            if restore_available(runtime):
                signals.add("candidate_recovery_available")
        except (ImportError, TypeError, ValueError):
            pass
    return frozenset(signals)


def note_signal(runtime: object, signal: str, *, reason: str = "") -> bool:
    """Record one host-owned signal; return whether it changed runtime state."""
    if signal not in HOST_SIGNALS:
        raise ValueError(f"unknown {MODE_ENV} signal: {signal}")
    if not enabled():
        return False
    current = set(getattr(runtime, "_tool_surface_signals", ()) or ())
    if signal in current:
        return False
    current.add(signal)
    setattr(runtime, "_tool_surface_signals", current)
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("tool_surface_signal", producer="kernel_tool_surface",
             signal=signal, reason=str(reason)[:200], dispatch=False)
    return True


def tool_allowed(name: str, *, runtime: object | None = None,
                 signals: object | None = None) -> bool:
    """Return whether a named capability is available in this frame."""
    if not enabled():
        return True
    required = TOOL_SIGNALS.get(name)
    if required is None:
        return True
    active = (signals_for(runtime) if signals is None else
              frozenset(str(signal) for signal in signals))
    return required in active


def runner_dispatchable(name: str, *, runtime: object | None = None) -> bool:
    """The runner's name-level dispatch-branch conditions, in ONE place (r43 review5).

    ``probes/react_repair.py`` routes a tool call to an execution branch only when
    this holds; otherwise it falls through to UNKNOWN_TOOL.  The green-lock menu
    authority consumes the same predicate, so it never offers an undispatchable tool.
    """
    if name == "shell":
        from kernel_shell_tool import service
        shell = service()
        return shell is not None and shell.settings.shell != "off"
    if name in {"read_ticket", "dispute"}:
        return (getattr(runtime, "stateframe_requirements_ledger_mode", "off") == "on"
                and tool_allowed(name, runtime=runtime))
    if name == "dispute_check":
        return (getattr(runtime, "stateframe_host_closure_mode", "off") == "on"
                or getattr(runtime, "stateframe_test_integrity_mode", "off") == "on")
    if name == "restore_candidate":
        try:
            from kernel_candidate_preservation import restore_available
            return bool(runtime is not None and restore_available(runtime))
        except (ImportError, TypeError, ValueError):
            return False
    return True


def dispatch_admissible(name: str, *, runtime: object | None = None,
                        signals: object | None = None) -> bool:
    """Share receipt, capability, and runner availability across menus and dispatch."""
    from kernel_continuation_evidence import has_execution_receipt_contract
    if name == "file_create" and new_files_forbidden(runtime):
        return False
    return (has_execution_receipt_contract(name)
            and runner_dispatchable(name, runtime=runtime)
            and tool_allowed(name, runtime=runtime, signals=signals))


def new_files_forbidden(runtime: object | None) -> bool:
    """True only when the task row EXPLICITLY forbids new product files (allow_new_files is False).

    The harness captures a created file only when the row allows new files, so a
    file_create on a forbidding row would be silently dropped from the candidate. Tool and capture contract must
    agree: such a row never offers or admits file_create. An absent key is not a prohibition.
    """
    owner = getattr(runtime, "owner", None) if runtime is not None else None
    row = getattr(owner, "task_row", None)
    if not isinstance(row, Mapping):
        row = getattr(runtime, "task_row", None)
    return isinstance(row, Mapping) and row.get("allow_new_files") is False


def _tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function", {})
    return str(function.get("name", "")) if isinstance(function, Mapping) else ""


def apply(tools: object, *, runtime: object | None = None,
          signals: object | None = None) -> object:
    """Filter the menu without mutating schemas or changing their order."""
    if not enabled() or not isinstance(tools, list):
        return tools
    active = (signals_for(runtime) if signals is None else
              frozenset(str(signal) for signal in signals))
    return [tool for tool in tools
            if dispatch_admissible(_tool_name(tool), runtime=runtime, signals=active)]


def schema_chars(tool: Mapping[str, object]) -> int:
    """Measure one schema using the request payload's JSON character encoding."""
    return len(json.dumps(tool, ensure_ascii=False))


__all__ = [
    "HOST_SIGNALS", "MODE_ENV", "TOOL_SIGNALS", "apply", "budget_guidance_enabled",
    "dispatch_admissible",
    "enabled", "note_signal", "schema_chars", "signals_for", "tool_allowed",
]
