"""RF02: an idempotent open-work menu delta at the v46 cutoff boundary.

This module owns only the pure menu calculation.  The completion owner remains
the sole authority that decides when the result is admitted and rendered.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_FL_CUTOFF_RULE"
SCHEMA = "stateframe-fl-cutoff-rule/1"
EVENT = "fl_cutoff_rule_menu"
CORE_MENU = ("edit", "str_replace", "read")
_TRUTHY = frozenset({"1", "on", "true", "yes"})


class CutoffRuleError(ValueError):
    """Raised when a menu does not have the shape required by RF02."""


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise CutoffRuleError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_fl_cutoff_rule_mode", None)
    if configured is not None:
        return str(configured).strip().lower() in _TRUTHY
    return mode() == "on"


def revision_changed(runtime: object | None = None) -> bool:
    """Return only an explicit public-revision change, failing closed otherwise."""
    if runtime is None:
        return False
    for name in (
        "stateframe_cutoff_revision_changed",
        "public_revision_changed",
        "revision_changed",
    ):
        value = getattr(runtime, name, None)
        if type(value) is bool:
            return value
    unchanged = getattr(runtime, "revision_unchanged", None)
    if type(unchanged) is bool:
        return not unchanged
    state = str(getattr(runtime, "revision_state", "") or "").strip().lower()
    return state in {"changed", "modified", "stale", "invalidated"}


_CLOSED_STATUSES = frozenset({"CLOSED", "PASS", "PASSED", "SATISFIED", "GREEN", "OK"})


def _rows(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        if any(key in value for key in (
                "id", "clause_id", "requirement_id", "obligation_id", "check_id",
                "status", "verdict", "result", "clause", "text")):
            return (value,)
        return tuple(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    if value in (None, ""):
        return ()
    return (value,)


def _row_id(value: object) -> str:
    if isinstance(value, Mapping):
        for key in ("id", "clause_id", "requirement_id", "obligation_id", "check_id"):
            candidate = value.get(key)
            if candidate not in (None, ""):
                return str(candidate).strip()
        return ""
    for name in ("id", "clause_id", "requirement_id", "obligation_id", "check_id"):
        candidate = getattr(value, name, None)
        if candidate not in (None, ""):
            return str(candidate).strip()
    return str(value).strip()


def _row_open(value: object) -> bool:
    if isinstance(value, Mapping):
        status = str(value.get("status", value.get("state", value.get("verdict", value.get("result", "")))))
    else:
        status = str(getattr(value, "status", getattr(value, "state", getattr(
            value, "verdict", getattr(value, "result", "")))))
    return status.strip().upper() not in _CLOSED_STATUSES


def _mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            converted = converter()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            converted = None
        if isinstance(converted, Mapping):
            return converted
    return None


def _ids_from(value: object) -> tuple[str, ...]:
    result: list[str] = []
    for row in _rows(value):
        identifier = _row_id(row)
        if identifier and _row_open(row) and _row_concrete(row) and identifier not in result:
            result.append(identifier)
    return tuple(result)


def _row_concrete(value: object) -> bool:
    if isinstance(value, Mapping):
        for key in ("concrete_required", "coverage_required"):
            if key in value:
                return bool(value[key])
        return True
    return bool(getattr(value, "concrete_required", False)
                or getattr(value, "check_ids", ())
                or getattr(value, "receipts", ())
                or getattr(value, "evidence_refs", ()))


def concrete_open_ids(runtime: object | None) -> tuple[str, ...]:
    """Return concrete requirements, excluding uncertified-submit bookkeeping.

    ``_v39_uncertified_submit_ids`` is an internal terminal alternative, not
    proof that model-repair work remains.  RF02 follows explicit clause/evidence
    rows so it cannot reopen repair tools after a successful terminal edit.
    """
    if runtime is None:
        return ()
    # Host done-facts are authoritative after an automatic verification.  An
    # explicit empty unresolved-gate list must suppress stale assessment rows.
    for name in ("done_facts_payload", "done_facts_claims", "host_done_facts"):
        facts = _mapping(getattr(runtime, name, None))
        if facts is not None and "unresolved_gating_items" in facts:
            return _ids_from(facts["unresolved_gating_items"])
    sources: list[object] = []
    for name in (
            "cutoff_concrete_required", "concrete_required", "open_concrete_required",
            "_challenge_budget_open_evidence", "_challenge_budget_unresolved_gating_items",
            "host_closure_assessments", "open_clause_rows"):
        value = getattr(runtime, name, None)
        if value not in (None, (), [], ""):
            sources.append(value)
    if not sources:
        try:
            from kernel_challenge_budget import open_clause_rows
            sources.append(open_clause_rows(runtime=runtime))
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
    if not sources:
        texts = _rows(getattr(runtime, "open_clause_texts", ()))
        ids = _rows(getattr(runtime, "open_requirement_ids", ()))
        sources.append(tuple({"id": _row_id(identifier), "text": str(text)}
                             for identifier, text in zip(ids, texts)))
    result: list[str] = []
    for source in sources:
        for identifier in _ids_from(source):
            if identifier not in result:
                result.append(identifier)
    return tuple(result)


def _tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    direct = tool.get("name")
    if direct:
        return str(direct)
    function = tool.get("function")
    return str(function.get("name", "")) if isinstance(function, Mapping) else ""


def names(tools: object) -> tuple[str, ...]:
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return ()
    result: list[str] = []
    for tool in tools:
        name = _tool_name(tool)
        if name:
            result.append(name)
    if len(set(result)) != len(result):
        raise CutoffRuleError("menu contains duplicate tool names")
    return tuple(result)


def desired_names(
        recorded: Sequence[str], *, available: Sequence[str] = (),
        revision_changed: bool = False,
) -> tuple[str, ...] | None:
    """Return RF02's exact menu delta, or ``None`` when it cannot be admitted.

    A clean core menu is returned in its recorded order to preserve byte
    identity.  A mismatched menu uses the generator's canonical order and keeps
    ``run_tests`` only for an explicitly changed public revision.
    """
    recorded_names = tuple(str(name) for name in recorded if str(name))
    available_names = tuple(str(name) for name in available if str(name))
    if len(set(recorded_names)) != len(recorded_names):
        raise CutoffRuleError("recorded menu contains duplicate tool names")
    if len(set(available_names)) != len(available_names):
        raise CutoffRuleError("available menu contains duplicate tool names")
    if all(name in recorded_names for name in CORE_MENU):
        if set(recorded_names) == set(CORE_MENU):
            return recorded_names
        source_names = recorded_names
    else:
        # A preceding terminal projection may have narrowed the current list
        # to done/run_tests.  Recover only from the retained full surface.
        if not all(name in available_names for name in CORE_MENU):
            return None
        source_names = recorded_names
    desired = list(CORE_MENU)
    if revision_changed and "run_tests" in source_names:
        desired.append("run_tests")
    return tuple(desired)


def project_tools(
        recorded_tools: Sequence[Mapping[str, object]],
        available_tools: Sequence[Mapping[str, object]], *,
        changed: bool,
) -> list[dict[str, object]] | None:
    """Project the exact available tool definitions in the RF02 menu order."""
    recorded_names = names(recorded_tools)
    available_names = names(available_tools)
    desired = desired_names(
        recorded_names, available=available_names, revision_changed=changed)
    if desired is None:
        return None
    by_name: dict[str, Mapping[str, object]] = {}
    for tool in (*recorded_tools, *available_tools):
        name = _tool_name(tool)
        if name and name not in by_name:
            by_name[name] = tool
    if any(name not in by_name for name in desired):
        return None
    if desired == recorded_names:
        return [copy.deepcopy(dict(tool)) for tool in recorded_tools]
    return [copy.deepcopy(dict(by_name[name])) for name in desired]


__all__ = [
    "CORE_MENU", "CutoffRuleError", "EVENT", "MODE_ENV", "SCHEMA",
    "concrete_open_ids", "desired_names", "enabled", "mode", "names", "project_tools",
    "revision_changed",
]
