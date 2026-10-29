"""Opt-in policy that promotes live aligned counterexamples to done blockers.

The existing readiness authority treats a valid aligned observation as
``repair_optional``.  This module is the small, separately gated policy arm
which changes only that disposition.  It owns no prose parser and accepts only
alignment evidence already produced by ``kernel_requirement_alignment``.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_REPAIR_BLOCKS_DONE"
K_NON_PROGRESS_RESPONSES = 6
STOPPED_INCOMPLETE = "STOPPED_INCOMPLETE"
BLOCKED_PHASE = "blocked_counterexample"

_CLOSED = frozenset({"PASS", "PASSED", "SATISFIED", "RESOLVED", "CLOSED"})
_INVALID_MARKERS = (
    "UNBOUND", "UNBINDABLE", "SELF_CONTRADICT", "SELF_CONTRADICTORY",
    "INVALID", "WRONG_BINDING", "REBOUND", "UNSATISFIABLE", "CONTRADICTS",
)


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def runtime_enabled(runtime: object | None) -> bool:
    if runtime is None:
        return False
    marker = getattr(runtime, "stateframe_repair_blocks_done_mode", None)
    if marker is not None:
        return marker == "on"
    try:
        return enabled()
    except ValueError:
        return False


def _field(row: Mapping[str, object], *names: str, default: object = None) -> object:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def _upper(value: object) -> str:
    if isinstance(value, Mapping):
        value = _field(value, "status", "value", "name", "reason", "type",
                       "classification", "binding", default="")
    return str(value or "").strip().upper().replace("-", "_")


def _check_id(row: Mapping[str, object]) -> str:
    return str(_field(row, "check_id", "obligation_id", "id", default="") or "")


def _has_literal_alignment(row: Mapping[str, object]) -> bool:
    evidence = row.get("alignment_evidence")
    if not bool(row.get("literal_grounded")) or not isinstance(evidence, Mapping):
        return False
    literals = evidence.get("matched_literals")
    if isinstance(literals, str):
        return bool(literals.strip())
    return isinstance(literals, Sequence) and not isinstance(literals, (bytes, bytearray)) \
        and any(str(item).strip() for item in literals)


def _has_repair_target(row: Mapping[str, object]) -> bool:
    for name in ("target_file", "target_path", "target_symbol", "symbol"):
        if str(row.get(name) or "").strip():
            return True
    for name in ("mutation_site", "mutation_site_refs", "mutation_sites", "mutation_sites_refs"):
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if any(item not in (None, "", {}) for item in value):
                return True
    return False


def _current_revision(row: Mapping[str, object]) -> set[str]:
    values: set[str] = set()
    revision = _field(row, "revision", "current_revision", "product_revision", default=None)
    if revision is not None:
        values.add(str(revision))
    aliases = row.get("revision_aliases")
    if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes, bytearray)):
        values.update(str(item) for item in aliases if item not in (None, ""))
    return values


def _invalid_classification(row: Mapping[str, object]) -> bool:
    if any(row.get(name) is True for name in (
            "invalid", "unbindable", "self_contradictory")):
        return True
    if row.get("valid") is False or row.get("binding_valid") is False:
        return True
    for name in ("classification", "demotion_reason", "reason", "binding_status",
                 "entrypoint_binding"):
        value = row.get(name)
        text = _upper(value)
        if any(marker in text for marker in _INVALID_MARKERS):
            return True
    binding = _field(row, "binding", "binding_status", "entrypoint_binding", default=None)
    if isinstance(binding, Mapping):
        binding = _field(binding, "status", "verdict", "binding", "value", default=None)
    if binding is not None and _upper(binding) != "BOUND":
        return True
    return False


def is_repair_blocking_counterexample(row: object, *, revision: object = None) -> bool:
    """Return true only for a current, executed, literal-aligned valid row."""
    if not isinstance(row, Mapping):
        return False
    if bool(row.get("counterexample_authority")) or _invalid_classification(row):
        return False
    status = _upper(_field(row, "status", "underlying_status", default="COUNTEREXAMPLE"))
    if status in _CLOSED:
        return False
    if _upper(_field(row, "execution", "execution_status", default="")) != "EXECUTED":
        return False
    if not _has_literal_alignment(row) or not _has_repair_target(row):
        return False
    if revision is None or str(revision) not in _current_revision(row):
        return False
    return True


def blocking_rows(rows: Sequence[Mapping[str, object]] | object,
                  revision: object = None, *, active: bool | None = None
                  ) -> tuple[Mapping[str, object], ...]:
    """Select and copy the rows promoted by the opt-in policy."""
    if active is None:
        active = enabled()
    if not active:
        return ()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return ()
    result: list[Mapping[str, object]] = []
    for item in rows:
        if is_repair_blocking_counterexample(item, revision=revision):
            result.append(copy.deepcopy(dict(item)))
    return tuple(result)


def clause_for(row: Mapping[str, object]) -> str:
    return str(_field(row, "clause", "expected", "requirement", default="") or "")


def repair_target_for(row: Mapping[str, object]) -> str:
    path = str(_field(row, "target_file", "target_path", "file", default="") or "")
    symbol = str(_field(row, "target_symbol", "symbol", default="") or "")
    if path and symbol:
        return f"{path}:{symbol}"
    if path or symbol:
        return path or symbol
    for name in ("mutation_site", "mutation_site_refs", "mutation_sites"):
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                if isinstance(item, Mapping):
                    item_path = str(_field(item, "path", "file", default="") or "")
                    item_symbol = str(_field(item, "symbol", "name", default="") or "")
                    if item_path and item_symbol:
                        return f"{item_path}:{item_symbol}"
                    if item_path:
                        return item_path
                elif item not in (None, ""):
                    return str(item)
    return "the repair site"


def blocker_ids(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_check_id(row) for row in rows if _check_id(row)))


def stopped_refusal(rows: Sequence[Mapping[str, object]] = (), *, count: int | None = None) -> str:
    details = []
    for row in rows:
        clause = clause_for(row)
        target = repair_target_for(row)
        if clause:
            details.append(f"{clause[:200]} (repair target: {target})")
        else:
            details.append(f"repair target: {target}")
    suffix = ("; " + "; ".join(details)) if details else ""
    count_text = f" after {count} non-progress responses" if count is not None else ""
    return (f"{STOPPED_INCOMPLETE}: completion is stopped{count_text}; "
            "the live counterexample remains unresolved" + suffix)


def _emit(runtime: object, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_repair_blocks_done", **fields)


def reset_progress(runtime: object) -> None:
    if runtime_enabled(runtime) and not getattr(runtime, "_repair_blocks_done_stopped", False):
        setattr(runtime, "_repair_blocks_done_non_progress", 0)


def is_stopped(runtime: object | None) -> bool:
    return bool(runtime is not None and runtime_enabled(runtime)
                and getattr(runtime, "_repair_blocks_done_stopped", False))


def terminal_stop_active(runtime: object | None) -> bool:
    """Legacy stop flag is authoritative only when hard-budget-only mode is off."""
    if not is_stopped(runtime):
        return False
    from kernel_terminal_policy import enabled as hard_budget_only_enabled
    return not hard_budget_only_enabled(runtime)


def record_non_progress(runtime: object, *, decision: object = None,
                        rows: Sequence[Mapping[str, object]] = (),
                        action_kind: str = "") -> bool:
    """Count blocked responses unless a done submission has already been sealed."""
    from kernel_done_is_submit_authority import done_sealed

    if done_sealed(runtime) is not None:
        return False
    if not runtime_enabled(runtime):
        return False
    phase = str(getattr(decision, "phase", "")) if decision is not None else ""
    if phase and phase != BLOCKED_PHASE:
        return False
    from kernel_terminal_policy import (
        enabled as hard_budget_only_enabled,
        queue_advisory,
    )
    hard_budget_only = hard_budget_only_enabled(runtime)
    if terminal_stop_active(runtime):
        return True
    count = int(getattr(runtime, "_repair_blocks_done_non_progress", 0) or 0) + 1
    if hard_budget_only:
        count = min(count, K_NON_PROGRESS_RESPONSES)
    setattr(runtime, "_repair_blocks_done_non_progress", count)
    ids = blocker_ids(rows)
    if decision is not None:
        ids = tuple(dict.fromkeys((*ids, *getattr(decision, "blocker_ids", ()))))
        rows = tuple(getattr(decision, "blockers", ()) or ()) or rows
    _emit(runtime, "repair_blocks_done_non_progress", action_kind=action_kind,
          refusal_count=count, check_ids=list(ids))
    if count < K_NON_PROGRESS_RESPONSES:
        return False
    if hard_budget_only:
        if queue_advisory(runtime, "repair_blocks_done_cap"):
            _emit(runtime, "repair_blocks_done_advisory", action_kind=action_kind,
                  refusal_count=count, check_ids=list(ids), advisory=True)
        return False
    setattr(runtime, "_repair_blocks_done_stopped", True)
    setattr(runtime, "system_outcome", STOPPED_INCOMPLETE)
    setattr(runtime, "completion_accepted", False)
    setattr(runtime, "v2_accepted_done", False)
    _emit(runtime, "repair_blocks_done_stopped_incomplete", action_kind=action_kind,
          refusal_count=count, check_ids=list(ids))
    return True


__all__ = [
    "BLOCKED_PHASE", "K_NON_PROGRESS_RESPONSES", "MODE_ENV", "STOPPED_INCOMPLETE",
    "blocking_rows", "blocker_ids", "clause_for", "enabled", "is_repair_blocking_counterexample",
    "is_stopped", "mode", "record_non_progress", "repair_target_for", "reset_progress",
    "runtime_enabled", "stopped_refusal", "terminal_stop_active",
]
