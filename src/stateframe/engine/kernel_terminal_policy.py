"""Run-ending policy, including typed no-progress stops for unbounded runs."""
from __future__ import annotations

import os
from collections.abc import MutableSequence

from kernel_step_limits import runtime_unbounded


MODE_ENV = "STATEFRAME_HARD_BUDGET_ONLY_TERMINALS"
_STATE_ATTR = "_hard_budget_terminal_advisories"

HARD_BUDGET_OUTCOMES = frozenset({
    "attempt_budget_exhausted", "bounded", "budget_exhausted", "ceiling_reached",
    "failed_budget", "fuel_exhausted", "response_budget_exhausted", "step_budget_exhausted",
    "tool_call_cap",
})
DONE_OUTCOMES = frozenset({"accepted", "completed", "uncertified_submit"})
FATAL_OUTCOMES = frozenset({
    "boundary_unrecoverable", "env_broken", "harness_contract", "harness_error",
    "harness_transport", "harden8_contract_error", "harden8_evidence_pin_error",
    "harden8_owner_finalization_error", "harden8_route_error", "invalidate_attempt",
    "quarantine_harness", "run_incomplete",
})
# User-configured hook policy ends the run: a stop hook's veto limit, or lost hook containment. These are
# explicit user decisions or safety stops, never heuristics, so they stay authoritative under hard-budget mode.
USER_POLICY_OUTCOMES = frozenset({"hook_blocked", "hook_containment"})

_ADVISORY_MESSAGES = {
    "blocked": "The host marked the current completion attempt blocked. Continue with the available repair, evidence, or verification actions until a hard budget is reached or completion is accepted.",
    "catalog_unknown": "The host could not classify the current catalog state. Continue by inspecting the available evidence and taking an admissible repair or verification action.",
    "completion_owner_exhausted": "The completion owner found no current submission route. Continue with available product, read, and verification actions; the hard run budgets still apply.",
    "continuation_blocked": "The continuation authority could not certify the current boundary. Continue with the available task actions and resolve the open work.",
    "effect_boundary_shadow": "The effect boundary reported an unresolved shadow result. Continue with an admissible action and use the remaining run budget.",
    "greenlock_empty_menu": "The green-lock projection produced no permitted tool. Continue with the available task context and request an admissible action; hard budgets still bound the run.",
    "owner_reserved_submit_cap": "The owner's reserved submission limit was reached without an accepted submission. Continue with available actions until a hard run budget is reached.",
    "postgreen_no_progress": "Recent post-green responses made no measurable progress. Choose a new repair, verification, or evidence action and continue.",
    "postgreen_stall": "The post-green stall signal fired. Continue with a different repair, verification, or evidence action.",
    "repair_blocks_done_cap": "Repeated completion attempts reached the repair-block advisory cap. Keep the repair tools available, address the reported blocker, and continue.",
    "response_reuse_stall": "The same verification was repeated without new evidence. Choose a useful repair or a fresh verification and continue.",
    "stage_exit_budget": "The stage exit budget was reached without a typed stage answer. Continue with the available stage actions; the run-wide hard budgets still apply.",
    "tool_call_loop": "A repeated tool-call pattern was detected. Change the next action and continue within the hard run budgets.",
    "unparsed_replies": "Several replies did not match the available action format. Use one available action now; the run remains active.",
    "done_stub_gate": "The completion check found a stub or placeholder. Repair the reported issue and continue.",
    "done_ship_tree_smoke": "The fresh import smoke found an issue in the current tree. Repair the reported issue and continue.",
    "done_probe_gate": "The probe completion check is still open. Resolve the reported requirement and continue.",
    "done_contract_audit": "The contract audit is still open. Resolve the reported contract issue and continue.",
    "done_turtle_defect": "The defect review found an unresolved issue. Repair or verify it and continue.",
    "done_public_review": "The public review found an unresolved issue. Address its evidence and continue.",
    "done_witness_challenge_cap": "The repeated completion challenge reached its advisory cap. Continue with the available product, evidence, or verification actions.",
    "v2_terminal": "The disposition authority marked this attempt incomplete. Continue working until a hard budget is reached or completion is accepted.",
    "generic_terminal": "A host policy requested a stop without a hard-budget or fatal-harness reason. Continue with the available task actions.",
}


def enabled(runtime: object | None = None, environ: dict[str, str] | None = None) -> bool:
    configured = getattr(runtime, "stateframe_hard_budget_only_terminals_mode", None)
    source = os.environ if environ is None else environ
    raw = configured if configured is not None else source.get(MODE_ENV, "on")
    value = str(raw or "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{MODE_ENV} must be on or off")
    return value == "on"


def terminal_category(outcome: object, runtime: object | None = None) -> str:
    """Classify only explicit end reasons; unknown/new reasons fail closed as heuristic."""
    value = str(outcome or "").strip().lower()
    if runtime_unbounded(runtime) and value in {
            "stalled", "tool_call_loop_breaker", "post_green_stall", "postgreen_no_progress"}:
        return "stuck"
    if value in HARD_BUDGET_OUTCOMES:
        return "budget"
    if value in DONE_OUTCOMES:
        return "done"
    if value in FATAL_OUTCOMES:
        return "fatal"
    if value in USER_POLICY_OUTCOMES:
        return "policy"
    return "heuristic"


def terminal_outcome_permitted(outcome: object, runtime: object | None = None) -> bool:
    return terminal_category(outcome, runtime) in {"budget", "done", "fatal", "policy", "stuck"}


def advisory_message(trigger: str) -> str:
    key = str(trigger or "generic_terminal").strip().lower()
    return _ADVISORY_MESSAGES.get(key, _ADVISORY_MESSAGES["generic_terminal"])


def _state(runtime: object) -> dict[str, object]:
    current = getattr(runtime, _STATE_ATTR, None)
    if isinstance(current, dict):
        seen = current.get("seen")
        pending = current.get("pending")
        if isinstance(seen, set) and isinstance(pending, list):
            return current
    current = {"seen": set(), "pending": []}
    try:
        setattr(runtime, _STATE_ATTR, current)
    except (AttributeError, TypeError):
        return {"seen": set(), "pending": []}
    return current


def queue_advisory(runtime: object, trigger: str) -> bool:
    """Queue one static, safe model-facing card per trigger for the current run."""
    if not enabled(runtime):
        return False
    key = str(trigger or "generic_terminal").strip().lower()[:80] or "generic_terminal"
    if key not in _ADVISORY_MESSAGES:
        key = "generic_terminal"
    state = _state(runtime)
    seen = state["seen"]
    pending = state["pending"]
    if not isinstance(seen, set) or not isinstance(pending, list):
        return False
    if key in seen:
        return False
    if len(seen) >= 64:
        key = "generic_terminal"
        if key in seen:
            return False
    seen.add(key)
    message = advisory_message(key)
    pending.append({"trigger": key, "message": message})
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        try:
            emit("run_terminal_advisory", producer="kernel_terminal_policy",
                 trigger=key, category="heuristic", dispatch=False)
        except Exception:  # noqa: BLE001 - telemetry cannot change continuation
            pass
    return True


def deliver_pending_advisories(runtime: object, *message_lists: object) -> int:
    """Append queued cards to each distinct mutable request-history list, once."""
    state = _state(runtime)
    pending_value = state.get("pending")
    if not isinstance(pending_value, list) or not pending_value:
        return 0
    pending = list(pending_value)
    destinations: list[MutableSequence[object]] = []
    seen_ids: set[int] = set()
    for value in message_lists:
        if (isinstance(value, MutableSequence) and id(value) not in seen_ids):
            seen_ids.add(id(value))
            destinations.append(value)
    if not destinations:
        return 0
    for card in pending:
        if not isinstance(card, dict) or not isinstance(card.get("message"), str):
            continue
        message = {"role": "user", "content": f"[kernel advisory] {card['message']}"}
        for destination in destinations:
            if not destination or destination[-1] != message:
                destination.append(dict(message))
    pending_value.clear()
    return len(pending)


def record_terminal(runtime: object, outcome: object, *, trigger: str,
                    messages: object = None) -> bool:
    """Keep an allowed terminal; downgrade a heuristic terminal to an advisory."""
    if not enabled(runtime) or terminal_outcome_permitted(outcome, runtime):
        setattr(runtime, "system_outcome", outcome)
        return True
    queue_advisory(runtime, trigger)
    deliver_pending_advisories(runtime, messages)
    return False


__all__ = [
    "DONE_OUTCOMES", "FATAL_OUTCOMES", "HARD_BUDGET_OUTCOMES", "MODE_ENV", "USER_POLICY_OUTCOMES",
    "advisory_message", "deliver_pending_advisories", "enabled", "queue_advisory",
    "record_terminal", "terminal_category", "terminal_outcome_permitted",
]
