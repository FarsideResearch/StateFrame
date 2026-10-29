"""Bounded, host-owned terminal choice for the v42 live request boundary."""
from __future__ import annotations

import copy
from contextvars import ContextVar
import hashlib
import json
import os
import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Callable
from pathlib import Path

from kernel_completion_authority import refusal_reason_codes as _authority_refusal_reason_codes
from kernel_harden8 import record_failure
from kernel_done_acceptance import done_accepted


MODE_ENV = "STATEFRAME_COMPLETION_OWNER"
NEXT_ACTION_OWNER_ENV = "STATEFRAME_NEXT_ACTION_OWNER"
NO_DONE_ONLY_COLLAPSE_ENV = "STATEFRAME_NO_DONE_ONLY_COLLAPSE"
STALL_RULE_ENV = "STATEFRAME_FL_STALL_RULE"
REQUEST_CONTRACT_REVISION_REBIND_ENV = "STATEFRAME_REQUEST_CONTRACT_REVISION_REBIND"
SCHEMA = "stateframe-completion-owner/1"
REQUEST_CONTRACT_SCHEMA = "stateframe-request-action-contract/1"
PROVIDER_REQUEST_DISPOSITION_SCHEMA = "stateframe-provider-request-disposition/1"
PROVIDER_REQUEST_NOT_DISPATCHED = "NOT_DISPATCHED"
PROVIDER_REQUEST_NOT_DISPATCHED_REASON_CODES = frozenset({
    "completion_owner_advisory",
    "completion_owner_boundary",
    "completion_owner_host_stop",
    "prepared_overflow",
    "response_budget_unavailable",
    "tool_call_loop_breaker",
})
LEAN_REQUEST_CONTRACT_KIND = "lean_stable_surface"
OWNER_BLOCK_SCHEMA = "stateframe-owner-block/1"
OWNER_MESSAGE_BLOCK_KEY = "_stateframe_owner_block"
TERMINAL_CHOICE_HEADER = "[COMPLETION OWNER]"
FINAL_SUBMIT_EXHAUSTED = "COMPLETION_OWNER_EXHAUSTED"
MAX_RESERVED_RESPONSES = 2
GREEN_TERMINAL_LINE = "public checks GREEN on this unchanged revision; submit with done"
NO_HOST_CHECKS_TERMINAL_LINE = (
    "NO HOST CHECKS: no public host check exists; Final step: call done now to submit "
    "this revision with that limitation disclosed; any open witness limitation is disclosed."
)
_STALL_RULE_TRUTHY = frozenset({"1", "on", "true", "yes"})
_STALL_RULE_TOOLS = ("edit", "str_replace", "read")
_STALL_RULE_EDIT_TOOLS = frozenset({"edit", "str_replace"})
_DECLARED_REPAIR_TOOLS = ("edit", "str_replace", "read")
_CHURN_DECISIONS = frozenset({"ALLOW", "VERIFY_REQUIRED", "RECOVERY_REQUIRED", "STOP_INCOMPLETE"})
MAX_DECLARED_REPAIR_RESPONSES = 3
_DECLARED_REPAIR_CUE_RE = re.compile(
    r"\b(?:outstanding|remaining|unresolved|open|still|must|need(?:s|ed)?)\b"
    r".{0,90}\b(?:repair|repairs|fix|fixes|correct|correction|issue|issues|"
    r"clause|clauses|work|failure|failing|wrong|missing)\b|"
    r"\b(?:repair|fix|correct|address)\w*\b.{0,90}\b(?:clause|clauses|"
    r"language|status|response|return|requirement|failure|issue)\b",
    re.IGNORECASE | re.DOTALL,
)
_DECLARED_REPAIR_RETRACTION_RE = re.compile(
    r"\b(?:no|none|nothing)\b.{0,60}\b(?:outstanding|remaining|unresolved|open|"
    r"repair|repairs|fix|issues?)\b|\b(?:all|every)\b.{0,60}\b(?:fixed|repaired|"
    r"resolved|complete|completed)\b",
    re.IGNORECASE | re.DOTALL,
)
_STALL_RULE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:::[A-Za-z_]\w*)?"
)
_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")
REFUSAL_REASON_CODES = frozenset({
    "ADMITTED_BASIS_UNCHANGED", "BASIS_CHANGED", "HOST_OBLIGATION",
    "NO_FEASIBLE_REPAIR_ACTION", "NEEDS_WORK", "COMMIT_FAILED",
    "COMPLETION_SEALED", "CLOSURE_UNAFFORDABLE", "SHIP_TREE_SMOKE",
    "ORACLE_TESTS_RED", "OPEN_COUNTEREXAMPLE", "BOUND_COUNTEREXAMPLE",
    "V2_DISPOSITION", "V2_TERMINAL", "STOPPED_INCOMPLETE", "UNIFIED_VIOLATED",
    "UNIFIED_DELIVERY_BLOCKED", "UNIFIED_UNKNOWN_UNCERTIFIED",
    "BEHAVIOR_WITNESS_REQUIRED", "BEHAVIOR_WITNESS_REQUEST_CAP",
    "BEHAVIOR_WITNESS_FALLBACK_UNAVAILABLE", "PUBLIC_RED_DONE",
    "PUBLIC_CHECKS_NOT_RUN_CURRENT_REVISION", "UNVERIFIED_DISCLOSED_CHECKS",
    "HSTATE_PLUS_DONE_CHALLENGE", "SUBMISSION_BLOCKED", "PREPUBLISH_DIFF_BLOCKED",
    "CONTINUATION_BLOCKED", "PENDING_SUBMISSION", "UNRESOLVED_SUBMISSION",
    "HOST_BINDING_FAILURE", "GREEN_LOCK_NO_PERMITTED_ACTION", "UNBOUND_OBSERVABLE",
    "NO_VERIFIER", "MISSING_CHECKS", "REQUIRED_EVIDENCE_GAP", "CLAUSE_COVERAGE",
    "COVERAGE_OPEN", "INTERNAL_COVERAGE_OPEN", "INTERNAL_METADATA_OPEN",
    "METADATA_OPEN", "INCONCLUSIVE", "UNKNOWN", "ADVISORY_ONLY",
    "LOW_COVERAGE_UNVERIFIED", "NAMED_LAYER_COVERAGE_UNVERIFIED",
    "RESPONSE_BUDGET_EXHAUSTED_READY", "POST_GREEN_STALL_READY",
    "POSTGREEN_BUDGET_READY", "READINESS_CERTIFIED_READY",
    "KEYED_SUBMISSION_PENDING", "KEYED_SUBMISSION_UNRESOLVED",
    "POSTGREEN_NO_PROGRESS_CAP", "CONTRACT_INCOMPATIBLE", "CONTRACT_CHECK_REQUIRED",
    "POSTGREEN_IDENTICAL_VERIFICATION",
    "CAPABILITY_DENIED", "DETREASON_SKEPTIC", "OPEN_COUNTEREXAMPLE_MENU_MISSING_EDIT",
    "READINESS_COUNTEREXAMPLE_OPEN",
} | set(_authority_refusal_reason_codes()))
_WITNESS_CODES = frozenset({
    "BEHAVIOR_WITNESS_REQUIRED", "BEHAVIOR_WITNESS_REQUEST_CAP",
    "BEHAVIOR_WITNESS_FALLBACK_UNAVAILABLE", "UNBOUND_OBSERVABLE",
    "NO_VERIFIER", "MISSING_CHECKS", "behavior_witness_pending",
})
_CHECK_CODES = frozenset({
    "ADMITTED_BASIS_UNCHANGED", "BASIS_CHANGED", "PUBLIC_CHECKS_NOT_RUN_CURRENT_REVISION",
    "UNVERIFIED_DISCLOSED_CHECKS", "UNIFIED_UNKNOWN_UNCERTIFIED", "MISSING_CHECKS",
    "CHECK_INCONCLUSIVE",
    "REQUIRED_EVIDENCE_GAP", "CLAUSE_COVERAGE", "COVERAGE_OPEN",
    "INTERNAL_COVERAGE_OPEN", "INTERNAL_METADATA_OPEN", "METADATA_OPEN",
    "INCONCLUSIVE", "UNKNOWN",
})
_REPAIR_CODES = frozenset({
    "HOST_OBLIGATION", "NEEDS_WORK", "SHIP_TREE_SMOKE", "ORACLE_TESTS_RED",
    "OPEN_COUNTEREXAMPLE", "BOUND_COUNTEREXAMPLE", "V2_DISPOSITION",
    "UNIFIED_VIOLATED", "PUBLIC_RED_DONE", "PRODUCT_VIOLATION", "UNRESOLVED_DEFECT",
    "TESTS_NOT_GREEN", "NONZERO_EXIT", "COUNTEREXAMPLE", "OPEN_BLOCKER",
    "INWORLD_OBLIGATION", "STUB_GATE", "PROBE_GATE", "CONTRACT_AUDIT", "SUBMISSION_BLOCKED",
    "DETREASON_SKEPTIC", "OPEN_COUNTEREXAMPLE_MENU_MISSING_EDIT", "READINESS_COUNTEREXAMPLE_OPEN",
})
_WORK_CODES = frozenset({"OPEN_WORK", "DECLARED_WORK"})
_QUESTION_CODES = frozenset({"QUESTION_CLAUSE"})
_CONTROL_CODES = frozenset({
    "CONTROL_FAULT", "CONTROL_CONFLICT", "NO_FEASIBLE_RECOVERY", "COMMIT",
    "HARNESS_ERROR", "SETUP_ERROR", "FUEL_EXHAUSTED", "BUDGET_EXHAUSTED",
    "DEADLINE", "COMPLETION_BLOCKED", "ADMITTED", "BOLT_GATE",
    "CONTINUATION_BLOCKER", "SIDEKICK_CHALLENGE", "PUBLIC_REVIEW", "UNVERIFIED",
    "TURTLE_DEFECT",
})
_RETRY_DONE_CODES = frozenset({"HSTATE_PLUS_DONE_CHALLENGE"})
_AUTHORITY_EVIDENCE_CODES = frozenset({
    "REQUIRED_EVIDENCE_GAP", "RECHECK_REQUIRED", "STALE_EVIDENCE", "STALE",
    "MISSING_SELECTOR", "NOT_EXECUTED", "AMBIGUOUS_CONTRACT", "UNBOUND_ENTRYPOINT",
    "UNBOUND_INVOCATION", "SKIPPED", "CONFLICTING_RECEIPTS", "CONFLICTING_RESULTS",
    "SELECTOR_ERROR", "VERIFY", "TESTS_NOT_RUN", "SOURCE_EVIDENCE", "READ_EVIDENCE",
    "EVIDENCE_GAP", "ORACLE_UNAVAILABLE", "LATE_VETO", "UNVERIFIED",
})
_CHECK_CODES = _CHECK_CODES | _AUTHORITY_EVIDENCE_CODES
_HOST_ONLY_CODES = frozenset({
    "NO_FEASIBLE_REPAIR_ACTION", "COMMIT_FAILED", "COMPLETION_SEALED",
    "CLOSURE_UNAFFORDABLE", "V2_TERMINAL", "STOPPED_INCOMPLETE",
    "UNIFIED_DELIVERY_BLOCKED", "CONTINUATION_BLOCKED", "PENDING_SUBMISSION",
    "UNRESOLVED_SUBMISSION", "HOST_BINDING_FAILURE", "GREEN_LOCK_NO_PERMITTED_ACTION",
    "PREPUBLISH_DIFF_BLOCKED", "ADVISORY_ONLY", "RESPONSE_BUDGET_EXHAUSTED_READY",
    "POST_GREEN_STALL_READY", "KEYED_SUBMISSION_PENDING", "KEYED_SUBMISSION_UNRESOLVED",
    "POSTGREEN_NO_PROGRESS_CAP", "CONTRACT_INCOMPATIBLE", "CONTRACT_CHECK_REQUIRED",
    "CAPABILITY_DENIED",
})
_COVERAGE_CODES = frozenset({
    "LOW_COVERAGE_UNVERIFIED", "NAMED_LAYER_COVERAGE_UNVERIFIED",
})

_REQUEST_ROUTES = frozenset({"repair", "verify", "submit", "host-stop"})
_HOST_STOP_OUTCOMES = frozenset({
    FINAL_SUBMIT_EXHAUSTED, "COMPLETION_OWNER_EXHAUSTED", "UNCERTIFIED_SUBMIT",
})
_ACTIVE_ADVISORY_EVALUATIONS: ContextVar[frozenset[int]] = ContextVar(
    "completion_owner_active_advisory_evaluations", default=frozenset(),
)
_OWNER_BLOCK_TYPES = frozenset({"terminal_line", "evidence_pin", "menu"})
_EVIDENCE_GAP_CODES = frozenset({
    "ADVISORY_ONLY", "CHECK_INCONCLUSIVE", "CLAUSE_COVERAGE", "COVERAGE_OPEN",
    "INCONCLUSIVE", "INTERNAL_COVERAGE_OPEN", "INTERNAL_METADATA_OPEN",
    "LOW_COVERAGE_UNVERIFIED", "METADATA_OPEN", "MISSING_CHECKS", "NO_VERIFIER",
    "NAMED_LAYER_COVERAGE_UNVERIFIED", "PUBLIC_CHECKS_NOT_RUN_CURRENT_REVISION",
    "REQUIRED_EVIDENCE_GAP", "UNBOUND_OBSERVABLE", "UNIFIED_UNKNOWN_UNCERTIFIED",
    "UNKNOWN", "UNVERIFIED_DISCLOSED_CHECKS",
})
_EVIDENCE_ROUTE_TOOLS = frozenset({
    "authored_check", "dispute_check", "probe", "read_check_receipt", "run_tests",
})
_PRODUCT_EDIT_TOOLS = frozenset({
    "edit", "file_create", "replace", "replace_module", "str_replace", "write",
})
_ADVISORY_ERROR_KIND = "completion_owner_advisory_gate_error"
_ADVISORY_FAILURE_CAP = 100_000


@dataclass(frozen=True)
class CompletionRouteDecision:
    """One bounded completion route shared by prompt, menu, diagnosis, and wire."""

    signature: str
    revision: str
    reason_category: str
    route: str
    reason_codes: tuple[str, ...]
    obligation_bindings: tuple[Mapping[str, object], ...]
    check_bindings: tuple[Mapping[str, object], ...]
    executed_checks: tuple[Mapping[str, object], ...]
    certified_acceptance: bool
    uncertified_submission_admissible: bool
    actions: tuple[Mapping[str, object], ...]
    advertised_actions: tuple[str, ...]
    dispatch_admissible_actions: tuple[str, ...]
    tool_schemas: tuple[Mapping[str, object], ...]
    advisory_degraded: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "completion-route-decision/1",
            "signature": self.signature,
            "revision": self.revision,
            "reason_category": self.reason_category,
            "route": self.route,
            "reason_codes": list(self.reason_codes),
            "obligation_bindings": [copy.deepcopy(dict(row)) for row in self.obligation_bindings],
            "check_bindings": [copy.deepcopy(dict(row)) for row in self.check_bindings],
            "executed_checks": [copy.deepcopy(dict(row)) for row in self.executed_checks],
            "certified_acceptance": self.certified_acceptance,
            "uncertified_submission_admissible": self.uncertified_submission_admissible,
            "actions": [copy.deepcopy(dict(row)) for row in self.actions],
            "advertised_actions": list(self.advertised_actions),
            "dispatch_admissible_actions": list(self.dispatch_admissible_actions),
            "tool_schemas": [copy.deepcopy(dict(row)) for row in self.tool_schemas],
            "advisory_degraded": (copy.deepcopy(dict(self.advisory_degraded))
                                  if isinstance(self.advisory_degraded, Mapping) else None),
        }


class RequestActionContractError(ValueError):
    """Raised when a final request cannot be represented by its owner decision."""

    status = "HARNESS_CONTRACT"
    boundary_recovery_eligible: bool = True


def _disposition_for(code: str) -> str:
    """Return the owner disposition for every declared refusal code."""
    normalized = str(code).strip().upper()
    if normalized in _HOST_ONLY_CODES:
        return "uncertified_submit"
    if normalized in _WITNESS_CODES:
        return "probe"
    if normalized in _COVERAGE_CODES:
        return "read"
    if normalized in _QUESTION_CODES:
        return "read"
    if normalized in _CONTROL_CODES:
        return "run_tests"
    if normalized in _WORK_CODES:
        return "work_update"
    if normalized in _CHECK_CODES:
        return "run_tests"
    if normalized in _REPAIR_CODES:
        return "edit"
    if normalized in _RETRY_DONE_CODES:
        return "done"
    return "uncertified_submit"


# Public audit surface: no refusal may fall through to the legacy done-only
# narrowing without a named owner disposition.
REFUSAL_DISPOSITIONS = {
    code: _disposition_for(code) for code in REFUSAL_REASON_CODES
}
_VIOLATION_STATUSES = frozenset({
    "COUNTEREXAMPLE", "ERROR", "FAIL", "FAILED", "PRODUCT_RED", "RED",
    "REGRESSION", "UNSATISFIED", "VIOLATED",
})
_UNCERTAINTY_REASONS = frozenset({
    "ADVISORY_ONLY", "CHECK_INCONCLUSIVE", "CLAUSE_COVERAGE", "COVERAGE_OPEN", "INCONCLUSIVE",
    "INTERNAL_COVERAGE_OPEN", "INTERNAL_METADATA_OPEN", "METADATA_OPEN",
    "MISSING_CHECKS", "NO_VERIFIER", "REQUIRED_EVIDENCE_GAP", "UNRESOLVED",
    "UNKNOWN", "UNBOUND_OBSERVABLE",
})
_DEMONSTRATED_FAILURE_CODES = frozenset({
    "BOUND_COUNTEREXAMPLE", "CONTRACT_AUDIT", "COUNTEREXAMPLE", "INWORLD_OBLIGATION",
    "NONZERO_EXIT", "OPEN_BLOCKER", "OPEN_COUNTEREXAMPLE", "ORACLE_TESTS_RED",
    "PRODUCT_VIOLATION", "PUBLIC_RED_DONE", "READINESS_COUNTEREXAMPLE_OPEN",
    "SHIP_TREE_SMOKE", "STUB_GATE", "TESTS_NOT_GREEN", "UNIFIED_VIOLATED",
    "UNRESOLVED_DEFECT",
})
_STATE_FIELDS = frozenset({
    "schema", "terminal_choice_offered", "terminal_choice_revision",
    "final_submit_reserved", "final_submit_used", "witness_used",
    "final_submit_response_count", "restatement_pending",
    "final_submit_request_pending", "done_only_offer_count",
    "progress_keys",
})
_LEGACY_STATE_FIELDS = _STATE_FIELDS - {
    "final_submit_response_count", "restatement_pending", "final_submit_request_pending",
    "done_only_offer_count", "progress_keys",
}


def mode(environ: MutableMapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    value = getattr(runtime, "stateframe_completion_owner_mode", None)
    return mode({MODE_ENV: str(value)}) == "on" if value is not None else mode() == "on"


def no_done_only_collapse_enabled(runtime: object | None = None) -> bool:
    value = getattr(runtime, "stateframe_no_done_only_collapse_mode", None)
    if value is None:
        value = os.environ.get(NO_DONE_ONLY_COLLAPSE_ENV, "off")
    normalized = str(value or "off").strip().lower()
    if normalized not in {"off", "on"}:
        raise ValueError(f"{NO_DONE_ONLY_COLLAPSE_ENV} must be off or on")
    if normalized != "on":
        return False
    try:
        from kernel_preedit_coverage_runtime import _zero_edits
        return bool(_zero_edits(runtime))
    except Exception:  # noqa: BLE001 - unknown mutation evidence preserves legacy collapse
        return False


def next_action_owner_enabled(runtime: object | None = None) -> bool:
    value = getattr(runtime, "stateframe_next_action_owner_mode", None) if runtime is not None else None
    if value is None:
        value = os.environ.get(NEXT_ACTION_OWNER_ENV, "off")
    value = str(value).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{NEXT_ACTION_OWNER_ENV} must be off or on")
    return value == "on"


def initial_state(*, witness_used: int = 0) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "terminal_choice_offered": False,
        "terminal_choice_revision": None,
        "final_submit_reserved": False,
        "final_submit_used": False,
        "witness_used": max(0, int(witness_used)),
        "final_submit_response_count": 0,
        "restatement_pending": False,
        "final_submit_request_pending": False,
        # This is deliberately independent from response charging.  A model
        # action between two terminal projections must not re-arm done-only.
        "done_only_offer_count": 0,
        # Bounded fingerprints of admitted product/information progress.  They
        # make a repeated read or repeated green receipt unable to re-arm the
        # terminal reserve indefinitely.
        "progress_keys": [],
    }


def _state(runtime: object) -> dict[str, object]:
    current = getattr(runtime, "_completion_owner_state", None)
    if isinstance(current, Mapping) and frozenset(current) in {_STATE_FIELDS, _LEGACY_STATE_FIELDS}:
        migrated = initial_state(witness_used=int(current.get("witness_used", 0) or 0))
        migrated.update(dict(current))
        migrated["final_submit_response_count"] = max(
            0,
            int(current.get(
                "final_submit_response_count",
                1 if bool(current.get("final_submit_used", False)) else 0,
            ) or 0),
        )
        migrated["restatement_pending"] = bool(current.get("restatement_pending", False))
        migrated["final_submit_request_pending"] = bool(
            current.get("final_submit_request_pending", False))
        migrated["done_only_offer_count"] = max(
            0, int(current.get("done_only_offer_count", 0) or 0))
        progress_keys = current.get("progress_keys", ())
        migrated["progress_keys"] = [
            str(item) for item in progress_keys
            if str(item)
        ] if isinstance(progress_keys, (list, tuple)) else []
        current = migrated
    else:
        current = initial_state(
            witness_used=int(getattr(runtime, "_stateframe_witness_gate_requests", 0) or 0))
    setattr(runtime, "_completion_owner_state", current)
    return current  # type: ignore[return-value]


def snapshot(runtime: object) -> dict[str, object] | None:
    if not enabled(runtime):
        return None
    return copy.deepcopy(_state(runtime))


def restore(runtime: object, value: object) -> None:
    if value is None:
        return
    if (not isinstance(value, Mapping)
            or frozenset(value) not in {_STATE_FIELDS, _LEGACY_STATE_FIELDS}
            or value.get("schema") != SCHEMA
            or type(value.get("terminal_choice_offered")) is not bool
            or type(value.get("final_submit_reserved")) is not bool
            or type(value.get("final_submit_used")) is not bool
            or not isinstance(value.get("terminal_choice_revision"), (str, type(None)))
            or type(value.get("witness_used")) is not int
            or value["witness_used"] < 0
            or ("final_submit_response_count" in value
                and (type(value["final_submit_response_count"]) is not int
                     or value["final_submit_response_count"] < 0))
            or ("restatement_pending" in value
                and type(value["restatement_pending"]) is not bool)
            or ("final_submit_request_pending" in value
                and type(value["final_submit_request_pending"]) is not bool)):
        raise ValueError("invalid completion-owner checkpoint")
    if ("done_only_offer_count" in value
            and (type(value["done_only_offer_count"]) is not int
                 or value["done_only_offer_count"] < 0)):
        raise ValueError("invalid completion-owner checkpoint")
    if ("progress_keys" in value
            and (not isinstance(value["progress_keys"], (list, tuple))
                 or any(not isinstance(item, str) for item in value["progress_keys"]))):
        raise ValueError("invalid completion-owner checkpoint")
    restored = initial_state(witness_used=int(value["witness_used"]))
    restored.update(copy.deepcopy(dict(value)))
    restored["final_submit_response_count"] = max(
        0,
        int(value.get(
            "final_submit_response_count",
            1 if bool(value.get("final_submit_used", False)) else 0,
        ) or 0),
    )
    restored["restatement_pending"] = bool(value.get("restatement_pending", False))
    restored["final_submit_request_pending"] = bool(
        value.get("final_submit_request_pending", False))
    restored["done_only_offer_count"] = max(
        0, int(value.get("done_only_offer_count", 0) or 0))
    restored["progress_keys"] = [
        str(item) for item in value.get("progress_keys", ())
        if str(item)
    ]
    setattr(runtime, "_completion_owner_state", restored)
    setattr(runtime, "_stateframe_witness_gate_requests", int(value["witness_used"]))


def _emit(runtime: object, kind: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_completion_owner", event_schema=SCHEMA,
             dispatch=False, **fields)


def record_provider_request_not_dispatched(
        runtime: object, *, request_id: object, reason_code: object) -> None:
    """Record why a prepared provider request stopped before transport dispatch."""
    if (not isinstance(request_id, str) or not request_id
            or request_id != request_id.strip() or len(request_id) > 128
            or not isinstance(reason_code, str)
            or reason_code not in PROVIDER_REQUEST_NOT_DISPATCHED_REASON_CODES):
        raise ValueError("invalid provider request dispatch disposition")
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if not callable(emit):
        raise RuntimeError("provider request disposition journal is unavailable")
    emit(
        "provider_request_not_dispatched",
        producer="kernel_completion_owner",
        event_schema=PROVIDER_REQUEST_DISPOSITION_SCHEMA,
        request_id=request_id,
        provider_request=True,
        disposition=PROVIDER_REQUEST_NOT_DISPATCHED,
        reason_code=reason_code,
        dispatch=False,
    )


def _common_sense_decision(runtime: object, tools: object = ()) -> object | None:
    """Ask the optional action governor at the shared owner boundary."""
    try:
        from kernel_common_sense_governor import decide

        surface = tools
        repair_names = {"edit", "str_replace", "read"}
        has_repair_surface = isinstance(surface, Sequence) and any(
            isinstance(tool, Mapping)
            and isinstance(tool.get("function"), Mapping)
            and str(tool["function"].get("name", "")) in repair_names
            for tool in surface
        )
        if not has_repair_surface:
            full_tools = getattr(runtime, "_completion_owner_full_tools", ())
            if isinstance(full_tools, Sequence) and full_tools:
                surface = full_tools
        return decide(runtime, surface)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        emit = getattr(getattr(runtime, "journal", None), "emit", None)
        if callable(emit):
            emit(
                "common_sense_governor_error",
                producer="kernel_completion_owner",
                event_schema=SCHEMA,
                error_type=type(error).__name__,
                message=f"{type(error).__name__}: {(str(error).splitlines() or ['<no message>'])[0][:240]}",
                dispatch=False,
            )
        return None


def _common_sense_actions(decision: object) -> tuple[dict[str, object], ...]:
    raw = getattr(decision, "actions", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    result: list[dict[str, object]] = []
    for action in raw:
        if isinstance(action, Mapping):
            value = dict(action)
        else:
            tool = str(getattr(action, "tool", ""))
            if not tool:
                continue
            arguments = getattr(action, "arguments", {})
            value = {
                "tool": tool,
                "reason": str(getattr(action, "reason", "")),
                "clears": str(getattr(action, "clears", "the open gate")),
                "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
            }
        if value.get("tool") and not any(item.get("tool") == value["tool"] for item in result):
            result.append(value)
    return tuple(result)


def _common_sense_enabled(runtime: object) -> bool:
    try:
        from kernel_common_sense_governor import enabled as governor_enabled
        return governor_enabled(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _premature_frame(runtime: object, tools: object = ()) -> Mapping[str, object] | None:
    try:
        from kernel_premature_rule import build_frame
        value = build_frame(runtime, tools)
        return value if isinstance(value, Mapping) else None
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _premature_actions(frame: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    menu = frame.get("menu", ())
    if not isinstance(menu, Sequence) or isinstance(menu, (str, bytes, bytearray)):
        return ()
    return tuple({
        "tool": str(tool),
        "arguments": {},
        "reason": "expose the unclosed public requirement without treating green as closure",
        "clears": "a clause-level closing receipt or the bounded terminal submission",
    } for tool in menu if str(tool))


def _advisory_degraded_fact(runtime: object) -> dict[str, object] | None:
    value = getattr(runtime, "_completion_owner_advisory_degraded", None)
    if not isinstance(value, Mapping):
        return None
    error_type = value.get("error_type")
    count = value.get("count")
    first_event = value.get("first_event")
    if (not isinstance(error_type, str) or type(count) is not int or count < 1
            or not isinstance(first_event, Mapping)):
        return None
    return {
        "type": "HOST_ADVISORY_DEGRADED",
        "error_type": error_type[:80],
        "count": min(count, _ADVISORY_FAILURE_CAP),
        "first_event": {
            "kind": str(first_event.get("kind", _ADVISORY_ERROR_KIND))[:96],
            "event_id": (str(first_event["event_id"])[:120]
                         if first_event.get("event_id") not in (None, "") else None),
        },
    }


def _record_advisory_failure(runtime: object, error: Exception) -> None:
    previous = _advisory_degraded_fact(runtime)
    count = min(_ADVISORY_FAILURE_CAP,
                (int(previous["count"]) if previous is not None else 0) + 1)
    error_type = (str(previous["error_type"]) if previous is not None
                  else type(error).__name__[:80])
    first_event = (copy.deepcopy(dict(previous["first_event"])) if previous is not None
                   else {"kind": _ADVISORY_ERROR_KIND, "event_id": None})
    traceback_frames = []
    current = error.__traceback__
    while current is not None and len(traceback_frames) < 32:
        frame = current.tb_frame
        frame_label = (f"{Path(frame.f_code.co_filename).name}:{current.tb_lineno}:"
                       f"{frame.f_code.co_name}")
        traceback_frames.append(frame_label[:240])
        current = current.tb_next
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        try:
            event = emit(
                _ADVISORY_ERROR_KIND,
                producer="kernel_completion_owner",
                event_schema=SCHEMA,
                error_type=type(error).__name__[:80],
                traceback_frames=traceback_frames,
                traceback_truncated=current is not None,
                failure_count=count,
                first_failure=count == 1,
                dispatch=False,
            )
            if count == 1 and isinstance(event, Mapping):
                event_id = event.get("event_id", event.get("sequence"))
                if event_id not in (None, ""):
                    first_event["event_id"] = str(event_id)[:120]
        except Exception:  # noqa: BLE001 - diagnostic failures stay advisory
            pass
    setattr(runtime, "_completion_owner_advisory_degraded", {
        "type": "HOST_ADVISORY_DEGRADED",
        "error_type": error_type,
        "count": count,
        "first_event": first_event,
    })


def advisory_degraded_notice(runtime: object) -> str | None:
    """Render one finite fact without serializing the runtime or its evidence graph."""
    fact = _advisory_degraded_fact(runtime)
    if fact is None:
        return None
    first = fact["first_event"]
    assert isinstance(first, Mapping)
    first_ref = first.get("event_id") or first.get("kind")
    return (
        "[HOST_ADVISORY_DEGRADED] The advisory gate raised "
        f"{fact['error_type']} {fact['count']} time(s); first event: {first_ref}. "
        "Advisory failure does not establish a product failure or remove an otherwise admitted repair route."
    )


def _advisory_gate_evaluation(runtime: object) -> Mapping[str, object] | None:
    """Read advisory-gate classification without creating a second terminal owner."""
    runtime_id = id(runtime)
    active = _ACTIVE_ADVISORY_EVALUATIONS.get()
    if runtime_id in active:
        # Runtime evidence callbacks can return through the owner.  The outer
        # advisory projection remains responsible for the complete result.
        return None
    token = _ACTIVE_ADVISORY_EVALUATIONS.set(active | {runtime_id})
    try:
        if getattr(runtime, "stateframe_advisory_gates_mode", "off") != "on":
            return None
        from kernel_advisory_gates import enabled as advisory_gates_enabled, evaluate
        if not advisory_gates_enabled(runtime):
            return None
        result = evaluate(runtime)
        return result if isinstance(result, Mapping) else None
    except Exception as error:  # noqa: BLE001 - advisory diagnostics cannot own completion
        _record_advisory_failure(runtime, error)
        return None
    finally:
        _ACTIVE_ADVISORY_EVALUATIONS.reset(token)


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class _ImmutableDict(dict[str, object]):
    """JSON-compatible deep-freeze boundary for the finalized contract."""

    def __init__(self, values: Mapping[str, object]) -> None:
        dict.__init__(self, values)

    @staticmethod
    def _reject(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RequestActionContractError("finalized request contract is immutable")

    __setitem__ = _reject
    __delitem__ = _reject
    clear = _reject
    pop = _reject
    popitem = _reject
    setdefault = _reject
    update = _reject
    __ior__ = _reject

    def __deepcopy__(self, memo: dict[int, object]) -> "_ImmutableDict":
        del memo
        return self


class _ImmutableList(list[object]):
    """List-compatible immutable container for JSON array fields."""

    def __init__(self, values: Sequence[object]) -> None:
        list.__init__(self, values)

    @staticmethod
    def _reject(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RequestActionContractError("finalized request contract is immutable")

    __setitem__ = _reject
    __delitem__ = _reject
    append = _reject
    clear = _reject
    extend = _reject
    insert = _reject
    pop = _reject
    remove = _reject
    reverse = _reject
    sort = _reject
    __iadd__ = _reject
    __imul__ = _reject

    def __deepcopy__(self, memo: dict[int, object]) -> "_ImmutableList":
        del memo
        return self


def _freeze_contract(value: object) -> object:
    if isinstance(value, Mapping):
        return _ImmutableDict({
            str(key): _freeze_contract(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return _ImmutableList(_freeze_contract(item) for item in value)
    return copy.deepcopy(value)


def _thaw_contract(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_contract(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_contract(item) for item in value]
    return copy.deepcopy(value)


def _contract_decision_digest(contract: Mapping[str, object]) -> str:
    return _digest({
        key: value for key, value in contract.items() if key != "decision_digest"
    })


def _verify_contract_integrity(contract: Mapping[str, object]) -> None:
    expected = contract.get("decision_digest")
    if not isinstance(expected, str) or _contract_decision_digest(contract) != expected:
        raise RequestActionContractError("finalized request contract was mutated")


def _owner_block_records(runtime: object) -> dict[str, dict[str, object]]:
    raw = getattr(runtime, "_completion_owner_owner_blocks", None)
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, dict[str, object]] = {}
    for block_type, block in raw.items():
        if str(block_type) not in _OWNER_BLOCK_TYPES or not isinstance(block, Mapping):
            continue
        if (block.get("schema") != OWNER_BLOCK_SCHEMA
                or block.get("type") != block_type
                or not str(block.get("id", ""))
                or not str(block.get("digest", ""))):
            continue
        result[str(block_type)] = {
            key: copy.deepcopy(value)
            for key, value in block.items()
            if key != "_value"
        }
    return result


def _owner_block(runtime: object, block_type: str) -> Mapping[str, object] | None:
    return _owner_block_records(runtime).get(block_type)


def _owner_block_value(runtime: object, block_type: str) -> object:
    raw = getattr(runtime, "_completion_owner_owner_blocks", None)
    block = raw.get(block_type) if isinstance(raw, Mapping) else None
    return copy.deepcopy(block.get("_value")) if isinstance(block, Mapping) else None


def record_owner_block(
        runtime: object,
        block_type: str,
        value: object,
        *,
        block_id: str | None = None,
        request_id: str | None = None,
) -> dict[str, object]:
    """Register host-authored control data without inspecting request text."""
    normalized_type = str(block_type).strip()
    if normalized_type not in _OWNER_BLOCK_TYPES:
        raise ValueError(f"unsupported owner block type: {normalized_type}")
    canonical_value = (
        _contract_tools(value) if normalized_type == "menu" else copy.deepcopy(value)
    )
    digest_value = [canonical_value] if normalized_type == "evidence_pin" else canonical_value
    digest = _digest(digest_value)
    record: dict[str, object] = {
        "schema": OWNER_BLOCK_SCHEMA,
        "type": normalized_type,
        "id": str(block_id or f"{normalized_type}:{digest}"),
        "digest": digest,
        "_value": canonical_value,
    }
    if request_id is not None:
        record["request_id"] = str(request_id)
    current = getattr(runtime, "_completion_owner_owner_blocks", None)
    blocks = dict(current) if isinstance(current, Mapping) else {}
    blocks[normalized_type] = record
    setattr(runtime, "_completion_owner_owner_blocks", blocks)
    return {
        key: copy.deepcopy(item) for key, item in record.items() if key != "_value"
    }


def _ensure_terminal_owner_block(
        runtime: object,
        selected: Sequence[Mapping[str, object]],
) -> None:
    """Keep a typed terminal record paired with every reserved done surface."""
    names = tuple(_tool_name(tool) for tool in selected)
    state = _state(runtime)
    if names != ("done",) or not (
            state["final_submit_reserved"] and state["final_submit_request_pending"]):
        return
    if _owner_block(runtime, "terminal_line") is not None:
        return
    fallback = (
        NO_HOST_CHECKS_TERMINAL_LINE if _no_host_checks(runtime) else
        "Final step: call done now to submit this revision (owner-admitted submit route)."
    )
    record_owner_block(runtime, "terminal_line", fallback, block_id="terminal:pending")


def _contract_terminal_line(runtime: object) -> str | None:
    value = _owner_block_value(runtime, "terminal_line")
    return value.strip() if isinstance(value, str) and value.strip() else None


def strip_owner_message_annotations(messages: object) -> list[dict[str, object]]:
    """Remove internal host-block markers before a request reaches transport."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return []
    return [
        {
            key: copy.deepcopy(value)
            for key, value in message.items()
            if key != OWNER_MESSAGE_BLOCK_KEY
        }
        for message in messages
        if isinstance(message, Mapping)
    ]


def _append_owner_message(messages: list[dict[str, object]], content: str,
                          *, marker: str) -> None:
    if not content or any(message.get("content") == content for message in messages):
        return
    messages.append({
        "role": "user",
        "content": content,
        OWNER_MESSAGE_BLOCK_KEY: marker,
    })


_PROGRESS_RESULTS = frozenset({"APPLIED", "PASS", "PASSED", "GREEN", "SUCCESS", "OK", "CLOSED"})


def _progress_key(record: object) -> tuple[str, str] | None:
    """Return a stable key only for settled, non-duplicate progress."""
    if not isinstance(record, Mapping):
        return None
    tool = str(record.get("tool", "")).strip().lower()
    result = str(record.get("result", record.get("result_class", ""))).strip().upper()

    # A successful native probe closes through the typed witness receipt, while
    # its action outcome is reported as COMPLETED rather than PASS.  Count that
    # receipt-bearing action as productive progress so the terminal stall cap
    # cannot erase the newly available submit route.
    if tool == "probe" and result == "COMPLETED" and record.get("receipt_ids"):
        return "clause_evidence", _digest({
            "tool": tool,
            "target": record.get("target"),
            "result": result,
            "receipt_ids": record.get("receipt_ids"),
        })

    # StepLedger attaches host-observed source evidence to the closed record.
    # That evidence is itself a new clause fact; do not require a synthetic
    # execution result when the read/recall path has already proved it.
    explicit_clause_evidence = record.get("new_clause_evidence") or record.get("clause_evidence")
    read_evidence = record.get("read_evidence")
    if explicit_clause_evidence or (
            tool in {"read", "recall_evidence"} and isinstance(read_evidence, Mapping)):
        return "clause_evidence", _digest({
            "tool": tool,
            "target": record.get("target"),
            "evidence": explicit_clause_evidence or read_evidence,
        })

    if result not in _PROGRESS_RESULTS:
        return None

    changes = record.get("changes")
    product_changes = []
    if isinstance(changes, Sequence) and not isinstance(changes, (str, bytes, bytearray)):
        product_changes = [
            {
                "path": item.get("path"),
                "before": item.get("before"),
                "after": item.get("after"),
                "role": item.get("role"),
            }
            for item in changes
            if isinstance(item, Mapping) and item.get("role", "product") == "product"
        ]
    if tool in {"edit", "str_replace", "file_create", "replace_module", "write", "replace"}:
        explicit_byte_change = any(
            record.get(name) is True for name in ("byte_changed", "file_changed", "artifact_change")
        )
        if product_changes or explicit_byte_change:
            return "product", _digest({"tool": tool, "changes": product_changes,
                                       "target": record.get("target")})
        return None
    if tool in {"run_tests", "test", "pytest"}:
        return "green_check", _digest({
            "tool": tool,
            "target": record.get("target"),
            "binding": record.get("after_binding"),
            "receipt_ids": record.get("receipt_ids"),
        })

    if tool == "probe":
        return "clause_evidence", _digest({
            "tool": tool,
            "target": record.get("target"),
            "result": result,
            "receipt_ids": record.get("receipt_ids"),
        })

    return None


def note_productive_progress(runtime: object, record: Mapping[str, object] | None = None) -> bool:
    """Reset terminal-cycle exhaustion after one new admitted fact or product change.

    This seam is called only after action closure.  Menu projection, NO_OP, and
    contract refusal therefore cannot re-arm the terminal owner.
    """
    if not enabled(runtime):
        return False
    key = _progress_key(record)
    if key is None:
        return False
    if isinstance(record, Mapping) and str(record.get("tool", "")).strip().lower() in _STALL_RULE_EDIT_TOOLS:
        setattr(runtime, "_completion_owner_repair_edit_count",
                int(getattr(runtime, "_completion_owner_repair_edit_count", 0) or 0) + 1)
        # Admitted-only counter (note_model_action also counts parsed/refused
        # edits): the open-requirement repair route keys on real progress.
        setattr(runtime, "_completion_owner_admitted_edit_count",
                int(getattr(runtime, "_completion_owner_admitted_edit_count", 0) or 0) + 1)
    state = _state(runtime)
    fingerprint = f"{key[0]}:{key[1]}"
    progress_keys = [str(item) for item in state.get("progress_keys", []) if str(item)]
    if fingerprint in progress_keys:
        return False
    progress_keys.append(fingerprint)
    state["progress_keys"] = progress_keys[-64:]
    state["done_only_offer_count"] = 0
    state["terminal_choice_offered"] = False
    state["terminal_choice_revision"] = None
    state["final_submit_reserved"] = False
    state["final_submit_used"] = False
    state["final_submit_response_count"] = 0
    state["restatement_pending"] = False
    state["final_submit_request_pending"] = False
    setattr(runtime, "_completion_owner_green_frame_counted", False)
    setattr(runtime, "_completion_owner_actionable", False)
    _emit(
        runtime,
        "completion_owner_progress_reset",
        progress_kind=key[0],
        target=str(record.get("target", "")) if isinstance(record, Mapping) else "",
        reason="admitted progress since the last terminal offer",
    )
    return True


def _revision(runtime: object) -> object:
    binding = getattr(runtime, "binding", None)
    try:
        return binding() if callable(binding) else binding
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _green_current(runtime: object) -> bool:
    # public_test_verdict owns per-call audit effects, including on a body hit.
    return _green_current_uncached(runtime)


def _green_current_uncached(runtime: object) -> bool:
    verdict_fn = getattr(runtime, "public_test_verdict", None)
    binding_fn = getattr(runtime, "binding", None)
    if not callable(verdict_fn) or not callable(binding_fn):
        return False
    try:
        verdict = verdict_fn()
        current = binding_fn()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    from kernel_host_cache import memo_runtime_body
    return memo_runtime_body(
        runtime, "green_current_body", lambda: _green_current_body(verdict, current),
        inputs=lambda: (verdict, current), binding=current,
    )


def _green_current_body(verdict: object, current: object) -> bool:
    if getattr(verdict, "state", None) != "SATISFIED" or getattr(verdict, "current_binding", None) != current:
        return False
    receipts = tuple(getattr(verdict, "trusted_receipts", ()) or ())
    return bool(receipts) and all(
        getattr(receipt, "binding", None) == current
        and getattr(receipt, "after_binding", None) == current
        and getattr(receipt, "exit_code", None) == 0
        and getattr(receipt, "binding_stable", None) is True
        for receipt in receipts
    )


def _product_parent_unchanged(runtime: object, *, reason: str) -> bool:
    """True until the current product identity is proven different from parent."""
    from kernel_green_lock import (
        current_public_verdict_is_green,
        note_unchanged_parent_green_suppressed,
        product_revision_change_from_parent,
    )
    changed, _revision, _comparison = product_revision_change_from_parent(runtime)
    from kernel_green_lock import PROVEN_PARENT_UNCHANGED as _PROVEN_UNCHANGED
    changed = changed or _comparison not in _PROVEN_UNCHANGED  # unknown parent identity is not "unchanged"
    if changed:
        return False
    # suppress ONLY the 936 condition, a public GREEN observed on the UNCHANGED parent. A no-green,
    # no-edit read loop keeps the stall rule's terminal path (J926: 226 turns left in a read loop must still end);
    # test_stallwire::test_terminal_offer_cannot_reexpand_to_full_menu pins that.
    if _green_current(runtime) or current_public_verdict_is_green(runtime):
        note_unchanged_parent_green_suppressed(runtime, reason=reason)
        return True
    return False


def postgreen_done_offer(runtime: object) -> tuple[bool, str]:
    """Require done on a provider request only after current green on a changed product."""
    if runtime is None or _model_turns_remaining(runtime) in (None, 0):
        return False, "UNKNOWN"
    try:
        from kernel_green_lock import product_revision_change_from_parent, current_public_verdict_is_green
        changed, revision, _comparison = product_revision_change_from_parent(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False, "UNKNOWN"
    # An unknown parent is not proof of an applied product edit. This keeps the
    # pre-edit terminal path closed while the live r2e unchanged-parent case
    # remains explicitly suppressed.
    if not changed or not (_green_current(runtime) or current_public_verdict_is_green(runtime)):
        return False, str(revision or "UNKNOWN")[:180]
    return True, str(revision or _route_revision(runtime))[:180]


def ensure_postgreen_done_on_wire(
        runtime: object,
        tools: object,
        *,
        request_id: str | None,
        capture: object | None = None,
) -> list[dict[str, object]]:
    """Restore a proposed typed done route; serialized capture certifies delivery."""
    selected = [copy.deepcopy(dict(tool)) for tool in _sequence(tools)
                if isinstance(tool, Mapping)]
    required, revision = postgreen_done_offer(runtime)
    if not required:
        return selected

    full_surface = _available_tools(runtime, selected)
    done_schema = next((copy.deepcopy(tool) for tool in full_surface
                        if _tool_name(tool) == "done"), None)
    if done_schema is None:
        raise RequestActionContractError(
            "post-green changed candidate has no admitted done schema at final wire")
    names = tuple(_tool_name(tool) for tool in selected)
    restored_by = getattr(capture, "postgreen_done_stripped_by", None)
    if "done" not in names:
        if not isinstance(restored_by, str) or not restored_by:
            restored_by = ("completion_owner_stall_rule"
                           if isinstance(getattr(runtime, "_stall_open_work_episode", None), Mapping)
                           else "final_wire_tool_projection")
        selected.append(done_schema)
        if capture is not None:
            setattr(capture, "postgreen_done_stripped_by", restored_by)

    request_key = str(request_id or "")
    seen = getattr(runtime, "_done_offered_after_green_request_ids", None)
    if not isinstance(seen, set):
        seen = set()
        setattr(runtime, "_done_offered_after_green_request_ids", seen)
    if request_key not in seen:
        if getattr(capture, "postgreen_done_stripped_by", None):
            _emit(runtime, "done_restoration_prepared",
                  request_id=request_id,
                  stripped_by=str(getattr(capture, "postgreen_done_stripped_by")))
        route = completion_route_decision(
            runtime, getattr(runtime, "_last_completion_decision", None),
            tools=getattr(runtime, "_completion_owner_full_tools", ()) or selected,
        )
        done_route = ("CERTIFIED_ACCEPTANCE" if route.certified_acceptance
                      else "UNCERTIFIED_SUBMIT")
        _emit(runtime, "done_offer_after_green_prepared",
              request_id=request_id, revision=revision, route=done_route)
        seen.add(request_key)
    return selected


def _clear_terminal_offer_state(runtime: object) -> None:
    state = _state(runtime)
    state["terminal_choice_offered"] = False
    state["final_submit_reserved"] = False
    state["final_submit_request_pending"] = False
    state["final_submit_response_count"] = 0
    state["done_only_offer_count"] = 0
    state["restatement_pending"] = False
    state["terminal_choice_revision"] = None


def _no_host_checks(runtime: object) -> bool:
    selectors_fn = getattr(runtime, "required_public_selectors", None)
    if callable(selectors_fn):
        try:
            return not tuple(selectors_fn())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False
    owner = getattr(runtime, "owner", None)
    if owner is None or not hasattr(owner, "f2p_ids"):
        return False
    return not tuple(str(item) for item in getattr(owner, "f2p_ids", ()) if str(item).strip())


def stall_rule_enabled(runtime: object | None = None) -> bool:
    """Return whether the request qualification stall-open-work rule is explicitly enabled."""
    configured = getattr(runtime, "stateframe_fl_stall_rule_mode", None)
    if configured is not None:
        return str(configured).strip().lower() in _STALL_RULE_TRUTHY
    return str(os.environ.get(STALL_RULE_ENV, "off") or "off").strip().lower() in _STALL_RULE_TRUTHY


_REUSED_GREEN_RULES = frozenset({"post_green_identical_rerun"})


def _greenlock_predispatch(
        runtime: object,
        tool: str,
        arguments: Mapping[str, object] | None = None,
) -> tuple[bool, str | None, bool]:
    """Ask live greenlock before publishing an owner action.

    The optional runtime hook is a narrow host/test seam.  Production runtimes
    use the canonical menu authority directly.  ``authority_active`` keeps the
    owner usable in reduced standalone test doubles with no greenlock object.
    """
    args = dict(arguments or {})
    capability_allowed, capability_reason = _current_tool_admission(runtime, tool)
    if not capability_allowed:
        return False, capability_reason or "tool_dispatch_unavailable", True
    hook = getattr(runtime, "greenlock_predispatch_admission", None)
    if callable(hook):
        try:
            result = hook(tool, args)
            if isinstance(result, tuple):
                allowed = bool(result[0])
                rule = str(result[1]) if len(result) > 1 and result[1] else None
            else:
                allowed, rule = bool(result), None
            return allowed, rule, True
        except Exception as error:
            record_failure(
                runtime,
                "route.greenlock_predispatch_hook",
                error,
                authority=True,
                disposition="deny",
                denied_actions=(tool, "done", "submit"),
            )
            return False, "greenlock_admission_error", True

    lock = getattr(runtime, "green_lock", None)
    if lock is None:
        return True, None, False

    try:
        from kernel_greenlock_menu_authority import applies, decide, refresh

        if not applies(lock):
            return True, None, False
        refresh(lock)
        # A released one-request restriction leaves its prior wire set in the
        # upstream snapshot. Ignore it only until this next owner-built menu is
        # finalized; the exact new menu is frozen before dispatch.
        previous_upstream = getattr(runtime, "_greenlock_menu_authority_upstream", None)
        try:
            from kernel_rejection_template_restrict import owner_projection_release_pending
            restriction_release_pending = owner_projection_release_pending(runtime)
        except (ImportError, AttributeError, TypeError, ValueError):
            restriction_release_pending = False
        if restriction_release_pending:
            setattr(runtime, "_greenlock_menu_authority_upstream", None)
        try:
            decision = decide(lock, tool, args, for_menu=True)
        finally:
            if restriction_release_pending:
                setattr(runtime, "_greenlock_menu_authority_upstream", previous_upstream)
        return bool(decision.allowed), str(decision.rule) if decision.rule else None, True
    except Exception as error:
        # Never publish an action that the live authority was not able to check.
        record_failure(
            runtime,
            "route.greenlock_predispatch_authority",
            error,
            authority=True,
            disposition="deny",
            denied_actions=(tool, "done", "submit"),
        )
        return False, "greenlock_admission_error", True


def greenlock_predispatch_admission(
        runtime: object,
        tool: str,
        arguments: Mapping[str, object] | None = None,
) -> bool:
    """Return whether ``tool`` is admissible at the current live boundary."""
    from kernel_capability_floor import enabled as capability_floor_enabled

    if tool == "done" and capability_floor_enabled(runtime):
        # Greenlock controls certification readiness. Submit stays executable so
        # terminal_result can return an honest uncertified artifact with its open
        # obligations; the tool runner gate remains mandatory.
        admitted, _reason = _current_tool_admission(runtime, tool)
        return admitted
    allowed, _rule, _active = _greenlock_predispatch(runtime, tool, arguments)
    return allowed


def _revision_digest(runtime: object) -> str:
    return _digest(_revision(runtime))


def _record_reused_green_verification(
        runtime: object,
        selectors: Sequence[str],
) -> bool:
    """Record exact current-green evidence when greenlock refused an identical rerun."""
    normalized = tuple(str(selector) for selector in selectors if str(selector))
    if not normalized or not _green_current(runtime):
        return False

    # A pending post-green edit must be cleared by the canonical receipt
    # helper.  A stale binding must never become completion evidence.
    try:
        from kernel_post_green_menu import note_reused_verification, pending_verification

        pending = bool(pending_verification(runtime))
    except (ImportError, AttributeError, TypeError, ValueError):
        note_reused_verification = None
        pending = False

    verdict = runtime.public_test_verdict()
    if pending:
        if note_reused_verification is None or not note_reused_verification(runtime, verdict, normalized):
            return False

    receipts = tuple(getattr(verdict, "trusted_receipts", ()) or ())
    covered = set()
    receipt_ids = []
    for receipt in receipts:
        if getattr(receipt, "binding", None) != verdict.current_binding:
            continue
        if getattr(receipt, "after_binding", None) != verdict.current_binding:
            continue
        if getattr(receipt, "exit_code", None) != 0 or getattr(receipt, "binding_stable", None) is not True:
            continue
        covered.update(str(item) for item in (getattr(receipt, "selectors", ()) or ()))
        receipt_id = getattr(receipt, "receipt_id", getattr(receipt, "id", None))
        if receipt_id:
            receipt_ids.append(str(receipt_id))
    if not set(normalized).issubset(covered):
        return False

    evidence = {
        "revision": _revision_digest(runtime),
        "binding": verdict.current_binding,
        "selectors": normalized,
        "receipt_ids": tuple(receipt_ids),
        "reason": "greenlock identical rerun refusal is the same current-green fact",
    }
    setattr(runtime, "_completion_owner_reused_verification", evidence)
    _emit(
        runtime,
        "completion_owner_reused_verification",
        selectors=list(normalized),
        receipt_ids=receipt_ids,
        revision=evidence["revision"],
    )
    return True


def completion_owner_reused_verification(runtime: object) -> bool:
    """Validate the owner’s named reused-verification evidence for this revision."""
    evidence = getattr(runtime, "_completion_owner_reused_verification", None)
    if not isinstance(evidence, Mapping) or evidence.get("revision") != _revision_digest(runtime):
        return False
    if not _green_current(runtime):
        return False
    try:
        from kernel_post_green_menu import host_verified_done, pending_verification

        if pending_verification(runtime) and not host_verified_done(runtime):
            return False
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    return True


def host_public_check_arguments(
        runtime: object,
        candidates: object = None,
        *,
        allow_candidate_fallback: bool = False,
) -> dict[str, object] | None:
    """Return dispatcher-validated public check arguments for an owner action.

    The completion owner is the last authority before a menu is rendered.  It
    therefore cannot pass through decision metadata that mixes a real selector
    with prose such as ``target exists: ...``.  Each selector must be one exact
    target accepted by the public dispatcher, point at an existing ``tests/``
    file, and survive validation again after the target list is assembled.

    ``{}`` means no host selector source was available, preserving reduced test
    doubles and legacy control actions.  ``None`` means a host selector source
    existed but contained no dispatcher-valid public selector and the action
    must not be rendered.
    """
    provider = getattr(runtime, "required_public_selectors", None)
    owner = getattr(runtime, "owner", None)
    host_source = None
    if callable(provider):
        try:
            host_source = provider()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            host_source = ()
    elif owner is not None and hasattr(owner, "f2p_ids"):
        host_source = getattr(owner, "f2p_ids", ())
    elif candidates is None:
        return {}
    host_values = host_source
    if isinstance(host_values, str):
        host_values = (host_values,)
    if not isinstance(host_values, Sequence) or isinstance(host_values, (bytes, bytearray)):
        host_values = () if host_source is not None else None
    source = host_source if host_source is not None else candidates
    if allow_candidate_fallback and host_source is not None and not host_values:
        # A pending post-green edit still needs a real check even when the
        # active acceptance record has no selector. Its prior successful
        # host-run selector is a safe candidate after normal path validation.
        source = candidates
    if isinstance(source, str):
        source = (source,)
    if not isinstance(source, Sequence) or isinstance(source, (bytes, bytearray)):
        return None
    values = tuple(str(item).strip() for item in source if str(item).strip())
    if not values:
        return None if host_source is not None else {}
    root_value = getattr(owner, "root", None)
    if root_value is None:
        return None
    try:
        root = Path(root_value).resolve()
        from kernel_public_evidence import requested_check_targets
        from kernel_live_projection import is_harness_path
        valid: list[str] = []
        for selector in values:
            parsed = requested_check_targets(root, selector)
            if parsed != [selector]:
                continue
            base = selector.split("::", 1)[0]
            resolved = (root / base).resolve()
            if (not resolved.is_relative_to(root)
                    or not resolved.is_file()
                    or not is_harness_path(Path(base))):
                continue
            if selector not in valid:
                valid.append(selector)
        if not valid:
            return None
        target = " ".join(valid)
        if requested_check_targets(root, target) != valid:
            return None
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return {"target": target, "selectors": valid}


def _closed(runtime: object) -> bool:
    ledger = getattr(runtime, "step_ledger", None)
    if ledger is not None and getattr(ledger, "pending", None) is not None:
        return False
    continuation = getattr(runtime, "continuation", None)
    state = getattr(getattr(continuation, "engine", None), "state", None)
    effects = getattr(state, "effects", {}) if state is not None else {}
    if isinstance(effects, Mapping):
        if any(getattr(effect, "status", None) in {"pending", "unknown"}
               for effect in effects.values()):
            return False
    return True


def _field(value: object, name: str, default: object = "") -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _assessments(runtime: object) -> tuple[object, ...]:
    values = getattr(runtime, "host_closure_assessments", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return ()
    return tuple(values)


def _assessment_is_context_only(value: object) -> bool:
    """Keep headings and non-normative ticket context out of executable gates."""
    if not isinstance(value, Mapping):
        fields = (
            "context_only", "executable", "role", "source_role", "classification",
            "disposition", "requirement_kind", "kind", "clause", "requirement",
            "description", "title", "source_text", "scope",
        )
        projected = {name: getattr(value, name) for name in fields
                     if hasattr(value, name)}
        if not projected:
            return False
        value = projected
    if value.get("context_only") is True or value.get("executable") is False:
        return True
    markers = (
        value.get("role"), value.get("source_role"), value.get("classification"),
        value.get("disposition"), value.get("requirement_kind"), value.get("kind"),
    )
    if any(str(marker or "").strip().lower() in {
            "context", "context_only", "heading", "ticket_heading", "non_normative",
    } for marker in markers):
        return True
    text = "\n".join(
        str(value.get(name, "") or "")
        for name in ("clause", "requirement", "description", "title", "source_text", "scope")
    ).strip()
    if not text:
        return False
    if text.lstrip().startswith("#"):
        return True
    try:
        from kernel_requirements_ledger import _is_normative
        return not _is_normative(text)
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def known_violation(runtime: object) -> bool:
    advisory_gates = _advisory_gate_evaluation(runtime)
    if advisory_gates is not None and advisory_gates.get("hard_blockers"):
        return True
    for assessment in _assessments(runtime):
        if _assessment_is_context_only(assessment):
            continue
        status = str(_field(assessment, "verdict", _field(assessment, "status", ""))).upper()
        reason = str(_field(assessment, "reason", "")).upper()
        if status in _VIOLATION_STATUSES or reason in _VIOLATION_STATUSES:
            return True
    current_revision = _revision(runtime)
    if isinstance(current_revision, Mapping):
        current_revision = current_revision.get(
            "product_revision", current_revision.get("artifact", current_revision.get("revision"))
        )
    for record in getattr(runtime, "behavior_witness_records", ()) or ():
        if not (isinstance(record, Mapping)
                and str(record.get("result", "")).upper() in {"FAIL", "COUNTEREXAMPLE"}):
            continue
        recorded_digest = record.get("revision_digest")
        if recorded_digest is not None:
            if current_revision is None or recorded_digest == _digest(current_revision):
                return True
            continue
        recorded_revision = record.get("revision")
        if (recorded_revision is None or current_revision is None
                or recorded_revision == current_revision
                or _digest(recorded_revision) == _digest(current_revision)):
            return True
    codes = {str(code).upper() for code in getattr(runtime, "_v39_uncertified_source_codes", ()) or ()}
    hard_codes = {"OPEN_COUNTEREXAMPLE", "PRODUCT_VIOLATION"}
    # GOVFIX owns concrete requirement alignment and its once-per-episode
    # refusal/release.  Advisory-gates must not turn that evidence gap into a
    # second permanent completion veto.
    if advisory_gates is None:
        hard_codes.add("REQUIREMENT_ALIGNMENT_OPEN")
    if codes & hard_codes:
        return True
    for name in ("open_counterexamples", "counterexamples", "live_counterexamples"):
        value = getattr(runtime, name, ())
        if value:
            return True
    return False


def open_ids(runtime: object) -> tuple[str, ...]:
    from kernel_host_cache import memo_runtime_fact, optimized_turn_facts_enabled
    if not optimized_turn_facts_enabled():
        return _open_ids_uncached(runtime)
    return memo_runtime_fact(
        runtime, "TURN_FACTS", lambda: _open_ids_uncached(runtime),
        fact_key="open_ids", reads_tools=False,
    )


def _open_ids_uncached(runtime: object) -> tuple[str, ...]:
    values: list[str] = [
        str(item) for item in getattr(runtime, "_v39_uncertified_submit_ids", ()) or ()
        if str(item).strip()
    ]
    for assessment in _assessments(runtime):
        if _assessment_is_context_only(assessment):
            continue
        obligation_id = str(_field(assessment, "obligation_id", _field(assessment, "id", "")))
        reason = str(_field(assessment, "reason", "")).upper()
        status = str(_field(assessment, "verdict", _field(assessment, "status", ""))).upper()
        if obligation_id and reason in _UNCERTAINTY_REASONS and status not in _VIOLATION_STATUSES:
            values.append(obligation_id)
    advisory_gates = _advisory_gate_evaluation(runtime)
    if advisory_gates is not None:
        values.extend(str(item) for item in advisory_gates.get("advisories", ()) if str(item).strip())
    try:
        from kernel_clause_discriminator import open_requirement_ids
        values.extend(open_requirement_ids(runtime))
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        # A failed optional adapter cannot authorize completion, but it also
        # must not crash the owner while legacy lanes are running.
        pass
    return tuple(dict.fromkeys(values))


def _green_open_work(runtime: object) -> bool:
    """Keep repair routes available while current green evidence has open work."""
    return _green_current(runtime) and bool(open_ids(runtime))


def _scrub_owner_off_menu_tool_names(
        runtime: object, messages: object, offered_names: Sequence[str],
) -> None:
    """Remove registered off-menu tool names from host-authored request notices."""
    if not isinstance(messages, list):
        return
    offered = {str(name).strip() for name in offered_names if str(name).strip()}
    full_surface = getattr(runtime, "_completion_owner_full_tools", ())
    registered = {
        _tool_name(tool) for tool in _sequence(full_surface)
        if isinstance(tool, Mapping) and _tool_name(tool)
    }
    terminal_line = _contract_terminal_line(runtime)
    advisory_markers = (
        "completion-route:", "advisory-degraded:", "no-progress-cap:",
        "stall-open-work", "declared-repair", "witness-evidence",
        "common-sense-control", "premature-rule", "next-action",
        "host-only", "common-sense-advisory",
    )
    for message in messages:
        if not isinstance(message, MutableMapping):
            continue
        marker = str(message.get(OWNER_MESSAGE_BLOCK_KEY, ""))
        content = message.get("content")
        if (not marker or marker.startswith("terminal:")
                or not marker.startswith(advisory_markers)
                or not isinstance(content, str) or content == terminal_line
                or _evidence_marker_present(content)):
            continue
        value = content
        for name in sorted(registered - offered, key=len, reverse=True):
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_]){re.escape(name)}"
                rf"(?:\s*\([^()\n]*\))?(?![A-Za-z0-9_])"
            )
            value = pattern.sub("", value)
        value = re.sub(r"\s+([,;:])", r"\1", value)
        value = re.sub(r"([,;:])\s*([,;:])", r"\2", value)
        value = re.sub(r"\s{2,}", " ", value).strip()
        message["content"] = value


def _reqwitness_open_ids(runtime: object) -> tuple[str, ...]:
    try:
        from kernel_clause_discriminator import open_requirement_ids
        return tuple(open_requirement_ids(runtime))
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        try:
            setattr(runtime, "_clause_discriminator_read_error", True)
        except (AttributeError, TypeError):
            pass
        return ()


def _authored_host_bound(receipt: Mapping[str, object]) -> bool:
    """W3: an authored receipt counts only with the dispatcher's host binding.

    The binding is created by ``kernel_clause_discriminator`` from host-owned
    obligation clauses (never from a proposal) and must either name this
    receipt's id or carry the matching host clause identity for an alias.
    """
    from kernel_authored_check import (
        BINDING_SCHEMA, binding_receipt_ids_are_host_consistent,
    )
    binding = receipt.get("host_obligation_binding")
    if not isinstance(binding, Mapping) or binding.get("schema") != BINDING_SCHEMA:
        return False
    ident = str(binding.get("obligation_id") or "").strip()
    receipt_ids = {
        str(receipt.get("requirement_id") or "").strip(),
        str(receipt.get("clause_id") or "").strip(),
    }
    alias = str(binding.get("host_requirement_id") or "").strip()
    return bool(ident and binding.get("clause_sha256")
                and binding_receipt_ids_are_host_consistent(receipt)
                and (ident in receipt_ids or alias in receipt_ids))


def _current_witness_receipt_closes_clause(runtime: object) -> bool:
    """Recognize a passing clause receipt at the runtime's current revision."""
    if getattr(runtime, "_clause_discriminator_read_error", False):
        return False
    try:
        from kernel_aha11_native_honesty import temporal_completion_authorized
        temporal_gate = temporal_completion_authorized(runtime, _revision(runtime))
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        temporal_gate = None
    if temporal_gate is not None:
        # The generic temporal bridge owns bounded partition coverage for an
        # opted-in clause; a single legacy witness receipt cannot close it.
        return temporal_gate
    if _reqwitness_open_ids(runtime):
        return False
    try:
        from kernel_clause_discriminator import clause_rows
        clauses = clause_rows(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    # An empty obligation list is not evidence of completion.  There must be a
    # current clause set, and every required clause must be receipt-covered.
    if not clauses or not all(row.get("covered") is True for row in clauses):
        return False
    # A turn counter is bookkeeping, not an executed observation.  It can only
    # support a receipt lookup below; it never closes a current clause alone.
    current_revision = _revision(runtime)
    if isinstance(current_revision, Mapping):
        current_revision = current_revision.get(
            "product_revision", current_revision.get("artifact", current_revision.get("revision"))
        )
    # A bare last-pass revision has no clause/check provenance and therefore
    # cannot close NEW_BEHAVIOR; only the typed receipts below may do so.
    for source_name in (
            "_clause_discriminator_receipts", "witness_evidence_receipts",
            "behavior_witness_records", "_row_role_authored_receipts"):
        for receipt in _sequence(getattr(runtime, source_name, ())):
            if not isinstance(receipt, Mapping):
                continue
            provenance = receipt.get("provenance")
            authored = isinstance(provenance, Mapping) and str(
                provenance.get("kind", "")).upper() == "AUTHORED_CHECK"
            if authored and (receipt.get("certifies_clause") is not True
                             or str(receipt.get("result", "")).upper() != "BEHAVIOR_FIXED"
                             or not _authored_host_bound(receipt)):
                continue
            if not authored and receipt.get("certifies_clause") is not True:
                continue
            result = str(receipt.get("outcome", receipt.get("result", receipt.get("status", "")))).upper()
            passed = result in {"PASS", "PASSED", "GREEN", "BEHAVIOR_FIXED"} \
                or receipt.get("exit_code") == 0
            reached = receipt.get("reached_observation") is True
            integrity = receipt.get("integrity") is True or (
                isinstance(receipt.get("test_integrity"), Mapping)
                and str(receipt["test_integrity"].get("status", "")).upper() in {"PASS", "CLEAN", "OK"}
            )
            check_identity = receipt.get("check_identity") or receipt.get("check_hash") or receipt.get("check_id") or receipt.get("selector")
            clause = receipt.get("clause_id") or receipt.get("requirement_id") or receipt.get("obligation_id") or receipt.get("clause_ids")
            if not (passed and reached and integrity and check_identity and clause):
                continue
            revision = receipt.get("revision")
            if (revision is not None and current_revision is not None
                    and (revision == current_revision
                         or _digest(revision) == _digest(current_revision))):
                return True
    for receipt in _journal_records(runtime):
        if not isinstance(receipt, Mapping) or receipt.get("kind") != "clause_discriminator_receipt":
            continue
        provenance = receipt.get("provenance")
        authored = isinstance(provenance, Mapping) and str(
            provenance.get("kind", "")).upper() == "AUTHORED_CHECK"
        if authored and (receipt.get("certifies_clause") is not True
                         or str(receipt.get("result", "")).upper() != "BEHAVIOR_FIXED"
                         or not _authored_host_bound(receipt)):
            continue
        if not authored and receipt.get("certifies_clause") is not True:
            continue
        if str(receipt.get("result", receipt.get("outcome", ""))).upper() not in {
                "PASS", "PASSED", "GREEN"}:
            continue
        if receipt.get("reached_observation") is not True:
            continue
        if not (receipt.get("integrity") is True or (
                isinstance(receipt.get("test_integrity"), Mapping)
                and str(receipt["test_integrity"].get("status", "")).upper()
                in {"PASS", "CLEAN", "OK"})):
            continue
        if not (receipt.get("check_identity") or receipt.get("check_hash")
                or receipt.get("check_id") or receipt.get("selector")):
            continue
        if not (receipt.get("clause_id") or receipt.get("requirement_id")
                or receipt.get("obligation_id") or receipt.get("clause_ids")):
            continue
        pin = receipt.get("evidence_pin")
        revision = pin.get("revision") if isinstance(pin, Mapping) else None
        if (revision is not None and current_revision is not None
                and (revision == current_revision
                     or _digest(revision) == _digest(current_revision))):
            return True
    return False


def recompute_authored_completion(runtime: object) -> dict[str, object]:
    """Refresh existing completion-owner inputs after a settled authored receipt."""
    current = _current_witness_receipt_closes_clause(runtime)
    projection = {"current_clause_closed": current,
                  "open_ids": list(open_ids(runtime))}
    setattr(runtime, "_completion_owner_authored_receipt_projection", copy.deepcopy(projection))
    _emit(runtime, "completion_owner_authored_receipt_recomputed", **projection)
    return projection


def _candidate(runtime: object) -> tuple[bool, str, tuple[str, ...]]:
    refresh = getattr(runtime, "refresh_green_revalidation_before_terminal", None)
    if callable(refresh):
        try:
            if refresh() is False:
                return False, _digest(_revision(runtime)), ()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False, _digest(_revision(runtime)), ()
    current = _revision(runtime)
    vector_fn = getattr(runtime, "pass_gates_vector", None)
    if callable(vector_fn):
        try:
            vector = vector_fn()
            if getattr(vector, "ready", False) is not True:
                return False, _digest(current), tuple(
                    str(getattr(item, "blocker_id", ""))
                    for item in getattr(vector, "blockers", ())
                    if str(getattr(item, "blocker_id", ""))
                )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False, _digest(current), ("PASS_GATES_VECTOR_UNAVAILABLE",)
    advisory_gates = _advisory_gate_evaluation(runtime)
    if advisory_gates is not None and advisory_gates.get("hard_blockers"):
        return False, _digest(current), ()
    if not _closed(runtime) or known_violation(runtime):
        return False, _digest(current), ()
    if not _no_host_checks(runtime) and not _green_current(runtime):
        return False, _digest(current), ()
    # Q3 (E9): green public tests are not a witness for a NEW_BEHAVIOR row.
    from kernel_row_role import certification_blocked
    if certification_blocked(runtime):
        _emit(runtime, "completion_owner_new_behavior_unwitnessed")
        return False, _digest(current), ()
    return True, _digest(current), open_ids(runtime)


def _green_terminal_ready(runtime: object) -> bool:
    """Return true only for current trusted green evidence with no product red."""
    if not _no_host_checks(runtime) and not _green_current(runtime):
        return False
    candidate, _, _ = _candidate(runtime)
    return candidate


def _done_tool(runtime: object, tools: object = ()) -> dict[str, object] | None:
    for tool in _available_tools(runtime, tools):
        if _tool_name(tool) == "done":
            return tool
    return None


def _pending_deferred_source_paths(runtime: object) -> tuple[str, ...]:
    raw_paths = getattr(runtime, "_request_deferred_source_paths", ())
    if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes, bytearray)):
        raw_paths = ()
    paths = tuple(dict.fromkeys(
        str(path).replace("\\", "/").strip() for path in raw_paths if str(path).strip()
    ))
    if not paths:
        plan = getattr(runtime, "_request_source_closure", None)
        raw_paths = plan.get("deferred_paths", ()) if isinstance(plan, Mapping) else ()
        if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes, bytearray)):
            return ()
        paths = tuple(dict.fromkeys(
            str(path).replace("\\", "/").strip() for path in raw_paths if str(path).strip()
        ))
    try:
        from kernel_read_reuse_visibility import proven_deferred_source_paths
        proven = proven_deferred_source_paths(runtime, paths)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        proven = set()
    return tuple(path for path in paths if path not in proven)


def _deferred_source_read_tool(runtime: object, tools: object) -> dict[str, object] | None:
    if not _pending_deferred_source_paths(runtime):
        return None
    read_tool = next((tool for tool in _available_tools(runtime, tools)
                      if _tool_name(tool) == "read"), None)
    # A bounded read is the retrieval route that makes a verified deferral
    # usable. Greenlock governs work actions; it must not remove this
    # read-only continuation from the reserved submit surface.
    if read_tool is None or not _current_tool_admission(runtime, "read")[0]:
        return None
    return copy.deepcopy(read_tool)


def _deferred_source_submit_tools(
        runtime: object, tools: object,
) -> list[dict[str, object]] | None:
    """Project the pending submit route to done plus its required read route."""
    if not _pending_deferred_source_paths(runtime):
        return None
    available = {_tool_name(tool): tool for tool in _available_tools(runtime, tools)}
    done = available.get("done")
    read = available.get("read")
    if (not isinstance(done, Mapping) or not isinstance(read, Mapping)
            or not _current_tool_admission(runtime, "read")[0]):
        return None
    return [copy.deepcopy(dict(done)), copy.deepcopy(dict(read))]


def _terminal_tools_with_deferred_read(
        runtime: object, tools: object, done_tool: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    done = done_tool or _done_tool(runtime, tools)
    if not isinstance(done, Mapping):
        return []
    selected = [copy.deepcopy(dict(done))]
    read_tool = _deferred_source_read_tool(runtime, tools)
    if read_tool is not None:
        selected.append(read_tool)
    return selected


def _terminal_presentation_tools(
        runtime: object, candidates: object, done_tool: Mapping[str, object],
) -> list[dict[str, object]]:
    preserved = _preserve_zero_edit_terminal_menu(runtime, candidates)
    if preserved is not None:
        return preserved
    return _terminal_tools_with_deferred_read(runtime, candidates, done_tool)


def _reserved_terminal_menu_names(runtime: object, names: Sequence[str]) -> bool:
    normalized = tuple(str(name) for name in names)
    if normalized == ("done",):
        return True
    if normalized == ("done", "read") and _pending_deferred_source_paths(runtime):
        return True
    return (no_done_only_collapse_enabled(runtime)
            and len(normalized) > 1 and normalized[0] == "done"
            and len(set(normalized)) == len(normalized)
            and set(normalized).issubset({"done", "edit", "str_replace", "read"}))


def _mark_owner_done_selection(runtime: object, reason: str) -> None:
    """Record that the owner itself selected a terminal ``done`` menu.

    The stamp is reset at the start of every ``terminal_tool_menu`` call, so it
    describes only the request currently being prepared.
    """
    setattr(runtime, "_completion_owner_done_selection", {"reason": str(reason)})


def owner_done_selection(runtime: object) -> dict[str, object] | None:
    """Return the owner's reserved terminal selection for this request, if any.

    Later presentation projections (stall menus, budget/diet passes) must treat
    this selection as authoritative and must not drop ``done`` from it.
    """
    stamp = getattr(runtime, "_completion_owner_done_selection", None)
    if not isinstance(stamp, Mapping):
        return None
    names = tuple(getattr(runtime, "completion_menu", ()) or ())
    if not _reserved_terminal_menu_names(runtime, names):
        return None
    if not getattr(runtime, "_question_done_available", False):
        return None
    dispatch = getattr(runtime, "_completion_owner_dispatch_tools", None)
    if not isinstance(dispatch, list) or [_tool_name(tool) for tool in dispatch] != list(names):
        return None
    return {"reason": str(stamp.get("reason", "")), "tools": copy.deepcopy(dispatch)}


def _set_done_only_menu(runtime: object, tools: object, *, reason: str) -> list[dict[str, object]]:
    from kernel_capability_floor import enabled as capability_floor_enabled

    floor_active = capability_floor_enabled(runtime)
    reqwitness_ids = _reqwitness_open_ids(runtime)
    if reqwitness_ids:
        limitation = _witness_limitation(runtime, tools)
        if limitation:
            done = _done_tool(runtime, tools)
            if done is not None:
                selected = _terminal_tools_with_deferred_read(runtime, tools, done)
                names = tuple(_tool_name(tool) for tool in selected)
                runtime.completion_menu = names
                runtime.offered_tools = names
                runtime._question_done_available = True
                runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
                _mark_owner_done_selection(runtime, "witness_limitation")
                if isinstance(tools, list):
                    tools[:] = copy.deepcopy(selected)
                _emit(runtime, "completion_owner_witness_route_exhausted",
                      offered=list(names), open_ids=list(reqwitness_ids),
                      reason=limitation)
                return selected
        available = _available_tools(runtime, tools)
        work = [tool for tool in available if _tool_name(tool) != "done"]
        runtime.completion_menu = tuple(_tool_name(tool) for tool in work)
        runtime._question_done_available = False
        runtime._completion_owner_dispatch_tools = copy.deepcopy(work)
        if isinstance(tools, list):
            tools[:] = copy.deepcopy(work)
        _emit(runtime, "completion_owner_clause_coverage_menu",
              offered=list(runtime.completion_menu), open_ids=list(reqwitness_ids),
              reason="unexercised public clause")
        return work
    state = _state(runtime)
    tracking_request = bool(
        state["terminal_choice_offered"] and state["final_submit_reserved"])
    selected_names = [
        _tool_name(tool) for tool in tools
    ] if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)) else []
    # The dispatch seam can project the same narrowed menu twice.  In legacy
    # mode, count provider-facing projections for its done-only cap; with the
    # capability floor, the episode-level hard budgets own stopping instead.
    if (tracking_request
            and (floor_active or state["done_only_offer_count"] < MAX_RESERVED_RESPONSES)
            and (getattr(runtime, "_completion_owner_green_frame_counted", False)
                 or _reserved_terminal_menu_names(runtime, selected_names))):
        runtime._question_done_available = True
        done = _done_tool(runtime, tools)
        selected = _terminal_tools_with_deferred_read(runtime, tools, done) if done is not None else [
            dict(tool) for tool in tools if isinstance(tool, Mapping)
        ]
        names = tuple(_tool_name(tool) for tool in selected)
        runtime.completion_menu = names
        runtime.offered_tools = names
        runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
        _mark_owner_done_selection(runtime, reason)
        if isinstance(tools, list):
            tools[:] = copy.deepcopy(selected)
        return selected
    if (tracking_request and not floor_active
            and state["done_only_offer_count"] >= MAX_RESERVED_RESPONSES):
        from kernel_terminal_policy import (
            enabled as hard_budget_only_enabled,
            queue_advisory,
        )
        if hard_budget_only_enabled(runtime):
            queue_advisory(runtime, "owner_reserved_submit_cap")
        else:
            runtime.system_outcome = FINAL_SUBMIT_EXHAUSTED
        runtime.completion_menu = ()
        runtime._question_done_available = False
        runtime._completion_owner_dispatch_tools = []
        if isinstance(tools, list):
            tools[:] = []
        _emit(runtime, "completion_owner_exhausted",
              reason="done_only_offer_cap",
              revision=state["terminal_choice_revision"],
              done_only_offer_count=state["done_only_offer_count"])
        return []
    done = _done_tool(runtime, tools)
    if done is None:
        return []
    if tracking_request:
        state["done_only_offer_count"] += 1
        setattr(runtime, "_completion_owner_green_frame_counted", True)
    selected = _terminal_tools_with_deferred_read(runtime, tools, done)
    names = tuple(_tool_name(tool) for tool in selected)
    runtime.completion_menu = names
    runtime.offered_tools = names
    runtime._question_done_available = True
    runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
    _mark_owner_done_selection(runtime, reason)
    if isinstance(tools, list):
        tools[:] = copy.deepcopy(selected)
    _emit(runtime, "completion_owner_green_frame_rendered",
          offered=list(names), host_line=GREEN_TERMINAL_LINE, reason=reason,
          done_only_offer_count=state["done_only_offer_count"])
    return selected


def _tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    return str(function.get("name", "")) if isinstance(function, Mapping) else ""


def remember_full_tools(runtime: object, tools: object) -> None:
    """Keep a recoverable owner surface across terminal menu projections."""
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return
    incoming = [copy.deepcopy(dict(tool)) for tool in tools if isinstance(tool, Mapping)]
    if not incoming:
        return
    current = getattr(runtime, "_completion_owner_full_tools", None)
    current_names = {_tool_name(tool) for tool in current} if isinstance(current, Sequence) else set()
    incoming_names = {_tool_name(tool) for tool in incoming}
    if (not isinstance(current, list) or not current
            or (not current_names - {"done"} and incoming_names - {"done"})):
        setattr(runtime, "_completion_owner_full_tools", incoming)


def register_tools(runtime: object, tools: object) -> list[dict[str, object]]:
    """Register one provider tool surface before any completion projection.

    The completion owner is the sole contract boundary for advertised actions;
    repo-intel and every legacy tool therefore enter through this validation
    rather than being appended after the request contract is frozen.
    """
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        raise RequestActionContractError("tool surface must be a sequence")
    registered: list[dict[str, object]] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        name = _tool_name(tool)
        if not name or name in names:
            raise RequestActionContractError("provider tool surface contains duplicate/empty names")
        registered.append(copy.deepcopy(dict(tool)))
        names.add(name)
    registered = canonicalize_tool_surface(registered, runtime=runtime)
    if isinstance(tools, list):
        tools[:] = copy.deepcopy(registered)
    remember_full_tools(runtime, registered)
    _emit(runtime, "completion_owner_tools_registered",
          tool_count=len(registered), tool_names=sorted(names))
    return registered


def _typed_probe_tool(tool: object) -> bool:
    """Return true only for the behavior-witness probe contract."""
    if _tool_name(tool) != "probe" or not isinstance(tool, Mapping):
        return False
    function = tool.get("function")
    parameters = function.get("parameters") if isinstance(function, Mapping) else None
    if not isinstance(parameters, Mapping):
        return False
    properties = parameters.get("properties")
    required = parameters.get("required")
    typed_fields = {"clause_id", "entrypoint", "input", "expected", "rationale"}
    discriminator_fields = {"clause_ref", "entrypoint", "input", "expected_relation"}
    branches = parameters.get("oneOf")
    branch_required_sets = (
        set(branch.get("required", ()))
        for branch in branches
        if isinstance(branch, Mapping)
    ) if isinstance(branches, Sequence) else ()
    branch_requires_typed = any(
        typed_fields <= required_set or discriminator_fields <= required_set
        for required_set in branch_required_sets
    )
    return (isinstance(properties, Mapping)
            and isinstance(required, Sequence)
            and not isinstance(required, (str, bytes, bytearray))
            and (typed_fields <= set(properties) or discriminator_fields <= set(properties))
            and (typed_fields <= set(required)
                 or discriminator_fields <= set(required)
                 or branch_requires_typed))


def _valid_witness_entrypoint(value: object) -> bool:
    text = str(value or "").strip()
    return bool(_ENTRYPOINT_RE.fullmatch(text)) and text != "public.module.function"


def _available_tools(runtime: object, tools: object = ()) -> list[dict[str, object]]:
    values: list[object] = []
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)):
        values.extend(tools)
    full = getattr(runtime, "_completion_owner_full_tools", ())
    if isinstance(full, Sequence) and not isinstance(full, (str, bytes, bytearray)):
        values.extend(full)
    result: list[dict[str, object]] = []
    names: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping):
            continue
        name = _tool_name(value)
        if name and name not in names:
            result.append(copy.deepcopy(dict(value)))
            names.add(name)
    return result


def _cutoff_rule_menu(runtime: object, tools: Sequence[Mapping[str, object]]) -> list[dict[str, object]] | None:
    """Return RF02's admitted menu, leaving all other owners authoritative."""
    try:
        from kernel_cutoff_rule import (
            concrete_open_ids,
            enabled as cutoff_enabled,
            project_tools,
            revision_changed as cutoff_revision_changed,
        )
        if not cutoff_enabled(runtime):
            return None
        candidate, _revision, _ids = _candidate(runtime)
        ids = concrete_open_ids(runtime)
        if not candidate or not ids:
            return None
        # Existing owners get first refusal.  RF02 must never replace a more
        # specific premature, stall, or current-counterexample recovery route.
        if _premature_frame(runtime, tools) is not None:
            return None
        if stall_open_work_frame(runtime, tools) is not None:
            return None
        try:
            from kernel_green_menu_open_obligation import open_counterexamples
            if open_counterexamples(runtime):
                return None
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
        available = _available_tools(runtime, tools)
        return project_tools(
            list(tools), available,
            changed=cutoff_revision_changed(runtime),
        )
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _cutoff_rule_offer_applies(runtime: object) -> bool:
    """Check RF02 before text projection, using the already-final menu names."""
    current = getattr(runtime, "completion_menu", ())
    if not isinstance(current, Sequence) or isinstance(current, (str, bytes, bytearray)) or not current:
        return False
    current_tools = [
        {"type": "function", "function": {"name": str(name)}}
        for name in current if str(name)
    ]
    return _cutoff_rule_menu(runtime, current_tools) is not None


def _apply_cutoff_rule_menu(runtime: object, tools: list[dict[str, object]]) -> list[dict[str, object]] | None:
    from kernel_cutoff_rule import (
        concrete_open_ids,
        revision_changed as cutoff_revision_changed,
    )
    projected = _cutoff_rule_menu(runtime, tools)
    if projected is None:
        return None
    current_names = tuple(_tool_name(tool) for tool in tools)
    runtime.completion_menu = tuple(_tool_name(tool) for tool in projected)
    runtime._question_done_available = False
    runtime._completion_owner_dispatch_tools = copy.deepcopy(projected)
    tools[:] = copy.deepcopy(projected)
    if current_names == runtime.completion_menu:
        setattr(runtime, "_completion_owner_actionable", True)
        return projected
    _emit(
        runtime,
        "fl_cutoff_rule_menu",
        rule_schema="stateframe-fl-cutoff-rule/1",
        offered=list(runtime.completion_menu),
        open_ids=list(concrete_open_ids(runtime)),
        revision_changed=cutoff_revision_changed(runtime),
        reason="current trusted green with concrete open work",
    )
    return projected


def _current_tool_admission(runtime: object, tool: str) -> tuple[bool, str | None]:
    """Apply the live capability/dispatch gates before publishing an action."""
    allows = getattr(runtime, "tool_surface_allows", None)
    if callable(allows):
        try:
            if not bool(allows(tool)):
                return False, "tool_surface_refused"
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False, "tool_surface_admission_error"
    try:
        from kernel_tool_surface import dispatch_admissible
        if not dispatch_admissible(tool, runtime=runtime):
            return False, "tool_runner_unavailable"
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False, "tool_runner_admission_error"
    return True, None


def _action_dispatch_admissible(
        runtime: object, tool: str, arguments: Mapping[str, object] | None = None,
) -> bool:
    """Use dispatch's capability and Greenlock gates for every owner menu."""
    admitted, _reason = _current_tool_admission(runtime, tool)
    if not admitted:
        return False
    if tool == "read_check_receipt":
        return True
    return greenlock_predispatch_admission(runtime, tool, arguments)


def capability_floor_action_admissible(runtime: object, tool: str) -> bool:
    """Join ordinary dispatch gates with IDEALPATH's current stage admission."""
    if not _action_dispatch_admissible(runtime, tool, {}):
        return False
    owner_factory = getattr(runtime, "_idealpath_owner", None)
    if not callable(owner_factory):
        return True
    try:
        idealpath_owner = owner_factory()
        if idealpath_owner is None:
            return True
        menu_admissible = getattr(idealpath_owner, "menu_admissible_actions", None)
        if not callable(menu_admissible):
            return False
        return tool in set(menu_admissible())
    except Exception:  # noqa: BLE001 - stage admission uncertainty fails closed
        return False


def _expanded_zero_edit_submit_menu(
        runtime: object, tools: object,
) -> list[dict[str, object]] | None:
    if not no_done_only_collapse_enabled(runtime):
        return None
    available = {_tool_name(tool): tool for tool in _available_tools(runtime, tools)}
    done = available.get("done")
    if not isinstance(done, Mapping) or not _action_dispatch_admissible(runtime, "done", {}):
        return None
    selected = [copy.deepcopy(dict(done))]
    for name in ("edit", "str_replace", "read"):
        tool = available.get(name)
        if isinstance(tool, Mapping) and _action_dispatch_admissible(runtime, name, {}):
            selected.append(copy.deepcopy(dict(tool)))
    return selected if len(selected) > 1 else None


def _preserve_zero_edit_terminal_menu(
        runtime: object, tools: object,
) -> list[dict[str, object]] | None:
    state = _state(runtime)
    if not (state.get("final_submit_reserved") or state.get("final_submit_request_pending")):
        return None
    if not no_done_only_collapse_enabled(runtime):
        return None
    current = _contract_tools(tools)
    names = tuple(_tool_name(tool) for tool in current)
    if not names or names[0] != "done":
        return None
    if len(set(names)) != len(names) or not set(names).issubset(
            {"done", "edit", "str_replace", "read"}):
        return None
    expanded = _expanded_zero_edit_submit_menu(runtime, tools)
    if expanded is not None:
        return expanded
    return current if len(names) > 1 else None


def _admitted_action_projections(
        runtime: object,
        actions: Sequence[Mapping[str, object]],
        tools: object = (),
        *,
        validate_arguments: bool = True,
) -> tuple[dict[str, object], ...]:
    """Filter policy actions through dispatch gates and optional payload checks.

    Some owners produce menu-level actions before the model has selected a
    concrete target or payload.  Those actions still need name-level
    capability/greenlock admission, while their eventual model call remains
    subject to ``validate_tool_arguments`` at the dispatch boundary.
    """
    available_tools = _available_tools(runtime, tools)
    available = {_tool_name(tool) for tool in available_tools}
    result: list[dict[str, object]] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        tool = str(action.get("tool", "")).strip()
        if not tool or tool not in available:
            continue
        arguments = action.get("arguments")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        if validate_arguments and validate_tool_arguments(available_tools, tool, arguments):
            continue
        if not _action_dispatch_admissible(runtime, tool, arguments):
            continue
        result.append(copy.deepcopy(dict(action)))
    return tuple(result)


def _stall_rule_signal(runtime: object) -> object | None:
    """Read STALLWIRE's settled current-red signal without owning its reducer."""
    try:
        from kernel_tokenlab_stall_breaker import stallwire_signal
        return stallwire_signal(runtime)
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def _stall_rule_probe_refused(runtime: object) -> bool:
    try:
        from kernel_tokenlab_stall_breaker import stallwire_probe_refused
        return stallwire_probe_refused(runtime)
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _stall_rule_no_progress_edit_state(runtime: object) -> Mapping[str, object] | None:
    try:
        from kernel_tokenlab_stall_breaker import stallwire_no_progress_edit_state
        state = stallwire_no_progress_edit_state(runtime)
        return state if isinstance(state, Mapping) else None
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def _stall_rule_no_progress_edit_cap(runtime: object) -> bool:
    from kernel_capability_floor import enabled as capability_floor_enabled

    if capability_floor_enabled(runtime):
        # Keep the measurement state, but never let its heuristic stop an episode.
        setattr(runtime, "_completion_owner_no_progress_edit_cap", False)
        return False
    state = _stall_rule_no_progress_edit_state(runtime)
    return bool(
        state and state.get("capped") and not state.get("pending_delivery")
    )


def _stall_rule_cap_notice(runtime: object) -> str:
    state = _stall_rule_no_progress_edit_state(runtime) or {}
    streak = max(0, int(state.get("streak", 0) or 0))
    action = str(state.get("tool", "settled action"))
    result = str(state.get("result", "NO_PROGRESS"))
    detail = str(state.get("detail", "the host reported no product/information progress"))
    ids = ", ".join(open_ids(runtime)) or "the open host state"
    return (
        f"{TERMINAL_CHOICE_HEADER} no-progress observation: {streak} consecutive settled actions "
        f"made no product/information progress (latest {action}={result}); "
        f"host detail: {detail}. Open requirements/obligations: [{ids}]."
    )


def _notify_v46_attempt_ceiling(runtime: object, terminal: Mapping[str, object]) -> None:
    """Keep enabled V46 ceiling mechanisms wired when this owner closes first."""
    if getattr(runtime, "_completion_owner_v46_ceiling_notified", False):
        return
    setattr(runtime, "_completion_owner_v46_ceiling_notified", True)
    try:
        from kernel_v46_mechanism_wiring import at_attempt_ceiling, before_completion
        before_completion(runtime, False, {})
        at_attempt_ceiling(runtime, terminal)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        # The completion outcome remains authoritative if an optional audit
        # adapter is unavailable; enabled adapters journal their own failures.
        return


def _stall_rule_host_impossible(runtime: object, ids: Sequence[str]) -> bool:
    """Recognize only explicit host impossibility classifications."""
    wanted = {str(item) for item in ids if str(item).strip()}
    for name in ("host_classified_impossibilities", "governor_impossibilities", "host_impossibilities"):
        value = getattr(runtime, name, None)
        if isinstance(value, Mapping):
            if any(str(key) in wanted and bool(flag) for key, flag in value.items()):
                return True
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if wanted.intersection(str(item) for item in value):
                return True
        elif isinstance(value, str) and value in wanted:
            return True
    return False


def _stall_rule_key(runtime: object, ids: Sequence[str], signal: object | None) -> str:
    signal_payload = signal.as_dict() if hasattr(signal, "as_dict") else None
    return _digest({
        "revision": _revision(runtime),
        "open_ids": tuple(str(item) for item in ids),
        "signal": signal_payload,
        "no_progress_edit": _stall_rule_no_progress_edit_state(runtime),
    })


def _stall_rule_episode(runtime: object, ids: Sequence[str], signal: object | None) -> dict[str, object]:
    key = _stall_rule_key(runtime, ids, signal)
    current = getattr(runtime, "_stall_open_work_episode", None)
    if not isinstance(current, dict) or current.get("key") != key:
        attempted_keys = getattr(runtime, "_stall_open_work_attempted_keys", set())
        if not isinstance(attempted_keys, set):
            attempted_keys = set()
            setattr(runtime, "_stall_open_work_attempted_keys", attempted_keys)
        current = {
            "key": key,
            "attempted": key in attempted_keys,
            "rendered": False,
        }
        setattr(runtime, "_stall_open_work_episode", current)
    return current


def note_stall_open_work_attempt(runtime: object, action: object,
                                 *, receipt: Mapping[str, object] | None = None) -> None:
    """Release one forced repair frame after a settled edit attempt."""
    if not stall_rule_enabled(runtime) or str(action).strip().lower() not in _STALL_RULE_EDIT_TOOLS:
        return
    ids = open_ids(runtime)
    signal = _stall_rule_signal(runtime)
    key = _stall_rule_key(runtime, ids, signal)
    no_progress_state = _stall_rule_no_progress_edit_state(runtime)
    if (no_progress_state is not None
            and int(no_progress_state.get("streak", 0) or 0)
            >= int(no_progress_state.get("notice_streak", 2) or 2)):
        # The settled no-op itself is the evidence that changes the frame. Do
        # not mark this new episode as an admitted repair attempt, or the menu
        # owner would suppress the concrete no-op line immediately after it is
        # needed.
        _emit(runtime, "completion_owner_stall_rule_attempted",
              episode_key=key, tool=str(action).strip().lower(),
              result=(receipt or {}).get("result", (receipt or {}).get("result_class", "UNKNOWN")),
              no_progress=True)
        return
    attempted_keys = getattr(runtime, "_stall_open_work_attempted_keys", set())
    if not isinstance(attempted_keys, set):
        attempted_keys = set()
        setattr(runtime, "_stall_open_work_attempted_keys", attempted_keys)
    attempted_keys.add(key)
    episode = _stall_rule_episode(runtime, ids, signal)
    episode["attempted"] = True
    episode["attempted_tool"] = str(action).strip().lower()
    _emit(runtime, "completion_owner_stall_rule_attempted",
          episode_key=key, tool=str(action).strip().lower(),
          result=(receipt or {}).get("result", (receipt or {}).get("result_class", "UNKNOWN")))


def _stall_rule_excerpt(runtime: object, signal: object | None) -> str:
    values: list[object] = []
    if signal is not None:
        failure = getattr(signal, "failure_excerpt", "")
        if failure:
            values.append(failure)
    for name in (
            "last_host_assertion", "last_host_observation", "last_observation",
            "latest_observation", "last_completed_observation", "last_test_result",
            "latest_test_result", "last_completed_receipt"):
        value = getattr(runtime, name, None)
        if value:
            if isinstance(value, Mapping):
                for field in ("claim", "observation", "message", "output", "detail", "failure_tail"):
                    if value.get(field):
                        values.append(value[field])
                        break
            else:
                values.append(value)
    for source_name in ("observations", "test_receipts", "test_results", "recent_test_feedback"):
        values_value = getattr(runtime, source_name, ())
        if isinstance(values_value, Sequence) and not isinstance(values_value, (str, bytes, bytearray)):
            for value in reversed(values_value):
                if isinstance(value, Mapping):
                    for field in ("claim", "observation", "message", "output", "detail", "failure_tail"):
                        if value.get(field):
                            values.append(value[field])
                            break
                elif value:
                    values.append(value)
    owner = getattr(runtime, "owner", None)
    receipts = getattr(owner, "_receipts", ())
    if isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes, bytearray)):
        for value in reversed(receipts):
            if isinstance(value, Mapping):
                for field in ("claim", "observation", "message", "output", "detail", "failure_tail"):
                    if value.get(field):
                        values.append(value[field])
                        break
    for value in values:
        text = " ".join(str(value).split())
        if text:
            return text if len(text) <= 360 else text[:359] + "…"
    return "no host assertion/observation excerpt recorded"


def _surface_text(value: object) -> str:
    if isinstance(value, Mapping):
        path = value.get("path", value.get("file", value.get("target", "")))
        symbol = value.get("symbol", value.get("function", ""))
        if path and symbol:
            return f"{path}::{symbol}"
        if path:
            return str(path)
        return ""
    return str(value).strip() if value else ""


def _stall_rule_surfaces(runtime: object, ids: Sequence[str], signal: object | None) -> tuple[str, ...]:
    surfaces: list[str] = []

    def add(value: object) -> None:
        text = _surface_text(value)
        if text and text not in surfaces:
            surfaces.append(text)

    if signal is not None:
        add({"path": getattr(signal, "implicated_path", ""),
             "symbol": getattr(signal, "implicated_symbol", "")})
    for name in ("host_implicated_surfaces", "implicated_surfaces", "stall_rule_surfaces"):
        value = getattr(runtime, name, ())
        if isinstance(value, Mapping):
            value = (value,)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                add(item)
        else:
            add(value)
    for assessment in _sequence(getattr(runtime, "host_closure_assessments", ())):
        if not isinstance(assessment, Mapping):
            continue
        identifier = str(assessment.get("obligation_id", assessment.get("id", "")))
        if ids and identifier not in ids:
            continue
        for field in ("surface", "path", "file", "target", "source", "entrypoint"):
            if assessment.get(field):
                add(assessment[field])
        clause = str(assessment.get("clause", assessment.get("requirement", "")) or "")
        for match in _STALL_RULE_PATH_RE.findall(clause):
            if not match.startswith("tests/"):
                add(match)
    path = ""
    target_path = getattr(runtime, "_target_relative", None)
    if callable(target_path):
        try:
            path = target_path()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            path = ""
        symbol = getattr(getattr(runtime, "owner", None), "fn", "")
        add({"path": path, "symbol": symbol})
    if surfaces:
        return tuple(surfaces)
    if not path:
        return ("target not localized",)
    return ("not recorded",)


def stall_open_work_frame(runtime: object, tools: object = ()) -> dict[str, object] | None:
    """Build the one live STALLRULE frame from host state and settled receipts."""
    if not stall_rule_enabled(runtime):
        return None
    # With no public host selector, the witness rail is explicitly unavailable;
    # the truthful fallback is the disclosed submit route, not repair churn.
    if _no_host_checks(runtime):
        return None
    if _stall_rule_no_progress_edit_cap(runtime) and not _no_host_checks(runtime):
        # No-op/rejected edit churn is a bounded host state, not permission to
        # keep publishing the same repair frame until the response budget dies.
        setattr(runtime, "_completion_owner_no_progress_edit_cap", True)
        _begin_uncertified_submit(runtime)
        return None
    available = _available_tools(runtime, tools)
    names = {_tool_name(tool) for tool in available}
    ids = open_ids(runtime)
    signal = _stall_rule_signal(runtime)
    no_progress_state = _stall_rule_no_progress_edit_state(runtime)
    from kernel_capability_floor import enabled as capability_floor_enabled
    no_progress_stall = bool(
        not capability_floor_enabled(runtime)
        and no_progress_state
        and int(no_progress_state.get("streak", 0) or 0)
        >= int(no_progress_state.get("notice_streak", 2) or 2)
        and not bool(no_progress_state.get("capped"))
    )
    red_stall = bool(signal is not None and getattr(runtime, "_stallwire_owner", None)
                     and getattr(runtime._stallwire_owner, "red_active", False))
    green_open = bool(_green_current(runtime) and ids)
    host_ready_no_open = bool(no_progress_stall and not ids and _green_terminal_ready(runtime)
                              and "done" in names)
    if ((not ids and not host_ready_no_open)
            or ("run_tests" not in names and not host_ready_no_open)
            or not (green_open or red_stall or no_progress_stall)):
        return None
    episode = _stall_rule_episode(runtime, ids, signal if red_stall else None)
    if bool(episode.get("attempted")) or _stall_rule_host_impossible(runtime, ids):
        return None
    excerpt = _stall_rule_excerpt(runtime, signal if red_stall else None)
    surfaces = _stall_rule_surfaces(runtime, ids, signal if red_stall else None)
    receipt_handle = _check_receipt_handle(runtime)
    diagnostic_available = bool(receipt_handle and "read_check_receipt" in names)
    menu = _STALL_RULE_TOOLS + (("read_check_receipt",) if diagnostic_available else ())
    lines: tuple[str, ...] = (
        f'Last host assertion excerpt (verbatim): "{excerpt}"',
        f"Host implicated surface: {', '.join(surfaces)}",
    )
    if no_progress_stall:
        # Keep productive alternatives visible; dropping edit must never make
        # premature done look like the only useful action.  Host-ready green
        # may retain done, while current RED never does.
        if host_ready_no_open:
            menu = ("done",)
        else:
            menu = ("str_replace", "read")
            if diagnostic_available:
                menu += ("read_check_receipt",)
        lines = (str(no_progress_state.get("notice", "")),)
    frame = {
        "episode_key": str(episode["key"]),
        "kind": "no-op-edit-stall" if no_progress_stall else (
            "red-stall" if red_stall else "green-open-work-stall"),
        "open_ids": tuple(ids),
        "menu": menu,
        "excerpt": excerpt,
        "surfaces": surfaces,
        "lines": lines,
    }
    runtime._stall_open_work_frame = frame
    return frame


def stall_open_work_actions(runtime: object, tools: object = ()) -> tuple[dict[str, object], ...]:
    frame = stall_open_work_frame(runtime, tools)
    if frame is None:
        return ()
    actions: list[dict[str, object]] = []
    for name in frame["menu"]:
        arguments: dict[str, object] = {}
        if name == "read_check_receipt":
            handle = _check_receipt_handle(runtime)
            if not handle:
                continue
            arguments = {"handle": handle, "stream": "assertion_context"}
        actions.append({
            "tool": name,
            "arguments": arguments,
            "reason": ("inspect the stored assertion without another run_tests decision"
                       if name == "read_check_receipt"
                       else "address the host-identified open requirement"),
            "clears": ("the readable check-evidence gap"
                       if name == "read_check_receipt"
                       else "the open requirement episode after a settled repair attempt"),
        })
    return tuple(actions)


def _stall_rule_fallback_tool(name: str) -> dict[str, object]:
    properties: dict[str, object] = {"path": {"type": "string"}}
    if name == "edit":
        properties["function_source"] = {"type": "string"}
    elif name == "str_replace":
        properties = {
            **properties,
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
        }
    return {"type": "function", "function": {
        "name": name,
        "parameters": {"type": "object", "properties": properties},
    }}


def stall_open_work_notice(runtime: object) -> str | None:
    frame = getattr(runtime, "_stall_open_work_frame", None)
    if not isinstance(frame, Mapping):
        frame = stall_open_work_frame(runtime)
    if not isinstance(frame, Mapping):
        return None
    return "\n".join(str(line) for line in frame["lines"])


def project_stall_open_work(runtime: object, tools: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Project STALLRULE's exact menu; this is the only renderer for the rule."""
    owner_selection = owner_done_selection(runtime)
    if owner_selection is not None:
        # The owner already chose a terminal done-only menu for this request;
        # a later stall projection must not replace it with a repair surface.
        selected = owner_selection["tools"]
        if isinstance(tools, list):
            tools[:] = copy.deepcopy(selected)
        return selected
    available = _available_tools(runtime, tools)
    if _stall_rule_probe_refused(runtime):
        available = [tool for tool in available if _tool_name(tool) != "probe"]
    actions = stall_open_work_actions(runtime, available)
    if not actions:
        return [dict(tool) for tool in available]
    by_name = {_tool_name(tool): tool for tool in available}
    menu_names = tuple(str(action.get("tool", "")) for action in actions)
    for name in menu_names:
        by_name.setdefault(name, _stall_rule_fallback_tool(name))
    selected = [dict(by_name[name]) for name in menu_names if name in by_name]
    runtime.completion_menu = tuple(_tool_name(tool) for tool in selected)
    runtime.offered_tools = tuple(runtime.completion_menu)
    runtime._question_done_available = False
    runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
    episode = getattr(runtime, "_stall_open_work_episode", None)
    if isinstance(episode, dict) and not episode.get("rendered"):
        episode["rendered"] = True
        _emit(runtime, "completion_owner_stall_rule_rendered",
              episode_key=episode.get("key"), offered=list(runtime.completion_menu),
              excerpt=getattr(runtime, "_stall_open_work_frame", {}).get("excerpt", ""))
    if isinstance(tools, list):
        tools[:] = copy.deepcopy(selected)
    return selected


def _preserve_open_requirement_route(
        runtime: object, selected: list[dict[str, object]], tools: object,
) -> list[dict[str, object]]:
    """Keep one admitted witness route when STALLRULE narrows an open clause.

    STALLRULE owns the product-stall repair surface, while the clause
    discriminator owns the typed witness route.  A green-but-open frame can
    legitimately need both owners; dropping the witness here strands the
    requirement even though the route was admitted before projection.
    """
    open_ids = _reqwitness_open_ids(runtime)
    if not open_ids:
        return selected
    actions = _witness_evidence_actions(runtime, tools)
    if not actions:
        return selected
    available = {_tool_name(tool): tool for tool in _available_tools(runtime, tools)}
    idealpath_owner = getattr(runtime, "_idealpath_owner", lambda: None)()
    selected_names = {_tool_name(tool) for tool in selected}
    route_arguments: dict[str, dict[str, object]] = {}
    for action in actions:
        name = str(action.get("tool", "")).strip()
        if not name or name not in available:
            continue
        if idealpath_owner is not None:
            admission = idealpath_owner.admit_action(name)
            if admission.get("admitted") is not True:
                continue
        arguments = action.get("arguments")
        if isinstance(arguments, Mapping):
            route_arguments[name] = dict(arguments)
        if name not in selected_names:
            selected.append(copy.deepcopy(available[name]))
            selected_names.add(name)
    if not route_arguments:
        return selected
    setattr(runtime, "_completion_owner_open_requirement_route_arguments", route_arguments)
    runtime.completion_menu = tuple(_tool_name(tool) for tool in selected)
    runtime.offered_tools = tuple(runtime.completion_menu)
    runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
    if isinstance(tools, list):
        tools[:] = copy.deepcopy(selected)
    _emit(runtime, "completion_owner_open_requirement_route_preserved",
          open_ids=list(open_ids), offered=list(runtime.completion_menu),
          routes=sorted(route_arguments))
    return selected


def _probe_remaining(runtime: object) -> int | None:
    try:
        from kernel_route_budget import _budget
        budget = _budget(runtime)
        remaining = budget.get("remaining")
        return int(remaining) if type(remaining) is int else None
    except (ImportError, AttributeError, TypeError, ValueError):
        remaining = getattr(runtime, "probe_remaining", None)
        return int(remaining) if type(remaining) is int else None


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _journal_records(runtime: object) -> tuple[object, ...]:
    """Fully staged journal replay.

    ``journal.records()`` is a generator that holds the writer's replay flag
    until exhausted. A short-circuited consumer (``any``) that keeps it bound
    leaves the journal refusing every later append, including the terminal
    events; a generator passed to ``_sequence`` was silently read as empty.
    """
    journal = getattr(runtime, "journal", None)
    from kernel_host_cache import cache_enabled, journal_records
    if cache_enabled("JOURNAL_RECORDS"):
        return journal_records(journal)
    records_fn = getattr(journal, "records", None)
    if callable(records_fn):
        return tuple(records_fn())
    return _sequence(records_fn)


def _decision_codes(runtime: object, decision: object | None) -> tuple[str, ...]:
    values: list[str] = []
    source = decision if decision is not None else getattr(runtime, "_last_completion_decision", None)
    if source is not None:
        accepted = _field(source, "accepted", None)
        if accepted is not True:
            reason_codes = (source.get("reason_codes", ())
                            if isinstance(source, Mapping)
                            else getattr(source, "reason_codes", ()))
            values.extend(str(item) for item in _sequence(reason_codes))
    preflight = getattr(runtime, "_stateframe_witness_gate_preflight", None)
    if isinstance(preflight, Mapping):
        values.extend(str(item) for item in _sequence(preflight.get("reason_codes", ())))
    return tuple(dict.fromkeys(item for item in values if item))


def _model_value_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content", value.get("text", ""))
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            return " ".join(_model_value_text(item) for item in content)
        return _model_value_text(content)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_model_value_text(item) for item in value)
    return str(value or "")


def _latest_model_response_text(runtime: object) -> str:
    """Read only the latest provider response, including structured done args."""
    capture = getattr(runtime, "_last_capture", None)
    if capture is None:
        capture = getattr(runtime, "_active_capture", None)
    values: list[str] = []
    if capture is not None:
        for name in ("provider_response_message", "response_content", "content"):
            value = _field(capture, name, None)
            if value:
                values.append(_model_value_text(value))
        calls = _field(capture, "provider_tool_calls", _field(capture, "tool_calls", ()))
        for call in _sequence(calls):
            if isinstance(call, Mapping):
                function = call.get("function")
                arguments = function.get("arguments") if isinstance(function, Mapping) else call.get("arguments")
                if arguments:
                    values.append(_model_value_text(arguments))
            else:
                arguments = getattr(call, "arguments", None)
                if arguments:
                    values.append(_model_value_text(arguments))
    if not values:
        for name in ("provider_response_message", "response_content", "last_model_response"):
            value = getattr(runtime, name, None)
            if value:
                values.append(_model_value_text(value))
    return "\n".join(item.strip() for item in values if item.strip())[:12000]


def _declared_repair_clause_ids(runtime: object) -> tuple[str, ...]:
    values: list[str] = list(open_ids(runtime))
    for source_name in (
            "host_closure_assessments", "behavior_witness_assessments", "obligation_assessments"):
        for assessment in _sequence(getattr(runtime, source_name, ())):
            if not isinstance(assessment, Mapping) or _assessment_is_context_only(assessment):
                continue
            identifier = str(assessment.get("obligation_id", assessment.get("id", ""))).strip()
            status = str(assessment.get("verdict", assessment.get("status", ""))).upper()
            reason = str(assessment.get("reason", assessment.get("reason_code", ""))).upper()
            if identifier and (status not in {"PASS", "PASSED", "SATISFIED", "RESOLVED", "COMPLETED"}
                               or reason in {"MISSING_CHECKS", "CLAUSE_COVERAGE", "COVERAGE_OPEN", "UNKNOWN"}):
                values.append(identifier)
    return tuple(dict.fromkeys(item for item in values if item))


def _green_lock_open_requirement_ids(runtime: object) -> tuple[str, ...]:
    """The green lock's own open-requirement view (the set that deferred it)."""
    try:
        from kernel_green_lock import _open_requirement_ids
        return tuple(str(item) for item in _open_requirement_ids(runtime) if str(item).strip())
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return ()


def _declared_repair_state(runtime: object) -> dict[str, object] | None:
    text = _latest_model_response_text(runtime)
    clause_ids = _declared_repair_clause_ids(runtime)
    if not text or not clause_ids:
        return None
    digest = _digest(text)
    current = getattr(runtime, "_completion_owner_declared_repair_state", None)
    if not isinstance(current, dict) or current.get("digest") != digest:
        cue = bool(_DECLARED_REPAIR_CUE_RE.search(text)
                   and not _DECLARED_REPAIR_RETRACTION_RE.search(text))
        # 10-06 batch 906 (sample task oracle, seq 3302/3316/3359): after hunk 1 the
        # public check was green, the green lock DEFERRED for open requirements
        # [obl-000, obl-001, R1, R2], yet the terminal choice collapsed the menu to
        # ``done`` and refused hunks 2-3.  An admitted product edit made during the
        # response just closed, while host-owned requirements remain open, is the
        # same repair signal as a prose declaration: keep the repair surface for
        # the next response.  Bounded: a response without a new admitted edit
        # falls back to the terminal choice.
        admitted_mark = int(getattr(runtime, "_completion_owner_admitted_edit_count", 0) or 0)
        previous_mark = int(current.get("admitted_edit_mark", 0) or 0) if isinstance(current, dict) else 0
        open_requirement_progress = (
            not cue and admitted_mark > previous_mark
            and bool(_green_lock_open_requirement_ids(runtime)))
        current = {
            "digest": digest,
            "active": cue or open_requirement_progress,
            "source": ("open_requirements_after_admitted_edit"
                       if open_requirement_progress else "model_declaration"),
            "admitted_edit_mark": admitted_mark,
            "clause_ids": list(clause_ids),
            "excerpt": text[:600],
            "repair_edit_baseline": int(getattr(runtime, "_completion_owner_repair_edit_count", 0) or 0),
            "response_count": int(getattr(runtime, "_completion_owner_declared_repair_responses", 0) or 0) + 1,
        }
        setattr(runtime, "_completion_owner_declared_repair_state", current)
        setattr(runtime, "_completion_owner_declared_repair_responses", current["response_count"])
        _emit(runtime, "completion_owner_declared_repairs_detected",
              clause_ids=list(clause_ids), response_count=current["response_count"],
              excerpt=current["excerpt"], source=current["source"],
              active=current["active"])
    if int(getattr(runtime, "_completion_owner_repair_edit_count", 0) or 0) > int(
            current.get("repair_edit_baseline", 0) or 0):
        current["active"] = False
        current["cleared_by"] = "admitted_repair_edit"
    return current


def _check_receipt_handle(runtime: object) -> str:
    """Return the newest host-owned output handle, preferring a current failure."""
    rows: list[Mapping[str, object]] = []
    latest = getattr(runtime, "_latest_check_receipt", None)
    if isinstance(latest, Mapping):
        rows.append(latest)
    for name in ("host_check_receipts", "host_repair_check_receipts", "test_receipts"):
        value = getattr(runtime, name, ())
        if isinstance(value, Mapping):
            value = tuple(value.values())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            rows.extend(item for item in reversed(value) if isinstance(item, Mapping))
    fallback = ""
    for row in rows:
        handle = row.get("full_output_handle") or row.get("check_output_receipt")
        if not handle:
            continue
        status = str(row.get(
            "outcome", row.get("status", row.get("result_class", row.get("result", ""))))).upper()
        if status in _DEMONSTRATED_FAILURE_CODES or status in {
                "BEHAVIOR_FAIL", "RED", "FAIL", "FAILED", "ERROR"}:
            return str(handle)
        fallback = fallback or str(handle)
    return fallback


def _declared_repair_actions(runtime: object, tools: object) -> tuple[dict[str, object], ...]:
    state = _declared_repair_state(runtime)
    if not state or not state.get("active"):
        return ()
    available = {_tool_name(tool): tool for tool in _available_tools(runtime, tools)}
    actions: list[dict[str, object]] = []
    for name in _DECLARED_REPAIR_TOOLS:
        if name not in available:
            continue
        admitted, _reason = _current_tool_admission(runtime, name)
        if not admitted or not greenlock_predispatch_admission(runtime, name, {}):
            continue
        actions.append({
            "tool": name,
            "arguments": {},
            "reason": "repair the model-declared outstanding clause work",
            "clears": "the declared repair after an admitted edit or model retraction",
        })
    return tuple(actions)


def _declared_repair_notice(runtime: object, actions: Sequence[Mapping[str, object]]) -> str:
    state = _declared_repair_state(runtime) or {}
    ids = ", ".join(str(item) for item in state.get("clause_ids", ()))
    cap = int(state.get("response_count", 0) or 0) >= MAX_DECLARED_REPAIR_RESPONSES
    suffix = " bounded repair response allowance is exhausted; exit remains incomplete." if cap else ""
    if state.get("source") == "open_requirements_after_admitted_edit":
        tools_text = " or ".join(str(item.get("tool", "")) for item in actions)
        return (f"{TERMINAL_CHOICE_HEADER} done withheld: your edit was admitted and the public "
                f"check is green, but requirements [{ids}] remain open. Work actions available: "
                f"{tools_text}. Apply the next repair edit; a response without a new admitted "
                "edit returns the submit choice.")
    return (f"{TERMINAL_CHOICE_HEADER} done withheld: the latest model response declares "
            f"outstanding repairs for open clauses [{ids}]. Work actions available: "
            f"{' or '.join(str(item.get('tool', '')) for item in actions)}. "
            "Admit a repair edit or retract the declaration before done is available."
            + suffix)


def _witness_evidence_actions(runtime: object, tools: object) -> tuple[dict[str, object], ...]:
    """Expose bounded clause witnesses through the owner evidence menu."""
    available = _available_tools(runtime, tools)
    available_names = {_tool_name(item) for item in available}
    authored_context = getattr(runtime, "_authored_check_context", None)
    authored_proposal = getattr(runtime, "_authored_check_proposal", None)
    if isinstance(authored_context, Mapping) and authored_proposal is None:
        authored_proposal = authored_context.get("proposal")
    if isinstance(authored_proposal, Mapping) and "authored_check" in available_names:
        proposal = authored_proposal.get("arguments", authored_proposal)
        required = {
            "check_id", "requirement_id", "clause", "requirement_quote",
            "path", "selector", "source", "write_authority",
        }
        if isinstance(proposal, Mapping) and required <= set(proposal):
            authored_action = {
                "tool": "authored_check",
                "arguments": copy.deepcopy(dict(proposal)),
                "reason": "create the missing public-bound check",
                "clears": "the missing evidence gate",
                "blocker": "EVIDENCE_MISSING",
            }
            admitted = _admitted_action_projections(
                runtime, (authored_action,), available, validate_arguments=True)
            if admitted:
                return admitted
    try:
        from kernel_aha11_native_honesty import project_open_clause_actions
        native_clauses = getattr(runtime, "aha11_open_clauses", ())
        if isinstance(native_clauses, Sequence) and native_clauses:
            binding = runtime.binding() if callable(getattr(runtime, "binding", None)) else {}
            binding = binding if isinstance(binding, Mapping) else {}
            routes = project_open_clause_actions(
                native_clauses,
                tuple(_tool_name(item) for item in available),
                revision=str(binding.get("product_revision", binding.get("artifact", "current"))),
                attempted=getattr(runtime, "aha11_route_attempts", ()),
            )
            actions = tuple(item["action"] for item in routes if isinstance(item.get("action"), Mapping))
            if actions:
                return _admitted_action_projections(runtime, actions, available)
            setattr(runtime, "aha11_unrunnable_routes", routes)
            return ()
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    try:
        from kernel_clause_discriminator import next_action as discriminator_action
        direct = discriminator_action(runtime, available)
        if direct is not None:
            return _admitted_action_projections(runtime, (direct,), available)
        # An open REQWITNESS clause must not fall through to the legacy review
        # rail when its executable discriminator route is unavailable or spent;
        # the owner needs to disclose the limitation and keep submit live.
        if _reqwitness_open_ids(runtime):
            return ()
        from kernel_clause_review import enabled as clause_review_enabled, next_actions
        if not clause_review_enabled(runtime):
            return ()
        return _admitted_action_projections(runtime, next_actions(runtime, available), available)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return ()


def _witness_limitation(runtime: object, tools: object = ()) -> str | None:
    """Describe why an open clause cannot currently take its witness route."""
    clause_ids = _reqwitness_open_ids(runtime)
    if not clause_ids:
        return None
    try:
        from kernel_clause_discriminator import witness_route
        available, reason = witness_route(runtime, _available_tools(runtime, tools))
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        available, reason = False, "witness_rail_unavailable"
    if available and _witness_evidence_actions(runtime, tools):
        return None
    if available:
        reason = "witness_action_not_admitted"
    labels = {
        "witness_rail_unavailable": "the witness rail is unavailable",
        "witness_route_budget_exhausted": "the witness route budget is exhausted",
        "continuation_limit_exhausted": "the shared continuation limit is exhausted",
        "continuation_tool_limit_exhausted": "the shared continuation tool limit is exhausted",
        "continuation_limit_unavailable": "the shared continuation limit could not admit a witness turn",
        "witness_turn_limit_exhausted": "the per-clause witness turn limit is exhausted",
        "witness_action_not_admitted": "the witness action is not admitted at this boundary",
    }
    detail = labels.get(str(reason), "the witness route is unavailable")
    ids = ", ".join(clause_ids)
    return (f"{TERMINAL_CHOICE_HEADER} witness limitation: open public clause(s) [{ids}] "
            f"were not exercised because {detail}; submit remains available with "
            "this limitation disclosed.")


def _witness_ids(runtime: object, decision: object | None) -> tuple[str, ...]:
    values: list[str] = []
    source = decision if decision is not None else getattr(runtime, "_last_completion_decision", None)
    context_ids = {
        str(_field(item, "obligation_id", _field(item, "id", ""))).strip()
        for item in (
            *_sequence(_field(source, "obligation_assessments", ())),
            *_sequence(getattr(runtime, "host_closure_assessments", ())),
            *_sequence(getattr(runtime, "behavior_witness_assessments", ())),
            *_sequence(getattr(runtime, "obligation_assessments", ())),
        )
        if _assessment_is_context_only(item)
    }
    for owner in (source, getattr(runtime, "_stateframe_witness_gate_preflight", None)):
        if isinstance(owner, Mapping):
            candidates = owner.get("witness_ids", owner.get("needs_work_ids", ()))
        else:
            candidates = getattr(owner, "witness_ids", getattr(owner, "needs_work_ids", ()))
        for value in _sequence(candidates):
            text = str(value).strip()
            if text and text not in context_ids and text not in values:
                values.append(text)
    for assessment in _sequence(getattr(runtime, "host_closure_assessments", ())):
        if not isinstance(assessment, Mapping) or _assessment_is_context_only(assessment):
            continue
        if str(assessment.get("kind", "")).lower() != "behavior":
            continue
        value = str(assessment.get("obligation_id", assessment.get("id", ""))).strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


def _witness_entrypoint(runtime: object, clause_id: str) -> str:
    for source_name in (
            "host_closure_assessments", "behavior_witness_assessments", "obligation_assessments",
    ):
        for assessment in _sequence(getattr(runtime, source_name, ())):
            if not isinstance(assessment, Mapping) or _assessment_is_context_only(assessment):
                continue
            identifier = str(assessment.get("obligation_id", assessment.get("id", ""))).strip()
            if identifier == clause_id:
                return str(assessment.get("entrypoint", assessment.get("public_entrypoint", "")) or "").strip()
    return ""


def legal_next_actions(runtime: object, *, decision: object | None = None,
                       tools: object = ()) -> tuple[dict[str, object], ...]:
    """Compute the single host-owned set of actions that can clear live gates."""
    green_open_work = _green_open_work(runtime)
    if not next_action_owner_enabled(runtime) and not green_open_work:
        return ()
    no_progress_cap = _stall_rule_no_progress_edit_cap(runtime) and not _no_host_checks(runtime)
    if no_progress_cap:
        setattr(runtime, "_completion_owner_no_progress_edit_cap", True)
        if not green_open_work:
            _begin_uncertified_submit(runtime)
            return ()
    if not (_current_witness_receipt_closes_clause(runtime) or _no_host_checks(runtime)):
        declared_repairs = _declared_repair_actions(runtime, tools)
        if declared_repairs and not green_open_work:
            return declared_repairs
    witness_receipt_closes_clause = _current_witness_receipt_closes_clause(runtime)
    stall_actions = () if (witness_receipt_closes_clause or _no_host_checks(runtime)) else stall_open_work_actions(runtime, tools)
    if stall_actions and not green_open_work:
        # Stall actions are menu-level routes: the model supplies the concrete
        # path/source payload after seeing this frame.  Applying the final-call
        # argument validator here silently drops every edit/read action whose
        # placeholder payload is empty, recreating a done-only dead-end.
        projected_stall_actions = list(_admitted_action_projections(
            runtime, stall_actions, tools, validate_arguments=False))
        handle = _check_receipt_handle(runtime)
        if (handle and "read_check_receipt" in {
                _tool_name(tool) for tool in _available_tools(runtime, tools)}
                and not any(item.get("tool") == "read_check_receipt"
                            for item in projected_stall_actions)):
            receipt_action = {
                "tool": "read_check_receipt",
                "arguments": {"handle": handle, "stream": "assertion_context"},
                "reason": "inspect the stored failing assertion without rerunning checks",
                "clears": "the diagnostic evidence gap",
            }
            projected_stall_actions.append(receipt_action)
        return tuple(projected_stall_actions)
    governor = _common_sense_decision(runtime, tools)
    if (governor is not None and not green_open_work
            and not bool(getattr(governor, "release_uncertified", False))):
        return _admitted_action_projections(
            runtime, _common_sense_actions(governor), tools, validate_arguments=False)
    if governor is None and not green_open_work and not _common_sense_enabled(runtime):
        premature = _premature_frame(runtime, tools)
        if premature is not None:
            return _admitted_action_projections(runtime, _premature_actions(premature), tools)
    codes = {str(code).strip().upper() for code in _decision_codes(runtime, decision)}
    raw_codes = {str(code).strip().upper() for code in _decision_codes(runtime, decision)}
    evidence_gap_only = bool(codes & _EVIDENCE_GAP_CODES) and not bool(
        codes & _DEMONSTRATED_FAILURE_CODES)
    witness_ids = _witness_ids(runtime, decision)
    preflight = getattr(runtime, "_stateframe_witness_gate_preflight", None)
    witness_pending = bool(codes & {code.upper() for code in _WITNESS_CODES})
    witness_pending = witness_pending or (
        isinstance(preflight, Mapping) and bool(preflight.get("requested"))
    )
    if not witness_pending:
        for assessment in _sequence(getattr(runtime, "host_closure_assessments", ())):
            if (isinstance(assessment, Mapping)
                    and not _assessment_is_context_only(assessment)
                    and str(assessment.get("kind", "")).lower() == "behavior"):
                if str(assessment.get("verdict", assessment.get("status", ""))).upper() not in {"PASS", "PASSED", "SATISFIED"}:
                    witness_pending = True
                    break
    names = {_tool_name(tool) for tool in _available_tools(runtime, tools)}
    typed_probe_available = any(
        _typed_probe_tool(tool) for tool in _available_tools(runtime, tools))
    remaining = _probe_remaining(runtime)
    actions: list[dict[str, object]] = []
    green_current = _green_current(runtime)
    demonstrated_failure = bool(codes & _DEMONSTRATED_FAILURE_CODES) or known_violation(runtime)
    blocked_rules: dict[str, str] = {}
    reused_clear = False

    def add(tool: str, *, reason: str, clears: str, arguments: Mapping[str, object] | None = None) -> None:
        nonlocal reused_clear
        if tool not in names or any(item["tool"] == tool for item in actions):
            return
        admitted, admission_reason = _current_tool_admission(runtime, tool)
        if not admitted:
            blocked_rules[tool] = admission_reason or "tool_admission_refused"
            return
        skeptic_clause_route = (
            bool(blocked_rules.get("run_tests")) and "DETREASON_SKEPTIC" in codes
        )
        if (tool in {"edit", "str_replace"} and green_current and not green_open_work
                and not demonstrated_failure and not skeptic_clause_route):
            # Green evidence without a witnessed current-revision failure is
            # not permission to manufacture repair work.
            return
        if tool == "probe" and (not typed_probe_available
                                 or (remaining is not None and remaining <= 0)):
            return
        if tool == "run_tests":
            validated = host_public_check_arguments(runtime, arguments)
            if validated is None:
                return
            arguments = validated
        if tool == "read_check_receipt":
            # This is a read-only evidence lookup.  It must remain reachable
            # even when a completion/green-lock owner is withholding mutation
            # or another completion decision.
            admitted, rule, authority_active = True, None, False
        else:
            admitted, rule, authority_active = _greenlock_predispatch(runtime, tool, arguments)
        if authority_active and not admitted:
            blocked_rules[tool] = rule or "greenlock_refused"
            if tool == "run_tests" and rule in _REUSED_GREEN_RULES:
                selectors = arguments.get("selectors", ()) if isinstance(arguments, Mapping) else ()
                if _record_reused_green_verification(runtime, tuple(str(item) for item in selectors)):
                    reused_clear = True
            return
        actions.append({"tool": tool, "arguments": copy.deepcopy(dict(arguments or {})),
                        "reason": reason, "clears": clears})

    source = decision if decision is not None else getattr(runtime, "_last_completion_decision", None)
    repair_action = _field(source, "repair_action", None) if source is not None else None
    if isinstance(repair_action, Mapping):
        repair_tool = str(repair_action.get("tool", "")).strip()
        if repair_tool and not (evidence_gap_only and repair_tool in _PRODUCT_EDIT_TOOLS):
            add(repair_tool, reason="use the host-selected clearing action",
                 clears="the refusal's host-selected gate",
                 arguments=(repair_action.get("arguments")
                            if isinstance(repair_action.get("arguments"), Mapping) else None))

    if green_open_work:
        # A green check does not close remaining requirements. Preserve each
        # ordinary repair/evidence action that is still admitted on this surface.
        for tool in ("edit", "str_replace", "read", "run_tests"):
            add(tool, reason="clear or inspect an open requirement after the green check",
                clears="the open requirement/obligation state")

    if reused_clear and not green_open_work:
        add("done", reason="reuse the admitted current-green verification",
            clears="the skeptic challenge using named current-revision evidence")
        if actions:
            return tuple(actions)

    if blocked_rules and "DETREASON_SKEPTIC" in codes:
        # Identical green reruns are not a clearing action.  Offer the host's
        # non-test interpretation route instead, subject to the same admission
        # predicate as every other owner action.
        add("read", reason="read the named clause/spec and record its interpretation",
            clears="the skeptic's open clause interpretation")

    if witness_pending:
        clause = ", ".join(witness_ids) if witness_ids else "the open behavior requirement"
        clause_id = witness_ids[0] if witness_ids else ""
        entrypoint = _witness_entrypoint(runtime, clause_id)
        if clause_id and _valid_witness_entrypoint(entrypoint):
            add("probe", reason="provide one typed public witness proposal",
                clears=f"the witness evidence gate for {clause}",
                arguments={"clause_id": clause_id, "entrypoint": entrypoint,
                           "input": {}, "expected": {}, "rationale": "derive from the requirement"})
        # A witness refusal can coexist with another live evidence/product
        # refusal. Keep the concrete observation route reachable so the
        # combined refusal cannot collapse back to probe-only or done-only.
        if codes - {code.upper() for code in _WITNESS_CODES}:
            add("read", reason="inspect the live requirement before clearing the witness",
                clears="the associated evidence or product gate")
    if codes & _RETRY_DONE_CODES:
        add("done", reason="retry the explicitly offered bounded challenge",
            clears="the one-shot done challenge")
    if codes & _QUESTION_CODES:
        for tool in ("read", "run_tests"):
            if tool in names:
                add(tool, reason="inspect the requested clause before answering it",
                    clears="the question evidence clause")
                break
    if codes & _CONTROL_CODES:
        for tool in ("run_tests", "read"):
            if tool in names:
                add(tool, reason="refresh the host control evidence",
                    clears="the control-plane refusal")
                break
    if codes & _WORK_CODES:
        for tool in ("work_update", "work"):
            if tool in names:
                add(tool, reason="close the declared work record",
                    clears="the open work obligation")
                break
    if codes & _COVERAGE_CODES:
        # Coverage producers carry a concrete read repair when available.  Keep
        # a verification fallback for reduced surfaces that omit read, but do
        # not pretend coverage uncertainty is a product defect requiring edit.
        add("read", reason="inspect the named or uncovered public requirement",
            clears="the coverage evidence gate")
        if "read" not in names:
            add("run_tests", reason="refresh the public evidence after inspection",
                clears="the coverage evidence gate")
    if codes & {code.upper() for code in _CHECK_CODES}:
        add("run_tests", reason="refresh the exact current public checks",
            clears="the current-revision evidence gate")
    if codes & {code.upper() for code in _REPAIR_CODES} and not evidence_gap_only:
        add("run_tests", reason="refresh the exact current public checks",
            clears="the current-revision evidence gate")
        add("edit", reason="repair the demonstrated current-revision failure",
            clears="the demonstrated failure before re-verification")
        add("str_replace", reason="repair the demonstrated current-revision failure",
            clears="the demonstrated failure before re-verification")
    receipt_handle = _check_receipt_handle(runtime)
    if receipt_handle:
        add(
            "read_check_receipt",
            reason="inspect the stored assertion without another run_tests decision",
            clears="the readable check-evidence gap",
            arguments={"handle": receipt_handle, "stream": "assertion_context"},
        )
    if reused_clear:
        actions.clear()
        add("done", reason="reuse the admitted current-green verification",
            clears="the skeptic challenge using named current-revision evidence")
        if actions:
            return tuple(actions)
    # Any refusal without a concrete model action is terminally owned by the
    # host.  In particular, never fall through to the legacy done-only offer.
    if not actions and raw_codes:
        return ()
    return tuple(actions)


def _next_action_notice(runtime: object, actions: Sequence[Mapping[str, object]],
                        decision: object | None) -> str:
    codes = _decision_codes(runtime, decision)
    reason = ", ".join(codes) if codes else "an outstanding completion gate"
    rendered: list[str] = []
    for action in actions:
        tool = str(action.get("tool", ""))
        if tool == "probe":
            arguments = action.get("arguments")
            mode = str(arguments.get("mode", "")) if isinstance(arguments, Mapping) else ""
            if mode in {"clause_review_propose", "clause_review_assess"}:
                episode_id = str(arguments.get("episode_id", "")) if isinstance(arguments, Mapping) else ""
                clause_ids = arguments.get("clause_ids", ()) if isinstance(arguments, Mapping) else ()
                rendered.append(
                    "probe(mode=" + json.dumps(mode, ensure_ascii=False)
                    + ", episode_id=" + json.dumps(episode_id, ensure_ascii=False)
                    + ", clause_ids=" + json.dumps(list(_sequence(clause_ids)), ensure_ascii=False) + ")")
                continue
            clause_id = str(arguments.get("clause_id", "")) if isinstance(arguments, Mapping) else ""
            entrypoint = str(arguments.get("entrypoint", "")) if isinstance(arguments, Mapping) else ""
            input_value = arguments.get("input", {}) if isinstance(arguments, Mapping) else {}
            expected_value = arguments.get("expected", {}) if isinstance(arguments, Mapping) else {}
            rationale = str(arguments.get("rationale", "")) if isinstance(arguments, Mapping) else ""
            rendered.append(
                "probe(clause_id=" + json.dumps(clause_id, ensure_ascii=False)
                + ", entrypoint=" + json.dumps(entrypoint, ensure_ascii=False)
                + ", input=" + json.dumps(input_value, sort_keys=True, ensure_ascii=False, default=str)
                + ", expected=" + json.dumps(expected_value, sort_keys=True, ensure_ascii=False, default=str)
                + ", rationale=" + json.dumps(rationale, ensure_ascii=False) + ")"
            )
        elif tool == "run_tests":
            arguments = action.get("arguments")
            target = str(arguments.get("target", "")) if isinstance(arguments, Mapping) else ""
            rendered.append("run_tests(target=" + json.dumps(target, ensure_ascii=False) + ")")
        else:
            rendered.append(tool + "()")
    return (f"{TERMINAL_CHOICE_HEADER} done refused: {reason}. "
            f"Next legal action: {' or '.join(rendered)}. "
            f"Success clears: {'; '.join(str(item.get('clears', 'the gate')) for item in actions)}.")


def _host_only_notice(runtime: object, decision: object | None) -> str:
    route = completion_route_decision(
        runtime, decision,
        tools=getattr(runtime, "_completion_owner_full_tools", ()),
    )
    typed_notice = completion_route_notice(runtime, route)
    if typed_notice:
        return typed_notice
    codes = _decision_codes(runtime, decision)
    reason = ", ".join(codes) if codes else "an outstanding host gate"
    return (f"{TERMINAL_CHOICE_HEADER} typed completion route: "
            f"reason_category={route.reason_category}; reason_codes={list(route.reason_codes)}; "
            f"menu_actions={list(route.advertised_actions)}. "
            f"Completion held: done refused: {reason}.")


def _begin_uncertified_submit(runtime: object) -> None:
    """End an unresolvable owner refusal once, without reopening ``done``."""
    if getattr(runtime, "_completion_owner_uncertified_submit_used", False):
        return
    setattr(runtime, "_completion_owner_uncertified_submit_used", True)
    setattr(runtime, "_completion_owner_uncertified_submit", True)
    setattr(runtime, "system_outcome", "UNCERTIFIED_SUBMIT")
    _emit(runtime, "completion_owner_uncertified_submit_started",
           reason_codes=list(_decision_codes(runtime, None)))


def _route_decision_data(decision: object) -> Mapping[str, object]:
    if isinstance(decision, Mapping):
        return decision
    convert = getattr(decision, "to_dict", None)
    if callable(convert):
        try:
            value = convert()
            return value if isinstance(value, Mapping) else {}
        except (AttributeError, TypeError, ValueError):
            return {}
    return {}


def _route_revision(runtime: object) -> str:
    revision = _revision(runtime)
    if isinstance(revision, Mapping):
        for key in ("product_revision", "artifact", "revision", "workspace_revision"):
            if revision.get(key) not in (None, ""):
                return str(revision[key])[:180]
    return str(revision or "UNKNOWN")[:180]


def _model_turns_remaining(runtime: object) -> int | None:
    limits = getattr(runtime, "continuation_limits", None)
    if limits is not None and getattr(limits, "limit", 0) is None:
        return None
    value = getattr(limits, "model_decisions_remaining", None)
    if type(value) is int:
        return max(0, value)
    owner = getattr(runtime, "owner", None)
    step = getattr(owner, "current_step", None)
    limit = getattr(runtime, "max_steps", getattr(runtime, "MAX_STEPS", None))
    if type(step) is int and type(limit) is int:
        return max(0, limit - step)
    return None


def _context_assessment_ids(decision: object, runtime: object) -> set[str]:
    rows = [
        *_sequence(_field(decision, "obligation_assessments", ())),
        *_sequence(getattr(runtime, "host_closure_assessments", ())),
        *_sequence(getattr(runtime, "behavior_witness_assessments", ())),
        *_sequence(getattr(runtime, "obligation_assessments", ())),
    ]
    return {
        str(_field(row, "obligation_id", _field(row, "id", ""))).strip()
        for row in rows if _assessment_is_context_only(row)
        and str(_field(row, "obligation_id", _field(row, "id", ""))).strip()
    }


def _route_bindings(decision: object, runtime: object) -> tuple[
        tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    context_ids = _context_assessment_ids(decision, runtime)
    obligation_rows: list[Mapping[str, object]] = []
    check_rows: list[Mapping[str, object]] = []
    seen_obligations: set[str] = set()
    seen_checks: set[str] = set()
    for raw_value in _sequence(_field(decision, "obligation_assessments", ())):
        raw = raw_value.to_dict() if hasattr(raw_value, "to_dict") else raw_value
        if not isinstance(raw, Mapping) or _assessment_is_context_only(raw):
            continue
        ident = str(raw.get("obligation_id", raw.get("id", "")) or "").strip()
        if ident and ident not in seen_obligations:
            row: dict[str, object] = {"obligation_id": ident}
            status = raw.get("verdict", raw.get("status"))
            reason = raw.get("reason", raw.get("reason_code"))
            if status not in (None, ""):
                row["status"] = str(status)[:80]
            if reason not in (None, ""):
                row["reason"] = str(reason)[:120]
            obligation_rows.append(row)
            seen_obligations.add(ident)
        refs = raw.get("check_bindings", raw.get("check_refs", raw.get("checks", ())))
        for check in _sequence(refs):
            if isinstance(check, str):
                check = {"check_id": check}
            if not isinstance(check, Mapping):
                continue
            selector = check.get("selector", check.get("test_selector", check.get("check_id")))
            check_id = check.get("check_id", check.get("id"))
            if selector in (None, "") and check_id in (None, ""):
                continue
            revision = check.get("revision", raw.get("revision", raw.get("current_revision")))
            key = _digest({"obligation": ident, "selector": selector,
                           "check_id": check_id, "revision": revision})
            if key in seen_checks:
                continue
            row = {"obligation_id": ident or None}
            if selector not in (None, ""):
                row["selector"] = str(selector)[:240]
            if check_id not in (None, ""):
                row["check_id"] = str(check_id)[:180]
            if revision not in (None, ""):
                row["revision"] = str(revision)[:180]
            check_rows.append(row)
            seen_checks.add(key)
    for raw_id in _sequence(_field(decision, "needs_work_ids", ())):
        ident = str(raw_id).strip()
        if ident and ident not in context_ids and ident not in seen_obligations:
            obligation_rows.append({"obligation_id": ident, "status": "OPEN"})
            seen_obligations.add(ident)
    return tuple(obligation_rows[:64]), tuple(check_rows[:64])


def _route_executed_checks(runtime: object) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    seen: set[str] = set()
    sources: list[object] = [getattr(getattr(runtime, "owner", None), "_receipts", ())]
    for name in ("host_check_receipts", "test_receipts", "probe_receipts", "integrity_receipts"):
        sources.append(getattr(runtime, name, ()))
    for source in sources:
        for raw in _sequence(source):
            if not isinstance(raw, Mapping):
                continue
            kind = str(raw.get("kind", raw.get("type", raw.get("tool", "")))).strip().lower()
            if kind in {
                    "done", "edit", "file_create", "grep", "locate", "read",
                    "read_check_receipt", "read_receipt", "recall_evidence",
                    "search", "source_read", "write"}:
                continue
            if not (kind in {"check", "pytest", "test", "tests", "probe", "verification",
                             "verify", "lint", "format", "public_check", "oracle_check"}
                    or raw.get("check_executed") is True
                    or "check_id" in raw or "exit_code" in raw or "returncode" in raw):
                continue
            selector_values: list[str] = []
            for key in ("selectors", "public_selectors", "selector", "test_selector"):
                selector_values.extend(str(value).strip() for value in _sequence(raw.get(key))
                                       if str(value).strip())
                if isinstance(raw.get(key), str) and raw[key].strip():
                    selector_values.append(raw[key].strip())
            selector_values = list(dict.fromkeys(selector_values))
            revision = next((raw.get(key) for key in (
                "current_revision", "workspace_revision", "product_revision", "revision",
            ) if raw.get(key) not in (None, "")), None)
            if revision is None:
                for binding_key in ("after_binding", "binding", "current_binding"):
                    binding = raw.get(binding_key)
                    if isinstance(binding, Mapping):
                        revision = next((binding.get(key) for key in (
                            "product_revision", "artifact", "revision", "workspace_revision",
                        ) if binding.get(key) not in (None, "")), None)
                        if revision is not None:
                            break
            status = str(raw.get(
                "status", raw.get("outcome", raw.get("result_class", raw.get("result", "")))
            )).upper()
            if type(raw.get("exit_code")) is int:
                status = "PASS" if raw["exit_code"] == 0 else "FAIL"
            if not selector_values or revision is None or status not in {
                    "PASS", "PASSED", "GREEN", "SUCCESS", "OK", "FAIL", "FAILED", "RED", "ERROR"}:
                continue
            receipt_id = str(raw.get("receipt_id", raw.get("event_id", raw.get("id", ""))))
            key = _digest({"receipt_id": receipt_id, "selectors": selector_values,
                           "revision": revision, "status": status})
            if key in seen:
                continue
            seen.add(key)
            rows.append({"receipt_id": receipt_id or None, "selectors": selector_values[:16],
                         "revision": str(revision)[:180], "outcome": status})
    return tuple(rows[-64:])


def _route_dispatchable(runtime: object, name: str, action: Mapping[str, object]) -> bool:
    if name == "done":
        if (action.get("route") in {"UNCERTIFIED_SUBMIT", "CERTIFIED_ACCEPTANCE"}
                and postgreen_done_offer(runtime)[0]):
            return True
        return _current_tool_admission(runtime, name)[0] and not _done_dependency_denied(runtime)
    return _action_dispatch_admissible(
        runtime, name,
        action.get("arguments") if isinstance(action.get("arguments"), Mapping) else None,
    )


def _postgreen_verification_pending(runtime: object) -> bool:
    """Read explicit post-green edit state without failing open on an unreadable state."""
    try:
        from kernel_post_green_menu import pending_verification
        return pending_verification(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return True


def _postgreen_green_selectors(runtime: object) -> tuple[str, ...]:
    """Reuse selectors only from a successful host check for post-edit verification."""
    for receipt in reversed(_route_executed_checks(runtime)):
        if str(receipt.get("outcome", "")).upper() not in {
                "PASS", "PASSED", "GREEN", "SUCCESS", "OK"}:
            continue
        selectors = receipt.get("selectors", ())
        values = tuple(str(item).strip() for item in _sequence(selectors) if str(item).strip())
        if values:
            return values
    return ()


def _postgreen_work_actions(
        runtime: object, decision: object | None, tools: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Keep live recovery work ahead of an uncertified post-green submit offer."""
    by_name = {_tool_name(tool) for tool in tools}
    verification_pending = _postgreen_verification_pending(runtime)
    repair_nag_active = verification_pending
    quieted_pending_route = False
    try:
        from kernel_post_green_menu import (
            host_verified_done,
            hostverified_quiet_enabled,
            pending_verification_active,
        )
        repair_nag_active = pending_verification_active(runtime)
        quieted_pending_route = (
            verification_pending
            and hostverified_quiet_enabled(runtime)
            and host_verified_done(runtime)
        )
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        # Unknown state keeps the repair recommendation fail-closed.
        pass
    actions: Sequence[Mapping[str, object]] = ()
    try:
        actions = legal_next_actions(runtime, decision=decision, tools=tools)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    binding_fn = getattr(runtime, "binding", None)
    verdict_fn = getattr(runtime, "public_test_verdict", None)
    if "run_tests" in by_name and callable(binding_fn) and callable(verdict_fn):
        try:
            current_binding = binding_fn()
            verdict = verdict_fn()
            verdict_binding = (
                verdict.get("current_binding") if isinstance(verdict, Mapping)
                else getattr(verdict, "current_binding", None)
            )
            # QUIET silences the post-green repair recommendation, not the
            # explicit verification route: CH-04 evidence stays armed until a
            # later receipt consumes it, even when the host already verified
            # the same request.
            if (verdict_binding != current_binding or repair_nag_active
                    or quieted_pending_route):
                quieted_repair_nag = quieted_pending_route and not repair_nag_active
                refresh = next((
                    dict(action) for action in actions
                    if isinstance(action, Mapping)
                    and str(action.get("tool", "")).strip() == "run_tests"
                ), None)
                if refresh is not None and _route_dispatchable(runtime, "run_tests", refresh):
                    if quieted_repair_nag:
                        refresh["reason"] = "keep the post-green verification route available"
                        refresh["clears"] = "the current-revision verification receipt"
                    return (refresh,)
                arguments = host_public_check_arguments(runtime)
                selectors = ()
                if arguments is None or not arguments.get("selectors"):
                    selectors = _postgreen_green_selectors(runtime)
                    if selectors:
                        arguments = host_public_check_arguments(
                            runtime, selectors, allow_candidate_fallback=True)
                refresh = None
                if isinstance(arguments, Mapping) and arguments.get("selectors"):
                    refresh = {
                        "tool": "run_tests",
                        "arguments": dict(arguments),
                        "reason": ("keep the post-green verification route available"
                                   if quieted_repair_nag else
                                   "refresh public check evidence for the current revision"),
                        "clears": ("the current-revision verification receipt"
                                   if quieted_repair_nag else
                                   "the stale revision binding on the latest public check"),
                    }
                if refresh is not None and _route_dispatchable(runtime, "run_tests", refresh):
                    return (refresh,)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
    return tuple(
        copy.deepcopy(dict(action)) for action in actions
        if isinstance(action, Mapping)
        and str(action.get("tool", "")).strip() != "done"
        and str(action.get("tool", "")).strip() in by_name
        and _route_dispatchable(runtime, str(action.get("tool", "")).strip(), action)
    )


def completion_route_decision(
        runtime: object, decision: object | None = None, *, tools: object = (),
) -> CompletionRouteDecision:
    """Resolve one revision-bound route and cache it for every completion projection."""
    from kernel_done_readiness_authority import cached_decision

    shared_done_decision = cached_decision(runtime)
    source = decision if decision is not None else getattr(runtime, "_last_completion_decision", None)
    product_parent_unchanged = _product_parent_unchanged(
        runtime, reason="completion_route")
    full_surface = getattr(runtime, "_completion_owner_full_tools", ())
    surface = _available_tools(runtime, full_surface or tools)
    decision_data = _route_decision_data(source)
    codes = _decision_codes(runtime, source)
    gate_recoveries: tuple[Mapping[str, object], ...] = ()
    gate_blocker_codes: tuple[str, ...] = ()
    gate_vector = None
    gate_factory = getattr(runtime, "pass_gates_vector", None)
    if callable(gate_factory):
        try:
            gate_vector = gate_factory(surface)
            if not getattr(gate_vector, "ready", False):
                gate_blocker_codes = tuple(str(getattr(blocker, "code", "HOST_BLOCKER"))
                                            for blocker in getattr(gate_vector, "blockers", ()))
                gate_recoveries = tuple(item for item in getattr(gate_vector, "recoveries", ())
                                        if isinstance(item, Mapping))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            gate_recoveries = ()
    revision = _route_revision(runtime)
    current_open_ids = open_ids(runtime)
    green_open_work = bool(_green_current(runtime) and current_open_ids)
    obligations, check_bindings = _route_bindings(source, runtime)
    executed_checks = _route_executed_checks(runtime)
    advisory_degraded = _advisory_degraded_fact(runtime)
    postgreen_done_required, _postgreen_revision = postgreen_done_offer(runtime)
    postgreen_verification_pending = _postgreen_verification_pending(runtime)
    tool_by_name = {_tool_name(tool): tool for tool in surface}
    owner_state = _state(runtime)
    from kernel_capability_floor import enabled as capability_floor_enabled

    # A soft done-only count cannot hide submit while the floor is active;
    # final reserved-response charging below remains a separate hard budget.
    done_capacity = (
        bool(owner_state.get("final_submit_request_pending"))
        or capability_floor_enabled(runtime)
        or int(owner_state.get("done_only_offer_count", 0) or 0) < MAX_RESERVED_RESPONSES
    )
    done_admissible = (
        "done" in tool_by_name and (postgreen_done_required or done_capacity)
        and (postgreen_done_required or _route_dispatchable(
            runtime, "done", {"tool": "done", "arguments": {}}))
    )
    shared_uncertified = False
    if (done_admissible and shared_done_decision is not None
            and getattr(shared_done_decision, "decision", None) == "ACCEPTED_UNCERTIFIED"):
        from kernel_done_readiness_authority import uncertified_gate_compatible

        shared_reason = str(getattr(shared_done_decision, "reason_code", ""))
        ordinary_codes = tuple(code for code in codes if code != shared_reason)
        shared_ids = tuple(dict.fromkeys((
            *open_ids(runtime),
            *tuple(getattr(shared_done_decision, "blocker_ids", ()) or ()),
            *tuple(getattr(shared_done_decision, "open_obligation_ids", ()) or ()),
        )))
        gate_ids = tuple(str(getattr(blocker, "blocker_id", ""))
                         for blocker in getattr(gate_vector, "blockers", ()) or ()
                         if str(getattr(blocker, "blocker_id", "")))
        shared_uncertified = uncertified_gate_compatible(
            runtime, shared_done_decision,
            reason_codes=ordinary_codes + gate_blocker_codes,
            open_work_ids=shared_ids + gate_ids,
            unresolved_gating_items=shared_ids + gate_ids,
            gate_vector=gate_vector,
        )
    if shared_uncertified:
        # The shared decision already records these IDs as host-owned debt.
        # Do not turn the same debt into an independent route refusal/recovery.
        codes = ()
        gate_recoveries = ()
    else:
        codes = tuple(dict.fromkeys((*codes, *gate_blocker_codes)))
    signature = _digest({
        "decision": decision_data,
        "shared_done_decision": (
            shared_done_decision.to_dict()
            if shared_done_decision is not None
            and callable(getattr(shared_done_decision, "to_dict", None))
            else None
        ),
        "shared_uncertified": shared_uncertified,
        "reason_codes": list(codes),
        "open_ids": list(current_open_ids),
        "green_open_work": green_open_work,
        "revision": revision,
        "tools": _digest(surface),
        "obligation_bindings": obligations,
        "check_bindings": check_bindings,
        "executed_checks": executed_checks,
        "advisory_degraded": advisory_degraded,
        "uncertified_submission_admissible": done_admissible,
        "postgreen_done_required": postgreen_done_required,
        "postgreen_verification_pending": postgreen_verification_pending,
        "product_parent_unchanged": product_parent_unchanged,
        "turns_remaining": _model_turns_remaining(runtime),
        "gate_recoveries": gate_recoveries,
    })
    cached = getattr(runtime, "_completion_owner_completion_decision", None)
    if (isinstance(cached, CompletionRouteDecision) and cached.signature == signature):
        return cached

    normalized_codes = {str(code).strip().upper() for code in codes}
    accepted = bool(_field(source, "accepted", False))
    certified = bool(accepted and _candidate(runtime)[0] and not current_open_ids)
    if certified:
        category = "CERTIFIED_ACCEPTANCE"
    elif normalized_codes & _DEMONSTRATED_FAILURE_CODES:
        category = "PRODUCT_FAILURE"
    elif normalized_codes & _EVIDENCE_GAP_CODES:
        category = "EVIDENCE_GAP"
    elif normalized_codes:
        category = "HOST_BLOCKER"
    else:
        category = "NO_REFUSAL"

    raw_actions: tuple[Mapping[str, object], ...] = ()
    if green_open_work or category not in {"CERTIFIED_ACCEPTANCE", "NO_REFUSAL"}:
        try:
            raw_actions = tuple(legal_next_actions(
                runtime, decision=source, tools=surface,
            ))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            raw_actions = ()
        raw_actions = (*raw_actions, *gate_recoveries)

    for recovery in gate_recoveries:
        recovery_tool = str(recovery.get("tool", "")).strip()
        recovery_args = recovery.get("arguments", {})
        allowed, refusal, _active = _greenlock_predispatch(
            runtime, recovery_tool,
            recovery_args if isinstance(recovery_args, Mapping) else None,
        )
        if recovery_tool not in tool_by_name or not allowed:
            _emit(runtime, "completion_route_recovery_refused",
                  tool=recovery_tool, reason=refusal or "schema_unavailable",
                  schema_available=recovery_tool in tool_by_name)

    if category == "EVIDENCE_GAP" and not green_open_work:
        selected_actions = tuple(
            action for action in raw_actions
            if str(action.get("tool", "")).strip() in _EVIDENCE_ROUTE_TOOLS
        )
    else:
        selected_actions = raw_actions
    selected_actions = tuple(
        action for action in selected_actions
        if str(action.get("tool", "")).strip() in tool_by_name
        and _route_dispatchable(runtime, str(action.get("tool", "")).strip(), action)
    )

    route = "NONE"
    actions: tuple[Mapping[str, object], ...] = selected_actions
    if green_open_work and selected_actions:
        route = "ACTION"
        actions = selected_actions
        if (done_admissible
                and not any(str(action.get("tool", "")).strip() == "done"
                            for action in actions)):
            actions = (*actions, {
                "tool": "done",
                "arguments": {},
                "route": "UNCERTIFIED_SUBMIT",
                "reason": "submit the green artifact while open requirements remain visible",
                "clears": "the bounded submission request without certifying correctness",
            })
    elif green_open_work:
        # No admitted action may mean the host cannot continue. Keep this a
        # blocker rather than rendering a verify or done-only route.
        route = "HOST_BLOCKER"
        actions = ()
    elif category == "CERTIFIED_ACCEPTANCE":
        route = "CERTIFIED_ACCEPTANCE"
        actions = ()
    elif product_parent_unchanged:
        route = "NONE"
        actions = ()
    elif category in {"PRODUCT_FAILURE", "HOST_BLOCKER"} and selected_actions:
        route = "ACTION"
    elif category == "EVIDENCE_GAP" and selected_actions:
        route = "ACTION"
    elif category == "EVIDENCE_GAP" and done_admissible:
        route = "UNCERTIFIED_SUBMIT"
        actions = ({
            "tool": "done",
            "arguments": {},
            "reason": "store the artifact while evidence obligations remain open",
            "clears": "the bounded submission request without certifying correctness",
        },)
    elif category in {"EVIDENCE_GAP", "HOST_BLOCKER", "PRODUCT_FAILURE"}:
        route = "HOST_BLOCKER"
        actions = ()

    if shared_uncertified:
        from kernel_done_readiness_authority import host_debt_disposition_text

        existing_actions = tuple(
            copy.deepcopy(dict(action)) for action in actions
            if str(action.get("tool", "")).strip() != "done"
        )
        done_action = {
            "tool": "done",
            "arguments": {},
            "route": "UNCERTIFIED_SUBMIT",
            "reason": host_debt_disposition_text(shared_done_decision),
            "clears": "the bounded submission request without certifying correctness",
        }
        actions = (*existing_actions, done_action)
        route = "ACTION" if existing_actions else "UNCERTIFIED_SUBMIT"

    if postgreen_done_required and done_admissible and not green_open_work:
        pending_work = () if certified else _postgreen_work_actions(runtime, source, surface)
        done_route = "CERTIFIED_ACCEPTANCE" if certified else "UNCERTIFIED_SUBMIT"
        done_action = {
            "tool": "done",
            "arguments": {},
            "route": done_route,
            "reason": ("submit the certified current revision" if certified else
                       "keep the changed green artifact submit-capable while certification is incomplete"),
            "clears": ("the certified completion" if certified else
                       "the bounded submission request without certifying correctness"),
        }
        if pending_work:
            # A changed revision invalidates the prior green receipt. Keep the
            # live verification/repair action available before any
            # uncertified terminal route, even when the done route remains
            # technically admissible as a non-certified storage operation.
            route = "ACTION"
            actions = (*pending_work, done_action)
        else:
            actions = tuple(action for action in actions if action.get("tool") != "done")
            if route == "ACTION":
                actions = (*actions, done_action)
            elif certified:
                route = "ACTION"
                actions = (done_action,)
            else:
                route = "UNCERTIFIED_SUBMIT"
                actions = (done_action,)

    advertised = tuple(dict.fromkeys(
        str(action.get("tool", "")).strip() for action in actions
        if str(action.get("tool", "")).strip()
    ))
    dispatchable = tuple(name for name in advertised if any(
        _route_dispatchable(runtime, name, action)
        for action in actions if str(action.get("tool", "")).strip() == name
    ))
    if advertised != dispatchable:
        route = "HOST_BLOCKER"
        actions = ()
        advertised = ()
        dispatchable = ()
    schemas = tuple(copy.deepcopy(dict(tool_by_name[name]))
                    for name in advertised if name in tool_by_name)
    if tuple(_tool_name(tool) for tool in schemas) != advertised:
        route = "HOST_BLOCKER"
        actions = ()
        advertised = ()
        dispatchable = ()
        schemas = ()
    result = CompletionRouteDecision(
        signature=signature,
        revision=revision,
        reason_category=category,
        route=route,
        reason_codes=tuple(codes),
        obligation_bindings=obligations,
        check_bindings=check_bindings,
        executed_checks=executed_checks,
        certified_acceptance=certified,
        uncertified_submission_admissible=done_admissible,
        actions=tuple(copy.deepcopy(dict(action)) for action in actions),
        advertised_actions=advertised,
        dispatch_admissible_actions=dispatchable,
        tool_schemas=schemas,
        advisory_degraded=advisory_degraded,
    )
    setattr(runtime, "_completion_owner_completion_decision", result)
    return result


def _shared_uncertified_submit_notice(runtime: object) -> str | None:
    from kernel_done_readiness_authority import (
        cached_decision, host_debt_disposition_text,
    )

    decision = cached_decision(runtime)
    if decision is None or getattr(decision, "decision", None) != "ACCEPTED_UNCERTIFIED":
        return None
    disposition = host_debt_disposition_text(decision)
    return (
        f"{TERMINAL_CHOICE_HEADER} Public checks are GREEN on this candidate. "
        f"{disposition} Call done to store this artifact as UNCERTIFIED_SUBMIT; certified success is not claimed."
    )


def _done_submit_next_actions_line(runtime: object,
                                   decision: CompletionRouteDecision | None) -> str:
    """STATEFRAME_DONE_IS_SUBMIT: keep the action menu the refusal panel used to carry.

    In b1431c7 HARM, removing "Next legal action" sent the model to read+read.
    Under done-is-submit nothing is refused, so the line lists what is available now
    (the route's concrete actions, then done) with no refusal wording. Fail-open: any
    error renders no line.
    """
    try:
        route = decision or completion_route_decision(runtime)
        rendered: list[str] = []
        for action in tuple(getattr(route, "actions", ()) or ()):
            tool = str(action.get("tool", "")) if isinstance(action, Mapping) else ""
            if not tool or tool in {"done", "probe"}:
                continue
            arguments = action.get("arguments")
            if tool == "run_tests" and isinstance(arguments, Mapping) and arguments.get("target"):
                item = "run_tests(target=" + json.dumps(str(arguments["target"]), ensure_ascii=False) + ")"
            else:
                item = tool + "()"
            if item not in rendered:
                rendered.append(item)
        rendered.append("done()")
        return "[host] Available now: " + " or ".join(rendered) + ".\n"
    except Exception:  # noqa: BLE001 - presentation only; never block the request
        return ""


def completion_route_notice(runtime: object,
                            decision: CompletionRouteDecision | None = None) -> str:
    """Render route prose only from the cached typed completion decision."""
    from kernel_done_is_submit_authority import enabled as done_submit_enabled
    if done_submit_enabled(runtime):
        return (
            _done_submit_next_actions_line(runtime, decision)
            + "[host] Calling done now captures and submits the current candidate, then ends the run. "
            "Readiness, certification, evidence currency, open items, counterexamples, coverage, "
            "and catalog status are recorded as labels for the grader."
        )
    submit_notice = _shared_uncertified_submit_notice(runtime)
    if submit_notice is not None:
        return submit_notice
    route = decision or completion_route_decision(runtime)
    typed_route = (
        f"Typed completion route: reason_category={route.reason_category}; "
        f"reason_codes={list(route.reason_codes)}; "
        f"menu_actions={list(route.advertised_actions)}. "
    )
    if route.route == "ACTION":
        source = getattr(runtime, "_last_completion_decision", None)
        notice = _next_action_notice(runtime, route.actions, source)
        done_routes = tuple(
            str(action.get("route")) for action in route.actions
            if action.get("tool") == "done" and action.get("route")
        )
        if "UNCERTIFIED_SUBMIT" in done_routes:
            notice += " done remains available as UNCERTIFIED_SUBMIT; open obligations stay unresolved."
        elif "CERTIFIED_ACCEPTANCE" in done_routes:
            notice += " done remains available to submit the certified revision."
        return notice + " " + typed_route
    if route.route == "UNCERTIFIED_SUBMIT":
        open_label = ", ".join(str(row.get("obligation_id", ""))
                                for row in route.obligation_bindings
                                if row.get("obligation_id"))
        return (
            f"{TERMINAL_CHOICE_HEADER} {typed_route}Evidence remains unresolved"
            f"{f' for [{open_label}]' if open_label else ''}. Use done to store this artifact "
            "as UNCERTIFIED_SUBMIT; open obligations are preserved and certified success is not claimed."
        )
    if route.route == "HOST_BLOCKER":
        reason = ", ".join(route.reason_codes) or "no owner-admitted recovery route"
        return (f"{TERMINAL_CHOICE_HEADER} {typed_route}HOST_COMPLETION_BLOCKER: {reason}. "
                "No model action is currently admissible; the artifact remains uncertified for host recovery.")
    return ""


def _route_menu_tools(runtime: object, route: CompletionRouteDecision,
                      tools: Sequence[Mapping[str, object]]) -> list[dict[str, object]] | None:
    if route.route not in {"ACTION", "UNCERTIFIED_SUBMIT", "HOST_BLOCKER"}:
        return None
    if route.route == "HOST_BLOCKER":
        from kernel_terminal_policy import (
            enabled as hard_budget_only_enabled,
            queue_advisory,
        )
        if hard_budget_only_enabled(runtime):
            queue_advisory(runtime, "completion_owner_exhausted")
            # Keep the upstream-admitted surface and append only the existing
            # dispatchable floor; the runner's hard budgets still bound retries.
            selected = _apply_capability_floor_to_owner_menu(runtime, tools, tools)
            setattr(runtime, "_completion_owner_host_stop_pending", not bool(selected))
            names = tuple(_tool_name(tool) for tool in selected)
            runtime.completion_menu = names
            runtime.offered_tools = names
            runtime._question_done_available = "done" in names
            runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
            runtime._completion_owner_actionable = bool(selected)
            runtime._completion_owner_done_selection = None
            if isinstance(tools, list):
                tools[:] = copy.deepcopy(selected)
            _emit(runtime, "completion_owner_typed_host_blocker",
                  reason_codes=list(route.reason_codes), turns_left=_model_turns_remaining(runtime),
                  advisory=True, offered=list(names))
            return selected
        else:
            runtime.system_outcome = "COMPLETION_OWNER_EXHAUSTED"
        runtime.completion_menu = ()
        runtime.offered_tools = ()
        runtime._question_done_available = False
        runtime._completion_owner_dispatch_tools = []
        if isinstance(tools, list):
            tools[:] = []
        _emit(runtime, "completion_owner_typed_host_blocker",
              reason_codes=list(route.reason_codes), turns_left=_model_turns_remaining(runtime))
        return []

    selected = [copy.deepcopy(dict(tool)) for tool in route.tool_schemas]
    names = tuple(_tool_name(tool) for tool in selected)
    if names != route.advertised_actions or names != route.dispatch_admissible_actions:
        _contract_error(runtime, "completion route action set differs from its final tool schemas",
                        request_id=None, boundary="owner",
                        route="completion-route", tool_names=names)
    if route.route == "ACTION" and any(name != "done" for name in names):
        _clear_terminal_offer_state(runtime)
        owner_blocks = getattr(runtime, "_completion_owner_owner_blocks", None)
        if isinstance(owner_blocks, dict):
            owner_blocks.pop("terminal_line", None)
        setattr(runtime, "_completion_owner_done_selection", None)
    if "done" in names and "read" not in names:
        read_tool = _deferred_source_read_tool(runtime, tools)
        if read_tool is not None:
            selected.append(read_tool)
            names = tuple(_tool_name(tool) for tool in selected)
    if names == ("done",):
        selected = _terminal_tools_with_deferred_read(runtime, selected, selected[0])
        names = tuple(_tool_name(tool) for tool in selected)
    runtime.completion_menu = names
    runtime.offered_tools = names
    runtime._question_done_available = "done" in names
    runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
    if isinstance(tools, list):
        tools[:] = copy.deepcopy(selected)
    if route.route == "UNCERTIFIED_SUBMIT":
        _begin_uncertified_submit(runtime)
        state = _state(runtime)
        revision = _digest(_revision(runtime))
        if not state["final_submit_request_pending"]:
            state["done_only_offer_count"] += 1
            state["final_submit_response_count"] = 0
            state["final_submit_used"] = False
        state["terminal_choice_revision"] = revision
        state["terminal_choice_offered"] = True
        state["final_submit_reserved"] = True
        state["final_submit_request_pending"] = True
        line = completion_route_notice(runtime, route)
        record_owner_block(runtime, "terminal_line", line,
                           block_id=f"terminal:uncertified:{route.signature}")
        _mark_owner_done_selection(runtime, "uncertified_evidence_gap_submission")
        setattr(runtime, "_completion_owner_actionable", False)
        _emit(runtime, "completion_owner_uncertified_submit_menu",
              offered=list(names), open_ids=list(_uncertified_open_ids(runtime)))
    else:
        setattr(runtime, "_completion_owner_actionable", True)
        _emit(runtime, "completion_owner_typed_action_menu",
              offered=list(names), reason_category=route.reason_category,
              reason_codes=list(route.reason_codes),
              actions=[copy.deepcopy(dict(action)) for action in route.actions])
    return selected


def _drop_conflicting_terminal_offer(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Remove only a prior host-authored terminal block, never matching content."""
    return [
        dict(message) for message in messages
        if message.get(OWNER_MESSAGE_BLOCK_KEY) is None
    ]


def _uncertified_open_ids(runtime: object) -> tuple[str, ...]:
    values = list(open_ids(runtime))
    if values:
        return tuple(values)
    codes = _decision_codes(runtime, None)
    return codes or ("uncertifiable host coverage",)


def _terminal_open_diagnostics(
        runtime: object, terminal: Mapping[str, object],
) -> tuple[tuple[str, ...], str]:
    """Return open work plus whether an empty result is actually authoritative."""
    explicit = terminal.get("open_ids")
    if isinstance(explicit, (list, tuple)) and not isinstance(explicit, (str, bytes, bytearray)):
        ids = tuple(str(item) for item in explicit if str(item).strip())
        status = str(terminal.get("open_ids_status") or ("known" if ids else "provided_empty"))
        return ids, status
    live = tuple(str(item) for item in open_ids(runtime) if str(item).strip())
    if live:
        setattr(runtime, "_completion_owner_last_open_ids", live)
        return live, "known"
    snapshot = getattr(runtime, "_completion_owner_last_open_ids", ())
    if isinstance(snapshot, (list, tuple)) and snapshot:
        return tuple(str(item) for item in snapshot if str(item).strip()), "last_authoritative"
    if bool(getattr(runtime, "_completion_owner_open_ids_verified_empty", False)):
        return (), "verified_empty"
    return (), "unknown"


def _add_terminal_open_diagnostics(
        runtime: object, terminal: dict[str, object], reason: str,
) -> dict[str, object]:
    ids, status = _terminal_open_diagnostics(runtime, terminal)
    terminal["open_ids_status"] = status
    if ids:
        terminal["open_ids"] = list(ids)
    terminal["completion_owner_reason"] = reason
    return terminal


def _render_completion_route_notice(
        runtime: object, outgoing: list[dict[str, object]],
        route_decision: CompletionRouteDecision,
) -> list[dict[str, object]]:
    outgoing = _drop_conflicting_terminal_offer(outgoing)
    notice = completion_route_notice(runtime, route_decision)
    if route_decision.route == "UNCERTIFIED_SUBMIT":
        record_owner_block(
            runtime, "terminal_line", notice,
            block_id=f"terminal:uncertified:{route_decision.signature}",
        )
        terminal = _owner_block(runtime, "terminal_line") or {}
        marker = str(terminal.get("id", "terminal:uncertified"))
    else:
        marker = f"completion-route:{route_decision.signature}"
    _append_owner_message(outgoing, notice, marker=marker)
    advisory_notice = advisory_degraded_notice(runtime)
    _append_owner_message(
        outgoing, advisory_notice,
        marker=f"advisory-degraded:{route_decision.signature}",
    )
    pending_no_progress = str(getattr(
        runtime, "_completion_owner_pending_no_progress_notice", "") or "")
    if pending_no_progress:
        _append_owner_message(
            outgoing, pending_no_progress,
            marker=f"no-progress-cap:{route_decision.signature}",
        )
        setattr(runtime, "_completion_owner_pending_no_progress_notice", None)
    setattr(runtime, "_completion_owner_actionable", route_decision.route == "ACTION")
    _emit(runtime, "completion_owner_route_decision_rendered",
          route=route_decision.route, reason_category=route_decision.reason_category,
          reason_codes=list(route_decision.reason_codes),
          advertised_actions=list(route_decision.advertised_actions),
          dispatch_admissible_actions=list(route_decision.dispatch_admissible_actions))
    return outgoing


def offer_terminal_choice(runtime: object, prepared: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Offer the existing done/challenge/submission route exactly once per revision."""
    outgoing = [dict(message) for message in prepared]
    if not enabled(runtime):
        return outgoing
    if _product_parent_unchanged(runtime, reason="completion_owner_terminal_choice"):
        outgoing = _drop_conflicting_terminal_offer(outgoing)
        _clear_terminal_offer_state(runtime)
        available = _available_tools(
            runtime, getattr(runtime, "_completion_owner_full_tools", ()))
        setattr(runtime, "_completion_owner_actionable",
                any(_tool_name(tool) != "done" for tool in available))
        return outgoing
    green_open_work = _green_open_work(runtime)
    if _stall_rule_no_progress_edit_cap(runtime) and not _no_host_checks(runtime):
        notice = _stall_rule_cap_notice(runtime)
        setattr(runtime, "_completion_owner_no_progress_edit_cap", True)
        _emit(runtime, "completion_owner_no_progress_edit_cap",
              notice=notice, open_ids=list(open_ids(runtime)))
        if green_open_work:
            # Keep the measured stall notice, then let the typed route publish
            # the repair actions that remain admissible for the open IDs.
            setattr(runtime, "_completion_owner_pending_no_progress_notice", notice)
        else:
            outgoing = _drop_conflicting_terminal_offer(outgoing)
            _begin_uncertified_submit(runtime)
            _notify_v46_attempt_ceiling(runtime, {
                "result": "UNCERTIFIED_SUBMIT", "accepted": False, "certified": False,
                "steps": getattr(getattr(runtime, "owner", None), "current_step", 0),
            })
            _append_owner_message(
                outgoing, notice, marker="no-progress-cap:terminal",
            )
            setattr(runtime, "_completion_owner_actionable", False)
            return outgoing
    if green_open_work:
        route_decision = completion_route_decision(
            runtime, getattr(runtime, "_last_completion_decision", None),
            tools=getattr(runtime, "_completion_owner_full_tools", ()),
        )
        if route_decision.route in {"ACTION", "UNCERTIFIED_SUBMIT", "HOST_BLOCKER"}:
            outgoing = _render_completion_route_notice(runtime, outgoing, route_decision)
            from kernel_capability_floor import enabled as capability_floor_enabled
            if (capability_floor_enabled(runtime)
                    and _shared_uncertified_submit_notice(runtime) is None):
                notice = stall_open_work_notice(runtime)
                if notice and not any(message.get("content") == notice for message in outgoing):
                    outgoing.append({"role": "user", "content": notice})
                    episode = getattr(runtime, "_stall_open_work_episode", None)
                    if isinstance(episode, dict) and not episode.get("notice_rendered"):
                        episode["notice_rendered"] = True
                        _emit(runtime, "completion_owner_stall_rule_notice",
                              offered=[str(action.get("tool", ""))
                                       for action in stall_open_work_actions(runtime)])
            return outgoing
    witness_receipt_closes_clause = _current_witness_receipt_closes_clause(runtime)
    stall_actions = () if (witness_receipt_closes_clause or _no_host_checks(runtime)) else stall_open_work_actions(runtime)
    if stall_actions:
        outgoing = _drop_conflicting_terminal_offer(outgoing)
        notice = stall_open_work_notice(runtime)
        if notice and not any(message.get("content") == notice for message in outgoing):
            outgoing.append({"role": "user", "content": notice})
        setattr(runtime, "_completion_owner_actionable", True)
        episode = getattr(runtime, "_stall_open_work_episode", None)
        if isinstance(episode, dict) and not episode.get("notice_rendered"):
            episode["notice_rendered"] = True
            _emit(runtime, "completion_owner_stall_rule_notice",
                  offered=[str(action.get("tool", "")) for action in stall_actions])
        return outgoing
    declared_repairs = () if (witness_receipt_closes_clause or _no_host_checks(runtime)) else _declared_repair_actions(
        runtime, getattr(runtime, "_completion_owner_full_tools", ()))
    if declared_repairs:
        outgoing = _drop_conflicting_terminal_offer(outgoing)
        state = _state(runtime)
        state["terminal_choice_offered"] = False
        state["final_submit_reserved"] = False
        state["final_submit_request_pending"] = False
        notice = _declared_repair_notice(runtime, declared_repairs)
        if not any(message.get("content") == notice for message in outgoing):
            outgoing.append({"role": "user", "content": notice})
        setattr(runtime, "_completion_owner_actionable", True)
        _emit(runtime, "completion_owner_declared_repair_menu",
              offered=[str(action.get("tool", "")) for action in declared_repairs],
              clause_ids=list((_declared_repair_state(runtime) or {}).get("clause_ids", ())))
        return outgoing
    review_actions = () if witness_receipt_closes_clause else _witness_evidence_actions(
        runtime, getattr(runtime, "_completion_owner_full_tools", ()))
    if review_actions:
        outgoing = _drop_conflicting_terminal_offer(outgoing)
        state = _state(runtime)
        state["terminal_choice_offered"] = False
        state["final_submit_reserved"] = False
        state["final_submit_request_pending"] = False
        notice = _next_action_notice(runtime, review_actions, None)
        if not any(message.get("content") == notice for message in outgoing):
            outgoing.append({"role": "user", "content": notice})
        setattr(runtime, "_completion_owner_actionable", True)
        _emit(runtime, "completion_owner_evidence_witness_menu",
              offered=[str(action.get("tool", "")) for action in review_actions])
        return outgoing
    witness_limitation = _witness_limitation(
        runtime, getattr(runtime, "_completion_owner_full_tools", ()))
    if witness_limitation:
        # Preserve the disclosure across later menu projections.  An unavailable
        # rail must not disappear merely because an edit refreshed the public
        # clause projection before the final submit turn.
        setattr(runtime, "_completion_owner_witness_limitation_notice", witness_limitation)
    else:
        prior_limitation = getattr(runtime, "_completion_owner_witness_limitation_notice", None)
        if isinstance(prior_limitation, str) and prior_limitation:
            witness_limitation = prior_limitation
    governor = _common_sense_decision(runtime)
    if governor is not None:
        if bool(getattr(governor, "release_uncertified", False)):
            setattr(runtime, "_common_sense_governor_release", True)
            if bool(getattr(governor, "advisory", False)):
                setattr(runtime, "_common_sense_governor_advisory", governor.control_line)
        else:
            outgoing = _drop_conflicting_terminal_offer(outgoing)
            state = _state(runtime)
            state["terminal_choice_offered"] = False
            state["final_submit_reserved"] = False
            state["final_submit_request_pending"] = False
            setattr(runtime, "_completion_owner_actionable", bool(_common_sense_actions(governor)))
            notice = str(getattr(governor, "control_line", "") or "")
            if notice and not any(message.get("content") == notice for message in outgoing):
                outgoing.append({"role": "user", "content": notice})
            dependency_key = str(getattr(governor, "dependency_key", ""))
            if getattr(runtime, "_common_sense_control_dependency_key", None) != dependency_key:
                setattr(runtime, "_common_sense_control_dependency_key", dependency_key)
                _emit(
                    runtime,
                    "completion_owner_common_sense_control",
                    rule_ids=list(getattr(governor, "rule_ids", ())),
                    actions=list(_common_sense_actions(governor)),
                    unmet_scope=list(getattr(governor, "unmet_scope", ())),
                )
            return outgoing
    if governor is None and not _common_sense_enabled(runtime):
        premature = _premature_frame(runtime)
        if premature is not None:
            outgoing = _drop_conflicting_terminal_offer(outgoing)
            state = _state(runtime)
            state["terminal_choice_offered"] = False
            state["final_submit_reserved"] = False
            state["final_submit_request_pending"] = False
            notice = str(premature.get("notice", "") or "")
            if notice and not any(message.get("content") == notice for message in outgoing):
                outgoing.append({"role": "user", "content": notice})
            setattr(runtime, "_completion_owner_actionable", True)
            _emit(runtime, "completion_owner_premature_rule_frame",
                  menu=list(premature.get("menu", ())),
                  clause_ids=list(premature.get("clause_ids", ())))
            return outgoing
    route_decision = completion_route_decision(
        runtime, getattr(runtime, "_last_completion_decision", None),
        tools=getattr(runtime, "_completion_owner_full_tools", ()),
    )
    if route_decision.route in {"ACTION", "UNCERTIFIED_SUBMIT", "HOST_BLOCKER"}:
        outgoing = _drop_conflicting_terminal_offer(outgoing)
        notice = completion_route_notice(runtime, route_decision)
        if route_decision.route == "UNCERTIFIED_SUBMIT":
            record_owner_block(
                runtime, "terminal_line", notice,
                block_id=f"terminal:uncertified:{route_decision.signature}",
            )
            terminal = _owner_block(runtime, "terminal_line") or {}
            marker = str(terminal.get("id", "terminal:uncertified"))
        else:
            marker = f"completion-route:{route_decision.signature}"
        if notice and not any(message.get("content") == notice for message in outgoing):
            outgoing.append({"role": "user", "content": notice, OWNER_MESSAGE_BLOCK_KEY: marker})
        advisory_notice = advisory_degraded_notice(runtime)
        if advisory_notice and not any(
                message.get("content") == advisory_notice for message in outgoing):
            outgoing.append({
                "role": "user", "content": advisory_notice,
                OWNER_MESSAGE_BLOCK_KEY: f"advisory-degraded:{route_decision.signature}",
            })
        setattr(runtime, "_completion_owner_actionable", route_decision.route == "ACTION")
        _emit(runtime, "completion_owner_route_decision_rendered",
              route=route_decision.route, reason_category=route_decision.reason_category,
              reason_codes=list(route_decision.reason_codes),
              advertised_actions=list(route_decision.advertised_actions),
              dispatch_admissible_actions=list(route_decision.dispatch_admissible_actions))
        return outgoing
    if getattr(runtime, "_completion_owner_uncertified_submit", False):
        # The host has already selected the one-shot uncertified terminal path;
        # do not render another refusal that could invite a further done turn.
        setattr(runtime, "_completion_owner_actionable", False)
        return outgoing
    candidate, candidate_revision, candidate_ids = _candidate(runtime)
    green_terminal_ready = bool(candidate)
    actions = () if green_terminal_ready else legal_next_actions(runtime)
    if actions:
        outgoing = _drop_conflicting_terminal_offer(outgoing)
        state = _state(runtime)
        state["terminal_choice_offered"] = False
        state["final_submit_reserved"] = False
        state["final_submit_request_pending"] = False
        notice = _next_action_notice(
            runtime, actions, getattr(runtime, "_last_completion_decision", None))
        if not any(message.get("content") == notice for message in outgoing):
            outgoing.append({"role": "user", "content": notice})
        setattr(runtime, "_completion_owner_actionable", True)
        _emit(runtime, "completion_owner_next_action_rendered",
              actions=[dict(action) for action in actions], reason_codes=list(_decision_codes(runtime, None)))
        return outgoing
    if _decision_codes(runtime, None) and not actions:
        outgoing = _drop_conflicting_terminal_offer(outgoing)
        state = _state(runtime)
        state["terminal_choice_offered"] = False
        state["final_submit_reserved"] = False
        state["final_submit_request_pending"] = False
        setattr(runtime, "_completion_owner_actionable", False)
        _begin_uncertified_submit(runtime)
        notice = _host_only_notice(runtime, getattr(runtime, "_last_completion_decision", None))
        if not any(message.get("content") == notice for message in outgoing):
            outgoing.append({"role": "user", "content": notice})
        _emit(runtime, "completion_owner_host_gate_held",
                  reason_codes=list(_decision_codes(runtime, None)))
        return outgoing
    if _cutoff_rule_offer_applies(runtime):
        # RF02 changes only the already-owned tool menu.  Do not append the
        # legacy done-only sentence when the menu owner will expose work.
        outgoing = _drop_conflicting_terminal_offer(outgoing)
        state = _state(runtime)
        state["terminal_choice_offered"] = False
        state["final_submit_reserved"] = False
        state["final_submit_request_pending"] = False
        setattr(runtime, "_completion_owner_actionable", True)
        return outgoing
    setattr(runtime, "_completion_owner_actionable", False)
    candidate, revision, ids = candidate, candidate_revision, candidate_ids
    if not candidate:
        return outgoing
    state = _state(runtime)
    if state["terminal_choice_revision"] != revision:
        state["terminal_choice_revision"] = revision
        state["terminal_choice_offered"] = False
        state["final_submit_reserved"] = False
        state["final_submit_used"] = False
        state["final_submit_response_count"] = 0
        state["restatement_pending"] = False
        state["final_submit_request_pending"] = False
    new_terminal_request = False
    if not state["terminal_choice_offered"]:
        state["terminal_choice_offered"] = True
        state["final_submit_reserved"] = True
        new_terminal_request = True
        _emit(runtime, "completion_owner_terminal_choice_offered",
              revision=revision, open_ids=list(ids),
              reserve=f"{MAX_RESERVED_RESPONSES}_final_submit_responses")
    if state["restatement_pending"]:
        _emit(runtime, "completion_owner_terminal_choice_restatement",
              revision=revision, response_count=state["final_submit_response_count"])
        state["restatement_pending"] = False
        new_terminal_request = True
    if new_terminal_request:
        continuation_limits = getattr(runtime, "continuation_limits", None)
        if continuation_limits is None:
            continuation_limits = getattr(runtime, "decision_budget", None)
        reserve_terminal = getattr(continuation_limits, "reserve_terminal", None)
        remaining_decisions = getattr(continuation_limits, "model_decisions_remaining", None)
        reserve_needed = not (
            (continuation_limits is not None and getattr(continuation_limits, "limit", 0) is None)
            or (type(remaining_decisions) is int and remaining_decisions > 0))
        if callable(reserve_terminal) and reserve_needed and not reserve_terminal():
            state["final_submit_reserved"] = False
            state["final_submit_request_pending"] = False
            setattr(runtime, "_completion_owner_actionable", False)
            _emit(runtime, "completion_owner_finalization_deferred",
                  reason="continuation_terminal_reserve_exhausted")
            return outgoing
        # Charging a reserved response is not proof that a provider request was
        # submitted. Keep this handoff durable until RequestCapture emits the
        # real request_submitted event.
        state["final_submit_request_pending"] = True
        # The live request can cross the menu seam more than once before it is
        # sent. Green framing counts that provider request once, not once per
        # projection.
        setattr(runtime, "_completion_owner_green_frame_counted", False)
    open_label = ", ".join(ids) if ids else "uncertifiable host coverage"
    advisory = getattr(runtime, "_common_sense_governor_advisory", None)
    if isinstance(advisory, str) and advisory and not any(
            message.get("content") == advisory for message in outgoing):
        outgoing.append({"role": "user", "content": advisory})
        setattr(runtime, "_common_sense_governor_advisory", None)
    if witness_limitation:
        _emit(runtime, "completion_owner_witness_limitation_disclosed",
              open_ids=list(_reqwitness_open_ids(runtime)), notice=witness_limitation)
    preserved_tools = (
        _expanded_zero_edit_submit_menu(
            runtime, getattr(runtime, "_completion_owner_full_tools", ()))
        if not _no_host_checks(runtime) else None
    )
    preserved_actions = ([_tool_name(tool) for tool in preserved_tools if _tool_name(tool) != "done"]
                         if preserved_tools else [])
    if preserved_actions:
        text = ("Final step: call done now to submit this revision when ready; "
                f"{', '.join(preserved_actions)} remain available for unresolved work "
                f"(uncertified; {'witness limitation disclosed; ' if witness_limitation else ''}"
                f"open: [{open_label}]).")
    else:
        text = ("Final step: call done now to submit this revision "
                f"(no public host checks exist; acceptance discloses that limitation; "
                f"{'witness limitation disclosed; ' if witness_limitation else ''}"
                f"open: [{open_label}])."
                if _no_host_checks(runtime) else
                "Final step: call done now to submit this revision "
                f"(uncertified; {'witness limitation disclosed; ' if witness_limitation else ''}"
                f"open: [{open_label}]). No other action is available.")
    shared_submit_notice = _shared_uncertified_submit_notice(runtime)
    if shared_submit_notice is not None:
        text = shared_submit_notice
    terminal_record = record_owner_block(
        runtime,
        "terminal_line",
        text,
        block_id=f"terminal:{revision}:{state['done_only_offer_count']}",
    )
    terminal_block_id = str(terminal_record["id"])
    if _no_host_checks(runtime) and not any(
            message.get(OWNER_MESSAGE_BLOCK_KEY) == terminal_block_id + ":no-host"
            for message in outgoing):
        outgoing.append({
            "role": "user", "content": NO_HOST_CHECKS_TERMINAL_LINE,
            OWNER_MESSAGE_BLOCK_KEY: terminal_block_id + ":no-host",
        })
    elif green_terminal_ready and shared_submit_notice is None and not any(
            message.get(OWNER_MESSAGE_BLOCK_KEY) == terminal_block_id + ":green"
            for message in outgoing):
        outgoing.append({
            "role": "user", "content": GREEN_TERMINAL_LINE,
            OWNER_MESSAGE_BLOCK_KEY: terminal_block_id + ":green",
        })
    if witness_limitation and not any(
            message.get(OWNER_MESSAGE_BLOCK_KEY) == terminal_block_id + ":witness"
            for message in outgoing):
        outgoing.append({
            "role": "user", "content": witness_limitation,
            OWNER_MESSAGE_BLOCK_KEY: terminal_block_id + ":witness",
        })
    if not any(message.get(OWNER_MESSAGE_BLOCK_KEY) == terminal_block_id
               for message in outgoing):
        outgoing.append({
            "role": "user", "content": text,
            OWNER_MESSAGE_BLOCK_KEY: terminal_block_id,
        })
    return outgoing


def before_provider_request(runtime: object, *, provider_request: bool = True) -> bool:
    """Run boundary feeders, then enforce the owner's reserved-response cap.

    Readiness and post-green budget modules may request the shared submitter;
    this owner remains the only terminal commitment/menu authority.
    """
    if not enabled(runtime):
        return True
    if provider_request and not done_accepted(runtime):
        from kernel_submit_at_readiness import at_request_boundary as readiness_boundary
        if readiness_boundary(runtime):
            return False
        from kernel_postgreen_budget import at_request_boundary as postgreen_boundary
        if postgreen_boundary(runtime):
            return False
    state = _state(runtime)
    from kernel_capability_floor import enabled as capability_floor_enabled

    # Retain the reserved final-response ceiling, but let the episode's hard
    # response/tool/time budgets govern repeated done-only offers under floor.
    if (((state["final_submit_reserved"]
          and state["final_submit_response_count"] >= MAX_RESERVED_RESPONSES)
         or (not capability_floor_enabled(runtime)
             and state["done_only_offer_count"] >= MAX_RESERVED_RESPONSES))
         and not state["final_submit_request_pending"]):
        accepted = bool(done_accepted(runtime))
        if not accepted:
            from kernel_terminal_policy import (
                enabled as hard_budget_only_enabled,
                queue_advisory,
            )
            if hard_budget_only_enabled(runtime):
                queue_advisory(runtime, "owner_reserved_submit_cap")
                _emit(runtime, "completion_owner_exhausted",
                      reason="reserved_final_submit_not_accepted", advisory=True)
                return True
            runtime.system_outcome = FINAL_SUBMIT_EXHAUSTED
            _emit(runtime, "completion_owner_exhausted", reason="reserved_final_submit_not_accepted")
        return False
    return True


def request_pending(runtime: object) -> bool:
    """Return whether the owner has prepared a reserved request not yet submitted."""
    if runtime is None:
        return False
    if not enabled(runtime):
        return False
    state = _state(runtime)
    return bool(state["final_submit_reserved"] and state["final_submit_request_pending"])


def reconcile_submitted_request(runtime: object) -> bool:
    """Close a reserved handoff when its durable wire event survived a crash."""
    if not request_pending(runtime):
        return False
    expected = getattr(runtime, "_completion_owner_pending_request_id", None)
    journal = getattr(runtime, "journal", None)
    source = getattr(journal, "records", None)
    try:
        records = source() if callable(source) else source
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        records = ()
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        return False
    for event in reversed(records):
        if not isinstance(event, Mapping) or event.get("kind") != "request_submitted":
            continue
        request_id = event.get("request_id")
        if expected is not None and request_id is not None and str(request_id) != str(expected):
            continue
        note_request_submitted(
            runtime,
            request_id=str(request_id) if request_id is not None else None,
            step=event.get("step") if isinstance(event.get("step"), int) else None,
        )
        _emit(runtime, "completion_owner_reserved_response_reconciled",
              request_id=request_id, step=event.get("step"),
              source="durable_request_submitted")
        return True
    return False


def _contract_tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    return str(function.get("name", "")) if isinstance(function, Mapping) else ""


def _contract_tools(tools: object) -> list[dict[str, object]]:
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return []
    return [copy.deepcopy(dict(tool)) for tool in tools if isinstance(tool, Mapping)]


@dataclass(frozen=True)
class ToolDispatch:
    """The validated, discriminated action handed to the legacy execution rail."""

    name: str
    arguments: dict[str, object]
    variant: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    """One source of truth for a tool's advertised contract and dispatch."""

    name: str
    schema: dict[str, object]
    validator: Callable[[object], tuple[str, ...]]
    dispatch: Callable[[object], ToolDispatch]
    valid_example: dict[str, object]
    invalid_example: dict[str, object]

    def advertised(self, description: str = "") -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description or f"{self.name} action",
                "parameters": copy.deepcopy(self.schema),
            },
        }


def _json_type_matches(value: object, expected: object) -> bool:
    if isinstance(expected, list):
        return any(_json_type_matches(value, item) for item in expected)
    return {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)


def validate_json_schema(
        schema: Mapping[str, object], value: object, path: str = "$") -> tuple[str, ...]:
    """Validate the JSON-Schema subset used by provider tools, including ``not``."""
    errors: list[str] = []
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path} must equal {schema['const']!r}")
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path} must be one of {json.dumps(enum, sort_keys=True)}")
    if "type" in schema and not _json_type_matches(value, schema["type"]):
        errors.append(f"{path} must be {schema['type']}")
        return tuple(errors)
    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            errors.append(f"{path} is shorter than minLength {minimum}")
        if isinstance(maximum, int) and len(value) > maximum:
            errors.append(f"{path} is longer than maxLength {maximum}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                matched = re.fullmatch(pattern, value) is not None
            except re.error:
                matched = False
            if not matched:
                errors.append(f"{path} does not match its required pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path} is below minimum {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path} is above maximum {maximum}")
        if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
            errors.append(f"{path} must be greater than {exclusive_minimum}")
    if isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            errors.append(f"{path} has fewer than {minimum} items")
        if isinstance(maximum, int) and len(value) > maximum:
            errors.append(f"{path} has more than {maximum} items")
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                errors.extend(validate_json_schema(items, item, f"{path}[{index}]"))
    if isinstance(value, Mapping):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        required = schema.get("required")
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes, bytearray)):
            errors.extend(
                f"{path}.{name} is required"
                for name in required if name not in value
            )
        additional = schema.get("additionalProperties", True)
        for name, item in value.items():
            child = properties.get(name)
            if isinstance(child, Mapping):
                errors.extend(validate_json_schema(child, item, f"{path}.{name}"))
            elif child is None and additional is False:
                errors.append(f"{path}.{name} is not allowed")
            elif child is None and isinstance(additional, Mapping):
                errors.extend(validate_json_schema(additional, item, f"{path}.{name}"))
    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword)
        if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes, bytearray)):
            continue
        branch_errors = [
            validate_json_schema(branch, value, path)
            for branch in branches if isinstance(branch, Mapping)
        ]
        matches = sum(not branch_error for branch_error in branch_errors)
        if keyword == "allOf":
            for branch_error in branch_errors:
                errors.extend(branch_error)
        elif keyword == "anyOf" and matches == 0:
            errors.append(f"{path} does not match any allowed schema variant")
        elif keyword == "oneOf" and matches != 1:
            errors.append(f"{path} must match exactly one schema variant (matched {matches})")
    negative = schema.get("not")
    if isinstance(negative, Mapping) and not validate_json_schema(negative, value, path):
        errors.append(f"{path} must not match the excluded schema")
    return tuple(errors)


def _object_schema(
        properties: Mapping[str, object], required: Sequence[str] = (), *,
        additional: object = True, one_of: Sequence[Mapping[str, object]] = ()) -> dict[str, object]:
    result: dict[str, object] = {
        "type": "object",
        "properties": copy.deepcopy(dict(properties)),
        "required": sorted({str(item) for item in required}),
        "additionalProperties": additional,
    }
    if one_of:
        result["oneOf"] = [copy.deepcopy(dict(branch)) for branch in one_of]
    return result


def _probe_branches(names: Sequence[str]) -> list[dict[str, object]]:
    branches: dict[str, dict[str, object]] = {
        "python": {
            "required": ["code"],
            "not": {"required": ["obligation"]},
        },
        "legacy": {
            "required": ["obligation", "code"],
            "not": {"required": ["mode"]},
        },
        "behavior_witness": {
            "required": ["clause_id", "entrypoint", "input", "expected", "rationale"],
            "not": {"required": ["code"]},
        },
        "clause_discriminator_propose": {
            "properties": {"mode": {"const": "clause_discriminator_propose"}},
            "required": ["mode", "clause_ref", "entrypoint", "input", "expected_relation"],
        },
        "clause_review_propose": {
            "properties": {"mode": {"const": "clause_review_propose"}},
            "required": ["mode", "episode_id", "packet"],
        },
        "clause_review_assess": {
            "properties": {"mode": {"const": "clause_review_assess"}},
            "required": ["mode", "episode_id", "assessment"],
        },
        "dependency_fact": {
            "properties": {"mode": {"const": "dependency_fact"}},
            "required": ["mode", "package", "target"],
        },
    }
    return [branches[name] for name in names if name in branches]


_PROBE_MODES = (
    "python", "legacy", "behavior_witness", "clause_discriminator_propose",
    "clause_review_propose", "clause_review_assess", "dependency_fact",
)


def _probe_schema(modes: Sequence[str] = _PROBE_MODES) -> dict[str, object]:
    properties: dict[str, object] = {
        "mode": {"type": "string", "enum": sorted(set(modes))},
        "obligation": {"type": "string"},
        "code": {"type": "string"},
        "question": {"type": "string"},
        "clause_id": {"type": "string"},
        "clause_ref": {"type": "string"},
        "source_unit_handle": {"type": "string", "minLength": 1},
        "source_revision": {"type": "string"},
        "check_identity": {"type": "string"},
        "oracle_identity": {"type": "string"},
        "entrypoint": {"type": "string"},
        "input": {"type": "object"},
        "expected": {"type": ["object", "string"]},
        "expected_relation": {"type": "object"},
        "rationale": {"type": "string"},
        "episode_id": {"type": "string"},
        "schema": {"type": "string"},
        "clause_ids": {"type": "array", "items": {"type": "string"}},
        "clauses": {"type": "array"},
        "packet": {"type": "object"},
        "assessment": {"type": "object"},
        "assembled_result": {"type": "object"},
        "receipt_ids": {"type": "array", "items": {"type": "string"}},
        "max_cases": {"type": "integer"},
        "max_rounds": {"type": "integer"},
        "token_contract": {"type": "object"},
        "token_count": {"type": "integer", "minimum": 0},
        "args": {"type": "array"},
        "kwargs": {"type": "object"},
        "projection": {"type": "string", "enum": ["body"]},
        "package": {"type": "string"},
        "module": {"type": "string"},
        "target": {"type": "string"},
        "call": {"type": "boolean"},
        "why": {"type": "string"},
    }
    return _object_schema(properties, one_of=_probe_branches(modes), additional=False)


def _property_map(names: Sequence[str], *, value: Mapping[str, object] | None = None) -> dict[str, object]:
    template = dict(value or {"type": "string"})
    return {name: copy.deepcopy(template) for name in names}


def _registry_schemas() -> tuple[tuple[str, dict[str, object]], ...]:
    def string(*names: str) -> dict[str, object]:
        return _property_map(names)

    def integer(*names: str) -> dict[str, object]:
        return _property_map(names, value={"type": "integer"})

    def boolean(*names: str) -> dict[str, object]:
        return _property_map(names, value={"type": "boolean"})

    schemas: list[tuple[str, dict[str, object]]] = []

    def inspect_parameters() -> dict[str, object]:
        try:
            # Kernel-owned copy first: releases ship without tools.* (10-05 L108 import crash).
            from kernel_nav_contract import inspect_schema
        except ImportError as exc:
            raise RequestActionContractError(
                "canonical inspect contract is unavailable"
            ) from exc
        return inspect_schema()

    def add(name: str, properties: Mapping[str, object], required: Sequence[str] = (), **kwargs: object) -> None:
        values = dict(properties)
        values.setdefault("why", {"type": "string"})
        schemas.append((name, _object_schema(values, required, **kwargs)))

    schemas.append(("inspect", inspect_parameters()))

    add("grep", {
        **string("pattern", "path", "file_glob", "glob", "cursor"),
        "mode": {"type": "string", "enum": ["regex", "literal", "path"]},
        **integer("page_size"),
    }, ("pattern",))
    add("list_files", {**string("path"), **integer("depth", "limit")})
    add("read", {**string("path", "cursor"), **integer("line", "column", "max_lines", "max_chars", "page_size")}, ("path",))
    add("edit", {**string("path", "function_source", "target_ref")}, ("path", "function_source"))
    add("str_replace", {**string("path", "old_str", "new_str"), "edits": {"type": "array"}}, ("path", "old_str", "new_str"))
    add("file_create", {**string("path", "content")}, ("path", "content"))
    add("run_tests", {**string("target")})
    add("reproduce", {
        **string("source", "test_name", "rationale"),
    }, ("source",))
    add("read_check_receipt", {**string("handle"), "stream": {"type": "string", "enum": ["combined", "stdout", "stderr", "assertion_context"]}, **integer("offset", "limit")}, ("handle",))
    add("done", {
        **string("summary"),
        "no_change_paths": {
            "type": "array", "maxItems": 128,
            "items": _object_schema({
                "path": {"type": "string", "minLength": 1, "maxLength": 512},
                "reason": {"type": "string", "minLength": 1, "maxLength": 400},
            }, ("path", "reason"), additional=False),
        },
        "claim": {"type": "object"},
        "evidence_refs": {"type": "array"},
    })
    add("file_list", {**string("path"), **integer("limit")})
    add("dependency_fact", {**string("package", "module", "target", "question"), "args": {"type": "array"}, "kwargs": {"type": "object"}, **boolean("call")}, ("package", "target"))
    add("where_defined", {**string("symbol")}, ("symbol",))
    add("who_calls", {**string("symbol"), **boolean("transitive")}, ("symbol",))
    add("references", {**string("symbol")}, ("symbol",))
    add("work", {"ops": {"type": "array"}}, ("ops",))
    add("decide", {**string("approach", "decisive_factor", "disqualifier")}, ("approach", "decisive_factor", "disqualifier"))
    add("recall_generated_artifact", {**string("reference")}, ("reference",))
    add("continuation_detour", {**string("path", "reason", "pending_decision"), **integer("start_line", "end_line")}, ("path", "start_line", "end_line", "reason", "pending_decision"))
    add("launch_job", {**string("cmd", "done_signal"), "expected_s": {"type": "number", "exclusiveMinimum": 0}}, ("cmd", "expected_s", "done_signal"))
    add("job_status", {**string("job_id")}, ("job_id",))
    add("job_output", {**string("job_id"), **integer("tail_lines")}, ("job_id",))
    add("job_kill", {**string("job_id")}, ("job_id",))
    add("read_ticket", {**string("ticket", "query")})
    add("dispute", {**string("check_id", "ground", "claim"), "evidence": {"type": ["object", "array", "string"]}}, ("check_id", "ground", "claim", "evidence"))
    add("dispute_check", {**string("check_id", "basis", "claim")}, ("check_id", "basis", "claim"))
    add("predict", {**string("path", "question")})
    add("registry_lookup", {**string("kind", "query")}, ("kind", "query"))
    add("replace_module", {**string("path", "expected_revision", "source", "candidate_id")}, ("path", "expected_revision"), one_of=(
        {"required": ["path", "expected_revision", "source"], "not": {"required": ["candidate_id"]}},
        {"required": ["path", "expected_revision", "candidate_id"], "not": {"required": ["source"]}},
    ))
    try:
        from kernel_native_tools import restore_candidate_schema
    except ImportError as exc:
        raise RequestActionContractError("canonical restore-candidate contract is unavailable") from exc
    schemas.append(("restore_candidate", restore_candidate_schema()))
    schemas.append(("probe", _probe_schema()))
    try:
        from kernel_authored_check import authored_check_schema
    except ImportError as exc:
        raise RequestActionContractError(
            "canonical authored-check contract is unavailable"
        ) from exc
    schemas.append(("authored_check", authored_check_schema()))
    return tuple(schemas)


def _schema_example(schema: Mapping[str, object]) -> object:
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return copy.deepcopy(enum[0])
    branches = schema.get("oneOf")
    if isinstance(branches, Sequence) and not isinstance(branches, (str, bytes, bytearray)):
        branch = next((item for item in branches if isinstance(item, Mapping)), None)
        if isinstance(branch, Mapping):
            return _schema_example(branch)
    expected = schema.get("type")
    if isinstance(expected, list):
        expected = next((item for item in expected if item != "null"), "null")
    if expected == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        names = required if isinstance(required, Sequence) and not isinstance(required, (str, bytes, bytearray)) else ()
        return {
            str(name): _schema_example(properties[name])
            for name in names
            if isinstance(properties, Mapping) and isinstance(properties.get(name), Mapping)
        }
    if expected is None and isinstance(schema.get("required"), Sequence):
        properties = schema.get("properties")
        return {
            str(name): _schema_example(properties[name])
            if isinstance(properties, Mapping) and isinstance(properties.get(name), Mapping)
            else "example"
            for name in schema["required"]
        }
    if expected == "array":
        items = schema.get("items")
        return [_schema_example(items)] if isinstance(items, Mapping) and schema.get("minItems", 0) else []
    if expected == "integer":
        return 1
    if expected == "number":
        return 1.0
    if expected == "boolean":
        return True
    if expected == "null":
        return None
    return "example"


def _invalid_example(schema: Mapping[str, object]) -> dict[str, object]:
    valid = _schema_example(schema)
    result = dict(valid) if isinstance(valid, Mapping) else {}
    branches = schema.get("oneOf")
    if isinstance(branches, Sequence) and not isinstance(branches, (str, bytes, bytearray)):
        first = next((item for item in branches if isinstance(item, Mapping)), None)
        if isinstance(first, Mapping):
            required = first.get("required")
            if isinstance(required, Sequence) and required:
                result.pop(str(required[0]), None)
                return result
    required = schema.get("required")
    if isinstance(required, Sequence) and required:
        result.pop(str(required[0]), None)
        return result
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for name, child in properties.items():
            if name == "why" or not isinstance(child, Mapping):
                continue
            result[str(name)] = {"invalid": True}
            return result
    if schema.get("additionalProperties") is False:
        return {"__unexpected__": True}
    return {"invalid": True}


def _probe_variant(arguments: Mapping[str, object]) -> str:
    mode = arguments.get("mode")
    if isinstance(mode, str) and mode:
        return mode
    if "clause_id" in arguments and "entrypoint" in arguments:
        return "behavior_witness"
    if "obligation" in arguments:
        return "legacy"
    return "python"


def _restore_variant(arguments: Mapping[str, object]) -> str:
    del arguments
    return "restore_candidate"


def _make_tool_spec(
        name: str, schema: Mapping[str, object], *,
        variant_selector: Callable[[Mapping[str, object]], str | None] | None = None,
) -> ToolSpec:
    frozen_schema = copy.deepcopy(dict(schema))

    def canonical_arguments(arguments: object) -> object:
        if name != "inspect":
            return arguments
        try:
            from kernel_nav_contract import (
                normalize_inspect_arguments,
                validate_inspect_semantics,
            )
            if not isinstance(arguments, Mapping):
                return arguments
            normalized = normalize_inspect_arguments(arguments)
            validate_inspect_semantics(normalized)
            return normalized
        except (ImportError, TypeError, ValueError) as exc:
            raise RequestActionContractError(
                f"tool '{name}' arguments failed canonical normalization: {type(exc).__name__}"
            ) from exc

    def validator(arguments: object) -> tuple[str, ...]:
        try:
            normalized = canonical_arguments(arguments)
        except RequestActionContractError as exc:
            return (str(exc),)
        return validate_json_schema(frozen_schema, normalized)

    def dispatch(arguments: object) -> ToolDispatch:
        errors = validator(arguments)
        if errors:
            detail = "; ".join(errors[:6])
            raise RequestActionContractError(
                f"tool '{name}' arguments rejected before dispatch: {detail}; "
                "return a JSON object matching the advertised schema")
        normalized = canonical_arguments(arguments)
        if not isinstance(normalized, Mapping):
            raise RequestActionContractError(
                f"tool '{name}' arguments rejected before dispatch: arguments must be an object")
        selector = variant_selector or (_restore_variant if name == "restore_candidate" else None)
        variant = selector(normalized) if selector is not None else None
        return ToolDispatch(name=name, arguments=dict(normalized), variant=variant)

    if name == "inspect":
        # The compact provider branches intentionally keep field schemas at
        # the root.  Supply examples explicitly so contract diagnostics do
        # not mistake a branch's tag-only projection for a runnable request.
        valid_example: Mapping[str, object] = {
            "op": "orient", "phase": "localize", "scope": "public",
        }
        invalid_example: Mapping[str, object] = {"op": "orient", "phase": "localize"}
    else:
        valid_example = (dict(_schema_example(frozen_schema))
                         if isinstance(_schema_example(frozen_schema), Mapping) else {})
        invalid_example = _invalid_example(frozen_schema)
    return ToolSpec(
        name=name,
        schema=frozen_schema,
        validator=validator,
        dispatch=dispatch,
        valid_example=dict(valid_example),
        invalid_example=dict(invalid_example),
    )


TOOL_SPECS: dict[str, ToolSpec] = {
    name: _make_tool_spec(name, schema, variant_selector=_probe_variant if name == "probe" else None)
    for name, schema in _registry_schemas()
}


def _merge_provider_schema(
        name: str, base: Mapping[str, object], provider: Mapping[str, object]) -> dict[str, object]:
    if name == "probe" and isinstance(provider.get("oneOf"), Sequence):
        return copy.deepcopy(dict(provider))
    result = copy.deepcopy(dict(base))
    base_properties = result.get("properties")
    provider_properties = provider.get("properties")
    if isinstance(base_properties, Mapping) and isinstance(provider_properties, Mapping):
        merged_properties = {**dict(base_properties), **copy.deepcopy(dict(provider_properties))}
        if provider.get("additionalProperties") is False and "why" not in provider_properties:
            merged_properties.pop("why", None)
        result["properties"] = merged_properties
    if isinstance(provider.get("required"), Sequence) and not isinstance(provider.get("required"), (str, bytes, bytearray)):
        current = list(result.get("required", ()))
        result["required"] = sorted({str(item) for item in [*current, *provider["required"]]})
    if "additionalProperties" in provider:
        result["additionalProperties"] = copy.deepcopy(provider["additionalProperties"])
    return result


def _spec_for_tool(tool: Mapping[str, object], *, probe_modes: Sequence[str] | None = None) -> ToolSpec:
    name = _contract_tool_name(tool)
    function = tool.get("function")
    provider = function.get("parameters") if isinstance(function, Mapping) else None
    base = TOOL_SPECS.get(name)
    if base is None:
        schema = (copy.deepcopy(dict(provider)) if isinstance(provider, Mapping)
                  else _object_schema({}, additional=False))
        return _make_tool_spec(name, schema)
    base_schema: Mapping[str, object] = base.schema
    if name == "probe" and probe_modes is not None:
        base_schema = _probe_schema(probe_modes)
    schema = (_merge_provider_schema(name, base_schema, provider)
              if isinstance(provider, Mapping) and name != "probe" else
              copy.deepcopy(dict(base_schema)))
    return _make_tool_spec(
        name, schema, variant_selector=_probe_variant if name == "probe" else None,
    )


def registered_tool_specs(tools: object) -> tuple[ToolSpec, ...]:
    """Resolve one exact ToolSpec for every tool in an advertised surface."""
    return tuple(_spec_for_tool(tool) for tool in _contract_tools(tools))


def validate_tool_arguments(tools: object, name: str, arguments: object) -> tuple[str, ...]:
    """Validate a model call against the exact schema that was advertised."""
    surface = _contract_tools(tools)
    tool = next((item for item in surface if _contract_tool_name(item) == name), None)
    if tool is None:
        offered = ", ".join(_contract_tool_name(item) for item in surface if _contract_tool_name(item)) or "none"
        return (f"tool '{name}' is not offered for the current state; choose one of: {offered}",)
    errors = _spec_for_tool(tool).validator(arguments)
    rendered: list[str] = []
    for error in errors:
        required = re.fullmatch(r"\$\.([A-Za-z_]\w*) is required", error)
        if required:
            rendered.append(f"missing required parameter: {required.group(1)}")
            continue
        typed = re.fullmatch(r"\$\.([A-Za-z_]\w*) must be (.+)", error)
        if typed:
            rendered.append(f"parameter {typed.group(1)} must be {typed.group(2)}")
            continue
        rendered.append(error)
    return tuple(rendered)


def dispatch_tool_call(tools: object, name: str, arguments: object) -> ToolDispatch:
    """Validate and route one native call through its owning ToolSpec."""
    surface = _contract_tools(tools)
    tool = next((item for item in surface if _contract_tool_name(item) == name), None)
    if tool is None:
        errors = validate_tool_arguments(surface, name, arguments)
        raise RequestActionContractError("; ".join(errors))
    return _spec_for_tool(tool).dispatch(arguments)


def _schema_variants(schema: Mapping[str, object]) -> tuple[frozenset[str], ...]:
    branches = schema.get("oneOf")
    if isinstance(branches, Sequence) and not isinstance(branches, (str, bytes, bytearray)):
        return tuple(
            frozenset(str(item) for item in branch.get("required", ()))
            for branch in branches if isinstance(branch, Mapping)
        )
    required = schema.get("required", ())
    return (frozenset(str(item) for item in required),)


def assert_tool_dispatch_contract(
        tools: object, *, runtime: object | None = None,
) -> None:
    """Prove every dispatch variant is present and required in its advertised schema."""
    del runtime  # retained for the public v46 call signature
    for tool in _contract_tools(tools):
        name = _contract_tool_name(tool)
        function = tool.get("function")
        parameters = function.get("parameters") if isinstance(function, Mapping) else None
        if not isinstance(parameters, Mapping):
            raise RequestActionContractError(
                f"dispatch contract missing parameters for advertised tool {name or '<unnamed>'}")
        properties = parameters.get("properties")
        property_names = set(properties) if isinstance(properties, Mapping) else set()
        expected = TOOL_SPECS.get(name)
        actual_variants = _schema_variants(parameters)
        if name == "probe" and not isinstance(parameters.get("oneOf"), Sequence):
            raise RequestActionContractError(
                "dispatch contract for probe requires explicit mode variants")
        expected_variants = (
            actual_variants if name == "probe"
            else _schema_variants(expected.schema if expected is not None else parameters)
        )
        for required in expected_variants:
            missing = set(required) - property_names
            if missing:
                raise RequestActionContractError(
                    f"dispatch contract for {name} requires advertised properties "
                    + ", ".join(sorted(missing)))
            if name == "probe":
                if not any(required.issubset(variant) for variant in actual_variants):
                    raise RequestActionContractError(
                        "dispatch contract for probe omits variant requiring "
                        + ", ".join(sorted(required)))
            elif isinstance(parameters.get("oneOf"), Sequence):
                if not any(required.issubset(variant) for variant in actual_variants):
                    raise RequestActionContractError(
                        f"dispatch contract for {name} omits variant requiring "
                        + ", ".join(sorted(required)))
            else:
                advertised_required = set(parameters.get("required", ()))
                missing_required = set(required) - advertised_required
                if missing_required:
                    raise RequestActionContractError(
                        f"dispatch contract for {name} must mark required arguments "
                        + ", ".join(sorted(missing_required)))
        if parameters.get("additionalProperties") is False:
            for variant in actual_variants:
                missing = set(variant) - property_names
                if missing:
                    raise RequestActionContractError(
                        f"advertised schema rejects dispatch arguments for {name}: "
                        + ", ".join(sorted(missing)))


def canonicalize_tool_surface(
        tools: object, *, runtime: object | None = None,
        probe_modes: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Generate every advertised schema from the owning ToolSpec registry."""
    result: list[dict[str, object]] = []
    specs: dict[str, ToolSpec] = {}
    for tool in _contract_tools(tools):
        name = _contract_tool_name(tool)
        function = tool.get("function")
        description = str(function.get("description") or "") if isinstance(function, Mapping) else ""
        spec = _spec_for_tool(tool, probe_modes=probe_modes)
        if not name:
            raise RequestActionContractError("provider tool surface contains an empty name")
        specs[name] = spec
        result.append(spec.advertised(description))
    if runtime is not None and _no_host_checks(runtime):
        # Preserve the frozen no-host terminal provider schema.  ``recall`` is
        # a host-backed indexed-run action and is admitted on every stateful
        # frame; this compatibility path has no host check/state frame yet.
        result = [tool for tool in result
                  if _contract_tool_name(tool) != "recall"]
        specs.pop("recall", None)
    assert_tool_dispatch_contract(result, runtime=runtime)
    if runtime is not None:
        setattr(runtime, "_completion_owner_tool_specs", specs)
    return result


def _contract_revision(runtime: object) -> str:
    # Wire contracts, memo keys, and stale checks share the live product revision;
    # the shared resolver excludes the request journal from product identity.
    from kernel_check_receipts import current_wire_revision

    return current_wire_revision(runtime)


def _request_contract_revision_rebind_mode(runtime: object | None = None) -> str:
    """Read legacy rebind mode, retained for optional shadow diagnostics."""
    try:
        configured = getattr(runtime, "stateframe_request_contract_revision_rebind_mode", None)
        if configured is None:
            configured = getattr(runtime, "stateframe_request_contract_revision_rebind", None)
        value = str(configured if configured is not None else
                    os.environ.get(REQUEST_CONTRACT_REVISION_REBIND_ENV, "off")).strip().lower()
    except Exception:
        return "off"
    return value if value in {"off", "shadow", "on"} else "off"


def _emit_request_contract_revision_rebind_shadow_mismatch(
        runtime: object, request_id: str, field: str,
) -> None:
    try:
        # _emit already stamps dispatch=False; passing it again raised TypeError (duplicate keyword) that the except
        # below swallowed, so every shadow mismatch was silently dropped.
        _emit(
            runtime, "request_contract_revision_rebind_shadow_mismatch",
            request_id=request_id, field=field,
        )
    except Exception:
        # Shadow telemetry must never change final-surface authoring.
        pass


def _contract_budget_generation(runtime: object) -> str | None:
    explicit = getattr(runtime, "budget_generation", None)
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    for budget_name in ("continuation_limits", "decision_budget"):
        budget = getattr(runtime, budget_name, None)
        if budget is None:
            continue
        generation = getattr(budget, "generation", None)
        if generation is None:
            generation = getattr(budget, "budget_generation", None)
        if generation is not None and str(generation).strip():
            return str(generation)
        stable_fields = (
            "limit", "mode", "arm", "tool_multiplier", "automatic_check_limit",
            "terminal_reserve_limit", "output_reservation_tokens",
            "sandbox_deadline_monotonic",
        )
        values = {
            field: getattr(budget, field, None)
            for field in stable_fields
        }
        if any(value is not None for value in values.values()):
            return _digest(values)
    return None


def _contract_host_text(messages: object) -> str:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return ""
    # Assistant reasoning is not a host instruction.  Only host-visible system,
    # user, and tool receipts can create the terminal-text/tool contradiction.
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, Mapping)
        and str(message.get("role", "")) in {"system", "user", "tool"}
    )


def _framefit5_owner_protocol(messages: object) -> Mapping[str, object] | None:
    """Read the typed owner protocol carried inside a FRAMEFIT5 StateFrame."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return None
    from kernel_cache_layout_view import semantic_messages
    messages = semantic_messages(messages)
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith("[StateFrame]\n"):
            continue
        try:
            payload = json.loads(content.split("\n", 1)[1])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        state = payload.get("state") if isinstance(payload, Mapping) else None
        framefit5 = state.get("framefit5") if isinstance(state, Mapping) else None
        protocol = framefit5.get("owner_protocol") if isinstance(framefit5, Mapping) else None
        if (isinstance(protocol, Mapping)
                and protocol.get("schema") == "stateframe-framefit5-owner/1"):
            return protocol
    return None


def _framefit5_terminal_line_present(messages: object, expected: str) -> bool:
    protocol = _framefit5_owner_protocol(messages)
    if protocol is not None:
        return protocol.get("terminal_line") == expected
    return expected in _contract_host_text(messages)


def _wire_evidence_pin(messages: object) -> tuple[str | None, int]:
    """Return the one serialized evidence block and its wire occurrence count."""
    protocol = _framefit5_owner_protocol(messages)
    if protocol is not None:
        value = protocol.get("evidence_pin")
        return (value, 1) if isinstance(value, str) and value else (None, 0)
    text = _contract_host_text(messages)
    start = "[EVIDENCEPIN]"
    end = "[/EVIDENCEPIN]"
    values: list[str] = []
    position = 0
    while True:
        begin = text.find(start, position)
        if begin < 0:
            break
        finish = text.find(end, begin + len(start))
        if finish < 0:
            break
        values.append(text[begin:finish + len(end)])
        position = finish + len(end)
    return (values[-1] if values else None), len(values)


def _wire_continuation_limits(messages: object) -> str | None:
    protocol = _framefit5_owner_protocol(messages)
    if protocol is not None:
        value = protocol.get("continuation_limits")
        return value if isinstance(value, str) and value else None
    text = _contract_host_text(messages)
    return next(
        (line.strip() for line in text.splitlines()
         if line.startswith("[CONTINUATION LIMITS |")),
        None,
    )


def _contract_evidence_pin(runtime: object) -> str | None:
    """Return only the digest of the host-authored typed evidence block."""
    block = _owner_block(runtime, "evidence_pin")
    digest = block.get("digest") if block is not None else None
    return str(digest) if digest else None


def _contract_evidence_pin_value(runtime: object) -> str | None:
    value = _owner_block_value(runtime, "evidence_pin")
    return value if isinstance(value, str) and value else None


def _evidence_marker_present(value: str) -> bool:
    return "[EVIDENCEPIN]" in value and "[/EVIDENCEPIN]" in value


def _terminal_line_requires_wire(value: str | None) -> bool:
    text = str(value or "")
    return "Final step: call done now" in text and (
        "this revision" in text or "NO HOST CHECKS:" in text
    )


def _contract_terminal_text(runtime: object) -> bool:
    """Return terminal status only from the host-owned typed terminal block."""
    return _owner_block(runtime, "terminal_line") is not None


def _contract_context_rows(runtime: object) -> tuple[Mapping[str, object], ...]:
    raw = getattr(runtime, "_completion_owner_required_context", None)
    if raw is None:
        raw = getattr(runtime, "required_action_context", ())
    if isinstance(raw, Mapping):
        raw = (raw,)
    return tuple(item for item in _sequence(raw) if isinstance(item, Mapping))


def _contract_context_status(
        runtime: object,
        messages: object,
        tool_names: Sequence[str],
) -> tuple[list[dict[str, object]], list[str]]:
    """Retain required context or prove the route that can acquire it."""
    text = _contract_host_text(messages)
    statuses: list[dict[str, object]] = []
    missing: list[str] = []
    for index, row in enumerate(_contract_context_rows(runtime)):
        identifier = str(row.get("id", row.get("name", f"context-{index}"))).strip()
        expected = row.get("text", row.get("source", row.get("content")))
        present = row.get("present") is True
        if not present and isinstance(expected, str) and expected:
            present = expected in text
        acquisition = str(row.get("acquisition_tool", row.get("route", ""))).strip()
        if not present and acquisition and acquisition in tool_names:
            statuses.append({
                "id": identifier,
                "status": "acquisition_route",
                "acquisition_tool": acquisition,
            })
            continue
        if not present:
            missing.append(identifier)
            continue
        statuses.append({"id": identifier, "status": "retained"})
    return statuses, missing


def _lean_source_scope(runtime: object) -> dict[str, object]:
    """Capture the live target and declared editable product paths for lean requests."""
    from kernel_request_source_window import editable_product_paths

    target_relative = getattr(runtime, "_target_relative", None)
    target = target_relative() if callable(target_relative) else ""
    return {
        "target_path": str(target or ""),
        "editable_product_paths": list(editable_product_paths(runtime)),
    }


def establish_lean_request_action_contract(
        runtime: object,
        messages: object,
        tools: object,
        *,
        request_id: str,
) -> dict[str, object]:
    """Freeze the complete stable lean surface without inheriting a routed menu."""
    if not str(request_id).strip():
        raise RequestActionContractError("lean request contract requires a request identity")
    schemas = canonicalize_tool_surface(tools, runtime=runtime)
    names = [_contract_tool_name(tool) for tool in schemas]
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        _contract_error(
            runtime, "lean request tools must be a nonempty unique stable surface",
            request_id=request_id, boundary="lean_prepare", route="lean", tool_names=names)
    if isinstance(tools, list):
        tools[:] = copy.deepcopy(schemas)
    source_scope = _lean_source_scope(runtime)
    required_context, missing_context = _contract_context_status(runtime, messages, names)
    if missing_context:
        _contract_error(
            runtime,
            "required action context is absent and has no retained acquisition route: "
            + ", ".join(missing_context),
            request_id=request_id, boundary="lean_prepare", route="lean", tool_names=names)
    record_owner_block(
        runtime, "menu", schemas, block_id=f"menu:{request_id}", request_id=request_id,
    )
    contract: dict[str, object] = {
        "schema": REQUEST_CONTRACT_SCHEMA,
        "contract_kind": LEAN_REQUEST_CONTRACT_KIND,
        "request_id": str(request_id),
        "revision": _contract_revision(runtime),
        "route": "lean",
        "unresolved_work": _contract_unresolved_work(runtime),
        "admitted_tool_schemas": schemas,
        "advertised_actions": list(names),
        "tool_schema_digest": _digest(schemas),
        "source_scope": source_scope,
        "source_scope_digest": _digest(source_scope),
        "required_context": required_context,
        "required_context_digest": _digest(_contract_context_rows(runtime)),
        "budget_generation": _contract_budget_generation(runtime),
        "terminal_text": False,
        "terminal_line": None,
        "host_stop": False,
        "owner_blocks": _owner_block_records(runtime),
    }
    evidence_pin_digest = _contract_evidence_pin(runtime)
    if evidence_pin_digest is not None:
        evidence_pin_value = _contract_evidence_pin_value(runtime)
        wire_pin, wire_pin_count = _wire_evidence_pin(messages)
        if (evidence_pin_value and _evidence_marker_present(evidence_pin_value)
                and (wire_pin_count != 1 or wire_pin != evidence_pin_value)):
            _contract_error(
                runtime, "typed evidence pin is absent or changed on the lean request",
                request_id=request_id, boundary="lean_prepare", route="lean", tool_names=names)
        contract["evidence_pin_schema"] = "stateframe-evidence-pin/1"
        contract["evidence_pin_digest"] = evidence_pin_digest
        if wire_pin is not None:
            contract["evidence_pin_wire_digest"] = _digest([wire_pin])
    budget_generation = contract.get("budget_generation")
    budget_object = getattr(runtime, "continuation_limits", None)
    if budget_object is None:
        budget_object = getattr(runtime, "decision_budget", None)
    if (budget_generation is not None
            and callable(getattr(budget_object, "render_continuation_limits", None))):
        wire_limits = _wire_continuation_limits(messages)
        if not wire_limits:
            _contract_error(
                runtime, "continuation limits are absent from the lean request",
                request_id=request_id, boundary="lean_prepare", route="lean", tool_names=names)
        contract["budget_wire_digest"] = _digest(wire_limits)
    contract["decision_digest"] = _contract_decision_digest(contract)
    frozen = _freeze_contract(contract)
    setattr(runtime, "_completion_owner_request_contract", frozen)
    runtime.completion_menu = tuple(names)
    runtime.offered_tools = tuple(names)
    runtime._question_done_available = "done" in names
    runtime._completion_owner_dispatch_tools = copy.deepcopy(schemas)
    runtime._completion_owner_actionable = bool(names)
    runtime._completion_owner_actionable = any(name != "done" for name in names)
    _emit(
        runtime, "provider_boundary_contract_decided", request_id=request_id,
        route="lean", revision=contract["revision"],
        unresolved_work=contract["unresolved_work"], advertised_actions=names,
        required_context=required_context, decision_digest=contract["decision_digest"],
    )
    _emit(
        runtime, "lean_action_contract_finalized", request_id=request_id,
        revision=contract["revision"], source_scope_digest=contract["source_scope_digest"],
        tool_schema_digest=contract["tool_schema_digest"], advertised_actions=names,
    )
    value = _thaw_contract(frozen)
    return value if isinstance(value, dict) else {}


def _contract_unresolved_work(runtime: object) -> list[str]:
    values = [*open_ids(runtime), *_decision_codes(runtime, None)]
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _empty_tool_route(runtime: object) -> tuple[str, str | None]:
    from kernel_terminal_policy import enabled as hard_budget_only_enabled

    if (hard_budget_only_enabled(runtime)
            and getattr(runtime, "_completion_owner_host_stop_pending", False)):
        return "host-stop", None
    return "", "a model-action request cannot have an empty tool list"


def _contract_route(runtime: object, messages: object, tool_names: Sequence[str], *,
                    provider_request: bool) -> tuple[str, str | None]:
    outcome = str(getattr(runtime, "system_outcome", "") or "")
    if outcome in _HOST_STOP_OUTCOMES and not tool_names:
        return "host-stop", None
    if not tool_names:
        return _empty_tool_route(runtime)
    if (_green_open_work(runtime)
            and {"edit", "str_replace", "read", "run_tests", "work", "work_update"}
            .intersection(tool_names)):
        return "repair", None
    terminal_text = _contract_terminal_text(runtime)
    if terminal_text and "done" not in tool_names:
        return "", "terminal text requires an admitted done tool"
    if terminal_text:
        return "submit", None
    state = _state(runtime)
    terminal_menu = tuple(str(name) for name in getattr(runtime, "completion_menu", ()) or ())
    if ("done" in tool_names
            and state.get("final_submit_request_pending")
            and _reserved_terminal_menu_names(runtime, terminal_menu)):
        return "submit", None
    if "run_tests" in tool_names and not {"edit", "str_replace", "work", "work_update"}.intersection(tool_names):
        return "verify", None
    return "repair", None


def align_tools_to_completion_route(runtime: object, tools: Sequence[object]) -> list[object]:
    """Make the submitted tool list BE the completion route's admitted schemas (one decision -> prose, menu, dispatch).

    r2b live 934/935 (10-08): the route decided its actions, but the request's tool surface was still built by the
    older terminal-menu path (str_replace/read/read_check_receipt, the J928 menu), so the new identity invariant fired
    and killed both runs at prepare(). The decision is the authority: when it routes ACTION/UNCERTIFIED_SUBMIT with
    concrete schemas, those ARE the menu. Other routes (and no runtime) leave the tools untouched.
    """
    original = list(tools or ())
    if runtime is None:
        return original
    try:
        decision = completion_route_decision(
            runtime, getattr(runtime, "_last_completion_decision", None),
            tools=getattr(runtime, "_completion_owner_full_tools", ()) or (),
        )
    except Exception:  # noqa: BLE001 - alignment is a projection; the contract check still runs after it
        return original
    if decision.route not in {"ACTION", "UNCERTIFIED_SUBMIT"} or not decision.tool_schemas:
        return original
    aligned = [copy.deepcopy(dict(schema)) for schema in decision.tool_schemas]
    aligned_names = tuple(_tool_name(tool) for tool in aligned)
    # Live benchmark r2AK (batch 1025, seq 6305): green with R1-R3 open, the ACTION route
    # aligned [edit, str_replace, read, ...] down to [read, read_check_receipt, done],
    # undoing owner_green_open_repair_kept one event later; with a declared file still
    # unedited the model could only re-read (read loop to the cap). While green evidence
    # has open work, the repair tools already on the menu stay on it.
    if _green_open_work(runtime):
        kept = [copy.deepcopy(dict(tool)) for tool in original
                if _tool_name(tool) in {"edit", "str_replace"} and _tool_name(tool) not in aligned_names]
        if kept:
            aligned = kept + aligned
            aligned_names = tuple(_tool_name(tool) for tool in aligned)
    if "done" in aligned_names and "read" not in aligned_names:
        read_tool = _deferred_source_read_tool(runtime, [*original, *aligned])
        if read_tool is not None:
            aligned.append(read_tool)
    if [_tool_name(tool) for tool in original] != [_tool_name(tool) for tool in aligned]:
        journal = getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if callable(emit):
            emit("completion_route_menu_aligned", producer="kernel_completion_owner", dispatch=False,
                 route=decision.route, before=[_tool_name(tool) for tool in original],
                 after=[_tool_name(tool) for tool in aligned])
    aligned_names = tuple(_tool_name(tool) for tool in aligned)
    runtime.completion_menu = aligned_names
    runtime.offered_tools = aligned_names
    runtime._question_done_available = "done" in aligned_names
    runtime._completion_owner_dispatch_tools = copy.deepcopy(aligned)
    runtime._completion_owner_actionable = any(name != "done" for name in aligned_names)
    return aligned


def _route_surface_strict() -> bool:
    """STATEFRAME_COMPLETION_ROUTE_STRICT=on makes a route/menu identity violation a hard contract error (tests, lab).
    Default OFF in live: r2b 934/935 died at prepare() on it (green-lock menu authority advertised tools the final-wire
    composer strips). A run must never die on a host projection mismatch; it is journaled as a finding instead."""
    return os.environ.get("STATEFRAME_COMPLETION_ROUTE_STRICT", "off").strip().lower() == "on"


def _route_surface_violation(runtime: object, message: str, *, request_id: str | None, boundary: str,
                             tool_names: Sequence[str]) -> None:
    if _route_surface_strict():
        _contract_error(runtime, message, request_id=request_id, boundary=boundary,
                        route="completion-route", tool_names=tool_names)
        return
    journal = getattr(runtime, "journal", None) if runtime is not None else None
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    try:
        decision = getattr(runtime, "_completion_owner_completion_decision", None)
        emit("completion_route_surface_mismatch", producer="kernel_completion_owner", dispatch=False,
             request_id=request_id, boundary=boundary, error=str(message)[:300],
             wire_actions=[str(name) for name in tool_names],
             route=getattr(decision, "route", None),
             route_actions=list(getattr(decision, "advertised_actions", ()) or ()),
             route_dispatchable=list(getattr(decision, "dispatch_admissible_actions", ()) or ()))
    except Exception:  # noqa: BLE001 - a finding must never become the crash it replaces
        return


def _completion_route_surface_error(
        runtime: object, tool_names: Sequence[str], *, request_id: object = None,
) -> str | None:
    if runtime is None:
        # Frozen-capture/provider-wire checks run without a runtime; the frozen contract already carries the
        # admitted surface, so there is no live completion route to compare against (10-08 regression).
        return None
    from kernel_host_cache import memo_runtime_fact, optimized_turn_facts_enabled
    if not optimized_turn_facts_enabled():
        return _completion_route_surface_error_uncached(
            runtime, tool_names, request_id=request_id)
    from kernel_done_readiness_authority import cached_decision
    try:
        # The memo must key on the shared SUBMIT decision: the uncached body reads it, so a cached
        # answer computed under an older decision would be stale (BUILD r2GU merge, W2 x DONEWIRE).
        shared_done_decision = cached_decision(runtime)
        decision_key = (id(shared_done_decision), getattr(shared_done_decision, "decision", None))
        normalized_names = tuple(str(name) for name in tool_names)
        dispatch_tools = getattr(runtime, "_completion_owner_dispatch_tools", ())
        full_tools = getattr(runtime, "_completion_owner_full_tools", ())
        route_tools = (*_sequence(dispatch_tools), *_sequence(full_tools))
    except Exception:
        return _completion_route_surface_error_uncached(
            runtime, tool_names, request_id=request_id)
    return memo_runtime_fact(
        runtime, "TURN_FACTS",
        lambda: _completion_route_surface_error_uncached(
            runtime, normalized_names, request_id=request_id),
        tools=route_tools,
        reads_tools=True,
        fact_key=("completion_route_surface_error", request_id, normalized_names, decision_key),
    )


def _completion_route_surface_error_uncached(
        runtime: object, tool_names: Sequence[str], *, request_id: object = None,
) -> str | None:
    from kernel_done_readiness_authority import cached_decision

    shared_done_decision = cached_decision(runtime)
    decision = completion_route_decision(
        runtime, getattr(runtime, "_last_completion_decision", None),
        tools=getattr(runtime, "_completion_owner_full_tools", ()) or (),
    )
    if (shared_done_decision is not None
            and getattr(shared_done_decision, "decision", None) == "ACCEPTED_UNCERTIFIED"
            and decision.uncertified_submission_admissible):
        done_routes = tuple(
            action for action in decision.actions
            if str(action.get("tool", "")).strip() == "done"
        )
        if (len(done_routes) != 1
                or done_routes[0].get("route") != "UNCERTIFIED_SUBMIT"):
            return "shared done readiness decision is missing from the completion route"
    # HOST_BLOCKER carries no actions by definition; the identity invariant applies to routes that advertise actions
    # (r2b 934/935: a blocker route vs a live working menu was a guaranteed false violation).
    if decision.route not in {"ACTION", "UNCERTIFIED_SUBMIT"}:
        return None
    try:
        from kernel_tool_surface import runner_dispatchable, tool_allowed
        dispatchable = tuple(
            name for name in tool_names
            if runner_dispatchable(name, runtime=runtime)
            and tool_allowed(name, runtime=runtime)
        )
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        dispatchable = ()
    advertised = tuple(str(name) for name in tool_names)
    deferred_read = (
        "read" in advertised
        and bool(_pending_deferred_source_paths(runtime))
        and tuple(name for name in advertised if name != "read")
        == decision.advertised_actions
    )
    route_names = tuple(name for name in advertised if not (deferred_read and name == "read"))
    route_dispatchable = tuple(
        name for name in dispatchable if not (deferred_read and name == "read")
    )
    try:
        from kernel_rejection_template_restrict import active_menu_tools
        restricted_tools = active_menu_tools(runtime, request_id)
    except (ImportError, AttributeError, TypeError, ValueError):
        restricted_tools = None
    expected = set(restricted_tools or ())
    if (restricted_tools is not None and expected
            and len(advertised) == len(expected) and set(advertised) == expected
            and len(route_names) == len(expected) and set(route_names) == expected
            and len(route_dispatchable) == len(expected)
            and set(route_dispatchable) == expected
            and expected.issubset(decision.advertised_actions)
            and expected.issubset(decision.dispatch_admissible_actions)):
        # Shared request-scoped menus may narrow the owner route only to actions
        # admitted and dispatchable by that same route.
        return None
    if (route_names != decision.advertised_actions
            or route_dispatchable != decision.dispatch_admissible_actions
            or route_names != route_dispatchable
            or (deferred_read and "read" not in dispatchable)):
        return (
            "completion route invariant failed: advertised actions, final tools, and "
            "dispatch-admissible actions must be identical"
        )
    return None


def _restricted_tools_admissible(
        runtime: object, tool_names: Sequence[str], *, require_route: bool,
) -> bool:
    """Require each restricted action to be host-dispatchable and route-admitted."""
    names = tuple(dict.fromkeys(str(name) for name in tool_names if str(name)))
    if not names:
        return False
    try:
        from kernel_tool_surface import dispatch_admissible
        if not all(dispatch_admissible(name, runtime=runtime) for name in names):
            return False
        full_surface = getattr(runtime, "_completion_owner_full_tools", ()) or ()
        if not full_surface:
            return True
        decision = completion_route_decision(
            runtime, getattr(runtime, "_last_completion_decision", None),
            tools=full_surface)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False
    route_admits_action = decision.route in {"ACTION", "UNCERTIFIED_SUBMIT"}
    if not route_admits_action:
        # NONE leaves the registered menu in force for legacy rejection repair.
        return decision.route == "NONE" and not require_route
    return set(names).issubset(decision.advertised_actions) and set(names).issubset(
        decision.dispatch_admissible_actions
    )


def _optional_restricted_tools_admissible(
        runtime: object, offered: Sequence[str], candidates: Sequence[str],
) -> tuple[str, ...]:
    """Keep optional verification actions only when dispatch and route admit them."""
    return tuple(
        name for name in candidates
        if offered.count(name) == 1
        and _restricted_tools_admissible(runtime, (name,), require_route=False)
    )


def _rejection_template_tool_admissible(runtime: object) -> bool:
    """Preserve the rejection repair's existing single-action admission rules."""
    return _restricted_tools_admissible(
        runtime, ("str_replace",), require_route=False
    )


def _rejection_menu_admission(
        runtime: object, messages: object, tool_names: Sequence[str], route: str,
        route_error: str | None, provider_request: bool,
) -> tuple[bool, str | None]:
    from kernel_rejection_template_restrict import rejection_pending
    if not rejection_pending(runtime):
        return True, None
    _, missing_context = _contract_context_status(runtime, messages, ["str_replace"])
    owner_admits = _rejection_template_tool_admissible(runtime)
    route_allows_repair = (
        route == "repair"
        or (route == "submit" and "str_replace" in tool_names and owner_admits)
    )
    admissible = bool(
        provider_request and route_allows_repair and route_error is None
        and not missing_context and owner_admits
    )
    if not provider_request:
        reason = "not_a_provider_request"
    elif route_error:
        reason = "owner_route_error"
    elif route not in {"repair", "submit"}:
        reason = f"owner_route_{route or 'unavailable'}"
    elif missing_context:
        reason = "required_context_not_retained_or_routable"
    elif not owner_admits:
        reason = "completion_owner_or_runner_disallows_str_replace"
    elif not route_allows_repair:
        reason = f"owner_route_{route or 'unavailable'}"
    else:
        reason = None
    return admissible, reason


def _read_pointer_menu_admission(
        runtime: object, messages: object, route: str, route_error: str | None,
        provider_request: bool,
) -> tuple[bool, str | None]:
    from kernel_rejection_template_restrict import (
        READ_POINTER_REQUIRED_TOOLS, read_pointer_menu_pending,
    )
    if not read_pointer_menu_pending(runtime):
        return True, None
    if _pending_deferred_source_paths(runtime):
        return False, "deferred_required_source_read_pending"
    _, missing_context = _contract_context_status(
        runtime, messages, list(READ_POINTER_REQUIRED_TOOLS))
    tools_admissible = _restricted_tools_admissible(
        runtime, READ_POINTER_REQUIRED_TOOLS, require_route=False
    )
    admissible = bool(
        provider_request and route == "repair" and route_error is None
        and not missing_context and tools_admissible
    )
    if not provider_request:
        reason = "not_a_provider_request"
    elif route_error:
        reason = "owner_route_error"
    elif route != "repair":
        reason = f"owner_route_{route or 'unavailable'}"
    elif missing_context:
        reason = "required_context_not_retained_or_routable"
    elif not tools_admissible:
        reason = "completion_owner_or_runner_disallows_menu_tool"
    else:
        reason = None
    return admissible, reason


def _noread_menu_admission(
        runtime: object, messages: object, tool_names: Sequence[str], route: str,
        route_error: str | None, provider_request: bool,
) -> tuple[bool, str | None]:
    from kernel_rejection_template_restrict import noread_menu_pending

    if not noread_menu_pending(runtime):
        return True, None
    if not provider_request:
        return False, "not_a_provider_request"
    if route_error:
        return False, "owner_route_error"
    if _pending_deferred_source_paths(runtime):
        return False, "deferred_required_source_read_pending"
    if "read" not in tool_names:
        return False, "read_not_offered"

    candidate = tuple(name for name in tool_names if name != "read")
    if not any(name != "done" for name in candidate):
        return False, "read_is_only_non_done_action"
    _, missing_context = _contract_context_status(runtime, messages, candidate)
    if missing_context:
        return False, "completion_route_requires_read"
    decision = getattr(runtime, "_completion_owner_completion_decision", None)
    route_actions = tuple(
        row for row in _sequence(getattr(decision, "actions", ()))
        if isinstance(row, Mapping)
    )
    read_clears = {
        str(row.get("clears", "")).strip()
        for row in route_actions
        if row.get("tool") == "read" and str(row.get("clears", "")).strip()
    }
    alternative_clears = {
        str(row.get("clears", "")).strip()
        for row in route_actions
        if row.get("tool") in candidate
        and row.get("tool") not in {"read", "done"}
        and str(row.get("clears", "")).strip()
    }
    if read_clears - alternative_clears:
        return False, "completion_route_requires_read"
    if not _restricted_tools_admissible(runtime, candidate, require_route=False):
        return False, "completion_owner_or_runner_disallows_menu_tool"
    if route not in {"repair", "verify", "submit"}:
        return False, f"owner_route_{route or 'unavailable'}"
    return True, None


def _contract_error(
        runtime: object | None,
        message: str,
        *,
        request_id: str | None,
        boundary: str,
        route: str | None = None,
        tool_names: Sequence[str] = (),
        boundary_recovery_eligible: bool = True,
) -> None:
    journal = getattr(runtime, "journal", None) if runtime is not None else None
    emit = getattr(journal, "emit", None)
    if callable(emit):
        advertised = list(tool_names)
        emit(
            "provider_boundary_contract_error",
            producer="kernel_completion_owner",
            event_schema=REQUEST_CONTRACT_SCHEMA,
            request_id=request_id,
            boundary=boundary,
            route=route,
            advertised_actions=advertised,
            status=RequestActionContractError.status,
            outcome=RequestActionContractError.status,
            failure_origin="harness",
            provider_calls=0,
            error=str(message)[:400],
            dispatch=False,
        )
    error = RequestActionContractError(str(message))
    error.boundary_recovery_eligible = boundary_recovery_eligible
    raise error


def _owner_offered_done(runtime: object) -> bool:
    """True when the completion owner itself selected/offered ``done``."""
    if isinstance(getattr(runtime, "_completion_owner_done_selection", None), Mapping):
        return True
    if getattr(runtime, "_question_done_available", False):
        return True
    return "done" in tuple(str(name) for name in getattr(runtime, "completion_menu", ()) or ())


def _done_dependency_denied(runtime: object) -> bool:
    gate = getattr(runtime, "_invalidation_gate", None)
    return isinstance(gate, Mapping) and gate.get("admitted") is False


def _reconcile_terminal_text_without_done(
        runtime: object,
        messages: object,
        tools: object,
        schemas: list,
        names: list,
        *,
        request_id: str | None,
        admitted_repair: bool = False,
) -> None:
    """Reconcile a terminal line after its ``done`` action is no longer admitted.

    Either recover the owner-admitted done schema (the owner offered it, or a
    witness receipt closes a clause), or withdraw the terminal line with a
    journal event. An owner-admitted rejected-edit repair may supersede a pending
    submit handoff. Never raise: projection drift is not a malformed surface.
    """
    recover = (not admitted_repair and (
        _current_witness_receipt_closes_clause(runtime)
        or (_owner_offered_done(runtime) and not _done_dependency_denied(runtime))))
    done = None
    if recover:
        full_surface = getattr(runtime, "_completion_owner_full_tools", ())
        done = next((copy.deepcopy(tool) for tool in _sequence(full_surface)
                     if isinstance(tool, Mapping) and _contract_tool_name(tool) == "done"), None)
    if done is not None:
        schemas.append(done)
        names.append("done")
        if isinstance(tools, list):
            tools[:] = copy.deepcopy(schemas)
        _emit(runtime, "completion_owner_terminal_done_recovered",
              request_id=request_id, advertised=list(names))
        return
    reason = (
        "rejection_template_repair_admitted" if admitted_repair else
        "done_not_offered_by_owner" if not recover else
        "done_schema_absent_from_full_surface"
    )
    if not admitted_repair and _done_dependency_denied(runtime):
        reason = "done_denied_by_dependency_gate"
    block = _owner_block(runtime, "terminal_line")
    block_id = str(block.get("id", "")) if isinstance(block, Mapping) else ""
    raw = getattr(runtime, "_completion_owner_owner_blocks", None)
    if isinstance(raw, Mapping):
        remaining = {key: value for key, value in raw.items() if key != "terminal_line"}
        setattr(runtime, "_completion_owner_owner_blocks", remaining)
    line = None
    if isinstance(raw, Mapping) and isinstance(raw.get("terminal_line"), Mapping):
        value = raw["terminal_line"].get("_value")
        line = value.strip() if isinstance(value, str) and value.strip() else None
    if isinstance(messages, list):
        messages[:] = [
            message for message in messages
            if not (isinstance(message, Mapping) and (
                (block_id and message.get("_stateframe_owner_block") == block_id)
                or (line is not None and str(message.get("content", "")).strip() == line)))
        ]
    if admitted_repair:
        _clear_terminal_offer_state(runtime)
    else:
        state = _state(runtime)
        state["final_submit_request_pending"] = False
    _emit(runtime, "completion_owner_terminal_text_withdrawn",
          request_id=request_id, reason=reason, advertised=list(names),
          terminal_block_id=block_id or None)


def _apply_capability_floor_to_request_contract(
        runtime: object, messages: object, tools: object,
        schemas: list[dict[str, object]], *,
        provider_request: bool, request_id: str | None,
) -> tuple[list[dict[str, object]], list[str], str | None, str | None] | None:
    from kernel_capability_floor import apply_floor, enabled as capability_floor_enabled

    if not capability_floor_enabled(runtime):
        return None
    candidates: list[Mapping[str, object]] = []
    candidate_names: set[str] = set()
    sources = (
        tools,
        getattr(runtime, "_readiness_full_tools", ()),
        getattr(runtime, "_completion_owner_full_tools", ()),
        getattr(runtime, "available_tools", ()),
    )
    for source in sources:
        if isinstance(source, Mapping):
            items = source.values()
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            items = source
        else:
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            name = _contract_tool_name(item)
            if name and name not in candidate_names:
                candidate_names.add(name)
                candidates.append(item)

    schemas = apply_floor(
        schemas, candidates,
        admissible=lambda name: capability_floor_action_admissible(runtime, name),
        runtime=runtime,
    )
    if isinstance(messages, list):
        try:
            from kernel_release_wire_claims import scope_turns_on_wire
            from kernel_step_contract import edit_nturn_scope_turns_enabled

            scoped_turns = (
                scope_turns_on_wire(messages) if edit_nturn_scope_turns_enabled() else []
            )
        except (ImportError, AttributeError, TypeError, ValueError):
            scoped_turns = []
        if scoped_turns:
            from kernel_capability_floor import SCOPE_TURN_PREFERRED_TOOLS, rank_tools

            schemas = rank_tools(schemas, SCOPE_TURN_PREFERRED_TOOLS)
    names = [_contract_tool_name(tool) for tool in schemas]
    if isinstance(tools, list):
        tools[:] = copy.deepcopy(schemas)
    runtime.completion_menu = tuple(names)
    runtime.offered_tools = tuple(names)
    runtime._question_done_available = "done" in names
    runtime._completion_owner_dispatch_tools = copy.deepcopy(schemas)
    route, route_error = _contract_route(
        runtime, messages, names, provider_request=provider_request)
    if "done" in names and any(name != "done" for name in names):
        terminal_line = _owner_block_value(runtime, "terminal_line")
        block = _owner_block(runtime, "terminal_line")
        if (isinstance(terminal_line, str)
                and "No other action is available." in terminal_line
                and isinstance(messages, list)):
            reconciled_line = terminal_line.replace(
                " No other action is available.", "").replace(
                    "No other action is available.", "").rstrip()
            block_id = str(block.get("id", "")) if isinstance(block, Mapping) else ""
            messages[:] = [
                ({**message, "content": reconciled_line}
                 if isinstance(message, Mapping) and (
                     (block_id and message.get("_stateframe_owner_block") == block_id)
                     or str(message.get("content", "")).strip() == terminal_line)
                 else message)
                for message in messages
            ]
            record_owner_block(
                runtime, "terminal_line", reconciled_line,
                block_id=block_id or None, request_id=request_id,
            )
            _emit(runtime, "completion_owner_terminal_text_reconciled",
                  request_id=request_id, reason="capability_floor_added_actions",
                  advertised=list(names), terminal_block_id=block_id or None)
    if isinstance(messages, list):
        from kernel_action_menu import synchronize_tool_instructions
        messages[:] = synchronize_tool_instructions(
            messages, schemas, getattr(runtime, "active_tool_restriction", None))
    return schemas, names, route, route_error


def establish_request_action_contract(
        runtime: object,
        messages: object,
        tools: object,
        *,
        request_id: str | None = None,
        provider_request: bool = True,
        preserve_input_surface: bool = False,
) -> dict[str, object]:
    """Create exactly one immutable owner decision for one prepared request."""
    current_id = str(request_id) if request_id is not None else None
    existing = getattr(runtime, "_completion_owner_request_contract", None)
    if (current_id is not None
            and isinstance(existing, Mapping)
            and existing.get("request_id") == current_id):
        _verify_contract_integrity(existing)
        value = _thaw_contract(existing)
        return value if isinstance(value, dict) else {}
    pending_wire = getattr(runtime, "_completion_owner_no_done_only_collapse_wire_pending", None)
    if (isinstance(pending_wire, Mapping)
            and pending_wire.get("request_id") != current_id):
        runtime._completion_owner_no_done_only_collapse_wire_pending = None

    schemas = canonicalize_tool_surface(tools, runtime=runtime)
    if isinstance(tools, list):
        tools[:] = copy.deepcopy(schemas)
    names = [_contract_tool_name(tool) for tool in schemas]
    state = _state(runtime)
    route_surface_error = _completion_route_surface_error(
        runtime, names, request_id=current_id)
    if route_surface_error:
        _route_surface_violation(runtime, route_surface_error, request_id=current_id,
                                 boundary="owner", tool_names=names)
    if (state.get("final_submit_reserved")
            and state.get("final_submit_request_pending")
            and not _contract_terminal_text(runtime)):
        fallback_terminal = (
            NO_HOST_CHECKS_TERMINAL_LINE if _no_host_checks(runtime) else
            "Final step: call done now to submit this revision (owner-admitted submit route)."
        )
        record_owner_block(runtime, "terminal_line", fallback_terminal, block_id="terminal:pending")
    if any(not name for name in names) or len(names) != len(set(names)):
        # Genuinely malformed surface: this stays a hard contract error.
        _contract_error(runtime, "advertised tool schemas must have unique function names",
                        request_id=current_id, boundary="owner", tool_names=names)
    if _contract_terminal_text(runtime) and "done" not in names:
        _reconcile_terminal_text_without_done(
            runtime, messages, tools, schemas, names, request_id=current_id)
    route, route_error = _contract_route(
        runtime, messages, names, provider_request=provider_request)
    from kernel_rejection_template_restrict import (
        active_policy_records,
        apply_menu as apply_shared_menu_restrictions,
        pending as shared_menu_restrictions_pending,
    )
    restriction_pending = shared_menu_restrictions_pending(runtime)
    restriction_admissible, admissible_reason = _rejection_menu_admission(
        runtime, messages, names, route, route_error, provider_request)
    pointer_menu_admissible, pointer_menu_reason = _read_pointer_menu_admission(
        runtime, messages, route, route_error, provider_request)
    rejection_optional_tools = _optional_restricted_tools_admissible(
        runtime, names, ("run_tests",))
    pointer_optional_tools = _optional_restricted_tools_admissible(
        runtime, names, ("authored_check", "run_tests"))
    noread_optional_tools = _optional_restricted_tools_admissible(
        runtime, names, ("run_tests",))
    noread_admissible, noread_reason = _noread_menu_admission(
        runtime, messages, names, route, route_error, provider_request)

    if preserve_input_surface:
        # Boundary recovery is constrained by the already-admitted menu. New
        # request policy projections must not expand or replace that surface.
        restricted_schemas, restriction_applied = schemas, False
    elif restriction_pending and not provider_request:
        # Native-batch captures drain actions already returned by an earlier
        # HTTP response; they do not consume the next model-request restriction.
        restricted_schemas, restriction_applied = schemas, False
    else:
        restricted_schemas, restriction_applied = apply_shared_menu_restrictions(
            runtime, schemas, request_id=current_id,
            admissible=restriction_admissible, reason=admissible_reason,
            policy_admissibility={
                "rejection_template_restrict": {
                    "admissible": restriction_admissible,
                    "reason": admissible_reason,
                    "optional_tools": rejection_optional_tools,
                },
                "read_pointer_edit_menu": {
                    "admissible": pointer_menu_admissible,
                    "reason": pointer_menu_reason,
                    "optional_tools": pointer_optional_tools,
                },
                "noread": {
                    "admissible": noread_admissible,
                    "reason": noread_reason,
                    "optional_tools": noread_optional_tools,
                },
            })
    if restriction_applied:
        schemas = restricted_schemas
        names = [_contract_tool_name(tool) for tool in schemas]
        if isinstance(tools, list):
            tools[:] = copy.deepcopy(schemas)
        runtime.completion_menu = tuple(names)
        runtime.offered_tools = tuple(names)
        runtime._question_done_available = False
        runtime._completion_owner_actionable = True
        runtime._completion_owner_done_selection = None
        runtime._completion_owner_dispatch_tools = copy.deepcopy(schemas)
        restriction_records = active_policy_records(runtime, current_id)
        active_components = {
            str(record.get("component"))
            for record in restriction_records if isinstance(record, Mapping)
        }
        existing_restriction = getattr(runtime, "active_tool_restriction", None)
        if isinstance(existing_restriction, Mapping):
            existing_rows = existing_restriction.get("restrictions")
            rows = existing_rows if isinstance(existing_rows, list) else [existing_restriction]
            combined_restrictions = [
                copy.deepcopy(dict(row)) for row in rows
                if isinstance(row, Mapping)
                and str(row.get("component")) not in active_components
            ]
        else:
            combined_restrictions = []
        combined_restrictions.extend(copy.deepcopy(restriction_records))
        active_restriction = (
            {"restrictions": combined_restrictions}
            if len(combined_restrictions) != 1 else combined_restrictions[0]
        )
        runtime.active_tool_restriction = active_restriction
        if isinstance(messages, list):
            from kernel_action_menu import synchronize_tool_instructions
            messages[:] = synchronize_tool_instructions(
                messages, schemas, active_restriction)
        if route == "submit" and "done" not in names:
            _reconcile_terminal_text_without_done(
                runtime, messages, tools, schemas, names,
                request_id=current_id, admitted_repair=True)
        route, route_error = _contract_route(
            runtime, messages, names, provider_request=provider_request)
    if route_error:
        _contract_error(runtime, route_error, request_id=current_id,
                        boundary="owner", route=route or None, tool_names=names)
    if route == "submit" and not preserve_input_surface:
        deferred_submit_tools = _deferred_source_submit_tools(runtime, schemas)
        deferred_read_surface = deferred_submit_tools is not None
        if deferred_read_surface:
            schemas = deferred_submit_tools
            names = [_contract_tool_name(tool) for tool in schemas]
            if isinstance(tools, list):
                tools[:] = copy.deepcopy(schemas)
            runtime.completion_menu = tuple(names)
            runtime.offered_tools = tuple(names)
            runtime._question_done_available = True
            runtime._completion_owner_dispatch_tools = copy.deepcopy(schemas)
            runtime._completion_owner_actionable = False
            _mark_owner_done_selection(runtime, "deferred_source_submit")
        preserved_terminal_menu = _preserve_zero_edit_terminal_menu(runtime, schemas)
        if preserved_terminal_menu is not None:
            schemas = preserved_terminal_menu
            names = [_contract_tool_name(tool) for tool in schemas]
            runtime.completion_menu = tuple(names)
            runtime.offered_tools = tuple(names)
            runtime._question_done_available = True
            runtime._completion_owner_dispatch_tools = copy.deepcopy(schemas)
            runtime._completion_owner_actionable = True
        if not (deferred_read_surface or preserved_terminal_menu is not None
                or (_no_host_checks(runtime) and any(name != "done" for name in names))):
            schemas = [tool for tool in schemas if _contract_tool_name(tool) == "done"]
            names = [_contract_tool_name(tool) for tool in schemas]
    floor_contract = (
        None if preserve_input_surface else
        _apply_capability_floor_to_request_contract(
            runtime, messages, tools, schemas, provider_request=provider_request,
            request_id=current_id,
        )
    )
    if floor_contract is not None:
        schemas, names, route, route_error = floor_contract
        if route_error:
            _contract_error(
                runtime, route_error, request_id=current_id,
                boundary="owner", route=route or None, tool_names=names,
            )
    _scrub_owner_off_menu_tool_names(runtime, messages, names)
    context, missing_context = _contract_context_status(runtime, messages, names)
    if missing_context:
        _contract_error(
            runtime,
            "required action context is absent and has no retained acquisition route: "
            + ", ".join(missing_context),
            request_id=current_id,
            boundary="owner",
            route=route,
            tool_names=names,
        )
    unresolved = _contract_unresolved_work(runtime)
    terminal_line = (
        NO_HOST_CHECKS_TERMINAL_LINE
        if route == "submit" and _no_host_checks(runtime)
        else _contract_terminal_line(runtime)
        or "Final step: call done now to submit this revision (owner-admitted submit route)."
        if route == "submit" else None
    )
    if (_terminal_line_requires_wire(terminal_line)
            and not _framefit5_terminal_line_present(messages, str(terminal_line))):
        _contract_error(
            runtime,
            "serialized terminal owner line is absent from the provider request",
            request_id=current_id,
            boundary="owner",
            route=route,
            tool_names=names,
        )
    record_owner_block(
        runtime,
        "menu",
        schemas,
        block_id=f"menu:{current_id or _digest(schemas)}",
        request_id=current_id,
    )
    terminal_text = _contract_terminal_text(runtime) or route == "submit"
    contract: dict[str, object] = {
        "schema": REQUEST_CONTRACT_SCHEMA,
        "request_id": current_id,
        "revision": _contract_revision(runtime),
        "unresolved_work": unresolved,
        "route": route,
        "admitted_tool_schemas": schemas,
        "advertised_actions": list(names),
        "required_context": context,
        "required_context_digest": _digest(_contract_context_rows(runtime)),
        "budget_generation": _contract_budget_generation(runtime),
        "terminal_text": terminal_text,
        "terminal_line": terminal_line,
        "host_stop": route == "host-stop",
        "owner_blocks": _owner_block_records(runtime),
        "terminal_block_id": (
            _owner_block(runtime, "terminal_line") or {}
        ).get("id"),
    }
    evidence_pin_digest = _contract_evidence_pin(runtime)
    if evidence_pin_digest is not None:
        evidence_pin_value = _contract_evidence_pin_value(runtime)
        wire_pin, wire_pin_count = _wire_evidence_pin(messages)
        # Host state is not delivery proof: the exact typed pin must survive
        # into serialized provider messages, exactly once.  Legacy synthetic
        # records without EVIDENCEPIN markers remain compatible; live pins
        # always carry the marker and therefore use the strict path.
        if (evidence_pin_value and _evidence_marker_present(evidence_pin_value)
                and (wire_pin_count != 1 or wire_pin != evidence_pin_value)):
            _contract_error(
                runtime,
                "typed evidence pin is absent or changed on the serialized wire"
                f" (wire_count={wire_pin_count};"
                f" wire_digest={_digest([wire_pin]) if wire_pin else 'none'};"
                f" typed_digest={_digest([evidence_pin_value])})",
                request_id=current_id,
                boundary="owner",
                route=route,
                tool_names=names,
            )
        contract["evidence_pin_schema"] = "stateframe-evidence-pin/1"
        contract["evidence_pin_digest"] = evidence_pin_digest
        if wire_pin is not None:
            contract["evidence_pin_wire_digest"] = _digest([wire_pin])
    budget_generation = contract.get("budget_generation")
    budget_object = getattr(runtime, "continuation_limits", None)
    if budget_object is None:
        budget_object = getattr(runtime, "decision_budget", None)
    if (budget_generation is not None
            and callable(getattr(budget_object, "render_continuation_limits", None))):
        wire_limits = _wire_continuation_limits(messages)
        if not wire_limits:
            _contract_error(
                runtime,
                "continuation limits are absent from the serialized wire",
                request_id=current_id,
                boundary="owner",
                route=route,
                tool_names=names,
            )
        contract["budget_wire_digest"] = _digest(wire_limits)
    probe_schema = next(
        (tool for tool in schemas if _contract_tool_name(tool) == "probe"), None)
    if probe_schema is not None:
        contract["evidence_operation"] = {
            "schema": "stateframe-clause-witness-contract/1",
            "owner": REQUEST_CONTRACT_SCHEMA,
            "tool": "probe",
            "modes": ["clause_review_propose", "clause_review_assess"],
            "dispatch": "kernel_clause_review",
            "receipt": "host-observed execution receipt",
            "assessment": "model references returned receipt IDs",
        }
    contract["tool_schema_digest"] = _digest(schemas)
    contract["decision_digest"] = _contract_decision_digest(contract)
    setattr(runtime, "_completion_owner_request_contract", _freeze_contract(contract))
    dispatch_names = tuple(
        _contract_tool_name(tool)
        for tool in _sequence(getattr(runtime, "_completion_owner_dispatch_tools", ()))
        if isinstance(tool, Mapping)
    )
    if (_green_open_work(runtime) and route == "repair"
            and any(name in {"edit", "str_replace", "read", "run_tests"} for name in names)
            and tuple(names) == tuple(getattr(runtime, "completion_menu", ()) or ())
            and tuple(names) == dispatch_names):
        _emit(runtime, "owner_green_open_repair_kept",
              request_id=current_id, open_ids=list(open_ids(runtime)),
              advertised_actions=list(names), dispatch_actions=list(dispatch_names),
              provider_request=provider_request)
    if (route == "submit" and no_done_only_collapse_enabled(runtime)
            and "done" in names and any(name != "done" for name in names)):
        marker = {"request_id": current_id, "actions": list(names)}
        runtime._completion_owner_no_done_only_collapse_wire_pending = marker
        applied_key = (current_id, tuple(names))
        if getattr(runtime, "_completion_owner_no_done_only_collapse_applied_key", None) != applied_key:
            runtime._completion_owner_no_done_only_collapse_applied_key = applied_key
            _emit(runtime, "no_done_only_collapse_applied",
                  request_id=current_id,
                  advertised_actions=list(names), mutation_count=0)
    _emit(
        runtime,
        "provider_boundary_contract_decided",
        request_id=current_id,
        route=route,
        revision=contract["revision"],
        unresolved_work=unresolved,
        advertised_actions=list(names),
        required_context=context,
        evidence_pin_present=evidence_pin_digest is not None,
        evidence_pin_digest=evidence_pin_digest,
        evidence_operation=contract.get("evidence_operation"),
        decision_digest=contract["decision_digest"],
    )
    value = _thaw_contract(getattr(runtime, "_completion_owner_request_contract"))
    return value if isinstance(value, dict) else {}


def _bind_final_menu_owner_block(
        runtime: object,
        contract: dict[str, object],
        schemas: list[object],
        request_id: str,
) -> bool:
    # Lean contracts intentionally carry no typed owner-block snapshot.
    # Their exact admitted schemas are still rebound below and checked at wire.
    if contract.get("contract_kind") == LEAN_REQUEST_CONTRACT_KIND:
        return True
    owner_blocks = contract.get("owner_blocks")
    if (not isinstance(owner_blocks, Mapping)
            or not isinstance(owner_blocks.get("menu"), Mapping)):
        return False
    menu_block = owner_blocks["menu"]
    if menu_block.get("digest") == _digest(schemas):
        return True
    # N4 authored this exact transport surface. Rebind its typed menu block while
    # preserving all other owner-block snapshots for strict drift checks.
    menu_record = record_owner_block(
        runtime, "menu", schemas,
        block_id=f"menu:{request_id or _digest(schemas)}",
        request_id=request_id or None,
    )
    refreshed_blocks = dict(owner_blocks)
    refreshed_blocks["menu"] = menu_record
    contract["owner_blocks"] = refreshed_blocks
    return True


def bind_final_tool_surface(
        runtime: object,
        capture: object,
        surface: Mapping[str, object],
        *,
        messages: object,
) -> bool:
    """Bind final request schemas and recompute any route derived from that menu."""
    schemas = surface.get("tools")
    names = surface.get("names")
    surface_digest = surface.get("tool_schema_digest")
    request_id = str(surface.get("request_id", "") or "")
    if (not isinstance(schemas, list)
            or not isinstance(names, (list, tuple))
            or not isinstance(surface_digest, str)
            or surface_digest != _digest(schemas)):
        return False
    normalized_names = [str(name) for name in names]
    if (not normalized_names or any(not name for name in normalized_names)
            or len(normalized_names) != len(set(normalized_names))):
        return False

    stored = getattr(runtime, "_completion_owner_request_contract", None)
    captured = getattr(capture, "request_action_contract", None)
    source = stored if isinstance(stored, Mapping) else captured
    if not isinstance(source, Mapping):
        return True
    try:
        _verify_contract_integrity(source)
        contract = _thaw_contract(source)
        if not isinstance(contract, dict):
            return False
        prior_request_id = contract.get("request_id")
        if (request_id and prior_request_id not in (None, "", request_id)):
            return False
        if prior_request_id in (None, "") and request_id:
            contract["request_id"] = request_id

        if contract.get("contract_kind") == LEAN_REQUEST_CONTRACT_KIND:
            if contract.get("route") != "lean":
                return False
        else:
            route, route_error = _contract_route(
                runtime, messages, normalized_names, provider_request=True,
            )
            if route_error:
                return False
            terminal_line = contract.get("terminal_line")
            if (_terminal_line_requires_wire(terminal_line)
                    and not _framefit5_terminal_line_present(messages, str(terminal_line))):
                return False
            contract["route"] = route
            contract["host_stop"] = route == "host-stop"
            contract["terminal_text"] = _contract_terminal_text(runtime) or route == "submit"

        if not _bind_final_menu_owner_block(runtime, contract, schemas, request_id):
            return False

        contract["admitted_tool_schemas"] = copy.deepcopy(schemas)
        contract["advertised_actions"] = normalized_names
        contract["tool_schema_digest"] = _digest(schemas)
        revision_mode = _request_contract_revision_rebind_mode(runtime)
        legacy_contract = copy.deepcopy(contract)
        legacy_contract["decision_digest"] = _contract_decision_digest(legacy_contract)
        try:
            rebound_contract = copy.deepcopy(legacy_contract)
            rebound_contract["revision"] = _contract_revision(runtime)
            rebound_contract["unresolved_work"] = _contract_unresolved_work(runtime)
            rebound_contract["decision_digest"] = _contract_decision_digest(rebound_contract)
        except Exception:
            contract = legacy_contract
            if revision_mode == "shadow":
                _emit_request_contract_revision_rebind_shadow_mismatch(
                    runtime, request_id, "rebind_error",
                )
        else:
            if revision_mode == "shadow":
                for field in ("revision", "unresolved_work", "decision_digest"):
                    if legacy_contract.get(field) != rebound_contract.get(field):
                        _emit_request_contract_revision_rebind_shadow_mismatch(
                            runtime, request_id, field,
                        )
            # Final-surface binding is a correctness boundary, not a feature flag.
            contract = rebound_contract
        frozen = _freeze_contract(contract)
        setattr(runtime, "_completion_owner_request_contract", frozen)
        setattr(capture, "request_action_contract", copy.deepcopy(contract))
        setattr(runtime, "_completion_owner_dispatch_tools", copy.deepcopy(schemas))
        if contract.get("contract_kind") != LEAN_REQUEST_CONTRACT_KIND:
            setattr(runtime, "completion_menu", tuple(normalized_names))
            setattr(runtime, "offered_tools", tuple(normalized_names))
            setattr(runtime, "_question_done_available", "done" in normalized_names)
        return True
    except (AttributeError, TypeError, ValueError):
        return False


def assert_request_action_contract(
        runtime: object | None,
        messages: object,
        tools: object,
        *,
        request_id: str | None = None,
        provider_request: bool = True,
        boundary: str = "provider",
        allow_host_stop: bool = False,
        frozen_contract: Mapping[str, object] | None = None,
        read_only: bool = False,
) -> dict[str, object]:
    """Fail closed if presentation, serialization, or dispatch drifted."""
    if isinstance(frozen_contract, Mapping):
        contract = _thaw_contract(frozen_contract)
    elif runtime is not None:
        stored_contract = getattr(runtime, "_completion_owner_request_contract", None)
        if not isinstance(stored_contract, Mapping):
            raise RequestActionContractError(
                "request action contract was not finalized before assertion")
        contract = _thaw_contract(stored_contract)
    else:
        raise RequestActionContractError(
            "request action contract was not finalized before assertion")
    if not isinstance(contract, dict):
        raise RequestActionContractError("finalized request contract has an invalid shape")
    lean_contract = contract.get("contract_kind") == LEAN_REQUEST_CONTRACT_KIND
    _verify_contract_integrity(contract)
    if (lean_contract and request_id is not None
            and contract.get("request_id") != request_id):
        message = "lean request action contract identity changed"
        if runtime is None:
            raise RequestActionContractError(message)
        _contract_error(runtime, message, request_id=request_id, boundary=boundary,
                        route="lean", tool_names=contract.get("advertised_actions", ()))
    raw_tools = (
        list(tools)
        if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray))
        else None
    )
    expected = [str(name) for name in contract.get("advertised_actions", ())]
    if read_only:
        projected = copy.deepcopy(messages)
        _scrub_owner_off_menu_tool_names(runtime, projected, expected)
        if projected != messages:
            raise RequestActionContractError("sealed owner presentation requires an off-menu rewrite")
    else:
        _scrub_owner_off_menu_tool_names(runtime, messages, expected)
    if raw_tools is None or any(not isinstance(tool, Mapping) for tool in raw_tools):
        message = "serialized tool surface is not a valid admitted schema list"
        if runtime is None:
            raise RequestActionContractError(message)
        _contract_error(runtime, message, request_id=request_id, boundary=boundary,
                        route=str(contract.get("route", "")), tool_names=expected)
    schemas = [copy.deepcopy(dict(tool)) for tool in raw_tools]
    names = [_contract_tool_name(tool) for tool in schemas]
    admitted_schemas = contract.get("admitted_tool_schemas")
    if (schemas != admitted_schemas
            or names != expected
            or _digest(schemas) != contract.get("tool_schema_digest")):
        if runtime is None:
            raise RequestActionContractError(
                "advertised actions differ from serialized/admitted tool schemas")
        _contract_error(
            runtime, "advertised actions differ from serialized/admitted tool schemas",
            request_id=request_id, boundary=boundary,
            route=str(contract.get("route", "")), tool_names=names)
    route_surface_error = None if lean_contract else _completion_route_surface_error(
        runtime, names, request_id=request_id or contract.get("request_id"),
    )
    if route_surface_error and _route_surface_strict():
        _contract_error(runtime, route_surface_error, request_id=request_id,
                        boundary=boundary,
                        route=str(contract.get("route", "")), tool_names=names)
    current_revision = _contract_revision(runtime) if runtime is not None else None
    if runtime is not None and (
            current_revision == "UNKNOWN" or current_revision != contract.get("revision")):
        _contract_error(
            runtime,
            "request action contract revision is stale",
            request_id=request_id,
            boundary=boundary,
            route=str(contract.get("route", "")),
            tool_names=names,
        )
    if runtime is not None and not lean_contract:
        current_route, route_error = _contract_route(
            runtime, messages, names, provider_request=provider_request)
        if route_error or current_route != contract.get("route"):
            message = route_error or "request action contract route drifted"
            _contract_error(
                runtime, message, request_id=request_id, boundary=boundary,
                route=str(contract.get("route", "")), tool_names=names)
    if runtime is not None and lean_contract:
        source_scope = _lean_source_scope(runtime)
        if (source_scope != contract.get("source_scope")
                or _digest(source_scope) != contract.get("source_scope_digest")):
            _contract_error(
                runtime, "lean editable source scope changed after preparation",
                request_id=request_id, boundary=boundary,
                route="lean", tool_names=names,
            )
    expected_context_digest = contract.get("required_context_digest")
    if runtime is not None and expected_context_digest != _digest(_contract_context_rows(runtime)):
        _contract_error(
            runtime,
            "required action context drifted from the owner contract",
            request_id=request_id,
            boundary=boundary,
            route=str(contract.get("route", "")),
            tool_names=names,
        )
    if runtime is not None and lean_contract:
        context, missing_context = _contract_context_status(runtime, messages, names)
        if missing_context or context != contract.get("required_context"):
            _contract_error(
                runtime, "required action context is absent or changed on the lean wire",
                request_id=request_id, boundary=boundary,
                route="lean", tool_names=names,
            )
    if runtime is not None:
        current_budget_generation = _contract_budget_generation(runtime)
        if current_budget_generation != contract.get("budget_generation"):
            _contract_error(
                runtime,
                "budget generation drifted from the owner contract",
                request_id=request_id,
                boundary=boundary,
                route=str(contract.get("route", "")),
                tool_names=names,
            )
    expected_owner_blocks = contract.get("owner_blocks", {})
    current_owner_blocks = (
        _owner_block_records(runtime) if runtime is not None
        else expected_owner_blocks
    )
    if current_owner_blocks != expected_owner_blocks:
        message = "typed owner block records drifted from the owner contract"
        if runtime is None:
            raise RequestActionContractError(message)
        _contract_error(
            runtime, message, request_id=request_id, boundary=boundary,
            route=str(contract.get("route", "")), tool_names=names)
    menu_block = (
        current_owner_blocks.get("menu")
        if isinstance(current_owner_blocks, Mapping) else None
    )
    if (isinstance(menu_block, Mapping)
            and menu_block.get("digest") != _digest(schemas)):
        message = "serialized menu drifted from the typed owner block"
        if runtime is None:
            raise RequestActionContractError(message)
        _contract_error(
            runtime, message, request_id=request_id, boundary=boundary,
            route=str(contract.get("route", "")), tool_names=names)
    expected_evidence_pin = contract.get("evidence_pin_digest")
    actual_evidence_pin = (
        _contract_evidence_pin(runtime) if runtime is not None else expected_evidence_pin
    )
    if ("evidence_pin_digest" in contract
            or actual_evidence_pin is not None) and actual_evidence_pin != expected_evidence_pin:
        message = "typed evidence pin drifted from the owner contract"
        if runtime is None:
            raise RequestActionContractError(message)
        _contract_error(
            runtime, message, request_id=request_id, boundary=boundary,
            route=str(contract.get("route", "")), tool_names=names)
    expected_wire_pin = contract.get("evidence_pin_wire_digest")
    if expected_wire_pin is not None:
        wire_pin, wire_pin_count = _wire_evidence_pin(messages)
        actual_wire_pin = _digest([wire_pin]) if wire_pin is not None and wire_pin_count == 1 else None
        if actual_wire_pin != expected_wire_pin:
            message = "serialized evidence pin was removed or changed"
            if runtime is None:
                raise RequestActionContractError(message)
            _contract_error(
                runtime, message, request_id=request_id, boundary=boundary,
                route=str(contract.get("route", "")), tool_names=names)
    expected_wire_budget = contract.get("budget_wire_digest")
    if expected_wire_budget is not None:
        wire_limits = _wire_continuation_limits(messages)
        actual_wire_budget = _digest(wire_limits) if wire_limits is not None else None
        if actual_wire_budget != expected_wire_budget:
            message = "serialized continuation limits were removed or changed"
            if runtime is None:
                error = RequestActionContractError(message)
                error.boundary_recovery_eligible = False
                raise error
            _contract_error(
                runtime, message, request_id=request_id, boundary=boundary,
                route=str(contract.get("route", "")), tool_names=names,
                boundary_recovery_eligible=False)
    expected_terminal_line = str(contract.get("terminal_line") or "").strip()
    if (_terminal_line_requires_wire(expected_terminal_line)
            and not _framefit5_terminal_line_present(messages, expected_terminal_line)):
        message = "serialized terminal owner line was removed or changed"
        if runtime is None:
            raise RequestActionContractError(message)
        _contract_error(
            runtime, message, request_id=request_id, boundary=boundary,
            route=str(contract.get("route", "")), tool_names=names)
    if (not names) and not bool(contract.get("host_stop")):
        message = "a model-action request cannot have an empty tool list"
        if runtime is None:
            raise RequestActionContractError(message)
        _contract_error(
            runtime, message, request_id=request_id, boundary=boundary,
            route=str(contract.get("route", "")), tool_names=names)
    if contract.get("host_stop") and not allow_host_stop:
        if runtime is None:
            raise RequestActionContractError(
                "host-stop outcome cannot be serialized as a model-action request")
        _contract_error(
            runtime, "host-stop outcome cannot be serialized as a model-action request",
            request_id=request_id, boundary=boundary, route="host-stop", tool_names=names)
    if bool(contract.get("terminal_text")) and "done" not in names:
        if runtime is None:
            raise RequestActionContractError(
                "terminal text was serialized without an admitted done tool")
        _contract_error(
            runtime, "terminal text was serialized without an admitted done tool",
            request_id=request_id, boundary=boundary,
            route=str(contract.get("route", "")), tool_names=names)
    if runtime is None:
        return contract
    key = (str(request_id), boundary)
    seen = getattr(runtime, "_completion_owner_request_contract_assertions", set())
    if not isinstance(seen, set):
        seen = set()
        setattr(runtime, "_completion_owner_request_contract_assertions", seen)
    if key not in seen:
        seen.add(key)
        _emit(
            runtime,
            "provider_boundary_contract_asserted",
            request_id=request_id,
            boundary=boundary,
            route=contract.get("route"),
            advertised_actions=expected,
            serialized_actions=names,
            admitted=True,
            decision_digest=contract.get("decision_digest"),
        )
    return copy.deepcopy(contract)


def request_action_contract(runtime: object) -> dict[str, object] | None:
    """Return the frozen decision for projections without giving them ownership."""
    contract = getattr(runtime, "_completion_owner_request_contract", None)
    value = _thaw_contract(contract) if isinstance(contract, Mapping) else None
    return value if isinstance(value, dict) else None


def request_action_presentation_contract(
        runtime: object, tools: object,
        messages: object = None) -> dict[str, object] | None:
    """Expose the owner's already-selected route to earlier text projections.

    The authoritative contract is intentionally frozen at the final prepared
    boundary.  Frame hygiene runs before that boundary, so it needs a
    read-only view of the pending done handoff to avoid stripping its text.
    This helper derives only from the owner's reserved state and current
    admitted menu; it never caches, mutates, or decides a request contract.
    """
    state = _state(runtime)
    pending = bool(
        state.get("final_submit_reserved")
        and state.get("final_submit_request_pending"))
    frozen = request_action_contract(runtime)
    if frozen is not None:
        frozen_names = [str(name) for name in frozen.get("advertised_actions", ())]
        frozen_terminal_done = (
            str(frozen.get("route", "")) == "submit"
            and bool(frozen.get("terminal_text"))
            and _reserved_terminal_menu_names(runtime, frozen_names)
        )
        if (not pending or not frozen_terminal_done
                or not _pending_deferred_source_paths(runtime)):
            return frozen
    owner_selected = _contract_tools(
        getattr(runtime, "_completion_owner_dispatch_tools", None))
    selected = owner_selected or _contract_tools(tools)
    if pending:
        candidates = [*owner_selected, *_contract_tools(tools),
                      *_contract_tools(getattr(runtime, "_completion_owner_full_tools", ())) ]
        done = next((tool for tool in candidates if _contract_tool_name(tool) == "done"), None)
        if done is not None:
            selected = _terminal_presentation_tools(runtime, candidates, done)
    selected_names = [_contract_tool_name(tool) for tool in selected]
    if not pending or not _reserved_terminal_menu_names(runtime, selected_names):
        return None
    if not _contract_terminal_text(runtime):
        fallback = (
            NO_HOST_CHECKS_TERMINAL_LINE if _no_host_checks(runtime) else
            "Final step: call done now to submit this revision (owner-admitted submit route)."
        )
        record_owner_block(runtime, "terminal_line", fallback, block_id="terminal:pending")
    terminal_record = _owner_block(runtime, "terminal_line") or {}
    terminal_line = (
        NO_HOST_CHECKS_TERMINAL_LINE if _no_host_checks(runtime)
        else _contract_terminal_line(runtime)
    )
    return {
        "schema": REQUEST_CONTRACT_SCHEMA,
        "route": "submit",
        "admitted_tool_schemas": selected,
        "advertised_actions": selected_names,
        "terminal_text": True,
        "terminal_line": terminal_line,
        "host_stop": False,
        "owner_blocks": _owner_block_records(runtime),
        "terminal_block_id": terminal_record.get("id"),
    }


def is_reserved_request(runtime: object, messages: object, tools: object) -> bool:
    """Recognize only the owner-owned terminal request at the provider boundary."""
    if not request_pending(runtime):
        return False
    if not _contract_terminal_text(runtime):
        return False
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return False
    tool_names = [
        str(tool.get("function", {}).get("name", ""))
        for tool in tools
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
    ]
    return _reserved_terminal_menu_names(runtime, tool_names)


def note_request_submitted(runtime: object, *, request_id: str | None = None,
                           step: int | None = None, body_sha256: str | None = None,
                           tool_names: Sequence[str] = (),
                           preserve_owner_blocks: bool = False) -> None:
    """Close the prepared-request handoff after the wire event is durable."""
    if not enabled(runtime):
        return
    state = _state(runtime)
    if not state["final_submit_request_pending"]:
        return
    expected_request_id = getattr(runtime, "_completion_owner_pending_request_id", None)
    if (expected_request_id is not None and request_id is not None
            and request_id != expected_request_id):
        return
    wire_marker = getattr(runtime, "_completion_owner_no_done_only_collapse_wire_pending", None)
    if (isinstance(wire_marker, Mapping)
            and wire_marker.get("request_id") == request_id
            and isinstance(body_sha256, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", body_sha256) is not None
            and tuple(str(name) for name in tool_names) == tuple(wire_marker.get("actions", ()))
            and no_done_only_collapse_enabled(runtime)):
        _emit(runtime, "no_done_only_collapse_fired",
              request_id=request_id, step=step, body_sha256=body_sha256,
              advertised_actions=list(tool_names), on_wire=True)
        runtime._completion_owner_no_done_only_collapse_wire_pending = None
    state["final_submit_request_pending"] = False
    runtime._completion_owner_pending_request_id = None
    _emit(runtime, "completion_owner_reserved_response_submitted",
          request_id=request_id, step=step,
          response_number=state["final_submit_response_count"])
    if preserve_owner_blocks:
        # The first physical attempt consumes the reservation, but the same
        # generation still authorizes an identical-byte transport retry.
        runtime._completion_owner_submitted_blocks_request_id = request_id
    else:
        setattr(runtime, "_completion_owner_owner_blocks", {})


def finish_request_submission(runtime: object, *, request_id: str) -> None:
    """Retire consumed owner presentation after all physical attempts finish."""
    if (runtime is not None
            and getattr(runtime, "_completion_owner_submitted_blocks_request_id", None) == request_id):
        runtime._completion_owner_owner_blocks = {}
        runtime._completion_owner_submitted_blocks_request_id = None


def charge_response(runtime: object, budget: object, *, step: int | None = None) -> bool:
    """Charge normally, or consume one of two reserved responses after a hard cap."""
    if not callable(getattr(budget, "charge_response", None)):
        return False
    if budget.charge_response(step=step):
        if enabled(runtime) and _state(runtime)["final_submit_reserved"]:
            state = _state(runtime)
            if state["final_submit_response_count"] < MAX_RESERVED_RESPONSES:
                state["final_submit_response_count"] += 1
                state["final_submit_used"] = True
        return True
    if not enabled(runtime):
        return False
    state = _state(runtime)
    if (not state["final_submit_reserved"]
            or state["final_submit_response_count"] >= MAX_RESERVED_RESPONSES):
        return False
    # The owner reserved this response before the normal charge failed. Extend
    # the accounting ceiling by exactly one unit and leave an audit trail.
    if getattr(budget, "continuation_limits_schema", None) == "stateframe-continuation-limits/1":
        return False
    if getattr(budget, "mode", "decision") != "response":
        return False
    if type(getattr(budget, "limit", None)) is not int:
        return False
    budget.limit += 1
    if hasattr(budget, "_response_reported"):
        budget._response_reported = False
    _emit(runtime, "completion_owner_reserved_response_released",
          step=step, response_budget=budget.limit,
          response_number=state["final_submit_response_count"] + 1,
          response_limit=MAX_RESERVED_RESPONSES)
    accepted = bool(budget.charge_response(step=step))
    if accepted:
        state["final_submit_response_count"] += 1
        state["final_submit_used"] = True
        state["restatement_pending"] = False
    return accepted


def note_model_action(runtime: object, action: str) -> None:
    """Record whether the first reserved response ignored the sole terminal action."""
    if not enabled(runtime):
        return
    normalized = str(action or "").strip().lower()
    # Witness capacity is settled by the admitted execution boundary.  A
    # parsed or otherwise ordinary ``probe`` action is not proof that a
    # clause was admitted, dispatched, or even runnable.
    if normalized in _STALL_RULE_EDIT_TOOLS:
        setattr(runtime, "_completion_owner_repair_edit_count",
                int(getattr(runtime, "_completion_owner_repair_edit_count", 0) or 0) + 1)
    state = _state(runtime)
    if not state["final_submit_reserved"] or not state["final_submit_response_count"]:
        return
    if normalized == "done":
        state["restatement_pending"] = False
        return
    if state["final_submit_response_count"] == 1:
        state["restatement_pending"] = True
        _emit(runtime, "completion_owner_non_done_response",
              action=normalized or "unknown", restatement="one_final_submit_response")


def handoff_churn_decision(runtime: object, decision: object) -> str:
    """Accept one budget decision without taking ownership of completion."""
    value = str(decision or "").strip().upper()
    if value not in _CHURN_DECISIONS:
        value = "STOP_INCOMPLETE"
    setattr(runtime, "_completion_owner_churn_decision", value)
    _emit(runtime, "completion_owner_churn_decision", decision=value)
    return value


def churn_decision(runtime: object) -> str:
    value = str(getattr(runtime, "_completion_owner_churn_decision", "ALLOW") or "ALLOW").upper()
    return value if value in _CHURN_DECISIONS else "STOP_INCOMPLETE"


def _apply_churn_decision(
        runtime: object,
        selected: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    decision = churn_decision(runtime)
    if decision == "ALLOW":
        return [copy.deepcopy(dict(tool)) for tool in selected]
    result = [copy.deepcopy(dict(tool)) for tool in selected if _tool_name(tool) != "done"]
    if decision == "RECOVERY_REQUIRED":
        try:
            from kernel_candidate_preservation import restore_available
            available = restore_available(runtime)
        except (ImportError, TypeError, ValueError):
            available = False
        if available and not any(_tool_name(tool) == "restore_candidate" for tool in result):
            restore = next((tool for tool in tools if _tool_name(tool) == "restore_candidate"), None)
            if isinstance(restore, Mapping):
                result.append(copy.deepcopy(dict(restore)))
    return result


def _dispatchable_owner_menu(
        runtime: object,
        selected: Sequence[Mapping[str, object]],
        full_tools: object,
        *, legal_actions: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Keep dispatch-admitted tools and retain other currently legal work routes."""
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for tool in selected:
        name = _tool_name(tool)
        if name and name not in seen and _action_dispatch_admissible(runtime, name, {}):
            result.append(copy.deepcopy(dict(tool)))
            seen.add(name)

    state = _state(runtime)
    result_names = tuple(_tool_name(tool) for tool in result)
    terminal_locked = bool(
        state.get("final_submit_reserved")
        or state.get("final_submit_request_pending")
        or (_reserved_terminal_menu_names(runtime, result_names)
            and getattr(runtime, "_question_done_available", False))
    )
    if terminal_locked:
        return result

    available = _available_tools(runtime, full_tools)
    actions = legal_actions if legal_actions is not None else legal_next_actions(
        runtime, tools=available,
    )
    admitted = _admitted_action_projections(
        runtime, actions, available, validate_arguments=False)
    by_name = {_tool_name(tool): tool for tool in available}
    for action in admitted:
        name = str(action.get("tool", "")).strip()
        if name in by_name and name not in seen:
            result.append(copy.deepcopy(by_name[name]))
            seen.add(name)
    return result


def _apply_capability_floor_to_owner_menu(
        runtime: object, selected: Sequence[Mapping[str, object]],
        current_tools: object = (),
) -> list[dict[str, object]]:
    from kernel_capability_floor import apply_floor, enabled as capability_floor_enabled

    if not capability_floor_enabled(runtime):
        return selected if isinstance(selected, list) else list(selected)
    if getattr(runtime, "system_outcome", None) == FINAL_SUBMIT_EXHAUSTED:
        return list(selected)

    sources = (
        current_tools,
        getattr(runtime, "_readiness_full_tools", ()),
        getattr(runtime, "_completion_owner_full_tools", ()),
        getattr(runtime, "available_tools", ()),
    )
    by_name: dict[str, Mapping[str, object]] = {}
    for source in sources:
        if isinstance(source, Mapping):
            items = source.values()
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            items = source
        else:
            continue
        for item in items:
            if isinstance(item, Mapping):
                name = _tool_name(item)
                if name:
                    by_name.setdefault(name, item)

    result = apply_floor(
        selected, list(by_name.values()),
        admissible=lambda name: capability_floor_action_admissible(runtime, name),
        runtime=runtime,
    )
    names = tuple(_tool_name(tool) for tool in result)
    runtime.completion_menu = names
    runtime.offered_tools = names
    runtime._question_done_available = "done" in names
    runtime._completion_owner_dispatch_tools = copy.deepcopy(result)
    runtime._completion_owner_actionable = bool(result)
    if isinstance(current_tools, list):
        current_tools[:] = copy.deepcopy(result)
    return result


def terminal_tool_menu(runtime: object, tools: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    # The done-selection stamp is request-scoped: only a selection made while
    # projecting THIS menu may claim authority over later projections.
    setattr(runtime, "_completion_owner_done_selection", None)
    setattr(runtime, "_completion_owner_host_stop_pending", False)
    from kernel_done_is_submit_authority import (
        candidate_exists as done_candidate_exists,
        enabled as done_submit_enabled,
        project_candidate_tools,
    )
    if done_submit_enabled(runtime) and done_candidate_exists(runtime):
        full_tools = (getattr(runtime, "_completion_owner_full_tools", None)
                      or tools)
        result = project_candidate_tools(runtime, tools, full_tools=full_tools)
        names = tuple(_tool_name(tool) for tool in result if _tool_name(tool))
        runtime.completion_menu = names
        runtime.offered_tools = names
        runtime._question_done_available = "done" in names
        runtime._completion_owner_dispatch_tools = copy.deepcopy(result)
        if isinstance(tools, list):
            tools[:] = copy.deepcopy(result)
        # (N4 menu == wire): under DONE_IS_SUBMIT this projection IS the menu authority for the
        # candidate turn. The green lock had frozen a different menu (probe/recall/where_defined ...), so wire truth
        # blocked GREEN turns with schema_authority_mismatch:green_lock_frozen_menu. Re-freeze to these tools.
        try:
            from kernel_greenlock_menu_authority import finalize_wire_menu_authority
            finalize_wire_menu_authority(runtime, result)
        except Exception:  # noqa: BLE001 - authority bookkeeping must not break menu composition
            pass
        return result
    floor_available_tools = copy.deepcopy(_contract_tools(tools))
    selected = _terminal_tool_menu_impl(runtime, tools)

    def finish(menu: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        result = _apply_capability_floor_to_owner_menu(runtime, menu, floor_available_tools)
        from kernel_done_readiness_authority import (
            attach_readiness_decision, decision_for_runtime,
            enabled as done_authority_enabled, project_menu,
            uncertified_host_debt_enabled,
        )
        if (done_authority_enabled(runtime)
                or (uncertified_host_debt_enabled()
                    and callable(getattr(runtime, "pass_gates_vector", None)))):
            vector = runtime.pass_gates_vector(floor_available_tools)
            readiness = getattr(runtime, "_readiness_decision", None)
            readiness = attach_readiness_decision(
                runtime, readiness, vector=vector, full_tools=floor_available_tools,
            ) if readiness is not None else None
            decision = decision_for_runtime(
                runtime, readiness=readiness, vector=vector,
                full_tools=floor_available_tools,
            )
            projected_menu = None
            if decision is not None:
                from kernel_done_readiness_authority import (
                    DoneReadinessContractError, abandon_decision,
                )
                try:
                    projected_menu = project_menu(
                        decision, result, floor_available_tools,
                        capability_admissible=lambda name: capability_floor_action_admissible(
                            runtime, name),
                    )
                except DoneReadinessContractError as error:
                    abandon_decision(runtime, decision, error)
            if projected_menu is not None:
                result = projected_menu
                names = tuple(_tool_name(tool) for tool in result)
                runtime.completion_menu = names
                runtime.offered_tools = names
                runtime._question_done_available = "done" in names
                runtime._completion_owner_dispatch_tools = copy.deepcopy(result)
                if isinstance(tools, list):
                    tools[:] = copy.deepcopy(result)
        if not result:
            from kernel_terminal_policy import (
                enabled as hard_budget_only_enabled,
                queue_advisory,
            )
            if hard_budget_only_enabled(runtime):
                queue_advisory(runtime, "completion_owner_exhausted")
                setattr(runtime, "_completion_owner_host_stop_pending", True)
        else:
            setattr(runtime, "_completion_owner_host_stop_pending", False)
        from kernel_capability_floor import enabled as capability_floor_enabled
        if capability_floor_enabled(runtime) and isinstance(tools, list):
            tools[:] = copy.deepcopy(result)
        return result

    selected = _apply_churn_decision(runtime, selected, tools)
    preserved_terminal_menu = _preserve_zero_edit_terminal_menu(runtime, selected)
    if preserved_terminal_menu is not None:
        selected = preserved_terminal_menu
        runtime._completion_owner_actionable = True
        _mark_owner_done_selection(runtime, "zero_edit_terminal_menu_preserved")
    # IDEALPATH owns the live stage menu.  Completion-owner projections may
    # narrow repair actions, but they must not silently remove the reducer's
    # current typed question/backtrack route before the provider boundary.
    idealpath_project = getattr(runtime, "_idealpath_project_menu", None)
    if callable(idealpath_project):
        selected = idealpath_project(selected)
    selected = _preserve_open_requirement_route(runtime, selected, tools)
    # Preservation is an enricher, not a second authority.  Re-run the frozen
    # IDEALPATH intersection so a route appended here cannot bypass admission
    # or resurrect a stale stage exit.
    if callable(idealpath_project):
        selected = idealpath_project(selected)
    selected = _dispatchable_owner_menu(runtime, selected, tools)
    runtime.completion_menu = tuple(_tool_name(tool) for tool in selected)
    runtime._question_done_available = "done" in runtime.completion_menu
    runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
    if isinstance(tools, list):
        tools[:] = copy.deepcopy(selected)
    # The completion owner remains the only menu authority, but it delegates
    # evidence freshness to the live dependency graph before exposing ``done``.
    try:
        from kernel_invalidation import runtime_completion_gate
        dependency_finding = runtime_completion_gate(runtime)
    except Exception:  # noqa: BLE001 - a missing graph must not expose done
        dependency_finding = {
            "admitted": False,
            "reason_code": "DEPGRAPH_ERROR",
            "finding": "dependency graph could not be evaluated",
            "next_action": "re-run the current public check before completion",
        }
    if dependency_finding.get("admitted") is False:
        selected = [tool for tool in selected if _tool_name(tool) != "done"]
        runtime.completion_menu = tuple(_tool_name(tool) for tool in selected)
        runtime._question_done_available = False
        runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
        runtime._invalidation_gate = copy.deepcopy(dependency_finding)
        if isinstance(tools, list):
            tools[:] = copy.deepcopy(selected)
        return finish(selected)
    if (enabled(runtime) and _product_parent_unchanged(
            runtime, reason="completion_owner_terminal_menu")
            and not _green_open_work(runtime)):
        _clear_terminal_offer_state(runtime)
        return finish(selected)
    if enabled(runtime):
        state = _state(runtime)
        from kernel_capability_floor import enabled as capability_floor_enabled

        # The legacy done-only cap is advisory once the floor owns the live
        # surface; do not turn its empty-menu result into an episode terminal.
        if (state["done_only_offer_count"] >= MAX_RESERVED_RESPONSES
                and not capability_floor_enabled(runtime)):
            if not selected:
                from kernel_terminal_policy import (
                    enabled as hard_budget_only_enabled,
                    queue_advisory,
                )
                if hard_budget_only_enabled(runtime):
                    queue_advisory(runtime, "owner_reserved_submit_cap")
                else:
                    runtime.system_outcome = FINAL_SUBMIT_EXHAUSTED
                runtime.completion_menu = ()
                record_owner_block(
                    runtime, "menu", selected,
                    block_id="menu:" + _digest(_contract_tools(selected)),
                )
                return finish(selected)
        # Keep the public owner boundary self-contained.  A private projection
        # may filter every selected tool after a live action was admitted; in
        # that case rebuild only the same admitted work route instead of
        # allowing the next contract seam to observe an empty menu.
        actions = legal_next_actions(runtime, tools=selected) if not selected else ()
        has_actions = bool(actions)
        if actions:
            action_names = {str(action.get("tool", "")) for action in actions}
            if action_names and not selected:
                selected = list(_admitted_action_projections(runtime, actions, tools))
            if has_actions and not selected:
                record_owner_block(
                    runtime, "menu", selected,
                    block_id="menu:" + _digest(_contract_tools(selected)),
                )
                return finish(selected)
        if _decision_codes(runtime, None) and not has_actions and not selected:
            runtime.completion_menu = ()
            record_owner_block(
                runtime, "menu", selected,
                block_id="menu:" + _digest(_contract_tools(selected)),
            )
            return finish(selected)
        selected_names = tuple(_tool_name(tool) for tool in selected)
        if "done" in selected_names:
            runtime.completion_menu = (
                ("done",) if selected_names == ("done",) else selected_names
            )
        _ensure_terminal_owner_block(runtime, selected)
        record_owner_block(
            runtime,
            "menu",
            selected,
            block_id="menu:" + _digest(_contract_tools(selected)),
        )
        # 10-06 batch 839: when no owner branch narrowed the menu, the dispatch surface was left
        # unregistered, so kernel_request_budget saw no protected completion surface and applied
        # its ordinary 5-tool cap -> every wire offered only {authored_check, done, edit,
        # file_create, read}; str_replace and run_tests were never offered and the frozen
        # "repair" contract perpetuated it.  The owner-selected legal set IS the dispatch
        # surface: register it so the budget protects it (as it did, accidentally, while the
        # gate vector was crashing).
        if selected and not isinstance(getattr(runtime, "_completion_owner_dispatch_tools", None), list):
            runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
        elif selected and {_tool_name(tool) for tool in selected} != {
                _tool_name(tool) for tool in runtime._completion_owner_dispatch_tools}:
            runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
    return finish(selected)


def _host_debt_menu_projection(
        runtime: object, selected: Sequence[Mapping[str, object]],
        full_tools: Sequence[Mapping[str, object]], vector: object,
) -> tuple[object, list[dict[str, object]]] | None:
    from kernel_done_readiness_authority import (
        DoneReadinessContractError, abandon_decision, decision_for_runtime,
        project_menu as project_done_menu, uncertified_host_debt_enabled,
    )

    if not uncertified_host_debt_enabled():
        return None
    decision = decision_for_runtime(runtime, vector=vector, full_tools=full_tools)
    if decision is None or decision.decision not in {"REFUSED", "ACCEPTED_UNCERTIFIED"}:
        return None
    try:
        projected = project_done_menu(
            decision, selected, full_tools,
            offer_done=decision.decision == "ACCEPTED_UNCERTIFIED",
            capability_admissible=lambda name: capability_floor_action_admissible(runtime, name),
        )
    except (AssertionError, DoneReadinessContractError) as error:
        abandon_decision(runtime, decision, error)
        return None
    return decision, projected


def _apply_host_debt_terminal_menu(
        runtime: object, tools: Sequence[Mapping[str, object]],
        selected: list[dict[str, object]], vector: object, gate_projector: Callable,
) -> list[dict[str, object]] | None:
    from kernel_done_readiness_authority import uncertified_host_debt_enabled

    if not uncertified_host_debt_enabled():
        return None
    if getattr(vector, "ready", False) is not True:
        selected = gate_projector(selected, vector)
        actions = legal_next_actions(
            runtime, tools=_available_tools(runtime, tools),
        )
        selected = _dispatchable_owner_menu(
            runtime, selected, tools, legal_actions=actions,
        )
    available = getattr(runtime, "_completion_owner_full_tools", ()) or tools
    shared_menu = _host_debt_menu_projection(runtime, selected, available, vector)
    if shared_menu is None:
        return None
    _decision, selected = shared_menu
    names = tuple(_tool_name(tool) for tool in selected)
    runtime.completion_menu = names
    runtime.offered_tools = names
    runtime._question_done_available = "done" in names
    runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
    setattr(runtime, "_completion_owner_actionable", any(
        name != "done" for name in names
    ))
    if isinstance(tools, list):
        tools[:] = copy.deepcopy(selected)
    return selected


def _terminal_tool_menu_impl(runtime: object, tools: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Make dispatch expose exactly the owner-selected legal action set."""
    selected = [dict(tool) for tool in tools]
    if not enabled(runtime):
        return selected
    # The host blocker vector is computed once for this artifact/evidence
    # identity and must gate every terminal projection, including a frozen
    # done-only contract.  Do this before owner-specific narrowing so an
    # unrelated current blocker cannot be hidden by a passing clause.
    pass_gates = getattr(runtime, "pass_gates_vector", None)
    if callable(pass_gates):
        try:
            from kernel_pass_gates_contract import project_menu
            vector = pass_gates(selected)
            shared_menu = _apply_host_debt_terminal_menu(
                runtime, tools, selected, vector, project_menu,
            )
            if shared_menu is not None:
                return shared_menu
            if getattr(vector, "ready", False) is not True:
                selected = project_menu(selected, vector)
                actions = legal_next_actions(
                    runtime, tools=_available_tools(runtime, tools),
                )
                selected = _dispatchable_owner_menu(
                    runtime, selected, tools, legal_actions=actions,
                )
                from kernel_done_readiness_authority import (
                    decision_for_runtime, enabled as done_authority_enabled,
                    project_menu as project_done_menu,
                )
                if done_authority_enabled(runtime):
                    available = (getattr(runtime, "_completion_owner_full_tools", ())
                                 or tools)
                    done_decision = decision_for_runtime(
                        runtime, vector=vector, full_tools=available,
                    )
                    uncertified_menu = None
                    if (done_decision is not None
                            and done_decision.decision == "ACCEPTED_UNCERTIFIED"):
                        from kernel_done_readiness_authority import (
                            DoneReadinessContractError, abandon_decision,
                        )
                        try:
                            uncertified_menu = project_done_menu(
                                done_decision, selected, available, offer_done=True,
                                capability_admissible=lambda name:
                                    capability_floor_action_admissible(runtime, name),
                            )
                        except DoneReadinessContractError as error:
                            abandon_decision(runtime, done_decision, error)
                    if uncertified_menu is not None:
                        selected = uncertified_menu
                    else:
                        selected = [tool for tool in selected
                                    if _tool_name(tool) != "done"]
                else:
                    selected = [tool for tool in selected if _tool_name(tool) != "done"]
                names = tuple(_tool_name(tool) for tool in selected)
                runtime.completion_menu = names
                runtime.offered_tools = names
                runtime._question_done_available = False
                runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
                owner_state = _state(runtime)
                terminal_reserved = bool(
                    owner_state.get("final_submit_reserved")
                    or owner_state.get("final_submit_request_pending")
                )
                setattr(runtime, "_completion_owner_actionable", not terminal_reserved and any(
                    str(action.get("tool", "")).strip() != "done"
                    for action in actions if isinstance(action, Mapping)
                ))
                if isinstance(tools, list):
                    tools[:] = copy.deepcopy(selected)
                return selected
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            # A missing gate projection cannot certify completion.  Preserve
            # only the existing recovery surface and record the host-owned
            # unsupported outcome instead of exposing a stale done route.
            selected = [tool for tool in selected if _tool_name(tool) != "done"]
            runtime.completion_menu = tuple(_tool_name(tool) for tool in selected)
            runtime._question_done_available = False
            runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
            runtime._pass_gates_last_surface = {
                "status": "HOST_UNSUPPORTED",
                "reason": "pass-gates vector unavailable",
            }
            if isinstance(tools, list):
                tools[:] = copy.deepcopy(selected)
            return selected
    route_decision = completion_route_decision(
        runtime, getattr(runtime, "_last_completion_decision", None),
        tools=getattr(runtime, "_completion_owner_full_tools", ()) or selected,
    )
    green_open_work = _green_open_work(runtime)
    if (_product_parent_unchanged(runtime, reason="completion_owner_terminal_menu")
            and not green_open_work):
        _clear_terminal_offer_state(runtime)
        selected = _dispatchable_owner_menu(runtime, selected, tools)
        runtime.completion_menu = tuple(_tool_name(tool) for tool in selected)
        runtime.offered_tools = tuple(runtime.completion_menu)
        runtime._question_done_available = "done" in runtime.completion_menu
        runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
        if isinstance(tools, list):
            tools[:] = copy.deepcopy(selected)
        return selected
    routed = _route_menu_tools(runtime, route_decision, tools)
    if routed is not None:
        return routed
    frozen_contract = request_action_contract(runtime)
    if frozen_contract is not None:
        frozen_tools = frozen_contract.get("admitted_tool_schemas", ())
        selected = _contract_tools(frozen_tools)
        if tuple(_tool_name(tool) for tool in selected) == ("done",):
            selected = _terminal_tools_with_deferred_read(runtime, selected, selected[0])
        if _reqwitness_open_ids(runtime) and _reserved_terminal_menu_names(
                runtime, tuple(_tool_name(tool) for tool in selected)):
            # A contract frozen before clause extraction must not preserve the
            # historical done-only projection while a public clause is open.
            return _set_done_only_menu(runtime, tools, reason="frozen contract has open clause")
        runtime.completion_menu = tuple(_contract_tool_name(tool) for tool in selected)
        runtime.offered_tools = tuple(runtime.completion_menu)
        runtime._question_done_available = "done" in runtime.completion_menu
        runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
        return selected
    if (_stall_rule_no_progress_edit_cap(runtime) and not _no_host_checks(runtime)
            and not green_open_work):
        setattr(runtime, "_completion_owner_no_progress_edit_cap", True)
        _begin_uncertified_submit(runtime)
        _notify_v46_attempt_ceiling(runtime, {
            "result": "UNCERTIFIED_SUBMIT", "accepted": False, "certified": False,
            "steps": getattr(getattr(runtime, "owner", None), "current_step", 0),
        })
        runtime.completion_menu = ()
        runtime._question_done_available = False
        runtime._completion_owner_dispatch_tools = []
        if isinstance(tools, list):
            tools[:] = []
        _emit(runtime, "completion_owner_no_progress_edit_cap",
              notice=_stall_rule_cap_notice(runtime), open_ids=list(open_ids(runtime)))
        return []
    state = _state(runtime)
    if (state["terminal_choice_offered"] and state["final_submit_reserved"]
            and state["final_submit_request_pending"]):
        # ``offer_terminal_choice`` may have rendered the done-only handoff
        # before this final provider-boundary projection.  Never let a later
        # menu stage re-expand that request to the full registry: plain text
        # such as ``done`` is not a structured tool call and otherwise settles
        # as an invalid proposal even though the prompt says no other action is
        # available.
        if not _no_host_checks(runtime):
            return _set_done_only_menu(runtime, selected, reason="reserved terminal offer")
    witness_receipt_closes_clause = _current_witness_receipt_closes_clause(runtime)
    if witness_receipt_closes_clause:
        candidates = [*selected, *(dict(tool) for tool in getattr(
            runtime, "_completion_owner_full_tools", ()) if isinstance(tool, Mapping))]
        done = [tool for tool in candidates
                if isinstance(tool.get("function"), Mapping)
                and tool["function"].get("name") == "done"]
        if done:
            done = _terminal_tools_with_deferred_read(runtime, candidates, done[0])
            names = tuple(_tool_name(tool) for tool in done)
            runtime.completion_menu = names
            runtime.offered_tools = names
            runtime._question_done_available = True
            runtime._completion_owner_dispatch_tools = copy.deepcopy(done)
            _mark_owner_done_selection(runtime, "witness_receipt_closes_clause")
            if isinstance(tools, list):
                tools[:] = copy.deepcopy(done)
            _emit(runtime, "completion_owner_terminal_menu_narrowed",
                  offered=list(names), reason="current clause witness receipt")
            return done
    if (not witness_receipt_closes_clause and not _no_host_checks(runtime)
            and stall_open_work_actions(runtime, selected)):
        projected = project_stall_open_work(runtime, selected)
        return [
            tool for tool in projected
            if _current_tool_admission(runtime, _tool_name(tool))[0]
            and greenlock_predispatch_admission(runtime, _tool_name(tool), {})
        ]
    declared_repairs = () if (witness_receipt_closes_clause or _no_host_checks(runtime)) else _declared_repair_actions(runtime, selected)
    if declared_repairs:
        runtime.completion_menu = tuple(str(action.get("tool", "")) for action in declared_repairs)
        runtime._question_done_available = False
        available = {_tool_name(tool): tool for tool in _available_tools(runtime, selected)}
        projected = [available[action["tool"]] for action in declared_repairs
                     if action.get("tool") in available]
        runtime._completion_owner_dispatch_tools = copy.deepcopy(projected)
        if isinstance(tools, list):
            tools[:] = copy.deepcopy(projected)
        _emit(runtime, "completion_owner_declared_repair_menu",
              offered=list(runtime.completion_menu),
              clause_ids=list((_declared_repair_state(runtime) or {}).get("clause_ids", ())))
        return projected
    review_actions = () if witness_receipt_closes_clause else _witness_evidence_actions(runtime, selected)
    if review_actions:
        available = {_tool_name(tool): tool for tool in _available_tools(runtime, selected)}
        projected = [available[action["tool"]] for action in review_actions
                     if action.get("tool") in available]
        runtime.completion_menu = tuple(_tool_name(tool) for tool in projected)
        runtime._question_done_available = False
        runtime._completion_owner_dispatch_tools = copy.deepcopy(projected)
        if isinstance(tools, list):
            tools[:] = copy.deepcopy(projected)
        _emit(runtime, "completion_owner_evidence_witness_menu",
              offered=list(runtime.completion_menu),
              actions=[dict(action) for action in review_actions])
        return projected
    governor = _common_sense_decision(runtime, selected)
    if governor is not None and not bool(getattr(governor, "release_uncertified", False)):
        # Common-sense actions are owner-level menu recommendations; they do
        # not know the model's concrete file/receipt payload yet.  Keep the
        # menu narrow only when the named route is dispatchable, then enforce
        # its exact schema in the real dispatch call.
        actions = _admitted_action_projections(
            runtime, _common_sense_actions(governor), selected, validate_arguments=False)
        has_common_sense_actions = bool(actions)
        if has_common_sense_actions:
            action_names = {str(action.get("tool", "")) for action in actions}
            available = _available_tools(runtime, selected)
            if not any(_tool_name(tool) in action_names for tool in available):
                available = _available_tools(
                    runtime, getattr(runtime, "_completion_owner_full_tools", ()))
            if str(getattr(governor, "frame_kind", "")) == "FLP1-B":
                by_name = {_tool_name(tool): tool for tool in available}
                selected = [by_name[name] for name in (
                    str(action.get("tool", "")) for action in actions
                ) if name in by_name]
            else:
                selected = [tool for tool in available if _tool_name(tool) in action_names]
            runtime.completion_menu = tuple(_tool_name(tool) for tool in selected)
            runtime._question_done_available = "done" in runtime.completion_menu
            runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
            if isinstance(tools, list):
                tools[:] = copy.deepcopy(selected)
            _emit(
                runtime,
                "completion_owner_next_action_menu",
                offered=list(runtime.completion_menu),
                actions=[dict(action) for action in actions],
            )
            return selected
        # A live governor rule with no admitted model action is a host-side
        # conflict, never permission to fall through to done-only.
        available = _available_tools(runtime, selected)
        if not any(_tool_name(tool) != "done" for tool in available):
            available = _available_tools(
                runtime, getattr(runtime, "_completion_owner_full_tools", ()))
        available = [tool for tool in available if _tool_name(tool) != "done"]
        runtime.completion_menu = tuple(_tool_name(tool) for tool in available)
        runtime._question_done_available = "done" in runtime.completion_menu
        runtime._completion_owner_dispatch_tools = copy.deepcopy(available)
        if isinstance(tools, list):
            tools[:] = copy.deepcopy(available)
        _emit(
            runtime,
            "completion_owner_common_sense_host_block",
            rule_ids=list(getattr(governor, "rule_ids", ())),
        )
        return available
    if governor is None and not _common_sense_enabled(runtime):
        premature = _premature_frame(runtime, selected)
        if premature is not None:
            action_names = [str(item) for item in premature.get("menu", ())]
            available = _available_tools(runtime, selected)
            by_name = {_tool_name(tool): tool for tool in available}
            selected = [
                by_name[name] for name in action_names
                if name in by_name
                and _current_tool_admission(runtime, name)[0]
                and greenlock_predispatch_admission(runtime, name, {})
            ]
            runtime.completion_menu = tuple(_tool_name(tool) for tool in selected)
            runtime._question_done_available = "done" in runtime.completion_menu
            runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
            if isinstance(tools, list):
                tools[:] = copy.deepcopy(selected)
            _emit(runtime, "completion_owner_premature_rule_menu",
                  offered=list(runtime.completion_menu))
            return selected
    cutoff_selected = _apply_cutoff_rule_menu(runtime, selected)
    if cutoff_selected is not None:
        return cutoff_selected
    if (_green_terminal_ready(runtime) and _no_host_checks(runtime)
            and not green_open_work):
        available = {_tool_name(tool): tool for tool in _available_tools(runtime, selected)}
        allowed = ("done", "run_tests", "edit", "str_replace", "read")
        projected = [available[name] for name in allowed if name in available]
        if tuple(_tool_name(tool) for tool in projected) == ("done",):
            projected = _terminal_tools_with_deferred_read(
                runtime, list(available.values()), projected[0],
            )
        runtime.completion_menu = tuple(_tool_name(tool) for tool in projected)
        runtime._question_done_available = "done" in runtime.completion_menu
        runtime._completion_owner_dispatch_tools = copy.deepcopy(projected)
        if isinstance(tools, list):
            tools[:] = copy.deepcopy(projected)
        _emit(runtime, "completion_owner_no_host_checks_menu",
              offered=list(runtime.completion_menu))
        return projected
    if _green_terminal_ready(runtime) and not green_open_work:
        return _set_done_only_menu(runtime, tools, reason="current trusted green on unchanged revision")
    if next_action_owner_enabled(runtime) or green_open_work:
        actions = legal_next_actions(runtime, tools=selected)
        has_next_actions = bool(actions)
        if has_next_actions:
            action_names = {str(action.get("tool", "")) for action in actions}
            available = _available_tools(runtime, selected)
            selected = [tool for tool in available if _tool_name(tool) in action_names]
            runtime.completion_menu = tuple(_tool_name(tool) for tool in selected)
            runtime._question_done_available = "done" in runtime.completion_menu
            runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
            if isinstance(tools, list):
                tools[:] = copy.deepcopy(selected)
            _emit(runtime, "completion_owner_next_action_menu",
                  offered=list(runtime.completion_menu),
                  actions=[dict(action) for action in actions])
            return selected
        if _decision_codes(runtime, None):
            _begin_uncertified_submit(runtime)
            full = _available_tools(runtime, selected)
            runtime.completion_menu = tuple(_tool_name(tool) for tool in full)
            runtime._question_done_available = "done" in runtime.completion_menu
            runtime._completion_owner_dispatch_tools = copy.deepcopy(full)
            if isinstance(tools, list):
                tools[:] = copy.deepcopy(full)
            codes = list(_decision_codes(runtime, None))
            digest = _digest({"codes": codes, "revision": _revision(runtime),
                              "offered": list(runtime.completion_menu)})
            if getattr(runtime, "_completion_owner_unclearable_refusal_digest", None) != digest:
                runtime._completion_owner_unclearable_refusal_digest = digest
                _emit(runtime, "completion_owner_unclearable_refusal",
                      reason_codes=codes, offered=list(runtime.completion_menu))
            return full
    if not state["terminal_choice_offered"] or not state["final_submit_reserved"]:
        return selected
    # StateFrameRuntime can apply the same owner menu once while building the
    # live dispatch surface and again at the provider-bound prepare seam.  The
    # second call is the same request, not another reserved response.  Reusing
    # an already narrowed done-only surface keeps the exhaustion cap tied to
    # real provider requests and prevents an empty third request.
    if state["done_only_offer_count"] and _reserved_terminal_menu_names(
            runtime, tuple(_tool_name(tool) for tool in selected)):
        names = tuple(_tool_name(tool) for tool in selected)
        runtime.completion_menu = names
        runtime.offered_tools = names
        runtime._question_done_available = True
        runtime._completion_owner_dispatch_tools = copy.deepcopy(selected)
        return selected
    from kernel_capability_floor import enabled as capability_floor_enabled

    # This count limits the legacy terminal-only surface, not the active run's
    # hard budget.  The floor keeps recovery schemas available after it trips.
    cap_exhausted = (
        state["done_only_offer_count"] >= MAX_RESERVED_RESPONSES
        and not capability_floor_enabled(runtime)
    )
    if cap_exhausted:
        from kernel_terminal_policy import (
            enabled as hard_budget_only_enabled,
            queue_advisory,
        )
        if hard_budget_only_enabled(runtime):
            queue_advisory(runtime, "owner_reserved_submit_cap")
        else:
            runtime.system_outcome = FINAL_SUBMIT_EXHAUSTED
        runtime.completion_menu = ()
        runtime._question_done_available = False
        runtime._completion_owner_dispatch_tools = []
        if isinstance(tools, list):
            tools[:] = []
        _emit(runtime, "completion_owner_exhausted",
              reason="done_only_offer_cap",
              revision=state["terminal_choice_revision"],
              done_only_offer_count=state["done_only_offer_count"])
        return []
    candidate, revision, _ = _candidate(runtime)
    if not candidate or state["terminal_choice_revision"] != revision:
        return selected
    full_surface = getattr(runtime, "_completion_owner_full_tools", ())
    candidates = [*selected, *(dict(tool) for tool in full_surface
                                if isinstance(tool, Mapping))]
    done = []
    for tool in candidates:
        if (isinstance(tool.get("function"), Mapping)
                and tool["function"].get("name") == "done"):
            done = [tool]
            break
    if not done:
        _emit(runtime, "completion_owner_terminal_choice_unavailable", reason="done_tool_missing")
        return []
    state["done_only_offer_count"] += 1
    prior_selected = list(selected)
    terminal_selected = _terminal_tools_with_deferred_read(runtime, candidates, done[0])
    names = tuple(_tool_name(tool) for tool in terminal_selected)
    runtime.completion_menu = names
    runtime.offered_tools = names
    runtime._question_done_available = True
    runtime._completion_owner_dispatch_tools = copy.deepcopy(terminal_selected)
    _mark_owner_done_selection(runtime, "terminal_choice")
    if isinstance(tools, list):
        # The request loop keeps this list as its live dispatch surface. Mutate
        # it so the menu narrowed here is the menu sent to the provider.
        tools[:] = copy.deepcopy(terminal_selected)
    _emit(runtime, "completion_owner_terminal_menu_narrowed",
          offered=list(names), removed=[
              str(tool.get("function", {}).get("name", "")) for tool in prior_selected
              if tool not in terminal_selected
          ], done_only_offer_count=state["done_only_offer_count"])
    return terminal_selected


def _legacy_terminal_result(runtime: object, result: Mapping[str, object]) -> dict[str, object]:
    """Project a truthful v42 terminal outcome without certifying green alone."""
    from kernel_done_is_submit_authority import terminal_from_seal

    sealed = terminal_from_seal(runtime)
    if sealed is not None:
        return sealed
    terminal = dict(result)
    if not enabled(runtime):
        return terminal
    pass_gates = getattr(runtime, "pass_gates_vector", None)
    if callable(pass_gates):
        try:
            from kernel_pass_gates_contract import project_terminal_result
            vector = pass_gates()
            if getattr(vector, "ready", False) is not True:
                return project_terminal_result(terminal, vector)
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            terminal.update({
                "accepted": False, "certified": False, "uncertified": True,
                "completion_owner_reason": "pass_gates_vector_unavailable",
            })
            return terminal
    try:
        from kernel_clause_review import terminal_projection as clause_review_projection
        clause_review = clause_review_projection(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        clause_review = None
    if isinstance(clause_review, Mapping):
        terminal["clause_review"] = copy.deepcopy(dict(clause_review))
        if clause_review.get("accepted") is False:
            terminal.update({
                "result": "INCOMPLETE", "outcome": "INCOMPLETE", "accepted": False,
                "certified": False, "completion_owner": SCHEMA,
                "completion_owner_reason": "bounded_clause_review_incomplete",
            })
            _add_terminal_open_diagnostics(
                runtime, terminal, "bounded_clause_review_incomplete")
            return terminal
    if _stall_rule_no_progress_edit_cap(runtime) and not _no_host_checks(runtime):
        _begin_uncertified_submit(runtime)
        ids = _uncertified_open_ids(runtime)
        outcome = {
            "result": "UNCERTIFIED_SUBMIT", "outcome": "UNCERTIFIED_SUBMIT",
            "accepted": False, "certified": False, "uncertified": True,
            "open_ids": list(ids), "completion_owner": SCHEMA,
            "open_ids_status": "known",
            "completion_owner_reason": "no_progress_edit_cap",
        }
        terminal.update(outcome)
        _emit(runtime, "completion_owner_no_progress_edit_cap_terminal",
              open_ids=list(ids))
        _notify_v46_attempt_ceiling(runtime, terminal)
        return terminal
    witness_receipt_closes_clause = _current_witness_receipt_closes_clause(runtime)
    declared_state = None if witness_receipt_closes_clause else _declared_repair_state(runtime)
    if (declared_state and declared_state.get("active")
            and int(declared_state.get("response_count", 0) or 0) >= MAX_DECLARED_REPAIR_RESPONSES):
        terminal.update({
            "result": "INCOMPLETE", "outcome": "INCOMPLETE", "accepted": False,
            "certified": False, "completion_owner": SCHEMA,
            "completion_owner_reason": "declared_repair_response_cap",
        })
        _add_terminal_open_diagnostics(runtime, terminal, "declared_repair_response_cap")
        return terminal
    governor = _common_sense_decision(runtime)
    if governor is not None and not bool(getattr(governor, "release_uncertified", False)):
        terminal.update({
            "result": "INCOMPLETE", "outcome": "INCOMPLETE", "accepted": False,
            "certified": False, "completion_owner": SCHEMA,
            "completion_owner_reason": "common_sense_open_evidence",
        })
        _add_terminal_open_diagnostics(runtime, terminal, "common_sense_open_evidence")
        return terminal
    candidate, revision, ids = _candidate(runtime)
    accepted = bool(done_accepted(runtime))
    decision = getattr(runtime, "_last_completion_decision", None)
    accepted = accepted or bool(getattr(decision, "accepted", False))
    violation = known_violation(runtime)
    events = _journal_records(runtime)
    keyed_confirmed = any(
        isinstance(event, Mapping)
        and event.get("kind") == "submission_idempotency"
        and event.get("phase") == "confirmed"
        for event in events
    )
    if _state(runtime)["final_submit_request_pending"]:
        terminal.update({
            "result": "INCOMPLETE", "outcome": "INCOMPLETE", "accepted": False,
            "certified": False, "completion_owner": SCHEMA,
            "completion_owner_reason": "reserved_response_not_submitted",
        })
        _add_terminal_open_diagnostics(runtime, terminal, "reserved_response_not_submitted")
        _emit(runtime, "completion_owner_unsent_reserved_response")
        return terminal
    if getattr(runtime, "_completion_owner_uncertified_submit", False):
        prior = getattr(runtime, "_completion_owner_uncertified_submit_result", None)
        if isinstance(prior, Mapping):
            terminal.update(copy.deepcopy(dict(prior)))
            _add_terminal_open_diagnostics(
                runtime,
                terminal,
                str(terminal.get("completion_owner_reason") or "uncertified_submit"),
            )
            return terminal
        ids = _uncertified_open_ids(runtime)
        outcome = {
            "result": "UNCERTIFIED_SUBMIT", "outcome": "UNCERTIFIED_SUBMIT",
            "accepted": False, "certified": False, "uncertified": True,
            "open_ids": list(ids), "completion_owner": SCHEMA,
            "open_ids_status": "known",
            "completion_owner_reason": "no_model_action_can_clear_refusal",
        }
        setattr(runtime, "_completion_owner_uncertified_submit_result", copy.deepcopy(outcome))
        terminal.update(outcome)
        _emit(runtime, "completion_owner_uncertified_submit",
              open_ids=list(ids), revision=revision)
        return terminal
    if violation or not candidate or not accepted or not _closed(runtime):
        terminal.update({
            "result": "INCOMPLETE", "outcome": "INCOMPLETE", "accepted": False,
            "certified": False, "completion_owner": SCHEMA,
            "completion_owner_reason": (
                "known_violation" if violation else
                "green_or_effects_not_current" if not candidate else
                "submission_not_accepted"),
        })
        _add_terminal_open_diagnostics(
            runtime,
            terminal,
            "known_violation" if violation else
            "green_or_effects_not_current" if not candidate else
            "submission_not_accepted",
        )
        return terminal
    premature = _premature_frame(runtime)
    if premature is not None:
        open_ids = [str(item) for item in premature.get("clause_ids", ()) if str(item)]
        terminal.update({
            "result": "UNCERTIFIED_SUBMIT", "outcome": "UNCERTIFIED_SUBMIT",
            "accepted": False, "certified": False, "uncertified": True,
            "open_ids": open_ids, "completion_owner": SCHEMA,
            "open_ids_status": "known" if open_ids else "verified_empty",
            "completion_owner_reason": "public_green_does_not_close_open_clauses",
        })
        _emit(runtime, "completion_owner_premature_rule_uncertified",
              open_ids=open_ids, menu=list(premature.get("menu", ())))
        return terminal
    if ids:
        if (getattr(runtime, "stateframe_submit_idempotent_mode", "off") == "on"
                and not keyed_confirmed
                and not bool(getattr(getattr(runtime, "continuation", None), "completed", False))):
            terminal.update({
                "result": "INCOMPLETE", "outcome": "INCOMPLETE", "accepted": False,
                "certified": False, "completion_owner": SCHEMA,
                "completion_owner_reason": "uncertified_submit_not_keyed",
            })
            _add_terminal_open_diagnostics(runtime, terminal, "uncertified_submit_not_keyed")
            return terminal
        terminal.update({
            "result": "UNCERTIFIED_SUBMIT", "outcome": "UNCERTIFIED_SUBMIT",
            "accepted": False, "certified": False, "uncertified": True,
            "open_ids": list(ids), "completion_owner": SCHEMA,
            "open_ids_status": "known",
            "completion_owner_revision": revision,
        })
        _emit(runtime, "completion_owner_uncertified_submit",
              open_ids=list(ids), revision=revision, keyed=keyed_confirmed)
        return terminal
    terminal.update({
        "accepted": True, "certified": True, "completion_owner": SCHEMA,
        "completion_owner_revision": revision,
    })
    _add_terminal_open_diagnostics(runtime, terminal, "accepted_submission")
    return terminal


def seal_terminal_disposition(runtime: object,
                              result: Mapping[str, object]) -> dict[str, object]:
    """Attach one terminal receipt and project every accepted field from it."""
    from kernel_done_is_submit_authority import terminal_from_seal

    sealed = terminal_from_seal(runtime)
    if sealed is not None:
        from kernel_terminal_disposition import mint, project
        return project(sealed, mint(runtime, sealed), runtime=runtime)
    from kernel_terminal_disposition import mint, project

    gate_result = dict(result)
    gate_vector = None
    gate_error = False
    pass_gates = getattr(runtime, "pass_gates_vector", None)
    if callable(pass_gates):
        try:
            from kernel_pass_gates_contract import project_terminal_result
            gate_vector = pass_gates()
            gate_result = project_terminal_result(gate_result, gate_vector)
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            gate_error = True
            gate_result["accepted"] = False
            gate_result["certified"] = False
            gate_result["completion_owner_reason"] = "PASS_GATES_VECTOR_UNAVAILABLE"

    existing = getattr(runtime, "_terminal_disposition_receipt", None)
    current_gate_identity = (
        str(getattr(gate_vector, "identity", "")) if gate_vector is not None else ""
    )
    existing_gate_identity = ""
    if isinstance(existing, Mapping) and isinstance(existing.get("pass_gates"), Mapping):
        existing_gate_identity = str(existing["pass_gates"].get("vector_identity", ""))
    result_artifact = str(
        gate_result.get("artifact_digest", gate_result.get("candidate_digest", "")) or ""
    )
    existing_artifact = str(existing.get("artifact_digest", "")) if isinstance(existing, Mapping) else ""
    expected_acceptance = "ACCEPTED" if gate_result.get("accepted") is True else "UNACCEPTED"
    expected_certification = "CERTIFIED" if gate_result.get("certified") is True else "UNCERTIFIED"
    receipt_state_changed = isinstance(existing, Mapping) and (
        existing_artifact != result_artifact
        or existing.get("acceptance") != expected_acceptance
        or existing.get("certification") != expected_certification
    )
    result_facts = gate_result.get("done_facts")
    result_facts_ref = (
        str(result_facts.get("evidence_digest"))
        if isinstance(result_facts, Mapping) and result_facts.get("evidence_digest")
        else ""
    )
    if (isinstance(existing, Mapping) and result_facts_ref
            and existing.get("done_facts_ref") != result_facts_ref):
        receipt_state_changed = True
    if (not isinstance(existing, Mapping)
            or (current_gate_identity and existing_gate_identity != current_gate_identity)
            or receipt_state_changed):
        existing = mint(runtime, gate_result)
        setattr(runtime, "_terminal_disposition_receipt", copy.deepcopy(dict(existing)))
        _emit(runtime, "terminal_disposition_minted",
              disposition_producer="kernel_terminal_disposition",
              receipt=copy.deepcopy(dict(existing)))
    projected = project(gate_result, existing, runtime=runtime)
    if gate_vector is not None:
        from kernel_pass_gates_contract import project_terminal_result
        projected = project_terminal_result(projected, gate_vector)
    elif gate_error:
        projected["accepted"] = False
        projected["certified"] = False
        projected["uncertified"] = True
        projected["completion_owner_reason"] = "PASS_GATES_VECTOR_UNAVAILABLE"
    return projected


def terminal_result(runtime: object, result: Mapping[str, object]) -> dict[str, object]:
    """Return the sealed submission or legacy terminal projection."""
    from kernel_done_is_submit_authority import terminal_from_seal
    from kernel_terminal_lattice import join_terminal

    sealed = terminal_from_seal(runtime)
    if sealed is not None:
        return join_terminal(
            runtime, sealed, writer="kernel_completion_owner.terminal_result",
        )
    terminal = _legacy_terminal_result(runtime, result)
    from kernel_done_readiness_authority import (
        apply_terminal_result, decision_for_runtime,
        enabled as done_authority_enabled,
    )
    done_decision = decision_for_runtime(runtime)
    terminal = apply_terminal_result(runtime, terminal, done_decision)
    try:
        from kernel_postgreen_protocol import terminal_projection
        settled = terminal_projection(runtime, terminal)
        if settled is not None:
            # Post-green settlement contributes diagnostics only.  The common
            # owner still evaluates candidate/evidence/requirements/effects
            # and mints the one terminal disposition below.
            terminal = settled
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    if (not enabled(runtime) and not done_authority_enabled(runtime)
            and done_decision is None):
        return join_terminal(
            runtime, terminal, writer="kernel_completion_owner.terminal_result",
        )
    # Host done facts are finalized by StateFrameRuntime after the bounded exit
    # is selected; minting before that would make a provisional result the
    # authority instead of the runner-derived facts.
    if (getattr(runtime, "stateframe_host_done_facts_mode", "off") == "on"
            and not isinstance(terminal.get("done_facts"), Mapping)):
        return join_terminal(
            runtime, terminal, writer="kernel_completion_owner.terminal_result",
        )
    return join_terminal(
        runtime, seal_terminal_disposition(runtime, terminal),
        writer="kernel_completion_owner.terminal_result",
    )


def request_menu_action_admissible(
        runtime: object | None, tool_name: object, request_id: object,
) -> bool:
    """Keep dispatch on the exact request-scoped menu accepted by the owner."""
    if runtime is None:
        return True
    from kernel_done_is_submit_authority import enabled as done_submit_enabled
    if (str(tool_name or "").strip() == "done"
            and done_submit_enabled(runtime)):
        return True
    current_id = str(request_id).strip() if request_id is not None else None
    if current_id == "":
        current_id = None
    contract = getattr(runtime, "_completion_owner_request_contract", None)
    if (current_id is not None and isinstance(contract, Mapping)
            and contract.get("request_id") == current_id):
        names = contract.get("advertised_actions")
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes, bytearray)):
            return str(tool_name or "").strip() in names
    from kernel_rejection_template_restrict import active_menu_tools

    allowed = active_menu_tools(runtime, current_id)
    return allowed is None or str(tool_name or "").strip() in allowed


__all__ = [
    "CompletionRouteDecision", "FINAL_SUBMIT_EXHAUSTED", "MAX_RESERVED_RESPONSES", "MODE_ENV", "NEXT_ACTION_OWNER_ENV",
    "NO_DONE_ONLY_COLLAPSE_ENV", "OWNER_BLOCK_SCHEMA", "OWNER_MESSAGE_BLOCK_KEY", "REQUEST_CONTRACT_SCHEMA",
    "RequestActionContractError", "STALL_RULE_ENV", "TOOL_SPECS", "ToolDispatch", "ToolSpec",
    "REFUSAL_DISPOSITIONS", "REFUSAL_REASON_CODES", "SCHEMA",
    "advisory_degraded_notice", "before_provider_request", "charge_response", "completion_route_decision",
    "completion_route_notice", "enabled",
    "initial_state", "known_violation", "mode", "offer_terminal_choice", "open_ids",
    "host_public_check_arguments", "is_reserved_request", "legal_next_actions",
    "next_action_owner_enabled", "greenlock_predispatch_admission",
    "completion_owner_reused_verification",
    "assert_tool_dispatch_contract", "canonicalize_tool_surface", "dispatch_tool_call",
    "registered_tool_specs", "validate_json_schema", "validate_tool_arguments",
    "note_model_action", "note_productive_progress", "note_request_submitted",
    "reconcile_submitted_request", "request_pending",
    "assert_request_action_contract", "establish_lean_request_action_contract",
    "establish_request_action_contract", "request_action_contract",
    "request_action_presentation_contract", "request_menu_action_admissible", "record_owner_block",
    "strip_owner_message_annotations",
    "remember_full_tools", "register_tools", "restore", "snapshot", "stall_open_work_actions",
    "stall_open_work_frame", "stall_open_work_notice", "stall_rule_enabled",
    "note_stall_open_work_attempt", "project_stall_open_work", "seal_terminal_disposition",
    "churn_decision", "handoff_churn_decision", "terminal_result", "terminal_tool_menu",
]
